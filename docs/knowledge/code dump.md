# Code Ideas

- LC Daily Para A+
    
    This code is looking at your daily parabolics like MSTR 11/21/24 or SMCI 2/16/24.
    
    I have two codes that get a good grouping of names when you look at names that are on both of these lists.
    But these codes are capturing the daily mold that we want to see, so can be good reference for how to get those.
    
    ```python
    # scan_daily.py  ── optimized with session reuse & parallel execution
    # ---------------------------------------------------------------------------
    import pandas as pd
    import numpy as np
    import requests
    from datetime import datetime
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # ─────────────────── Configuration ─────────────────── #
    session  = requests.Session()
    API_KEY  = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    BASE_URL = 'https://api.polygon.io'
    
    # ─────────────────── Data Fetching ─────────────────── #
    def fetch_aggregates(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download daily bars from Polygon.io and return a clean DataFrame."""
        url  = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        resp = session.get(url, params={'apiKey': API_KEY})
        resp.raise_for_status()
        data = resp.json().get('results', [])
        if not data:
            return pd.DataFrame()
    
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['t'], unit='ms')
        df.rename(columns={'o':'Open','h':'High','l':'Low','c':'Close','v':'Volume'}, inplace=True)
        df.set_index('Date', inplace=True)
        return df[['Open','High','Low','Close','Volume']]
    
    # ─────────────────── Metric Computations ─────────────────── #
    def compute_emas(df: pd.DataFrame, spans=(9, 20)) -> pd.DataFrame:
        for span in spans:
            df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
        return df
    
    def compute_atr(df: pd.DataFrame, period: int = 30) -> pd.DataFrame:
        hi_lo   = df['High'] - df['Low']
        hi_prev = (df['High'] - df['Close'].shift(1)).abs()
        lo_prev = (df['Low']  - df['Close'].shift(1)).abs()
        df['TR'] = pd.concat([hi_lo, hi_prev, lo_prev], axis=1).max(axis=1)
        df['ATR_raw']        = df['TR'].rolling(window=period, min_periods=period).mean()
        df['ATR']            = df['ATR_raw'].shift(1)
        df['ATR_Pct_Change'] = df['ATR_raw'].pct_change().shift(1) * 100
        return df
    
    def compute_volume(df: pd.DataFrame, period: int = 30) -> pd.DataFrame:
        df['VOL_AVG_raw'] = df['Volume'].rolling(window=period, min_periods=period).mean()
        df['VOL_AVG']     = df['VOL_AVG_raw'].shift(1)
        df['Prev_Volume'] = df['Volume'].shift(1)
        return df
    
    def compute_slopes(df: pd.DataFrame, span: int, windows=(3, 5, 15)) -> pd.DataFrame:
        for w in windows:
            df[f'Slope_{span}_{w}d'] = (
                (df[f'EMA_{span}'] - df[f'EMA_{span}'].shift(w)) / df[f'EMA_{span}'].shift(w)
            ) * 100
        return df
    
    def compute_custom_50d_slope(df: pd.DataFrame, span: int = 9,
                                 start_shift: int = 4, end_shift: int = 50) -> pd.DataFrame:
        """Slope from day-4 back to day-50 (positive ⇒ up-trend)."""
        col = f'Slope_{span}_{start_shift}to{end_shift}d'
        df[col] = (
            (df[f'EMA_{span}'].shift(start_shift) - df[f'EMA_{span}'].shift(end_shift))
            / df[f'EMA_{span}'].shift(end_shift)
        ) * 100
        return df
    
    def compute_gap(df: pd.DataFrame) -> pd.DataFrame:
        df['Gap']          = (df['Open'] - df['Close'].shift(1)).abs()
        df['Gap_over_ATR'] = df['Gap'] / df['ATR']
        return df
    
    def compute_div_ema_atr(df: pd.DataFrame, spans=(9, 20)) -> pd.DataFrame:
        for span in spans:
            df[f'High_over_EMA{span}_div_ATR'] = (df['High'] - df[f'EMA_{span}']) / df['ATR']
        return df
    
    def compute_pct_changes(df: pd.DataFrame) -> pd.DataFrame:
        low7  = df['Low'].rolling(window=7 , min_periods=7 ).min()
        low14 = df['Low'].rolling(window=14, min_periods=14).min()
        df['Pct_7d_low_div_ATR']  = ((df['Close'] - low7 ) / low7 ) / df['ATR'] * 100
        df['Pct_14d_low_div_ATR'] = ((df['Close'] - low14) / low14) / df['ATR'] * 100
        return df
    
    def compute_range_position(df: pd.DataFrame) -> pd.DataFrame:
        df['Upper_70_Range'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']) * 100
        return df
    
    def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        df = (df.pipe(compute_emas)
                .pipe(compute_atr)
                .pipe(compute_volume)
                .pipe(compute_slopes, span=9)
                .pipe(compute_custom_50d_slope, span=9, start_shift=4, end_shift=50)
                .pipe(compute_gap)
                .pipe(compute_div_ema_atr)
                .pipe(compute_pct_changes)
                .pipe(compute_range_position))
    
        # multi-day references
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_Open']  = df['Open'].shift(1)
        df['Prev_High']  = df['High'].shift(1)
        df['Close_D3']   = df['Close'].shift(3)
        df['Close_D4']   = df['Close'].shift(4)
    
        # previous-day % gain
        df['Prev_Gain_Pct'] = (df['Prev_Close'] - df['Prev_Open']) / df['Prev_Open'] * 100
    
        # 1-, 2-, 3-day moves vs ATR
        df['Pct_1d']         = df['Close'].pct_change() * 100
        df['Pct_1d_div_ATR'] = df['Pct_1d'] / df['ATR']
        df['Move2d_div_ATR'] = (df['Prev_Close'] - df['Close_D3']) / df['ATR']
        df['Move3d_div_ATR'] = (df['Prev_Close'] - df['Close_D4']) / df['ATR']
    
        # misc ratios
        df['Range_over_ATR']  = df['TR'] / df['ATR']
        df['Vol_over_AVG']    = df['Volume'] / df['VOL_AVG']
        df['Close_over_EMA9'] = df['Close'] / df['EMA_9']
        df['Open_over_EMA9']  = df['Open']  / df['EMA_9']
        return df
    
    # ─────────────────── Scan Logic ─────────────────── #
    def scan_daily_para(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        defaults = {
            'atr_mult'              : 4,
            'vol_mult'              : 2,
            'slope3d_min'           : 10,
            'slope5d_min'           : 20,
            'slope15d_min'          : 40,
            'slope50d_min'        : 60,  # optional long-trend filter
            'high_ema9_mult'        : 4,
            'high_ema20_mult'       : 5,
            'pct7d_low_div_atr_min' : 0.5,
            'pct14d_low_div_atr_min': 1.5,
            'gap_div_atr_min'       : 0.5,
            'open_over_ema9_min'    : 1.25,
            'atr_pct_change_min'    : 5,
            'prev_close_min'        : 15.0,
            'prev_gain_pct_min'     : 0.25,  # ← new trigger threshold
            'pct2d_div_atr_min'     : 2,
            'pct3d_div_atr_min'     : 3,
        }
        if params:
            defaults.update(params)
        d = defaults
    
        df_m = compute_all_metrics(df.copy())
    
        cond = (
            (df_m['TR']            / df_m['ATR']        >= d['atr_mult']) &
            (df_m['Volume']        / df_m['VOL_AVG']    >= d['vol_mult']) &
            (df_m['Prev_Volume']   / df_m['VOL_AVG']    >= d['vol_mult']) &
            (df_m['Slope_9_3d']    >= d['slope3d_min']) &
            (df_m['Slope_9_5d']    >= d['slope5d_min']) &
            (df_m['Slope_9_15d']   >= d['slope15d_min']) &
            (df_m['High_over_EMA9_div_ATR']  >= d['high_ema9_mult']) &
            (df_m['High_over_EMA20_div_ATR'] >= d['high_ema20_mult']) &
            (df_m['Pct_7d_low_div_ATR']      >= d['pct7d_low_div_atr_min']) &
            (df_m['Pct_14d_low_div_ATR']     >= d['pct14d_low_div_atr_min']) &
            (df_m['Gap_over_ATR']            >= d['gap_div_atr_min']) &
            (df_m['Open'] / df_m['EMA_9']    >= d['open_over_ema9_min']) &
            (df_m['ATR_Pct_Change']          >= d['atr_pct_change_min']) &
            (df_m['Prev_Close']              >  d['prev_close_min']) &
            (df_m['Move2d_div_ATR']          >= d['pct2d_div_atr_min']) &
            (df_m['Move3d_div_ATR']          >= d['pct3d_div_atr_min']) &
            # new trigger rule: previous-day gain ≥ threshold
            (df_m['Prev_Gain_Pct']           >= d['prev_gain_pct_min']) &
            # gap-up rule
            (df_m['Open'] > df_m['Prev_High'])
            # optional long-trend filter
            #& (df_m['Slope_9_4to50d'] >= d['slope50d_min'])
        )
        return df_m.loc[cond]
    
    # ─────────────────── Worker & Parallel Scan ─────────────────── #
    def fetch_and_scan(symbol: str, start_date: str, end_date: str, params: dict) -> list[tuple[str, str]]:
        df = fetch_aggregates(symbol, start_date, end_date)
        if df.empty:
            return []
        hits = scan_daily_para(df, params)
        return [(symbol, d.strftime('%Y-%m-%d')) for d in hits.index]
    
    # ─────────────────── Main ─────────────────── #
    if __name__ == '__main__':
        custom_params = {
            'atr_mult'              : 4,
            'vol_mult'              : 2.0,
            'slope3d_min'           : 10,
            'slope5d_min'           : 20,
            'slope15d_min'          : 50,
            'high_ema9_mult'        : 4,
            'high_ema20_mult'       : 5,
            'pct7d_low_div_atr_min' : 0.5,
            'pct14d_low_div_atr_min': 1.5,
            'gap_div_atr_min'       : 0.5,
            'open_over_ema9_min'    : 1.0,
            'atr_pct_change_min'    : 5,
            'prev_close_min'        : 10.0,
            'prev_gain_pct_min'     : 0.25,   # adjust if needed
            'pct2d_div_atr_min'     : 2,
            'pct3d_div_atr_min'     : 3,
            'slope50d_min'        : 60,
        }
    
        symbols = ['MSTR', 'SMCI', 'DJT', 'BABA', 'TCOM', 'AMC', 'SOXL', 'MRVL', 'TGT', 'DOCU',
            'ZM', 'DIS', 'NFLX', 'RKT', 'SNAP', 'RBLX', 'META', 'SE', 'NVDA', 'AAPL',
            'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'AMD', 'INTC', 'BA', 'PYPL', 'QCOM', 'ORCL',
            'T', 'CSCO', 'VZ', 'KO', 'PEP', 'MRK', 'PFE', 'ABBV', 'JNJ', 'CRM',
            'BAC', 'C', 'JPM', 'WMT', 'CVX', 'XOM', 'COP', 'RTX', 'SPGI', 'GS',
            'HD', 'LOW', 'COST', 'UNH', 'NEE', 'NKE', 'LMT', 'HON', 'CAT', 'MMM',
            'LIN', 'ADBE', 'AVGO', 'TXN', 'ACN', 'UPS', 'BLK', 'PM', 'MO', 'ELV',
            'VRTX', 'ZTS', 'NOW', 'ISRG', 'PLD', 'MS', 'MDT', 'WM', 'GE', 'IBM',
            'BKNG', 'FDX', 'ADP', 'EQIX', 'DHR', 'SNPS', 'REGN', 'SYK', 'TMO', 'CVS',
            'INTU', 'SCHW', 'CI', 'APD', 'SO', 'MMC', 'ICE', 'FIS', 'ADI', 'CSX',
            'LRCX', 'GILD', 'RIVN', 'LCID', 'PLTR', 'SNOW', 'SPY', 'QQQ', 'DIA', 'IWM',
            'TQQQ', 'SQQQ', 'ARKK', 'LABU', 'TECL', 'UVXY', 'XLE', 'XLK', 'XLF', 'IBB',
            'KWEB', 'TAN', 'XOP', 'EEM', 'HYG', 'EFA', 'USO', 'GLD', 'SLV', 'BITO',
            'RIOT', 'MARA', 'COIN', 'SQ', 'AFRM', 'DKNG', 'SHOP', 'UPST', 'CLF', 'AA',
            'F', 'GM', 'ROKU', 'WBD', 'WBA', 'PARA', 'PINS', 'LYFT', 'BYND', 'RDDT',
            'GME', 'VKTX', 'APLD', 'KGEI', 'INOD', 'LMB', 'AMR', 'PMTS', 'SAVA', 'CELH',
            'ESOA', 'IVT', 'MOD', 'SKYE', 'AR', 'VIXY', 'TECS', 'LABD', 'SPXS', 'SPXL',
            'DRV', 'TZA', 'FAZ', 'WEBS', 'PSQ', 'SDOW', 'MSTU', 'MSTZ', 'NFLU', 'BTCL',
            'BTCZ', 'ETU', 'ETQ', 'FAS', 'TNA', 'NUGT', 'TSLL', 'NVDU', 'AMZU', 'MSFU',
            'UVIX', 'CRCL', 'SBET','MRNA','TIGR','PLUG','AXON','FUTU','CGC','UVXY']
    
        start_date = '2020-01-01'
        end_date   = datetime.today().strftime('%Y-%m-%d')
    
        with ThreadPoolExecutor(max_workers=5) as exe:
            futures = {
                exe.submit(fetch_and_scan, s, start_date, end_date, custom_params): s
                for s in symbols
            }
            for fut in as_completed(futures):
                for sym, hit_date in fut.result():
                    print(f"{sym} {hit_date}")
    
    ```
    
    ```python
    # scan_daily.py  ── optimized with session reuse & parallel execution
    # ---------------------------------------------------------------------------
    import pandas as pd
    import numpy as np
    import requests
    from datetime import datetime
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # ─────────────────── Configuration ─────────────────── #
    session  = requests.Session()
    API_KEY  = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    BASE_URL = 'https://api.polygon.io'
    
    # ─────────────────── Data Fetching ─────────────────── #
    def fetch_aggregates(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download daily bars from Polygon.io and return a clean DataFrame."""
        url   = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        resp  = session.get(url, params={'apiKey': API_KEY})
        resp.raise_for_status()
        data  = resp.json().get('results', [])
        if not data:
            return pd.DataFrame()
    
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['t'], unit='ms')
        df.rename(columns={'o':'Open','h':'High','l':'Low','c':'Close','v':'Volume'}, inplace=True)
        df.set_index('Date', inplace=True)
        return df[['Open','High','Low','Close','Volume']]
    
    # ─────────────────── Metric Computations ─────────────────── #
    def compute_emas(df: pd.DataFrame, spans=(9, 20)) -> pd.DataFrame:
        for span in spans:
            df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
        return df
    
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        hi_lo   = df['High'] - df['Low']
        hi_prev = (df['High'] - df['Close'].shift(1)).abs()
        lo_prev = (df['Low']  - df['Close'].shift(1)).abs()
        df['TR']             = pd.concat([hi_lo, hi_prev, lo_prev], axis=1).max(axis=1)
        df['ATR_raw']        = df['TR'].rolling(window=period, min_periods=period).mean()
        df['ATR']            = df['ATR_raw'].shift(1)
        df['ATR_Pct_Change'] = df['ATR_raw'].pct_change().shift(1) * 100
        return df
    
    def compute_volume(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df['VOL_AVG_raw'] = df['Volume'].rolling(window=period, min_periods=period).mean()
        df['VOL_AVG']     = df['VOL_AVG_raw'].shift(1)
        df['Prev_Volume'] = df['Volume'].shift(1)
        return df
    
    def compute_slopes(df: pd.DataFrame, span: int, windows=(3, 5, 15)) -> pd.DataFrame:
        for w in windows:
            df[f'Slope_{span}_{w}d'] = (
                (df[f'EMA_{span}'] - df[f'EMA_{span}'].shift(w)) / df[f'EMA_{span}'].shift(w)
            ) * 100
        return df
    
    def compute_custom_50d_slope(df: pd.DataFrame, span: int = 9,
                                 start_shift: int = 4, end_shift: int = 50) -> pd.DataFrame:
        """
        50-day slope measured from day-4 back to day-50.
        Positive = up-slope over that period.
        """
        col_name = f'Slope_{span}_{start_shift}to{end_shift}d'
        df[col_name] = (
            (df[f'EMA_{span}'].shift(start_shift) - df[f'EMA_{span}'].shift(end_shift))
            / df[f'EMA_{span}'].shift(end_shift)
        ) * 100
        return df
    
    def compute_gap(df: pd.DataFrame) -> pd.DataFrame:
        df['Gap']          = (df['Open'] - df['Close'].shift(1)).abs()
        df['Gap_over_ATR'] = df['Gap'] / df['ATR']
        return df
    
    def compute_div_ema_atr(df: pd.DataFrame, spans=(9, 20)) -> pd.DataFrame:
        for span in spans:
            df[f'High_over_EMA{span}_div_ATR'] = (df['High'] - df[f'EMA_{span}']) / df['ATR']
        return df
    
    def compute_pct_changes(df: pd.DataFrame) -> pd.DataFrame:
        low7  = df['Low'].rolling(window=7,  min_periods=7).min()
        low14 = df['Low'].rolling(window=14, min_periods=14).min()
        df['Pct_7d_low_div_ATR']  = ((df['Close'] - low7)  / low7)  / df['ATR'] * 100
        df['Pct_14d_low_div_ATR'] = ((df['Close'] - low14) / low14) / df['ATR'] * 100
        return df
    
    def compute_range_position(df: pd.DataFrame) -> pd.DataFrame:
        df['Upper_70_Range'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']) * 100
        return df
    
    def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        df = compute_emas(df)
        df = compute_atr(df)
        df = compute_volume(df)
        df = compute_slopes(df, span=9)
        df = compute_custom_50d_slope(df, span=9, start_shift=4, end_shift=50)  # NEW
        df = compute_gap(df)
        df = compute_div_ema_atr(df)
        df = compute_pct_changes(df)
        df = compute_range_position(df)
    
        # multi-day references
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_Open']  = df['Open'].shift(1)       # NEW
        df['Prev_High']  = df['High'].shift(1)       # NEW
        df['Close_D2']   = df['Close'].shift(2)
        df['Close_D3']   = df['Close'].shift(3)
        df['Close_D4']   = df['Close'].shift(4)
    
        # single-day % move vs ATR
        df['Pct_1d']         = df['Close'].pct_change() * 100
        df['Pct_1d_div_ATR'] = df['Pct_1d'] / df['ATR']
    
        # 2- & 3-day dollar move vs ATR
        df['Move2d_div_ATR'] = (df['Prev_Close'] - df['Close_D3']) / df['ATR']
        df['Move3d_div_ATR'] = (df['Prev_Close'] - df['Close_D4']) / df['ATR']
    
        # misc ratios
        df['Range_over_ATR']  = df['TR'] / df['ATR']
        df['Vol_over_AVG']    = df['Volume'] / df['VOL_AVG']
        df['Close_over_EMA9'] = df['Close'] / df['EMA_9']
        df['Open_over_EMA9']  = df['Open']  / df['EMA_9']
        return df
    
    # ─────────────────── Scan Logic ─────────────────── #
    def scan_daily_para(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        defaults = {
            'atr_mult':              4,
            'vol_mult':              2,
            'slope3d_min':          10,
            'slope5d_min':          20,
            'slope15d_min':         40,
            # NEW: 50-day custom slope threshold (optional)
            # 'slope50d_min':       60,
            'high_ema9_mult':        4,
            'high_ema20_mult':       5,
            'pct7d_low_div_atr_min': 0.5,
            'pct14d_low_div_atr_min':1.5,
            'gap_div_atr_min':       0.5,
            'open_over_ema9_min':    1.25,
            'atr_pct_change_min':   10,
            'prev_close_min':       15.0,
            # removed pct1d_div_atr_min filter (per request)
            'pct2d_div_atr_min':     2,
            'pct3d_div_atr_min':     3,
        }
        if params:
            defaults.update(params)
        d = defaults
    
        df_m = compute_all_metrics(df.copy())
    
        cond = (
            # existing filters
            (df_m['TR']            / df_m['ATR']   >= d['atr_mult']) &
            (df_m['Volume']        / df_m['VOL_AVG'] >= d['vol_mult']) &
            (df_m['Prev_Volume']   >= d['vol_mult'] * df_m['VOL_AVG'].mean()) &
            (df_m['Slope_9_3d']    >= d['slope3d_min']) &
            (df_m['Slope_9_5d']    >= d['slope5d_min']) &
            (df_m['Slope_9_15d']   >= d['slope15d_min']) &
            (df_m['High_over_EMA9_div_ATR']  >= d['high_ema9_mult']) &
            (df_m['High_over_EMA20_div_ATR'] >= d['high_ema20_mult']) &
            (df_m['Pct_7d_low_div_ATR']      >= d['pct7d_low_div_atr_min']) &
            (df_m['Pct_14d_low_div_ATR']     >= d['pct14d_low_div_atr_min']) &
            (df_m['Gap_over_ATR']            >= d['gap_div_atr_min']) &
            (df_m['Open'] / df_m['EMA_9']    >= d['open_over_ema9_min']) &
            (df_m['ATR_Pct_Change']          >= d['atr_pct_change_min']) &
            (df_m['Prev_Close']              >  d['prev_close_min']) &
            (df_m['Move2d_div_ATR']          >= d['pct2d_div_atr_min']) &
            (df_m['Move3d_div_ATR']          >= d['pct3d_div_atr_min'])
            # user-requested NEW filters
            & (df_m['Prev_Close'] > df_m['Prev_Open'])   # day-1 candle must be green
            & (df_m['Open']       > df_m['Prev_High'])   # today’s open > yesterday’s high
            # OPTIONAL custom 50-day slope filter
            # & (df_m['Slope_9_4to50d'] >= d['slope50d_min'])
        )
        return df_m.loc[cond]
    
    # ─────────────────── Worker & Parallel Scan ─────────────────── #
    def fetch_and_scan(symbol: str, start_date: str, end_date: str, params: dict) -> list[tuple[str, str]]:
        df = fetch_aggregates(symbol, start_date, end_date)
        if df.empty:
            return []
        hits = scan_daily_para(df, params)
        return [(symbol, d.strftime('%Y-%m-%d')) for d in hits.index]
    
    # ─────────────────── Main ─────────────────── #
    if __name__ == '__main__':
        custom_params = {
            'atr_mult':               2.5,
            'vol_mult':               2.0,
            'slope3d_min':           10,
            'slope5d_min':           20,
            'slope15d_min':          50,
            'high_ema9_mult':         3.5,
            'high_ema20_mult':        5,
            'pct7d_low_div_atr_min':  0.5,
            'pct14d_low_div_atr_min': 1.5,
            'gap_div_atr_min':        0.5,
            'open_over_ema9_min':     1.0,
            'atr_pct_change_min':    9,
            'prev_close_min':        10.0,
            # pct1d_div_atr_min removed
            'pct2d_div_atr_min':      2,
            'pct3d_div_atr_min':      2.5,
            # 'slope50d_min':         60,
        }
    
        symbols    = ['MSTR', 'SMCI', 'DJT', 'BABA', 'TCOM', 'AMC', 'SOXL', 'MRVL', 'TGT', 'DOCU',
        'ZM', 'DIS', 'NFLX', 'RKT', 'SNAP', 'RBLX', 'META', 'SE', 'NVDA', 'AAPL',
        'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'AMD', 'INTC', 'BA', 'PYPL', 'QCOM', 'ORCL',
        'T', 'CSCO', 'VZ', 'KO', 'PEP', 'MRK', 'PFE', 'ABBV', 'JNJ', 'CRM',
        'BAC', 'C', 'JPM', 'WMT', 'CVX', 'XOM', 'COP', 'RTX', 'SPGI', 'GS',
        'HD', 'LOW', 'COST', 'UNH', 'NEE', 'NKE', 'LMT', 'HON', 'CAT', 'MMM',
        'LIN', 'ADBE', 'AVGO', 'TXN', 'ACN', 'UPS', 'BLK', 'PM', 'MO', 'ELV',
        'VRTX', 'ZTS', 'NOW', 'ISRG', 'PLD', 'MS', 'MDT', 'WM', 'GE', 'IBM',
        'BKNG', 'FDX', 'ADP', 'EQIX', 'DHR', 'SNPS', 'REGN', 'SYK', 'TMO', 'CVS',
        'INTU', 'SCHW', 'CI', 'APD', 'SO', 'MMC', 'ICE', 'FIS', 'ADI', 'CSX',
        'LRCX', 'GILD', 'RIVN', 'LCID', 'PLTR', 'SNOW', 'SPY', 'QQQ', 'DIA', 'IWM',
        'TQQQ', 'SQQQ', 'ARKK', 'LABU', 'TECL', 'UVXY', 'XLE', 'XLK', 'XLF', 'IBB',
        'KWEB', 'TAN', 'XOP', 'EEM', 'HYG', 'EFA', 'USO', 'GLD', 'SLV', 'BITO',
        'RIOT', 'MARA', 'COIN', 'SQ', 'AFRM', 'DKNG', 'SHOP', 'UPST', 'CLF', 'AA',
        'F', 'GM', 'ROKU', 'WBD', 'WBA', 'PARA', 'PINS', 'LYFT', 'BYND', 'RDDT',
        'GME', 'VKTX', 'APLD', 'KGEI', 'INOD', 'LMB', 'AMR', 'PMTS', 'SAVA', 'CELH',
        'ESOA', 'IVT', 'MOD', 'SKYE', 'AR', 'VIXY', 'TECS', 'LABD', 'SPXS', 'SPXL',
        'DRV', 'TZA', 'FAZ', 'WEBS', 'PSQ', 'SDOW', 'MSTU', 'MSTZ', 'NFLU', 'BTCL',
        'BTCZ', 'ETU', 'ETQ', 'FAS', 'TNA', 'NUGT', 'TSLL', 'NVDU', 'AMZU', 'MSFU',
        'UVIX', 'CRCL', 'SBET','MRNA','TIGR']
        start_date = '2020-01-01'
        end_date   = datetime.today().strftime('%Y-%m-%d')
    
        with ThreadPoolExecutor(max_workers=5) as exe:
            futures = {
                exe.submit(fetch_and_scan, sym, start_date, end_date, custom_params): sym
                for sym in symbols
            }
            for fut in as_completed(futures):
                for sym, date in fut.result():
                    print(f"{sym} {date}")
    
    ```
    
- OG Scans
    
    These scans are capturing a lot of the daily molds properly and also are a great reference to see how to scan all available tickers in the market with Polygon.
    
    When we originally tried to make large cap scans, these are the ones that we ended up with—a great baseline for the new setups we are trying to make.
    
    ```python
    import pandas as pd
    import requests
    import time
    import numpy as np
    import pandas_market_calendars as mcal
    import aiohttp
    import asyncio
    import pandas as pd
    from multiprocessing import Pool, cpu_count
    from concurrent.futures import ProcessPoolExecutor
    from tabulate import tabulate
    import webbrowser
    import plotly.graph_objects as go
    import sys
    from concurrent.futures import ThreadPoolExecutor
    import dask
    from dask.distributed import Client, as_completed
    import dask.dataframe as dd
    import datetime
    import logging
    import backoff
    
    nyse = mcal.get_calendar('NYSE')
    executor = ThreadPoolExecutor() 
    
    import warnings
    warnings.filterwarnings("ignore")
    
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    DATE = "2025-01-17"
    
    # Replace with your Polygon API Key
    API_KEY = '4r6MZNWLy2ucmhVI7fY8MrvXfXTSmxpy'
    BASE_URL = "https://api.polygon.io"
    
    def adjust_daily(df):
        # df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['t'], unit='ms')
        df['date'] = df['date'].dt.date          
        df['pdc'] = df['c'].shift(1)
        df['high_low'] = df['h'] - df['l']  # High - Low
        df['high_pdc'] = abs(df['h'] - df['pdc'])  # High - Previous Day Close
        df['low_pdc'] = abs(df['l'] - df['pdc'])  # Low - Previous Day Close
        # True Range (TR) is the max of high-low, high-pdc, low-pdc
        df['true_range'] = df[['high_low', 'high_pdc', 'low_pdc']].max(axis=1)
        # Calculate the ATR (using a 14-day rolling window)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        # Drop intermediate columns for clean output
        df.drop(['high_low', 'high_pdc', 'low_pdc'], axis=1, inplace=True)
    
             
        df['h1'] = df['h'].shift(1)
        df['h2'] = df['h'].shift(2)
    
        df['h3'] = df['h'].shift(3)
    
        df['c1'] = df['c'].shift(1)
        df['c2'] = df['c'].shift(2)
    
        df['o1'] = df['o'].shift(1)
        df['o2'] = df['o'].shift(2)
        
        df['l1'] = df['l'].shift(1)
        df['l2'] = df['l'].shift(2)
    
        df['v1'] = df['v'].shift(1)
        df['v2'] = df['v'].shift(2)
        
        df['dol_v'] = (df['c'] * df['v'])
        df['dol_v1'] = df['dol_v'].shift(1)
        df['dol_v2'] = df['dol_v'].shift(2)
    
        df['close_range'] = (df['c'] - df['l'])/(df['h'] - df['l'])
        df['close_range1'] = df['close_range'].shift(1)
    
        df['gap_atr'] = ((df['o'] - df['pdc'])/df['atr'])
        df['gap_atr1'] = ((df['o1'] - df['c2'])/df['atr'])
    
        df['gap_pdh_atr'] = ((df['o'] - df['h1'])/df['atr'])
        
        df['high_chg'] = (df['h'] - df['o'])
        df['high_chg_atr'] = ((df['h'] - df['o'])/df['atr'])
        # df['high_chg_atr'] = round(df['high_chg_atr'], 2)
        df['high_chg_atr1'] = ((df['h1'] - df['o1'])/df['atr'])
    
        df['high_chg_from_pdc_atr'] = ((df['h'] - df['c1'])/df['atr'])
        df['high_chg_from_pdc_atr1'] = ((df['h1'] - df['c2'])/df['atr'])
    
        df['pct_change'] = round(((df['c'] / df['c1']) - 1)*100, 2)
        
        df['ema9'] = df['c'].ewm(span=9, adjust=False).mean().fillna(0)
        df['ema20'] = df['c'].ewm(span=20, adjust=False).mean().fillna(0)
        df['ema50'] = df['c'].ewm(span=50, adjust=False).mean().fillna(0)
        df['ema200'] = df['c'].ewm(span=200, adjust=False).mean().fillna(0)
    
        df['ema20_2'] = df['ema20'].shift(2)
        
        df['dist_h_9ema'] = (df['h'] - df['ema9'])
        df['dist_h_20ema'] = (df['h'] - df['ema20'])
        df['dist_h_50ema'] = (df['h'] - df['ema50'])
        df['dist_h_200ema'] = (df['h'] - df['ema200'])
    
        df['dist_h_9ema1'] = df['dist_h_9ema'].shift(1)
        df['dist_h_20ema1'] = df['dist_h_20ema'].shift(1)
        df['dist_h_50ema1'] = df['dist_h_50ema'].shift(1)
        df['dist_h_200ema1'] = df['dist_h_200ema'].shift(1)
    
        df['dist_h_9ema_atr'] = df['dist_h_9ema'] /df['atr']
        df['dist_h_20ema_atr'] = df['dist_h_20ema'] /df['atr']
        df['dist_h_50ema_atr'] = df['dist_h_50ema'] /df['atr']
        df['dist_h_200ema_atr'] = df['dist_h_200ema'] /df['atr']
    
        df['dist_h_9ema_atr1'] = df['dist_h_9ema1'] /df['atr']
        df['dist_h_20ema_atr1'] = df['dist_h_20ema1'] /df['atr']
        df['dist_h_50ema_atr1'] = df['dist_h_50ema1'] /df['atr']
        df['dist_h_200ema_atr1'] = df['dist_h_200ema1'] /df['atr']
    
        df['dist_h_9ema2'] = df['dist_h_9ema'].shift(2)
        df['dist_h_9ema3'] = df['dist_h_9ema'].shift(3)
        df['dist_h_9ema4'] = df['dist_h_9ema'].shift(4)
    
        df['dist_h_20ema2'] = df['dist_h_20ema'].shift(2)
        df['dist_h_20ema3'] = df['dist_h_20ema'].shift(3)
        df['dist_h_20ema4'] = df['dist_h_20ema'].shift(4)
        df['dist_h_20ema5'] = df['dist_h_20ema'].shift(5)
    
        df['dist_h_9ema_atr2'] = df['dist_h_9ema2'] /df['atr']
        df['dist_h_9ema_atr3'] = df['dist_h_9ema3'] /df['atr']
        df['dist_h_9ema_atr4'] = df['dist_h_9ema4'] /df['atr']
        
        df['dist_h_20ema_atr2'] = df['dist_h_20ema2'] /df['atr']
        df['dist_h_20ema_atr3'] = df['dist_h_20ema3'] /df['atr']
        df['dist_h_20ema_atr4'] = df['dist_h_20ema4'] /df['atr']
    
        df['lowest_low_20'] = df['l'].rolling(window=20, min_periods=1).min()
        df['lowest_low_20_2'] = df['lowest_low_20'].shift(2)
        df['h_dist_to_lowest_low_20_atr'] = ((df['h'] - df['lowest_low_20'])/df['atr'])
    
        df['lowest_low_30'] = df['l'].rolling(window=30, min_periods=1).min()
        df['lowest_low_30_1'] = df['lowest_low_30'].shift(1)
    
        df['h_dist_to_lowest_low_30'] = (df['h'] - df['lowest_low_30'])
    
        df['lowest_low_5'] = df['l'].rolling(window=5, min_periods=1).min()
        df['h_dist_to_lowest_low_5_atr'] = ((df['h'] - df['lowest_low_5'])/df['atr'])
    
        df['highest_high_100'] = df['h'].rolling(window=100).max()
        df['highest_high_100_1'] = df['highest_high_100'].shift(1)
        
        df['highest_high_250'] = df['h'].rolling(window=250, min_periods=1).max()
        df['highest_high_250_1'] = df['highest_high_250'].shift(1)
    
        df['highest_high_5'] = df['h'].rolling(window=5, min_periods=1).max()
    
        df['highest_high_50'] = df['h'].rolling(window=50, min_periods=1).max()
        df['highest_high_50_1'] = df['highest_high_50'].shift(1)
        df['highest_high_50_4'] = df['highest_high_50'].shift(4)
    
        df['highest_high_20'] = df['h'].rolling(window=20, min_periods=1).max()
        df['highest_high_20_2'] = df['highest_high_20'].shift(2)
    
        df['highest_high_100'] = df['h'].rolling(window=100, min_periods=1).max()
        df['highest_high_100_4'] = df['highest_high_100'].shift(4)
    
        return df
    def adjust_intraday(df):
        #df=pd.DataFrame(results)
        df['date_time']=pd.to_datetime(df['t']*1000000).dt.tz_localize('UTC')
        df['date_time']=df['date_time'].dt.tz_convert('US/Eastern')
    
        #df['Date Time'] = pd.to_datetime(df['Date Time'])
    
        # format the datetime objects to "yyyy-mm-dd hh:mm:ss" format
        df['date_time'] = df['date_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['date_time'] = pd.to_datetime(df['date_time'])
    
        # df=df.set_index(['date_time']).asfreq('1min')
        # df.v = df.v.fillna(0)
        # df[['c']] = df[['c']].ffill()
        # df['h'].fillna(df['c'], inplace=True)
        # df['l'].fillna(df['c'], inplace=True)
        # df['o'].fillna(df['c'], inplace=True)
        # df=df.between_time('04:00', '20:00')
        # df = df.reset_index(level=0)
    
        df['time'] = pd.to_datetime(df['date_time']).dt.time
        df['date'] = pd.to_datetime(df['date_time']).dt.date
    
        # daily_v_sum = df.groupby(df['date_time'].dt.date)['v'].sum()
        # valid_dates = daily_v_sum[daily_v_sum > 0].index
        # df = df[df['date_time'].dt.date.isin(valid_dates)]
        # # df = df.reset_index(level=0)
        # df = df.reset_index(drop=True)
    
        
    
        df['time_int'] = df['date_time'].dt.hour * 100 + df['date_time'].dt.minute
        df['date_int'] = df['date_time'].dt.strftime('%Y%m%d').astype(int)
    
        
        
        df['date_time_int'] = df['date_int'].astype(str) + '.' + df['time_int'].astype(str)
        df['date_time_int'] = df['date_time_int'].astype(float)
    
        
        df['v_sum'] = df.groupby('date')['v'].cumsum()
        
        df['hod_all'] = df.groupby(df['date'])['h'].cummax().fillna(0)
    
        
    
        return df
    
    def check_high_lvl_filter_lc(df):
        # df = df1.iloc[-1]
        # df = df1.tail(1)
    
        df['lc_frontside_d3_extended_1'] = ((df['h'] >= df['h1']) & 
                            (df['h1'] >= df['h2']) & 
    
                            (df['high_chg_atr1'] >= 0.5) & 
                            (df['gap_atr1'] >= 0.2) & 
                            (df['close_range1'] >= 0.6) &                  
                            (df['c1'] >= df['o1']) & 
                            (df['dist_h_9ema_atr1'] >= 1.5) & 
                            (df['dist_h_20ema_atr1'] >= 3) & 
    
                            (df['high_chg_atr'] >= 0.7) & 
                            (df['gap_atr'] >= 0.2) & 
                            (df['close_range'] >= 0.6) & 
                            (df['dist_h_9ema_atr'] >= 2) & 
                            (df['dist_h_50ema_atr'] >= 4) & 
                            (df['v_ua'] >= 10000000) & 
                            (df['dol_v'] >= 500000000) & 
                            (df['c_ua'] >= 20) & 
    
                            ((df['h'] >= df['highest_high_250']) &
                            (df['ema9'] >= df['ema20']) & 
                            (df['ema20'] >= df['ema50']) & 
                            (df['ema50'] >= df['ema200']))
                            
                            ).astype(int)
        df['lc_backside_d3_extended_1'] = ((df['h'] >= df['h1']) & 
                            (df['h1'] >= df['h2']) & 
    
                            (df['high_chg_atr1'] >= 0.5) & 
                            (df['gap_atr1'] >= 0.2) & 
                            (df['close_range1'] >= 0.6) &                  
                            (df['c1'] >= df['o1']) & 
                            (df['dist_h_9ema_atr1'] >= 1.5) & 
                            (df['dist_h_20ema_atr1'] >= 3) & 
    
                            (df['high_chg_atr'] >= 0.7) & 
                            (df['gap_atr'] >= 0.2) & 
                            (df['close_range'] >= 0.6) & 
                            (df['dist_h_9ema_atr'] >= 2) & 
                            (df['dist_h_50ema_atr'] >= 4) & 
                            (df['v_ua'] >= 10000000) & 
                            (df['dol_v'] >= 500000000) & 
                            (df['c_ua'] >= 20) & 
    
                            ((df['ema9'] < df['ema20']) | 
                            (df['ema20'] < df['ema50']) | 
                            (df['ema50'] < df['ema200']) |
                            (df['h'] < df['highest_high_250']))
                                                    
                            ).astype(int)
        
        df['lc_frontside_d3_extended_2'] = ((df['h'] >= df['h1']) & 
                            (df['h1'] >= df['h2']) & 
    
                            (df['high_chg_atr1'] >= 0.3) & 
                            (df['close_range1'] >= 0.5) &                  
                            (df['c1'] >= df['o1']) & 
                            (df['dist_h_9ema_atr1'] >= 1) &                     
                            (df['high_chg1'] >= df['high_chg']*0.4) & 
    
                            (df['high_chg_atr'] >= 0.6) & 
                            (df['gap_atr'] >= 0.2) & 
                            (df['close_range'] >= 0.4) & 
                            (df['dist_h_9ema_atr'] >= 1.5) & 
                            (df['dist_h_20ema_atr'] >= 3) & 
                            (df['v_ua'] >= 10000000) & 
                            (df['dol_v'] >= 500000000) & 
                            (df['c_ua'] >= 20) & 
    
                            ((df['h'] >= df['highest_high_250']) &
                            (df['ema9'] >= df['ema20']) & 
                            (df['ema20'] >= df['ema50']) & 
                            (df['ema50'] >= df['ema200']))
                            
                            ).astype(int)  
        df['lc_backside_d3_extended_2'] = ((df['h'] >= df['h1']) & 
                            (df['h1'] >= df['h2']) & 
    
                            (df['high_chg_atr1'] >= 0.3) & 
                            (df['close_range1'] >= 0.5) &                  
                            (df['c1'] >= df['o1']) & 
                            (df['dist_h_9ema_atr1'] >= 1) &                     
                            (df['high_chg1'] >= df['high_chg']*0.4) & 
    
                            (df['high_chg_atr'] >= 0.6) & 
                            (df['gap_atr'] >= 0.2) & 
                            (df['close_range'] >= 0.4) & 
                            (df['dist_h_9ema_atr'] >= 1.5) & 
                            (df['dist_h_20ema_atr'] >= 3) & 
                            (df['v_ua'] >= 10000000) & 
                            (df['dol_v'] >= 500000000) & 
                            (df['c_ua'] >= 20) & 
    
                            ((df['ema9'] < df['ema20']) | 
                            (df['ema20'] < df['ema50']) | 
                            (df['ema50'] < df['ema200']) |
                            (df['h'] < df['highest_high_250']))
                            
                            ).astype(int)
        
        df['lc_frontside_d4_para'] = ((df['h'] >= df['h1']) & 
                            (df['h1'] >= df['h2']) & 
                            
                            (df['dist_h_9ema_atr2'] >= 1) & 
                            (df['dist_h_9ema_atr3'] >= 1) & 
                            (df['dist_h_9ema_atr4'] >= 1) & 
                            
                            (df['c2'] >= df['o2']) & 
                            (df['dist_h_20ema_atr2'] >= 1.5) & 
    
                            (df['high_chg_atr1'] >= 0.3) & 
                            (df['gap_atr1'] >= 0) & 
                            (df['close_range1'] >= 0.3) &                  
                            (df['c1'] >= df['o1']) & 
                            (df['dist_h_9ema_atr1'] >= 1.5) & 
                            (df['dist_h_20ema_atr1'] >= 3) & 
    
                            (df['high_chg_atr'] >= 0.5) & 
                            (df['gap_atr'] >= 0.25) & 
                            (df['close_range'] >= 0.3) &                   
                            (df['c'] >= df['o']) & 
                            (df['dist_h_9ema_atr'] >= 2) & 
                            (df['dist_h_50ema_atr'] >= 4) & 
                            (df['v_ua'] >= 10000000) & 
                            (df['dol_v'] >= 500000000) & 
                            (df['c_ua'] >= 20) & 
    
                            ((df['h'] >= df['highest_high_250']) &
                            (df['ema9'] >= df['ema20']) & 
                            (df['ema20'] >= df['ema50']) & 
                            (df['ema50'] >= df['ema200']))
                            
                            ).astype(int)       
        df['lc_backside_d4_para'] = ((df['h'] >= df['h1']) & 
                            (df['h1'] >= df['h2']) & 
                            
                            (df['dist_h_9ema_atr2'] >= 1) & 
                            (df['dist_h_9ema_atr3'] >= 1) & 
                            (df['dist_h_9ema_atr4'] >= 1) & 
                            
                            (df['c2'] >= df['o2']) & 
                            (df['dist_h_20ema_atr2'] >= 1.5) & 
    
                            (df['high_chg_atr1'] >= 0.3) & 
                            (df['gap_atr1'] >= 0) & 
                            (df['close_range1'] >= 0.3) &                  
                            (df['c1'] >= df['o1']) & 
                            (df['dist_h_9ema_atr1'] >= 1.5) & 
                            (df['dist_h_20ema_atr1'] >= 3) & 
    
                            (df['high_chg_atr'] >= 0.5) & 
                            (df['gap_atr'] >= 0.25) & 
                            (df['close_range'] >= 0.3) &                   
                            (df['c'] >= df['o']) & 
                            (df['dist_h_9ema_atr'] >= 2) & 
                            (df['dist_h_50ema_atr'] >= 4) & 
                            (df['v_ua'] >= 10000000) & 
                            (df['dol_v'] >= 500000000) & 
                            (df['c_ua'] >= 20) & 
    
                            ((df['ema9'] < df['ema20']) | 
                            (df['ema20'] < df['ema50']) | 
                            (df['ema50'] < df['ema200']) |
                            (df['h'] < df['highest_high_250']))
                            
                            ).astype(int)
    
        df['lc_frontside_d3_uptrend'] = ((df['h'] >= df['h1']) & 
                            (df['h1'] >= df['h2']) & 
    
                            (df['high_chg_atr1'] >= 0.3) & 
                            (df['close_range1'] >= 0.5) &                  
                            (df['c1'] >= df['o1']) & 
                            (df['dist_h_9ema_atr1'] >= 2) & 
                            (df['dist_h_20ema_atr1'] >= 3) & 
    
                            (df['high_chg_atr'] >= 0.5) & 
                            (df['gap_atr'] >= -0.2) & 
                            (df['close_range'] >= 0.5) & 
                            (df['dist_h_9ema_atr'] >= 2) & 
                            (df['dist_h_20ema_atr'] >= 3) & 
                            (df['dist_h_200ema_atr'] >= 7) & 
                            (df['v_ua'] >= 10000000) & 
                            (df['dol_v'] >= 500000000) & 
                            (df['c_ua'] >= 20) & 
    
                            ((df['h'] >= df['highest_high_250']) &
                            (df['ema9'] >= df['ema20']) & 
                            (df['ema20'] >= df['ema50']) & 
                            (df['ema50'] >= df['ema200']))
                            
                            ).astype(int)         
        df['lc_backside_d3'] = ((df['h'] >= df['h1']) & 
                                (df['h1'] >= df['h2']) & 
    
                                (df['high_chg_atr1'] >= 1) & 
                                (df['close_range1'] >= 0.5) &                  
                                (df['c1'] >= df['o1']) & 
                                (df['dist_h_9ema_atr1'] >= 1) & 
                                (df['dist_h_20ema_atr1'] >= 2) & 
    
                                (df['high_chg_atr'] >= 1) & 
                                (df['close_range'] >= 0.5) & 
                                (df['dist_h_9ema_atr'] >= 1) & 
                                (df['dist_h_20ema_atr'] >= 2) & 
                                (df['v_ua'] >= 10000000) & 
                                (df['dol_v'] >= 500000000) & 
                                (df['c_ua'] >= 20) & 
    
                                ((df['ema9'] < df['ema20']) | 
                                (df['ema20'] < df['ema50']) | 
                                (df['ema50'] < df['ema200']) |
                                (df['h'] < df['highest_high_250']))
                                        
                            ).astype(int)
    
        
        df['lc_frontside_d2_uptrend'] = ((df['high_chg_atr'] >= 0.75) & 
                                (df['close_range'] >= 0.7) &           
                                (df['c'] >= df['o']) & 
                                (df['dist_h_9ema_atr'] >= 1.5) & 
                                (df['dist_h_20ema_atr'] >= 3) & 
                                (df['v_ua'] >= 10000000) & 
                                (df['dol_v'] >= 500000000) & 
                                (df['c_ua'] >= 2000000000) & 
                                (df['h_dist_to_lowest_low_20_atr'] >= 5) & 
                                (df['h'] >= df['highest_high_20']) &
    
                                ((df['h'] >= df['highest_high_250']) &
                                (df['ema9'] >= df['ema20']) & 
                                (df['ema20'] >= df['ema50']) & 
                                (df['ema50'] >= df['ema200']))
                                
                                ).astype(int)        
        df['lc_frontside_d2'] = ((df['high_chg_atr'] >= 1.5) & 
                                (df['close_range'] >= 0.5) &                   
                                (df['c'] >= df['o']) & 
                                (df['v_ua'] >= 10000000) & 
                                (df['dol_v'] >= 500000000) & 
                                (df['c_ua'] >= 20) & 
    
                                ((df['h'] >= df['highest_high_250']) &
                                (df['ema9'] >= df['ema20']) & 
                                (df['ema20'] >= df['ema50']) & 
                                (df['ema50'] >= df['ema200']))
    
                                ).astype(int)
        df['lc_backside_d2'] = ((df['high_chg_atr'] >= 1.5) & 
                                (df['close_range'] >= 0.5) &           
                                (df['c'] >= df['o']) & 
                                (df['v_ua'] >= 10000000) & 
                                (df['dol_v'] >= 500000000) & 
                                (df['c_ua'] >= 20) & 
                                (df['h'] >= df['highest_high_5']) &
    
                                ((df['ema9'] < df['ema20']) | 
                                (df['ema20'] < df['ema50']) | 
                                (df['ema50'] < df['ema200']) |
                                (df['h'] < df['highest_high_250']))
                                
                                ).astype(int)
        
        
        df['lc_fbo'] = (((df['high_chg_atr'] >= 0.5) | (df['high_chg_from_pdc_atr'] >= 0.5)) & 
                        (df['h'] >= df['h1']) & 
                        (df['close_range'] >= 0.3) &           
                        (df['c'] >= df['o']) & 
                        (df['v_ua'] >= 10000000) & 
                        (df['dol_v'] >= 500000000) & 
                        (df['c_ua'] >= 2000000000) & 
                        ((df['h_dist_to_lowest_low_20_atr'] >= 4) | (df['h_dist_to_lowest_low_5_atr'] >= 2)) & 
                        (df['h'] >= df['highest_high_50_4'] - df['atr']*1) & 
                        (df['h'] <= df['highest_high_50_4'] + df['atr']*1) & 
                        
                        (df['highest_high_50_4'] >= df['highest_high_100_4']) & 
    
                        (df['h1'] < df['highest_high_50_4']) & 
                        (df['h2'] < df['highest_high_50_4']) & 
                        (df['h3'] < df['highest_high_50_4']) & 
    
                        (df['ema9'] >= df['ema20']) & 
                        (df['ema20'] >= df['ema50']) & 
                        (df['ema50'] >= df['ema200'])
                        
                        ).astype(int)
    
       
    
        columns_to_check = ['lc_frontside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_frontside_d3_extended_2', 'lc_backside_d3_extended_2', 'lc_frontside_d4_para', 'lc_backside_d4_para',
         'lc_frontside_d3_uptrend', 'lc_backside_d3', 'lc_frontside_d2_uptrend', 'lc_frontside_d2', 'lc_backside_d2', 'lc_fbo']
    
        df2 = df[df[columns_to_check].any(axis=1)]
        return df2 
    
    async def fetch_intraday_data(session, ticker, start_date, end_date):
        url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}'
        params = {'apiKey': API_KEY}
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return pd.DataFrame(data['results'])
            else:
                print(f"Failed to fetch data for {ticker}: {response.status}")
                return pd.DataFrame()
            
    
    def get_min_price_lc(df):
        ### LC Min Price
        df['lc_frontside_d3_extended_1_min_price'] = round((df['c'] + df['atr']*.25), 2)
        df['lc_backside_d3_extended_1_min_price'] = round(np.maximum(
                df['c'] + df['atr']*1, df['h'] + df['atr']*0.75
            ), 2)
    
        df['lc_frontside_d3_extended_2_min_price'] = round(np.maximum(
                        df['c'] + df['atr']*0.75,
                        np.maximum(df['h'], df['ema50'] + df['atr']*5)
                        ), 2)   
        df['lc_backside_d3_extended_2_min_price'] = round(np.maximum(
                        df['c'] + df['atr']*1,
                        np.maximum(df['h'] + df['atr']*0.75, df['ema50'] + df['atr']*5)
                        ), 2)   
        
        
        df['lc_frontside_d4_para_min_price'] = round((df['c'] + df['atr']*0.25), 2)
        # df['lc_backside_d4_para_min_price'] = round((df['c'] + df['atr']*0.25), 2)
        df['lc_backside_d4_para_min_price'] = round(np.maximum(
                df['c'] + df['atr']*1, df['h'] + df['atr']*0.75
            ), 2)
    
        df['lc_frontside_d3_uptrend_min_price'] = round((df['c'] + df['atr']*0.2), 2)
        df['lc_backside_d3_min_price'] = round(np.maximum(
                df['c'] + df['atr']*1, df['h'] + df['atr']*0.75
            ), 2)
        
        df['lc_frontside_d2_uptrend_min_price'] = round(np.maximum(
                df['c'] + df['atr']*0.5, df['h']
            ), 2)
        
        df['lc_frontside_d2_min_price'] = round((df['c'] + df['atr']*2), 2)
        df['lc_backside_d2_min_price'] = round((df['c'] + df['atr']*2), 2)
        # df['lc_backside_d2_min_price'] = round(np.maximum(
        #         df['c'] + df['atr']*1, df['h'] + df['atr']*0.75
        #     ), 2)
    
        df['lc_fbo_min_price'] = round(np.maximum(
                df['c'] + df['atr']*0.5, df['h']
            ), 2)
            
    
        df['lc_frontside_d3_extended_1_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
        df['lc_backside_d3_extended_1_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
        df['lc_frontside_d3_extended_2_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
        df['lc_backside_d3_extended_2_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
        df['lc_frontside_d4_para_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
        df['lc_backside_d4_para_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
        df['lc_frontside_d3_uptrend_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
        df['lc_backside_d3_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
        df['lc_frontside_d2_uptrend_min_price'] = round((df['c'] + df['d1_range']*.5), 2)
        df['lc_frontside_d2_min_price'] = round((df['c'] + df['d1_range']*.5), 2)
        df['lc_frontside_d2_min_price'] = round((df['c'] + df['d1_range']*.5), 2)
        df['lc_backside_d2_min_price'] = round((df['c'] + df['d1_range']*.5), 2)
    
        
    
        columns_to_check = ['lc_frontside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_frontside_d3_extended_2', 'lc_backside_d3_extended_2', 'lc_frontside_d4_para', 'lc_backside_d4_para',
         'lc_frontside_d3_uptrend', 'lc_backside_d3', 'lc_frontside_d2_uptrend', 'lc_frontside_d2', 'lc_backside_d2', 'lc_fbo']
    
        df['lowest_min_price'] = df.apply(lambda row: min([row[col + '_min_price'] for col in columns_to_check if row[col] == 1]), axis=1)
    
        for col in columns_to_check:
            min_price_col = f"{col}_min_price"
            min_pct_col = f"{col}_min_pct"
            df[min_pct_col] = round((df[min_price_col] / df['c'] - 1) * 100, 2)
    
        return df
    
    async def fetch_intial_stock_list(session, date, adj):
        url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}?adjusted={adj}&apiKey={API_KEY}"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if 'results' in data:
                    df = pd.DataFrame(data['results'])
                    # df = df[(df['v_ua'] >= 2000000) & (df['c'] >= df['o']) & (df['c'] * df['v'] >= 20000000) & (((df['c'] * df['l']) / (df['h'] * df['l'])) >= 0.3)]
                    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
                    df.rename(columns={'T': 'ticker'}, inplace=True)
                    # print(url)
                    return df
    
    def process_stock_data_grouped(df):
        # Group by 'ticker' and apply the calculations to each group
        df_grouped = df.groupby('ticker').apply(lambda group: compute_indicators(group))
        # Reset index to undo the multi-index caused by groupby
        df_grouped.reset_index(drop=True, inplace=True)
        return df_grouped
    def compute_indicators(group):
        # Make sure to sort by date to respect chronological order
        group = group.sort_values(by='date')
    
        # Calculating various financial metrics and indicators
        group['pdc'] = group['c'].shift(1)
        group['high_low'] = group['h'] - group['l']
        group['high_pdc'] = abs(group['h'] - group['pdc'])
        group['low_pdc'] = abs(group['l'] - group['pdc'])
        group['true_range'] = group[['high_low', 'high_pdc', 'low_pdc']].max(axis=1)
        group['atr'] = group['true_range'].rolling(window=14).mean()
    
        
        group['d1_range'] = abs(group['h'] - group['l'])
        
        # Shifting high, close, open, low, and volume
        for i in range(1, 4):  # Example for up to 3 days shift
            group[f'h{i}'] = group['h'].shift(i).fillna(0)
            if i <= 2:  # Up to 2 days shift for close, open, low, volume
                group[f'c{i}'] = group['c'].shift(i).fillna(0)
                group[f'o{i}'] = group['o'].shift(i).fillna(0)
                group[f'l{i}'] = group['l'].shift(i).fillna(0)
                group[f'v{i}'] = group['v'].shift(i).fillna(0)
    
        # Dollar volume and other calculations
        # group['dol_v'] = group['c'] * group['v']
        group['dol_v1'] = group['dol_v'].shift(1)
        group['dol_v2'] = group['dol_v'].shift(2)
        
        # group['close_range'] = (group['c'] - group['l']) / (group['h'] - group['l'])
        group['close_range1'] = group['close_range'].shift(1)
        
        # Gap metrics related to ATR
        group['gap_atr'] = ((group['o'] - group['pdc']) / group['atr'])
        group['gap_atr1'] = ((group['o1'] - group['c2']) / group['atr'])
        group['gap_pdh_atr'] = ((group['o'] - group['h1']) / group['atr'])
    
        # High change metrics normalized by ATR
        group['high_chg'] = (group['h'] - group['o'])
        group['high_chg_atr'] = (group['high_chg'] / group['atr'])
        group['high_chg_atr1'] = ((group['h1'] - group['o1']) / group['atr'])
    
        # High change from previous day close normalized by ATR
        group['high_chg_from_pdc_atr'] = ((group['h'] - group['c1']) / group['atr'])
        group['high_chg_from_pdc_atr1'] = ((group['h1'] - group['c2']) / group['atr'])
    
        # Percentage change in close price from the previous day
        group['pct_change'] = ((group['c'] / group['c1'] - 1) * 100).round(2)
    
        # Calculate EMAs and fill NAs.
        for period in [9, 20, 50, 200]:
            group[f'ema{period}'] = group['c'].ewm(span=period, adjust=False).mean().fillna(0)
            
            # Calculate distances from EMAs for the high price.
            group[f'dist_h_{period}ema'] = group['h'] - group[f'ema{period}']
            
            # Normalize these distances by the ATR.
            group[f'dist_h_{period}ema_atr'] = group[f'dist_h_{period}ema'] / group['atr']
            
            # Apply shifts to the calculated distances and normalize again by ATR.
            for dist in range(1, 5):
                group[f'dist_h_{period}ema{dist}'] = group[f'dist_h_{period}ema'].shift(dist)
                group[f'dist_h_{period}ema_atr{dist}'] = group[f'dist_h_{period}ema{dist}'] / group['atr']
    
        # Rolling maximums and minimums
        for window in [5, 20, 50, 100, 250]:
            group[f'lowest_low_{window}'] = group['l'].rolling(window=window, min_periods=1).min()
            group[f'highest_high_{window}'] = group['h'].rolling(window=window, min_periods=1).max()
            if window in [20, 50, 100, 250]:  # Shifting previous highs for selected windows
                for dist in range(1, 5):
                    if window == 20 or window == 50:
                        group[f'highest_high_{window}_{dist}'] = group[f'highest_high_{window}'].shift(dist)
        
        
        group['lowest_low_30'] = group['l'].rolling(window=30, min_periods=1).min()
        group['lowest_low_30_1'] = group['lowest_low_30'].shift(1)
    
        group['highest_high_100_1'] = group['highest_high_100'].shift(1)
        group['highest_high_100_4'] = group['highest_high_100'].shift(4)
        group['highest_high_250_1'] = group['highest_high_250'].shift(1)
        group['lowest_low_20_2'] = group['lowest_low_20'].shift(2)
    
        group['lowest_low_20_ua'] = group['l_ua'].rolling(window=20, min_periods=1).min()
    
        group['h_dist_to_lowest_low_30'] = (group['h'] - group['lowest_low_30']) / group['atr']
    
        group['h_dist_to_lowest_low_20_atr'] = (group['h'] - group['lowest_low_20']) / group['atr']
        group['h_dist_to_lowest_low_5_atr'] = (group['h'] - group['lowest_low_5']) / group['atr']
        
        group['ema20_2'] = group['ema20'].shift(2)
    
        # Drop intermediate columns
        columns_to_drop = ['high_low', 'high_pdc', 'low_pdc']
        group.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        return group
    
    def compute_indicators1(df):
        # Sorting by 'ticker' and 'date' to respect chronological order for each ticker
        df = df.sort_values(by=['ticker', 'date'])
        
        # Calculating previous day's close
        df['pdc'] = df.groupby('ticker')['c'].shift(1)
    
        # Calculating ranges and true range
        df['high_low'] = df['h'] - df['l']
        df['high_pdc'] = (df['h'] - df['pdc']).abs()
        df['low_pdc'] = (df['l'] - df['pdc']).abs()
        df['true_range'] = df[['high_low', 'high_pdc', 'low_pdc']].max(axis=1)
        df['atr'] = df.groupby('ticker')['true_range'].transform(lambda x: x.rolling(window=14).mean())
    
        
        df['d1_range'] = abs(df['h'] - df['l'])
    
        # Shifting values for high, close, open, low, volume
        for i in range(1, 4):
            df[f'h{i}'] = df.groupby('ticker')['h'].shift(i).fillna(0)
            if i <= 2:  # Limiting to 2 days shift for close, open, low, volume
                df[f'c{i}'] = df.groupby('ticker')['c'].shift(i).fillna(0)
                df[f'o{i}'] = df.groupby('ticker')['o'].shift(i).fillna(0)
                df[f'l{i}'] = df.groupby('ticker')['l'].shift(i).fillna(0)
                df[f'v{i}'] = df.groupby('ticker')['v'].shift(i).fillna(0)
    
        # Dollar volume calculations and shifts
        df['dol_v'] = df['c'] * df['v']
        df['dol_v1'] = df.groupby('ticker')['dol_v'].shift(1)
        df['dol_v2'] = df.groupby('ticker')['dol_v'].shift(2)
    
        # Close range calculations and shifts
        df['close_range'] = (df['c'] - df['l']) / (df['h'] - df['o'])
        df['close_range1'] = df.groupby('ticker')['close_range'].shift(1)
    
        # Gap metrics related to ATR
        df['gap_atr'] = ((df['o'] - df['pdc']) / df['atr'])
        df['gap_atr1'] = ((df['o1'] - df['c2']) / df['atr'])
        df['gap_pdh_atr'] = ((df['o'] - df['h1']) / df['atr'])
    
        # High change metrics normalized by ATR
        df['high_chg'] = df['h'] - df['o']
        df['high_chg1'] = df['h1'] - df['o1']
        df['high_chg_atr'] = df['high_chg'] / df['atr']
        df['high_chg_atr1'] = ((df['h1'] - df['o1']) / df['atr'])
    
        # High change from previous day close normalized by ATR
        df['high_chg_from_pdc_atr'] = ((df['h'] - df['c1']) / df['atr'])
        df['high_chg_from_pdc_atr1'] = ((df['h1'] - df['c2']) / df['atr'])
    
        # Percentage change in close price from the previous day
        df['pct_change'] = ((df['c'] / df['c1'] - 1) * 100).round(2)
    
        # Calculating EMAs
        for period in [9, 20, 50, 200]:
            df[f'ema{period}'] = df.groupby('ticker')['c'].transform(lambda x: x.ewm(span=period, adjust=False).mean().fillna(0))
            df[f'dist_h_{period}ema'] = df['h'] - df[f'ema{period}']
            df[f'dist_h_{period}ema_atr'] = df[f'dist_h_{period}ema'] / df['atr']
    
            # Apply shifts to the calculated distances and normalize again by ATR
            for dist in range(1, 5):
                df[f'dist_h_{period}ema{dist}'] = df.groupby('ticker')[f'dist_h_{period}ema'].shift(dist)
                df[f'dist_h_{period}ema_atr{dist}'] = df[f'dist_h_{period}ema{dist}'] / df['atr']
    
        # Calculate rolling maximums and minimums
        for window in [5, 20, 50, 100, 250]:
            df[f'lowest_low_{window}'] = df.groupby('ticker')['l'].transform(lambda x: x.rolling(window=window, min_periods=1).min())
            df[f'highest_high_{window}'] = df.groupby('ticker')['h'].transform(lambda x: x.rolling(window=window, min_periods=1).max())
    
            # Shifting previous highs for selected windows
            for dist in range(1, 5):
                df[f'highest_high_{window}_{dist}'] = df.groupby('ticker')[f'highest_high_{window}'].shift(dist)
    
        # Calculate rolling minimums for the low prices with shifts
        df['lowest_low_30'] = df.groupby('ticker')['l'].transform(lambda x: x.rolling(window=30, min_periods=1).min())
        df['lowest_low_30_1'] = df.groupby('ticker')['lowest_low_30'].shift(1)
    
        # Calculate rolling maximums for high prices with multiple shifts
        df['highest_high_100_1'] = df.groupby('ticker')['highest_high_100'].shift(1)
        df['highest_high_100_4'] = df.groupby('ticker')['highest_high_100'].shift(4)
        df['highest_high_250_1'] = df.groupby('ticker')['highest_high_250'].shift(1)
        df['lowest_low_20_2'] = df.groupby('ticker')['lowest_low_20'].shift(2)
    
        # Assuming l_ua is a predefined column in your DataFrame
        df['lowest_low_20_ua'] = df.groupby('ticker')['l_ua'].transform(lambda x: x.rolling(window=20, min_periods=1).min())
    
        # Calculate distances from the lowest lows normalized by ATR
        df['h_dist_to_lowest_low_30'] = (df['h'] - df['lowest_low_30'])
        df['h_dist_to_lowest_low_30_atr'] = (df['h'] - df['lowest_low_30']) / df['atr']
        df['h_dist_to_lowest_low_20_atr'] = (df['h'] - df['lowest_low_20']) / df['atr']
        df['h_dist_to_lowest_low_5_atr'] = (df['h'] - df['lowest_low_5']) / df['atr']
    
        # Shifting EMAs
        df['ema20_2'] = df.groupby('ticker')['ema20'].shift(2)
    
        df['v_ua1'] = df.groupby('ticker')['v_ua'].shift(1)
        df['v_ua2'] = df.groupby('ticker')['v_ua'].shift(2)
    
        # Drop intermediate columns
        columns_to_drop = ['high_low', 'high_pdc', 'low_pdc']
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
        return df
    
    def calculate_trading_days(date):
        # Define a range of trading days (30 days before and after the given date)
        start_date = date - pd.Timedelta(days=30)
        end_date = date + pd.Timedelta(days=30)
    
        # Get the trading schedule for the range
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        trading_days = schedule.index
    
        # Ensure the date exists in the trading days
        if date not in trading_days:
            return pd.NaT, pd.NaT  # Return NaT if the date is not a trading day
    
        # Find the location of the given date in the trading days
        idx = trading_days.get_loc(date)
    
        # Calculate the next trading day and the fourth previous trading day
        date_plus_1 = trading_days[idx + 1] if idx + 1 < len(trading_days) else pd.NaT
        date_minus_4 = trading_days[idx - 4] if idx - 4 >= 0 else pd.NaT
    
        return date_plus_1, date_minus_4
    def get_offsets(date):
        if date not in trading_days_map:
            return pd.NaT, pd.NaT
        idx = trading_days_map[date]
        date_plus_1 = trading_days_list[idx + 1] if idx + 1 < len(trading_days_list) else pd.NaT
        date_minus_4 = trading_days_list[idx - 4] if idx - 4 >= 0 else pd.NaT
        date_minus_30 = trading_days_list[idx - 30] if idx - 30 >= 0 else pd.NaT
        return date_plus_1, date_minus_4, date_minus_30
    
    def fetch_intraday_data(ticker, start_date, end_date):
        url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/30/minute/{start_date}/{end_date}?adjusted=true'
        params = {'apiKey': API_KEY}
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    return pd.DataFrame(data['results'])
                else:
                    print(f"No results for {ticker}")
            else:
                print(f"Error fetching data for {ticker}: {response.status_code}")
        except Exception as e:
            print(f"Exception for {ticker}: {e}")
        return pd.DataFrame()
    
    def process_lc_row(row):
        """Process a single row for intraday data and calculations."""
        try:
            ticker = row['ticker']
            date = row['date']
    
            date_minus_4 = row['date_minus_4']
            date_plus_1 = row['date_plus_1']
    
            # Fetch and adjust intraday data
            intraday_data = fetch_intraday_data(ticker, date_minus_4, date_plus_1)
            intraday_data = adjust_intraday(intraday_data)
    
            date_plus_1_formatted = datetime.datetime.strptime(date_plus_1, '%Y-%m-%d').date()
    
            intraday_data_before = intraday_data[intraday_data['date'] != date_plus_1_formatted]
            
            resp_v = intraday_data_before[intraday_data_before['time_int'] == 900].set_index('date')['v_sum']
            resp_o = intraday_data_before[intraday_data_before['time_int'] == 930].set_index('date')['o']
            resp_v, resp_o = resp_v.align(resp_o, fill_value=0)
            pm_dol_vol_all = resp_o * resp_v
            avg_pm_dol_vol = pm_dol_vol_all.mean()
    
            # Add results to row
            row['avg_5d_pm_dol_vol'] = avg_pm_dol_vol
            if avg_pm_dol_vol >= 10000000:
                row['valid_pm_liq'] = 1
            else:
                row['valid_pm_liq'] = 0
            # row['valid_pm_liq'] = 1 if avg_pm_dol_vol >= 10000000 else 0
    
            
            intraday_data_after = intraday_data[intraday_data['date'] == date_plus_1_formatted]      
                        
    
            pm_df = intraday_data_after[intraday_data_after['time_int'] <= 900].set_index('time')
            open_df = intraday_data_after[intraday_data_after['time_int'] <= 930].set_index('time')
    
            if not pm_df.empty:
                resp_v = pm_df['v_sum'][-1]
                resp_o = open_df['o'][-1]
                pm_dol_vol_next_day = resp_o * resp_v
                pmh_next_day = pm_df['hod_all'][-1]
            else:
                pm_dol_vol_next_day = 0
                pmh_next_day = 0
                resp_o = 0
            
            row['pmh_next_day'] = pmh_next_day
            row['open_next_day'] = resp_o
            row['pm_dol_vol_next_day'] = pm_dol_vol_next_day
    
        except Exception as e:
            print(f"Error processing LC Row {row['ticker']} on {row['date']}: {e}")
            row['avg_5d_pm_dol_vol'] = None
            row['valid_pm_liq'] = None
            row['pmh_next_day'] = 0
            row['open_next_day'] = 0
            row['pm_dol_vol_next_day'] = 0
    
        return row
    
    def dates_before_after(df):
        global trading_days_map, trading_days_list
    
        start_date = df['date'].min() - pd.Timedelta(days=60)
        end_date = df['date'].max() + pd.Timedelta(days=30)
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        trading_days = pd.Series(schedule.index, index=schedule.index)
    
        # Map trading days for faster lookup
        trading_days_list = trading_days.index.to_list()
        trading_days_map = {day: idx for idx, day in enumerate(trading_days_list)}
    
        results = df_lc['date'].map(get_offsets)
        df_lc[['date_plus_1', 'date_minus_4', 'date_minus_30']] = pd.DataFrame(results.tolist(), index=df_lc.index)
    
        # Format dates as strings if needed
        df_lc['date_plus_1'] = df_lc['date_plus_1'].dt.strftime('%Y-%m-%d')
        df_lc['date_minus_4'] = df_lc['date_minus_4'].dt.strftime('%Y-%m-%d')
        df_lc['date_minus_30'] = df_lc['date_minus_30'].dt.strftime('%Y-%m-%d')
    
        return df
    
    def check_next_day_valid_lc(df):
        # Ensure minimum price columns exist for each check column
    
        columns_to_check = ['lc_backside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_frontside_d3_extended_2', 'lc_backside_d3_extended_2', 'lc_frontside_d4_para', 'lc_backside_d4_para',
         'lc_frontside_d3_uptrend', 'lc_backside_d3', 'lc_frontside_d2_uptrend', 'lc_frontside_d2', 'lc_backside_d2', 'lc_fbo']
        
        for col in columns_to_check:
            min_price_col = col + '_min_price'
            if col in df.columns and min_price_col in df.columns:
                # Vectorized condition checks
                condition = (df[col] == 1) & (df['open_next_day'] >= df[min_price_col]) & (df['pm_dol_vol_next_day'] >= 10000000)
                df[col] = condition.astype(int)  # Update the column based on conditions
    
        df = df[~(df[columns_to_check] == 0).all(axis=1)]
    
        return df
    
    def check_lc_pm_liquidity(df):
        df.loc[
            ((df['lc_frontside_d3_extended_1'] == 1) | (df['lc_backside_d3_extended_1'] == 1) | (df['lc_frontside_d3_extended_2'] == 1) | (df['lc_backside_d3_extended_2'] == 1) | (df['lc_frontside_d4_para'] == 1) | (df['lc_backside_d4_para'] == 1) | 
             (df['lc_frontside_d3_uptrend'] == 1) | (df['lc_backside_d3'] == 1) | (df['lc_frontside_d2_uptrend'] == 1) | (df['lc_fbo'] == 1)) & 
            (df['valid_pm_liq'] != 1),
            ['lc_frontside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_frontside_d3_extended_2', 'lc_backside_d3_extended_2', 'lc_frontside_d4_para', 'lc_backside_d4_para', 
            'lc_frontside_d3_uptrend', 'lc_backside_d3', 'lc_frontside_d2_uptrend', 'lc_fbo']
            ] = 0
        
        df.loc[
            ((df['lc_frontside_d2'] == 1) | (df['lc_backside_d2'] == 1)) & 
            (df['valid_pm_liq'] == 0),
            ['lc_frontside_d2', 'lc_backside_d2']
            ] = 0
        
    
        columns_to_check = ['lc_frontside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_frontside_d3_extended_2', 'lc_backside_d3_extended_2', 'lc_frontside_d4_para', 'lc_backside_d4_para',
         'lc_frontside_d3_uptrend', 'lc_backside_d3', 'lc_frontside_d2_uptrend', 'lc_frontside_d2', 'lc_backside_d2', 'lc_fbo']
        
        # Drop rows where all specified columns are 0
        df = df[~(df[columns_to_check] == 0).all(axis=1)]
        
        # df_lc = df_lc[df_lc['valid_pm_liq'] == 1]
    
        return df
    
    def process_dataframe(func, data):
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            processed_rows = list(executor.map(func, data))
        return pd.DataFrame(processed_rows)
    
    async def main():
        global df_lc, df_sc
        ### Get Main List
        all_results = []
        adj = "true"
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_intial_stock_list(session, date, adj) for date in DATES]
            # results = await asyncio.gather(*tasks)
            # all_results = [result for result in results if result is not None]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # all_results = [result for result in results if isinstance(result, pd.DataFrame)]
    
            all_results = []
            retry_tasks = []
    
            # Check results and prepare for retry if necessary
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # print(f"Retrying failed task for date: {DATES[i]} due to {result}")
                    retry_tasks.append(fetch_intial_stock_list(session, DATES[i], adj))
                else:
                    all_results.append(result)
    
            # Retry failed tasks
            if retry_tasks:
                retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
                # Merge retry results, assuming they are in the same order as retry_tasks
                for retry_result in retry_results:
                    if not isinstance(retry_result, Exception):
                        all_results.append(retry_result)
                    else:
                        print(f"Failed after retry: {retry_result}")
    
        df_a = pd.concat(all_results, ignore_index=True)
        all_results = []
        adj = "false"
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_intial_stock_list(session, date, adj) for date in DATES]
            # results = await asyncio.gather(*tasks)
            # all_results = [result for result in results if result is not None]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # all_results = [result for result in results if isinstance(result, pd.DataFrame)]
    
            
            all_results = []
            retry_tasks = []
    
            # Check results and prepare for retry if necessary
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # print(f"Retrying failed task for date: {DATES[i]} due to {result}")
                    retry_tasks.append(fetch_intial_stock_list(session, DATES[i], adj))
                else:
                    all_results.append(result)
    
            # Retry failed tasks
            if retry_tasks:
                retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
                # Merge retry results, assuming they are in the same order as retry_tasks
                for retry_result in retry_results:
                    if not isinstance(retry_result, Exception):
                        all_results.append(retry_result)
                    else:
                        print(f"Failed after retry: {retry_result}")
    
        df_ua = pd.concat(all_results, ignore_index=True)
        df_ua.rename(columns={col: col + '_ua' if col not in ['date', 'ticker'] else col for col in df_ua.columns}, inplace=True)
    
        
    
        print("done 1")
    
        df = pd.merge(df_a, df_ua, on=['date', 'ticker'], how='inner')
        
        df = df.drop(columns=['vw', 't', 'n', 'vw_ua', 't_ua', 'n_ua'])
        df = df.sort_values(by='date')
    
        df['date'] = pd.to_datetime(df['date'])
        df['close_range'] = (df['c'] - df['l']) / (df['h'] - df['l'])
        df['dol_v'] = (df['c'] * df['v']) 
        df['pre_conditions'] = (
            (df['c_ua'] >= 5) &
            (df['v_ua'] >= 10000000) &
            (df['dol_v'] >= 500000000) &
            (df['c'] > df['o']) &
            (df['close_range'] >= 0.3)
        )
    
        ticker_meet_conditions = df.groupby('ticker')['pre_conditions'].any()
        filtered_tickers = ticker_meet_conditions[ticker_meet_conditions].index
        df = df[df['ticker'].isin(filtered_tickers)]
    
        print(df)
        print("done 2")
    
        
    
        df = df.select_dtypes(include=['floating']).round(2).join(df.select_dtypes(exclude=['floating']))
    
        # df = process_stock_data_grouped(df)    
        df = compute_indicators1(df)
        df = df.sort_values(by='date')
        df = df[df['pre_conditions'] == True]
    
        
    
        df = df.select_dtypes(include=['floating']).round(2).join(df.select_dtypes(exclude=['floating']))
    
        # df = df[(df['date'] >= START_DATE_DT) & (df['date'] <= END_DATE_DT)]
        # df = df.reset_index(drop=True)
    
        print(df)
        print("done 3")
    
        df_lc = check_high_lvl_filter_lc(df)
    
        df = dates_before_after(df)
    
        print("done 4")
    
        
        # df = df[(df['date'] >= start_date_70_days_before) & (df['date'] <= END_DATE_DT)]
        # df = df.reset_index(drop=True)
    
        rows_lc = df_lc.to_dict(orient='records')
    
        # Use ProcessPoolExecutor to process both dataframes concurrently
        with ProcessPoolExecutor() as executor:
            future_lc = executor.submit(process_dataframe, process_lc_row, rows_lc)
    
            df_lc = future_lc.result()
    
        print("done 5")
    
        # Continue with further processing
        df_lc = get_min_price_lc(df_lc)
        df_lc = check_next_day_valid_lc(df_lc)
        df_lc = check_lc_pm_liquidity(df_lc)
    
        print("done 6")
        
    
        df_lc = df_lc[(df_lc['date'] >= START_DATE_DT) & (df_lc['date'] <= END_DATE_DT)]
        df_lc = df_lc.reset_index(drop=True)
    
        # Output the final dataframes
        print(df_lc)
        df_lc.to_csv("lc_backtest.csv")
    
            
    
            
    if __name__ == "__main__":
        START_DATE = '2025-05-01'  # Specify the start date
        END_DATE = '2025-06-05'  
    
        START_DATE_DT = pd.to_datetime(START_DATE)
        END_DATE_DT = pd.to_datetime(END_DATE)
    
        start_date_70_days_before = pd.Timestamp(START_DATE) - pd.DateOffset(days=70)
        start_date_70_days_before = pd.to_datetime(start_date_70_days_before)
    
        start_date_300_days_before = pd.Timestamp(START_DATE) - pd.DateOffset(days=400)
        start_date_300_days_before = str(start_date_300_days_before)[:10]
                                                                              
        schedule = nyse.schedule(start_date=start_date_300_days_before, end_date=END_DATE)
        DATES = nyse.valid_days(start_date=start_date_300_days_before, end_date=END_DATE)
        DATES = [date.strftime('%Y-%m-%d') for date in nyse.valid_days(start_date=start_date_300_days_before, end_date=END_DATE)]
    
        asyncio.run(main())
        # asyncio.run(main())
    
        print("Getting Stocks")
    
    ```
    
- analyze and debug scan
    
    When trying to find scans and looking at specific dates, it's really important to be able to debug and analyze those specific dates and see why certain parameters aren't hitting or are hitting.
    
    ```python
    # analyze_events.py  • full, working version
    import pandas as pd, numpy as np, requests
    from datetime import datetime, timedelta
    
    API_KEY  = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    BASE_URL = 'https://api.polygon.io'
    
    def fetch_aggregates(ticker, start_date, end_date):
        url  = f'{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}'
        r    = requests.get(url, params={'apiKey': API_KEY}); r.raise_for_status()
        rows = r.json().get('results', [])
        if not rows: return pd.DataFrame()
        df = (pd.DataFrame(rows)
                .assign(Date=lambda d: pd.to_datetime(d['t'], unit='ms'))
                .rename(columns={'o':'Open','h':'High','l':'Low','c':'Close','v':'Volume'})
                .set_index('Date')[['Open','High','Low','Close','Volume']])
        return df
    
    # ── metrics helpers (exactly as last message, unchanged) ───────────────────────
    def compute_emas(df, spans=(9,20,50)):
        for s in spans: df[f'EMA_{s}'] = df['Close'].ewm(span=s, adjust=False).mean()
        return df
    
    def compute_atr(df, period=30):
        hi_lo = df['High']-df['Low']
        hi_pc = (df['High']-df['Close'].shift()).abs()
        lo_pc = (df['Low'] -df['Close'].shift()).abs()
        df['TR']      = pd.concat([hi_lo,hi_pc,lo_pc],axis=1).max(axis=1)
        df['ATR_raw'] = df['TR'].rolling(period, min_periods=period).mean()
        df['ATR']     = df['ATR_raw'].shift()
        df['ATR_Pct_Change'] = df['ATR_raw'].pct_change().shift()*100
        return df
    
    def compute_volume(df, period=30):
        df['VOL_AVG_raw'] = df['Volume'].rolling(period,min_periods=period).mean()
        df['VOL_AVG']     = df['VOL_AVG_raw'].shift()
        df['Prev_Volume'] = df['Volume'].shift()
        return df
    
    def compute_slopes(df, span=9, wins=(3,5,15)):
        for w in wins:
            col=f'EMA_{span}'
            df[f'Slope_{span}_{w}d']=(df[col]-df[col].shift(w))/df[col].shift(w)*100
        return df
    
    def compute_slope_adj(df, span=9, window=50, lag=4):
        col=f'EMA_{span}'
        df[f'Slope_{span}_{window}d_adj'] = (
            df[col].shift(lag)-df[col].shift(window+lag)
        )/df[col].shift(window+lag)*100
        return df
    
    def compute_gap(df):
        df['Gap']=(df['Open']-df['Close'].shift()).abs()
        df['Gap_over_ATR']=df['Gap']/df['ATR']; return df
    
    def compute_div_ema_atr(df, spans=(9,20)):
        for s in spans: df[f'High_over_EMA{s}_div_ATR']=(df['High']-df[f'EMA_{s}'])/df['ATR']
        return df
    
    def compute_pct_low_moves(df):
        low7  = df['Low'].rolling(7 ,min_periods=7 ).min()
        low14 = df['Low'].rolling(14,min_periods=14).min()
        df['Pct_7d_low_div_ATR'] = ((df['Close']-low7 )/low7 )/df['ATR']*100
        df['Pct_14d_low_div_ATR']= ((df['Close']-low14)/low14)/df['ATR']*100
        return df
    
    def compute_range_pos(df):
        df['Upper_70_Range']=(df['Close']-df['Low'])/(df['High']-df['Low'])*100
        return df
    
    def compute_all_metrics(df):
        df=(df.pipe(compute_emas)
              .pipe(compute_atr)
              .pipe(compute_volume)
              .pipe(compute_slopes,span=9)
              .pipe(compute_slope_adj,span=9,window=50,lag=4)
              .pipe(compute_gap)
              .pipe(compute_div_ema_atr)
              .pipe(compute_pct_low_moves)
              .pipe(compute_range_pos))
        df['Prev_Close']=df['Close'].shift()
        df['Close_D3']=df['Close'].shift(3)
        df['Close_D4']=df['Close'].shift(4)
        df['Pct_1d']=df['Close'].pct_change()*100
        df['Pct_1d_div_ATR']=df['Pct_1d']/df['ATR']
        df['Move2d_div_ATR']=(df['Prev_Close']-df['Close_D3'])/df['ATR']
        df['Move3d_div_ATR']=(df['Prev_Close']-df['Close_D4'])/df['ATR']
        df['Open_gt_PrevHigh']=df['Open']>df['High'].shift()
        df['TrigGreen']       =df['Close'].shift()>df['Open'].shift()
        df['Range_over_ATR']=df['TR']/df['ATR']
        df['Vol_over_AVG']=df['Volume']/df['VOL_AVG']
        df['Close_over_EMA9']=df['Close']/df['EMA_9']
        df['Open_over_EMA9'] =df['Open'] /df['EMA_9']
        return df
    
    # ── analysis driver ───────────────────────────────────────────────────────────
    def analyze_known_events(evts, lookback=90):
        rows=[]
        for tk,date in evts:
            end   = pd.to_datetime(date)+timedelta(days=1)
            start = end - timedelta(days=lookback)
            df    = fetch_aggregates(tk,start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'))
            if df.empty: continue
            dfm   = compute_all_metrics(df)
            idx   = dfm.index.strftime('%Y-%m-%d')
            if date not in idx: continue
            i            = idx.tolist().index(date)
            row_trig     = dfm.iloc[i]      # D-0
            row_d1       = dfm.iloc[i-1]    # D-1  ← fixed line
            rows.append({
                'Ticker':tk,'Date (D-1)':(pd.to_datetime(date)-timedelta(days=1)).strftime('%Y-%m-%d'),
                'Prev_Close':row_d1['Prev_Close'],'Prev_Volume':row_d1['Prev_Volume'],
                'ATR':row_d1['ATR'],'ATR_%Δ':row_d1['ATR_Pct_Change'],
                'Slope_9_3d':row_d1['Slope_9_3d'],'Slope_9_5d':row_d1['Slope_9_5d'],
                'Slope_9_15d':row_d1['Slope_9_15d'],'Slope_9_50d_adj':row_d1['Slope_9_50d_adj'],
                'Open/EMA9':row_d1['Open_over_EMA9'],'Range/ATR':row_d1['Range_over_ATR'],
                'Vol/Avg':row_d1['Vol_over_AVG'],
                'High/EMA9/ATR':row_d1['High_over_EMA9_div_ATR'],
                'High/EMA20/ATR':row_d1['High_over_EMA20_div_ATR'],
                'Pct7d/ATR':row_d1['Pct_7d_low_div_ATR'],'Pct14d/ATR':row_d1['Pct_14d_low_div_ATR'],
                'Gap/ATR':row_d1['Gap_over_ATR'],'Upper70%':row_d1['Upper_70_Range'],
                'Close/EMA9':row_d1['Close_over_EMA9'],
                'Move2d/ATR':row_d1['Move2d_div_ATR'],'Move3d/ATR':row_d1['Move3d_div_ATR'],
                'Open>PrevHigh?':bool(row_trig['Open_gt_PrevHigh']),
                'TrigGreen?':bool(row_trig['TrigGreen'])
            })
        return pd.DataFrame(rows)
    
    if __name__=='__main__':
        events = [
            ("TSLL", "2024-12-17"),
            ("TSLL", "2024-12-09"),
            ("PLTR", "2024-12-09"),
            ("KWEB", "2024-09-30"),
            ("NVDA", "2024-03-08"),
            ("SOXL", "2024-05-23"),
            ("MSTR", "2024-02-15"),
        ]
        df=analyze_known_events(events)
        pd.set_option('display.max_columns',None, 'display.width',0)
        print('\nKnown-event metrics:\n',df)
    
    ```
    
    ```python
    # debug_scan.py ── mirrors scan_daily.py for one-off inspection
    import pandas as pd, requests
    from datetime import timedelta, datetime
    
    API_KEY  = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    BASE_URL = 'https://api.polygon.io'
    
    # ---------------- data fetch ----------------
    def fetch_aggregates(tkr: str, start: str, end: str) -> pd.DataFrame:
        url = f'{BASE_URL}/v2/aggs/ticker/{tkr}/range/1/day/{start}/{end}'
        r   = requests.get(url, params={'apiKey': API_KEY}); r.raise_for_status()
        js  = r.json().get('results', [])
        if not js: return pd.DataFrame()
        df  = pd.DataFrame(js)
        df['Date'] = pd.to_datetime(df['t'], unit='ms')
        df.rename(columns={'o':'Open','h':'High','l':'Low','c':'Close','v':'Volume'}, inplace=True)
        df.set_index('Date', inplace=True)
        return df[['Open','High','Low','Close','Volume']]
    
    # ---------------- indicators (identical to scan) ---------------
    def compute_emas(df, spans=(9, 20)):
        for s in spans: df[f'EMA_{s}'] = df['Close'].ewm(span=s, adjust=False).mean()
        return df
    
    def compute_atr(df, period=14):
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift()).abs(),
            (df['Low']  - df['Close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['TR'] = tr
        df['ATR_raw'] = tr.rolling(period, min_periods=period).mean()
        df['ATR']     = df['ATR_raw'].shift(1)
        df['ATR_Pct_Change'] = df['ATR_raw'].pct_change().shift(1) * 100
        return df
    
    def compute_volume(df, period=14):
        df['VOL_AVG_raw'] = df['Volume'].rolling(period, min_periods=period).mean()
        df['VOL_AVG']     = df['VOL_AVG_raw'].shift(1)
        df['Prev_Volume'] = df['Volume'].shift(1)
        return df
    
    def compute_slopes(df, span=9, win=(3,5,15)):
        ema = f'EMA_{span}'
        for w in win:
            df[f'Slope_{span}_{w}d'] = (df[ema] - df[ema].shift(w)) / df[ema].shift(w) * 100
        df['Slope_9_4to50d'] = (df[ema].shift(4) - df[ema].shift(50)) / df[ema].shift(50) * 100
        return df
    
    def compute_gap(df):
        df['Gap']          = (df['Open'] - df['Close'].shift()).abs()
        df['Gap_over_ATR'] = df['Gap'] / df['ATR']
        return df
    
    def compute_div_ema_atr(df, spans=(9,20)):
        for s in spans:
            df[f'High_over_EMA{s}_div_ATR'] = (df['High'] - df[f'EMA_{s}']) / df['ATR']
        return df
    
    def compute_pct_changes(df):
        low7  = df['Low'].rolling(7 , min_periods=7 ).min()
        low14 = df['Low'].rolling(14, min_periods=14).min()
        df['Pct_7d_low_div_ATR']  = ((df['Close'] - low7 ) / low7 ) / df['ATR'] * 100
        df['Pct_14d_low_div_ATR'] = ((df['Close'] - low14) / low14) / df['ATR'] * 100
        return df
    
    def compute_range_pos(df):
        df['Upper_70_Range'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']) * 100
        return df
    
    def compute_all_metrics(df):
        df = (df.pipe(compute_emas)
                .pipe(compute_atr)
                .pipe(compute_volume)
                .pipe(compute_slopes)
                .pipe(compute_gap)
                .pipe(compute_div_ema_atr)
                .pipe(compute_pct_changes)
                .pipe(compute_range_pos))
    
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_Open']  = df['Open'].shift(1)
        df['Prev_High']  = df['High'].shift(1)
        df['Close_D3']   = df['Close'].shift(3)
        df['Close_D4']   = df['Close'].shift(4)
    
        df['Prev_Gain_Pct'] = (df['Prev_Close'] - df['Prev_Open']) / df['Prev_Open'] * 100
        df['Move2d_div_ATR'] = (df['Prev_Close'] - df['Close_D3']) / df['ATR']
        df['Move3d_div_ATR'] = (df['Prev_Close'] - df['Close_D4']) / df['ATR']
        return df
    
    # --------------- thresholds (same as scan defaults/custom) ---------------
    P = {
        'atr_mult':4, 'vol_mult':2,
        'slope3d_min':10, 'slope5d_min':20, 'slope15d_min':40,
        'high_ema9_mult':4, 'high_ema20_mult':5,
        'pct7d_low_div_atr_min':0.5, 'pct14d_low_div_atr_min':1.5,
        'gap_div_atr_min':0.5, 'open_over_ema9_min':1.25,
        'atr_pct_change_min':10, 'prev_close_min':15,
        'pct2d_div_atr_min':2, 'pct3d_div_atr_min':3,
    }
    
    # --------------- events to inspect ---------------
    EVENTS = [
        ('DJT',  '2024-10-29'),
        ('SMCI', '2024-02-16'),
        ('MSTR', '2024-11-21'),
    ]
    
    # --------------- run loop ---------------
    def show(symbol, date_str):
        start = (pd.to_datetime(date_str) - timedelta(days=90)).strftime('%Y-%m-%d')
        end   = (pd.to_datetime(date_str) + timedelta(days=1)).strftime('%Y-%m-%d')
        df = fetch_aggregates(symbol, start, end)
        if df.empty:
            print(f"\n{symbol}: no data"); return
        dm = compute_all_metrics(df)
        dm.index = dm.index.strftime('%Y-%m-%d')
        if date_str not in dm.index:
            print(f"\n{symbol}: date missing"); return
        row = dm.loc[date_str]
    
        print(f"\n── {symbol}  {date_str} ──")
        pd.set_option('display.width', None)
        print(row.to_frame('Value').T)
    
        def ok(lbl,val,thr): print(f"{lbl:24s}: {val:8.3f} vs {thr:8.3f} → {'PASS' if val>=thr else 'FAIL'}")
    
        print("\nChecks")
        ok('Range/ATR',      row['TR'] / row['ATR'],           P['atr_mult'])
        ok('Vol/Avg',        row['Volume'] / row['VOL_AVG'],   P['vol_mult'])
        ok('PrevVol/Avg',    row['Prev_Volume']/row['VOL_AVG'],P['vol_mult'])
        ok('Slope_9_3d',     row['Slope_9_3d'],                P['slope3d_min'])
        ok('Slope_9_5d',     row['Slope_9_5d'],                P['slope5d_min'])
        ok('Slope_9_15d',    row['Slope_9_15d'],               P['slope15d_min'])
        ok('High/EMA9/ATR',  row['High_over_EMA9_div_ATR'],    P['high_ema9_mult'])
        ok('High/EMA20/ATR', row['High_over_EMA20_div_ATR'],   P['high_ema20_mult'])
        ok('Pct7d_low/ATR',  row['Pct_7d_low_div_ATR'],        P['pct7d_low_div_atr_min'])
        ok('Pct14d_low/ATR', row['Pct_14d_low_div_ATR'],       P['pct14d_low_div_atr_min'])
        ok('Gap/ATR',        row['Gap_over_ATR'],              P['gap_div_atr_min'])
        ok('Open/EMA9',      row['Open'] / row['EMA_9'],       P['open_over_ema9_min'])
        ok('ATR %Δ (D-1)',   row['ATR_Pct_Change'],            P['atr_pct_change_min'])
        ok('Prev_Close',     row['Prev_Close'],                P['prev_close_min'])
        ok('Move2d/ATR',     row['Move2d_div_ATR'],            P['pct2d_div_atr_min'])
        ok('Move3d/ATR',     row['Move3d_div_ATR'],            P['pct3d_div_atr_min'])
        print(f"Prev_Close > Prev_Open?  : {row['Prev_Close'] > row['Prev_Open']}")
        print(f"Open > Prev_High?        : {row['Open'] > row['Prev_High']}")
    
    if __name__ == '__main__':
        for sym, d in EVENTS:
            show(sym, d)
    
    ```
    
- dev band scans
    
    This scan shows how the depth bands could potentially be used inside of the scan as a confirmation.
    
    ```python
    # scan_bucket_A_frontside_AH_or_PRE_touch.py
    # Bucket A — Front-side blowoff into 72-band in AH (prev 16–20) or PRE (04–10:15).
    # Trade at the touch (AH/PRE) or at the open if first touch was earlier and we’re still extended.
    # Includes: daily-strength filters, liquidity guards, and EMA9 “angle” grades at the HH
    # for 15m / 30m / 1H / 2H. Prints a compact table.
    
    import pandas as pd, numpy as np, requests
    from datetime import datetime, timedelta, time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # ───────── config ─────────
    session  = requests.Session()
    API_KEY  = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
    BASE_URL = "https://api.polygon.io"
    ET_TZ    = "US/Eastern"
    
    # Deviation threshold & proximity
    DEV_THR   = 6.9
    OPEN_PROX = 0.5          # 09:30 open needs Dev72 >= DEV_THR - OPEN_PROX
    
    LOOKBACK_CAL_DAYS = 25   # intraday fetch window
    
    WINDOWS = {
        "AH":   ((16,0),(20,0)),   # previous day (ET)
        "PRE1": ((4,0),(7,0)),
        "PRE2": ((7,0),(10,15)),   # GTZ
        "REG":  ((9,30),(16,0)),
    }
    
    # Liquidity thresholds
    LIQ_THR = {
        "min_price": 5.0,
        "adv20_dollar_min": 50_000_000,
        "reg10_med_min":    2_000_000,   # $/bar 1H REG median over ~10 days
        "pre10_med_min":      300_000,   # $/bar 15m PRE median over ~10 days
        "d0_pre_sum_min":     400_000,   # $ sum in PRE (04:00–09:30) on D0
    }
    
    # EMA9 “angle” windows (bars ending at HH)
    SLOPE_BARS = {"15m": 3, "30m": 3, "1h": 3, "2h": 2}
    BAR_MINS   = {"15m": 15, "30m": 30, "1h": 60, "2h": 120}
    
    # ───────── fetch helpers ─────────
    def fetch_daily(tkr: str, start: str, end: str) -> pd.DataFrame:
        url = f"{BASE_URL}/v2/aggs/ticker/{tkr}/range/1/day/{start}/{end}"
        r   = session.get(url, params={"apiKey": API_KEY, "adjusted": "true", "sort": "asc", "limit": 50000})
        r.raise_for_status()
        rows = r.json().get("results", [])
        if not rows: return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .assign(Date=lambda d: pd.to_datetime(d["t"], unit="ms"))
            .rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
            .set_index("Date")[["Open","High","Low","Close","Volume"]]
            .sort_index()
        )
    
    def fetch_intraday(tkr: str, mult: int, span: str, start: str, end: str) -> pd.DataFrame:
        url = f"{BASE_URL}/v2/aggs/ticker/{tkr}/range/{mult}/{span}/{start}/{end}"
        r   = session.get(url, params={"apiKey": API_KEY, "adjusted": "true", "sort": "asc", "limit": 50000})
        r.raise_for_status()
        rows = r.json().get("results", [])
        if not rows: return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .assign(ts=lambda d: pd.to_datetime(d["t"], unit="ms", utc=True).dt.tz_convert(ET_TZ))
            .rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
            .set_index("ts")[["Open","High","Low","Close","Volume"]]
            .sort_index()
        )
    
    # ───────── daily strength ─────────
    def add_daily_strength_metrics(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
        for w in (5, 15):
            df[f"Slope_9_{w}d"] = (df["EMA_9"].shift(1) - df["EMA_9"].shift(w+1)) / df["EMA_9"].shift(w+1) * 100
        rng = (df["Close"] - df["Low"]) / (df["High"] - df["Low"]).replace(0, np.nan) * 100
        df["Upper_70_Range_prev"] = rng.shift(1)
        df["Vol_avg20"] = df["Volume"].rolling(20, min_periods=20).mean()
        df["ADV20_$"]   = (df["Close"] * df["Volume"]).rolling(20, min_periods=20).mean()
        df["Vol_over_AVG_prev"] = df["Volume"].shift(1) / df["Vol_avg20"].shift(1)
        df["Pct_1d_prev"] = df["Close"].pct_change().shift(1) * 100
        df["Close_over_EMA9_prev"] = df["Close"].shift(1) / df["EMA_9"].shift(1)
        df["Prev_Close"] = df["Close"].shift(1)
        return df
    
    def passes_daily_strength(df: pd.DataFrame, d0: pd.Timestamp) -> bool:
        try:
            row = df.loc[d0]
        except KeyError:
            return False
        return bool(
            (row["Pct_1d_prev"] >= 4.0) and
            (row["Upper_70_Range_prev"] >= 70) and
            (row["Vol_over_AVG_prev"] >= 1.5) and
            (row["Slope_9_5d"] >= 6) and
            (row["Slope_9_15d"] >= 15) and
            (row["Close_over_EMA9_prev"] >= 1.01)
        )
    
    # ───────── intraday cloud + liquidity ─────────
    def add_cloud_metrics(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"]  - df["Close"].shift()).abs()
        ], axis=1).max(axis=1)
        df["ATR_72"] = tr.rolling(72, min_periods=30).mean()
        df["EMA_72"] = df["Close"].ewm(span=72, adjust=False).mean()
        df["Dev72Mult_High"] = (df["High"] - df["EMA_72"]) / df["ATR_72"].replace(0, np.nan)
        df["Dev72Mult_Open"] = (df["Open"] - df["EMA_72"]) / df["ATR_72"].replace(0, np.nan)
        df["DollarVol"] = df["Close"] * df["Volume"]
        df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()   # for angle calc
        df["tod"]  = df.index.tz_convert(ET_TZ).time
        return df
    
    def _ts_local(d: datetime.date, hh: int, mm: int) -> pd.Timestamp:
        return pd.Timestamp.combine(d, time(hh, mm)).tz_localize(ET_TZ)
    
    def window_flags(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp, dev_thr=DEV_THR) -> dict:
        w = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        if w.empty:
            return {"max_dev": np.nan, "first_touchTS": None}
        hit = w[w["Dev72Mult_High"] >= dev_thr]
        first = hit.index[0] if not hit.empty else None
        return {"max_dev": float(w["Dev72Mult_High"].max()), "first_touchTS": first}
    
    def earliest_touch(*touches):
        ts = [t["first_touchTS"] for t in touches if t["first_touchTS"] is not None]
        if not ts: return None
        return min(ts)
    
    def intraday_liquidity(df: pd.DataFrame, d0_et: pd.Timestamp) -> dict:
        if df.empty:
            return {"REG_med_$": np.nan, "PRE_med_$": np.nan, "D0_pre_sum_$": np.nan}
        def in_tod(t, start, end): return (t >= start) and (t <= end)
        start_date = (d0_et - timedelta(days=15)).date()
        mask_hist = (df.index.date < d0_et.date()) & (df.index.date >= start_date)
    
        reg_start, reg_end = WINDOWS["REG"][0], WINDOWS["REG"][1]
        reg_start, reg_end = time(*reg_start), time(*reg_end)
        pre_start, pre_end = WINDOWS["PRE2"][0], WINDOWS["PRE2"][1]
        pre_start, pre_end = time(*pre_start), time(*pre_end)
    
        reg = df[mask_hist & df["tod"].apply(lambda t: in_tod(t, reg_start, reg_end))]
        pre = df[mask_hist & df["tod"].apply(lambda t: in_tod(t, pre_start, pre_end))]
    
        reg_med = float(reg["DollarVol"].median()) if not reg.empty else np.nan
        pre_med = float(pre["DollarVol"].median()) if not pre.empty else np.nan
    
        d0_pre = df[(df.index.date == d0_et.date()) & df["tod"].apply(lambda t: time(4,0) <= t <= time(9,30))]
        d0_pre_sum = float(d0_pre["DollarVol"].sum()) if not d0_pre.empty else np.nan
    
        return {"REG_med_$": reg_med, "PRE_med_$": pre_med, "D0_pre_sum_$": d0_pre_sum}
    
    # ───────── HH + EMA9 angle helpers ─────────
    def _nearest_le_idx(idx, ts):
        if ts in idx: return ts
        pos = idx.searchsorted(ts, side="right") - 1
        if pos < 0: return None
        return idx[pos]
    
    def highest_high_segment(df: pd.DataFrame, seg_start: pd.Timestamp, seg_end: pd.Timestamp) -> dict:
        w = df.loc[(df.index >= seg_start) & (df.index <= seg_end)]
        if w.empty:
            return {"HH": np.nan, "HH_TS": None, "HH_Dev": np.nan}
        i = w["High"].idxmax()
        return {
            "HH": float(w.at[i, "High"]),
            "HH_TS": i,
            "HH_Dev": float(w.at[i, "Dev72Mult_High"]) if pd.notna(w.at[i, "Dev72Mult_High"]) else np.nan,
        }
    
    def ema9_angle_at_ts(df: pd.DataFrame, ts: pd.Timestamp, bars: int, bar_minutes: int) -> dict:
        """EMA9 slope & angle ending at *ts*:
           - pct_per_hr and angle(deg) = atan(pct_per_hr)
           - dev_per_hr (ATR72-normalized) and angle(deg)
        """
        if df.empty or ts is None or bars <= 0:
            return {"pct_per_hr": np.nan, "dev_per_hr": np.nan,
                    "angle_pct_deg_hr": np.nan, "angle_dev_deg_hr": np.nan}
    
        ts_i = _nearest_le_idx(df.index, ts)
        if ts_i is None:
            return {"pct_per_hr": np.nan, "dev_per_hr": np.nan,
                    "angle_pct_deg_hr": np.nan, "angle_dev_deg_hr": np.nan}
    
        loc = df.index.get_loc(ts_i)
        if isinstance(loc, slice):
            return {"pct_per_hr": np.nan, "dev_per_hr": np.nan,
                    "angle_pct_deg_hr": np.nan, "angle_dev_deg_hr": np.nan}
    
        prev = loc - bars
        if prev < 0:
            return {"pct_per_hr": np.nan, "dev_per_hr": np.nan,
                    "angle_pct_deg_hr": np.nan, "angle_dev_deg_hr": np.nan}
    
        ema_now = df["EMA_9"].iloc[loc]
        ema_prev= df["EMA_9"].iloc[prev]
        if pd.isna(ema_now) or pd.isna(ema_prev) or ema_prev == 0:
            return {"pct_per_hr": np.nan, "dev_per_hr": np.nan,
                    "angle_pct_deg_hr": np.nan, "angle_dev_deg_hr": np.nan}
    
        # % slope per bar → per hour
        pct_per_bar = (ema_now - ema_prev) / ema_prev / bars * 100.0
        pct_per_hr  = pct_per_bar * (60.0 / max(1, bar_minutes))
    
        # deviation-normalized slope
        atr_mean = df["ATR_72"].iloc[prev:loc+1].mean()
        dev_per_bar = ((ema_now - ema_prev) / max(1e-12, atr_mean)) / bars
        dev_per_hr  = dev_per_bar * (60.0 / max(1, bar_minutes))
    
        angle_pct_deg_hr = float(np.degrees(np.arctan(pct_per_hr)))
        angle_dev_deg_hr = float(np.degrees(np.arctan(dev_per_hr)))
    
        return {"pct_per_hr": float(pct_per_hr), "dev_per_hr": float(dev_per_hr),
                "angle_pct_deg_hr": angle_pct_deg_hr, "angle_dev_deg_hr": angle_dev_deg_hr}
    
    # ───────── main scan ─────────
    def scan_bucket_A(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        daily = fetch_daily(symbol, start_date, end_date)
        if daily.empty: return pd.DataFrame()
        daily = add_daily_strength_metrics(daily)
    
        out = []
        for d0 in daily.index[1:]:
            if not passes_daily_strength(daily, d0):
                continue
            row_d = daily.loc[d0]
            if (row_d["Prev_Close"] < LIQ_THR["min_price"]) or (row_d["ADV20_$"] < LIQ_THR["adv20_dollar_min"]):
                continue
    
            d0_et = pd.Timestamp(d0).tz_localize(ET_TZ)
            start = (d0_et - timedelta(days=LOOKBACK_CAL_DAYS)).strftime("%Y-%m-%d")
            end   = (d0_et + timedelta(days=1)).strftime("%Y-%m-%d")
    
            # intraday frames
            h1  = add_cloud_metrics(fetch_intraday(symbol, 1,  "hour",   start, end))
            m15 = add_cloud_metrics(fetch_intraday(symbol, 15, "minute", start, end))
            m30 = add_cloud_metrics(fetch_intraday(symbol, 30, "minute", start, end))
            h2  = add_cloud_metrics(fetch_intraday(symbol, 2,  "hour",   start, end))
            if h1.empty or m15.empty:
                continue
    
            prev_date = (d0_et - timedelta(days=1)).date()
            ah_s, ah_e   = _ts_local(prev_date, *WINDOWS["AH"][0]),   _ts_local(prev_date, *WINDOWS["AH"][1])
            pre1_s, pre1_e = _ts_local(d0_et.date(), *WINDOWS["PRE1"][0]), _ts_local(d0_et.date(), *WINDOWS["PRE1"][1])
            pre2_s, pre2_e = _ts_local(d0_et.date(), *WINDOWS["PRE2"][0]), _ts_local(d0_et.date(), *WINDOWS["PRE2"][1])
    
            # touches
            h1_ah   = window_flags(h1,  ah_s, ah_e)
            m15_ah  = window_flags(m15, ah_s, ah_e)
            h1_pre1 = window_flags(h1,  pre1_s, pre1_e)
            m15_pre1= window_flags(m15, pre1_s, pre1_e)
            h1_pre2 = window_flags(h1,  pre2_s, pre2_e)
            m15_pre2= window_flags(m15, pre2_s, pre2_e)
    
            # open proximity
            t0930 = _ts_local(d0_et.date(), 9, 30)
            r0930 = m15.loc[t0930] if t0930 in m15.index else None
            open_dev = None
            open_prox_ok = False
            if r0930 is not None and not pd.isna(r0930["Dev72Mult_Open"]):
                open_dev = float(r0930["Dev72Mult_Open"])
                open_prox_ok = (open_dev >= (DEV_THR - OPEN_PROX))
    
            # validity
            touch_in_pre2 = (h1_pre2["first_touchTS"] is not None) or (m15_pre2["first_touchTS"] is not None)
            touch_earlier = any([d["first_touchTS"] is not None for d in (h1_ah, m15_ah, h1_pre1, m15_pre1)])
            valid = touch_in_pre2 or (touch_earlier and (touch_in_pre2 or open_prox_ok))
    
            # liquidity gates
            liq1h = intraday_liquidity(h1,  d0_et)
            liq15 = intraday_liquidity(m15, d0_et)
            liq_ok = (
                (liq1h["REG_med_$"] >= LIQ_THR["reg10_med_min"]) and
                (liq15["PRE_med_$"] >= LIQ_THR["pre10_med_min"]) and
                (liq15["D0_pre_sum_$"] >= LIQ_THR["d0_pre_sum_min"])
            )
            if not (valid and liq_ok):
                continue
    
            # HH window (prev 16:00 → D0 10:15)
            seg_start = ah_s
            seg_end   = pre2_e
    
            hh_1h  = highest_high_segment(h1,  seg_start, seg_end)
            hh_15  = highest_high_segment(m15, seg_start, seg_end)
            hh_30  = highest_high_segment(m30, seg_start, seg_end) if not m30.empty else {"HH":np.nan,"HH_TS":None,"HH_Dev":np.nan}
            hh_2h  = highest_high_segment(h2,  seg_start, seg_end) if not h2.empty else {"HH":np.nan,"HH_TS":None,"HH_Dev":np.nan}
    
            # angles
            a_1h  = ema9_angle_at_ts(h1,  hh_1h["HH_TS"],  SLOPE_BARS["1h"],  BAR_MINS["1h"])  if hh_1h["HH_TS"]  is not None else {}
            a_15  = ema9_angle_at_ts(m15, hh_15["HH_TS"],  SLOPE_BARS["15m"], BAR_MINS["15m"]) if hh_15["HH_TS"]  is not None else {}
            a_30  = ema9_angle_at_ts(m30, hh_30["HH_TS"],  SLOPE_BARS["30m"], BAR_MINS["30m"]) if hh_30["HH_TS"]  is not None else {}
            a_2h  = ema9_angle_at_ts(h2,  hh_2h["HH_TS"],  SLOPE_BARS["2h"],  BAR_MINS["2h"])  if hh_2h["HH_TS"]  is not None else {}
    
            fmt = "%m-%d %H:%M"  # compact timestamps
            row = {
                "Ticker": symbol,
                "Date": d0.strftime("%Y-%m-%d"),
                # D-1 context
                "D-1%": round(float(row_d["Pct_1d_prev"]), 2),
                "D-1 Upper%": round(float(row_d["Upper_70_Range_prev"]), 1),
                "D-1 Vol/Avg": round(float(row_d["Vol_over_AVG_prev"]), 2),
                "Slope9_5d": round(float(row_d["Slope_9_5d"]), 2),
                "Slope9_15d": round(float(row_d["Slope_9_15d"]), 2),
                # touches
                "AH FirstTouch":   h1_ah["first_touchTS"].strftime(fmt)    if h1_ah["first_touchTS"]   is not None else (m15_ah["first_touchTS"].strftime(fmt)   if m15_ah["first_touchTS"]   is not None else None),
                "PRE1 FirstTouch": h1_pre1["first_touchTS"].strftime(fmt)  if h1_pre1["first_touchTS"] is not None else (m15_pre1["first_touchTS"].strftime(fmt) if m15_pre1["first_touchTS"] is not None else None),
                "PRE2 FirstTouch": h1_pre2["first_touchTS"].strftime(fmt)  if h1_pre2["first_touchTS"] is not None else (m15_pre2["first_touchTS"].strftime(fmt) if m15_pre2["first_touchTS"] is not None else None),
                "OpenDev@09:30": round(open_dev, 2) if open_dev is not None else np.nan,
                "OpenProxOK": bool(open_prox_ok),
                # liquidity
                "ADV20_$": round(float(row_d["ADV20_$"]), 0),
                "REG_MedDV$_10d": round(liq1h["REG_med_$"], 0) if pd.notna(liq1h["REG_med_$"]) else np.nan,
                "PRE_MedDV$_10d": round(liq15["PRE_med_$"], 0) if pd.notna(liq15["PRE_med_$"]) else np.nan,
                "D0_PRE_SumDV$_toOpen": round(liq15["D0_pre_sum_$"], 0) if pd.notna(liq15["D0_pre_sum_$"]) else np.nan,
                # HH + Dev72 at HH
                "HH_1H": hh_1h["HH"], "HH_1H_dev": hh_1h["HH_Dev"], "HH_1H_time": hh_1h["HH_TS"].strftime(fmt) if hh_1h["HH_TS"] else None,
                "HH_15m": hh_15["HH"], "HH_15m_dev": hh_15["HH_Dev"], "HH_15m_time": hh_15["HH_TS"].strftime(fmt) if hh_15["HH_TS"] else None,
                "HH_30m": hh_30["HH"], "HH_30m_dev": hh_30["HH_Dev"], "HH_30m_time": hh_30["HH_TS"].strftime(fmt) if hh_30["HH_TS"] else None,
                "HH_2H": hh_2h["HH"], "HH_2H_dev": hh_2h["HH_Dev"], "HH_2H_time": hh_2h["HH_TS"].strftime(fmt) if hh_2h["HH_TS"] else None,
                # EMA9 angles (hour-normalized)
                "EMA9_PctSlope_1H_per_hr": a_1h.get("pct_per_hr", np.nan),
                "EMA9_AnglePct_1H_deg_hr": a_1h.get("angle_pct_deg_hr", np.nan),
                "EMA9_DevSlope_1H_per_hr": a_1h.get("dev_per_hr", np.nan),
                "EMA9_AngleDev_1H_deg_hr": a_1h.get("angle_dev_deg_hr", np.nan),
                "EMA9_PctSlope_15m_per_hr": a_15.get("pct_per_hr", np.nan),
                "EMA9_AnglePct_15m_deg_hr": a_15.get("angle_pct_deg_hr", np.nan),
                "EMA9_DevSlope_15m_per_hr": a_15.get("dev_per_hr", np.nan),
                "EMA9_AngleDev_15m_deg_hr": a_15.get("angle_dev_deg_hr", np.nan),
                "EMA9_PctSlope_30m_per_hr": a_30.get("pct_per_hr", np.nan),
                "EMA9_AnglePct_30m_deg_hr": a_30.get("angle_pct_deg_hr", np.nan),
                "EMA9_DevSlope_30m_per_hr": a_30.get("dev_per_hr", np.nan),
                "EMA9_AngleDev_30m_deg_hr": a_30.get("angle_dev_deg_hr", np.nan),
                "EMA9_PctSlope_2H_per_hr": a_2h.get("pct_per_hr", np.nan),
                "EMA9_AnglePct_2H_deg_hr": a_2h.get("angle_pct_deg_hr", np.nan),
                "EMA9_DevSlope_2H_per_hr": a_2h.get("dev_per_hr", np.nan),
                "EMA9_AngleDev_2H_deg_hr": a_2h.get("angle_dev_deg_hr", np.nan),
            }
            out.append(row)
    
        return pd.DataFrame(out)
    
    # ───────── compact printing helpers ─────────
    ROUND = 2  # decimals for floats
    
    SHORT_NAMES = {
        "D-1%":"D1%", "D-1 Upper%":"D1U%", "D-1 Vol/Avg":"D1V",
        "Slope9_5d":"s9_5d", "Slope9_15d":"s9_15d",
        "AH FirstTouch":"AH", "PRE1 FirstTouch":"PRE1", "PRE2 FirstTouch":"PRE2",
        "OpenDev@09:30":"ODev", "OpenProxOK":"OProx",
        "ADV20_$":"ADV20$", "REG_MedDV$_10d":"H1$Med", "PRE_MedDV$_10d":"P15$Med",
        "D0_PRE_SumDV$_toOpen":"PRE$Sum",
        "HH_1H":"HH1H", "HH_1H_dev":"Dev1H", "HH_1H_time":"T1H",
        "HH_15m":"HH15", "HH_15m_dev":"Dev15", "HH_15m_time":"T15",
        "HH_30m":"HH30", "HH_30m_dev":"Dev30", "HH_30m_time":"T30",
        "HH_2H":"HH2H", "HH_2H_dev":"Dev2H", "HH_2H_time":"T2H",
        "EMA9_PctSlope_1H_per_hr":"Pct1H", "EMA9_AnglePct_1H_deg_hr":"Ang%1H",
        "EMA9_DevSlope_1H_per_hr":"DevS1H", "EMA9_AngleDev_1H_deg_hr":"AngD1H",
        "EMA9_PctSlope_15m_per_hr":"Pct15", "EMA9_AnglePct_15m_deg_hr":"Ang%15",
        "EMA9_DevSlope_15m_per_hr":"DevS15", "EMA9_AngleDev_15m_deg_hr":"AngD15",
        "EMA9_PctSlope_30m_per_hr":"Pct30", "EMA9_AnglePct_30m_deg_hr":"Ang%30",
        "EMA9_DevSlope_30m_per_hr":"DevS30", "EMA9_AngleDev_30m_deg_hr":"AngD30",
        "EMA9_PctSlope_2H_per_hr":"Pct2H", "EMA9_AnglePct_2H_deg_hr":"Ang%2H",
        "EMA9_DevSlope_2H_per_hr":"DevS2H", "EMA9_AngleDev_2H_deg_hr":"AngD2H",
    }
    
    SHORT_ORDER = [
        "Ticker","Date","D1%","D1U%","D1V","s9_5d","s9_15d",
        "AH","PRE1","PRE2","ODev","OProx",
        "ADV20$","H1$Med","P15$Med","PRE$Sum",
        "HH1H","Dev1H","T1H","Pct1H","Ang%1H","DevS1H","AngD1H",
        "HH15","Dev15","T15","Pct15","Ang%15","DevS15","AngD15",
        "HH30","Dev30","T30","Pct30","Ang%30","DevS30","AngD30",
        "HH2H","Dev2H","T2H","Pct2H","Ang%2H","DevS2H","AngD2H",
    ]
    
    def compact_for_print(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # shorten any time-like columns (already compact, but keep just in case)
        for c in [c for c in d.columns if c.endswith("_time") or c in ("AH FirstTouch","PRE1 FirstTouch","PRE2 FirstTouch")]:
            d[c] = d[c].astype(str)
        # rename & round
        d = d.rename(columns=SHORT_NAMES)
        num_cols = d.select_dtypes(include=[float, int]).columns
        d[num_cols] = d[num_cols].round(ROUND)
        # order
        cols = [c for c in SHORT_ORDER if c in d.columns]
        other = [c for c in d.columns if c not in cols]
        return d[cols + other]
    
    # ───────── run CLI ─────────
    if __name__ == "__main__":
        symbols = ['MSTR','SMCI','DJT','BABA','TCOM','AMC','SOXL','MRVL','TGT','DOCU','ZM','DIS','NFLX','RKT','SNAP','RBLX','META','SE','NVDA','AAPL','MSFT','GOOGL','AMZN','TSLA','AMD','INTC','BA','PYPL','QCOM','ORCL','T','CSCO','VZ','KO','PEP','MRK','PFE','ABBV','JNJ','CRM','BAC','C','JPM','WMT','CVX','XOM','COP','RTX','SPGI','GS','HD','LOW','COST','UNH','NEE','NKE','LMT','HON','CAT','MMM','LIN','ADBE','AVGO','TXN','ACN','UPS','BLK','PM','MO','ELV','VRTX','ZTS','NOW','ISRG','PLD','MS','MDT','WM','GE','IBM','BKNG','FDX','ADP','EQIX','DHR','SNPS','REGN','SYK','TMO','CVS','INTU','SCHW','CI','APD','SO','MMC','ICE','FIS','ADI','CSX','LRCX','GILD','RIVN','LCID','PLTR','SNOW','SPY','QQQ','DIA','IWM','TQQQ','SQQQ','ARKK','LABU','TECL','UVXY','XLE','XLK','XLF','IBB','KWEB','TAN','XOP','EEM','HYG','EFA','USO','GLD','SLV','BITO','RIOT','MARA','COIN','SQ','AFRM','DKNG','SHOP','UPST','CLF','AA','F','GM','ROKU','WBD','WBA','PARA','PINS','LYFT','BYND','RDDT','GME','VKTX','APLD','KGEI','INOD','LMB','AMR','PMTS','SAVA','CELH','ESOA','IVT','MOD','SKYE','AR','VIXY','TECS','LABD','SPXS','SPXL','DRV','TZA','FAZ','WEBS','PSQ','SDOW','MSTU','MSTZ','NFLU','BTCL','BTCZ','ETU','ETQ','FAS','TNA','NUGT','TSLL','NVDU','AMZU','MSFU','UVIX','CRCL','SBET','MRNA','TIGR']
        start   = "2023-01-01"
        end     = datetime.today().strftime("%Y-%m-%d")
        MAX_WORKERS = 5
    
        def _worker(sym):
            try:
                return scan_bucket_A(sym, start, end)
            except Exception:
                return None
    
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futs = {exe.submit(_worker, s): s for s in symbols}
            for fut in as_completed(futs):
                df = fut.result()
                if df is not None and hasattr(df, "empty") and not df.empty:
                    results.append(df)
    
        if results:
            out = pd.concat(results, ignore_index=True).sort_values(["Date","Ticker"], ascending=[False, True])
            out = compact_for_print(out)
            pd.set_option("display.max_columns", None, "display.max_colwidth", 14)
            print("\nBucket A — compact view (AH/PRE 72-band touch + liquidity + HH & EMA9 angles):\n")
            print(out.to_string(index=False))
        else:
            print("No hits for given symbols/date range.")
    ```
    
- chart layouts
    
    This is a code of how we like our charts to look with indicators, color, candlesticks, etc. This is a great guide for building any form of charts.
    
    ```python
    import matplotlib
    matplotlib.use("Qt5Agg")
    
    import requests
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from datetime import datetime, timedelta
    from mplfinance.original_flavor import candlestick_ohlc
    import pandas_market_calendars as mcal
    import matplotlib.dates as mdates
    from zoneinfo import ZoneInfo
    
    plt.style.use('dark_background')
    
    def generate_and_save_chart(TICKER, TARGET_DATE, API_KEY):
        TICKER = TICKER.upper()
        today = pd.Timestamp(datetime.now().date())
    
        def fetch_polygon_agg(ticker, multiplier, timespan, from_date, to_date, api_key):
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
            params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
            response = requests.get(url, params=params)
            data = response.json()
            if "results" not in data:
                raise ValueError("Polygon API error or no data found")
            df = pd.DataFrame(data["results"])
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("US/Eastern")
            df.set_index("timestamp", inplace=True)
            df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
            return df[["Open", "High", "Low", "Close", "Volume"]]
    
        def add_bands(df, ema_short, ema_long, du1, du2, dl1, dl2, prefix=""):
            df[f"{prefix}emaS"] = df["Close"].ewm(span=ema_short, min_periods=0).mean()
            df[f"{prefix}emaL"] = df["Close"].ewm(span=ema_long, min_periods=0).mean()
            tr = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - df["Close"].shift()).abs(),
                (df["Low"] - df["Close"].shift()).abs()
            ], axis=1).max(axis=1)
            df[f"{prefix}ATR_S"] = tr.rolling(ema_short, min_periods=1).mean()
            df[f"{prefix}ATR_L"] = tr.rolling(ema_long, min_periods=1).mean()
            df[f"{prefix}UPP1"] = df[f"{prefix}emaS"] + du1 * df[f"{prefix}ATR_S"]
            df[f"{prefix}UPP2"] = df[f"{prefix}emaS"] + du2 * df[f"{prefix}ATR_S"]
            df[f"{prefix}LOW1"] = df[f"{prefix}emaL"] - dl1 * df[f"{prefix}ATR_L"]
            df[f"{prefix}LOW2"] = df[f"{prefix}emaL"] - dl2 * df[f"{prefix}ATR_L"]
    
        def compress_market_time(df, target_date):
            df = df.copy()
            df = df[((df.index.time >= pd.to_datetime("04:00").time()) & (df.index.time < pd.to_datetime("09:30").time())) |
                    ((df.index.time >= pd.to_datetime("09:30").time()) & (df.index.time <= pd.to_datetime("20:00").time()))]
            df = df[df.index.dayofweek < 5]
            df = df[df.index.date == target_date.date()]
            df["timestamp_est"] = df.index.tz_convert("US/Eastern")
            return df
    
        def remove_fake_wicks(df):
            return df[~((df["High"] == df["Low"]) & (df["Open"] == df["High"]) & (df["Close"] == df["High"]))]
    
        start_target = pd.to_datetime(TARGET_DATE)
        prev_day = start_target - timedelta(days=5)
        next_day = start_target + timedelta(days=1)
        sd_f = today - pd.Timedelta(days=365)
        ed = today
    
        df_2m_raw = remove_fake_wicks(fetch_polygon_agg(TICKER, 2, "minute", prev_day.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d"), API_KEY))
        df_5m_raw = remove_fake_wicks(fetch_polygon_agg(TICKER, 5, "minute", prev_day.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d"), API_KEY))
        df_15m_raw = remove_fake_wicks(fetch_polygon_agg(TICKER, 15, "minute", prev_day.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d"), API_KEY))
        df_daily = remove_fake_wicks(fetch_polygon_agg(TICKER, 1, "day", sd_f.strftime("%Y-%m-%d"), ed.strftime("%Y-%m-%d"), API_KEY))
    
        add_bands(df_15m_raw, 9, 20, 1.0, 0.5, 2.0, 2.5, prefix="s_")
        add_bands(df_5m_raw, 9, 20, 1.0, 0.5, 2.0, 2.5, prefix="s_")
        add_bands(df_2m_raw, 9, 20, 1.0, 0.5, 2.0, 2.5, prefix="q_")
        df_daily["ema9"] = df_daily["Close"].ewm(span=9, min_periods=0).mean()
        df_daily["ema20"] = df_daily["Close"].ewm(span=20, min_periods=0).mean()
        df_5m_raw["EMA_161"] = df_5m_raw["Close"].ewm(span=161, adjust=False, min_periods=0).mean()
        df_5m_raw["EMA_222"] = df_5m_raw["Close"].ewm(span=222, adjust=False, min_periods=0).mean()
    
        df_2m = compress_market_time(df_2m_raw, TARGET_DATE)
        df_5m = compress_market_time(df_5m_raw, TARGET_DATE)
        df_15m = compress_market_time(df_15m_raw, TARGET_DATE)
    
        df_5m["EMA_161"] = df_5m_raw.loc[df_5m.index, "EMA_161"].values
        df_5m["EMA_222"] = df_5m_raw.loc[df_5m.index, "EMA_222"].values
    
        for df in [df_5m, df_2m, df_15m]:
            df["date_only"] = df.index.date
            df["cum_vol"] = df.groupby("date_only")["Volume"].cumsum()
            df["cum_vol_price"] = (df["Close"] * df["Volume"]).groupby(df["date_only"]).cumsum()
            df["VWAP"] = df["cum_vol_price"] / df["cum_vol"]
    
        df_daily = df_daily.copy()
        df_daily["timestamp_est"] = df_daily.index.tz_convert("US/Eastern")
    
        output_dir = os.path.expanduser("~/Desktop/d1 charts")
        os.makedirs(output_dir, exist_ok=True)
    
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8), constrained_layout=True)
    
        visible_high = df_5m["High"].max()
        visible_low = df_5m["Low"].min()
        ax1.set_ylim(visible_low * 0.995, visible_high * 1.005)
        start_time = pd.Timestamp(f"{TARGET_DATE.date()} 07:00:00", tz="US/Eastern")
        end_time = pd.Timestamp(f"{TARGET_DATE.date()} 17:00:00", tz="US/Eastern")
        ax1.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
    
        ax1.xaxis_date()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=ZoneInfo("US/Eastern")))
        for i in range(len(df_5m) - 1):
            t1 = df_5m.index[i]
            if pd.to_datetime("04:00").time() <= t1.time() < pd.to_datetime("09:30").time():
                ax1.axvspan(mdates.date2num(df_5m["timestamp_est"].iloc[i]), mdates.date2num(df_5m["timestamp_est"].iloc[i + 1]), color="#444444", alpha=0.6)
            if pd.to_datetime("16:00").time() <= t1.time() < pd.to_datetime("20:00").time():
                ax1.axvspan(mdates.date2num(df_5m["timestamp_est"].iloc[i]), mdates.date2num(df_5m["timestamp_est"].iloc[i + 1]), color="#333333", alpha=0.5)
    
        candlestick_ohlc(ax1, list(zip(mdates.date2num(df_5m["timestamp_est"]), df_5m["Open"], df_5m["High"], df_5m["Low"], df_5m["Close"])),
                         width=0.0017, colorup="white", colordown="red")
        for col1, col2, color, alpha in [
            ("s_UPP1", "s_UPP2", "pink", 0.3),
            ("s_LOW1", "s_LOW2", "cyan", 0.3)
        ]:
            ax1.fill_between(mdates.date2num(df_5m["timestamp_est"]), df_5m[col1], df_5m[col2], color=color, alpha=alpha)
        ax1.fill_between(mdates.date2num(df_5m["timestamp_est"]), df_5m["EMA_161"], df_5m["EMA_222"],
                         color="gold", alpha=0.3, label="J-Line (EMA 161/222)")
    
        for col, style, label, color in [("s_emaS", "--", "EMA9", "yellow"), ("s_emaL", "--", "EMA20", "lightskyblue"), ("VWAP", "--", "VWAP", "purple")]:
            ax1.plot(mdates.date2num(df_5m["timestamp_est"]), df_5m[col], linestyle=style, label=label, color=color)
        ax1.set_title(f"{TICKER} {TARGET_DATE.date()} 5M")
        ax1.legend()
    
        ax2.xaxis_date()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=ZoneInfo("US/Eastern")))
    
        start_time = pd.Timestamp(f"{TARGET_DATE.date()} 07:00:00", tz="US/Eastern")
        end_time = pd.Timestamp(f"{TARGET_DATE.date()} 16:00:00", tz="US/Eastern")
        ax2.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
    
        for i in range(len(df_2m) - 1):
            t1 = df_2m.index[i]
            if pd.to_datetime("04:00").time() <= t1.time() < pd.to_datetime("09:30").time():
                ax2.axvspan(mdates.date2num(df_2m["timestamp_est"].iloc[i]), mdates.date2num(df_2m["timestamp_est"].iloc[i + 1]), color="#444444", alpha=0.6)
            if pd.to_datetime("16:00").time() <= t1.time() < pd.to_datetime("20:00").time():
                ax2.axvspan(mdates.date2num(df_2m["timestamp_est"].iloc[i]), mdates.date2num(df_2m["timestamp_est"].iloc[i + 1]), color="#333333", alpha=0.5)
    
        candlestick_ohlc(ax2, list(zip(mdates.date2num(df_2m["timestamp_est"]), df_2m["Open"], df_2m["High"], df_2m["Low"], df_2m["Close"])),
                         width=0.0007, colorup="white", colordown="red")
        for col1, col2, color, alpha in [
            ("q_UPP1", "q_UPP2", "red", 0.4),
            ("q_LOW1", "q_LOW2", "lightgreen", 0.4)
        ]:
            ax2.fill_between(mdates.date2num(df_2m["timestamp_est"]), df_2m[col1], df_2m[col2], color=color, alpha=alpha)
        for col, style, label, color in [("q_emaS", "--", "EMA9", "yellow"), ("q_emaL", "--", "EMA20", "lightskyblue"), ("VWAP", "--", "VWAP", "purple")]:
            ax2.plot(mdates.date2num(df_2m["timestamp_est"]), df_2m[col], linestyle=style, label=label, color=color)
        ax2.set_title(f"{TICKER} {TARGET_DATE.date()} 2M")
        ax2.legend()
    
        fig1.savefig(os.path.join(output_dir, f"{TICKER}_{TARGET_DATE.date()}_1.png"))
        plt.close(fig1)
    
        # Skipping daily and 15m chart here intentionally to stay within length limit. Let me know to continue.
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(22, 8), constrained_layout=True)
    
        ax3.xaxis_date()
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
        candlestick_ohlc(ax3, list(zip(mdates.date2num(df_daily["timestamp_est"]), df_daily["Open"], df_daily["High"], df_daily["Low"], df_daily["Close"])),
                         width=0.5, colorup="white", colordown="red")
        ax3.plot(mdates.date2num(df_daily["timestamp_est"]), df_daily["ema9"], linestyle="--", color="yellow", label="EMA9")
        ax3.plot(mdates.date2num(df_daily["timestamp_est"]), df_daily["ema20"], linestyle="--", color="lightskyblue", label="EMA20")
        ax3.set_title("Daily Chart")
        ax3.legend()
    
        visible_high_15m = df_15m["High"].max()
        visible_low_15m = df_15m["Low"].min()
        ax4.set_ylim(visible_low_15m * 0.995, visible_high_15m * 1.005)
    
        ax4.xaxis_date()
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=ZoneInfo("US/Eastern")))
    
        for i in range(len(df_15m) - 1):
            t1 = df_15m.index[i]
            if pd.to_datetime("04:00").time() <= t1.time() < pd.to_datetime("09:30").time():
                ax4.axvspan(mdates.date2num(df_15m["timestamp_est"].iloc[i]), mdates.date2num(df_15m["timestamp_est"].iloc[i + 1]), color="#444444", alpha=0.6)
            if pd.to_datetime("16:00").time() <= t1.time() < pd.to_datetime("20:00").time():
                ax4.axvspan(mdates.date2num(df_15m["timestamp_est"].iloc[i]), mdates.date2num(df_15m["timestamp_est"].iloc[i + 1]), color="#333333", alpha=0.5)
    
        candlestick_ohlc(ax4, list(zip(mdates.date2num(df_15m["timestamp_est"]), df_15m["Open"], df_15m["High"], df_15m["Low"], df_15m["Close"])),
                         width=0.005, colorup="white", colordown="red")
        for col1, col2, color, alpha in [
            ("s_UPP1", "s_UPP2", "pink", 0.3),
            ("s_LOW1", "s_LOW2", "cyan", 0.3)
        ]:
            ax4.fill_between(mdates.date2num(df_15m["timestamp_est"]), df_15m[col1], df_15m[col2], color=color, alpha=alpha)
        for col, style, label, color in [("s_emaS", "--", "EMA9", "yellow"), ("s_emaL", "--", "EMA20", "lightskyblue"), ("VWAP", "--", "VWAP", "purple")]:
            ax4.plot(mdates.date2num(df_15m["timestamp_est"]), df_15m[col], linestyle=style, label=label, color=color)
        ax4.set_title(f"{TICKER} {TARGET_DATE.date()} 15M")
        ax4.legend()
    
        fig2.savefig(os.path.join(output_dir, f"{TICKER}_{TARGET_DATE.date()}_2.png"))
        plt.close(fig2)
    
    API_KEY = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
    ticker_dates = [
        ("BMNR", "2025-07-01"),
        ("SRFM", "2025-06-25"),
        ("KLTO", "2025-06-10"),
        ("MBRX", "2025-06-04"),
        ("SBET", "2025-05-30"),
        ("LVWR", "2025-05-28"),
        ("ASST", "2025-05-08"),
        ("OMEX", "2025-04-25"),
        ("SBEV", "2025-04-16"),
        ("ICCT", "2025-04-01"),
        ("MLGO", "2025-04-01"),
        ("IBO", "2025-03-21"),
        ("DWTX", "2025-03-12"),
        ("TRNR", "2025-03-03"),
        ("AIFF", "2025-02-12"),
        ("DOMH", "2025-02-11"),
        ("QNTM", "2025-02-06"),
        ("CRNC", "2025-01-06"),
        ("NITO", "2025-01-03"),
        ("FFIE", "2024-12-31"),
        ("RVSN", "2024-12-30"),
        ("INTZ", "2024-12-30"),
        ("ABAT", "2024-12-27"),
        ("SES", "2024-12-27"),
        ("NUKK", "2024-12-19"),
        ("VRAR", "2024-12-19"),
        ("RR", "2024-12-18"),
        ("LAES", "2024-12-12"),
        ("QUBT", "2024-11-25"),
        ("QMCO", "2024-11-25"),
        ("SPAI", "2024-11-21"),
        ("DXYZ", "2024-11-11"),
        ("GNPX", "2024-10-22"),
        ("BIVI", "2024-10-21"),
        ("VERB", "2024-10-14"),
        ("BOF", "2024-08-29"),
        ("GDC", "2024-08-23"),
        ("FFIE", "2024-08-23"),
        ("APDN", "2024-08-19"),
        ("SERV", "2024-07-30"),
        ("SERV", "2024-07-22"),
        ("ZAPP", "2024-07-09"),
        ("LGVN", "2024-06-13"),
        ("FFIE", "2024-05-17"),
        ("AMC", "2024-05-14"),
        ("GME", "2024-05-14"),
        ("SMFL", "2024-04-23"),
        ("WISA", "2024-04-17"),
        ("AISP", "2024-03-06"),
        ("HOLO", "2024-02-08"),
        ("MINM", "2024-02-01"),
        ("RVSN", "2024-01-30"),
        
    ]
    
    for ticker, date_str in ticker_dates:
        try:
            generate_and_save_chart(ticker, pd.to_datetime(date_str), API_KEY)
            print(f"Saved: {ticker} on {date_str}")
        except Exception as e:
            print(f"Failed: {ticker} on {date_str} -> {e}")
        
    ```
    
    ```python
    import matplotlib
    matplotlib.use("Qt5Agg")
    
    import requests
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from datetime import datetime, timedelta
    from mplfinance.original_flavor import candlestick_ohlc
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from zoneinfo import ZoneInfo
    
    plt.style.use('dark_background')
    
    # ──────────────────────────────────────────────────────────────────────────────
    def idx_formatter_factory(timestamps, fmt):
        def _fmt(x, pos):
            i = int(round(x))
            if i < 0 or i >= len(timestamps):
                return ""
            ts = timestamps[i]
            return ts.strftime(fmt)
        return _fmt
    
    def plot_candles_no_gaps(ax, df, width=0.8, timefmt='%H:%M', shade_prepost=False):
        """Plot candles using integer indices to remove calendar gaps.
           df must include columns: Open, High, Low, Close, timestamp_est.
        """
        if df.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            return None
    
        x = np.arange(len(df), dtype=float)  # integer positions
        tuples = list(zip(x, df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values))
        candlestick_ohlc(ax, tuples, width=width, colorup="white", colordown="red")
    
        # Shading only where bars exist
        if shade_prepost and len(df) > 1:
            for i in range(len(df)-1):
                t = df.index[i].time()
                if (pd.to_datetime("04:00").time() <= t < pd.to_datetime("09:30").time()):
                    ax.axvspan(x[i], x[i+1], color="#444444", alpha=0.6)
                if (pd.to_datetime("16:00").time() <= t < pd.to_datetime("20:00").time()):
                    ax.axvspan(x[i], x[i+1], color="#333333", alpha=0.5)
    
        # Adaptive tick formatting — FIX: use .dt.to_pydatetime() for Series
        ts_list = df["timestamp_est"].dt.to_pydatetime().tolist()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(idx_formatter_factory(ts_list, timefmt)))
    
        # Tight y-lims
        y_max = df["High"].max()
        y_min = df["Low"].min()
        ax.set_ylim(y_min * 0.995, y_max * 1.005)
    
        return x
    
    # ──────────────────────────────────────────────────────────────────────────────
    def generate_and_save_chart(TICKER, TARGET_DATE, API_KEY):
        TICKER = TICKER.upper()
    
        def fetch_polygon_agg(ticker, multiplier, timespan, from_date, to_date, api_key):
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
            params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if "results" not in data:
                raise ValueError(f"Polygon API error or no data found: {data}")
            df = pd.DataFrame(data["results"])
            if df.empty:
                return df
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("US/Eastern")
            df = df.sort_values("timestamp")
            df.set_index("timestamp", inplace=True)
            df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
            return df[["Open", "High", "Low", "Close", "Volume"]]
    
        def add_bands(df, ema_short, ema_long, du1, du2, dl1, dl2, prefix=""):
            if df.empty:
                return
            df[f"{prefix}emaS"] = df["Close"].ewm(span=ema_short, min_periods=0).mean()
            df[f"{prefix}emaL"] = df["Close"].ewm(span=ema_long, min_periods=0).mean()
            tr = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - df["Close"].shift()).abs(),
                (df["Low"] - df["Close"].shift()).abs()
            ], axis=1).max(axis=1)
            df[f"{prefix}ATR_S"] = tr.rolling(ema_short, min_periods=1).mean()
            df[f"{prefix}ATR_L"] = tr.rolling(ema_long, min_periods=1).mean()
            df[f"{prefix}UPP1"] = df[f"{prefix}emaS"] + du1 * df[f"{prefix}ATR_S"]
            df[f"{prefix}UPP2"] = df[f"{prefix}emaS"] + du2 * df[f"{prefix}ATR_S"]
            df[f"{prefix}LOW1"] = df[f"{prefix}emaL"] - dl1 * df[f"{prefix}ATR_L"]
            df[f"{prefix}LOW2"] = df[f"{prefix}emaL"] - dl2 * df[f"{prefix}ATR_L"]
    
        def compress_market_time(df, target_date):
            df = df.copy()
            # Keep only 04:00–20:00 Eastern on the target date, weekdays only
            df = df[((df.index.time >= pd.to_datetime("04:00").time()) & (df.index.time < pd.to_datetime("09:30").time())) | 
                    ((df.index.time >= pd.to_datetime("09:30").time()) & (df.index.time <= pd.to_datetime("20:00").time()))]
            df = df[df.index.dayofweek < 5]
            df = df[df.index.date == target_date.date()]
            df["timestamp_est"] = df.index.tz_convert("US/Eastern")
            return df
    
        def filter_session_hours(df):
            """For HOURLY: remove weekends and hours outside 04:00–20:00 ET."""
            if df.empty:
                return df
            df = df.copy()
            df = df[df.index.dayofweek < 5]
            start_t = pd.to_datetime("04:00").time()
            end_t   = pd.to_datetime("20:00").time()
            mask = (df.index.time >= start_t) & (df.index.time <= end_t)
            df = df[mask]
            df["timestamp_est"] = df.index.tz_convert("US/Eastern")
            return df
    
        def remove_fake_wicks(df):
            if df.empty:
                return df
            return df[~((df["High"] == df["Low"]) & (df["Open"] == df["High"]) & (df["Close"] == df["High"]))]
    
        start_target = pd.to_datetime(TARGET_DATE)
        prev_day = start_target - timedelta(days=5)
        next_day = start_target + timedelta(days=1)
        daily_start = (start_target - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        daily_end = start_target.strftime("%Y-%m-%d")
    
        # Hourly window: 6 days before to target day
        hour_start = (start_target - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        hour_end   = (start_target + pd.Timedelta(days=0)).strftime("%Y-%m-%d")
    
        # Fetch data
        df_5m_raw   = remove_fake_wicks(fetch_polygon_agg(TICKER, 5,  "minute", prev_day.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d"), API_KEY))
        df_15m_raw  = remove_fake_wicks(fetch_polygon_agg(TICKER, 15, "minute", prev_day.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d"), API_KEY))
        df_hour_raw = remove_fake_wicks(fetch_polygon_agg(TICKER, 1,  "hour",   hour_start, hour_end, API_KEY))
        df_daily    = remove_fake_wicks(fetch_polygon_agg(TICKER, 1,  "day",    daily_start, daily_end, API_KEY))
    
        # Indicators
        for d in (df_15m_raw, df_5m_raw, df_hour_raw):
            add_bands(d, 72, 89, 6.9, 9.6, 4.2, 5.5, prefix="l_")
            add_bands(d, 9, 20, 1.0, 0.5, 2.0, 2.5, prefix="s_")
    
        # Daily EMA
        if not df_daily.empty:
            df_daily["ema9"]  = df_daily["Close"].ewm(span=9,  min_periods=0).mean()
            df_daily["ema20"] = df_daily["Close"].ewm(span=20, min_periods=0).mean()
    
        # Intraday compress for target date + VWAP
        df_5m   = compress_market_time(df_5m_raw,   start_target)
        df_15m  = compress_market_time(df_15m_raw,  start_target)
    
        # Hourly — filter to trading session to *remove all non-trading gaps*
        df_hour = filter_session_hours(df_hour_raw)
    
        # Ensure timestamp_est for all
        for d in [df_5m, df_15m, df_daily]:
            if not d.empty:
                d["timestamp_est"] = d.index.tz_convert("US/Eastern")
    
        # VWAPs (per day) for intraday + hourly
        for d in [df_5m, df_15m, df_hour]:
            if d.empty:
                continue
            d["date_only"] = d.index.date
            d["cum_vol"] = d.groupby("date_only")["Volume"].cumsum()
            d["cum_vol_price"] = (d["Close"] * d["Volume"]).groupby(d["date_only"]).cumsum()
            d["VWAP"] = d["cum_vol_price"] / d["cum_vol"]
    
        # Daily up to target
        if not df_daily.empty:
            df_daily = df_daily[df_daily.index.date <= start_target.date()].copy()
    
        # Robust output dir
        default_dir = os.path.expanduser("~/Desktop/lc setups")
        try:
            os.makedirs(default_dir, exist_ok=True)
            output_dir = default_dir
        except Exception as e:
            print(f"Warning: couldn't create Desktop path ({e}). Using current directory.")
            output_dir = os.getcwd()
    
        # Helper to save & close
        def _save_fig(fig, path):
            try:
                fig.savefig(path, dpi=150, bbox_inches='tight')
                print("Saved:", path)
            finally:
                plt.close(fig)
    
        # ── DAILY (own file)
        figD, axD = plt.subplots(figsize=(12, 7))
        xD = plot_candles_no_gaps(axD, df_daily, width=0.6, timefmt='%Y-%m-%d', shade_prepost=False)
        if xD is not None and not df_daily.empty:
            axD.plot(np.arange(len(df_daily)), df_daily["ema9"],  linestyle="--", label="EMA9")
            axD.plot(np.arange(len(df_daily)), df_daily["ema20"], linestyle="--", label="EMA20")
            axD.set_title(f"{TICKER} Daily")
            axD.legend()
        fD = os.path.join(output_dir, f"{TICKER}_{start_target.date()}_daily.png")
        _save_fig(figD, fD)
    
        # ── 1H (own file)
        figH, axH = plt.subplots(figsize=(12, 7))
        xH = plot_candles_no_gaps(axH, df_hour, width=0.6, timefmt='%m-%d %H:%M', shade_prepost=True)
        if xH is not None and not df_hour.empty:
            for col1, col2, alpha in [
                ("s_UPP1", "s_UPP2", 0.3),
                ("s_LOW1", "s_LOW2", 0.3),
                ("l_UPP1", "l_UPP2", 0.25),
                ("l_LOW1", "l_LOW2", 0.25),
            ]:
                axH.fill_between(np.arange(len(df_hour)), df_hour[col1], df_hour[col2], alpha=alpha)
            for series, style, label in [
                ("s_emaS", "--", "EMA9"),
                ("s_emaL", "--", "EMA20"),
                ("VWAP",  "--", "VWAP"),
            ]:
                axH.plot(np.arange(len(df_hour)), df_hour[series], linestyle=style, label=label)
            axH.set_title(f"{TICKER} 1H (session-only, no gaps)")
            axH.legend()
        fH = os.path.join(output_dir, f"{TICKER}_{start_target.date()}_1h.png")
        _save_fig(figH, fH)
    
        # ── 15M (own file)
        fig15, ax15 = plt.subplots(figsize=(12, 7))
        x15 = plot_candles_no_gaps(ax15, df_15m, width=0.6, timefmt='%H:%M', shade_prepost=True)
        if x15 is not None and not df_15m.empty:
            for col1, col2, alpha in [
                ("s_UPP1", "s_UPP2", 0.3),
                ("s_LOW1", "s_LOW2", 0.3),
                ("l_UPP1", "l_UPP2", 0.25),
                ("l_LOW1", "l_LOW2", 0.25),
            ]:
                ax15.fill_between(np.arange(len(df_15m)), df_15m[col1], df_15m[col2], alpha=alpha)
            for series, style, label in [
                ("s_emaS", "--", "EMA9"),
                ("s_emaL", "--", "EMA20"),
                ("VWAP",  "--", "VWAP"),
            ]:
                ax15.plot(np.arange(len(df_15m)), df_15m[series], linestyle=style, label=label)
            ax15.set_title(f"{TICKER} {start_target.date()} 15M")
            ax15.legend()
        f15 = os.path.join(output_dir, f"{TICKER}_{start_target.date()}_15m.png")
        _save_fig(fig15, f15)
    
        # ── 5M (own file)
        fig5, ax5 = plt.subplots(figsize=(12, 7))
        x5 = plot_candles_no_gaps(ax5, df_5m, width=0.6, timefmt='%H:%M', shade_prepost=True)
        if x5 is not None and not df_5m.empty:
            for col1, col2, alpha in [
                ("s_UPP1", "s_UPP2", 0.3),
                ("s_LOW1", "s_LOW2", 0.3),
                ("l_UPP1", "l_UPP2", 0.25),
                ("l_LOW1", "l_LOW2", 0.25),
            ]:
                ax5.fill_between(np.arange(len(df_5m)), df_5m[col1], df_5m[col2], alpha=alpha)
            for series, style, label in [
                ("s_emaS", "--", "EMA9"),
                ("s_emaL", "--", "EMA20"),
                ("VWAP",  "--", "VWAP"),
            ]:
                ax5.plot(np.arange(len(df_5m)), df_5m[series], linestyle=style, label=label)
            ax5.set_title(f"{TICKER} {start_target.date()} 5M")
            ax5.legend()
        f5 = os.path.join(output_dir, f"{TICKER}_{start_target.date()}_5m.png")
        _save_fig(fig5, f5)
    
    # ──────────────────────────────────────────────────────────────────────────────
    API_KEY = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
    ticker_dates = [
        ("NVDA", "2024-06-20"),
        ("NVDA", "2024-03-08"),
        ("NVDA", "2025-07-31"),
        ("NVDA", "2024-07-11"),
        ("MSTR", "2024-11-21"),
        ("PLTR", "2024-12-09"),
    ]
    
    if __name__ == "__main__":
        for ticker, date_str in ticker_dates:
            try:
                generate_and_save_chart(ticker, pd.to_datetime(date_str), API_KEY)
            except Exception as e:
                print(f"Failed: {ticker} on {date_str} -> {e}")
    
    ```
    
- exec
    
    This is code that did some execution.
    
    ```python
    import matplotlib
    matplotlib.use("Qt5Agg")
    
    import requests
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from mplfinance.original_flavor import candlestick_ohlc
    
    plt.style.use('dark_background')
    
    # === CONFIG ===
    API_KEY = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
    TICKER = "SRFM"
    TARGET_DATE = pd.Timestamp("2025-06-25")
    END_DATE = (TARGET_DATE + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    # === DATA FETCH ===
    def fetch_polygon_agg(ticker, multiplier, timespan, from_date, to_date, api_key):
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()
        if "results" not in data:
            raise ValueError("Polygon API error or no data found")
        df = pd.DataFrame(data["results"])
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("US/Eastern")
        df.set_index("timestamp", inplace=True)
        df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]
    
    # === BAND LOGIC ===
    def add_bands(df, ema_short, ema_long, du1, du2, dl1, dl2, prefix=""):
        df[f"{prefix}emaS"] = df["Close"].ewm(span=ema_short).mean()
        df[f"{prefix}emaL"] = df["Close"].ewm(span=ema_long).mean()
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs()
        ], axis=1).max(axis=1)
        df[f"{prefix}ATR_S"] = tr.rolling(ema_short).mean()
        df[f"{prefix}ATR_L"] = tr.rolling(ema_long).mean()
        df[f"{prefix}UPP1"] = df[f"{prefix}emaS"] + du1 * df[f"{prefix}ATR_S"]
        df[f"{prefix}UPP2"] = df[f"{prefix}emaS"] + du2 * df[f"{prefix}ATR_S"]
        df[f"{prefix}LOW1"] = df[f"{prefix}emaL"] - dl1 * df[f"{prefix}ATR_L"]
        df[f"{prefix}LOW2"] = df[f"{prefix}emaL"] - dl2 * df[f"{prefix}ATR_L"]
    
    # === TIME FILTERING ===
    def compress_market_time(df):
        df = df.copy()
        df = df[((df.index.time >= pd.to_datetime("04:00").time()) & (df.index.time < pd.to_datetime("08:00").time())) |
                ((df.index.time >= pd.to_datetime("09:30").time()) & (df.index.time <= pd.to_datetime("20:00").time()))]
        df = df[df.index.dayofweek < 5]
        df["compressed_time"] = pd.Series(range(len(df)), index=df.index)
        df["datetime"] = df.index.tz_convert("US/Eastern")
        return df
    
    # === OHLC FOR MPL ===
    def to_ohlc(df, xcol):
        return list(zip(df[xcol], df["Open"], df["High"], df["Low"], df["Close"]))
    
    # === FETCH ===
    start = (TARGET_DATE - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    full_15m = fetch_polygon_agg(TICKER, 15, "minute", start, END_DATE, API_KEY)
    full_5m = fetch_polygon_agg(TICKER, 5, "minute", start, END_DATE, API_KEY)
    full_2m = fetch_polygon_agg(TICKER, 2, "minute", start, END_DATE, API_KEY)
    
    # === BANDS ===
    add_bands(full_15m, 72, 89, 6.6, 9.6, 6.6, 9.6)
    add_bands(full_15m, 9, 20, 0.1, 0.5, 2.0, 2.5, prefix="s_")
    add_bands(full_5m, 72, 89, 7.9, 6.0, 7.9, 6.0)
    add_bands(full_5m, 9, 20, 1, 0.5, 2.0, 2.5, prefix="s_")
    add_bands(full_2m, 9, 20, 3.3, 2.4, 2.5, 3.3, prefix="r_")
    add_bands(full_2m, 9, 20, 1, 0.5, 2.0, 2.5, prefix="q_")
    
    # === VWAP Calculation (per day grouping) ===
    for df in [full_5m, full_2m]:
        df["date_only"] = df.index.date
        df["cum_vol"] = df.groupby("date_only")["Volume"].cumsum()
        df["cum_vol_price"] = df.groupby("date_only").apply(lambda x: (x["Close"] * x["Volume"]).cumsum()).reset_index(level=0, drop=True)
        df["VWAP"] = df["cum_vol_price"] / df["cum_vol"]
    
    # === COMPRESS ===
    df_15m = compress_market_time(full_15m)
    df_5m = compress_market_time(full_5m)
    df_2m = compress_market_time(full_2m)
    
    # === ENTRY LOGIC ===
    band = df_15m.between_time("07:00", "12:00")
    hit15 = band[(band["High"] >= band["UPP1"]) & (band["Open"] < band["UPP1"])].head(1)
    entries15 = [(hit15["compressed_time"].iloc[0], hit15["High"].iloc[0])] if not hit15.empty else []
    
    entries2 = []
    entries5 = []
    
    if not hit15.empty:
        upp1_price = hit15["UPP1"].iloc[0]
    
        # Find first 2M candle where High >= UPP1
        for i in range(len(df_2m)):
            row = df_2m.iloc[i]
            if row["High"] >= upp1_price:
                t_hit = row.name  # timestamp when 2M actually hits UPP1
                df2_after = df_2m.loc[t_hit:]
                break
        else:
            df2_after = pd.DataFrame()  # fallback if not found
    
        # Proceed with bar break detection only after confirmed hit
        for i in range(2, len(df2_after)):
            curr = df2_after.iloc[i]
            prev = df2_after.iloc[i - 1]
            if curr["High"] == curr["Low"] == curr["Open"] == curr["Close"]:
                continue
            if curr["Low"] < prev["Low"]:
                entries2.append((curr["compressed_time"], prev["Low"]))
                break
    
            df5_after = df_5m.loc[t_hit:]
            prev_low = None
            for i in range(len(df5_after)):
                curr = df5_after.iloc[i]
    
                if curr["High"] == curr["Low"] == curr["Open"] == curr["Close"]:
                    continue
    
                if prev_low is not None and curr["Low"] < prev_low:
                    # map to 2M chart
                    t5 = curr.name
                    df2_after_5m = df_2m.loc[t5:]
                    for k in range(len(df2_after_5m)):
                        mapped = df2_after_5m.iloc[k]
                        entries2.append((mapped["compressed_time"], prev_low))  # plot 5M break on 2M chart
                        break
                    break  # stop after first 5M break is handled
    
                prev_low = curr["Low"]
    
            # fallback in case no break happened — plot the first bar anyway
            if not entries5 and len(df5_after) > 0:
                first = df5_after.iloc[0]
                entries5.append((first["compressed_time"], first["Low"]))
    # === THIRD ENTRY: VWAP break -> Dev band pop (after) -> 2M bar break ===
    entry3 = None
    
    if not hit15.empty and not df_5m.empty and "VWAP" in df_5m.columns:
        # Step 1: First 5M VWAP close break
        vwap_triggered = df_5m.loc[t_hit:]
        vwap_triggered = vwap_triggered[vwap_triggered["Close"] < vwap_triggered["VWAP"]]
    
        if not vwap_triggered.empty:
            vwap_trigger_time = vwap_triggered.index[0]
    
            # Step 2: From NEXT 5M candle onward, look for dev band pop
            post_vwap_after = df_5m[df_5m.index > vwap_trigger_time]
            dev_hit = post_vwap_after[post_vwap_after["High"] >= post_vwap_after["s_UPP2"]]
    
            if not dev_hit.empty:
                dev_hit_time = dev_hit.index[0]
    
                # Step 3: From dev_hit_time, find first 2M bar break
                df2_after = df_2m.loc[dev_hit_time:]
                for i in range(2, len(df2_after)):
                    curr = df2_after.iloc[i]
                    prev = df2_after.iloc[i - 1]
                    if curr["High"] == curr["Low"] == curr["Open"] == curr["Close"]:
                        continue
                    if curr["Low"] < prev["Low"]:
                        entry3 = (curr["compressed_time"], prev["Low"])  # plot 2M break
                        break
    
    # === FOURTH ENTRY: Post-VWAP 5M bar low break -> Dev band pop -> 2M bar break ===
    entry4 = None
    
    if not hit15.empty and not df_5m.empty and "VWAP" in df_5m.columns and "s_UPP2" in df_5m.columns:
        if not vwap_triggered.empty:
            # Start from the bar *after* the VWAP close
            vwap_idx = df_5m.index.get_loc(vwap_trigger_time)
            df5_post_vwap = df_5m.iloc[vwap_idx + 1:]
    
            trigger_idx = None
            for i in range(1, len(df5_post_vwap)):
                curr = df5_post_vwap.iloc[i]
                prev = df5_post_vwap.iloc[i - 1]
                if curr["Close"] < prev["Low"]:
                    trigger_idx = df5_post_vwap.index[i]
                    break
    
            if trigger_idx:
                # Look for dev band hit AFTER trigger
                df5_after_trigger = df_5m[df_5m.index > trigger_idx]
                dev_hit_5m = df5_after_trigger[df5_after_trigger["High"] >= df5_after_trigger["s_UPP2"]]
    
                if not dev_hit_5m.empty:
                    dev_hit_time = dev_hit_5m.index[0]
    
                    # Look for 2M bar break entry — fix timing by starting from index 1
                    df2_after = df_2m.loc[dev_hit_time:]
                    for j in range(1, len(df2_after)):
                        curr = df2_after.iloc[j]
                        prev = df2_after.iloc[j - 1]
                        if curr["High"] == curr["Low"] == curr["Open"] == curr["Close"]:
                            continue
                        if curr["Low"] < prev["Low"]:
                            entry4 = (curr["compressed_time"], prev["Low"])
                            break
    
    # === PLOT ===
    ohlc_15 = to_ohlc(df_15m, "compressed_time")
    ohlc_5 = to_ohlc(df_5m, "compressed_time")
    ohlc_2 = to_ohlc(df_2m, "compressed_time")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(26, 8), sharey=False, sharex=False, constrained_layout=True)
    
    for ax, ohlc, df, title in zip(
        [ax1, ax2, ax3], [ohlc_15, ohlc_5, ohlc_2], [df_15m, df_5m, df_2m], ["15M Chart", "5M Chart", "2M Chart"]):
    
        for i in range(len(df) - 1):
            hour = df.index[i].time()
            if (hour >= pd.to_datetime("04:00").time() and hour < pd.to_datetime("09:30").time()) or \
               (hour >= pd.to_datetime("16:00").time() and hour < pd.to_datetime("20:00").time()):
                ax.axvspan(df["compressed_time"].iloc[i], df["compressed_time"].iloc[i + 1], color="#222222", alpha=0.85)
    
        candlestick_ohlc(ax, ohlc, width=0.5, colorup="white", colordown="red")
    
        for col1, col2, color, alpha in [
            ("UPP1", "UPP2", "lightcoral", 0.5),
            ("LOW1", "LOW2", "lightgreen", 0.5),
            ("s_UPP1", "s_UPP2", "pink", 0.3),
            ("s_LOW1", "s_LOW2", "cyan", 0.3)
        ]:
            if col1 in df.columns and col2 in df.columns:
                ax.fill_between(df["compressed_time"], df[col1], df[col2], color=color, alpha=alpha)
    
            # Plot EMA and VWAP lines
        for col, style, label, color in [
            ("s_emaS", "--", "EMA9", "yellow"),
            ("s_emaL", "--", "EMA20", "lightskyblue"),
            ("VWAP", "--", "VWAP", "purple")  # <- ADD THIS LINE
        ]:
            if col in df.columns:
                ax.plot(df["compressed_time"], df[col], linestyle=style, label=label, color=color)
    
        if title == "2M Chart":
            ax.fill_between(df["compressed_time"], df["r_UPP1"], df["r_UPP2"], color="lightcoral", alpha=0.6)
            ax.fill_between(df["compressed_time"], df["r_LOW1"], df["r_LOW2"], color="green", alpha=0.6)
            ax.fill_between(df["compressed_time"], df["q_UPP1"], df["q_UPP2"], color="red", alpha=0.4)
            ax.fill_between(df["compressed_time"], df["q_LOW1"], df["q_LOW2"], color="lightgreen", alpha=0.4)
            for t, p in entries2:
                ax.plot(t, p, marker="v", color="red", markersize=10)
            if entry3:
                t, p = entry3
                ax.plot(t, p, marker="v", color="#FF4500", markersize=10)
            if entry4:
                t, p = entry4
                ax.plot(t, p, marker="v", color="#FF4500", markersize=10)
    
        #if title == "5M Chart":
         #   for t, p in entries5:
          #      ax.plot(t, p, marker="v", color="red", markersize=10)
    
        if title == "15M Chart":
            for t, p in entries15:
                ax.plot(t, p, marker="v", color="gold", markersize=10)
    
        ax.set_title(title)
        tick_idx = np.linspace(0, len(df) - 1, 10, dtype=int)
        ax.set_xticks(df["compressed_time"].iloc[tick_idx])
        labels = [ts.strftime("%b %d\n%H:%M") for ts in df.index[tick_idx]]
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.xaxis.grid(False)
    
    plt.show()
    
    ```
    

- simple backtest code
    
    ```python
    import pandas as pd
    import numpy as np
    import talib
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
    
    # Load the filtered data
    data_path = '/Users/md/Dropbox/dev/github/hyper-liquid-trading-bots/backtests/gap strategies/BTC-USD-1h-2018-1-01T00_00.csv'
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    
    # Bollinger Band Breakout Strategy (Short Only)
    class BollingerBandBreakoutShort(Strategy):
        window = 21
        num_std = 2.7
        take_profit = 0.05  # 5%
        stop_loss = 0.03    # 3%
    
        def init(self):
            # Calculate Bollinger Bands using TA-Lib
            self.upper_band, self.middle_band, self.lower_band = self.I(talib.BBANDS, self.data.Close, self.window, self.num_std, self.num_std)
    
        def next(self):
            if len(self.data) < self.window:
                return
    
            # Check for breakout below lower band
            if self.data.Close[-1] < self.lower_band[-1] and not self.position:
                self.sell(sl=self.data.Close[-1] * (1 + self.stop_loss),
                          tp=self.data.Close[-1] * (1 - self.take_profit))
    
    # Ensure necessary columns are present and rename correctly
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Unnamed: 6']
    
    # Drop the unnecessary column
    data.drop(columns=['Unnamed: 6'], inplace=True)
    
    # Create and configure the backtest
    bt = Backtest(data, BollingerBandBreakoutShort, cash=100000, commission=0.002)
    
    # Run the backtest with default parameters and print the results
    stats_default = bt.run()
    print("Default Parameters Results:")
    print(stats_default)
    
    # Now perform the optimization
    optimization_results = bt.optimize(
        window=range(10, 20, 5),
        num_std=[round(i, 1) for i in np.arange(1.5, 3.5, 0.1)],
        take_profit=[i / 100 for i in range(1, 7, 1)],  # Optimize TP from 1% to 9%
        stop_loss=[i / 100 for i in range(1, 7, 1)],    # Optimize SL from 1% to 9%
        maximize='Equity Final [$]',
        constraint=lambda param: param.window > 0 and param.num_std > 0  # Ensure valid parameters
    )
    
    # Print the optimization results
    print(optimization_results)
    
    # Print the best optimized values
    print("Best Parameters:")
    print("Window:", optimization_results._strategy.window)
    print("Number of Standard Deviations:", optimization_results._strategy.num_std)
    print("Take Profit:", optimization_results._strategy.take_profit)
    print("Stop Loss:", optimization_results._strategy.stop_loss)
    ```