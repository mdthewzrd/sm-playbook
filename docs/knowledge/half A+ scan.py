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





