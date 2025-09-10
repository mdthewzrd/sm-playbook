"""
Backside Pop Scanner V9 - Enhanced Daily Trigger & Deviation Bands
V9 Enhancements:
- Fixed daily trigger: close below same day low OR previous candle low
- Enhanced deviation band detection using 0.5 multiplier (more sensitive)
- Improved indexing for previous candle lookups
- All V8 functionality maintained with bug fixes
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# CONFIGURATION
API_KEY = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
BASE_URL = "https://api.polygon.io"

# DATE RANGE AND TICKERS - EDIT THESE
START_DATE = "2024-01-01"
END_DATE = "2025-09-01"
MAX_WORKERS = 16

TICKER_UNIVERSE = ['HOOD', 'MSTR', 'SMCI', 'IBIT', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META',"SOXL", "MRVL", "TGT", "DOCU", "ZM", "DIS", "NFLX", "AMC", "RKT", "SNAP", "RBLX", "META", "SE", "NVDA", 
               "SMCI", "MSTR", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "AMD", "NFLX", "INTC", "BABA", "BA", 
               "PYPL", "QCOM", "ORCL", "T", "CSCO", "VZ", "KO", "PEP", "MRK", "PFE", "ABBV", "JNJ", "DIS", "CRM", 
               "BAC", "C", "JPM", "WMT", "CVX", "XOM", "COP", "RTX", "SPGI", "GS", "TGT", "HD", "LOW", "COST", "UNH", 
               "NEE", "NKE", "LMT", "HON", "CAT", "MMM", "LIN", "ADBE", "AVGO", "TXN", "ACN", "UPS", "BLK", "PM", "MO", 
               "ELV", "VRTX", "ZTS", "NOW", "ISRG", "PLD", "MS", "MDT", "WM", "GE", "IBM", "BKNG", "FDX", "ADP", "EQIX", 
               "DHR", "SNPS", "REGN", "SYK", "TMO", "CVS", "INTU", "SCHW", "CI", "APD", "SO", "MMC", "ICE", "FIS", 
               "ADI", "CSX", "LRCX", "GILD", "RIVN", "LCID", "PLTR", "SNOW", "SPY", "QQQ", "DIA", "IWM", "TQQQ", 
               "SQQQ", "ARKK", "SOXL", "LABU", "TECL", "UVXY", "XLE", "XLK", "XLF", "IBB", "KWEB", "TAN", "XOP", 
               "EEM", "HYG", "EFA", "USO", "GLD", "SLV", "BITO", "RIOT", "MARA", "COIN", "SQ", "AFRM", "DKNG", "ZM", 
               "DOCU", "SHOP", "UPST", "CLF", "AA", "F", "GM", "ROKU", "WBD", "WBA", "PARA", "PINS", "LYFT", "SNAP", 
               "BYND", "DJT", "RDDT", "GME", "VKTX", "APLD", "KGEI", "INOD", "LMB", "AMR", "PMTS", "SAVA", "CELH", 
               "ESOA", "IVT", "MOD", "SKYE", "AR", "VIXY", "TECS", "LABD", "SPXS", "SPXL", "DRV", "TZA", "FAZ", "WEBS", 
               "PSQ", "SDOW", "GME", "VKTX", "MSTU", "MSTZ", "NFLU", "BTCL", "BTCZ", "ETU", "ETQ", "SOXL", "TECL", 
               "FAS", "SPXL", "TNA", "NUGT", "TSLL", "NVDU", "AMZU", "MSFU", "NFLU", "TQQQ", "SPXL", "SOXL", "FAS", 
               "TECL", "UVXY", "UVIX"]

class BacksidePopScannerV9:
    """Backside Pop Scanner V9 with Enhanced Daily Trigger & Deviation Bands"""
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        self.ticker_universe = TICKER_UNIVERSE
        self.max_workers = MAX_WORKERS
        
        # V9 Enhanced scan thresholds
        self.scan_thresholds = {
            'min_trend_atr': 6.0,           
            'min_gap_atr': 1.0,             
            'min_extension_atr': 2.0,       
            'min_range_close_pct': 70.0,    
            'min_volume_multiple': 1.0,     
            'min_change_atr': 0.5,         
            'max_downtrend_slope': -1.0,    
            'min_ema_extension_pct': 20.0,  
            'min_fade_atr': 2.0,            
            'min_days_since_high': 1.0,       
            'max_days_since_high': 30.0,      
            'min_price': 10.0,              
            'max_price': 1000.0,            
            'min_volume': 1_000_000,
            'require_dev_band_upper': True,   # D0 open/high must be above 0.5 deviation band 
            'require_daily_trigger': True,    # Must have close below low after daily high is set
            'require_d_minus_1_green': True   # D-1 must be green (close > open)
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'backside-scanner-v9'})

    def fetch_daily_data_cached(self, symbol, start_date, end_date):
        """Fetch daily OHLCV data from Polygon with caching"""
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {'adjusted': 'true', 'sort': 'asc', 'apikey': self.api_key}
        
        try:
            time.sleep(0.012)  # Rate limiting
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
                    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                    return df[['date', 'open', 'high', 'low', 'close', 'volume']]
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def calculate_indicators_with_dev_bands(self, df):
        """Calculate indicators with V9 enhanced 9/20 EMA deviation bands"""
        if df.empty or len(df) < 10:
            return df
            
        df = df.copy().sort_values('date').reset_index(drop=True)
        
        # Vectorized calculations
        df['pdc'] = df['close'].shift(1)
        df['true_range'] = np.maximum.reduce([
            df['high'] - df['low'],
            np.abs(df['high'] - df['pdc']),
            np.abs(df['low'] - df['pdc'])
        ])
        
        # Adaptive ATR calculation
        if len(df) >= 200:
            df['atr'] = df['true_range'].rolling(200, min_periods=50).mean()
        elif len(df) >= 50:
            df['atr'] = df['true_range'].rolling(50, min_periods=14).mean()
        else:
            df['atr'] = df['true_range'].rolling(14, min_periods=5).mean()
        
        # EMAs (vectorized)
        df['ema_9'] = df['close'].ewm(span=9, min_periods=5).mean()
        df['ema_20'] = df['close'].ewm(span=20, min_periods=10).mean()
        df['ema_89'] = df['close'].ewm(span=89, min_periods=20).mean()
        
        # Vectorized gap, range, and volume calculations
        df['gap_dollars'] = df['open'] - df['pdc']
        df['gap_atr'] = df['gap_dollars'] / df['atr']
        df['range_dollars'] = df['high'] - df['low']
        df['close_range'] = np.where(df['range_dollars'] > 0, 
                                   (df['close'] - df['low']) / df['range_dollars'], 0)
        df['price_change'] = df['close'] - df['pdc']
        df['price_change_atr'] = df['price_change'] / df['atr']
        df['open_to_prev_low'] = df['open'] - df['low'].shift(1)
        df['extension_atr'] = df['open_to_prev_low'] / df['atr']
        df['avg_volume_20'] = df['volume'].rolling(20, min_periods=5).mean()
        df['volume_multiple'] = df['volume'] / df['avg_volume_20']
        
        # V9 Enhanced 9/20 EMA Deviation Bands
        df['ATR_9'] = df['true_range'].rolling(9, min_periods=3).mean()
        df['dev_band_upper_1'] = df['ema_9'] + 1.0 * df['ATR_9']
        df['dev_band_upper_2'] = df['ema_9'] + 0.5 * df['ATR_9']  # V9: More sensitive 0.5 band
        df['dev_band_lower_1'] = df['ema_20'] - 2.0 * df['ATR_9']
        df['dev_band_lower_2'] = df['ema_20'] - 2.4 * df['ATR_9']
        
        # V9: Use 0.5 deviation band for better sensitivity
        df['open_above_dev_upper'] = df['open'] > df['dev_band_upper_2']
        df['high_above_dev_upper'] = df['high'] > df['dev_band_upper_2']
        
        return df

    def find_daily_high_and_trigger(self, df, setup_idx):
        """V9 Enhanced: Find daily high and check for daily trigger (close below low or previous candle low)"""
        if setup_idx < 20:
            return None, None, False, None, None
            
        # Look back for the trend period to find the daily high
        lookback_data = df.iloc[max(0, setup_idx - 120):setup_idx].copy()
        if lookback_data.empty:
            return None, None, False, None, None
        
        # Find the highest high in the lookback period
        daily_high_idx = lookback_data['high'].idxmax()
        daily_high_date = lookback_data.loc[daily_high_idx, 'date']
        daily_high_value = lookback_data.loc[daily_high_idx, 'high']
        daily_high_low = lookback_data.loc[daily_high_idx, 'low']  # The low of the daily high bar
        
        # V9 Enhanced: Check for daily trigger - close below the low of the daily high bar OR previous candle's low
        # Can happen on the same day as the high OR after the high is set
        
        # First check if the high day itself has a trigger
        daily_high_close = lookback_data.loc[daily_high_idx, 'close']
        
        # Get previous candle's low (day before the high day) - V9 Fixed indexing
        # Find the position of the high day in the lookback_data
        high_day_position = lookback_data.index.get_loc(daily_high_idx)
        if high_day_position > 0:
            prev_candle_idx = lookback_data.index[high_day_position - 1]
            prev_candle_low = lookback_data.loc[prev_candle_idx, 'low']
        else:
            prev_candle_low = float('inf')  # No previous candle
        
        # V9: Trigger can be: close < same day low OR close < previous candle low
        same_day_trigger = daily_high_close < daily_high_low
        prev_candle_trigger = daily_high_close < prev_candle_low
        
        if same_day_trigger or prev_candle_trigger:
            # Trigger happened on the same day as the high
            daily_trigger = True
            trigger_date = daily_high_date
            trigger_close = daily_high_close
        else:
            # Check for trigger on days AFTER the high is set
            post_high_data = df.iloc[daily_high_idx + 1:setup_idx]
            if post_high_data.empty:
                daily_trigger = False
                trigger_date = None
                trigger_close = None
            else:
                # V9: Look for any close below the low of the daily high bar OR previous candle low
                trigger_bars = post_high_data[
                    (post_high_data['close'] < daily_high_low) | 
                    (post_high_data['close'] < prev_candle_low)
                ]
                
                if trigger_bars.empty:
                    daily_trigger = False
                    trigger_date = None
                    trigger_close = None
                else:
                    daily_trigger = True
                    first_trigger_idx = trigger_bars.index[0]
                    trigger_date = trigger_bars.loc[first_trigger_idx, 'date']
                    trigger_close = trigger_bars.loc[first_trigger_idx, 'close']
        
        return daily_high_date, daily_high_value, daily_trigger, trigger_date, trigger_close

    def find_trend_and_euphoric_high_fast(self, df, setup_idx):
        """Optimized trend and euphoric high detection"""
        if setup_idx < 20:
            return None, None, None, None
            
        # Look for 9/20 EMA cross (vectorized)
        pre_setup = df.iloc[:setup_idx].copy()
        ema_9_above_20 = (pre_setup['ema_9'] > pre_setup['ema_20']).astype(int)
        cross_signal = ema_9_above_20.diff()
        
        # Find last bullish cross
        cross_indices = pre_setup[cross_signal == 1].index
        if len(cross_indices) == 0:
            return None, None, None, None
            
        trend_start_idx = cross_indices[-1]
        trend_start_date = pre_setup.loc[trend_start_idx, 'date']
        trend_start_price = pre_setup.loc[trend_start_idx, 'close']
        
        # Find euphoric high (maximum high since trend start)
        trend_data = pre_setup.iloc[trend_start_idx:]
        if trend_data.empty:
            return None, None, None, None
            
        euphoric_idx = trend_data['high'].idxmax()
        euphoric_date = trend_data.loc[euphoric_idx, 'date']
        euphoric_high = trend_data.loc[euphoric_idx, 'high']
        
        return trend_start_date, trend_start_price, euphoric_date, euphoric_high

    def scan_single_ticker_v9(self, symbol, start_date, end_date):
        """V9 Enhanced scan for single ticker with all fixes"""
        # Fetch data with extended lookback
        extended_start = (pd.to_datetime(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
        df = self.fetch_daily_data_cached(symbol, extended_start, end_date)
        
        if df.empty or len(df) < 50:
            return []
        
        # Calculate indicators with V9 deviation bands
        df = self.calculate_indicators_with_dev_bands(df)
        
        # Find setups in date range
        target_start = pd.to_datetime(start_date).date()
        target_end = pd.to_datetime(end_date).date()
        setups = []
        
        for idx in range(1, len(df)):
            current_date = df.iloc[idx]['date']
            if not (target_start <= current_date <= target_end):
                continue
                
            prev_idx = idx - 1
            d_minus_1 = df.iloc[prev_idx]
            d_0 = df.iloc[idx]
            
            # V9 CRITICAL FILTERS
            
            # 1. Basic filters
            if not (self.scan_thresholds['min_price'] <= d_0['close'] <= self.scan_thresholds['max_price']):
                continue
            if d_minus_1['volume'] < self.scan_thresholds['min_volume']:
                continue
            if pd.isna(d_minus_1['atr']) or d_minus_1['atr'] <= 0:
                continue
            
            # 2. D-1 must be green
            if d_minus_1['close'] <= d_minus_1['open']:
                continue
            
            # 3. V9: D0 open/high must be above 0.5 deviation band (more sensitive)
            d0_open_above_dev = d_0['open_above_dev_upper'] if not pd.isna(d_0['open_above_dev_upper']) else False
            d0_high_above_dev = d_0['high_above_dev_upper'] if not pd.isna(d_0['high_above_dev_upper']) else False
            
            if not (d0_open_above_dev or d0_high_above_dev):
                continue
            
            # 4. V9 Enhanced: Daily trigger requirement (fixed logic)
            daily_high_date, daily_high_value, daily_trigger, trigger_date, trigger_close = \
                self.find_daily_high_and_trigger(df, idx)
            
            if not daily_trigger:
                continue
            
            # 5. Find trend and euphoric high
            trend_start_date, trend_start_price, euphoric_date, euphoric_high = \
                self.find_trend_and_euphoric_high_fast(df, idx)
            
            if not all([trend_start_date, trend_start_price, euphoric_date, euphoric_high]):
                continue
            
            # Calculate setup metrics
            days_since_high = (d_minus_1['date'] - euphoric_date).days
            trend_atr_multiples = (euphoric_high - trend_start_price) / d_minus_1['atr']
            fade_atr = (euphoric_high - d_minus_1['close']) / d_minus_1['atr']
            downtrend_slope = (d_minus_1['close'] - euphoric_high) / max(days_since_high, 1)
            
            gap_atr = d_0['gap_atr'] if not pd.isna(d_0['gap_atr']) else 0
            extension_atr = d_0['extension_atr'] if not pd.isna(d_0['extension_atr']) else 0
            range_close_pct = d_minus_1['close_range'] * 100
            volume_multiple = d_minus_1['volume_multiple'] if not pd.isna(d_minus_1['volume_multiple']) else 0
            change_atr = abs(d_minus_1['price_change_atr']) if not pd.isna(d_minus_1['price_change_atr']) else 0
            ema_extension_pct = ((d_0['open'] - d_0['ema_89']) / d_0['ema_89']) * 100 if d_0['ema_89'] > 0 else 0
            
            # V9 Enhanced scan criteria
            scan_criteria = all([
                trend_atr_multiples >= self.scan_thresholds['min_trend_atr'],
                gap_atr >= self.scan_thresholds['min_gap_atr'],
                extension_atr >= self.scan_thresholds['min_extension_atr'],
                range_close_pct >= self.scan_thresholds['min_range_close_pct'],
                volume_multiple >= self.scan_thresholds['min_volume_multiple'],
                change_atr >= self.scan_thresholds['min_change_atr'],
                downtrend_slope <= self.scan_thresholds['max_downtrend_slope'],
                ema_extension_pct >= self.scan_thresholds['min_ema_extension_pct'],
                fade_atr >= self.scan_thresholds['min_fade_atr'],
                self.scan_thresholds['min_days_since_high'] <= days_since_high <= self.scan_thresholds['max_days_since_high']
            ])
            
            if scan_criteria:
                setup = {
                    'symbol': symbol,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'trend_atr': round(trend_atr_multiples, 2),
                    'gap_atr': round(gap_atr, 2),
                    'extension_atr': round(extension_atr, 2),
                    'range_close_pct': round(range_close_pct, 1),
                    'volume_multiple': round(volume_multiple, 2),
                    'change_atr': round(change_atr, 2),
                    'fade_atr': round(fade_atr, 2),
                    'days_since_high': days_since_high,
                    'trend_start': trend_start_date.strftime('%Y-%m-%d'),
                    'euphoric_high': euphoric_date.strftime('%Y-%m-%d'),
                    'daily_high_date': daily_high_date.strftime('%Y-%m-%d') if daily_high_date else None,
                    'daily_trigger_date': trigger_date.strftime('%Y-%m-%d') if trigger_date else None,
                    'dev_band_upper_05': round(d_0['dev_band_upper_2'], 2) if not pd.isna(d_0['dev_band_upper_2']) else None,
                    'd0_open': round(d_0['open'], 2),
                    'd0_high': round(d_0['high'], 2),
                    'scanner_version': 'V9'
                }
                setups.append(setup)
        
        return setups

    def run_scan_v9(self, start_date=None, end_date=None, tickers=None):
        """V9 Enhanced scanner with all fixes and improvements"""
        start_date = start_date or START_DATE
        end_date = end_date or END_DATE
        tickers = tickers or self.ticker_universe
        
        print(f"🚀 BACKSIDE POP SCANNER V9 - ENHANCED DAILY TRIGGER & DEVIATION BANDS")
        print(f"📅 Scanning {len(tickers)} tickers from {start_date} to {end_date}")
        print(f"🔧 V9 Enhancements:")
        print(f"   • Fixed daily trigger: close < same day low OR previous candle low")
        print(f"   • Enhanced 0.5 deviation band detection (more sensitive)")
        print(f"   • Improved indexing for previous candle lookups")
        print(f"   • All V8 functionality with bug fixes")
        print("=" * 80)
        
        all_setups = []
        processed = 0
        errors = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.scan_single_ticker_v9, ticker, start_date, end_date): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    setups = future.result()
                    if setups:
                        all_setups.extend(setups)
                        print(f"✅ {ticker}: {len(setups)} setups found")
                    else:
                        print(f"⚪ {ticker}: No setups")
                except Exception as e:
                    print(f"❌ {ticker}: Error - {str(e)[:50]}")
                    errors += 1
                
                processed += 1
                if processed % 20 == 0:
                    print(f"📊 Progress: {processed}/{len(tickers)} ({processed/len(tickers)*100:.1f}%)")
        
        print("\n" + "=" * 80)
        print(f"🎯 V9 SCAN COMPLETE")
        print(f"📊 Results: {len(all_setups)} setups found")
        print(f"⚠️  Errors: {errors} tickers failed")
        
        if all_setups:
            df_results = pd.DataFrame(all_setups)
            df_results = df_results.sort_values(['date', 'trend_atr'], ascending=[False, False])
            
            print(f"\n🏆 TOP V9 SETUPS:")
            print(df_results.head(10)[['symbol', 'date', 'trend_atr', 'gap_atr', 'extension_atr', 'daily_trigger_date']].to_string(index=False))
            
            return df_results
        else:
            print("No setups found matching V9 criteria.")
            return pd.DataFrame()

# Example Usage
if __name__ == "__main__":
    # Initialize V9 scanner
    scanner = BacksidePopScannerV9()
    
    # Run V9 enhanced scan
    results = scanner.run_scan_v9()
    
    if not results.empty:
        print(f"\n💾 Save results with:")
        print(f"results.to_csv('backside_pop_v9_results.csv', index=False)")