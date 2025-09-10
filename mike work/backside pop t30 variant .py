"""
Backside Pop Scanner V10 - Fade Pattern Analysis
V10 Enhancements:
- Removed daily trigger requirement
- Added multiple red days detection (consecutive down days)
- Added big outlier fade day detection (high volume + wide range)
- Enhanced fade pattern analysis for better setup identification
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
START_DATE = "2022-01-01"
END_DATE = "2025-09-01"
MAX_WORKERS = 16

TICKER_UNIVERSE = ['HOOD', 'MSTR', 'SMCI','COIN', 'IBIT', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META',"SOXL", "MRVL", "TGT", "DOCU", "ZM", "DIS", "NFLX", "AMC", "RKT", "SNAP", "RBLX", "META", "SE", "NVDA", 
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

class BacksidePopScannerV10:
    """Backside Pop Scanner V10 with Fade Pattern Analysis"""
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        self.ticker_universe = TICKER_UNIVERSE
        self.max_workers = MAX_WORKERS
        
        # V10 Enhanced scan thresholds
        self.scan_thresholds = {
            'min_trend_atr': 8.0,           
            'min_gap_atr': 0.70,             
            'min_extension_atr': 2.0,       
            'min_range_close_pct': 70.0,    
            'min_volume_multiple': 1.0,     
            'min_change_atr': 0.1,         
            'max_downtrend_slope': -0.50,    
            'min_ema_extension_pct': 10.0,  
            'min_fade_atr': 2.0,            
            'min_days_since_high': 1.0,       
            'max_days_since_high': 30.0,      
            'min_price': 10.0,              
            'max_price': 1000.0,            
            'min_volume': 10_000_000,
            'require_dev_band_upper': True,   # D0 open/high must be above 0.5 deviation band 
            'require_d_minus_1_green': True,  # D-1 must be green (close > open)
            
            # V10: Fade Pattern Requirements (replaces daily trigger)
            'min_red_days_consecutive': 3,    # Need at least 2+ consecutive red days
            'outlier_volume_multiple': 2,   # OR volume 2.5x+ average for outlier fade day
            'outlier_range_atr': 2,         # OR range 3.0+ ATR for outlier fade day
            'outlier_fade_atr': 2.0,          # OR fade 4.0+ ATR for outlier fade day
            'fade_lookback_days': 10          # Look back N days for fade analysis
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'backside-scanner-v10'})

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
        """Calculate indicators with V10 enhanced 9/20 EMA deviation bands"""
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
        df['range_atr'] = df['range_dollars'] / df['atr']  # V10: Added range in ATR
        df['close_range'] = np.where(df['range_dollars'] > 0, 
                                   (df['close'] - df['low']) / df['range_dollars'], 0)
        df['price_change'] = df['close'] - df['pdc']
        df['price_change_atr'] = df['price_change'] / df['atr']
        df['open_to_prev_low'] = df['open'] - df['low'].shift(1)
        df['extension_atr'] = df['open_to_prev_low'] / df['atr']
        df['avg_volume_20'] = df['volume'].rolling(20, min_periods=5).mean()
        df['volume_multiple'] = df['volume'] / df['avg_volume_20']
        
        # V10: Red/Green day classification
        df['is_red_day'] = df['close'] < df['open']
        df['is_green_day'] = df['close'] > df['open']
        
        # V10 Enhanced 9/20 EMA Deviation Bands
        df['ATR_9'] = df['true_range'].rolling(9, min_periods=3).mean()
        df['dev_band_upper_1'] = df['ema_9'] + 1.0 * df['ATR_9']
        df['dev_band_upper_2'] = df['ema_9'] + 0.35 * df['ATR_9']  # V10: 0.5 band for sensitivity
        df['dev_band_lower_1'] = df['ema_20'] - 2.0 * df['ATR_9']
        df['dev_band_lower_2'] = df['ema_20'] - 2.4 * df['ATR_9']
        
        # V10: Use 0.5 deviation band for better sensitivity
        df['open_above_dev_upper'] = df['open'] > df['dev_band_upper_2']
        df['high_above_dev_upper'] = df['high'] > df['dev_band_upper_2']
        
        # T30 Range Filter - Don't gap too high in recent range
        # Calculate 30-day rolling high and low
        df['rolling_high_30'] = df['high'].rolling(30, min_periods=10).max()
        df['rolling_low_30'] = df['low'].rolling(30, min_periods=10).min()
        
        # Calculate the range from high to low
        df['range_30_days'] = df['rolling_high_30'] - df['rolling_low_30']
        
        # Calculate where Day 0 open sits in this range (0 = at low, 100 = at high)
        df['open_position_in_range_pct'] = np.where(
            df['range_30_days'] > 0,
            ((df['open'] - df['rolling_low_30']) / df['range_30_days'] * 100),
            50.0  # Default to middle if no range
        )
        
        # Filter: D0 open should NOT be above 60th percentile (40% down from highs)
        df['open_not_too_high_in_range'] = df['open_position_in_range_pct'] <= 80.0
        
        return df

    def analyze_fade_pattern(self, df, setup_idx, euphoric_high):
        """V10: Analyze fade pattern - multiple red days OR big outlier fade day"""
        if setup_idx < self.scan_thresholds['fade_lookback_days']:
            return False, None, None
        
        # Look back from euphoric high to setup day
        euphoric_idx = None
        for i in range(setup_idx):
            if abs(df.iloc[i]['high'] - euphoric_high) < 0.01:  # Find euphoric high row
                euphoric_idx = i
                break
        
        if euphoric_idx is None or euphoric_idx >= setup_idx - 1:
            return False, None, None
        
        # Analyze fade from euphoric high to setup day
        fade_data = df.iloc[euphoric_idx:setup_idx].copy()
        if len(fade_data) < 2:
            return False, None, None
        
        # Method 1: Check for consecutive red days
        consecutive_reds = 0
        max_consecutive_reds = 0
        red_streak_start = None
        
        for i, row in fade_data.iterrows():
            if row['is_red_day']:
                if consecutive_reds == 0:
                    red_streak_start = row['date']
                consecutive_reds += 1
                max_consecutive_reds = max(max_consecutive_reds, consecutive_reds)
            else:
                consecutive_reds = 0
        
        has_multiple_red_days = max_consecutive_reds >= self.scan_thresholds['min_red_days_consecutive']
        
        # Method 2: Check for outlier fade day (high volume + wide range + big fade)
        has_outlier_fade_day = False
        outlier_fade_info = None
        
        for i, row in fade_data.iterrows():
            if pd.isna(row['volume_multiple']) or pd.isna(row['range_atr']) or pd.isna(row['atr']):
                continue
                
            # Calculate fade from euphoric high to this day's close
            fade_from_high = (euphoric_high - row['close']) / row['atr']
            
            # Check outlier criteria
            is_high_volume = row['volume_multiple'] >= self.scan_thresholds['outlier_volume_multiple']
            is_wide_range = row['range_atr'] >= self.scan_thresholds['outlier_range_atr']
            is_big_fade = fade_from_high >= self.scan_thresholds['outlier_fade_atr']
            
            # Need at least 2 of 3 outlier conditions
            outlier_conditions = sum([is_high_volume, is_wide_range, is_big_fade])
            
            if outlier_conditions >= 2:
                has_outlier_fade_day = True
                outlier_fade_info = {
                    'date': row['date'],
                    'volume_multiple': row['volume_multiple'],
                    'range_atr': row['range_atr'],
                    'fade_atr': fade_from_high,
                    'conditions_met': outlier_conditions
                }
                break
        
        # Return True if either method is satisfied
        fade_pattern_valid = has_multiple_red_days or has_outlier_fade_day
        
        fade_info = {
            'multiple_red_days': has_multiple_red_days,
            'max_consecutive_reds': max_consecutive_reds,
            'red_streak_start': red_streak_start,
            'outlier_fade_day': has_outlier_fade_day,
            'outlier_info': outlier_fade_info
        }
        
        return fade_pattern_valid, fade_info, "multiple_red" if has_multiple_red_days else "outlier_fade"

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

    def scan_single_ticker_v10(self, symbol, start_date, end_date):
        """V10 Enhanced scan with fade pattern analysis (no daily trigger)"""
        # Fetch data with extended lookback
        extended_start = (pd.to_datetime(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
        df = self.fetch_daily_data_cached(symbol, extended_start, end_date)
        
        if df.empty or len(df) < 50:
            return []
        
        # Calculate indicators with V10 deviation bands
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
            
            # V10 CRITICAL FILTERS
            
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
            
            # 3. V10: D0 open/high must be above 0.5 deviation band (more sensitive)
            d0_open_above_dev = d_0['open_above_dev_upper'] if not pd.isna(d_0['open_above_dev_upper']) else False
            d0_high_above_dev = d_0['high_above_dev_upper'] if not pd.isna(d_0['high_above_dev_upper']) else False
            
            if not (d0_open_above_dev or d0_high_above_dev):
                continue
            
            # 3b. T30 Range Filter: D0 open must NOT be above 60th percentile of recent range
            open_not_too_high = d_0['open_not_too_high_in_range'] if not pd.isna(d_0['open_not_too_high_in_range']) else True
            
            if not open_not_too_high:
                continue  # Skip if gapping too high in 30-day range
            
            # 4. Find trend and euphoric high
            trend_start_date, trend_start_price, euphoric_date, euphoric_high = \
                self.find_trend_and_euphoric_high_fast(df, idx)
            
            if not all([trend_start_date, trend_start_price, euphoric_date, euphoric_high]):
                continue
            
            # 5. V10: Check fade pattern (replaces daily trigger)
            fade_valid, fade_info, fade_type = self.analyze_fade_pattern(df, idx, euphoric_high)
            
            if not fade_valid:
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
            
            # V10 Enhanced scan criteria
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
                # T30 Range metrics for analysis
                open_range_pct = d_0['open_position_in_range_pct'] if not pd.isna(d_0['open_position_in_range_pct']) else None
                range_high_30 = d_0['rolling_high_30'] if not pd.isna(d_0['rolling_high_30']) else None
                range_low_30 = d_0['rolling_low_30'] if not pd.isna(d_0['rolling_low_30']) else None
                
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
                    'fade_type': fade_type,
                    'consecutive_reds': fade_info['max_consecutive_reds'] if fade_info['multiple_red_days'] else 0,
                    'outlier_volume': round(fade_info['outlier_info']['volume_multiple'], 2) if fade_info['outlier_fade_day'] and fade_info['outlier_info'] else None,
                    'outlier_range_atr': round(fade_info['outlier_info']['range_atr'], 2) if fade_info['outlier_fade_day'] and fade_info['outlier_info'] else None,
                    'dev_band_upper_05': round(d_0['dev_band_upper_2'], 2) if not pd.isna(d_0['dev_band_upper_2']) else None,
                    'd0_open': round(d_0['open'], 2),
                    'd0_high': round(d_0['high'], 2),
                    # T30 Range Analysis
                    'open_range_position_pct': round(open_range_pct, 1) if open_range_pct is not None else None,
                    't30_range_high': round(range_high_30, 2) if range_high_30 is not None else None,
                    't30_range_low': round(range_low_30, 2) if range_low_30 is not None else None,
                    'scanner_version': 'V10_T30'
                }
                setups.append(setup)
        
        return setups

    def run_scan_v10(self, start_date=None, end_date=None, tickers=None):
        """V10 Enhanced scanner with fade pattern analysis"""
        start_date = start_date or START_DATE
        end_date = end_date or END_DATE
        tickers = tickers or self.ticker_universe
        
        print(f"🚀 BACKSIDE POP SCANNER V10 T30 - FADE PATTERN ANALYSIS + RANGE FILTER")
        print(f"📅 Scanning {len(tickers)} tickers from {start_date} to {end_date}")
        print(f"🔧 V10 T30 Enhancements:")
        print(f"   • Removed daily trigger requirement")
        print(f"   • Added multiple red days detection ({self.scan_thresholds['min_red_days_consecutive']}+ consecutive)")
        print(f"   • Added outlier fade day detection (volume {self.scan_thresholds['outlier_volume_multiple']}x + range {self.scan_thresholds['outlier_range_atr']}+ ATR)")
        print(f"   • Enhanced fade pattern analysis for better setup identification")
        print(f"   • T30 Range Filter: D0 open must be ≤60th percentile of 30-day range (≤40% down from highs)")
        print("=" * 80)
        
        all_setups = []
        processed = 0
        errors = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.scan_single_ticker_v10, ticker, start_date, end_date): ticker 
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
        print(f"🎯 V10 T30 SCAN COMPLETE")
        print(f"📊 Results: {len(all_setups)} setups found (T30 range filtered)")
        print(f"⚠️  Errors: {errors} tickers failed")
        
        if all_setups:
            df_results = pd.DataFrame(all_setups)
            df_results = df_results.sort_values(['date', 'trend_atr'], ascending=[False, False])
            
            print(f"\n🏆 ALL V10 T30 SETUPS:")
            display_cols = ['symbol', 'date', 'trend_atr', 'gap_atr', 'extension_atr', 'fade_type', 'open_range_position_pct', 'consecutive_reds']
            print(df_results[display_cols].to_string(index=False))
            
            # V10: Fade pattern breakdown
            fade_counts = df_results['fade_type'].value_counts()
            print(f"\n📊 FADE PATTERN BREAKDOWN:")
            for fade_type, count in fade_counts.items():
                print(f"   {fade_type}: {count} setups")
            
            return df_results
        else:
            print("No setups found matching V10 criteria.")
            return pd.DataFrame()

# Example Usage
if __name__ == "__main__":
    # Initialize V10 scanner
    scanner = BacksidePopScannerV10()
    
    # Run V10 enhanced scan
    results = scanner.run_scan_v10()
    
    if not results.empty:
        # Save results to CSV
        results.to_csv('backside_pop_v10_t30_results.csv', index=False)
        print(f"\n💾 T30 Range Filtered results saved to: backside_pop_v10_t30_results.csv")
        
        # Show range position summary
        if 'open_range_position_pct' in results.columns:
            avg_range_pos = results['open_range_position_pct'].mean()
            print(f"📊 Average D0 open position in T30 range: {avg_range_pos:.1f}% (all ≤60%)")
            print(f"📊 Range: {results['open_range_position_pct'].min():.1f}% - {results['open_range_position_pct'].max():.1f}%")