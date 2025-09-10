"""
Backside Pop Scanner V11 - B-Grade Setups (Non-A+ Names)
V11 Enhancements:
- Excludes A+ tickers from V10 to avoid duplicates
- Loosened parameters for B-grade setups
- Focuses on lesser-known names with potential
- Based on V10 fade pattern analysis but with relaxed thresholds
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
END_DATE = "2025-09-03"
MAX_WORKERS = 16

# FULL TICKER UNIVERSE (ALL NAMES)
TICKER_UNIVERSE = ['HOOD', 'COIN', 'MSTR','DJT','BABA','F','GME', 'SMCI', 'IBIT', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META',"SOXL", "MRVL", "TGT", "DOCU", "ZM", "DIS", "NFLX", "AMC", "RKT", "SNAP", "RBLX", "META", "SE", "NVDA", 
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
               "TECL", "UVXY", "UVIX", "AI", "PATH", "U", "SOFI", "OPEN", "WISH", "CLOV", "SPCE", "NKLA", "WKHS", "GOEV", "RIDE", "SKLZ", "DDOG", "CRWD", "NET", "OKTA", "ZS", "ESTC", "MDB", "GTLB",
               "BILL", "TEAM", "HUBS", "ZEN", "PD", "DOCN", "FROG", "SMAR", "JAMF", "SUMO", "BIGC", "CART", "PRGS", "NCNO", "VERX", "PAGS", "STNE", "NU", "MELI"]

class BacksidePopScannerV11:
    """Backside Pop Scanner V11 - B-Grade Setups (Excludes V10 Qualifiers)
    
    V11 Logic:
    - Scans ALL tickers with relaxed parameters
    - For each potential setup, checks if it would also qualify for V10
    - Only includes setups that FAIL V10 criteria but PASS V11 criteria
    - Result: True B-grade setups that wouldn't make the V10 cut
    """
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        # Use ALL tickers - we'll filter by setup quality, not ticker names
        self.ticker_universe = TICKER_UNIVERSE
        self.max_workers = MAX_WORKERS
        
        # V11 More relaxed scan thresholds for B-grade setups
        self.scan_thresholds_v11 = {
            'min_trend_atr': 4.0,           # More relaxed from V10's 4.0
            'min_gap_atr': 0.4,             # More relaxed from V10's 0.5  
            'min_extension_atr': 1.0,       # More relaxed from V10's 1.5
            'min_range_close_pct': 60.0,    # More relaxed from V10's 70.0
            'min_volume_multiple': 0.7,     # More relaxed from V10's 0.7
            'min_change_atr': 0.4,          # More relaxed from V10's 0.5
            'max_downtrend_slope': -0.40,   # Much more lenient from V10's -0.50
            'min_ema_extension_pct': 5.0,   # More relaxed from V10's 10.0
            'min_fade_atr': 2.0,            # More relaxed from V10's 2.0
            'min_days_since_high': 1.0,       
            'max_days_since_high': 30.0,    # Extended from V10's 30.0  
            'min_price': 10.0,               # More relaxed from V10's 10.0
            'max_price': 1000.0,            
            'min_volume': 10_000_000,         # Share volume minimum 
            'min_dollar_volume_20d': 20_000_000,  # $20M minimum 20-day avg dollar volume for liquidity
            'require_dev_band_upper': True,   
            'require_d_minus_1_green': True,  
            
            # V11: Very lenient fade pattern requirements
            'min_red_days_consecutive': 4,    
            'outlier_volume_multiple': 2.0,   # More lenient
            'outlier_range_atr': 2.0,         # More lenient
            'outlier_fade_atr': 2.0,          # More lenient
            'fade_lookback_days': 15          # Extended
        }
        
        # V10 Original strict thresholds for filtering out A+ setups
        self.scan_thresholds_v10 = {
            'min_trend_atr': 4.0,           
            'min_gap_atr': 0.5,             
            'min_extension_atr': 1.5,       
            'min_range_close_pct': 70.0,    
            'min_volume_multiple': 0.7,     
            'min_change_atr': 0.5,         
            'max_downtrend_slope': -0.50,    
            'min_ema_extension_pct': 10.0,  
            'min_fade_atr': 2.0,            
            'min_days_since_high': 1.0,       
            'max_days_since_high': 30.0,      
            'min_price': 10.0,              
            'max_price': 1000.0,            
            'min_volume': 10_000_000,
            'require_dev_band_upper': True,   
            'require_d_minus_1_green': True,  
            
            'min_red_days_consecutive': 2,    
            'outlier_volume_multiple': 2.0,   
            'outlier_range_atr': 2.0,         
            'outlier_fade_atr': 2.0,          
            'fade_lookback_days': 10          
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'backside-scanner-v11'})

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
        """Calculate indicators with V11 enhanced 9/20 EMA deviation bands"""
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
        df['range_atr'] = df['range_dollars'] / df['atr']
        df['close_range'] = np.where(df['range_dollars'] > 0, 
                                   (df['close'] - df['low']) / df['range_dollars'], 0)
        df['price_change'] = df['close'] - df['pdc']
        df['price_change_atr'] = df['price_change'] / df['atr']
        df['open_to_prev_low'] = df['open'] - df['low'].shift(1)
        df['extension_atr'] = df['open_to_prev_low'] / df['atr']
        df['avg_volume_20'] = df['volume'].rolling(20, min_periods=5).mean()
        df['volume_multiple'] = df['volume'] / df['avg_volume_20']
        
        # Dollar volume calculation for liquidity filter
        df['dollar_volume'] = df['close'] * df['volume']
        df['avg_dollar_volume_20d'] = df['dollar_volume'].rolling(20, min_periods=5).mean()
        
        # V11: Red/Green day classification
        df['is_red_day'] = df['close'] < df['open']
        df['is_green_day'] = df['close'] > df['open']
        
        # V11 Enhanced 9/20 EMA Deviation Bands
        df['ATR_9'] = df['true_range'].rolling(9, min_periods=3).mean()
        df['dev_band_upper_1'] = df['ema_9'] + 1.0 * df['ATR_9']
        df['dev_band_upper_2'] = df['ema_9'] + 0.5 * df['ATR_9']
        df['dev_band_lower_1'] = df['ema_20'] - 2.0 * df['ATR_9']
        df['dev_band_lower_2'] = df['ema_20'] - 2.4 * df['ATR_9']
        
        # V11: Use 0.5 deviation band for better sensitivity
        df['open_above_dev_upper'] = df['open'] > df['dev_band_upper_2']
        df['high_above_dev_upper'] = df['high'] > df['dev_band_upper_2']
        
        return df

    def would_qualify_for_v10(self, df, idx, euphoric_high, trend_start_price, euphoric_date):
        """Check if this setup would also qualify for V10 (A+ criteria)"""
        prev_idx = idx - 1
        d_minus_1 = df.iloc[prev_idx]
        d_0 = df.iloc[idx]
        
        # V10 basic filters
        if not (self.scan_thresholds_v10['min_price'] <= d_0['close'] <= self.scan_thresholds_v10['max_price']):
            return False
        if d_minus_1['volume'] < self.scan_thresholds_v10['min_volume']:
            return False
        if pd.isna(d_minus_1['atr']) or d_minus_1['atr'] <= 0:
            return False
        
        # V10 D-1 must be green
        if d_minus_1['close'] <= d_minus_1['open']:
            return False
        
        # V10 D0 open/high must be above 0.5 deviation band
        d0_open_above_dev = d_0['open_above_dev_upper'] if not pd.isna(d_0['open_above_dev_upper']) else False
        d0_high_above_dev = d_0['high_above_dev_upper'] if not pd.isna(d_0['high_above_dev_upper']) else False
        
        if not (d0_open_above_dev or d0_high_above_dev):
            return False
        
        # V10 fade pattern check
        fade_valid, _, _ = self.analyze_fade_pattern_v10(df, idx, euphoric_high)
        if not fade_valid:
            return False
        
        # Calculate V10 setup metrics
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
        
        # V10 scan criteria (strict)
        v10_criteria = all([
            trend_atr_multiples >= self.scan_thresholds_v10['min_trend_atr'],
            gap_atr >= self.scan_thresholds_v10['min_gap_atr'],
            extension_atr >= self.scan_thresholds_v10['min_extension_atr'],
            range_close_pct >= self.scan_thresholds_v10['min_range_close_pct'],
            volume_multiple >= self.scan_thresholds_v10['min_volume_multiple'],
            change_atr >= self.scan_thresholds_v10['min_change_atr'],
            downtrend_slope <= self.scan_thresholds_v10['max_downtrend_slope'],
            ema_extension_pct >= self.scan_thresholds_v10['min_ema_extension_pct'],
            fade_atr >= self.scan_thresholds_v10['min_fade_atr'],
            self.scan_thresholds_v10['min_days_since_high'] <= days_since_high <= self.scan_thresholds_v10['max_days_since_high']
        ])
        
        return v10_criteria

    def analyze_fade_pattern_v10(self, df, setup_idx, euphoric_high):
        """V10 fade pattern analysis (stricter)"""
        if setup_idx < self.scan_thresholds_v10['fade_lookback_days']:
            return False, None, None
        
        # Look back from euphoric high to setup day
        euphoric_idx = None
        for i in range(setup_idx):
            if abs(df.iloc[i]['high'] - euphoric_high) < 0.01:
                euphoric_idx = i
                break
        
        if euphoric_idx is None or euphoric_idx >= setup_idx - 1:
            return False, None, None
        
        # Analyze fade from euphoric high to setup day
        fade_data = df.iloc[euphoric_idx:setup_idx].copy()
        if len(fade_data) < 2:
            return False, None, None
        
        # Method 1: Check for consecutive red days (V10 strict)
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
        
        # V10: If claiming multiple red days, need at least 2+ AND they must have higher volume/range  
        has_multiple_red_days = False
        if max_consecutive_reds >= 2:  # Actually need 2+ consecutive red days
            # Check if the red days had higher than normal volume or range
            red_days_quality = False
            for i, row in fade_data.iterrows():
                if row['is_red_day']:
                    vol_multiple = row['volume_multiple'] if not pd.isna(row['volume_multiple']) else 1.0
                    range_atr = row['range_atr'] if not pd.isna(row['range_atr']) else 1.0
                    
                    # Red days need elevated volume (1.5x+) OR wide range (2.0+ ATR) for V10
                    if vol_multiple >= 1.5 or range_atr >= 2.0:
                        red_days_quality = True
                        break
            
            has_multiple_red_days = red_days_quality
        
        # Method 2: Check for outlier fade day (V10 strict)
        has_outlier_fade_day = False
        outlier_fade_info = None
        
        for i, row in fade_data.iterrows():
            if pd.isna(row['volume_multiple']) or pd.isna(row['range_atr']) or pd.isna(row['atr']):
                continue
                
            fade_from_high = (euphoric_high - row['close']) / row['atr']
            
            is_high_volume = row['volume_multiple'] >= self.scan_thresholds_v10['outlier_volume_multiple']
            is_wide_range = row['range_atr'] >= self.scan_thresholds_v10['outlier_range_atr']
            is_big_fade = fade_from_high >= self.scan_thresholds_v10['outlier_fade_atr']
            
            # V10: Need at least 2 of 3 outlier conditions (strict)
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
        
        fade_pattern_valid = has_multiple_red_days or has_outlier_fade_day
        
        fade_info = {
            'multiple_red_days': has_multiple_red_days,
            'max_consecutive_reds': max_consecutive_reds,
            'red_streak_start': red_streak_start,
            'outlier_fade_day': has_outlier_fade_day,
            'outlier_info': outlier_fade_info
        }
        
        return fade_pattern_valid, fade_info, "multiple_red" if has_multiple_red_days else "outlier_fade"

    def analyze_fade_pattern(self, df, setup_idx, euphoric_high):
        """V11: Analyze fade pattern with relaxed criteria"""
        if setup_idx < self.scan_thresholds_v11['fade_lookback_days']:
            return False, None, None
        
        # Look back from euphoric high to setup day
        euphoric_idx = None
        for i in range(setup_idx):
            if abs(df.iloc[i]['high'] - euphoric_high) < 0.01:
                euphoric_idx = i
                break
        
        if euphoric_idx is None or euphoric_idx >= setup_idx - 1:
            return False, None, None
        
        # Analyze fade from euphoric high to setup day
        fade_data = df.iloc[euphoric_idx:setup_idx].copy()
        if len(fade_data) < 2:
            return False, None, None
        
        # Method 1: Check for consecutive red days (relaxed to 1+)
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
        
        # V11: If claiming multiple red days, need at least 2+ AND they must have higher volume/range
        has_multiple_red_days = False
        if max_consecutive_reds >= 2:  # Actually need 2+ consecutive red days
            # Check if the red days had higher than normal volume or range
            red_days_quality = False
            for i, row in fade_data.iterrows():
                if row['is_red_day']:
                    vol_multiple = row['volume_multiple'] if not pd.isna(row['volume_multiple']) else 1.0
                    range_atr = row['range_atr'] if not pd.isna(row['range_atr']) else 1.0
                    
                    # Red days need elevated volume (1.5x+) OR wide range (1.8+ ATR)
                    if vol_multiple >= 1.5 or range_atr >= 1.8:
                        red_days_quality = True
                        break
            
            has_multiple_red_days = red_days_quality
        
        # Method 2: Check for outlier fade day (relaxed criteria)
        has_outlier_fade_day = False
        outlier_fade_info = None
        
        for i, row in fade_data.iterrows():
            if pd.isna(row['volume_multiple']) or pd.isna(row['range_atr']) or pd.isna(row['atr']):
                continue
                
            # Calculate fade from euphoric high to this day's close
            fade_from_high = (euphoric_high - row['close']) / row['atr']
            
            # Check outlier criteria (relaxed)
            is_high_volume = row['volume_multiple'] >= self.scan_thresholds_v11['outlier_volume_multiple']
            is_wide_range = row['range_atr'] >= self.scan_thresholds_v11['outlier_range_atr']
            is_big_fade = fade_from_high >= self.scan_thresholds_v11['outlier_fade_atr']
            
            # V11: Need at least 2 of 3 outlier conditions (more restrictive)
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

    def scan_single_ticker_v11(self, symbol, start_date, end_date):
        """V11 B-Grade scan - excludes setups that would qualify for V10"""
        # Fetch data with extended lookback
        extended_start = (pd.to_datetime(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
        df = self.fetch_daily_data_cached(symbol, extended_start, end_date)
        
        if df.empty or len(df) < 50:
            return []
        
        # Calculate indicators with V11 deviation bands
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
            
            # V11 RELAXED FILTERS FOR B-GRADE
            
            # 1. Basic filters (relaxed)
            if not (self.scan_thresholds_v11['min_price'] <= d_0['close'] <= self.scan_thresholds_v11['max_price']):
                continue
            if d_minus_1['volume'] < self.scan_thresholds_v11['min_volume']:
                continue
            if pd.isna(d_minus_1['atr']) or d_minus_1['atr'] <= 0:
                continue
            
            # 1b. Dollar volume liquidity filter - D-1 must have sufficient 20d avg dollar volume
            d_minus_1_dollar_vol_20d = d_minus_1['avg_dollar_volume_20d'] if not pd.isna(d_minus_1['avg_dollar_volume_20d']) else 0
            if d_minus_1_dollar_vol_20d < self.scan_thresholds_v11['min_dollar_volume_20d']:
                continue
            
            # 2. D-1 must be green
            if d_minus_1['close'] <= d_minus_1['open']:
                continue
            
            # 3. V11: D0 open/high must be above 0.5 deviation band
            d0_open_above_dev = d_0['open_above_dev_upper'] if not pd.isna(d_0['open_above_dev_upper']) else False
            d0_high_above_dev = d_0['high_above_dev_upper'] if not pd.isna(d_0['high_above_dev_upper']) else False
            
            if not (d0_open_above_dev or d0_high_above_dev):
                continue
            
            # 4. Find trend and euphoric high
            trend_start_date, trend_start_price, euphoric_date, euphoric_high = \
                self.find_trend_and_euphoric_high_fast(df, idx)
            
            if not all([trend_start_date, trend_start_price, euphoric_date, euphoric_high]):
                continue
            
            # 5. V11: Check fade pattern (relaxed)
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
            
            # CRITICAL: Exclude setups that would qualify for V10 (A+ setups)
            if self.would_qualify_for_v10(df, idx, euphoric_high, trend_start_price, euphoric_date):
                continue  # Skip this setup - it belongs in V10, not V11
            
            # V11 Relaxed scan criteria for B-grade setups
            scan_criteria = all([
                trend_atr_multiples >= self.scan_thresholds_v11['min_trend_atr'],
                gap_atr >= self.scan_thresholds_v11['min_gap_atr'],
                extension_atr >= self.scan_thresholds_v11['min_extension_atr'],
                range_close_pct >= self.scan_thresholds_v11['min_range_close_pct'],
                volume_multiple >= self.scan_thresholds_v11['min_volume_multiple'],
                change_atr >= self.scan_thresholds_v11['min_change_atr'],
                downtrend_slope <= self.scan_thresholds_v11['max_downtrend_slope'],
                ema_extension_pct >= self.scan_thresholds_v11['min_ema_extension_pct'],
                fade_atr >= self.scan_thresholds_v11['min_fade_atr'],
                self.scan_thresholds_v11['min_days_since_high'] <= days_since_high <= self.scan_thresholds_v11['max_days_since_high']
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
                    'fade_type': fade_type,
                    'consecutive_reds': fade_info['max_consecutive_reds'] if fade_info['multiple_red_days'] else 0,
                    'outlier_volume': round(fade_info['outlier_info']['volume_multiple'], 2) if fade_info['outlier_fade_day'] and fade_info['outlier_info'] else None,
                    'outlier_range_atr': round(fade_info['outlier_info']['range_atr'], 2) if fade_info['outlier_fade_day'] and fade_info['outlier_info'] else None,
                    'dev_band_upper_05': round(d_0['dev_band_upper_2'], 2) if not pd.isna(d_0['dev_band_upper_2']) else None,
                    'd0_open': round(d_0['open'], 2),
                    'd0_high': round(d_0['high'], 2),
                    # Liquidity metrics
                    'd_minus_1_avg_dollar_vol_20d': round(d_minus_1_dollar_vol_20d / 1_000_000, 1),  # In millions
                    'scanner_version': 'V11',
                    'setup_grade': 'B'
                }
                setups.append(setup)
        
        return setups

    def run_scan_v11(self, start_date=None, end_date=None, tickers=None):
        """V11 B-Grade scanner with relaxed parameters"""
        start_date = start_date or START_DATE
        end_date = end_date or END_DATE
        tickers = tickers or self.ticker_universe
        
        print(f"🚀 BACKSIDE POP SCANNER V11 - B-GRADE SETUPS (EXCLUDES V10 QUALIFIERS)")
        print(f"📅 Scanning {len(tickers)} tickers from {start_date} to {end_date}")
        print(f"🔧 V11 B-Grade Logic:")
        print(f"   • Scans ALL tickers with relaxed parameters")
        print(f"   • Excludes any ticker/date combo that would qualify for V10")
        print(f"   • Min trend ATR: {self.scan_thresholds_v11['min_trend_atr']} (vs 4.0 in V10)")
        print(f"   • Min gap ATR: {self.scan_thresholds_v11['min_gap_atr']} (vs 0.5 in V10)")
        print(f"   • Min volume: {self.scan_thresholds_v11['min_volume']:,} (vs 10M in V10)")
        print(f"   • Min price: ${self.scan_thresholds_v11['min_price']} (vs $10 in V10)")
        print(f"   • RESULT: Only B-grade setups that missed V10 criteria")
        print("=" * 80)
        
        all_setups = []
        processed = 0
        errors = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.scan_single_ticker_v11, ticker, start_date, end_date): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    setups = future.result()
                    if setups:
                        all_setups.extend(setups)
                        print(f"✅ {ticker}: {len(setups)} B-grade setups found (excluded V10 qualifiers)")
                    else:
                        print(f"⚪ {ticker}: No B-grade setups (after V10 exclusion)")
                except Exception as e:
                    print(f"❌ {ticker}: Error - {str(e)[:50]}")
                    errors += 1
                
                processed += 1
                if processed % 20 == 0:
                    print(f"📊 Progress: {processed}/{len(tickers)} ({processed/len(tickers)*100:.1f}%)")
        
        print("\n" + "=" * 80)
        print(f"🎯 V11 B-GRADE SCAN COMPLETE")
        print(f"📊 Results: {len(all_setups)} B-grade setups found (V10 qualifiers excluded)")
        print(f"⚠️  Errors: {errors} tickers failed")
        
        if all_setups:
            df_results = pd.DataFrame(all_setups)
            df_results = df_results.sort_values(['date', 'trend_atr'], ascending=[False, False])
            
            print(f"\n🏆 ALL V11 B-GRADE SETUPS:")
            display_cols = ['symbol', 'date', 'trend_atr', 'gap_atr', 'extension_atr', 'fade_type', 'consecutive_reds', 'setup_grade']
            print(df_results[display_cols].to_string(index=False))
            
            # V11: Fade pattern breakdown
            fade_counts = df_results['fade_type'].value_counts()
            print(f"\n📊 B-GRADE FADE PATTERN BREAKDOWN:")
            for fade_type, count in fade_counts.items():
                print(f"   {fade_type}: {count} setups")
            
            return df_results
        else:
            print("No B-grade setups found matching V11 criteria.")
            return pd.DataFrame()

# Example Usage
if __name__ == "__main__":
    # Initialize V11 B-Grade scanner
    scanner = BacksidePopScannerV11()
    
    # Run V11 B-Grade scan
    results = scanner.run_scan_v11()
    
    if not results.empty:
        # Save results to CSV
        results.to_csv('backside_pop_v11_b_grade_results.csv', index=False)
        print(f"\n💾 B-Grade results saved to: backside_pop_v11_b_grade_results.csv")
        print(f"📋 Total B-grade setups: {len(results)}")
        print(f"🔍 These setups would NOT have qualified for V10 (excluded by setup quality, not ticker name)")
        print(f"🎯 V11 captures the setups that V10 missed due to stricter criteria")