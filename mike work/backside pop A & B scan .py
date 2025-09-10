"""
Combined Backside Pop Scanner V10 + V11 Enhanced
Features:
- Runs both V10 (A+ setups) and V11 (B-grade setups) in single scan
- All results combined into one output with clear version labeling
- Enhanced analysis columns for both scanner versions
- Additional VWAP analysis columns: D0 open vs VWAP, D0 high vs VWAP
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

# FULL TICKER UNIVERSE
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

class CombinedBacksideScanner:
    """Combined V10 + V11 Backside Pop Scanner with Enhanced Analysis"""
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        self.ticker_universe = TICKER_UNIVERSE
        self.max_workers = MAX_WORKERS
        
        # V10 scan thresholds (A+ setups)
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
            'max_d0_range_position': 65.0,   
            'min_red_days_consecutive': 2,    
            'outlier_volume_multiple': 2,   
            'outlier_range_atr': 2,         
            'outlier_fade_atr': 2.0,          
            'fade_lookback_days': 10          
        }
        
        # V11 scan thresholds (B-grade setups)
        self.scan_thresholds_v11 = {
            'min_trend_atr': 4.0,           
            'min_gap_atr': 0.4,             
            'min_extension_atr': 1.0,       
            'min_range_close_pct': 60.0,    
            'min_volume_multiple': 0.7,     
            'min_change_atr': 0.4,          
            'max_downtrend_slope': -0.40,   
            'min_ema_extension_pct': 5.0,   
            'min_fade_atr': 2.0,            
            'min_days_since_high': 1.0,       
            'max_days_since_high': 30.0,      
            'min_price': 10.0,               
            'max_price': 1000.0,            
            'min_volume': 10_000_000,         
            'min_dollar_volume_20d': 20_000_000,  
            'require_dev_band_upper': True,   
            'require_d_minus_1_green': True,  
            'min_red_days_consecutive': 4,    
            'outlier_volume_multiple': 2.0,   
            'outlier_range_atr': 2.0,         
            'outlier_fade_atr': 2.0,          
            'fade_lookback_days': 15          
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'combined-backside-scanner'})

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
        """Calculate indicators with enhanced 9/20 EMA deviation bands"""
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
        
        # Dollar volume calculation
        df['dollar_volume'] = df['close'] * df['volume']
        df['avg_dollar_volume_20d'] = df['dollar_volume'].rolling(20, min_periods=5).mean()
        
        # Red/Green day classification
        df['is_red_day'] = df['close'] < df['open']
        df['is_green_day'] = df['close'] > df['open']
        
        # Enhanced 9/20 EMA Deviation Bands
        df['ATR_9'] = df['true_range'].rolling(9, min_periods=3).mean()
        df['dev_band_upper_1'] = df['ema_9'] + 1.0 * df['ATR_9']
        df['dev_band_upper_2'] = df['ema_9'] + 0.5 * df['ATR_9']
        df['dev_band_lower_1'] = df['ema_20'] - 2.0 * df['ATR_9']
        df['dev_band_lower_2'] = df['ema_20'] - 2.4 * df['ATR_9']
        
        df['open_above_dev_upper'] = df['open'] > df['dev_band_upper_2']
        df['high_above_dev_upper'] = df['high'] > df['dev_band_upper_2']
        
        return df

    def calculate_vwap_anchored(self, df, anchor_idx):
        """Calculate anchored VWAP from a specific date index"""
        if anchor_idx >= len(df):
            return pd.Series([np.nan] * len(df), index=df.index)
        
        vwap_series = pd.Series([np.nan] * len(df), index=df.index)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        cum_vol = 0
        cum_pv = 0
        
        for i in range(anchor_idx, len(df)):
            cum_vol += df.iloc[i]['volume']
            cum_pv += typical_price.iloc[i] * df.iloc[i]['volume']
            
            if cum_vol > 0:
                vwap_series.iloc[i] = cum_pv / cum_vol
        
        return vwap_series

    def find_big_red_day(self, df, euphoric_idx, setup_idx):
        """Find the biggest red day between euphoric high and setup day"""
        if euphoric_idx >= setup_idx - 1:
            return None, None, None
        
        fade_data = df.iloc[euphoric_idx:setup_idx].copy()
        red_days = fade_data[fade_data['is_red_day']].copy()
        
        if red_days.empty:
            return None, None, None
        
        red_days['dollar_drop'] = red_days['open'] - red_days['close']
        biggest_red_idx = red_days['dollar_drop'].idxmax()
        
        return biggest_red_idx, red_days.loc[biggest_red_idx, 'date'], red_days.loc[biggest_red_idx, 'dollar_drop']

    def is_all_time_high(self, df, high_idx):
        """Check if the high at high_idx is an all-time high in the dataset"""
        if high_idx <= 0:
            return False
        
        current_high = df.iloc[high_idx]['high']
        prior_highs = df.iloc[:high_idx]['high']
        
        return current_high >= prior_highs.max()

    def calculate_candle_body_ratio(self, row):
        """Calculate candle body size relative to total range"""
        total_range = row['high'] - row['low']
        if total_range <= 0:
            return 0
        
        body_size = abs(row['close'] - row['open'])
        return body_size / total_range

    def is_lowest_low_in_period(self, df, day_idx, lookback_days=5):
        """Check if day's low is lowest in last N days"""
        if day_idx < lookback_days:
            return False
        
        current_low = df.iloc[day_idx]['low']
        period_data = df.iloc[max(0, day_idx - lookback_days):day_idx + 1]
        
        return current_low <= period_data['low'].min()

    def calculate_range_position(self, df, setup_idx, euphoric_high):
        """Calculate D0 open position within range from euphoric high to lowest low after high"""
        if setup_idx < 2:
            return None
        
        euphoric_idx = None
        for i in range(setup_idx):
            if abs(df.iloc[i]['high'] - euphoric_high) < 0.01:
                euphoric_idx = i
                break
        
        if euphoric_idx is None:
            return None
        
        range_data = df.iloc[euphoric_idx:setup_idx]
        if len(range_data) < 1:
            return None
        
        range_high = range_data['high'].max()
        range_low = range_data['low'].min()
        
        d0_open = df.iloc[setup_idx]['open']
        if range_high <= range_low:
            return None
            
        range_position_pct = ((d0_open - range_low) / (range_high - range_low)) * 100
        
        return range_position_pct

    def calculate_setup_grade(self, metrics, version='V10'):
        """Calculate setup grade based on key metrics"""
        grade = 0
        
        if version == 'V10':
            # A+ grade thresholds
            trend_atr = metrics.get('trend_atr', 0)
            if trend_atr >= 8:
                grade += 30
            elif trend_atr >= 6:
                grade += 25
            elif trend_atr >= 4:
                grade += 20
            
            ext_atr = metrics.get('extension_atr', 0)
            if ext_atr >= 3:
                grade += 25
            elif ext_atr >= 2:
                grade += 20
            elif ext_atr >= 1.5:
                grade += 15
            
            fade_atr = metrics.get('fade_atr', 0)
            if fade_atr >= 4:
                grade += 25
            elif fade_atr >= 3:
                grade += 20
            elif fade_atr >= 2:
                grade += 15
            
            vol_mult = metrics.get('volume_multiple', 0)
            if vol_mult >= 2:
                grade += 20
            elif vol_mult >= 1.5:
                grade += 15
            elif vol_mult >= 1:
                grade += 10
        else:
            # B-grade thresholds (more lenient)
            trend_atr = metrics.get('trend_atr', 0)
            if trend_atr >= 6:
                grade += 30
            elif trend_atr >= 4.5:
                grade += 25
            elif trend_atr >= 3.5:
                grade += 20
            elif trend_atr >= 2.5:
                grade += 15
            
            ext_atr = metrics.get('extension_atr', 0)
            if ext_atr >= 2.5:
                grade += 25
            elif ext_atr >= 1.8:
                grade += 20
            elif ext_atr >= 1.2:
                grade += 15
            elif ext_atr >= 0.8:
                grade += 10
            
            fade_atr = metrics.get('fade_atr', 0)
            if fade_atr >= 3.5:
                grade += 25
            elif fade_atr >= 2.5:
                grade += 20
            elif fade_atr >= 1.8:
                grade += 15
            elif fade_atr >= 1.2:
                grade += 10
            
            vol_mult = metrics.get('volume_multiple', 0)
            if vol_mult >= 1.8:
                grade += 20
            elif vol_mult >= 1.3:
                grade += 15
            elif vol_mult >= 0.9:
                grade += 10
            elif vol_mult >= 0.6:
                grade += 5
        
        return min(grade, 100)

    def calculate_fade_grade(self, fade_info, metrics, version='V10'):
        """Calculate fade grade based on fade pattern quality"""
        grade = 0
        
        if version == 'V10':
            # A+ fade grading
            if fade_info and fade_info.get('multiple_red_days', False):
                consecutive_reds = fade_info.get('max_consecutive_reds', 0)
                if consecutive_reds >= 4:
                    grade += 40
                elif consecutive_reds >= 3:
                    grade += 30
                elif consecutive_reds >= 2:
                    grade += 20
            
            if fade_info and fade_info.get('outlier_fade_day', False):
                outlier_info = fade_info.get('outlier_info', {})
                if outlier_info:
                    conditions_met = outlier_info.get('conditions_met', 0)
                    if conditions_met >= 3:
                        grade += 30
                    elif conditions_met >= 2:
                        grade += 20
            
            fade_atr = metrics.get('fade_atr', 0)
            if fade_atr >= 5:
                grade += 30
            elif fade_atr >= 4:
                grade += 25
            elif fade_atr >= 3:
                grade += 20
            elif fade_atr >= 2:
                grade += 15
        else:
            # B-grade fade grading (more lenient)
            if fade_info and fade_info.get('multiple_red_days', False):
                consecutive_reds = fade_info.get('max_consecutive_reds', 0)
                if consecutive_reds >= 3:
                    grade += 40
                elif consecutive_reds >= 2:
                    grade += 30
                elif consecutive_reds >= 1:
                    grade += 20
            
            if fade_info and fade_info.get('outlier_fade_day', False):
                outlier_info = fade_info.get('outlier_info', {})
                if outlier_info:
                    conditions_met = outlier_info.get('conditions_met', 0)
                    if conditions_met >= 3:
                        grade += 30
                    elif conditions_met >= 2:
                        grade += 20
                    elif conditions_met >= 1:
                        grade += 10
            
            fade_atr = metrics.get('fade_atr', 0)
            if fade_atr >= 4:
                grade += 30
            elif fade_atr >= 3:
                grade += 25
            elif fade_atr >= 2.2:
                grade += 20
            elif fade_atr >= 1.5:
                grade += 15
            elif fade_atr >= 1.0:
                grade += 10
        
        return min(grade, 100)

    def analyze_fade_pattern(self, df, setup_idx, euphoric_high, version='V10'):
        """Analyze fade pattern with version-specific criteria"""
        thresholds = self.scan_thresholds_v10 if version == 'V10' else self.scan_thresholds_v11
        
        if setup_idx < thresholds['fade_lookback_days']:
            return False, None, None
        
        euphoric_idx = None
        for i in range(setup_idx):
            if abs(df.iloc[i]['high'] - euphoric_high) < 0.01:
                euphoric_idx = i
                break
        
        if euphoric_idx is None or euphoric_idx >= setup_idx - 1:
            return False, None, None
        
        fade_data = df.iloc[euphoric_idx:setup_idx].copy()
        if len(fade_data) < 2:
            return False, None, None
        
        # Method 1: Check for consecutive red days
        consecutive_reds = 0
        max_consecutive_reds = 0
        red_streak_start = None
        has_red_below_ema9 = False
        
        for i, row in fade_data.iterrows():
            if row['is_red_day']:
                if consecutive_reds == 0:
                    red_streak_start = row['date']
                consecutive_reds += 1
                max_consecutive_reds = max(max_consecutive_reds, consecutive_reds)
                
                if not pd.isna(row['ema_9']) and row['close'] < row['ema_9']:
                    has_red_below_ema9 = True
            else:
                consecutive_reds = 0
        
        # Version-specific red day requirements
        if version == 'V10':
            has_multiple_red_days = (max_consecutive_reds >= thresholds['min_red_days_consecutive'] and 
                                   has_red_below_ema9)
        else:
            # V11: More relaxed
            has_multiple_red_days = False
            if max_consecutive_reds >= 2:
                red_days_quality = False
                for i, row in fade_data.iterrows():
                    if row['is_red_day']:
                        vol_multiple = row['volume_multiple'] if not pd.isna(row['volume_multiple']) else 1.0
                        range_atr = row['range_atr'] if not pd.isna(row['range_atr']) else 1.0
                        
                        if vol_multiple >= 1.5 or range_atr >= 1.8:
                            red_days_quality = True
                            break
                
                has_multiple_red_days = red_days_quality
        
        # Method 2: Check for outlier fade day
        has_outlier_fade_day = False
        outlier_fade_info = None
        
        for i, row in fade_data.iterrows():
            if pd.isna(row['volume_multiple']) or pd.isna(row['range_atr']) or pd.isna(row['atr']):
                continue
                
            fade_from_high = (euphoric_high - row['close']) / row['atr']
            
            is_high_volume = row['volume_multiple'] >= thresholds['outlier_volume_multiple']
            is_wide_range = row['range_atr'] >= thresholds['outlier_range_atr']
            is_big_fade = fade_from_high >= thresholds['outlier_fade_atr']
            
            if version == 'V10':
                closed_below_ema9 = not pd.isna(row['ema_9']) and row['close'] < row['ema_9']
                outlier_conditions = sum([is_high_volume, is_wide_range, is_big_fade])
                
                if outlier_conditions >= 2 and closed_below_ema9:
                    has_outlier_fade_day = True
                    outlier_fade_info = {
                        'date': row['date'],
                        'volume_multiple': row['volume_multiple'],
                        'range_atr': row['range_atr'],
                        'fade_atr': fade_from_high,
                        'conditions_met': outlier_conditions,
                        'closed_below_ema9': closed_below_ema9
                    }
                    break
            else:
                # V11: More relaxed
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
            'has_red_below_ema9': has_red_below_ema9,
            'outlier_fade_day': has_outlier_fade_day,
            'outlier_info': outlier_fade_info
        }
        
        return fade_pattern_valid, fade_info, "multiple_red" if has_multiple_red_days else "outlier_fade"

    def find_trend_and_euphoric_high_fast(self, df, setup_idx):
        """Optimized trend and euphoric high detection"""
        if setup_idx < 20:
            return None, None, None, None
            
        pre_setup = df.iloc[:setup_idx].copy()
        ema_9_above_20 = (pre_setup['ema_9'] > pre_setup['ema_20']).astype(int)
        cross_signal = ema_9_above_20.diff()
        
        cross_indices = pre_setup[cross_signal == 1].index
        if len(cross_indices) == 0:
            return None, None, None, None
            
        trend_start_idx = cross_indices[-1]
        trend_start_date = pre_setup.loc[trend_start_idx, 'date']
        trend_start_price = pre_setup.loc[trend_start_idx, 'close']
        
        trend_data = pre_setup.iloc[trend_start_idx:]
        if trend_data.empty:
            return None, None, None, None
            
        euphoric_idx = trend_data['high'].idxmax()
        euphoric_date = trend_data.loc[euphoric_idx, 'date']
        euphoric_high = trend_data.loc[euphoric_idx, 'high']
        
        return trend_start_date, trend_start_price, euphoric_date, euphoric_high

    def scan_single_ticker_combined(self, symbol, start_date, end_date):
        """Combined scan for both V10 and V11 setups"""
        # Fetch data with extended lookback
        extended_start = (pd.to_datetime(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
        df = self.fetch_daily_data_cached(symbol, extended_start, end_date)
        
        if df.empty or len(df) < 50:
            return []
        
        # Calculate indicators
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
            
            # Basic filters (common to both)
            if not (10.0 <= d_0['close'] <= 1000.0):
                continue
            if d_minus_1['volume'] < 10_000_000:
                continue
            if pd.isna(d_minus_1['atr']) or d_minus_1['atr'] <= 0:
                continue
            if d_minus_1['close'] <= d_minus_1['open']:
                continue
            
            d0_open_above_dev = d_0['open_above_dev_upper'] if not pd.isna(d_0['open_above_dev_upper']) else False
            d0_high_above_dev = d_0['high_above_dev_upper'] if not pd.isna(d_0['high_above_dev_upper']) else False
            
            if not (d0_open_above_dev or d0_high_above_dev):
                continue
            
            # Find trend and euphoric high
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
            
            range_position_pct = self.calculate_range_position(df, idx, euphoric_high)
            d_minus_1_dollar_vol_20d = d_minus_1['avg_dollar_volume_20d'] if not pd.isna(d_minus_1['avg_dollar_volume_20d']) else 0
            
            # Check V10 criteria first (A+ setups)
            fade_valid_v10, fade_info_v10, fade_type_v10 = self.analyze_fade_pattern(df, idx, euphoric_high, 'V10')
            
            v10_criteria = all([
                fade_valid_v10,
                trend_atr_multiples >= self.scan_thresholds_v10['min_trend_atr'],
                gap_atr >= self.scan_thresholds_v10['min_gap_atr'],
                extension_atr >= self.scan_thresholds_v10['min_extension_atr'],
                range_close_pct >= self.scan_thresholds_v10['min_range_close_pct'],
                volume_multiple >= self.scan_thresholds_v10['min_volume_multiple'],
                change_atr >= self.scan_thresholds_v10['min_change_atr'],
                downtrend_slope <= self.scan_thresholds_v10['max_downtrend_slope'],
                ema_extension_pct >= self.scan_thresholds_v10['min_ema_extension_pct'],
                fade_atr >= self.scan_thresholds_v10['min_fade_atr'],
                self.scan_thresholds_v10['min_days_since_high'] <= days_since_high <= self.scan_thresholds_v10['max_days_since_high'],
                range_position_pct is not None and range_position_pct <= self.scan_thresholds_v10['max_d0_range_position']
            ])
            
            if v10_criteria:
                # This is a V10 A+ setup
                setups.append(self.create_setup_object(
                    symbol, current_date, df, idx, prev_idx, d_minus_1, d_0,
                    trend_start_date, trend_start_price, euphoric_date, euphoric_high,
                    fade_info_v10, fade_type_v10, 'V10'
                ))
            else:
                # Check V11 criteria (B-grade setups)
                fade_valid_v11, fade_info_v11, fade_type_v11 = self.analyze_fade_pattern(df, idx, euphoric_high, 'V11')
                
                v11_criteria = all([
                    fade_valid_v11,
                    trend_atr_multiples >= self.scan_thresholds_v11['min_trend_atr'],
                    gap_atr >= self.scan_thresholds_v11['min_gap_atr'],
                    extension_atr >= self.scan_thresholds_v11['min_extension_atr'],
                    range_close_pct >= self.scan_thresholds_v11['min_range_close_pct'],
                    volume_multiple >= self.scan_thresholds_v11['min_volume_multiple'],
                    change_atr >= self.scan_thresholds_v11['min_change_atr'],
                    downtrend_slope <= self.scan_thresholds_v11['max_downtrend_slope'],
                    ema_extension_pct >= self.scan_thresholds_v11['min_ema_extension_pct'],
                    fade_atr >= self.scan_thresholds_v11['min_fade_atr'],
                    self.scan_thresholds_v11['min_days_since_high'] <= days_since_high <= self.scan_thresholds_v11['max_days_since_high'],
                    d_minus_1_dollar_vol_20d >= self.scan_thresholds_v11['min_dollar_volume_20d']
                ])
                
                if v11_criteria:
                    # This is a V11 B-grade setup
                    setups.append(self.create_setup_object(
                        symbol, current_date, df, idx, prev_idx, d_minus_1, d_0,
                        trend_start_date, trend_start_price, euphoric_date, euphoric_high,
                        fade_info_v11, fade_type_v11, 'V11'
                    ))
        
        return setups

    def create_setup_object(self, symbol, current_date, df, idx, prev_idx, d_minus_1, d_0,
                           trend_start_date, trend_start_price, euphoric_date, euphoric_high,
                           fade_info, fade_type, version):
        """Create enhanced setup object with all analysis parameters"""
        
        # Calculate core metrics
        days_since_high = (d_minus_1['date'] - euphoric_date).days
        trend_atr_multiples = (euphoric_high - trend_start_price) / d_minus_1['atr']
        fade_atr = (euphoric_high - d_minus_1['close']) / d_minus_1['atr']
        
        gap_atr = d_0['gap_atr'] if not pd.isna(d_0['gap_atr']) else 0
        extension_atr = d_0['extension_atr'] if not pd.isna(d_0['extension_atr']) else 0
        range_close_pct = d_minus_1['close_range'] * 100
        volume_multiple = d_minus_1['volume_multiple'] if not pd.isna(d_minus_1['volume_multiple']) else 0
        
        # Find indices for analysis
        euphoric_idx = None
        for i in range(idx):
            if abs(df.iloc[i]['high'] - euphoric_high) < 0.01:
                euphoric_idx = i
                break
        
        # Big Red Day Analysis
        brd_idx, brd_date, brd_drop = self.find_big_red_day(df, euphoric_idx, idx)
        brd_is_highest_high = False
        brd_atr_ext_9ema = None
        brd_atr_ext_20ema = None
        
        if brd_idx is not None:
            brd_row = df.iloc[brd_idx]
            brd_is_highest_high = (brd_idx == euphoric_idx)
            
            if not pd.isna(brd_row['ema_9']) and not pd.isna(brd_row['atr']):
                brd_atr_ext_9ema = (brd_row['high'] - brd_row['ema_9']) / brd_row['atr']
            if not pd.isna(brd_row['ema_20']) and not pd.isna(brd_row['atr']):
                brd_atr_ext_20ema = (brd_row['high'] - brd_row['ema_20']) / brd_row['atr']
        
        # Highest High Analysis
        hh_atr_ext_9ema = None
        hh_atr_ext_20ema = None
        hh_is_ath = False
        
        if euphoric_idx is not None:
            hh_row = df.iloc[euphoric_idx]
            hh_is_ath = self.is_all_time_high(df, euphoric_idx)
            
            if not pd.isna(hh_row['ema_9']) and not pd.isna(hh_row['atr']):
                hh_atr_ext_9ema = (euphoric_high - hh_row['ema_9']) / hh_row['atr']
            if not pd.isna(hh_row['ema_20']) and not pd.isna(hh_row['atr']):
                hh_atr_ext_20ema = (euphoric_high - hh_row['ema_20']) / hh_row['atr']
        
        # D-1 Trigger Analysis
        d1_candle_body_ratio = self.calculate_candle_body_ratio(d_minus_1)
        d1_low_is_5day_low = self.is_lowest_low_in_period(df, prev_idx, 5)
        
        # Anchored VWAP Analysis (Enhanced with D0 open and high)
        vwap_from_hh = None
        vwap_from_brd = None
        d0_open_vs_vwap_hh = None
        d0_high_vs_vwap_hh = None
        d0_open_vs_vwap_brd = None
        d0_high_vs_vwap_brd = None
        
        if euphoric_idx is not None:
            vwap_series_hh = self.calculate_vwap_anchored(df, euphoric_idx)
            if not pd.isna(vwap_series_hh.iloc[idx]):
                vwap_from_hh = vwap_series_hh.iloc[idx]
                d0_open_vs_vwap_hh = ((d_0['open'] - vwap_from_hh) / vwap_from_hh) * 100
                d0_high_vs_vwap_hh = ((d_0['high'] - vwap_from_hh) / vwap_from_hh) * 100
        
        if brd_idx is not None:
            vwap_series_brd = self.calculate_vwap_anchored(df, brd_idx)
            if not pd.isna(vwap_series_brd.iloc[idx]):
                vwap_from_brd = vwap_series_brd.iloc[idx]
                d0_open_vs_vwap_brd = ((d_0['open'] - vwap_from_brd) / vwap_from_brd) * 100
                d0_high_vs_vwap_brd = ((d_0['high'] - vwap_from_brd) / vwap_from_brd) * 100
        
        # D0 Distance from EMAs
        d0_open_vs_9ema_atr = None
        d0_open_vs_20ema_atr = None
        
        if not pd.isna(d_0['ema_9']) and not pd.isna(d_0['atr']):
            d0_open_vs_9ema_atr = (d_0['open'] - d_0['ema_9']) / d_0['atr']
        if not pd.isna(d_0['ema_20']) and not pd.isna(d_0['atr']):
            d0_open_vs_20ema_atr = (d_0['open'] - d_0['ema_20']) / d_0['atr']
        
        # Setup and Fade Grades
        setup_metrics = {
            'trend_atr': trend_atr_multiples,
            'extension_atr': extension_atr,
            'fade_atr': fade_atr,
            'volume_multiple': volume_multiple
        }
        
        setup_grade = self.calculate_setup_grade(setup_metrics, version)
        fade_grade = self.calculate_fade_grade(fade_info, setup_metrics, version)
        
        # Range position
        range_position_pct = self.calculate_range_position(df, idx, euphoric_high)
        
        # Dollar volume
        d_minus_1_dollar_vol_20d = d_minus_1['avg_dollar_volume_20d'] if not pd.isna(d_minus_1['avg_dollar_volume_20d']) else 0
        
        # Create enhanced setup object
        setup = {
            # Core identification
            'symbol': symbol,
            'date': current_date.strftime('%Y-%m-%d'),
            'scanner_version': version,
            'setup_grade_type': 'A+' if version == 'V10' else 'B',
            
            # Key grades and dates
            'setup_grade': setup_grade,
            'fade_grade': fade_grade,
            'highest_high_date': euphoric_date.strftime('%Y-%m-%d'),
            'brd_date': brd_date.strftime('%Y-%m-%d') if brd_date else None,
            
            # Core metrics
            'trend_atr': round(trend_atr_multiples, 2),
            'gap_atr': round(gap_atr, 2),
            'extension_atr': round(extension_atr, 2),
            'fade_atr': round(fade_atr, 2),
            'days_since_high': days_since_high,
            'fade_type': fade_type,
            
            # Big Red Day Analysis
            'brd_is_highest_high': brd_is_highest_high,
            'brd_atr_ext_9ema': round(brd_atr_ext_9ema, 2) if brd_atr_ext_9ema else None,
            'brd_atr_ext_20ema': round(brd_atr_ext_20ema, 2) if brd_atr_ext_20ema else None,
            
            # Highest High Analysis
            'hh_atr_ext_9ema': round(hh_atr_ext_9ema, 2) if hh_atr_ext_9ema else None,
            'hh_atr_ext_20ema': round(hh_atr_ext_20ema, 2) if hh_atr_ext_20ema else None,
            'hh_is_ath': hh_is_ath,
            
            # D-1 Trigger Analysis
            'd1_candle_body_ratio': round(d1_candle_body_ratio, 3),
            'd1_low_is_5day_low': d1_low_is_5day_low,
            'd1_range_close_pct': round(range_close_pct, 1),
            'd1_volume_multiple': round(volume_multiple, 2),
            
            # Enhanced Anchored VWAP Analysis
            'd0_open_vs_vwap_hh_pct': round(d0_open_vs_vwap_hh, 2) if d0_open_vs_vwap_hh else None,
            'd0_high_vs_vwap_hh_pct': round(d0_high_vs_vwap_hh, 2) if d0_high_vs_vwap_hh else None,
            'd0_open_vs_vwap_brd_pct': round(d0_open_vs_vwap_brd, 2) if d0_open_vs_vwap_brd else None,
            'd0_high_vs_vwap_brd_pct': round(d0_high_vs_vwap_brd, 2) if d0_high_vs_vwap_brd else None,
            
            # D0 EMA Distance Analysis
            'd0_open_vs_9ema_atr': round(d0_open_vs_9ema_atr, 2) if d0_open_vs_9ema_atr else None,
            'd0_open_vs_20ema_atr': round(d0_open_vs_20ema_atr, 2) if d0_open_vs_20ema_atr else None,
            
            # Additional context
            'consecutive_reds': fade_info['max_consecutive_reds'] if fade_info['multiple_red_days'] else 0,
            'range_position_pct': round(range_position_pct, 1) if range_position_pct is not None else None,
            'd_minus_1_avg_dollar_vol_20d': round(d_minus_1_dollar_vol_20d / 1_000_000, 1) if d_minus_1_dollar_vol_20d > 0 else None,
            
            # Price levels
            'd0_open': round(d_0['open'], 2),
            'd0_high': round(d_0['high'], 2),
            'euphoric_high_price': round(euphoric_high, 2)
        }
        
        return setup

    def run_combined_scan(self, start_date=None, end_date=None, tickers=None):
        """Run combined V10 + V11 scanner"""
        start_date = start_date or START_DATE
        end_date = end_date or END_DATE
        tickers = tickers or self.ticker_universe
        
        print(f"🚀 COMBINED BACKSIDE POP SCANNER V10 + V11 ENHANCED")
        print(f"📅 Scanning {len(tickers)} tickers from {start_date} to {end_date}")
        print(f"🔧 Combined Features:")
        print(f"   • V10: A+ setups with strict criteria")
        print(f"   • V11: B-grade setups with relaxed criteria (excludes V10 qualifiers)")
        print(f"   • Enhanced VWAP analysis: D0 open AND high vs anchored VWAP")
        print(f"   • All results combined with version labeling")
        print("=" * 80)
        
        all_setups = []
        processed = 0
        errors = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.scan_single_ticker_combined, ticker, start_date, end_date): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    setups = future.result()
                    if setups:
                        v10_count = len([s for s in setups if s['scanner_version'] == 'V10'])
                        v11_count = len([s for s in setups if s['scanner_version'] == 'V11'])
                        all_setups.extend(setups)
                        print(f"✅ {ticker}: {v10_count} V10 (A+), {v11_count} V11 (B-grade)")
                    else:
                        print(f"⚪ {ticker}: No setups")
                except Exception as e:
                    print(f"❌ {ticker}: Error - {str(e)[:50]}")
                    errors += 1
                
                processed += 1
                if processed % 20 == 0:
                    print(f"📊 Progress: {processed}/{len(tickers)} ({processed/len(tickers)*100:.1f}%)")
        
        print("\n" + "=" * 80)
        print(f"🎯 COMBINED SCAN COMPLETE")
        print(f"📊 Total Results: {len(all_setups)} setups found")
        print(f"⚠️  Errors: {errors} tickers failed")
        
        if all_setups:
            df_results = pd.DataFrame(all_setups)
            df_results = df_results.sort_values(['scanner_version', 'setup_grade', 'fade_grade'], ascending=[True, False, False])
            
            # Summary statistics
            v10_count = len(df_results[df_results['scanner_version'] == 'V10'])
            v11_count = len(df_results[df_results['scanner_version'] == 'V11'])
            
            print(f"\n📊 COMBINED RESULTS BREAKDOWN:")
            print(f"   V10 A+ Setups: {v10_count}")
            print(f"   V11 B-grade Setups: {v11_count}")
            print(f"   Total Setups: {len(df_results)}")
            
            print(f"\n🏆 TOP COMBINED SETUPS BY VERSION AND GRADE:")
            display_cols = ['scanner_version', 'setup_grade_type', 'symbol', 'date', 'setup_grade', 'fade_grade', 'highest_high_date', 'brd_date']
            print(df_results[display_cols].head(20).to_string(index=False))
            
            # Enhanced statistics by version
            for version in ['V10', 'V11']:
                version_data = df_results[df_results['scanner_version'] == version]
                if not version_data.empty:
                    print(f"\n📊 {version} ANALYSIS:")
                    print(f"   Count: {len(version_data)}")
                    print(f"   Avg Setup Grade: {version_data['setup_grade'].mean():.1f}")
                    print(f"   Avg Fade Grade: {version_data['fade_grade'].mean():.1f}")
                    print(f"   All-Time Highs: {version_data['hh_is_ath'].sum()}/{len(version_data)} ({version_data['hh_is_ath'].mean()*100:.1f}%)")
                    print(f"   BRD = Highest High: {version_data['brd_is_highest_high'].sum()}/{len(version_data)} ({version_data['brd_is_highest_high'].mean()*100:.1f}%)")
            
            return df_results
        else:
            print("No setups found matching combined criteria.")
            return pd.DataFrame()

# Example Usage
if __name__ == "__main__":
    # Initialize combined scanner
    scanner = CombinedBacksideScanner()
    
    # Run combined scan
    results = scanner.run_combined_scan()
    
    if not results.empty:
        # Save combined results to CSV
        results.to_csv('backside_pop_combined_v10_v11_enhanced_results.csv', index=False)
        print(f"\n💾 Combined results saved to: backside_pop_combined_v10_v11_enhanced_results.csv")
        
        # Display all columns for analysis
        print(f"\n📋 ALL ENHANCED ANALYSIS COLUMNS:")
        for col in results.columns:
            print(f"   • {col}")
        
        print(f"\n🎯 SUMMARY:")
        print(f"   • Single output file with both V10 and V11 results")
        print(f"   • Clear version labeling (scanner_version column)")
        print(f"   • Enhanced VWAP analysis with D0 open AND high")
        print(f"   • Complete analysis parameter coverage")
        print(f"   • NEW: Optimization filters applied - expect fewer but higher quality results")