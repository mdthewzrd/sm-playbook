"""
Backside Pop Scanner V10 Enhanced - Analysis Columns
Enhanced with additional analysis parameters for optimization research
SCAN CRITERIA UNCHANGED - only adding analysis columns for parameter research
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

TICKER_UNIVERSE = ['HOOD', 'COIN', 'MSTR', 'SMCI', 'IBIT', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META',"SOXL", "MRVL", "TGT", "DOCU", "ZM", "DIS", "NFLX", "AMC", "RKT", "SNAP", "RBLX", "META", "SE", "NVDA", 
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

class BacksidePopScannerV10Enhanced:
    """Backside Pop Scanner V10 Enhanced with Additional Analysis Columns"""
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        self.ticker_universe = TICKER_UNIVERSE
        self.max_workers = MAX_WORKERS
        
        # V10 scan thresholds - UNCHANGED
        self.scan_thresholds = {
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
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'backside-scanner-v10-enhanced'})

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
        df['range_atr'] = df['range_dollars'] / df['atr']
        df['close_range'] = np.where(df['range_dollars'] > 0, 
                                   (df['close'] - df['low']) / df['range_dollars'], 0)
        df['price_change'] = df['close'] - df['pdc']
        df['price_change_atr'] = df['price_change'] / df['atr']
        df['open_to_prev_low'] = df['open'] - df['low'].shift(1)
        df['extension_atr'] = df['open_to_prev_low'] / df['atr']
        df['avg_volume_20'] = df['volume'].rolling(20, min_periods=5).mean()
        df['volume_multiple'] = df['volume'] / df['avg_volume_20']
        
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
        
        # Calculate VWAP from anchor point forward
        vwap_series = pd.Series([np.nan] * len(df), index=df.index)
        
        # Calculate typical price and volume
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate cumulative values from anchor point
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
        
        # Look for biggest red day in fade period
        fade_data = df.iloc[euphoric_idx:setup_idx].copy()
        red_days = fade_data[fade_data['is_red_day']].copy()
        
        if red_days.empty:
            return None, None, None
        
        # Find biggest red day by absolute dollar drop
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
        
        # Find euphoric high index
        euphoric_idx = None
        for i in range(setup_idx):
            if abs(df.iloc[i]['high'] - euphoric_high) < 0.01:
                euphoric_idx = i
                break
        
        if euphoric_idx is None:
            return None
        
        # Get range from euphoric high to setup day
        range_data = df.iloc[euphoric_idx:setup_idx]
        if len(range_data) < 1:
            return None
        
        # Get highest high and lowest low in this range
        range_high = range_data['high'].max()
        range_low = range_data['low'].min()
        
        # Calculate D0 open position in this range
        d0_open = df.iloc[setup_idx]['open']
        if range_high <= range_low:
            return None
            
        range_position_pct = ((d0_open - range_low) / (range_high - range_low)) * 100
        
        return range_position_pct

    def analyze_fade_pattern(self, df, setup_idx, euphoric_high):
        """V10: Analyze fade pattern - multiple red days OR big outlier fade day"""
        if setup_idx < self.scan_thresholds['fade_lookback_days']:
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
        
        has_multiple_red_days = (max_consecutive_reds >= self.scan_thresholds['min_red_days_consecutive'] and 
                               has_red_below_ema9)
        
        # Method 2: Check for outlier fade day
        has_outlier_fade_day = False
        outlier_fade_info = None
        
        for i, row in fade_data.iterrows():
            if pd.isna(row['volume_multiple']) or pd.isna(row['range_atr']) or pd.isna(row['atr']):
                continue
                
            fade_from_high = (euphoric_high - row['close']) / row['atr']
            
            is_high_volume = row['volume_multiple'] >= self.scan_thresholds['outlier_volume_multiple']
            is_wide_range = row['range_atr'] >= self.scan_thresholds['outlier_range_atr']
            is_big_fade = fade_from_high >= self.scan_thresholds['outlier_fade_atr']
            
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

    def calculate_setup_grade(self, metrics):
        """Calculate setup grade based on key metrics"""
        # Grade based on trend strength, extension, and fade quality
        grade = 0
        
        # Trend ATR component (0-30 points)
        trend_atr = metrics.get('trend_atr', 0)
        if trend_atr >= 8:
            grade += 30
        elif trend_atr >= 6:
            grade += 25
        elif trend_atr >= 4:
            grade += 20
        
        # Extension ATR component (0-25 points)
        ext_atr = metrics.get('extension_atr', 0)
        if ext_atr >= 3:
            grade += 25
        elif ext_atr >= 2:
            grade += 20
        elif ext_atr >= 1.5:
            grade += 15
        
        # Fade ATR component (0-25 points)
        fade_atr = metrics.get('fade_atr', 0)
        if fade_atr >= 4:
            grade += 25
        elif fade_atr >= 3:
            grade += 20
        elif fade_atr >= 2:
            grade += 15
        
        # Volume multiple component (0-20 points)
        vol_mult = metrics.get('volume_multiple', 0)
        if vol_mult >= 2:
            grade += 20
        elif vol_mult >= 1.5:
            grade += 15
        elif vol_mult >= 1:
            grade += 10
        
        return min(grade, 100)  # Cap at 100

    def calculate_fade_grade(self, fade_info, metrics):
        """Calculate fade grade based on fade pattern quality"""
        grade = 0
        
        # Red days component (0-40 points)
        if fade_info and fade_info.get('multiple_red_days', False):
            consecutive_reds = fade_info.get('max_consecutive_reds', 0)
            if consecutive_reds >= 4:
                grade += 40
            elif consecutive_reds >= 3:
                grade += 30
            elif consecutive_reds >= 2:
                grade += 20
        
        # Outlier fade component (0-30 points)
        if fade_info and fade_info.get('outlier_fade_day', False):
            outlier_info = fade_info.get('outlier_info', {})
            if outlier_info:
                conditions_met = outlier_info.get('conditions_met', 0)
                if conditions_met >= 3:
                    grade += 30
                elif conditions_met >= 2:
                    grade += 20
        
        # Fade ATR component (0-30 points)
        fade_atr = metrics.get('fade_atr', 0)
        if fade_atr >= 5:
            grade += 30
        elif fade_atr >= 4:
            grade += 25
        elif fade_atr >= 3:
            grade += 20
        elif fade_atr >= 2:
            grade += 15
        
        return min(grade, 100)  # Cap at 100

    def scan_single_ticker_v10_enhanced(self, symbol, start_date, end_date):
        """V10 Enhanced scan with additional analysis columns - SCAN CRITERIA UNCHANGED"""
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
            
            # V10 CRITICAL FILTERS - UNCHANGED
            if not (self.scan_thresholds['min_price'] <= d_0['close'] <= self.scan_thresholds['max_price']):
                continue
            if d_minus_1['volume'] < self.scan_thresholds['min_volume']:
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
            
            # Check fade pattern (UNCHANGED)
            fade_valid, fade_info, fade_type = self.analyze_fade_pattern(df, idx, euphoric_high)
            
            if not fade_valid:
                continue
            
            # Calculate existing setup metrics
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
            
            # V10 Enhanced scan criteria - UNCHANGED
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
                self.scan_thresholds['min_days_since_high'] <= days_since_high <= self.scan_thresholds['max_days_since_high'],
                range_position_pct is not None and range_position_pct <= self.scan_thresholds['max_d0_range_position']
            ])
            
            if scan_criteria:
                # ========== NEW ANALYSIS PARAMETERS ==========
                
                # Find euphoric high index for calculations
                euphoric_idx = None
                for i in range(idx):
                    if abs(df.iloc[i]['high'] - euphoric_high) < 0.01:
                        euphoric_idx = i
                        break
                
                # 1. Big Red Day Analysis
                brd_idx, brd_date, brd_drop = self.find_big_red_day(df, euphoric_idx, idx)
                brd_is_highest_high = False
                brd_atr_ext_9ema = None
                brd_atr_ext_20ema = None
                
                if brd_idx is not None:
                    brd_row = df.iloc[brd_idx]
                    brd_is_highest_high = (brd_idx == euphoric_idx)
                    
                    # Big red day ATR extensions from EMAs using high of that day
                    if not pd.isna(brd_row['ema_9']) and not pd.isna(brd_row['atr']):
                        brd_atr_ext_9ema = (brd_row['high'] - brd_row['ema_9']) / brd_row['atr']
                    if not pd.isna(brd_row['ema_20']) and not pd.isna(brd_row['atr']):
                        brd_atr_ext_20ema = (brd_row['high'] - brd_row['ema_20']) / brd_row['atr']
                
                # 2. Highest High Day Analysis
                hh_atr_ext_9ema = None
                hh_atr_ext_20ema = None
                hh_is_ath = False
                
                if euphoric_idx is not None:
                    hh_row = df.iloc[euphoric_idx]
                    hh_is_ath = self.is_all_time_high(df, euphoric_idx)
                    
                    # Highest high day ATR extensions from EMAs
                    if not pd.isna(hh_row['ema_9']) and not pd.isna(hh_row['atr']):
                        hh_atr_ext_9ema = (euphoric_high - hh_row['ema_9']) / hh_row['atr']
                    if not pd.isna(hh_row['ema_20']) and not pd.isna(hh_row['atr']):
                        hh_atr_ext_20ema = (euphoric_high - hh_row['ema_20']) / hh_row['atr']
                
                # 3. D-1 Trigger Day Analysis
                d1_candle_body_ratio = self.calculate_candle_body_ratio(d_minus_1)
                d1_low_is_5day_low = self.is_lowest_low_in_period(df, prev_idx, 5)
                
                # 4. Anchored VWAP Analysis
                vwap_from_hh = None
                vwap_from_brd = None
                d0_open_vs_vwap_hh = None
                d0_open_vs_vwap_brd = None
                
                if euphoric_idx is not None:
                    vwap_series_hh = self.calculate_vwap_anchored(df, euphoric_idx)
                    if not pd.isna(vwap_series_hh.iloc[idx]):
                        vwap_from_hh = vwap_series_hh.iloc[idx]
                        d0_open_vs_vwap_hh = ((d_0['open'] - vwap_from_hh) / vwap_from_hh) * 100
                
                if brd_idx is not None:
                    vwap_series_brd = self.calculate_vwap_anchored(df, brd_idx)
                    if not pd.isna(vwap_series_brd.iloc[idx]):
                        vwap_from_brd = vwap_series_brd.iloc[idx]
                        d0_open_vs_vwap_brd = ((d_0['open'] - vwap_from_brd) / vwap_from_brd) * 100
                
                # 5. D0 Distance from EMAs
                d0_open_vs_9ema_atr = None
                d0_open_vs_20ema_atr = None
                
                if not pd.isna(d_0['ema_9']) and not pd.isna(d_0['atr']):
                    d0_open_vs_9ema_atr = (d_0['open'] - d_0['ema_9']) / d_0['atr']
                if not pd.isna(d_0['ema_20']) and not pd.isna(d_0['atr']):
                    d0_open_vs_20ema_atr = (d_0['open'] - d_0['ema_20']) / d_0['atr']
                
                # 6. Setup and Fade Grades
                setup_metrics = {
                    'trend_atr': trend_atr_multiples,
                    'extension_atr': extension_atr,
                    'fade_atr': fade_atr,
                    'volume_multiple': volume_multiple
                }
                
                setup_grade = self.calculate_setup_grade(setup_metrics)
                fade_grade = self.calculate_fade_grade(fade_info, setup_metrics)
                
                # ========== ENHANCED SETUP OBJECT ==========
                setup = {
                    # Core identification
                    'symbol': symbol,
                    'date': current_date.strftime('%Y-%m-%d'),
                    
                    # Key grades and dates
                    'setup_grade': setup_grade,
                    'fade_grade': fade_grade,
                    'highest_high_date': euphoric_date.strftime('%Y-%m-%d'),
                    'brd_date': brd_date.strftime('%Y-%m-%d') if brd_date else None,
                    
                    # Original V10 metrics (core ones)
                    'trend_atr': round(trend_atr_multiples, 2),
                    'gap_atr': round(gap_atr, 2),
                    'extension_atr': round(extension_atr, 2),
                    'fade_atr': round(fade_atr, 2),
                    'days_since_high': days_since_high,
                    'fade_type': fade_type,
                    
                    # NEW: Big Red Day Analysis
                    'brd_is_highest_high': brd_is_highest_high,
                    'brd_atr_ext_9ema': round(brd_atr_ext_9ema, 2) if brd_atr_ext_9ema else None,
                    'brd_atr_ext_20ema': round(brd_atr_ext_20ema, 2) if brd_atr_ext_20ema else None,
                    
                    # NEW: Highest High Analysis
                    'hh_atr_ext_9ema': round(hh_atr_ext_9ema, 2) if hh_atr_ext_9ema else None,
                    'hh_atr_ext_20ema': round(hh_atr_ext_20ema, 2) if hh_atr_ext_20ema else None,
                    'hh_is_ath': hh_is_ath,
                    
                    # NEW: D-1 Trigger Analysis
                    'd1_candle_body_ratio': round(d1_candle_body_ratio, 3),
                    'd1_low_is_5day_low': d1_low_is_5day_low,
                    'd1_range_close_pct': round(range_close_pct, 1),
                    'd1_volume_multiple': round(volume_multiple, 2),
                    
                    # NEW: Anchored VWAP Analysis
                    'd0_open_vs_vwap_hh_pct': round(d0_open_vs_vwap_hh, 2) if d0_open_vs_vwap_hh else None,
                    'd0_open_vs_vwap_brd_pct': round(d0_open_vs_vwap_brd, 2) if d0_open_vs_vwap_brd else None,
                    
                    # NEW: D0 EMA Distance Analysis
                    'd0_open_vs_9ema_atr': round(d0_open_vs_9ema_atr, 2) if d0_open_vs_9ema_atr else None,
                    'd0_open_vs_20ema_atr': round(d0_open_vs_20ema_atr, 2) if d0_open_vs_20ema_atr else None,
                    
                    # Additional context
                    'consecutive_reds': fade_info['max_consecutive_reds'] if fade_info['multiple_red_days'] else 0,
                    'range_position_pct': round(range_position_pct, 1) if range_position_pct is not None else None,
                    'scanner_version': 'V10_Enhanced'
                }
                setups.append(setup)
        
        return setups

    def run_scan_v10_enhanced(self, start_date=None, end_date=None, tickers=None):
        """V10 Enhanced scanner with additional analysis columns"""
        start_date = start_date or START_DATE
        end_date = end_date or END_DATE
        tickers = tickers or self.ticker_universe
        
        print(f"🚀 BACKSIDE POP SCANNER V10 ENHANCED - ANALYSIS COLUMNS")
        print(f"📅 Scanning {len(tickers)} tickers from {start_date} to {end_date}")
        print(f"🔧 V10 Enhanced Features:")
        print(f"   • SCAN CRITERIA UNCHANGED - same setups as V10")
        print(f"   • Added setup_grade and fade_grade calculations")
        print(f"   • Added big red day analysis and ATR extensions")
        print(f"   • Added highest high analysis and all-time high detection")
        print(f"   • Added D-1 trigger analysis (candle body ratio, 5-day low)")
        print(f"   • Added anchored VWAP analysis from key dates")
        print(f"   • Added D0 distance from EMAs in ATR terms")
        print("=" * 80)
        
        all_setups = []
        processed = 0
        errors = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.scan_single_ticker_v10_enhanced, ticker, start_date, end_date): ticker 
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
        print(f"🎯 V10 ENHANCED SCAN COMPLETE")
        print(f"📊 Results: {len(all_setups)} setups found")
        print(f"⚠️  Errors: {errors} tickers failed")
        
        if all_setups:
            df_results = pd.DataFrame(all_setups)
            df_results = df_results.sort_values(['setup_grade', 'fade_grade'], ascending=[False, False])
            
            print(f"\n🏆 TOP ENHANCED SETUPS BY GRADE:")
            display_cols = ['symbol', 'date', 'setup_grade', 'fade_grade', 'highest_high_date', 'brd_date', 'fade_type']
            print(df_results[display_cols].head(20).to_string(index=False))
            
            # Enhanced statistics
            print(f"\n📊 ENHANCED ANALYSIS BREAKDOWN:")
            print(f"   Average Setup Grade: {df_results['setup_grade'].mean():.1f}")
            print(f"   Average Fade Grade: {df_results['fade_grade'].mean():.1f}")
            print(f"   All-Time Highs: {df_results['hh_is_ath'].sum()}/{len(df_results)} ({df_results['hh_is_ath'].mean()*100:.1f}%)")
            print(f"   BRD = Highest High: {df_results['brd_is_highest_high'].sum()}/{len(df_results)} ({df_results['brd_is_highest_high'].mean()*100:.1f}%)")
            print(f"   D-1 Low = 5-Day Low: {df_results['d1_low_is_5day_low'].sum()}/{len(df_results)} ({df_results['d1_low_is_5day_low'].mean()*100:.1f}%)")
            
            return df_results
        else:
            print("No setups found matching V10 criteria.")
            return pd.DataFrame()

# Example Usage
if __name__ == "__main__":
    # Initialize V10 Enhanced scanner
    scanner = BacksidePopScannerV10Enhanced()
    
    # Run V10 enhanced scan with analysis columns
    results = scanner.run_scan_v10_enhanced()
    
    if not results.empty:
        # Save enhanced results to CSV
        results.to_csv('backside_pop_v10_enhanced_results.csv', index=False)
        print(f"\n💾 Enhanced results saved to: backside_pop_v10_enhanced_results.csv")
        
        # Display all columns for analysis
        print(f"\n📋 ALL ANALYSIS COLUMNS:")
        for col in results.columns:
            print(f"   • {col}")