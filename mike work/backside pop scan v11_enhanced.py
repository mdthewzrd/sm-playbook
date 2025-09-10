"""
Backside Pop Scanner V11 Enhanced - With Sam's Analysis Parameters
Uses Original V11 Parameters - No V10 Exclusion - Just Adds Analysis Metrics
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

class BacksidePopScannerV11Enhanced:
    """Enhanced Backside Pop Scanner V11 - Original V11 Parameters + Sam's Analysis Metrics"""
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        self.ticker_universe = TICKER_UNIVERSE
        self.max_workers = MAX_WORKERS
        
        # V11 Original relaxed scan thresholds (NO V10 EXCLUSION)
        self.scan_thresholds = {
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
            
            'min_red_days_consecutive': 2,    # Use V11's relaxed requirement
            'outlier_volume_multiple': 2.0,   
            'outlier_range_atr': 2.0,         
            'outlier_fade_atr': 2.0,          
            'fade_lookback_days': 15          
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'backside-scanner-v11-enhanced'})

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

    def calculate_anchored_vwap(self, df, anchor_idx):
        """Calculate anchored VWAP from a specific day"""
        if anchor_idx >= len(df):
            return pd.Series([np.nan] * len(df))
        
        vwap_series = pd.Series([np.nan] * len(df))
        
        cumulative_volume = 0
        cumulative_pv = 0
        
        for i in range(anchor_idx, len(df)):
            typical_price = (df.iloc[i]['high'] + df.iloc[i]['low'] + df.iloc[i]['close']) / 3
            volume = df.iloc[i]['volume']
            
            cumulative_volume += volume
            cumulative_pv += typical_price * volume
            
            if cumulative_volume > 0:
                vwap_series.iloc[i] = cumulative_pv / cumulative_volume
        
        return vwap_series

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

    def find_first_red_day_after_high(self, df, euphoric_idx, setup_idx):
        """Find the first red day after euphoric high (FRD)"""
        if euphoric_idx is None or euphoric_idx >= setup_idx:
            return None, None
        
        for i in range(euphoric_idx + 1, setup_idx):
            if df.iloc[i]['is_red_day']:
                return i, df.iloc[i]['date']
        
        return None, None

    def find_biggest_red_day_after_high(self, df, euphoric_idx, setup_idx):
        """Find the biggest red day after euphoric high by range"""
        if euphoric_idx is None or euphoric_idx >= setup_idx:
            return None, None
        
        max_range_atr = 0
        big_red_idx = None
        
        for i in range(euphoric_idx + 1, setup_idx):
            if df.iloc[i]['is_red_day'] and not pd.isna(df.iloc[i]['range_atr']):
                if df.iloc[i]['range_atr'] > max_range_atr:
                    max_range_atr = df.iloc[i]['range_atr']
                    big_red_idx = i
        
        if big_red_idx is not None:
            return big_red_idx, df.iloc[big_red_idx]['date']
        
        return None, None

    def calculate_sam_parameters(self, df, setup_idx, euphoric_idx, euphoric_high, euphoric_date):
        """Calculate Sam's requested analysis parameters"""
        if setup_idx < 5 or euphoric_idx is None:
            return {}
        
        # Get key rows
        d_minus_1 = df.iloc[setup_idx - 1]
        d_0 = df.iloc[setup_idx]
        euphoric_row = df.iloc[euphoric_idx]
        
        params = {}
        
        # 1. FRD (First Red Day) Analysis
        frd_idx, frd_date = self.find_first_red_day_after_high(df, euphoric_idx, setup_idx)
        if frd_idx is not None:
            frd_row = df.iloc[frd_idx]
            params['frd_date'] = frd_date.strftime('%Y-%m-%d')
            params['frd_high_9ema_atr'] = (frd_row['high'] - frd_row['ema_9']) / frd_row['atr'] if not pd.isna(frd_row['ema_9']) and not pd.isna(frd_row['atr']) else None
            params['frd_high_20ema_atr'] = (frd_row['high'] - frd_row['ema_20']) / frd_row['atr'] if not pd.isna(frd_row['ema_20']) and not pd.isna(frd_row['atr']) else None
        else:
            params['frd_date'] = None
            params['frd_high_9ema_atr'] = None
            params['frd_high_20ema_atr'] = None
        
        # 2. Highest High Day ATR Extensions from EMAs
        params['highest_high_date'] = euphoric_date.strftime('%Y-%m-%d')
        params['hh_high_9ema_atr'] = (euphoric_high - euphoric_row['ema_9']) / euphoric_row['atr'] if not pd.isna(euphoric_row['ema_9']) and not pd.isna(euphoric_row['atr']) else None
        params['hh_high_20ema_atr'] = (euphoric_high - euphoric_row['ema_20']) / euphoric_row['atr'] if not pd.isna(euphoric_row['ema_20']) and not pd.isna(euphoric_row['atr']) else None
        
        # 3. Big Red Day Analysis
        big_red_idx, big_red_date = self.find_biggest_red_day_after_high(df, euphoric_idx, setup_idx)
        if big_red_idx is not None:
            big_red_row = df.iloc[big_red_idx]
            params['big_red_date'] = big_red_date.strftime('%Y-%m-%d')
            params['big_red_high_9ema_atr'] = (big_red_row['high'] - big_red_row['ema_9']) / big_red_row['atr'] if not pd.isna(big_red_row['ema_9']) and not pd.isna(big_red_row['atr']) else None
            params['big_red_high_20ema_atr'] = (big_red_row['high'] - big_red_row['ema_20']) / big_red_row['atr'] if not pd.isna(big_red_row['ema_20']) and not pd.isna(big_red_row['atr']) else None
        else:
            params['big_red_date'] = None
            params['big_red_high_9ema_atr'] = None
            params['big_red_high_20ema_atr'] = None
        
        # 4. All-time high check (simplified to 1-year lookback)
        lookback_1yr = max(0, euphoric_idx - 252)
        year_data = df.iloc[lookback_1yr:euphoric_idx + 1]
        params['is_ath'] = euphoric_high >= year_data['high'].max() if not year_data.empty else False
        
        # 5. D-1 Trigger Day Analysis
        # Body vs range ratio
        d_minus_1_body = abs(d_minus_1['close'] - d_minus_1['open'])
        d_minus_1_range = d_minus_1['range_dollars']
        params['d1_body_range_ratio'] = d_minus_1_body / d_minus_1_range if d_minus_1_range > 0 else 0
        
        # D-1 close position in range (0-100%)
        params['d1_close_position'] = d_minus_1['close_range'] * 100
        
        # 6. D-1 low vs D-5 analysis
        if setup_idx >= 6:
            d5_to_d1_lows = [df.iloc[i]['low'] for i in range(setup_idx - 5, setup_idx)]
            params['d1_is_lowest_low_5d'] = d_minus_1['low'] == min(d5_to_d1_lows)
        else:
            params['d1_is_lowest_low_5d'] = False
        
        # 7. Anchored VWAP from highest high
        vwap_series = self.calculate_anchored_vwap(df, euphoric_idx)
        if not pd.isna(vwap_series.iloc[setup_idx]):
            params['d0_open_avwap_atr'] = (d_0['open'] - vwap_series.iloc[setup_idx]) / d_0['atr'] if not pd.isna(d_0['atr']) else None
        else:
            params['d0_open_avwap_atr'] = None
        
        # Anchored VWAP from big red day
        if big_red_idx is not None and not pd.isna(vwap_series.iloc[setup_idx]):
            big_red_vwap_series = self.calculate_anchored_vwap(df, big_red_idx)
            params['d0_open_brd_avwap_atr'] = (d_0['open'] - big_red_vwap_series.iloc[setup_idx]) / d_0['atr'] if not pd.isna(big_red_vwap_series.iloc[setup_idx]) and not pd.isna(d_0['atr']) else None
        else:
            params['d0_open_brd_avwap_atr'] = None
        
        # 8. Gap and pre-market analysis (simplified without pre-market data)
        params['gap_atr'] = d_0['gap_atr'] if not pd.isna(d_0['gap_atr']) else 0
        
        # D0 open vs EMAs
        params['d0_open_9ema_atr'] = (d_0['open'] - d_0['ema_9']) / d_0['atr'] if not pd.isna(d_0['ema_9']) and not pd.isna(d_0['atr']) else None
        params['d0_open_20ema_atr'] = (d_0['open'] - d_0['ema_20']) / d_0['atr'] if not pd.isna(d_0['ema_20']) and not pd.isna(d_0['atr']) else None
        
        return params

    def analyze_fade_pattern(self, df, setup_idx, euphoric_high):
        """V11: Analyze fade pattern with relaxed criteria"""
        if setup_idx < self.scan_thresholds['fade_lookback_days']:
            return False, None, None
        
        # Find euphoric high index
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
        
        # Method 1: Check for consecutive red days (relaxed)
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
        
        has_multiple_red_days = False
        if max_consecutive_reds >= self.scan_thresholds['min_red_days_consecutive']:
            # Check if the red days had higher than normal volume or range
            red_days_quality = False
            for i, row in fade_data.iterrows():
                if row['is_red_day']:
                    vol_multiple = row['volume_multiple'] if not pd.isna(row['volume_multiple']) else 1.0
                    range_atr = row['range_atr'] if not pd.isna(row['range_atr']) else 1.0
                    
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
                
            fade_from_high = (euphoric_high - row['close']) / row['atr']
            
            is_high_volume = row['volume_multiple'] >= self.scan_thresholds['outlier_volume_multiple']
            is_wide_range = row['range_atr'] >= self.scan_thresholds['outlier_range_atr']
            is_big_fade = fade_from_high >= self.scan_thresholds['outlier_fade_atr']
            
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
        """V11 Enhanced scan with original V11 parameters + Sam's metrics"""
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
            
            # V11 RELAXED FILTERS (ORIGINAL V11 PARAMETERS)
            
            # 1. Basic filters (relaxed)
            if not (self.scan_thresholds['min_price'] <= d_0['close'] <= self.scan_thresholds['max_price']):
                continue
            if d_minus_1['volume'] < self.scan_thresholds['min_volume']:
                continue
            if pd.isna(d_minus_1['atr']) or d_minus_1['atr'] <= 0:
                continue
            
            # 1b. Dollar volume liquidity filter
            d_minus_1_dollar_vol_20d = d_minus_1['avg_dollar_volume_20d'] if not pd.isna(d_minus_1['avg_dollar_volume_20d']) else 0
            if d_minus_1_dollar_vol_20d < self.scan_thresholds['min_dollar_volume_20d']:
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
            
            # Find euphoric high index
            euphoric_idx = None
            for i in range(idx):
                if abs(df.iloc[i]['high'] - euphoric_high) < 0.01:
                    euphoric_idx = i
                    break
            
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
            
            # V11 Relaxed scan criteria (ORIGINAL V11 PARAMETERS)
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
                # Calculate Sam's parameters
                sam_params = self.calculate_sam_parameters(df, idx, euphoric_idx, euphoric_high, euphoric_date)
                
                setup = {
                    'symbol': symbol,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'setup_grade': 'B',  # V11 setups are B-grade
                    'fade_grade': 'TBD',  # Will be determined by actual fade
                    
                    # Sam's key parameters
                    'highest_high_date': sam_params.get('highest_high_date'),
                    'frd_date': sam_params.get('frd_date'),
                    'big_red_date': sam_params.get('big_red_date'),
                    
                    # FRD Analysis
                    'frd_high_9ema_atr': round(sam_params.get('frd_high_9ema_atr', 0), 2) if sam_params.get('frd_high_9ema_atr') is not None else None,
                    'frd_high_20ema_atr': round(sam_params.get('frd_high_20ema_atr', 0), 2) if sam_params.get('frd_high_20ema_atr') is not None else None,
                    
                    # Highest High Analysis  
                    'hh_high_9ema_atr': round(sam_params.get('hh_high_9ema_atr', 0), 2) if sam_params.get('hh_high_9ema_atr') is not None else None,
                    'hh_high_20ema_atr': round(sam_params.get('hh_high_20ema_atr', 0), 2) if sam_params.get('hh_high_20ema_atr') is not None else None,
                    'is_ath': sam_params.get('is_ath', False),
                    
                    # Big Red Day Analysis
                    'big_red_high_9ema_atr': round(sam_params.get('big_red_high_9ema_atr', 0), 2) if sam_params.get('big_red_high_9ema_atr') is not None else None,
                    'big_red_high_20ema_atr': round(sam_params.get('big_red_high_20ema_atr', 0), 2) if sam_params.get('big_red_high_20ema_atr') is not None else None,
                    
                    # D-1 Trigger Analysis
                    'd1_body_range_ratio': round(sam_params.get('d1_body_range_ratio', 0), 3),
                    'd1_close_position': round(sam_params.get('d1_close_position', 0), 1),
                    'd1_is_lowest_low_5d': sam_params.get('d1_is_lowest_low_5d', False),
                    
                    # Anchored VWAP Analysis
                    'd0_open_avwap_atr': round(sam_params.get('d0_open_avwap_atr', 0), 2) if sam_params.get('d0_open_avwap_atr') is not None else None,
                    'd0_open_brd_avwap_atr': round(sam_params.get('d0_open_brd_avwap_atr', 0), 2) if sam_params.get('d0_open_brd_avwap_atr') is not None else None,
                    
                    # Gap and EMA Analysis
                    'gap_atr': round(sam_params.get('gap_atr', 0), 2),
                    'd0_open_9ema_atr': round(sam_params.get('d0_open_9ema_atr', 0), 2) if sam_params.get('d0_open_9ema_atr') is not None else None,
                    'd0_open_20ema_atr': round(sam_params.get('d0_open_20ema_atr', 0), 2) if sam_params.get('d0_open_20ema_atr') is not None else None,
                    
                    # Original core metrics
                    'trend_atr': round(trend_atr_multiples, 2),
                    'extension_atr': round(extension_atr, 2),
                    'fade_atr': round(fade_atr, 2),
                    'fade_type': fade_type,
                    'consecutive_reds': fade_info['max_consecutive_reds'] if fade_info['multiple_red_days'] else 0,
                    'scanner_version': 'V11_Enhanced_Fixed'
                }
                setups.append(setup)
        
        return setups

    def run_scan_v11(self, start_date=None, end_date=None, tickers=None):
        """V11 Enhanced scanner with original V11 parameters + Sam's metrics"""
        start_date = start_date or START_DATE
        end_date = end_date or END_DATE
        tickers = tickers or self.ticker_universe
        
        print(f"🚀 BACKSIDE POP SCANNER V11 ENHANCED FIXED - ORIGINAL V11 PARAMS + SAM'S METRICS")
        print(f"📅 Scanning {len(tickers)} tickers from {start_date} to {end_date}")
        print(f"🔧 Enhanced Logic (FIXED):")
        print(f"   • Uses ORIGINAL V11 relaxed parameters")
        print(f"   • NO V10 EXCLUSION - just adds Sam's analysis metrics")
        print(f"   • Min gap ATR: {self.scan_thresholds['min_gap_atr']} (relaxed)")
        print(f"   • Min extension ATR: {self.scan_thresholds['min_extension_atr']} (relaxed)")
        print(f"   • Min red days consecutive: {self.scan_thresholds['min_red_days_consecutive']} (relaxed)")
        print(f"   • Enhanced with FRD, Big Red Day, ATH analysis, Anchored VWAP, etc.")
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
                        print(f"✅ {ticker}: {len(setups)} enhanced setups found")
                    else:
                        print(f"⚪ {ticker}: No setups")
                except Exception as e:
                    print(f"❌ {ticker}: Error - {str(e)[:50]}")
                    errors += 1
                
                processed += 1
                if processed % 20 == 0:
                    print(f"📊 Progress: {processed}/{len(tickers)} ({processed/len(tickers)*100:.1f}%)")
        
        print("\n" + "=" * 80)
        print(f"🎯 V11 ENHANCED FIXED SCAN COMPLETE")
        print(f"📊 Results: {len(all_setups)} setups found (original V11 params)")
        print(f"⚠️  Errors: {errors} tickers failed")
        
        if all_setups:
            df_results = pd.DataFrame(all_setups)
            df_results = df_results.sort_values(['date', 'trend_atr'], ascending=[False, False])
            
            print(f"\n🏆 V11 ENHANCED SETUPS - KEY METRICS:")
            # Focus on Sam's key columns
            key_cols = ['symbol', 'date', 'setup_grade', 'fade_grade', 'highest_high_date', 'frd_date', 'big_red_date',
                       'hh_high_9ema_atr', 'frd_high_9ema_atr', 'big_red_high_9ema_atr', 'is_ath', 'd1_is_lowest_low_5d']
            
            available_cols = [col for col in key_cols if col in df_results.columns]
            print(df_results[available_cols].to_string(index=False))
            
            print(f"\n📊 ENHANCED ANALYSIS SUMMARY:")
            print(f"   • ATH setups: {df_results['is_ath'].sum()} / {len(df_results)}")
            print(f"   • D-1 lowest low 5d: {df_results['d1_is_lowest_low_5d'].sum()} / {len(df_results)}")
            print(f"   • Setups with FRD: {df_results['frd_date'].notna().sum()} / {len(df_results)}")
            print(f"   • Setups with Big Red Day: {df_results['big_red_date'].notna().sum()} / {len(df_results)}")
            
            return df_results
        else:
            print("No setups found matching V11 criteria.")
            return pd.DataFrame()

# Example Usage
if __name__ == "__main__":
    # Initialize V11 enhanced scanner
    scanner = BacksidePopScannerV11Enhanced()
    
    # Run V11 enhanced scan
    results = scanner.run_scan_v11()
    
    if not results.empty:
        # Save results to CSV
        results.to_csv('main_scan_v11_enhanced_fixed_results.csv', index=False)
        print(f"\n💾 Enhanced results saved to: main_scan_v11_enhanced_fixed_results.csv")
        print(f"📋 Total setups: {len(results)}")
        print(f"📋 Total columns: {len(results.columns)} (enhanced with Sam's parameters)")