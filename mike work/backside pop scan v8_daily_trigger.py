"""
Backside Pop Scanner V8 - Daily Trigger Required
Enhanced with daily trigger requirement: must have a close below a low AFTER the daily high is set
This ensures a proper daily-level breakdown before the backside pop setup
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
START_DATE = "2024-12-01"
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

class BacksidePopScannerV8:
    """Backside Pop Scanner V8 with Daily Trigger Requirement"""
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        self.ticker_universe = TICKER_UNIVERSE
        self.max_workers = MAX_WORKERS
        
        # Data cache for performance optimization
        self.data_cache = {}
        
        # Enhanced scan thresholds with daily trigger requirement
        self.scan_thresholds = {
            'min_trend_atr': 6.0,           
            'min_gap_atr': 0.4,             
            'min_extension_atr': 1.0,       
            'min_range_close_pct': 70.0,    
            'min_volume_multiple': 0.7,     
            'min_change_atr': 0.25,         
            'max_downtrend_slope': -1.0,    
            'min_ema_extension_pct': 20.0,  
            'min_fade_atr': 1.5,            
            'min_days_since_high': 1.0,       
            'max_days_since_high': 30.0,      
            'min_price': 10.0,              
            'max_price': 1000.0,            
            'min_volume': 1_000_000,
            'require_dev_band_upper': True,   # D0 open/high must be above 9/20 deviation band upper 1
            'require_daily_trigger': True     # NEW: Must have close below low after daily high is set
        }
        
        # Grading system
        self.grading_system = {
            'weights': {
                'trend_weight': 25, 'gap_weight': 15, 'extension_weight': 20,  
                'range_weight': 15, 'volume_weight': 10, 'change_weight': 10, 'slope_weight': 5        
            },
            'grade_breakpoints': {'A+': 90, 'A': 80, 'B+': 70, 'B': 60, 'C+': 50, 'C': 40},
            'trend_scoring': {'A+_min': 12.0, 'A_min': 8.0, 'B+_min': 6.0, 'B_min': 4.0},
            'gap_scoring': {'A+_min': 0.8, 'A_min': 0.6, 'B+_min': 0.4, 'B_min': 0.3},
            'extension_scoring': {'A+_min': 1.5, 'A_min': 1.2, 'B+_min': 0.8, 'B_min': 0.5},
            'range_scoring': {'A+_min': 75.0, 'A_min': 65.0, 'B+_min': 50.0, 'B_min': 40.0},
            'volume_scoring': {'A+_min': 1.5, 'A_min': 1.2, 'B+_min': 1.0, 'B_min': 0.8},
            'change_scoring': {'A+_min': 1.0, 'A_min': 0.8, 'B+_min': 0.5, 'B_min': 0.3},
            'downtrend_slope_scoring': {'A+_max': -4.0, 'A_max': -2.0, 'B+_max': -1.0, 'B_max': -0.5}
        }

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'backside-scanner-v8'})

    def fetch_daily_data_cached(self, symbol, start_date, end_date):
        """Cached data fetching with connection pooling"""
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {'adjusted': 'true', 'sort': 'asc', 'apikey': self.api_key}
        
        try:
            time.sleep(0.02)
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
                    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                    
                    # Cache the result
                    self.data_cache[cache_key] = df
                    return df
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def calculate_indicators_with_dev_bands(self, df):
        """Calculate indicators with 9/20 EMA deviation bands and daily trigger detection"""
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
        
        # 9/20 EMA Deviation Bands
        df['ATR_9'] = df['true_range'].rolling(9, min_periods=3).mean()
        df['dev_band_upper_1'] = df['ema_9'] + 1.0 * df['ATR_9']
        df['dev_band_upper_2'] = df['ema_9'] + 0.5 * df['ATR_9']
        df['dev_band_lower_1'] = df['ema_20'] - 2.0 * df['ATR_9']
        df['dev_band_lower_2'] = df['ema_20'] - 2.4 * df['ATR_9']
        
        # Check if D0 open/high is above upper deviation band
        df['open_above_dev_upper_1'] = df['open'] > df['dev_band_upper_1']
        df['high_above_dev_upper_1'] = df['high'] > df['dev_band_upper_1']
        
        return df

    def find_daily_high_and_trigger(self, df, setup_idx):
        """Find the daily high and check for daily trigger (close below low after high)"""
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
        
        # Check for daily trigger: close below the low of the daily high bar OR previous candle's low
        # Can happen on the same day as the high OR after the high is set
        
        # First check if the high day itself has a trigger
        daily_high_close = lookback_data.loc[daily_high_idx, 'close']
        
        # Get previous candle's low (day before the high day)
        # Find the position of the high day in the lookback_data
        high_day_position = lookback_data.index.get_loc(daily_high_idx)
        if high_day_position > 0:
            prev_candle_idx = lookback_data.index[high_day_position - 1]
            prev_candle_low = lookback_data.loc[prev_candle_idx, 'low']
        else:
            prev_candle_low = float('inf')  # No previous candle
        
        # Trigger can be: close < same day low OR close < previous candle low
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
                # Look for any close below the low of the daily high bar OR previous candle low
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

    def scan_symbol_optimized(self, symbol, scan_date, lookback_days=400):
        """Optimized single symbol scanning with daily trigger requirement"""
        try:
            scan_date = pd.to_datetime(scan_date).date()
            start_date = scan_date - timedelta(days=lookback_days)
            end_date = scan_date + timedelta(days=10)
            
            # Fetch cached data
            df = self.fetch_daily_data_cached(symbol, start_date.strftime('%Y-%m-%d'), 
                                            end_date.strftime('%Y-%m-%d'))
            
            if df.empty or len(df) < 30:
                return None
            
            # Calculate indicators with deviation bands
            df = self.calculate_indicators_with_dev_bands(df)
            
            # Find setup day
            setup_mask = df['date'] == scan_date
            if not setup_mask.any():
                return None
            
            setup_idx = df[setup_mask].index[0]
            if setup_idx == 0:
                return None
            
            prev_idx = setup_idx - 1
            d_minus_1 = df.iloc[prev_idx]
            d_0 = df.iloc[setup_idx]
            
            # Fast pre-filters (fail fast)
            if (d_0['close'] < self.scan_thresholds['min_price'] or 
                d_0['close'] > self.scan_thresholds['max_price'] or
                d_minus_1['volume'] < self.scan_thresholds['min_volume'] or
                pd.isna(d_minus_1['atr']) or d_minus_1['atr'] <= 0 or
                d_minus_1['close'] <= d_minus_1['open']):  # D-1 must be green
                return None
            
            # CRITICAL: D0 open OR high must be above 9/20 deviation band upper (use 0.5 band)
            if self.scan_thresholds['require_dev_band_upper']:
                # Use 0.5 multiplier band (upper_2) to match chart visual
                d0_open_above_dev = d_0['open'] > d_0['dev_band_upper_2'] if not pd.isna(d_0['dev_band_upper_2']) else False
                d0_high_above_dev = d_0['high'] > d_0['dev_band_upper_2'] if not pd.isna(d_0['dev_band_upper_2']) else False
                
                if not (d0_open_above_dev or d0_high_above_dev):
                    return None
            
            # CRITICAL: Check for daily trigger (close below low after daily high)
            daily_high_date, daily_high_value, daily_trigger, trigger_date, trigger_close = \
                self.find_daily_high_and_trigger(df, setup_idx)
            
            if self.scan_thresholds['require_daily_trigger'] and not daily_trigger:
                return None
            
            # Find trend and euphoric high (optimized)
            trend_start_date, trend_start_price, euphoric_date, euphoric_high = \
                self.find_trend_and_euphoric_high_fast(df, setup_idx)
            
            if not all([trend_start_date, trend_start_price, euphoric_date, euphoric_high]):
                return None
            
            # Calculate metrics (vectorized where possible)
            days_since_high = (d_minus_1['date'] - euphoric_date).days
            trend_atr_multiples = (euphoric_high - trend_start_price) / d_minus_1['atr']
            fade_atr = (euphoric_high - d_minus_1['close']) / d_minus_1['atr']
            downtrend_slope = (d_minus_1['close'] - euphoric_high) / max(days_since_high, 1)
            
            # Calculate days since daily trigger
            days_since_trigger = 0
            if trigger_date:
                days_since_trigger = (d_minus_1['date'] - trigger_date).days
            
            # Compile setup data
            setup_data = {
                'trend_atr_multiples': trend_atr_multiples,
                'gap_atr': d_0['gap_atr'] if not pd.isna(d_0['gap_atr']) else 0,
                'extension_atr': d_0['extension_atr'] if not pd.isna(d_0['extension_atr']) else 0,
                'range_close_pct': d_minus_1['close_range'] * 100,
                'volume_multiple': d_minus_1['volume_multiple'] if not pd.isna(d_minus_1['volume_multiple']) else 0,
                'change_atr': abs(d_minus_1['price_change_atr']) if not pd.isna(d_minus_1['price_change_atr']) else 0,
                'downtrend_slope': downtrend_slope,
                'ema_extension_pct': ((d_0['open'] - d_0['ema_89']) / d_0['ema_89']) * 100 if d_0['ema_89'] > 0 else 0,
                'fade_atr': fade_atr,
                'days_since_high': days_since_high,
                # Deviation band metrics (using 0.5 band)
                'd0_open_above_dev_upper_2': d0_open_above_dev,
                'd0_high_above_dev_upper_2': d0_high_above_dev,
                'd0_dev_band_upper_2': d_0['dev_band_upper_2'] if not pd.isna(d_0['dev_band_upper_2']) else 0,
                'd0_open_price': d_0['open'],
                'd0_high_price': d_0['high'],
                # Daily trigger metrics
                'daily_high_date': daily_high_date,
                'daily_high_value': daily_high_value,
                'daily_trigger': daily_trigger,
                'trigger_date': trigger_date,
                'trigger_close': trigger_close,
                'days_since_trigger': days_since_trigger
            }
            
            # Enhanced criteria check
            passes_criteria = all([
                setup_data['trend_atr_multiples'] >= self.scan_thresholds['min_trend_atr'],
                setup_data['gap_atr'] >= self.scan_thresholds['min_gap_atr'],
                setup_data['extension_atr'] >= self.scan_thresholds['min_extension_atr'],
                setup_data['range_close_pct'] >= self.scan_thresholds['min_range_close_pct'],
                setup_data['volume_multiple'] >= self.scan_thresholds['min_volume_multiple'],
                setup_data['change_atr'] >= self.scan_thresholds['min_change_atr'],
                setup_data['downtrend_slope'] <= self.scan_thresholds['max_downtrend_slope'],
                setup_data['ema_extension_pct'] >= self.scan_thresholds['min_ema_extension_pct'],
                setup_data['fade_atr'] >= self.scan_thresholds['min_fade_atr'],
                self.scan_thresholds['min_days_since_high'] <= days_since_high <= self.scan_thresholds['max_days_since_high']
            ])
            
            if not passes_criteria:
                return None
            
            # Fast scoring calculation
            scoring = self.calculate_setup_score_fast(setup_data)
            
            # Performance tracking
            performance = {}
            if setup_idx + 5 < len(df):
                d0_open, d0_close = d_0['open'], d_0['close']
                d5_close = df.iloc[setup_idx + 5]['close']
                performance = {
                    'intraday_fade_pct': ((d0_close - d0_open) / d0_open) * 100,
                    'swing_fade_5d_pct': ((d5_close - d0_open) / d0_open) * 100
                }
            
            return {
                'symbol': symbol,
                'scan_date': scan_date.strftime('%Y-%m-%d'),
                'passes_scan': True,
                'total_score': scoring['total_score'],
                'grade': scoring['grade'],
                'component_scores': scoring['component_scores'],
                'setup_metrics': setup_data,
                'performance': performance,
                'trend_context': {
                    'trend_start': trend_start_date.strftime('%Y-%m-%d'),
                    'euphoric_high': euphoric_date.strftime('%Y-%m-%d'),
                    'days_since_high': days_since_high
                }
            }
            
        except Exception:
            return None

    def calculate_setup_score_fast(self, setup_data):
        """Optimized scoring calculation"""
        scores = {}
        weights = self.grading_system['weights']
        
        # Vectorized scoring where possible
        for component in ['trend', 'gap', 'extension', 'range', 'volume', 'change']:
            value = setup_data.get(f"{component}_atr_multiples" if component == 'trend' else 
                                 f"{component}_close_pct" if component == 'range' else 
                                 f"{component}_multiple" if component == 'volume' else 
                                 f"{component}_atr", 0)
            
            scoring_config = self.grading_system.get(f"{component}_scoring", {})
            weight = weights.get(f"{component}_weight", 0)
            
            if value >= scoring_config.get('A+_min', 999):
                scores[component] = weight
            elif value >= scoring_config.get('A_min', 999):
                scores[component] = int(weight * 0.85)
            elif value >= scoring_config.get('B+_min', 999):
                scores[component] = int(weight * 0.70)
            elif value >= scoring_config.get('B_min', 999):
                scores[component] = int(weight * 0.50)
            else:
                scores[component] = 0
        
        # Handle slope separately (lower is better)
        slope_value = setup_data.get('downtrend_slope', 0)
        slope_config = self.grading_system['downtrend_slope_scoring']
        slope_weight = weights['slope_weight']
        
        if slope_value <= slope_config['A+_max']:
            scores['slope'] = slope_weight
        elif slope_value <= slope_config['A_max']:
            scores['slope'] = int(slope_weight * 0.85)
        elif slope_value <= slope_config['B+_max']:
            scores['slope'] = int(slope_weight * 0.70)
        elif slope_value <= slope_config['B_max']:
            scores['slope'] = int(slope_weight * 0.50)
        else:
            scores['slope'] = 0
        
        total_score = sum(scores.values())
        
        # Fast grade assignment
        if total_score >= 90: grade = 'A+'
        elif total_score >= 80: grade = 'A'
        elif total_score >= 70: grade = 'B+'
        elif total_score >= 60: grade = 'B'
        elif total_score >= 50: grade = 'C+'
        else: grade = 'C'
        
        return {'total_score': total_score, 'component_scores': scores, 'grade': grade}

    def generate_date_range(self, start_date, end_date):
        """Fast date range generation"""
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()
        date_range = pd.date_range(start=start, end=end, freq='B')  # Business days only
        return [d.date() for d in date_range]

    def run_historical_scan_threaded(self, start_date, end_date):
        """High-performance threaded historical scan with daily trigger requirement"""
        print(f"BACKSIDE POP SCANNER V8 - DAILY TRIGGER REQUIRED")
        print(f"Date Range: {start_date} to {end_date}")
        print(f"Tickers: {len(self.ticker_universe)} symbols")
        print(f"Threading: {self.max_workers} workers")
        print(f"CRITICAL FILTERS:")
        print(f"  • D0 open/high must be above 9/20 deviation band upper")
        print(f"  • D-1 must be GREEN (close > open)")
        print(f"  • DAILY TRIGGER: Must have close below low AFTER daily high is set")
        print("-" * 80)
        
        # Generate all scan tasks
        trading_dates = self.generate_date_range(start_date, end_date)
        all_tasks = [(symbol, date) for date in trading_dates for symbol in self.ticker_universe]
        total_tasks = len(all_tasks)
        
        print(f"Total scans: {total_tasks:,} ({len(trading_dates)} dates × {len(self.ticker_universe)} symbols)")
        print("Starting threaded scan...\n")
        
        all_results = []
        completed = 0
        start_time = time.time()
        
        # Execute with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.scan_symbol_optimized, symbol, date): (symbol, date)
                for symbol, date in all_tasks
            }
            
            # Process results as they complete
            for future in as_completed(future_to_task):
                symbol, scan_date = future_to_task[future]
                completed += 1
                
                try:
                    result = future.result(timeout=15)
                    if result:
                        all_results.append(result)
                        trigger_status = "🟢" if result['setup_metrics']['daily_trigger'] else "🔴"
                        dev_status = "✓" if (result['setup_metrics']['d0_open_above_dev_upper_1'] or 
                                           result['setup_metrics']['d0_high_above_dev_upper_1']) else "✗"
                        
                        trigger_info = ""
                        if result['setup_metrics']['trigger_date']:
                            trigger_info = f"Trigger: {result['setup_metrics']['trigger_date']} (${result['setup_metrics']['trigger_close']:.2f})"
                        
                        print(f"{trigger_status}{dev_status} {symbol} {scan_date} - {result['grade']} ({result['total_score']}) | "
                              f"High: ${result['setup_metrics']['daily_high_value']:.2f} | {trigger_info}")
                    
                    # Progress updates
                    if completed % 500 == 0 or completed == total_tasks:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        remaining = (total_tasks - completed) / rate if rate > 0 else 0
                        progress = (completed / total_tasks) * 100
                        
                        print(f"\nProgress: {completed:,}/{total_tasks:,} ({progress:.1f}%) | "
                              f"Found: {len(all_results)} setups | "
                              f"Rate: {rate:.1f}/sec | "
                              f"ETA: {remaining/60:.1f}min\n")
                        
                except Exception:
                    pass
        
        # Final results
        elapsed_total = time.time() - start_time
        all_results.sort(key=lambda x: (x['scan_date'], -x['total_score']))
        
        print(f"\n{'='*80}")
        print(f"SCAN COMPLETE - V8 WITH DAILY TRIGGER")
        print(f"{'='*80}")
        print(f"Total time: {elapsed_total/60:.1f} minutes")
        print(f"Average rate: {total_tasks/elapsed_total:.1f} scans/second")
        print(f"Setups found: {len(all_results)}")
        print(f"Hit rate: {len(all_results)/total_tasks*100:.4f}%")
        
        if all_results:
            # Performance summary
            grades = {}
            trigger_count = 0
            for result in all_results:
                grades[result['grade']] = grades.get(result['grade'], 0) + 1
                if result['setup_metrics']['daily_trigger']:
                    trigger_count += 1
            
            print(f"Grade distribution: {dict(sorted(grades.items()))}")
            print(f"Daily triggers found: {trigger_count}/{len(all_results)} ({trigger_count/len(all_results)*100:.1f}%)")
            
            # Performance metrics
            perf_results = [r for r in all_results if r['performance'] and r['performance'].get('swing_fade_5d_pct')]
            if perf_results:
                avg_perf = np.mean([r['performance']['swing_fade_5d_pct'] for r in perf_results])
                success_count = len([r for r in perf_results if r['performance']['swing_fade_5d_pct'] < -5])
                success_rate = success_count / len(perf_results) * 100
                print(f"Avg 5-day performance: {avg_perf:.1f}%")
                print(f"Success rate (>5% fade): {success_rate:.1f}%")
            
            # Results table with daily trigger info
            print(f"\n{'Date':<12} {'Symbol':<6} {'Score':<5} {'Grade':<5} {'DailyHigh':<10} {'TriggerDate':<12} {'TriggerClose':<12} {'DaysToTrig':<10} {'Perf':<6}")
            print("-" * 120)
            
            for result in all_results:
                date_str = result['scan_date']
                symbol = result['symbol']
                score = result['total_score']
                grade = result['grade']
                daily_high = f"${result['setup_metrics']['daily_high_value']:.2f}"
                trigger_date = str(result['setup_metrics']['trigger_date']) if result['setup_metrics']['trigger_date'] else "None"
                trigger_close = f"${result['setup_metrics']['trigger_close']:.2f}" if result['setup_metrics']['trigger_close'] else "N/A"
                days_to_trig = str(result['setup_metrics']['days_since_trigger']) if result['setup_metrics']['trigger_date'] else "N/A"
                
                # Performance
                if result['performance'] and result['performance'].get('swing_fade_5d_pct'):
                    perf = f"{result['performance']['swing_fade_5d_pct']:.1f}%"
                else:
                    perf = "N/A"
                
                print(f"{date_str:<12} {symbol:<6} {score:<5} {grade:<5} {daily_high:<10} {trigger_date:<12} {trigger_close:<12} {days_to_trig:<10} {perf:<6}")
                
        else:
            print("No setups found. Consider adjusting scan thresholds.")
        
        # Cache statistics
        print(f"\nCache efficiency: {len(self.data_cache)} symbols cached")
        
        return all_results

def main():
    """Run the V8 scanner with daily trigger requirement"""
    scanner = BacksidePopScannerV8()
    scanner.run_historical_scan_threaded(START_DATE, END_DATE)

if __name__ == "__main__":
    main()