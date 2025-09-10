"""
Backside Pop Strategy Analyzer V8 - Daily Trigger Enhanced
Complete SM Playbook Implementation with Daily Trigger Requirement
Enhanced to match V8 scanner with 9/20 deviation bands and daily trigger logic

Framework: Lingua Trading Language
Strategy Type: Short failed breakout strategy  
Context: Extended names attempting to reclaim previous highs in backside territory
Target Win Rate: 65%+
Risk/Reward: 1:2.2 average

V8 Enhancements:
- 9/20 EMA deviation bands (matching chart code)
- Daily trigger requirement (close below low after daily high)
- D-1 must be green requirement
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# SM Playbook Infrastructure Integration
API_KEY = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"  # Polygon API Key
BASE_URL = "https://api.polygon.io"

class BacksidePopAnalyzerV8:
    """
    Complete Backside Pop Strategy Analyzer V8
    Implements V8 scanner logic with daily trigger and deviation band requirements
    """
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        
        # V8 Enhanced scan thresholds (matching V8 scanner)
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
            'require_daily_trigger': True,    # Must have close below low after daily high is set
            'require_d_minus_1_green': True   # D-1 must be green (close > open)
        }
        
        # Grading System Configuration (Total: 100 points)
        self.component_weights = {
            'trend_score': 25,      # Trend size + slope analysis
            'gap_score': 15,        # Gap behavior 
            'extension_score': 20,  # D0 open to D-1 low extension
            'range_score': 15,      # D-1 range close position
            'volume_score': 10,     # Volume multiple
            'change_score': 10,     # D-1 price change
            'slope_score': 5        # Downtrend slope confirmation
        }
        
        # Grade Thresholds
        self.grade_thresholds = {
            'a_plus_min': 90,
            'a_min': 80,
            'b_plus_min': 70,
            'b_min': 60,
            'c_plus_min': 50
        }

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'backside-analyzer-v8'})

    def fetch_daily_data_cached(self, symbol, start_date, end_date):
        """Fetch daily OHLCV data from Polygon (matching V8 scanner)"""
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
                    return df[['date', 'open', 'high', 'low', 'close', 'volume']]
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def calculate_indicators_with_dev_bands(self, df):
        """Calculate indicators with 9/20 EMA deviation bands (matching V8 scanner)"""
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
        
        # 9/20 EMA Deviation Bands (matching V8 scanner)
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
        """Find the daily high and check for daily trigger (matching V8 scanner)"""
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
        """Find trend and euphoric high (matching V8 scanner)"""
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

    def analyze_setup_v8(self, symbol, setup_date, debug=True, lookback_days=400):
        """Complete setup analysis for V8 with daily trigger and deviation bands"""
        
        # Calculate date range for analysis
        setup_date = pd.to_datetime(setup_date).date()
        start_date = setup_date - timedelta(days=lookback_days)
        end_date = setup_date + timedelta(days=10)
        
        if debug:
            print(f"\n{'='*80}")
            print(f"BACKSIDE POP ANALYZER V8 - {symbol} on {setup_date}")
            print(f"{'='*80}")
        
        # Fetch data
        df = self.fetch_daily_data_cached(symbol, start_date.strftime('%Y-%m-%d'), 
                                        end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            return {'error': f'No data available for {symbol}'}
        
        if debug:
            print(f"✓ Fetched {len(df)} days of data")
        
        # Calculate indicators with deviation bands
        df = self.calculate_indicators_with_dev_bands(df)
        
        # Find setup day data
        setup_row = df[df['date'] == setup_date]
        if setup_row.empty:
            return {'error': f'No data for {symbol} on {setup_date}'}
        
        setup_idx = setup_row.index[0]
        if setup_idx == 0:
            return {'error': f'Insufficient historical data for {symbol}'}
        
        prev_idx = setup_idx - 1
        d_minus_1 = df.iloc[prev_idx]
        d_0 = df.iloc[setup_idx]
        
        if debug:
            print(f"✓ Setup day found at index {setup_idx}")
        
        # V8 CRITICAL FILTERS CHECK
        filter_results = {}
        
        # 1. Basic price/volume filters
        filter_results['price_range'] = (self.scan_thresholds['min_price'] <= d_0['close'] <= self.scan_thresholds['max_price'])
        filter_results['volume_check'] = d_minus_1['volume'] >= self.scan_thresholds['min_volume']
        filter_results['atr_valid'] = not (pd.isna(d_minus_1['atr']) or d_minus_1['atr'] <= 0)
        
        # 2. D-1 must be green
        filter_results['d_minus_1_green'] = d_minus_1['close'] > d_minus_1['open']
        
        # 3. D0 open/high must be above deviation band upper (try both 0.5 and 1.0)
        d0_open_above_dev_1 = d_0['open_above_dev_upper_1'] if not pd.isna(d_0['open_above_dev_upper_1']) else False
        d0_high_above_dev_1 = d_0['high_above_dev_upper_1'] if not pd.isna(d_0['high_above_dev_upper_1']) else False
        
        # Also check 0.5 band (upper_2)
        d0_open_above_dev_2 = d_0['open'] > d_0['dev_band_upper_2'] if not pd.isna(d_0['dev_band_upper_2']) else False
        d0_high_above_dev_2 = d_0['high'] > d_0['dev_band_upper_2'] if not pd.isna(d_0['dev_band_upper_2']) else False
        
        # Use the 0.5 band (more permissive) to match what you see on chart
        filter_results['dev_band_check'] = d0_open_above_dev_2 or d0_high_above_dev_2
        filter_results['dev_band_1_0_check'] = d0_open_above_dev_1 or d0_high_above_dev_1
        
        # 4. Daily trigger requirement
        daily_high_date, daily_high_value, daily_trigger, trigger_date, trigger_close = \
            self.find_daily_high_and_trigger(df, setup_idx)
        filter_results['daily_trigger'] = daily_trigger
        
        # 5. Find trend and euphoric high
        trend_start_date, trend_start_price, euphoric_date, euphoric_high = \
            self.find_trend_and_euphoric_high_fast(df, setup_idx)
        filter_results['trend_found'] = all([trend_start_date, trend_start_price, euphoric_date, euphoric_high])
        
        if debug:
            print(f"\n🔍 V8 CRITICAL FILTERS:")
            print(f"  ✓ Price Range: {filter_results['price_range']} (${d_0['close']:.2f})")
            print(f"  ✓ Volume Check: {filter_results['volume_check']} ({d_minus_1['volume']:,})")
            print(f"  ✓ ATR Valid: {filter_results['atr_valid']} (ATR: {d_minus_1['atr']:.2f})")
            print(f"  ✓ D-1 Green: {filter_results['d_minus_1_green']} (${d_minus_1['open']:.2f} → ${d_minus_1['close']:.2f})")
            print(f"  ✓ Dev Band 0.5: {filter_results['dev_band_check']} (Open: ${d_0['open']:.2f}, High: ${d_0['high']:.2f})")
            print(f"     - Upper Band 0.5: ${d_0['dev_band_upper_2']:.2f} {'✓' if d0_open_above_dev_2 or d0_high_above_dev_2 else '✗'}")
            print(f"     - Upper Band 1.0: ${d_0['dev_band_upper_1']:.2f} {'✓' if d0_open_above_dev_1 or d0_high_above_dev_1 else '✗'}")
            print(f"     - EMA9: ${d_0['ema_9']:.2f}, ATR_9: ${d_0['ATR_9']:.2f}")
            print(f"  ✓ Daily Trigger: {filter_results['daily_trigger']} (High: {daily_high_date}, Trigger: {trigger_date})")
            print(f"  ✓ Trend Found: {filter_results['trend_found']} (Start: {trend_start_date}, High: {euphoric_date})")
        
        # Calculate metrics if we have valid trend data
        if filter_results['trend_found']:
            days_since_high = (d_minus_1['date'] - euphoric_date).days
            trend_atr_multiples = (euphoric_high - trend_start_price) / d_minus_1['atr']
            fade_atr = (euphoric_high - d_minus_1['close']) / d_minus_1['atr']
            downtrend_slope = (d_minus_1['close'] - euphoric_high) / max(days_since_high, 1)
        else:
            days_since_high = 0
            trend_atr_multiples = 0
            fade_atr = 0
            downtrend_slope = 0
        
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
            'days_since_high': days_since_high
        }
        
        # V8 Scan criteria check
        scan_criteria = all([
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
        
        # All V8 filters combined
        passes_all_v8_filters = all(filter_results.values()) and scan_criteria
        
        if debug:
            print(f"\n📊 SETUP METRICS:")
            print(f"  Trend: {setup_data['trend_atr_multiples']:.2f} ATR (need ≥{self.scan_thresholds['min_trend_atr']})")
            print(f"  Gap: {setup_data['gap_atr']:.2f} ATR (need ≥{self.scan_thresholds['min_gap_atr']})")
            print(f"  Extension: {setup_data['extension_atr']:.2f} ATR (need ≥{self.scan_thresholds['min_extension_atr']})")
            print(f"  Range Close: {setup_data['range_close_pct']:.1f}% (need ≥{self.scan_thresholds['min_range_close_pct']}%)")
            print(f"  Volume: {setup_data['volume_multiple']:.2f}x (need ≥{self.scan_thresholds['min_volume_multiple']}x)")
            print(f"  Change: {setup_data['change_atr']:.2f} ATR (need ≥{self.scan_thresholds['min_change_atr']})")
            print(f"  Fade: {setup_data['fade_atr']:.2f} ATR (need ≥{self.scan_thresholds['min_fade_atr']})")
            print(f"  Days Since High: {days_since_high} (need {self.scan_thresholds['min_days_since_high']}-{self.scan_thresholds['max_days_since_high']})")
            
            print(f"\n🎯 V8 SCAN RESULT:")
            print(f"  Passes All Filters: {'✅ YES' if passes_all_v8_filters else '❌ NO'}")
            print(f"  Passes Scan Criteria: {'✅ YES' if scan_criteria else '❌ NO'}")
        
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
            
            if debug and performance:
                print(f"\n📈 PERFORMANCE:")
                print(f"  Intraday: {performance['intraday_fade_pct']:.1f}%")
                print(f"  5-Day: {performance['swing_fade_5d_pct']:.1f}%")
        
        # Compile final results
        result = {
            'symbol': symbol,
            'setup_date': setup_date.strftime('%Y-%m-%d'),
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # V8 Filter Results
            'v8_filters': filter_results,
            'passes_all_v8_filters': passes_all_v8_filters,
            'passes_scan_criteria': scan_criteria,
            
            # Grading Results
            'total_score': scoring['total_score'],
            'grade': scoring['grade'],
            'tier': scoring['tier'],
            'component_scores': scoring['component_scores'],
            
            # Raw Metrics
            'setup_metrics': setup_data,
            
            # V8 Specific Data
            'deviation_bands': {
                'd0_dev_band_upper_1': d_0['dev_band_upper_1'] if not pd.isna(d_0['dev_band_upper_1']) else 0,
                'd0_dev_band_upper_2': d_0['dev_band_upper_2'] if not pd.isna(d_0['dev_band_upper_2']) else 0,
                'd0_ema_9': d_0['ema_9'] if not pd.isna(d_0['ema_9']) else 0,
                'd0_atr_9': d_0['ATR_9'] if not pd.isna(d_0['ATR_9']) else 0,
                'd0_open_price': d_0['open'],
                'd0_high_price': d_0['high'],
                'd0_open_above_dev_1': d0_open_above_dev_1,
                'd0_high_above_dev_1': d0_high_above_dev_1,
                'd0_open_above_dev_2': d0_open_above_dev_2,
                'd0_high_above_dev_2': d0_high_above_dev_2
            },
            
            'daily_trigger_data': {
                'daily_high_date': daily_high_date,
                'daily_high_value': daily_high_value,
                'daily_trigger': daily_trigger,
                'trigger_date': trigger_date,
                'trigger_close': trigger_close,
                'days_since_trigger': days_since_trigger
            },
            
            # Context Information
            'trend_context': {
                'trend_start_date': trend_start_date.strftime('%Y-%m-%d') if trend_start_date else None,
                'euphoric_high_date': euphoric_date.strftime('%Y-%m-%d') if euphoric_date else None,
                'days_since_high': days_since_high,
                'euphoric_high_price': float(euphoric_high) if euphoric_high else None,
                'backside_context': euphoric_high and d_0['high'] < euphoric_high
            },
            
            # Performance Tracking (if available)
            'performance': performance,
        }
        
        if debug:
            print(f"\n{'='*80}")
            print(f"ANALYSIS COMPLETE - {symbol} {scoring['grade']} ({scoring['total_score']} points)")
            if passes_all_v8_filters:
                print("✅ WOULD BE DETECTED BY V8 SCANNER")
            else:
                print("❌ WOULD NOT BE DETECTED BY V8 SCANNER")
            print(f"{'='*80}")
        
        return result

    def calculate_setup_score_fast(self, setup_data):
        """Calculate setup score (matching V8 scanner logic)"""
        scores = {}
        
        # Trend scoring (25 points max)
        trend_atr = setup_data.get('trend_atr_multiples', 0)
        if trend_atr >= 12.0:     scores['trend'] = 25
        elif trend_atr >= 8.0:    scores['trend'] = 21
        elif trend_atr >= 6.0:    scores['trend'] = 18
        elif trend_atr >= 4.0:    scores['trend'] = 13
        else:                     scores['trend'] = 0
        
        # Gap scoring (15 points max)
        gap_atr = setup_data.get('gap_atr', 0)
        if gap_atr >= 0.8:        scores['gap'] = 15
        elif gap_atr >= 0.6:      scores['gap'] = 13
        elif gap_atr >= 0.4:      scores['gap'] = 11
        elif gap_atr >= 0.3:      scores['gap'] = 8
        else:                     scores['gap'] = 0
        
        # Extension scoring (20 points max)
        ext_atr = setup_data.get('extension_atr', 0)
        if ext_atr >= 1.5:        scores['extension'] = 20
        elif ext_atr >= 1.2:      scores['extension'] = 17
        elif ext_atr >= 0.8:      scores['extension'] = 14
        elif ext_atr >= 0.5:      scores['extension'] = 10
        else:                     scores['extension'] = 0
        
        # Range scoring (15 points max)
        range_pct = setup_data.get('range_close_pct', 0)
        if range_pct >= 75.0:     scores['range'] = 15
        elif range_pct >= 65.0:   scores['range'] = 13
        elif range_pct >= 50.0:   scores['range'] = 11
        elif range_pct >= 40.0:   scores['range'] = 8
        else:                     scores['range'] = 0
        
        # Volume scoring (10 points max)
        vol_mult = setup_data.get('volume_multiple', 0)
        if vol_mult >= 1.5:       scores['volume'] = 10
        elif vol_mult >= 1.2:     scores['volume'] = 8
        elif vol_mult >= 1.0:     scores['volume'] = 7
        elif vol_mult >= 0.8:     scores['volume'] = 5
        else:                     scores['volume'] = 0
        
        # Change scoring (10 points max)
        change_atr = setup_data.get('change_atr', 0)
        if change_atr >= 1.0:     scores['change'] = 10
        elif change_atr >= 0.8:   scores['change'] = 8
        elif change_atr >= 0.5:   scores['change'] = 7
        elif change_atr >= 0.3:   scores['change'] = 5
        else:                     scores['change'] = 0
        
        # Slope scoring (5 points max) - lower is better
        slope = setup_data.get('downtrend_slope', 0)
        if slope <= -4.0:         scores['slope'] = 5
        elif slope <= -2.0:       scores['slope'] = 4
        elif slope <= -1.0:       scores['slope'] = 4
        elif slope <= -0.5:       scores['slope'] = 3
        else:                     scores['slope'] = 0
        
        total_score = sum(scores.values())
        
        # Grade assignment
        if total_score >= 90: grade, tier = 'A+', 'PRIME'
        elif total_score >= 80: grade, tier = 'A', 'PRIME'
        elif total_score >= 70: grade, tier = 'B+', 'GOOD'
        elif total_score >= 60: grade, tier = 'B', 'GOOD'
        elif total_score >= 50: grade, tier = 'C+', 'WATCHLIST'
        else: grade, tier = 'C', 'WATCHLIST'
        
        return {
            'total_score': total_score,
            'component_scores': scores,
            'grade': grade,
            'tier': tier
        }

    def debug_setup_detailed(self, symbol, setup_date):
        """Detailed debug analysis matching V8 scanner logic"""
        print(f"\n🔬 DETAILED V8 DEBUG ANALYSIS")
        print(f"Symbol: {symbol}")
        print(f"Date: {setup_date}")
        print("-" * 60)
        
        result = self.analyze_setup_v8(symbol, setup_date, debug=True)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return result
        
        # Summary of why it passed or failed V8 filters
        print(f"\n🎯 V8 SCANNER COMPATIBILITY:")
        
        v8_filters = result['v8_filters']
        failed_filters = []
        
        if not v8_filters['price_range']:
            failed_filters.append("Price not in range")
        if not v8_filters['volume_check']:
            failed_filters.append("Volume too low")
        if not v8_filters['atr_valid']:
            failed_filters.append("Invalid ATR")
        if not v8_filters['d_minus_1_green']:
            failed_filters.append("D-1 not green")
        if not v8_filters['dev_band_check']:
            failed_filters.append("Not above deviation band")
        if not v8_filters['daily_trigger']:
            failed_filters.append("No daily trigger")
        if not v8_filters['trend_found']:
            failed_filters.append("No valid trend found")
        if not result['passes_scan_criteria']:
            failed_filters.append("Failed scan criteria")
        
        if failed_filters:
            print("❌ FAILED FILTERS:")
            for failure in failed_filters:
                print(f"  • {failure}")
        else:
            print("✅ ALL V8 FILTERS PASSED")
        
        return result

# Example Usage and Testing
if __name__ == "__main__":
    # Initialize V8 analyzer
    analyzer = BacksidePopAnalyzerV8()
    
    print("BACKSIDE POP ANALYZER V8 - DAILY TRIGGER ENHANCED")
    print("=" * 60)
    
    # Test some examples - EDIT THESE TICKERS AND DATES
    test_examples = [
        {'symbol': 'HOOD', 'date': '2025-03-03'},
        {'symbol': 'MSTR', 'date': '2024-12-05'}, 
        {'symbol': 'TSLA', 'date': '2024-07-15'},
        # Add your own tickers and dates here:
        # {'symbol': 'NVDA', 'date': '2024-11-15'},
        # {'symbol': 'SMCI', 'date': '2024-10-20'},
    ]
    
    for example in test_examples:
        try:
            result = analyzer.debug_setup_detailed(example['symbol'], example['date'])
            print("\n" + "="*60)
        except Exception as e:
            print(f"Error analyzing {example['symbol']}: {e}")
    
    print("\nV8 ANALYZER TESTING COMPLETE")
    print("Use analyzer.debug_setup_detailed(symbol, date) for detailed analysis")
    print("=" * 60)