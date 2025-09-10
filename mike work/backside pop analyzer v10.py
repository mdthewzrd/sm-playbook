"""
Backside Pop Strategy Analyzer V10 - Fade Pattern Analysis
Complete analysis tool matching V10 scanner with fade pattern requirements
Shows all parameters to debug why setups aren't hitting
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

class BacksidePopAnalyzerV10:
    """
    Complete Backside Pop Strategy Analyzer V10
    Implements V10 scanner logic with fade pattern analysis (no daily trigger)
    """
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        
        # V10 Enhanced scan thresholds (matching V10 scanner exactly)
        self.scan_thresholds = {
            'min_trend_atr': 6.0,           
            'min_gap_atr': 0.7,             # Updated to match scanner
            'min_extension_atr': 2.0,       
            'min_range_close_pct': 70.0,    
            'min_volume_multiple': 0.7,     # Updated to match scanner
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
            'require_d_minus_1_green': True,  # D-1 must be green (close > open)
            
            # V10: Fade Pattern Requirements (replaces daily trigger)
            'min_red_days_consecutive': 2,    # Need at least 2+ consecutive red days
            'outlier_volume_multiple': 2.5,   # OR volume 2.5x+ average for outlier fade day
            'outlier_range_atr': 3.0,         # OR range 3.0+ ATR for outlier fade day
            'outlier_fade_atr': 4.0,          # OR fade 4.0+ ATR for outlier fade day
            'fade_lookback_days': 10          # Look back N days for fade analysis
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'backside-analyzer-v10'})

    def fetch_daily_data_cached(self, symbol, start_date, end_date):
        """Fetch daily OHLCV data from Polygon (matching V10 scanner)"""
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
        """Calculate indicators with V10 enhanced 9/20 EMA deviation bands (matching V10 scanner)"""
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
        df['dev_band_upper_2'] = df['ema_9'] + 0.5 * df['ATR_9']  # V10: 0.5 band for sensitivity
        df['dev_band_lower_1'] = df['ema_20'] - 2.0 * df['ATR_9']
        df['dev_band_lower_2'] = df['ema_20'] - 2.4 * df['ATR_9']
        
        # V10: Use 0.5 deviation band for better sensitivity
        df['open_above_dev_upper'] = df['open'] > df['dev_band_upper_2']
        df['high_above_dev_upper'] = df['high'] > df['dev_band_upper_2']
        
        return df

    def analyze_fade_pattern_detailed(self, df, setup_idx, euphoric_high, euphoric_date):
        """V10: Detailed fade pattern analysis with full debugging info"""
        if setup_idx < self.scan_thresholds['fade_lookback_days']:
            return False, {}, None
        
        # Find euphoric high row
        euphoric_idx = None
        for i in range(setup_idx):
            if (df.iloc[i]['date'] == euphoric_date and 
                abs(df.iloc[i]['high'] - euphoric_high) < 0.01):
                euphoric_idx = i
                break
        
        if euphoric_idx is None or euphoric_idx >= setup_idx - 1:
            return False, {
                'error': f'Could not find euphoric high row for {euphoric_date} with high {euphoric_high}',
                'euphoric_idx_found': euphoric_idx,
                'setup_idx': setup_idx
            }, None
        
        # Analyze fade from euphoric high to setup day
        fade_data = df.iloc[euphoric_idx:setup_idx].copy()
        if len(fade_data) < 2:
            return False, {
                'error': 'Insufficient fade data',
                'fade_period_days': len(fade_data)
            }, None
        
        # Method 1: Detailed consecutive red days analysis
        red_day_analysis = []
        consecutive_reds = 0
        max_consecutive_reds = 0
        red_streak_start = None
        current_streak_start = None
        
        for i, row in fade_data.iterrows():
            is_red = row['is_red_day']
            red_day_analysis.append({
                'date': row['date'],
                'open': row['open'],
                'close': row['close'],
                'is_red': is_red,
                'streak_position': consecutive_reds + 1 if is_red else 0
            })
            
            if is_red:
                if consecutive_reds == 0:
                    red_streak_start = row['date']
                    current_streak_start = row['date']
                consecutive_reds += 1
                max_consecutive_reds = max(max_consecutive_reds, consecutive_reds)
            else:
                consecutive_reds = 0
        
        has_multiple_red_days = max_consecutive_reds >= self.scan_thresholds['min_red_days_consecutive']
        
        # Method 2: Detailed outlier fade day analysis
        outlier_day_analysis = []
        has_outlier_fade_day = False
        best_outlier_info = None
        best_outlier_score = 0
        
        for i, row in fade_data.iterrows():
            if pd.isna(row['volume_multiple']) or pd.isna(row['range_atr']) or pd.isna(row['atr']):
                continue
                
            # Calculate fade from euphoric high to this day's close
            fade_from_high = (euphoric_high - row['close']) / row['atr']
            
            # Check outlier criteria
            is_high_volume = row['volume_multiple'] >= self.scan_thresholds['outlier_volume_multiple']
            is_wide_range = row['range_atr'] >= self.scan_thresholds['outlier_range_atr']
            is_big_fade = fade_from_high >= self.scan_thresholds['outlier_fade_atr']
            
            # Count outlier conditions met
            outlier_conditions = sum([is_high_volume, is_wide_range, is_big_fade])
            
            outlier_info = {
                'date': row['date'],
                'volume_multiple': round(row['volume_multiple'], 2),
                'range_atr': round(row['range_atr'], 2),
                'fade_atr': round(fade_from_high, 2),
                'is_high_volume': is_high_volume,
                'is_wide_range': is_wide_range,
                'is_big_fade': is_big_fade,
                'conditions_met': outlier_conditions,
                'is_outlier': outlier_conditions >= 2
            }
            outlier_day_analysis.append(outlier_info)
            
            # Track best outlier day
            if outlier_conditions >= 2 and outlier_conditions > best_outlier_score:
                has_outlier_fade_day = True
                best_outlier_info = outlier_info.copy()
                best_outlier_score = outlier_conditions
        
        # Return detailed analysis
        fade_pattern_valid = has_multiple_red_days or has_outlier_fade_day
        
        detailed_fade_info = {
            'fade_period_days': len(fade_data),
            'euphoric_idx': euphoric_idx,
            'euphoric_date': euphoric_date,
            'euphoric_high': euphoric_high,
            
            # Red days analysis
            'multiple_red_days': has_multiple_red_days,
            'max_consecutive_reds': max_consecutive_reds,
            'red_streak_start': red_streak_start,
            'red_day_analysis': red_day_analysis,
            'red_days_required': self.scan_thresholds['min_red_days_consecutive'],
            
            # Outlier analysis
            'outlier_fade_day': has_outlier_fade_day,
            'best_outlier_info': best_outlier_info,
            'outlier_day_analysis': outlier_day_analysis,
            'outlier_volume_required': self.scan_thresholds['outlier_volume_multiple'],
            'outlier_range_required': self.scan_thresholds['outlier_range_atr'],
            'outlier_fade_required': self.scan_thresholds['outlier_fade_atr'],
            
            # Overall result
            'fade_valid': fade_pattern_valid,
            'fade_type': "multiple_red" if has_multiple_red_days else ("outlier_fade" if has_outlier_fade_day else "none")
        }
        
        return fade_pattern_valid, detailed_fade_info, detailed_fade_info['fade_type']

    def find_trend_and_euphoric_high_fast(self, df, setup_idx):
        """Find trend and euphoric high (matching V10 scanner)"""
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

    def analyze_setup_v10(self, symbol, setup_date, debug=True, lookback_days=400):
        """Complete setup analysis for V10 with fade pattern requirements"""
        
        # Calculate date range for analysis
        setup_date = pd.to_datetime(setup_date).date()
        start_date = setup_date - timedelta(days=lookback_days)
        end_date = setup_date + timedelta(days=10)
        
        if debug:
            print(f"\n{'='*80}")
            print(f"BACKSIDE POP ANALYZER V10 - {symbol} on {setup_date}")
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
        
        # V10 CRITICAL FILTERS CHECK
        filter_results = {}
        
        # 1. Basic price/volume filters
        filter_results['price_range'] = (self.scan_thresholds['min_price'] <= d_0['close'] <= self.scan_thresholds['max_price'])
        filter_results['volume_check'] = d_minus_1['volume'] >= self.scan_thresholds['min_volume']
        filter_results['atr_valid'] = not (pd.isna(d_minus_1['atr']) or d_minus_1['atr'] <= 0)
        
        # 2. D-1 must be green
        filter_results['d_minus_1_green'] = d_minus_1['close'] > d_minus_1['open']
        
        # 3. V10: D0 open/high must be above 0.5 deviation band
        d0_open_above_dev = d_0['open_above_dev_upper'] if not pd.isna(d_0['open_above_dev_upper']) else False
        d0_high_above_dev = d_0['high_above_dev_upper'] if not pd.isna(d_0['high_above_dev_upper']) else False
        
        filter_results['dev_band_check'] = d0_open_above_dev or d0_high_above_dev
        
        # 4. Find trend and euphoric high
        trend_start_date, trend_start_price, euphoric_date, euphoric_high = \
            self.find_trend_and_euphoric_high_fast(df, setup_idx)
        filter_results['trend_found'] = all([trend_start_date, trend_start_price, euphoric_date, euphoric_high])
        
        # 5. V10: Fade pattern analysis (replaces daily trigger)
        if filter_results['trend_found']:
            fade_valid, fade_info, fade_type = self.analyze_fade_pattern_detailed(
                df, setup_idx, euphoric_high, euphoric_date)
            filter_results['fade_pattern'] = fade_valid
        else:
            fade_valid, fade_info, fade_type = False, {'error': 'No trend found'}, None
            filter_results['fade_pattern'] = False
        
        if debug:
            print(f"\n🔍 V10 CRITICAL FILTERS:")
            print(f"  ✓ Price Range: {filter_results['price_range']} (${d_0['close']:.2f})")
            print(f"  ✓ Volume Check: {filter_results['volume_check']} ({d_minus_1['volume']:,} >= {self.scan_thresholds['min_volume']:,})")
            print(f"  ✓ ATR Valid: {filter_results['atr_valid']} (ATR: {d_minus_1['atr']:.2f})")
            print(f"  ✓ D-1 Green: {filter_results['d_minus_1_green']} (${d_minus_1['open']:.2f} → ${d_minus_1['close']:.2f})")
            print(f"  ✓ Dev Band 0.5: {filter_results['dev_band_check']} (Open: ${d_0['open']:.2f}, High: ${d_0['high']:.2f})")
            if not pd.isna(d_0['dev_band_upper_2']):
                print(f"     - Upper Band 0.5: ${d_0['dev_band_upper_2']:.2f} {'✓' if d0_open_above_dev or d0_high_above_dev else '✗'}")
                print(f"     - EMA9: ${d_0['ema_9']:.2f}, ATR_9: ${d_0['ATR_9']:.2f}")
            print(f"  ✓ Trend Found: {filter_results['trend_found']} (Start: {trend_start_date}, High: {euphoric_date})")
            print(f"  ✓ Fade Pattern: {filter_results['fade_pattern']} (Type: {fade_type})")
        
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
        
        # V10 Scan criteria check
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
        
        # All V10 filters combined
        passes_all_v10_filters = all(filter_results.values()) and scan_criteria
        
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
            
            # V10: Detailed Fade Pattern Analysis
            if 'error' not in fade_info:
                print(f"\n🔄 FADE PATTERN ANALYSIS:")
                print(f"  Pattern Valid: {fade_info['fade_valid']} (Type: {fade_info.get('fade_type', 'none')})")
                print(f"  Fade Period: {fade_info['fade_period_days']} days")
                
                # Red days analysis
                print(f"\n  📉 RED DAYS ANALYSIS:")
                print(f"    Max Consecutive: {fade_info['max_consecutive_reds']} (need ≥{fade_info['red_days_required']})")
                print(f"    Multiple Red Days: {fade_info['multiple_red_days']}")
                if fade_info['red_day_analysis']:
                    print(f"    Daily breakdown:")
                    for day in fade_info['red_day_analysis'][-5:]:  # Show last 5 days
                        print(f"      {day['date']}: O:{day['open']:.2f} C:{day['close']:.2f} Red:{day['is_red']} Streak:{day['streak_position']}")
                
                # Outlier analysis
                print(f"\n  💥 OUTLIER FADE ANALYSIS:")
                print(f"    Has Outlier Day: {fade_info['outlier_fade_day']}")
                print(f"    Required: Vol≥{fade_info['outlier_volume_required']}x, Range≥{fade_info['outlier_range_required']}ATR, Fade≥{fade_info['outlier_fade_required']}ATR")
                if fade_info['best_outlier_info']:
                    best = fade_info['best_outlier_info']
                    print(f"    Best Outlier ({best['date']}): Vol:{best['volume_multiple']}x Range:{best['range_atr']}ATR Fade:{best['fade_atr']}ATR")
                    print(f"    Conditions Met: {best['conditions_met']}/3 (Vol:{best['is_high_volume']} Range:{best['is_wide_range']} Fade:{best['is_big_fade']})")
                
                if fade_info['outlier_day_analysis']:
                    print(f"    All outlier candidates:")
                    for day in fade_info['outlier_day_analysis'][-3:]:  # Show best 3
                        print(f"      {day['date']}: Vol:{day['volume_multiple']}x Range:{day['range_atr']}ATR Fade:{day['fade_atr']}ATR [{day['conditions_met']}/3]")
            else:
                print(f"\n🔄 FADE PATTERN ERROR: {fade_info['error']}")
            
            print(f"\n🎯 V10 SCAN RESULT:")
            print(f"  Passes All Filters: {'✅ YES' if passes_all_v10_filters else '❌ NO'}")
            print(f"  Passes Scan Criteria: {'✅ YES' if scan_criteria else '❌ NO'}")
        
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
            
            # V10 Filter Results
            'v10_filters': filter_results,
            'passes_all_v10_filters': passes_all_v10_filters,
            'passes_scan_criteria': scan_criteria,
            
            # Raw Metrics
            'setup_metrics': setup_data,
            
            # V10 Specific Data
            'deviation_bands': {
                'd0_dev_band_upper_2': d_0['dev_band_upper_2'] if not pd.isna(d_0['dev_band_upper_2']) else 0,
                'd0_ema_9': d_0['ema_9'] if not pd.isna(d_0['ema_9']) else 0,
                'd0_atr_9': d_0['ATR_9'] if not pd.isna(d_0['ATR_9']) else 0,
                'd0_open_price': d_0['open'],
                'd0_high_price': d_0['high'],
                'd0_open_above_dev': d0_open_above_dev,
                'd0_high_above_dev': d0_high_above_dev
            },
            
            'fade_pattern_data': fade_info,
            
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
            print(f"ANALYSIS COMPLETE - {symbol}")
            if passes_all_v10_filters:
                print("✅ WOULD BE DETECTED BY V10 SCANNER")
            else:
                print("❌ WOULD NOT BE DETECTED BY V10 SCANNER")
            print(f"{'='*80}")
        
        return result

    def debug_setup_detailed(self, symbol, setup_date):
        """Detailed debug analysis matching V10 scanner logic"""
        print(f"\n🔬 DETAILED V10 DEBUG ANALYSIS")
        print(f"Symbol: {symbol}")
        print(f"Date: {setup_date}")
        print("-" * 60)
        
        result = self.analyze_setup_v10(symbol, setup_date, debug=True)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return result
        
        # Summary of why it passed or failed V10 filters
        print(f"\n🎯 V10 SCANNER COMPATIBILITY:")
        
        v10_filters = result['v10_filters']
        failed_filters = []
        
        if not v10_filters['price_range']:
            failed_filters.append("Price not in range")
        if not v10_filters['volume_check']:
            failed_filters.append("Volume too low")
        if not v10_filters['atr_valid']:
            failed_filters.append("Invalid ATR")
        if not v10_filters['d_minus_1_green']:
            failed_filters.append("D-1 not green")
        if not v10_filters['dev_band_check']:
            failed_filters.append("Not above deviation band")
        if not v10_filters['fade_pattern']:
            failed_filters.append("Fade pattern not valid")
        if not v10_filters['trend_found']:
            failed_filters.append("No valid trend found")
        if not result['passes_scan_criteria']:
            failed_filters.append("Failed scan criteria")
        
        if failed_filters:
            print("❌ FAILED FILTERS:")
            for failure in failed_filters:
                print(f"  • {failure}")
        else:
            print("✅ ALL V10 FILTERS PASSED")
        
        return result

# Example Usage and Testing
if __name__ == "__main__":
    # Initialize V10 analyzer
    analyzer = BacksidePopAnalyzerV10()
    
    print("BACKSIDE POP ANALYZER V10 - FADE PATTERN ANALYSIS")
    print("=" * 60)
    
    # Test specific examples that should be hitting
    test_examples = [
        {'symbol': 'COIN', 'date': '2024-01-11'},
        {'symbol': 'COIN', 'date': '2024-04-24'}, 
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
    
    print("\nV10 ANALYZER TESTING COMPLETE")
    print("Use analyzer.debug_setup_detailed(symbol, date) for detailed analysis")
    print("=" * 60)