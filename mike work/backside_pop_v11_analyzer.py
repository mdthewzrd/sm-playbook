"""
Backside Pop V11 Parameter Analyzer
Shows all parameter values for each setup to help fine-tune criteria
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
START_DATE = "2023-08-01"
END_DATE = "2023-08-04"
MAX_WORKERS = 16

# ANALYZE SPECIFIC TICKERS (Add the ones you want to examine)
ANALYZE_TICKERS = ['AMD']

class BacksidePopAnalyzer:
    """Analyzer to show all parameter values for V11 setups"""
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        self.max_workers = MAX_WORKERS
        
        # V11 thresholds for reference (UPDATED TO MATCH YOUR CURRENT V11)
        self.scan_thresholds_v11 = {
            'min_trend_atr': 4.0,           
            'min_gap_atr': 0.4,             
            'min_extension_atr': 0.8,       
            'min_range_close_pct': 60.0,    
            'min_volume_multiple': 0.7,     
            'min_change_atr': 0.4,          
            'max_downtrend_slope': -0.20,   
            'min_ema_extension_pct': 0.0,   
            'min_fade_atr': 2.0,            
            'min_days_since_high': 1.0,       
            'max_days_since_high': 40.0,      
            'min_price': 3.0,               
            'max_price': 1000.0,            
            'min_volume': 10_000_000,        
            'require_dev_band_upper': True,   
            'require_d_minus_1_green': True,  
            
            'min_red_days_consecutive': 3,    
            'outlier_volume_multiple': 2.0,   # UPDATED TO MATCH V11
            'outlier_range_atr': 2.0,         # UPDATED TO MATCH V11
            'outlier_fade_atr': 2.0,          # UPDATED TO MATCH V11
            'fade_lookback_days': 15          
        }
        
        # V10 thresholds for comparison
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
        self.session.headers.update({'User-Agent': 'backside-analyzer'})

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
        """Calculate all indicators for analysis"""
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
        
        # Use 0.5 deviation band for better sensitivity
        df['open_above_dev_upper'] = df['open'] > df['dev_band_upper_2']
        df['high_above_dev_upper'] = df['high'] > df['dev_band_upper_2']
        
        return df

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

    def analyze_fade_pattern_detailed(self, df, setup_idx, euphoric_high):
        """Detailed fade pattern analysis with all parameter values"""
        if setup_idx < self.scan_thresholds_v11['fade_lookback_days']:
            return None
        
        # Look back from euphoric high to setup day
        euphoric_idx = None
        for i in range(setup_idx):
            if abs(df.iloc[i]['high'] - euphoric_high) < 0.01:
                euphoric_idx = i
                break
        
        if euphoric_idx is None or euphoric_idx >= setup_idx - 1:
            return None
        
        # Analyze fade from euphoric high to setup day
        fade_data = df.iloc[euphoric_idx:setup_idx].copy()
        if len(fade_data) < 2:
            return None
        
        # Method 1: Consecutive red days analysis
        consecutive_reds = 0
        max_consecutive_reds = 0
        red_streak_start = None
        red_days_detail = []
        
        for i, row in fade_data.iterrows():
            if row['is_red_day']:
                if consecutive_reds == 0:
                    red_streak_start = row['date']
                consecutive_reds += 1
                max_consecutive_reds = max(max_consecutive_reds, consecutive_reds)
                
                # Capture red day details
                red_days_detail.append({
                    'date': row['date'],
                    'volume_multiple': row['volume_multiple'] if not pd.isna(row['volume_multiple']) else 0,
                    'range_atr': row['range_atr'] if not pd.isna(row['range_atr']) else 0,
                    'price_change_pct': ((row['close'] - row['open']) / row['open'] * 100),
                    'high_to_close_fade': ((row['high'] - row['close']) / row['high'] * 100)
                })
            else:
                consecutive_reds = 0
        
        # Check if multiple red days qualify (need 2+ with volume/range)
        has_quality_multiple_reds = False
        if max_consecutive_reds >= 2:
            for red_day in red_days_detail:
                if red_day['volume_multiple'] >= 1.5 or red_day['range_atr'] >= 1.8:
                    has_quality_multiple_reds = True
                    break
        
        # Method 2: Outlier fade day analysis
        outlier_fade_candidates = []
        
        for i, row in fade_data.iterrows():
            if pd.isna(row['volume_multiple']) or pd.isna(row['range_atr']) or pd.isna(row['atr']):
                continue
                
            # Calculate fade from euphoric high to this day's close
            fade_from_high = (euphoric_high - row['close']) / row['atr']
            
            # Check outlier criteria
            is_high_volume = row['volume_multiple'] >= self.scan_thresholds_v11['outlier_volume_multiple']
            is_wide_range = row['range_atr'] >= self.scan_thresholds_v11['outlier_range_atr']
            is_big_fade = fade_from_high >= self.scan_thresholds_v11['outlier_fade_atr']
            
            outlier_conditions = sum([is_high_volume, is_wide_range, is_big_fade])
            
            # DEBUG: Print detailed outlier analysis for each day
            if row['date'].strftime('%Y-%m-%d') >= '2025-08-01':  # Focus on recent dates
                print(f"    🔍 DEBUG - Outlier analysis for {row['date']}:")
                print(f"        Volume: {row['volume_multiple']:.2f}x ≥ {self.scan_thresholds_v11['outlier_volume_multiple']:.1f}? {is_high_volume}")
                print(f"        Range: {row['range_atr']:.2f} ATR ≥ {self.scan_thresholds_v11['outlier_range_atr']:.1f}? {is_wide_range}")  
                print(f"        Fade: {fade_from_high:.2f} ATR ≥ {self.scan_thresholds_v11['outlier_fade_atr']:.1f}? {is_big_fade}")
                print(f"        Conditions met: {outlier_conditions}/3 (need ≥2 to qualify)")
                print(f"        Qualifies: {outlier_conditions >= 2}")
            
            outlier_fade_candidates.append({
                'date': row['date'],
                'volume_multiple': round(row['volume_multiple'], 2),
                'range_atr': round(row['range_atr'], 2),
                'fade_from_high_atr': round(fade_from_high, 2),
                'is_high_volume': is_high_volume,
                'is_wide_range': is_wide_range,
                'is_big_fade': is_big_fade,
                'outlier_conditions_met': outlier_conditions,
                'qualifies_as_outlier': outlier_conditions >= 2,
                'euphoric_high_used': euphoric_high,
                'close_price': row['close'],
                'atr_used': row['atr']
            })
        
        # Find best outlier candidate
        best_outlier = None
        for candidate in outlier_fade_candidates:
            if candidate['qualifies_as_outlier']:
                best_outlier = candidate
                break
        
        return {
            'fade_data_days': len(fade_data),
            'max_consecutive_reds': max_consecutive_reds,
            'red_days_detail': red_days_detail,
            'has_quality_multiple_reds': has_quality_multiple_reds,
            'outlier_fade_candidates': outlier_fade_candidates,
            'best_outlier_candidate': best_outlier,
            'fade_pattern_qualifies': has_quality_multiple_reds or (best_outlier is not None),
            'qualifying_method': 'multiple_red' if has_quality_multiple_reds else 'outlier_fade' if best_outlier else 'none'
        }

    def analyze_single_ticker_detailed(self, symbol, start_date, end_date):
        """Detailed analysis of a single ticker showing all parameter values"""
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
        detailed_analysis = []
        
        for idx in range(1, len(df)):
            current_date = df.iloc[idx]['date']
            if not (target_start <= current_date <= target_end):
                continue
                
            prev_idx = idx - 1
            d_minus_1 = df.iloc[prev_idx]
            d_0 = df.iloc[idx]
            
            # Basic filters
            if not (self.scan_thresholds_v11['min_price'] <= d_0['close'] <= self.scan_thresholds_v11['max_price']):
                continue
            if d_minus_1['volume'] < self.scan_thresholds_v11['min_volume']:
                continue
            if pd.isna(d_minus_1['atr']) or d_minus_1['atr'] <= 0:
                continue
            
            # D-1 must be green
            d_minus_1_green = d_minus_1['close'] > d_minus_1['open']
            if not d_minus_1_green:
                continue
            
            # D0 deviation band check
            d0_open_above_dev = d_0['open_above_dev_upper'] if not pd.isna(d_0['open_above_dev_upper']) else False
            d0_high_above_dev = d_0['high_above_dev_upper'] if not pd.isna(d_0['high_above_dev_upper']) else False
            
            if not (d0_open_above_dev or d0_high_above_dev):
                continue
            
            # Find trend and euphoric high
            trend_start_date, trend_start_price, euphoric_date, euphoric_high = \
                self.find_trend_and_euphoric_high_fast(df, idx)
            
            if not all([trend_start_date, trend_start_price, euphoric_date, euphoric_high]):
                continue
            
            # Detailed fade pattern analysis
            fade_analysis = self.analyze_fade_pattern_detailed(df, idx, euphoric_high)
            
            if fade_analysis is None:
                continue
            
            # Calculate all setup metrics
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
            
            # Check all V11 criteria
            v11_criteria = {
                'trend_atr': {'value': round(trend_atr_multiples, 2), 'threshold': self.scan_thresholds_v11['min_trend_atr'], 'passes': trend_atr_multiples >= self.scan_thresholds_v11['min_trend_atr']},
                'gap_atr': {'value': round(gap_atr, 2), 'threshold': self.scan_thresholds_v11['min_gap_atr'], 'passes': gap_atr >= self.scan_thresholds_v11['min_gap_atr']},
                'extension_atr': {'value': round(extension_atr, 2), 'threshold': self.scan_thresholds_v11['min_extension_atr'], 'passes': extension_atr >= self.scan_thresholds_v11['min_extension_atr']},
                'range_close_pct': {'value': round(range_close_pct, 1), 'threshold': self.scan_thresholds_v11['min_range_close_pct'], 'passes': range_close_pct >= self.scan_thresholds_v11['min_range_close_pct']},
                'volume_multiple': {'value': round(volume_multiple, 2), 'threshold': self.scan_thresholds_v11['min_volume_multiple'], 'passes': volume_multiple >= self.scan_thresholds_v11['min_volume_multiple']},
                'change_atr': {'value': round(change_atr, 2), 'threshold': self.scan_thresholds_v11['min_change_atr'], 'passes': change_atr >= self.scan_thresholds_v11['min_change_atr']},
                'downtrend_slope': {'value': round(downtrend_slope, 3), 'threshold': self.scan_thresholds_v11['max_downtrend_slope'], 'passes': downtrend_slope <= self.scan_thresholds_v11['max_downtrend_slope']},
                'ema_extension_pct': {'value': round(ema_extension_pct, 1), 'threshold': self.scan_thresholds_v11['min_ema_extension_pct'], 'passes': ema_extension_pct >= self.scan_thresholds_v11['min_ema_extension_pct']},
                'fade_atr': {'value': round(fade_atr, 2), 'threshold': self.scan_thresholds_v11['min_fade_atr'], 'passes': fade_atr >= self.scan_thresholds_v11['min_fade_atr']},
                'days_since_high': {'value': days_since_high, 'threshold': f"{self.scan_thresholds_v11['min_days_since_high']}-{self.scan_thresholds_v11['max_days_since_high']}", 'passes': self.scan_thresholds_v11['min_days_since_high'] <= days_since_high <= self.scan_thresholds_v11['max_days_since_high']},
            }
            
            all_criteria_pass = all([criteria['passes'] for criteria in v11_criteria.values()])
            fade_pattern_passes = fade_analysis['fade_pattern_qualifies']
            
            overall_qualifies = all_criteria_pass and fade_pattern_passes
            
            analysis_result = {
                'symbol': symbol,
                'date': current_date.strftime('%Y-%m-%d'),
                'overall_qualifies': overall_qualifies,
                'price': round(d_0['close'], 2),
                'trend_start': trend_start_date.strftime('%Y-%m-%d'),
                'euphoric_high_date': euphoric_date.strftime('%Y-%m-%d'),
                'euphoric_high_price': round(euphoric_high, 2),
                'v11_criteria': v11_criteria,
                'fade_analysis': fade_analysis,
                'raw_data': {
                    'd_minus_1': {
                        'date': (current_date - timedelta(days=1)).strftime('%Y-%m-%d'),
                        'open': round(d_minus_1['open'], 2),
                        'high': round(d_minus_1['high'], 2),
                        'low': round(d_minus_1['low'], 2),
                        'close': round(d_minus_1['close'], 2),
                        'volume': int(d_minus_1['volume']),
                        'was_green': d_minus_1_green
                    },
                    'd_0': {
                        'date': current_date.strftime('%Y-%m-%d'),
                        'open': round(d_0['open'], 2),
                        'high': round(d_0['high'], 2),
                        'low': round(d_0['low'], 2),
                        'close': round(d_0['close'], 2),
                        'volume': int(d_0['volume']),
                        'dev_band_upper': round(d_0['dev_band_upper_2'], 2) if not pd.isna(d_0['dev_band_upper_2']) else None,
                        'open_above_dev': d0_open_above_dev,
                        'high_above_dev': d0_high_above_dev
                    }
                }
            }
            
            detailed_analysis.append(analysis_result)
        
        return detailed_analysis

    def run_detailed_analysis(self, tickers=None, start_date=None, end_date=None):
        """Run detailed analysis on specified tickers"""
        start_date = start_date or START_DATE
        end_date = end_date or END_DATE
        tickers = tickers or ANALYZE_TICKERS
        
        print(f"🔍 BACKSIDE POP V11 PARAMETER ANALYZER")
        print(f"📅 Analyzing {len(tickers)} tickers from {start_date} to {end_date}")
        print(f"🎯 Showing ALL parameter values to help fine-tune criteria")
        print("=" * 80)
        
        all_analysis = []
        
        for ticker in tickers:
            print(f"🔍 Analyzing {ticker}...")
            analysis = self.analyze_single_ticker_detailed(ticker, start_date, end_date)
            if analysis:
                all_analysis.extend(analysis)
                print(f"✅ {ticker}: Found {len(analysis)} potential setups")
            else:
                print(f"⚪ {ticker}: No setups found")
        
        print("\n" + "=" * 80)
        print(f"📊 DETAILED ANALYSIS RESULTS")
        print(f"Found {len(all_analysis)} potential setups")
        
        if all_analysis:
            self.display_detailed_results(all_analysis)
        
        return all_analysis

    def display_detailed_results(self, analysis_results):
        """Display detailed results in a readable format"""
        qualifying_setups = [r for r in analysis_results if r['overall_qualifies']]
        non_qualifying_setups = [r for r in analysis_results if not r['overall_qualifies']]
        
        print(f"\n✅ QUALIFYING SETUPS ({len(qualifying_setups)}):")
        print("=" * 120)
        
        for setup in qualifying_setups:
            self.print_setup_details(setup, "QUALIFYING")
        
        print(f"\n❌ NON-QUALIFYING SETUPS ({len(non_qualifying_setups)}):")
        print("=" * 120)
        
        for setup in non_qualifying_setups:
            self.print_setup_details(setup, "NON-QUALIFYING")

    def print_setup_details(self, setup, status):
        """Print detailed setup information"""
        print(f"\n📊 {status}: {setup['symbol']} on {setup['date']} (${setup['price']})")
        print(f"📈 Trend: {setup['trend_start']} → Euphoric High: {setup['euphoric_high_date']} (${setup['euphoric_high_price']})")
        
        print(f"\n🎯 V11 CRITERIA:")
        for criterion, data in setup['v11_criteria'].items():
            status_icon = "✅" if data['passes'] else "❌"
            print(f"   {status_icon} {criterion}: {data['value']} (≥{data['threshold']})" if 'min' in str(data['threshold']) or 'max' not in str(data['threshold']) else f"   {status_icon} {criterion}: {data['value']} (≤{data['threshold']})")
        
        print(f"\n🔍 FADE PATTERN ANALYSIS:")
        fade = setup['fade_analysis']
        print(f"   📅 Fade lookback days: {fade['fade_data_days']}")
        print(f"   🔴 Max consecutive red days: {fade['max_consecutive_reds']}")
        print(f"   ✅ Quality multiple reds: {fade['has_quality_multiple_reds']}")
        print(f"   🎯 Qualifying method: {fade['qualifying_method']}")
        print(f"   ✅ Overall fade qualifies: {fade['fade_pattern_qualifies']}")
        
        if fade['red_days_detail']:
            print(f"   📊 RED DAYS DETAIL:")
            for i, red_day in enumerate(fade['red_days_detail']):
                print(f"      Day {i+1} ({red_day['date']}): Vol={red_day['volume_multiple']:.2f}x, Range={red_day['range_atr']:.2f}ATR, Change={red_day['price_change_pct']:.1f}%, Fade={red_day['high_to_close_fade']:.1f}%")
        
        if fade['outlier_fade_candidates']:
            print(f"   📊 OUTLIER FADE CANDIDATES:")
            for i, candidate in enumerate(fade['outlier_fade_candidates']):
                status = "✅ QUALIFIES" if candidate['qualifies_as_outlier'] else "❌ No"
                print(f"      Day {i+1} ({candidate['date']}): {status} - Vol={candidate['volume_multiple']}x, Range={candidate['range_atr']}ATR, Fade={candidate['fade_from_high_atr']}ATR")
                print(f"         Conditions: Vol≥{self.scan_thresholds_v11['outlier_volume_multiple']}?{candidate['is_high_volume']}, Range≥{self.scan_thresholds_v11['outlier_range_atr']}?{candidate['is_wide_range']}, Fade≥{self.scan_thresholds_v11['outlier_fade_atr']}?{candidate['is_big_fade']} ({candidate['outlier_conditions_met']}/3)")
        
        print(f"\n📈 RAW DATA:")
        d_minus_1 = setup['raw_data']['d_minus_1']
        d_0 = setup['raw_data']['d_0']
        print(f"   D-1 ({d_minus_1['date']}): O={d_minus_1['open']} H={d_minus_1['high']} L={d_minus_1['low']} C={d_minus_1['close']} V={d_minus_1['volume']:,} Green={d_minus_1['was_green']}")
        print(f"   D0  ({d_0['date']}): O={d_0['open']} H={d_0['high']} L={d_0['low']} C={d_0['close']} V={d_0['volume']:,}")
        print(f"   D0 Dev Band: {d_0['dev_band_upper']} | Open>{d_0['open_above_dev']}, High>{d_0['high_above_dev']}")
        
        print("-" * 120)

# Example Usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = BacksidePopAnalyzer()
    
    # Run detailed analysis
    results = analyzer.run_detailed_analysis()
    
    print(f"\n💾 Analysis complete. Use this data to fine-tune your V11 parameters!")
    print(f"🎯 Focus on the 'OUTLIER FADE CANDIDATES' section to adjust outlier criteria")
    print(f"📊 Look at volume_multiple, range_atr, and fade_atr values to set better thresholds")