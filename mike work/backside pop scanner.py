"""
Configurable Backside Pop Scanner
Fully customizable parameters and grading system
Test Universe: HOOD, MSTR, SMCI, IBIT

Usage:
1. Set your scan thresholds (minimum values to qualify)
2. Set your grading thresholds (A+, A, B+, etc.)
3. Run scan on test universe
4. Adjust parameters based on results
"""

import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings("ignore")

API_KEY = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
BASE_URL = "https://api.polygon.io"

class ConfigurableBacksideScanner:
    """
    Fully configurable Backside Pop scanner with adjustable parameters
    """
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        
        # Default scan thresholds (minimum values to qualify for scan)
        self.scan_thresholds = {
            'min_trend_atr': 5.0,           # Minimum trend size in ATR
            'min_gap_atr': 0.4,             # Minimum gap size in ATR
            'min_extension_atr': 0.8,       # Minimum extension in ATR
            'min_range_close_pct': 60.0,    # Minimum D-1 close range %
            'min_volume_multiple': 0.7,     # Minimum volume multiple
            'min_change_atr': 0.25,          # Minimum D-1 change in ATR
            'max_downtrend_slope': -0.15,    # Maximum downtrend slope (more negative = steeper)
            'min_ema_extension_pct': 10.0,   # Minimum % above EMA89
            'min_fade_atr': 1.5,            # Minimum fade from high in ATR
            'min_days_since_high': 1,       # Minimum days since euphoric high
            'max_days_since_high': 90       # Maximum days since euphoric high
        }
        
        # Grading system thresholds
        self.grading_system = {
            # Component weights (must total 100)
            'weights': {
                'trend_weight': 20,      # Trend size + slope
                'gap_weight': 15,        # Gap behavior
                'extension_weight': 15,  # Extension analysis  
                'range_weight': 15,      # D-1 range close
                'volume_weight': 15,     # Volume multiple
                'change_weight': 10,     # D-1 price change
                'slope_weight': 10        # Downtrend slope
            },
            
            # Grade breakpoints (0-100 scale)
            'grade_breakpoints': {
                'A+': 85,
                'A': 70,
                'B+': 60,
                'B': 50,
                'C+': 40,
                'C': 30
            },
            
            # Component scoring thresholds
            'trend_scoring': {
                'A+_min': 9.0,         # ATR multiples for A+ 
                'A_min': 6.0,           # ATR multiples for A
                'B+_min': 5.0,          # ATR multiples for B+
                'B_min': 4.0            # ATR multiples for B
            },
            
            'slope_scoring': {
                'A+_min': 1.5,          # % per day for A+
                'A_min': 1.0,           # % per day for A  
                'B+_min': 0.75,          # % per day for B+
                'B_min': 0.5           # % per day for B
            },
            
            'gap_scoring': {
                'A+_min': 0.8,          # ATR for A+
                'A_min': 0.5,           # ATR for A
                'B+_min': 0.4,          # ATR for B+
                'B_min': 0.3            # ATR for B
            },
            
            'extension_scoring': {
                'A+_min': 1.5,          # ATR for A+
                'A_min': 1.2,           # ATR for A
                'B+_min': 1,          # ATR for B+
                'B_min': 0.5            # ATR for B
            },
            
            'range_scoring': {
                'A+_min': 85.0,         # % for A+
                'A_min': 75.0,          # % for A
                'B+_min': 60.0,         # % for B+
                'B_min': 50.0           # % for B
            },
            
            'volume_scoring': {
                'A+_min': 1.25,          # Multiple for A+
                'A_min': 1.0,           # Multiple for A
                'B+_min': 0.75,          # Multiple for B+
                'B_min': 0.5            # Multiple for B
            },
            
            'change_scoring': {
                'A+_min': 0.5,          # ATR for A+
                'A_min': 0.25,           # ATR for A
                'B+_min': 0.15,          # ATR for B+
                'B_min': 0.05            # ATR for B
            },
            
            'downtrend_slope_scoring': {
                'A+_max': -15.0,         # Slope for A+ (more negative)
                'A_max': -5.0,          # Slope for A
                'B+_max': -2.0,         # Slope for B+
                'B_max': -1.0           # Slope for B
            }
        }

    def update_scan_thresholds(self, **kwargs):
        """Update scan threshold parameters"""
        for key, value in kwargs.items():
            if key in self.scan_thresholds:
                self.scan_thresholds[key] = value
                print(f"Updated {key}: {value}")
            else:
                print(f"Warning: {key} not found in scan_thresholds")

    def update_grading_weights(self, **kwargs):
        """Update component weights (must total 100)"""
        for key, value in kwargs.items():
            if key in self.grading_system['weights']:
                self.grading_system['weights'][key] = value
                print(f"Updated weight {key}: {value}")
            else:
                print(f"Warning: {key} not found in weights")
        
        total_weight = sum(self.grading_system['weights'].values())
        if total_weight != 100:
            print(f"WARNING: Total weights = {total_weight}, should be 100")

    def update_grade_breakpoints(self, **kwargs):
        """Update grade breakpoint thresholds"""
        for key, value in kwargs.items():
            if key in self.grading_system['grade_breakpoints']:
                self.grading_system['grade_breakpoints'][key] = value
                print(f"Updated grade {key}: {value}")
            else:
                print(f"Warning: {key} not found in grade_breakpoints")

    def update_component_scoring(self, component, **kwargs):
        """Update scoring thresholds for a specific component"""
        component_key = f"{component}_scoring"
        if component_key in self.grading_system:
            for key, value in kwargs.items():
                if key in self.grading_system[component_key]:
                    self.grading_system[component_key][key] = value
                    print(f"Updated {component} {key}: {value}")
                else:
                    print(f"Warning: {key} not found in {component_key}")
        else:
            print(f"Warning: {component_key} not found")

    def print_current_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("CURRENT SCANNER CONFIGURATION")
        print("="*60)
        
        print("\nSCAN THRESHOLDS (Minimum to qualify):")
        for key, value in self.scan_thresholds.items():
            print(f"  {key}: {value}")
        
        print(f"\nGRADING WEIGHTS (Total: {sum(self.grading_system['weights'].values())}):")
        for key, value in self.grading_system['weights'].items():
            print(f"  {key}: {value}")
        
        print("\nGRADE BREAKPOINTS:")
        for key, value in self.grading_system['grade_breakpoints'].items():
            print(f"  {key}: {value}+ points")

    def fetch_daily_data(self, symbol, start_date, end_date):
        """Fetch daily OHLCV data from Polygon"""
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
                    df = df.rename(columns={
                        'o': 'open', 'h': 'high', 'l': 'low', 
                        'c': 'close', 'v': 'volume'
                    })
                    return df[['date', 'open', 'high', 'low', 'close', 'volume']]
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_lingua_indicators(self, df):
        """Calculate all Lingua framework indicators"""
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # Previous day values
        df['pdc'] = df['close'].shift(1)
        
        # ATR Calculation
        df['high_low'] = df['high'] - df['low']
        df['high_pdc'] = abs(df['high'] - df['pdc'])
        df['low_pdc'] = abs(df['low'] - df['pdc'])
        df['true_range'] = df[['high_low', 'high_pdc', 'low_pdc']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        # EMAs
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_89'] = df['close'].ewm(span=89).mean()
        
        # Gap Analysis
        df['gap_dollars'] = df['open'] - df['pdc']
        df['gap_atr'] = df['gap_dollars'] / df['atr']
        
        # Range Analysis
        df['range_dollars'] = df['high'] - df['low']
        df['close_range'] = (df['close'] - df['low']) / df['range_dollars']
        
        # Price Change Metrics
        df['price_change'] = df['close'] - df['pdc']
        df['price_change_atr'] = df['price_change'] / df['atr']
        
        # Extension Metrics
        df['open_to_prev_low'] = df['open'] - df['low'].shift(1)
        df['extension_atr'] = df['open_to_prev_low'] / df['atr']
        
        # Volume Analysis
        df['avg_volume_20'] = df['volume'].rolling(20).mean()
        df['volume_multiple'] = df['volume'] / df['avg_volume_20']
        
        # Rolling Highs for Structure
        for period in [50, 100, 250]:
            df[f'highest_high_{period}'] = df['high'].rolling(period).max()
        
        return df

    def identify_trend_start(self, df):
        """Identify trend start using 9/20 EMA cross"""
        df = df.copy()
        df['ema_9_above_20'] = (df['ema_9'] > df['ema_20']).astype(int)
        df['cross_signal'] = df['ema_9_above_20'].diff()
        
        # Find last bullish cross
        cross_dates = df[df['cross_signal'] == 1]
        if not cross_dates.empty:
            last_cross = cross_dates.iloc[-1]
            return last_cross['date'], last_cross['close']
        return None, None

    def identify_euphoric_high(self, df, trend_start_date):
        """Identify euphoric top"""
        if trend_start_date is None:
            return None, None
            
        trend_data = df[df['date'] >= trend_start_date].copy()
        if trend_data.empty:
            return None, None
            
        max_high_idx = trend_data['high'].idxmax()
        max_high_row = trend_data.loc[max_high_idx]
        return max_high_row['date'], max_high_row['high']

    def calculate_trend_slope(self, start_price, end_price, duration_days):
        """Calculate trend slope"""
        if duration_days == 0 or start_price == 0:
            return 0
        total_change_pct = (end_price - start_price) / start_price
        return (total_change_pct / duration_days) * 100

    def calculate_downtrend_slope(self, df, high_date, analysis_date):
        """Calculate downtrend slope"""
        high_row = df[df['date'] == high_date]
        analysis_row = df[df['date'] == analysis_date]
        
        if high_row.empty or analysis_row.empty:
            return 0
            
        high_price = high_row['high'].iloc[0]
        analysis_price = analysis_row['close'].iloc[0]
        days_diff = (analysis_date - high_date).days
        
        if days_diff == 0:
            return 0
        
        return (analysis_price - high_price) / days_diff

    def score_component(self, component, value):
        """Score a component based on its value and thresholds"""
        scoring_config = self.grading_system.get(f"{component}_scoring", {})
        weight = self.grading_system['weights'].get(f"{component}_weight", 0)
        
        if component == 'downtrend_slope':
            # For slope, more negative = better score
            if value <= scoring_config.get('A+_max', -999):
                return weight
            elif value <= scoring_config.get('A_max', -999):
                return int(weight * 0.85)
            elif value <= scoring_config.get('B+_max', -999):
                return int(weight * 0.70)
            elif value <= scoring_config.get('B_max', -999):
                return int(weight * 0.50)
            else:
                return 0
        else:
            # For other components, higher = better score
            if value >= scoring_config.get('A+_min', 999):
                return weight
            elif value >= scoring_config.get('A_min', 999):
                return int(weight * 0.85)
            elif value >= scoring_config.get('B+_min', 999):
                return int(weight * 0.70)
            elif value >= scoring_config.get('B_min', 999):
                return int(weight * 0.50)
            else:
                return 0

    def calculate_setup_score(self, setup_data):
        """Calculate complete setup score"""
        scores = {}
        
        # Component scoring
        scores['trend'] = self.score_component('trend', setup_data.get('trend_atr_multiples', 0))
        scores['gap'] = self.score_component('gap', setup_data.get('gap_atr', 0))
        scores['extension'] = self.score_component('extension', setup_data.get('extension_atr', 0))
        scores['range'] = self.score_component('range', setup_data.get('range_close_pct', 0))
        scores['volume'] = self.score_component('volume', setup_data.get('volume_multiple', 0))
        scores['change'] = self.score_component('change', setup_data.get('change_atr', 0))
        scores['slope'] = self.score_component('downtrend_slope', setup_data.get('downtrend_slope', 0))
        
        total_score = sum(scores.values())
        
        # Determine grade
        grade_breakpoints = self.grading_system['grade_breakpoints']
        if total_score >= grade_breakpoints['A+']:
            grade = 'A+'
        elif total_score >= grade_breakpoints['A']:
            grade = 'A'
        elif total_score >= grade_breakpoints['B+']:
            grade = 'B+'
        elif total_score >= grade_breakpoints['B']:
            grade = 'B'
        elif total_score >= grade_breakpoints['C+']:
            grade = 'C+'
        else:
            grade = 'C'
        
        return {
            'total_score': total_score,
            'component_scores': scores,
            'grade': grade
        }

    def passes_scan_criteria(self, setup_data):
        """Check if setup passes minimum scan criteria"""
        criteria = [
            setup_data.get('trend_atr_multiples', 0) >= self.scan_thresholds['min_trend_atr'],
            setup_data.get('gap_atr', 0) >= self.scan_thresholds['min_gap_atr'],
            setup_data.get('extension_atr', 0) >= self.scan_thresholds['min_extension_atr'],
            setup_data.get('range_close_pct', 0) >= self.scan_thresholds['min_range_close_pct'],
            setup_data.get('volume_multiple', 0) >= self.scan_thresholds['min_volume_multiple'],
            setup_data.get('change_atr', 0) >= self.scan_thresholds['min_change_atr'],
            setup_data.get('downtrend_slope', 0) <= self.scan_thresholds['max_downtrend_slope'],
            setup_data.get('ema_extension_pct', 0) >= self.scan_thresholds['min_ema_extension_pct'],
            setup_data.get('fade_atr', 0) >= self.scan_thresholds['min_fade_atr'],
            self.scan_thresholds['min_days_since_high'] <= setup_data.get('days_since_high', 0) <= self.scan_thresholds['max_days_since_high']
        ]
        return all(criteria)

    def scan_symbol(self, symbol, scan_date, lookback_days=180):
        """Scan a single symbol for the setup"""
        scan_date = pd.to_datetime(scan_date).date()
        start_date = scan_date - timedelta(days=lookback_days)
        end_date = scan_date + timedelta(days=10)
        
        # Fetch data
        df = self.fetch_daily_data(symbol, start_date.strftime('%Y-%m-%d'), 
                                  end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            return {'symbol': symbol, 'error': 'No data available'}
        
        # Calculate indicators
        df = self.calculate_lingua_indicators(df)
        
        # Find setup day
        setup_row = df[df['date'] == scan_date]
        if setup_row.empty:
            return {'symbol': symbol, 'error': f'No data for {scan_date}'}
        
        setup_idx = setup_row.index[0]
        prev_day_idx = setup_idx - 1
        
        if prev_day_idx < 0:
            return {'symbol': symbol, 'error': 'Insufficient historical data'}
        
        d_minus_1 = df.iloc[prev_day_idx]
        d_0 = df.iloc[setup_idx]
        
        # Trend analysis
        trend_start_date, trend_start_price = self.identify_trend_start(df[:setup_idx])
        euphoric_date, euphoric_high = self.identify_euphoric_high(df, trend_start_date)
        
        # Calculate metrics
        if trend_start_date and euphoric_high and trend_start_price:
            trend_duration = (euphoric_date - trend_start_date).days if euphoric_date else 0
            trend_atr_multiples = (euphoric_high - trend_start_price) / d_minus_1['atr'] if d_minus_1['atr'] > 0 else 0
            trend_slope = self.calculate_trend_slope(trend_start_price, euphoric_high, trend_duration)
        else:
            trend_atr_multiples = 0
            trend_slope = 0
            trend_duration = 0
        
        # Fade analysis
        if euphoric_date and euphoric_high:
            days_since_high = (d_minus_1['date'] - euphoric_date).days
            fade_atr = (euphoric_high - d_minus_1['close']) / d_minus_1['atr'] if d_minus_1['atr'] > 0 else 0
            downtrend_slope = self.calculate_downtrend_slope(df, euphoric_date, d_minus_1['date'])
        else:
            days_since_high = 0
            fade_atr = 0
            downtrend_slope = 0
        
        # Compile setup data
        setup_data = {
            'trend_atr_multiples': trend_atr_multiples,
            'trend_slope': trend_slope,
            'gap_atr': d_0['gap_atr'],
            'extension_atr': d_0['extension_atr'],
            'range_close_pct': d_minus_1['close_range'] * 100,
            'volume_multiple': d_minus_1['volume_multiple'],
            'change_atr': d_minus_1['price_change_atr'],
            'downtrend_slope': downtrend_slope,
            'ema_extension_pct': ((d_0['open'] - d_0['ema_89']) / d_0['ema_89']) * 100 if d_0['ema_89'] > 0 else 0,
            'fade_atr': fade_atr,
            'days_since_high': days_since_high
        }
        
        # Check scan criteria
        passes_scan = self.passes_scan_criteria(setup_data)
        
        # Calculate score
        scoring = self.calculate_setup_score(setup_data)
        
        # Performance tracking (if future data available)
        performance = {}
        if setup_idx + 5 < len(df):
            d0_open = d_0['open']
            d0_close = d_0['close']
            d5_close = df.iloc[min(setup_idx + 5, len(df) - 1)]['close']
            
            performance = {
                'intraday_fade_pct': ((d0_close - d0_open) / d0_open) * 100,
                'swing_fade_5d_pct': ((d5_close - d0_open) / d0_open) * 100
            }
        
        return {
            'symbol': symbol,
            'scan_date': scan_date.strftime('%Y-%m-%d'),
            'passes_scan': passes_scan,
            'total_score': scoring['total_score'],
            'grade': scoring['grade'],
            'component_scores': scoring['component_scores'],
            'setup_metrics': setup_data,
            'performance': performance,
            'trend_context': {
                'trend_start': trend_start_date.strftime('%Y-%m-%d') if trend_start_date else None,
                'euphoric_high': euphoric_date.strftime('%Y-%m-%d') if euphoric_date else None,
                'days_since_high': days_since_high
            }
        }

    def scan_test_universe(self, scan_date):
        """Scan the test universe: HOOD, MSTR, SMCI, IBIT"""
        test_symbols = ['HOOD', 'MSTR', 'SMCI', 'IBIT']
        results = []
        
        print(f"\nScanning test universe for {scan_date}")
        print("-" * 50)
        
        for symbol in test_symbols:
            print(f"Scanning {symbol}...")
            result = self.scan_symbol(symbol, scan_date)
            results.append(result)
        
        return results

    def print_scan_results(self, results):
        """Print formatted scan results"""
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            print("No valid scan results")
            return
        
        print(f"\n{'='*80}")
        print("SCAN RESULTS SUMMARY")
        print(f"{'='*80}")
        
        # Summary table
        print(f"{'Symbol':<6} {'Pass':<5} {'Score':<5} {'Grade':<5} {'Trend':<6} {'Gap':<5} {'Ext':<5} {'Range':<6} {'Vol':<4} {'Fade':<5}")
        print("-" * 80)
        
        for result in valid_results:
            symbol = result['symbol']
            passes = "✓" if result['passes_scan'] else "✗"
            score = result['total_score']
            grade = result['grade']
            trend = f"{result['setup_metrics']['trend_atr_multiples']:.1f}"
            gap = f"{result['setup_metrics']['gap_atr']:.2f}"
            ext = f"{result['setup_metrics']['extension_atr']:.2f}"
            range_close = f"{result['setup_metrics']['range_close_pct']:.0f}%"
            vol = f"{result['setup_metrics']['volume_multiple']:.1f}"
            fade = f"{result['setup_metrics']['fade_atr']:.1f}" if result['setup_metrics']['fade_atr'] > 0 else "N/A"
            
            print(f"{symbol:<6} {passes:<5} {score:<5} {grade:<5} {trend:<6} {gap:<5} {ext:<5} {range_close:<6} {vol:<4} {fade:<5}")
        
        # Detailed breakdown
        print(f"\n{'='*80}")
        print("DETAILED BREAKDOWN")
        print(f"{'='*80}")
        
        for result in valid_results:
            if 'error' in result:
                continue
                
            print(f"\n{result['symbol']} - {result['scan_date']}")
            print(f"Score: {result['total_score']}/100 (Grade: {result['grade']})")
            print(f"Passes Scan: {result['passes_scan']}")
            
            print("Component Scores:")
            for comp, score in result['component_scores'].items():
                weight = self.grading_system['weights'].get(f"{comp}_weight", 0)
                print(f"  {comp.capitalize()}: {score}/{weight}")
            
            print("Key Metrics:")
            metrics = result['setup_metrics']
            print(f"  Trend: {metrics['trend_atr_multiples']:.2f} ATR, Slope: {metrics['trend_slope']:.2f}%/day")
            print(f"  Gap: {metrics['gap_atr']:.2f} ATR, Extension: {metrics['extension_atr']:.2f} ATR")
            print(f"  Range Close: {metrics['range_close_pct']:.1f}%, Volume: {metrics['volume_multiple']:.1f}x")
            print(f"  Fade: {metrics['fade_atr']:.2f} ATR in {metrics['days_since_high']} days")
            
            if result['performance']:
                perf = result['performance']
                print(f"Performance: {perf['intraday_fade_pct']:.1f}% (intraday), {perf['swing_fade_5d_pct']:.1f}% (5-day)")

def interactive_configuration():
    """Interactive configuration helper"""
    scanner = ConfigurableBacksideScanner()
    
    print("="*60)
    print("CONFIGURABLE BACKSIDE POP SCANNER")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. View current configuration")
        print("2. Update scan thresholds") 
        print("3. Update grading weights")
        print("4. Update grade breakpoints")
        print("5. Update component scoring")
        print("6. Run scan on test universe")
        print("7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            scanner.print_current_config()
            
        elif choice == '2':
            print("\nCurrent scan thresholds:")
            for key, value in scanner.scan_thresholds.items():
                print(f"  {key}: {value}")
            
            param = input("\nEnter parameter to update (or 'back'): ").strip()
            if param == 'back':
                continue
            if param in scanner.scan_thresholds:
                try:
                    new_value = float(input(f"Enter new value for {param}: "))
                    scanner.update_scan_thresholds(**{param: new_value})
                except ValueError:
                    print("Invalid value")
            else:
                print("Parameter not found")
                
        elif choice == '3':
            print("\nCurrent weights:")
            for key, value in scanner.grading_system['weights'].items():
                print(f"  {key}: {value}")
            
            param = input("\nEnter weight to update (or 'back'): ").strip()
            if param == 'back':
                continue
            if param in scanner.grading_system['weights']:
                try:
                    new_value = float(input(f"Enter new value for {param}: "))
                    scanner.update_grading_weights(**{param: new_value})
                except ValueError:
                    print("Invalid value")
            else:
                print("Weight not found")
                
        elif choice == '4':
            print("\nCurrent grade breakpoints:")
            for key, value in scanner.grading_system['grade_breakpoints'].items():
                print(f"  {key}: {value}")
            
            param = input("\nEnter grade to update (or 'back'): ").strip()
            if param == 'back':
                continue
            if param in scanner.grading_system['grade_breakpoints']:
                try:
                    new_value = float(input(f"Enter new value for {param}: "))
                    scanner.update_grade_breakpoints(**{param: new_value})
                except ValueError:
                    print("Invalid value")
            else:
                print("Grade not found")
                
        elif choice == '5':
            components = ['trend', 'gap', 'extension', 'range', 'volume', 'change', 'downtrend_slope']
            print(f"\nAvailable components: {', '.join(components)}")
            
            comp = input("Enter component to update (or 'back'): ").strip()
            if comp == 'back':
                continue
            if comp in components:
                scoring_key = f"{comp}_scoring"
                if scoring_key in scanner.grading_system:
                    print(f"\nCurrent {comp} scoring:")
                    for key, value in scanner.grading_system[scoring_key].items():
                        print(f"  {key}: {value}")
                    
                    param = input(f"\nEnter {comp} threshold to update: ").strip()
                    if param in scanner.grading_system[scoring_key]:
                        try:
                            new_value = float(input(f"Enter new value for {param}: "))
                            scanner.update_component_scoring(comp, **{param: new_value})
                        except ValueError:
                            print("Invalid value")
                    else:
                        print("Threshold not found")
            else:
                print("Component not found")
                
        elif choice == '6':
            scan_date = input("Enter scan date (YYYY-MM-DD) or press Enter for 2025-03-03: ").strip()
            if not scan_date:
                scan_date = '2025-03-03'
            
            try:
                results = scanner.scan_test_universe(scan_date)
                scanner.print_scan_results(results)
            except Exception as e:
                print(f"Error running scan: {e}")
                
        elif choice == '7':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    interactive_configuration()