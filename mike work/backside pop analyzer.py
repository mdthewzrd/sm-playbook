"""
Backside Pop Strategy Analyzer
Complete SM Playbook Implementation with Grading System

Framework: Lingua Trading Language
Strategy Type: Short failed breakout strategy  
Context: Extended names attempting to reclaim previous highs in backside territory
Target Win Rate: 65%+
Risk/Reward: 1:2.2 average

Based on A+ Examples: HOOD 3/3/25, MSTR 12/5/24, SMCI 3/26/24, IBIT 3/3/25
"""

import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
import pandas_market_calendars as mcal
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings("ignore")

# SM Playbook Infrastructure Integration
API_KEY = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"  # Polygon API Key
BASE_URL = "https://api.polygon.io"
nyse = mcal.get_calendar('NYSE')

class BacksidePopAnalyzer:
    """
    Complete Backside Pop Strategy Analyzer
    Implements Lingua Framework with SM Playbook grading methodology
    """
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        
        # A+ Example Benchmarks (from HOOD 3/3/25 analysis)
        self.a_plus_benchmarks = {
            'trend_atr_multiples': 15.4,
            'trend_slope_normalized': 2.8,  # % per day
            'gap_atr': 0.86,
            'extension_atr': 1.73, 
            'range_close_pct': 0.672,
            'volume_multiple': 1.4,
            'change_atr': 0.6,
            'downtrend_slope': -6.2,
            'fade_from_high_atr': 7.4,
            'ema_extension_pct': 35.6
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
        
        # Scan Thresholds (Inclusive - 50-60% of A+ values)
        self.scan_thresholds = {
            'min_trend_atr': 5,           # Scan anything 5+ ATR trend
            'min_gap_atr': 0.4,           # Scan anything 0.4+ ATR gap
            'min_extension_atr': 0.75,    # Scan anything 0.75+ ATR extension  
            'min_range_close': 0.50,      # Scan anything 50%+ range close
            'min_volume_multiple': 0.75,  # Scan anything 0.75x+ volume
            'min_change_atr': 0.4,        # Scan anything 0.4+ ATR change
            'max_downtrend_slope': -1,    # Scan anything steeper than -1
            'min_ema_extension': 10       # Scan anything 10%+ above EMA89
        }
        
        # Grade Thresholds
        self.grade_thresholds = {
            'a_plus_min': 90,
            'a_min': 80,
            'b_plus_min': 70,
            'b_min': 60,
            'c_plus_min': 50
        }

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
        df['pdc'] = df['close'].shift(1)  # Previous day close
        
        # ATR Calculation (14-period)
        df['high_low'] = df['high'] - df['low']
        df['high_pdc'] = abs(df['high'] - df['pdc'])
        df['low_pdc'] = abs(df['low'] - df['pdc'])
        df['true_range'] = df[['high_low', 'high_pdc', 'low_pdc']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        # EMA Calculations (Lingua Framework)
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_89'] = df['close'].ewm(span=89).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
        # EMA Cloud System (Means)
        df['ema_cloud_9_20_top'] = np.maximum(df['ema_9'], df['ema_20'])
        df['ema_cloud_9_20_bottom'] = np.minimum(df['ema_9'], df['ema_20'])
        df['ema_cloud_bullish'] = (df['ema_9'] > df['ema_20']).astype(int)
        
        # Deviation Bands (Extremes)
        df['ema_cloud_mid'] = (df['ema_9'] + df['ema_20']) / 2
        df['std_dev'] = df['close'].rolling(20).std()
        df['dev_band_upper'] = df['ema_cloud_mid'] + (df['std_dev'] * 2.0)
        df['dev_band_lower'] = df['ema_cloud_mid'] - (df['std_dev'] * 2.0)
        
        # Gap Analysis
        df['gap_dollars'] = df['open'] - df['pdc']
        df['gap_atr'] = df['gap_dollars'] / df['atr']
        df['gap_percent'] = (df['open'] / df['pdc'] - 1) * 100
        
        # Range Analysis  
        df['range_dollars'] = df['high'] - df['low']
        df['range_atr'] = df['range_dollars'] / df['atr']
        df['close_range'] = (df['close'] - df['low']) / df['range_dollars']
        
        # Price Change Metrics
        df['price_change'] = df['close'] - df['pdc']
        df['price_change_atr'] = df['price_change'] / df['atr']
        df['price_change_percent'] = (df['close'] / df['pdc'] - 1) * 100
        
        # Extension Metrics
        df['high_to_open'] = df['high'] - df['open']
        df['high_to_open_atr'] = df['high_to_open'] / df['atr']
        df['open_to_prev_low'] = df['open'] - df['low'].shift(1)
        df['extension_atr'] = df['open_to_prev_low'] / df['atr']
        
        # Volume Analysis
        df['avg_volume_20'] = df['volume'].rolling(20).mean()
        df['volume_multiple'] = df['volume'] / df['avg_volume_20']
        df['dollar_volume'] = df['close'] * df['volume']
        
        # Rolling Highs/Lows for Structure
        for period in [5, 20, 50, 100, 250]:
            df[f'highest_high_{period}'] = df['high'].rolling(period).max()
            df[f'lowest_low_{period}'] = df['low'].rolling(period).min()
        
        # Distance from EMAs (normalized by ATR)
        for ema_period in [9, 20, 50, 89, 200]:
            df[f'dist_from_ema_{ema_period}'] = df['high'] - df[f'ema_{ema_period}']
            df[f'dist_from_ema_{ema_period}_atr'] = df[f'dist_from_ema_{ema_period}'] / df['atr']
        
        return df

    def identify_trend_start(self, df):
        """Identify trend start using 9/20 EMA cross confirmation"""
        df = df.copy()
        
        # EMA Cross Detection
        df['ema_9_above_20'] = (df['ema_9'] > df['ema_20']).astype(int)
        df['cross_signal'] = df['ema_9_above_20'].diff()
        
        # Find last bullish cross with 3+ day confirmation
        cross_dates = df[df['cross_signal'] == 1].copy()
        
        for idx, cross in cross_dates.iterrows():
            # Check next 3 days for confirmation
            confirmation_period = df[(df['date'] >= cross['date']) & 
                                   (df['date'] <= cross['date'] + pd.Timedelta(days=5))]
            
            if len(confirmation_period) >= 3:
                confirmed_days = (confirmation_period['close'] > confirmation_period['open']).sum()
                if confirmed_days >= 2:  # At least 2 of 3 days green
                    return cross['date'], cross['close']
        
        return None, None

    def identify_euphoric_high(self, df, trend_start_date):
        """Identify euphoric top - highest high since trend start"""
        if trend_start_date is None:
            return None, None, None
            
        trend_data = df[df['date'] >= trend_start_date].copy()
        if trend_data.empty:
            return None, None, None
            
        # Find highest high during trend
        max_high_idx = trend_data['high'].idxmax()
        max_high_row = trend_data.loc[max_high_idx]
        
        return max_high_row['date'], max_high_row['high'], max_high_idx

    def calculate_trend_slope(self, trend_start_price, trend_end_price, trend_duration_days):
        """Calculate normalized trend slope (% per day)"""
        if trend_duration_days == 0 or trend_start_price == 0:
            return 0
            
        total_change_pct = (trend_end_price - trend_start_price) / trend_start_price
        slope_per_day = (total_change_pct / trend_duration_days) * 100
        return slope_per_day

    def calculate_downtrend_slope(self, df, high_date, analysis_date):
        """Calculate slope from euphoric high to analysis date"""
        high_row = df[df['date'] == high_date]
        analysis_row = df[df['date'] == analysis_date]
        
        if high_row.empty or analysis_row.empty:
            return 0
            
        high_price = high_row['high'].iloc[0]
        analysis_price = analysis_row['close'].iloc[0]
        days_diff = (analysis_date - high_date).days
        
        if days_diff == 0:
            return 0
            
        price_change = analysis_price - high_price
        slope = price_change / days_diff
        return slope

    def score_trend_quality(self, trend_atr, trend_slope):
        """Score trend size and slope (25 points max)"""
        # Trend size scoring (20 points)
        if trend_atr >= 15:     size_score = 20    # A+ (HOOD level)
        elif trend_atr >= 10:   size_score = 17    # A
        elif trend_atr >= 8:    size_score = 15    # B+  
        elif trend_atr >= 5:    size_score = 12    # B
        else:                   size_score = 0
        
        # Slope scoring (5 points) - steeper trends score higher
        if trend_slope >= 3.0:     slope_score = 5    # A+ (MSTR-like parabolic)
        elif trend_slope >= 2.0:   slope_score = 4    # A
        elif trend_slope >= 1.5:   slope_score = 3    # B+
        elif trend_slope >= 1.0:   slope_score = 2    # B
        else:                      slope_score = 1
        
        return size_score + slope_score

    def score_gap_behavior(self, gap_atr):
        """Score gap behavior (15 points max)"""
        if gap_atr >= 0.7:      return 15    # A+ 
        elif gap_atr >= 0.5:    return 13    # A (solid gap)
        elif gap_atr >= 0.4:    return 10    # B+ (scan threshold)
        elif gap_atr >= 0.3:    return 7     # B
        else:                   return 0

    def score_extension(self, extension_atr):
        """Score extension analysis (20 points max)"""
        if extension_atr >= 1.5:    return 20    # A+ (HOOD level)
        elif extension_atr >= 1.0:  return 17    # A
        elif extension_atr >= 0.75: return 14    # B+ (scan threshold)
        elif extension_atr >= 0.5:  return 10    # B
        else:                       return 0

    def score_range_close(self, range_close_pct):
        """Score range close position (15 points max)"""
        if range_close_pct >= 0.80:    return 15    # A+ (strong close)
        elif range_close_pct >= 0.65:  return 13    # A (HOOD level)
        elif range_close_pct >= 0.50:  return 10    # B+ (scan threshold)
        elif range_close_pct >= 0.40:  return 7     # B
        else:                          return 0

    def score_volume_behavior(self, volume_multiple):
        """Score volume multiple (10 points max)"""
        if volume_multiple >= 1.3:     return 10    # A+ (high volume)
        elif volume_multiple >= 1.0:   return 8     # A
        elif volume_multiple >= 0.75:  return 6     # B+ (scan threshold)
        elif volume_multiple >= 0.5:   return 4     # B
        else:                          return 0

    def score_price_change(self, change_atr):
        """Score D-1 price change (10 points max)"""
        if change_atr >= 1.0:       return 10    # A+ (big move)
        elif change_atr >= 0.5:     return 8     # A
        elif change_atr >= 0.4:     return 6     # B+ (scan threshold)
        elif change_atr >= 0.3:     return 4     # B
        else:                       return 0

    def score_downtrend_slope(self, slope_value):
        """Score downtrend slope (5 points max)"""
        if slope_value <= -5:       return 5    # A+ (HOOD level steep decline)
        elif slope_value <= -3:     return 4    # A (solid downtrend)
        elif slope_value <= -1:     return 3    # B+ (scan threshold)
        elif slope_value <= 0:      return 2    # B (some decline)
        else:                       return 0    # Not declining

    def calculate_setup_score(self, setup_data):
        """Calculate complete setup score (0-100 points)"""
        scores = {}
        
        # Component scoring
        scores['trend_score'] = self.score_trend_quality(
            setup_data.get('trend_atr_multiples', 0),
            setup_data.get('trend_slope', 0)
        )
        
        scores['gap_score'] = self.score_gap_behavior(
            setup_data.get('gap_atr', 0)
        )
        
        scores['extension_score'] = self.score_extension(
            setup_data.get('extension_atr', 0)
        )
        
        scores['range_score'] = self.score_range_close(
            setup_data.get('range_close_pct', 0)
        )
        
        scores['volume_score'] = self.score_volume_behavior(
            setup_data.get('volume_multiple', 0)
        )
        
        scores['change_score'] = self.score_price_change(
            setup_data.get('change_atr', 0)
        )
        
        scores['slope_score'] = self.score_downtrend_slope(
            setup_data.get('downtrend_slope', 0)
        )
        
        # Total score
        total_score = sum(scores.values())
        
        # Generate grade
        if total_score >= self.grade_thresholds['a_plus_min']:
            grade = 'A+'
            tier = 'PRIME'
        elif total_score >= self.grade_thresholds['a_min']:
            grade = 'A'
            tier = 'PRIME'
        elif total_score >= self.grade_thresholds['b_plus_min']:
            grade = 'B+'
            tier = 'GOOD'
        elif total_score >= self.grade_thresholds['b_min']:
            grade = 'B'
            tier = 'GOOD'
        elif total_score >= self.grade_thresholds['c_plus_min']:
            grade = 'C+'
            tier = 'WATCHLIST'
        else:
            grade = 'C'
            tier = 'WATCHLIST'
        
        return {
            'total_score': total_score,
            'component_scores': scores,
            'grade': grade,
            'tier': tier
        }

    def analyze_setup(self, symbol, setup_date, lookback_days=180):
        """Complete setup analysis for a given symbol and date"""
        
        # Calculate date range for analysis
        setup_date = pd.to_datetime(setup_date).date()
        start_date = setup_date - timedelta(days=lookback_days)
        end_date = setup_date + timedelta(days=10)  # Include some forward data for performance
        
        # Fetch data
        df = self.fetch_daily_data(symbol, start_date.strftime('%Y-%m-%d'), 
                                  end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            return {'error': f'No data available for {symbol}'}
        
        # Calculate indicators
        df = self.calculate_lingua_indicators(df)
        
        # Find setup day data
        setup_row = df[df['date'] == setup_date]
        if setup_row.empty:
            return {'error': f'No data for {symbol} on {setup_date}'}
        
        setup_idx = setup_row.index[0]
        prev_day_idx = setup_idx - 1
        
        if prev_day_idx < 0:
            return {'error': f'Insufficient historical data for {symbol}'}
        
        # Get D-1 and D0 data
        d_minus_1 = df.iloc[prev_day_idx]
        d_0 = df.iloc[setup_idx]
        
        # Trend analysis
        trend_start_date, trend_start_price = self.identify_trend_start(df[:setup_idx])
        euphoric_date, euphoric_high, euphoric_idx = self.identify_euphoric_high(df, trend_start_date)
        
        # Calculate trend metrics
        if trend_start_date and euphoric_high and trend_start_price:
            trend_duration = (euphoric_date - trend_start_date).days if euphoric_date else 0
            trend_atr_multiples = (euphoric_high - trend_start_price) / d_minus_1['atr'] if d_minus_1['atr'] > 0 else 0
            trend_slope = self.calculate_trend_slope(trend_start_price, euphoric_high, trend_duration)
        else:
            trend_atr_multiples = 0
            trend_slope = 0
            trend_duration = 0
        
        # Calculate downtrend slope (from high to D-1)
        if euphoric_date:
            downtrend_slope = self.calculate_downtrend_slope(df, euphoric_date, d_minus_1['date'])
        else:
            downtrend_slope = 0
        
        # Compile setup data
        setup_data = {
            'trend_atr_multiples': trend_atr_multiples,
            'trend_slope': trend_slope,
            'gap_atr': d_0['gap_atr'],
            'extension_atr': d_0['extension_atr'],
            'range_close_pct': d_minus_1['close_range'],
            'volume_multiple': d_minus_1['volume_multiple'],
            'change_atr': d_minus_1['price_change_atr'],
            'downtrend_slope': downtrend_slope,
            'ema_extension_pct': ((d_0['open'] - d_0['ema_89']) / d_0['ema_89']) * 100 if d_0['ema_89'] > 0 else 0
        }
        
        # Calculate scoring
        scoring_results = self.calculate_setup_score(setup_data)
        
        # Calculate performance (if future data available)
        performance_data = {}
        if setup_idx + 5 < len(df):  # Ensure we have enough forward data
            d0_open = d_0['open']
            d0_close = d_0['close']
            d3_close = df.iloc[min(setup_idx + 3, len(df) - 1)]['close']
            d5_close = df.iloc[min(setup_idx + 5, len(df) - 1)]['close']
            
            performance_data = {
                'intraday_fade_pct': ((d0_close - d0_open) / d0_open) * 100,
                'swing_fade_3d_pct': ((d3_close - d0_open) / d0_open) * 100,
                'swing_fade_5d_pct': ((d5_close - d0_open) / d0_open) * 100
            }
        
        # Compile final results
        result = {
            'symbol': symbol,
            'setup_date': setup_date.strftime('%Y-%m-%d'),
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # Grading Results
            'total_score': scoring_results['total_score'],
            'grade': scoring_results['grade'],
            'tier': scoring_results['tier'],
            'component_scores': scoring_results['component_scores'],
            
            # Raw Metrics
            'setup_metrics': setup_data,
            
            # Context Information
            'trend_context': {
                'trend_start_date': trend_start_date.strftime('%Y-%m-%d') if trend_start_date else None,
                'euphoric_high_date': euphoric_date.strftime('%Y-%m-%d') if euphoric_date else None,
                'trend_duration_days': trend_duration,
                'euphoric_high_price': float(euphoric_high) if euphoric_high else None,
                'backside_context': euphoric_high and d_0['high'] < euphoric_high
            },
            
            # Market Structure
            'market_structure': {
                'above_ema_89': d_0['open'] > d_0['ema_89'],
                'ema_cloud_bullish': d_0['ema_cloud_bullish'] == 1,
                'distance_from_highs_atr': (d_0['open'] - d_0['highest_high_250']) / d_0['atr'] if d_0['atr'] > 0 else 0
            },
            
            # Performance Tracking (if available)
            'performance': performance_data,
            
            # Qualification Status
            'qualification': {
                'meets_scan_criteria': all([
                    setup_data['trend_atr_multiples'] >= self.scan_thresholds['min_trend_atr'],
                    setup_data['gap_atr'] >= self.scan_thresholds['min_gap_atr'],
                    setup_data['extension_atr'] >= self.scan_thresholds['min_extension_atr'],
                    setup_data['range_close_pct'] >= self.scan_thresholds['min_range_close'],
                    setup_data['volume_multiple'] >= self.scan_thresholds['min_volume_multiple'],
                    setup_data['change_atr'] >= self.scan_thresholds['min_change_atr'],
                    setup_data['downtrend_slope'] <= self.scan_thresholds['max_downtrend_slope'],
                    setup_data['ema_extension_pct'] >= self.scan_thresholds['min_ema_extension']
                ]),
                'qualification_stage': 'D-1_CLOSE',
                'confirmation_stage': 'D0_OPEN'
            }
        }
        
        return result

    def analyze_a_plus_examples(self):
        """Analyze all A+ examples to validate grading system"""
        a_plus_examples = [
            {'symbol': 'HOOD', 'date': '2025-03-03'},
            {'symbol': 'MSTR', 'date': '2024-12-05'}, 
            {'symbol': 'SMCI', 'date': '2024-03-26'},
            {'symbol': 'IBIT', 'date': '2025-03-03'}
        ]
        
        results = []
        for example in a_plus_examples:
            print(f"Analyzing A+ example: {example['symbol']} on {example['date']}")
            result = self.analyze_setup(example['symbol'], example['date'])
            results.append(result)
            
        return results

    def batch_analyze(self, symbol_date_list, max_workers=5):
        """Analyze multiple setups concurrently"""
        def analyze_single(params):
            symbol, date = params
            return self.analyze_setup(symbol, date)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(analyze_single, symbol_date_list))
        
        return results

    def extract_raw_parameters(self, symbol, setup_date, lookback_days=180):
        """Extract raw parameter values without any scoring - just the mold data"""
        
        # Calculate date range for analysis
        setup_date = pd.to_datetime(setup_date).date()
        start_date = setup_date - timedelta(days=lookback_days)
        end_date = setup_date + timedelta(days=10)  # Include some forward data for performance
        
        # Fetch data
        df = self.fetch_daily_data(symbol, start_date.strftime('%Y-%m-%d'), 
                                  end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            return {'error': f'No data available for {symbol}'}
        
        # Calculate indicators
        df = self.calculate_lingua_indicators(df)
        
        # Find setup day data
        setup_row = df[df['date'] == setup_date]
        if setup_row.empty:
            return {'error': f'No data for {symbol} on {setup_date}'}
        
        setup_idx = setup_row.index[0]
        prev_day_idx = setup_idx - 1
        
        if prev_day_idx < 0:
            return {'error': f'Insufficient historical data for {symbol}'}
        
        # Get D-1 and D0 data
        d_minus_1 = df.iloc[prev_day_idx]
        d_0 = df.iloc[setup_idx]
        
        # Trend analysis
        trend_start_date, trend_start_price = self.identify_trend_start(df[:setup_idx])
        euphoric_date, euphoric_high, euphoric_idx = self.identify_euphoric_high(df, trend_start_date)
        
        # Calculate trend metrics
        if trend_start_date and euphoric_high and trend_start_price:
            trend_duration = (euphoric_date - trend_start_date).days if euphoric_date else 0
            trend_atr_multiples = (euphoric_high - trend_start_price) / d_minus_1['atr'] if d_minus_1['atr'] > 0 else 0
            trend_slope = self.calculate_trend_slope(trend_start_price, euphoric_high, trend_duration)
            trend_total_move_pct = ((euphoric_high - trend_start_price) / trend_start_price) * 100 if trend_start_price > 0 else 0
        else:
            trend_atr_multiples = 0
            trend_slope = 0
            trend_duration = 0
            trend_total_move_pct = 0
        
        # Calculate downtrend slope (from high to D-1)
        if euphoric_date:
            downtrend_slope = self.calculate_downtrend_slope(df, euphoric_date, d_minus_1['date'])
            days_since_high = (d_minus_1['date'] - euphoric_date).days
            fade_from_high_dollars = euphoric_high - d_minus_1['close'] if euphoric_high else 0
            fade_from_high_atr = fade_from_high_dollars / d_minus_1['atr'] if d_minus_1['atr'] > 0 else 0
            fade_from_high_pct = (fade_from_high_dollars / euphoric_high) * 100 if euphoric_high > 0 else 0
        else:
            downtrend_slope = 0
            days_since_high = 0
            fade_from_high_dollars = 0
            fade_from_high_atr = 0
            fade_from_high_pct = 0
        
        # Calculate performance (if future data available)
        performance_data = {}
        if setup_idx + 5 < len(df):  # Ensure we have enough forward data
            d0_open = d_0['open']
            d0_close = d_0['close'] 
            d1_close = df.iloc[min(setup_idx + 1, len(df) - 1)]['close']
            d3_close = df.iloc[min(setup_idx + 3, len(df) - 1)]['close']
            d5_close = df.iloc[min(setup_idx + 5, len(df) - 1)]['close']
            
            performance_data = {
                'intraday_fade_pct': ((d0_close - d0_open) / d0_open) * 100,
                'next_day_fade_pct': ((d1_close - d0_open) / d0_open) * 100,
                'swing_fade_3d_pct': ((d3_close - d0_open) / d0_open) * 100,
                'swing_fade_5d_pct': ((d5_close - d0_open) / d0_open) * 100
            }
        
        # Raw parameter extraction - no scoring, just the values
        raw_params = {
            'symbol': symbol,
            'setup_date': setup_date.strftime('%Y-%m-%d'),
            
            # TREND PARAMETERS
            'trend_start_date': trend_start_date.strftime('%Y-%m-%d') if trend_start_date else None,
            'trend_start_price': float(trend_start_price) if trend_start_price else None,
            'trend_duration_days': trend_duration,
            'trend_atr_multiples': round(trend_atr_multiples, 2),
            'trend_total_move_pct': round(trend_total_move_pct, 1),
            'trend_slope_pct_per_day': round(trend_slope, 2),
            
            # EUPHORIC TOP PARAMETERS
            'euphoric_high_date': euphoric_date.strftime('%Y-%m-%d') if euphoric_date else None,
            'euphoric_high_price': float(euphoric_high) if euphoric_high else None,
            'days_since_high': days_since_high,
            'fade_from_high_dollars': round(fade_from_high_dollars, 2),
            'fade_from_high_atr': round(fade_from_high_atr, 2),
            'fade_from_high_pct': round(fade_from_high_pct, 1),
            
            # D-1 (DAY MINUS ONE) PARAMETERS
            'd_minus_1_open': round(d_minus_1['open'], 2),
            'd_minus_1_high': round(d_minus_1['high'], 2),
            'd_minus_1_low': round(d_minus_1['low'], 2),
            'd_minus_1_close': round(d_minus_1['close'], 2),
            'd_minus_1_volume': int(d_minus_1['volume']),
            'd_minus_1_range_dollars': round(d_minus_1['range_dollars'], 2),
            'd_minus_1_range_atr': round(d_minus_1['range_atr'], 2),
            'd_minus_1_close_range_pct': round(d_minus_1['close_range'] * 100, 1),
            'd_minus_1_price_change_dollars': round(d_minus_1['price_change'], 2),
            'd_minus_1_price_change_atr': round(d_minus_1['price_change_atr'], 2),
            'd_minus_1_price_change_pct': round(d_minus_1['price_change_percent'], 1),
            'd_minus_1_volume_multiple': round(d_minus_1['volume_multiple'], 2),
            'd_minus_1_dollar_volume': int(d_minus_1['dollar_volume']),
            'd_minus_1_atr': round(d_minus_1['atr'], 2),
            
            # D0 (SETUP DAY) PARAMETERS
            'd0_open': round(d_0['open'], 2),
            'd0_high': round(d_0['high'], 2),
            'd0_low': round(d_0['low'], 2),
            'd0_close': round(d_0['close'], 2),
            'd0_gap_dollars': round(d_0['gap_dollars'], 2),
            'd0_gap_atr': round(d_0['gap_atr'], 2),
            'd0_gap_pct': round(d_0['gap_percent'], 1),
            'd0_extension_dollars': round(d_0['open_to_prev_low'], 2),
            'd0_extension_atr': round(d_0['extension_atr'], 2),
            'd0_high_to_open_dollars': round(d_0['high_to_open'], 2),
            'd0_high_to_open_atr': round(d_0['high_to_open_atr'], 2),
            
            # EMA RELATIONSHIPS (D0)
            'd0_ema_9': round(d_0['ema_9'], 2),
            'd0_ema_20': round(d_0['ema_20'], 2),
            'd0_ema_50': round(d_0['ema_50'], 2),
            'd0_ema_89': round(d_0['ema_89'], 2),
            'd0_ema_200': round(d_0['ema_200'], 2),
            'd0_above_ema_9': d_0['open'] > d_0['ema_9'],
            'd0_above_ema_20': d_0['open'] > d_0['ema_20'],
            'd0_above_ema_50': d_0['open'] > d_0['ema_50'],
            'd0_above_ema_89': d_0['open'] > d_0['ema_89'],
            'd0_above_ema_200': d_0['open'] > d_0['ema_200'],
            'd0_ema_extension_pct': round(((d_0['open'] - d_0['ema_89']) / d_0['ema_89']) * 100, 1) if d_0['ema_89'] > 0 else 0,
            
            # EMA CLOUD STATUS
            'd0_ema_cloud_bullish': d_0['ema_cloud_bullish'] == 1,
            'd0_in_ema_cloud': (d_0['open'] >= d_0['ema_cloud_9_20_bottom']) and (d_0['open'] <= d_0['ema_cloud_9_20_top']),
            
            # MARKET STRUCTURE
            'downtrend_slope': round(downtrend_slope, 2),
            'd0_vs_250d_high': round(((d_0['open'] - d_0['highest_high_250']) / d_0['highest_high_250']) * 100, 1) if d_0['highest_high_250'] > 0 else 0,
            'd0_vs_100d_high': round(((d_0['open'] - d_0['highest_high_100']) / d_0['highest_high_100']) * 100, 1) if d_0['highest_high_100'] > 0 else 0,
            'd0_vs_50d_high': round(((d_0['open'] - d_0['highest_high_50']) / d_0['highest_high_50']) * 100, 1) if d_0['highest_high_50'] > 0 else 0,
            'backside_context': euphoric_high and d_0['open'] < euphoric_high,
            
            # DEVIATION BANDS
            'd0_dev_band_upper': round(d_0['dev_band_upper'], 2),
            'd0_dev_band_lower': round(d_0['dev_band_lower'], 2),
            'd0_above_upper_dev_band': d_0['open'] > d_0['dev_band_upper'],
            'd0_below_lower_dev_band': d_0['open'] < d_0['dev_band_lower'],
            
            # PERFORMANCE TRACKING
            'performance': performance_data
        }
        
        return raw_params

    def print_parameter_table(self, results):
        """Print a clean table of raw parameters for easy comparison"""
        if not results:
            print("No results to display")
            return
            
        # Filter valid results
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            print("No valid results to display")
            return
        
        print(f"\n{'='*120}")
        print("RAW PARAMETER VALUES - BACKSIDE POP MOLD ANALYSIS")
        print(f"{'='*120}")
        
        # Key parameters to display in table format
        key_params = [
            ('Symbol', 'symbol'),
            ('Date', 'setup_date'),
            ('Trend ATR', 'trend_atr_multiples'),
            ('Trend %', 'trend_total_move_pct'),
            ('Slope %/day', 'trend_slope_pct_per_day'),
            ('Days Since High', 'days_since_high'),
            ('Fade ATR', 'fade_from_high_atr'),
            ('Fade %', 'fade_from_high_pct'),
            ('D-1 Range %', 'd_minus_1_close_range_pct'),
            ('D-1 Vol Multiple', 'd_minus_1_volume_multiple'),
            ('D-1 Change ATR', 'd_minus_1_price_change_atr'),
            ('D0 Gap ATR', 'd0_gap_atr'),
            ('D0 Extension ATR', 'd0_extension_atr'),
            ('EMA89 Ext %', 'd0_ema_extension_pct'),
            ('Downtrend Slope', 'downtrend_slope')
        ]
        
        # Print header
        header = ""
        for label, _ in key_params:
            header += f"{label:>15}"
        print(header)
        print("-" * len(header))
        
        # Print data rows
        for result in valid_results:
            row = ""
            for _, param_key in key_params:
                value = result.get(param_key, 'N/A')
                if isinstance(value, (int, float)) and value != 'N/A':
                    row += f"{value:>15.2f}"
                else:
                    row += f"{str(value):>15}"
            print(row)
        
        print(f"\n{'='*120}")
        print("DETAILED BREAKDOWN BY NAME:")
        print(f"{'='*120}")
        
        # Print detailed breakdown for each name
        for result in valid_results:
            print(f"\n{result['symbol']} on {result['setup_date']}:")
            print("-" * 40)
            
            print("TREND CONTEXT:")
            print(f"  Start: {result['trend_start_date']} at ${result['trend_start_price']}")
            print(f"  High: {result['euphoric_high_date']} at ${result['euphoric_high_price']}")
            print(f"  Duration: {result['trend_duration_days']} days")
            print(f"  Total Move: {result['trend_atr_multiples']} ATR ({result['trend_total_move_pct']}%)")
            print(f"  Slope: {result['trend_slope_pct_per_day']}% per day")
            
            print("FADE CONTEXT:")
            print(f"  Days Since High: {result['days_since_high']}")
            print(f"  Fade Amount: {result['fade_from_high_atr']} ATR ({result['fade_from_high_pct']}%)")
            print(f"  Downtrend Slope: {result['downtrend_slope']}")
            
            print("D-1 SETUP:")
            print(f"  OHLC: ${result['d_minus_1_open']:.2f} / ${result['d_minus_1_high']:.2f} / ${result['d_minus_1_low']:.2f} / ${result['d_minus_1_close']:.2f}")
            print(f"  Range: ${result['d_minus_1_range_dollars']:.2f} ({result['d_minus_1_range_atr']:.2f} ATR)")
            print(f"  Close Range: {result['d_minus_1_close_range_pct']:.1f}%")
            print(f"  Change: {result['d_minus_1_price_change_atr']:.2f} ATR ({result['d_minus_1_price_change_pct']:.1f}%)")
            print(f"  Volume: {result['d_minus_1_volume_multiple']:.2f}x average")
            
            print("D0 SETUP:")
            print(f"  Open: ${result['d0_open']:.2f}")
            print(f"  Gap: {result['d0_gap_atr']:.2f} ATR ({result['d0_gap_pct']:.1f}%)")
            print(f"  Extension: {result['d0_extension_atr']:.2f} ATR")
            print(f"  Above EMA89: {result['d0_above_ema_89']} ({result['d0_ema_extension_pct']:.1f}% ext)")
            print(f"  Backside Context: {result['backside_context']}")
            
            if result['performance']:
                print("PERFORMANCE:")
                perf = result['performance']
                print(f"  Intraday: {perf['intraday_fade_pct']:.1f}%")
                print(f"  Next Day: {perf['next_day_fade_pct']:.1f}%")
                print(f"  3-Day: {perf['swing_fade_3d_pct']:.1f}%")
                print(f"  5-Day: {perf['swing_fade_5d_pct']:.1f}%")

    def analyze_mold_patterns(self):
        """Analyze A+ examples to understand the mold patterns"""
        a_plus_examples = [
            {'symbol': 'HOOD', 'date': '2025-03-03'},
            {'symbol': 'TSLA', 'date': '2024-07-15'}, 
            {'symbol': 'DJT', 'date': '2024-11-05'},
        ]
        
        results = []
        for example in a_plus_examples:
            print(f"Extracting parameters for: {example['symbol']} on {example['date']}")
            result = self.extract_raw_parameters(example['symbol'], example['date'])
            if 'error' not in result:
                results.append(result)
            else:
                print(f"Error with {example['symbol']}: {result['error']}")
        
        return results


# Example Usage and Testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = BacksidePopAnalyzer()
    
    print("EXTRACTING RAW PARAMETER VALUES FOR A+ EXAMPLES")
    print("=" * 60)
    
    # Analyze all A+ examples to see the raw mold
    mold_results = analyzer.analyze_mold_patterns()
    
    # Print the parameter comparison table
    analyzer.print_parameter_table(mold_results)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - Review the parameter values above")
    print("Use these values to set your scan thresholds and grading criteria")
    print("=" * 60)