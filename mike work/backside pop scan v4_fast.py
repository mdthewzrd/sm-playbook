"""
Fast Backside Pop Scanner V4 - Pre-filtered for Speed
Uses multiple baseline filters to only scan the most promising setups
"""

import pandas as pd
import requests
import time
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# CONFIGURATION
API_KEY = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
BASE_URL = "https://api.polygon.io"

# DATE RANGE - EDIT THESE
START_DATE = "2024-12-01"
END_DATE = "2025-04-01"

class FastBacksidePopScanner:
    """Fast Backside Pop Scanner with Heavy Pre-filtering"""
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        self.ticker_universe = []
        
        # SPEED OPTIMIZATION: Strict pre-filters to reduce scan volume by 90%+
        self.pre_filters = {
            'min_price': 15.0,              # Higher price floor
            'max_price': 500.0,             # Lower price ceiling  
            'min_volume': 2_000_000,        # Higher volume requirement
            'min_dollar_volume': 50_000_000, # Much higher dollar volume ($50M)
            'min_atr_pct': 3.0,             # Minimum ATR as % of price (volatility)
            'max_recent_low_days': 60,      # Must have hit recent low within 60 days
            'min_trend_size': 15.0,         # Minimum trend size in %
        }
        
        # Scan thresholds - Keep same as V3 but only run on pre-filtered candidates
        self.scan_thresholds = {
            'min_trend_atr': 6.0,
            'min_gap_atr': 0.4,
            'min_extension_atr': 1,
            'min_range_close_pct': 70.0,
            'min_volume_multiple': 0.7,
            'min_change_atr': 0.25,
            'max_downtrend_slope': -1.0,
            'min_ema_extension_pct': 20.0,
            'min_fade_atr': 1.5,
            'min_days_since_high': 1,
            'max_days_since_high': 30,
            'min_price': 10.0,
            'max_price': 1000.0,
            'min_volume': 1_000_000
        }
        
        # Grading system (same as V3)
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

    def get_high_volume_tickers(self):
        """Get top liquid tickers - much faster than API crawling"""
        # Pre-selected high-volume, volatile tickers likely to have backside pops
        high_volume_tickers = [
            # Mega cap tech (high volume)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
            
            # High beta/volatile large caps
            'AMD', 'MSTR', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'SNOW', 'ZM',
            'CRWD', 'OKTA', 'DDOG', 'NET', 'MDB', 'TEAM', 'WDAY', 'NOW',
            
            # Meme/Reddit favorites
            'GME', 'AMC', 'BB', 'WISH', 'CLOV', 'SPCE', 'PLUG', 'FCEL',
            
            # High volume financials
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'SCHW', 'BLK',
            
            # Biotech volatility 
            'MRNA', 'BNTX', 'GILD', 'BIIB', 'VRTX', 'REGN', 'AMGN', 'ILMN',
            
            # Energy momentum
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'OXY',
            
            # Consumer volatile
            'ROKU', 'PTON', 'DOCU', 'ZI', 'UBER', 'LYFT', 'ABNB', 'DASH',
            
            # Retail/e-commerce
            'SHOP', 'ETSY', 'EBAY', 'PINS', 'SNAP', 'TWTR', 'SPOT', 'SQ',
            
            # Industrial/transport
            'CAT', 'BA', 'UNP', 'CSX', 'NSC', 'UPS', 'FDX', 'LUV', 'DAL', 'AAL',
            
            # Semiconductors
            'INTC', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX', 'KLAC',
            
            # Healthcare/pharma
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY', 'MRK',
            
            # Communication
            'VZ', 'T', 'TMUS', 'DIS', 'CMCSA', 'CHTR', 'PARA',
            
            # Recent IPOs/high beta
            'RIVN', 'LCID', 'SOFI', 'UPST', 'AFRM', 'OPEN', 'RKLB', 'SPRT',
            
            # Energy transition
            'ENPH', 'SEDG', 'FSLR', 'RUN', 'NOVA', 'ARRY', 'ORA',
            
            # Cloud/SaaS  
            'CRM', 'ORCL', 'ADBE', 'INTU', 'PANW', 'FTNT', 'ZS', 'ESTC'
        ]
        
        print(f"Using {len(high_volume_tickers)} pre-selected high-volume tickers")
        return high_volume_tickers

    def fast_pre_filter_ticker(self, ticker, sample_date):
        """Lightning fast pre-filter - reject 80%+ of ticker-date combinations immediately"""
        try:
            # Fetch minimal data for speed
            end_date = pd.to_datetime(sample_date) + timedelta(days=5)
            start_date = pd.to_datetime(sample_date) - timedelta(days=90)
            
            df = self.fetch_daily_data(ticker, start_date.strftime('%Y-%m-%d'), 
                                     end_date.strftime('%Y-%m-%d'))
            
            if df.empty or len(df) < 60:
                return False
                
            # Find sample date
            sample_row = df[df['date'] == pd.to_datetime(sample_date).date()]
            if sample_row.empty:
                return False
                
            sample_idx = sample_row.index[0] 
            sample_data = df.iloc[sample_idx]
            
            # SPEED FILTER 1: Basic price/volume
            if not (self.pre_filters['min_price'] <= sample_data['close'] <= self.pre_filters['max_price']):
                return False
                
            if sample_data['volume'] < self.pre_filters['min_volume']:
                return False
                
            # SPEED FILTER 2: Dollar volume
            dollar_volume = sample_data['volume'] * sample_data['close']
            if dollar_volume < self.pre_filters['min_dollar_volume']:
                return False
            
            # SPEED FILTER 3: Recent volatility (ATR check)
            recent_data = df.iloc[max(0, sample_idx-20):sample_idx+1]
            if len(recent_data) < 10:
                return False
                
            recent_data = recent_data.copy()
            recent_data['range'] = recent_data['high'] - recent_data['low']
            avg_range = recent_data['range'].mean()
            atr_pct = (avg_range / sample_data['close']) * 100
            
            if atr_pct < self.pre_filters['min_atr_pct']:
                return False
                
            # SPEED FILTER 4: Must have recent significant high (trend requirement)
            lookback_data = df.iloc[max(0, sample_idx-60):sample_idx+1]
            high_price = lookback_data['high'].max()
            low_price = lookback_data['low'].min()
            trend_pct = ((high_price - low_price) / low_price) * 100
            
            if trend_pct < self.pre_filters['min_trend_size']:
                return False
                
            # SPEED FILTER 5: Must be near recent low (backside setup requirement)
            recent_low = lookback_data.tail(30)['low'].min()
            if sample_data['close'] > recent_low * 1.20:  # More than 20% above recent low
                return False
            
            return True
            
        except Exception:
            return False

    def fetch_daily_data(self, symbol, start_date, end_date):
        """Fetch daily OHLCV data from Polygon"""
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {'adjusted': 'true', 'sort': 'asc', 'apikey': self.api_key}
        
        try:
            time.sleep(0.02)  # Faster rate limiting
            response = requests.get(url, params=params, timeout=10)
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
        except Exception:
            return pd.DataFrame()

    def calculate_lingua_indicators(self, df):
        """Calculate indicators (same as V3 but optimized)"""
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # Previous day values
        df['pdc'] = df['close'].shift(1)
        
        # ATR Calculation
        df['high_low'] = df['high'] - df['low']
        df['high_pdc'] = abs(df['high'] - df['pdc'])
        df['low_pdc'] = abs(df['low'] - df['pdc'])
        df['true_range'] = df[['high_low', 'high_pdc', 'low_pdc']].max(axis=1)
        
        if len(df) >= 200:
            df['atr'] = df['true_range'].rolling(window=200).mean()
        elif len(df) >= 50:
            df['atr'] = df['true_range'].rolling(window=50).mean()
        else:
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
        
        return df

    def identify_trend_start(self, df):
        """Identify trend start using 9/20 EMA cross"""
        df = df.copy()
        df['ema_9_above_20'] = (df['ema_9'] > df['ema_20']).astype(int)
        df['cross_signal'] = df['ema_9_above_20'].diff()
        
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

    def scan_symbol(self, symbol, scan_date, lookback_days=400):
        """Scan a single symbol for the setup (same as V3)"""
        try:
            scan_date = pd.to_datetime(scan_date).date()
            start_date = scan_date - timedelta(days=lookback_days)
            end_date = scan_date + timedelta(days=10)
            
            # Fetch data
            df = self.fetch_daily_data(symbol, start_date.strftime('%Y-%m-%d'), 
                                      end_date.strftime('%Y-%m-%d'))
            
            if df.empty or len(df) < 30:
                return None
            
            # Calculate indicators
            df = self.calculate_lingua_indicators(df)
            
            # Find setup day
            setup_row = df[df['date'] == scan_date]
            if setup_row.empty:
                return None
            
            setup_idx = setup_row.index[0]
            prev_day_idx = setup_idx - 1
            
            if prev_day_idx < 0:
                return None
            
            d_minus_1 = df.iloc[prev_day_idx]
            d_0 = df.iloc[setup_idx]
            
            # Basic filters
            if not (self.scan_thresholds['min_price'] <= d_0['close'] <= self.scan_thresholds['max_price']):
                return None
            
            if d_minus_1['volume'] < self.scan_thresholds['min_volume']:
                return None
            
            if pd.isna(d_minus_1['atr']) or d_minus_1['atr'] <= 0:
                return None
            
            # CRITICAL: D-1 must be GREEN (close > open)
            if d_minus_1['close'] <= d_minus_1['open']:
                return None
            
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
                'gap_atr': d_0['gap_atr'] if not pd.isna(d_0['gap_atr']) else 0,
                'extension_atr': d_0['extension_atr'] if not pd.isna(d_0['extension_atr']) else 0,
                'range_close_pct': d_minus_1['close_range'] * 100 if not pd.isna(d_minus_1['close_range']) else 0,
                'volume_multiple': d_minus_1['volume_multiple'] if not pd.isna(d_minus_1['volume_multiple']) else 0,
                'change_atr': abs(d_minus_1['price_change_atr']) if not pd.isna(d_minus_1['price_change_atr']) else 0,
                'downtrend_slope': downtrend_slope,
                'ema_extension_pct': ((d_0['open'] - d_0['ema_89']) / d_0['ema_89']) * 100 if d_0['ema_89'] > 0 else 0,
                'fade_atr': fade_atr,
                'days_since_high': days_since_high
            }
            
            # Check scan criteria
            passes_scan = self.passes_scan_criteria(setup_data)
            
            if not passes_scan:
                return None
            
            # Calculate score
            scoring = self.calculate_setup_score(setup_data)
            
            # Performance tracking
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
            
        except Exception:
            return None

    def generate_date_range(self, start_date, end_date):
        """Generate list of trading dates between start and end date"""
        try:
            start = pd.to_datetime(start_date).date()
            end = pd.to_datetime(end_date).date()
            
            # Generate all dates in range
            date_range = pd.date_range(start=start, end=end, freq='D')
            
            # Filter to weekdays only (simple trading day approximation)
            trading_dates = [d.date() for d in date_range if d.weekday() < 5]
            
            return trading_dates
        except Exception:
            return []

    def run_fast_historical_scan(self, start_date, end_date):
        """Run FAST historical scan with heavy pre-filtering"""
        print(f"FAST BACKSIDE POP SCANNER V4")
        print(f"Date Range: {start_date} to {end_date}")
        print("-" * 60)
        
        # Generate trading dates
        trading_dates = self.generate_date_range(start_date, end_date)
        print(f"Scanning {len(trading_dates)} trading dates...")
        
        # Use pre-selected high-volume tickers
        self.ticker_universe = self.get_high_volume_tickers()
        print(f"Ticker Universe: {len(self.ticker_universe)} high-volume tickers")
        print("-" * 60)
        
        # SPEED OPTIMIZATION: Pre-filter ticker-date combinations
        print("🚀 PRE-FILTERING ticker-date combinations for speed...")
        
        valid_combinations = []
        total_combinations = len(self.ticker_universe) * len(trading_dates)
        checked = 0
        
        for date in trading_dates:
            for ticker in self.ticker_universe:
                checked += 1
                if checked % 1000 == 0:
                    progress = (checked / total_combinations) * 100
                    print(f"Pre-filter progress: {checked}/{total_combinations} ({progress:.1f}%) - {len(valid_combinations)} valid combinations")
                
                # Fast pre-filter
                if self.fast_pre_filter_ticker(ticker, date):
                    valid_combinations.append((ticker, date))
        
        print(f"🎯 PRE-FILTERING COMPLETE:")
        print(f"   Total combinations: {total_combinations:,}")
        print(f"   Valid combinations: {len(valid_combinations):,}")
        print(f"   Filtering efficiency: {((total_combinations - len(valid_combinations)) / total_combinations) * 100:.1f}% filtered out")
        print("-" * 60)
        
        if not valid_combinations:
            print("❌ No valid combinations found. Consider loosening pre-filters.")
            return
            
        # Now run full scan only on pre-filtered combinations
        print(f"🔍 RUNNING FULL SCAN on {len(valid_combinations):,} pre-filtered combinations...")
        
        all_results = []
        
        for i, (ticker, date) in enumerate(valid_combinations):
            if i % 100 == 0 or i == len(valid_combinations) - 1:
                progress = (i / len(valid_combinations)) * 100
                print(f"Scan progress: {i+1}/{len(valid_combinations)} ({progress:.1f}%) - {len(all_results)} setups found")
            
            result = self.scan_symbol(ticker, date)
            if result:
                all_results.append(result)
                print(f"✅ Found: {ticker} {date} - Grade: {result['grade']} (Score: {result['total_score']})")
        
        # Sort results
        all_results.sort(key=lambda x: (x['scan_date'], -x['total_score']))
        
        print(f"\n🎯 FAST SCAN COMPLETE")
        print(f"Pre-filtered combinations scanned: {len(valid_combinations):,}")
        print(f"Total setups found: {len(all_results)}")
        
        if all_results:
            # Grade distribution
            grades = {}
            for result in all_results:
                grade = result['grade']
                grades[grade] = grades.get(grade, 0) + 1
            
            print(f"Grade distribution: {dict(sorted(grades.items()))}")
            
            print(f"\nALL SETUPS FOUND:")
            print(f"{'Date':<12} {'Symbol':<6} {'Score':<5} {'Grade':<5} {'Trend':<6} {'Gap':<5} {'Ext':<5} {'Range':<6} {'Perf':<6}")
            print("-" * 75)
            
            for result in all_results:
                date_str = result['scan_date']
                symbol = result['symbol']
                score = result['total_score']
                grade = result['grade']
                trend = f"{result['setup_metrics']['trend_atr_multiples']:.1f}"
                gap = f"{result['setup_metrics']['gap_atr']:.2f}"
                ext = f"{result['setup_metrics']['extension_atr']:.2f}"
                range_pct = f"{result['setup_metrics']['range_close_pct']:.0f}%"
                
                # Performance
                if result['performance'] and 'swing_fade_5d_pct' in result['performance']:
                    perf = f"{result['performance']['swing_fade_5d_pct']:.1f}%"
                else:
                    perf = "N/A"
                
                print(f"{date_str:<12} {symbol:<6} {score:<5} {grade:<5} {trend:<6} {gap:<5} {ext:<5} {range_pct:<6} {perf:<6}")
        else:
            print("No setups found. Consider loosening scan thresholds.")

def main():
    """Run the fast scanner"""
    scanner = FastBacksidePopScanner()
    scanner.run_fast_historical_scan(START_DATE, END_DATE)

if __name__ == "__main__":
    main()