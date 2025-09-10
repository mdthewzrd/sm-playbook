"""
Clean Backside Pop Scanner V3 - Dynamic Ticker Universe with Liquidity Filtering
Based on V2 but now scans all NASDAQ and NYSE tickers that meet liquidity parameters
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

class BacksidePopScanner:
    """Clean Backside Pop Scanner with Dynamic Ticker Universe"""
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        self.ticker_universe = []  # Will be populated dynamically
        
        # Scan thresholds - LOWER VALUES FOR BETTER DETECTION
        self.scan_thresholds = {
            'min_trend_atr': 6.0,           # Reduced from 5.0
            'min_gap_atr': 0.4,             # Reduced from 0.3
            'min_extension_atr': 1,       # Reduced from 0.5
            'min_range_close_pct': 70.0,    # Reduced from 40.0
            'min_volume_multiple': 0.7,     # Reduced from 0.8
            'min_change_atr': 0.25,          # Reduced from 0.3
            'max_downtrend_slope': -1.0,    # Less strict (closer to 0)
            'min_ema_extension_pct': 20.0,   # Reduced from 5.0
            'min_fade_atr': 1.5,            # Reduced from 2.0
            'min_days_since_high': 1,       
            'max_days_since_high': 30,      # Increased from 60
            'min_price': 10.0,               # Reduced from 10.0
            'max_price': 1000.0,            # Increased from 500.0
            'min_volume': 1_000_000           # Reduced from 1M
        }
        
        # Liquidity parameters for ticker filtering
        self.liquidity_filters = {
            'min_price': 10.0,
            'max_price': 1000.0,
            'min_volume': 1_000_000,
            'min_pm_dollar_volume': 15_000_000,  # $15M minimum premarket dollar volume
        }
        
        # Grading system
        self.grading_system = {
            'weights': {
                'trend_weight': 25,      
                'gap_weight': 15,        
                'extension_weight': 20,  
                'range_weight': 15,      
                'volume_weight': 10,     
                'change_weight': 10,     
                'slope_weight': 5        
            },
            
            'grade_breakpoints': {
                'A+': 90, 'A': 80, 'B+': 70, 'B': 60, 'C+': 50, 'C': 40
            },
            
            'trend_scoring': {'A+_min': 12.0, 'A_min': 8.0, 'B+_min': 6.0, 'B_min': 4.0},
            'gap_scoring': {'A+_min': 0.8, 'A_min': 0.6, 'B+_min': 0.4, 'B_min': 0.3},
            'extension_scoring': {'A+_min': 1.5, 'A_min': 1.2, 'B+_min': 0.8, 'B_min': 0.5},
            'range_scoring': {'A+_min': 75.0, 'A_min': 65.0, 'B+_min': 50.0, 'B_min': 40.0},
            'volume_scoring': {'A+_min': 1.5, 'A_min': 1.2, 'B+_min': 1.0, 'B_min': 0.8},
            'change_scoring': {'A+_min': 1.0, 'A_min': 0.8, 'B+_min': 0.5, 'B_min': 0.3},
            'downtrend_slope_scoring': {'A+_max': -4.0, 'A_max': -2.0, 'B+_max': -1.0, 'B_max': -0.5}
        }

    def fetch_all_tickers(self):
        """Fetch all NASDAQ and NYSE tickers from Polygon API"""
        url = f"{self.base_url}/v3/reference/tickers"
        all_tickers = []
        
        # Try different approaches to get tickers
        exchanges = ['XNAS', 'XNYS']  # NASDAQ and NYSE separately
        
        for exchange in exchanges:
            params = {
                'market': 'stocks',
                'exchange': exchange,
                'active': 'true',
                'sort': 'ticker',
                'order': 'asc',
                'limit': 1000,
                'apikey': self.api_key
            }
            
            try:
                page_count = 0
                while page_count < 20:  # Limit pages to avoid infinite loop
                    time.sleep(0.1)  # Rate limiting
                    response = requests.get(url, params=params, timeout=30)
                    
                    print(f"Fetching {exchange} tickers (page {page_count + 1})...")
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get('results', [])
                        
                        if not results:
                            print(f"No more results for {exchange}")
                            break
                        
                        # Filter tickers based on basic criteria
                        for ticker_data in results:
                            ticker = ticker_data.get('ticker', '')
                            
                            # Basic filtering
                            if (ticker and 
                                len(ticker) <= 5 and  # Skip complex tickers
                                '.' not in ticker and  # Skip tickers with dots
                                '-' not in ticker and  # Skip tickers with dashes
                                not ticker.endswith('W') and  # Skip warrants
                                not ticker.endswith('U')):  # Skip units
                                all_tickers.append(ticker)
                        
                        print(f"Found {len(results)} tickers on page {page_count + 1}, total so far: {len(all_tickers)}")
                        
                        # Check for next page
                        if 'next_url' in data and data['next_url']:
                            # Extract cursor from next_url
                            next_url = data['next_url']
                            if 'cursor=' in next_url:
                                cursor = next_url.split('cursor=')[1].split('&')[0]
                                params['cursor'] = cursor
                                page_count += 1
                            else:
                                break
                        else:
                            break
                    else:
                        print(f"Error fetching {exchange} tickers: {response.status_code}")
                        if response.status_code == 429:
                            print("Rate limited, waiting...")
                            time.sleep(2)
                            continue
                        break
                        
            except Exception as e:
                print(f"Error fetching {exchange}: {e}")
                continue
        
        # Remove duplicates and sort
        all_tickers = sorted(list(set(all_tickers)))
        print(f"Total unique tickers found: {len(all_tickers)}")
        
        # If we didn't get many tickers, try a different approach
        if len(all_tickers) < 100:
            print("Low ticker count, trying alternative method...")
            all_tickers.extend(self.get_common_tickers())
            all_tickers = sorted(list(set(all_tickers)))
        
        return all_tickers

    def get_common_tickers(self):
        """Fallback method to get common tickers if API fails"""
        # Common high-volume tickers across different sectors
        common_tickers = [
            # Tech giants
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ADBE',
            'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
            'AXP', 'BLK', 'SCHW', 'CME', 'ICE', 'SPGI', 'MCO', 'AON', 'MMC', 'AJG',
            
            # Healthcare/Pharma
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'CVS', 'MRK',
            'LLY', 'MDT', 'GILD', 'AMGN', 'VRTX', 'REGN', 'BIIB', 'ILMN', 'MRNA', 'BNTX',
            
            # Consumer/Retail
            'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'SBUX', 'NKE', 'COST', 'TGT',
            'LOW', 'TJX', 'AMGN', 'EL', 'CL', 'KMB', 'GIS', 'K', 'CPB', 'CAG',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'HES', 'DVN',
            'OXY', 'BKR', 'HAL', 'KMI', 'WMB', 'OKE', 'EPD', 'ET', 'MPLX', 'ENB',
            
            # Industrial
            'CAT', 'BA', 'HON', 'UNP', 'RTX', 'LMT', 'GE', 'MMM', 'DE', 'EMR',
            'ITW', 'CSX', 'NSC', 'UPS', 'FDX', 'LUV', 'DAL', 'AAL', 'UAL', 'JBLU',
            
            # Communication
            'VZ', 'T', 'TMUS', 'DIS', 'CMCSA', 'CHTR', 'NFLX', 'SPOT', 'ROKU', 'TWTR',
            
            # Materials
            'LIN', 'APD', 'ECL', 'SHW', 'DD', 'DOW', 'PPG', 'NEM', 'FCX', 'VALE',
            
            # Real Estate
            'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'MAA', 'UDR',
            
            # Utilities  
            'NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'ES', 'PEG',
            
            # Meme/Popular stocks
            'GME', 'AMC', 'HOOD', 'PLTR', 'WISH', 'CLOV', 'BB', 'NOK', 'SNDL', 'TLRY',
            'SPCE', 'PLUG', 'FCEL', 'RIDE', 'NKLA', 'LCID', 'RIVN', 'F', 'GM', 'FORD',
            
            # ETFs and Popular
            'SPY', 'QQQ', 'IWM', 'ARKK', 'ARKW', 'ARKG', 'SQQQ', 'TQQQ', 'SPXL', 'UPRO',
            'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'AGG', 'LQD', 'HYG', 'JNK', 'TLT',
            
            # Recent popular/volatile
            'MSTR', 'COIN', 'RBLX', 'SNOW', 'ZM', 'PTON', 'DOCU', 'ZI', 'CRWD', 'OKTA',
            'DDOG', 'NET', 'FSLY', 'ESTC', 'MDB', 'TEAM', 'WDAY', 'NOW', 'SPLK', 'VEEV',
            
            # Biotech/Small cap popular
            'SGEN', 'BMRN', 'ALNY', 'IONS', 'SRPT', 'EXAS', 'ARCT', 'FOLD', 'BLUE', 'SAGE',
            'NTLA', 'CRSP', 'EDIT', 'BEAM', 'PRIME', 'VERV', 'CGEM', 'SANA', 'FATE', 'CRBU',
        ]
        
        print(f"Added {len(common_tickers)} common tickers as fallback")
        return common_tickers

    def filter_tickers_by_liquidity(self, tickers, sample_date):
        """Filter tickers based on liquidity requirements including premarket dollar volume"""
        filtered_tickers = []
        total_tickers = len(tickers)
        
        print(f"Filtering {total_tickers} tickers for liquidity on {sample_date}...")
        
        for i, ticker in enumerate(tickers):
            if i % 100 == 0:
                print(f"Progress: {i}/{total_tickers} ({(i/total_tickers)*100:.1f}%)")
            
            # Fetch recent data to check liquidity
            end_date = pd.to_datetime(sample_date) + timedelta(days=1)
            start_date = pd.to_datetime(sample_date) - timedelta(days=30)
            
            df = self.fetch_daily_data(ticker, 
                                     start_date.strftime('%Y-%m-%d'), 
                                     end_date.strftime('%Y-%m-%d'))
            
            if not df.empty and len(df) >= 20:
                # Get most recent data
                latest = df.iloc[-1]
                
                # Check basic price and volume filters
                if (self.liquidity_filters['min_price'] <= latest['close'] <= self.liquidity_filters['max_price'] and
                    latest['volume'] >= self.liquidity_filters['min_volume']):
                    
                    # Check average volume and dollar volume over last 20 days
                    recent_data = df.tail(20)
                    avg_volume = recent_data['volume'].mean()
                    avg_price = recent_data['close'].mean()
                    avg_dollar_volume = avg_volume * avg_price
                    
                    if (avg_volume >= self.liquidity_filters['min_volume'] and
                        avg_dollar_volume >= self.liquidity_filters['min_pm_dollar_volume']):
                        filtered_tickers.append(ticker)
        
        print(f"Liquidity filtering complete: {len(filtered_tickers)} tickers passed")
        return filtered_tickers

    def build_dynamic_ticker_universe(self, sample_date):
        """Build ticker universe dynamically based on liquidity"""
        print("Building dynamic ticker universe...")
        
        # Fetch all available tickers
        print("Fetching all NASDAQ and NYSE tickers...")
        all_tickers = self.fetch_all_tickers()
        print(f"Found {len(all_tickers)} potential tickers")
        
        if not all_tickers:
            print("No tickers found, falling back to manual list")
            return ['HOOD', 'MSTR', 'SMCI', 'IBIT', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META']
        
        # Filter by liquidity
        filtered_tickers = self.filter_tickers_by_liquidity(all_tickers, sample_date)
        
        print(f"Final ticker universe: {len(filtered_tickers)} tickers")
        return filtered_tickers

    def fetch_daily_data(self, symbol, start_date, end_date):
        """Fetch daily OHLCV data from Polygon"""
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'apikey': self.api_key
        }
        
        try:
            time.sleep(0.05)
            response = requests.get(url, params=params, timeout=15)
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
            return pd.DataFrame()

    def calculate_lingua_indicators(self, df):
        """Calculate all indicators with fallback ATR"""
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # Previous day values
        df['pdc'] = df['close'].shift(1)
        
        # ATR Calculation with fallback
        df['high_low'] = df['high'] - df['low']
        df['high_pdc'] = abs(df['high'] - df['pdc'])
        df['low_pdc'] = abs(df['low'] - df['pdc'])
        df['true_range'] = df[['high_low', 'high_pdc', 'low_pdc']].max(axis=1)
        
        # Try 200-period ATR first, fallback to shorter periods
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
        """Scan a single symbol for the setup"""
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
            
        except Exception as e:
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
        except Exception as e:
            print(f"Error generating date range: {e}")
            return []

    def run_historical_scan(self, start_date, end_date):
        """Run historical scan with dynamic ticker universe"""
        print(f"BACKSIDE POP HISTORICAL SCAN V3")
        print(f"Date Range: {start_date} to {end_date}")
        print("-" * 60)
        
        # Generate trading dates
        trading_dates = self.generate_date_range(start_date, end_date)
        print(f"Scanning {len(trading_dates)} trading dates...")
        
        # Build dynamic ticker universe using the first scan date as sample
        sample_date = trading_dates[0] if trading_dates else start_date
        self.ticker_universe = self.build_dynamic_ticker_universe(sample_date)
        
        print(f"Ticker Universe: {len(self.ticker_universe)} tickers")
        print("-" * 60)
        
        all_results = []
        total_scans = len(self.ticker_universe) * len(trading_dates)
        current_scan = 0
        
        # Scan each combination
        for date in trading_dates:
            for symbol in self.ticker_universe:
                current_scan += 1
                if current_scan % 1000 == 0 or current_scan == total_scans:
                    progress = (current_scan / total_scans) * 100
                    print(f"Progress: {current_scan}/{total_scans} ({progress:.1f}%) - {len(all_results)} setups found")
                
                result = self.scan_symbol(symbol, date)
                if result:
                    all_results.append(result)
                    print(f"Found: {symbol} {date} - Grade: {result['grade']} (Score: {result['total_score']})")
        
        # Sort results
        all_results.sort(key=lambda x: (x['scan_date'], -x['total_score']))
        
        print(f"\nSCAN COMPLETE")
        print(f"Total tickers scanned: {len(self.ticker_universe)}")
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
                if result['performance'] and result['performance']['swing_fade_5d_pct']:
                    perf = f"{result['performance']['swing_fade_5d_pct']:.1f}%"
                else:
                    perf = "N/A"
                
                print(f"{date_str:<12} {symbol:<6} {score:<5} {grade:<5} {trend:<6} {gap:<5} {ext:<5} {range_pct:<6} {perf:<6}")
        else:
            print("No setups found. Consider lowering scan thresholds.")

def main():
    """Run the scanner"""
    scanner = BacksidePopScanner()
    scanner.run_historical_scan(START_DATE, END_DATE)

if __name__ == "__main__":
    main()