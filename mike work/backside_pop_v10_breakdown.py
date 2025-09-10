"""
Backside Pop Scanner V10 - Breakdown Variant
Looks for consolidation/support breaks with high volume as catalyst, then pops after breakdown
Different from standard V10 which looks for extension setups
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

# DATE RANGE AND TICKERS
START_DATE = "2024-01-01"
END_DATE = "2025-09-01"
MAX_WORKERS = 16

TICKER_UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'CRM', 'ADBE', 'PYPL', 'INTC', 'CSCO', 'PEP', 'AVGO', 'CMCSA', 'TXN', 'QCOM', 'COST', 'TMUS', 'HON', 'UNP', 'SBUX', 'AMAT', 'INTU', 'BKNG', 'ISRG', 'ADP', 'GILD', 'AMT', 'MU', 'VRTX', 'LRCX', 'FISV', 'CSX', 'ADI', 'REGN', 'ATVI', 'MDLZ', 'KLAC', 'ORLY', 'SNPS', 'CDNS', 'MAR', 'MRVL', 'FTNT', 'ASML', 'CRWD', 'ADSK', 'NXPI', 'WDAY', 'ABNB', 'TEAM', 'DXCM', 'MELI', 'KHC', 'EXC', 'CSGP', 'FANG', 'CHTR', 'PANW', 'AEP', 'KDP', 'PAYX', 'ROST', 'ODFL', 'FAST', 'VRSK', 'CTSH', 'BKR', 'EA', 'DDOG', 'CPRT', 'PCAR', 'XEL', 'EBAY', 'GEHC', 'MNST', 'MRNA', 'AZN', 'COIN']

class BacksidePopV10Breakdown:
    """V10 Breakdown variant - looks for consolidation breaks with volume catalyst"""
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        self.ticker_universe = TICKER_UNIVERSE
        self.max_workers = MAX_WORKERS
        
        # Breakdown scan thresholds with easy on/off toggles
        self.scan_thresholds = {
            # Standard Backside Pop Parameters (from V10/V11)
            'min_trend_atr': 4.0,           
            'min_gap_atr': 0.5,             
            'min_extension_atr': 1.5,       
            'min_range_close_pct': 70.0,    # D-1 range close percentage
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
            'min_dollar_volume_20d': 20_000_000,
            
            # Filter Toggles - Easy On/Off Controls
            'require_dev_band_upper': True,   # D0 open/high must be above 0.5 deviation band
            'require_d_minus_1_green': True,  # D-1 must be green (close > open)
            'require_bearish_ema': True,      # Daily 9/20 must be bearish (backside)
            'require_trend_atr': True,        # Enable trend ATR filter
            'require_gap_atr': True,          # Enable gap ATR filter
            'require_extension_atr': True,    # Enable extension ATR filter
            'require_range_close_pct': True,  # Enable D-1 range close filter
            'require_volume_multiple': True,  # Enable volume multiple filter
            'require_change_atr': True,       # Enable change ATR filter
            'require_downtrend_slope': True,  # Enable downtrend slope filter
            'require_ema_extension_pct': True, # Enable EMA extension filter
            'require_fade_atr': True,         # Enable fade ATR filter
            'require_days_since_high': True,  # Enable days since high filter
            'require_consolidation_breakdown': True,  # Enable consolidation breakdown pattern
            'require_range_expansion': True,  # Enable range expansion filter  
            
            # Breakdown Pattern Requirements (replaces fade pattern)
            'min_breakdown_volume_multiple': 0.5,  # Breakdown day needs elevated volume
            'min_breakdown_range_atr': 0.5,        # Breakdown day needs wide range
            'breakdown_lookback_days': 100,         # Look back for consolidation period
            'min_consolidation_days': 7,          # Minimum consolidation period (2 weeks)
            'max_consolidation_days': 100,          # Maximum consolidation period (2 months)
            'consolidation_range_tolerance': 0.05, # 5% tolerance for consolidation boundaries
            
            # Increased Range Requirements (higher volatility periods)
            'min_recent_range_expansion': 1.1,     # Recent ATR should be 1.2x+ historical
            'range_expansion_lookback': 252,       # Look back 1 year for range comparison
            
            # Trend Duration Validation
            'min_trend_duration_days': 3,
            'max_trend_duration_days': 60
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'backside-v10-breakdown'})

    def fetch_daily_data_cached(self, symbol, start_date, end_date):
        """Fetch daily OHLCV data from Polygon"""
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {'adjusted': 'true', 'sort': 'asc', 'apikey': self.api_key}
        
        try:
            time.sleep(0.012)
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
        """Calculate indicators with enhanced 9/20 EMA deviation bands"""
        if df.empty or len(df) < 10:
            return df
            
        df = df.copy().sort_values('date').reset_index(drop=True)
        
        # Basic calculations
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
        
        # EMAs
        df['ema_9'] = df['close'].ewm(span=9, min_periods=5).mean()
        df['ema_20'] = df['close'].ewm(span=20, min_periods=10).mean()
        df['ema_89'] = df['close'].ewm(span=89, min_periods=20).mean()
        
        # Basic calculations
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
        
        # Dollar volume
        df['dollar_volume'] = df['close'] * df['volume']
        df['avg_dollar_volume_20d'] = df['dollar_volume'].rolling(20, min_periods=5).mean()
        
        # Red/Green classification
        df['is_red_day'] = df['close'] < df['open']
        df['is_green_day'] = df['close'] > df['open']
        
        # Deviation bands
        df['ATR_9'] = df['true_range'].rolling(9, min_periods=3).mean()
        df['dev_band_upper_1'] = df['ema_9'] + 1.0 * df['ATR_9']
        df['dev_band_upper_2'] = df['ema_9'] + 0.5 * df['ATR_9']
        df['dev_band_lower_1'] = df['ema_20'] - 2.0 * df['ATR_9']
        df['dev_band_lower_2'] = df['ema_20'] - 2.4 * df['ATR_9']
        
        df['open_above_dev_upper'] = df['open'] > df['dev_band_upper_2']
        df['high_above_dev_upper'] = df['high'] > df['dev_band_upper_2']
        
        # Backside EMA condition (9 below 20)
        df['ema_bearish'] = df['ema_9'] < df['ema_20']
        
        # Range expansion calculation (recent vs historical volatility)
        df['atr_60d'] = df['true_range'].rolling(60, min_periods=20).mean()  # Recent ATR
        df['atr_252d'] = df['true_range'].rolling(252, min_periods=100).mean()  # Historical ATR
        df['range_expansion_ratio'] = df['atr_60d'] / df['atr_252d']
        
        return df

    def detect_consolidation_breakdown(self, df, setup_idx):
        """Detect consolidation period and breakdown with volume catalyst"""
        if setup_idx < self.scan_thresholds['breakdown_lookback_days']:
            return False, None
        
        # Look back for consolidation period
        lookback_start = max(0, setup_idx - self.scan_thresholds['breakdown_lookback_days'])
        analysis_data = df.iloc[lookback_start:setup_idx].copy()
        
        if len(analysis_data) < self.scan_thresholds['min_consolidation_days']:
            return False, None
        
        # Find potential consolidation periods
        consolidation_found = False
        breakdown_info = None
        
        # Look for periods where price stayed within a range
        for start_idx in range(len(analysis_data) - self.scan_thresholds['min_consolidation_days']):
            for end_idx in range(start_idx + self.scan_thresholds['min_consolidation_days'], 
                               min(start_idx + self.scan_thresholds['max_consolidation_days'], len(analysis_data))):
                
                consolidation_period = analysis_data.iloc[start_idx:end_idx]
                
                # Calculate consolidation boundaries
                period_high = consolidation_period['high'].max()
                period_low = consolidation_period['low'].min()
                consolidation_range = (period_high - period_low) / period_low
                
                # Check if this was a reasonable consolidation (not too wide)
                if consolidation_range > 0.2:  # Skip if range > 20%
                    continue
                
                # Check if most days stayed within the range
                within_range_count = 0
                for _, row in consolidation_period.iterrows():
                    if (period_low * (1 - self.scan_thresholds['consolidation_range_tolerance']) <= 
                        row['low'] <= period_high * (1 + self.scan_thresholds['consolidation_range_tolerance'])):
                        within_range_count += 1
                
                consolidation_ratio = within_range_count / len(consolidation_period)
                
                # If >70% of days were within consolidation range, it's valid
                if consolidation_ratio >= 0.7:
                    # Check for breakdown after consolidation
                    post_consolidation = analysis_data.iloc[end_idx:]
                    
                    for _, breakdown_day in post_consolidation.iterrows():
                        # Check if this day broke below consolidation with volume
                        volume_mult = breakdown_day['volume_multiple'] if not pd.isna(breakdown_day['volume_multiple']) else 0
                        range_atr = breakdown_day['range_atr'] if not pd.isna(breakdown_day['range_atr']) else 0
                        
                        broke_below = breakdown_day['low'] < period_low * 0.98  # 2% below consolidation
                        high_volume = volume_mult >= self.scan_thresholds['min_breakdown_volume_multiple']
                        wide_range = range_atr >= self.scan_thresholds['min_breakdown_range_atr']
                        
                        if broke_below and high_volume and wide_range:
                            consolidation_found = True
                            breakdown_info = {
                                'consolidation_start_date': consolidation_period.iloc[0]['date'],
                                'consolidation_end_date': consolidation_period.iloc[-1]['date'],
                                'consolidation_days': len(consolidation_period),
                                'consolidation_high': period_high,
                                'consolidation_low': period_low,
                                'consolidation_range_pct': consolidation_range * 100,
                                'breakdown_date': breakdown_day['date'],
                                'breakdown_volume_multiple': volume_mult,
                                'breakdown_range_atr': range_atr,
                                'breakdown_low': breakdown_day['low']
                            }
                            break
                    
                    if consolidation_found:
                        break
                
                if consolidation_found:
                    break
            
            if consolidation_found:
                break
        
        return consolidation_found, breakdown_info

    def find_trend_and_euphoric_high_fast(self, df, setup_idx):
        """Find trend and euphoric high with duration validation"""
        if setup_idx < 20:
            return None, None, None, None
            
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
        
        # Find euphoric high
        trend_data = pre_setup.iloc[trend_start_idx:]
        if trend_data.empty:
            return None, None, None, None
            
        euphoric_idx = trend_data['high'].idxmax()
        euphoric_date = trend_data.loc[euphoric_idx, 'date']
        euphoric_high = trend_data.loc[euphoric_idx, 'high']
        
        # Validate trend duration
        trend_duration = (euphoric_date - trend_start_date).days
        if not (self.scan_thresholds['min_trend_duration_days'] <= trend_duration <= self.scan_thresholds['max_trend_duration_days']):
            return None, None, None, None
        
        return trend_start_date, trend_start_price, euphoric_date, euphoric_high

    def scan_single_ticker_breakdown(self, symbol, start_date, end_date):
        """V10 Breakdown scan for consolidation break setups"""
        # Fetch data with extended lookback
        extended_start = (pd.to_datetime(start_date) - timedelta(days=500)).strftime('%Y-%m-%d')
        df = self.fetch_daily_data_cached(symbol, extended_start, end_date)
        
        if df.empty or len(df) < 100:
            return []
        
        # Calculate indicators
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
            
            # BREAKDOWN FILTERS
            
            # 1. Basic filters
            if not (self.scan_thresholds['min_price'] <= d_0['close'] <= self.scan_thresholds['max_price']):
                continue
            if d_minus_1['volume'] < self.scan_thresholds['min_volume']:
                continue
            if pd.isna(d_minus_1['atr']) or d_minus_1['atr'] <= 0:
                continue
            
            # 2. D-1 must be green
            if d_minus_1['close'] <= d_minus_1['open']:
                continue
                
            # 3. Dollar volume liquidity filter
            d_minus_1_dollar_vol_20d = d_minus_1['avg_dollar_volume_20d'] if not pd.isna(d_minus_1['avg_dollar_volume_20d']) else 0
            if d_minus_1_dollar_vol_20d < self.scan_thresholds['min_dollar_volume_20d']:
                continue
            
            # 4. Must be on backside - daily 9/20 bearish
            ema_bearish = d_0['ema_bearish'] if not pd.isna(d_0['ema_bearish']) else False
            if not ema_bearish:
                continue
            
            # 5. D0 open/high must be above 0.5 deviation band
            d0_open_above_dev = d_0['open_above_dev_upper'] if not pd.isna(d_0['open_above_dev_upper']) else False
            d0_high_above_dev = d_0['high_above_dev_upper'] if not pd.isna(d_0['high_above_dev_upper']) else False
            
            if not (d0_open_above_dev or d0_high_above_dev):
                continue
            
            # 6. Range expansion requirement (increased volatility) - TOGGLEABLE
            if self.scan_thresholds['require_range_expansion']:
                range_expansion = d_0['range_expansion_ratio'] if not pd.isna(d_0['range_expansion_ratio']) else 1.0
                if range_expansion < self.scan_thresholds['min_recent_range_expansion']:
                    continue
            
            # 7. Find trend and euphoric high
            trend_start_date, trend_start_price, euphoric_date, euphoric_high = \
                self.find_trend_and_euphoric_high_fast(df, idx)
            
            if not all([trend_start_date, trend_start_price, euphoric_date, euphoric_high]):
                continue
            
            # 8. Check for consolidation breakdown pattern - TOGGLEABLE
            if self.scan_thresholds['require_consolidation_breakdown']:
                breakdown_valid, breakdown_info = self.detect_consolidation_breakdown(df, idx)
                if not breakdown_valid:
                    continue
            else:
                # Create dummy breakdown info if not required
                breakdown_info = {
                    'consolidation_start_date': trend_start_date,
                    'consolidation_end_date': euphoric_date, 
                    'consolidation_days': (euphoric_date - trend_start_date).days,
                    'consolidation_high': euphoric_high,
                    'consolidation_low': d_minus_1['close'],
                    'consolidation_range_pct': 0.0,
                    'breakdown_date': current_date,
                    'breakdown_volume_multiple': d_minus_1['volume_multiple'] if not pd.isna(d_minus_1['volume_multiple']) else 0,
                    'breakdown_range_atr': d_minus_1['range_atr'] if not pd.isna(d_minus_1['range_atr']) else 0,
                    'breakdown_low': d_0['low']
                }
            
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
            
            # TOGGLEABLE Breakdown scan criteria - each can be enabled/disabled
            scan_criteria = []
            
            # Trend ATR filter
            if self.scan_thresholds['require_trend_atr']:
                scan_criteria.append(trend_atr_multiples >= self.scan_thresholds['min_trend_atr'])
            
            # Gap ATR filter
            if self.scan_thresholds['require_gap_atr']:
                scan_criteria.append(gap_atr >= self.scan_thresholds['min_gap_atr'])
            
            # Extension ATR filter
            if self.scan_thresholds['require_extension_atr']:
                scan_criteria.append(extension_atr >= self.scan_thresholds['min_extension_atr'])
            
            # Range close percentage filter
            if self.scan_thresholds['require_range_close_pct']:
                scan_criteria.append(range_close_pct >= self.scan_thresholds['min_range_close_pct'])
            
            # Volume multiple filter
            if self.scan_thresholds['require_volume_multiple']:
                scan_criteria.append(volume_multiple >= self.scan_thresholds['min_volume_multiple'])
            
            # Change ATR filter
            if self.scan_thresholds['require_change_atr']:
                scan_criteria.append(change_atr >= self.scan_thresholds['min_change_atr'])
            
            # Downtrend slope filter
            if self.scan_thresholds['require_downtrend_slope']:
                scan_criteria.append(downtrend_slope <= self.scan_thresholds['max_downtrend_slope'])
            
            # EMA extension filter
            if self.scan_thresholds['require_ema_extension_pct']:
                scan_criteria.append(ema_extension_pct >= self.scan_thresholds['min_ema_extension_pct'])
            
            # Fade ATR filter
            if self.scan_thresholds['require_fade_atr']:
                scan_criteria.append(fade_atr >= self.scan_thresholds['min_fade_atr'])
            
            # Days since high filter
            if self.scan_thresholds['require_days_since_high']:
                scan_criteria.append(self.scan_thresholds['min_days_since_high'] <= days_since_high <= self.scan_thresholds['max_days_since_high'])
            
            # If no criteria enabled, pass all (shouldn't happen but safe fallback)
            if not scan_criteria:
                scan_criteria = [True]
            
            if all(scan_criteria):
                # Get range expansion for output (even if filter disabled)
                range_expansion = d_0['range_expansion_ratio'] if not pd.isna(d_0['range_expansion_ratio']) else 1.0
                
                setup = {
                    'symbol': symbol,
                    'date': current_date.strftime('%Y-%m-%d'),
                    
                    # Core backside pop metrics
                    'trend_atr': round(trend_atr_multiples, 2),
                    'gap_atr': round(gap_atr, 2),
                    'extension_atr': round(extension_atr, 2),
                    'range_close_pct': round(range_close_pct, 1),  # D-1 range close percentage
                    'volume_multiple': round(volume_multiple, 2),
                    'change_atr': round(change_atr, 2),
                    'fade_atr': round(fade_atr, 2),
                    'downtrend_slope': round(downtrend_slope, 3),
                    'ema_extension_pct': round(ema_extension_pct, 1),
                    'days_since_high': days_since_high,
                    
                    # Date references
                    'trend_start': trend_start_date.strftime('%Y-%m-%d'),
                    'euphoric_high': euphoric_date.strftime('%Y-%m-%d'),
                    
                    # Pattern info
                    'pattern_type': 'consolidation_breakdown' if self.scan_thresholds['require_consolidation_breakdown'] else 'standard_backside',
                    'consolidation_days': breakdown_info['consolidation_days'],
                    'consolidation_start': breakdown_info['consolidation_start_date'].strftime('%Y-%m-%d'),
                    'consolidation_end': breakdown_info['consolidation_end_date'].strftime('%Y-%m-%d'),
                    'consolidation_range_pct': round(breakdown_info['consolidation_range_pct'], 1),
                    'breakdown_date': breakdown_info['breakdown_date'].strftime('%Y-%m-%d'),
                    'breakdown_volume': round(breakdown_info['breakdown_volume_multiple'], 1),
                    'breakdown_range_atr': round(breakdown_info['breakdown_range_atr'], 1),
                    
                    # Technical levels and OHLC data
                    'range_expansion_ratio': round(range_expansion, 2),
                    'dev_band_upper_05': round(d_0['dev_band_upper_2'], 2) if not pd.isna(d_0['dev_band_upper_2']) else None,
                    'd0_open': round(d_0['open'], 2),
                    'd0_high': round(d_0['high'], 2),
                    'd0_close': round(d_0['close'], 2),
                    'd_minus_1_open': round(d_minus_1['open'], 2),
                    'd_minus_1_high': round(d_minus_1['high'], 2),
                    'd_minus_1_low': round(d_minus_1['low'], 2),
                    'd_minus_1_close': round(d_minus_1['close'], 2),
                    
                    # Volume and liquidity
                    'dollar_volume_20d_m': round(d_minus_1_dollar_vol_20d / 1_000_000, 1) if d_minus_1_dollar_vol_20d > 0 else None,
                    
                    'scanner_version': 'V10_Breakdown_Toggleable'
                }
                setups.append(setup)
        
        return setups

    def run_breakdown_scan(self, start_date=None, end_date=None, tickers=None):
        """V10 Breakdown scanner"""
        start_date = start_date or START_DATE
        end_date = end_date or END_DATE
        tickers = tickers or self.ticker_universe
        
        print(f"🚀 BACKSIDE POP SCANNER V10 - BREAKDOWN VARIANT")
        print(f"📅 Scanning {len(tickers)} tickers from {start_date} to {end_date}")
        print(f"🔧 V10 Breakdown Features:")
        print(f"   • Must be on backside (daily 9 EMA < 20 EMA)")
        print(f"   • Consolidation break with volume catalyst ({self.scan_thresholds['min_breakdown_volume_multiple']}x+ volume)")
        print(f"   • Consolidation period: {self.scan_thresholds['min_consolidation_days']}-{self.scan_thresholds['max_consolidation_days']} days")
        print(f"   • Increased range expansion ({self.scan_thresholds['min_recent_range_expansion']}x+ recent vs historical volatility)")
        print(f"   • Trend duration validation ({self.scan_thresholds['min_trend_duration_days']}-{self.scan_thresholds['max_trend_duration_days']} days)")
        print(f"   • ${self.scan_thresholds['min_dollar_volume_20d']/1_000_000:.0f}M minimum dollar volume")
        print("=" * 80)
        
        all_setups = []
        processed = 0
        errors = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.scan_single_ticker_breakdown, ticker, start_date, end_date): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    setups = future.result()
                    if setups:
                        all_setups.extend(setups)
                        print(f"✅ {ticker}: {len(setups)} breakdown setups found")
                    else:
                        print(f"⚪ {ticker}: No breakdown setups")
                except Exception as e:
                    print(f"❌ {ticker}: Error - {str(e)[:50]}")
                    errors += 1
                
                processed += 1
                if processed % 20 == 0:
                    print(f"📊 Progress: {processed}/{len(tickers)} ({processed/len(tickers)*100:.1f}%)")
        
        print("\n" + "=" * 80)
        print(f"🎯 V10 BREAKDOWN SCAN COMPLETE")
        print(f"📊 Results: {len(all_setups)} breakdown setups found")
        print(f"⚠️  Errors: {errors} tickers failed")
        
        if all_setups:
            df_results = pd.DataFrame(all_setups)
            df_results = df_results.sort_values(['date', 'trend_atr'], ascending=[False, False])
            
            print(f"\n🏆 ALL V10 BREAKDOWN SETUPS:")
            display_cols = ['symbol', 'date', 'trend_atr', 'gap_atr', 'extension_atr', 'consolidation_days', 'breakdown_volume', 'range_expansion_ratio']
            print(df_results[display_cols].to_string(index=False))
            
            # Consolidation duration breakdown
            if 'consolidation_days' in df_results.columns:
                avg_consolidation = df_results['consolidation_days'].mean()
                print(f"\n📊 CONSOLIDATION ANALYSIS:")
                print(f"   Average consolidation duration: {avg_consolidation:.1f} days")
                print(f"   Range: {df_results['consolidation_days'].min()}-{df_results['consolidation_days'].max()} days")
            
            return df_results
        else:
            print("No breakdown setups found matching V10 criteria.")
            return pd.DataFrame()

# Example Usage
if __name__ == "__main__":
    # Initialize V10 breakdown scanner
    scanner = BacksidePopV10Breakdown()
    
    # Run breakdown scan
    results = scanner.run_breakdown_scan()
    
    if not results.empty:
        # Save results to CSV
        results.to_csv('backside_pop_v10_breakdown_results.csv', index=False)
        print(f"\n💾 Breakdown results saved to: backside_pop_v10_breakdown_results.csv")