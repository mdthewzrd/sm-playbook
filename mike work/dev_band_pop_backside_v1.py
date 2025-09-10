"""
Dev Band Pop Backside Scanner V1
Simple scanner for liquid names popping back into 0.4 upper deviation bands after being on backside
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
START_DATE = "2025-01-01"
END_DATE = "2025-09-01"
MAX_WORKERS = 16

TICKER_UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'CRM', 'ADBE', 'PYPL', 'INTC', 'CSCO', 'PEP', 'AVGO', 'CMCSA', 'TXN', 'QCOM', 'COST', 'TMUS', 'HON', 'UNP', 'SBUX', 'AMAT', 'INTU', 'BKNG', 'ISRG', 'ADP', 'GILD', 'AMT', 'MU', 'VRTX', 'LRCX', 'FISV', 'CSX', 'ADI', 'REGN', 'ATVI', 'MDLZ', 'KLAC', 'ORLY', 'SNPS', 'CDNS', 'MAR', 'MRVL', 'FTNT', 'ASML', 'CRWD', 'ADSK', 'NXPI', 'WDAY', 'ABNB', 'TEAM', 'DXCM', 'MELI', 'KHC', 'EXC', 'CSGP', 'FANG', 'CHTR', 'PANW', 'AEP', 'KDP', 'PAYX', 'ROST', 'ODFL', 'FAST', 'VRSK', 'CTSH', 'BKR', 'EA', 'DDOG', 'CPRT', 'PCAR', 'XEL', 'EBAY', 'GEHC', 'MNST', 'MRNA', 'AZN', 'COIN']

class DevBandPopBacksideV1:
    """Simple scanner for backside names popping into 0.4 upper dev bands"""
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        self.ticker_universe = TICKER_UNIVERSE
        self.max_workers = MAX_WORKERS
        
        # Simple scan criteria
        self.scan_thresholds = {
            # Basic filters
            'min_price': 10.0,
            'max_price': 1000.0,
            'min_volume': 10_000_000,
            'min_dollar_volume_20d': 20_000_000,  # Liquidity filter
            
            # Volume and gap criteria
            'min_volume_multiple': 1.1,      # Simple volume requirement
            'min_gap_over_close_atr': 0.4,   # Gap over previous close in ATR
            'min_gap_over_high_atr': 0.05,    # Gap over previous high in ATR
            'min_extension_to_low_atr': 1.250, # Extension from D0 open to D-1 low in ATR
            
            # Dev band criteria - 0.4 multiplier for sensitivity
            'dev_band_multiplier': 0.35,
            
            # Backside criteria - must have been below dev band recently AND 9/20 bearish
            'lookback_days': 15,             # Look back for backside confirmation
            'min_days_below_band': 3,        # Must have been below band for at least 3 days
            'require_bearish_ema': True,     # 9 EMA must be below 20 EMA (bearish)
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'dev-band-pop-backside-v1'})

    def fetch_daily_data_cached(self, symbol, start_date, end_date):
        """Fetch daily OHLCV data from Polygon"""
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

    def calculate_simple_indicators(self, df):
        """Calculate simple indicators for dev band analysis"""
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
        
        # Simple ATR
        df['atr'] = df['true_range'].rolling(14, min_periods=5).mean()
        
        # EMAs for trend context
        df['ema_9'] = df['close'].ewm(span=9, min_periods=5).mean()
        df['ema_20'] = df['close'].ewm(span=20, min_periods=10).mean()
        
        # Enhanced gap and extension calculations
        df['gap_over_close_dollars'] = df['open'] - df['pdc']
        df['gap_over_close_atr'] = df['gap_over_close_dollars'] / df['atr']
        
        # Gap over previous high
        df['prev_high'] = df['high'].shift(1)
        df['gap_over_high_dollars'] = df['open'] - df['prev_high']  
        df['gap_over_high_atr'] = df['gap_over_high_dollars'] / df['atr']
        
        # Extension from D0 open to D-1 low
        df['prev_low'] = df['low'].shift(1)
        df['extension_to_prev_low_dollars'] = df['open'] - df['prev_low']
        df['extension_to_prev_low_atr'] = df['extension_to_prev_low_dollars'] / df['atr']
        
        df['avg_volume_20'] = df['volume'].rolling(20, min_periods=5).mean()
        df['volume_multiple'] = df['volume'] / df['avg_volume_20']
        
        # Dollar volume for liquidity
        df['dollar_volume'] = df['close'] * df['volume']
        df['avg_dollar_volume_20d'] = df['dollar_volume'].rolling(20, min_periods=5).mean()
        
        # Dev band calculation with 0.4 multiplier
        df['ATR_9'] = df['true_range'].rolling(9, min_periods=3).mean()
        df['dev_band_upper_04'] = df['ema_9'] + self.scan_thresholds['dev_band_multiplier'] * df['ATR_9']
        
        # Track position relative to dev band
        df['open_above_dev_04'] = df['open'] > df['dev_band_upper_04']
        df['high_above_dev_04'] = df['high'] > df['dev_band_upper_04']
        df['close_below_dev_04'] = df['close'] < df['dev_band_upper_04']
        
        # Backside EMA condition (9 below 20)
        df['ema_bearish'] = df['ema_9'] < df['ema_20']
        
        return df

    def check_backside_context(self, df, setup_idx):
        """Check if stock was recently on backside (below dev band)"""
        if setup_idx < self.scan_thresholds['lookback_days']:
            return False
        
        # Look back to see if stock was below dev band recently
        lookback_start = max(0, setup_idx - self.scan_thresholds['lookback_days'])
        recent_data = df.iloc[lookback_start:setup_idx]
        
        # Count days below dev band
        days_below_band = recent_data['close_below_dev_04'].sum()
        
        return days_below_band >= self.scan_thresholds['min_days_below_band']

    def scan_single_ticker(self, symbol, start_date, end_date):
        """Scan single ticker for dev band pop setups"""
        # Fetch data with extended lookback
        extended_start = (pd.to_datetime(start_date) - timedelta(days=100)).strftime('%Y-%m-%d')
        df = self.fetch_daily_data_cached(symbol, extended_start, end_date)
        
        if df.empty or len(df) < 30:
            return []
        
        # Calculate indicators
        df = self.calculate_simple_indicators(df)
        
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
            
            # SIMPLE FILTERS
            
            # 1. Basic price and volume filters
            if not (self.scan_thresholds['min_price'] <= d_0['close'] <= self.scan_thresholds['max_price']):
                continue
            if d_minus_1['volume'] < self.scan_thresholds['min_volume']:
                continue
            if pd.isna(d_minus_1['atr']) or d_minus_1['atr'] <= 0:
                continue
            
            # 2. Dollar volume liquidity filter
            d_minus_1_dollar_vol = d_minus_1['avg_dollar_volume_20d'] if not pd.isna(d_minus_1['avg_dollar_volume_20d']) else 0
            if d_minus_1_dollar_vol < self.scan_thresholds['min_dollar_volume_20d']:
                continue
            
            # 3. Volume multiple requirement
            volume_multiple = d_minus_1['volume_multiple'] if not pd.isna(d_minus_1['volume_multiple']) else 0
            if volume_multiple < self.scan_thresholds['min_volume_multiple']:
                continue
            
            # 4. Enhanced gap requirements - must gap over BOTH close AND high
            gap_over_close_atr = d_0['gap_over_close_atr'] if not pd.isna(d_0['gap_over_close_atr']) else 0
            gap_over_high_atr = d_0['gap_over_high_atr'] if not pd.isna(d_0['gap_over_high_atr']) else 0
            
            if gap_over_close_atr < self.scan_thresholds['min_gap_over_close_atr']:
                continue
            if gap_over_high_atr < self.scan_thresholds['min_gap_over_high_atr']:
                continue
            
            # 4b. Extension requirement - good extension from D0 open to D-1 low
            extension_to_low_atr = d_0['extension_to_prev_low_atr'] if not pd.isna(d_0['extension_to_prev_low_atr']) else 0
            if extension_to_low_atr < self.scan_thresholds['min_extension_to_low_atr']:
                continue
            
            # 5. Must be popping into 0.4 dev band
            d0_open_above_dev = d_0['open_above_dev_04'] if not pd.isna(d_0['open_above_dev_04']) else False
            d0_high_above_dev = d_0['high_above_dev_04'] if not pd.isna(d_0['high_above_dev_04']) else False
            
            if not (d0_open_above_dev or d0_high_above_dev):
                continue
            
            # 6. Must be on backside - 9 EMA below 20 EMA (bearish)
            ema_bearish = d_0['ema_bearish'] if not pd.isna(d_0['ema_bearish']) else False
            if not ema_bearish:
                continue
            
            # 7. Check backside context - must have been below band recently
            if not self.check_backside_context(df, idx):
                continue
            
            # Calculate setup metrics
            dev_band_value = d_0['dev_band_upper_04'] if not pd.isna(d_0['dev_band_upper_04']) else 0
            ema_9_value = d_0['ema_9'] if not pd.isna(d_0['ema_9']) else 0
            ema_20_value = d_0['ema_20'] if not pd.isna(d_0['ema_20']) else 0
            
            # EMA trend context (should always be bearish now)
            ema_trend = "bearish"  # Always bearish due to filter above
            
            setup = {
                'symbol': symbol,
                'date': current_date.strftime('%Y-%m-%d'),
                'gap_over_close_atr': round(gap_over_close_atr, 2),
                'gap_over_high_atr': round(gap_over_high_atr, 2),
                'extension_to_low_atr': round(extension_to_low_atr, 2),
                'volume_multiple': round(volume_multiple, 2),
                'd0_open': round(d_0['open'], 2),
                'd0_high': round(d_0['high'], 2),
                'd0_close': round(d_0['close'], 2),
                'd_minus_1_high': round(d_minus_1['high'], 2),
                'd_minus_1_low': round(d_minus_1['low'], 2),
                'd_minus_1_close': round(d_minus_1['close'], 2),
                'dev_band_04': round(dev_band_value, 2),
                'ema_9': round(ema_9_value, 2),
                'ema_20': round(ema_20_value, 2),
                'ema_trend': ema_trend,
                'dollar_volume_20d_m': round(d_minus_1_dollar_vol / 1_000_000, 1) if d_minus_1_dollar_vol > 0 else None,
                'scanner_version': 'DevBandPopBacksideV1_Enhanced'
            }
            setups.append(setup)
        
        return setups

    def run_scan(self, start_date=None, end_date=None, tickers=None):
        """Run dev band pop backside scan"""
        start_date = start_date or START_DATE
        end_date = end_date or END_DATE
        tickers = tickers or self.ticker_universe
        
        print(f"🚀 DEV BAND POP BACKSIDE SCANNER V1")
        print(f"📅 Scanning {len(tickers)} tickers from {start_date} to {end_date}")
        print(f"🔧 Features:")
        print(f"   • Simple scan for liquid names popping into 0.{int(self.scan_thresholds['dev_band_multiplier']*10)} upper dev bands")
        print(f"   • Must be on backside (9 EMA < 20 EMA) and have been below dev band recently")
        print(f"   • Recent backside context ({self.scan_thresholds['min_days_below_band']}+ days below band in last {self.scan_thresholds['lookback_days']} days)")
        print(f"   • Gap over close {self.scan_thresholds['min_gap_over_close_atr']}+ ATR and over high {self.scan_thresholds['min_gap_over_high_atr']}+ ATR")
        print(f"   • Extension from D0 open to D-1 low {self.scan_thresholds['min_extension_to_low_atr']}+ ATR")
        print(f"   • Volume {self.scan_thresholds['min_volume_multiple']}x+")
        print(f"   • ${self.scan_thresholds['min_dollar_volume_20d']/1_000_000:.0f}M minimum dollar volume")
        print("=" * 80)
        
        all_setups = []
        processed = 0
        errors = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.scan_single_ticker, ticker, start_date, end_date): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    setups = future.result()
                    if setups:
                        all_setups.extend(setups)
                        print(f"✅ {ticker}: {len(setups)} dev band pop setups found")
                    else:
                        print(f"⚪ {ticker}: No setups")
                except Exception as e:
                    print(f"❌ {ticker}: Error - {str(e)[:50]}")
                    errors += 1
                
                processed += 1
                if processed % 20 == 0:
                    print(f"📊 Progress: {processed}/{len(tickers)} ({processed/len(tickers)*100:.1f}%)")
        
        print("\n" + "=" * 80)
        print(f"🎯 DEV BAND POP BACKSIDE SCAN COMPLETE")
        print(f"📊 Results: {len(all_setups)} setups found")
        print(f"⚠️  Errors: {errors} tickers failed")
        
        if all_setups:
            df_results = pd.DataFrame(all_setups)
            df_results = df_results.sort_values(['date', 'gap_over_close_atr'], ascending=[False, False])
            
            print(f"\n🏆 ALL DEV BAND POP BACKSIDE SETUPS:")
            display_cols = ['symbol', 'date', 'gap_over_close_atr', 'gap_over_high_atr', 'extension_to_low_atr', 'volume_multiple', 'd0_open', 'dev_band_04']
            print(df_results[display_cols].to_string(index=False))
            
            # EMA trend breakdown
            trend_counts = df_results['ema_trend'].value_counts()
            print(f"\n📊 EMA TREND BREAKDOWN:")
            for trend, count in trend_counts.items():
                print(f"   {trend}: {count} setups")
            
            return df_results
        else:
            print("No dev band pop backside setups found.")
            return pd.DataFrame()

# Example Usage
if __name__ == "__main__":
    # Initialize scanner
    scanner = DevBandPopBacksideV1()
    
    # Run scan
    results = scanner.run_scan()
    
    if not results.empty:
        # Save results to CSV
        results.to_csv('dev_band_pop_backside_v1_results.csv', index=False)
        print(f"\n💾 Results saved to: dev_band_pop_backside_v1_results.csv")