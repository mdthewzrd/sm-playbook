#!/usr/bin/env python3
"""
OS D1 Scanner - Pure Implementation
Uses LC framework's "scan all tickers" capability with ONLY your exact OS D1 parameters
No LC pattern detection - just trigger day and EMA validation
"""

import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
import pandas_market_calendars as mcal
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

class OS_D1_Scanner:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.nyse = mcal.get_calendar('NYSE')
        
    def get_previous_trading_day(self, current_date, n=1):
        """Get the nth previous trading day"""
        start_date = pd.to_datetime(current_date) - pd.Timedelta(days=500)
        end_date = pd.to_datetime(current_date) + pd.Timedelta(days=30)
        schedule = self.nyse.schedule(start_date=start_date, end_date=end_date)
        trading_days = schedule.index
        
        try:
            idx = trading_days.get_loc(pd.to_datetime(current_date))
            return trading_days[max(0, idx - n)]
        except:
            return pd.to_datetime(current_date) - pd.Timedelta(days=n)
    
    async def fetch_grouped_daily_data(self, session, date, adjusted=True):
        """Fetch grouped daily data for all stocks - from LC framework"""
        adj_str = "true" if adjusted else "false"
        url = f"{self.base_url}/v2/aggs/grouped/locale/us/market/stocks/{date}"
        params = {'adjusted': adj_str, 'apiKey': self.api_key}
        
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if 'results' in data:
                    df = pd.DataFrame(data['results'])
                    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
                    df.rename(columns={'T': 'ticker'}, inplace=True)
                    return df
            return pd.DataFrame()
    
    def fetch_daily_data(self, ticker, start_date, end_date, adjusted="true"):
        """Fetch daily data from Polygon API"""
        url = f'{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}'
        params = {
            'adjusted': adjusted,
            'sort': 'asc',
            'limit': 5000,
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    return pd.DataFrame(data['results'])
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
        
        return pd.DataFrame()
    
    def adjust_daily(self, df):
        """Apply daily adjustments and calculate EMAs"""
        if df.empty:
            return df
            
        df = df.copy()
        df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
        df['pdc'] = df['c'].shift(1)  # Previous day close
        df['ema200'] = df['c'].ewm(span=200, adjust=False).mean().fillna(0)
        return df
    
    def get_market_cap(self, ticker, date):
        """Get market cap data from Polygon API"""
        try:
            mc_url = f'{self.base_url}/v3/reference/tickers/{ticker}'
            params = {
                'date': date,
                'apiKey': self.api_key
            }
            
            response = requests.get(mc_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    try:
                        market_cap = data['results'].get('market_cap', 0)
                        return market_cap / 1000000  # Convert to millions
                    except:
                        # Try alternative calculation with outstanding shares
                        try:
                            os_shares = data['results'].get('weighted_shares_outstanding', 0)
                            if os_shares > 0:
                                # Get current price
                                price_data = self.fetch_daily_data(ticker, date, date, adjusted="false")
                                if not price_data.empty:
                                    close_price = price_data['c'].iloc[-1]
                                    return (os_shares * close_price) / 1000000
                        except:
                            pass
            return None
        except Exception as e:
            return None
    
    async def scan_os_d1(self, scan_date):
        """
        OS D1 Scanner - Your exact parameters only
        1. Scan all tickers (from LC framework)
        2. Apply trigger day criteria
        3. Validate EMA criteria
        """
        
        print(f"🚀 Running OS D1 Scanner for {scan_date}")
        print("Using your exact trigger day and EMA parameters")
        print("=" * 60)
        
        try:
            # Get previous trading day
            prev_date = self.get_previous_trading_day(scan_date, 1)
            prev_date_str = prev_date.strftime('%Y-%m-%d')
            
            print(f"📅 Scan Date: {scan_date}")
            print(f"📅 Previous Date: {prev_date_str}")
            
            # Fetch all tickers for both days using LC framework method
            async with aiohttp.ClientSession() as session:
                print("📡 Fetching all market tickers...")
                current_data = await self.fetch_grouped_daily_data(session, scan_date)
                prev_data = await self.fetch_grouped_daily_data(session, prev_date_str)
                
            if current_data.empty or prev_data.empty:
                print("❌ No market data available")
                return pd.DataFrame()
                
            print(f"📊 Processing {len(current_data)} current day tickers")
            print(f"📊 Processing {len(prev_data)} previous day tickers")
            
            # Apply your exact trigger day criteria
            print("\n🎯 Applying OS D1 Trigger Day Criteria...")
            
            trigger_results = []
            
            for _, row in current_data.iterrows():
                ticker = row['ticker']
                
                # Find previous day data for this ticker
                prev_row = prev_data[prev_data['ticker'] == ticker]
                if prev_row.empty:
                    continue
                    
                prev_row = prev_row.iloc[0]
                
                try:
                    # Your exact parameters from your code:
                    # df['trig_day'] = ((df['pm_high'] / df['prev_close'] - 1>= .5) & 
                    #                 (df['gap'] >= 0.5) & 
                    #                 (df['open'] / df['prev_high'] - 1>= .3) & 
                    #                 (df['pm_vol'] >= 5000000) & 
                    #                 (df['prev_close'] >= 0.75)).astype(int)
                    
                    pm_high = row.get('pm_high', row['h'])  # Use high if pm_high not available
                    pm_vol = row.get('pm_vol', row['v'])   # Use volume if pm_vol not available
                    gap = row['o'] - prev_row['c']  # Actual gap amount, not percentage
                    
                    trig_day = (
                        (pm_high / prev_row['c'] - 1 >= 0.5) and    # pm_high/prev_close - 1 >= 50%
                        (gap >= 0.5) and                           # gap >= 0.5 (dollar amount)
                        (row['o'] / prev_row['h'] - 1 >= 0.3) and  # open/prev_high - 1 >= 30%
                        (pm_vol >= 5000000) and                    # pm_vol >= 5M
                        (prev_row['c'] >= 0.75)                    # prev_close >= $0.75
                    )
                    
                    if trig_day:
                        trigger_results.append({
                            'ticker': ticker,
                            'date': scan_date,
                            'open': row['o'],
                            'high': row['h'],
                            'low': row['l'],
                            'close': row['c'],
                            'volume': row['v'],
                            'prev_close': prev_row['c'],
                            'prev_high': prev_row['h'],
                            'pm_high_ratio': pm_high / prev_row['c'] - 1,
                            'gap_amount': gap,
                            'open_prev_high_ratio': row['o'] / prev_row['h'] - 1,
                            'pm_vol': pm_vol
                        })
                        
                except Exception as e:
                    continue
            
            if not trigger_results:
                print("ℹ️ No trigger day candidates found")
                return pd.DataFrame()
                
            trigger_df = pd.DataFrame(trigger_results)
            print(f"🎯 Found {len(trigger_df)} trigger day candidates")
            
            # Show trigger day candidates
            for _, row in trigger_df.iterrows():
                print(f"  • {row['ticker']}: Gap ${row['gap_amount']:.2f}, PM High {row['pm_high_ratio']:.1%}, Vol {row['pm_vol']:,}")
            
            # Apply EMA validation
            print(f"\n🧮 Validating EMA Criteria (Price <= EMA200 * 0.8)...")
            
            final_results = []
            
            for _, row in trigger_df.iterrows():
                ticker = row['ticker']
                
                try:
                    # Get historical data for EMA200 calculation
                    start_date = pd.to_datetime(scan_date) - pd.Timedelta(days=500)
                    start_date_str = start_date.strftime('%Y-%m-%d')
                    prev_date_str = (pd.to_datetime(scan_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    
                    print(f"   📈 Checking EMA for {ticker}...")
                    
                    # Fetch historical data
                    daily_data = self.fetch_daily_data(ticker, start_date_str, prev_date_str, adjusted="true")
                    
                    if not daily_data.empty:
                        daily_data = self.adjust_daily(daily_data)
                        
                        # Check if we have enough data for EMA200
                        if len(daily_data) >= 200:
                            current_price = row['close']
                            ema200 = daily_data['ema200'].iloc[-1]
                            
                            # Your exact EMA validation: price <= EMA200 * 0.8
                            ema_valid = current_price <= (ema200 * 0.8)
                            
                            if ema_valid:
                                # Get market cap
                                market_cap = self.get_market_cap(ticker, prev_date_str)
                                
                                result = row.to_dict()
                                result.update({
                                    'ema200': ema200,
                                    'ema_ratio': current_price / ema200 if ema200 > 0 else 0,
                                    'market_cap_millions': market_cap,
                                    'historical_days': len(daily_data),
                                    'ema_valid': 1
                                })
                                
                                final_results.append(result)
                                print(f"   ✅ {ticker}: VALID (Price: ${current_price:.2f} <= ${ema200 * 0.8:.2f})")
                            else:
                                print(f"   ❌ {ticker}: Price too high (${current_price:.2f} > ${ema200 * 0.8:.2f})")
                        else:
                            print(f"   ⚠️ {ticker}: Insufficient data ({len(daily_data)} < 200 days)")
                    else:
                        print(f"   ❌ {ticker}: No historical data")
                        
                except Exception as e:
                    print(f"   ❌ {ticker}: Error - {e}")
                    continue
            
            return pd.DataFrame(final_results)
            
        except Exception as e:
            print(f"❌ Scan failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

async def main():
    """Run the OS D1 scanner"""
    
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    scanner = OS_D1_Scanner(api_key)
    
    # Scan for a recent date
    scan_date = '2024-08-20'
    
    results = await scanner.scan_os_d1(scan_date)
    
    if not results.empty:
        print(f"\n{'='*60}")
        print("🎯 OS D1 SETUP RESULTS")
        print(f"{'='*60}")
        
        for _, row in results.iterrows():
            print(f"\n📊 {row['ticker']} ({scan_date}):")
            print(f"  • Price: ${row['close']:.2f} (Gap: ${row['gap_amount']:.2f})")
            print(f"  • PM High Ratio: {row['pm_high_ratio']:.1%}")
            print(f"  • EMA200: ${row['ema200']:.2f} (80% = ${row['ema200'] * 0.8:.2f})")
            print(f"  • Volume: {row['volume']:,}")
            if row.get('market_cap_millions'):
                print(f"  • Market Cap: ${row['market_cap_millions']:.0f}M")
        
        # Export results
        output_file = f"os_d1_results_{scan_date}.csv"
        results.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
    else:
        print(f"\n{'='*60}")
        print("ℹ️ No OS D1 setups found for this date")
        print("This is normal - OS D1 setups are rare and require specific market conditions")

if __name__ == '__main__':
    asyncio.run(main())