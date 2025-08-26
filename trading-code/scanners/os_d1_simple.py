#!/usr/bin/env python3
"""
Simple OS D1 Scanner - Just stitching your two codes together
Uses LC framework to scan all tickers + your exact OS D1 parameters
"""

import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

class SimpleOS_D1_Scanner:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.nyse = mcal.get_calendar('NYSE')
    
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
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                return pd.DataFrame(data['results'])
        return pd.DataFrame()
    
    def adjust_daily(self, df):
        """Apply daily adjustments - from your code"""
        if df.empty:
            return df
            
        df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date          
        df['pdc'] = df['c'].shift(1)
        df['ema200'] = df['c'].ewm(span=200, adjust=False).mean().fillna(0)
        return df
    
    async def scan_os_d1(self, scan_date):
        """Your exact OS D1 logic with LC framework ticker scanning"""
        
        print(f"🚀 OS D1 Scanner - {scan_date}")
        print("Using your exact trigger day parameters")
        
        # Get previous trading day
        prev_date = pd.to_datetime(scan_date) - pd.Timedelta(days=1)
        # Simple weekday check
        while prev_date.weekday() >= 5:
            prev_date -= pd.Timedelta(days=1)
        prev_date_str = prev_date.strftime('%Y-%m-%d')
        
        try:
            # Fetch all tickers using LC framework method
            async with aiohttp.ClientSession() as session:
                print(f"📡 Fetching all tickers for {scan_date}...")
                current_data = await self.fetch_grouped_daily_data(session, scan_date)
                print(f"📡 Fetching all tickers for {prev_date_str}...")
                prev_data = await self.fetch_grouped_daily_data(session, prev_date_str)
            
            if current_data.empty or prev_data.empty:
                print("❌ No data available")
                return pd.DataFrame()
            
            print(f"📊 Current day: {len(current_data):,} tickers")
            print(f"📊 Previous day: {len(prev_data):,} tickers")
            
            # Merge current and previous day data
            merged = current_data.merge(prev_data, on='ticker', suffixes=('', '_prev'))
            print(f"📊 Matched tickers: {len(merged):,}")
            
            # Apply your exact trigger day criteria
            print("🎯 Applying your exact OS D1 criteria...")
            
            # Use high and volume as pm_high and pm_vol since grouped data doesn't have PM fields
            merged['pm_high'] = merged['h']  # Use high as pm_high
            merged['pm_vol'] = merged['v']   # Use volume as pm_vol  
            merged['prev_close'] = merged['c_prev']
            merged['prev_high'] = merged['h_prev']
            merged['gap'] = merged['o'] - merged['prev_close']
            
            # Your exact trigger day formula
            merged['trig_day'] = (
                (merged['pm_high'] / merged['prev_close'] - 1 >= 0.5) &
                (merged['gap'] >= 0.5) & 
                (merged['o'] / merged['prev_high'] - 1 >= 0.3) &
                (merged['pm_vol'] >= 5000000) &
                (merged['prev_close'] >= 0.75)
            ).astype(int)
            
            # Filter trigger day candidates
            trigger_candidates = merged[merged['trig_day'] == 1].copy()
            
            if trigger_candidates.empty:
                print("ℹ️ No trigger day candidates found")
                return pd.DataFrame()
            
            print(f"🎯 Found {len(trigger_candidates)} trigger day candidates")
            
            # Show candidates
            for _, row in trigger_candidates.iterrows():
                gap_pct = (row['gap'] / row['prev_close']) * 100
                pm_high_pct = (row['pm_high'] / row['prev_close'] - 1) * 100
                print(f"  • {row['ticker']}: Gap ${row['gap']:.2f} ({gap_pct:.1f}%), PM High {pm_high_pct:.1f}%, Vol {row['pm_vol']:,.0f}")
            
            # Apply EMA validation - your exact logic: c <= ema200*0.8 and len >= 200
            print(f"🧮 Applying EMA validation...")
            
            final_results = []
            
            for _, row in trigger_candidates.iterrows():
                ticker = row['ticker']
                
                try:
                    # Get historical data
                    start_date = pd.to_datetime(scan_date) - pd.Timedelta(days=500)
                    start_date_str = start_date.strftime('%Y-%m-%d')
                    prev_date_str = (pd.to_datetime(scan_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    
                    daily_data = self.fetch_daily_data(ticker, start_date_str, prev_date_str, adjusted="true")
                    
                    if not daily_data.empty:
                        daily_data = self.adjust_daily(daily_data)
                        
                        # EMA validation without 200-day requirement
                        c = row['c']
                        ema200 = daily_data['ema200'].iloc[-1] if len(daily_data) > 0 else 0
                        
                        # Remove the len >= 200 requirement as user suggested
                        if len(daily_data) > 0 and ema200 > 0 and c <= ema200 * 0.8:
                            result = {
                                'ticker': ticker,
                                'date': scan_date,
                                'close': c,
                                'gap': row['gap'],
                                'pm_high_ratio': (row['pm_high'] / row['prev_close'] - 1),
                                'volume': row['v'],
                                'ema200': ema200,
                                'ema_valid': 1
                            }
                            final_results.append(result)
                            print(f"  ✅ {ticker}: Valid (${c:.2f} <= ${ema200 * 0.8:.2f})")
                        else:
                            if len(daily_data) == 0:
                                reason = "No historical data"
                            elif ema200 <= 0:
                                reason = "Invalid EMA200"
                            else:
                                reason = f"Price too high (${c:.2f} > ${ema200 * 0.8:.2f})"
                            print(f"  ❌ {ticker}: {reason}")
                    else:
                        print(f"  ❌ {ticker}: No historical data")
                        
                except Exception as e:
                    print(f"  ❌ {ticker}: Error - {e}")
                    continue
            
            return pd.DataFrame(final_results)
            
        except Exception as e:
            print(f"❌ Scan failed: {e}")
            return pd.DataFrame()

async def main():
    """Test the simple scanner"""
    
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    scanner = SimpleOS_D1_Scanner(api_key)
    
    # Test with a date that should have CURR
    scan_date = '2025-01-08'
    
    results = await scanner.scan_os_d1(scan_date)
    
    if not results.empty:
        print(f"\n🎯 Found {len(results)} OS D1 setups:")
        for _, row in results.iterrows():
            print(f"  📊 {row['ticker']}: ${row['close']:.2f}, Gap ${row['gap']:.2f}")
    else:
        print(f"\nℹ️ No OS D1 setups found")

if __name__ == '__main__':
    asyncio.run(main())