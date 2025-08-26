#!/usr/bin/env python3
"""
OS D1 Scanner without EMA validation - just trigger day criteria
To see all candidates that meet your exact trigger day parameters
"""

import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta

class OS_D1_NoEMA_Scanner:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
    
    async def fetch_grouped_daily_data(self, session, date, adjusted=True):
        """Fetch grouped daily data for all stocks"""
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
    
    async def scan_trigger_day_only(self, scan_date):
        """Find all tickers meeting your exact trigger day criteria"""
        
        print(f"🚀 OS D1 Trigger Day Scanner - {scan_date}")
        print("Your exact criteria: pm_high/prev_close-1≥50%, gap≥$0.5, open/prev_high-1≥30%, vol≥5M, prev_close≥$0.75")
        
        # Get previous trading day
        prev_date = pd.to_datetime(scan_date) - pd.Timedelta(days=1)
        while prev_date.weekday() >= 5:
            prev_date -= pd.Timedelta(days=1)
        prev_date_str = prev_date.strftime('%Y-%m-%d')
        
        try:
            # Fetch all tickers
            async with aiohttp.ClientSession() as session:
                current_data = await self.fetch_grouped_daily_data(session, scan_date)
                prev_data = await self.fetch_grouped_daily_data(session, prev_date_str)
            
            if current_data.empty or prev_data.empty:
                print("❌ No data available")
                return pd.DataFrame()
            
            # Merge data
            merged = current_data.merge(prev_data, on='ticker', suffixes=('', '_prev'))
            print(f"📊 Processing {len(merged):,} tickers")
            
            # Apply your exact trigger day criteria
            merged['pm_high'] = merged['h']  # Using high as pm_high
            merged['pm_vol'] = merged['v']   # Using volume as pm_vol  
            merged['prev_close'] = merged['c_prev']
            merged['prev_high'] = merged['h_prev']
            merged['gap'] = merged['o'] - merged['prev_close']
            
            # Your exact formula
            merged['trig_day'] = (
                (merged['pm_high'] / merged['prev_close'] - 1 >= 0.5) &
                (merged['gap'] >= 0.5) & 
                (merged['o'] / merged['prev_high'] - 1 >= 0.3) &
                (merged['pm_vol'] >= 5000000) &
                (merged['prev_close'] >= 0.75)
            ).astype(int)
            
            # Get all trigger day candidates
            results = merged[merged['trig_day'] == 1].copy()
            
            if results.empty:
                print("ℹ️ No trigger day candidates found")
                return pd.DataFrame()
            
            # Add calculated fields
            results['gap_pct'] = (results['gap'] / results['prev_close']) * 100
            results['pm_high_pct'] = (results['pm_high'] / results['prev_close'] - 1) * 100
            results['open_prev_high_pct'] = (results['o'] / results['prev_high'] - 1) * 100
            
            # Sort by PM high ratio (most impressive first)
            results = results.sort_values('pm_high_pct', ascending=False)
            
            print(f"🎯 Found {len(results)} trigger day setups (NO EMA filtering):")
            print(f"{'Ticker':<8} {'Close':<8} {'Gap $':<8} {'Gap %':<8} {'PM High %':<12} {'Volume':<12}")
            print("=" * 80)
            
            for _, row in results.iterrows():
                print(f"{row['ticker']:<8} ${row['c']:<7.2f} ${row['gap']:<7.2f} {row['gap_pct']:<7.1f}% {row['pm_high_pct']:<11.1f}% {row['pm_vol']:<12,.0f}")
            
            # Check if CURR is in the results
            curr_found = results[results['ticker'] == 'CURR']
            if not curr_found.empty:
                print(f"\n✅ CURR found in results! This confirms the trigger day criteria are working.")
            else:
                print(f"\n⚠️ CURR not found. Let's check why...")
            
            return results
            
        except Exception as e:
            print(f"❌ Scan failed: {e}")
            return pd.DataFrame()

async def main():
    """Test the no-EMA scanner"""
    
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    scanner = OS_D1_NoEMA_Scanner(api_key)
    
    # Test with CURR date
    scan_date = '2025-01-08'
    
    results = await scanner.scan_trigger_day_only(scan_date)
    
    if not results.empty:
        output_file = f"trigger_day_candidates_{scan_date}.csv"
        results.to_csv(output_file, index=False)
        print(f"\n💾 All {len(results)} candidates saved to: {output_file}")
    
if __name__ == '__main__':
    asyncio.run(main())