#!/usr/bin/env python3
"""
OS D1 Full 2025 Scanner - All Valid Names
Scan every trading day in 2025 for small cap day one gappers to fade
"""

import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta
import pandas_market_calendars as mcal

class OS_D1_Full2025_Scanner:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.nyse = mcal.get_calendar('NYSE')
    
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
    
    def get_trading_days_2025(self):
        """Get all trading days in 2025"""
        start_date = '2025-01-01'
        end_date = '2025-12-31'
        
        schedule = self.nyse.schedule(start_date=start_date, end_date=end_date)
        trading_days = [date.strftime('%Y-%m-%d') for date in schedule.index]
        
        return trading_days
    
    async def scan_single_date(self, session, scan_date, prev_date):
        """Scan a single date for OS D1 setups"""
        
        try:
            # Fetch data for both days
            current_data = await self.fetch_grouped_daily_data(session, scan_date)
            prev_data = await self.fetch_grouped_daily_data(session, prev_date)
            
            if current_data.empty or prev_data.empty:
                return pd.DataFrame()
            
            # Merge data
            merged = current_data.merge(prev_data, on='ticker', suffixes=('', '_prev'))
            
            # Apply your exact trigger day criteria
            merged['pm_high'] = merged['h']  # Using high as pm_high
            merged['pm_vol'] = merged['v']   # Using volume as pm_vol  
            merged['prev_close'] = merged['c_prev']
            merged['prev_high'] = merged['h_prev']
            merged['gap'] = merged['o'] - merged['prev_close']
            
            # Your exact OS D1 formula
            merged['trig_day'] = (
                (merged['pm_high'] / merged['prev_close'] - 1 >= 0.5) &
                (merged['gap'] >= 0.5) & 
                (merged['o'] / merged['prev_high'] - 1 >= 0.3) &
                (merged['pm_vol'] >= 5000000) &
                (merged['prev_close'] >= 0.75)
            ).astype(int)
            
            # Get trigger day candidates
            results = merged[merged['trig_day'] == 1].copy()
            
            if results.empty:
                return pd.DataFrame()
            
            # Add calculated fields
            results['scan_date'] = scan_date
            results['gap_pct'] = (results['gap'] / results['prev_close']) * 100
            results['pm_high_pct'] = (results['pm_high'] / results['prev_close'] - 1) * 100
            results['open_prev_high_pct'] = (results['o'] / results['prev_high'] - 1) * 100
            
            # Select relevant columns
            final_results = results[[
                'ticker', 'scan_date', 'c', 'gap', 'gap_pct', 'pm_high_pct', 
                'pm_vol', 'prev_close', 'prev_high', 'o', 'h', 'l'
            ]].copy()
            
            return final_results
            
        except Exception as e:
            print(f"❌ Error scanning {scan_date}: {e}")
            return pd.DataFrame()
    
    async def scan_all_2025(self):
        """Scan all trading days in 2025 for OS D1 setups"""
        
        print("🚀 OS D1 Full 2025 Scanner")
        print("Scanning ALL trading days in 2025 for small cap day one gappers")
        print("=" * 80)
        
        # Get all trading days
        trading_days = self.get_trading_days_2025()
        print(f"📅 Total trading days in 2025: {len(trading_days)}")
        
        all_results = []
        processed_days = 0
        days_with_setups = 0
        
        # Create a single session for all requests
        async with aiohttp.ClientSession() as session:
            
            # Process in batches to avoid overwhelming the API
            batch_size = 10  # Process 10 days at a time
            
            for i in range(0, len(trading_days), batch_size):
                batch = trading_days[i:i+batch_size]
                batch_tasks = []
                
                print(f"\n📊 Processing batch {i//batch_size + 1}/{(len(trading_days)-1)//batch_size + 1}")
                print(f"   Days {i+1}-{min(i+batch_size, len(trading_days))}: {batch[0]} to {batch[-1]}")
                
                for j, scan_date in enumerate(batch):
                    # Get previous trading day
                    if i + j > 0:
                        prev_date = trading_days[i + j - 1]
                    else:
                        # For first day, get previous trading day from 2024
                        prev_date_dt = pd.to_datetime(scan_date) - pd.Timedelta(days=1)
                        while prev_date_dt.weekday() >= 5:  # Weekend
                            prev_date_dt -= pd.Timedelta(days=1)
                        prev_date = prev_date_dt.strftime('%Y-%m-%d')
                    
                    # Create task for this date
                    task = self.scan_single_date(session, scan_date, prev_date)
                    batch_tasks.append(task)
                
                # Execute batch
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process batch results
                for j, result in enumerate(batch_results):
                    scan_date = batch[j]
                    processed_days += 1
                    
                    if isinstance(result, Exception):
                        print(f"   ❌ {scan_date}: {result}")
                        continue
                    
                    if not result.empty:
                        days_with_setups += 1
                        setup_count = len(result)
                        all_results.append(result)
                        
                        print(f"   ✅ {scan_date}: {setup_count} setups")
                        
                        # Show top setups for this day
                        top_setups = result.nlargest(3, 'pm_high_pct')
                        for _, row in top_setups.iterrows():
                            print(f"      📊 {row['ticker']}: Gap {row['gap_pct']:.1f}%, PM High {row['pm_high_pct']:.1f}%")
                    else:
                        print(f"   ℹ️ {scan_date}: 0 setups")
                
                # Add small delay between batches
                await asyncio.sleep(0.5)
        
        # Compile final results
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            final_df = final_df.sort_values(['scan_date', 'pm_high_pct'], ascending=[True, False])
            
            print(f"\n{'='*80}")
            print(f"🎯 2025 OS D1 SMALL CAP DAY ONE GAPPER RESULTS")
            print(f"{'='*80}")
            print(f"📅 Trading days processed: {processed_days}")
            print(f"📈 Days with setups: {days_with_setups}")
            print(f"🎯 Total OS D1 setups found: {len(final_df)}")
            print(f"📊 Average setups per setup day: {len(final_df) / days_with_setups:.1f}")
            print(f"🔥 Setup frequency: {days_with_setups / processed_days:.1%}")
            
            # Show top 20 setups by PM High %
            print(f"\n🏆 TOP 20 BIGGEST GAP UPS (Fade Candidates):")
            print(f"{'Date':<12} {'Ticker':<8} {'Close':<8} {'Gap %':<8} {'PM High %':<12} {'Volume':<12}")
            print("=" * 85)
            
            top_20 = final_df.nlargest(20, 'pm_high_pct')
            for _, row in top_20.iterrows():
                print(f"{row['scan_date']:<12} {row['ticker']:<8} ${row['c']:<7.2f} {row['gap_pct']:<7.1f}% {row['pm_high_pct']:<11.1f}% {row['pm_vol']:<12,.0f}")
            
            # Monthly breakdown
            print(f"\n📅 2025 MONTHLY BREAKDOWN:")
            final_df['month'] = pd.to_datetime(final_df['scan_date']).dt.strftime('%Y-%m')
            monthly_counts = final_df.groupby('month').size().sort_index()
            
            for month, count in monthly_counts.items():
                month_name = pd.to_datetime(month).strftime('%B %Y')
                print(f"   • {month_name}: {count} setups")
            
            # Export complete results
            output_file = "os_d1_all_2025_setups.csv"
            final_df.to_csv(output_file, index=False)
            print(f"\n💾 Complete results saved to: {output_file}")
            
            # Show stats
            avg_gap = final_df['gap_pct'].mean()
            avg_pm_high = final_df['pm_high_pct'].mean()
            avg_volume = final_df['pm_vol'].mean()
            
            print(f"\n📊 2025 SMALL CAP GAPPER STATISTICS:")
            print(f"   • Average gap: {avg_gap:.1f}%")
            print(f"   • Average PM high: {avg_pm_high:.1f}%") 
            print(f"   • Average volume: {avg_volume:,.0f}")
            
            return final_df
        else:
            print(f"\nℹ️ No OS D1 setups found in 2025")
            return pd.DataFrame()

async def main():
    """Run the full 2025 OS D1 scanner"""
    
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    scanner = OS_D1_Full2025_Scanner(api_key)
    
    results = await scanner.scan_all_2025()
    
    if not results.empty:
        print(f"\n✅ Found {len(results)} small cap day one gappers in 2025!")
        print(f"These are your fade candidates - small caps that gapped up on day one hype")
    else:
        print(f"\n❌ No setups found")

if __name__ == '__main__':
    asyncio.run(main())