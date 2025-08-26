#!/usr/bin/env python3
"""
Run OS D1 Scanner across date range: 8/25/24 - 1/1/25
"""

import pandas as pd
import asyncio
from datetime import datetime, timedelta
import sys
import os

# Add the scanner directory to the path
sys.path.append(os.path.dirname(__file__))

from os_d1_pure_scanner import OS_D1_Scanner

async def run_os_d1_date_range():
    """Run OS D1 scanner across the specified date range"""
    
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    scanner = OS_D1_Scanner(api_key)
    
    # Date range: 8/25/24 to 1/1/25
    start_date = datetime(2024, 8, 25)
    end_date = datetime(2025, 1, 1)
    
    print(f"🚀 Running OS D1 Scanner for Date Range")
    print(f"📅 From: {start_date.strftime('%Y-%m-%d')}")
    print(f"📅 To: {end_date.strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    all_results = []
    total_days = 0
    days_with_setups = 0
    
    # Generate list of trading days to scan
    current_date = start_date
    scan_dates = []
    
    while current_date <= end_date:
        # Only scan weekdays (Monday=0, Sunday=6)
        if current_date.weekday() < 5:  # Monday to Friday
            scan_dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    print(f"📊 Total trading days to scan: {len(scan_dates)}")
    print(f"🔍 Starting scan...\n")
    
    for i, scan_date in enumerate(scan_dates):
        print(f"📅 Day {i+1}/{len(scan_dates)}: Scanning {scan_date}")
        
        try:
            # Run the scanner for this date
            results = await scanner.scan_os_d1(scan_date)
            
            if not results.empty:
                days_with_setups += 1
                print(f"   ✅ Found {len(results)} OS D1 setup(s)")
                
                # Add scan date to results
                results['scan_date'] = scan_date
                all_results.append(results)
                
                # Show the setups found
                for _, row in results.iterrows():
                    print(f"      📊 {row['ticker']}: ${row['close']:.2f}, Gap: ${row['gap_amount']:.2f}")
            else:
                print(f"   ℹ️ No setups found")
            
            total_days += 1
            
        except Exception as e:
            print(f"   ❌ Error scanning {scan_date}: {e}")
            continue
    
    # Compile final results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        print(f"\n{'='*60}")
        print(f"🎯 OS D1 SCANNER RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"📊 Total Days Scanned: {total_days}")
        print(f"🎯 Days with OS D1 Setups: {days_with_setups}")
        print(f"📈 Total OS D1 Setups Found: {len(final_df)}")
        
        if len(final_df) > 0:
            print(f"\n📊 Setup Statistics:")
            print(f"   • Average setups per setup day: {len(final_df) / days_with_setups:.1f}")
            print(f"   • Setup frequency: {days_with_setups / total_days:.1%}")
            
            # Show price and volume statistics
            avg_price = final_df['close'].mean()
            avg_gap = final_df['gap_amount'].mean()
            avg_volume = final_df['volume'].mean()
            
            print(f"   • Average setup price: ${avg_price:.2f}")
            print(f"   • Average gap amount: ${avg_gap:.2f}")
            print(f"   • Average volume: {avg_volume:,.0f}")
            
            # Show top setups by gap size
            print(f"\n🏆 Top 10 Setups by Gap Amount:")
            top_setups = final_df.nlargest(10, 'gap_amount')
            
            for _, row in top_setups.iterrows():
                print(f"   • {row['ticker']} ({row['scan_date']}): ${row['close']:.2f}, Gap ${row['gap_amount']:.2f}")
            
            # Show monthly breakdown
            print(f"\n📅 Monthly Breakdown:")
            final_df['month'] = pd.to_datetime(final_df['scan_date']).dt.strftime('%Y-%m')
            monthly_counts = final_df.groupby('month').size()
            
            for month, count in monthly_counts.items():
                print(f"   • {month}: {count} setups")
            
            # Export results
            output_file = f"os_d1_results_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            final_df.to_csv(output_file, index=False)
            print(f"\n💾 Complete results saved to: {output_file}")
            
            return final_df
        
    else:
        print(f"\nℹ️ No OS D1 setups found in the entire date range")
        print(f"This indicates your criteria are working correctly - OS D1 setups are rare!")
        
    return pd.DataFrame()

if __name__ == '__main__':
    asyncio.run(run_os_d1_date_range())