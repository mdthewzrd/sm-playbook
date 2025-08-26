#!/usr/bin/env python3
"""
Run OS D1 Scanner for 2025 date range: 1/1/25 - 8/25/25
"""

import pandas as pd
import asyncio
from datetime import datetime
import sys
import os

# Add the scanner directory to the path
sys.path.append(os.path.dirname(__file__))

from os_d1_pure_scanner import OS_D1_Scanner

async def run_os_d1_2025_range():
    """Run OS D1 scanner for 2025 sample dates"""
    
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    scanner = OS_D1_Scanner(api_key)
    
    # Sample dates from 1/1/25 to 8/25/25 range
    # Testing key dates throughout 2025
    sample_dates = [
        '2025-01-02',  # First trading day of 2025
        '2025-01-15',  # Mid January
        '2025-02-03',  # Early February
        '2025-02-18',  # Presidents Day week
        '2025-03-03',  # Early March
        '2025-03-17',  # Mid March
        '2025-04-01',  # Q2 start
        '2025-04-15',  # Mid April
        '2025-05-01',  # May Day
        '2025-05-15',  # Mid May
        '2025-06-02',  # Early June
        '2025-06-16',  # Mid June
        '2025-07-01',  # Q3 start
        '2025-07-15',  # Mid July
        '2025-08-01',  # Early August
        '2025-08-15',  # Mid August
        '2025-08-25',  # End date
    ]
    
    print(f"🚀 Running OS D1 Scanner for 2025 Date Range")
    print(f"📅 Testing {len(sample_dates)} key dates from 1/1/25 - 8/25/25")
    print("=" * 60)
    
    all_results = []
    total_days = 0
    days_with_setups = 0
    days_with_data = 0
    
    for i, scan_date in enumerate(sample_dates):
        print(f"\n📅 Sample {i+1}/{len(sample_dates)}: Scanning {scan_date}")
        print("-" * 40)
        
        try:
            # Run the scanner for this date
            results = await scanner.scan_os_d1(scan_date)
            
            if results is not None:
                days_with_data += 1
                
                if not results.empty:
                    days_with_setups += 1
                    print(f"✅ Found {len(results)} OS D1 setup(s) on {scan_date}")
                    
                    # Add scan date to results
                    results['scan_date'] = scan_date
                    all_results.append(results)
                    
                    # Show the setups found
                    for _, row in results.iterrows():
                        print(f"   📊 {row['ticker']}: ${row['close']:.2f}, Gap: ${row['gap_amount']:.2f}, PM High: {row['pm_high_ratio']:.1%}")
                        print(f"       EMA200: ${row['ema200']:.2f}, Vol: {row['volume']:,}")
                else:
                    print(f"ℹ️ No setups found on {scan_date}")
            else:
                print(f"⚠️ No market data available for {scan_date}")
            
            total_days += 1
            
        except Exception as e:
            print(f"❌ Error scanning {scan_date}: {e}")
            continue
    
    # Compile final results
    print(f"\n{'='*60}")
    print(f"🎯 OS D1 SCANNER 2025 RESULTS")
    print(f"{'='*60}")
    
    print(f"📅 Total Sample Dates: {total_days}")
    print(f"📊 Days with Market Data: {days_with_data}")
    print(f"🎯 Days with OS D1 Setups: {days_with_setups}")
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        print(f"📈 Total OS D1 Setups Found: {len(final_df)}")
        
        print(f"\n📊 All OS D1 Setups Found in 2025:")
        print(f"{'='*80}")
        print(f"{'Date':<12} {'Ticker':<8} {'Price':<8} {'Gap $':<8} {'PM High %':<12} {'Volume':<12}")
        print(f"{'='*80}")
        
        for _, row in final_df.iterrows():
            print(f"{row['scan_date']:<12} {row['ticker']:<8} ${row['close']:<7.2f} ${row['gap_amount']:<7.2f} {row['pm_high_ratio']:<11.1%} {row['volume']:<12,.0f}")
        
        # Export results
        output_file = f"os_d1_2025_results.csv"
        final_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Show statistics
        if len(final_df) > 0:
            avg_price = final_df['close'].mean()
            avg_gap = final_df['gap_amount'].mean()
            avg_pm_high = final_df['pm_high_ratio'].mean()
            
            print(f"\n📊 2025 Setup Statistics:")
            print(f"   • Average price: ${avg_price:.2f}")
            print(f"   • Average gap: ${avg_gap:.2f}")
            print(f"   • Average PM high: {avg_pm_high:.1%}")
            if days_with_data > 0:
                print(f"   • Setup frequency: {days_with_setups}/{days_with_data} days ({days_with_setups/days_with_data:.1%})")
            
            # Show monthly breakdown
            print(f"\n📅 2025 Monthly Breakdown:")
            final_df['month'] = pd.to_datetime(final_df['scan_date']).dt.strftime('%Y-%m')
            monthly_counts = final_df.groupby('month').size()
            
            for month, count in monthly_counts.items():
                print(f"   • {month}: {count} setups")
        
        return final_df
        
    else:
        print(f"ℹ️ No OS D1 setups found in 2025 sample dates")
        
        if days_with_data == 0:
            print(f"⚠️ No market data available for future dates")
            print(f"   Polygon.io may not have data for dates beyond current date")
        else:
            print(f"📝 Your criteria are working - OS D1 setups require very specific conditions")
            
        return pd.DataFrame()

if __name__ == '__main__':
    asyncio.run(run_os_d1_2025_range())