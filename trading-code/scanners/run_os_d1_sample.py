#!/usr/bin/env python3
"""
Run OS D1 Scanner for sample dates in the range
Testing a few key dates to validate the scanner works
"""

import pandas as pd
import asyncio
from datetime import datetime
import sys
import os

# Add the scanner directory to the path
sys.path.append(os.path.dirname(__file__))

from os_d1_pure_scanner import OS_D1_Scanner

async def run_os_d1_sample_dates():
    """Run OS D1 scanner for key sample dates in the range"""
    
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    scanner = OS_D1_Scanner(api_key)
    
    # Sample dates from your requested range (8/25/24 - 1/1/25)
    # Testing key dates that might have volatile market conditions
    sample_dates = [
        '2024-08-26',  # Monday after 8/25
        '2024-09-03',  # Post Labor Day
        '2024-09-18',  # Fed meeting week
        '2024-10-01',  # Q4 start
        '2024-10-31',  # End of October
        '2024-11-05',  # Election week
        '2024-11-29',  # Post Thanksgiving
        '2024-12-02',  # December start
        '2024-12-16',  # Mid December
        '2024-12-30',  # End of year
    ]
    
    print(f"🚀 Running OS D1 Scanner for Sample Dates")
    print(f"📅 Testing {len(sample_dates)} key dates from 8/25/24 - 1/1/25 range")
    print("=" * 60)
    
    all_results = []
    total_days = 0
    days_with_setups = 0
    
    for i, scan_date in enumerate(sample_dates):
        print(f"\n📅 Sample {i+1}/{len(sample_dates)}: Scanning {scan_date}")
        print("-" * 40)
        
        try:
            # Run the scanner for this date
            results = await scanner.scan_os_d1(scan_date)
            
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
            
            total_days += 1
            
        except Exception as e:
            print(f"❌ Error scanning {scan_date}: {e}")
            continue
    
    # Compile final results
    print(f"\n{'='*60}")
    print(f"🎯 OS D1 SCANNER SAMPLE RESULTS")
    print(f"{'='*60}")
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        print(f"📅 Sample Dates Scanned: {total_days}")
        print(f"🎯 Days with OS D1 Setups: {days_with_setups}")
        print(f"📈 Total OS D1 Setups Found: {len(final_df)}")
        
        print(f"\n📊 All OS D1 Setups Found:")
        print(f"{'='*80}")
        print(f"{'Date':<12} {'Ticker':<8} {'Price':<8} {'Gap $':<8} {'PM High %':<12} {'Volume':<12}")
        print(f"{'='*80}")
        
        for _, row in final_df.iterrows():
            print(f"{row['scan_date']:<12} {row['ticker']:<8} ${row['close']:<7.2f} ${row['gap_amount']:<7.2f} {row['pm_high_ratio']:<11.1%} {row['volume']:<12,.0f}")
        
        # Export results
        output_file = f"os_d1_sample_results.csv"
        final_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Show statistics
        if len(final_df) > 0:
            avg_price = final_df['close'].mean()
            avg_gap = final_df['gap_amount'].mean()
            avg_pm_high = final_df['pm_high_ratio'].mean()
            
            print(f"\n📊 Setup Statistics:")
            print(f"   • Average price: ${avg_price:.2f}")
            print(f"   • Average gap: ${avg_gap:.2f}")
            print(f"   • Average PM high: {avg_pm_high:.1%}")
            print(f"   • Setup frequency: {days_with_setups}/{total_days} days ({days_with_setups/total_days:.1%})")
        
        return final_df
        
    else:
        print(f"ℹ️ No OS D1 setups found in sample dates")
        print(f"📝 Your criteria are working - OS D1 setups require very specific conditions:")
        print(f"   • PM High ≥ 50% above previous close")
        print(f"   • Gap ≥ $0.50")
        print(f"   • Open ≥ 30% above previous high")
        print(f"   • Volume ≥ 5M")
        print(f"   • Previous close ≥ $0.75")
        print(f"   • Current price ≤ 80% of EMA200")
        
        return pd.DataFrame()

if __name__ == '__main__':
    asyncio.run(run_os_d1_sample_dates())