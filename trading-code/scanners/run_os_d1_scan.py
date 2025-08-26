#!/usr/bin/env python3
"""
Simple OS D1 LC Scanner - Direct Implementation
Uses your exact LC scan parameters to scan all available tickers
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta

# Add the scanner directory to the path
sys.path.append(os.path.dirname(__file__))

from os_d1_lc_scanner import OS_D1_LC_Scanner

async def run_os_d1_scan():
    """Run the OS D1 LC scan with your exact parameters"""
    
    print("🚀 Running OS D1 Long Continuation Scanner")
    print("=" * 50)
    print("Using exact parameters from your Notion SM Playbook OS D1 setup")
    
    # Initialize scanner
    api_key = os.getenv('POLYGON_API_KEY', 'Fm7brz4s23eSocDErnL68cE7wspz2K1I')
    scanner = OS_D1_LC_Scanner(api_key)
    
    # Use a recent trading date for the scan
    # Go back a few days to ensure data is available
    scan_date = datetime.now() - timedelta(days=3)
    
    # Make sure it's a weekday
    while scan_date.weekday() >= 5:  # Weekend
        scan_date -= timedelta(days=1)
    
    start_date = scan_date.strftime('%Y-%m-%d')
    end_date = start_date  # Single day scan
    
    print(f"📅 Scan Date: {start_date}")
    print(f"🔍 Scanning all available tickers for LC setups...")
    
    try:
        # Run the scanner
        results_df = await scanner.scan_dates(start_date, end_date)
        
        if results_df is not None and len(results_df) > 0:
            print(f"\n📊 Scan Results:")
            print(f"   Total tickers processed: {len(results_df):,}")
            
            # Check for LC setups using your exact criteria
            lc_columns = ['lc_frontside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_fbo']
            lc_setups = results_df[results_df[lc_columns].sum(axis=1) > 0]
            
            if len(lc_setups) > 0:
                print(f"🎯 LC Setups Found: {len(lc_setups)}")
                print(f"\n{'='*80}")
                print(f"{'Ticker':<10} {'Setup Type':<25} {'Price':<10} {'Volume':<15} {'Gap ATR':<10}")
                print(f"{'='*80}")
                
                for _, row in lc_setups.iterrows():
                    setup_types = []
                    for col in lc_columns:
                        if row[col] == 1:
                            if 'frontside_d3_extended_1' in col:
                                setup_types.append('Frontside D3 Extended')
                            elif 'backside_d3_extended_1' in col:
                                setup_types.append('Backside D3 Extended')
                            elif 'fbo' in col:
                                setup_types.append('Failed Breakout')
                    
                    setup_type_str = ', '.join(setup_types)
                    price = f"${row['c']:.2f}"
                    volume = f"{row['v']:,}"
                    gap_atr = f"{row.get('gap_atr', 0):.2f}" if 'gap_atr' in row else "N/A"
                    
                    print(f"{row['ticker']:<10} {setup_type_str:<25} {price:<10} {volume:<15} {gap_atr:<10}")
                
                # Export results
                output_file = f"os_d1_lc_results_{start_date}.csv"
                lc_setups.to_csv(output_file, index=False)
                print(f"\n💾 Results saved to: {output_file}")
                
                # Show key statistics
                print(f"\n📈 Setup Statistics:")
                
                # Pattern breakdown
                frontside_count = (lc_setups['lc_frontside_d3_extended_1'] == 1).sum()
                backside_count = (lc_setups['lc_backside_d3_extended_1'] == 1).sum()
                fbo_count = (lc_setups['lc_fbo'] == 1).sum()
                
                if frontside_count > 0:
                    print(f"   • Frontside D3 Extended: {frontside_count}")
                if backside_count > 0:
                    print(f"   • Backside D3 Extended: {backside_count}")
                if fbo_count > 0:
                    print(f"   • Failed Breakout: {fbo_count}")
                
                # Price and volume ranges
                avg_price = lc_setups['c'].mean()
                avg_volume = lc_setups['v'].mean()
                print(f"   • Average Price: ${avg_price:.2f}")
                print(f"   • Average Volume: {avg_volume:,.0f}")
                
            else:
                print(f"ℹ️ No LC setups found for {start_date}")
                print("   This is normal - LC setups are specific patterns that don't occur daily")
                
                # Show some statistics about what was processed
                print(f"\n📊 Scan Statistics:")
                if 'v' in results_df.columns:
                    high_volume_count = (results_df['v'] >= 10000000).sum()
                    print(f"   • High volume stocks (>10M): {high_volume_count:,}")
                
                if 'c' in results_df.columns:
                    high_price_count = (results_df['c'] >= 20).sum()
                    print(f"   • Stocks above $20: {high_price_count:,}")
                
        else:
            print("❌ No data returned from scanner")
            
    except Exception as e:
        print(f"❌ Scan failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n✅ OS D1 LC scan completed successfully!")
    print(f"📝 Ready to proceed with entry/exit logic implementation")
    
    return True

if __name__ == '__main__':
    asyncio.run(run_os_d1_scan())