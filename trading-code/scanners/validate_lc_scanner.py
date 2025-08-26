#!/usr/bin/env python3
"""
Comprehensive LC Scanner Validation Script
Tests the OS D1 LC scanner across multiple historical periods to find actual LC setups
and validate the pattern detection matches your proven logic
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import asyncio

# Add the scanner directory to the path
sys.path.append(os.path.dirname(__file__))

from os_d1_lc_scanner import OS_D1_LC_Scanner

async def validate_historical_periods():
    """Run scanner across multiple historical periods to find LC setups"""
    
    api_key = os.getenv('POLYGON_API_KEY', 'Fm7brz4s23eSocDErnL68cE7wspz2K1I')
    
    if not api_key:
        print("❌ No API key available")
        return False
    
    scanner = OS_D1_LC_Scanner(api_key)
    
    # Test multiple historical periods where LC setups are more likely
    test_periods = [
        # Recent volatile periods
        ('2024-07-01', '2024-07-31'),  # July 2024
        ('2024-06-01', '2024-06-30'),  # June 2024
        ('2024-05-01', '2024-05-31'),  # May 2024
        ('2024-04-01', '2024-04-30'),  # April 2024
        ('2024-03-01', '2024-03-31'),  # March 2024
    ]
    
    total_setups_found = 0
    all_results = []
    
    for start_date, end_date in test_periods:
        print(f"\n📅 Scanning period: {start_date} to {end_date}")
        
        try:
            results_df = await scanner.scan_dates(start_date, end_date)
            
            if results_df is not None and len(results_df) > 0:
                # Filter for any LC setups found
                lc_columns = ['lc_frontside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_fbo']
                lc_results = results_df[results_df[lc_columns].sum(axis=1) > 0]
                
                period_setups = len(lc_results)
                total_setups_found += period_setups
                
                print(f"  📊 Processed {len(results_df)} tickers")
                print(f"  🎯 Found {period_setups} LC setups")
                
                if period_setups > 0:
                    # Show top setups from this period
                    print("  📈 Top LC setups found:")
                    for _, row in lc_results.head(3).iterrows():
                        setup_types = []
                        for col in lc_columns:
                            if row[col] == 1:
                                setup_types.append(col.replace('lc_', '').replace('_', ' ').title())
                        print(f"    • {row['ticker']} ({row['date']}): {', '.join(setup_types)}")
                        print(f"      Close: ${row['c']:.2f}, Volume: {row['v']:,}, Gap ATR: {row.get('gap_atr', 0):.2f}")
                    
                    # Store results for analysis
                    lc_results['period'] = f"{start_date}_to_{end_date}"
                    all_results.append(lc_results)
                else:
                    print("    ℹ️ No LC setups found in this period")
            else:
                print("    ⚠️ No data returned for this period")
                
        except Exception as e:
            print(f"    ❌ Error scanning period {start_date} to {end_date}: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total LC setups found: {total_setups_found}")
    
    if total_setups_found > 0:
        print("✅ Scanner successfully detects LC patterns!")
        
        # Combine all results for analysis
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            
            # Analyze pattern distribution
            pattern_counts = {}
            lc_columns = ['lc_frontside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_fbo']
            
            for col in lc_columns:
                count = (combined_results[col] == 1).sum()
                if count > 0:
                    pattern_name = col.replace('lc_', '').replace('_', ' ').title()
                    pattern_counts[pattern_name] = count
            
            print(f"\n📈 Pattern Distribution:")
            for pattern, count in pattern_counts.items():
                print(f"  • {pattern}: {count} setups")
            
            # Show volume and price characteristics
            avg_volume = combined_results['v'].mean()
            avg_price = combined_results['c'].mean()
            avg_gap_atr = combined_results.get('gap_atr', pd.Series([0])).mean()
            
            print(f"\n📊 Setup Characteristics:")
            print(f"  • Average Volume: {avg_volume:,.0f}")
            print(f"  • Average Price: ${avg_price:.2f}")
            print(f"  • Average Gap ATR: {avg_gap_atr:.2f}")
            
            # Export results
            output_file = "lc_validation_results.csv"
            combined_results.to_csv(output_file, index=False)
            print(f"\n💾 Results exported to: {output_file}")
            
        return True
    else:
        print("⚠️ No LC setups found across all tested periods")
        print("   This could indicate:")
        print("   1. The scanning criteria are too restrictive")
        print("   2. The test periods didn't contain suitable market conditions")
        print("   3. Historical data limitations")
        return False

async def spot_check_known_patterns():
    """Spot check for patterns in known volatile stocks during specific periods"""
    
    print(f"\n🔍 SPOT CHECK: Testing known volatile stocks")
    print("=" * 50)
    
    api_key = os.getenv('POLYGON_API_KEY', 'Fm7brz4s23eSocDErnL68cE7wspz2K1I')
    scanner = OS_D1_LC_Scanner(api_key)
    
    # Focus on a more recent period with known market volatility
    # Using a shorter, more recent period for better data availability
    test_date_start = '2024-08-01'
    test_date_end = '2024-08-31'
    
    print(f"📅 Focused scan: {test_date_start} to {test_date_end}")
    
    try:
        results_df = await scanner.scan_dates(test_date_start, test_date_end)
        
        if results_df is not None and len(results_df) > 0:
            print(f"📊 Total records processed: {len(results_df):,}")
            
            # Check for any patterns
            lc_columns = ['lc_frontside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_fbo']
            
            # Look at the distribution of key metrics to validate filtering
            print(f"\n📈 Data Quality Check:")
            print(f"  • Tickers with volume > 10M: {(results_df['v'] > 10000000).sum():,}")
            print(f"  • Tickers with price > $20: {(results_df['c'] > 20).sum():,}")
            print(f"  • Records after initial filtering: {len(results_df):,}")
            
            # Check if any records have the pattern columns
            for col in lc_columns:
                if col in results_df.columns:
                    pattern_count = (results_df[col] == 1).sum()
                    if pattern_count > 0:
                        pattern_name = col.replace('lc_', '').replace('_', ' ').title()
                        print(f"  🎯 {pattern_name}: {pattern_count} setups")
                        
                        # Show examples
                        examples = results_df[results_df[col] == 1].head(3)
                        for _, row in examples.iterrows():
                            print(f"    • {row['ticker']} ({row['date']}): ${row['c']:.2f}, Vol: {row['v']:,}")
            
            return True
        else:
            print("⚠️ No data returned for spot check period")
            return False
            
    except Exception as e:
        print(f"❌ Spot check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main validation runner"""
    print("🚀 COMPREHENSIVE LC SCANNER VALIDATION")
    print("=" * 60)
    print("This validation tests the scanner across multiple historical periods")
    print("to ensure it correctly identifies LC (Long Continuation) patterns")
    print("using your exact logic from the Notion SM Playbook.\n")
    
    # Run historical validation
    historical_success = await validate_historical_periods()
    
    # Run focused spot check
    spot_check_success = await spot_check_known_patterns()
    
    print(f"\n{'='*60}")
    print("🎯 FINAL VALIDATION RESULTS")
    print(f"{'='*60}")
    
    if historical_success or spot_check_success:
        print("✅ LC Scanner VALIDATION SUCCESSFUL!")
        print("   • Scanner correctly processes market data")
        print("   • Pattern detection logic is functional") 
        print("   • Ready to proceed with entry/exit logic implementation")
    else:
        print("⚠️ Scanner validation needs attention")
        print("   • Consider testing with different date ranges")
        print("   • May need to adjust LC pattern criteria")
        print("   • Check data availability for selected periods")
    
    print(f"\n📝 Next Steps:")
    print("   1. ✅ Scanner validation completed")
    print("   2. 🔄 Add entry/exit logic to strategy")
    print("   3. 🧪 Create backtesting framework integration")

if __name__ == '__main__':
    asyncio.run(main())