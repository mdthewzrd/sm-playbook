"""
V11 Quality Analysis Tool
Compares original V11 vs Enhanced V11 to demonstrate quality improvements
"""

import pandas as pd

def analyze_v11_quality_improvements():
    """Analyze and display the key quality improvements in V11 Enhanced"""
    
    print("🔍 V11 QUALITY ANALYSIS - PARAMETER COMPARISON")
    print("=" * 80)
    
    # Parameter comparison table
    comparison_data = [
        ["Parameter", "Original V11", "Enhanced V11", "Quality Impact"],
        ["Min Gap ATR", "0.4", "0.5", "↑ Filters smaller gaps"],
        ["Min Extension ATR", "0.8", "1.2", "↑ Requires better extension"], 
        ["Min Range Close %", "60%", "65%", "↑ Stronger closes"],
        ["Min Volume Multiple", "0.7x", "0.8x", "↑ Better volume"],
        ["Min Change ATR", "0.4", "0.5", "↑ Larger moves"],
        ["Max Downtrend Slope", "-0.20", "-0.30", "↑ Deeper declines"],
        ["Min EMA Extension %", "0%", "5%", "↑ Must be extended"],
        ["Min Price", "$3.00", "$5.00", "↑ Higher quality names"],
        ["Min Volume", "10M", "15M", "↑ More liquid"],
        ["Min Dollar Volume", "$20M", "$25M", "↑ Better liquidity"],
        ["Min Red Days", "1", "2", "↑ Real fade pattern"],
        ["Min Outlier Fade ATR", "2.0", "2.5", "↑ Bigger fade days"],
        ["Max Days Since High", "40", "35", "↑ Fresher setups"],
        ["Fade Lookback", "15", "12", "↑ More focused"],
    ]
    
    for row in comparison_data:
        if row[0] == "Parameter":
            print(f"{row[0]:<20} {row[1]:<15} {row[2]:<15} {row[3]}")
            print("-" * 80)
        else:
            print(f"{row[0]:<20} {row[1]:<15} {row[2]:<15} {row[3]}")
    
    print("\n🎯 KEY QUALITY IMPROVEMENTS:")
    print("✅ Gap Requirements: Back to 0.5 ATR (from 0.4) - filters weak gaps")
    print("✅ Extension: Increased to 1.2 ATR (from 0.8) - better setups")
    print("✅ EMA Extension: Now requires 5% (from 0%) - must be extended") 
    print("✅ Price Filter: $5 minimum (from $3) - higher quality names")
    print("✅ Range Close: 65% minimum (from 60%) - stronger closes")
    print("✅ Red Days: Requires 2+ (from 1) - real fade patterns")
    print("✅ Fade ATR: 2.5 minimum (from 2.0) - bigger fade days")
    print("✅ Strong Close Filter: New requirement for top 80% range closes")
    print("✅ Volume Quality Score: New metric combining volume + range")
    print("✅ Trend Duration: 3-60 days validation for better quality")
    
    print("\n📊 EXPECTED IMPACT:")
    print("• Fewer total setups (quality over quantity)")
    print("• Higher success rate for B-grade opportunities")
    print("• Less 'bullshit' noise from weak setups") 
    print("• Better risk/reward profiles")
    print("• More focused on actionable opportunities")

if __name__ == "__main__":
    analyze_v11_quality_improvements()