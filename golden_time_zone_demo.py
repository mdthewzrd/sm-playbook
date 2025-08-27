#!/usr/bin/env python3
"""
Golden Time Zone Implementation Demo
Shows how the BRF strategy now filters signals to 8:30-11:30 AM ET only
"""

from datetime import datetime
import pytz

def is_golden_time_zone(test_time=None):
    """Check if time is within Golden Time Zone (8:30-11:30 AM ET)"""
    try:
        et = pytz.timezone('US/Eastern')
        current_et = test_time or datetime.now(et)
        
        # Skip weekends
        if current_et.weekday() >= 5:
            return False
        
        # Golden Time Zone: 8:30 AM - 11:30 AM ET
        golden_start = current_et.replace(hour=8, minute=30, second=0, microsecond=0)
        golden_end = current_et.replace(hour=11, minute=30, second=0, microsecond=0)
        
        return golden_start <= current_et <= golden_end
    except Exception:
        return False

def demo_golden_time_zone():
    print("🕰️  BRF Strategy - Golden Time Zone Filter")
    print("=" * 50)
    print("✅ RULE ADDED: Strategy only looks for extensions during 8:30-11:30 AM ET")
    print()
    
    # Test different times throughout the day
    test_times = [
        ("7:00", 7, 0),   # Pre-market
        ("8:00", 8, 0),   # Just before golden time
        ("8:30", 8, 30),  # Golden time starts ✅
        ("9:15", 9, 15),  # Prime morning hours ✅
        ("10:00", 10, 0), # Peak golden time ✅
        ("11:00", 11, 0), # Still golden time ✅
        ("11:30", 11, 30), # Golden time ends ✅
        ("12:00", 12, 0), # Lunch period
        ("14:00", 14, 0), # Afternoon
        ("15:30", 15, 30) # Market close approach
    ]
    
    et = pytz.timezone('US/Eastern')
    base_date = datetime.now(et).replace(hour=9, minute=0, second=0, microsecond=0)
    
    print("Time (ET)  | Status      | Reason")
    print("-" * 40)
    
    for time_str, hour, minute in test_times:
        test_time = base_date.replace(hour=hour, minute=minute)
        is_golden = is_golden_time_zone(test_time)
        
        if is_golden:
            status = "✅ ACCEPTED"
            reason = "Within Golden Time Zone"
        else:
            if hour < 8 or (hour == 8 and minute < 30):
                reason = "Before Golden Time Zone"
            elif hour > 11 or (hour == 11 and minute > 30):
                reason = "After Golden Time Zone"
            else:
                reason = "Outside Golden Time Zone"
            status = "❌ REJECTED"
        
        print(f"{time_str:<9} | {status:<11} | {reason}")
    
    print()
    print("📊 Impact on Strategy:")
    print("• Reduces false signals from low-liquidity periods")
    print("• Focuses on high-volume morning momentum")
    print("• Avoids lunch-time chop and late-day volatility")
    print("• Aligns with traditional 'Golden Hours' of day trading")
    print()
    print("🎯 Implementation:")
    print("• Added to strategy YAML configuration")
    print("• Enforced in execution system validation")
    print("• Applied to backtesting analysis")
    print("• Integrated with risk controls")

if __name__ == "__main__":
    demo_golden_time_zone()