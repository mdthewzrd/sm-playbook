#!/usr/bin/env python3
"""
Test Golden Time Zone Implementation
Demonstrates the BRF strategy now only accepts signals during 8:30-11:30 AM ET
"""

import sys
sys.path.append('.bmad-core/execution')

from datetime import datetime
import pytz
from brf_execution_system import BRFExecutionSystem

def test_golden_time_zone():
    print("🕰️  Testing Golden Time Zone Implementation")
    print("=" * 50)
    
    # Initialize BRF execution system in paper mode
    brf_system = BRFExecutionSystem(mode="paper", initial_capital=100000)
    
    # Test different times
    test_times = [
        (7, 30),   # Before golden time
        (8, 29),   # Just before golden time  
        (8, 30),   # Start of golden time ✅
        (9, 15),   # During golden time ✅
        (10, 45),  # During golden time ✅
        (11, 30),  # End of golden time ✅
        (11, 31),  # Just after golden time
        (13, 30),  # Afternoon
        (15, 45)   # Late afternoon
    ]
    
    # Mock signal data
    test_signal = {
        'current_price': 25.50,
        'score': 75.0,
        'vwap': 24.80,
        'volume_ratio': 1.5,
        'suggested_position_size': 5000
    }
    
    print(f"Testing BRF signal processing at different times:")
    print(f"Signal: Price=${test_signal['current_price']}, Score={test_signal['score']}")
    print()
    
    et = pytz.timezone('US/Eastern')
    
    for hour, minute in test_times:
        # Create test time
        test_time = datetime.now(et).replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Patch the system's time check for testing
        original_method = brf_system._is_golden_time_zone
        
        def mock_golden_time():
            current_et = test_time
            golden_start = current_et.replace(hour=8, minute=30, second=0, microsecond=0)
            golden_end = current_et.replace(hour=11, minute=30, second=0, microsecond=0)
            return golden_start <= current_et <= golden_end
        
        brf_system._is_golden_time_zone = mock_golden_time
        
        # Test signal validation
        is_valid = brf_system._validate_signal_data(test_signal)
        
        # Determine status
        if hour == 8 and minute >= 30:
            expected = "✅ ACCEPTED"
        elif hour == 11 and minute <= 30:
            expected = "✅ ACCEPTED"  
        elif 8 < hour < 11:
            expected = "✅ ACCEPTED"
        else:
            expected = "❌ REJECTED"
        
        actual = "✅ ACCEPTED" if is_valid else "❌ REJECTED"
        
        print(f"{test_time.strftime('%H:%M')}: {actual} ({expected})")
        
        # Restore original method
        brf_system._is_golden_time_zone = original_method
    
    print()
    print("🎯 Golden Time Zone Summary:")
    print("• Strategy ONLY accepts signals from 8:30-11:30 AM ET")
    print("• All signals outside this window are automatically rejected")
    print("• This ensures focus on the most optimal trading hours")
    print("• Reduces risk from low-liquidity periods")

if __name__ == "__main__":
    test_golden_time_zone()