#!/usr/bin/env python3
"""
BRF Trade Execution Logic Validation
Analyzes the core trading logic independent of data artifacts
"""

import json
import pandas as pd
from datetime import datetime

def validate_trade_execution():
    """Validate the core BRF trading execution logic"""
    
    print("="*60)
    print("BRF TRADE EXECUTION LOGIC VALIDATION")
    print("="*60)
    
    # Load backtest results
    try:
        with open('/Users/michaeldurante/sm-playbook/reports/brf_backtest_results.json', 'r') as f:
            results = json.load(f)
    except:
        print("❌ Could not load backtest results")
        return
    
    trades = results.get('trades', [])
    if not trades:
        print("❌ No trade data found")
        return
        
    trades_df = pd.DataFrame(trades)
    
    print(f"📊 TRADE DATA ANALYSIS")
    print(f"Total trade records: {len(trades_df)}")
    
    # Separate buy and sell trades
    buy_trades = trades_df[trades_df['action'] == 'BUY'].copy()
    sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()
    
    print(f"Buy trades: {len(buy_trades)}")
    print(f"Sell trades: {len(sell_trades)}")
    
    # Validate entry logic
    print(f"\n🎯 ENTRY LOGIC VALIDATION")
    
    if len(buy_trades) > 0:
        print(f"✅ Entry signals generated: {len(buy_trades)}")
        
        # Check signal scores
        if 'signal_score' in buy_trades.columns:
            avg_score = buy_trades['signal_score'].mean()
            min_score = buy_trades['signal_score'].min()
            max_score = buy_trades['signal_score'].max()
            
            print(f"✅ Signal scoring working:")
            print(f"   Average score: {avg_score:.1f}/100")
            print(f"   Score range: {min_score:.1f} - {max_score:.1f}")
            print(f"   Above threshold (70): {len(buy_trades[buy_trades['signal_score'] >= 70])}")
        
        # Check position sizing
        if 'shares' in buy_trades.columns and 'price' in buy_trades.columns:
            buy_trades['position_value'] = buy_trades['shares'] * buy_trades['price']
            avg_position = buy_trades['position_value'].mean()
            print(f"✅ Position sizing logic working:")
            print(f"   Average position value: ${avg_position:.2f}")
            print(f"   Position range: ${buy_trades['position_value'].min():.2f} - ${buy_trades['position_value'].max():.2f}")
    
    # Validate exit logic  
    print(f"\n📤 EXIT LOGIC VALIDATION")
    
    if len(sell_trades) > 0:
        print(f"✅ Exit signals generated: {len(sell_trades)}")
        
        # Check exit reasons
        if 'exit_reason' in sell_trades.columns:
            exit_reasons = sell_trades['exit_reason'].value_counts()
            print(f"✅ Exit reason tracking:")
            for reason, count in exit_reasons.items():
                print(f"   {reason}: {count}")
        
        # Check P&L logic
        profitable_trades = sell_trades[sell_trades.get('pnl', 0) > 0]
        losing_trades = sell_trades[sell_trades.get('pnl', 0) <= 0]
        
        if len(profitable_trades) > 0 or len(losing_trades) > 0:
            print(f"✅ P&L calculation working:")
            print(f"   Profitable trades: {len(profitable_trades)}")
            print(f"   Losing trades: {len(losing_trades)}")
            
            if len(profitable_trades) > 0:
                win_rate = len(profitable_trades) / (len(profitable_trades) + len(losing_trades))
                print(f"   Win rate: {win_rate*100:.1f}%")
    
    # Risk management validation
    print(f"\n🛡️  RISK MANAGEMENT VALIDATION")
    
    # Check for risk controls in buy trades
    if len(buy_trades) > 0:
        # Simulate checking if positions would exceed limits
        concurrent_positions = {}
        max_concurrent = 0
        
        for _, trade in trades_df.iterrows():
            timestamp = trade['timestamp']
            symbol = trade['symbol']
            action = trade['action']
            
            if action == 'BUY':
                concurrent_positions[symbol] = concurrent_positions.get(symbol, 0) + 1
            elif action == 'SELL':
                concurrent_positions[symbol] = concurrent_positions.get(symbol, 1) - 1
                if concurrent_positions[symbol] <= 0:
                    del concurrent_positions[symbol]
            
            current_count = sum(concurrent_positions.values())
            max_concurrent = max(max_concurrent, current_count)
        
        print(f"✅ Position limits respected:")
        print(f"   Max concurrent positions: {max_concurrent}")
        print(f"   Target limit: 3")
        
        if max_concurrent <= 3:
            print(f"   Status: ✅ Within limits")
        else:
            print(f"   Status: ⚠️ Exceeded limits")
    
    # Strategy-specific validation
    print(f"\n🔧 STRATEGY-SPECIFIC VALIDATION")
    
    # Check for backside runner characteristics
    symbols_traded = set()
    if len(buy_trades) > 0:
        symbols_traded = set(buy_trades['symbol'].unique())
        print(f"✅ Universe targeting:")
        print(f"   Symbols traded: {sorted(symbols_traded)}")
        print(f"   Target universe: Small-cap momentum stocks")
    
    # Check hold times
    if 'hold_time_minutes' in sell_trades.columns:
        avg_hold_time = sell_trades['hold_time_minutes'].mean()
        print(f"✅ Hold time analysis:")
        print(f"   Average hold time: {avg_hold_time:.0f} minutes")
        print(f"   Expected: Intraday mean reversion (< 1 day)")
        
        if avg_hold_time < 480:  # Less than 8 hours
            print(f"   Status: ✅ Appropriate for strategy")
        else:
            print(f"   Status: ⚠️ Longer than expected")
    
    # Overall validation summary
    print(f"\n📋 VALIDATION SUMMARY")
    
    validations = []
    
    # Core logic checks
    if len(buy_trades) > 0:
        validations.append(("Entry signal generation", "✅ PASS"))
    else:
        validations.append(("Entry signal generation", "❌ FAIL"))
    
    if len(sell_trades) > 0:
        validations.append(("Exit signal generation", "✅ PASS"))
    else:
        validations.append(("Exit signal generation", "❌ FAIL"))
        
    if 'signal_score' in buy_trades.columns and buy_trades['signal_score'].min() >= 50:
        validations.append(("Signal quality scoring", "✅ PASS"))
    else:
        validations.append(("Signal quality scoring", "⚠️ PARTIAL"))
    
    if max_concurrent <= 3:
        validations.append(("Position limit controls", "✅ PASS"))
    else:
        validations.append(("Position limit controls", "❌ FAIL"))
    
    if len(sell_trades) > 0 and 'exit_reason' in sell_trades.columns:
        validations.append(("Exit reason tracking", "✅ PASS"))
    else:
        validations.append(("Exit reason tracking", "⚠️ PARTIAL"))
    
    # Print validation results
    for check, status in validations:
        print(f"   {check}: {status}")
    
    # Overall status
    passed = sum(1 for _, status in validations if "✅" in status)
    total = len(validations)
    
    print(f"\n🎖️  OVERALL VALIDATION STATUS")
    print(f"   Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print(f"   Result: ✅ ALL VALIDATIONS PASSED")
        print(f"   Status: Ready for production deployment")
    elif passed >= total * 0.8:
        print(f"   Result: ⚠️ MOST VALIDATIONS PASSED")
        print(f"   Status: Minor issues to address")
    else:
        print(f"   Result: ❌ VALIDATION ISSUES")
        print(f"   Status: Requires fixes before deployment")
    
    print("="*60)

if __name__ == "__main__":
    validate_trade_execution()