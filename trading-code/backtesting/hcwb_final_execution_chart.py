#!/usr/bin/env python3
"""
HCWB Final OS D1 Execution Chart - Corrected 3 key breaks only
Shows: 2 starters stopped + 2 profitable 5m breaks (including post-HOD winner)
"""

import matplotlib
matplotlib.use("Qt5Agg")

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, time
from matplotlib.ticker import MaxNLocator

plt.style.use('dark_background')

def fetch_intraday_data(ticker, date, api_key):
    """Fetch intraday data from Polygon API"""
    trade_date = pd.to_datetime(date).strftime('%Y-%m-%d')
    
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{trade_date}/{date}'
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                df = pd.DataFrame(data['results'])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_convert('US/Eastern')
                df = df.set_index('timestamp')
                
                # Full trading session: 4:00 AM - 8:00 PM ET
                session_start = time(4, 0)
                session_end = time(20, 0) 
                df = df[(df.index.time >= session_start) & (df.index.time <= session_end)]
                
                df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
                return df
                
    except Exception as e:
        print(f"Error: {e}")
    
    return pd.DataFrame()

def resample_to_timeframe(df, timeframe):
    """Resample to different timeframes"""
    return df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

def add_indicators(df):
    """Add technical indicators"""
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    return df

def create_hcwb_final_chart():
    """Create HCWB final corrected OS D1 execution chart"""
    
    ticker = 'HCWB'
    date = '2025-08-25'
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    
    print(f"📊 Creating FINAL CORRECTED OS D1 execution chart for {ticker}")
    print("   3 key breaks only: 2 starters + 2 profitable 5m breaks")
    
    # Fetch the data
    df_1m = fetch_intraday_data(ticker, date, api_key)
    if df_1m.empty:
        print("❌ No data available")
        return
    
    # Create timeframes
    df_5m = resample_to_timeframe(df_1m, '5min')
    
    # Add indicators
    df_1m = add_indicators(df_1m)
    df_5m = add_indicators(df_5m)
    
    # Setup information
    setup_info = {
        'prev_close': 3.45,
        'pm_high': 5.78,
        'gap_pct': 55.9,
        'hod': 7.35
    }
    
    print(f"   📊 Gap: {setup_info['gap_pct']:.1f}%, PM High: ${setup_info['pm_high']:.2f}")
    
    # Final corrected trade sequence (only 4 trades total)
    trade_sequence = [
        {
            'name': 'Starter 1',
            'entry_time': '09:35:00',
            'entry_price': 5.30,
            'stop_price': 5.83,
            'stop_time': '09:46:00',
            'pnl': -0.530,
            'color': '#ff4444',
            'result': 'STOPPED'
        },
        {
            'name': 'Starter 2',
            'entry_time': '09:50:00', 
            'entry_price': 6.27,
            'stop_price': 6.57,
            'stop_time': '09:55:00',
            'pnl': -0.300,
            'color': '#ff6666',
            'result': 'STOPPED'
        },
        {
            'name': '5m Break 2',
            'entry_time': '09:52:00',
            'entry_price': 5.78,
            'exit_time': '11:31:00',
            'exit_price': 5.32,
            'pnl': 0.460,
            'color': '#44ff44',
            'result': 'PROFIT'
        },
        {
            'name': '5m Break 3 (Post-HOD Winner)',
            'entry_time': '10:26:00',
            'entry_price': 6.27,
            'exit_time': '11:31:00', 
            'exit_price': 5.32,
            'pnl': 0.950,
            'color': '#00ff88',
            'result': 'PROFIT'
        }
    ]
    
    # Create the chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14), gridspec_kw={'height_ratios': [4, 1]})
    
    # Price chart with candlesticks
    x = range(len(df_1m))
    
    # Plot candlesticks
    colors = ['white' if df_1m.iloc[i]['close'] >= df_1m.iloc[i]['open'] else 'red' for i in range(len(df_1m))]
    
    for i in range(len(df_1m)):
        row = df_1m.iloc[i]
        ax1.plot([i, i], [row['low'], row['high']], color=colors[i], linewidth=1, alpha=0.8)
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        if body_height > 0:
            rect = plt.Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                               facecolor=colors[i], alpha=0.8, edgecolor=colors[i])
            ax1.add_patch(rect)
    
    # Add indicators
    ax1.plot(x, df_1m['ema_20'], '--', color='#ff8800', linewidth=2, label='EMA 20', alpha=0.8)
    ax1.plot(x, df_1m['vwap'], '--', color='#ffff00', linewidth=2, label='VWAP', alpha=0.8)
    
    # Key levels
    ax1.axhline(y=setup_info['pm_high'], color='#ff00ff', linewidth=3, 
               alpha=0.9, label=f'PM High: ${setup_info["pm_high"]:.2f}')
    ax1.axhline(y=setup_info['hod'], color='#ff0088', linewidth=3,
               alpha=0.9, label=f'High of Day: ${setup_info["hod"]:.2f}')
    ax1.axhline(y=setup_info['prev_close'], color='#888888', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Prev Close: ${setup_info["prev_close"]:.2f}')
    
    # Mark market open
    market_open_idx = None
    for i, timestamp in enumerate(df_1m.index):
        if timestamp.time() >= time(9, 30):
            market_open_idx = i
            break
    
    if market_open_idx:
        ax1.axvline(x=market_open_idx, color='#ffffff', linestyle=':', linewidth=2, alpha=0.8, label='Market Open')
        ax1.axvspan(0, market_open_idx, color='#444444', alpha=0.3, label='Pre-Market')
    
    # Mark HOD time
    hod_idx = None
    for i, timestamp in enumerate(df_1m.index):
        if timestamp.time() >= time(10, 0):
            hod_idx = i
            break
    if hod_idx:
        ax1.axvline(x=hod_idx, color='#ff0088', linestyle=':', linewidth=2, alpha=0.8, label='HOD Time')
    
    # Plot all trade entries and exits
    for i, trade in enumerate(trade_sequence, 1):
        # Entry point
        entry_time = pd.to_datetime(f"2025-08-25 {trade['entry_time']}").tz_localize('US/Eastern')
        entry_idx = None
        
        for j, timestamp in enumerate(df_1m.index):
            if timestamp >= entry_time:
                entry_idx = j
                break
        
        if entry_idx:
            # Entry marker
            ax1.scatter(entry_idx, trade['entry_price'], s=250, color=trade['color'], 
                       marker='v', edgecolor='white', linewidth=2, zorder=10)
            
            # Entry label
            ax1.annotate(f"{trade['name']}\nSHORT: ${trade['entry_price']:.2f}\nP&L: ${trade['pnl']:+.3f}", 
                        xy=(entry_idx, trade['entry_price']),
                        xytext=(15, 25), textcoords='offset points',
                        fontsize=10, fontweight='bold', color=trade['color'],
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='black', alpha=0.8, edgecolor=trade['color']))
            
            # Exit point
            if trade['result'] == 'STOPPED':
                # Stop out
                stop_time = pd.to_datetime(f"2025-08-25 {trade['stop_time']}").tz_localize('US/Eastern')
                stop_idx = None
                
                for j, timestamp in enumerate(df_1m.index):
                    if timestamp >= stop_time:
                        stop_idx = j
                        break
                
                if stop_idx:
                    ax1.scatter(stop_idx, trade['stop_price'], s=200, color='red', 
                               marker='x', linewidth=4, zorder=10)
                    
                    # Draw stop line
                    ax1.plot([entry_idx, stop_idx], [trade['entry_price'], trade['stop_price']], 
                            '--', color='red', linewidth=2, alpha=0.7)
            
            else:  # PROFIT
                # Profit exit
                exit_time = pd.to_datetime(f"2025-08-25 {trade['exit_time']}").tz_localize('US/Eastern')
                exit_idx = None
                
                for j, timestamp in enumerate(df_1m.index):
                    if timestamp >= exit_time:
                        exit_idx = j
                        break
                
                if exit_idx:
                    ax1.scatter(exit_idx, trade['exit_price'], s=200, color='green', 
                               marker='^', edgecolor='white', linewidth=2, zorder=10)
                    
                    # Draw profit line
                    ax1.plot([entry_idx, exit_idx], [trade['entry_price'], trade['exit_price']], 
                            '-', color='green', linewidth=3, alpha=0.8)
    
    # Format x-axis
    time_labels = []
    time_indices = []
    for i in range(0, len(df_1m), max(1, len(df_1m) // 12)):
        time_labels.append(df_1m.index[i].strftime('%H:%M'))
        time_indices.append(i)
    
    ax1.set_xticks(time_indices)
    ax1.set_xticklabels(time_labels, rotation=45)
    
    # Calculate total P&L
    total_pnl = sum([trade['pnl'] for trade in trade_sequence])
    
    ax1.set_title(f'{ticker} FINAL OS D1 EXECUTION | Net P&L: ${total_pnl:+.3f}/share | 3 Key Breaks Only', 
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Volume chart
    volume_colors = ['#444444' if df_1m.index[i].time() < time(9, 30) else colors[i] 
                    for i in range(len(df_1m))]
    ax2.bar(x, df_1m['volume'], color=volume_colors, alpha=0.8)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)
    
    # Add execution summary
    summary_text = f"""
FINAL OS D1 EXECUTION RESULTS
═══════════════════════════════
SETUP: Gap +{setup_info['gap_pct']:.1f}%, HOD ${setup_info['hod']:.2f}

CORRECTED TRADE SEQUENCE:
1. Starter 1: SHORT @ $5.30 → STOPPED @ $5.83
   Result: ${trade_sequence[0]['pnl']:+.3f}/share (full loss) ❌

2. Starter 2: SHORT @ $6.27 → STOPPED @ $6.57  
   Result: ${trade_sequence[1]['pnl']:+.3f}/share (full loss) ❌

3. 5m Break 2: SHORT @ $5.78 → EXIT @ $5.32
   Result: ${trade_sequence[2]['pnl']:+.3f}/share ✅

4. 5m Break 3 (Post-HOD): SHORT @ $6.27 → EXIT @ $5.32
   Result: ${trade_sequence[3]['pnl']:+.3f}/share ✅ BIG WINNER

NET P&L: ${total_pnl:+.3f}/share

FINAL INSIGHTS:
✓ Only 3 key 5m bar breaks (not every new high)
✓ 2 full-size starter losses as expected  
✓ 5m Break 2 (09:45 level) profitable
✓ Post-HOD break (10:20) = biggest winner
✓ Break after true HOD captures gap exhaustion
✓ Entry when price breaks below 5m candle low

CORRECTED OS D1 LOGIC:
• First break at each significant level only
• Post-HOD breaks are especially valid
• Gap exhaustion creates winning opportunities
• Proper stop placement prevents overlosses
    """
    
    ax1.text(0.72, 0.98, summary_text, transform=ax1.transAxes,
             verticalalignment='top', fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#001100', alpha=0.95))
    
    plt.tight_layout()
    
    # Save chart
    chart_path = f'/Users/michaeldurante/sm-playbook/trading-code/backtesting/{ticker}_FINAL_OS_D1_Execution.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()
    
    print(f"✅ Final OS D1 execution chart saved: {chart_path}")
    print(f"\n🏆 FINAL EXECUTION SUMMARY:")
    print(f"   📊 Total Trades: {len(trade_sequence)}")
    print(f"   📈 Winners: 2")  
    print(f"   📉 Losers: 2")
    print(f"   💰 Net P&L: ${total_pnl:+.3f}/share")
    print(f"   🎯 Strategy: Corrected OS D1 - 3 key breaks only")

if __name__ == '__main__':
    create_hcwb_final_chart()