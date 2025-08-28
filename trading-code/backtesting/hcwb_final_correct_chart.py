#!/usr/bin/env python3
"""
HCWB Final Correct OS D1 Chart - Proper 5m breaks with correct stop logic
09:45 break = LOSER (stop 1c over $6.48 high gets hit)
10:00 break = WINNER (post-HOD break works)
"""

import matplotlib
matplotlib.use("Qt5Agg")

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, time
from matplotlib.patches import Rectangle

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
    
    return df

def create_final_correct_chart():
    """Create final correct HCWB execution chart with proper stops"""
    
    ticker = 'HCWB'
    date = '2025-08-25'
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    
    print(f"📊 Creating FINAL CORRECT OS D1 execution chart")
    print("   09:45 break = LOSER (stop 1c over high gets hit)")
    print("   10:00 break = WINNER")
    
    # Fetch the data
    df_1m = fetch_intraday_data(ticker, date, api_key)
    if df_1m.empty:
        print("❌ No data available")
        return
    
    # Add indicators
    df_1m = add_indicators(df_1m)
    
    # Setup information
    setup_info = {
        'prev_close': 3.45,
        'pm_high': 5.78,
        'gap_pct': 55.9,
        'hod': 7.35
    }
    
    # Corrected trade sequence with proper stops
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
            'name': '5m Break 1 (09:45)',
            'entry_time': '09:52:00',  # Break below $5.79
            'entry_price': 5.78,
            'stop_price': 6.49,  # 1c over 09:45 candle high ($6.48)
            'stop_time': '09:55:00',  # Gets stopped when price hits $6.49+
            'pnl': -0.710,  # Stopped for loss
            'color': '#ff8888',
            'result': 'STOPPED'
        },
        {
            'name': '5m Break 2 (10:00 HOD Winner)',
            'entry_time': '10:16:00',  # Break below $6.70
            'entry_price': 6.69,
            'exit_time': '11:31:00',
            'exit_price': 5.32,
            'pnl': 1.370,  # Big winner
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
    ax1.axhline(y=6.49, color='#ffaa00', linewidth=2, linestyle=':', 
               alpha=0.9, label='09:45 Stop: $6.49 (1c over $6.48)')
    ax1.axhline(y=setup_info['prev_close'], color='#888888', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Prev Close: ${setup_info["prev_close"]:.2f}')
    
    # Mark market open and HOD time
    market_open_idx = None
    hod_idx = None
    
    for i, timestamp in enumerate(df_1m.index):
        if timestamp.time() >= time(9, 30) and market_open_idx is None:
            market_open_idx = i
            ax1.axvline(x=i, color='#ffffff', linestyle=':', linewidth=2, alpha=0.8, label='Market Open')
            ax1.axvspan(0, i, color='#444444', alpha=0.3, label='Pre-Market')
        if timestamp.time() >= time(10, 0) and hod_idx is None:
            hod_idx = i
            ax1.axvline(x=i, color='#ff0088', linestyle=':', linewidth=2, alpha=0.8, label='HOD Time (10:00)')
    
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
    
    ax1.set_title(f'{ticker} FINAL CORRECT OS D1 | Net P&L: ${total_pnl:+.3f}/share | Proper Stop Logic', 
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='center right', fontsize=9)
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
FINAL CORRECT OS D1 RESULTS
══════════════════════════
SETUP: Gap +{setup_info['gap_pct']:.1f}%, HOD ${setup_info['hod']:.2f}

CORRECTED SEQUENCE (PROPER STOPS):
1. Starter 1: SHORT @ $5.30 → STOPPED @ $5.83
   Result: ${trade_sequence[0]['pnl']:+.3f}/share ❌

2. Starter 2: SHORT @ $6.27 → STOPPED @ $6.57  
   Result: ${trade_sequence[1]['pnl']:+.3f}/share ❌

3. 5m Break 1 (09:45): SHORT @ $5.78 → STOPPED @ $6.49
   Result: ${trade_sequence[2]['pnl']:+.3f}/share ❌
   Stop: 1c over 09:45 candle high ($6.48)

4. 5m Break 2 (10:00): SHORT @ $6.69 → EXIT @ $5.32
   Result: ${trade_sequence[3]['pnl']:+.3f}/share ✅ BIG WINNER

NET P&L: ${total_pnl:+.3f}/share

CORRECTED LOGIC:
✓ 09:45 5m break: Entry @ $5.78 (below $5.79 low)
✓ Stop @ $6.49 (1c over $6.48 high) = GETS HIT
✓ 10:00 5m break: Entry @ $6.69 (below $6.70 low) 
✓ Post-HOD break captures gap exhaustion = WINNER

PROPER STOP PLACEMENT:
• Stop = 1c over the 5m candle's high
• NOT over future highs or arbitrary levels
• 09:45 candle high = $6.48, so stop = $6.49
• Price goes above $6.49 → stop gets hit
    """
    
    ax1.text(0.72, 0.98, summary_text, transform=ax1.transAxes,
             verticalalignment='top', fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#001100', alpha=0.95))
    
    plt.tight_layout()
    
    # Save chart
    chart_path = f'/Users/michaeldurante/sm-playbook/trading-code/backtesting/{ticker}_FINAL_CORRECT_Execution.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()
    
    print(f"✅ Final correct execution chart saved: {chart_path}")
    print(f"\n🏆 CORRECTED EXECUTION SUMMARY:")
    print(f"   📊 Total Trades: {len(trade_sequence)}")
    print(f"   📈 Winners: 1")  
    print(f"   📉 Losers: 3")
    print(f"   💰 Net P&L: ${total_pnl:+.3f}/share")
    print(f"   🎯 09:45 break STOPPED at $6.49 (1c over $6.48 high)")
    print(f"   🎯 10:00 break WINNER (post-HOD gap exhaustion)")

if __name__ == '__main__':
    create_final_correct_chart()