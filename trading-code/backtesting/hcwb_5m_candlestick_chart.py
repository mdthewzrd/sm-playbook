#!/usr/bin/env python3
"""
HCWB 5-Minute Candlestick Chart - Show actual 5m bars for break identification
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

def resample_to_5min(df):
    """Resample to 5-minute bars"""
    return df.resample('5min').agg({
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

def create_5m_candlestick_chart():
    """Create 5-minute candlestick chart for HCWB"""
    
    ticker = 'HCWB'
    date = '2025-08-25'
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    
    print(f"📊 Creating 5-MINUTE CANDLESTICK chart for {ticker}")
    print("   Shows actual 5m bars for break identification")
    
    # Fetch the data
    df_1m = fetch_intraday_data(ticker, date, api_key)
    if df_1m.empty:
        print("❌ No data available")
        return
    
    # Create 5-minute bars
    df_5m = resample_to_5min(df_1m)
    
    # Add indicators
    df_5m = add_indicators(df_5m)
    
    # Focus on market session
    market_5m = df_5m[df_5m.index.time >= time(9, 30)]
    market_5m = market_5m.head(40)  # Show first 40 bars (3+ hours)
    
    # Setup information
    setup_info = {
        'prev_close': 3.45,
        'pm_high': 5.78,
        'gap_pct': 55.9,
        'hod': market_5m['high'].max()
    }
    
    print(f"   📊 Gap: {setup_info['gap_pct']:.1f}%, PM High: ${setup_info['pm_high']:.2f}")
    print(f"   📊 High of Day: ${setup_info['hod']:.2f}")
    
    # Identify new high bars
    running_high = 0
    new_high_bars = []
    
    for i, (timestamp, bar) in enumerate(market_5m.iterrows()):
        if bar['high'] > running_high:
            running_high = bar['high']
            new_high_bars.append(i)
    
    # Create the chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14), gridspec_kw={'height_ratios': [4, 1]})
    
    # Plot 5-minute candlesticks
    x = range(len(market_5m))
    
    for i, (timestamp, bar) in enumerate(market_5m.iterrows()):
        # Determine color
        color = 'white' if bar['close'] >= bar['open'] else 'red'
        
        # Plot high-low line
        ax1.plot([i, i], [bar['low'], bar['high']], color=color, linewidth=2, alpha=0.8)
        
        # Plot body
        body_height = abs(bar['close'] - bar['open'])
        body_bottom = min(bar['open'], bar['close'])
        
        if body_height > 0.01:  # Only draw body if significant
            rect = Rectangle((i - 0.4, body_bottom), 0.8, body_height,
                           facecolor=color, alpha=0.8, edgecolor=color, linewidth=1)
            ax1.add_patch(rect)
        else:
            # Doji - draw small line
            ax1.plot([i-0.4, i+0.4], [bar['close'], bar['close']], color=color, linewidth=2)
        
        # Highlight new high bars
        if i in new_high_bars:
            # Add glow effect around new high bars
            ax1.scatter(i, bar['high'], s=100, color='yellow', marker='*', 
                       zorder=10, alpha=0.8, edgecolor='orange', linewidth=2)
            
            # Label the bar
            is_hod = bar['high'] == setup_info['hod']
            label_text = f"${bar['high']:.2f}\n{'HOD' if is_hod else 'NEW HIGH'}"
            ax1.annotate(label_text, 
                        xy=(i, bar['high']), xytext=(0, 15), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold', 
                        color='yellow' if not is_hod else 'orange',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    # Add indicators
    ax1.plot(x, market_5m['ema_20'], '--', color='#ff8800', linewidth=2, label='EMA 20', alpha=0.8)
    ax1.plot(x, market_5m['vwap'], '--', color='#ffff00', linewidth=2, label='VWAP', alpha=0.8)
    
    # Key levels
    ax1.axhline(y=setup_info['pm_high'], color='#ff00ff', linewidth=3, 
               alpha=0.9, label=f'PM High: ${setup_info["pm_high"]:.2f}')
    ax1.axhline(y=setup_info['prev_close'], color='#888888', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Prev Close: ${setup_info["prev_close"]:.2f}')
    
    # Mark HOD time
    hod_time = None
    for i, (timestamp, bar) in enumerate(market_5m.iterrows()):
        if bar['high'] == setup_info['hod']:
            hod_time = i
            ax1.axvline(x=i, color='#ff0088', linestyle=':', linewidth=3, alpha=0.9, 
                       label=f'HOD Time: {timestamp.strftime("%H:%M")}')
            break
    
    # Format x-axis with 5m timestamps
    time_labels = []
    time_indices = []
    for i in range(0, len(market_5m), max(1, len(market_5m) // 15)):
        time_labels.append(market_5m.index[i].strftime('%H:%M'))
        time_indices.append(i)
    
    ax1.set_xticks(time_indices)
    ax1.set_xticklabels(time_labels, rotation=45)
    
    ax1.set_title(f'{ticker} 5-MINUTE CANDLESTICK CHART | Gap: +{setup_info["gap_pct"]:.1f}% | HOD: ${setup_info["hod"]:.2f}', 
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='center right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Volume chart (5m volume)
    volume_colors = ['white' if market_5m.iloc[i]['close'] >= market_5m.iloc[i]['open'] else 'red' 
                    for i in range(len(market_5m))]
    ax2.bar(x, market_5m['volume'], color=volume_colors, alpha=0.8)
    ax2.set_ylabel('Volume (5m)')
    ax2.set_xlabel('Time (5-minute bars)')
    ax2.grid(True, alpha=0.3)
    
    # Add analysis text
    analysis_text = f"""
5-MINUTE BAR ANALYSIS
════════════════════
Each candle = 5 minutes of price action

NEW HIGH BARS (⭐ marked):
• These are 5m bar break candidates
• Entry when price breaks below these lows
• Only first break at each level counts

POST-HOD ANALYSIS:
• HOD at {market_5m[market_5m['high'] == setup_info['hod']].index[0].strftime('%H:%M')} 
• Look for 5m bars breaking below previous lows
• Post-HOD breaks capture gap exhaustion

BREAK IDENTIFICATION:
1. Find 5m candle that sets new high
2. Wait for price to break below that candle's low  
3. Enter SHORT on the break
4. Stop above recent highs or 1c over highest high after

VISUAL GUIDE:
⭐ = 5m bar that set new session high
🔴/⚪ = Bearish/Bullish 5m candle
📊 Key levels marked with horizontal lines
    """
    
    ax1.text(0.72, 0.98, analysis_text, transform=ax1.transAxes,
             verticalalignment='top', fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#001100', alpha=0.95))
    
    plt.tight_layout()
    
    # Save chart
    chart_path = f'/Users/michaeldurante/sm-playbook/trading-code/backtesting/{ticker}_5min_Candlestick_Chart.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()
    
    print(f"✅ 5-minute candlestick chart saved: {chart_path}")
    print(f"\n📊 NEW HIGH BARS IDENTIFIED:")
    
    for i in new_high_bars:
        timestamp = market_5m.index[i]
        bar = market_5m.iloc[i]
        is_hod = bar['high'] == setup_info['hod']
        print(f"   {timestamp.strftime('%H:%M')}: High ${bar['high']:.2f}, Low ${bar['low']:.2f} {'(HOD)' if is_hod else ''}")

if __name__ == '__main__':
    create_5m_candlestick_chart()