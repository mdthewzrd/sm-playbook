#!/usr/bin/env python3

"""Create detailed August 19th chart showing enhanced pyramiding strategy"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'indicators'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time, timedelta
from polygon import RESTClient
from dual_deviation_cloud import DualDeviationCloud
import ta

def create_august_19_pyramiding_chart():
    """Create detailed chart showing August 19th enhanced pyramiding strategy"""
    
    # Initialize data client
    api_key = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
    polygon_client = RESTClient(api_key)
    
    # Initialize indicators
    indicator_2m = DualDeviationCloud({
        'ema_fast_length': 9,
        'ema_slow_length': 20,
        'positive_dev_1': 1.0,
        'positive_dev_2': 0.5,
        'negative_dev_1': 2.0,
        'negative_dev_2': 2.4
    })
    
    symbol = "IBIT"
    
    print("Downloading data for August 19th pyramiding chart...")
    
    # Get extended data for proper EMA calculation
    aggs_2m = []
    for agg in polygon_client.get_aggs(
        ticker=symbol,
        multiplier=2,
        timespan='minute',
        from_='2025-08-16',
        to='2025-08-19',
        adjusted=True,
        sort='asc',
        limit=50000
    ):
        aggs_2m.append({
            'timestamp': pd.Timestamp(agg.timestamp, unit='ms', tz='America/New_York'),
            'open': agg.open,
            'high': agg.high,
            'low': agg.low,
            'close': agg.close,
            'volume': agg.volume
        })
    
    # Convert to DataFrame
    df_2m = pd.DataFrame(aggs_2m)
    df_2m.set_index('timestamp', inplace=True)
    df_2m.sort_index(inplace=True)
    
    # Calculate indicators
    indicators_2m = indicator_2m.calculate(df_2m)
    
    # Add TA-Lib indicators
    indicators_2m['rsi_14'] = ta.momentum.RSIIndicator(df_2m['close'], window=14).rsi()
    indicators_2m['rsi_7'] = ta.momentum.RSIIndicator(df_2m['close'], window=7).rsi()
    indicators_2m['volume_sma'] = df_2m['volume'].rolling(window=20).mean()
    indicators_2m['volume_ratio'] = df_2m['volume'] / indicators_2m['volume_sma']
    
    # Filter to August 19th only
    aug_19 = pd.Timestamp('2025-08-19').date()
    aug_19_data = df_2m[df_2m.index.date == aug_19]
    aug_19_indicators = indicators_2m[indicators_2m.index.date == aug_19]
    
    print("Creating August 19th pyramiding chart...")
    
    # Set up the plot
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16), gridspec_kw={'height_ratios': [4, 1, 1]})
    
    # Plot 2M candlesticks
    for timestamp, row in aug_19_data.iterrows():
        color = 'lime' if row['close'] >= row['open'] else 'red'
        # High-low line
        ax1.plot([timestamp, timestamp], [row['low'], row['high']], 
                color='white', linewidth=0.5, alpha=0.7)
        # Open-close body
        ax1.plot([timestamp, timestamp], [row['open'], row['close']], 
                color=color, linewidth=2, alpha=0.8)
    
    # Plot EMAs and deviation bands
    ax1.plot(aug_19_indicators.index, aug_19_indicators['fast_ema'], 
            color='cyan', label='2M EMA9', linewidth=2, alpha=0.9)
    ax1.plot(aug_19_indicators.index, aug_19_indicators['slow_ema'], 
            color='yellow', label='2M EMA20', linewidth=2, alpha=0.9)
    
    # Deviation bands
    ax1.plot(aug_19_indicators.index, aug_19_indicators['upper_band_1'], 
            color='orange', label='Upper Band 1', linewidth=1.5, alpha=0.8)
    ax1.plot(aug_19_indicators.index, aug_19_indicators['lower_band_1'], 
            color='purple', label='Lower Band 1', linewidth=1.5, alpha=0.8)
    ax1.plot(aug_19_indicators.index, aug_19_indicators['upper_band_2'], 
            color='red', label='Upper Band 2', linewidth=1, alpha=0.6)
    ax1.plot(aug_19_indicators.index, aug_19_indicators['lower_band_2'], 
            color='blue', label='Lower Band 2', linewidth=1, alpha=0.6)
    
    # Fill between bands
    ax1.fill_between(aug_19_indicators.index, aug_19_indicators['upper_band_1'], 
                    aug_19_indicators['lower_band_1'], alpha=0.1, color='gray')
    
    # === MARK KEY EVENTS ===
    
    # Route Start (9:00 AM)
    route_start_time = pd.Timestamp('2025-08-19 09:00:00', tz='America/New_York')
    ax1.axvline(route_start_time, color='gold', linestyle='--', linewidth=2, alpha=0.8)
    ax1.annotate('🚀 ROUTE START\n9:00 AM', xy=(route_start_time, 65.6),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='gold'))
    
    # 5M Bearish Cross (9:36 AM) - Session Start
    session_start_time = pd.Timestamp('2025-08-19 09:36:00', tz='America/New_York')
    ax1.axvline(session_start_time, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax1.annotate('📉 5M BEARISH CROSS\nSCALPING START\n9:36 AM',
                xy=(session_start_time, 65.5),
                xytext=(10, 30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # 5M Bullish Cross (3:36 PM) - Session End
    session_end_time = pd.Timestamp('2025-08-19 15:36:00', tz='America/New_York')
    ax1.axvline(session_end_time, color='lime', linestyle='--', linewidth=2, alpha=0.8)
    ax1.annotate('📈 5M BULLISH CROSS\nSCALPING END\n3:36 PM',
                xy=(session_end_time, 64.2),
                xytext=(-80, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lime', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='lime'))
    
    # === ENHANCED PYRAMIDING STRATEGY ENTRIES ===
    
    # Define the enhanced strategy positions with break of lows entries
    pyramid_positions = [
        # Early entries
        {'entry': '09:36', 'entry_price': 65.61, 'size': 1200, 'type': 'bearish_cross', 'level': 1},
        {'entry': '09:54', 'entry_price': 65.44, 'size': 1500, 'type': 'ema20_pop_pyramid', 'level': 2},
        
        # NEW: Break of lows entries (after initial weakness)
        {'entry': '10:20', 'entry_price': 65.12, 'size': 1300, 'type': 'break_of_lows', 'level': 3},
        {'entry': '10:35', 'entry_price': 64.95, 'size': 1400, 'type': 'close_below_lows', 'level': 4},
        
        # Deviation band pyramid entries (now levels 5-9)
        {'entry': '11:20', 'entry_price': 64.67, 'size': 500, 'type': 'dev_band_1', 'level': 5},
        {'entry': '11:22', 'entry_price': 64.73, 'size': 400, 'type': 'dev_band_1', 'level': 6},
        {'entry': '11:24', 'entry_price': 64.72, 'size': 320, 'type': 'dev_band_1', 'level': 7},
        {'entry': '11:26', 'entry_price': 64.77, 'size': 256, 'type': 'dev_band_1', 'level': 8},
        {'entry': '11:28', 'entry_price': 64.73, 'size': 307, 'type': 'dev_band_2', 'level': 9},
        
        # All covered at 15:36 on 5M bullish cross (trend break)
        {'exit': '15:36', 'exit_price': 64.21, 'exit_reason': '5m_bullish_cross_exit'}
    ]
    
    # Calculate total pyramid P&L
    total_pnl = 0
    total_size = 0
    
    # Entry colors for different entry types and pyramid levels
    pyramid_colors = ['darkred', 'red', 'blue', 'navy', 'orange', 'yellow', 'pink', 'magenta', 'violet']
    entry_type_colors = {
        'bearish_cross': 'darkred',
        'ema20_pop_pyramid': 'red', 
        'break_of_lows': 'blue',
        'close_below_lows': 'navy',
        'dev_band_1': 'orange',
        'dev_band_2': 'yellow'
    }
    
    # Plot pyramid entries
    for i, pos in enumerate(pyramid_positions[:-1]):  # Exclude exit info
        entry_time = pd.Timestamp(f'2025-08-19 {pos["entry"]}:00', tz='America/New_York')
        
        # Entry marker - color by type, size scales with position size
        color = entry_type_colors.get(pos['type'], pyramid_colors[pos['level']-1])
        marker_size = 100 + (pos['size'] / 5)  # Scale with position size
        
        # Different markers for different entry types
        if pos['type'] == 'bearish_cross':
            marker = '>'
        elif pos['type'] == 'ema20_pop_pyramid':
            marker = 's'
        elif pos['type'] == 'break_of_lows':
            marker = '<'
        elif pos['type'] == 'close_below_lows':
            marker = 'D'
        else:
            marker = 'v'
        
        ax1.scatter(entry_time, pos['entry_price'], 
                   color=color, s=marker_size, marker=marker,
                   zorder=10, edgecolor='white', linewidth=1.5, alpha=0.9)
        
        # Calculate stop loss level for this entry type
        if pos['type'] == 'bearish_cross':
            stop_level = pos['entry_price'] + 0.25  # 25¢ stop above cross entry
        elif pos['type'] == 'ema20_pop_pyramid':
            stop_level = pos['entry_price'] + 0.20  # 20¢ stop above EMA20 pop
        elif pos['type'] == 'break_of_lows':
            stop_level = pos['entry_price'] + 0.18  # 18¢ stop above break of lows
        elif pos['type'] == 'close_below_lows':
            stop_level = pos['entry_price'] + 0.15  # 15¢ stop above close below
        else:  # dev_band entries
            stop_level = pos['entry_price'] + 0.12  # 12¢ stop above dev band entries
            
        # Draw stop loss line
        ax1.axhline(stop_level, color=color, linestyle=':', alpha=0.6, linewidth=1,
                   xmin=0.02, xmax=0.98)
        
        # Add entry label with stop info
        risk_per_share = stop_level - pos['entry_price']
        total_risk = risk_per_share * pos['size']
        
        ax1.annotate(f"#{pos['level']}\n{pos['size']} shares\nStop: ${stop_level:.2f}\nRisk: ${total_risk:.0f}", 
                    xy=(entry_time, pos['entry_price']),
                    xytext=(5, -35), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7),
                    ha='center', fontsize=8, fontweight='bold')
        
        # Calculate P&L for this position
        exit_price = pyramid_positions[-1]['exit_price']
        pnl = (pos['entry_price'] - exit_price) * pos['size']
        total_pnl += pnl
        total_size += pos['size']
    
    # Plot single exit for all positions
    exit_info = pyramid_positions[-1]
    exit_time = pd.Timestamp(f'2025-08-19 {exit_info["exit"]}:00', tz='America/New_York')
    
    ax1.scatter(exit_time, exit_info['exit_price'],
               color='lime', s=200, marker='^',
               zorder=10, edgecolor='white', linewidth=2, alpha=0.9)
    
    # Draw lines from each entry to the single exit
    for pos in pyramid_positions[:-1]:
        entry_time = pd.Timestamp(f'2025-08-19 {pos["entry"]}:00', tz='America/New_York')
        ax1.plot([entry_time, exit_time], 
                [pos['entry_price'], exit_info['exit_price']],
                color='lime', linewidth=2, alpha=0.6)
    
    # Add exit annotation at trend break
    ax1.annotate(f'🔄 TREND BREAK EXIT\n5M Bullish Cross\nAll Positions Covered\n${total_pnl:.0f} Total P&L', 
                xy=(exit_time, exit_info['exit_price']),
                xytext=(-60, 30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lime', alpha=0.9),
                ha='center', fontweight='bold', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='lime', lw=2))
    
    # Highlight active scalping session
    ax1.axvspan(session_start_time, session_end_time, alpha=0.1, color='yellow', label='Active Session')
    
    # RSI subplot
    ax2.plot(aug_19_indicators.index, aug_19_indicators['rsi_14'], 
            color='cyan', label='RSI(14)', linewidth=1.5)
    ax2.plot(aug_19_indicators.index, aug_19_indicators['rsi_7'], 
            color='orange', label='RSI(7)', linewidth=1.5)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought')
    ax2.axhline(30, color='lime', linestyle='--', alpha=0.7, label='Oversold') 
    ax2.axhline(50, color='gray', linestyle='-', alpha=0.5)
    ax2.set_ylim(20, 80)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylabel('RSI')
    
    # Volume subplot
    colors = ['lime' if c >= o else 'red' for o, c in zip(aug_19_data['open'], aug_19_data['close'])]
    ax3.bar(aug_19_data.index, aug_19_data['volume'], color=colors, alpha=0.7,
            width=pd.Timedelta(minutes=1.5))
    
    # Volume ratio overlay
    ax3_twin = ax3.twinx()
    ax3_twin.plot(aug_19_indicators.index, aug_19_indicators['volume_ratio'],
                 color='white', linewidth=1, alpha=0.8, label='Vol Ratio')
    ax3_twin.axhline(1.0, color='gray', linestyle='-', alpha=0.5)
    ax3_twin.set_ylabel('Volume Ratio', color='white', fontsize=10)
    
    # Formatting
    ax1.set_title('IBIT August 19th, 2025 - Enhanced Pyramiding Scalping Strategy\n' +
                  f'9-Level Pyramid: {total_size} Shares | P&L: ${total_pnl:.0f} | Break of Lows + Trend Exit', 
                  fontsize=16, pad=20, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax3.set_ylabel('Volume', fontsize=12)
    ax3.set_xlabel('Time (EST)', fontsize=12)
    
    # Format x-axis
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
        
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
    
    # Add strategy summary
    summary_text = (
        "ENHANCED PYRAMIDING STRATEGY:\n"
        "• Route Start: 9:00 AM\n"
        "• Session: 9:36 AM - 3:36 PM\n"
        f"• Total Pyramid Levels: {len(pyramid_positions)-1}\n"
        "• Entry #1: Bearish Cross @ 9:36 AM (►)\n"
        "• Entry #2: EMA20 Pop @ 9:54 AM (■)\n"
        "• Entry #3: Break of Lows @ 10:20 AM (◄)\n"
        "• Entry #4: Close Below @ 10:35 AM (♦)\n"
        "• Entries #5-9: Dev Bands 11:20-11:28 (▼)\n"
        f"• Total Position: {total_size} shares\n"
        f"• Total P&L: ${total_pnl:.0f}\n"
        "• Exit: 5M Bullish Cross @ 3:36 PM\n"
        "• Risk Mgmt: Stop Lines (25¢/20¢/15¢)"
    )
    
    ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
            facecolor='black', alpha=0.8, edgecolor='white'))
    
    plt.tight_layout()
    
    # Save chart
    chart_path = "/Users/michaeldurante/sm-playbook/trading-code/backtesting/August_19_2025_Enhanced_Pyramiding_Chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    print(f"August 19th Enhanced Pyramiding Chart saved to: {chart_path}")
    return chart_path

if __name__ == "__main__":
    create_august_19_pyramiding_chart()