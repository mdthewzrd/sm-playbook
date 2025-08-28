#!/usr/bin/env python3

"""Create detailed chart for August 19th scalping opportunities"""

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

def create_august_19_scalping_chart():
    """Create detailed chart showing August 19th scalping opportunities"""
    
    # Initialize data client
    api_key = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
    polygon_client = RESTClient(api_key)
    
    # Initialize indicators
    indicator_calc = DualDeviationCloud({
        'ema_fast_length': 9,
        'ema_slow_length': 20,
        'positive_dev_1': 1.0,
        'positive_dev_2': 0.5,
        'negative_dev_1': 2.0,
        'negative_dev_2': 2.4
    })
    
    symbol = "IBIT"
    
    print("Downloading data for August 19th chart...")
    
    # Get extended data for proper EMA calculation
    aggs_5m = []
    for agg in polygon_client.get_aggs(
        ticker=symbol,
        multiplier=5,
        timespan='minute',
        from_='2025-08-16',
        to='2025-08-19',
        adjusted=True,
        sort='asc',
        limit=50000
    ):
        aggs_5m.append({
            'timestamp': pd.Timestamp(agg.timestamp, unit='ms', tz='America/New_York'),
            'open': agg.open,
            'high': agg.high,
            'low': agg.low,
            'close': agg.close,
            'volume': agg.volume
        })
    
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
    
    aggs_1h = []
    for agg in polygon_client.get_aggs(
        ticker=symbol,
        multiplier=60,
        timespan='minute',
        from_='2025-08-16',
        to='2025-08-19',
        adjusted=True,
        sort='asc',
        limit=50000
    ):
        aggs_1h.append({
            'timestamp': pd.Timestamp(agg.timestamp, unit='ms', tz='America/New_York'),
            'open': agg.open,
            'high': agg.high,
            'low': agg.low,
            'close': agg.close,
            'volume': agg.volume
        })
    
    # Convert to DataFrames
    df_5m = pd.DataFrame(aggs_5m)
    df_5m.set_index('timestamp', inplace=True)
    df_5m.sort_index(inplace=True)
    
    df_2m = pd.DataFrame(aggs_2m)
    df_2m.set_index('timestamp', inplace=True)
    df_2m.sort_index(inplace=True)
    
    df_1h = pd.DataFrame(aggs_1h)
    df_1h.set_index('timestamp', inplace=True)
    df_1h.sort_index(inplace=True)
    
    # Calculate indicators
    indicators_5m = indicator_calc.calculate(df_5m)
    indicators_2m = indicator_calc.calculate(df_2m)
    indicators_1h = indicator_calc.calculate(df_1h)
    
    # Filter to August 19th
    aug_19 = pd.Timestamp('2025-08-19').date()
    
    aug_19_5m = indicators_5m[indicators_5m.index.date == aug_19]
    aug_19_2m = indicators_2m[indicators_2m.index.date == aug_19]
    
    print("Creating chart...")
    
    # Set up the plot
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14), gridspec_kw={'height_ratios': [4, 1]})
    
    # Plot 2M candlesticks
    for timestamp, row in aug_19_2m.iterrows():
        color = 'lime' if row['close'] >= row['open'] else 'red'
        # High-low line
        ax1.plot([timestamp, timestamp], [row['low'], row['high']], color='white', linewidth=0.5, alpha=0.7)
        # Open-close body
        ax1.plot([timestamp, timestamp], [row['open'], row['close']], color=color, linewidth=2, alpha=0.8)
    
    # Plot EMAs and deviation bands (using 2M data for detail)
    ax1.plot(aug_19_2m.index, aug_19_2m['fast_ema'], color='cyan', label='2M EMA9', linewidth=2, alpha=0.9)
    ax1.plot(aug_19_2m.index, aug_19_2m['slow_ema'], color='yellow', label='2M EMA20', linewidth=2, alpha=0.9)
    
    # Deviation bands
    ax1.plot(aug_19_2m.index, aug_19_2m['upper_band_1'], color='orange', label='Upper Band 1', linewidth=1.5, alpha=0.8)
    ax1.plot(aug_19_2m.index, aug_19_2m['lower_band_1'], color='purple', label='Lower Band 1', linewidth=1.5, alpha=0.8)
    ax1.plot(aug_19_2m.index, aug_19_2m['upper_band_2'], color='red', label='Upper Band 2', linewidth=1, alpha=0.6)
    ax1.plot(aug_19_2m.index, aug_19_2m['lower_band_2'], color='blue', label='Lower Band 2', linewidth=1, alpha=0.6)
    
    # Fill between bands
    ax1.fill_between(aug_19_2m.index, aug_19_2m['upper_band_1'], aug_19_2m['lower_band_1'], 
                     alpha=0.1, color='gray', label='Deviation Channel')
    
    # === MARK KEY EVENTS ===
    
    # 1. Route Start (9:00 AM)
    route_start_time = pd.Timestamp('2025-08-19 09:00:00', tz='America/New_York')
    if route_start_time in aug_19_5m.index:
        route_price = aug_19_5m.loc[route_start_time]['high']
        ax1.axvline(route_start_time, color='gold', linestyle='--', linewidth=2, alpha=0.8, label='Route Start (9:00 AM)')
        ax1.annotate('🚀 ROUTE START\n9:00 AM', xy=(route_start_time, route_price), 
                    xytext=(10, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='gold'))
    
    # 2. 5M Bearish Cross (9:35 AM) - Session Start
    bearish_cross_time = pd.Timestamp('2025-08-19 09:35:00', tz='America/New_York')
    if bearish_cross_time in aug_19_5m.index:
        cross_price = aug_19_5m.loc[bearish_cross_time]['close']
        ax1.axvline(bearish_cross_time, color='red', linestyle='--', linewidth=2, alpha=0.8, label='5M Bearish Cross')
        ax1.annotate('📉 5M BEARISH CROSS\nSCALPING START\n9:35 AM', 
                    xy=(bearish_cross_time, cross_price), 
                    xytext=(10, 30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    # 3. 5M Bullish Cross (3:35 PM) - Session End
    bullish_cross_time = pd.Timestamp('2025-08-19 15:35:00', tz='America/New_York')
    if bullish_cross_time in aug_19_5m.index:
        cross_price = aug_19_5m.loc[bullish_cross_time]['close']
        ax1.axvline(bullish_cross_time, color='lime', linestyle='--', linewidth=2, alpha=0.8, label='5M Bullish Cross')
        ax1.annotate('📈 5M BULLISH CROSS\nSCALPING END\n3:35 PM', 
                    xy=(bullish_cross_time, cross_price), 
                    xytext=(-80, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lime', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='lime'))
    
    # === MARK SCALPING OPPORTUNITIES - ENHANCED STRATEGY RESULTS ===
    
    # Define the actual scalps from enhanced strategy
    scalps = [
        {'entry': '09:54', 'exit': '12:28', 'entry_price': 65.44, 'exit_price': 64.46, 'pnl': 979.00, 'type': 'EMA20_POP'},
        {'entry': '12:36', 'exit': '12:36', 'entry_price': 64.50, 'exit_price': 64.50, 'pnl': 0.00, 'type': 'DEV_BAND'},
        {'entry': '12:38', 'exit': '12:38', 'entry_price': 64.34, 'exit_price': 64.34, 'pnl': 0.00, 'type': 'EMA20_POP'},
        {'entry': '12:52', 'exit': '14:20', 'entry_price': 64.42, 'exit_price': 64.27, 'pnl': 148.70, 'type': 'EMA20_POP'},
        {'entry': '14:24', 'exit': '15:36', 'entry_price': 64.31, 'exit_price': 64.38, 'pnl': -70.00, 'type': 'EMA20_POP'}
    ]
    
    for i, scalp in enumerate(scalps, 1):
        entry_time = pd.Timestamp(f'2025-08-19 {scalp["entry"]}:00', tz='America/New_York')
        exit_time = pd.Timestamp(f'2025-08-19 {scalp["exit"]}:00', tz='America/New_York')
        
        # Entry marker (red down arrow, different shapes for entry types)
        entry_marker = 'v' if scalp['type'] == 'DEV_BAND' else 's'  # Triangle for dev band, square for EMA20
        entry_color = 'red' if scalp['type'] == 'DEV_BAND' else 'orange'
        ax1.scatter(entry_time, scalp['entry_price'], color=entry_color, s=150, marker=entry_marker, 
                   zorder=10, edgecolor='white', linewidth=1)
        
        # Exit marker (green up arrow for profit, red for loss)
        exit_color = 'lime' if scalp['pnl'] > 0 else 'red'
        ax1.scatter(exit_time, scalp['exit_price'], color=exit_color, s=150, marker='^', 
                   zorder=10, edgecolor='white', linewidth=1)
        
        # Draw line between entry and exit
        ax1.plot([entry_time, exit_time], [scalp['entry_price'], scalp['exit_price']], 
                color=exit_color, linewidth=3, alpha=0.7, linestyle='-')
        
        # Add P&L label
        mid_time = entry_time + (exit_time - entry_time) / 2
        mid_price = (scalp['entry_price'] + scalp['exit_price']) / 2
        pnl_text = f"${scalp['pnl']:.0f}"
        ax1.annotate(pnl_text, xy=(mid_time, mid_price), 
                    xytext=(0, 15), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=exit_color, alpha=0.8),
                    ha='center', fontweight='bold')
    
    # Highlight scalping session period
    session_start = pd.Timestamp('2025-08-19 09:35:00', tz='America/New_York')
    session_end = pd.Timestamp('2025-08-19 15:35:00', tz='America/New_York')
    ax1.axvspan(session_start, session_end, alpha=0.1, color='yellow', label='Active Scalping Session')
    
    # Highlight market hours
    market_open = pd.Timestamp('2025-08-19 09:30:00', tz='America/New_York')
    market_close = pd.Timestamp('2025-08-19 16:00:00', tz='America/New_York')
    ax1.axvspan(market_open, market_close, alpha=0.05, color='white', label='Market Hours')
    
    # Volume subplot
    volume_data = aug_19_2m.copy()
    colors = ['lime' if c >= o else 'red' for o, c in zip(volume_data['open'], volume_data['close'])]
    ax2.bar(volume_data.index, volume_data['volume'], color=colors, alpha=0.7, width=pd.Timedelta(minutes=1.5))
    
    # Formatting
    ax1.set_title('IBIT August 19th, 2025 - Session-Based Scalping Opportunities\n' +
                  'Route Start → 5M Cross → Deviation Band Scalps → Session End', 
                  fontsize=16, pad=20, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Time (EST)', fontsize=12)
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
        
        # Rotate labels
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
    
    # Add summary text box
    summary_text = (
        "ENHANCED SCALPING SESSION RESULTS:\n"
        "• Route Start: 9:00 AM\n"
        "• Session: 9:36 AM - 3:36 PM (6 hours)\n"
        "• Total Scalps: 5 opportunities\n"
        "• Win Rate: 60% (3W/2L)\n"
        "• Total P&L: +$1,057.70\n"
        "• Big Morning Entry: 9:54 AM (+$979)\n"
        "• Entry Types: DEV_BAND (▼) vs EMA20_POP (■)"
    )
    
    ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='black', alpha=0.8, edgecolor='white'))
    
    plt.tight_layout()
    
    # Save chart
    chart_path = "/Users/michaeldurante/sm-playbook/trading-code/backtesting/August_19_2025_Scalping_Opportunities.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Chart saved to: {chart_path}")
    
    return chart_path

if __name__ == "__main__":
    create_august_19_scalping_chart()