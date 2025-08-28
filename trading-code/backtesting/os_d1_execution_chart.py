#!/usr/bin/env python3
"""
OS D1 Execution Chart - Clean visualization of backtest results
Uses the user's preferred matplotlib styling
"""

import matplotlib
matplotlib.use("Qt5Agg")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

plt.style.use('dark_background')

def create_execution_chart():
    """Create clean execution chart for OS D1 backtest results"""
    
    # Load the results
    try:
        trades_df = pd.read_csv('os_d1_complete_backtest_results.csv')
        setups_df = pd.read_csv('os_d1_scanned_setups.csv')
    except FileNotFoundError:
        print("❌ Results files not found. Please run the backtest first.")
        return
    
    # Convert dates
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    setups_df['date'] = pd.to_datetime(setups_df['date'])
    
    # Create main figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('OS D1 Strategy Backtest Execution Results', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # Chart 1: Cumulative P&L
    trades_sorted = trades_df.sort_values('date')
    trades_sorted['cumulative_pnl'] = trades_sorted['pnl_r'].cumsum()
    
    ax1.plot(range(len(trades_sorted)), trades_sorted['cumulative_pnl'], 
             linewidth=4, marker='o', markersize=8, color='#00ff88', 
             markerfacecolor='white', markeredgecolor='#00ff88', markeredgewidth=2)
    ax1.axhline(y=0, color='white', linestyle='--', alpha=0.6, linewidth=2)
    ax1.set_title('Cumulative P&L', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('P&L (R)', fontsize=14)
    ax1.set_xlabel('Trade Number', fontsize=14)
    ax1.grid(True, alpha=0.3, linewidth=1)
    
    # Add final P&L annotation
    final_pnl = trades_sorted['cumulative_pnl'].iloc[-1]
    ax1.text(0.05, 0.95, f'Final P&L: {final_pnl:.2f}R', 
             transform=ax1.transAxes, fontsize=14, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#00aa44', alpha=0.8))
    
    # Chart 2: Individual Trade Results
    colors = ['#00ff88' if x > 0 else '#ff4444' for x in trades_sorted['pnl_r']]
    bars = ax2.bar(range(len(trades_sorted)), trades_sorted['pnl_r'], 
                   color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    ax2.axhline(y=0, color='white', linestyle='--', alpha=0.6, linewidth=2)
    ax2.set_title('Individual Trade P&L', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('P&L (R)', fontsize=14)
    ax2.set_xlabel('Trade Number', fontsize=14)
    ax2.grid(True, alpha=0.3, linewidth=1)
    
    # Add ticker labels on bars
    for i, (bar, ticker, pnl) in enumerate(zip(bars, trades_sorted['ticker'], trades_sorted['pnl_r'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.1),
                f'{ticker}\n{pnl:.2f}R', ha='center', 
                va='bottom' if height >= 0 else 'top', 
                fontweight='bold', fontsize=10)
    
    # Chart 3: Setup Types Performance
    entry_stats = trades_df.groupby('spot_type').agg({
        'pnl_r': ['count', lambda x: (x > 0).sum(), 'sum', 'mean']
    }).round(2)
    entry_stats.columns = ['Total', 'Winners', 'Total_PnL', 'Avg_PnL']
    entry_stats['Win_Rate'] = (entry_stats['Winners'] / entry_stats['Total'] * 100).round(1)
    
    # Create grouped bar chart
    x_pos = np.arange(len(entry_stats))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, entry_stats['Win_Rate'], width, 
                   label='Win Rate (%)', color='#00aaff', alpha=0.8)
    bars2 = ax3.bar(x_pos + width/2, entry_stats['Avg_PnL']*20, width,  # Scale for visibility
                   label='Avg P&L (×20)', color='#ff8800', alpha=0.8)
    
    ax3.set_title('Performance by Entry Type', fontsize=16, fontweight='bold', pad=20)
    ax3.set_ylabel('Win Rate (%) / Scaled Avg P&L', fontsize=14)
    ax3.set_xlabel('Entry Type', fontsize=14)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(entry_stats.index, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, linewidth=1)
    
    # Add value labels
    for bar, rate in zip(bars1, entry_stats['Win_Rate']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Chart 4: Gap vs Performance Scatter
    winners_mask = trades_df['pnl_r'] > 0
    
    # Plot winners
    scatter1 = ax4.scatter(trades_df[winners_mask]['gap_pct'], 
                          trades_df[winners_mask]['pnl_r'],
                          s=150, color='#00ff88', alpha=0.8, 
                          edgecolors='white', linewidth=2, label='Winners')
    
    # Plot losers
    scatter2 = ax4.scatter(trades_df[~winners_mask]['gap_pct'], 
                          trades_df[~winners_mask]['pnl_r'],
                          s=150, color='#ff4444', alpha=0.8, 
                          edgecolors='white', linewidth=2, label='Losers')
    
    # Add ticker labels to points
    for _, trade in trades_df.iterrows():
        ax4.annotate(trade['ticker'], 
                    (trade['gap_pct'], trade['pnl_r']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', alpha=0.8)
    
    ax4.axhline(y=0, color='white', linestyle='--', alpha=0.6, linewidth=2)
    ax4.set_title('Gap % vs Trade P&L', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xlabel('Gap %', fontsize=14)
    ax4.set_ylabel('P&L (R)', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3, linewidth=1)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    # Save the chart
    chart_path = '/Users/michaeldurante/sm-playbook/trading-code/backtesting/OS_D1_Execution_Chart.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()
    
    print(f"✅ Execution chart saved: {chart_path}")
    
    # Print performance summary
    print("\n" + "="*70)
    print("📊 OS D1 BACKTEST EXECUTION SUMMARY")
    print("="*70)
    
    total_trades = len(trades_df)
    total_setups = len(setups_df)
    total_pnl = trades_df['pnl_r'].sum()
    avg_pnl = trades_df['pnl_r'].mean()
    win_rate = (trades_df['pnl_r'] > 0).mean()
    winners = trades_df[trades_df['pnl_r'] > 0]['pnl_r']
    losers = trades_df[trades_df['pnl_r'] < 0]['pnl_r']
    
    print(f"🎯 SCANNING RESULTS:")
    print(f"   • Total OS D1 setups found: {total_setups}")
    print(f"   • Trades executed: {total_trades}")
    print(f"   • Execution rate: {total_trades/total_setups:.1%}")
    
    print(f"\n📈 TRADE PERFORMANCE:")
    print(f"   • Total P&L: {total_pnl:.2f}R")
    print(f"   • Average P&L per trade: {avg_pnl:.2f}R")
    print(f"   • Win rate: {win_rate:.1%}")
    if not winners.empty:
        print(f"   • Average winner: {winners.mean():.2f}R")
        print(f"   • Best trade: {winners.max():.2f}R")
    if not losers.empty:
        print(f"   • Average loser: {losers.mean():.2f}R")
        print(f"   • Worst trade: {losers.min():.2f}R")
        print(f"   • Profit factor: {abs(winners.mean()/losers.mean()):.2f}")
    
    print(f"\n🎯 ENTRY TYPE BREAKDOWN:")
    for entry_type, stats in entry_stats.iterrows():
        print(f"   • {entry_type}: {stats['Total']} trades, "
              f"{stats['Win_Rate']:.0f}% win rate, "
              f"{stats['Avg_PnL']:.2f}R avg")
    
    print(f"\n📋 INDIVIDUAL TRADES:")
    for i, (_, trade) in enumerate(trades_sorted.iterrows(), 1):
        status = "🟢 WIN " if trade['pnl_r'] > 0 else "🔴 LOSS"
        print(f"   {i:2d}. {trade['ticker']} ({trade['date'].strftime('%m/%d')}) - "
              f"{status} {trade['pnl_r']:+.2f}R | "
              f"{trade['spot_type']} | Gap: {trade['gap_pct']:.0f}%")

if __name__ == '__main__':
    create_execution_chart()