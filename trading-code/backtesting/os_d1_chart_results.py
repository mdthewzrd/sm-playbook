#!/usr/bin/env python3
"""
OS D1 Backtest Results Visualization
Creates comprehensive charts showing execution results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Set style
plt.style.use('dark_background')

def create_execution_charts():
    """Create comprehensive charts of OS D1 backtest execution"""
    
    # Load the results
    try:
        trades_df = pd.read_csv('os_d1_complete_backtest_results.csv')
        setups_df = pd.read_csv('os_d1_scanned_setups.csv')
    except FileNotFoundError:
        print("❌ Results files not found. Please run the backtest first.")
        return
    
    # Convert date columns
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    setups_df['date'] = pd.to_datetime(setups_df['date'])
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('OS D1 Strategy Backtest Execution Results', fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Cumulative P&L Chart
    ax1 = plt.subplot(3, 3, 1)
    trades_df_sorted = trades_df.sort_values('date')
    trades_df_sorted['cumulative_pnl'] = trades_df_sorted['pnl_r'].cumsum()
    
    ax1.plot(trades_df_sorted['date'], trades_df_sorted['cumulative_pnl'], 
             linewidth=3, marker='o', markersize=8, color='#00ff88')
    ax1.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax1.set_title('Cumulative P&L Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('P&L (R)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add final P&L text
    final_pnl = trades_df_sorted['cumulative_pnl'].iloc[-1]
    ax1.text(0.05, 0.95, f'Final P&L: {final_pnl:.2f}R', 
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7))
    
    # 2. Trade P&L Distribution
    ax2 = plt.subplot(3, 3, 2)
    winners = trades_df[trades_df['pnl_r'] > 0]['pnl_r']
    losers = trades_df[trades_df['pnl_r'] < 0]['pnl_r']
    
    ax2.hist([winners, losers], bins=10, label=['Winners', 'Losers'], 
             color=['#00ff88', '#ff4444'], alpha=0.8, edgecolor='white')
    ax2.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('P&L (R)', fontsize=12)
    ax2.set_ylabel('Number of Trades', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Win Rate by Entry Type
    ax3 = plt.subplot(3, 3, 3)
    entry_stats = trades_df.groupby('spot_type').agg({
        'pnl_r': ['count', lambda x: (x > 0).sum(), 'mean']
    }).round(2)
    entry_stats.columns = ['Total_Trades', 'Winners', 'Avg_PnL']
    entry_stats['Win_Rate'] = (entry_stats['Winners'] / entry_stats['Total_Trades'] * 100).round(1)
    
    colors = ['#00ff88', '#ff8800', '#8888ff']
    bars = ax3.bar(entry_stats.index, entry_stats['Win_Rate'], color=colors, alpha=0.8)
    ax3.set_title('Win Rate by Entry Type', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Win Rate (%)', fontsize=12)
    ax3.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, rate in zip(bars, entry_stats['Win_Rate']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate}%', ha='center', va='bottom', fontweight='bold')
    
    ax3.grid(True, alpha=0.3)
    
    # 4. Daily Setups Found
    ax4 = plt.subplot(3, 3, 4)
    daily_setups = setups_df.groupby('date').size()
    
    ax4.bar(daily_setups.index, daily_setups.values, color='#00aaff', alpha=0.8)
    ax4.set_title('OS D1 Setups Found by Date', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of Setups', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. Gap vs PM High Scatter
    ax5 = plt.subplot(3, 3, 5)
    winners_mask = trades_df['pnl_r'] > 0
    
    ax5.scatter(trades_df[winners_mask]['gap_pct'], trades_df[winners_mask]['pm_high_pct'], 
               s=abs(trades_df[winners_mask]['pnl_r']) * 100, color='#00ff88', 
               alpha=0.7, label='Winners', edgecolors='white')
    ax5.scatter(trades_df[~winners_mask]['gap_pct'], trades_df[~winners_mask]['pm_high_pct'], 
               s=abs(trades_df[~winners_mask]['pnl_r']) * 100, color='#ff4444', 
               alpha=0.7, label='Losers', edgecolors='white')
    
    ax5.set_title('Gap % vs PM High % (Size = |P&L|)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Gap %', fontsize=12)
    ax5.set_ylabel('PM High %', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Trade Timeline
    ax6 = plt.subplot(3, 3, 6)
    for i, (_, trade) in enumerate(trades_df_sorted.iterrows()):
        color = '#00ff88' if trade['pnl_r'] > 0 else '#ff4444'
        ax6.barh(i, trade['pnl_r'], color=color, alpha=0.8)
        ax6.text(trade['pnl_r'] + 0.05, i, f"{trade['ticker']}", 
                va='center', fontweight='bold', fontsize=10)
    
    ax6.set_title('Individual Trade Results', fontsize=14, fontweight='bold')
    ax6.set_xlabel('P&L (R)', fontsize=12)
    ax6.set_ylabel('Trade #', fontsize=12)
    ax6.axvline(x=0, color='white', linestyle='--', alpha=0.5)
    ax6.grid(True, alpha=0.3)
    
    # 7. Performance Metrics Table
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('off')
    
    # Calculate key metrics
    total_trades = len(trades_df)
    total_pnl = trades_df['pnl_r'].sum()
    avg_pnl = trades_df['pnl_r'].mean()
    win_rate = (trades_df['pnl_r'] > 0).mean()
    avg_winner = trades_df[trades_df['pnl_r'] > 0]['pnl_r'].mean()
    avg_loser = trades_df[trades_df['pnl_r'] < 0]['pnl_r'].mean()
    max_winner = trades_df['pnl_r'].max()
    max_loser = trades_df['pnl_r'].min()
    
    metrics_text = f"""
    PERFORMANCE METRICS
    {'='*25}
    Total Trades: {total_trades}
    Total Setups Found: {len(setups_df)}
    
    Total P&L: {total_pnl:.2f}R
    Average P&L: {avg_pnl:.2f}R
    Win Rate: {win_rate:.1%}
    
    Average Winner: {avg_winner:.2f}R
    Average Loser: {avg_loser:.2f}R
    Best Trade: {max_winner:.2f}R
    Worst Trade: {max_loser:.2f}R
    
    Profit Factor: {abs(avg_winner/avg_loser):.2f}
    """
    
    ax7.text(0.1, 0.9, metrics_text, transform=ax7.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#333333', alpha=0.8))
    
    # 8. Entry Time Distribution
    ax8 = plt.subplot(3, 3, 8)
    trades_df['entry_hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
    trades_df['entry_minute'] = pd.to_datetime(trades_df['entry_time']).dt.minute
    trades_df['entry_decimal'] = trades_df['entry_hour'] + trades_df['entry_minute']/60
    
    ax8.hist(trades_df['entry_decimal'], bins=8, color='#ff8800', alpha=0.8, edgecolor='white')
    ax8.set_title('Trade Entry Time Distribution', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Hour of Day', fontsize=12)
    ax8.set_ylabel('Number of Trades', fontsize=12)
    ax8.grid(True, alpha=0.3)
    
    # Format x-axis to show times
    ax8.set_xticks([9.5, 10, 10.5])
    ax8.set_xticklabels(['9:30', '10:00', '10:30'])
    
    # 9. Confidence vs Performance
    ax9 = plt.subplot(3, 3, 9)
    ax9.scatter(trades_df['confidence'], trades_df['pnl_r'], 
               s=100, alpha=0.7, c=trades_df['pnl_r'], 
               cmap='RdYlGn', edgecolors='white')
    
    ax9.set_title('Entry Confidence vs P&L', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Entry Confidence', fontsize=12)
    ax9.set_ylabel('P&L (R)', fontsize=12)
    ax9.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax9.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(ax9.collections[0], ax=ax9)
    cbar.set_label('P&L (R)', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the chart
    plt.savefig('/Users/michaeldurante/sm-playbook/trading-code/backtesting/OS_D1_Execution_Chart.png', 
                dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()
    
    print("✅ Execution chart saved as: OS_D1_Execution_Chart.png")
    
    # Print trade details
    print("\n" + "="*60)
    print("📋 DETAILED TRADE RESULTS")
    print("="*60)
    
    for i, (_, trade) in enumerate(trades_df_sorted.iterrows(), 1):
        status = "🟢 WIN" if trade['pnl_r'] > 0 else "🔴 LOSS"
        print(f"{i:2d}. {trade['ticker']} ({trade['date'].strftime('%m/%d')}) - "
              f"{status} {trade['pnl_r']:+.2f}R | "
              f"{trade['spot_type']} | Gap: {trade['gap_pct']:.0f}% | "
              f"Entry: {trade['entry_time']}")

if __name__ == '__main__':
    create_execution_charts()