#!/usr/bin/env python3
"""
Create Weekly BRF Execution Chart
Visual representation of the Golden Time Zone performance
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set dark theme
plt.style.use('dark_background')

def create_weekly_chart():
    # Weekly data from our test
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    setups = [37, 54, 28, 42, 23]
    avg_scores = [82.4, 77.1, 77.8, 81.4, 80.8]
    symbols = [7, 8, 6, 8, 7]
    
    # Hourly distribution
    hours = ['09:00', '10:00', '11:00']
    hourly_counts = [99, 51, 34]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('📊 BRF Strategy - Weekly Execution Analysis\nGolden Time Zone: 8:30-11:30 AM ET', 
                 fontsize=16, fontweight='bold', color='white')
    
    # 1. Daily Setup Count
    bars1 = ax1.bar(days, setups, color=['#00ff88', '#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24'], alpha=0.8)
    ax1.set_title('📅 Daily BRF Setup Count', fontsize=12, fontweight='bold', color='white')
    ax1.set_ylabel('Number of Setups', color='white')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, setups):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(value), ha='center', va='bottom', color='white', fontweight='bold')
    
    # 2. Average Scores
    bars2 = ax2.bar(days, avg_scores, color=['#ff9f43', '#10ac84', '#ee5a6f', '#0984e3', '#a29bfe'], alpha=0.8)
    ax2.set_title('🎯 Average Setup Scores', fontsize=12, fontweight='bold', color='white')
    ax2.set_ylabel('Average Score', color='white')
    ax2.set_ylim(70, 85)  # Focus on the score range
    ax2.grid(True, alpha=0.3)
    
    # Add score labels
    for bar, score in zip(bars2, avg_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{score:.1f}', ha='center', va='bottom', color='white', fontweight='bold')
    
    # 3. Hourly Distribution
    bars3 = ax3.bar(hours, hourly_counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.8)
    ax3.set_title('⏰ Golden Time Zone Distribution', fontsize=12, fontweight='bold', color='white')
    ax3.set_ylabel('Setup Count', color='white')
    ax3.grid(True, alpha=0.3)
    
    # Add percentage labels
    total_hourly = sum(hourly_counts)
    for bar, count in zip(bars3, hourly_counts):
        pct = (count / total_hourly) * 100
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', color='white', fontweight='bold')
    
    # 4. Symbol Coverage
    symbol_names = ['TSLA', 'AMZN', 'IXHL', 'PLTR', 'NVDA', 'COMM', 'SRFM', 'GOOGL']
    symbol_counts = [43, 30, 24, 23, 21, 19, 14, 10]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(symbol_names)))
    wedges, texts, autotexts = ax4.pie(symbol_counts, labels=symbol_names, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax4.set_title('📈 Setup Distribution by Symbol', fontsize=12, fontweight='bold', color='white')
    
    # Style the pie chart text
    for text in texts:
        text.set_color('white')
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Adjust layout and add summary stats
    plt.tight_layout()
    
    # Add summary text box
    summary_text = """
📊 WEEKLY SUMMARY
• Total Setups: 184
• Average Score: 79.7
• Golden Time Compliance: 100%
• Active Symbols: 8/8
• Peak Hour: 9:00-10:00 AM (54%)
"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, color='white',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#2c3e50', alpha=0.8))
    
    # Save the chart
    plt.savefig('/Users/michaeldurante/sm-playbook/reports/weekly_brf_execution_chart.png', 
                dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    plt.show()
    
    print("📊 Weekly BRF execution chart created successfully!")
    print("💾 Saved to: /Users/michaeldurante/sm-playbook/reports/weekly_brf_execution_chart.png")

if __name__ == "__main__":
    create_weekly_chart()