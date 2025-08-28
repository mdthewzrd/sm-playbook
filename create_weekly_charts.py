#!/usr/bin/env python3
"""
Create Weekly BRF Strategy Charts
Visualizes the golden time zone filtering and weekly execution results
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pytz
import numpy as np

plt.style.use('dark_background')

def create_weekly_charts():
    """Create comprehensive charts for the weekly BRF test results"""
    
    # Load results
    try:
        results_df = pd.read_csv('/Users/michaeldurante/sm-playbook/reports/weekly_brf_test_results.csv')
        print(f"📊 Loaded {len(results_df)} signals for visualization")
    except FileNotFoundError:
        print("❌ Results file not found. Please run weekly test first.")
        return
    
    # Convert timestamp column
    results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('🎯 BRF Strategy Weekly Execution Results - Golden Time Zone Analysis', 
                 fontsize=16, fontweight='bold', color='white')
    
    # Create grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[2, 1])
    
    # 1. Main Timeline Chart
    ax1 = fig.add_subplot(gs[0, :])
    create_timeline_chart(ax1, results_df)
    
    # 2. Daily Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    create_daily_distribution(ax2, results_df)
    
    # 3. Hourly Distribution  
    ax3 = fig.add_subplot(gs[1, 1])
    create_hourly_distribution(ax3, results_df)
    
    # 4. Symbol Performance
    ax4 = fig.add_subplot(gs[2, 0])
    create_symbol_performance(ax4, results_df)
    
    # 5. Score Distribution
    ax5 = fig.add_subplot(gs[2, 1])
    create_score_distribution(ax5, results_df)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save chart
    output_path = '/Users/michaeldurante/sm-playbook/reports/weekly_brf_execution_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"📈 Chart saved to: {output_path}")
    
    plt.show()

def create_timeline_chart(ax, df):
    """Create main timeline chart showing signal distribution"""
    
    # Golden time zone shading
    days = pd.date_range('2025-08-25', periods=5, freq='D')
    for day in days:
        if day.weekday() < 5:  # Weekdays only
            golden_start = day.replace(hour=8, minute=30)
            golden_end = day.replace(hour=11, minute=30)
            ax.axvspan(golden_start, golden_end, alpha=0.3, color='gold', label='Golden Time Zone' if day == days[0] else "")
    
    # Plot signals by symbol with different colors and markers
    symbols = df['symbol'].unique()
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd', '#98fb98', '#f0e68c']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    
    for i, symbol in enumerate(symbols):
        symbol_data = df[df['symbol'] == symbol]
        ax.scatter(symbol_data['timestamp'], [i] * len(symbol_data), 
                  c=colors[i % len(colors)], marker=markers[i % len(markers)], 
                  s=symbol_data['score'], alpha=0.8, 
                  label=f'{symbol} ({len(symbol_data)} signals)', zorder=5)
        
        # Add score annotations
        for _, row in symbol_data.iterrows():
            ax.annotate(f"{row['score']:.0f}", 
                       (row['timestamp'], i), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='white', alpha=0.8)
    
    ax.set_ylim(-0.5, len(symbols) - 0.5)
    ax.set_yticks(range(len(symbols)))
    ax.set_yticklabels(symbols)
    ax.set_ylabel('Symbol', fontweight='bold')
    ax.set_title('🕰️ Golden Time Zone Signal Timeline (Bubble Size = BRF Score)', 
                fontweight='bold', pad=20)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)

def create_daily_distribution(ax, df):
    """Create daily distribution bar chart"""
    
    # Get day names
    df['day'] = df['timestamp'].dt.day_name()
    daily_counts = df['day'].value_counts()
    
    # Ensure all weekdays are represented
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    daily_data = [daily_counts.get(day, 0) for day in weekdays]
    
    bars = ax.bar(weekdays, daily_data, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('📅 Daily Signal Distribution', fontweight='bold')
    ax.set_ylabel('Golden Time Signals')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

def create_hourly_distribution(ax, df):
    """Create hourly distribution chart"""
    
    # Extract hour from timestamp
    df['hour'] = df['timestamp'].dt.hour
    hourly_counts = df['hour'].value_counts().sort_index()
    
    # Create pie chart
    colors = ['#ff6b6b', '#4ecdc4']
    wedges, texts, autotexts = ax.pie(hourly_counts.values, 
                                     labels=[f'{h}:00' for h in hourly_counts.index],
                                     autopct='%1.1f%%', 
                                     colors=colors,
                                     startangle=90)
    
    # Style the text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('⏰ Hourly Distribution\n(Golden Time Only)', fontweight='bold')

def create_symbol_performance(ax, df):
    """Create symbol performance chart"""
    
    symbol_counts = df['symbol'].value_counts()
    symbol_avg_scores = df.groupby('symbol')['score'].mean()
    
    # Create horizontal bar chart
    y_pos = np.arange(len(symbol_counts))
    bars = ax.barh(y_pos, symbol_counts.values, color='#45b7d1', alpha=0.7)
    
    # Add average score labels
    for i, (symbol, count) in enumerate(symbol_counts.items()):
        avg_score = symbol_avg_scores[symbol]
        ax.text(count + 0.1, i, f'Avg: {avg_score:.1f}', 
               va='center', fontsize=9, color='white')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(symbol_counts.index)
    ax.set_xlabel('Golden Time Signals')
    ax.set_title('📊 Symbol Performance', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

def create_score_distribution(ax, df):
    """Create BRF score distribution histogram"""
    
    scores = df['score']
    
    # Create histogram
    bins = np.arange(70, scores.max() + 5, 2)
    n, bins, patches = ax.hist(scores, bins=bins, color='#96ceb4', alpha=0.7, edgecolor='black')
    
    # Color code the bars
    for i, patch in enumerate(patches):
        if bins[i] >= 80:
            patch.set_facecolor('#ff6b6b')  # High score - red
        elif bins[i] >= 75:
            patch.set_facecolor('#ffeaa7')  # Medium score - yellow
        else:
            patch.set_facecolor('#96ceb4')  # Lower score - green
    
    # Add statistics
    avg_score = scores.mean()
    ax.axvline(avg_score, color='white', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(avg_score + 0.5, max(n) * 0.8, f'Avg: {avg_score:.1f}', 
           color='white', fontweight='bold', rotation=90)
    
    ax.set_xlabel('BRF Score')
    ax.set_ylabel('Count')
    ax.set_title('📈 Score Distribution\n(Golden Time Signals)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

def create_additional_analytics_chart():
    """Create additional analytics visualization"""
    
    # Create a summary comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='black')
    fig.suptitle('🎯 Golden Time Zone Impact Analysis', 
                 fontsize=16, fontweight='bold', color='white')
    
    # Simulation data for before/after golden time zone implementation
    before_data = {
        'Total Signals': 19,
        'Low Quality': 13,
        'High Quality': 6,
        'Daily Average': 3.8,
        'Win Rate': 0.45,  # Estimated
    }
    
    after_data = {
        'Total Signals': 6,
        'Low Quality': 0,
        'High Quality': 6,
        'Daily Average': 1.2,
        'Win Rate': 0.72,  # Estimated improvement
    }
    
    # Chart 1: Signal Quality Comparison
    categories = ['Total\nSignals', 'High Quality\nSignals', 'Daily\nAverage']
    before_values = [before_data['Total Signals'], before_data['High Quality'], before_data['Daily Average']]
    after_values = [after_data['Total Signals'], after_data['High Quality'], after_data['Daily Average']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, before_values, width, label='Before Golden Time', color='#ff6b6b', alpha=0.7)
    bars2 = ax1.bar(x + width/2, after_values, width, label='After Golden Time', color='#4ecdc4', alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Count')
    ax1.set_title('📊 Signal Quality Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Time Distribution
    time_periods = ['Pre-Market\n(4-8:30)', 'Golden Time\n(8:30-11:30)', 'Regular Hours\n(11:30-16)', 'After Hours\n(16-20)']
    signal_distribution = [0, 6, 8, 5]  # Based on the test results
    acceptance_rate = [0, 100, 0, 0]  # Only golden time accepted
    
    ax2_twin = ax2.twinx()
    
    bars = ax2.bar(time_periods, signal_distribution, color='#45b7d1', alpha=0.7, label='Total Signals')
    line = ax2_twin.plot(time_periods, acceptance_rate, color='#ffeaa7', marker='o', 
                        linewidth=3, markersize=8, label='Acceptance Rate (%)')
    
    ax2.set_ylabel('Signal Count', color='#45b7d1')
    ax2_twin.set_ylabel('Acceptance Rate (%)', color='#ffeaa7')
    ax2.set_title('⏰ Time Period Analysis', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Save additional chart
    output_path = '/Users/michaeldurante/sm-playbook/reports/golden_time_impact_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"📈 Impact analysis chart saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    print("🎨 Creating Weekly BRF Strategy Charts...")
    create_weekly_charts()
    print("\n🔍 Creating Golden Time Zone Impact Analysis...")
    create_additional_analytics_chart()
    print("\n✅ Chart generation complete!")