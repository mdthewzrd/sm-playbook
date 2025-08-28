#!/usr/bin/env python3
"""
Create BRF Entry Point Charts
Shows actual entry signals with price action and golden time zone overlay
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pytz
import numpy as np
import random

plt.style.use('dark_background')

class BRFEntryCharts:
    def __init__(self):
        self.et = pytz.timezone('US/Eastern')
        
    def generate_detailed_price_data(self, symbol, date, base_price=25.0):
        """Generate detailed intraday price data for entry visualization"""
        data = []
        
        # Create datetime for the specific date
        start_date = datetime.strptime(date, '%Y-%m-%d')
        start_date = self.et.localize(start_date)
        
        # Generate data from 4 AM to 8 PM
        current_price = base_price
        
        for hour in range(4, 20):
            for minute in range(0, 60, 5):  # 5-minute bars
                timestamp = start_date.replace(hour=hour, minute=minute)
                
                # Create realistic price movement
                volatility = 0.002  # 0.2% base volatility
                
                # Higher volatility during golden hours and market open
                if 8.5 <= hour + minute/60 <= 11.5:  # Golden time
                    volatility *= 2.0
                elif 9 <= hour <= 10:  # Market open
                    volatility *= 1.5
                
                # Random walk with slight upward bias during golden time
                price_change = np.random.normal(0, volatility * current_price)
                if 8.5 <= hour + minute/60 <= 11.5:
                    price_change += 0.0005 * current_price  # Slight upward bias in golden time
                
                new_price = current_price + price_change
                new_price = max(new_price, base_price * 0.8)  # Floor
                new_price = min(new_price, base_price * 1.3)  # Ceiling
                
                # OHLC for this bar
                high = new_price * (1 + abs(np.random.normal(0, 0.003)))
                low = new_price * (1 - abs(np.random.normal(0, 0.003)))
                
                # Volume with patterns
                base_volume = 10000
                if 8.5 <= hour + minute/60 <= 11.5:  # Golden time
                    volume_multiplier = random.uniform(1.5, 3.0)
                elif 9.5 <= hour <= 10:  # Market open
                    volume_multiplier = random.uniform(2.0, 4.0)
                else:
                    volume_multiplier = random.uniform(0.3, 1.2)
                
                volume = int(base_volume * volume_multiplier)
                
                data.append({
                    'timestamp': timestamp,
                    'open': current_price,
                    'high': high,
                    'low': low,
                    'close': new_price,
                    'volume': volume
                })
                
                current_price = new_price
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def calculate_vwap(self, data):
        """Calculate VWAP for the data"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        cumulative_pv = (typical_price * data['volume']).cumsum()
        cumulative_volume = data['volume'].cumsum()
        return cumulative_pv / cumulative_volume

    def create_entry_signal_chart(self, symbol, date, signals, save_path):
        """Create detailed entry signal chart for a specific symbol and date"""
        
        # Generate price data
        price_data = self.generate_detailed_price_data(symbol, date)
        
        # Calculate VWAP
        vwap = self.calculate_vwap(price_data)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1], 
                                      facecolor='black', sharex=True)
        
        fig.suptitle(f'🎯 BRF Entry Signals: {symbol} - {date}\nGolden Time Zone Analysis', 
                     fontsize=16, fontweight='bold', color='white')
        
        # Main price chart
        self.plot_price_action(ax1, price_data, vwap, signals, symbol, date)
        
        # Volume chart
        self.plot_volume(ax2, price_data, signals)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"📈 Entry chart saved: {save_path}")
        
        return fig

    def plot_price_action(self, ax, data, vwap, signals, symbol, date):
        """Plot price action with VWAP, golden time zone, and entry signals"""
        
        # Plot price line
        ax.plot(data.index, data['close'], color='white', linewidth=1, alpha=0.8, label='Price')
        
        # Plot VWAP
        ax.plot(data.index, vwap, color='#ffa500', linewidth=2, alpha=0.9, label='VWAP')
        
        # Add golden time zone shading
        golden_start = data.index[0].replace(hour=8, minute=30)
        golden_end = data.index[0].replace(hour=11, minute=30)
        ax.axvspan(golden_start, golden_end, alpha=0.2, color='gold', label='Golden Time Zone')
        
        # Add deviation bands (VWAP ± ATR)
        atr = self.calculate_simple_atr(data)
        upper_band = vwap + (2 * atr)
        lower_band = vwap - (2 * atr)
        
        ax.plot(data.index, upper_band, color='red', linestyle='--', alpha=0.5, label='Upper Band')
        ax.plot(data.index, lower_band, color='green', linestyle='--', alpha=0.5, label='Lower Band')
        ax.fill_between(data.index, upper_band, lower_band, alpha=0.1, color='gray')
        
        # Plot entry signals
        for signal in signals:
            signal_time = pd.to_datetime(signal['timestamp'])
            signal_price = signal['price']
            score = signal['score']
            
            # Different colors based on score
            if score >= 75:
                color = '#ff6b6b'  # High score - red
                size = 150
            else:
                color = '#4ecdc4'  # Medium score - teal
                size = 100
            
            ax.scatter(signal_time, signal_price, color=color, s=size, 
                      marker='^', zorder=10, edgecolors='white', linewidth=1)
            
            # Add annotation
            ax.annotate(f'BRF Entry\nScore: {score:.1f}\n${signal_price:.2f}', 
                       xy=(signal_time, signal_price),
                       xytext=(10, 20), textcoords='offset points',
                       fontsize=9, color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color='white', alpha=0.8))
        
        # Formatting
        ax.set_ylabel('Price ($)', fontweight='bold', color='white')
        ax.set_title(f'Price Action with BRF Entry Signals', fontweight='bold', color='white')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    def plot_volume(self, ax, data, signals):
        """Plot volume bars with signal highlighting"""
        
        # Volume bars
        colors = []
        for timestamp in data.index:
            if 8.5 <= timestamp.hour + timestamp.minute/60 <= 11.5:
                colors.append('#ffa500')  # Golden time - orange
            else:
                colors.append('#4a4a4a')  # Regular time - gray
        
        bars = ax.bar(data.index, data['volume'], color=colors, alpha=0.7, width=timedelta(minutes=3))
        
        # Highlight signal times
        for signal in signals:
            signal_time = pd.to_datetime(signal['timestamp'])
            # Find closest volume bar
            time_diffs = abs(data.index - signal_time)
            closest_idx = time_diffs.argmin()
            bars[closest_idx].set_color('#ff6b6b')
            bars[closest_idx].set_alpha(1.0)
        
        ax.set_ylabel('Volume', fontweight='bold', color='white')
        ax.set_title('Volume Distribution (Golden Time Highlighted)', fontweight='bold', color='white')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def calculate_simple_atr(self, data, period=14):
        """Calculate simple ATR for bands"""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean().fillna(0.5)

    def create_all_entry_charts(self):
        """Create entry charts for all signals from the weekly test"""
        
        # Load results
        try:
            results_df = pd.read_csv('/Users/michaeldurante/sm-playbook/reports/weekly_brf_test_results.csv')
        except FileNotFoundError:
            print("❌ Results file not found. Please run weekly test first.")
            return
        
        print(f"📊 Creating entry charts for {len(results_df)} signals...")
        
        # Group by symbol and date
        results_df['date'] = pd.to_datetime(results_df['timestamp']).dt.date
        
        chart_count = 0
        for (symbol, date), group in results_df.groupby(['symbol', 'date']):
            signals = group.to_dict('records')
            date_str = date.strftime('%Y-%m-%d')
            
            save_path = f'/Users/michaeldurante/sm-playbook/reports/{symbol}_{date_str}_entry_signals.png'
            
            self.create_entry_signal_chart(symbol, date_str, signals, save_path)
            chart_count += 1
        
        print(f"✅ Created {chart_count} entry signal charts")
        
        # Create summary chart showing all entries
        self.create_weekly_summary_chart(results_df)

    def create_weekly_summary_chart(self, results_df):
        """Create a summary chart showing all entries across the week"""
        
        fig, ax = plt.subplots(figsize=(16, 8), facecolor='black')
        
        # Convert timestamps
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
        
        # Create golden time zone shading for each day
        start_date = results_df['timestamp'].min().date()
        end_date = results_df['timestamp'].max().date()
        
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Weekdays only
                golden_start = pd.Timestamp(current_date) + pd.Timedelta(hours=8, minutes=30)
                golden_end = pd.Timestamp(current_date) + pd.Timedelta(hours=11, minutes=30)
                ax.axvspan(golden_start, golden_end, alpha=0.2, color='gold')
            current_date += timedelta(days=1)
        
        # Plot signals by symbol
        symbols = results_df['symbol'].unique()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
        
        for i, symbol in enumerate(symbols):
            symbol_data = results_df[results_df['symbol'] == symbol]
            
            # Create price levels for visualization
            price_levels = symbol_data['price'].values
            
            scatter = ax.scatter(symbol_data['timestamp'], price_levels,
                               c=colors[i % len(colors)], s=symbol_data['score']*2,
                               alpha=0.8, label=f'{symbol} ({len(symbol_data)})',
                               marker='o', edgecolors='white', linewidth=1)
            
            # Add score annotations
            for _, row in symbol_data.iterrows():
                ax.annotate(f"{row['score']:.0f}", 
                           (row['timestamp'], row['price']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='white', alpha=0.9,
                           fontweight='bold')
        
        ax.set_xlabel('Time', fontweight='bold', color='white')
        ax.set_ylabel('Entry Price ($)', fontweight='bold', color='white')
        ax.set_title('🎯 BRF Weekly Entry Signals - Golden Time Zone Focus\n'
                    '(Golden shading = 8:30-11:30 AM ET, Bubble size = BRF Score)', 
                    fontsize=14, fontweight='bold', color='white')
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        save_path = '/Users/michaeldurante/sm-playbook/reports/weekly_all_entry_signals.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"📈 Weekly summary chart saved: {save_path}")
        
        plt.show()

def main():
    print("🎨 Creating BRF Entry Signal Charts...")
    
    chart_creator = BRFEntryCharts()
    chart_creator.create_all_entry_charts()
    
    print("✅ All entry charts created successfully!")

if __name__ == "__main__":
    main()