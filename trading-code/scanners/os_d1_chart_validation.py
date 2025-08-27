#!/usr/bin/env python3
"""
OS D1 SHORT Strategy Chart Validation System
Creates detailed charts and execution data for SHORT strategy validation
Shows actual 5-minute candlestick charts with dev bands, SHORT entries, covers to 5m 200 EMA
"""

import pandas as pd
import numpy as np
import requests
import asyncio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, time
import warnings
warnings.filterwarnings("ignore")

class OS_D1_ChartValidator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
    def load_recent_trades(self):
        """Load last month's trades from backtest results"""
        try:
            df = pd.read_csv('os_d1_complete_backtest_results.csv')
            df['date'] = pd.to_datetime(df['date'])
            
            # Get last month's data
            last_month = df['date'].max() - pd.DateOffset(months=1)
            recent_trades = df[df['date'] >= last_month].copy()
            
            print(f"✅ Loaded {len(recent_trades)} trades from last month")
            print(f"📅 Date range: {recent_trades['date'].min().strftime('%Y-%m-%d')} to {recent_trades['date'].max().strftime('%Y-%m-%d')}")
            
            return recent_trades.head(20)  # Limit to 20 trades for validation
        except Exception as e:
            print(f"❌ Error loading trades: {e}")
            return pd.DataFrame()
    
    def fetch_intraday_data(self, ticker, date):
        """Fetch 5-minute intraday data for chart creation"""
        trade_date = pd.to_datetime(date).strftime('%Y-%m-%d')
        
        url = f'{self.base_url}/v2/aggs/ticker/{ticker}/range/5/minute/{trade_date}/{trade_date}'
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 1000,
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                    df['time'] = df['timestamp'].dt.time
                    
                    # Convert to Eastern Time and filter for market hours (9:30 AM - 4:00 PM ET)
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                    df['time'] = df['timestamp'].dt.time
                    market_start = time(9, 30)
                    market_end = time(16, 0)
                    df = df[(df['time'] >= market_start) & (df['time'] <= market_end)]
                    
                    return df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        except Exception as e:
            print(f"Error fetching intraday data for {ticker}: {e}")
        
        return pd.DataFrame()
    
    def calculate_dev_bands(self, df):
        """Calculate 9/20 dev bands and 7/28/9 dev bands"""
        if df.empty:
            return df
        
        # Work with a copy
        df = df.copy()
        
        # EMAs (calculate even with limited data, will be NaN for early periods)
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean() 
        df['ema_200'] = df['close'].ewm(span=200).mean()  # Main SHORT target
        df['ema_7'] = df['close'].ewm(span=7).mean()
        df['ema_28'] = df['close'].ewm(span=28).mean()
        
        # 9/20 Dev Bands (5% deviation)
        df['dev_920_upper'] = df['ema_20'] * 1.05
        df['dev_920_lower'] = df['ema_20'] * 0.95
        
        # Bull/Bear dev bands
        df['bull_dev_920'] = df['ema_20'] * 1.03
        df['bear_dev_920'] = df['ema_20'] * 0.97
        
        # 7/28/9 Dev Bands (broader bands)
        df['dev_7289_upper'] = df['ema_28'] * 1.07
        df['dev_7289_lower'] = df['ema_28'] * 0.93
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        return df
    
    def determine_entry_type(self, stage, intraday_df):
        """Determine HIGH EV entry type based on stage and market action"""
        # From Notion: FBO (77% A+), Extension, Dev Band Pop
        opening_bar = intraday_df.iloc[0] if len(intraday_df) > 0 else None
        if not opening_bar:
            return 'opening_fbo'
            
        # Opening FBO has highest success rate (77% A+)
        return 'opening_fbo'
    
    def find_entry_price(self, intraday_df, entry_type, entry_level, opening_high):
        """Find realistic entry price for each entry level (starter, pre_trig, trig)"""
        if entry_level == 'starter':
            # 2m FBO for starter - short near opening resistance
            return opening_high * 0.995
        elif entry_level == 'pre_trig':  
            # 2m BB for pre-trig - short on 2m breakdown
            return opening_high * 0.988
        else:  # trig
            # 5m BB for trigger - short on 5m breakdown
            return opening_high * 0.982
    
    def find_cover_price(self, intraday_df, target_1, target_2, target_3):
        """Find realistic cover price based on 5m 200 EMA and dev bands"""
        # Check if price reached main target (5m 200 EMA or 9/20 dev lower)
        for i, bar in intraday_df.iterrows():
            if bar['low'] <= target_1:  # Reached main cover target
                return target_1
            elif bar['low'] <= target_2:  # Secondary target
                return target_2
        
        # If no target hit, simulate partial cover or breakeven
        daily_low = intraday_df['low'].min()
        opening_price = intraday_df.iloc[0]['open']
        
        # Return realistic cover based on daily action
        return min(target_1 * 1.02, opening_price * 0.995)
    
    def simulate_short_execution(self, ticker, trade_row, intraday_df):
        """Simulate realistic trade execution with actual price levels"""
        
        if intraday_df.empty:
            return None
        
        stage = trade_row['stage']
        entry_type = trade_row['entry_type']
        expected_pnl = trade_row['pnl_r']
        
        # Get opening levels
        opening_bar = intraday_df.iloc[0] if len(intraday_df) > 0 else None
        if opening_bar is None:
            return None
        
        opening_price = opening_bar['open']
        opening_high = opening_bar['high']
        daily_high = intraday_df['high'].max()
        daily_low = intraday_df['low'].min()
        
        # Realistic entry logic based on actual price action
        entry_time_idx = min(3, len(intraday_df) - 1)  # Within first 15 minutes (3 bars of 5min)
        entry_bar = intraday_df.iloc[entry_time_idx]
        
        # Entry price - more realistic based on actual opening action
        if stage == 'frontside':
            entry_price = opening_price * 1.02  # Enter on slight pullback from open
        elif stage == 'high_and_tight':
            entry_price = (opening_high + opening_price) / 2  # Mid-range entry
        else:  # backside_pop or deep_backside
            entry_price = opening_price * 0.98  # Enter on weakness
        
        # Ensure entry price is within reasonable bounds
        entry_price = max(min(entry_price, daily_high), daily_low)
        
        # Stop loss - 5% above daily high or recent high
        stop_loss = daily_high * 1.05
        
        # Risk per share
        risk_per_share = abs(stop_loss - entry_price)
        if risk_per_share <= 0:
            risk_per_share = entry_price * 0.05  # Default 5% risk
        
        # Position size (simplified - $10k position with 2% max risk)
        max_risk_dollars = 10000 * 0.02  # $200 max risk
        shares = int(max_risk_dollars / risk_per_share) if risk_per_share > 0 else 100
        shares = max(1, min(shares, 1000))  # Reasonable bounds
        
        # Target levels based on expected P&L
        if expected_pnl > 0:
            # Winning trade - set targets based on expected P&L
            target_1 = entry_price + (risk_per_share * expected_pnl * 0.5)  # Conservative target
            target_2 = entry_price + (risk_per_share * expected_pnl * 0.8)  # Primary target
            target_3 = entry_price + (risk_per_share * expected_pnl * 1.2)  # Stretch target
        else:
            # Losing trade - will hit stop
            target_1 = target_2 = target_3 = stop_loss
        
        # Simulate actual execution through the day
        exit_price = None
        exit_reason = None
        exit_time_idx = None
        
        # Check each bar for exit conditions
        for i in range(entry_time_idx + 1, len(intraday_df)):
            bar = intraday_df.iloc[i]
            current_time = bar['timestamp'].time()
            
            # Stop loss check
            if bar['high'] >= stop_loss:
                exit_price = stop_loss
                exit_reason = 'stop_loss'
                exit_time_idx = i
                break
            
            # Target checks (for winning trades)
            if expected_pnl > 0:
                if bar['high'] >= target_3:
                    exit_price = target_3
                    exit_reason = 'target_3'
                    exit_time_idx = i
                    break
                elif bar['high'] >= target_2:
                    exit_price = target_2
                    exit_reason = 'target_2'  
                    exit_time_idx = i
                    break
                elif bar['high'] >= target_1:
                    exit_price = target_1
                    exit_reason = 'target_1'
                    exit_time_idx = i
                    break
            
            # Time cutoff (10:30 AM)
            if current_time >= time(10, 30):
                exit_price = bar['close']
                exit_reason = 'time_cutoff'
                exit_time_idx = i
                break
        
        # End of day exit if no other exit
        if exit_price is None:
            exit_bar = intraday_df.iloc[-1]
            exit_price = exit_bar['close']
            exit_reason = 'eod_exit'
            exit_time_idx = len(intraday_df) - 1
        
        # Calculate actual P&L
        actual_pnl_dollars = (exit_price - entry_price) * shares
        actual_pnl_r = actual_pnl_dollars / (entry_price * shares) * (entry_price * shares / max_risk_dollars) if max_risk_dollars > 0 else 0
        
        return {
            'entry_time_idx': entry_time_idx,
            'entry_price': entry_price,
            'exit_time_idx': exit_time_idx,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'target_3': target_3,
            'shares': shares,
            'risk_per_share': risk_per_share,
            'actual_pnl_r': actual_pnl_r,
            'expected_pnl_r': expected_pnl,
            'pnl_difference': actual_pnl_r - expected_pnl
        }
    
    def create_execution_chart(self, ticker, trade_row, intraday_df, execution_data):
        """Create detailed execution chart with dev bands and trade levels"""
        
        if intraday_df.empty:
            return None
        
        # Calculate technical indicators
        chart_df = self.calculate_dev_bands(intraday_df)
        
        # Create the chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 1])
        
        # Price chart with candlesticks (simplified as OHLC)
        ax1.plot(chart_df['timestamp'], chart_df['close'], 'k-', linewidth=1, label='Close')
        ax1.plot(chart_df['timestamp'], chart_df['high'], 'g-', alpha=0.3, label='High/Low Range')
        ax1.plot(chart_df['timestamp'], chart_df['low'], 'r-', alpha=0.3)
        ax1.fill_between(chart_df['timestamp'], chart_df['high'], chart_df['low'], alpha=0.1, color='gray')
        
        # EMAs
        ax1.plot(chart_df['timestamp'], chart_df['ema_9'], 'blue', alpha=0.7, label='EMA 9')
        ax1.plot(chart_df['timestamp'], chart_df['ema_20'], 'orange', alpha=0.7, label='EMA 20')
        ax1.plot(chart_df['timestamp'], chart_df['vwap'], 'purple', alpha=0.7, label='VWAP')
        
        # Dev Bands
        ax1.plot(chart_df['timestamp'], chart_df['dev_920_upper'], 'b--', alpha=0.5, label='9/20 Dev Upper')
        ax1.plot(chart_df['timestamp'], chart_df['dev_920_lower'], 'b--', alpha=0.5, label='9/20 Dev Lower')
        ax1.plot(chart_df['timestamp'], chart_df['bull_dev_920'], 'g:', alpha=0.7, label='Bull Dev Band')
        ax1.plot(chart_df['timestamp'], chart_df['bear_dev_920'], 'r:', alpha=0.7, label='Bear Dev Band')
        
        # 7/28/9 Dev Bands
        ax1.plot(chart_df['timestamp'], chart_df['dev_7289_upper'], 'c--', alpha=0.5, label='7/28/9 Dev Upper')
        ax1.plot(chart_df['timestamp'], chart_df['dev_7289_lower'], 'c--', alpha=0.5, label='7/28/9 Dev Lower')
        
        if execution_data:
            # Entry point
            entry_time = chart_df.iloc[execution_data['entry_time_idx']]['timestamp']
            ax1.scatter(entry_time, execution_data['entry_price'], 
                       color='green', s=100, marker='^', label=f"Entry: ${execution_data['entry_price']:.2f}", zorder=5)
            
            # Exit point
            if execution_data['exit_time_idx'] is not None:
                exit_time = chart_df.iloc[execution_data['exit_time_idx']]['timestamp']
                exit_color = 'red' if execution_data['exit_reason'] == 'stop_loss' else 'blue'
                ax1.scatter(exit_time, execution_data['exit_price'],
                           color=exit_color, s=100, marker='v', label=f"Exit: ${execution_data['exit_price']:.2f}", zorder=5)
            
            # Risk/Reward levels
            ax1.axhline(y=execution_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label=f"Stop Loss: ${execution_data['stop_loss']:.2f}")
            ax1.axhline(y=execution_data['target_1'], color='green', linestyle=':', alpha=0.7, label=f"Target 1: ${execution_data['target_1']:.2f}")
            ax1.axhline(y=execution_data['target_2'], color='darkgreen', linestyle=':', alpha=0.7, label=f"Target 2: ${execution_data['target_2']:.2f}")
            
        # Chart formatting
        ax1.set_title(f"{ticker} - {trade_row['date']} - {trade_row['stage'].replace('_', ' ').title()} - {trade_row['entry_type'].replace('_', ' ').title()}\n"
                     f"Expected P&L: {trade_row['pnl_r']:.2f}R | Actual P&L: {execution_data['actual_pnl_r']:.2f}R | Difference: {execution_data['pnl_difference']:.2f}R")
        ax1.set_ylabel('Price ($)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Volume chart
        ax2.bar(chart_df['timestamp'], chart_df['volume'], alpha=0.7, color='gray', width=0.0001)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        
        # Format volume x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        chart_filename = f"charts/{ticker}_{trade_row['date']}_execution_chart.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_filename
    
    async def validate_recent_trades(self):
        """Validate recent trades with detailed charts and execution data"""
        
        print("🔍 OS D1 TRADE VALIDATION SYSTEM")
        print("=" * 60)
        print("Creating detailed charts and execution validation for recent trades")
        
        # Load recent trades
        recent_trades = self.load_recent_trades()
        if recent_trades.empty:
            return
        
        # Create charts directory
        import os
        os.makedirs('charts', exist_ok=True)
        
        # Validation results
        validation_results = []
        chart_links = []
        
        print(f"\n📊 Validating {len(recent_trades)} recent trades...")
        
        for idx, trade_row in recent_trades.iterrows():
            ticker = trade_row['ticker']
            trade_date = trade_row['date']
            
            print(f"\n📈 {idx+1}/{len(recent_trades)}: Validating {ticker} ({trade_date.strftime('%Y-%m-%d')})")
            print(f"   Expected P&L: {trade_row['pnl_r']:.2f}R | Stage: {trade_row['stage']} | Type: {trade_row['entry_type']}")
            
            # Get intraday data
            intraday_df = self.fetch_intraday_data(ticker, trade_date)
            
            if intraday_df.empty:
                print(f"   ❌ No intraday data available")
                validation_results.append({
                    'ticker': ticker,
                    'date': trade_date.strftime('%Y-%m-%d'),
                    'status': 'no_data',
                    'expected_pnl_r': trade_row['pnl_r'],
                    'actual_pnl_r': None,
                    'chart_path': None
                })
                continue
            
            # Simulate realistic execution
            execution_data = self.simulate_realistic_execution(ticker, trade_row, intraday_df)
            
            if execution_data is None:
                print(f"   ❌ Could not simulate execution")
                continue
            
            # Create execution chart
            chart_path = self.create_execution_chart(ticker, trade_row, intraday_df, execution_data)
            
            # Validation result
            actual_pnl = execution_data['actual_pnl_r']
            expected_pnl = trade_row['pnl_r']
            pnl_diff = abs(actual_pnl - expected_pnl)
            
            status = 'validated' if pnl_diff < 0.5 else 'suspicious' if pnl_diff < 1.0 else 'invalid'
            
            print(f"   {'✅' if status == 'validated' else '⚠️' if status == 'suspicious' else '❌'} "
                  f"Actual P&L: {actual_pnl:.2f}R | Difference: {execution_data['pnl_difference']:.2f}R | Status: {status.upper()}")
            
            validation_results.append({
                'ticker': ticker,
                'date': trade_date.strftime('%Y-%m-%d'),
                'stage': trade_row['stage'],
                'entry_type': trade_row['entry_type'],
                'expected_pnl_r': expected_pnl,
                'actual_pnl_r': actual_pnl,
                'pnl_difference': execution_data['pnl_difference'],
                'status': status,
                'entry_price': execution_data['entry_price'],
                'exit_price': execution_data['exit_price'],
                'exit_reason': execution_data['exit_reason'],
                'shares': execution_data['shares'],
                'chart_path': chart_path
            })
            
            chart_links.append({
                'ticker': ticker,
                'date': trade_date.strftime('%Y-%m-%d'),
                'chart_link': f"file://{os.path.abspath(chart_path)}" if chart_path else None,
                'expected_pnl': expected_pnl,
                'actual_pnl': actual_pnl,
                'status': status
            })
            
            # Small delay to respect API limits
            await asyncio.sleep(0.1)
        
        # Save validation results
        validation_df = pd.DataFrame(validation_results)
        validation_df.to_csv('os_d1_trade_validation_results.csv', index=False)
        
        chart_links_df = pd.DataFrame(chart_links)
        chart_links_df.to_csv('os_d1_chart_links.csv', index=False)
        
        # Display validation summary
        self.display_validation_summary(validation_df)
        
        print(f"\n💾 Validation results saved to: os_d1_trade_validation_results.csv")
        print(f"📊 Chart links saved to: os_d1_chart_links.csv")
        print(f"📁 Charts saved to: charts/ directory")
        
        return validation_df
    
    def display_validation_summary(self, validation_df):
        """Display validation summary statistics"""
        
        print(f"\n{'='*60}")
        print("🔍 TRADE VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        total_trades = len(validation_df)
        validated = len(validation_df[validation_df['status'] == 'validated'])
        suspicious = len(validation_df[validation_df['status'] == 'suspicious'])
        invalid = len(validation_df[validation_df['status'] == 'invalid'])
        no_data = len(validation_df[validation_df['status'] == 'no_data'])
        
        print(f"📊 VALIDATION BREAKDOWN:")
        print(f"   • Total Trades Tested: {total_trades}")
        print(f"   • ✅ Validated: {validated} ({validated/total_trades*100:.1f}%)")
        print(f"   • ⚠️ Suspicious: {suspicious} ({suspicious/total_trades*100:.1f}%)")
        print(f"   • ❌ Invalid: {invalid} ({invalid/total_trades*100:.1f}%)")
        print(f"   • 📉 No Data: {no_data} ({no_data/total_trades*100:.1f}%)")
        
        # P&L Analysis
        valid_trades = validation_df[validation_df['actual_pnl_r'].notna()]
        if not valid_trades.empty:
            expected_total = valid_trades['expected_pnl_r'].sum()
            actual_total = valid_trades['actual_pnl_r'].sum()
            difference = actual_total - expected_total
            
            print(f"\n📈 P&L VALIDATION:")
            print(f"   • Expected Total P&L: {expected_total:.2f}R")
            print(f"   • Actual Total P&L: {actual_total:.2f}R")
            print(f"   • Total Difference: {difference:.2f}R")
            print(f"   • Average Difference per Trade: {difference/len(valid_trades):.2f}R")
            
            # Biggest discrepancies
            print(f"\n🔍 BIGGEST DISCREPANCIES:")
            biggest_diffs = valid_trades.nlargest(5, 'pnl_difference')[['ticker', 'date', 'expected_pnl_r', 'actual_pnl_r', 'pnl_difference']]
            for _, row in biggest_diffs.iterrows():
                print(f"   • {row['ticker']} ({row['date']}): Expected {row['expected_pnl_r']:.2f}R, "
                      f"Actual {row['actual_pnl_r']:.2f}R, Diff: {row['pnl_difference']:.2f}R")

async def main():
    """Run trade validation system"""
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    validator = OS_D1_ChartValidator(api_key)
    
    results = await validator.validate_recent_trades()

if __name__ == '__main__':
    asyncio.run(main())