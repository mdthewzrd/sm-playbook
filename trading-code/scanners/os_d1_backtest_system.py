#!/usr/bin/env python3
"""
OS D1 Small Cap Fade Strategy - Complete Backtesting System

This implements the full OS D1 strategy from the Notion document:
1. Scanner finds small cap day one gappers (already complete - 434 setups)
2. Enter SHORT positions to fade the gap up
3. Take profits as the stock fades back down (mean reversion)

Strategy Logic from Document:
- Entry: Short near the gap high/pre-market high 
- Stop Loss: Above the high of day
- Take Profit: Target fade back to previous day's levels
- Position Sizing: Risk-based sizing
- Time Frame: Intraday fade (same day exit preferred)
"""

import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

class OS_D1_BacktestSystem:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
        # Strategy parameters from OS D1 document
        self.position_size = 10000  # $10K per position
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        self.stop_loss_buffer = 0.05  # 5% above HOD for stop
        self.take_profit_targets = [0.5, 0.7, 0.85]  # Fade targets (50%, 70%, 85% back to prev close)
        self.max_hold_time_minutes = 390  # Exit by end of day if not stopped out
        
    def load_os_d1_setups(self):
        """Load OS D1 setups from 2024 backtest results"""
        try:
            # First try to load 2025 setups
            df = pd.read_csv('os_d1_all_2025_setups.csv')
            df['scan_date'] = pd.to_datetime(df['scan_date'])
            print(f"✅ Loaded {len(df)} OS D1 setups from 2025")
            return df
        except FileNotFoundError:
            try:
                # Fallback to 2024 backtest results
                df = pd.read_csv('os_d1_2024_backtest_results.csv')
                df['scan_date'] = pd.to_datetime(df['date'])
                # Convert to match expected format
                df['prev_close'] = df['prev_close']
                df['h'] = df['day_high'] 
                print(f"✅ Loaded {len(df)} OS D1 setups from 2024 backtest")
                return df
            except FileNotFoundError:
                print("❌ No OS D1 setups files found. Run the scanner first.")
                return pd.DataFrame()
    
    def fetch_intraday_data(self, ticker, date, timespan='1', multiplier=1):
        """Fetch intraday data for backtesting the fade"""
        
        # Convert date to start/end timestamps for market hours
        trade_date = pd.to_datetime(date).strftime('%Y-%m-%d')
        
        url = f'{self.base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/minute/{trade_date}/{trade_date}'
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000,
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
                    
                    # Filter for market hours (9:30 AM - 4:00 PM ET)
                    market_start = pd.to_datetime('09:30:00').time()
                    market_end = pd.to_datetime('16:00:00').time()
                    
                    df = df[(df['time'] >= market_start) & (df['time'] <= market_end)]
                    
                    return df[['timestamp', 'o', 'h', 'l', 'c', 'v']].rename(columns={
                        'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
                    })
        except Exception as e:
            print(f"Error fetching intraday data for {ticker}: {e}")
        
        return pd.DataFrame()
    
    def calculate_entry_price(self, intraday_df, setup_row):
        """Calculate entry price for the fade short"""
        
        if intraday_df.empty:
            return None, None
        
        # Entry strategy: Short near the gap high or first 15 minutes high
        # This simulates shorting the initial gap fade
        
        first_15_min = intraday_df.head(15)  # First 15 minutes
        
        if len(first_15_min) == 0:
            return None, None
        
        # Entry price: Average of first 15 min high (conservative entry)
        entry_price = first_15_min['high'].mean()
        
        # Entry time: Around 15 minutes after open
        entry_time = first_15_min['timestamp'].iloc[-1] if len(first_15_min) > 0 else intraday_df['timestamp'].iloc[0]
        
        return entry_price, entry_time
    
    def simulate_fade_trade(self, ticker, setup_row, intraday_df):
        """Simulate the complete fade trade for one setup"""
        
        if intraday_df.empty:
            return None
        
        # Calculate entry
        entry_price, entry_time = self.calculate_entry_price(intraday_df, setup_row)
        
        if entry_price is None:
            return None
        
        # Calculate position size based on risk
        prev_close = setup_row['prev_close']
        gap_high = setup_row['h']  # High of the day
        
        # Stop loss: 5% above the gap high
        stop_loss = gap_high * (1 + self.stop_loss_buffer)
        
        # Risk per share
        risk_per_share = abs(stop_loss - entry_price)
        
        if risk_per_share <= 0:
            return None
        
        # Position size calculation
        max_risk_dollars = self.position_size * self.max_risk_per_trade
        shares = int(max_risk_dollars / risk_per_share)
        
        if shares <= 0:
            return None
        
        # Calculate take profit levels (fade targets)
        take_profit_1 = prev_close + (entry_price - prev_close) * (1 - self.take_profit_targets[0])  # 50% fade
        take_profit_2 = prev_close + (entry_price - prev_close) * (1 - self.take_profit_targets[1])  # 70% fade
        take_profit_3 = prev_close + (entry_price - prev_close) * (1 - self.take_profit_targets[2])  # 85% fade
        
        # Simulate the trade through the day
        entry_idx = intraday_df[intraday_df['timestamp'] >= entry_time].index
        
        if len(entry_idx) == 0:
            return None
        
        trade_data = intraday_df.loc[entry_idx[0]:].copy()
        
        exit_price = None
        exit_time = None
        exit_reason = None
        
        # Check each minute for exit conditions
        for idx, row in trade_data.iterrows():
            current_price = row['low']  # Use low for short covering (conservative)
            current_time = row['timestamp']
            
            # Check stop loss (price goes against us)
            if row['high'] >= stop_loss:
                exit_price = stop_loss
                exit_time = current_time
                exit_reason = 'stop_loss'
                break
            
            # Check take profit levels (price fades down)
            if current_price <= take_profit_3:
                exit_price = take_profit_3
                exit_time = current_time
                exit_reason = 'take_profit_3'
                break
            elif current_price <= take_profit_2:
                exit_price = take_profit_2
                exit_time = current_time
                exit_reason = 'take_profit_2'
                break
            elif current_price <= take_profit_1:
                exit_price = take_profit_1
                exit_time = current_time
                exit_reason = 'take_profit_1'
                break
        
        # If no exit by end of day, exit at close
        if exit_price is None:
            exit_price = trade_data['close'].iloc[-1]
            exit_time = trade_data['timestamp'].iloc[-1]
            exit_reason = 'eod_exit'
        
        # Calculate P&L (short position)
        pnl_per_share = entry_price - exit_price  # Short P&L
        total_pnl = pnl_per_share * shares
        
        # Calculate metrics
        hold_time_minutes = (exit_time - entry_time).total_seconds() / 60
        
        return {
            'ticker': ticker,
            'date': setup_row['scan_date'].strftime('%Y-%m-%d'),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'shares': shares,
            'pnl_per_share': pnl_per_share,
            'total_pnl': total_pnl,
            'exit_reason': exit_reason,
            'hold_time_minutes': hold_time_minutes,
            'stop_loss': stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'take_profit_3': take_profit_3,
            'prev_close': prev_close,
            'gap_high': gap_high,
            'gap_pct': setup_row['gap_pct'],
            'pm_high_pct': setup_row['pm_high_pct']
        }
    
    async def backtest_os_d1_strategy(self, start_date='2024-08-01', end_date='2024-08-31', max_trades=20):
        """Backtest the complete OS D1 fade strategy"""
        
        print("🚀 OS D1 Small Cap Fade Strategy Backtesting")
        print("=" * 60)
        print("Strategy: Short small cap day one gappers expecting fade")
        
        # Load setups
        setups_df = self.load_os_d1_setups()
        
        if setups_df.empty:
            return
        
        # Filter by date range
        mask = (setups_df['scan_date'] >= start_date) & (setups_df['scan_date'] <= end_date)
        test_setups = setups_df[mask].copy()
        
        print(f"📊 Testing {len(test_setups)} setups from {start_date} to {end_date}")
        
        # Limit number of trades for testing
        if len(test_setups) > max_trades:
            test_setups = test_setups.head(max_trades)
            print(f"📉 Limited to first {max_trades} setups for testing")
        
        # Run backtests
        all_trades = []
        successful_trades = 0
        failed_data_fetches = 0
        
        for idx, setup_row in test_setups.iterrows():
            ticker = setup_row['ticker']
            date = setup_row['scan_date']
            
            print(f"📈 {successful_trades + failed_data_fetches + 1}/{len(test_setups)}: Testing {ticker} on {date.strftime('%Y-%m-%d')}")
            
            # Get intraday data
            intraday_df = self.fetch_intraday_data(ticker, date)
            
            if intraday_df.empty:
                print(f"   ❌ No intraday data available")
                failed_data_fetches += 1
                continue
            
            # Simulate trade
            trade_result = self.simulate_fade_trade(ticker, setup_row, intraday_df)
            
            if trade_result is None:
                print(f"   ❌ Could not simulate trade")
                continue
            
            all_trades.append(trade_result)
            successful_trades += 1
            
            # Show trade result
            pnl = trade_result['total_pnl']
            reason = trade_result['exit_reason']
            hold_time = trade_result['hold_time_minutes']
            
            print(f"   {'✅' if pnl > 0 else '❌'} P&L: ${pnl:,.2f} | Exit: {reason} | Hold: {hold_time:.0f}min")
            
            # Small delay to respect API limits
            await asyncio.sleep(0.1)
        
        # Analysis
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            self.analyze_backtest_results(trades_df, successful_trades, failed_data_fetches)
            
            # Save results
            output_file = f"os_d1_backtest_results_{start_date}_{end_date}.csv"
            trades_df.to_csv(output_file, index=False)
            print(f"\n💾 Backtest results saved to: {output_file}")
            
            return trades_df
        else:
            print("\n❌ No successful trades to analyze")
            return pd.DataFrame()
    
    def analyze_backtest_results(self, trades_df, successful_trades, failed_data_fetches):
        """Analyze and display backtest results"""
        
        print(f"\n{'='*60}")
        print("🎯 OS D1 FADE STRATEGY BACKTEST RESULTS")
        print(f"{'='*60}")
        
        # Basic stats
        total_pnl = trades_df['total_pnl'].sum()
        avg_pnl = trades_df['total_pnl'].mean()
        win_rate = (trades_df['total_pnl'] > 0).mean()
        
        print(f"📊 PERFORMANCE SUMMARY:")
        print(f"   • Total trades: {successful_trades}")
        print(f"   • Failed data fetches: {failed_data_fetches}")
        print(f"   • Total P&L: ${total_pnl:,.2f}")
        print(f"   • Average P&L per trade: ${avg_pnl:,.2f}")
        print(f"   • Win rate: {win_rate:.1%}")
        
        # Win/Loss breakdown
        winners = trades_df[trades_df['total_pnl'] > 0]
        losers = trades_df[trades_df['total_pnl'] <= 0]
        
        if len(winners) > 0:
            avg_winner = winners['total_pnl'].mean()
            print(f"   • Average winner: ${avg_winner:,.2f}")
        
        if len(losers) > 0:
            avg_loser = losers['total_pnl'].mean()
            print(f"   • Average loser: ${avg_loser:,.2f}")
        
        # Exit reason breakdown
        print(f"\n📈 EXIT REASON BREAKDOWN:")
        exit_reasons = trades_df['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            pct = count / len(trades_df) * 100
            avg_pnl_reason = trades_df[trades_df['exit_reason'] == reason]['total_pnl'].mean()
            print(f"   • {reason}: {count} trades ({pct:.1f}%) | Avg P&L: ${avg_pnl_reason:,.2f}")
        
        # Hold time analysis
        avg_hold_time = trades_df['hold_time_minutes'].mean()
        print(f"\n⏱️ TIMING ANALYSIS:")
        print(f"   • Average hold time: {avg_hold_time:.0f} minutes")
        
        # Best and worst trades
        best_trade = trades_df.loc[trades_df['total_pnl'].idxmax()]
        worst_trade = trades_df.loc[trades_df['total_pnl'].idxmin()]
        
        print(f"\n🏆 BEST TRADE:")
        print(f"   • {best_trade['ticker']} ({best_trade['date']}): ${best_trade['total_pnl']:,.2f}")
        print(f"   • Gap: {best_trade['gap_pct']:.1f}% | Exit: {best_trade['exit_reason']}")
        
        print(f"\n💀 WORST TRADE:")
        print(f"   • {worst_trade['ticker']} ({worst_trade['date']}): ${worst_trade['total_pnl']:,.2f}")
        print(f"   • Gap: {worst_trade['gap_pct']:.1f}% | Exit: {worst_trade['exit_reason']}")

async def main():
    """Run the OS D1 backtest system"""
    
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    backtest_system = OS_D1_BacktestSystem(api_key)
    
    # Run backtest on first part of 2025 data
    results = await backtest_system.backtest_os_d1_strategy(
        start_date='2025-01-01', 
        end_date='2025-03-31',
        max_trades=30  # Test first 30 setups
    )

if __name__ == '__main__':
    asyncio.run(main())