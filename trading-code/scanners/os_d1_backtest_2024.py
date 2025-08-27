#!/usr/bin/env python3
"""
OS D1 Small Cap Fade Strategy - 2024 Historical Backtest

Since 2025 intraday data isn't available, this creates a simulation using 2024 historical data
and the same OS D1 logic, plus simulates intraday fade behavior using daily data patterns
"""

import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

class OS_D1_2024_Backtest:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
        # Strategy parameters
        self.position_size = 10000  # $10K per position
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        self.stop_loss_buffer = 0.05  # 5% above HOD for stop
        
    async def fetch_grouped_daily_data(self, session, date, adjusted=True):
        """Fetch grouped daily data"""
        adj_str = "true" if adjusted else "false"
        url = f"{self.base_url}/v2/aggs/grouped/locale/us/market/stocks/{date}"
        params = {'adjusted': adj_str, 'apiKey': self.api_key}
        
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if 'results' in data:
                    df = pd.DataFrame(data['results'])
                    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
                    df.rename(columns={'T': 'ticker'}, inplace=True)
                    return df
            return pd.DataFrame()
    
    async def find_2024_os_d1_setups(self, start_date='2024-01-01', end_date='2024-12-31', max_days=250):
        """Find OS D1 setups in 2024 historical data"""
        
        print("🔍 Finding OS D1 setups in 2024 historical data...")
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = [d.strftime('%Y-%m-%d') for d in date_range if d.weekday() < 5]
        
        # Limit for testing
        if len(trading_days) > max_days:
            trading_days = trading_days[:max_days]
            print(f"📉 Limited to first {max_days} days for comprehensive analysis")
        
        all_setups = []
        
        async with aiohttp.ClientSession() as session:
            for i, scan_date in enumerate(trading_days[1:]):  # Skip first day (need previous day)
                prev_date = trading_days[i]  # Previous trading day
                
                print(f"📅 {i+1}/{len(trading_days)-1}: Scanning {scan_date}")
                
                try:
                    # Fetch data for both days
                    current_data = await self.fetch_grouped_daily_data(session, scan_date)
                    prev_data = await self.fetch_grouped_daily_data(session, prev_date)
                    
                    if current_data.empty or prev_data.empty:
                        continue
                    
                    # Merge data
                    merged = current_data.merge(prev_data, on='ticker', suffixes=('', '_prev'))
                    
                    # Apply OS D1 criteria
                    merged['pm_high'] = merged['h']
                    merged['pm_vol'] = merged['v']
                    merged['prev_close'] = merged['c_prev']
                    merged['prev_high'] = merged['h_prev']
                    merged['gap'] = merged['o'] - merged['prev_close']
                    
                    # OS D1 trigger day formula
                    merged['trig_day'] = (
                        (merged['pm_high'] / merged['prev_close'] - 1 >= 0.5) &
                        (merged['gap'] >= 0.5) & 
                        (merged['o'] / merged['prev_high'] - 1 >= 0.3) &
                        (merged['pm_vol'] >= 5000000) &
                        (merged['prev_close'] >= 0.75)
                    ).astype(int)
                    
                    # Get setups for this date
                    day_setups = merged[merged['trig_day'] == 1].copy()
                    
                    if not day_setups.empty:
                        day_setups['scan_date'] = scan_date
                        day_setups['gap_pct'] = (day_setups['gap'] / day_setups['prev_close']) * 100
                        day_setups['pm_high_pct'] = (day_setups['pm_high'] / day_setups['prev_close'] - 1) * 100
                        
                        all_setups.append(day_setups[['ticker', 'scan_date', 'c', 'gap', 'gap_pct', 'pm_high_pct', 
                                                     'pm_vol', 'prev_close', 'h', 'l', 'o']])
                        
                        print(f"   ✅ Found {len(day_setups)} setups")
                        for _, row in day_setups.head(3).iterrows():
                            print(f"      📊 {row['ticker']}: Gap {row['gap_pct']:.1f}%, PM High {row['pm_high_pct']:.1f}%")
                    else:
                        print(f"   ℹ️ No setups found")
                    
                    # Small delay
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    continue
        
        if all_setups:
            final_df = pd.concat(all_setups, ignore_index=True)
            final_df['scan_date'] = pd.to_datetime(final_df['scan_date'])
            print(f"\n✅ Found {len(final_df)} total OS D1 setups in 2024")
            return final_df
        else:
            print(f"\n❌ No OS D1 setups found")
            return pd.DataFrame()
    
    def simulate_daily_fade_trade(self, setup_row):
        """Simulate fade trade using daily OHLC data"""
        
        # Extract key values
        ticker = setup_row['ticker']
        open_price = setup_row['o']
        high_price = setup_row['h']
        low_price = setup_row['l']
        close_price = setup_row['c']
        prev_close = setup_row['prev_close']
        
        # Entry: Short near the gap open (more realistic entry point)
        entry_price = open_price  # Enter at the gap open
        
        # Stop loss: Above the high with buffer
        stop_loss = high_price * (1 + self.stop_loss_buffer)
        
        # Position sizing
        risk_per_share = stop_loss - entry_price
        if risk_per_share <= 0:
            return None
        
        max_risk_dollars = self.position_size * self.max_risk_per_trade
        shares = int(max_risk_dollars / risk_per_share)
        
        if shares <= 0:
            return None
        
        # Fade targets
        fade_50 = prev_close + (entry_price - prev_close) * 0.5  # 50% fade back to prev close
        fade_70 = prev_close + (entry_price - prev_close) * 0.3  # 70% fade
        fade_85 = prev_close + (entry_price - prev_close) * 0.15  # 85% fade
        
        # Simulate trade outcome based on daily range
        exit_price = None
        exit_reason = None
        
        # Check if we got stopped out (stop loss was hit)
        if high_price >= stop_loss:  # Stop loss was hit
            # We were stopped out
            exit_price = stop_loss
            exit_reason = 'stop_loss'
        
        # Check if we hit fade targets (low went down to our targets)
        elif low_price <= fade_85:
            exit_price = fade_85
            exit_reason = 'fade_85'
        elif low_price <= fade_70:
            exit_price = fade_70
            exit_reason = 'fade_70'
        elif low_price <= fade_50:
            exit_price = fade_50
            exit_reason = 'fade_50'
        else:
            # Exit at close
            exit_price = close_price
            exit_reason = 'eod_close'
        
        # Calculate P&L (short position)
        pnl_per_share = entry_price - exit_price
        total_pnl = pnl_per_share * shares
        
        return {
            'ticker': ticker,
            'date': setup_row['scan_date'].strftime('%Y-%m-%d'),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'pnl_per_share': pnl_per_share,
            'total_pnl': total_pnl,
            'exit_reason': exit_reason,
            'stop_loss': stop_loss,
            'fade_50': fade_50,
            'fade_70': fade_70,
            'fade_85': fade_85,
            'prev_close': prev_close,
            'gap_pct': setup_row['gap_pct'],
            'pm_high_pct': setup_row['pm_high_pct'],
            'day_high': high_price,
            'day_low': low_price,
            'day_close': close_price
        }
    
    async def run_2024_backtest(self):
        """Run the complete 2024 OS D1 backtest"""
        
        print("🚀 OS D1 2024 Historical Backtest")
        print("=" * 60)
        
        # Find 2024 setups
        setups_df = await self.find_2024_os_d1_setups()
        
        if setups_df.empty:
            return
        
        print(f"\n📊 Backtesting {len(setups_df)} setups...")
        
        # Run simulations
        all_trades = []
        
        for idx, setup_row in setups_df.iterrows():
            ticker = setup_row['ticker']
            date = setup_row['scan_date']
            
            trade_result = self.simulate_daily_fade_trade(setup_row)
            
            if trade_result is None:
                continue
            
            all_trades.append(trade_result)
            
            # Show progress
            if len(all_trades) % 10 == 0:
                print(f"📈 Processed {len(all_trades)} trades...")
        
        # Analyze results
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            self.analyze_results(trades_df)
            
            # Save results
            output_file = "os_d1_2024_backtest_results.csv"
            trades_df.to_csv(output_file, index=False)
            print(f"\n💾 Results saved to: {output_file}")
            
            return trades_df
        else:
            print("❌ No successful trades")
            return pd.DataFrame()
    
    def analyze_results(self, trades_df):
        """Analyze backtest results"""
        
        print(f"\n{'='*60}")
        print("🎯 OS D1 2024 BACKTEST RESULTS")
        print(f"{'='*60}")
        
        # Performance metrics
        total_trades = len(trades_df)
        total_pnl = trades_df['total_pnl'].sum()
        avg_pnl = trades_df['total_pnl'].mean()
        win_rate = (trades_df['total_pnl'] > 0).mean()
        
        print(f"📊 PERFORMANCE:")
        print(f"   • Total trades: {total_trades}")
        print(f"   • Total P&L: ${total_pnl:,.2f}")
        print(f"   • Average P&L: ${avg_pnl:,.2f}")
        print(f"   • Win rate: {win_rate:.1%}")
        
        # Winners vs losers
        winners = trades_df[trades_df['total_pnl'] > 0]
        losers = trades_df[trades_df['total_pnl'] <= 0]
        
        if len(winners) > 0:
            print(f"   • Average winner: ${winners['total_pnl'].mean():,.2f}")
        if len(losers) > 0:
            print(f"   • Average loser: ${losers['total_pnl'].mean():,.2f}")
        
        # Exit reasons
        print(f"\n📈 EXIT BREAKDOWN:")
        exit_counts = trades_df['exit_reason'].value_counts()
        for reason, count in exit_counts.items():
            pct = count / total_trades * 100
            avg_pnl = trades_df[trades_df['exit_reason'] == reason]['total_pnl'].mean()
            print(f"   • {reason}: {count} ({pct:.1f}%) | Avg P&L: ${avg_pnl:,.2f}")
        
        # Best/worst trades
        best_trade = trades_df.loc[trades_df['total_pnl'].idxmax()]
        worst_trade = trades_df.loc[trades_df['total_pnl'].idxmin()]
        
        print(f"\n🏆 BEST TRADE: {best_trade['ticker']} ({best_trade['date']})")
        print(f"   P&L: ${best_trade['total_pnl']:,.2f} | Gap: {best_trade['gap_pct']:.1f}% | Exit: {best_trade['exit_reason']}")
        
        print(f"\n💀 WORST TRADE: {worst_trade['ticker']} ({worst_trade['date']})")
        print(f"   P&L: ${worst_trade['total_pnl']:,.2f} | Gap: {worst_trade['gap_pct']:.1f}% | Exit: {worst_trade['exit_reason']}")

async def main():
    """Run 2024 historical backtest"""
    
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    backtest = OS_D1_2024_Backtest(api_key)
    
    results = await backtest.run_2024_backtest()

if __name__ == '__main__':
    asyncio.run(main())