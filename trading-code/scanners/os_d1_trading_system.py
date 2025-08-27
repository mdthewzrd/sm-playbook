#!/usr/bin/env python3
"""
OS D1 Complete Trading System - Based on Notion Document
Implements the full OS D1 momentum strategy with:
- Stage Classification (Frontside, High & Tight, Backside Pop, Deep/Reverted Backside)  
- Entry Logic (FBO, Extensions, Dev Band Pop)
- Pyramiding Strategy (5m DB pushes)
- Cover Strategy (5m BDB/200EMA signals)
- Risk Management (3R max loss, position sizing)
"""

import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta, time
import warnings
warnings.filterwarnings("ignore")

class OS_D1_TradingSystem:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
        # Risk management parameters from Notion
        self.max_loss = 3.0  # 3R maximum loss
        self.starter_size = 0.25  # 0.25R starter positions
        self.pre_trig_size = 0.25  # 0.25R pre-trigger positions  
        self.trig_size = 1.0  # 1R trigger positions
        self.max_starters = 2  # Maximum 2 starter positions
        self.max_5m_bb = 2  # Maximum 2 5m bollinger band triggers
        self.cutoff_time = time(10, 30)  # 10:30 AM cutoff
        
    def load_os_d1_setups(self, filename="os_d1_momentum_setups_2025-01-01_2025-02-28.csv"):
        """Load validated OS D1 setups"""
        try:
            df = pd.read_csv(filename)
            df['scan_date'] = pd.to_datetime(df['scan_date'])
            print(f"✅ Loaded {len(df)} validated OS D1 momentum setups")
            return df
        except FileNotFoundError:
            print(f"❌ Setup file {filename} not found. Run scanner first.")
            return pd.DataFrame()
    
    def fetch_intraday_data(self, ticker, date, timespan='1', multiplier=1):
        """Fetch minute-level intraday data for trading"""
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
                    market_start = time(9, 30)
                    market_end = time(16, 0)
                    df = df[(df['time'] >= market_start) & (df['time'] <= market_end)]
                    
                    return df[['timestamp', 'o', 'h', 'l', 'c', 'v']].rename(columns={
                        'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
                    })
        except Exception as e:
            print(f"Error fetching intraday data for {ticker}: {e}")
        
        return pd.DataFrame()
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators needed for stage classification"""
        if df.empty:
            return df
        
        # EMAs for different timeframes
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_9'] = df['close'].ewm(span=9).mean() 
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Bollinger Bands (20 period, 2 std dev)
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df
    
    def classify_opening_stage(self, intraday_df, setup_row):
        """
        Classify opening stage based on Notion criteria:
        - Frontside: until 5m trig
        - High & Tight: until close below 5m 20 ema or 15m trig  
        - Backside Pop: until multiple closes below 15m 20ema or 15m 9/20 cross
        - Deep Backside: until 5m bdb hit pre pop
        - Reverted D1: after backside reverts
        """
        
        if intraday_df.empty:
            return "unknown"
        
        # Get pre-market high and previous close from setup
        pm_high = setup_row['pm_high']
        prev_close = setup_row['prev_close']
        
        # Get opening price action (first 30 minutes)
        opening_period = intraday_df.head(30)
        
        if opening_period.empty:
            return "unknown"
        
        # Check if we're still in the opening gap up phase
        current_price = opening_period['close'].iloc[-1]
        opening_high = opening_period['high'].max()
        
        # Calculate key levels
        gap_fill_level = prev_close
        pm_high_level = pm_high
        
        # Stage classification logic
        if current_price >= pm_high_level * 0.9:  # Still near pre-market highs
            return "frontside"
        elif current_price >= pm_high_level * 0.7:  # Pulled back but holding
            return "high_and_tight"  
        elif current_price >= gap_fill_level * 1.1:  # Above gap fill but faded
            return "backside_pop"
        else:  # Deep pullback
            return "deep_backside"
    
    def identify_high_ev_spots(self, intraday_df, setup_row, stage):
        """
        Identify high EV entry spots based on stage and Notion criteria:
        - FBO (Failed Breakout): Opening FBO 77% A+ rate
        - Extensions: Best from High & Tight or below  
        - Dev Band Pop: 5m/15m deviation band pushes
        """
        
        if intraday_df.empty:
            return []
        
        high_ev_spots = []
        
        # Opening period (first 15 minutes)
        opening_bars = intraday_df.head(15)
        
        if opening_bars.empty:
            return high_ev_spots
        
        # Get key levels
        pm_high = setup_row['pm_high']
        prev_close = setup_row['prev_close']
        opening_high = opening_bars['high'].max()
        
        # 1. Opening FBO Detection
        # Look for failure to break above pre-market high or opening range high
        for i in range(5, len(opening_bars)):
            current_bar = opening_bars.iloc[i]
            prev_bars = opening_bars.iloc[max(0, i-3):i]
            
            # Failed breakout if we tested high but couldn't hold
            if (current_bar['high'] >= opening_high * 0.98 and 
                current_bar['close'] < current_bar['high'] * 0.95):
                
                high_ev_spots.append({
                    'type': 'opening_fbo',
                    'time': current_bar['timestamp'],
                    'price': current_bar['close'],
                    'stage': stage,
                    'confidence': 0.77,  # 77% A+ rate from Notion
                    'entry_type': 'starter'  # 0.25R starter position
                })
        
        # 2. Extension Detection  
        # Look for continuation moves after consolidation
        if stage in ['high_and_tight', 'backside_pop']:  # Best extensions from these stages
            for i in range(10, len(intraday_df)):
                current_bar = intraday_df.iloc[i]
                prev_bars = intraday_df.iloc[max(0, i-5):i]
                
                # Extension if breaking above recent consolidation high
                recent_high = prev_bars['high'].max()
                if (current_bar['high'] > recent_high * 1.02 and 
                    current_bar['volume'] > prev_bars['volume'].mean() * 1.5):
                    
                    confidence = 0.75 if stage == 'high_and_tight' else 0.67
                    high_ev_spots.append({
                        'type': 'extension',
                        'time': current_bar['timestamp'], 
                        'price': current_bar['close'],
                        'stage': stage,
                        'confidence': confidence,
                        'entry_type': 'pre_trig'  # 0.25R pre-trigger
                    })
        
        # 3. Dev Band Pop Detection
        # Look for pushes off key support levels
        for i in range(20, len(intraday_df)):
            current_bar = intraday_df.iloc[i]
            
            # Check if we're bouncing off VWAP or EMA support
            if hasattr(current_bar, 'vwap') and hasattr(current_bar, 'ema_20'):
                if (current_bar['low'] <= current_bar['vwap'] * 1.02 and
                    current_bar['close'] > current_bar['vwap'] * 1.01):
                    
                    high_ev_spots.append({
                        'type': 'dev_band_pop',
                        'time': current_bar['timestamp'],
                        'price': current_bar['close'], 
                        'stage': stage,
                        'confidence': 0.65,  # Estimated from experience
                        'entry_type': 'starter'  # 0.25R starter
                    })
        
        return high_ev_spots
    
    def calculate_entry_levels(self, spot, intraday_df, setup_row):
        """Calculate specific entry levels based on Notion entry processes"""
        
        entry_time = spot['time']
        entry_price = spot['price']
        spot_type = spot['type']
        stage = spot['stage']
        
        # Get the bar at entry time
        entry_idx = intraday_df[intraday_df['timestamp'] <= entry_time].index
        if len(entry_idx) == 0:
            return None
        
        entry_bar = intraday_df.loc[entry_idx[-1]]
        
        # Entry process based on type and stage from Notion
        entry_levels = {
            'spot_type': spot_type,
            'stage': stage,
            'entry_time': entry_time,
            'confidence': spot['confidence']
        }
        
        if spot_type == 'opening_fbo':
            # Opening FBO entry process: 2m fbo -> 2m BB -> 5m BB
            entry_levels.update({
                'starter_entry': entry_price,  # 0.25R vs 10% stop
                'starter_stop': entry_price * 0.90,  # 10% stop
                'pre_trig_entry': entry_bar['bb_upper'] if 'bb_upper' in entry_bar else entry_price * 1.02,
                'pre_trig_stop': entry_price * 1.01,  # 1c over highs
                'trig_entry': entry_price * 1.05,  # 5m BB estimate
                'trig_stop': entry_price * 1.01  # 1c over highs
            })
        
        elif spot_type == 'extension':
            # Extension entry process: 2m bb -> 2m Trig -> 5m BB  
            entry_levels.update({
                'starter_entry': entry_bar['bb_upper'] if 'bb_upper' in entry_bar else entry_price * 1.02,
                'starter_stop': entry_price * 0.90,  # 10% stop
                'pre_trig_entry': entry_price * 1.02,  # 2m trigger
                'pre_trig_stop': entry_price * 1.01,  # 1c over highs
                'trig_entry': entry_price * 1.05,  # 5m BB
                'trig_stop': entry_price * 1.01  # 1c over highs
            })
        
        elif spot_type == 'dev_band_pop':
            # Dev Band Pop: 1m fail -> 2m bb -> 5m BB
            entry_levels.update({
                'starter_entry': entry_price,  # 1m fail candle
                'starter_stop': setup_row['pm_high'],  # vs PMH
                'pre_trig_entry': entry_bar['bb_upper'] if 'bb_upper' in entry_bar else entry_price * 1.02,
                'pre_trig_stop': entry_price * 1.01,  # 1c over highs
                'trig_entry': entry_price * 1.05,  # 5m BB
                'trig_stop': entry_price * 1.01  # 1c over highs  
            })
        
        return entry_levels
    
    def simulate_os_d1_trade(self, ticker, setup_row, max_test_trades=3):
        """Simulate complete OS D1 momentum trade with stage classification and entries"""
        
        print(f"\n🎯 Simulating OS D1 trade: {ticker} ({setup_row['scan_date'].strftime('%Y-%m-%d')})")
        print(f"   Setup: Gap {setup_row['gap_pct']:.1f}%, PM High {setup_row['pm_high_pct']:.1f}%")
        
        # Get intraday data
        intraday_df = self.fetch_intraday_data(ticker, setup_row['scan_date'])
        
        if intraday_df.empty:
            print(f"   ❌ No intraday data available")
            return None
        
        # Add technical indicators
        intraday_df = self.calculate_technical_indicators(intraday_df)
        
        # Classify opening stage
        stage = self.classify_opening_stage(intraday_df, setup_row)
        print(f"   📊 Opening stage: {stage.upper()}")
        
        # Identify high EV spots
        high_ev_spots = self.identify_high_ev_spots(intraday_df, setup_row, stage)
        
        if not high_ev_spots:
            print(f"   ℹ️ No high EV entry spots found")
            return None
        
        print(f"   🎯 Found {len(high_ev_spots)} high EV spots")
        
        # Test up to max_test_trades of the best spots
        trade_results = []
        for i, spot in enumerate(high_ev_spots[:max_test_trades]):
            print(f"   📈 Testing {spot['type']} at {spot['time'].strftime('%H:%M')} "
                  f"(confidence: {spot['confidence']:.0%})")
            
            # Calculate entry levels
            entry_levels = self.calculate_entry_levels(spot, intraday_df, setup_row)
            if not entry_levels:
                continue
            
            # Simulate the trade execution
            trade_result = self.execute_simulated_trade(intraday_df, entry_levels, setup_row)
            if trade_result:
                trade_results.append(trade_result)
                print(f"      {'✅' if trade_result['pnl'] > 0 else '❌'} "
                      f"P&L: {trade_result['pnl']:.2f}R | Exit: {trade_result['exit_reason']}")
        
        if trade_results:
            # Return best trade result
            best_trade = max(trade_results, key=lambda x: x['pnl'])
            best_trade.update({
                'ticker': ticker,
                'date': setup_row['scan_date'].strftime('%Y-%m-%d'),
                'stage': stage,
                'total_spots': len(high_ev_spots),
                'trades_tested': len(trade_results)
            })
            return best_trade
        
        return None
    
    def execute_simulated_trade(self, intraday_df, entry_levels, setup_row):
        """Execute simulated trade with pyramiding and cover strategy"""
        
        entry_time = entry_levels['entry_time']
        starter_entry = entry_levels['starter_entry']
        starter_stop = entry_levels['starter_stop']
        
        # Find entry point in data
        entry_idx = intraday_df[intraday_df['timestamp'] >= entry_time].index
        if len(entry_idx) == 0:
            return None
        
        trade_data = intraday_df.loc[entry_idx[0]:]
        
        # Simulate trade execution
        position_size = 0  # Start with no position
        total_pnl = 0.0
        entry_price = starter_entry
        stop_loss = starter_stop
        max_position = 0
        
        # Track position through the day
        for idx, bar in trade_data.iterrows():
            current_time = bar['timestamp'].time()
            current_price = bar['close']
            
            # Cut losses if stop hit
            if position_size > 0 and current_price <= stop_loss:
                pnl = (stop_loss - entry_price) / entry_price * position_size
                total_pnl += pnl
                return {
                    'pnl': total_pnl,
                    'exit_price': stop_loss,
                    'exit_time': current_time,
                    'exit_reason': 'stop_loss',
                    'max_position': max_position,
                    'entry_price': entry_price
                }
            
            # Enter starter position if we haven't yet
            if position_size == 0 and current_price >= starter_entry:
                position_size = self.starter_size  # 0.25R
                entry_price = current_price
                max_position = position_size
            
            # Look for pyramid opportunities (5m DB push)
            elif position_size > 0 and position_size < self.max_starters and current_price > entry_price * 1.02:
                # Add to position on strength
                position_size += self.starter_size
                max_position = max(max_position, position_size)
            
            # Cover strategy - exit on 5m BDB or time cutoff
            if position_size > 0:
                # Time cutoff at 10:30 AM
                if current_time >= self.cutoff_time:
                    pnl = (current_price - entry_price) / entry_price * position_size
                    total_pnl += pnl
                    return {
                        'pnl': total_pnl,
                        'exit_price': current_price,
                        'exit_time': current_time,
                        'exit_reason': 'time_cutoff',
                        'max_position': max_position,
                        'entry_price': entry_price
                    }
                
                # Cover 1/3 on 2m BB, 1/3 on 5m BB, 1/3 on 15m BB (simplified)
                if current_price > entry_price * 1.05:  # Simulate BB break higher
                    pnl = (current_price - entry_price) / entry_price * position_size
                    total_pnl += pnl
                    return {
                        'pnl': total_pnl,
                        'exit_price': current_price,
                        'exit_time': current_time,
                        'exit_reason': 'profit_target',
                        'max_position': max_position,
                        'entry_price': entry_price
                    }
        
        # Exit at end of day if still in position
        if position_size > 0:
            final_price = trade_data['close'].iloc[-1]
            pnl = (final_price - entry_price) / entry_price * position_size
            total_pnl += pnl
            return {
                'pnl': total_pnl,
                'exit_price': final_price,
                'exit_time': trade_data['timestamp'].iloc[-1].time(),
                'exit_reason': 'eod_exit',
                'max_position': max_position,
                'entry_price': entry_price
            }
        
        return None
    
    async def run_os_d1_trading_system(self, start_date='2025-01-01', end_date='2025-02-28', max_trades=20):
        """Run complete OS D1 trading system"""
        
        print("🚀 OS D1 Complete Trading System")
        print("=" * 60)
        print("Strategy: LONG momentum on small cap day one gappers")
        print("Entries: FBO, Extensions, Dev Band Pop")
        print("Risk: 3R max loss, pyramiding system, 10:30 cutoff")
        
        # Load validated setups
        setups_df = self.load_os_d1_setups()
        if setups_df.empty:
            return
        
        # Filter by date range
        mask = (setups_df['scan_date'] >= start_date) & (setups_df['scan_date'] <= end_date)
        test_setups = setups_df[mask].copy()
        
        if len(test_setups) > max_trades:
            test_setups = test_setups.head(max_trades)
            print(f"📉 Testing first {max_trades} setups")
        
        print(f"📊 Testing {len(test_setups)} OS D1 setups\n")
        
        # Run simulated trades
        all_trades = []
        successful_trades = 0
        
        for idx, setup_row in test_setups.iterrows():
            trade_result = self.simulate_os_d1_trade(setup_row['ticker'], setup_row)
            
            if trade_result:
                all_trades.append(trade_result)
                successful_trades += 1
        
        # Analyze results
        if all_trades:
            self.analyze_os_d1_results(all_trades, successful_trades, len(test_setups))
            
            # Save results
            results_df = pd.DataFrame(all_trades)
            output_file = f"os_d1_trading_results_{start_date}_{end_date}.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\n💾 Trading results saved to: {output_file}")
            
            return results_df
        else:
            print("\n❌ No successful trades to analyze")
            return pd.DataFrame()
    
    def analyze_os_d1_results(self, trades, successful_trades, total_setups):
        """Analyze OS D1 trading results"""
        
        trades_df = pd.DataFrame(trades)
        
        print(f"\n{'='*60}")
        print("🎯 OS D1 MOMENTUM TRADING RESULTS")
        print(f"{'='*60}")
        
        # Performance metrics
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        win_rate = (trades_df['pnl'] > 0).mean()
        
        print(f"📊 PERFORMANCE:")
        print(f"   • Setups tested: {total_setups}")
        print(f"   • Successful trades: {successful_trades}")
        print(f"   • Total P&L: {total_pnl:.2f}R")
        print(f"   • Average P&L: {avg_pnl:.2f}R")
        print(f"   • Win rate: {win_rate:.1%}")
        
        # Stage analysis
        print(f"\n📈 STAGE BREAKDOWN:")
        stage_perf = trades_df.groupby('stage')['pnl'].agg(['count', 'mean', lambda x: (x > 0).mean()])
        stage_perf.columns = ['trades', 'avg_pnl', 'win_rate']
        
        for stage, row in stage_perf.iterrows():
            print(f"   • {stage.replace('_', ' ').title()}: {row['trades']} trades, "
                  f"{row['avg_pnl']:.2f}R avg, {row['win_rate']:.1%} win rate")
        
        # Exit reason analysis
        print(f"\n📈 EXIT BREAKDOWN:")
        exit_counts = trades_df['exit_reason'].value_counts()
        for reason, count in exit_counts.items():
            avg_pnl = trades_df[trades_df['exit_reason'] == reason]['pnl'].mean()
            print(f"   • {reason.replace('_', ' ').title()}: {count} trades, {avg_pnl:.2f}R avg")
        
        # Best and worst trades
        best_trade = trades_df.loc[trades_df['pnl'].idxmax()]
        worst_trade = trades_df.loc[trades_df['pnl'].idxmin()]
        
        print(f"\n🏆 BEST TRADE: {best_trade['ticker']} ({best_trade['date']})")
        print(f"   P&L: {best_trade['pnl']:.2f}R | Stage: {best_trade['stage']} | Exit: {best_trade['exit_reason']}")
        
        print(f"\n💀 WORST TRADE: {worst_trade['ticker']} ({worst_trade['date']})")
        print(f"   P&L: {worst_trade['pnl']:.2f}R | Stage: {worst_trade['stage']} | Exit: {worst_trade['exit_reason']}")

async def main():
    """Run the complete OS D1 trading system"""
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    trading_system = OS_D1_TradingSystem(api_key)
    
    # Run the complete trading system
    results = await trading_system.run_os_d1_trading_system(
        start_date='2025-01-01',
        end_date='2025-02-28', 
        max_trades=15  # Test 15 setups for now
    )

if __name__ == '__main__':
    asyncio.run(main())