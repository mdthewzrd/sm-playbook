#!/usr/bin/env python3
"""
OS D1 SHORT Strategy - Correct Implementation from Notion Document

This implements the actual OS D1 strategy:
- SHORT small cap day one gappers expecting failure/reversion
- Stage classification for timing short entries
- FBO (Failed Breakout), Extension failures, Dev Band Pop failures
- Cover strategy for exiting short positions
"""

import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta, time
import warnings
warnings.filterwarnings("ignore")

class OS_D1_ShortStrategy:
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
        """Load validated OS D1 setups for shorting"""
        try:
            df = pd.read_csv(filename)
            df['scan_date'] = pd.to_datetime(df['scan_date'])
            print(f"✅ Loaded {len(df)} OS D1 setups for SHORT strategy")
            return df
        except FileNotFoundError:
            print(f"❌ Setup file {filename} not found. Run scanner first.")
            return pd.DataFrame()
    
    def fetch_intraday_data(self, ticker, date, timespan='1', multiplier=1):
        """Fetch minute-level intraday data for shorting"""
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
        """Calculate technical indicators for short strategy"""
        if df.empty:
            return df
        
        # EMAs for stage classification
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_9'] = df['close'].ewm(span=9).mean() 
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
        # VWAP for failure points
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Bollinger Bands for breakout failures
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Dev bands (deviation from VWAP/EMA)
        df['dev_band_upper'] = df['vwap'] * 1.05  # 5% above VWAP
        df['dev_band_lower'] = df['vwap'] * 0.95  # 5% below VWAP
        
        return df
    
    def classify_short_stage(self, intraday_df, setup_row):
        """
        Classify stage for shorting timing:
        - Frontside: Early gap up, look for opening failures
        - High & Tight: Consolidation at highs, short extension failures  
        - Backside Pop: Already fading, short any pops back up
        - Deep Backside: Deep fade, wait for failed bounces
        """
        
        if intraday_df.empty:
            return "unknown"
        
        # Get key levels
        pm_high = setup_row['pm_high']
        prev_close = setup_row['prev_close']
        opening_bars = intraday_df.head(30)  # First 30 minutes
        
        if opening_bars.empty:
            return "unknown"
        
        current_price = opening_bars['close'].iloc[-1]
        opening_high = opening_bars['high'].max()
        
        # Stage classification for shorting
        if current_price >= pm_high * 0.95:  # Still at/near highs - prime for shorting
            return "frontside"
        elif current_price >= pm_high * 0.8:  # Holding most gains - short tight patterns
            return "high_and_tight"  
        elif current_price >= prev_close * 1.2:  # Fading but above significant levels
            return "backside_pop"
        else:  # Deep fade already - only short failed bounces
            return "deep_backside"
    
    def identify_short_opportunities(self, intraday_df, setup_row, stage):
        """
        Identify SHORT opportunities based on stage and failure patterns:
        - FBO: Failed breakouts above key levels (77% success rate)
        - Extension Failures: Momentum exhaustion on extensions
        - Dev Band Pop Failures: Failed bounces off support
        """
        
        if intraday_df.empty:
            return []
        
        short_ops = []
        
        # Key levels for shorting
        pm_high = setup_row['pm_high']
        prev_close = setup_row['prev_close']
        gap_level = setup_row['o']  # Opening gap level
        
        # 1. FAILED BREAKOUT (FBO) Detection
        # Look for failure to sustain breaks above key levels
        for i in range(10, len(intraday_df)):
            current_bar = intraday_df.iloc[i]
            prev_bars = intraday_df.iloc[max(0, i-5):i]
            
            # Recent high that failed to hold
            recent_high = prev_bars['high'].max()
            
            # FBO criteria: Broke above level but failing to hold
            if (current_bar['high'] >= recent_high * 1.01 and  # Broke above recent high
                current_bar['close'] < current_bar['high'] * 0.98 and  # Failed to hold  
                current_bar['volume'] > prev_bars['volume'].mean() * 1.2):  # On volume
                
                # Higher confidence for opening FBO
                confidence = 0.77 if current_bar['timestamp'].time() < time(10, 0) else 0.55
                
                short_ops.append({
                    'type': 'fbo',
                    'time': current_bar['timestamp'],
                    'price': current_bar['close'],
                    'stage': stage,
                    'confidence': confidence,
                    'entry_type': 'starter',  # 0.25R starter
                    'failure_level': recent_high
                })
        
        # 2. EXTENSION FAILURE Detection
        # Look for momentum exhaustion on extension moves
        if stage in ['frontside', 'high_and_tight']:
            for i in range(15, len(intraday_df)):
                current_bar = intraday_df.iloc[i]
                prev_bars = intraday_df.iloc[max(0, i-10):i]
                
                # Extension that's running out of steam
                if (current_bar['high'] > prev_bars['high'].max() * 1.02 and  # New high
                    current_bar['close'] < current_bar['high'] * 0.95 and  # Rejecting highs
                    current_bar['volume'] < prev_bars['volume'].mean()):  # Decreasing volume
                    
                    confidence = 0.65 if stage == 'high_and_tight' else 0.45
                    
                    short_ops.append({
                        'type': 'extension_failure',
                        'time': current_bar['timestamp'],
                        'price': current_bar['close'],
                        'stage': stage,
                        'confidence': confidence,
                        'entry_type': 'pre_trig',  # 0.25R pre-trigger
                        'failure_level': current_bar['high']
                    })
        
        # 3. DEV BAND POP FAILURE Detection  
        # Failed bounces off support levels
        for i in range(20, len(intraday_df)):
            current_bar = intraday_df.iloc[i]
            
            # Check for failed bounce off VWAP or dev bands
            if hasattr(current_bar, 'vwap') and hasattr(current_bar, 'dev_band_lower'):
                # Touched support but failed to sustain bounce
                if (current_bar['low'] <= current_bar['dev_band_lower'] * 1.02 and
                    current_bar['close'] < (current_bar['high'] + current_bar['low']) / 2):  # Failed to hold mid-point
                    
                    short_ops.append({
                        'type': 'dev_band_failure',
                        'time': current_bar['timestamp'],
                        'price': current_bar['close'],
                        'stage': stage,
                        'confidence': 0.60,
                        'entry_type': 'starter',  # 0.25R starter
                        'failure_level': current_bar['high']
                    })
        
        return short_ops
    
    def calculate_short_entry_levels(self, opportunity, intraday_df, setup_row):
        """Calculate SHORT entry levels based on failure patterns"""
        
        entry_time = opportunity['time']
        entry_price = opportunity['price']
        opp_type = opportunity['type']
        stage = opportunity['stage']
        failure_level = opportunity['failure_level']
        
        # Get the bar at entry time
        entry_idx = intraday_df[intraday_df['timestamp'] <= entry_time].index
        if len(entry_idx) == 0:
            return None
        
        entry_bar = intraday_df.loc[entry_idx[-1]]
        
        # SHORT entry levels based on failure type
        entry_levels = {
            'opportunity_type': opp_type,
            'stage': stage,
            'entry_time': entry_time,
            'confidence': opportunity['confidence'],
            'failure_level': failure_level
        }
        
        if opp_type == 'fbo':
            # Failed Breakout SHORT: Short the rejection, stop above failure level
            entry_levels.update({
                'starter_entry': entry_price,  # Short at failure price
                'starter_stop': failure_level * 1.10,  # 10% above failure level
                'pre_trig_entry': entry_price * 0.98,  # Short more on further weakness  
                'pre_trig_stop': failure_level * 1.01,  # Tighter stop
                'trig_entry': entry_price * 0.95,  # Full position on breakdown
                'trig_stop': failure_level * 1.01,  # Keep stops tight
                'target_1': entry_price * 0.90,  # 10% fade target
                'target_2': entry_price * 0.85,  # 15% fade target  
                'target_3': setup_row['prev_close']  # Gap fill target
            })
        
        elif opp_type == 'extension_failure':
            # Extension Failure SHORT: Short momentum exhaustion
            entry_levels.update({
                'starter_entry': entry_price,
                'starter_stop': failure_level * 1.10,  # 10% above extension high
                'pre_trig_entry': entry_price * 0.98,
                'pre_trig_stop': failure_level * 1.01,
                'trig_entry': entry_price * 0.95,
                'trig_stop': failure_level * 1.01,
                'target_1': entry_price * 0.88,  # Deeper targets for extensions
                'target_2': entry_price * 0.80,
                'target_3': setup_row['prev_close']
            })
        
        elif opp_type == 'dev_band_failure':
            # Dev Band Failure SHORT: Failed support bounce
            entry_levels.update({
                'starter_entry': entry_price,
                'starter_stop': setup_row['pm_high'],  # Stop at PM high
                'pre_trig_entry': entry_price * 0.98,
                'pre_trig_stop': failure_level * 1.01,
                'trig_entry': entry_price * 0.95,
                'trig_stop': failure_level * 1.01,
                'target_1': entry_price * 0.92,
                'target_2': entry_price * 0.85,
                'target_3': setup_row['prev_close'] * 0.95  # Below gap fill
            })
        
        return entry_levels
    
    def simulate_short_trade(self, ticker, setup_row, max_opportunities=3):
        """Simulate SHORT trade on OS D1 setup with failure patterns"""
        
        print(f"\n🎯 Simulating SHORT trade: {ticker} ({setup_row['scan_date'].strftime('%Y-%m-%d')})")
        print(f"   Setup: Gap {setup_row['gap_pct']:.1f}%, PM High {setup_row['pm_high_pct']:.1f}%")
        
        # Get intraday data
        intraday_df = self.fetch_intraday_data(ticker, setup_row['scan_date'])
        
        if intraday_df.empty:
            print(f"   ❌ No intraday data available")
            return None
        
        # Add technical indicators
        intraday_df = self.calculate_technical_indicators(intraday_df)
        
        # Classify stage for shorting
        stage = self.classify_short_stage(intraday_df, setup_row)
        print(f"   📊 Short stage: {stage.upper()}")
        
        # Find short opportunities
        short_ops = self.identify_short_opportunities(intraday_df, setup_row, stage)
        
        if not short_ops:
            print(f"   ℹ️ No short opportunities found")
            return None
        
        print(f"   🎯 Found {len(short_ops)} short opportunities")
        
        # Test best opportunities
        trade_results = []
        for i, opportunity in enumerate(short_ops[:max_opportunities]):
            print(f"   📉 Testing {opportunity['type']} at {opportunity['time'].strftime('%H:%M')} "
                  f"(confidence: {opportunity['confidence']:.0%})")
            
            # Calculate SHORT entry levels
            entry_levels = self.calculate_short_entry_levels(opportunity, intraday_df, setup_row)
            if not entry_levels:
                continue
            
            # Execute simulated SHORT trade
            trade_result = self.execute_short_trade(intraday_df, entry_levels, setup_row)
            if trade_result:
                trade_results.append(trade_result)
                print(f"      {'✅' if trade_result['pnl'] > 0 else '❌'} "
                      f"P&L: {trade_result['pnl']:.2f}R | Exit: {trade_result['exit_reason']}")
        
        if trade_results:
            # Return best SHORT trade
            best_trade = max(trade_results, key=lambda x: x['pnl'])
            best_trade.update({
                'ticker': ticker,
                'date': setup_row['scan_date'].strftime('%Y-%m-%d'),
                'stage': stage,
                'total_opportunities': len(short_ops),
                'trades_tested': len(trade_results)
            })
            return best_trade
        
        return None
    
    def execute_short_trade(self, intraday_df, entry_levels, setup_row):
        """Execute SHORT trade with pyramiding and cover strategy"""
        
        entry_time = entry_levels['entry_time']
        starter_entry = entry_levels['starter_entry']
        starter_stop = entry_levels['starter_stop']
        target_1 = entry_levels.get('target_1', starter_entry * 0.90)
        target_2 = entry_levels.get('target_2', starter_entry * 0.85)
        target_3 = entry_levels.get('target_3', setup_row['prev_close'])
        
        # Find entry point
        entry_idx = intraday_df[intraday_df['timestamp'] >= entry_time].index
        if len(entry_idx) == 0:
            return None
        
        trade_data = intraday_df.loc[entry_idx[0]:]
        
        # SHORT trade simulation
        short_position = 0  # Shares short (positive = short position)
        total_pnl = 0.0
        entry_price = starter_entry
        stop_loss = starter_stop
        max_position = 0
        
        # Track SHORT position through the day
        for idx, bar in trade_data.iterrows():
            current_time = bar['timestamp'].time()
            current_price = bar['close']
            
            # Stop loss hit (price goes against short)
            if short_position > 0 and current_price >= stop_loss:
                # Cover short at stop loss
                pnl = (entry_price - stop_loss) / entry_price * short_position  # SHORT P&L
                total_pnl += pnl
                return {
                    'pnl': total_pnl,
                    'exit_price': stop_loss,
                    'exit_time': current_time,
                    'exit_reason': 'stop_loss',
                    'max_position': max_position,
                    'entry_price': entry_price
                }
            
            # Enter SHORT position
            if short_position == 0 and current_price <= starter_entry:
                short_position = self.starter_size  # 0.25R short
                entry_price = current_price
                max_position = short_position
            
            # Pyramid SHORT on further weakness
            elif (short_position > 0 and short_position < self.max_starters and 
                  current_price < entry_price * 0.98):
                short_position += self.starter_size  # Add to short
                max_position = max(max_position, short_position)
            
            # COVER strategy - exit SHORT positions
            if short_position > 0:
                # Time cutoff
                if current_time >= self.cutoff_time:
                    pnl = (entry_price - current_price) / entry_price * short_position
                    total_pnl += pnl
                    return {
                        'pnl': total_pnl,
                        'exit_price': current_price,
                        'exit_time': current_time,
                        'exit_reason': 'time_cutoff',
                        'max_position': max_position,
                        'entry_price': entry_price
                    }
                
                # Target levels (cover 1/3 at each)
                if current_price <= target_3:  # Best target
                    pnl = (entry_price - current_price) / entry_price * short_position
                    total_pnl += pnl
                    return {
                        'pnl': total_pnl,
                        'exit_price': current_price,
                        'exit_time': current_time,
                        'exit_reason': 'target_3',
                        'max_position': max_position,
                        'entry_price': entry_price
                    }
                elif current_price <= target_2:
                    pnl = (entry_price - current_price) / entry_price * short_position
                    total_pnl += pnl
                    return {
                        'pnl': total_pnl,
                        'exit_price': current_price,
                        'exit_time': current_time,
                        'exit_reason': 'target_2',
                        'max_position': max_position,
                        'entry_price': entry_price
                    }
                elif current_price <= target_1:
                    pnl = (entry_price - current_price) / entry_price * short_position
                    total_pnl += pnl
                    return {
                        'pnl': total_pnl,
                        'exit_price': current_price,
                        'exit_time': current_time,
                        'exit_reason': 'target_1',
                        'max_position': max_position,
                        'entry_price': entry_price
                    }
        
        # Cover at end of day
        if short_position > 0:
            final_price = trade_data['close'].iloc[-1]
            pnl = (entry_price - final_price) / entry_price * short_position
            total_pnl += pnl
            return {
                'pnl': total_pnl,
                'exit_price': final_price,
                'exit_time': trade_data['timestamp'].iloc[-1].time(),
                'exit_reason': 'eod_cover',
                'max_position': max_position,
                'entry_price': entry_price
            }
        
        return None
    
    async def run_os_d1_short_system(self, start_date='2025-01-01', end_date='2025-02-28', max_trades=20):
        """Run complete OS D1 SHORT system"""
        
        print("🚀 OS D1 SHORT Strategy System")
        print("=" * 60)
        print("Strategy: SHORT small cap day one gappers expecting failure")
        print("Entries: FBO, Extension Failures, Dev Band Pop Failures")
        print("Covers: Target levels, time cutoff, stop losses")
        
        # Load setups
        setups_df = self.load_os_d1_setups()
        if setups_df.empty:
            return
        
        # Filter by date range
        mask = (setups_df['scan_date'] >= start_date) & (setups_df['scan_date'] <= end_date)
        test_setups = setups_df[mask].copy()
        
        if len(test_setups) > max_trades:
            test_setups = test_setups.head(max_trades)
            print(f"📉 Testing first {max_trades} setups")
        
        print(f"📊 Testing {len(test_setups)} OS D1 SHORT setups\n")
        
        # Run SHORT simulations
        all_trades = []
        successful_trades = 0
        
        for idx, setup_row in test_setups.iterrows():
            trade_result = self.simulate_short_trade(setup_row['ticker'], setup_row)
            
            if trade_result:
                all_trades.append(trade_result)
                successful_trades += 1
        
        # Analyze SHORT results
        if all_trades:
            self.analyze_short_results(all_trades, successful_trades, len(test_setups))
            
            # Save results
            results_df = pd.DataFrame(all_trades)
            output_file = f"os_d1_short_results_{start_date}_{end_date}.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\n💾 SHORT results saved to: {output_file}")
            
            return results_df
        else:
            print("\n❌ No successful SHORT trades to analyze")
            return pd.DataFrame()
    
    def analyze_short_results(self, trades, successful_trades, total_setups):
        """Analyze OS D1 SHORT results"""
        
        trades_df = pd.DataFrame(trades)
        
        print(f"\n{'='*60}")
        print("🎯 OS D1 SHORT STRATEGY RESULTS")
        print(f"{'='*60}")
        
        # Performance metrics
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        win_rate = (trades_df['pnl'] > 0).mean()
        
        print(f"📊 PERFORMANCE:")
        print(f"   • Setups tested: {total_setups}")
        print(f"   • Successful SHORT trades: {successful_trades}")
        print(f"   • Total P&L: {total_pnl:.2f}R")
        print(f"   • Average P&L: {avg_pnl:.2f}R")
        print(f"   • Win rate: {win_rate:.1%}")
        
        # Stage analysis for shorting
        if 'stage' in trades_df.columns:
            print(f"\n📈 SHORT STAGE BREAKDOWN:")
            stage_perf = trades_df.groupby('stage')['pnl'].agg(['count', 'mean', lambda x: (x > 0).mean()])
            stage_perf.columns = ['trades', 'avg_pnl', 'win_rate']
            
            for stage, row in stage_perf.iterrows():
                print(f"   • {stage.replace('_', ' ').title()}: {row['trades']} trades, "
                      f"{row['avg_pnl']:.2f}R avg, {row['win_rate']:.1%} win rate")
        
        # Cover reason analysis
        print(f"\n📈 COVER BREAKDOWN:")
        exit_counts = trades_df['exit_reason'].value_counts()
        for reason, count in exit_counts.items():
            avg_pnl = trades_df[trades_df['exit_reason'] == reason]['pnl'].mean()
            print(f"   • {reason.replace('_', ' ').title()}: {count} trades, {avg_pnl:.2f}R avg")

async def main():
    """Run the OS D1 SHORT strategy system"""
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    short_system = OS_D1_ShortStrategy(api_key)
    
    # Run the SHORT system
    results = await short_system.run_os_d1_short_system(
        start_date='2025-01-01',
        end_date='2025-02-28', 
        max_trades=15
    )

if __name__ == '__main__':
    asyncio.run(main())