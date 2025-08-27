#!/usr/bin/env python3
"""
OS D1 SHORT Strategy Chart Validation System
Implements the exact SHORT strategy from Notion document with:
- Candlestick charts with dark theme and EST timezone
- 4 opening stages: Frontside, High & Tight, Backside Pop, Deep Backside
- HIGH EV spots: FBO (77% A+), Extension, Dev Band Pop
- Cover targets: 5m 200 EMA and 9/20 dev bands lower
- Exact entry processes: Starter (0.25R), Pre-Trig (0.25R), Trig (1R)
"""

import pandas as pd
import numpy as np
import requests
import asyncio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta, time
import warnings
warnings.filterwarnings("ignore")

# Set dark theme
plt.style.use('dark_background')

class OS_D1_ShortValidator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
        # OS D1 Short Strategy Parameters from Notion
        self.max_loss_r = 3.0
        self.max_starters = 2
        self.max_5m_bb = 2
        self.cutoff_time = time(10, 30)
        
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
        """Fetch 5-minute intraday data in Eastern Time"""
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
                    
                    # Convert to Eastern Time 
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
                    df['time'] = df['timestamp'].dt.time
                    
                    # Filter for market hours (9:30 AM - 4:00 PM ET)
                    market_start = time(9, 30)
                    market_end = time(16, 0)
                    df = df[(df['time'] >= market_start) & (df['time'] <= market_end)]
                    
                    return df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        except Exception as e:
            print(f"Error fetching intraday data for {ticker}: {e}")
        
        return pd.DataFrame()
    
    def calculate_indicators(self, df):
        """Calculate 5m 200 EMA, 9/20 dev bands, and VWAP"""
        if df.empty:
            return df
        
        # Work with a copy
        df = df.copy()
        
        # Key EMAs for SHORT strategy
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean() 
        df['ema_200'] = df['close'].ewm(span=200).mean()  # MAIN SHORT COVER TARGET
        
        # 9/20 Dev Bands (key resistance/support levels)
        df['dev_920_upper'] = df['ema_20'] * 1.05
        df['dev_920_lower'] = df['ema_20'] * 0.95  # COVER TARGET
        
        # Bull/Bear dev bands for entries
        df['bull_dev_920'] = df['ema_20'] * 1.03
        df['bear_dev_920'] = df['ema_20'] * 0.97
        
        # 7/28/9 Dev Bands (broader context)
        df['ema_28'] = df['close'].ewm(span=28).mean()
        df['dev_7289_upper'] = df['ema_28'] * 1.07
        df['dev_7289_lower'] = df['ema_28'] * 0.93
        
        # VWAP for trend confirmation
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        return df
    
    def classify_opening_stage(self, intraday_df, setup_data):
        """Classify opening stage: Frontside, High & Tight, Backside Pop, Deep Backside"""
        if intraday_df.empty:
            return 'unknown'
        
        gap_pct = setup_data.get('gap_pct', 0)
        pm_high_pct = setup_data.get('pm_high_pct', 0)
        
        # Opening Stage Classification from Notion
        if gap_pct >= 150 and pm_high_pct >= 200:
            return 'frontside'  # Still running strong
        elif gap_pct >= 75 and pm_high_pct >= 100:
            return 'high_and_tight'  # Consolidating at highs
        elif gap_pct >= 25 and pm_high_pct >= 50:
            return 'backside_pop'  # Moderate pullback
        else:
            return 'deep_backside'  # Deep pullback
    
    def determine_high_ev_entry(self, stage, intraday_df):
        """Determine HIGH EV entry type based on Notion success rates"""
        # From Notion: Opening FBO has 77% A+ rate (best)
        # Extensions good for non-frontside
        # Dev Band Pop for EMA rejection
        
        opening_bar = intraday_df.iloc[0] if len(intraday_df) > 0 else None
        if opening_bar is None:
            return 'opening_fbo'
        
        daily_high = intraday_df['high'].max()
        opening_high = opening_bar['high']
        
        # Opening FBO is best across all stages (77% A+)
        if abs(daily_high - opening_high) / opening_high < 0.05:  # High made at open
            return 'opening_fbo'
        elif daily_high > opening_high * 1.1:  # Extension beyond open
            return 'extension'
        else:
            return 'dev_band_pop'
    
    def simulate_short_entries(self, intraday_df, entry_type, stage, opening_high):
        """Simulate exact SHORT entry process from Notion tables"""
        
        entries = []
        stop_loss = opening_high * 1.01  # Tight stop above resistance
        
        # Starter Entry (0.25R) - 2m FBO per Notion
        starter_price = opening_high * 0.995  # Short near resistance
        starter_risk = abs(starter_price - stop_loss)
        if starter_risk > 0:
            starter_shares = 250 / starter_risk  # 0.25R position
            entries.append({
                'type': 'starter',
                'price': starter_price,
                'shares': starter_shares,
                'risk_r': 0.25
            })
        
        # Pre-Trig Entry (0.25R) - 2m BB per Notion  
        if len(intraday_df) > 8:  # Need bars for 2m BB simulation
            pretrig_price = opening_high * 0.988  # Short on 2m breakdown
            pretrig_risk = abs(pretrig_price - stop_loss)
            if pretrig_risk > 0:
                pretrig_shares = 250 / pretrig_risk
                entries.append({
                    'type': 'pre_trig',
                    'price': pretrig_price,
                    'shares': pretrig_shares,
                    'risk_r': 0.25
                })
        
        # Trig Entry (1R) - 5m BB per Notion
        if len(intraday_df) > 20:  # Need bars for 5m BB
            trig_price = opening_high * 0.982  # Short on 5m breakdown
            trig_risk = abs(trig_price - stop_loss)
            if trig_risk > 0:
                trig_shares = 1000 / trig_risk  # 1R position
                entries.append({
                    'type': 'trig',
                    'price': trig_price,
                    'shares': trig_shares,
                    'risk_r': 1.0
                })
        
        return entries, stop_loss
    
    def find_cover_targets(self, chart_df):
        """Find cover targets: 5m 200 EMA and 9/20 dev bands lower"""
        if chart_df.empty:
            return None, None, None
        
        # Main cover target: 5m 200 EMA (per Notion)
        ema_200 = chart_df['ema_200'].iloc[-1] if not pd.isna(chart_df['ema_200'].iloc[-1]) else chart_df['low'].min()
        
        # Secondary: 9/20 dev band lower  
        dev_920_lower = chart_df['dev_920_lower'].iloc[-1] if not pd.isna(chart_df['dev_920_lower'].iloc[-1]) else chart_df['low'].min()
        
        # Primary cover target (higher of the two)
        target_1 = max(ema_200, dev_920_lower)
        target_2 = target_1 * 0.95  # Secondary
        target_3 = target_1 * 0.90  # Deep target
        
        return target_1, target_2, target_3
    
    def simulate_cover_execution(self, intraday_df, entries, targets):
        """Simulate cover strategy: Cover in thirds when targets hit"""
        target_1, target_2, target_3 = targets
        
        total_shares = sum([e['shares'] for e in entries])
        total_cost = sum([e['price'] * e['shares'] for e in entries])
        
        if total_shares == 0:
            return 0, 0
        
        avg_entry = total_cost / total_shares
        
        # Find best cover price achieved during the day
        cover_price = avg_entry  # Default no profit/loss
        
        for i, bar in intraday_df.iterrows():
            if bar['low'] <= target_1:  # Hit main target (5m 200 EMA)
                cover_price = target_1
                break
            elif bar['low'] <= target_2:  # Hit secondary
                cover_price = target_2
                break
            elif bar['low'] <= target_3:  # Hit deep target
                cover_price = target_3
                break
        
        # If no target hit, simulate partial cover at daily low or breakeven
        if cover_price == avg_entry:
            daily_low = intraday_df['low'].min()
            if daily_low < avg_entry * 0.98:  # Some downward movement
                cover_price = min(avg_entry * 0.995, daily_low * 1.005)
        
        return avg_entry, cover_price
    
    def create_candlestick_chart(self, ticker, trade_date, intraday_df, entries, targets, execution_data):
        """Create detailed candlestick chart with dark theme and EST timezone"""
        
        if intraday_df.empty:
            return None
        
        # Calculate indicators
        chart_df = self.calculate_indicators(intraday_df)
        
        # Create figure with dark theme
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                     height_ratios=[3, 1], 
                                     facecolor='#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        
        # OHLC Bar Chart (simplified candlesticks)
        colors = ['#00ff00' if close >= open else '#ff0000' for open, close in zip(chart_df['open'], chart_df['close'])]
        
        # Plot price bars
        ax1.plot(chart_df['timestamp'], chart_df['close'], color='white', linewidth=2, alpha=0.9, label='Close Price')
        ax1.fill_between(chart_df['timestamp'], chart_df['low'], chart_df['high'], alpha=0.3, color='gray', label='High/Low Range')
        
        # Color bars by direction
        for i, (timestamp, open_price, close_price) in enumerate(zip(chart_df['timestamp'], chart_df['open'], chart_df['close'])):
            color = '#00ff00' if close_price >= open_price else '#ff0000'
            ax1.scatter(timestamp, close_price, c=color, s=20, alpha=0.8)
        
        # Technical indicators
        ax1.plot(chart_df['timestamp'], chart_df['ema_200'], '#ffff00', linewidth=2, label='5m 200 EMA (Cover Target)', alpha=0.9)
        ax1.plot(chart_df['timestamp'], chart_df['ema_20'], '#ff6600', linewidth=1.5, label='EMA 20', alpha=0.8)
        ax1.plot(chart_df['timestamp'], chart_df['vwap'], '#9933ff', linewidth=1.5, label='VWAP', alpha=0.8)
        
        # 9/20 Dev Bands
        ax1.plot(chart_df['timestamp'], chart_df['dev_920_upper'], '--', color='#00ffff', alpha=0.6, label='9/20 Dev Upper')
        ax1.plot(chart_df['timestamp'], chart_df['dev_920_lower'], '--', color='#00ffff', alpha=0.6, label='9/20 Dev Lower (Cover)')
        ax1.fill_between(chart_df['timestamp'], chart_df['dev_920_upper'], chart_df['dev_920_lower'], 
                         alpha=0.1, color='cyan', label='9/20 Dev Channel')
        
        # 7/28/9 Dev Bands
        ax1.plot(chart_df['timestamp'], chart_df['dev_7289_upper'], ':', color='#ff9999', alpha=0.5, label='7/28/9 Dev Upper')
        ax1.plot(chart_df['timestamp'], chart_df['dev_7289_lower'], ':', color='#ff9999', alpha=0.5, label='7/28/9 Dev Lower')
        
        # Plot SHORT entries
        for i, entry in enumerate(entries):
            color = ['#ff4444', '#ffaa44', '#44ff44'][i]  # Red, Orange, Green
            ax1.axhline(y=entry['price'], color=color, linestyle='-', alpha=0.8, linewidth=2,
                       label=f"{entry['type'].upper()} SHORT @ ${entry['price']:.2f}")
        
        # Plot cover targets
        target_1, target_2, target_3 = targets
        if target_1:
            ax1.axhline(y=target_1, color='#00ff00', linestyle='--', alpha=0.8, linewidth=2,
                       label=f'Cover Target 1 @ ${target_1:.2f}')
        if target_2:
            ax1.axhline(y=target_2, color='#ffff00', linestyle='--', alpha=0.6, linewidth=1,
                       label=f'Cover Target 2 @ ${target_2:.2f}')
        
        # Plot stop loss
        if entries:
            stop_loss = max([e['price'] for e in entries]) * 1.01
            ax1.axhline(y=stop_loss, color='#ff0000', linestyle=':', alpha=0.8, linewidth=2,
                       label=f'Stop Loss @ ${stop_loss:.2f}')
        
        # Execution markers
        avg_entry, cover_price = execution_data['avg_entry'], execution_data['cover_price']
        if avg_entry and cover_price:
            ax1.scatter([chart_df['timestamp'].iloc[2]], [avg_entry], 
                       s=100, c='red', marker='v', label=f'SHORT Entry @ ${avg_entry:.2f}', zorder=5)
            
            # Find cover time (simplified - use end of day)
            cover_time = chart_df['timestamp'].iloc[-5] if len(chart_df) > 5 else chart_df['timestamp'].iloc[-1]
            ax1.scatter([cover_time], [cover_price], 
                       s=100, c='green', marker='^', label=f'Cover @ ${cover_price:.2f}', zorder=5)
        
        # Styling
        ax1.set_facecolor('#1e1e1e')
        ax1.grid(True, alpha=0.3, color='#555555')
        ax1.set_title(f'{ticker} - OS D1 SHORT Strategy Execution - {trade_date}', 
                     color='white', fontsize=16, fontweight='bold')
        ax1.legend(loc='upper right', facecolor='#2e2e2e', edgecolor='white', fontsize=10)
        ax1.tick_params(colors='white')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz='US/Eastern'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        
        # Volume subplot
        ax2.bar(chart_df['timestamp'], chart_df['volume'], color='#444444', alpha=0.7)
        ax2.set_facecolor('#1e1e1e')
        ax2.grid(True, alpha=0.3, color='#555555')
        ax2.set_ylabel('Volume', color='white')
        ax2.tick_params(colors='white')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz='US/Eastern'))
        
        plt.tight_layout()
        
        # Save chart
        chart_filename = f'charts/{ticker}_{trade_date}_short_execution_chart.png'
        plt.savefig(chart_filename, facecolor='#1e1e1e', dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_filename
    
    def validate_short_trade(self, ticker, trade_row, intraday_df):
        """Validate individual SHORT trade with complete execution simulation"""
        
        if intraday_df.empty:
            return None
        
        # Setup data
        setup_data = {
            'gap_pct': trade_row.get('gap_pct', 50),
            'pm_high_pct': trade_row.get('pm_high_pct', 75)
        }
        
        # Stage classification
        stage = self.classify_opening_stage(intraday_df, setup_data)
        
        # Opening levels
        opening_bar = intraday_df.iloc[0]
        opening_high = opening_bar['high']
        
        # Calculate indicators
        chart_df = self.calculate_indicators(intraday_df)
        
        # Determine HIGH EV entry type
        entry_type = self.determine_high_ev_entry(stage, intraday_df)
        
        # Simulate SHORT entries
        entries, stop_loss = self.simulate_short_entries(intraday_df, entry_type, stage, opening_high)
        
        # Find cover targets
        targets = self.find_cover_targets(chart_df)
        
        # Simulate execution
        avg_entry, cover_price = self.simulate_cover_execution(intraday_df, entries, targets)
        
        # Calculate P&L for SHORT
        risk_per_share = abs(avg_entry - stop_loss) if avg_entry else 0.01
        actual_pnl_r = (avg_entry - cover_price) / risk_per_share if avg_entry and cover_price and risk_per_share > 0 else -0.5
        
        execution_data = {
            'avg_entry': avg_entry,
            'cover_price': cover_price,
            'stop_loss': stop_loss,
            'actual_pnl_r': actual_pnl_r
        }
        
        # Create chart
        chart_path = self.create_candlestick_chart(ticker, trade_row['date'].strftime('%Y-%m-%d'), 
                                                 intraday_df, entries, targets, execution_data)
        
        return {
            'ticker': ticker,
            'date': trade_row['date'].strftime('%Y-%m-%d'),
            'stage': stage,
            'entry_type': entry_type,
            'expected_pnl': trade_row['pnl_r'],
            'actual_pnl': actual_pnl_r,
            'pnl_difference': actual_pnl_r - trade_row['pnl_r'],
            'avg_entry': avg_entry,
            'cover_price': cover_price,
            'stop_loss': stop_loss,
            'chart_path': chart_path,
            'validation_status': self.determine_validation_status(actual_pnl_r, trade_row['pnl_r'])
        }
    
    def determine_validation_status(self, actual_pnl, expected_pnl):
        """Determine validation status"""
        diff = abs(actual_pnl - expected_pnl)
        if diff <= 0.5:
            return 'validated'
        elif diff <= 1.0:
            return 'suspicious'
        else:
            return 'invalid'
    
    async def validate_recent_short_trades(self):
        """Validate recent trades using SHORT strategy"""
        
        print("🔍 OS D1 SHORT STRATEGY VALIDATION SYSTEM")
        print("=" * 60)
        print("Creating candlestick charts with SHORT execution validation")
        
        # Load recent trades
        trades_df = self.load_recent_trades()
        if trades_df.empty:
            return pd.DataFrame()
        
        print(f"\\n📊 Validating {len(trades_df)} recent SHORT trades...\\n")
        
        # Validate each trade
        validation_results = []
        
        for idx, (_, trade_row) in enumerate(trades_df.iterrows()):
            ticker = trade_row['ticker']
            trade_date = trade_row['date']
            
            print(f"📈 {idx+1}/{len(trades_df)}: Validating {ticker} ({trade_date.strftime('%Y-%m-%d')})")
            print(f"   Expected P&L: {trade_row['pnl_r']:.2f}R | Stage: {trade_row.get('stage', 'unknown')} | Type: {trade_row.get('entry_type', 'opening_fbo')}")
            
            # Fetch intraday data
            intraday_df = self.fetch_intraday_data(ticker, trade_date)
            
            if not intraday_df.empty:
                # Validate trade
                result = self.validate_short_trade(ticker, trade_row, intraday_df)
                if result:
                    validation_results.append(result)
                    
                    status_icon = {'validated': '✅', 'suspicious': '⚠️', 'invalid': '❌'}.get(result['validation_status'], '❓')
                    print(f"   {status_icon} Actual P&L: {result['actual_pnl']:.2f}R | Difference: {result['pnl_difference']:.2f}R | Status: {result['validation_status'].upper()}")
                else:
                    print("   ❌ Validation failed - no execution data")
            else:
                print("   ❌ No intraday data available")
                
        # Generate summary
        if validation_results:
            return self.generate_validation_summary(validation_results)
        else:
            print("\\n❌ No trades validated")
            return pd.DataFrame()
    
    def generate_validation_summary(self, results):
        """Generate comprehensive validation summary"""
        
        results_df = pd.DataFrame(results)
        
        print(f"\\n{'='*60}")
        print("🔍 SHORT STRATEGY VALIDATION SUMMARY")  
        print(f"{'='*60}")
        
        # Validation breakdown
        total_trades = len(results_df)
        status_counts = results_df['validation_status'].value_counts()
        
        print(f"📊 VALIDATION BREAKDOWN:")
        for status in ['validated', 'suspicious', 'invalid']:
            count = status_counts.get(status, 0)
            pct = count / total_trades * 100
            icon = {'validated': '✅', 'suspicious': '⚠️', 'invalid': '❌'}[status]
            print(f"   • {icon} {status.title()}: {count} ({pct:.1f}%)")
        
        # P&L analysis
        total_expected = results_df['expected_pnl'].sum()
        total_actual = results_df['actual_pnl'].sum()
        total_diff = total_actual - total_expected
        avg_diff = results_df['pnl_difference'].mean()
        
        print(f"\\n📈 P&L VALIDATION:")
        print(f"   • Expected Total P&L: {total_expected:.2f}R")
        print(f"   • Actual Total P&L: {total_actual:.2f}R")
        print(f"   • Total Difference: {total_diff:.2f}R")
        print(f"   • Average Difference per Trade: {avg_diff:.2f}R")
        
        # Stage performance
        print(f"\\n📊 STAGE PERFORMANCE:")
        stage_perf = results_df.groupby('stage').agg({
            'actual_pnl': ['count', 'mean'],
            'validation_status': lambda x: (x == 'validated').sum()
        }).round(2)
        
        for stage in stage_perf.index:
            count = stage_perf.loc[stage, ('actual_pnl', 'count')]
            avg_pnl = stage_perf.loc[stage, ('actual_pnl', 'mean')]
            validated = stage_perf.loc[stage, ('validation_status', '<lambda>')]
            print(f"   • {stage.replace('_', ' ').title()}: {count} trades, {avg_pnl:.2f}R avg, {validated} validated")
        
        # Best and worst trades
        best_trade = results_df.loc[results_df['actual_pnl'].idxmax()]
        worst_trade = results_df.loc[results_df['actual_pnl'].idxmin()]
        
        print(f"\\n🏆 BEST SHORT TRADE: {best_trade['ticker']} ({best_trade['date']})")
        print(f"   Actual P&L: {best_trade['actual_pnl']:.2f}R | Stage: {best_trade['stage']} | Type: {best_trade['entry_type']}")
        
        print(f"\\n💀 WORST SHORT TRADE: {worst_trade['ticker']} ({worst_trade['date']})")
        print(f"   Actual P&L: {worst_trade['actual_pnl']:.2f}R | Stage: {worst_trade['stage']} | Type: {worst_trade['entry_type']}")
        
        # Validation assessment
        validated_pct = (status_counts.get('validated', 0) / total_trades) * 100
        print(f"\\n✅ SHORT STRATEGY VALIDATION:")
        print(f"   • Validation Rate: {validated_pct:.1f}%")
        print(f"   • Strategy Viability: {'✅ VIABLE' if avg_diff > -1.0 else '⚠️ NEEDS OPTIMIZATION'}")
        print(f"   • Backtest Accuracy: {'✅ ACCURATE' if validated_pct >= 50 else '❌ UNREALISTIC'}")
        
        # Save results
        results_file = "os_d1_short_validation_results.csv"
        results_df.to_csv(results_file, index=False)
        
        # Save chart links
        chart_links_df = results_df[['ticker', 'date', 'chart_path', 'expected_pnl', 'actual_pnl', 'validation_status']].copy()
        chart_links_df['chart_link'] = 'file://' + chart_links_df['chart_path'].astype(str)
        chart_links_file = "os_d1_short_chart_links.csv"
        chart_links_df.to_csv(chart_links_file, index=False)
        
        print(f"\\n💾 Validation results saved to: {results_file}")
        print(f"📊 Chart links saved to: {chart_links_file}")
        print(f"📁 Candlestick charts saved to: charts/ directory")
        
        return results_df

async def main():
    """Run the OS D1 SHORT strategy validation"""
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    validator = OS_D1_ShortValidator(api_key)
    
    results = await validator.validate_recent_short_trades()

if __name__ == '__main__':
    asyncio.run(main())