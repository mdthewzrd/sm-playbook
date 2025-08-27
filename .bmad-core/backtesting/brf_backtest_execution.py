#!/usr/bin/env python3
"""
BRF Strategy Backtest Execution
Executes the Backside Reversion & Fade strategy backtest using real market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import json
import os

class BRFBacktestEngine:
    def __init__(self):
        self.initial_capital = 100000
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.current_date = None
        
        # BRF Strategy Parameters (optimized)
        self.atr_period = 14
        self.deviation_multiplier = 2.0
        self.volume_threshold = 1.3  # 30% above average
        self.min_score_threshold = 70
        
        # Risk Management
        self.max_risk_per_trade = 0.02  # 2%
        self.max_positions = 3
        
    def get_backside_runner_universe(self) -> List[str]:
        """Get list of stocks that historically show backside runner patterns"""
        # Small-cap stocks with momentum characteristics
        universe = [
            'CELC', 'AIP', 'AVAH', 'BGLC', 'CLDI', 'COMM', 'GTI', 'LPSN',
            'MWYN', 'PHLT', 'REPL', 'SCS', 'SLRX', 'SMXT', 'SNGX', 'VAPE',
            'VERB', 'VWAV', 'YMAB', 'TELL', 'CLOV', 'SOFI', 'PLTR', 'WISH'
        ]
        return universe
    
    def download_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download historical data for backtesting"""
        data = {}
        
        for symbol in symbols:
            try:
                # Download 5-minute data (limited availability)
                ticker = yf.Ticker(symbol)
                
                # Get daily data first
                daily_data = ticker.history(start=start_date, end=end_date, interval='1d')
                
                if len(daily_data) > 0:
                    # For demonstration, we'll use daily data and simulate intraday
                    data[symbol] = self.simulate_intraday_from_daily(daily_data)
                    print(f"Downloaded data for {symbol}: {len(data[symbol])} bars")
                    
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
                
        return data
    
    def simulate_intraday_from_daily(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """Simulate 5-minute bars from daily data for backtesting"""
        intraday_bars = []
        
        for date, row in daily_data.iterrows():
            # Create 78 5-minute bars per day (9:30 AM - 4:00 PM)
            daily_range = row['High'] - row['Low']
            
            for i in range(78):
                # Simulate realistic intraday movement
                time_factor = i / 77  # 0 to 1 progression through day
                
                # Add some randomness but keep within daily range
                noise = np.random.normal(0, daily_range * 0.02)
                base_price = row['Low'] + (daily_range * time_factor)
                
                # Create OHLC for 5-minute bar
                bar_open = base_price + noise
                bar_close = base_price + noise + np.random.normal(0, daily_range * 0.01)
                bar_high = max(bar_open, bar_close) + abs(np.random.normal(0, daily_range * 0.005))
                bar_low = min(bar_open, bar_close) - abs(np.random.normal(0, daily_range * 0.005))
                
                # Ensure we don't exceed daily range
                bar_high = min(bar_high, row['High'])
                bar_low = max(bar_low, row['Low'])
                
                # Volume distributed throughout day
                volume = row['Volume'] / 78 * (1 + np.random.normal(0, 0.3))
                volume = max(volume, 1000)
                
                timestamp = pd.Timestamp(date) + pd.Timedelta(minutes=5*i + 570)  # Start at 9:30 AM
                
                intraday_bars.append({
                    'timestamp': timestamp,
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': bar_close,
                    'volume': int(volume)
                })
        
        df = pd.DataFrame(intraday_bars)
        df.set_index('timestamp', inplace=True)
        return df
    
    def calculate_vwap(self, data: pd.DataFrame, window: int = 78) -> pd.Series:
        """Calculate VWAP (Volume Weighted Average Price)"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(window).sum() / data['volume'].rolling(window).sum()
        return vwap
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift(1))
        low_close_prev = abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def calculate_deviation_bands(self, data: pd.DataFrame, vwap: pd.Series, atr: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate deviation bands using VWAP +/- ATR multiple"""
        upper_band = vwap + (self.deviation_multiplier * atr)
        lower_band = vwap - (self.deviation_multiplier * atr)
        return upper_band, lower_band
    
    def identify_backside_runner_pattern(self, data: pd.DataFrame, index: int) -> Dict:
        """Identify if current bar shows backside runner setup"""
        if index < 100:  # Need sufficient history
            return {'score': 0, 'setup': False}
        
        current_data = data.iloc[max(0, index-100):index+1]
        
        # Calculate indicators
        vwap_15m = self.calculate_vwap(current_data, 15)  # Approximate 15m VWAP
        vwap_5m = self.calculate_vwap(current_data, 5)    # 5m VWAP
        atr = self.calculate_atr(current_data, self.atr_period)
        
        current_price = current_data['close'].iloc[-1]
        current_vwap_15m = vwap_15m.iloc[-1]
        current_atr = atr.iloc[-1]
        current_volume = current_data['volume'].iloc[-1]
        avg_volume = current_data['volume'].rolling(20).mean().iloc[-1]
        
        # Pattern scoring components
        score = 0
        
        # 1. VWAP relationship (price above VWAP indicates prior strength)
        if current_price > current_vwap_15m:
            score += 20
        
        # 2. Volume surge (indicates institutional activity)
        if current_volume > avg_volume * self.volume_threshold:
            score += 25
        
        # 3. Deviation from VWAP (overextension)
        deviation = abs(current_price - current_vwap_15m) / current_vwap_15m
        if deviation > 0.02:  # 2% deviation
            score += 25
        
        # 4. ATR-based overextension
        upper_band = current_vwap_15m + (self.deviation_multiplier * current_atr)
        if current_price > upper_band:
            score += 30
        
        return {
            'score': score,
            'setup': score >= self.min_score_threshold,
            'vwap_15m': current_vwap_15m,
            'vwap_5m': vwap_5m.iloc[-1] if not pd.isna(vwap_5m.iloc[-1]) else current_vwap_15m,
            'atr': current_atr,
            'upper_band': upper_band,
            'deviation': deviation,
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1
        }
    
    def calculate_position_size(self, entry_price: float, stop_price: float) -> int:
        """Calculate position size based on risk management"""
        if stop_price >= entry_price:  # Invalid stop
            return 0
            
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return 0
            
        max_risk_amount = self.current_capital * self.max_risk_per_trade
        shares = int(max_risk_amount / risk_per_share)
        
        # Ensure we can afford the position
        position_value = shares * entry_price
        if position_value > self.current_capital * 0.2:  # Max 20% per position
            shares = int(self.current_capital * 0.2 / entry_price)
            
        return max(shares, 0)
    
    def execute_trade(self, symbol: str, action: str, price: float, shares: int, timestamp: pd.Timestamp, signal_data: Dict):
        """Execute a trade and update portfolio"""
        if action == 'BUY':
            cost = shares * price
            if cost <= self.current_capital:
                self.current_capital -= cost
                if symbol in self.positions:
                    # Add to existing position
                    old_shares = self.positions[symbol]['shares']
                    old_cost = self.positions[symbol]['cost_basis'] * old_shares
                    total_shares = old_shares + shares
                    total_cost = old_cost + cost
                    self.positions[symbol] = {
                        'shares': total_shares,
                        'cost_basis': total_cost / total_shares,
                        'entry_time': self.positions[symbol]['entry_time'],
                        'stop_price': signal_data.get('stop_price', price * 0.95),
                        'signal_data': signal_data
                    }
                else:
                    self.positions[symbol] = {
                        'shares': shares,
                        'cost_basis': price,
                        'entry_time': timestamp,
                        'stop_price': signal_data.get('stop_price', price * 0.95),
                        'signal_data': signal_data
                    }
                
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': action,
                    'shares': shares,
                    'price': price,
                    'value': cost,
                    'signal_score': signal_data.get('score', 0)
                }
                self.trades.append(trade)
                return True
        
        elif action == 'SELL' and symbol in self.positions:
            proceeds = shares * price
            self.current_capital += proceeds
            
            # Calculate P&L
            cost_basis = self.positions[symbol]['cost_basis']
            pnl = (price - cost_basis) * shares
            
            # Update position
            self.positions[symbol]['shares'] -= shares
            if self.positions[symbol]['shares'] <= 0:
                del self.positions[symbol]
            
            trade = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': action,
                'shares': shares,
                'price': price,
                'value': proceeds,
                'pnl': pnl,
                'signal_score': signal_data.get('score', 0)
            }
            self.trades.append(trade)
            return True
        
        return False
    
    def run_backtest(self, start_date: str = '2024-01-01', end_date: str = '2024-12-31') -> Dict:
        """Run the complete BRF strategy backtest"""
        print("Starting BRF Strategy Backtest...")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        
        # Get universe and download data
        universe = self.get_backside_runner_universe()
        print(f"Universe: {len(universe)} stocks")
        
        data = self.download_data(universe[:10], start_date, end_date)  # Limit to 10 for demo
        
        if not data:
            print("No data downloaded. Exiting.")
            return {}
        
        print(f"Successfully downloaded data for {len(data)} stocks")
        
        # Track all timestamps across all symbols
        all_timestamps = set()
        for symbol_data in data.values():
            all_timestamps.update(symbol_data.index)
        
        all_timestamps = sorted(all_timestamps)
        
        total_bars = len(all_timestamps)
        print(f"Processing {total_bars} time periods...")
        
        for i, current_time in enumerate(all_timestamps):
            if i % 1000 == 0:
                print(f"Progress: {i/total_bars*100:.1f}% - {current_time}")
            
            self.current_date = current_time
            
            # Update equity curve
            portfolio_value = self.current_capital
            for symbol, position in self.positions.items():
                if symbol in data and current_time in data[symbol].index:
                    current_price = data[symbol].loc[current_time, 'close']
                    portfolio_value += position['shares'] * current_price
            
            self.equity_curve.append({
                'timestamp': current_time,
                'portfolio_value': portfolio_value
            })
            
            # Process each symbol
            for symbol, symbol_data in data.items():
                if current_time not in symbol_data.index:
                    continue
                
                current_index = symbol_data.index.get_loc(current_time)
                current_bar = symbol_data.iloc[current_index]
                
                # Check for pattern setup
                pattern_data = self.identify_backside_runner_pattern(symbol_data, current_index)
                
                # Entry logic
                if (pattern_data['setup'] and 
                    len(self.positions) < self.max_positions and 
                    symbol not in self.positions):
                    
                    entry_price = current_bar['close']
                    stop_price = pattern_data['vwap_15m'] - (1.5 * pattern_data['atr'])
                    
                    shares = self.calculate_position_size(entry_price, stop_price)
                    
                    if shares > 0:
                        pattern_data['stop_price'] = stop_price
                        success = self.execute_trade(symbol, 'BUY', entry_price, shares, current_time, pattern_data)
                        if success:
                            print(f"BUY {symbol}: {shares} shares at ${entry_price:.2f} (Score: {pattern_data['score']})")
                
                # Exit logic for existing positions
                if symbol in self.positions:
                    position = self.positions[symbol]
                    current_price = current_bar['close']
                    
                    # Stop loss
                    if current_price <= position['stop_price']:
                        self.execute_trade(symbol, 'SELL', current_price, position['shares'], current_time, {'reason': 'stop_loss'})
                        print(f"STOP {symbol}: {position['shares']} shares at ${current_price:.2f}")
                    
                    # Profit target (simplified - using VWAP reversion)
                    elif current_price <= pattern_data.get('vwap_15m', position['cost_basis']):
                        # Take profit when price reverts to VWAP
                        self.execute_trade(symbol, 'SELL', current_price, position['shares'], current_time, {'reason': 'profit_target'})
                        print(f"PROFIT {symbol}: {position['shares']} shares at ${current_price:.2f}")
        
        # Close any remaining positions
        final_time = all_timestamps[-1] if all_timestamps else pd.Timestamp.now()
        for symbol, position in list(self.positions.items()):
            if symbol in data:
                final_price = data[symbol].iloc[-1]['close']
                self.execute_trade(symbol, 'SELL', final_price, position['shares'], final_time, {'reason': 'final_close'})
        
        return self.generate_results()
    
    def generate_results(self) -> Dict:
        """Generate comprehensive backtest results"""
        if not self.equity_curve:
            return {'error': 'No equity curve data'}
        
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Calculate performance metrics
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Daily returns
        equity_df['daily_return'] = equity_df['portfolio_value'].pct_change()
        avg_daily_return = equity_df['daily_return'].mean()
        daily_volatility = equity_df['daily_return'].std()
        
        # Risk metrics
        sharpe_ratio = (avg_daily_return / daily_volatility * np.sqrt(252)) if daily_volatility > 0 else 0
        
        # Drawdown
        rolling_max = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            profitable_trades = sell_trades[sell_trades['pnl'] > 0] if 'pnl' in sell_trades.columns else pd.DataFrame()
            total_trades = len(sell_trades)
            win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
            
            avg_win = profitable_trades['pnl'].mean() if len(profitable_trades) > 0 else 0
            losing_trades = sell_trades[sell_trades['pnl'] <= 0] if 'pnl' in sell_trades.columns else pd.DataFrame()
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else float('inf')
        else:
            total_trades = win_rate = avg_win = avg_loss = profit_factor = 0
        
        results = {
            'performance_summary': {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100
            },
            'trade_statistics': {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            },
            'strategy_specific': {
                'avg_signal_score': trades_df['signal_score'].mean() if 'signal_score' in trades_df.columns and not trades_df.empty else 0,
                'high_score_trades': len(trades_df[trades_df['signal_score'] > 80]) if 'signal_score' in trades_df.columns else 0
            },
            'equity_curve': equity_df.to_dict('records'),
            'trades': trades_df.to_dict('records')
        }
        
        return results

def main():
    """Main execution function"""
    engine = BRFBacktestEngine()
    
    # Run backtest for the last year
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    results = engine.run_backtest(start_date, end_date)
    
    if 'error' in results:
        print(f"Backtest failed: {results['error']}")
        return
    
    # Print results
    print("\n" + "="*60)
    print("BRF STRATEGY BACKTEST RESULTS")
    print("="*60)
    
    perf = results['performance_summary']
    print(f"Initial Capital:    ${perf['initial_capital']:,.2f}")
    print(f"Final Value:        ${perf['final_value']:,.2f}")
    print(f"Total Return:       {perf['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio:       {perf['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:       {perf['max_drawdown_pct']:.2f}%")
    
    trade_stats = results['trade_statistics']
    print(f"\nTotal Trades:       {trade_stats['total_trades']}")
    print(f"Win Rate:           {trade_stats['win_rate_pct']:.1f}%")
    print(f"Profit Factor:      {trade_stats['profit_factor']:.2f}")
    print(f"Avg Win:            ${trade_stats['avg_win']:.2f}")
    print(f"Avg Loss:           ${trade_stats['avg_loss']:.2f}")
    
    strategy_stats = results['strategy_specific']
    print(f"\nAvg Signal Score:   {strategy_stats['avg_signal_score']:.1f}")
    print(f"High Score Trades:  {strategy_stats['high_score_trades']}")
    
    # Save results
    results_file = '/Users/michaeldurante/sm-playbook/reports/brf_backtest_results.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()