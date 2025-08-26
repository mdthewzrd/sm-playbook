"""
Backtesting Engine for SM Playbook Trading System

This module provides the core backtesting functionality for evaluating
trading strategies using historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005   # 0.05%
    max_positions: int = 10
    position_size_method: str = 'fixed_risk'  # 'fixed_risk', 'fixed_amount', 'percent'
    risk_per_trade: float = 0.01  # 1%
    benchmark_symbol: str = 'SPY'
    timeframe: str = '1D'


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0
    direction: str = 'long'  # 'long' or 'short'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    commission_paid: float = 0
    slippage_cost: float = 0
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    trade_id: Optional[str] = None
    strategy_name: Optional[str] = None
    
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_date is None
    
    def duration_days(self) -> Optional[int]:
        """Get trade duration in days."""
        if self.exit_date is None:
            return None
        return (self.exit_date - self.entry_date).days


class BacktestEngine:
    """
    Main backtesting engine for strategy evaluation.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.trades = []
        self.portfolio_history = []
        self.current_portfolio = {}
        self.cash = config.initial_capital
        self.total_value = config.initial_capital
        self.daily_returns = []
        self.benchmark_returns = []
        self.current_date = None
        self.trade_counter = 0
        
    def run_backtest(
        self, 
        strategy_func: Callable,
        data: Dict[str, pd.DataFrame],
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run backtest with given strategy and data.
        
        Args:
            strategy_func: Function that generates trading signals
            data: Dictionary of symbol -> OHLCV DataFrame
            benchmark_data: Benchmark price data for comparison
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Validate data
        if not self._validate_data(data):
            raise ValueError("Invalid data provided for backtesting")
        
        # Get date range from data
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        date_range = sorted(list(all_dates))
        
        # Filter date range based on config
        date_range = [
            d for d in date_range 
            if self.config.start_date <= d <= self.config.end_date
        ]
        
        if not date_range:
            raise ValueError("No data available in specified date range")
        
        # Initialize portfolio tracking
        self._initialize_portfolio(date_range[0])
        
        # Run backtest day by day
        for current_date in date_range:
            self.current_date = current_date
            
            # Get current market data
            current_data = {}
            for symbol, df in data.items():
                if current_date in df.index:
                    current_data[symbol] = df.loc[current_date]
            
            if not current_data:
                continue
                
            # Generate signals from strategy
            try:
                signals = strategy_func(current_data, self.current_portfolio, current_date)
                if signals:
                    self._process_signals(signals, current_data)
            except Exception as e:
                logger.error(f"Error generating signals on {current_date}: {e}")
                continue
            
            # Update portfolio
            self._update_portfolio(current_data)
            
            # Record daily performance
            self._record_daily_performance(current_date, benchmark_data)
        
        # Close any remaining open trades
        self._close_open_trades(date_range[-1], data)
        
        # Generate results
        results = self._generate_results()
        
        logger.info(f"Backtest completed. Total trades: {len(self.trades)}")
        return results
    
    def _validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate input data."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for symbol, df in data.items():
            if not isinstance(df, pd.DataFrame):
                logger.error(f"Data for {symbol} is not a DataFrame")
                return False
                
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns for {symbol}: {missing_cols}")
                return False
                
            if df.empty:
                logger.error(f"Empty data for symbol: {symbol}")
                return False
        
        return True
    
    def _initialize_portfolio(self, start_date: datetime):
        """Initialize portfolio tracking."""
        self.current_date = start_date
        self.cash = self.config.initial_capital
        self.total_value = self.config.initial_capital
        self.current_portfolio = {}
        
    def _process_signals(self, signals: List[Dict], current_data: Dict):
        """Process trading signals."""
        for signal in signals:
            try:
                if signal.get('action') == 'buy':
                    self._execute_buy_signal(signal, current_data)
                elif signal.get('action') == 'sell':
                    self._execute_sell_signal(signal, current_data)
                elif signal.get('action') == 'close':
                    self._execute_close_signal(signal, current_data)
            except Exception as e:
                logger.error(f"Error processing signal {signal}: {e}")
    
    def _execute_buy_signal(self, signal: Dict, current_data: Dict):
        """Execute buy signal."""
        symbol = signal['symbol']
        if symbol not in current_data:
            return
        
        # Check if already have position
        if symbol in self.current_portfolio:
            logger.debug(f"Already have position in {symbol}, skipping buy signal")
            return
        
        # Check position limits
        if len(self.current_portfolio) >= self.config.max_positions:
            logger.debug(f"Maximum positions ({self.config.max_positions}) reached")
            return
        
        # Calculate position size
        current_price = current_data[symbol]['close']
        stop_loss = signal.get('stop_loss', current_price * 0.95)
        
        position_size = self._calculate_position_size(
            current_price, stop_loss, signal.get('risk_amount')
        )
        
        if position_size <= 0:
            logger.debug(f"Invalid position size for {symbol}")
            return
        
        # Calculate costs
        trade_value = position_size * current_price
        commission = trade_value * self.config.commission
        slippage = trade_value * self.config.slippage
        total_cost = trade_value + commission + slippage
        
        # Check if we have enough cash
        if total_cost > self.cash:
            logger.debug(f"Insufficient cash for {symbol} trade: need {total_cost}, have {self.cash}")
            return
        
        # Execute trade
        trade = Trade(
            symbol=symbol,
            entry_date=self.current_date,
            entry_price=current_price,
            quantity=position_size,
            direction='long',
            stop_loss=stop_loss,
            take_profit=signal.get('take_profit'),
            commission_paid=commission,
            slippage_cost=slippage,
            trade_id=f"{symbol}_{self.trade_counter}",
            strategy_name=signal.get('strategy_name', 'default')
        )
        
        self.trades.append(trade)
        self.current_portfolio[symbol] = trade
        self.cash -= total_cost
        self.trade_counter += 1
        
        logger.debug(f"Opened long position: {symbol} @ {current_price:.2f}, size: {position_size:.2f}")
    
    def _execute_sell_signal(self, signal: Dict, current_data: Dict):
        """Execute sell signal."""
        symbol = signal['symbol']
        if symbol not in self.current_portfolio:
            logger.debug(f"No position in {symbol} to sell")
            return
        
        self._close_position(symbol, current_data[symbol]['close'], 'signal')
    
    def _execute_close_signal(self, signal: Dict, current_data: Dict):
        """Execute close signal."""
        symbol = signal['symbol']
        if symbol in self.current_portfolio:
            self._close_position(symbol, current_data[symbol]['close'], 'close_signal')
    
    def _close_position(self, symbol: str, exit_price: float, exit_reason: str = 'manual'):
        """Close an existing position."""
        if symbol not in self.current_portfolio:
            return
        
        trade = self.current_portfolio[symbol]
        
        # Calculate P&L
        trade_value = trade.quantity * exit_price
        commission = trade_value * self.config.commission
        slippage = trade_value * self.config.slippage
        
        gross_pnl = (exit_price - trade.entry_price) * trade.quantity
        net_pnl = gross_pnl - commission - slippage - trade.commission_paid - trade.slippage_cost
        
        pnl_percent = (exit_price / trade.entry_price - 1) * 100
        
        # Update trade
        trade.exit_date = self.current_date
        trade.exit_price = exit_price
        trade.pnl = net_pnl
        trade.pnl_percent = pnl_percent
        trade.commission_paid += commission
        trade.slippage_cost += slippage
        
        # Update cash
        self.cash += trade_value - commission - slippage
        
        # Remove from portfolio
        del self.current_portfolio[symbol]
        
        logger.debug(f"Closed position: {symbol} @ {exit_price:.2f}, P&L: {net_pnl:.2f} ({pnl_percent:.2f}%)")
    
    def _calculate_position_size(
        self, 
        entry_price: float, 
        stop_loss: float,
        risk_amount: Optional[float] = None
    ) -> float:
        """Calculate position size based on risk management rules."""
        
        if self.config.position_size_method == 'fixed_risk':
            if stop_loss is None or stop_loss >= entry_price:
                return 0
            
            risk_amount = risk_amount or (self.total_value * self.config.risk_per_trade)
            risk_per_share = entry_price - stop_loss
            position_size = risk_amount / risk_per_share
            
        elif self.config.position_size_method == 'fixed_amount':
            fixed_amount = risk_amount or 10000  # Default $10k per trade
            position_size = fixed_amount / entry_price
            
        elif self.config.position_size_method == 'percent':
            percent = risk_amount or 0.1  # Default 10%
            dollar_amount = self.total_value * percent
            position_size = dollar_amount / entry_price
            
        else:
            raise ValueError(f"Unknown position sizing method: {self.config.position_size_method}")
        
        return max(0, position_size)
    
    def _update_portfolio(self, current_data: Dict):
        """Update portfolio values and check stop losses."""
        # Check stop losses and take profits
        positions_to_close = []
        
        for symbol, trade in self.current_portfolio.items():
            if symbol not in current_data:
                continue
                
            current_price = current_data[symbol]['close']
            
            # Check stop loss
            if trade.stop_loss and current_price <= trade.stop_loss:
                positions_to_close.append((symbol, current_price, 'stop_loss'))
                continue
                
            # Check take profit
            if trade.take_profit and current_price >= trade.take_profit:
                positions_to_close.append((symbol, current_price, 'take_profit'))
                continue
        
        # Close positions that hit stops
        for symbol, exit_price, reason in positions_to_close:
            self._close_position(symbol, exit_price, reason)
        
        # Calculate total portfolio value
        portfolio_value = self.cash
        for symbol, trade in self.current_portfolio.items():
            if symbol in current_data:
                current_price = current_data[symbol]['close']
                position_value = trade.quantity * current_price
                portfolio_value += position_value
        
        self.total_value = portfolio_value
    
    def _record_daily_performance(self, date: datetime, benchmark_data: Optional[pd.DataFrame]):
        """Record daily performance metrics."""
        # Calculate daily return
        if len(self.portfolio_history) > 0:
            prev_value = self.portfolio_history[-1]['total_value']
            daily_return = (self.total_value / prev_value - 1) * 100
        else:
            daily_return = 0
        
        self.daily_returns.append(daily_return)
        
        # Record portfolio snapshot
        portfolio_snapshot = {
            'date': date,
            'total_value': self.total_value,
            'cash': self.cash,
            'positions': len(self.current_portfolio),
            'daily_return': daily_return
        }
        
        self.portfolio_history.append(portfolio_snapshot)
        
        # Calculate benchmark return if available
        if benchmark_data is not None and date in benchmark_data.index:
            if len(self.benchmark_returns) > 0:
                prev_benchmark = benchmark_data.loc[benchmark_data.index[
                    benchmark_data.index.get_loc(date) - 1
                ]]['close']
                current_benchmark = benchmark_data.loc[date]['close']
                benchmark_return = (current_benchmark / prev_benchmark - 1) * 100
            else:
                benchmark_return = 0
            
            self.benchmark_returns.append(benchmark_return)
    
    def _close_open_trades(self, final_date: datetime, data: Dict[str, pd.DataFrame]):
        """Close any remaining open trades at the end of backtest."""
        for symbol in list(self.current_portfolio.keys()):
            if symbol in data and final_date in data[symbol].index:
                final_price = data[symbol].loc[final_date]['close']
                self._close_position(symbol, final_price, 'backtest_end')
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive backtest results."""
        if not self.trades:
            return {'error': 'No trades executed during backtest period'}
        
        # Basic trade statistics
        closed_trades = [t for t in self.trades if not t.is_open()]
        winning_trades = [t for t in closed_trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl and t.pnl < 0]
        
        total_trades = len(closed_trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # P&L calculations
        total_pnl = sum(t.pnl for t in closed_trades if t.pnl)
        gross_profit = sum(t.pnl for t in winning_trades if t.pnl)
        gross_loss = sum(t.pnl for t in losing_trades if t.pnl)
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # Return calculations
        total_return = (self.total_value / self.config.initial_capital - 1) * 100
        
        # Risk metrics
        daily_returns_array = np.array(self.daily_returns)
        volatility = np.std(daily_returns_array) * np.sqrt(252)  # Annualized
        
        # Calculate max drawdown
        portfolio_values = [p['total_value'] for p in self.portfolio_history]
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        mean_return = np.mean(daily_returns_array) * 252
        sharpe_ratio = (mean_return - 2) / volatility if volatility > 0 else 0
        
        # Trade duration analysis
        durations = [t.duration_days() for t in closed_trades if t.duration_days() is not None]
        avg_duration = np.mean(durations) if durations else 0
        
        results = {
            'backtest_config': self.config.__dict__,
            'performance_summary': {
                'total_return': total_return,
                'total_pnl': total_pnl,
                'final_portfolio_value': self.total_value,
                'initial_capital': self.config.initial_capital,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio
            },
            'trade_statistics': {
                'total_trades': total_trades,
                'winning_trades': win_count,
                'losing_trades': loss_count,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'average_duration_days': avg_duration
            },
            'detailed_trades': [
                {
                    'symbol': t.symbol,
                    'entry_date': t.entry_date,
                    'exit_date': t.exit_date,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'quantity': t.quantity,
                    'pnl': t.pnl,
                    'pnl_percent': t.pnl_percent,
                    'duration_days': t.duration_days(),
                    'strategy_name': t.strategy_name
                } for t in closed_trades
            ],
            'portfolio_history': self.portfolio_history,
            'daily_returns': self.daily_returns,
            'benchmark_returns': self.benchmark_returns
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save backtest results to file."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary to CSV
        summary_df = pd.DataFrame([results['performance_summary']])
        summary_df.to_csv(f"{output_path}_summary.csv", index=False)
        
        # Save trades to CSV
        if results['detailed_trades']:
            trades_df = pd.DataFrame(results['detailed_trades'])
            trades_df.to_csv(f"{output_path}_trades.csv", index=False)
        
        # Save portfolio history to CSV
        if results['portfolio_history']:
            portfolio_df = pd.DataFrame(results['portfolio_history'])
            portfolio_df.to_csv(f"{output_path}_portfolio.csv", index=False)
        
        logger.info(f"Backtest results saved to {output_path}")


def create_sample_strategy():
    """Create a sample trading strategy for testing."""
    
    def simple_ma_strategy(current_data: Dict, portfolio: Dict, current_date: datetime) -> List[Dict]:
        """
        Simple moving average crossover strategy.
        
        Args:
            current_data: Dict of symbol -> current OHLCV data
            portfolio: Current portfolio positions
            current_date: Current date
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # This is a placeholder - in real implementation, you would:
        # 1. Calculate technical indicators
        # 2. Apply strategy rules
        # 3. Generate buy/sell signals
        
        return signals
    
    return simple_ma_strategy