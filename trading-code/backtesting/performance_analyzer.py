"""
Performance Analysis Module for SM Playbook Trading System

This module provides comprehensive performance analysis and metrics
calculation for backtesting results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    # Return metrics
    total_return: float
    annualized_return: float
    monthly_returns: List[float]
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    downside_deviation: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: Optional[float] = None
    
    # Trade statistics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    largest_win: float
    largest_loss: float
    
    # Advanced metrics
    beta: Optional[float] = None
    alpha: Optional[float] = None
    tracking_error: Optional[float] = None
    var_95: float = 0
    cvar_95: float = 0


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        
    def analyze_backtest_results(
        self, 
        backtest_results: Dict[str, Any],
        benchmark_returns: Optional[List[float]] = None
    ) -> PerformanceMetrics:
        """
        Analyze backtest results and calculate performance metrics.
        
        Args:
            backtest_results: Results from BacktestEngine
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        logger.info("Starting performance analysis")
        
        # Extract data from backtest results
        daily_returns = np.array(backtest_results.get('daily_returns', []))
        portfolio_history = backtest_results.get('portfolio_history', [])
        trades = backtest_results.get('detailed_trades', [])
        
        if len(daily_returns) == 0:
            raise ValueError("No daily returns data available for analysis")
        
        # Calculate basic return metrics
        total_return = self._calculate_total_return(portfolio_history)
        annualized_return = self._calculate_annualized_return(daily_returns)
        monthly_returns = self._calculate_monthly_returns(portfolio_history)
        
        # Calculate risk metrics
        volatility = self._calculate_volatility(daily_returns)
        max_drawdown, max_dd_duration = self._calculate_max_drawdown(portfolio_history)
        downside_deviation = self._calculate_downside_deviation(daily_returns)
        
        # Calculate risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns, volatility)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns, downside_deviation)
        calmar_ratio = self._calculate_calmar_ratio(annualized_return, max_drawdown)
        
        # Calculate trade statistics
        trade_stats = self._calculate_trade_statistics(trades)
        
        # Calculate VaR and CVaR
        var_95 = self._calculate_var(daily_returns, 0.95)
        cvar_95 = self._calculate_cvar(daily_returns, 0.95)
        
        # Calculate benchmark-relative metrics if benchmark provided
        beta = None
        alpha = None
        information_ratio = None
        tracking_error = None
        
        if benchmark_returns and len(benchmark_returns) == len(daily_returns):
            benchmark_array = np.array(benchmark_returns)
            beta = self._calculate_beta(daily_returns, benchmark_array)
            alpha = self._calculate_alpha(daily_returns, benchmark_array, beta)
            information_ratio = self._calculate_information_ratio(daily_returns, benchmark_array)
            tracking_error = self._calculate_tracking_error(daily_returns, benchmark_array)
        
        metrics = PerformanceMetrics(
            # Return metrics
            total_return=total_return,
            annualized_return=annualized_return,
            monthly_returns=monthly_returns,
            
            # Risk metrics
            volatility=volatility,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            downside_deviation=downside_deviation,
            
            # Risk-adjusted metrics
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            
            # Trade statistics
            total_trades=trade_stats['total_trades'],
            win_rate=trade_stats['win_rate'],
            profit_factor=trade_stats['profit_factor'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            avg_trade=trade_stats['avg_trade'],
            largest_win=trade_stats['largest_win'],
            largest_loss=trade_stats['largest_loss'],
            
            # Advanced metrics
            beta=beta,
            alpha=alpha,
            tracking_error=tracking_error,
            var_95=var_95,
            cvar_95=cvar_95
        )
        
        logger.info("Performance analysis completed")
        return metrics
    
    def _calculate_total_return(self, portfolio_history: List[Dict]) -> float:
        """Calculate total return from portfolio history."""
        if len(portfolio_history) < 2:
            return 0
        
        initial_value = portfolio_history[0]['total_value']
        final_value = portfolio_history[-1]['total_value']
        return (final_value / initial_value - 1) * 100
    
    def _calculate_annualized_return(self, daily_returns: np.ndarray) -> float:
        """Calculate annualized return."""
        if len(daily_returns) == 0:
            return 0
        
        daily_return_fraction = daily_returns / 100
        cumulative_return = np.prod(1 + daily_return_fraction)
        trading_days = len(daily_returns)
        years = trading_days / 252
        
        if years <= 0:
            return 0
        
        annualized = (cumulative_return ** (1 / years) - 1) * 100
        return annualized
    
    def _calculate_monthly_returns(self, portfolio_history: List[Dict]) -> List[float]:
        """Calculate monthly returns."""
        if len(portfolio_history) < 2:
            return []
        
        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Resample to monthly
        monthly = df['total_value'].resample('M').last()
        monthly_returns = monthly.pct_change().dropna() * 100
        
        return monthly_returns.tolist()
    
    def _calculate_volatility(self, daily_returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        if len(daily_returns) == 0:
            return 0
        return np.std(daily_returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, portfolio_history: List[Dict]) -> Tuple[float, int]:
        """Calculate maximum drawdown and its duration."""
        if len(portfolio_history) < 2:
            return 0, 0
        
        values = [p['total_value'] for p in portfolio_history]
        peak = values[0]
        max_dd = 0
        max_dd_duration = 0
        current_dd_duration = 0
        
        for value in values:
            if value > peak:
                peak = value
                current_dd_duration = 0
            else:
                drawdown = (peak - value) / peak * 100
                max_dd = max(max_dd, drawdown)
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
        
        return max_dd, max_dd_duration
    
    def _calculate_downside_deviation(self, daily_returns: np.ndarray) -> float:
        """Calculate downside deviation."""
        if len(daily_returns) == 0:
            return 0
        
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) == 0:
            return 0
        
        return np.std(negative_returns) * np.sqrt(252)
    
    def _calculate_sharpe_ratio(self, daily_returns: np.ndarray, volatility: float) -> float:
        """Calculate Sharpe ratio."""
        if volatility == 0 or len(daily_returns) == 0:
            return 0
        
        mean_return = np.mean(daily_returns) * 252
        return (mean_return - self.risk_free_rate * 100) / volatility
    
    def _calculate_sortino_ratio(self, daily_returns: np.ndarray, downside_deviation: float) -> float:
        """Calculate Sortino ratio."""
        if downside_deviation == 0 or len(daily_returns) == 0:
            return 0
        
        mean_return = np.mean(daily_returns) * 252
        return (mean_return - self.risk_free_rate * 100) / downside_deviation
    
    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0
        return annualized_return / max_drawdown
    
    def _calculate_trade_statistics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive trade statistics."""
        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'avg_win': 0, 'avg_loss': 0, 'avg_trade': 0,
                'largest_win': 0, 'largest_loss': 0
            }
        
        # Filter completed trades with P&L data
        completed_trades = [t for t in trades if t.get('pnl') is not None]
        
        if not completed_trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'avg_win': 0, 'avg_loss': 0, 'avg_trade': 0,
                'largest_win': 0, 'largest_loss': 0
            }
        
        pnls = [t['pnl'] for t in completed_trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        total_trades = len(completed_trades)
        win_count = len(winning_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        avg_trade = np.mean(pnls)
        
        largest_win = max(pnls) if pnls else 0
        largest_loss = min(pnls) if pnls else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
    
    def _calculate_var(self, daily_returns: np.ndarray, confidence: float) -> float:
        """Calculate Value at Risk."""
        if len(daily_returns) == 0:
            return 0
        return np.percentile(daily_returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, daily_returns: np.ndarray, confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(daily_returns) == 0:
            return 0
        
        var = self._calculate_var(daily_returns, confidence)
        tail_returns = daily_returns[daily_returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return np.mean(tail_returns)
    
    def _calculate_beta(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate beta relative to benchmark."""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0
        
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return 0
        
        return covariance / benchmark_variance
    
    def _calculate_alpha(
        self, 
        returns: np.ndarray, 
        benchmark_returns: np.ndarray, 
        beta: float
    ) -> float:
        """Calculate alpha (excess return)."""
        if len(returns) != len(benchmark_returns):
            return 0
        
        strategy_return = np.mean(returns) * 252
        benchmark_return = np.mean(benchmark_returns) * 252
        
        alpha = strategy_return - (self.risk_free_rate * 100 + beta * (benchmark_return - self.risk_free_rate * 100))
        return alpha
    
    def _calculate_information_ratio(
        self, 
        returns: np.ndarray, 
        benchmark_returns: np.ndarray
    ) -> float:
        """Calculate information ratio."""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0
        
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(252)
        
        if tracking_error == 0:
            return 0
        
        return np.mean(active_returns) * 252 / tracking_error
    
    def _calculate_tracking_error(
        self, 
        returns: np.ndarray, 
        benchmark_returns: np.ndarray
    ) -> float:
        """Calculate tracking error."""
        if len(returns) != len(benchmark_returns):
            return 0
        
        active_returns = returns - benchmark_returns
        return np.std(active_returns) * np.sqrt(252)
    
    def generate_performance_report(
        self, 
        metrics: PerformanceMetrics, 
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            metrics: PerformanceMetrics object
            output_path: Optional path to save report
            
        Returns:
            Formatted performance report as string
        """
        report = []
        report.append("="*60)
        report.append("           PERFORMANCE ANALYSIS REPORT")
        report.append("="*60)
        report.append("")
        
        # Return Metrics
        report.append("RETURN METRICS:")
        report.append("-" * 40)
        report.append(f"Total Return:              {metrics.total_return:>10.2f}%")
        report.append(f"Annualized Return:         {metrics.annualized_return:>10.2f}%")
        report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS:")
        report.append("-" * 40)
        report.append(f"Volatility (Annualized):   {metrics.volatility:>10.2f}%")
        report.append(f"Maximum Drawdown:          {metrics.max_drawdown:>10.2f}%")
        report.append(f"Max DD Duration (days):    {metrics.max_drawdown_duration:>10}")
        report.append(f"Downside Deviation:        {metrics.downside_deviation:>10.2f}%")
        report.append(f"Value at Risk (95%):       {metrics.var_95:>10.2f}%")
        report.append(f"Conditional VaR (95%):     {metrics.cvar_95:>10.2f}%")
        report.append("")
        
        # Risk-Adjusted Metrics
        report.append("RISK-ADJUSTED METRICS:")
        report.append("-" * 40)
        report.append(f"Sharpe Ratio:              {metrics.sharpe_ratio:>10.2f}")
        report.append(f"Sortino Ratio:             {metrics.sortino_ratio:>10.2f}")
        report.append(f"Calmar Ratio:              {metrics.calmar_ratio:>10.2f}")
        
        if metrics.information_ratio is not None:
            report.append(f"Information Ratio:         {metrics.information_ratio:>10.2f}")
        
        report.append("")
        
        # Trade Statistics
        report.append("TRADE STATISTICS:")
        report.append("-" * 40)
        report.append(f"Total Trades:              {metrics.total_trades:>10}")
        report.append(f"Win Rate:                  {metrics.win_rate*100:>10.2f}%")
        report.append(f"Profit Factor:             {metrics.profit_factor:>10.2f}")
        report.append(f"Average Win:               ${metrics.avg_win:>10.2f}")
        report.append(f"Average Loss:              ${metrics.avg_loss:>10.2f}")
        report.append(f"Average Trade:             ${metrics.avg_trade:>10.2f}")
        report.append(f"Largest Win:               ${metrics.largest_win:>10.2f}")
        report.append(f"Largest Loss:              ${metrics.largest_loss:>10.2f}")
        report.append("")
        
        # Benchmark Comparison (if available)
        if metrics.beta is not None:
            report.append("BENCHMARK COMPARISON:")
            report.append("-" * 40)
            report.append(f"Beta:                      {metrics.beta:>10.2f}")
            report.append(f"Alpha:                     {metrics.alpha:>10.2f}%")
            report.append(f"Tracking Error:            {metrics.tracking_error:>10.2f}%")
            report.append("")
        
        report.append("="*60)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Performance report saved to {output_path}")
        
        return report_text
    
    def create_performance_charts(
        self, 
        backtest_results: Dict[str, Any], 
        output_dir: str = "charts"
    ):
        """
        Create visualization charts for performance analysis.
        
        Args:
            backtest_results: Results from BacktestEngine
            output_dir: Directory to save charts
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            portfolio_history = backtest_results.get('portfolio_history', [])
            daily_returns = backtest_results.get('daily_returns', [])
            trades = backtest_results.get('detailed_trades', [])
            
            if not portfolio_history:
                logger.warning("No portfolio history available for charting")
                return
            
            # Create output directory
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Chart 1: Portfolio value over time
            self._create_portfolio_value_chart(portfolio_history, f"{output_dir}/portfolio_value.png")
            
            # Chart 2: Daily returns distribution
            if daily_returns:
                self._create_returns_distribution_chart(daily_returns, f"{output_dir}/returns_distribution.png")
            
            # Chart 3: Drawdown chart
            self._create_drawdown_chart(portfolio_history, f"{output_dir}/drawdown.png")
            
            # Chart 4: Monthly returns heatmap
            self._create_monthly_returns_heatmap(portfolio_history, f"{output_dir}/monthly_returns.png")
            
            # Chart 5: Trade analysis
            if trades:
                self._create_trade_analysis_chart(trades, f"{output_dir}/trade_analysis.png")
            
            logger.info(f"Performance charts created in {output_dir}")
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available. Skipping chart creation.")
        except Exception as e:
            logger.error(f"Error creating charts: {e}")
    
    def _create_portfolio_value_chart(self, portfolio_history: List[Dict], output_path: str):
        """Create portfolio value over time chart."""
        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['total_value'], linewidth=2, color='blue')
        plt.title('Portfolio Value Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_returns_distribution_chart(self, daily_returns: List[float], output_path: str):
        """Create daily returns distribution chart."""
        plt.figure(figsize=(12, 8))
        
        # Histogram
        plt.subplot(2, 2, 1)
        plt.hist(daily_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Daily Returns Distribution')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        plt.subplot(2, 2, 2)
        stats.probplot(daily_returns, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normal Distribution)')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 3)
        plt.boxplot(daily_returns, vert=True)
        plt.title('Daily Returns Box Plot')
        plt.ylabel('Daily Return (%)')
        plt.grid(True, alpha=0.3)
        
        # Time series
        plt.subplot(2, 2, 4)
        plt.plot(daily_returns, linewidth=1, color='red')
        plt.title('Daily Returns Time Series')
        plt.xlabel('Trading Day')
        plt.ylabel('Daily Return (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_drawdown_chart(self, portfolio_history: List[Dict], output_path: str):
        """Create drawdown chart."""
        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate drawdown
        peak = df['total_value'].expanding().max()
        drawdown = (df['total_value'] - peak) / peak * 100
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(df['date'], drawdown, 0, color='red', alpha=0.3)
        plt.plot(df['date'], drawdown, color='red', linewidth=1)
        plt.title('Portfolio Drawdown Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_monthly_returns_heatmap(self, portfolio_history: List[Dict], output_path: str):
        """Create monthly returns heatmap."""
        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Calculate monthly returns
        monthly_values = df['total_value'].resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna() * 100
        
        if len(monthly_returns) < 2:
            logger.warning("Insufficient data for monthly returns heatmap")
            return
        
        # Create pivot table for heatmap
        monthly_returns_df = monthly_returns.to_frame('returns')
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        
        pivot = monthly_returns_df.pivot(index='year', columns='month', values='returns')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Monthly Return (%)'})
        plt.title('Monthly Returns Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Year', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_trade_analysis_chart(self, trades: List[Dict], output_path: str):
        """Create trade analysis charts."""
        df = pd.DataFrame(trades)
        
        # Filter completed trades with P&L
        completed_trades = df[df['pnl'].notna()]
        
        if completed_trades.empty:
            logger.warning("No completed trades available for trade analysis chart")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Trade P&L over time
        plt.subplot(2, 3, 1)
        completed_trades['entry_date'] = pd.to_datetime(completed_trades['entry_date'])
        plt.scatter(completed_trades['entry_date'], completed_trades['pnl'], 
                   c=completed_trades['pnl'], cmap='RdYlGn', alpha=0.7)
        plt.title('Trade P&L Over Time')
        plt.xlabel('Entry Date')
        plt.ylabel('P&L ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # P&L distribution
        plt.subplot(2, 3, 2)
        plt.hist(completed_trades['pnl'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        plt.title('Trade P&L Distribution')
        plt.xlabel('P&L ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Win/Loss ratio
        plt.subplot(2, 3, 3)
        wins = len(completed_trades[completed_trades['pnl'] > 0])
        losses = len(completed_trades[completed_trades['pnl'] < 0])
        plt.pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%', colors=['green', 'red'])
        plt.title('Win/Loss Ratio')
        
        # Trade duration
        if 'duration_days' in completed_trades.columns:
            plt.subplot(2, 3, 4)
            duration_data = completed_trades['duration_days'].dropna()
            if not duration_data.empty:
                plt.hist(duration_data, bins=20, alpha=0.7, color='orange', edgecolor='black')
                plt.title('Trade Duration Distribution')
                plt.xlabel('Duration (Days)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
        
        # Cumulative P&L
        plt.subplot(2, 3, 5)
        cumulative_pnl = completed_trades.sort_values('entry_date')['pnl'].cumsum()
        plt.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='blue')
        plt.title('Cumulative P&L by Trade')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative P&L ($)')
        plt.grid(True, alpha=0.3)
        
        # P&L by symbol (if available)
        plt.subplot(2, 3, 6)
        if 'symbol' in completed_trades.columns:
            symbol_pnl = completed_trades.groupby('symbol')['pnl'].sum().sort_values(ascending=False)
            if len(symbol_pnl) <= 20:  # Only show if reasonable number of symbols
                symbol_pnl.plot(kind='bar', color='skyblue')
                plt.title('P&L by Symbol')
                plt.xlabel('Symbol')
                plt.ylabel('Total P&L ($)')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()