"""
Performance Monitoring System for SM Playbook Trading System

Real-time monitoring of trading performance metrics, P&L tracking,
and strategy performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time."""
    timestamp: datetime
    portfolio_value: float
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    total_return: float
    drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    active_positions: int
    total_trades: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_return': self.total_return,
            'drawdown': self.drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'active_positions': self.active_positions,
            'total_trades': self.total_trades
        }


class PerformanceMonitor:
    """
    Real-time performance monitoring system.
    
    Tracks portfolio performance, calculates key metrics, and provides
    alerts when performance thresholds are breached.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize performance monitor.
        
        Args:
            initial_capital: Starting portfolio value
        """
        self.initial_capital = initial_capital
        self.performance_history = []
        self.daily_snapshots = []
        self.alerts = []
        
        # Performance thresholds
        self.drawdown_alert_threshold = 0.05  # 5%
        self.daily_loss_alert_threshold = 0.02  # 2%
        self.win_rate_alert_threshold = 0.35  # 35%
        
        # State tracking
        self.peak_value = initial_capital
        self.session_start_value = initial_capital
        self.last_snapshot_time = None
        self.is_monitoring = False
        
        # Performance caches
        self._performance_cache = {}
        self._cache_expiry = None
        
        logger.info(f"Performance monitor initialized with capital: ${initial_capital:,.2f}")
    
    async def start_monitoring(self, update_interval: int = 30):
        """
        Start real-time performance monitoring.
        
        Args:
            update_interval: Update interval in seconds
        """
        if self.is_monitoring:
            logger.warning("Performance monitoring already active")
            return
        
        self.is_monitoring = True
        self.session_start_value = self.get_current_portfolio_value()
        
        logger.info(f"Starting performance monitoring (update interval: {update_interval}s)")
        
        try:
            while self.is_monitoring:
                await self._update_performance_snapshot()
                await self._check_alerts()
                await asyncio.sleep(update_interval)
        except Exception as e:
            logger.error(f"Error in performance monitoring loop: {e}")
            self.is_monitoring = False
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        
        # Generate session summary
        if self.performance_history:
            await self._generate_session_summary()
        
        logger.info("Performance monitoring stopped")
    
    async def _update_performance_snapshot(self):
        """Update performance snapshot with current data."""
        try:
            current_time = datetime.now()
            
            # Get current portfolio data
            portfolio_value = self.get_current_portfolio_value()
            positions = self.get_current_positions()
            trades = self.get_completed_trades()
            
            # Calculate metrics
            metrics = await self._calculate_current_metrics(
                portfolio_value, positions, trades
            )
            
            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=current_time,
                portfolio_value=portfolio_value,
                daily_pnl=portfolio_value - self.session_start_value,
                unrealized_pnl=metrics['unrealized_pnl'],
                realized_pnl=metrics['realized_pnl'],
                total_return=((portfolio_value / self.initial_capital) - 1) * 100,
                drawdown=self._calculate_current_drawdown(portfolio_value),
                sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
                win_rate=metrics.get('win_rate', 0.0),
                profit_factor=metrics.get('profit_factor', 0.0),
                active_positions=len(positions),
                total_trades=len(trades)
            )
            
            # Add to history
            self.performance_history.append(snapshot)
            
            # Update daily snapshot if new day
            if self._is_new_trading_day():
                self.daily_snapshots.append(snapshot)
            
            self.last_snapshot_time = current_time
            
            # Clear performance cache
            self._performance_cache.clear()
            
            logger.debug(f"Performance snapshot updated: P&L ${snapshot.daily_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating performance snapshot: {e}")
    
    async def _calculate_current_metrics(
        self,
        portfolio_value: float,
        positions: List[Dict],
        trades: List[Dict]
    ) -> Dict[str, float]:
        """Calculate current performance metrics."""
        metrics = {}
        
        try:
            # Unrealized P&L from positions
            unrealized_pnl = sum(
                pos.get('unrealized_pnl', 0) for pos in positions
            )
            metrics['unrealized_pnl'] = unrealized_pnl
            
            # Realized P&L from completed trades
            realized_pnl = sum(
                trade.get('pnl', 0) for trade in trades
                if trade.get('exit_date') is not None
            )
            metrics['realized_pnl'] = realized_pnl
            
            # Trade statistics
            if trades:
                closed_trades = [t for t in trades if t.get('exit_date')]
                
                if closed_trades:
                    winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
                    losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]
                    
                    metrics['win_rate'] = len(winning_trades) / len(closed_trades)
                    
                    gross_profit = sum(t['pnl'] for t in winning_trades)
                    gross_loss = abs(sum(t['pnl'] for t in losing_trades))
                    
                    metrics['profit_factor'] = (
                        gross_profit / gross_loss if gross_loss > 0 else float('inf')
                    )
            
            # Sharpe ratio (simplified daily calculation)
            if len(self.performance_history) > 30:
                daily_returns = self._calculate_daily_returns()
                if len(daily_returns) > 1:
                    metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(daily_returns)
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
        
        return metrics
    
    def _calculate_current_drawdown(self, current_value: float) -> float:
        """Calculate current drawdown percentage."""
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        if self.peak_value > 0:
            return ((self.peak_value - current_value) / self.peak_value) * 100
        return 0.0
    
    def _calculate_daily_returns(self) -> List[float]:
        """Calculate daily returns from performance history."""
        if len(self.daily_snapshots) < 2:
            return []
        
        returns = []
        for i in range(1, len(self.daily_snapshots)):
            prev_value = self.daily_snapshots[i-1].portfolio_value
            curr_value = self.daily_snapshots[i].portfolio_value
            
            if prev_value > 0:
                daily_return = ((curr_value / prev_value) - 1) * 100
                returns.append(daily_return)
        
        return returns
    
    def _calculate_sharpe_ratio(self, daily_returns: List[float]) -> float:
        """Calculate Sharpe ratio from daily returns."""
        if not daily_returns or len(daily_returns) < 2:
            return 0.0
        
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe (assuming 252 trading days)
        risk_free_rate = 2.0  # 2% annual risk-free rate
        sharpe = ((mean_return * 252) - risk_free_rate) / (std_return * np.sqrt(252))
        
        return sharpe
    
    async def _check_alerts(self):
        """Check for performance alerts."""
        if not self.performance_history:
            return
        
        current_snapshot = self.performance_history[-1]
        
        # Drawdown alert
        if current_snapshot.drawdown > self.drawdown_alert_threshold * 100:
            await self._create_alert(
                'DRAWDOWN_WARNING',
                f'Drawdown exceeded {self.drawdown_alert_threshold*100:.1f}%: {current_snapshot.drawdown:.2f}%',
                'high'
            )
        
        # Daily loss alert
        daily_loss_pct = (current_snapshot.daily_pnl / self.session_start_value) * 100
        if daily_loss_pct < -self.daily_loss_alert_threshold * 100:
            await self._create_alert(
                'DAILY_LOSS_WARNING',
                f'Daily loss exceeded {self.daily_loss_alert_threshold*100:.1f}%: {daily_loss_pct:.2f}%',
                'high'
            )
        
        # Win rate alert
        if (current_snapshot.total_trades > 10 and 
            current_snapshot.win_rate < self.win_rate_alert_threshold):
            await self._create_alert(
                'LOW_WIN_RATE',
                f'Win rate below {self.win_rate_alert_threshold*100:.1f}%: {current_snapshot.win_rate*100:.1f}%',
                'medium'
            )
    
    async def _create_alert(self, alert_type: str, message: str, priority: str):
        """Create a performance alert."""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'priority': priority,
            'acknowledged': False
        }
        
        self.alerts.append(alert)
        
        logger.warning(f"Performance Alert [{priority.upper()}]: {message}")
        
        # Could integrate with external alerting systems here
        # (email, Slack, SMS, etc.)
    
    def _is_new_trading_day(self) -> bool:
        """Check if we've entered a new trading day."""
        if not self.daily_snapshots:
            return True
        
        last_snapshot = self.daily_snapshots[-1]
        current_date = datetime.now().date()
        last_date = last_snapshot.timestamp.date()
        
        return current_date > last_date
    
    def get_current_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        if not self.performance_history:
            return {'error': 'No performance data available'}
        
        current = self.performance_history[-1]
        
        # Calculate some additional metrics
        days_active = (datetime.now() - self.performance_history[0].timestamp).days
        avg_daily_return = current.total_return / max(1, days_active)
        
        return {
            'current_value': current.portfolio_value,
            'initial_capital': self.initial_capital,
            'total_return': current.total_return,
            'daily_pnl': current.daily_pnl,
            'unrealized_pnl': current.unrealized_pnl,
            'realized_pnl': current.realized_pnl,
            'max_drawdown': max(s.drawdown for s in self.performance_history),
            'current_drawdown': current.drawdown,
            'sharpe_ratio': current.sharpe_ratio,
            'win_rate': current.win_rate,
            'profit_factor': current.profit_factor,
            'active_positions': current.active_positions,
            'total_trades': current.total_trades,
            'days_active': days_active,
            'avg_daily_return': avg_daily_return,
            'alerts_count': len([a for a in self.alerts if not a['acknowledged']])
        }
    
    def get_performance_chart_data(self, days: int = 30) -> Dict[str, List]:
        """Get data for performance charts."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_snapshots = [
            s for s in self.performance_history 
            if s.timestamp >= cutoff_date
        ]
        
        if not recent_snapshots:
            return {'timestamps': [], 'values': [], 'pnl': [], 'drawdown': []}
        
        return {
            'timestamps': [s.timestamp.isoformat() for s in recent_snapshots],
            'values': [s.portfolio_value for s in recent_snapshots],
            'pnl': [s.daily_pnl for s in recent_snapshots],
            'drawdown': [s.drawdown for s in recent_snapshots]
        }
    
    def get_alerts(self, acknowledged: Optional[bool] = None) -> List[Dict]:
        """Get performance alerts."""
        if acknowledged is None:
            return self.alerts
        
        return [a for a in self.alerts if a['acknowledged'] == acknowledged]
    
    def acknowledge_alert(self, alert_index: int):
        """Acknowledge a performance alert."""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index]['acknowledged'] = True
            logger.info(f"Alert {alert_index} acknowledged")
    
    def save_performance_data(self, file_path: str):
        """Save performance data to file."""
        try:
            data = {
                'initial_capital': self.initial_capital,
                'performance_history': [s.to_dict() for s in self.performance_history],
                'daily_snapshots': [s.to_dict() for s in self.daily_snapshots],
                'alerts': self.alerts,
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'monitoring_active': self.is_monitoring,
                    'peak_value': self.peak_value
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Performance data saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def load_performance_data(self, file_path: str):
        """Load performance data from file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.initial_capital = data['initial_capital']
            self.peak_value = data['metadata']['peak_value']
            
            # Reconstruct performance history
            self.performance_history = []
            for snapshot_data in data['performance_history']:
                snapshot_data['timestamp'] = datetime.fromisoformat(snapshot_data['timestamp'])
                snapshot = PerformanceSnapshot(**snapshot_data)
                self.performance_history.append(snapshot)
            
            # Reconstruct daily snapshots
            self.daily_snapshots = []
            for snapshot_data in data['daily_snapshots']:
                snapshot_data['timestamp'] = datetime.fromisoformat(snapshot_data['timestamp'])
                snapshot = PerformanceSnapshot(**snapshot_data)
                self.daily_snapshots.append(snapshot)
            
            self.alerts = data['alerts']
            
            logger.info(f"Performance data loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
    
    async def _generate_session_summary(self):
        """Generate session summary when monitoring stops."""
        if not self.performance_history:
            return
        
        first = self.performance_history[0]
        last = self.performance_history[-1]
        
        session_duration = last.timestamp - first.timestamp
        
        summary = f"""
        ======= PERFORMANCE SESSION SUMMARY =======
        Session Duration: {session_duration}
        Initial Value: ${self.initial_capital:,.2f}
        Final Value: ${last.portfolio_value:,.2f}
        Total Return: {last.total_return:.2f}%
        Daily P&L: ${last.daily_pnl:.2f}
        Max Drawdown: {max(s.drawdown for s in self.performance_history):.2f}%
        Win Rate: {last.win_rate*100:.1f}%
        Total Trades: {last.total_trades}
        Alerts Generated: {len(self.alerts)}
        ==========================================
        """
        
        logger.info(summary)
    
    # Placeholder methods for integration with actual trading system
    def get_current_portfolio_value(self) -> float:
        """Get current portfolio value - override in implementation."""
        # This would integrate with actual position manager
        return self.initial_capital + sum(
            s.daily_pnl for s in self.performance_history[-1:] if self.performance_history
        )
    
    def get_current_positions(self) -> List[Dict]:
        """Get current positions - override in implementation."""
        # This would integrate with actual position manager
        return []
    
    def get_completed_trades(self) -> List[Dict]:
        """Get completed trades - override in implementation."""
        # This would integrate with actual trade history
        return []


async def main():
    """Example usage of performance monitor."""
    monitor = PerformanceMonitor(initial_capital=100000.0)
    
    # Start monitoring
    await monitor.start_monitoring(update_interval=10)
    
    # Monitor would run until stopped
    # await monitor.stop_monitoring()


if __name__ == '__main__':
    asyncio.run(main())