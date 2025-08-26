"""
Risk Management Module for SM Playbook Trading System

This module provides comprehensive risk management functionality
including position sizing, risk limits, and portfolio risk monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    # Position limits
    max_position_size: float = 50000.0  # Maximum single position value
    max_positions: int = 10  # Maximum number of positions
    position_risk_limit: float = 0.02  # 2% risk per position
    
    # Portfolio limits
    max_portfolio_value: float = 1000000.0
    max_portfolio_risk: float = 0.08  # 8% total portfolio risk
    max_sector_concentration: float = 0.25  # 25% max in single sector
    max_correlation_exposure: float = 0.30  # 30% max in correlated positions
    
    # Daily limits
    max_daily_loss: float = 10000.0
    max_daily_trades: int = 20
    max_daily_volume: float = 200000.0  # Total dollar volume
    
    # Drawdown limits
    max_drawdown: float = 0.15  # 15% max drawdown
    drawdown_alert_level: float = 0.08  # Alert at 8% drawdown
    
    # Volatility limits
    max_portfolio_volatility: float = 0.25  # 25% annualized volatility
    volatility_lookback_days: int = 30
    
    # Time-based limits
    position_hold_limit_days: int = 30  # Max days to hold position
    intraday_risk_limit: float = 0.03  # 3% intraday risk limit


@dataclass
class RiskMetrics:
    """Container for calculated risk metrics."""
    portfolio_value: float
    portfolio_risk: float
    daily_pnl: float
    unrealized_pnl: float
    max_drawdown: float
    current_drawdown: float
    portfolio_volatility: float
    var_95: float  # Value at Risk 95%
    risk_level: RiskLevel
    risk_score: float  # 0-100 risk score
    violations: List[str]  # List of risk limit violations


class RiskManager:
    """
    Comprehensive risk management system for trading operations.
    """
    
    def __init__(self, limits: RiskLimits):
        """
        Initialize risk manager.
        
        Args:
            limits: Risk limit configuration
        """
        self.limits = limits
        self.risk_history = []
        self.violation_history = []
        self.position_history = {}
        
        # Risk monitoring state
        self.daily_trade_count = 0
        self.daily_volume = 0.0
        self.session_start_value = None
        self.peak_portfolio_value = 0.0
        self.last_risk_check = None
        
        logger.info("Risk manager initialized with limits")
    
    async def pre_trade_check(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        portfolio_value: float = None,
        current_positions: List[Dict] = None
    ) -> bool:
        """
        Perform pre-trade risk checks.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Entry price
            side: 'long' or 'short'
            portfolio_value: Current portfolio value
            current_positions: List of current positions
            
        Returns:
            True if trade passes all risk checks, False otherwise
        """
        try:
            violations = []
            
            # Calculate position value
            position_value = abs(quantity * price)
            
            # Check maximum position size
            if position_value > self.limits.max_position_size:
                violations.append(f"Position size ${position_value:.2f} exceeds limit ${self.limits.max_position_size:.2f}")
            
            # Check maximum positions count
            if current_positions and len(current_positions) >= self.limits.max_positions:
                violations.append(f"Position count {len(current_positions)} at maximum {self.limits.max_positions}")
            
            # Check daily trade limit
            if self.daily_trade_count >= self.limits.max_daily_trades:
                violations.append(f"Daily trade limit {self.limits.max_daily_trades} reached")
            
            # Check daily volume limit
            if self.daily_volume + position_value > self.limits.max_daily_volume:
                violations.append(f"Daily volume limit ${self.limits.max_daily_volume:.2f} would be exceeded")
            
            # Portfolio-level checks if portfolio data available
            if portfolio_value and current_positions:
                # Check portfolio risk
                estimated_risk = self._calculate_portfolio_risk_with_new_position(
                    current_positions, symbol, quantity, price
                )
                
                if estimated_risk > self.limits.max_portfolio_risk:
                    violations.append(f"Portfolio risk {estimated_risk:.2%} exceeds limit {self.limits.max_portfolio_risk:.2%}")
                
                # Check portfolio value limit
                if portfolio_value > self.limits.max_portfolio_value:
                    violations.append(f"Portfolio value ${portfolio_value:.2f} exceeds limit ${self.limits.max_portfolio_value:.2f}")
            
            # Log violations
            if violations:
                logger.warning(f"Pre-trade risk check failed for {symbol}: {violations}")
                self.violation_history.extend(violations)
                return False
            
            # Update daily counters
            self.daily_trade_count += 1
            self.daily_volume += position_value
            
            logger.debug(f"Pre-trade risk check passed for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error in pre-trade risk check: {e}")
            return False
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        portfolio_value: float,
        risk_method: str = 'fixed_risk'
    ) -> float:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            portfolio_value: Current portfolio value
            risk_method: Position sizing method
            
        Returns:
            Position size (quantity of shares/units)
        """
        try:
            if risk_method == 'fixed_risk':
                # Risk-based position sizing
                if not stop_loss or stop_loss <= 0 or stop_loss == entry_price:
                    logger.warning(f"Invalid stop loss for {symbol}, using default sizing")
                    return self.limits.max_position_size / entry_price
                
                risk_amount = portfolio_value * self.limits.position_risk_limit
                risk_per_unit = abs(entry_price - stop_loss)
                position_size = risk_amount / risk_per_unit
                
                # Cap by maximum position value
                max_quantity = self.limits.max_position_size / entry_price
                position_size = min(position_size, max_quantity)
                
            elif risk_method == 'fixed_amount':
                # Fixed dollar amount
                position_size = self.limits.max_position_size / entry_price
                
            elif risk_method == 'volatility_adjusted':
                # Adjust based on historical volatility
                # This would require historical price data
                base_size = self.limits.max_position_size / entry_price
                # Placeholder - would calculate actual volatility adjustment
                volatility_factor = 1.0
                position_size = base_size * volatility_factor
                
            else:
                raise ValueError(f"Unknown risk method: {risk_method}")
            
            # Ensure positive size
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        method: str = 'atr',
        atr_value: Optional[float] = None,
        risk_percentage: float = 0.02
    ) -> float:
        """
        Calculate appropriate stop loss level.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            side: 'long' or 'short'
            method: Stop loss calculation method
            atr_value: ATR value for ATR-based stops
            risk_percentage: Risk percentage for percentage-based stops
            
        Returns:
            Stop loss price
        """
        try:
            if method == 'atr' and atr_value:
                # ATR-based stop loss
                multiplier = 2.0  # 2x ATR
                if side == 'long':
                    stop_loss = entry_price - (atr_value * multiplier)
                else:  # short
                    stop_loss = entry_price + (atr_value * multiplier)
                    
            elif method == 'percentage':
                # Percentage-based stop loss
                if side == 'long':
                    stop_loss = entry_price * (1 - risk_percentage)
                else:  # short
                    stop_loss = entry_price * (1 + risk_percentage)
                    
            elif method == 'fixed_amount':
                # Fixed dollar amount
                fixed_amount = entry_price * 0.05  # 5% default
                if side == 'long':
                    stop_loss = entry_price - fixed_amount
                else:  # short
                    stop_loss = entry_price + fixed_amount
                    
            else:
                # Default percentage stop
                if side == 'long':
                    stop_loss = entry_price * 0.95  # 5% stop
                else:  # short
                    stop_loss = entry_price * 1.05  # 5% stop
            
            # Ensure stop loss is reasonable
            if side == 'long' and stop_loss >= entry_price:
                stop_loss = entry_price * 0.95
            elif side == 'short' and stop_loss <= entry_price:
                stop_loss = entry_price * 1.05
            
            return max(0, stop_loss)
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            if side == 'long':
                return entry_price * 0.95
            else:
                return entry_price * 1.05
    
    def assess_portfolio_risk(
        self,
        positions: List[Dict],
        portfolio_value: float,
        price_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> RiskMetrics:
        """
        Assess comprehensive portfolio risk metrics.
        
        Args:
            positions: List of current positions
            portfolio_value: Current portfolio value
            price_data: Historical price data for volatility calculations
            
        Returns:
            RiskMetrics object with comprehensive risk assessment
        """
        try:
            violations = []
            
            # Calculate basic metrics
            total_exposure = sum(abs(p.get('quantity', 0) * p.get('current_price', 0)) for p in positions)
            unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)
            
            # Calculate daily P&L if session start value available
            daily_pnl = 0.0
            if self.session_start_value:
                daily_pnl = portfolio_value - self.session_start_value
            
            # Track peak value and calculate drawdown
            if portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = portfolio_value
            
            current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            
            # Calculate portfolio risk (sum of individual position risks)
            portfolio_risk = 0.0
            for position in positions:
                pos_value = abs(position.get('quantity', 0) * position.get('current_price', 0))
                pos_risk = pos_value / portfolio_value if portfolio_value > 0 else 0
                portfolio_risk += pos_risk
            
            # Calculate portfolio volatility if price data available
            portfolio_volatility = 0.0
            if price_data and len(positions) > 0:
                portfolio_volatility = self._calculate_portfolio_volatility(positions, price_data)
            
            # Calculate VaR (simplified)
            var_95 = self._calculate_portfolio_var(positions, portfolio_value)
            
            # Check violations
            if portfolio_risk > self.limits.max_portfolio_risk:
                violations.append(f"Portfolio risk {portfolio_risk:.2%} exceeds limit")
            
            if current_drawdown > self.limits.max_drawdown:
                violations.append(f"Drawdown {current_drawdown:.2%} exceeds limit")
            
            if daily_pnl < -self.limits.max_daily_loss:
                violations.append(f"Daily loss ${-daily_pnl:.2f} exceeds limit")
            
            if len(positions) > self.limits.max_positions:
                violations.append(f"Position count {len(positions)} exceeds limit")
            
            # Calculate risk level and score
            risk_level, risk_score = self._calculate_risk_level(
                portfolio_risk, current_drawdown, daily_pnl, portfolio_volatility
            )
            
            metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                portfolio_risk=portfolio_risk,
                daily_pnl=daily_pnl,
                unrealized_pnl=unrealized_pnl,
                max_drawdown=self.limits.max_drawdown,
                current_drawdown=current_drawdown,
                portfolio_volatility=portfolio_volatility,
                var_95=var_95,
                risk_level=risk_level,
                risk_score=risk_score,
                violations=violations
            )
            
            # Store in history
            self.risk_history.append(metrics)
            if violations:
                self.violation_history.extend(violations)
            
            self.last_risk_check = datetime.now()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return RiskMetrics(
                portfolio_value=portfolio_value,
                portfolio_risk=0.0,
                daily_pnl=0.0,
                unrealized_pnl=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                portfolio_volatility=0.0,
                var_95=0.0,
                risk_level=RiskLevel.MEDIUM,
                risk_score=50.0,
                violations=['Error in risk calculation']
            )
    
    def _calculate_portfolio_risk_with_new_position(
        self,
        current_positions: List[Dict],
        new_symbol: str,
        new_quantity: float,
        new_price: float
    ) -> float:
        """Calculate portfolio risk including a new position."""
        # Simplified calculation - in practice would consider correlations
        current_risk = sum(
            abs(p.get('quantity', 0) * p.get('current_price', 0)) 
            for p in current_positions
        )
        new_position_value = abs(new_quantity * new_price)
        
        total_value = current_risk + new_position_value  # Simplified
        return (current_risk + new_position_value) / total_value if total_value > 0 else 0
    
    def _calculate_portfolio_volatility(
        self,
        positions: List[Dict],
        price_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate portfolio volatility based on historical data."""
        try:
            # Simplified volatility calculation
            # In practice, would calculate weighted portfolio volatility
            # considering correlations between positions
            
            if not positions:
                return 0.0
            
            total_weight = 0.0
            weighted_volatility = 0.0
            
            for position in positions:
                symbol = position.get('symbol')
                if symbol not in price_data:
                    continue
                
                # Calculate position weight
                pos_value = abs(position.get('quantity', 0) * position.get('current_price', 0))
                
                # Calculate individual asset volatility
                prices = price_data[symbol]['close']
                returns = prices.pct_change().dropna()
                
                if len(returns) < 30:
                    continue
                
                volatility = returns.std() * np.sqrt(252)  # Annualized
                weight = pos_value
                
                weighted_volatility += weight * volatility
                total_weight += weight
            
            return weighted_volatility / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.0
    
    def _calculate_portfolio_var(self, positions: List[Dict], portfolio_value: float) -> float:
        """Calculate portfolio Value at Risk (simplified)."""
        try:
            # Simplified VaR calculation
            # In practice, would use historical simulation or Monte Carlo
            
            if not positions or portfolio_value <= 0:
                return 0.0
            
            # Assume normal distribution with estimated volatility
            daily_vol = 0.02  # 2% daily volatility assumption
            z_score_95 = -1.645  # 95% confidence level
            
            var_95 = portfolio_value * daily_vol * z_score_95
            return var_95
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def _calculate_risk_level(
        self,
        portfolio_risk: float,
        drawdown: float,
        daily_pnl: float,
        volatility: float
    ) -> Tuple[RiskLevel, float]:
        """Calculate overall risk level and score."""
        try:
            risk_score = 0.0
            
            # Portfolio risk component (0-30 points)
            risk_ratio = portfolio_risk / self.limits.max_portfolio_risk
            risk_score += min(30, risk_ratio * 30)
            
            # Drawdown component (0-25 points)
            dd_ratio = drawdown / self.limits.max_drawdown
            risk_score += min(25, dd_ratio * 25)
            
            # Daily P&L component (0-25 points)
            if daily_pnl < 0:
                loss_ratio = abs(daily_pnl) / self.limits.max_daily_loss
                risk_score += min(25, loss_ratio * 25)
            
            # Volatility component (0-20 points)
            vol_ratio = volatility / self.limits.max_portfolio_volatility
            risk_score += min(20, vol_ratio * 20)
            
            # Determine risk level
            if risk_score < 25:
                risk_level = RiskLevel.LOW
            elif risk_score < 50:
                risk_level = RiskLevel.MEDIUM
            elif risk_score < 75:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            return risk_level, risk_score
            
        except Exception as e:
            logger.error(f"Error calculating risk level: {e}")
            return RiskLevel.MEDIUM, 50.0
    
    def reset_daily_counters(self):
        """Reset daily risk counters (call at start of trading day)."""
        self.daily_trade_count = 0
        self.daily_volume = 0.0
        logger.info("Daily risk counters reset")
    
    def set_session_start_value(self, value: float):
        """Set the session start portfolio value."""
        self.session_start_value = value
        if value > self.peak_portfolio_value:
            self.peak_portfolio_value = value
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk management summary."""
        return {
            'limits': {
                'max_position_size': self.limits.max_position_size,
                'max_positions': self.limits.max_positions,
                'max_portfolio_risk': self.limits.max_portfolio_risk,
                'max_daily_loss': self.limits.max_daily_loss,
                'max_drawdown': self.limits.max_drawdown
            },
            'daily_counters': {
                'trade_count': self.daily_trade_count,
                'volume': self.daily_volume,
                'trades_remaining': max(0, self.limits.max_daily_trades - self.daily_trade_count),
                'volume_remaining': max(0, self.limits.max_daily_volume - self.daily_volume)
            },
            'session_tracking': {
                'session_start_value': self.session_start_value,
                'peak_portfolio_value': self.peak_portfolio_value,
                'last_risk_check': self.last_risk_check
            },
            'violation_count': len(self.violation_history),
            'recent_violations': self.violation_history[-10:] if self.violation_history else []
        }


def create_conservative_limits() -> RiskLimits:
    """Create conservative risk limits for safe trading."""
    return RiskLimits(
        max_position_size=25000.0,
        max_positions=5,
        position_risk_limit=0.01,  # 1% per position
        max_portfolio_risk=0.05,   # 5% total risk
        max_daily_loss=2500.0,
        max_drawdown=0.10          # 10% max drawdown
    )


def create_aggressive_limits() -> RiskLimits:
    """Create aggressive risk limits for higher risk tolerance."""
    return RiskLimits(
        max_position_size=100000.0,
        max_positions=15,
        position_risk_limit=0.025,  # 2.5% per position
        max_portfolio_risk=0.12,    # 12% total risk
        max_daily_loss=15000.0,
        max_drawdown=0.20           # 20% max drawdown
    )