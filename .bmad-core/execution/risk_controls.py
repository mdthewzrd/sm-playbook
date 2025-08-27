"""
BRF Strategy Real-Time Risk Control System
Implements comprehensive risk management and position monitoring
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class RiskEvent(Enum):
    POSITION_SIZE_BREACH = "position_size_breach"
    PORTFOLIO_DRAWDOWN = "portfolio_drawdown"
    SINGLE_STOCK_LOSS = "single_stock_loss"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    CORRELATION_RISK = "correlation_risk"
    VOLATILITY_SPIKE = "volatility_spike"
    MARKET_HOURS = "market_hours"

@dataclass
class RiskAlert:
    event_type: RiskEvent
    level: RiskLevel
    symbol: Optional[str]
    message: str
    value: float
    threshold: float
    timestamp: datetime
    action_required: bool = False

class BRFRiskControls:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.risk_limits = {
            'max_position_size_pct': 0.05,      # 5% max per position
            'max_portfolio_drawdown_pct': 0.15,  # 15% max portfolio drawdown
            'max_daily_loss_pct': 0.05,          # 5% max daily loss
            'max_single_stock_loss_pct': 0.03,   # 3% max loss per stock
            'max_correlation_exposure': 0.20,     # 20% max in correlated positions
            'max_open_positions': 10,            # Max 10 concurrent positions
            'min_cash_reserve_pct': 0.10         # 10% minimum cash reserve
        }
        
        self.daily_pnl = 0.0
        self.daily_start_capital = initial_capital
        self.risk_alerts: List[RiskAlert] = []
        self.position_limits: Dict[str, float] = {}
        self.trading_halted = False
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def check_pre_trade_risk(self, symbol: str, order_value: float, 
                           current_positions: Dict, portfolio_value: float) -> Tuple[bool, List[RiskAlert]]:
        """Comprehensive pre-trade risk checks"""
        alerts = []
        can_trade = True
        
        # 1. Position size check
        position_pct = order_value / portfolio_value
        if position_pct > self.risk_limits['max_position_size_pct']:
            alert = RiskAlert(
                event_type=RiskEvent.POSITION_SIZE_BREACH,
                level=RiskLevel.HIGH,
                symbol=symbol,
                message=f"Position size {position_pct:.2%} exceeds limit {self.risk_limits['max_position_size_pct']:.2%}",
                value=position_pct,
                threshold=self.risk_limits['max_position_size_pct'],
                timestamp=datetime.now(),
                action_required=True
            )
            alerts.append(alert)
            can_trade = False
        
        # 2. Cash reserve check
        total_position_value = sum(pos.quantity * pos.current_price for pos in current_positions.values())
        available_cash = portfolio_value - total_position_value
        cash_pct = available_cash / portfolio_value
        
        if cash_pct < self.risk_limits['min_cash_reserve_pct']:
            alert = RiskAlert(
                event_type=RiskEvent.DAILY_LOSS_LIMIT,
                level=RiskLevel.MEDIUM,
                symbol=None,
                message=f"Cash reserve {cash_pct:.2%} below minimum {self.risk_limits['min_cash_reserve_pct']:.2%}",
                value=cash_pct,
                threshold=self.risk_limits['min_cash_reserve_pct'],
                timestamp=datetime.now(),
                action_required=False
            )
            alerts.append(alert)
        
        # 3. Daily loss limit check
        daily_loss_pct = abs(self.daily_pnl) / self.daily_start_capital
        if self.daily_pnl < 0 and daily_loss_pct > self.risk_limits['max_daily_loss_pct']:
            alert = RiskAlert(
                event_type=RiskEvent.DAILY_LOSS_LIMIT,
                level=RiskLevel.CRITICAL,
                symbol=None,
                message=f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.risk_limits['max_daily_loss_pct']:.2%}",
                value=daily_loss_pct,
                threshold=self.risk_limits['max_daily_loss_pct'],
                timestamp=datetime.now(),
                action_required=True
            )
            alerts.append(alert)
            can_trade = False
            self.trading_halted = True
        
        # 4. Maximum positions check
        if len(current_positions) >= self.risk_limits['max_open_positions']:
            alert = RiskAlert(
                event_type=RiskEvent.POSITION_SIZE_BREACH,
                level=RiskLevel.HIGH,
                symbol=None,
                message=f"Maximum positions {len(current_positions)} reached",
                value=len(current_positions),
                threshold=self.risk_limits['max_open_positions'],
                timestamp=datetime.now(),
                action_required=True
            )
            alerts.append(alert)
            can_trade = False
        
        # 5. Market hours check
        if not self.is_market_hours():
            alert = RiskAlert(
                event_type=RiskEvent.MARKET_HOURS,
                level=RiskLevel.HIGH,
                symbol=None,
                message="Trading outside market hours",
                value=0,
                threshold=0,
                timestamp=datetime.now(),
                action_required=True
            )
            alerts.append(alert)
            can_trade = False
        
        self.risk_alerts.extend(alerts)
        return can_trade, alerts
    
    def monitor_position_risk(self, positions: Dict, current_prices: Dict[str, float]) -> List[RiskAlert]:
        """Monitor existing positions for risk violations"""
        alerts = []
        
        for symbol, position in positions.items():
            current_price = current_prices.get(symbol, position.current_price)
            position.current_price = current_price
            
            # Update unrealized P&L
            position.unrealized_pnl = position.quantity * (current_price - position.avg_cost)
            
            # 1. Single stock loss check
            loss_pct = position.unrealized_pnl / (position.quantity * position.avg_cost)
            if loss_pct < -self.risk_limits['max_single_stock_loss_pct']:
                alert = RiskAlert(
                    event_type=RiskEvent.SINGLE_STOCK_LOSS,
                    level=RiskLevel.HIGH,
                    symbol=symbol,
                    message=f"Position loss {abs(loss_pct):.2%} exceeds limit {self.risk_limits['max_single_stock_loss_pct']:.2%}",
                    value=abs(loss_pct),
                    threshold=self.risk_limits['max_single_stock_loss_pct'],
                    timestamp=datetime.now(),
                    action_required=True
                )
                alerts.append(alert)
        
        # 2. Portfolio drawdown check
        total_unrealized = sum(pos.unrealized_pnl for pos in positions.values())
        portfolio_value = self.initial_capital + total_unrealized + self.daily_pnl
        drawdown_pct = (self.initial_capital - portfolio_value) / self.initial_capital
        
        if drawdown_pct > self.risk_limits['max_portfolio_drawdown_pct']:
            alert = RiskAlert(
                event_type=RiskEvent.PORTFOLIO_DRAWDOWN,
                level=RiskLevel.CRITICAL,
                symbol=None,
                message=f"Portfolio drawdown {drawdown_pct:.2%} exceeds limit {self.risk_limits['max_portfolio_drawdown_pct']:.2%}",
                value=drawdown_pct,
                threshold=self.risk_limits['max_portfolio_drawdown_pct'],
                timestamp=datetime.now(),
                action_required=True
            )
            alerts.append(alert)
            self.trading_halted = True
        
        self.risk_alerts.extend(alerts)
        return alerts
    
    def check_correlation_risk(self, positions: Dict, sector_data: Dict[str, str] = None) -> List[RiskAlert]:
        """Check for excessive correlation exposure"""
        alerts = []
        
        if not sector_data:
            return alerts
        
        # Group positions by sector
        sector_exposure = {}
        total_exposure = sum(pos.quantity * pos.current_price for pos in positions.values())
        
        for symbol, position in positions.items():
            sector = sector_data.get(symbol, "Unknown")
            position_value = position.quantity * position.current_price
            sector_exposure[sector] = sector_exposure.get(sector, 0) + position_value
        
        # Check each sector exposure
        for sector, exposure in sector_exposure.items():
            exposure_pct = exposure / total_exposure
            if exposure_pct > self.risk_limits['max_correlation_exposure']:
                alert = RiskAlert(
                    event_type=RiskEvent.CORRELATION_RISK,
                    level=RiskLevel.MEDIUM,
                    symbol=None,
                    message=f"Sector exposure ({sector}) {exposure_pct:.2%} exceeds limit {self.risk_limits['max_correlation_exposure']:.2%}",
                    value=exposure_pct,
                    threshold=self.risk_limits['max_correlation_exposure'],
                    timestamp=datetime.now(),
                    action_required=False
                )
                alerts.append(alert)
        
        self.risk_alerts.extend(alerts)
        return alerts
    
    def emergency_liquidation_check(self, positions: Dict) -> bool:
        """Determine if emergency liquidation is required"""
        critical_alerts = [a for a in self.risk_alerts if a.level == RiskLevel.CRITICAL]
        
        # Emergency liquidation triggers
        triggers = [
            RiskEvent.PORTFOLIO_DRAWDOWN,
            RiskEvent.DAILY_LOSS_LIMIT
        ]
        
        for alert in critical_alerts:
            if alert.event_type in triggers and alert.action_required:
                self.logger.critical(f"EMERGENCY LIQUIDATION TRIGGERED: {alert.message}")
                return True
        
        return False
    
    def get_position_size_limit(self, symbol: str, setup_score: float, 
                              portfolio_value: float, volatility: float = None) -> float:
        """Calculate dynamic position size limit based on risk factors"""
        base_limit = self.risk_limits['max_position_size_pct']
        
        # Adjust based on setup score (70-100 range)
        score_multiplier = min(1.0, (setup_score - 70) / 30 * 0.5 + 0.5)  # 0.5-1.0
        
        # Adjust based on volatility if available
        volatility_multiplier = 1.0
        if volatility:
            # Reduce size for high volatility stocks
            volatility_multiplier = max(0.5, 1.0 - (volatility - 0.02) * 10)  # Reduce for vol > 2%
        
        adjusted_limit = base_limit * score_multiplier * volatility_multiplier
        return min(adjusted_limit, base_limit)  # Never exceed absolute limit
    
    def is_market_hours(self) -> bool:
        """Check if current time is within market hours (9:30 AM - 4:00 PM ET)"""
        now = datetime.now()
        
        # Skip weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET (simplified, no holiday checks)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def is_golden_time_zone(self) -> bool:
        """Check if current time is within Golden Time Zone (8:30-11:30 AM ET)"""
        import pytz
        
        try:
            # Get current Eastern Time
            et = pytz.timezone('US/Eastern')
            current_et = datetime.now(et)
            
            # Skip weekends
            if current_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Golden Time Zone: 8:30 AM - 11:30 AM ET
            golden_start = current_et.replace(hour=8, minute=30, second=0, microsecond=0)
            golden_end = current_et.replace(hour=11, minute=30, second=0, microsecond=0)
            
            return golden_start <= current_et <= golden_end
            
        except Exception as e:
            self.logger.error(f"Error checking Golden Time Zone: {e}")
            return False
    
    def update_daily_pnl(self, realized_pnl: float):
        """Update daily P&L tracking"""
        self.daily_pnl += realized_pnl
        
        # Reset daily tracking at market open
        now = datetime.now()
        if now.hour == 9 and now.minute == 30:
            self.daily_pnl = 0.0
            self.daily_start_capital = self.initial_capital
            self.trading_halted = False
            self.risk_alerts = []  # Clear old alerts
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        recent_alerts = [a for a in self.risk_alerts 
                        if a.timestamp > datetime.now() - timedelta(hours=1)]
        
        alert_counts = {
            'critical': len([a for a in recent_alerts if a.level == RiskLevel.CRITICAL]),
            'high': len([a for a in recent_alerts if a.level == RiskLevel.HIGH]),
            'medium': len([a for a in recent_alerts if a.level == RiskLevel.MEDIUM]),
            'low': len([a for a in recent_alerts if a.level == RiskLevel.LOW])
        }
        
        return {
            'trading_halted': self.trading_halted,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.daily_start_capital,
            'alert_counts': alert_counts,
            'recent_alerts': len(recent_alerts),
            'market_hours': self.is_market_hours(),
            'risk_limits': self.risk_limits
        }
    
    def override_risk_halt(self, reason: str = "Manual override"):
        """Manual override for risk halt (use with extreme caution)"""
        self.trading_halted = False
        self.logger.warning(f"RISK HALT OVERRIDDEN: {reason}")
        
        # Log override for audit trail
        override_alert = RiskAlert(
            event_type=RiskEvent.DAILY_LOSS_LIMIT,  # Generic event type
            level=RiskLevel.CRITICAL,
            symbol=None,
            message=f"Risk halt manually overridden: {reason}",
            value=0,
            threshold=0,
            timestamp=datetime.now(),
            action_required=False
        )
        self.risk_alerts.append(override_alert)