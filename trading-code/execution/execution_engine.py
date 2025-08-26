"""
Execution Engine for SM Playbook Trading System

This module provides the main execution engine that coordinates
trade execution, risk management, and position monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
from pathlib import Path

from .risk_manager import RiskManager, RiskLimits
from .position_manager import PositionManager
from .order_manager import OrderManager, Order, OrderType, OrderStatus
from .signal_processor import SignalProcessor, Signal

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for the trading system."""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


@dataclass
class ExecutionConfig:
    """Configuration for execution engine."""
    mode: ExecutionMode = ExecutionMode.PAPER
    max_positions: int = 10
    default_position_size: float = 10000.0  # Default position size in dollars
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    max_order_value: float = 100000.0
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    order_timeout_minutes: int = 60
    market_hours_only: bool = True
    allowed_symbols: Optional[List[str]] = None
    
    # Risk management
    max_daily_loss: float = 5000.0
    max_portfolio_risk: float = 0.08  # 8%
    position_risk_limit: float = 0.02  # 2% per position
    
    # Execution timing
    min_time_between_trades: int = 60  # seconds
    market_open_delay: int = 300  # 5 minutes after market open
    market_close_buffer: int = 900  # 15 minutes before market close


class ExecutionEngine:
    """
    Main execution engine for the SM Playbook trading system.
    
    Coordinates signal processing, risk management, order execution,
    and position monitoring.
    """
    
    def __init__(self, config: ExecutionConfig):
        """
        Initialize execution engine.
        
        Args:
            config: Execution configuration
        """
        self.config = config
        self.is_running = False
        self.last_trade_time = None
        
        # Initialize components
        self.risk_manager = RiskManager(self._create_risk_limits())
        self.position_manager = PositionManager()
        self.order_manager = OrderManager()
        self.signal_processor = SignalProcessor()
        
        # State tracking
        self.daily_pnl = 0.0
        self.session_start_time = None
        self.processed_signals = []
        self.execution_stats = {
            'signals_processed': 0,
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_rejected': 0,
            'total_commission': 0.0,
            'total_slippage': 0.0
        }
        
        # Event hooks
        self.on_signal_received: Optional[Callable] = None
        self.on_order_filled: Optional[Callable] = None
        self.on_position_opened: Optional[Callable] = None
        self.on_position_closed: Optional[Callable] = None
        self.on_risk_breach: Optional[Callable] = None
        
        logger.info(f"Execution engine initialized in {config.mode.value} mode")
    
    def _create_risk_limits(self) -> RiskLimits:
        """Create risk limits from configuration."""
        return RiskLimits(
            max_position_size=self.config.default_position_size * 2,
            max_portfolio_value=self.config.max_order_value * 10,
            max_daily_loss=self.config.max_daily_loss,
            max_portfolio_risk=self.config.max_portfolio_risk,
            position_risk_limit=self.config.position_risk_limit,
            max_positions=self.config.max_positions
        )
    
    async def start(self):
        """Start the execution engine."""
        if self.is_running:
            logger.warning("Execution engine is already running")
            return
        
        logger.info("Starting execution engine...")
        self.is_running = True
        self.session_start_time = datetime.now()
        self.daily_pnl = 0.0
        
        # Reset daily counters
        self.execution_stats.update({
            'signals_processed': 0,
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_rejected': 0
        })
        
        # Start monitoring tasks
        await asyncio.gather(
            self._monitor_positions(),
            self._monitor_orders(),
            self._monitor_risk()
        )
    
    async def stop(self):
        """Stop the execution engine."""
        if not self.is_running:
            logger.warning("Execution engine is not running")
            return
        
        logger.info("Stopping execution engine...")
        self.is_running = False
        
        # Cancel any pending orders
        await self.order_manager.cancel_all_orders()
        
        # Generate session summary
        self._generate_session_summary()
    
    async def process_signal(self, signal: Signal) -> bool:
        """
        Process a trading signal.
        
        Args:
            signal: Trading signal to process
            
        Returns:
            True if signal was processed successfully, False otherwise
        """
        try:
            logger.debug(f"Processing signal: {signal}")
            
            if not self.is_running:
                logger.warning("Cannot process signal - engine not running")
                return False
            
            # Call signal received hook
            if self.on_signal_received:
                await self.on_signal_received(signal)
            
            # Validate signal
            if not await self._validate_signal(signal):
                logger.warning(f"Signal validation failed: {signal}")
                return False
            
            # Process signal based on action
            success = False
            if signal.action == 'buy' or signal.action == 'long':
                success = await self._process_buy_signal(signal)
            elif signal.action == 'sell' or signal.action == 'short':
                success = await self._process_sell_signal(signal)
            elif signal.action == 'close':
                success = await self._process_close_signal(signal)
            else:
                logger.error(f"Unknown signal action: {signal.action}")
                return False
            
            if success:
                self.processed_signals.append(signal)
                self.execution_stats['signals_processed'] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing signal {signal}: {e}")
            return False
    
    async def _validate_signal(self, signal: Signal) -> bool:
        """Validate a trading signal."""
        # Check if symbol is allowed
        if (self.config.allowed_symbols and 
            signal.symbol not in self.config.allowed_symbols):
            logger.warning(f"Symbol {signal.symbol} not in allowed symbols")
            return False
        
        # Check market hours if required
        if self.config.market_hours_only:
            if not self._is_market_hours():
                logger.warning("Signal received outside market hours")
                return False
        
        # Check minimum time between trades
        if (self.last_trade_time and 
            (datetime.now() - self.last_trade_time).total_seconds() < 
            self.config.min_time_between_trades):
            logger.warning("Signal too soon after last trade")
            return False
        
        # Validate signal data
        if not signal.price or signal.price <= 0:
            logger.error(f"Invalid price in signal: {signal.price}")
            return False
        
        return True
    
    async def _process_buy_signal(self, signal: Signal) -> bool:
        """Process a buy/long signal."""
        try:
            # Check if we already have a position
            current_position = self.position_manager.get_position(signal.symbol)
            if current_position and current_position.quantity > 0:
                logger.debug(f"Already have long position in {signal.symbol}")
                return False
            
            # Calculate position size
            position_size = await self._calculate_position_size(signal)
            if position_size <= 0:
                logger.warning(f"Invalid position size calculated: {position_size}")
                return False
            
            # Pre-trade risk check
            if not await self.risk_manager.pre_trade_check(
                signal.symbol, position_size, signal.price, 'long'
            ):
                logger.warning(f"Pre-trade risk check failed for {signal.symbol}")
                return False
            
            # Create and submit order
            order = Order(
                symbol=signal.symbol,
                order_type=OrderType.MARKET,
                side='buy',
                quantity=position_size,
                price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy_name=signal.strategy_name,
                signal_id=signal.signal_id
            )
            
            success = await self.order_manager.submit_order(order)
            if success:
                self.execution_stats['orders_submitted'] += 1
                self.last_trade_time = datetime.now()
                logger.info(f"Buy order submitted: {order}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing buy signal: {e}")
            return False
    
    async def _process_sell_signal(self, signal: Signal) -> bool:
        """Process a sell/short signal."""
        try:
            # Check if we have a long position to close
            current_position = self.position_manager.get_position(signal.symbol)
            if current_position and current_position.quantity > 0:
                # Close long position
                return await self._close_position(signal.symbol, signal.price, 'sell_signal')
            
            # For short selling (if enabled)
            if not self.config.mode == ExecutionMode.LIVE:  # Only allow in live mode for now
                logger.warning("Short selling not enabled in current mode")
                return False
            
            # Calculate short position size
            position_size = await self._calculate_position_size(signal)
            if position_size <= 0:
                return False
            
            # Pre-trade risk check for short
            if not await self.risk_manager.pre_trade_check(
                signal.symbol, position_size, signal.price, 'short'
            ):
                logger.warning(f"Pre-trade risk check failed for short {signal.symbol}")
                return False
            
            # Create and submit short order
            order = Order(
                symbol=signal.symbol,
                order_type=OrderType.MARKET,
                side='sell_short',
                quantity=position_size,
                price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy_name=signal.strategy_name,
                signal_id=signal.signal_id
            )
            
            success = await self.order_manager.submit_order(order)
            if success:
                self.execution_stats['orders_submitted'] += 1
                self.last_trade_time = datetime.now()
                logger.info(f"Short order submitted: {order}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing sell signal: {e}")
            return False
    
    async def _process_close_signal(self, signal: Signal) -> bool:
        """Process a close position signal."""
        try:
            return await self._close_position(signal.symbol, signal.price, 'close_signal')
        except Exception as e:
            logger.error(f"Error processing close signal: {e}")
            return False
    
    async def _close_position(self, symbol: str, price: float, reason: str = 'manual') -> bool:
        """Close a position."""
        try:
            position = self.position_manager.get_position(symbol)
            if not position:
                logger.warning(f"No position to close for {symbol}")
                return False
            
            # Create close order
            order = Order(
                symbol=symbol,
                order_type=OrderType.MARKET,
                side='sell' if position.quantity > 0 else 'buy_to_cover',
                quantity=abs(position.quantity),
                price=price,
                reason=reason
            )
            
            success = await self.order_manager.submit_order(order)
            if success:
                self.execution_stats['orders_submitted'] += 1
                logger.info(f"Close order submitted: {order} (reason: {reason})")
            
            return success
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    async def _calculate_position_size(self, signal: Signal) -> float:
        """Calculate appropriate position size based on risk management."""
        try:
            # Get current portfolio value
            portfolio_value = self.position_manager.get_portfolio_value()
            
            # Use signal-specific position size if provided
            if signal.position_size:
                return signal.position_size
            
            # Risk-based position sizing
            if signal.stop_loss and signal.stop_loss > 0:
                risk_amount = portfolio_value * self.config.position_risk_limit
                risk_per_share = abs(signal.price - signal.stop_loss)
                if risk_per_share > 0:
                    position_size = risk_amount / risk_per_share
                    return min(position_size, self.config.default_position_size / signal.price)
            
            # Default fixed dollar amount
            return self.config.default_position_size / signal.price
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def _monitor_positions(self):
        """Monitor positions for stop losses and take profits."""
        while self.is_running:
            try:
                positions = self.position_manager.get_all_positions()
                
                for position in positions:
                    # Check stop loss
                    if (position.stop_loss and 
                        ((position.quantity > 0 and position.current_price <= position.stop_loss) or
                         (position.quantity < 0 and position.current_price >= position.stop_loss))):
                        
                        logger.info(f"Stop loss triggered for {position.symbol}")
                        await self._close_position(position.symbol, position.current_price, 'stop_loss')
                    
                    # Check take profit
                    if (position.take_profit and
                        ((position.quantity > 0 and position.current_price >= position.take_profit) or
                         (position.quantity < 0 and position.current_price <= position.take_profit))):
                        
                        logger.info(f"Take profit triggered for {position.symbol}")
                        await self._close_position(position.symbol, position.current_price, 'take_profit')
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_orders(self):
        """Monitor order status and handle fills."""
        while self.is_running:
            try:
                # Check for filled orders
                filled_orders = await self.order_manager.get_filled_orders()
                
                for order in filled_orders:
                    await self._handle_order_fill(order)
                
                # Check for expired orders
                expired_orders = await self.order_manager.get_expired_orders(
                    self.config.order_timeout_minutes
                )
                
                for order in expired_orders:
                    await self.order_manager.cancel_order(order.order_id)
                    logger.warning(f"Cancelled expired order: {order}")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in order monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _handle_order_fill(self, order: Order):
        """Handle a filled order."""
        try:
            self.execution_stats['orders_filled'] += 1
            
            # Calculate commission and slippage
            commission = order.filled_quantity * order.fill_price * self.config.commission_rate
            slippage = order.filled_quantity * abs(order.fill_price - order.price) * self.config.slippage_rate
            
            self.execution_stats['total_commission'] += commission
            self.execution_stats['total_slippage'] += slippage
            
            # Update position
            if order.side in ['buy', 'sell_short']:
                # Opening position
                await self.position_manager.open_position(
                    symbol=order.symbol,
                    quantity=order.filled_quantity if order.side == 'buy' else -order.filled_quantity,
                    entry_price=order.fill_price,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    strategy_name=order.strategy_name
                )
                
                if self.on_position_opened:
                    await self.on_position_opened(order)
                    
                logger.info(f"Position opened: {order.symbol} @ {order.fill_price}")
                
            else:  # sell, buy_to_cover
                # Closing position
                pnl = await self.position_manager.close_position(
                    symbol=order.symbol,
                    exit_price=order.fill_price,
                    exit_time=datetime.now()
                )
                
                self.daily_pnl += pnl
                
                if self.on_position_closed:
                    await self.on_position_closed(order, pnl)
                    
                logger.info(f"Position closed: {order.symbol} @ {order.fill_price}, P&L: ${pnl:.2f}")
            
            # Call order filled hook
            if self.on_order_filled:
                await self.on_order_filled(order)
                
        except Exception as e:
            logger.error(f"Error handling order fill: {e}")
    
    async def _monitor_risk(self):
        """Monitor risk limits and take action if breached."""
        while self.is_running:
            try:
                # Check daily loss limit
                if self.daily_pnl <= -self.config.max_daily_loss:
                    logger.critical(f"Daily loss limit breached: ${self.daily_pnl:.2f}")
                    await self._handle_risk_breach('daily_loss_limit')
                
                # Check portfolio risk
                portfolio_risk = self.position_manager.get_portfolio_risk()
                if portfolio_risk > self.config.max_portfolio_risk:
                    logger.warning(f"Portfolio risk elevated: {portfolio_risk:.2%}")
                    await self._handle_risk_breach('portfolio_risk')
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _handle_risk_breach(self, risk_type: str):
        """Handle risk limit breach."""
        logger.critical(f"Risk breach detected: {risk_type}")
        
        if self.on_risk_breach:
            await self.on_risk_breach(risk_type)
        
        # Take protective action based on risk type
        if risk_type == 'daily_loss_limit':
            # Close all positions and stop trading for the day
            await self._emergency_close_all()
            await self.stop()
            
        elif risk_type == 'portfolio_risk':
            # Close riskiest positions
            await self._reduce_portfolio_risk()
    
    async def _emergency_close_all(self):
        """Emergency close all positions."""
        logger.critical("Executing emergency close of all positions")
        
        positions = self.position_manager.get_all_positions()
        for position in positions:
            await self._close_position(
                position.symbol, 
                position.current_price, 
                'emergency_close'
            )
    
    async def _reduce_portfolio_risk(self):
        """Reduce portfolio risk by closing high-risk positions."""
        positions = self.position_manager.get_all_positions()
        
        # Sort by risk (largest unrealized loss first)
        positions.sort(key=lambda p: p.unrealized_pnl)
        
        # Close positions with largest losses
        for position in positions[:3]:  # Close up to 3 worst positions
            if position.unrealized_pnl < 0:
                await self._close_position(
                    position.symbol,
                    position.current_price,
                    'risk_reduction'
                )
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        now = datetime.now()
        
        # Basic market hours check (9:30 AM - 4:00 PM ET, weekdays)
        if now.weekday() > 4:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Add buffers
        effective_open = market_open + timedelta(seconds=self.config.market_open_delay)
        effective_close = market_close - timedelta(seconds=self.config.market_close_buffer)
        
        return effective_open <= now <= effective_close
    
    def _generate_session_summary(self):
        """Generate and log session summary."""
        if not self.session_start_time:
            return
        
        session_duration = datetime.now() - self.session_start_time
        
        summary = f"""
        ======= EXECUTION SESSION SUMMARY =======
        Session Duration: {session_duration}
        Signals Processed: {self.execution_stats['signals_processed']}
        Orders Submitted: {self.execution_stats['orders_submitted']}
        Orders Filled: {self.execution_stats['orders_filled']}
        Orders Rejected: {self.execution_stats['orders_rejected']}
        Daily P&L: ${self.daily_pnl:.2f}
        Total Commission: ${self.execution_stats['total_commission']:.2f}
        Total Slippage: ${self.execution_stats['total_slippage']:.2f}
        Active Positions: {len(self.position_manager.get_all_positions())}
        ========================================
        """
        
        logger.info(summary)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current execution engine status."""
        return {
            'is_running': self.is_running,
            'mode': self.config.mode.value,
            'session_start_time': self.session_start_time,
            'daily_pnl': self.daily_pnl,
            'active_positions': len(self.position_manager.get_all_positions()),
            'pending_orders': len(self.order_manager.get_pending_orders()),
            'execution_stats': self.execution_stats.copy(),
            'portfolio_value': self.position_manager.get_portfolio_value(),
            'portfolio_risk': self.position_manager.get_portfolio_risk()
        }


def create_default_config() -> ExecutionConfig:
    """Create default execution configuration."""
    return ExecutionConfig(
        mode=ExecutionMode.PAPER,
        max_positions=5,
        default_position_size=10000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_daily_loss=5000.0,
        enable_stop_loss=True,
        enable_take_profit=True
    )