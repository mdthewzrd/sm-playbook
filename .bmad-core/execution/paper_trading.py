"""
BRF Strategy Paper Trading System
Complete simulation environment for testing BRF strategy without real money
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
import asyncio
from datetime import datetime, timedelta
import random
import logging
from dataclasses import dataclass

from .order_manager import BRFOrderManager, Order, OrderStatus, OrderType
from .risk_controls import BRFRiskControls, RiskAlert
from .position_tracker import BRFPositionTracker, TradeExecution

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    spread: float

class PaperTradingEngine:
    def __init__(self, initial_capital: float = 100000, enable_slippage: bool = True):
        self.initial_capital = initial_capital
        self.enable_slippage = enable_slippage
        
        # Initialize core systems
        self.order_manager = BRFOrderManager(initial_capital)
        self.risk_controls = BRFRiskControls(initial_capital)
        self.position_tracker = BRFPositionTracker("data/paper_trading")
        
        # Paper trading specific settings
        self.latency_ms = random.randint(50, 200)  # Simulated latency
        self.slippage_bps = 2  # 2 basis points average slippage
        self.fill_probability = 0.95  # 95% fill rate
        
        # Market simulation
        self.market_data: Dict[str, MarketData] = {}
        self.trading_active = False
        self.simulation_speed = 1.0  # 1.0 = real-time, higher = faster
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Paper Trading Engine initialized with ${initial_capital:,}")
    
    def process_brf_signal(self, symbol: str, setup_data: Dict) -> List[str]:
        """Process BRF signal in paper trading environment"""
        # Pre-trade risk checks
        portfolio_value = self.get_portfolio_value()
        order_value = setup_data.get('suggested_position_size', 5000)
        
        can_trade, risk_alerts = self.risk_controls.check_pre_trade_risk(
            symbol, order_value, self.order_manager.positions, portfolio_value
        )
        
        if not can_trade:
            self.logger.warning(f"Trade blocked by risk controls: {symbol}")
            for alert in risk_alerts:
                self.logger.warning(f"Risk Alert: {alert.message}")
            return []
        
        # Generate orders using order manager
        orders = self.order_manager.process_brf_signal(symbol, setup_data)
        
        # Submit orders to paper trading system
        order_ids = []
        for order in orders:
            order_id = self.submit_paper_order(order)
            if order_id:
                order_ids.append(order_id)
        
        return order_ids
    
    def submit_paper_order(self, order: Order) -> Optional[str]:
        """Submit order to paper trading system with realistic simulation"""
        # Generate order ID
        order.order_id = f"PAPER_{order.symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Add to pending orders
        self.order_manager.pending_orders.append(order)
        
        # Schedule order processing with simulated latency
        asyncio.create_task(self._process_paper_order(order))
        
        self.logger.info(f"Paper Order Submitted: {order.symbol} {order.side.upper()} "
                        f"{order.quantity} @ ${order.price:.2f} [ID: {order.order_id}]")
        
        return order.order_id
    
    async def _process_paper_order(self, order: Order):
        """Process paper order with realistic fills and slippage"""
        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000)
        
        # Check if order should be filled
        if random.random() > self.fill_probability:
            order.status = OrderStatus.REJECTED
            self.logger.warning(f"Paper Order Rejected: {order.order_id}")
            return
        
        # Get current market data
        market_data = self.market_data.get(order.symbol)
        if not market_data:
            # Generate synthetic market data if not available
            market_data = self._generate_market_data(order.symbol, order.price)
        
        # Calculate fill price with slippage
        fill_price = self._calculate_fill_price(order, market_data)
        
        # Execute the fill
        order.status = OrderStatus.FILLED
        order.filled_qty = order.quantity
        order.avg_fill_price = fill_price
        order.timestamp = datetime.now()
        
        # Update order manager position
        self.order_manager.update_position(order)
        
        # Add trade to position tracker
        trade = TradeExecution(
            timestamp=order.timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_qty,
            price=order.avg_fill_price,
            trade_type="entry" if order.side == "buy" else "exit",
            setup_score=None,  # Would be passed from signal
            strategy_context={"paper_trading": True}
        )
        self.position_tracker.add_trade(trade)
        
        # Remove from pending orders
        self.order_manager.pending_orders.remove(order)
        self.order_manager.order_history.append(order)
        
        self.logger.info(f"Paper Order Filled: {order.symbol} {order.side.upper()} "
                        f"{order.filled_qty} @ ${order.avg_fill_price:.2f}")
        
        # Check for automatic exit order generation
        if order.side == "buy":
            self._generate_exit_orders(order.symbol)
    
    def _calculate_fill_price(self, order: Order, market_data: MarketData) -> float:
        """Calculate realistic fill price with slippage"""
        if not self.enable_slippage:
            return order.price
        
        # Base slippage calculation
        slippage_factor = self.slippage_bps / 10000
        
        if order.order_type == OrderType.MARKET:
            # Market orders use bid/ask with slippage
            if order.side == "buy":
                fill_price = market_data.ask * (1 + slippage_factor)
            else:
                fill_price = market_data.bid * (1 - slippage_factor)
        else:
            # Limit orders can get price improvement or slippage
            price_impact = random.uniform(-slippage_factor, slippage_factor)
            fill_price = order.price * (1 + price_impact)
        
        return round(fill_price, 2)
    
    def _generate_market_data(self, symbol: str, reference_price: float) -> MarketData:
        """Generate synthetic market data for paper trading"""
        spread_pct = 0.001  # 0.1% spread
        spread = reference_price * spread_pct
        
        bid = reference_price - spread / 2
        ask = reference_price + spread / 2
        
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            price=reference_price,
            volume=random.randint(10000, 100000),
            bid=bid,
            ask=ask,
            spread=spread
        )
    
    def _generate_exit_orders(self, symbol: str):
        """Automatically generate exit orders for new positions"""
        position = self.order_manager.positions.get(symbol)
        if not position:
            return
        
        # Create phased exit orders
        exit_orders = self.order_manager.create_exit_orders(symbol, position, "profit_target")
        stop_orders = self.order_manager.create_exit_orders(symbol, position, "stop_loss")
        
        # Submit exit orders
        for order in exit_orders + stop_orders:
            self.submit_paper_order(order)
        
        self.logger.info(f"Exit orders generated for {symbol}: "
                        f"{len(exit_orders)} profit targets, {len(stop_orders)} stops")
    
    def update_market_prices(self, price_updates: Dict[str, float]):
        """Update market prices and trigger risk monitoring"""
        # Update market data
        for symbol, price in price_updates.items():
            self.market_data[symbol] = self._generate_market_data(symbol, price)
        
        # Update position tracker
        self.position_tracker.update_market_prices(price_updates)
        
        # Run risk monitoring
        risk_alerts = self.risk_controls.monitor_position_risk(
            self.order_manager.positions, price_updates
        )
        
        # Handle critical risk alerts
        if self.risk_controls.emergency_liquidation_check(self.order_manager.positions):
            self.emergency_liquidation()
    
    def emergency_liquidation(self):
        """Execute emergency liquidation of all positions"""
        self.logger.critical("EMERGENCY LIQUIDATION TRIGGERED")
        
        for symbol, position in self.order_manager.positions.items():
            # Create market sell order for full position
            liquidation_order = Order(
                symbol=symbol,
                side="sell",
                quantity=position.quantity,
                order_type=OrderType.MARKET,
                timestamp=datetime.now()
            )
            
            self.submit_paper_order(liquidation_order)
        
        # Halt all trading
        self.trading_active = False
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        cash_value = self.order_manager.available_capital
        position_value = 0
        
        for position in self.order_manager.positions.values():
            current_price = position.current_price
            position_value += position.quantity * current_price
        
        return cash_value + position_value
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        portfolio_summary = self.order_manager.get_portfolio_summary()
        risk_summary = self.risk_controls.get_risk_summary()
        position_summary = self.position_tracker.get_portfolio_summary()
        performance_report = self.position_tracker.get_performance_report(30)
        
        total_return = (self.get_portfolio_value() - self.initial_capital) / self.initial_capital
        
        return {
            'paper_trading': True,
            'portfolio_value': self.get_portfolio_value(),
            'initial_capital': self.initial_capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'cash_available': portfolio_summary['available_capital'],
            'positions_count': portfolio_summary['positions_count'],
            'trading_active': self.trading_active,
            'risk_status': risk_summary,
            'position_metrics': position_summary,
            'performance_30d': performance_report,
            'total_trades': len(self.position_tracker.trade_history),
            'pending_orders': len(self.order_manager.pending_orders)
        }
    
    def run_backtest_simulation(self, historical_data: Dict[str, pd.DataFrame], 
                              brf_signals: Dict[str, List[Dict]]) -> Dict:
        """Run complete backtest simulation with historical data"""
        self.logger.info("Starting backtest simulation...")
        
        results = {
            'trades_executed': 0,
            'positions_opened': 0,
            'positions_closed': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'daily_returns': []
        }
        
        # Process each symbol's historical data
        for symbol, data in historical_data.items():
            if symbol not in brf_signals:
                continue
            
            symbol_signals = brf_signals[symbol]
            
            # Process each signal chronologically
            for signal in symbol_signals:
                signal_time = pd.to_datetime(signal['timestamp'])
                
                # Find corresponding price data
                price_data = data[data.index <= signal_time].iloc[-1] if len(data) > 0 else None
                if price_data is None:
                    continue
                
                # Create setup data for order manager
                setup_data = {
                    'current_price': price_data['close'],
                    'vwap': signal.get('vwap', price_data['close']),
                    'score': signal.get('score', 75),
                    'suggested_position_size': 5000
                }
                
                # Process signal
                order_ids = self.process_brf_signal(symbol, setup_data)
                if order_ids:
                    results['trades_executed'] += len(order_ids)
                    results['positions_opened'] += 1
        
        # Calculate final results
        final_summary = self.get_performance_summary()
        results.update({
            'final_portfolio_value': final_summary['portfolio_value'],
            'total_return': final_summary['total_return_pct'],
            'positions_closed': len(self.position_tracker.closed_positions),
            'win_rate': final_summary['performance_30d']['win_rate'],
            'total_pnl': final_summary['portfolio_value'] - self.initial_capital
        })
        
        self.logger.info(f"Backtest completed: {results['trades_executed']} trades, "
                        f"{results['total_return']:.2f}% return")
        
        return results
    
    def export_paper_trading_report(self, file_path: str):
        """Export comprehensive paper trading report"""
        summary = self.get_performance_summary()
        
        # Create detailed report
        report = f"""
# Paper Trading Report - BRF Strategy

## Portfolio Summary
- **Initial Capital**: ${self.initial_capital:,}
- **Current Portfolio Value**: ${summary['portfolio_value']:,.2f}
- **Total Return**: {summary['total_return_pct']:.2f}%
- **Available Cash**: ${summary['cash_available']:,.2f}
- **Active Positions**: {summary['positions_count']}

## Performance Metrics (30 Days)
- **Total Trades**: {summary['total_trades']}
- **Win Rate**: {summary['performance_30d']['win_rate']:.1%}
- **Avg Win**: ${summary['performance_30d']['avg_win']:.2f}
- **Avg Loss**: ${summary['performance_30d']['avg_loss']:.2f}
- **Best Day**: ${summary['performance_30d']['best_day']:.2f}
- **Worst Day**: ${summary['performance_30d']['worst_day']:.2f}

## Risk Metrics
- **Trading Halted**: {summary['risk_status']['trading_halted']}
- **Daily P&L**: ${summary['risk_status']['daily_pnl']:.2f}
- **Market Hours**: {summary['risk_status']['market_hours']}
- **Critical Alerts**: {summary['risk_status']['alert_counts']['critical']}

## Position Analysis  
- **Pyramid Stats**: {summary['position_metrics']['pyramid_stats']}
- **Exit Stats**: {summary['position_metrics']['exit_phase_stats']}
- **Largest Position**: ${summary['position_metrics']['largest_position']:,.2f} ({summary['position_metrics']['largest_position_symbol']})

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        with open(file_path, 'w') as f:
            f.write(report)
        
        # Export trade log
        log_file = file_path.replace('.md', '_trades.csv')
        self.position_tracker.export_trade_log(log_file)
        
        self.logger.info(f"Paper trading report exported to {file_path}")
    
    def reset_simulation(self):
        """Reset paper trading simulation to initial state"""
        self.order_manager = BRFOrderManager(self.initial_capital)
        self.risk_controls = BRFRiskControls(self.initial_capital) 
        self.position_tracker = BRFPositionTracker("data/paper_trading_reset")
        self.market_data = {}
        self.trading_active = True
        
        self.logger.info("Paper trading simulation reset to initial state")