"""
BRF Strategy Complete Execution System
Integrates all execution components for live and paper trading
"""

import pandas as pd
from typing import Dict, List, Optional, Union
import asyncio
from datetime import datetime
import logging
from pathlib import Path

# Import all execution components
from .order_manager import BRFOrderManager, Order, OrderStatus
from .risk_controls import BRFRiskControls, RiskLevel, RiskAlert
from .position_tracker import BRFPositionTracker, TradeExecution
from .paper_trading import PaperTradingEngine

# Import backtesting and strategy components
import sys
sys.path.append(str(Path(__file__).parent.parent))

class BRFExecutionSystem:
    """
    Main execution system that coordinates all BRF strategy components
    Handles both live trading and paper trading modes
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 mode: str = "paper",  # "paper" or "live"
                 data_path: str = "data/execution"):
        
        self.initial_capital = initial_capital
        self.mode = mode
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize core systems
        if mode == "paper":
            self.paper_engine = PaperTradingEngine(initial_capital)
            self.order_manager = self.paper_engine.order_manager
            self.risk_controls = self.paper_engine.risk_controls
            self.position_tracker = self.paper_engine.position_tracker
        else:
            # Live trading mode
            self.order_manager = BRFOrderManager(initial_capital)
            self.risk_controls = BRFRiskControls(initial_capital)
            self.position_tracker = BRFPositionTracker(str(self.data_path / "live"))
            self.paper_engine = None
        
        # System status
        self.system_active = False
        self.last_signal_time = None
        self.performance_metrics = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"BRF Execution System initialized - Mode: {mode.upper()}, Capital: ${initial_capital:,}")
    
    async def start_system(self):
        """Start the execution system"""
        self.system_active = True
        self.logger.info("BRF Execution System STARTED")
        
        # Start monitoring tasks
        if self.mode == "live":
            # In live mode, start risk monitoring and position updates
            asyncio.create_task(self._monitor_positions())
            asyncio.create_task(self._monitor_risk())
    
    def stop_system(self):
        """Stop the execution system"""
        self.system_active = False
        self.logger.info("BRF Execution System STOPPED")
    
    def process_brf_signal(self, symbol: str, signal_data: Dict) -> Dict:
        """
        Process incoming BRF signal and execute trades
        
        Args:
            symbol: Stock symbol
            signal_data: BRF signal data including score, price, vwap, etc.
            
        Returns:
            Execution result with order IDs and status
        """
        if not self.system_active:
            return {'status': 'error', 'message': 'System not active'}
        
        self.last_signal_time = datetime.now()
        
        # Log signal reception
        self.logger.info(f"BRF Signal Received: {symbol} - Score: {signal_data.get('score', 'N/A')}")
        
        # Validate signal data
        if not self._validate_signal_data(signal_data):
            return {'status': 'error', 'message': 'Invalid signal data'}
        
        # Process through appropriate engine
        if self.mode == "paper":
            order_ids = self.paper_engine.process_brf_signal(symbol, signal_data)
        else:
            order_ids = self._process_live_signal(symbol, signal_data)
        
        # Log execution result
        result = {
            'status': 'success' if order_ids else 'blocked',
            'symbol': symbol,
            'order_ids': order_ids,
            'timestamp': self.last_signal_time,
            'mode': self.mode
        }
        
        self.logger.info(f"BRF Signal Processed: {symbol} - {len(order_ids)} orders submitted")
        return result
    
    def _validate_signal_data(self, signal_data: Dict) -> bool:
        """Validate incoming signal data"""
        required_fields = ['current_price', 'score', 'vwap']
        
        for field in required_fields:
            if field not in signal_data:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        # Validate score range
        score = signal_data.get('score', 0)
        if not (70 <= score <= 100):
            self.logger.warning(f"Score {score} outside valid range (70-100)")
            return False
        
        # Golden Time Zone validation (8:30 AM - 11:30 AM ET)
        if not self._is_golden_time_zone():
            self.logger.info("Signal outside Golden Time Zone (8:30-11:30 AM ET) - rejected")
            return False
        
        return True
    
    def _is_golden_time_zone(self) -> bool:
        """Check if current time is within the Golden Time Zone (8:30-11:30 AM ET)"""
        from datetime import datetime
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
            
            is_golden_time = golden_start <= current_et <= golden_end
            
            if not is_golden_time:
                self.logger.debug(f"Current time {current_et.strftime('%H:%M')} outside Golden Time Zone (08:30-11:30)")
            
            return is_golden_time
            
        except Exception as e:
            self.logger.error(f"Error checking Golden Time Zone: {e}")
            return False  # Fail safe - reject if unable to determine time
    
    def _process_live_signal(self, symbol: str, signal_data: Dict) -> List[str]:
        """Process signal in live trading mode"""
        # Pre-trade risk checks
        portfolio_value = self._get_portfolio_value()
        suggested_size = signal_data.get('suggested_position_size', 5000)
        
        can_trade, risk_alerts = self.risk_controls.check_pre_trade_risk(
            symbol, suggested_size, self.order_manager.positions, portfolio_value
        )
        
        if not can_trade:
            self.logger.warning(f"Trade blocked by risk controls: {symbol}")
            return []
        
        # Generate and submit orders
        orders = self.order_manager.process_brf_signal(symbol, signal_data)
        order_ids = []
        
        for order in orders:
            # In live mode, this would connect to actual broker
            order_id = self._submit_live_order(order)
            if order_id:
                order_ids.append(order_id)
        
        return order_ids
    
    def _submit_live_order(self, order: Order) -> Optional[str]:
        """Submit order to live broker (placeholder)"""
        # TODO: Implement actual broker integration
        # For now, just simulate order submission
        order.order_id = f"LIVE_{order.symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"LIVE ORDER SUBMITTED: {order.symbol} {order.side} "
                        f"{order.quantity} @ ${order.price:.2f}")
        
        return order.order_id
    
    async def _monitor_positions(self):
        """Monitor positions in live mode"""
        while self.system_active:
            try:
                # Get current market prices (would connect to data feed)
                price_updates = await self._get_current_prices()
                
                # Update position tracker
                if price_updates:
                    self.position_tracker.update_market_prices(price_updates)
                
                # Sleep before next update
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_risk(self):
        """Monitor risk controls in live mode"""
        while self.system_active:
            try:
                # Get current prices
                price_updates = await self._get_current_prices()
                
                if price_updates:
                    # Run risk monitoring
                    risk_alerts = self.risk_controls.monitor_position_risk(
                        self.order_manager.positions, price_updates
                    )
                    
                    # Handle critical alerts
                    if self.risk_controls.emergency_liquidation_check(self.order_manager.positions):
                        await self._emergency_liquidation()
                
                await asyncio.sleep(1)  # Check risk every second
                
            except Exception as e:
                self.logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _get_current_prices(self) -> Dict[str, float]:
        """Get current market prices (placeholder for live data feed)"""
        # TODO: Implement actual market data connection
        # For now, return empty dict
        return {}
    
    async def _emergency_liquidation(self):
        """Execute emergency liquidation"""
        self.logger.critical("EMERGENCY LIQUIDATION INITIATED")
        
        for symbol, position in self.order_manager.positions.items():
            # Create emergency sell order
            emergency_order = Order(
                symbol=symbol,
                side="sell",
                quantity=position.quantity,
                order_type="market",
                timestamp=datetime.now()
            )
            
            # Submit immediately
            order_id = self._submit_live_order(emergency_order)
            if order_id:
                self.logger.critical(f"Emergency liquidation order: {symbol} - {order_id}")
        
        # Stop all trading
        self.system_active = False
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        if self.mode == "paper":
            return self.paper_engine.get_portfolio_value()
        else:
            # Calculate for live mode
            cash = self.order_manager.available_capital
            position_value = sum(
                pos.quantity * pos.current_price 
                for pos in self.order_manager.positions.values()
            )
            return cash + position_value
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        if self.mode == "paper":
            performance = self.paper_engine.get_performance_summary()
        else:
            performance = {
                'portfolio_value': self._get_portfolio_value(),
                'positions_count': len(self.order_manager.positions),
                'available_capital': self.order_manager.available_capital
            }
        
        # Add system-specific metrics
        performance.update({
            'system_active': self.system_active,
            'mode': self.mode,
            'last_signal_time': self.last_signal_time,
            'initial_capital': self.initial_capital
        })
        
        return performance
    
    def get_position_details(self, symbol: str = None) -> Union[Dict, List[Dict]]:
        """Get detailed position information"""
        if symbol:
            return self.position_tracker.get_position_details(symbol)
        else:
            # Return all positions
            all_positions = []
            for symbol in self.order_manager.positions.keys():
                pos_details = self.position_tracker.get_position_details(symbol)
                if pos_details:
                    all_positions.append(pos_details)
            return all_positions
    
    def get_risk_report(self) -> Dict:
        """Get current risk status report"""
        return self.risk_controls.get_risk_summary()
    
    def get_performance_report(self, days: int = 30) -> Dict:
        """Get performance report for specified period"""
        return self.position_tracker.get_performance_report(days)
    
    def export_execution_report(self, file_path: str):
        """Export comprehensive execution report"""
        if self.mode == "paper":
            self.paper_engine.export_paper_trading_report(file_path)
        else:
            # Create live trading report
            status = self.get_system_status()
            risk_report = self.get_risk_report()
            performance = self.get_performance_report()
            
            report_content = f"""
# BRF Strategy Execution Report - LIVE TRADING

## System Status
- **Mode**: {status['mode'].upper()}
- **System Active**: {status['system_active']}
- **Portfolio Value**: ${status['portfolio_value']:,.2f}
- **Active Positions**: {status['positions_count']}
- **Last Signal**: {status['last_signal_time']}

## Risk Status
- **Trading Halted**: {risk_report['trading_halted']}
- **Daily P&L**: ${risk_report['daily_pnl']:.2f}
- **Market Hours**: {risk_report['market_hours']}

## Performance ({performance['period_days']} days)
- **Total Trades**: {performance['total_trades']}
- **Win Rate**: {performance['win_rate']:.1%}
- **Total P&L**: ${performance['total_realized_pnl']:.2f}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            with open(file_path, 'w') as f:
                f.write(report_content)
        
        self.logger.info(f"Execution report exported to {file_path}")
    
    def switch_mode(self, new_mode: str):
        """Switch between paper and live trading modes"""
        if new_mode == self.mode:
            return
        
        self.logger.warning(f"Switching mode from {self.mode.upper()} to {new_mode.upper()}")
        
        # Stop current system
        self.stop_system()
        
        # Reinitialize with new mode
        self.mode = new_mode
        self.__init__(self.initial_capital, new_mode, str(self.data_path))
        
    def force_liquidate_position(self, symbol: str) -> bool:
        """Force liquidation of specific position"""
        if symbol not in self.order_manager.positions:
            self.logger.error(f"No position found for {symbol}")
            return False
        
        position = self.order_manager.positions[symbol]
        
        # Create market sell order
        liquidation_order = Order(
            symbol=symbol,
            side="sell",
            quantity=position.quantity,
            order_type="market",
            timestamp=datetime.now()
        )
        
        # Submit order
        if self.mode == "paper":
            order_id = self.paper_engine.submit_paper_order(liquidation_order)
        else:
            order_id = self._submit_live_order(liquidation_order)
        
        if order_id:
            self.logger.info(f"Forced liquidation initiated: {symbol} - {order_id}")
            return True
        
        return False
    
    def update_risk_limits(self, new_limits: Dict):
        """Update risk control limits"""
        old_limits = self.risk_controls.risk_limits.copy()
        
        for key, value in new_limits.items():
            if key in self.risk_controls.risk_limits:
                self.risk_controls.risk_limits[key] = value
                self.logger.info(f"Risk limit updated: {key} = {value}")
        
        self.logger.info(f"Risk limits updated: {len(new_limits)} changes")
    
    def get_system_health(self) -> Dict:
        """Get system health check results"""
        health = {
            'overall_status': 'healthy',
            'components': {},
            'warnings': [],
            'errors': []
        }
        
        # Check each component
        try:
            # Order Manager
            health['components']['order_manager'] = {
                'status': 'healthy',
                'pending_orders': len(self.order_manager.pending_orders),
                'positions': len(self.order_manager.positions)
            }
        except Exception as e:
            health['components']['order_manager'] = {'status': 'error', 'message': str(e)}
            health['errors'].append(f"Order Manager: {e}")
        
        try:
            # Risk Controls
            risk_summary = self.risk_controls.get_risk_summary()
            health['components']['risk_controls'] = {
                'status': 'halted' if risk_summary['trading_halted'] else 'healthy',
                'critical_alerts': risk_summary['alert_counts']['critical'],
                'market_hours': risk_summary['market_hours']
            }
            
            if risk_summary['trading_halted']:
                health['warnings'].append("Trading halted by risk controls")
        except Exception as e:
            health['components']['risk_controls'] = {'status': 'error', 'message': str(e)}
            health['errors'].append(f"Risk Controls: {e}")
        
        try:
            # Position Tracker
            portfolio_summary = self.position_tracker.get_portfolio_summary()
            health['components']['position_tracker'] = {
                'status': 'healthy',
                'total_positions': portfolio_summary['total_positions'],
                'unrealized_pnl': portfolio_summary['total_unrealized_pnl']
            }
        except Exception as e:
            health['components']['position_tracker'] = {'status': 'error', 'message': str(e)}
            health['errors'].append(f"Position Tracker: {e}")
        
        # Determine overall status
        if health['errors']:
            health['overall_status'] = 'error'
        elif health['warnings']:
            health['overall_status'] = 'warning'
        
        return health