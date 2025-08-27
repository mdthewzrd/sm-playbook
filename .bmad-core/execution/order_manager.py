"""
BRF Strategy Order Management System
Handles position sizing, pyramiding, and order execution logic
"""

import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    timestamp: Optional[pd.Timestamp] = None

@dataclass
class Position:
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_levels: List[Dict] = None
    
    def __post_init__(self):
        if self.entry_levels is None:
            self.entry_levels = []

class BRFOrderManager:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Order] = []
        self.order_history: List[Order] = []
        self.max_position_size = 0.05  # 5% max per position
        self.pyramid_levels = 3  # Max 3 entries per position
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              setup_score: float, pyramid_level: int = 1) -> int:
        """Calculate position size based on BRF strategy rules"""
        # Base allocation: 2-5% of capital based on setup score
        base_allocation_pct = 0.02 + (setup_score - 70) * 0.001  # 2-5%
        base_allocation = self.available_capital * base_allocation_pct
        
        # Pyramid sizing: 50%, 30%, 20% allocation
        pyramid_weights = [0.5, 0.3, 0.2]
        if pyramid_level > len(pyramid_weights):
            return 0
            
        allocated_amount = base_allocation * pyramid_weights[pyramid_level - 1]
        shares = int(allocated_amount / entry_price)
        
        return max(1, shares)  # Minimum 1 share
    
    def create_brf_entry_orders(self, symbol: str, setup_data: Dict) -> List[Order]:
        """Create pyramiding entry orders for BRF setup"""
        orders = []
        current_price = setup_data['current_price']
        setup_score = setup_data['score']
        vwap = setup_data['vwap']
        
        # Entry levels for pyramiding
        entry_levels = [
            current_price,  # Initial entry at current price
            current_price + (current_price - vwap) * 0.1,  # 10% extension
            current_price + (current_price - vwap) * 0.2   # 20% extension
        ]
        
        for i, entry_price in enumerate(entry_levels, 1):
            quantity = self.calculate_position_size(symbol, entry_price, setup_score, i)
            
            if quantity > 0:
                order = Order(
                    symbol=symbol,
                    side="buy",
                    quantity=quantity,
                    order_type=OrderType.LIMIT,
                    price=entry_price,
                    timestamp=pd.Timestamp.now()
                )
                orders.append(order)
        
        return orders
    
    def create_exit_orders(self, symbol: str, position: Position, 
                          exit_reason: str = "profit_target") -> List[Order]:
        """Create exit orders based on BRF strategy rules"""
        orders = []
        
        if exit_reason == "profit_target":
            # Phased profit taking: 30%, 50%, 20%
            profit_phases = [0.3, 0.5, 0.2]
            current_qty = position.quantity
            
            for i, phase_pct in enumerate(profit_phases):
                phase_qty = int(current_qty * phase_pct)
                if phase_qty > 0:
                    # Calculate target price based on ATR and risk/reward
                    target_price = position.avg_cost * (1 + 0.02 * (i + 1))  # 2%, 4%, 6%
                    
                    order = Order(
                        symbol=symbol,
                        side="sell",
                        quantity=phase_qty,
                        order_type=OrderType.LIMIT,
                        price=target_price,
                        timestamp=pd.Timestamp.now()
                    )
                    orders.append(order)
        
        elif exit_reason == "stop_loss":
            # Full position stop loss
            stop_price = position.avg_cost * 0.98  # 2% stop loss
            
            order = Order(
                symbol=symbol,
                side="sell",
                quantity=position.quantity,
                order_type=OrderType.STOP,
                stop_price=stop_price,
                timestamp=pd.Timestamp.now()
            )
            orders.append(order)
        
        return orders
    
    def process_brf_signal(self, symbol: str, setup_data: Dict) -> List[Order]:
        """Process BRF signal and generate appropriate orders"""
        orders = []
        
        # Check if we already have a position
        existing_position = self.positions.get(symbol)
        
        if existing_position is None:
            # New position - create entry orders
            entry_orders = self.create_brf_entry_orders(symbol, setup_data)
            orders.extend(entry_orders)
            
            self.logger.info(f"BRF Signal: {symbol} - Created {len(entry_orders)} entry orders")
            
        elif len(existing_position.entry_levels) < self.pyramid_levels:
            # Existing position - add pyramid level if conditions met
            current_price = setup_data['current_price']
            last_entry = existing_position.entry_levels[-1]['price']
            
            # Only pyramid if price extended further (10% higher than last entry)
            if current_price > last_entry * 1.1:
                pyramid_level = len(existing_position.entry_levels) + 1
                quantity = self.calculate_position_size(
                    symbol, current_price, setup_data['score'], pyramid_level
                )
                
                if quantity > 0:
                    order = Order(
                        symbol=symbol,
                        side="buy",
                        quantity=quantity,
                        order_type=OrderType.LIMIT,
                        price=current_price,
                        timestamp=pd.Timestamp.now()
                    )
                    orders.append(order)
                    
                    self.logger.info(f"BRF Pyramid: {symbol} Level {pyramid_level}")
        
        return orders
    
    def update_position(self, order: Order):
        """Update position after order fill"""
        if order.status != OrderStatus.FILLED:
            return
        
        symbol = order.symbol
        
        if order.side == "buy":
            if symbol in self.positions:
                # Update existing position
                pos = self.positions[symbol]
                total_cost = (pos.quantity * pos.avg_cost) + (order.filled_qty * order.avg_fill_price)
                total_qty = pos.quantity + order.filled_qty
                pos.avg_cost = total_cost / total_qty
                pos.quantity = total_qty
                
                # Record entry level
                pos.entry_levels.append({
                    'price': order.avg_fill_price,
                    'quantity': order.filled_qty,
                    'timestamp': order.timestamp
                })
            else:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=order.filled_qty,
                    avg_cost=order.avg_fill_price,
                    current_price=order.avg_fill_price,
                    entry_levels=[{
                        'price': order.avg_fill_price,
                        'quantity': order.filled_qty,
                        'timestamp': order.timestamp
                    }]
                )
            
            # Update available capital
            self.available_capital -= order.filled_qty * order.avg_fill_price
            
        elif order.side == "sell":
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.quantity -= order.filled_qty
                
                # Calculate realized P&L
                realized_pnl = order.filled_qty * (order.avg_fill_price - pos.avg_cost)
                pos.realized_pnl += realized_pnl
                
                # Update available capital
                self.available_capital += order.filled_qty * order.avg_fill_price
                
                # Remove position if fully closed
                if pos.quantity <= 0:
                    del self.positions[symbol]
                
                self.logger.info(f"Position Update: {symbol} - Realized P&L: ${realized_pnl:.2f}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        total_value = self.available_capital
        total_unrealized = 0
        total_realized = sum(pos.realized_pnl for pos in self.positions.values())
        
        for pos in self.positions.values():
            position_value = pos.quantity * pos.current_price
            total_value += position_value
            total_unrealized += pos.unrealized_pnl
        
        return {
            'total_value': total_value,
            'available_capital': self.available_capital,
            'positions_count': len(self.positions),
            'total_unrealized_pnl': total_unrealized,
            'total_realized_pnl': total_realized,
            'total_return': (total_value - self.initial_capital) / self.initial_capital
        }
    
    def submit_orders(self, orders: List[Order]) -> List[str]:
        """Submit orders to broker (placeholder for actual implementation)"""
        order_ids = []
        
        for order in orders:
            # In real implementation, this would connect to broker API
            order.order_id = f"BRF_{order.symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            self.pending_orders.append(order)
            order_ids.append(order.order_id)
            
            self.logger.info(
                f"Order Submitted: {order.symbol} {order.side.upper()} "
                f"{order.quantity} @ ${order.price:.2f} [ID: {order.order_id}]"
            )
        
        return order_ids