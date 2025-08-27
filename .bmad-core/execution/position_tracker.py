"""
BRF Strategy Position Tracking System
Real-time monitoring and management of trading positions
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

@dataclass
class TradeExecution:
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    price: float
    trade_type: str  # 'entry', 'pyramid', 'exit'
    setup_score: Optional[float] = None
    strategy_context: Optional[Dict] = None

@dataclass
class PositionMetrics:
    symbol: str
    entry_date: datetime
    current_quantity: int
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float
    total_cost_basis: float
    current_value: float
    days_held: int
    max_profit: float
    max_loss: float
    pyramid_levels: int
    exit_levels_completed: int

class BRFPositionTracker:
    def __init__(self, data_path: str = "data/positions"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Active positions and trade history
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[TradeExecution] = []
        self.closed_positions: List[Dict] = []
        
        # Performance tracking
        self.daily_pnl: Dict[str, float] = {}  # Date -> PnL
        self.position_metrics: Dict[str, PositionMetrics] = {}
        
        # BRF-specific tracking
        self.setup_tracking: Dict[str, Dict] = {}  # Track original setup data
        self.pyramid_tracking: Dict[str, List] = {}  # Track pyramid entries
        self.exit_tracking: Dict[str, List] = {}  # Track exit phases
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load existing data
        self.load_positions()
    
    def add_trade(self, trade: TradeExecution):
        """Add new trade execution to tracking"""
        self.trade_history.append(trade)
        
        # Update position based on trade
        if trade.side == 'buy':
            self._process_buy_trade(trade)
        else:
            self._process_sell_trade(trade)
        
        # Save to disk
        self.save_positions()
        
        self.logger.info(f"Trade Added: {trade.symbol} {trade.side.upper()} "
                        f"{trade.quantity} @ ${trade.price:.2f}")
    
    def _process_buy_trade(self, trade: TradeExecution):
        """Process buy trade and update position"""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = {
                'symbol': symbol,
                'entries': [],
                'exits': [],
                'total_quantity': 0,
                'total_cost': 0,
                'avg_entry_price': 0,
                'realized_pnl': 0,
                'first_entry_date': trade.timestamp,
                'status': 'open'
            }
            
            # Track original BRF setup
            if trade.setup_score and trade.strategy_context:
                self.setup_tracking[symbol] = {
                    'setup_score': trade.setup_score,
                    'entry_date': trade.timestamp,
                    'context': trade.strategy_context
                }
        
        # Add entry to position
        position = self.positions[symbol]
        entry_data = {
            'timestamp': trade.timestamp,
            'quantity': trade.quantity,
            'price': trade.price,
            'trade_type': trade.trade_type,
            'setup_score': trade.setup_score
        }
        position['entries'].append(entry_data)
        
        # Update position totals
        position['total_quantity'] += trade.quantity
        position['total_cost'] += trade.quantity * trade.price
        position['avg_entry_price'] = position['total_cost'] / position['total_quantity']
        
        # Track pyramid levels
        if symbol not in self.pyramid_tracking:
            self.pyramid_tracking[symbol] = []
        self.pyramid_tracking[symbol].append(entry_data)
        
        self.logger.info(f"Position Updated: {symbol} - Total Qty: {position['total_quantity']}, "
                        f"Avg Price: ${position['avg_entry_price']:.2f}")
    
    def _process_sell_trade(self, trade: TradeExecution):
        """Process sell trade and update position"""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            self.logger.error(f"Cannot sell {symbol}: No position found")
            return
        
        position = self.positions[symbol]
        
        # Add exit to position
        exit_data = {
            'timestamp': trade.timestamp,
            'quantity': trade.quantity,
            'price': trade.price,
            'trade_type': trade.trade_type
        }
        position['exits'].append(exit_data)
        
        # Calculate realized P&L for this exit
        avg_cost = position['avg_entry_price']
        realized_pnl = trade.quantity * (trade.price - avg_cost)
        position['realized_pnl'] += realized_pnl
        
        # Update position quantity
        position['total_quantity'] -= trade.quantity
        
        # Track exit phases
        if symbol not in self.exit_tracking:
            self.exit_tracking[symbol] = []
        self.exit_tracking[symbol].append(exit_data)
        
        # Close position if fully exited
        if position['total_quantity'] <= 0:
            position['status'] = 'closed'
            position['close_date'] = trade.timestamp
            self.closed_positions.append(position.copy())
            
            # Remove from active positions
            del self.positions[symbol]
            
            self.logger.info(f"Position Closed: {symbol} - Total Realized P&L: "
                           f"${position['realized_pnl']:.2f}")
        
        # Update daily P&L tracking
        today = trade.timestamp.date().isoformat()
        self.daily_pnl[today] = self.daily_pnl.get(today, 0) + realized_pnl
    
    def update_market_prices(self, price_data: Dict[str, float]):
        """Update current market prices for all positions"""
        for symbol, position in self.positions.items():
            if symbol in price_data:
                current_price = price_data[symbol]
                
                # Calculate unrealized P&L
                unrealized_pnl = position['total_quantity'] * (current_price - position['avg_entry_price'])
                
                # Update position metrics
                metrics = self.calculate_position_metrics(symbol, current_price)
                self.position_metrics[symbol] = metrics
                
                # Track high water marks
                if 'max_profit' not in position:
                    position['max_profit'] = 0
                    position['max_loss'] = 0
                
                position['max_profit'] = max(position['max_profit'], unrealized_pnl)
                position['max_loss'] = min(position['max_loss'], unrealized_pnl)
                position['current_price'] = current_price
                position['unrealized_pnl'] = unrealized_pnl
    
    def calculate_position_metrics(self, symbol: str, current_price: float) -> PositionMetrics:
        """Calculate comprehensive position metrics"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        entry_date = position['first_entry_date']
        days_held = (datetime.now() - entry_date).days
        
        unrealized_pnl = position['total_quantity'] * (current_price - position['avg_entry_price'])
        unrealized_pnl_pct = unrealized_pnl / position['total_cost']
        
        metrics = PositionMetrics(
            symbol=symbol,
            entry_date=entry_date,
            current_quantity=position['total_quantity'],
            avg_entry_price=position['avg_entry_price'],
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            realized_pnl=position['realized_pnl'],
            total_cost_basis=position['total_cost'],
            current_value=position['total_quantity'] * current_price,
            days_held=days_held,
            max_profit=position.get('max_profit', 0),
            max_loss=position.get('max_loss', 0),
            pyramid_levels=len(position['entries']),
            exit_levels_completed=len(position['exits'])
        )
        
        return metrics
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        total_positions = len(self.positions)
        total_unrealized = sum(pos.get('unrealized_pnl', 0) for pos in self.positions.values())
        total_realized = sum(pos['realized_pnl'] for pos in self.positions.values())
        total_realized += sum(pos['realized_pnl'] for pos in self.closed_positions)
        
        # Calculate position distribution
        position_values = {}
        total_value = 0
        
        for symbol, position in self.positions.items():
            current_price = position.get('current_price', position['avg_entry_price'])
            value = position['total_quantity'] * current_price
            position_values[symbol] = value
            total_value += value
        
        # BRF-specific metrics
        pyramid_stats = self._calculate_pyramid_stats()
        exit_phase_stats = self._calculate_exit_stats()
        
        return {
            'total_positions': total_positions,
            'total_unrealized_pnl': total_unrealized,
            'total_realized_pnl': total_realized,
            'total_pnl': total_unrealized + total_realized,
            'total_position_value': total_value,
            'largest_position': max(position_values.values()) if position_values else 0,
            'largest_position_symbol': max(position_values, key=position_values.get) if position_values else None,
            'position_distribution': position_values,
            'pyramid_stats': pyramid_stats,
            'exit_phase_stats': exit_phase_stats,
            'closed_positions_count': len(self.closed_positions)
        }
    
    def _calculate_pyramid_stats(self) -> Dict:
        """Calculate pyramid entry statistics"""
        pyramid_levels = []
        setup_scores = []
        
        for symbol, entries in self.pyramid_tracking.items():
            pyramid_levels.append(len(entries))
            for entry in entries:
                if entry.get('setup_score'):
                    setup_scores.append(entry['setup_score'])
        
        return {
            'avg_pyramid_levels': sum(pyramid_levels) / len(pyramid_levels) if pyramid_levels else 0,
            'max_pyramid_levels': max(pyramid_levels) if pyramid_levels else 0,
            'avg_setup_score': sum(setup_scores) / len(setup_scores) if setup_scores else 0,
            'positions_with_pyramids': len([p for p in pyramid_levels if p > 1])
        }
    
    def _calculate_exit_stats(self) -> Dict:
        """Calculate exit phase statistics"""
        exit_phases = []
        for symbol, exits in self.exit_tracking.items():
            exit_phases.append(len(exits))
        
        return {
            'avg_exit_phases': sum(exit_phases) / len(exit_phases) if exit_phases else 0,
            'positions_with_partial_exits': len([p for p in exit_phases if p > 1]),
            'total_exit_executions': sum(exit_phases)
        }
    
    def get_performance_report(self, days: int = 30) -> Dict:
        """Generate performance report for specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter recent trades
        recent_trades = [t for t in self.trade_history if t.timestamp >= cutoff_date]
        recent_closed = [p for p in self.closed_positions 
                        if p.get('close_date') and p['close_date'] >= cutoff_date]
        
        # Calculate metrics
        total_trades = len(recent_trades)
        winning_positions = len([p for p in recent_closed if p['realized_pnl'] > 0])
        losing_positions = len([p for p in recent_closed if p['realized_pnl'] < 0])
        
        win_rate = winning_positions / (winning_positions + losing_positions) if (winning_positions + losing_positions) > 0 else 0
        
        # Daily P&L for the period
        recent_daily_pnl = {date: pnl for date, pnl in self.daily_pnl.items() 
                           if datetime.fromisoformat(date).date() >= cutoff_date.date()}
        
        return {
            'period_days': days,
            'total_trades': total_trades,
            'positions_closed': len(recent_closed),
            'winning_positions': winning_positions,
            'losing_positions': losing_positions,
            'win_rate': win_rate,
            'total_realized_pnl': sum(p['realized_pnl'] for p in recent_closed),
            'avg_win': sum(p['realized_pnl'] for p in recent_closed if p['realized_pnl'] > 0) / max(winning_positions, 1),
            'avg_loss': sum(p['realized_pnl'] for p in recent_closed if p['realized_pnl'] < 0) / max(losing_positions, 1),
            'daily_pnl': recent_daily_pnl,
            'best_day': max(recent_daily_pnl.values()) if recent_daily_pnl else 0,
            'worst_day': min(recent_daily_pnl.values()) if recent_daily_pnl else 0
        }
    
    def save_positions(self):
        """Save position data to disk"""
        data = {
            'positions': self.positions,
            'closed_positions': self.closed_positions,
            'daily_pnl': self.daily_pnl,
            'setup_tracking': self.setup_tracking,
            'pyramid_tracking': self.pyramid_tracking,
            'exit_tracking': self.exit_tracking,
            'last_updated': datetime.now().isoformat()
        }
        
        # Convert datetime objects to strings for JSON serialization
        data_str = json.dumps(data, default=str, indent=2)
        
        file_path = self.data_path / "positions.json"
        with open(file_path, 'w') as f:
            f.write(data_str)
    
    def load_positions(self):
        """Load position data from disk"""
        file_path = self.data_path / "positions.json"
        
        if not file_path.exists():
            return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.positions = data.get('positions', {})
            self.closed_positions = data.get('closed_positions', [])
            self.daily_pnl = data.get('daily_pnl', {})
            self.setup_tracking = data.get('setup_tracking', {})
            self.pyramid_tracking = data.get('pyramid_tracking', {})
            self.exit_tracking = data.get('exit_tracking', {})
            
            self.logger.info(f"Loaded {len(self.positions)} active positions and "
                           f"{len(self.closed_positions)} closed positions")
            
        except Exception as e:
            self.logger.error(f"Error loading positions: {e}")
    
    def export_trade_log(self, file_path: str):
        """Export detailed trade log to CSV"""
        trade_data = []
        
        for trade in self.trade_history:
            trade_data.append({
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'trade_type': trade.trade_type,
                'setup_score': trade.setup_score,
                'value': trade.quantity * trade.price
            })
        
        df = pd.DataFrame(trade_data)
        df.to_csv(file_path, index=False)
        self.logger.info(f"Trade log exported to {file_path}")
    
    def get_position_details(self, symbol: str) -> Dict:
        """Get detailed information for specific position"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        metrics = self.position_metrics.get(symbol)
        setup_data = self.setup_tracking.get(symbol, {})
        pyramid_data = self.pyramid_tracking.get(symbol, [])
        exit_data = self.exit_tracking.get(symbol, [])
        
        return {
            'position': position,
            'metrics': asdict(metrics) if metrics else None,
            'original_setup': setup_data,
            'pyramid_entries': pyramid_data,
            'exit_phases': exit_data,
            'entry_count': len(pyramid_data),
            'exit_count': len(exit_data)
        }