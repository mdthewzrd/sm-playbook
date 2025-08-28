"""
IBIT GTZ EMA Reversion Short Strategy - CORRECTED VERSION

Proper implementation based on actual strategy rules:
1. Route Start: Price hits 1H EMA(9) after 8 AM
2. Initial Entry: 2M bar break of lows 
3. Stop: 10% above high-of-day so far
4. First Add: 5M bar break → full size vs 1¢ above highest high
5. Pyramids: 5M/2M dev band hits → 2M bar breaks with available risk
6. Targets: 15M dev bands (partial), 1H dev bands (main)
"""

import matplotlib
matplotlib.use("Qt5Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
import os
from polygon import RESTClient
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.ticker import FuncFormatter, MaxNLocator
warnings.filterwarnings('ignore')

plt.style.use('dark_background')

# Import custom indicator
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'indicators'))
from indicators.base_indicator import BaseIndicator
from indicators.dual_deviation_cloud import DualDeviationCloud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StrategyState:
    """Strategy state management"""
    route_start_established: bool = False
    high_of_day: float = 0.0
    last_swing_high_5m: float = 0.0
    vwap_break_confirmed: bool = False
    consecutive_5m_breaks: int = 0
    current_date: Optional[datetime] = None
    
    def reset_daily(self):
        """Reset state for new trading day"""
        self.route_start_established = False
        self.high_of_day = 0.0
        self.last_swing_high_5m = 0.0
        self.vwap_break_confirmed = False
        self.consecutive_5m_breaks = 0


@dataclass
class TradeRecord:
    """Trade execution record"""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0.0
    stop_loss: float = 0.0
    trade_type: str = "initial"  # initial, add, pyramid
    risk_amount: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    
    def is_open(self) -> bool:
        return self.exit_time is None


class IBITGTZCorrectedStrategy:
    """Corrected IBIT GTZ Strategy Implementation"""
    
    def __init__(self, config_path: str):
        """Initialize strategy"""
        self.config = self._load_config(config_path)['simulation_environment']
        self.state = StrategyState()
        self.trades = []
        self.open_positions = []
        self.indicators = {}
        self.data = {}
        
        # Strategy parameters
        self.base_risk = 1000  # $1000 base risk per trade
        self.max_consecutive_breaks = 3
        
        # Initialize Polygon client
        self.polygon_api_key = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
        self.polygon_client = RESTClient(self.polygon_api_key)
        
        logger.info(f"Initialized CORRECTED IBIT GTZ Strategy")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load backtest configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config['backtest_configuration']
    
    def download_data(self) -> Dict[str, pd.DataFrame]:
        """Download extended hours data"""
        symbol = self.config['symbol']
        start_date = self.config['start_date']
        end_date = self.config['end_date']
        
        logger.info(f"Downloading {symbol} data from {start_date} to {end_date}")
        
        # Extend date range
        extended_start = (pd.to_datetime(start_date) - timedelta(days=3)).strftime('%Y-%m-%d')
        extended_end = (pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        
        timeframes = {
            '1h': (60, 'minute'),
            '15m': (15, 'minute'), 
            '5m': (5, 'minute'),
            '2m': (2, 'minute')
        }
        
        data = {}
        for tf_name, (multiplier, timespan) in timeframes.items():
            try:
                logger.info(f"Downloading {tf_name} data...")
                
                aggs = []
                for agg in self.polygon_client.get_aggs(
                    ticker=symbol,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=extended_start,
                    to=extended_end,
                    adjusted=True,
                    limit=50000
                ):
                    timestamp_et = pd.to_datetime(agg.timestamp, unit='ms', utc=True).tz_convert('US/Eastern')
                    
                    aggs.append({
                        'timestamp': timestamp_et,
                        'open': agg.open,
                        'high': agg.high,
                        'low': agg.low,
                        'close': agg.close,
                        'volume': agg.volume,
                        'vwap': agg.vwap if hasattr(agg, 'vwap') and agg.vwap else agg.close
                    })
                
                if not aggs:
                    continue
                
                df = pd.DataFrame(aggs)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Filter to extended hours (4 AM - 8 PM ET) and weekdays
                extended_hours_mask = (
                    (df.index.time >= time(4, 0)) & 
                    (df.index.time <= time(20, 0)) &
                    (df.index.dayofweek < 5)
                )
                df = df[extended_hours_mask]
                
                # Calculate daily VWAP
                if 'vwap' not in df.columns or df['vwap'].isna().all():
                    df['date'] = df.index.date
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
                    df['cum_vol_price'] = (typical_price * df['volume']).groupby(df['date']).cumsum()
                    df['vwap'] = df['cum_vol_price'] / df['cum_vol']
                    df = df.drop(['date', 'cum_vol', 'cum_vol_price'], axis=1)
                
                data[tf_name] = df
                logger.info(f"Downloaded {len(df)} bars for {tf_name}")
                
            except Exception as e:
                logger.error(f"Error downloading {tf_name} data: {e}")
                continue
        
        self.data = data
        return data
    
    def calculate_indicators(self):
        """Calculate DualDeviationCloud for all timeframes"""
        logger.info("Calculating indicators for all timeframes")
        
        for timeframe, df in self.data.items():
            logger.info(f"Processing {timeframe} timeframe indicators")
            
            # Initialize DualDeviationCloud
            ddc = DualDeviationCloud({
                'ema_fast_length': 9,
                'ema_slow_length': 20,
                'positive_dev_1': 1.0,
                'positive_dev_2': 0.5,
                'negative_dev_1': 2.0,
                'negative_dev_2': 2.4
            })
            
            indicator_data = ddc.calculate(df)
            self.indicators[timeframe] = indicator_data
            
            logger.info(f"Completed indicators for {timeframe}")
    
    def check_route_start(self, dt: datetime) -> bool:
        """Check if price hits 1H EMA(9) after 8 AM"""
        if dt.time() < time(8, 0):
            return False
            
        if self.state.route_start_established:
            return False
            
        try:
            # Get 1H data
            h1_data = self.indicators['1h']
            h1_times = h1_data.index[h1_data.index <= dt]
            if len(h1_times) == 0:
                return False
                
            h1_current = h1_data.loc[h1_times[-1]]
            h1_ema9 = h1_current['fast_ema']
            
            # Get current 5M bar to check if high hit EMA
            m5_data = self.indicators['5m']
            m5_times = m5_data.index[m5_data.index <= dt]
            if len(m5_times) == 0:
                return False
                
            current_5m = m5_data.loc[m5_times[-1]]
            
            # Route Start: price hits 1H EMA(9)
            if current_5m['high'] >= h1_ema9:
                self.state.high_of_day = max(self.state.high_of_day, current_5m['high'])
                logger.info(f"Route Start: 5M high {current_5m['high']:.2f} hit 1H EMA9 {h1_ema9:.2f} at {dt}")
                return True
                
        except Exception as e:
            logger.error(f"Error checking route start at {dt}: {e}")
            
        return False
    
    def check_2m_bar_break(self, dt: datetime, target_type: str = "low") -> bool:
        """Check for 2M bar break of lows"""
        try:
            m2_data = self.indicators['2m']
            m2_times = m2_data.index[m2_data.index <= dt]
            
            if len(m2_times) < 2:
                return False
                
            current_bar = m2_data.loc[m2_times[-1]]
            prev_bar = m2_data.loc[m2_times[-2]]
            
            if target_type == "low":
                return current_bar['low'] < prev_bar['low']
            else:
                return current_bar['high'] > prev_bar['high']
                
        except Exception as e:
            logger.error(f"Error checking 2M bar break at {dt}: {e}")
            return False
    
    def check_5m_bar_break(self, dt: datetime) -> bool:
        """Check for 5M bar break of lows"""
        try:
            m5_data = self.indicators['5m']
            m5_times = m5_data.index[m5_data.index <= dt]
            
            if len(m5_times) < 2:
                return False
                
            current_bar = m5_data.loc[m5_times[-1]]
            prev_bar = m5_data.loc[m5_times[-2]]
            
            return current_bar['low'] < prev_bar['low']
                
        except Exception as e:
            logger.error(f"Error checking 5M bar break at {dt}: {e}")
            return False
    
    def check_vwap_break(self, dt: datetime) -> bool:
        """Check for 5M close below VWAP after market open"""
        if dt.time() < time(9, 30):
            return False
            
        try:
            m5_data = self.indicators['5m']
            m5_times = m5_data.index[m5_data.index <= dt]
            
            if len(m5_times) == 0:
                return False
                
            current_bar = m5_data.loc[m5_times[-1]]
            return current_bar['close'] < current_bar['vwap']
                
        except Exception as e:
            logger.error(f"Error checking VWAP break at {dt}: {e}")
            return False
    
    def check_dev_band_hit(self, dt: datetime, timeframe: str) -> bool:
        """Check if price hit 9/20 deviation bands"""
        try:
            tf_data = self.indicators[timeframe]
            tf_times = tf_data.index[tf_data.index <= dt]
            
            if len(tf_times) == 0:
                return False
                
            current_bar = tf_data.loc[tf_times[-1]]
            
            # Check if high touched upper bands or low touched lower bands
            return (current_bar['high'] >= current_bar['upper_band_2'] or 
                    current_bar['low'] <= current_bar['lower_band_2'])
                
        except Exception as e:
            logger.error(f"Error checking dev bands at {dt}: {e}")
            return False
    
    def calculate_position_size(self, entry_price: float, stop_price: float, risk_amount: float) -> float:
        """Calculate position size based on risk"""
        if stop_price <= entry_price:  # For shorts, stop should be above entry
            return 0
            
        risk_per_share = stop_price - entry_price
        if risk_per_share <= 0:
            return 0
            
        return risk_amount / risk_per_share
    
    def calculate_available_risk(self) -> float:
        """Calculate available risk from stop movement"""
        if not self.open_positions:
            return self.base_risk
            
        # Calculate risk freed up by moving stops
        total_freed_risk = 0
        for trade in self.open_positions:
            # Simplified: assume we can free up some risk as stops move
            total_freed_risk += self.base_risk * 0.5  # 50% of original risk available
            
        return min(total_freed_risk, self.base_risk * 2)  # Cap at 2x base risk
    
    def execute_trade(self, dt: datetime, entry_price: float, trade_type: str, risk_amount: float) -> TradeRecord:
        """Execute a trade with proper sizing"""
        
        # Calculate stop loss
        if trade_type == "initial":
            stop_loss = self.state.high_of_day * 1.1  # 10% above high of day
        elif trade_type == "add":
            stop_loss = self.state.high_of_day + 0.01  # 1¢ above highest high
        else:  # pyramid
            stop_loss = self.state.last_swing_high_5m + 0.05  # 5¢ above last swing high
        
        # Calculate position size
        quantity = self.calculate_position_size(entry_price, stop_loss, risk_amount)
        
        if quantity <= 0:
            logger.warning(f"Invalid position size calculated for {trade_type} at {dt}")
            return None
        
        trade = TradeRecord(
            entry_time=dt,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            trade_type=trade_type,
            risk_amount=risk_amount
        )
        
        self.trades.append(trade)
        self.open_positions.append(trade)
        
        logger.info(f"Executed {trade_type} SHORT at {dt}: ${entry_price:.2f}, Size: {quantity:.0f}, "
                   f"Stop: ${stop_loss:.2f}, Risk: ${risk_amount:.0f}")
        return trade
    
    def manage_positions(self, dt: datetime):
        """Manage open positions"""
        if not self.open_positions:
            return
            
        try:
            # Get current data
            m5_data = self.indicators['5m']
            m5_times = m5_data.index[m5_data.index <= dt]
            if len(m5_times) == 0:
                return
                
            current_bar = m5_data.loc[m5_times[-1]]
            
            positions_to_close = []
            
            for trade in self.open_positions:
                # Check stop loss
                if current_bar['high'] >= trade.stop_loss:
                    positions_to_close.append((trade, trade.stop_loss, "stop_loss"))
                    continue
                
                # Check targets (simplified)
                # 15M dev band partial exit
                if self.check_dev_band_hit(dt, '15m'):
                    positions_to_close.append((trade, current_bar['low'], "partial_target"))
                    continue
                    
                # 1H dev band main target
                if self.check_dev_band_hit(dt, '1h'):
                    positions_to_close.append((trade, current_bar['low'], "main_target"))
                    continue
            
            # Execute exits
            for trade, exit_price, reason in positions_to_close:
                self.close_position(trade, dt, exit_price, reason)
                
        except Exception as e:
            logger.error(f"Error managing positions at {dt}: {e}")
    
    def close_position(self, trade: TradeRecord, exit_time: datetime, exit_price: float, exit_reason: str):
        """Close a position"""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        
        # Calculate P&L for short position
        trade.pnl = (trade.entry_price - exit_price) * trade.quantity
        
        # Remove from open positions
        if trade in self.open_positions:
            self.open_positions.remove(trade)
        
        logger.info(f"Closed {trade.trade_type} at {exit_time}: ${exit_price:.2f}, "
                   f"P&L: ${trade.pnl:.2f}, Reason: {exit_reason}")
    
    def update_swing_highs(self, dt: datetime):
        """Update 5M swing highs for stop management"""
        try:
            m5_data = self.indicators['5m']
            m5_times = m5_data.index[m5_data.index <= dt]
            
            if len(m5_times) >= 5:  # Need at least 5 bars to determine swing
                recent_highs = m5_data.loc[m5_times[-5:]]['high']
                self.state.last_swing_high_5m = recent_highs.max()
                
        except Exception as e:
            logger.error(f"Error updating swing highs at {dt}: {e}")
    
    def run_backtest(self):
        """Run the corrected backtest"""
        logger.info("Starting CORRECTED IBIT GTZ Strategy Backtest")
        
        # Download and prepare data
        self.download_data()
        self.calculate_indicators()
        
        # Get 5M timestamps for main loop
        m5_data = self.indicators['5m']
        timestamps = m5_data.index
        
        logger.info(f"Processing {len(timestamps)} 5-minute bars")
        
        for i, current_time in enumerate(timestamps):
            
            # New trading day
            if (self.state.current_date is None or 
                current_time.date() != self.state.current_date):
                
                # Close positions at end of day
                if self.state.current_date is not None:
                    final_price = m5_data.loc[current_time]['close']
                    for trade in self.open_positions[:]:
                        self.close_position(trade, current_time, final_price, "end_of_day")
                
                self.state.reset_daily()
                self.state.current_date = current_time.date()
                logger.info(f"New trading day: {current_time.date()}")
            
            # Update high of day and swing highs
            current_bar = m5_data.loc[current_time]
            self.state.high_of_day = max(self.state.high_of_day, current_bar['high'])
            self.update_swing_highs(current_time)
            
            # Check for route start
            if self.check_route_start(current_time):
                self.state.route_start_established = True
            
            # Trading logic after route start
            if self.state.route_start_established:
                
                # 1. Initial Entry: 2M bar break after route start
                if (len(self.open_positions) == 0 and 
                    self.check_2m_bar_break(current_time, "low")):
                    
                    self.execute_trade(current_time, current_bar['close'], 
                                     "initial", self.base_risk)
                
                # 2. First Add: 5M bar break → full size
                elif (len(self.open_positions) == 1 and 
                      self.open_positions[0].trade_type == "initial" and
                      self.check_5m_bar_break(current_time)):
                    
                    self.execute_trade(current_time, current_bar['close'], 
                                     "add", self.base_risk * 2)
                
                # 3. VWAP break confirmation
                if (current_time.time() >= time(9, 30) and 
                    self.check_vwap_break(current_time)):
                    self.state.vwap_break_confirmed = True
                
                # 4. Pyramid opportunities after VWAP break
                if (self.state.vwap_break_confirmed and
                    len(self.open_positions) > 0 and
                    len(self.open_positions) < 4):  # Max 4 positions
                    
                    # Check for dev band hits on 5M or 2M
                    if (self.check_dev_band_hit(current_time, '5m') or 
                        self.check_dev_band_hit(current_time, '2m')):
                        
                        # Enter on 2M bar break
                        if self.check_2m_bar_break(current_time, "low"):
                            available_risk = self.calculate_available_risk()
                            
                            self.execute_trade(current_time, current_bar['close'], 
                                             "pyramid", available_risk)
                
                # 5. Handle consecutive 5M breaks (up to 3)
                if self.check_5m_bar_break(current_time):
                    if current_bar['high'] > self.state.high_of_day * 0.999:  # New highs
                        self.state.consecutive_5m_breaks += 1
                        
                        if (self.state.consecutive_5m_breaks <= self.max_consecutive_breaks and
                            current_bar['high'] > m5_data.loc[m5_data.index <= current_time]['fast_ema'].iloc[-1]):
                            
                            # Valid for entry if above 9 EMA
                            if len(self.open_positions) < 4:
                                available_risk = self.calculate_available_risk()
                                self.execute_trade(current_time, current_bar['close'], 
                                                 "pyramid", available_risk)
            
            # Manage existing positions
            self.manage_positions(current_time)
            
            # Progress update
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(timestamps)} bars. Open: {len(self.open_positions)}")
        
        logger.info(f"Backtest completed. Total trades: {len(self.trades)}")
        self.generate_results()
        
        return self.trades
    
    def generate_results(self):
        """Generate results summary"""
        if not self.trades:
            logger.info("No trades executed")
            return
        
        closed_trades = [t for t in self.trades if not t.is_open()]
        
        if not closed_trades:
            logger.info("No completed trades")
            return
        
        # Calculate statistics
        total_pnl = sum(t.pnl for t in closed_trades)
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(closed_trades)
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        logger.info("\n" + "="*50)
        logger.info("CORRECTED BACKTEST RESULTS")
        logger.info("="*50)
        logger.info(f"Total Trades: {len(closed_trades)}")
        logger.info(f"Total P&L: ${total_pnl:.2f}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Average Win: ${avg_win:.2f}")
        logger.info(f"Average Loss: ${avg_loss:.2f}")
        logger.info(f"Winning Trades: {len(winning_trades)}")
        logger.info(f"Losing Trades: {len(losing_trades)}")
        
        # Trade type breakdown
        for trade_type in ['initial', 'add', 'pyramid']:
            type_trades = [t for t in closed_trades if t.trade_type == trade_type]
            if type_trades:
                type_pnl = sum(t.pnl for t in type_trades)
                logger.info(f"{trade_type.title()} trades: {len(type_trades)}, P&L: ${type_pnl:.2f}")
        
        logger.info("="*50)


def main():
    """Main execution"""
    config_path = "/Users/michaeldurante/sm-playbook/config/backtests/IBIT_GTZ_Short_v1_Test.json"
    
    try:
        strategy = IBITGTZCorrectedStrategy(config_path)
        trades = strategy.run_backtest()
        return strategy, trades
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    strategy, trades = main()