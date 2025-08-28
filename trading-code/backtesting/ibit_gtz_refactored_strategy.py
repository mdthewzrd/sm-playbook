#!/usr/bin/env python3

"""
IBIT GTZ EMA Reversion Short v1 - REFACTORED IMPLEMENTATION
Critical bug fixes addressing state management and multi-timeframe logic
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'indicators'))

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, time, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from polygon import RESTClient
from dual_deviation_cloud import DualDeviationCloud

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DayState:
    """State management for each trading day - THE MASTER GATEKEEPER"""
    short_bias_active: bool = False
    route_start_established: bool = False  # CRITICAL: Master flag for all entries
    entry_window_open: bool = False
    intraday_high: float = 0.0
    vwap_break_occurred: bool = False  # For pyramiding condition
    primary_target_hit: bool = False  # For trailing stop mode
    date: Optional[datetime] = None
    
    def reset_for_new_day(self, date: datetime):
        """Reset all state variables for new trading day"""
        self.short_bias_active = False
        self.route_start_established = False
        self.entry_window_open = False
        self.intraday_high = 0.0
        self.vwap_break_occurred = False
        self.primary_target_hit = False
        self.date = date
        logger.info(f"State reset for new day: {date.strftime('%Y-%m-%d')}")

@dataclass
class Position:
    """Track individual position details"""
    entry_time: datetime
    entry_price: float
    size: int
    position_type: str  # 'starter', 'main', 'pyramid'
    stop_price: float
    target_price: float
    unrealized_pnl: float = 0.0
    is_closed: bool = False
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl: float = 0.0
    exit_reason: str = ""

class IBITGTZRefactoredStrategy:
    """
    REFACTORED IMPLEMENTATION with proper state management
    Following the exact specification logic
    """
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize data client
        api_key = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
        self.polygon_client = RESTClient(api_key)
        
        # Initialize indicators
        self.indicator_calc = DualDeviationCloud({
            'ema_fast_length': 9,
            'ema_slow_length': 20,
            'positive_dev_1': 1.0,
            'positive_dev_2': 0.5,
            'negative_dev_1': 2.0,
            'negative_dev_2': 2.4
        })
        
        # State management - THE CRITICAL FIX
        self.day_state = DayState()
        
        # Data storage
        self.data = {}
        self.indicators = {}
        
        # Position tracking
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        
        # Risk parameters
        self.base_position_size = 1000  # Base risk per trade
        self.max_pyramids = 4
        
        logger.info("Initialized REFACTORED IBIT GTZ Strategy with proper state management")

    def download_data(self):
        """Download multi-timeframe data"""
        symbol = self.config['backtest_configuration']['simulation_environment']['symbol']
        start_date = self.config['backtest_configuration']['simulation_environment']['start_date']
        end_date = self.config['backtest_configuration']['simulation_environment']['end_date']
        
        logger.info(f"Downloading {symbol} data from {start_date} to {end_date}")
        
        # Extend date range for extended hours data
        extended_start = (pd.to_datetime(start_date) - timedelta(days=3)).strftime('%Y-%m-%d')
        extended_end = (pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Define timeframe mappings for Polygon API
        timeframes = {
            '1h': (60, 'minute'),
            '15m': (15, 'minute'), 
            '5m': (5, 'minute'),
            '2m': (2, 'minute')
        }
        
        for tf_name, (multiplier, timespan) in timeframes.items():
            logger.info(f"Downloading {tf_name} data...")
            
            # Get aggregates from Polygon
            aggs = []
            for agg in self.polygon_client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=extended_start,
                to=extended_end,
                adjusted=True,
                sort='asc',
                limit=50000
            ):
                aggs.append({
                    'timestamp': pd.Timestamp(agg.timestamp, unit='ms', tz='America/New_York'),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                })
            
            if not aggs:
                raise ValueError(f"No data received for {tf_name}")
            
            # Convert to DataFrame
            df = pd.DataFrame(aggs)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Filter to requested date range but keep extended hours
            start_filter = pd.Timestamp(start_date, tz='America/New_York')
            end_filter = pd.Timestamp(end_date, tz='America/New_York') + timedelta(days=1)
            df = df[(df.index >= start_filter) & (df.index < end_filter)]
            
            self.data[tf_name] = df
            logger.info(f"Downloaded {len(df)} bars for {tf_name}")
            
            if tf_name == '5m':
                logger.info(f"5M data range: {df.index[0]} to {df.index[-1]}")

    def calculate_indicators(self):
        """Calculate indicators for all timeframes"""
        logger.info("Calculating indicators for all timeframes")
        
        for timeframe, data in self.data.items():
            logger.info(f"Processing {timeframe} timeframe indicators")
            
            # Calculate basic indicators
            indicators_df = self.indicator_calc.calculate(data)
            
            # Add VWAP calculation
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            volume_price = typical_price * data['volume']
            
            # Session-based VWAP (reset daily at 4 AM EST)
            vwap_values = []
            cumulative_volume_price = 0
            cumulative_volume = 0
            current_date = None
            
            for i, (timestamp, row) in enumerate(data.iterrows()):
                # Reset VWAP at 4 AM each day
                trading_date = timestamp.date()
                if timestamp.time() == time(4, 0) or trading_date != current_date:
                    cumulative_volume_price = 0
                    cumulative_volume = 0
                    current_date = trading_date
                
                cumulative_volume_price += volume_price.iloc[i]
                cumulative_volume += row['volume']
                
                vwap = cumulative_volume_price / cumulative_volume if cumulative_volume > 0 else row['close']
                vwap_values.append(vwap)
            
            indicators_df['vwap'] = vwap_values
            
            # Store indicators
            self.indicators[timeframe] = indicators_df
            logger.info(f"Completed indicators for {timeframe}")

    def check_short_bias(self, current_time: datetime) -> bool:
        """
        CRITICAL: Check 1H EMA regime for short bias
        This is the foundational filter
        """
        try:
            # Get 1H data up to current time
            h1_data = self.indicators['1h']
            h1_times = h1_data.index[h1_data.index <= current_time]
            
            if len(h1_times) == 0:
                return False
            
            current_h1 = h1_data.loc[h1_times[-1]]
            
            # SHORT BIAS: EMA(9) < EMA(20) on 1H
            short_bias = current_h1['fast_ema'] < current_h1['slow_ema']
            
            return short_bias
            
        except Exception as e:
            logger.error(f"Error checking short bias at {current_time}: {e}")
            return False

    def check_route_start(self, current_time: datetime, current_5m_bar: pd.Series) -> bool:
        """
        CRITICAL: Route Start Logic - THE MASTER GATEKEEPER
        Only between 8 AM - 12 PM EST, if price trades above 1H EMA(9)
        """
        try:
            # Must be in GTZ window (8 AM - 12 PM EST)
            current_time_only = current_time.time()
            if not (time(8, 0) <= current_time_only <= time(12, 0)):
                return False
            
            # Must have short bias active
            if not self.day_state.short_bias_active:
                return False
            
            # Get current 1H EMA(9) level
            h1_data = self.indicators['1h']
            h1_times = h1_data.index[h1_data.index <= current_time]
            
            if len(h1_times) == 0:
                return False
            
            current_h1 = h1_data.loc[h1_times[-1]]
            h1_ema9 = current_h1['fast_ema']
            
            # Route Start: Price trades ABOVE 1H EMA(9)
            if current_5m_bar['high'] > h1_ema9:
                self.day_state.route_start_established = True
                self.day_state.entry_window_open = True
                
                # Record intraday high
                self.day_state.intraday_high = max(
                    self.day_state.intraday_high, 
                    current_5m_bar['high']
                )
                
                logger.info(f"ROUTE START ESTABLISHED at {current_time}: "
                           f"5M High {current_5m_bar['high']:.2f} > 1H EMA9 {h1_ema9:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking route start at {current_time}: {e}")
            return False

    def check_entry_window_cutoff(self, current_time: datetime) -> bool:
        """
        Check if entry window should close due to lower deviation band touch
        """
        try:
            for timeframe in ['5m', '15m', '1h']:
                tf_data = self.indicators[timeframe]
                tf_times = tf_data.index[tf_data.index <= current_time]
                
                if len(tf_times) == 0:
                    continue
                
                current_bar = tf_data.loc[tf_times[-1]]
                
                # Check if price touched lower deviation band
                if current_bar['low'] <= current_bar['lower_band_1']:
                    logger.info(f"Entry window closed: {timeframe} lower band touched at {current_time}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking entry window cutoff: {e}")
            return False

    def check_2m_bar_break(self, current_time: datetime) -> Optional[float]:
        """Check for 2M bar break (Starter Entry)"""
        try:
            m2_data = self.indicators['2m']
            m2_times = m2_data.index[m2_data.index <= current_time]
            
            if len(m2_times) < 2:
                return None
            
            current_bar = m2_data.loc[m2_times[-1]]
            prev_bar = m2_data.loc[m2_times[-2]]
            
            # Break of previous 2M bar's low
            if current_bar['low'] < prev_bar['low']:
                return current_bar['close']
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking 2M bar break: {e}")
            return None

    def check_5m_bar_break(self, current_time: datetime) -> Optional[float]:
        """Check for 5M bar break (Main Entry)"""
        try:
            m5_data = self.indicators['5m']
            m5_times = m5_data.index[m5_data.index <= current_time]
            
            if len(m5_times) < 2:
                return None
            
            current_bar = m5_data.loc[m5_times[-1]]
            prev_bar = m5_data.loc[m5_times[-2]]
            
            # Break of previous 5M bar's low
            if current_bar['low'] < prev_bar['low']:
                return current_bar['close']
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking 5M bar break: {e}")
            return None

    def check_vwap_break(self, current_time: datetime) -> bool:
        """Check if 5M bar closed below VWAP (enables pyramiding)"""
        try:
            m5_data = self.indicators['5m']
            m5_times = m5_data.index[m5_data.index <= current_time]
            
            if len(m5_times) == 0:
                return False
            
            current_bar = m5_data.loc[m5_times[-1]]
            
            # 5M close below VWAP
            if current_bar['close'] < current_bar['vwap']:
                if not self.day_state.vwap_break_occurred:
                    self.day_state.vwap_break_occurred = True
                    logger.info(f"VWAP break occurred at {current_time} - Pyramiding enabled")
                return True
            
            return self.day_state.vwap_break_occurred
            
        except Exception as e:
            logger.error(f"Error checking VWAP break: {e}")
            return False

    def check_pyramiding_opportunity(self, current_time: datetime) -> Optional[float]:
        """Check for pyramiding opportunities after VWAP break"""
        try:
            if not self.day_state.vwap_break_occurred:
                return None
            
            # Check 5M pullback to EMA cloud
            m5_data = self.indicators['5m']
            m5_times = m5_data.index[m5_data.index <= current_time]
            
            if len(m5_times) == 0:
                return None
            
            current_5m = m5_data.loc[m5_times[-1]]
            
            # Pullback to 5M EMA cloud (between EMA9 and EMA20)
            ema_cloud_low = min(current_5m['fast_ema'], current_5m['slow_ema'])
            ema_cloud_high = max(current_5m['fast_ema'], current_5m['slow_ema'])
            
            if ema_cloud_low <= current_5m['high'] <= ema_cloud_high:
                return current_5m['close']
            
            # Also check 2M deviation bands
            m2_data = self.indicators['2m']
            m2_times = m2_data.index[m2_data.index <= current_time]
            
            if len(m2_times) > 0:
                current_2m = m2_data.loc[m2_times[-1]]
                if current_2m['low'] <= current_2m['upper_band_1']:
                    return current_2m['close']
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking pyramiding: {e}")
            return None

    def calculate_position_size(self, entry_price: float, stop_price: float) -> int:
        """Calculate position size based on risk"""
        try:
            risk_per_share = abs(stop_price - entry_price)
            if risk_per_share == 0:
                return 0
            
            position_size = int(self.base_position_size / risk_per_share)
            return max(position_size, 1)  # Minimum 1 share
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1

    def execute_trade(self, current_time: datetime, entry_price: float, 
                     trade_type: str, stop_price: float):
        """Execute a trade with proper position sizing"""
        try:
            # Calculate position size
            position_size = self.calculate_position_size(entry_price, stop_price)
            
            # Calculate target (1H lower deviation band)
            h1_data = self.indicators['1h']
            h1_times = h1_data.index[h1_data.index <= current_time]
            
            target_price = entry_price * 0.99  # Default 1% target
            if len(h1_times) > 0:
                current_h1 = h1_data.loc[h1_times[-1]]
                target_price = current_h1['lower_band_1']
            
            # Create position
            position = Position(
                entry_time=current_time,
                entry_price=entry_price,
                size=position_size,
                position_type=trade_type,
                stop_price=stop_price,
                target_price=target_price
            )
            
            self.positions.append(position)
            
            logger.info(f"Executed {trade_type} SHORT at {current_time}: "
                       f"${entry_price:.2f}, Size: {position_size}, "
                       f"Stop: ${stop_price:.2f}, Target: ${target_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")

    def update_positions(self, current_time: datetime, current_bar: pd.Series):
        """Update all open positions"""
        positions_to_close = []
        
        for i, position in enumerate(self.positions):
            if position.is_closed:
                continue
            
            # Update unrealized P&L (short position)
            position.unrealized_pnl = (position.entry_price - current_bar['close']) * position.size
            
            # Check stop loss
            if current_bar['high'] >= position.stop_price:
                positions_to_close.append((i, position.stop_price, "stop_loss"))
                continue
            
            # Check primary target (1H lower deviation band)
            if not self.day_state.primary_target_hit and current_bar['low'] <= position.target_price:
                # Cover 75% at target
                if position.position_type in ['main', 'starter']:
                    positions_to_close.append((i, position.target_price, "primary_target"))
                    self.day_state.primary_target_hit = True
                    logger.info(f"Primary target hit at {current_time}")
                continue
            
            # Check trailing stop mode (after primary target hit)
            if self.day_state.primary_target_hit:
                # Check for 5M EMA20 reclaim and swing high break
                m5_data = self.indicators['5m']
                m5_times = m5_data.index[m5_data.index <= current_time]
                
                if len(m5_times) >= 2:
                    current_5m = m5_data.loc[m5_times[-1]]
                    prev_5m = m5_data.loc[m5_times[-2]]
                    
                    # Trailing stop: 5M close above EMA20 + swing high break
                    if (current_5m['close'] > current_5m['slow_ema'] and 
                        current_5m['high'] > prev_5m['high']):
                        positions_to_close.append((i, current_bar['close'], "trailing_stop"))
        
        # Close positions
        for i, exit_price, reason in positions_to_close:
            self.close_position(i, current_time, exit_price, reason)

    def close_position(self, position_index: int, exit_time: datetime, 
                      exit_price: float, reason: str):
        """Close a position"""
        position = self.positions[position_index]
        position.is_closed = True
        position.exit_time = exit_time
        position.exit_price = exit_price
        position.exit_reason = reason
        
        # Calculate realized P&L (short position)
        position.realized_pnl = (position.entry_price - exit_price) * position.size
        
        self.closed_positions.append(position)
        
        logger.info(f"Closed {position.position_type} at {exit_time}: "
                   f"${exit_price:.2f}, P&L: ${position.realized_pnl:.2f}, "
                   f"Reason: {reason}")

    def run_backtest(self):
        """
        MAIN BACKTEST LOOP with proper state management
        """
        logger.info("Starting REFACTORED IBIT GTZ Strategy Backtest")
        
        # Download and prepare data
        self.download_data()
        self.calculate_indicators()
        
        # Main processing loop on 5M timeframe
        m5_data = self.indicators['5m']
        total_bars = len(m5_data)
        
        logger.info(f"Processing {total_bars} 5-minute bars")
        
        current_date = None
        
        for i, (current_time, current_bar) in enumerate(m5_data.iterrows()):
            # Progress logging
            if i % 500 == 0:
                open_positions = len([p for p in self.positions if not p.is_closed])
                logger.info(f"Processed {i}/{total_bars} bars. Open positions: {open_positions}")
            
            # Check for new trading day (reset state at 4 AM)
            trading_date = current_time.date()
            if (current_time.time() == time(4, 0) or 
                (trading_date != current_date and current_time.time() >= time(4, 0))):
                
                if current_date is not None:
                    # Close all positions at end of day
                    for j, position in enumerate(self.positions):
                        if not position.is_closed:
                            self.close_position(j, current_time, current_bar['close'], "end_of_day")
                
                # Reset state for new day
                self.day_state.reset_for_new_day(current_time)
                current_date = trading_date
                logger.info(f"New trading day: {trading_date}")
            
            # STEP 1: Check short bias (1H EMA regime)
            if not self.day_state.short_bias_active:
                self.day_state.short_bias_active = self.check_short_bias(current_time)
                if self.day_state.short_bias_active:
                    logger.info(f"Short bias activated at {current_time}")
            
            # STEP 2: Check Route Start (THE CRITICAL GATE)
            if (self.day_state.short_bias_active and 
                not self.day_state.route_start_established):
                self.check_route_start(current_time, current_bar)
            
            # STEP 3: Check entry window cutoff
            if (self.day_state.route_start_established and 
                self.day_state.entry_window_open):
                if self.check_entry_window_cutoff(current_time):
                    self.day_state.entry_window_open = False
            
            # STEP 4: Entry logic (ONLY if route_start_established is TRUE)
            if (self.day_state.route_start_established and 
                self.day_state.entry_window_open):
                
                # Calculate stops
                initial_stop = self.day_state.intraday_high + 0.01
                
                # Check for Starter Entry (2M bar break)
                starter_price = self.check_2m_bar_break(current_time)
                if starter_price and len([p for p in self.positions if not p.is_closed and p.position_type == 'starter']) == 0:
                    self.execute_trade(current_time, starter_price, "starter", initial_stop)
                
                # Check for Main Entry (5M bar break)
                main_price = self.check_5m_bar_break(current_time)
                if main_price and len([p for p in self.positions if not p.is_closed and p.position_type == 'main']) == 0:
                    self.execute_trade(current_time, main_price, "main", initial_stop)
            
            # STEP 5: Check VWAP break for pyramiding
            if self.day_state.route_start_established:
                self.check_vwap_break(current_time)
            
            # STEP 6: Pyramiding logic (after VWAP break)
            if (self.day_state.route_start_established and 
                self.day_state.vwap_break_occurred):
                
                pyramid_count = len([p for p in self.positions if not p.is_closed and p.position_type == 'pyramid'])
                if pyramid_count < self.max_pyramids:
                    pyramid_price = self.check_pyramiding_opportunity(current_time)
                    if pyramid_price:
                        # Calculate trailing stop above recent 5M swing high
                        trailing_stop = current_bar['high'] + 0.05
                        self.execute_trade(current_time, pyramid_price, "pyramid", trailing_stop)
            
            # STEP 7: Update all positions
            self.update_positions(current_time, current_bar)
        
        # Final cleanup
        for j, position in enumerate(self.positions):
            if not position.is_closed:
                self.close_position(j, current_time, current_bar['close'], "backtest_end")
        
        return self.generate_performance_summary()

    def generate_performance_summary(self):
        """Generate performance summary"""
        if not self.closed_positions:
            logger.warning("No closed positions to analyze")
            return
        
        total_trades = len(self.closed_positions)
        total_pnl = sum(p.realized_pnl for p in self.closed_positions)
        
        winning_trades = [p for p in self.closed_positions if p.realized_pnl > 0]
        losing_trades = [p for p in self.closed_positions if p.realized_pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = np.mean([p.realized_pnl for p in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([p.realized_pnl for p in losing_trades]) if losing_trades else 0
        
        logger.info("="*60)
        logger.info("REFACTORED STRATEGY PERFORMANCE SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Total P&L: ${total_pnl:.2f}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Average Win: ${avg_win:.2f}")
        logger.info(f"Average Loss: ${avg_loss:.2f}")
        logger.info(f"Winning Trades: {len(winning_trades)}")
        logger.info(f"Losing Trades: {len(losing_trades)}")
        
        # Breakdown by trade type
        for trade_type in ['starter', 'main', 'pyramid']:
            type_trades = [p for p in self.closed_positions if p.position_type == trade_type]
            if type_trades:
                type_pnl = sum(p.realized_pnl for p in type_trades)
                logger.info(f"{trade_type.title()} trades: {len(type_trades)}, P&L: ${type_pnl:.2f}")
        
        logger.info("="*60)
        
        return self.closed_positions

    def generate_chart(self):
        """Generate trading chart with execution markers"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Use 5M data for chart
        chart_data = self.indicators['5m'].copy()
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), 
                                       gridspec_kw={'height_ratios': [3, 1]}, 
                                       facecolor='black')
        
        fig.suptitle('IBIT GTZ Refactored Strategy - 5 Minute Chart (Extended Hours)\n2025-08-13 to 2025-08-27', 
                    fontsize=16, color='white', y=0.98)
        
        # Main price chart
        ax1.set_facecolor('black')
        ax1.grid(True, alpha=0.3, color='gray')
        
        # Plot candlesticks manually (no gaps)
        times = chart_data.index
        opens = chart_data['open'].values
        highs = chart_data['high'].values
        lows = chart_data['low'].values
        closes = chart_data['close'].values
        
        # Create x-axis positions (no gaps)
        x_positions = range(len(times))
        
        # Plot candlesticks
        for i, (x, o, h, l, c) in enumerate(zip(x_positions, opens, highs, lows, closes)):
            color = 'lime' if c >= o else 'red'
            
            # Wick
            ax1.plot([x, x], [l, h], color=color, linewidth=0.5, alpha=0.7)
            
            # Body
            body_height = abs(c - o)
            body_bottom = min(c, o)
            
            if body_height > 0.001:  # Only draw if there's a meaningful body
                ax1.add_patch(Rectangle((x - 0.4, body_bottom), 0.8, body_height, 
                                      facecolor=color, edgecolor=color, alpha=0.8))
            else:
                # Doji - draw a line
                ax1.plot([x - 0.4, x + 0.4], [c, c], color=color, linewidth=1)
        
        # Plot indicators
        ax1.plot(x_positions, chart_data['fast_ema'], color='cyan', linewidth=1, label='EMA(9)', alpha=0.8)
        ax1.plot(x_positions, chart_data['slow_ema'], color='orange', linewidth=1, label='EMA(20)', alpha=0.8)
        ax1.plot(x_positions, chart_data['vwap'], color='yellow', linewidth=1.5, label='VWAP', alpha=0.9)
        
        # Plot deviation bands
        ax1.plot(x_positions, chart_data['upper_band_1'], color='red', linewidth=1, alpha=0.6, label='Upper Bands')
        ax1.plot(x_positions, chart_data['upper_band_2'], color='red', linewidth=1, alpha=0.4)
        ax1.plot(x_positions, chart_data['lower_band_1'], color='lime', linewidth=1, alpha=0.6, label='Lower Bands')
        ax1.plot(x_positions, chart_data['lower_band_2'], color='lime', linewidth=1, alpha=0.4)
        
        # Fill EMA cloud
        ax1.fill_between(x_positions, chart_data['fast_ema'], chart_data['slow_ema'], 
                        alpha=0.1, color='blue', label='EMA Cloud')
        
        # Add pre/post market shading
        for i, timestamp in enumerate(times):
            time_only = timestamp.time()
            # Pre-market (4 AM - 9:30 AM) and post-market (4 PM - 8 PM)
            if (time_only >= pd.to_datetime('04:00').time() and time_only < pd.to_datetime('09:30').time()) or \
               (time_only >= pd.to_datetime('16:00').time() and time_only <= pd.to_datetime('20:00').time()):
                ax1.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='gray')
        
        # Plot trade markers
        entry_markers = {'starter': [], 'main': [], 'pyramid': []}
        exit_markers = []
        
        for position in self.closed_positions:
            # Find entry time in chart data
            entry_idx = None
            for i, timestamp in enumerate(times):
                if timestamp >= position.entry_time:
                    entry_idx = i
                    break
            
            if entry_idx is not None:
                # Entry marker
                trade_type = position.position_type
                if trade_type in entry_markers:
                    entry_markers[trade_type].append((entry_idx, position.entry_price, 
                                                    f"${position.realized_pnl:.0f}"))
            
            # Find exit time
            if position.exit_time:
                exit_idx = None
                for i, timestamp in enumerate(times):
                    if timestamp >= position.exit_time:
                        exit_idx = i
                        break
                
                if exit_idx is not None:
                    exit_markers.append((exit_idx, position.exit_price, 
                                       f"${position.realized_pnl:.0f}"))
        
        # Plot entry markers
        colors = {'starter': 'red', 'main': 'magenta', 'pyramid': 'cyan'}
        for trade_type, markers in entry_markers.items():
            if markers:
                x_vals, y_vals, labels = zip(*markers)
                ax1.scatter(x_vals, y_vals, color=colors[trade_type], 
                          marker='v', s=50, alpha=0.8, label=f'{trade_type.title()} Entry')
                
                # Add P&L annotations for significant trades
                for x, y, label in markers:
                    pnl_value = float(label.replace('$', ''))
                    if abs(pnl_value) > 5000:  # Only annotate big trades
                        ax1.annotate(label, (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=8, 
                                   color='white', alpha=0.9)
        
        # Plot exit markers (green triangles for profits, red for losses)
        if exit_markers:
            for x, y, label in exit_markers:
                pnl_value = float(label.replace('$', ''))
                color = 'lime' if pnl_value > 0 else 'red'
                ax1.scatter([x], [y], color=color, marker='^', s=30, alpha=0.7)
        
        # Format main chart
        ax1.set_ylabel('Price ($)', color='white', fontsize=12)
        ax1.tick_params(colors='white')
        ax1.legend(loc='upper left', facecolor='black', edgecolor='white', 
                  labelcolor='white', fontsize=10)
        
        # Set x-axis labels to show dates
        n_labels = 10
        label_indices = [i * len(x_positions) // n_labels for i in range(n_labels)]
        label_dates = [times[i].strftime('%m/%d %H:%M') for i in label_indices]
        
        ax1.set_xlim(0, len(x_positions))
        ax1.set_xticks(label_indices)
        ax1.set_xticklabels(label_dates, rotation=45, color='white')
        
        # Volume chart
        ax2.set_facecolor('black')
        ax2.grid(True, alpha=0.3, color='gray')
        
        volumes = chart_data['volume'].values
        volume_colors = ['lime' if c >= o else 'red' 
                        for c, o in zip(closes, opens)]
        
        ax2.bar(x_positions, volumes, color=volume_colors, alpha=0.6, width=0.8)
        ax2.set_ylabel('Volume', color='white', fontsize=10)
        ax2.tick_params(colors='white')
        ax2.set_yscale('log')
        ax2.set_xlim(0, len(x_positions))
        
        # Match x-axis with main chart
        ax2.set_xticks(label_indices)
        ax2.set_xticklabels(label_dates, rotation=45, color='white')
        
        # Style the figure
        fig.patch.set_facecolor('black')
        plt.tight_layout()
        
        # Save chart
        chart_path = "/Users/michaeldurante/sm-playbook/trading-code/backtesting/IBIT_GTZ_Refactored_Strategy_Chart.png"
        plt.savefig(chart_path, facecolor='black', edgecolor='none', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Chart saved to: {chart_path}")
        return chart_path


def main():
    """Main execution"""
    config_path = "/Users/michaeldurante/sm-playbook/config/backtests/IBIT_GTZ_Short_v1_Test.json"
    
    try:
        strategy = IBITGTZRefactoredStrategy(config_path)
        results = strategy.run_backtest()
        
        # Generate chart
        chart_path = strategy.generate_chart()
        logger.info(f"Refactored strategy backtest completed. Chart: {chart_path}")
        
        return strategy, results
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    strategy, results = main()