#!/usr/bin/env python3

"""
IBIT GTZ Route Start Scalping Strategy
Uses Route Start as setup trigger, then scalps deviation band pops on 2M/5M timeframes
Intraday-only trades (close all at 4 PM)
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
from dataclasses import dataclass
from typing import List, Optional
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
    """State management for each trading day"""
    short_bias_active: bool = False
    route_start_established: bool = False  # Route Start occurred (1H setup)
    bearish_cross_triggered: bool = False  # 5M bearish cross occurred
    scalping_session_active: bool = False  # Active scalping session (5M 9 < 20)
    date: Optional[datetime] = None
    
    def reset_for_new_day(self, date: datetime):
        """Reset all state variables for new trading day"""
        self.short_bias_active = False
        self.route_start_established = False
        self.bearish_cross_triggered = False
        self.scalping_session_active = False
        self.date = date

@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    size: int = 1000
    direction: str = "short"  # Always short for this strategy
    pnl: float = 0.0
    exit_reason: str = ""

class IBITScalpingStrategy:
    """
    Route Start Scalping strategy for IBIT
    - Setup: Route Start (price hits 1H EMA9 with short bias) enables scalping for the day
    - Entry: Scalp pops into 2M/5M deviation bands after Route Start
    - Exit: Quick scalps with intraday-only trades (close all at 4 PM)
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
        
        # State management
        self.day_state = DayState()
        
        # Data storage
        self.data_1h = None
        self.data_5m = None
        self.data_2m = None
        self.indicators_1h = None
        self.indicators_5m = None
        self.indicators_2m = None
        
        # Trade tracking
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        
        logger.info("Initialized IBIT Route Start Scalping Strategy")

    def download_data(self):
        """Download 1H, 5-minute and 2-minute data"""
        symbol = self.config['backtest_configuration']['simulation_environment']['symbol']
        start_date = self.config['backtest_configuration']['simulation_environment']['start_date']
        end_date = self.config['backtest_configuration']['simulation_environment']['end_date']
        
        logger.info(f"Downloading {symbol} data from {start_date} to {end_date}")
        
        # Extend date range for extended hours data
        extended_start = (pd.to_datetime(start_date) - timedelta(days=3)).strftime('%Y-%m-%d')
        extended_end = (pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Download 1H data for Route Start
        logger.info("Downloading 1H data...")
        aggs_1h = []
        for agg in self.polygon_client.get_aggs(
            ticker=symbol,
            multiplier=60,
            timespan='minute',
            from_=extended_start,
            to=extended_end,
            adjusted=True,
            sort='asc',
            limit=50000
        ):
            aggs_1h.append({
                'timestamp': pd.Timestamp(agg.timestamp, unit='ms', tz='America/New_York'),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume
            })
        
        # Download 5M data for scalping
        logger.info("Downloading 5M data...")
        aggs_5m = []
        for agg in self.polygon_client.get_aggs(
            ticker=symbol,
            multiplier=5,
            timespan='minute',
            from_=extended_start,
            to=extended_end,
            adjusted=True,
            sort='asc',
            limit=50000
        ):
            aggs_5m.append({
                'timestamp': pd.Timestamp(agg.timestamp, unit='ms', tz='America/New_York'),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume
            })
        
        # Download 2M data for scalping
        logger.info("Downloading 2M data...")
        aggs_2m = []
        for agg in self.polygon_client.get_aggs(
            ticker=symbol,
            multiplier=2,
            timespan='minute',
            from_=extended_start,
            to=extended_end,
            adjusted=True,
            sort='asc',
            limit=50000
        ):
            aggs_2m.append({
                'timestamp': pd.Timestamp(agg.timestamp, unit='ms', tz='America/New_York'),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume
            })
        
        if not aggs_1h or not aggs_5m or not aggs_2m:
            raise ValueError("No data received")
        
        # Convert to DataFrames
        df_1h = pd.DataFrame(aggs_1h)
        df_1h.set_index('timestamp', inplace=True)
        df_1h.sort_index(inplace=True)
        
        df_5m = pd.DataFrame(aggs_5m)
        df_5m.set_index('timestamp', inplace=True)
        df_5m.sort_index(inplace=True)
        
        df_2m = pd.DataFrame(aggs_2m)
        df_2m.set_index('timestamp', inplace=True)
        df_2m.sort_index(inplace=True)
        
        # Filter to requested date range
        start_filter = pd.Timestamp(start_date, tz='America/New_York')
        end_filter = pd.Timestamp(end_date, tz='America/New_York') + timedelta(days=1)
        
        self.data_1h = df_1h[(df_1h.index >= start_filter) & (df_1h.index < end_filter)]
        self.data_5m = df_5m[(df_5m.index >= start_filter) & (df_5m.index < end_filter)]
        self.data_2m = df_2m[(df_2m.index >= start_filter) & (df_2m.index < end_filter)]
        
        logger.info(f"Downloaded {len(self.data_1h)} 1H bars, {len(self.data_5m)} 5M bars, {len(self.data_2m)} 2M bars")

    def calculate_indicators(self):
        """Calculate EMA indicators for all timeframes"""
        logger.info("Calculating indicators...")
        
        # Calculate 1H indicators for Route Start
        self.indicators_1h = self.indicator_calc.calculate(self.data_1h)
        
        # Calculate 5M indicators for scalping
        self.indicators_5m = self.indicator_calc.calculate(self.data_5m)
        
        # Calculate 2M indicators for scalping
        self.indicators_2m = self.indicator_calc.calculate(self.data_2m)
        
        logger.info("Completed all indicators")

    def check_5m_bearish_cross(self, current_time: datetime) -> bool:
        """Check for 5M bearish EMA cross (9 crossing below 20)"""
        try:
            # Get 5M data up to current time
            bar_times = self.indicators_5m.index[self.indicators_5m.index <= current_time]
            if len(bar_times) < 2:
                return False
            
            current_bar = self.indicators_5m.loc[bar_times[-1]]
            previous_bar = self.indicators_5m.loc[bar_times[-2]]
            
            # Bearish cross: EMA9 was above or equal to EMA20, now below
            cross = (previous_bar['fast_ema'] >= previous_bar['slow_ema'] and 
                    current_bar['fast_ema'] < current_bar['slow_ema'])
            
            if cross:
                logger.info(f"5M BEARISH CROSS at {current_time}: EMA9 {current_bar['fast_ema']:.2f} < EMA20 {current_bar['slow_ema']:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking 5M bearish cross: {e}")
            return False

    def check_5m_bullish_cross(self, current_time: datetime) -> bool:
        """Check for 5M bullish EMA cross (9 crossing above 20) - ends scalping session"""
        try:
            # Get 5M data up to current time
            bar_times = self.indicators_5m.index[self.indicators_5m.index <= current_time]
            if len(bar_times) < 2:
                return False
            
            current_bar = self.indicators_5m.loc[bar_times[-1]]
            previous_bar = self.indicators_5m.loc[bar_times[-2]]
            
            # Bullish cross: EMA9 was below EMA20, now above
            cross = (previous_bar['fast_ema'] < previous_bar['slow_ema'] and 
                    current_bar['fast_ema'] > current_bar['slow_ema'])
            
            if cross:
                logger.info(f"5M BULLISH CROSS at {current_time}: EMA9 {current_bar['fast_ema']:.2f} > EMA20 {current_bar['slow_ema']:.2f} - ENDING SCALPING SESSION")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking 5M bullish cross: {e}")
            return False

    def is_5m_bearish(self, current_time: datetime) -> bool:
        """Check if 5M EMA9 is currently below EMA20"""
        try:
            bar_times = self.indicators_5m.index[self.indicators_5m.index <= current_time]
            if len(bar_times) == 0:
                return False
            
            current_bar = self.indicators_5m.loc[bar_times[-1]]
            return current_bar['fast_ema'] < current_bar['slow_ema']
            
        except Exception as e:
            logger.error(f"Error checking 5M bearish state: {e}")
            return False

    def check_route_start_setup(self, current_time: datetime, current_5m_bar) -> bool:
        """Check if Route Start occurred to enable scalping for the day"""
        try:
            # Must be in GTZ window (8 AM - 12 PM EST)
            current_time_only = current_time.time()
            if not (time(8, 0) <= current_time_only <= time(12, 0)):
                return False
            
            # Skip if Route Start already established for the day
            if self.day_state.route_start_established:
                return False
            
            # Get current 1H EMA(9) level
            h1_times = self.indicators_1h.index[self.indicators_1h.index <= current_time]
            if len(h1_times) == 0:
                return False
            
            current_h1 = self.indicators_1h.loc[h1_times[-1]]
            h1_ema9 = current_h1['fast_ema']
            
            # Check short bias (EMA9 < EMA20 on 1H)
            short_bias = current_h1['fast_ema'] < current_h1['slow_ema']
            if not short_bias:
                return False
            
            # Route Start Setup: 5M high hits 1H EMA(9) - ESTABLISHES ROUTE START
            if current_5m_bar['high'] > h1_ema9:
                logger.info(f"ROUTE START SETUP at {current_time}: 5M High {current_5m_bar['high']:.2f} > 1H EMA9 {h1_ema9:.2f} - ROUTE START ESTABLISHED")
                self.day_state.route_start_established = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking route start setup: {e}")
            return False

    def check_5m_break_of_lows(self, current_time: datetime) -> bool:
        """Check if 5M has broken previous lows (confirmation for entry)"""
        try:
            # Get recent 5M data
            bar_times = self.indicators_5m.index[self.indicators_5m.index <= current_time]
            if len(bar_times) < 3:  # Need at least 3 bars for comparison
                return False
            
            current_bar = self.indicators_5m.loc[bar_times[-1]]
            previous_bar = self.indicators_5m.loc[bar_times[-2]]
            
            # Check if current 5M low broke the previous 5M low
            return current_bar['low'] < previous_bar['low']
            
        except Exception as e:
            logger.error(f"Error checking 5M break of lows: {e}")
            return False

    def check_scalping_entry(self, current_time: datetime) -> Optional[float]:
        """Check for scalping entries: deviation bands OR EMA20 pops with 5M confirmation"""
        try:
            # Only scalp during active scalping sessions
            if not self.day_state.scalping_session_active:
                return None
            
            # Skip if already in a trade
            if self.current_trade is not None:
                return None
            
            # Must be during market hours for scalping
            if not self.is_market_hours(current_time):
                return None
            
            # Must have 5M break of previous lows for confirmation
            if not self.check_5m_break_of_lows(current_time):
                return None
            
            # Get 2M data
            bar_times = self.indicators_2m.index[self.indicators_2m.index <= current_time]
            if len(bar_times) == 0:
                return None
            
            current_bar = self.indicators_2m.loc[bar_times[-1]]
            
            # Primary Entry: Short when price pops into upper_band_1 (9/20 upper dev band)
            if current_bar['high'] >= current_bar['upper_band_1']:
                logger.info(f"DEV BAND SCALP ENTRY at {current_time}: High {current_bar['high']:.2f} >= Upper Band 1 {current_bar['upper_band_1']:.2f}, 5M broke lows")
                return current_bar['close']
            
            # Backup Entry: Short when price pops into 2M EMA20 (for multiple pushes)
            elif current_bar['high'] >= current_bar['slow_ema']:
                logger.info(f"EMA20 POP SCALP ENTRY at {current_time}: High {current_bar['high']:.2f} >= 2M EMA20 {current_bar['slow_ema']:.2f}, 5M broke lows")
                return current_bar['close']
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking scalping entry: {e}")
            return None

    def check_band_scalp_exit(self, current_time: datetime) -> Optional[str]:
        """Check for band scalp exits at lower deviation band"""
        try:
            # Only check exit if we have an open trade
            if self.current_trade is None:
                return None
            
            # Get 2M data
            bar_times = self.indicators_2m.index[self.indicators_2m.index <= current_time]
            if len(bar_times) == 0:
                return None
            
            current_bar = self.indicators_2m.loc[bar_times[-1]]
            
            # Exit: Cover when price hits lower_band_1 (9/20 lower dev band)
            if current_bar['low'] <= current_bar['lower_band_1']:
                logger.info(f"BAND SCALP EXIT at {current_time}: Low {current_bar['low']:.2f} <= Lower Band 1 {current_bar['lower_band_1']:.2f}")
                return "lower_band_cover"
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking band scalp exit: {e}")
            return None

    def is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during regular market hours (9:30 AM - 4 PM EST)"""
        time_only = timestamp.time()
        return time(9, 30) <= time_only <= time(16, 0)
    
    def is_end_of_day(self, timestamp: datetime) -> bool:
        """Check if it's 4 PM (close all positions)"""
        return timestamp.time() >= time(16, 0)

    def run_backtest(self):
        """Run the Route Start Scalping strategy backtest"""
        logger.info("Starting Route Start Scalping Strategy Backtest")
        
        # Download and prepare data
        self.download_data()
        self.calculate_indicators()
        
        current_date = None
        
        # Main backtest loop - use 2M timeframe for higher frequency scalping
        for i, (timestamp, row) in enumerate(self.indicators_2m.iterrows()):
            # Skip if not enough data
            if i == 0:
                continue
            
            # Check for new trading day (reset state at 4 AM)
            trading_date = timestamp.date()
            if (timestamp.time() == time(4, 0) or 
                (trading_date != current_date and timestamp.time() >= time(4, 0))):
                
                # Close any overnight position at day start
                if self.current_trade is not None:
                    self.current_trade.exit_time = timestamp
                    self.current_trade.exit_price = row['close']
                    self.current_trade.pnl = (self.current_trade.entry_price - self.current_trade.exit_price) * self.current_trade.size
                    self.current_trade.exit_reason = "overnight_close"
                    
                    logger.info(f"OVERNIGHT CLOSE at {timestamp}: ${row['close']:.2f}, P&L: ${self.current_trade.pnl:.2f}")
                    self.trades.append(self.current_trade)
                    self.current_trade = None
                
                self.day_state.reset_for_new_day(timestamp)
                current_date = trading_date
                logger.info(f"New trading day: {trading_date}")
            
            # Step 1: Check for Route Start setup (1H setup)
            if not self.day_state.route_start_established:
                setup_times_5m = self.indicators_5m.index[self.indicators_5m.index <= timestamp]
                if len(setup_times_5m) > 0:
                    current_5m_bar = self.indicators_5m.loc[setup_times_5m[-1]]
                    self.check_route_start_setup(timestamp, current_5m_bar)
            
            # Step 2: Check for 5M bearish cross (starts scalping session)
            if (self.day_state.route_start_established and 
                not self.day_state.bearish_cross_triggered and 
                self.check_5m_bearish_cross(timestamp)):
                self.day_state.bearish_cross_triggered = True
                self.day_state.scalping_session_active = True
                logger.info(f"SCALPING SESSION STARTED at {timestamp}")
            
            # Step 3: Check for 5M bullish cross (ends scalping session)  
            if (self.day_state.scalping_session_active and 
                self.check_5m_bullish_cross(timestamp)):
                self.day_state.scalping_session_active = False
                logger.info(f"SCALPING SESSION ENDED at {timestamp}")
                
                # Close any open trade when session ends
                if self.current_trade is not None:
                    self.current_trade.exit_time = timestamp
                    self.current_trade.exit_price = row['close']
                    self.current_trade.pnl = (self.current_trade.entry_price - self.current_trade.exit_price) * self.current_trade.size
                    self.current_trade.exit_reason = "session_end_bullish_cross"
                    
                    logger.info(f"SESSION END CLOSE at {timestamp}: ${row['close']:.2f}, P&L: ${self.current_trade.pnl:.2f}")
                    self.trades.append(self.current_trade)
                    self.current_trade = None
            
            # Step 4: Validate scalping session is still active (5M 9 < 20)
            if (self.day_state.scalping_session_active and 
                not self.is_5m_bearish(timestamp)):
                self.day_state.scalping_session_active = False
                logger.info(f"SCALPING SESSION ENDED at {timestamp} - 5M no longer bearish")
                
                # Close any open trade
                if self.current_trade is not None:
                    self.current_trade.exit_time = timestamp
                    self.current_trade.exit_price = row['close']
                    self.current_trade.pnl = (self.current_trade.entry_price - self.current_trade.exit_price) * self.current_trade.size
                    self.current_trade.exit_reason = "session_end_5m_not_bearish"
                    
                    logger.info(f"SESSION END CLOSE at {timestamp}: ${row['close']:.2f}, P&L: ${self.current_trade.pnl:.2f}")
                    self.trades.append(self.current_trade)
                    self.current_trade = None
            
            # Step 5: Entry logic during active scalping sessions (enhanced)
            if self.day_state.scalping_session_active:
                entry_price = self.check_scalping_entry(timestamp)
                if entry_price is not None:
                    self.current_trade = Trade(
                        entry_time=timestamp,
                        entry_price=entry_price,
                        direction="short"
                    )
                    logger.info(f"ENHANCED SCALP SHORT ENTRY at {timestamp}: ${entry_price:.2f}")
            
            # Step 6: Exit logic - Cover at lower deviation band
            if self.current_trade is not None:
                exit_reason = self.check_band_scalp_exit(timestamp)
                
                if exit_reason is not None:
                    self.current_trade.exit_time = timestamp
                    self.current_trade.exit_price = row['close']
                    self.current_trade.pnl = (self.current_trade.entry_price - self.current_trade.exit_price) * self.current_trade.size
                    self.current_trade.exit_reason = exit_reason
                    
                    logger.info(f"SESSION SCALP COVER at {timestamp}: ${row['close']:.2f}, P&L: ${self.current_trade.pnl:.2f} ({exit_reason})")
                    
                    self.trades.append(self.current_trade)
                    self.current_trade = None
            
            # End-of-day close: Close all positions at 4 PM
            if self.current_trade is not None and self.is_end_of_day(timestamp):
                self.current_trade.exit_time = timestamp
                self.current_trade.exit_price = row['close']
                self.current_trade.pnl = (self.current_trade.entry_price - self.current_trade.exit_price) * self.current_trade.size
                self.current_trade.exit_reason = "end_of_day_close"
                
                logger.info(f"END OF DAY CLOSE at {timestamp}: ${row['close']:.2f}, P&L: ${self.current_trade.pnl:.2f}")
                
                self.trades.append(self.current_trade)
                self.current_trade = None
                
                # End scalping session at market close
                self.day_state.scalping_session_active = False
        
        # Close any remaining open trade at end
        if self.current_trade is not None:
            final_row = self.indicators_2m.iloc[-1]
            final_timestamp = self.indicators_2m.index[-1]
            
            self.current_trade.exit_time = final_timestamp
            self.current_trade.exit_price = final_row['close']
            self.current_trade.pnl = (self.current_trade.entry_price - self.current_trade.exit_price) * self.current_trade.size
            self.current_trade.exit_reason = "end_of_backtest"
            
            logger.info(f"Final trade closed at {final_timestamp}: ${final_row['close']:.2f}, P&L: ${self.current_trade.pnl:.2f}")
            
            self.trades.append(self.current_trade)
            self.current_trade = None
        
        return self.generate_performance_summary()

    def generate_performance_summary(self):
        """Generate performance summary statistics"""
        if not self.trades:
            logger.info("No trades executed")
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        total_pnl = sum(trade.pnl for trade in self.trades)
        winning_trades = [trade for trade in self.trades if trade.pnl > 0]
        losing_trades = [trade for trade in self.trades if trade.pnl <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100
        avg_win = sum(trade.pnl for trade in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(trade.pnl for trade in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Log performance summary
        logger.info("============================================================")
        logger.info("ENHANCED SESSION-BASED SCALPING STRATEGY PERFORMANCE")
        logger.info("============================================================")
        logger.info(f"Total Trades: {len(self.trades)}")
        logger.info(f"Total P&L: ${total_pnl:.2f}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Average Win: ${avg_win:.2f}")
        logger.info(f"Average Loss: ${avg_loss:.2f}")
        logger.info(f"Winning Trades: {len(winning_trades)}")
        logger.info(f"Losing Trades: {len(losing_trades)}")
        logger.info("")
        logger.info("Individual Trades:")
        
        for i, trade in enumerate(self.trades, 1):
            logger.info(f"Trade {i}: {trade.entry_time.strftime('%m/%d %H:%M')} -> {trade.exit_time.strftime('%m/%d %H:%M')}, "
                       f"Entry: ${trade.entry_price:.2f}, Exit: ${trade.exit_price:.2f}, P&L: ${trade.pnl:.2f} ({trade.exit_reason})")
        
        logger.info("============================================================")
        
        return {
            'total_trades': len(self.trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'trades': self.trades
        }

    def create_chart(self):
        """Create visual chart of the strategy"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle
        
        # Set up the plot style
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot 5M candlesticks and indicators
        df_plot = self.indicators_5m.copy()
        
        # Plot candlesticks
        for timestamp, row in df_plot.iterrows():
            color = 'lime' if row['close'] >= row['open'] else 'red'
            ax1.plot([timestamp, timestamp], [row['low'], row['high']], color='white', linewidth=0.8, alpha=0.7)
            ax1.plot([timestamp, timestamp], [row['open'], row['close']], color=color, linewidth=3, alpha=0.8)
        
        # Plot EMAs and deviation bands
        ax1.plot(df_plot.index, df_plot['fast_ema'], color='cyan', label='EMA9', linewidth=1.5, alpha=0.8)
        ax1.plot(df_plot.index, df_plot['slow_ema'], color='yellow', label='EMA20', linewidth=1.5, alpha=0.8)
        ax1.plot(df_plot.index, df_plot['upper_band_1'], color='orange', label='Upper Band 1', linewidth=1, alpha=0.6)
        ax1.plot(df_plot.index, df_plot['upper_band_2'], color='red', label='Upper Band 2', linewidth=1, alpha=0.6)
        
        # Mark trade entries and exits
        for trade in self.trades:
            # Entry marker
            ax1.scatter(trade.entry_time, trade.entry_price, color='red', s=100, marker='v', 
                       label='Entry' if trade == self.trades[0] else "", zorder=5, alpha=0.9)
            
            # Exit marker
            if trade.exit_time:
                color = 'lime' if trade.pnl > 0 else 'red'
                ax1.scatter(trade.exit_time, trade.exit_price, color=color, s=100, marker='^', 
                           label='Exit' if trade == self.trades[0] else "", zorder=5, alpha=0.9)
                
                # Draw line connecting entry to exit
                ax1.plot([trade.entry_time, trade.exit_time], [trade.entry_price, trade.exit_price], 
                        color=color, linewidth=2, alpha=0.7, linestyle='--')
        
        # Highlight market hours
        for date in pd.date_range(df_plot.index[0].date(), df_plot.index[-1].date()):
            market_open = pd.Timestamp.combine(date, time(9, 30)).tz_localize('America/New_York')
            market_close = pd.Timestamp.combine(date, time(16, 0)).tz_localize('America/New_York')
            
            if market_open <= df_plot.index[-1] and market_close >= df_plot.index[0]:
                ax1.axvspan(market_open, market_close, alpha=0.1, color='white', label='Market Hours' if date == pd.date_range(df_plot.index[0].date(), df_plot.index[-1].date())[0] else "")
        
        ax1.set_title('IBIT Session-Based Scalping Strategy - 5 Minute Chart (Extended Hours)\n2025-08-13 to 2025-08-27', 
                     fontsize=14, pad=20)
        ax1.set_ylabel('Price ($)')
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Volume subplot
        volume_data = self.data_5m.copy()
        colors = ['lime' if c >= o else 'red' for o, c in zip(volume_data['open'], volume_data['close'])]
        ax2.bar(volume_data.index, volume_data['volume'], color=colors, alpha=0.7, width=pd.Timedelta(minutes=3))
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date/Time (EST)')
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            
            # Rotate labels
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = "/Users/michaeldurante/sm-playbook/trading-code/backtesting/IBIT_Session_Based_Scalping_Chart.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='black')
        logger.info(f"Chart saved to: {chart_path}")
        
        return chart_path

def main():
    """Main execution function"""
    try:
        # Configuration
        config_path = "/Users/michaeldurante/sm-playbook/config/backtests/IBIT_GTZ_Short_v1_Test.json"
        
        # Initialize and run strategy
        strategy = IBITScalpingStrategy(config_path)
        results = strategy.run_backtest()
        
        # Create chart
        chart_path = strategy.create_chart()
        logger.info(f"Session-based scalping backtest completed. Chart: {chart_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise

if __name__ == "__main__":
    main()