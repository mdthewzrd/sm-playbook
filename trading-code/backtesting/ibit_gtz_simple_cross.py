#!/usr/bin/env python3

"""
IBIT GTZ Simple Strategy - 5M EMA Cross Only
Clean implementation using just 5-minute 9/20 EMA crossovers
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
    route_start_established: bool = False  # Master flag that enables scalping
    scalping_active: bool = False  # Active scalping window
    date: Optional[datetime] = None
    
    def reset_for_new_day(self, date: datetime):
        """Reset all state variables for new trading day"""
        self.short_bias_active = False
        self.route_start_established = False
        self.scalping_active = False
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
        """Download 1H and 5-minute data"""
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
        
        # Download 5M data for entries/exits
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

    def check_bearish_cross(self, current_idx: int) -> bool:
        """Check for bearish EMA cross (9 below 20)"""
        if current_idx == 0:
            return False
        
        current = self.indicators_5m.iloc[current_idx]
        previous = self.indicators_5m.iloc[current_idx - 1]
        
        # Bearish cross: EMA9 was above EMA20, now below
        cross = (previous['fast_ema'] >= previous['slow_ema'] and 
                current['fast_ema'] < current['slow_ema'])
        
        return cross

    def check_bullish_cross(self, current_idx: int) -> bool:
        """Check for bullish EMA cross (9 above 20)"""
        if current_idx == 0:
            return False
        
        current = self.indicators_5m.iloc[current_idx]
        previous = self.indicators_5m.iloc[current_idx - 1]
        
        # Bullish cross: EMA9 was below EMA20, now above
        cross = (previous['fast_ema'] <= previous['slow_ema'] and 
                current['fast_ema'] > current['slow_ema'])
        
        return cross

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
            
            # Route Start Setup: 5M high hits 1H EMA(9) - ENABLES SCALPING
            if current_5m_bar['high'] > h1_ema9:
                logger.info(f"ROUTE START SETUP at {current_time}: 5M High {current_5m_bar['high']:.2f} > 1H EMA9 {h1_ema9:.2f} - SCALPING ENABLED")
                self.day_state.route_start_established = True
                self.day_state.scalping_active = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking route start setup: {e}")
            return False

    def check_deviation_band_entry(self, current_time: datetime, timeframe: str = "2m") -> Optional[float]:
        """Check for scalping entries on pops into deviation bands"""
        try:
            # Only scalp if Route Start is established and scalping is active
            if not (self.day_state.route_start_established and self.day_state.scalping_active):
                return None
            
            # Skip if already in a trade
            if self.current_trade is not None:
                return None
            
            # Must be during market hours for scalping
            if not self.is_market_hours(current_time):
                return None
            
            # Get the appropriate timeframe data
            if timeframe == "2m":
                indicators = self.indicators_2m
                bar_times = indicators.index[indicators.index <= current_time]
            else:  # 5m
                indicators = self.indicators_5m
                bar_times = indicators.index[indicators.index <= current_time]
            
            if len(bar_times) == 0:
                return None
            
            current_bar = indicators.loc[bar_times[-1]]
            
            # Check for pop into upper deviation band (short entry)
            # Using upper_band_2 (tighter band) for more frequent scalps
            if current_bar['high'] >= current_bar['upper_band_2']:
                logger.info(f"SCALP ENTRY ({timeframe.upper()}) at {current_time}: High {current_bar['high']:.2f} >= Upper Band {current_bar['upper_band_2']:.2f}")
                return current_bar['close']  # Enter at close
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking deviation band entry: {e}")
            return None

    def check_quick_exit(self, current_time: datetime, entry_time: datetime, entry_price: float, current_price: float) -> Optional[str]:
        """Check for quick scalp exits"""
        try:
            # Quick profit target (0.20 point move)
            profit_target = 0.20
            if (entry_price - current_price) >= profit_target:
                return f"profit_target_{profit_target}"
            
            # Quick stop loss (0.15 point move against us)
            stop_loss = 0.15
            if (current_price - entry_price) >= stop_loss:
                return f"stop_loss_{stop_loss}"
            
            # Time-based exit (5 minutes max hold)
            time_diff = (current_time - entry_time).total_seconds() / 60
            if time_diff >= 5:
                return "time_exit_5min"
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking quick exit: {e}")
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
                    self.current_trade.exit_reason = \"overnight_close\"\n                    \n                    logger.info(f\"OVERNIGHT CLOSE at {timestamp}: ${row['close']:.2f}, P&L: ${self.current_trade.pnl:.2f}\")\n                    self.trades.append(self.current_trade)\n                    self.current_trade = None\n                \n                self.day_state.reset_for_new_day(timestamp)\n                current_date = trading_date\n                logger.info(f\"New trading day: {trading_date}\")\n            \n            # Setup logic: Check for Route Start (enables scalping for the day)\n            if not self.day_state.route_start_established:\n                # Check 5M data for Route Start setup\n                setup_times_5m = self.indicators_5m.index[self.indicators_5m.index <= timestamp]\n                if len(setup_times_5m) > 0:\n                    current_5m_bar = self.indicators_5m.loc[setup_times_5m[-1]]\n                    self.check_route_start_setup(timestamp, current_5m_bar)\n            \n            # Entry logic: Scalp deviation band pops (only after Route Start)\n            if self.day_state.scalping_active:\n                entry_price = self.check_deviation_band_entry(timestamp, \"2m\")\n                if entry_price is not None:\n                    self.current_trade = Trade(\n                        entry_time=timestamp,\n                        entry_price=entry_price,\n                        direction=\"short\"\n                    )\n                    logger.info(f\"SCALP SHORT ENTRY at {timestamp}: ${entry_price:.2f}\")\n            \n            # Exit logic: Quick scalp exits\n            if self.current_trade is not None:\n                exit_reason = self.check_quick_exit(\n                    timestamp, \n                    self.current_trade.entry_time, \n                    self.current_trade.entry_price, \n                    row['close']\n                )\n                \n                if exit_reason is not None:\n                    self.current_trade.exit_time = timestamp\n                    self.current_trade.exit_price = row['close']\n                    self.current_trade.pnl = (self.current_trade.entry_price - self.current_trade.exit_price) * self.current_trade.size\n                    self.current_trade.exit_reason = exit_reason\n                    \n                    logger.info(f\"SCALP EXIT at {timestamp}: ${row['close']:.2f}, P&L: ${self.current_trade.pnl:.2f} ({exit_reason})\")\n                    \n                    self.trades.append(self.current_trade)\n                    self.current_trade = None\n            \n            # End-of-day close: Close all positions at 4 PM\n            if self.current_trade is not None and self.is_end_of_day(timestamp):\n                self.current_trade.exit_time = timestamp\n                self.current_trade.exit_price = row['close']\n                self.current_trade.pnl = (self.current_trade.entry_price - self.current_trade.exit_price) * self.current_trade.size\n                self.current_trade.exit_reason = \"end_of_day_close\"\n                \n                logger.info(f\"END OF DAY CLOSE at {timestamp}: ${row['close']:.2f}, P&L: ${self.current_trade.pnl:.2f}\")\n                \n                self.trades.append(self.current_trade)\n                self.current_trade = None\n                \n                # Disable scalping until next Route Start\n                self.day_state.scalping_active = False"}
        
        # Close any remaining open trade at end
        if self.current_trade is not None:
            final_row = self.indicators_5m.iloc[-1]
            final_timestamp = self.indicators_5m.index[-1]
            
            self.current_trade.exit_time = final_timestamp
            self.current_trade.exit_price = final_row['close']
            self.current_trade.pnl = (self.current_trade.entry_price - self.current_trade.exit_price) * self.current_trade.size
            self.current_trade.exit_reason = "end_of_backtest"
            
            logger.info(f"Final trade closed at {final_timestamp}: ${final_row['close']:.2f}, P&L: ${self.current_trade.pnl:.2f}")
            
            self.trades.append(self.current_trade)
            self.current_trade = None
        
        return self.generate_performance_summary()

    def generate_performance_summary(self):
        """Generate performance summary"""
        if not self.trades:
            logger.warning("No trades executed")
            return
        
        total_trades = len(self.trades)
        total_pnl = sum(t.pnl for t in self.trades)
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        logger.info("="*60)
        logger.info("SIMPLE EMA CROSS STRATEGY PERFORMANCE")
        logger.info("="*60)
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Total P&L: ${total_pnl:.2f}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Average Win: ${avg_win:.2f}")
        logger.info(f"Average Loss: ${avg_loss:.2f}")
        logger.info(f"Winning Trades: {len(winning_trades)}")
        logger.info(f"Losing Trades: {len(losing_trades)}")
        
        # Show individual trades
        logger.info("\nIndividual Trades:")
        for i, trade in enumerate(self.trades):
            logger.info(f"Trade {i+1}: {trade.entry_time.strftime('%m/%d %H:%M')} -> "
                       f"{trade.exit_time.strftime('%m/%d %H:%M')}, "
                       f"Entry: ${trade.entry_price:.2f}, Exit: ${trade.exit_price:.2f}, "
                       f"P&L: ${trade.pnl:.2f}")
        
        logger.info("="*60)
        
        return self.trades

    def generate_chart(self):
        """Generate simple chart with EMA crosses and trades"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Use 5M data for chart
        chart_data = self.indicators_5m.copy()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), 
                                       gridspec_kw={'height_ratios': [3, 1]}, 
                                       facecolor='black')
        
        fig.suptitle('IBIT Simple EMA Cross Strategy - 5 Minute Chart\n2025-08-13 to 2025-08-27', 
                    fontsize=16, color='white', y=0.98)
        
        # Main price chart
        ax1.set_facecolor('black')
        ax1.grid(True, alpha=0.3, color='gray')
        
        # Plot candlesticks manually
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
            
            if body_height > 0.001:
                ax1.add_patch(Rectangle((x - 0.4, body_bottom), 0.8, body_height, 
                                      facecolor=color, edgecolor=color, alpha=0.8))
            else:
                # Doji
                ax1.plot([x - 0.4, x + 0.4], [c, c], color=color, linewidth=1)
        
        # Plot EMAs
        ax1.plot(x_positions, chart_data['fast_ema'], color='cyan', linewidth=2, label='EMA(9)')
        ax1.plot(x_positions, chart_data['slow_ema'], color='orange', linewidth=2, label='EMA(20)')
        
        # Fill EMA cloud
        ax1.fill_between(x_positions, chart_data['fast_ema'], chart_data['slow_ema'], 
                        alpha=0.2, color='blue', label='EMA Cloud')
        
        # Add market hours shading
        for i, timestamp in enumerate(times):
            time_only = timestamp.time()
            # Pre-market (4 AM - 9:30 AM) and post-market (4 PM - 8 PM)
            if not (time(9, 30) <= time_only <= time(16, 0)):
                ax1.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='gray')
        
        # Plot trade markers
        entry_markers = []
        exit_markers = []
        
        for trade in self.trades:
            # Find entry and exit indices
            entry_idx = None
            exit_idx = None
            
            for i, timestamp in enumerate(times):
                if timestamp >= trade.entry_time and entry_idx is None:
                    entry_idx = i
                if trade.exit_time and timestamp >= trade.exit_time and exit_idx is None:
                    exit_idx = i
                    break
            
            if entry_idx is not None:
                entry_markers.append((entry_idx, trade.entry_price, f"${trade.pnl:.0f}"))
            
            if exit_idx is not None:
                exit_markers.append((exit_idx, trade.exit_price, f"${trade.pnl:.0f}"))
        
        # Plot entry markers (red arrows down for short entries)
        if entry_markers:
            x_vals, y_vals, labels = zip(*entry_markers)
            ax1.scatter(x_vals, y_vals, color='red', marker='v', s=80, alpha=0.9, label='Short Entry')
            
            # Add P&L annotations for significant trades
            for x, y, label in entry_markers:
                pnl_value = float(label.replace('$', ''))
                if abs(pnl_value) > 2000:  # Only annotate bigger trades
                    ax1.annotate(label, (x, y), xytext=(5, -15), 
                               textcoords='offset points', fontsize=9, 
                               color='white', alpha=0.9)
        
        # Plot exit markers (green arrows up for exits)
        if exit_markers:
            x_vals, y_vals, labels = zip(*exit_markers)
            colors = ['lime' if float(label.replace('$', '')) > 0 else 'red' for label in labels]
            ax1.scatter(x_vals, y_vals, c=colors, marker='^', s=60, alpha=0.8, label='Exit')
        
        # Format main chart
        ax1.set_ylabel('Price ($)', color='white', fontsize=12)
        ax1.tick_params(colors='white')
        ax1.legend(loc='upper left', facecolor='black', edgecolor='white', 
                  labelcolor='white', fontsize=10)
        
        # Set x-axis labels
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
        volume_colors = ['lime' if c >= o else 'red' for c, o in zip(closes, opens)]
        
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
        chart_path = "/Users/michaeldurante/sm-playbook/trading-code/backtesting/IBIT_Simple_EMA_Cross_Chart.png"
        plt.savefig(chart_path, facecolor='black', edgecolor='none', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Chart saved to: {chart_path}")
        return chart_path


def main():
    """Main execution"""
    config_path = "/Users/michaeldurante/sm-playbook/config/backtests/IBIT_GTZ_Short_v1_Test.json"
    
    try:
        strategy = IBITSimpleCrossStrategy(config_path)
        results = strategy.run_backtest()
        
        # Generate chart
        chart_path = strategy.generate_chart()
        logger.info(f"Simple EMA cross backtest completed. Chart: {chart_path}")
        
        return strategy, results
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    strategy, results = main()