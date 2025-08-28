"""
IBIT GTZ EMA Reversion Short Strategy Backtesting Script

This script implements the complete IBIT_GTZ_EMA_Reversion_Short_v1 strategy
as defined in the strategy specification. It handles multi-timeframe analysis,
state management, pyramiding, and generates comprehensive trade visualization.

Based on specifications:
- Strategy: /.bmad-core/data/strategies/IBIT_GTZ_EMA_Reversion_Short_v1.json
- Indicator: /.bmad-core/data/indicators/DualDeviationCloud.json  
- Config: /config/backtests/IBIT_GTZ_Short_v1_Test.json
"""

import matplotlib
matplotlib.use("Qt5Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
import os
from polygon import RESTClient
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.ticker import FuncFormatter, MaxNLocator
warnings.filterwarnings('ignore')

plt.style.use('dark_background')

# Import our custom indicator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'indicators'))

# Import base indicator first to avoid import issues
from indicators.base_indicator import BaseIndicator
from indicators.dual_deviation_cloud import DualDeviationCloud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StrategyState:
    """Manages the strategy's state variables"""
    short_bias_active: bool = False
    route_start_established: bool = False
    entry_window_open: bool = False
    intraday_high: float = 0.0
    current_date: Optional[datetime] = None
    
    def reset_daily(self):
        """Reset state at end of trading day (18:00 EST)"""
        self.short_bias_active = False
        self.route_start_established = False
        self.entry_window_open = False
        self.intraday_high = 0.0


@dataclass 
class TradeRecord:
    """Represents a single trade execution"""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0.0
    stop_loss: float = 0.0
    trade_type: str = "starter"  # "starter", "main", "pyramid"
    exit_reason: str = ""
    pnl: float = 0.0
    
    def is_open(self) -> bool:
        return self.exit_time is None


class IBITGTZStrategy:
    """
    Implementation of IBIT GTZ EMA Reversion Short Strategy v1
    """
    
    def __init__(self, config_path: str):
        """Initialize strategy with configuration"""
        self.full_config = self._load_config(config_path)
        self.config = self.full_config['simulation_environment']
        self.state = StrategyState()
        self.trades = []
        self.open_positions = []
        self.indicators = {}
        self.data = {}
        self.results = {}
        
        # Strategy parameters
        self.max_pyramids = 4
        self.position_size = 100  # Base position size
        self.setup_start_hour = 8   # 8 AM EST
        self.setup_end_hour = 12    # 12 PM EST
        self.trading_day_end_hour = 18  # 6 PM EST
        
        # Initialize Polygon client
        self.polygon_api_key = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
        self.polygon_client = RESTClient(self.polygon_api_key)
        
        logger.info(f"Initialized IBIT GTZ Short Strategy")
        logger.info(f"Backtest period: {self.config['start_date']} to {self.config['end_date']}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load backtest configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config['backtest_configuration']
    
    def download_data(self) -> Dict[str, pd.DataFrame]:
        """Download multi-timeframe data for IBIT using Polygon API with extended hours"""
        symbol = self.config['symbol']
        start_date = self.config['start_date']
        end_date = self.config['end_date']
        
        logger.info(f"Downloading {symbol} extended hours data from {start_date} to {end_date}")
        
        # Extend date range to ensure we get full extended hours data
        extended_start = (pd.to_datetime(start_date) - timedelta(days=3)).strftime('%Y-%m-%d')
        extended_end = (pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Define timeframe mappings for Polygon API
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
                
                # Get aggregates from Polygon
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
                    # Convert to Eastern Time
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
                    logger.warning(f"No data received for {tf_name} timeframe")
                    continue
                
                # Create DataFrame
                df = pd.DataFrame(aggs)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Filter to extended hours (4 AM - 8 PM ET) and remove weekends
                extended_hours_mask = (
                    (df.index.time >= time(4, 0)) & 
                    (df.index.time <= time(20, 0)) &
                    (df.index.dayofweek < 5)
                )
                df = df[extended_hours_mask]
                
                # Calculate VWAP if not provided - daily reset
                if 'vwap' not in df.columns or df['vwap'].isna().all():
                    df['date'] = df.index.date
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
                    df['cum_vol_price'] = (typical_price * df['volume']).groupby(df['date']).cumsum()
                    df['vwap'] = df['cum_vol_price'] / df['cum_vol']
                    df = df.drop(['date', 'cum_vol', 'cum_vol_price'], axis=1)
                
                data[tf_name] = df
                logger.info(f"Downloaded {len(df)} bars for {tf_name} timeframe (extended hours)")
                
            except Exception as e:
                logger.error(f"Error downloading {tf_name} data: {e}")
                # Try to continue with other timeframes
                continue
        
        if not data:
            raise ValueError("No data could be downloaded from Polygon API")
        
        self.data = data
        return data
    
    def calculate_indicators(self):
        """Calculate all required indicators for each timeframe"""
        logger.info("Calculating indicators for all timeframes")
        
        for timeframe, df in self.data.items():
            logger.info(f"Processing {timeframe} timeframe indicators")
            
            # Initialize DualDeviationCloud indicator
            ddc = DualDeviationCloud({
                'ema_fast_length': 9,
                'ema_slow_length': 20,
                'positive_dev_1': 1.0,
                'positive_dev_2': 0.5,
                'negative_dev_1': 2.0,
                'negative_dev_2': 2.4
            })
            
            # Calculate indicator values
            indicator_data = ddc.calculate(df)
            
            # Add additional EMAs
            indicator_data['ema_72'] = df['close'].ewm(span=72, adjust=False).mean()
            indicator_data['ema_89'] = df['close'].ewm(span=89, adjust=False).mean()
            
            # Store results
            self.indicators[timeframe] = indicator_data
            
            logger.info(f"Completed indicators for {timeframe}")
    
    def is_trading_hours(self, dt: datetime) -> bool:
        """Check if time is within extended trading hours (4:00 AM - 8:00 PM EST)"""
        market_time = dt.time()
        return time(4, 0) <= market_time <= time(20, 0)
    
    def is_setup_window(self, dt: datetime) -> bool:
        """Check if time is within setup identification window (8 AM - 12 PM EST)"""
        market_time = dt.time()
        return time(self.setup_start_hour, 0) <= market_time <= time(self.setup_end_hour, 0)
    
    def check_regime_filter(self, dt: datetime) -> bool:
        """
        Check 1H regime filter: short_bias_active = EMA(9) < EMA(20)
        """
        try:
            # Get 1H data at current time
            h1_data = self.indicators['1h']
            
            # Find the most recent 1H bar
            available_times = h1_data.index[h1_data.index <= dt]
            if len(available_times) == 0:
                return False
                
            current_time = available_times[-1]
            current_bar = h1_data.loc[current_time]
            
            # Check if fast EMA is below slow EMA (bearish bias)
            return current_bar['fast_ema'] < current_bar['slow_ema']
            
        except Exception as e:
            logger.error(f"Error checking regime filter at {dt}: {e}")
            return False
    
    def check_route_start(self, dt: datetime) -> bool:
        """
        Check Route Start condition:
        - short_bias_active is TRUE
        - time between 08:00-12:00 EST  
        - price trades above EMA(9, '1H') (check high of current 5M bar)
        """
        if not self.state.short_bias_active:
            return False
            
        if not self.is_setup_window(dt):
            return False
        
        try:
            # Get current 1H data
            h1_data = self.indicators['1h']
            available_times = h1_data.index[h1_data.index <= dt]
            if len(available_times) == 0:
                return False
                
            current_time = available_times[-1]
            current_bar = h1_data.loc[current_time]
            h1_ema9 = current_bar['fast_ema']
            
            # Get current 5M bar (use high to check if price traded above EMA)
            m5_data = self.indicators['5m']
            m5_times = m5_data.index[m5_data.index <= dt]
            if len(m5_times) == 0:
                return False
                
            current_5m = m5_data.loc[m5_times[-1]]
            
            # Check if the HIGH of current 5M bar traded above 1H EMA(9)
            if current_5m['high'] > h1_ema9:
                # Record session high - use the actual high that exceeded EMA
                session_high = max(self.state.intraday_high, current_5m['high'])
                self.state.intraday_high = session_high
                
                logger.info(f"Route Start: 5M high {current_5m['high']:.2f} > 1H EMA9 {h1_ema9:.2f} at {dt}")
                return True
                
        except Exception as e:
            logger.error(f"Error checking route start at {dt}: {e}")
            
        return False
    
    def check_entry_window_cutoff(self, dt: datetime) -> bool:
        """
        Check if entry window should be closed:
        Price touches lower deviation band on 5M, 15M, or 1H
        """
        try:
            timeframes_to_check = ['5m', '15m', '1h']
            
            for tf in timeframes_to_check:
                if tf not in self.indicators:
                    continue
                    
                tf_data = self.indicators[tf]
                available_times = tf_data.index[tf_data.index <= dt]
                if len(available_times) == 0:
                    continue
                    
                current_bar = tf_data.loc[available_times[-1]]
                
                # Check if low touches or crosses below lower deviation bands
                if (current_bar['low'] <= current_bar['lower_band_1'] or 
                    current_bar['low'] <= current_bar['lower_band_2']):
                    logger.info(f"Entry window cutoff triggered on {tf} at {dt}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error checking entry cutoff at {dt}: {e}")
            
        return False
    
    def check_starter_entry(self, dt: datetime) -> bool:
        """
        Check starter entry condition: break of previous 2M bar's low
        """
        try:
            m2_data = self.indicators['2m']
            available_times = m2_data.index[m2_data.index <= dt]
            
            if len(available_times) < 2:
                return False
                
            current_bar = m2_data.loc[available_times[-1]]
            prev_bar = m2_data.loc[available_times[-2]]
            
            # Check if current low breaks previous bar's low
            return current_bar['low'] < prev_bar['low']
            
        except Exception as e:
            logger.error(f"Error checking starter entry at {dt}: {e}")
            return False
    
    def check_main_entry(self, dt: datetime) -> bool:
        """
        Check main entry condition: break of previous 5M bar's low
        """
        try:
            m5_data = self.indicators['5m']
            available_times = m5_data.index[m5_data.index <= dt]
            
            if len(available_times) < 2:
                return False
                
            current_bar = m5_data.loc[available_times[-1]]
            prev_bar = m5_data.loc[available_times[-2]]
            
            # Check if current low breaks previous bar's low
            return current_bar['low'] < prev_bar['low']
            
        except Exception as e:
            logger.error(f"Error checking main entry at {dt}: {e}")
            return False
    
    def check_pyramiding_condition(self, dt: datetime) -> bool:
        """
        Check if pyramiding is enabled: 5M bar closed below VWAP
        """
        try:
            m5_data = self.indicators['5m']
            available_times = m5_data.index[m5_data.index <= dt]
            
            if len(available_times) == 0:
                return False
                
            current_bar = m5_data.loc[available_times[-1]]
            return current_bar['close'] < current_bar['vwap']
            
        except Exception as e:
            logger.error(f"Error checking pyramiding condition at {dt}: {e}")
            return False
    
    def check_pyramid_entry(self, dt: datetime) -> bool:
        """
        Check pyramid entry: pullback to 5M EMA cloud or 2M deviation bands
        """
        try:
            # Check 5M EMA cloud pullback
            m5_data = self.indicators['5m']
            m5_times = m5_data.index[m5_data.index <= dt]
            if len(m5_times) > 0:
                m5_bar = m5_data.loc[m5_times[-1]]
                
                # Check if price is pulling back to cloud area
                if (m5_bar['cloud_lower_boundary'] <= m5_bar['close'] <= m5_bar['cloud_upper_boundary']):
                    return True
            
            # Check 2M deviation band pullback
            m2_data = self.indicators['2m']
            m2_times = m2_data.index[m2_data.index <= dt]
            if len(m2_times) > 0:
                m2_bar = m2_data.loc[m2_times[-1]]
                
                # Check if price is near deviation bands
                if (m2_bar['upper_band_2'] <= m2_bar['close'] <= m2_bar['upper_band_1']):
                    return True
                    
        except Exception as e:
            logger.error(f"Error checking pyramid entry at {dt}: {e}")
            
        return False
    
    def execute_trade(self, dt: datetime, entry_price: float, trade_type: str) -> TradeRecord:
        """Execute a trade entry with proper stop loss placement"""
        
        # Calculate stop loss - wider stops to avoid getting stopped on minor moves
        if trade_type == "main":
            # Main entries: use intraday high + wider buffer
            stop_loss = max(self.state.intraday_high + 0.25, entry_price + 0.75)  # At least 75 cents
        else:
            # Pyramid entries: wider stops since we're adding to winner
            stop_loss = entry_price + 1.00  # $1.00 above entry for pyramids
        
        trade = TradeRecord(
            entry_time=dt,
            entry_price=entry_price,
            quantity=self.position_size,
            stop_loss=stop_loss,
            trade_type=trade_type
        )
        
        self.trades.append(trade)
        self.open_positions.append(trade)
        
        logger.info(f"Executed {trade_type} SHORT at {dt}: ${entry_price:.2f}, Stop: ${stop_loss:.2f} (+${stop_loss-entry_price:.2f})")
        return trade
    
    def check_primary_target_exit(self, dt: datetime) -> bool:
        """
        Check primary target: price touches 1H Lower Deviation Band
        """
        try:
            h1_data = self.indicators['1h']
            available_times = h1_data.index[h1_data.index <= dt]
            if len(available_times) == 0:
                return False
                
            current_bar = h1_data.loc[available_times[-1]]
            return current_bar['low'] <= current_bar['lower_band_1']
            
        except Exception as e:
            logger.error(f"Error checking primary target at {dt}: {e}")
            return False
    
    def check_trailing_stop_exit(self, dt: datetime) -> bool:
        """
        Check trailing stop: price reclaims 5M 20 EMA AND breaks above recent swing high
        """
        try:
            m5_data = self.indicators['5m']
            available_times = m5_data.index[m5_data.index <= dt]
            if len(available_times) < 5:
                return False
                
            current_bar = m5_data.loc[available_times[-1]]
            
            # Check if price reclaimed 5M 20 EMA (slow_ema)
            if current_bar['close'] <= current_bar['slow_ema']:
                return False
            
            # Check if price broke above recent swing high
            recent_bars = m5_data.loc[available_times[-5:]]
            recent_high = recent_bars['high'].max()
            
            return current_bar['high'] > recent_high
            
        except Exception as e:
            logger.error(f"Error checking trailing stop at {dt}: {e}")
            return False
    
    def manage_positions(self, dt: datetime):
        """Manage open positions - check stops and exits"""
        if not self.open_positions:
            return
            
        try:
            # Get current price data
            m5_data = self.indicators['5m']
            m5_times = m5_data.index[m5_data.index <= dt]
            if len(m5_times) == 0:
                return
                
            current_bar = m5_data.loc[m5_times[-1]]
            current_price = current_bar['close']
            current_high = current_bar['high']  # For stop loss check
            current_low = current_bar['low']    # For target check
            
            positions_to_close = []
            
            for trade in self.open_positions:
                exit_reason = ""
                exit_price = current_price
                
                # Check stop loss - for SHORT positions, stop when price goes ABOVE stop
                if current_high >= trade.stop_loss:
                    exit_reason = "stop_loss"
                    exit_price = min(trade.stop_loss, current_high)  # Exit at stop or high, whichever is lower
                    
                # Check primary target - for shorts, target when price goes lower
                elif self.check_primary_target_exit(dt):
                    exit_reason = "primary_target"
                    exit_price = current_low  # Exit at the low for shorts
                    
                # Check trailing stop mode 
                elif self.check_trailing_stop_exit(dt):
                    exit_reason = "trailing_stop"
                    exit_price = current_price
                
                if exit_reason:
                    positions_to_close.append((trade, exit_price, exit_reason))
            
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
        
        # Calculate P&L (for short position)
        trade.pnl = (trade.entry_price - exit_price) * trade.quantity
        
        # Remove from open positions
        if trade in self.open_positions:
            self.open_positions.remove(trade)
        
        logger.info(f"Closed {trade.trade_type} position at {exit_time}: ${exit_price:.2f}, P&L: ${trade.pnl:.2f}, Reason: {exit_reason}")
    
    def run_backtest(self):
        """Run the complete backtest"""
        logger.info("Starting IBIT GTZ Short Strategy Backtest")
        
        # Download and prepare data
        self.download_data()
        self.calculate_indicators()
        
        # Get all 5-minute timestamps for iteration
        m5_data = self.indicators['5m']
        timestamps = m5_data.index
        
        logger.info(f"Processing {len(timestamps)} 5-minute bars")
        
        for i, current_time in enumerate(timestamps):
            
            # Check if it's a new trading day (reset state at market open each day)
            if (self.state.current_date is None or 
                current_time.date() != self.state.current_date):
                if self.state.current_date is not None:
                    # Close any remaining positions at end of day
                    final_price = m5_data.loc[current_time]['close'] if current_time in m5_data.index else None
                    if final_price:
                        for trade in self.open_positions[:]:
                            self.close_position(trade, current_time, final_price, "end_of_day")
                
                self.state.reset_daily()
                self.state.current_date = current_time.date()
                logger.info(f"New trading day: {current_time.date()}")
            
            # Update current state date
            if self.state.current_date is None:
                self.state.current_date = current_time.date()
            
            # Skip if outside trading hours
            if not self.is_trading_hours(current_time):
                continue
            
            # Step 1: Check regime filter (1H timeframe)
            previous_bias = self.state.short_bias_active
            self.state.short_bias_active = self.check_regime_filter(current_time)
            
            # Debug regime changes
            if previous_bias != self.state.short_bias_active and current_time.time() == time(9, 30):
                try:
                    h1_data = self.indicators['1h']
                    available_times = h1_data.index[h1_data.index <= current_time]
                    if len(available_times) > 0:
                        current_bar = h1_data.loc[available_times[-1]]
                        logger.info(f"Regime on {current_time.date()}: Short Bias = {self.state.short_bias_active}, "
                                   f"EMA9={current_bar['fast_ema']:.2f}, EMA20={current_bar['slow_ema']:.2f}")
                except Exception as e:
                    logger.debug(f"Regime debug error: {e}")
            
            # Step 2: Check for Route Start setup
            if (self.state.short_bias_active and 
                not self.state.route_start_established and
                self.check_route_start(current_time)):
                
                self.state.route_start_established = True
                self.state.entry_window_open = True
                logger.info(f"Route Start established at {current_time}")
            
            # Debug missing route starts at market open
            elif (current_time.time() == time(9, 30) and 
                  current_time.weekday() < 5):  # Weekday
                logger.info(f"Market open {current_time.date()}: Short Bias = {self.state.short_bias_active}, "
                           f"Setup Window = {self.is_setup_window(current_time)}")
            
            # Step 3: Check entry window cutoff
            if (self.state.entry_window_open and 
                self.check_entry_window_cutoff(current_time)):
                self.state.entry_window_open = False
                logger.info(f"Entry window closed at {current_time}")
            
            # Step 4: Simplified Entry Logic - 9/20 EMA breaks with re-entry capability
            if (self.state.route_start_established and 
                self.state.entry_window_open and
                len(self.open_positions) < self.max_pyramids):
                
                current_bar = m5_data.loc[current_time]
                current_price = current_bar['close']
                current_low = current_bar['low']
                
                # Get 5M indicators for EMA levels
                m5_indicators = self.indicators['5m']
                if current_time not in m5_indicators.index:
                    continue
                    
                current_indicators = m5_indicators.loc[current_time]
                ema9 = current_indicators['fast_ema']
                ema20 = current_indicators['slow_ema']
                vwap = current_indicators['vwap']
                
                # Entry condition: price breaks below 9 EMA (for shorts)
                if current_low < ema9:
                    
                    # Always allow entry if no positions (includes re-entry after stop out)
                    if len(self.open_positions) == 0:
                        self.execute_trade(current_time, current_price, "main")
                    
                    # Pyramid entries: additional breaks when already positioned
                    elif len(self.open_positions) > 0 and len(self.open_positions) < self.max_pyramids:
                        
                        # Pyramid on breaks below 20 EMA or significant moves below VWAP
                        if (current_low < ema20 or 
                            (current_price < vwap and current_low < current_price * 0.998)):  # 0.2% break
                            self.execute_trade(current_time, current_price, "pyramid")
            
            # Step 5: Manage existing positions
            self.manage_positions(current_time)
            
            # Progress update every 1000 bars
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(timestamps)} bars. Open positions: {len(self.open_positions)}")
        
        # Close any remaining open positions
        final_price = m5_data.iloc[-1]['close']
        for trade in self.open_positions[:]:
            self.close_position(trade, timestamps[-1], final_price, "backtest_end")
        
        logger.info(f"Backtest completed. Total trades: {len(self.trades)}")
        
        # Generate results
        self.generate_results()
        
        return self.results
    
    def generate_results(self):
        """Generate comprehensive backtest results"""
        if not self.trades:
            self.results = {'error': 'No trades executed'}
            return
        
        closed_trades = [t for t in self.trades if not t.is_open()]
        
        if not closed_trades:
            self.results = {'error': 'No completed trades'}
            return
        
        # Calculate basic statistics
        total_pnl = sum(t.pnl for t in closed_trades)
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades and sum(t.pnl for t in losing_trades) != 0 else float('inf')
        
        # Trade type analysis
        trade_types = {}
        for trade_type in ['starter', 'main', 'pyramid']:
            type_trades = [t for t in closed_trades if t.trade_type == trade_type]
            if type_trades:
                trade_types[trade_type] = {
                    'count': len(type_trades),
                    'total_pnl': sum(t.pnl for t in type_trades),
                    'win_rate': len([t for t in type_trades if t.pnl > 0]) / len(type_trades)
                }
        
        self.results = {
            'total_trades': len(closed_trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'trade_types': trade_types,
            'trades': closed_trades
        }
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("="*50)
        logger.info(f"Total Trades: {len(closed_trades)}")
        logger.info(f"Total P&L: ${total_pnl:.2f}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Average Win: ${avg_win:.2f}")
        logger.info(f"Average Loss: ${avg_loss:.2f}")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Winning Trades: {len(winning_trades)}")
        logger.info(f"Losing Trades: {len(losing_trades)}")
        logger.info("="*50)
    
    def idx_formatter_factory(self, timestamps, fmt):
        """Create formatter for integer indices to timestamp labels"""
        def _fmt(x, pos):
            i = int(round(x))
            if i < 0 or i >= len(timestamps):
                return ""
            ts = timestamps[i]
            return ts.strftime(fmt)
        return _fmt

    def plot_candles_no_gaps(self, ax, df, width=0.8, timefmt='%H:%M', shade_prepost=False):
        """Plot candles using integer indices to remove calendar gaps"""
        if df.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            return None

        x = np.arange(len(df), dtype=float)  # integer positions
        tuples = list(zip(x, df["open"].values, df["high"].values, df["low"].values, df["close"].values))
        candlestick_ohlc(ax, tuples, width=width, colorup="white", colordown="red")

        # Shading pre/post market sessions
        if shade_prepost and len(df) > 1:
            for i in range(len(df)-1):
                t = df.index[i].time()
                # Pre-market: 4:00 AM - 9:30 AM
                if (time(4, 0) <= t < time(9, 30)):
                    ax.axvspan(x[i], x[i+1], color="#444444", alpha=0.6)
                # After hours: 4:00 PM - 8:00 PM  
                if (time(16, 0) <= t < time(20, 0)):
                    ax.axvspan(x[i], x[i+1], color="#333333", alpha=0.5)

        # Format x-axis with timestamps
        ts_list = df.index.to_pydatetime().tolist()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(self.idx_formatter_factory(ts_list, timefmt)))

        # Tight y-limits
        y_max = df["high"].max()
        y_min = df["low"].min()
        ax.set_ylim(y_min * 0.995, y_max * 1.005)

        return x

    def compress_market_time(self, df, target_dates=None):
        """Keep only extended hours data (4 AM - 8 PM) with no gaps"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Filter to extended hours: 4:00 AM - 8:00 PM ET
        extended_hours_mask = (
            (df.index.time >= time(4, 0)) & 
            (df.index.time <= time(20, 0))
        )
        df = df[extended_hours_mask]
        
        # Remove weekends
        df = df[df.index.dayofweek < 5]
        
        # If specific dates provided, filter to those
        if target_dates:
            if isinstance(target_dates, (str, datetime)):
                target_dates = [pd.to_datetime(target_dates).date()]
            elif isinstance(target_dates, list):
                target_dates = [pd.to_datetime(d).date() for d in target_dates]
            
            df = df[df.index.date.isin(target_dates) if hasattr(df.index.date, 'isin') 
                   else df.index.to_series().dt.date.isin(target_dates)]
        
        return df

    def generate_chart(self, output_path: str = None):
        """Generate 5-minute candlestick chart with trade markers using no-gaps plotting"""
        logger.info("Generating execution chart...")
        
        if '5m' not in self.data:
            logger.error("No 5-minute data available for charting")
            return
        
        # Get all trading dates from the backtest period
        start_date = pd.to_datetime(self.config['start_date'])
        end_date = pd.to_datetime(self.config['end_date'])
        
        # Prepare chart data with no gaps (extended hours 4 AM - 8 PM)
        chart_data = self.data['5m'].copy()
        indicators_data = self.indicators['5m'].copy()
        
        # Filter to extended hours and backtest date range
        date_mask = (chart_data.index.date >= start_date.date()) & (chart_data.index.date <= end_date.date())
        extended_hours_mask = (
            (chart_data.index.time >= time(4, 0)) & 
            (chart_data.index.time <= time(20, 0))
        )
        
        combined_mask = date_mask & extended_hours_mask
        chart_data = chart_data[combined_mask]
        indicators_data = indicators_data[combined_mask]
        
        if chart_data.empty:
            logger.error("No chart data after filtering")
            return
        
        # Create figure with subplots
        fig, (ax_price, ax_volume) = plt.subplots(2, 1, figsize=(20, 14), 
                                                  gridspec_kw={'height_ratios': [4, 1]}, 
                                                  sharex=True)
        
        # Plot candlesticks with no gaps
        x_indices = self.plot_candles_no_gaps(ax_price, chart_data, width=0.8, 
                                             timefmt='%m/%d %H:%M', shade_prepost=True)
        
        if x_indices is None:
            logger.error("Failed to plot candlesticks")
            return
        
        # Plot indicators using integer indices
        x = np.arange(len(indicators_data))
        
        # EMAs
        ax_price.plot(x, indicators_data['fast_ema'], 
                     label='EMA(9)', color='cyan', linewidth=2, alpha=0.8)
        ax_price.plot(x, indicators_data['slow_ema'], 
                     label='EMA(20)', color='orange', linewidth=2, alpha=0.8)
        ax_price.plot(x, indicators_data['vwap'], 
                     label='VWAP', color='purple', linewidth=2, linestyle='--', alpha=0.7)
        
        # Deviation bands with fills
        ax_price.fill_between(x, indicators_data['upper_band_1'], indicators_data['upper_band_2'], 
                             alpha=0.2, color='red', label='Upper Bands')
        ax_price.fill_between(x, indicators_data['lower_band_1'], indicators_data['lower_band_2'], 
                             alpha=0.2, color='green', label='Lower Bands')
        
        # EMA Cloud
        ax_price.fill_between(x, indicators_data['cloud_lower_boundary'], indicators_data['cloud_upper_boundary'],
                             alpha=0.15, color='blue', label='EMA Cloud')
        
        # Plot trade markers
        entry_colors = {'starter': 'lime', 'main': 'cyan', 'pyramid': 'orange'}
        exit_colors = {'stop_loss': 'red', 'primary_target': 'green', 
                      'trailing_stop': 'yellow', 'end_of_day': 'gray', 'backtest_end': 'gray'}
        
        for trade in self.trades:
            if not trade.is_open():
                # Find index positions for trade times
                try:
                    entry_idx = chart_data.index.get_loc(trade.entry_time)
                    exit_idx = chart_data.index.get_loc(trade.exit_time)
                    
                    # Entry marker
                    entry_color = entry_colors.get(trade.trade_type, 'white')
                    ax_price.scatter(entry_idx, trade.entry_price, 
                                   color=entry_color, marker='v', s=200, zorder=10,
                                   edgecolors='black', linewidths=2)
                    
                    # Exit marker  
                    exit_color = exit_colors.get(trade.exit_reason, 'white')
                    ax_price.scatter(exit_idx, trade.exit_price,
                                   color=exit_color, marker='^', s=200, zorder=10,
                                   edgecolors='black', linewidths=2)
                    
                    # Connection line
                    ax_price.plot([entry_idx, exit_idx], [trade.entry_price, trade.exit_price],
                                 color='gray', alpha=0.5, linewidth=2, linestyle='-')
                    
                    # P&L annotation
                    mid_idx = (entry_idx + exit_idx) // 2
                    mid_price = (trade.entry_price + trade.exit_price) / 2
                    pnl_text = f"${trade.pnl:.1f}" if abs(trade.pnl) >= 0.1 else f"${trade.pnl:.2f}"
                    ax_price.annotate(pnl_text, (mid_idx, mid_price), 
                                     xytext=(5, 5), textcoords='offset points',
                                     fontsize=8, color='white', alpha=0.8)
                    
                except (KeyError, IndexError):
                    # Trade time not in chart data range
                    continue
        
        # Volume subplot  
        ax_volume.bar(x, chart_data['volume'], width=0.8, alpha=0.6, color='gray')
        ax_volume.set_ylabel('Volume', fontsize=12)
        ax_volume.set_yscale('log')
        
        # Chart formatting
        ax_price.set_title(f'IBIT GTZ Short Strategy - 5 Minute Chart (Extended Hours)\n'
                          f'{self.config["start_date"]} to {self.config["end_date"]}', 
                          fontsize=16, fontweight='bold', color='white')
        ax_price.set_ylabel('Price ($)', fontsize=12)
        
        # Legend
        ax_price.legend(loc='upper left', fontsize=10, framealpha=0.7)
        
        # Grid  
        ax_price.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax_volume.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Tight layout
        plt.tight_layout()
        
        # Save chart
        if output_path is None:
            output_path = f"IBIT_GTZ_Short_Strategy_Chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
        logger.info(f"Chart saved to: {output_path}")
        
        plt.show()


def main():
    """Main execution function"""
    
    # Configuration file path
    config_path = "/Users/michaeldurante/sm-playbook/config/backtests/IBIT_GTZ_Short_v1_Test.json"
    
    try:
        # Initialize strategy
        strategy = IBITGTZStrategy(config_path)
        
        # Run backtest
        results = strategy.run_backtest()
        
        # Generate visualization
        chart_path = "/Users/michaeldurante/sm-playbook/trading-code/backtesting/IBIT_GTZ_Short_Backtest_Chart.png"
        strategy.generate_chart(chart_path)
        
        return strategy, results
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    strategy, results = main()