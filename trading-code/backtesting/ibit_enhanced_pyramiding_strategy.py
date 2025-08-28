#!/usr/bin/env python3

"""
Enhanced IBIT Pyramiding Scalping Strategy with TA-Lib and Lingua Indicators

This strategy implements:
- Multiple entry pyramiding on deviation band pops
- Partial covers using multi-timeframe deviation bands  
- Position recycling within sessions
- TA-Lib indicators for enhanced signal quality
- Lingua concepts for optimal entry/exit timing
- Professional backtesting.py framework
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'indicators'))

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from polygon import RESTClient

# Import trading libraries
from backtesting import Backtest, Strategy
import ta  # TA-Lib alternative
from dual_deviation_cloud import DualDeviationCloud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Lingua indicators with absolute import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'indicators'))
try:
    from lingua_indicators import LinguaATRBands, MultiTimeframeMomentum, VolumeProfile
    LINGUA_AVAILABLE = True
except ImportError:
    LINGUA_AVAILABLE = False
    logger.warning("Lingua indicators not available, using basic indicators only")

@dataclass
class Position:
    """Represents a pyramided position"""
    entry_time: datetime
    entry_price: float
    size: int
    entry_type: str  # 'primary', 'pyramid', 'recycle'
    timeframe: str   # '2M', '5M'
    is_covered: bool = False
    cover_price: Optional[float] = None
    cover_time: Optional[datetime] = None
    pnl: float = 0.0

@dataclass
class SessionState:
    """Tracks scalping session state"""
    is_active: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    route_start_time: Optional[datetime] = None
    positions: List[Position] = None
    total_size: int = 0
    max_pyramid_levels: int = 5
    base_size: int = 1000
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = []

class EnhancedPyramidingStrategy:
    """Enhanced pyramiding scalping strategy with multi-timeframe analysis"""
    
    def __init__(self):
        """Initialize the enhanced pyramiding strategy"""
        self.api_key = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
        self.polygon_client = RESTClient(self.api_key)
        
        # Initialize indicators
        self.indicator_2m = DualDeviationCloud({
            'ema_fast_length': 9,
            'ema_slow_length': 20,
            'positive_dev_1': 1.0,
            'positive_dev_2': 0.5,
            'negative_dev_1': 2.0,
            'negative_dev_2': 2.4
        })
        
        self.indicator_5m = DualDeviationCloud({
            'ema_fast_length': 9,
            'ema_slow_length': 20,
            'positive_dev_1': 1.2,
            'positive_dev_2': 0.6,
            'negative_dev_1': 2.2,
            'negative_dev_2': 2.6
        })
        
        # Lingua indicators (if available)
        if LINGUA_AVAILABLE:
            self.atr_bands = LinguaATRBands(period=20, atr_period=14, multiplier=1.5)
            self.mtf_momentum = MultiTimeframeMomentum(short_period=5, medium_period=14, long_period=30)
            self.volume_profile = VolumeProfile(lookback_period=50, price_buckets=20)
        else:
            self.atr_bands = None
            self.mtf_momentum = None
            self.volume_profile = None
        
        # Session tracking
        self.session = SessionState()
        self.daily_reset_done = {}
        
        # Performance tracking
        self.all_positions = []
        self.daily_sessions = {}  # Track sessions by date for charting
        
    def download_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download multi-timeframe data"""
        logger.info(f"Downloading {symbol} data from {start_date} to {end_date}")
        
        timeframes = {
            '1H': 60,
            '5M': 5,
            '2M': 2
        }
        
        data = {}
        
        for tf_name, multiplier in timeframes.items():
            logger.info(f"Downloading {tf_name} data...")
            
            aggs = []
            for agg in self.polygon_client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan='minute',
                from_=start_date,
                to=end_date,
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
            
            df = pd.DataFrame(aggs)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            data[tf_name] = df
            
        logger.info(f"Downloaded {len(data['1H'])} 1H, {len(data['5M'])} 5M, {len(data['2M'])} 2M bars")
        return data
        
    def calculate_all_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculate all indicators for each timeframe"""
        logger.info("Calculating indicators...")
        
        indicators = {}
        
        # Calculate deviation cloud indicators
        indicators['1H'] = self.indicator_2m.calculate(data['1H'])  # Use 2M settings for 1H
        indicators['5M'] = self.indicator_5m.calculate(data['5M'])
        indicators['2M'] = self.indicator_2m.calculate(data['2M'])
        
        # Add TA-Lib indicators
        for tf in ['1H', '5M', '2M']:
            df = data[tf].copy()
            
            # RSI with different periods
            indicators[tf]['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            indicators[tf]['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            indicators[tf]['macd'] = macd.macd()
            indicators[tf]['macd_signal'] = macd.macd_signal()
            indicators[tf]['macd_diff'] = macd.macd_diff()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            indicators[tf]['stoch_k'] = stoch.stoch()
            indicators[tf]['stoch_d'] = stoch.stoch_signal()
            
            # Volume indicators
            indicators[tf]['volume_sma'] = df['volume'].rolling(window=20).mean()
            indicators[tf]['volume_ratio'] = df['volume'] / indicators[tf]['volume_sma']
            
            # ATR for position sizing
            indicators[tf]['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
        # Calculate Lingua indicators on key timeframes (if available)
        if LINGUA_AVAILABLE and self.atr_bands is not None:
            try:
                lingua_2m = self.atr_bands.calculate(data['2M'])
                lingua_5m_momentum = self.mtf_momentum.calculate(data['5M'])
                lingua_2m_volume = self.volume_profile.calculate(data['2M'])
                
                # Merge Lingua indicators
                for col in lingua_2m.columns:
                    indicators['2M'][col] = lingua_2m[col]
                    
                for col in lingua_5m_momentum.columns:
                    indicators['5M'][col] = lingua_5m_momentum[col]
                    
                for col in lingua_2m_volume.columns:
                    indicators['2M'][col] = lingua_2m_volume[col]
            except Exception as e:
                logger.warning(f"Error calculating Lingua indicators: {e}")
        
        logger.info("Completed all indicators")
        return indicators
        
    def check_route_start(self, current_time: datetime, data_1h: pd.DataFrame, 
                         indicators_1h: pd.DataFrame, data_5m: pd.DataFrame) -> bool:
        """Check for Route Start setup with enhanced criteria"""
        # GTZ window check (8 AM - 12 PM)
        if not (time(8, 0) <= current_time.time() <= time(12, 0)):
            return False
            
        # Get current 1H bar data
        h1_times = indicators_1h.index[indicators_1h.index <= current_time]
        if len(h1_times) == 0:
            return False
            
        current_h1 = indicators_1h.loc[h1_times[-1]]
        h1_ema9 = current_h1['fast_ema']
        h1_ema20 = current_h1['slow_ema']
        
        # Must have 1H short bias
        short_bias = h1_ema9 < h1_ema20
        if not short_bias:
            return False
            
        # Get current 5M bar
        m5_times = data_5m.index[data_5m.index <= current_time]
        if len(m5_times) == 0:
            return False
            
        current_5m = data_5m.loc[m5_times[-1]]
        
        # Route Start: 5M high > 1H EMA9
        if current_5m['high'] > h1_ema9:
            logger.info(f"ROUTE START at {current_time}: 5M High {current_5m['high']:.2f} > 1H EMA9 {h1_ema9:.2f}")
            return True
            
        return False
        
    def check_5m_cross(self, current_time: datetime, indicators_5m: pd.DataFrame, 
                      cross_type: str = 'bearish') -> bool:
        """Check for 5M EMA cross with enhanced validation"""
        m5_times = indicators_5m.index[indicators_5m.index <= current_time]
        if len(m5_times) < 2:
            return False
            
        current = indicators_5m.loc[m5_times[-1]]
        previous = indicators_5m.loc[m5_times[-2]]
        
        if cross_type == 'bearish':
            cross = (previous['fast_ema'] >= previous['slow_ema'] and 
                    current['fast_ema'] < current['slow_ema'])
        else:  # bullish
            cross = (previous['fast_ema'] < previous['slow_ema'] and 
                    current['fast_ema'] > current['slow_ema'])
                    
        return cross
        
    def get_entry_signal_strength(self, current_time: datetime, timeframe: str,
                                data: pd.DataFrame, indicators: pd.DataFrame) -> float:
        """Calculate entry signal strength using multiple indicators"""
        times = indicators.index[indicators.index <= current_time]
        if len(times) == 0:
            return 0.0
            
        current_bar = indicators.loc[times[-1]]
        current_data = data.loc[times[-1]]
        
        strength = 0.0
        factors = 0
        
        # Deviation band position strength
        if 'upper_band_1' in current_bar:
            band_range = current_bar['upper_band_1'] - current_bar['lower_band_1']
            if band_range > 0:
                price_position = (current_data['high'] - current_bar['lower_band_1']) / band_range
                if price_position > 0.8:  # Near upper band
                    strength += min(2.0, (price_position - 0.8) * 10)  # Max 2 points
                    factors += 1
        
        # RSI divergence strength
        if 'rsi_14' in current_bar:
            rsi = current_bar['rsi_14']
            if rsi > 70:  # Overbought for shorts
                strength += min(1.5, (rsi - 70) / 20)
                factors += 1
                
        # Volume confirmation
        if 'volume_ratio' in current_bar:
            vol_ratio = current_bar['volume_ratio']
            if vol_ratio > 1.2:  # Above average volume
                strength += min(1.0, (vol_ratio - 1.0) * 2)
                factors += 1
                
        # MACD divergence
        if 'macd_diff' in current_bar and len(times) >= 2:
            prev_bar = indicators.loc[times[-2]]
            if (current_bar['macd_diff'] < 0 and prev_bar['macd_diff'] >= 0):
                strength += 1.0
                factors += 1
                
        # Lingua ATR bands
        if 'lingua_atr_upper' in current_bar:
            if current_data['high'] >= current_bar['lingua_atr_upper']:
                strength += 1.5
                factors += 1
                
        # Stochastic overbought
        if 'stoch_k' in current_bar:
            if current_bar['stoch_k'] > 80:
                strength += 0.5
                factors += 1
                
        return strength / max(1, factors) if factors > 0 else 0.0
        
    def check_5m_break_of_lows_entry(self, current_time: datetime, data_5m: pd.DataFrame, 
                                   indicators_5m: pd.DataFrame) -> Optional[str]:
        """Check for 5M break of lows entry after bearish cross"""
        if not self.session.is_active:
            return None
            
        # Must have at least 2 positions (bearish cross + ema20 pop)
        active_positions = [p for p in self.session.positions if not p.is_covered]
        if len(active_positions) < 2:
            return None
            
        # Get current 5M data
        m5_times = data_5m.index[data_5m.index <= current_time]
        if len(m5_times) < 10:  # Need history to find recent lows
            return None
            
        current_5m = data_5m.loc[m5_times[-1]]
        
        # Look back 20 periods to find recent low after session start
        lookback_start = max(0, len(m5_times) - 20)
        session_start_idx = None
        
        # Find session start in 5M data
        for i, timestamp in enumerate(m5_times):
            if timestamp >= self.session.start_time:
                session_start_idx = i
                break
                
        if session_start_idx is None:
            return None
            
        # Get recent data since session start
        recent_data = data_5m.loc[m5_times[session_start_idx:]]
        if len(recent_data) < 5:
            return None
            
        # Find the recent low (excluding current bar)
        recent_low = recent_data['low'][:-1].min()  # Exclude current bar
        recent_low_price = recent_low
        
        # Check for break of lows entry
        if current_5m['low'] < recent_low_price:
            return 'break_of_lows'
            
        # Check for close below lows entry  
        elif current_5m['close'] < recent_low_price:
            return 'close_below_lows'
            
        return None

    def check_pyramiding_entry(self, current_time: datetime, current_price: float,
                              timeframe: str, data: pd.DataFrame, indicators: pd.DataFrame) -> Optional[str]:
        """Check for pyramiding entry opportunities"""
        if len(self.session.positions) >= self.session.max_pyramid_levels:
            return None
            
        times = indicators.index[indicators.index <= current_time]
        if len(times) == 0:
            return None
            
        current_bar = indicators.loc[times[-1]]
        current_data = data.loc[times[-1]]
        
        entry_type = None
        
        # Primary entry: Deviation band pop
        if 'upper_band_1' in current_bar and current_data['high'] >= current_bar['upper_band_1']:
            entry_type = 'dev_band_1'
            
        # Secondary entry: Upper band 2 for stronger moves
        elif 'upper_band_2' in current_bar and current_data['high'] >= current_bar['upper_band_2']:
            entry_type = 'dev_band_2'
            
        # EMA20 pop backup (prioritize if we have bearish cross entry)
        elif current_data['high'] >= current_bar['slow_ema']:
            # Check if we already have a bearish cross entry
            has_cross_entry = any(p.entry_type == 'bearish_cross' for p in self.session.positions if not p.is_covered)
            if has_cross_entry and len([p for p in self.session.positions if not p.is_covered]) == 1:
                entry_type = 'ema20_pop_pyramid'  # Priority pyramid after cross
            elif len(self.session.positions) == 0:
                entry_type = 'ema20_pop'  # Backup if no positions
            
        # Lingua ATR band entry
        elif 'lingua_atr_upper' in current_bar and current_data['high'] >= current_bar['lingua_atr_upper']:
            entry_type = 'lingua_atr'
            
        # Volume profile resistance entry
        elif 'volume_poc' in current_bar and current_data['high'] >= current_bar['volume_poc']:
            last_position_price = self.session.positions[-1].entry_price if self.session.positions else 0
            if current_price > last_position_price + 0.10:  # Only if price moved up significantly
                entry_type = 'volume_poc'
                
        return entry_type
        
    def check_partial_cover_signals(self, current_time: datetime, position: Position,
                                   current_price: float, data: pd.DataFrame, 
                                   indicators: pd.DataFrame) -> Optional[str]:
        """Check for partial cover opportunities"""
        if position.is_covered:
            return None
            
        times = indicators.index[indicators.index <= current_time]
        if len(times) == 0:
            return None
            
        current_bar = indicators.loc[times[-1]]
        current_data = data.loc[times[-1]]
        
        # Profit threshold check
        profit_per_share = position.entry_price - current_price
        profit_pct = profit_per_share / position.entry_price * 100
        
        # Quick scalp covers (for smaller profits)
        if 'lower_band_1' in current_bar and current_data['low'] <= current_bar['lower_band_1']:
            if profit_pct > 0.1:  # At least 0.1% profit
                return 'lower_band_1'
                
        # Deeper covers for larger profits
        elif 'lower_band_2' in current_bar and current_data['low'] <= current_bar['lower_band_2']:
            if profit_pct > 0.3:  # At least 0.3% profit
                return 'lower_band_2'
                
        # Lingua ATR lower band
        elif 'lingua_atr_lower' in current_bar and current_data['low'] <= current_bar['lingua_atr_lower']:
            if profit_pct > 0.2:
                return 'lingua_atr_lower'
                
        # Volume profile support
        elif 'volume_poc' in current_bar and current_data['low'] <= current_bar['volume_poc']:
            if profit_pct > 0.15:
                return 'volume_poc_support'
                
        # RSI oversold for covers
        elif 'rsi_7' in current_bar and current_bar['rsi_7'] < 30:
            if profit_pct > 0.1:
                return 'rsi_oversold'
                
        return None
        
    def calculate_position_size(self, entry_type: str, current_atr: float, 
                              existing_positions: int) -> int:
        """Calculate dynamic position size based on entry type and risk"""
        base_size = self.session.base_size
        
        # Size modifiers based on entry type
        size_multipliers = {
            'bearish_cross': 1.2,           # Larger for cross entries
            'ema20_pop_pyramid': 1.5,      # Largest for the key pyramid entry
            'break_of_lows': 1.3,           # Large for break of lows
            'close_below_lows': 1.4,       # Larger for close below confirmation
            'dev_band_1': 1.0,              # Standard size
            'dev_band_2': 1.5,              # Larger for stronger signals
            'ema20_pop': 0.8,               # Smaller for backup signals
            'lingua_atr': 1.2,              # Medium size for ATR signals
            'volume_poc': 0.6,              # Smaller for volume levels
        }
        
        multiplier = size_multipliers.get(entry_type, 1.0)
        
        # Reduce size for additional pyramid levels
        pyramid_reduction = 0.8 ** existing_positions
        
        # ATR-based sizing (reduce size in high volatility)
        atr_modifier = max(0.5, min(1.5, 1.0 / (current_atr * 100)))
        
        final_size = int(base_size * multiplier * pyramid_reduction * atr_modifier)
        return max(100, final_size)  # Minimum 100 shares
        
    def execute_entry(self, current_time: datetime, entry_price: float, 
                     entry_type: str, timeframe: str, position_size: int) -> Position:
        """Execute a pyramided entry"""
        pyramid_level = len(self.session.positions)
        position_type = 'primary' if pyramid_level == 0 else 'pyramid'
        
        position = Position(
            entry_time=current_time,
            entry_price=entry_price,
            size=position_size,
            entry_type=entry_type,
            timeframe=timeframe
        )
        
        self.session.positions.append(position)
        self.session.total_size += position_size
        
        logger.info(f"{position_type.upper()} ENTRY #{pyramid_level+1} at {current_time}: "
                   f"${entry_price:.2f}, Size: {position_size}, Type: {entry_type}")
        
        return position
        
    def execute_partial_cover(self, position: Position, current_time: datetime, 
                            cover_price: float, cover_reason: str, 
                            cover_percentage: float = 1.0) -> float:
        """Execute partial or full cover"""
        if position.is_covered:
            return 0.0
            
        cover_size = int(position.size * cover_percentage)
        pnl = (position.entry_price - cover_price) * cover_size
        
        position.is_covered = True
        position.cover_time = current_time
        position.cover_price = cover_price
        position.pnl = pnl
        
        self.session.total_size -= cover_size
        
        logger.info(f"PARTIAL COVER at {current_time}: ${cover_price:.2f}, "
                   f"Size: {cover_size}, P&L: ${pnl:.2f}, Reason: {cover_reason}")
        
        return pnl
        
    def run_backtest(self, symbol: str = "IBIT", start_date: str = "2025-08-13", 
                    end_date: str = "2025-08-27") -> Dict:
        """Run the enhanced pyramiding backtest"""
        logger.info("Starting Enhanced Pyramiding Scalping Strategy Backtest")
        
        # Download data
        data = self.download_data(symbol, start_date, end_date)
        
        # Calculate indicators
        indicators = self.calculate_all_indicators(data)
        
        # Get 2M data for main execution loop
        df_2m = data['2M']
        indicators_2m = indicators['2M']
        
        total_pnl = 0.0
        
        for current_time, row in df_2m.iterrows():
            trading_date = current_time.date()
            
            # Daily reset
            if trading_date not in self.daily_reset_done:
                self.daily_reset_done[trading_date] = True
                self.session = SessionState()
                self.daily_sessions[trading_date] = {'positions': [], 'session_periods': []}
                logger.info(f"New trading day: {trading_date}")
                
            # Skip if outside market hours
            if not (time(4, 0) <= current_time.time() <= time(20, 0)):
                continue
                
            # Route Start check
            if not self.session.route_start_time:
                if self.check_route_start(current_time, data['1H'], indicators['1H'], data['5M']):
                    self.session.route_start_time = current_time
                    self.daily_sessions[trading_date]['route_start_time'] = current_time
                    logger.info(f"ROUTE START ESTABLISHED at {current_time}")
                    
            # Wait for Route Start
            if not self.session.route_start_time:
                continue
                
            # Session start check (5M bearish cross)
            if not self.session.is_active:
                if self.check_5m_cross(current_time, indicators['5M'], 'bearish'):
                    self.session.is_active = True
                    self.session.start_time = current_time
                    self.daily_sessions[trading_date]['session_periods'].append({'start': current_time})
                    
                    # IMMEDIATE ENTRY #1: Short on bearish cross
                    signal_strength = self.get_entry_signal_strength(current_time, '2M', df_2m, indicators_2m)
                    if signal_strength > 0.5:  # Lower threshold for cross entry
                        current_atr = indicators_2m.loc[current_time, 'atr'] if 'atr' in indicators_2m.columns else 0.5
                        cross_entry_size = self.calculate_position_size('bearish_cross', current_atr, 0)
                        cross_position = self.execute_entry(current_time, row['close'], 'bearish_cross', '5M', cross_entry_size)
                        logger.info(f"🔥 BEARISH CROSS ENTRY at {current_time}")
                    
                    logger.info(f"SCALPING SESSION STARTED at {current_time}")
                    
            # Skip if session not active
            if not self.session.is_active:
                continue
                
            # Session end check (5M bullish cross) - COVERS ALL POSITIONS
            if self.check_5m_cross(current_time, indicators['5M'], 'bullish'):
                # Close ALL remaining positions on trend break
                positions_covered = 0
                for position in self.session.positions:
                    if not position.is_covered:
                        pnl = self.execute_partial_cover(position, current_time, row['close'], '5m_bullish_cross_exit')
                        total_pnl += pnl
                        positions_covered += 1
                
                logger.info(f"🔄 5M BULLISH CROSS - TREND BREAK EXIT: Covered {positions_covered} positions at {current_time}")
                        
                # Update session tracking
                if self.daily_sessions[trading_date]['session_periods']:
                    self.daily_sessions[trading_date]['session_periods'][-1]['end'] = current_time
                self.daily_sessions[trading_date]['positions'].extend(self.session.positions)
                        
                self.session.is_active = False
                self.session.end_time = current_time
                logger.info(f"SCALPING SESSION ENDED at {current_time}")
                continue
                
            # Check for partial covers (reduced priority - mainly for risk management)
            for position in self.session.positions:
                if not position.is_covered:
                    cover_signal = self.check_partial_cover_signals(
                        current_time, position, row['close'], df_2m, indicators_2m
                    )
                    # Only take partial covers on extreme moves or major profit levels
                    if cover_signal and cover_signal in ['lower_band_2', 'rsi_oversold'] and (position.entry_price - row['close']) > 0.5:
                        pnl = self.execute_partial_cover(position, current_time, row['close'], cover_signal)
                        total_pnl += pnl
                        
            # Check for 5M break of lows entries first (priority after cross and EMA20 pop)
            break_of_lows_entry = self.check_5m_break_of_lows_entry(current_time, data['5M'], indicators['5M'])
            if break_of_lows_entry:
                # Execute break of lows entry
                current_atr = indicators_2m.loc[current_time, 'atr'] if 'atr' in indicators_2m.columns else 0.5
                position_size = self.calculate_position_size(break_of_lows_entry, current_atr, len(self.session.positions))
                position = self.execute_entry(current_time, row['close'], break_of_lows_entry, '5M', position_size)
                logger.info(f"💥 {break_of_lows_entry.upper().replace('_', ' ')} ENTRY at {current_time}")
                
            # Check for other pyramiding entries
            elif len(self.session.positions) > 0:  # Only after we have initial positions
                entry_type = self.check_pyramiding_entry(current_time, row['close'], '2M', df_2m, indicators_2m)
                if entry_type:
                    # Calculate signal strength
                    signal_strength = self.get_entry_signal_strength(current_time, '2M', df_2m, indicators_2m)
                    
                    if signal_strength > 1.0:  # Minimum signal strength threshold
                        # Calculate position size
                        current_atr = indicators_2m.loc[current_time, 'atr'] if 'atr' in indicators_2m.columns else 0.5
                        position_size = self.calculate_position_size(entry_type, current_atr, len(self.session.positions))
                        
                        # Execute entry
                        position = self.execute_entry(current_time, row['close'], entry_type, '2M', position_size)
                    
        # Close any remaining positions at end of backtest
        for position in self.session.positions:
            if not position.is_covered:
                final_price = df_2m.iloc[-1]['close']
                pnl = self.execute_partial_cover(position, df_2m.index[-1], final_price, 'backtest_end')
                total_pnl += pnl
                
        # Collect all positions for analysis
        self.all_positions = []
        for date in self.daily_reset_done.keys():
            # This is simplified - in a full implementation we'd track daily positions
            pass
            
        # Collect all remaining positions
        for date, session_data in self.daily_sessions.items():
            self.all_positions.extend(session_data.get('positions', []))
        
        # Generate chart
        chart_path = self.create_enhanced_chart(data, indicators)
        
        # Generate performance report
        results = self.generate_performance_report(total_pnl)
        results['chart_path'] = chart_path
        
        logger.info("Enhanced Pyramiding Strategy Backtest Completed")
        return results
        
    def generate_performance_report(self, total_pnl: float) -> Dict:
        """Generate comprehensive performance report"""
        # This is a simplified version - would be much more detailed in practice
        total_trades = len(self.all_positions)
        winning_trades = len([p for p in self.all_positions if p.pnl > 0])
        
        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': winning_trades / max(1, total_trades) * 100,
            'strategy': 'Enhanced Pyramiding Scalping'
        }
        
    def create_enhanced_chart(self, data: Dict[str, pd.DataFrame], indicators: Dict[str, pd.DataFrame]) -> str:
        """Create comprehensive chart showing pyramiding strategy performance"""
        logger.info("Creating Enhanced Pyramiding Strategy Chart...")
        
        # Set up the plot with multiple subplots
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 16), 
                                           gridspec_kw={'height_ratios': [4, 1, 1]})
        
        # Get 2M data for main chart
        df_2m = data['2M']
        indicators_2m = indicators['2M']
        
        # Plot candlesticks on 2M timeframe
        for timestamp, row in df_2m.iterrows():
            color = 'lime' if row['close'] >= row['open'] else 'red'
            # High-low line
            ax1.plot([timestamp, timestamp], [row['low'], row['high']], 
                    color='white', linewidth=0.3, alpha=0.6)
            # Open-close body  
            ax1.plot([timestamp, timestamp], [row['open'], row['close']], 
                    color=color, linewidth=1.5, alpha=0.8)
        
        # Plot EMAs and deviation bands
        ax1.plot(indicators_2m.index, indicators_2m['fast_ema'], 
                color='cyan', label='2M EMA9', linewidth=2, alpha=0.9)
        ax1.plot(indicators_2m.index, indicators_2m['slow_ema'], 
                color='yellow', label='2M EMA20', linewidth=2, alpha=0.9)
        
        # Deviation bands
        ax1.plot(indicators_2m.index, indicators_2m['upper_band_1'], 
                color='orange', label='Upper Band 1', linewidth=1.5, alpha=0.8)
        ax1.plot(indicators_2m.index, indicators_2m['lower_band_1'], 
                color='purple', label='Lower Band 1', linewidth=1.5, alpha=0.8)
        ax1.plot(indicators_2m.index, indicators_2m['upper_band_2'], 
                color='red', label='Upper Band 2', linewidth=1, alpha=0.6)
        ax1.plot(indicators_2m.index, indicators_2m['lower_band_2'], 
                color='blue', label='Lower Band 2', linewidth=1, alpha=0.6)
        
        # Fill between bands
        ax1.fill_between(indicators_2m.index, indicators_2m['upper_band_1'], 
                        indicators_2m['lower_band_1'], alpha=0.1, color='gray')
        
        # Plot all positions from daily sessions
        total_positions = 0
        total_pnl = 0.0
        pyramid_colors = ['red', 'orange', 'yellow', 'pink', 'magenta']
        cover_colors = ['lime', 'green', 'lightgreen', 'darkgreen', 'forestgreen']
        
        for date, session_data in self.daily_sessions.items():
            if 'positions' not in session_data:
                continue
                
            for i, position in enumerate(session_data['positions']):
                if position.is_covered:
                    total_positions += 1
                    total_pnl += position.pnl
                    
                    # Entry marker - different shapes for different types
                    entry_color = pyramid_colors[min(i, len(pyramid_colors)-1)]
                    entry_marker = 'v' if 'dev_band' in position.entry_type else 's'
                    entry_size = 100 + (position.size / 10)  # Scale marker size
                    
                    ax1.scatter(position.entry_time, position.entry_price, 
                              color=entry_color, s=entry_size, marker=entry_marker,
                              zorder=10, edgecolor='white', linewidth=1,
                              alpha=0.9)
                    
                    # Exit marker  
                    cover_color = cover_colors[min(i, len(cover_colors)-1)]
                    ax1.scatter(position.cover_time, position.cover_price,
                              color=cover_color, s=entry_size, marker='^',
                              zorder=10, edgecolor='white', linewidth=1,
                              alpha=0.9)
                    
                    # Draw line between entry and exit
                    line_color = 'lime' if position.pnl > 0 else 'red'
                    ax1.plot([position.entry_time, position.cover_time], 
                            [position.entry_price, position.cover_price],
                            color=line_color, linewidth=2, alpha=0.7)
                    
                    # Add P&L label for significant trades
                    if abs(position.pnl) > 50:  # Only label significant P&Ls
                        mid_time = position.entry_time + (position.cover_time - position.entry_time) / 2
                        mid_price = (position.entry_price + position.cover_price) / 2
                        pnl_text = f"${position.pnl:.0f}"
                        ax1.annotate(pnl_text, xy=(mid_time, mid_price),
                                   xytext=(0, 15), textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.2', 
                                           facecolor=line_color, alpha=0.8),
                                   ha='center', fontweight='bold', fontsize=8)
        
        # Mark Route Starts and Session periods
        df_5m = data['5M'] 
        indicators_5m = indicators['5M']
        
        for date, session_data in self.daily_sessions.items():
            if 'route_start_time' in session_data and session_data['route_start_time']:
                ax1.axvline(session_data['route_start_time'], color='gold', 
                           linestyle='--', linewidth=2, alpha=0.8, label='Route Start')
                
            if 'session_periods' in session_data:
                for period in session_data['session_periods']:
                    if 'end' in period:  # Only plot complete sessions
                        ax1.axvspan(period['start'], period['end'], 
                                   alpha=0.1, color='yellow', label='Active Session')
        
        # RSI subplot
        ax2.plot(indicators_2m.index, indicators_2m['rsi_14'], 
                color='cyan', label='RSI(14)', linewidth=1.5)
        ax2.plot(indicators_2m.index, indicators_2m['rsi_7'], 
                color='orange', label='RSI(7)', linewidth=1.5)
        ax2.axhline(70, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(30, color='lime', linestyle='--', alpha=0.7)
        ax2.axhline(50, color='gray', linestyle='-', alpha=0.5)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper right')
        ax2.set_ylabel('RSI', fontsize=12)
        
        # Volume subplot
        volume_data = df_2m.copy()
        colors = ['lime' if c >= o else 'red' for o, c in zip(volume_data['open'], volume_data['close'])]
        ax3.bar(volume_data.index, volume_data['volume'], color=colors, alpha=0.7, 
                width=pd.Timedelta(minutes=1.5))
        
        # Volume ratio line
        ax3_twin = ax3.twinx()
        ax3_twin.plot(indicators_2m.index, indicators_2m['volume_ratio'], 
                     color='white', linewidth=1, alpha=0.8, label='Vol Ratio')
        ax3_twin.axhline(1.0, color='gray', linestyle='-', alpha=0.5)
        ax3_twin.set_ylabel('Volume Ratio', color='white', fontsize=10)
        
        # Formatting
        ax1.set_title('Enhanced IBIT Pyramiding Scalping Strategy - Multi-Entry Analysis\n' +
                      f'Total Positions: {total_positions} | Total P&L: ${total_pnl:.2f}', 
                      fontsize=16, pad=20, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='upper right', framealpha=0.9, fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax3.set_ylabel('Volume', fontsize=12)
        ax3.set_xlabel('Time (EST)', fontsize=12)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')
        
        # Add strategy summary
        summary_text = (
            f"ENHANCED PYRAMIDING RESULTS:\n"
            f"• Strategy: Multi-Entry Deviation Band Scalping\n"
            f"• Max Pyramid Levels: 5 per session\n" 
            f"• Entry Types: DevBand1(▼), DevBand2(▼), EMA20(■)\n"
            f"• Partial Covers: Lower Bands + RSI Oversold\n"
            f"• Total Positions: {total_positions}\n"
            f"• Total P&L: ${total_pnl:.2f}\n"
            f"• Win Rate: {len([p for session in self.daily_sessions.values() if 'positions' in session for p in session['positions'] if p.pnl > 0]) / max(1, total_positions) * 100:.1f}%"
        )
        
        ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                facecolor='black', alpha=0.8, edgecolor='white'))
        
        plt.tight_layout()
        
        # Save chart
        chart_path = "/Users/michaeldurante/sm-playbook/trading-code/backtesting/IBIT_Enhanced_Pyramiding_Strategy_Chart.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        logger.info(f"Enhanced Pyramiding Chart saved to: {chart_path}")
        return chart_path

if __name__ == "__main__":
    strategy = EnhancedPyramidingStrategy()
    results = strategy.run_backtest()
    
    print("="*60)
    print("ENHANCED PYRAMIDING SCALPING STRATEGY RESULTS")
    print("="*60)
    print(f"Total P&L: ${results['total_pnl']:.2f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Chart saved to: {results['chart_path']}")
    print("="*60)