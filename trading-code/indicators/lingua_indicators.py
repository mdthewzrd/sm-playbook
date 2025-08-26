"""
Lingua Custom Indicators for SM Playbook Trading System

Custom indicators specifically designed for the Lingua trading methodology
and BMAT system implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from .base_indicator import BaseIndicator
import logging

logger = logging.getLogger(__name__)


class LinguaATRBands(BaseIndicator):
    """
    Lingua ATR Bands - Modified Bollinger Bands using ATR for volatility.
    
    This indicator creates bands around a moving average using ATR
    instead of standard deviation, providing better volatility adjustment.
    """
    
    def __init__(self, period: int = 20, atr_period: int = 14, multiplier: float = 2.0):
        """
        Initialize Lingua ATR Bands.
        
        Args:
            period: Period for moving average
            atr_period: Period for ATR calculation
            multiplier: ATR multiplier for band width
        """
        params = {
            'period': period,
            'atr_period': atr_period, 
            'multiplier': multiplier,
            'min_periods': max(period, atr_period)
        }
        super().__init__('LinguaATRBands', params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR Bands."""
        if not self.validate_data(data):
            raise ValueError("Invalid data for LinguaATRBands calculation")
        
        period = self.params['period']
        atr_period = self.params['atr_period']
        multiplier = self.params['multiplier']
        
        # Calculate moving average (middle band)
        middle = data['close'].rolling(window=period).mean()
        
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=atr_period).mean()
        
        # Calculate bands
        upper = middle + (atr * multiplier)
        lower = middle - (atr * multiplier)
        
        # Band width and position
        band_width = (upper - lower) / middle * 100
        band_position = (data['close'] - lower) / (upper - lower) * 100
        
        result = pd.DataFrame({
            'lingua_atr_upper': upper,
            'lingua_atr_middle': middle,
            'lingua_atr_lower': lower,
            'lingua_atr_width': band_width,
            'lingua_atr_position': band_position
        }, index=data.index)
        
        self._results = result
        self._is_calculated = True
        logger.debug(f"Calculated {self.name} for {len(data)} periods")
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on ATR Bands."""
        if self._results is None:
            self.calculate(data)
        
        signals = pd.DataFrame(index=data.index)
        
        # Band squeeze detection (low volatility)
        squeeze_threshold = self.params.get('squeeze_threshold', 10)
        signals['squeeze'] = self._results['lingua_atr_width'] < squeeze_threshold
        
        # Breakout signals
        signals['breakout_long'] = (
            (data['close'] > self._results['lingua_atr_upper']) &
            (data['close'].shift() <= self._results['lingua_atr_upper'].shift())
        )
        
        signals['breakout_short'] = (
            (data['close'] < self._results['lingua_atr_lower']) &
            (data['close'].shift() >= self._results['lingua_atr_lower'].shift())
        )
        
        # Mean reversion signals
        signals['mean_revert_long'] = (
            (data['close'] < self._results['lingua_atr_lower']) &
            (data['close'] > data['close'].shift())  # Price starting to recover
        )
        
        signals['mean_revert_short'] = (
            (data['close'] > self._results['lingua_atr_upper']) &
            (data['close'] < data['close'].shift())  # Price starting to decline
        )
        
        # Signal strength based on distance from middle
        signals['signal_strength'] = np.abs(
            self._results['lingua_atr_position'] - 50
        ) / 50
        
        return signals


class RSIGradient(BaseIndicator):
    """
    RSI Gradient - Rate of change of RSI over time.
    
    This indicator measures the momentum of momentum, providing
    early signals for RSI direction changes.
    """
    
    def __init__(self, rsi_period: int = 14, gradient_period: int = 3):
        """
        Initialize RSI Gradient.
        
        Args:
            rsi_period: Period for RSI calculation
            gradient_period: Period for gradient calculation
        """
        params = {
            'rsi_period': rsi_period,
            'gradient_period': gradient_period,
            'min_periods': rsi_period + gradient_period
        }
        super().__init__('RSIGradient', params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI Gradient."""
        if not self.validate_data(data):
            raise ValueError("Invalid data for RSI Gradient calculation")
        
        rsi_period = self.params['rsi_period']
        gradient_period = self.params['gradient_period']
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate gradient (rate of change)
        rsi_gradient = rsi.diff(gradient_period) / gradient_period
        
        # Smooth gradient
        smooth_gradient = rsi_gradient.rolling(window=3).mean()
        
        # Gradient momentum (second derivative)
        gradient_momentum = rsi_gradient.diff()
        
        result = pd.DataFrame({
            'rsi': rsi,
            'rsi_gradient': rsi_gradient,
            'rsi_gradient_smooth': smooth_gradient,
            'rsi_momentum': gradient_momentum
        }, index=data.index)
        
        self._results = result
        self._is_calculated = True
        logger.debug(f"Calculated {self.name} for {len(data)} periods")
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on RSI Gradient."""
        if self._results is None:
            self.calculate(data)
        
        signals = pd.DataFrame(index=data.index)
        
        # Gradient direction changes
        signals['gradient_bullish'] = (
            (self._results['rsi_gradient'] > 0) &
            (self._results['rsi_gradient'].shift() <= 0)
        )
        
        signals['gradient_bearish'] = (
            (self._results['rsi_gradient'] < 0) &
            (self._results['rsi_gradient'].shift() >= 0)
        )
        
        # Strong gradient signals
        gradient_threshold = 1.0
        signals['strong_bullish'] = self._results['rsi_gradient'] > gradient_threshold
        signals['strong_bearish'] = self._results['rsi_gradient'] < -gradient_threshold
        
        # Momentum divergence
        signals['momentum_divergence'] = (
            (self._results['rsi_gradient'] > 0) &
            (self._results['rsi_momentum'] < 0)
        ) | (
            (self._results['rsi_gradient'] < 0) &
            (self._results['rsi_momentum'] > 0)
        )
        
        # Signal strength based on gradient magnitude
        signals['signal_strength'] = np.abs(self._results['rsi_gradient']) / 5.0
        signals['signal_strength'] = np.clip(signals['signal_strength'], 0, 1)
        
        return signals


class MultiTimeframeMomentum(BaseIndicator):
    """
    Multi-Timeframe Momentum - Composite momentum across multiple timeframes.
    
    This indicator aggregates momentum signals from different timeframes
    to provide a comprehensive view of market direction.
    """
    
    def __init__(self, short_period: int = 5, medium_period: int = 14, long_period: int = 30):
        """
        Initialize Multi-Timeframe Momentum.
        
        Args:
            short_period: Short-term momentum period
            medium_period: Medium-term momentum period  
            long_period: Long-term momentum period
        """
        params = {
            'short_period': short_period,
            'medium_period': medium_period,
            'long_period': long_period,
            'min_periods': long_period
        }
        super().__init__('MultiTimeframeMomentum', params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Multi-Timeframe Momentum."""
        if not self.validate_data(data):
            raise ValueError("Invalid data for Multi-Timeframe Momentum calculation")
        
        short = self.params['short_period']
        medium = self.params['medium_period']
        long_period = self.params['long_period']
        
        # Calculate momentum for each timeframe (price change percentage)
        short_momentum = (data['close'] / data['close'].shift(short) - 1) * 100
        medium_momentum = (data['close'] / data['close'].shift(medium) - 1) * 100
        long_momentum = (data['close'] / data['close'].shift(long_period) - 1) * 100
        
        # Normalize momentum values (z-score)
        short_norm = (short_momentum - short_momentum.rolling(50).mean()) / short_momentum.rolling(50).std()
        medium_norm = (medium_momentum - medium_momentum.rolling(50).mean()) / medium_momentum.rolling(50).std()
        long_norm = (long_momentum - long_momentum.rolling(50).mean()) / long_momentum.rolling(50).std()
        
        # Composite momentum (weighted average)
        weights = {'short': 0.5, 'medium': 0.3, 'long': 0.2}
        composite = (
            short_norm * weights['short'] +
            medium_norm * weights['medium'] +
            long_norm * weights['long']
        )
        
        # Momentum alignment score
        alignment = (
            np.sign(short_momentum) + 
            np.sign(medium_momentum) + 
            np.sign(long_momentum)
        ) / 3
        
        result = pd.DataFrame({
            'momentum_short': short_momentum,
            'momentum_medium': medium_momentum,
            'momentum_long': long_momentum,
            'momentum_composite': composite,
            'momentum_alignment': alignment
        }, index=data.index)
        
        self._results = result
        self._is_calculated = True
        logger.debug(f"Calculated {self.name} for {len(data)} periods")
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on Multi-Timeframe Momentum."""
        if self._results is None:
            self.calculate(data)
        
        signals = pd.DataFrame(index=data.index)
        
        # Momentum alignment signals
        signals['aligned_bullish'] = self._results['momentum_alignment'] > 0.5
        signals['aligned_bearish'] = self._results['momentum_alignment'] < -0.5
        
        # Strong momentum signals
        momentum_threshold = 1.0
        signals['strong_momentum_long'] = self._results['momentum_composite'] > momentum_threshold
        signals['strong_momentum_short'] = self._results['momentum_composite'] < -momentum_threshold
        
        # Momentum reversal signals
        signals['momentum_reversal_long'] = (
            (self._results['momentum_composite'] > 0) &
            (self._results['momentum_composite'].shift() <= 0) &
            (self._results['momentum_short'] > self._results['momentum_medium'])
        )
        
        signals['momentum_reversal_short'] = (
            (self._results['momentum_composite'] < 0) &
            (self._results['momentum_composite'].shift() >= 0) &
            (self._results['momentum_short'] < self._results['momentum_medium'])
        )
        
        # Signal strength based on composite momentum magnitude
        signals['signal_strength'] = np.abs(self._results['momentum_composite']) / 2.0
        signals['signal_strength'] = np.clip(signals['signal_strength'], 0, 1)
        
        return signals


class VolumeProfile(BaseIndicator):
    """
    Volume Profile - Distribution of volume across price levels.
    
    This indicator identifies areas of high and low volume activity,
    providing support and resistance levels based on volume.
    """
    
    def __init__(self, lookback_period: int = 50, price_buckets: int = 20):
        """
        Initialize Volume Profile.
        
        Args:
            lookback_period: Period to look back for volume analysis
            price_buckets: Number of price levels to analyze
        """
        params = {
            'lookback_period': lookback_period,
            'price_buckets': price_buckets,
            'min_periods': lookback_period
        }
        super().__init__('VolumeProfile', params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Profile."""
        if not self.validate_data(data):
            raise ValueError("Invalid data for Volume Profile calculation")
        
        lookback = self.params['lookback_period']
        buckets = self.params['price_buckets']
        
        def calculate_volume_profile(window_data):
            """Calculate volume profile for a window of data."""
            if len(window_data) < 2:
                return np.nan, np.nan, np.nan
                
            price_min = window_data['low'].min()
            price_max = window_data['high'].max()
            
            if price_max == price_min:
                return price_min, 0, 0
                
            # Create price buckets
            price_levels = np.linspace(price_min, price_max, buckets + 1)
            volume_at_price = np.zeros(buckets)
            
            # Distribute volume across price levels
            for _, row in window_data.iterrows():
                # Assume volume is distributed evenly across the bar's range
                bar_range = row['high'] - row['low']
                if bar_range > 0:
                    # Find which buckets this bar spans
                    start_bucket = np.digitize(row['low'], price_levels) - 1
                    end_bucket = np.digitize(row['high'], price_levels) - 1
                    
                    start_bucket = max(0, min(start_bucket, buckets - 1))
                    end_bucket = max(0, min(end_bucket, buckets - 1))
                    
                    # Distribute volume
                    bucket_count = max(1, end_bucket - start_bucket + 1)
                    volume_per_bucket = row['volume'] / bucket_count
                    
                    for i in range(start_bucket, end_bucket + 1):
                        volume_at_price[i] += volume_per_bucket
            
            # Find POC (Point of Control) - price level with highest volume
            poc_index = np.argmax(volume_at_price)
            poc_price = price_levels[poc_index] + (price_levels[1] - price_levels[0]) / 2
            
            # Calculate VWAP for the period
            vwap = (window_data['close'] * window_data['volume']).sum() / window_data['volume'].sum()
            
            # Volume-weighted standard deviation
            variance = ((window_data['close'] - vwap) ** 2 * window_data['volume']).sum() / window_data['volume'].sum()
            vol_std = np.sqrt(variance)
            
            return poc_price, vwap, vol_std
        
        # Rolling calculation of volume profile metrics
        poc_prices = []
        vwap_values = []
        vol_stds = []
        
        for i in range(len(data)):
            if i < lookback - 1:
                poc_prices.append(np.nan)
                vwap_values.append(np.nan)
                vol_stds.append(np.nan)
            else:
                window_data = data.iloc[i - lookback + 1:i + 1]
                poc, vwap, vol_std = calculate_volume_profile(window_data)
                poc_prices.append(poc)
                vwap_values.append(vwap)
                vol_stds.append(vol_std)
        
        # Calculate value area (typically 70% of volume)
        value_area_high = np.array(vwap_values) + np.array(vol_stds)
        value_area_low = np.array(vwap_values) - np.array(vol_stds)
        
        result = pd.DataFrame({
            'volume_poc': poc_prices,
            'volume_vwap': vwap_values,
            'volume_std': vol_stds,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low
        }, index=data.index)
        
        self._results = result
        self._is_calculated = True
        logger.debug(f"Calculated {self.name} for {len(data)} periods")
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on Volume Profile."""
        if self._results is None:
            self.calculate(data)
        
        signals = pd.DataFrame(index=data.index)
        
        # POC support/resistance signals
        signals['poc_support'] = (
            (data['low'] <= self._results['volume_poc']) &
            (data['close'] > self._results['volume_poc'])
        )
        
        signals['poc_resistance'] = (
            (data['high'] >= self._results['volume_poc']) &
            (data['close'] < self._results['volume_poc'])
        )
        
        # Value area signals
        signals['above_value_area'] = data['close'] > self._results['value_area_high']
        signals['below_value_area'] = data['close'] < self._results['value_area_low']
        signals['in_value_area'] = (
            (data['close'] >= self._results['value_area_low']) &
            (data['close'] <= self._results['value_area_high'])
        )
        
        # VWAP signals
        signals['above_vwap'] = data['close'] > self._results['volume_vwap']
        signals['below_vwap'] = data['close'] < self._results['volume_vwap']
        
        # Volume profile strength (based on how close price is to POC)
        price_distance = np.abs(data['close'] - self._results['volume_poc'])
        signals['signal_strength'] = 1 - (price_distance / (self._results['value_area_high'] - self._results['value_area_low']))
        signals['signal_strength'] = np.clip(signals['signal_strength'], 0, 1)
        
        return signals