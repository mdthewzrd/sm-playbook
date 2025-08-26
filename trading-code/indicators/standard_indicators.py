"""
Standard Technical Indicators for SM Playbook Trading System

Implementation of common technical indicators using the BaseIndicator
framework for consistency with the BMAT methodology.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_indicator import BaseIndicator
import logging

logger = logging.getLogger(__name__)


class StandardMovingAverage(BaseIndicator):
    """
    Standard Moving Average (SMA, EMA, WMA).
    """
    
    def __init__(self, period: int = 20, ma_type: str = 'sma'):
        """
        Initialize Moving Average.
        
        Args:
            period: Period for moving average calculation
            ma_type: Type of MA ('sma', 'ema', 'wma')
        """
        params = {
            'period': period,
            'ma_type': ma_type.lower(),
            'min_periods': period
        }
        super().__init__(f'MovingAverage_{ma_type.upper()}_{period}', params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Moving Average."""
        if not self.validate_data(data):
            raise ValueError("Invalid data for Moving Average calculation")
        
        period = self.params['period']
        ma_type = self.params['ma_type']
        
        if ma_type == 'sma':
            ma_value = data['close'].rolling(window=period).mean()
        elif ma_type == 'ema':
            ma_value = data['close'].ewm(span=period).mean()
        elif ma_type == 'wma':
            weights = np.arange(1, period + 1)
            ma_value = data['close'].rolling(window=period).apply(
                lambda x: np.average(x, weights=weights), raw=True
            )
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")
        
        result = pd.DataFrame({
            f'ma_{ma_type}_{period}': ma_value
        }, index=data.index)
        
        self._results = result
        self._is_calculated = True
        logger.debug(f"Calculated {self.name} for {len(data)} periods")
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on Moving Average."""
        if self._results is None:
            self.calculate(data)
        
        signals = pd.DataFrame(index=data.index)
        ma_col = self._results.columns[0]
        
        # Price vs MA signals
        signals['above_ma'] = data['close'] > self._results[ma_col]
        signals['below_ma'] = data['close'] < self._results[ma_col]
        
        # MA crossover signals
        signals['bullish_cross'] = (
            (data['close'] > self._results[ma_col]) &
            (data['close'].shift() <= self._results[ma_col].shift())
        )
        
        signals['bearish_cross'] = (
            (data['close'] < self._results[ma_col]) &
            (data['close'].shift() >= self._results[ma_col].shift())
        )
        
        # Signal strength based on distance from MA
        distance = np.abs(data['close'] - self._results[ma_col]) / self._results[ma_col]
        signals['signal_strength'] = np.clip(distance * 10, 0, 1)
        
        return signals


class StandardRSI(BaseIndicator):
    """
    Standard Relative Strength Index (RSI).
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize RSI.
        
        Args:
            period: Period for RSI calculation
        """
        params = {
            'period': period,
            'min_periods': period + 1
        }
        super().__init__(f'RSI_{period}', params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI."""
        if not self.validate_data(data):
            raise ValueError("Invalid data for RSI calculation")
        
        period = self.params['period']
        
        # Calculate price changes
        delta = data['close'].diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        result = pd.DataFrame({
            f'rsi_{period}': rsi
        }, index=data.index)
        
        self._results = result
        self._is_calculated = True
        logger.debug(f"Calculated {self.name} for {len(data)} periods")
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on RSI."""
        if self._results is None:
            self.calculate(data)
        
        signals = pd.DataFrame(index=data.index)
        rsi_col = self._results.columns[0]
        rsi_values = self._results[rsi_col]
        
        # RSI level signals
        signals['oversold'] = rsi_values < 30
        signals['overbought'] = rsi_values > 70
        signals['neutral'] = (rsi_values >= 30) & (rsi_values <= 70)
        
        # RSI crossover signals
        signals['oversold_recovery'] = (
            (rsi_values > 30) & (rsi_values.shift() <= 30)
        )
        signals['overbought_decline'] = (
            (rsi_values < 70) & (rsi_values.shift() >= 70)
        )
        
        # RSI momentum signals
        signals['rsi_bullish'] = (
            (rsi_values > 50) & (rsi_values > rsi_values.shift())
        )
        signals['rsi_bearish'] = (
            (rsi_values < 50) & (rsi_values < rsi_values.shift())
        )
        
        # Signal strength based on RSI extremes
        signals['signal_strength'] = np.where(
            rsi_values < 30, (30 - rsi_values) / 30,
            np.where(rsi_values > 70, (rsi_values - 70) / 30, 0)
        )
        
        return signals


class StandardBollingerBands(BaseIndicator):
    """
    Standard Bollinger Bands.
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands.
        
        Args:
            period: Period for moving average and standard deviation
            std_dev: Standard deviation multiplier
        """
        params = {
            'period': period,
            'std_dev': std_dev,
            'min_periods': period
        }
        super().__init__(f'BollingerBands_{period}_{std_dev}', params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        if not self.validate_data(data):
            raise ValueError("Invalid data for Bollinger Bands calculation")
        
        period = self.params['period']
        std_dev = self.params['std_dev']
        
        # Calculate middle band (SMA)
        middle = data['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = data['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        # Calculate band width and position
        band_width = (upper - lower) / middle * 100
        band_position = (data['close'] - lower) / (upper - lower) * 100
        
        result = pd.DataFrame({
            f'bb_upper_{period}': upper,
            f'bb_middle_{period}': middle,
            f'bb_lower_{period}': lower,
            f'bb_width_{period}': band_width,
            f'bb_position_{period}': band_position
        }, index=data.index)
        
        self._results = result
        self._is_calculated = True
        logger.debug(f"Calculated {self.name} for {len(data)} periods")
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on Bollinger Bands."""
        if self._results is None:
            self.calculate(data)
        
        signals = pd.DataFrame(index=data.index)
        period = self.params['period']
        
        upper_col = f'bb_upper_{period}'
        middle_col = f'bb_middle_{period}'
        lower_col = f'bb_lower_{period}'
        width_col = f'bb_width_{period}'
        position_col = f'bb_position_{period}'
        
        # Band touch signals
        signals['upper_touch'] = data['close'] >= self._results[upper_col]
        signals['lower_touch'] = data['close'] <= self._results[lower_col]
        
        # Mean reversion signals
        signals['mean_revert_long'] = (
            (data['close'] < self._results[lower_col]) &
            (data['close'] > data['close'].shift())
        )
        signals['mean_revert_short'] = (
            (data['close'] > self._results[upper_col]) &
            (data['close'] < data['close'].shift())
        )
        
        # Squeeze detection
        squeeze_threshold = 10
        signals['squeeze'] = self._results[width_col] < squeeze_threshold
        
        # Breakout signals
        signals['breakout_long'] = (
            (data['close'] > self._results[upper_col]) &
            (data['close'].shift() <= self._results[upper_col].shift())
        )
        signals['breakout_short'] = (
            (data['close'] < self._results[lower_col]) &
            (data['close'].shift() >= self._results[lower_col].shift())
        )
        
        # Signal strength based on band position
        signals['signal_strength'] = np.abs(
            self._results[position_col] - 50
        ) / 50
        
        return signals


class StandardATR(BaseIndicator):
    """
    Standard Average True Range (ATR).
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize ATR.
        
        Args:
            period: Period for ATR calculation
        """
        params = {
            'period': period,
            'min_periods': period + 1
        }
        super().__init__(f'ATR_{period}', params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR."""
        if not self.validate_data(data):
            raise ValueError("Invalid data for ATR calculation")
        
        period = self.params['period']
        
        # Calculate True Range
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        # Calculate ATR
        atr = true_range.rolling(window=period).mean()
        
        # ATR as percentage of price
        atr_percent = (atr / data['close']) * 100
        
        # ATR percentile rank over 252 periods (1 year)
        atr_percentile = atr.rolling(window=252).rank(pct=True) * 100
        
        result = pd.DataFrame({
            f'atr_{period}': atr,
            f'atr_percent_{period}': atr_percent,
            f'atr_percentile_{period}': atr_percentile
        }, index=data.index)
        
        self._results = result
        self._is_calculated = True
        logger.debug(f"Calculated {self.name} for {len(data)} periods")
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on ATR."""
        if self._results is None:
            self.calculate(data)
        
        signals = pd.DataFrame(index=data.index)
        period = self.params['period']
        
        atr_col = f'atr_{period}'
        atr_percent_col = f'atr_percent_{period}'
        atr_percentile_col = f'atr_percentile_{period}'
        
        # Volatility regime signals
        signals['low_volatility'] = self._results[atr_percentile_col] < 20
        signals['high_volatility'] = self._results[atr_percentile_col] > 80
        signals['normal_volatility'] = (
            (self._results[atr_percentile_col] >= 20) &
            (self._results[atr_percentile_col] <= 80)
        )
        
        # Volatility expansion signals
        signals['volatility_expansion'] = (
            self._results[atr_col] > self._results[atr_col].rolling(10).mean() * 1.5
        )
        
        signals['volatility_contraction'] = (
            self._results[atr_col] < self._results[atr_col].rolling(10).mean() * 0.7
        )
        
        # Signal strength based on volatility extremes
        signals['signal_strength'] = np.where(
            self._results[atr_percentile_col] > 80, 
            (self._results[atr_percentile_col] - 80) / 20,
            np.where(
                self._results[atr_percentile_col] < 20,
                (20 - self._results[atr_percentile_col]) / 20,
                0
            )
        )
        
        return signals


class StandardMACD(BaseIndicator):
    """
    Standard MACD (Moving Average Convergence Divergence).
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize MACD.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
        """
        params = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'min_periods': slow_period + signal_period
        }
        super().__init__(f'MACD_{fast_period}_{slow_period}_{signal_period}', params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD."""
        if not self.validate_data(data):
            raise ValueError("Invalid data for MACD calculation")
        
        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        signal_period = self.params['signal_period']
        
        # Calculate EMAs
        ema_fast = data['close'].ewm(span=fast_period).mean()
        ema_slow = data['close'].ewm(span=slow_period).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        result = pd.DataFrame({
            f'macd_line_{fast_period}_{slow_period}': macd_line,
            f'macd_signal_{fast_period}_{slow_period}_{signal_period}': signal_line,
            f'macd_histogram_{fast_period}_{slow_period}_{signal_period}': histogram
        }, index=data.index)
        
        self._results = result
        self._is_calculated = True
        logger.debug(f"Calculated {self.name} for {len(data)} periods")
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on MACD."""
        if self._results is None:
            self.calculate(data)
        
        signals = pd.DataFrame(index=data.index)
        
        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        signal_period = self.params['signal_period']
        
        macd_col = f'macd_line_{fast_period}_{slow_period}'
        signal_col = f'macd_signal_{fast_period}_{slow_period}_{signal_period}'
        hist_col = f'macd_histogram_{fast_period}_{slow_period}_{signal_period}'
        
        # MACD crossover signals
        signals['bullish_crossover'] = (
            (self._results[macd_col] > self._results[signal_col]) &
            (self._results[macd_col].shift() <= self._results[signal_col].shift())
        )
        
        signals['bearish_crossover'] = (
            (self._results[macd_col] < self._results[signal_col]) &
            (self._results[macd_col].shift() >= self._results[signal_col].shift())
        )
        
        # Zero line crossovers
        signals['zero_line_bullish'] = (
            (self._results[macd_col] > 0) &
            (self._results[macd_col].shift() <= 0)
        )
        
        signals['zero_line_bearish'] = (
            (self._results[macd_col] < 0) &
            (self._results[macd_col].shift() >= 0)
        )
        
        # Histogram signals
        signals['histogram_bullish'] = self._results[hist_col] > 0
        signals['histogram_bearish'] = self._results[hist_col] < 0
        
        signals['histogram_increasing'] = (
            self._results[hist_col] > self._results[hist_col].shift()
        )
        
        signals['histogram_decreasing'] = (
            self._results[hist_col] < self._results[hist_col].shift()
        )
        
        # Signal strength based on MACD line magnitude
        signals['signal_strength'] = np.abs(self._results[macd_col]) / data['close'] * 1000
        signals['signal_strength'] = np.clip(signals['signal_strength'], 0, 1)
        
        return signals