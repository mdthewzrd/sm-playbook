"""
DualDeviationCloud Indicator Implementation

A cloud formed between a fast and slow EMA, with two upper and two lower 
deviation bands calculated from the cloud. The multipliers for each band 
are independently configurable.

Based on specification: /.bmad-core/data/indicators/DualDeviationCloud.json
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

from base_indicator import BaseIndicator

logger = logging.getLogger(__name__)


class DualDeviationCloud(BaseIndicator):
    """
    DualDeviationCloud indicator implementation.
    
    Creates a cloud between fast and slow EMAs with configurable deviation bands.
    The strategy primarily uses 'lower_band_1' as its profit target.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize DualDeviationCloud indicator.
        
        Args:
            params: Dictionary containing:
                - ema_fast_length (int): Length for fast EMA (default: 9)
                - ema_slow_length (int): Length for slow EMA (default: 20)
                - positive_dev_1 (float): First positive deviation multiplier (default: 1.0)
                - positive_dev_2 (float): Second positive deviation multiplier (default: 0.5)
                - negative_dev_1 (float): First negative deviation multiplier (default: 2.0)
                - negative_dev_2 (float): Second negative deviation multiplier (default: 2.4)
        """
        default_params = {
            'ema_fast_length': 9,
            'ema_slow_length': 20,
            'positive_dev_1': 1.0,
            'positive_dev_2': 0.5,
            'negative_dev_1': 2.0,
            'negative_dev_2': 2.4
        }
        
        if params:
            default_params.update(params)
            
        super().__init__('DualDeviationCloud', default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the DualDeviationCloud indicator values.
        
        Args:
            data: OHLCV data as pandas DataFrame
            
        Returns:
            DataFrame with all indicator components
        """
        if not self.validate_data(data):
            raise ValueError("Invalid data provided for DualDeviationCloud calculation")
        
        result_df = data.copy()
        
        # Extract parameters
        fast_length = self.params['ema_fast_length']
        slow_length = self.params['ema_slow_length']
        pos_dev_1 = self.params['positive_dev_1']
        pos_dev_2 = self.params['positive_dev_2']
        neg_dev_1 = self.params['negative_dev_1']
        neg_dev_2 = self.params['negative_dev_2']
        
        # Calculate EMAs
        result_df['fast_ema'] = data['close'].ewm(span=fast_length, adjust=False).mean()
        result_df['slow_ema'] = data['close'].ewm(span=slow_length, adjust=False).mean()
        
        # Calculate cloud boundaries
        result_df['cloud_upper_boundary'] = np.maximum(
            result_df['fast_ema'], 
            result_df['slow_ema']
        )
        result_df['cloud_lower_boundary'] = np.minimum(
            result_df['fast_ema'], 
            result_df['slow_ema']
        )
        
        # Calculate standard deviation using slow_length period
        result_df['std_dev'] = data['close'].rolling(window=slow_length).std()
        
        # Calculate deviation bands
        result_df['upper_band_1'] = (
            result_df['cloud_upper_boundary'] + 
            (result_df['std_dev'] * pos_dev_1)
        )
        result_df['upper_band_2'] = (
            result_df['cloud_upper_boundary'] + 
            (result_df['std_dev'] * pos_dev_2)
        )
        result_df['lower_band_1'] = (
            result_df['cloud_lower_boundary'] - 
            (result_df['std_dev'] * neg_dev_1)
        )
        result_df['lower_band_2'] = (
            result_df['cloud_lower_boundary'] - 
            (result_df['std_dev'] * neg_dev_2)
        )
        
        self._results = result_df
        self._is_calculated = True
        
        return result_df
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on DualDeviationCloud values.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            DataFrame with signal columns
        """
        if not self._is_calculated:
            data = self.calculate(data)
        else:
            data = self._results
        
        signals = pd.DataFrame(index=data.index)
        
        # Basic signals for cloud interaction
        signals['above_cloud'] = data['close'] > data['cloud_upper_boundary']
        signals['below_cloud'] = data['close'] < data['cloud_lower_boundary']
        signals['in_cloud'] = ~(signals['above_cloud'] | signals['below_cloud'])
        
        # Band touch signals
        signals['touch_upper_1'] = data['high'] >= data['upper_band_1']
        signals['touch_upper_2'] = data['high'] >= data['upper_band_2']
        signals['touch_lower_1'] = data['low'] <= data['lower_band_1']
        signals['touch_lower_2'] = data['low'] <= data['lower_band_2']
        
        # Entry window cutoff signal (for strategy implementation)
        signals['entry_cutoff'] = (
            signals['touch_lower_1'] | 
            signals['touch_lower_2']
        )
        
        return signals
    
    def get_minimum_periods(self) -> int:
        """
        Get minimum periods required for calculation.
        
        Returns:
            Maximum of slow_length (for EMA and StdDev)
        """
        return max(self.params['ema_slow_length'], 20)
    
    def get_bands_for_timeframe(self, data: pd.DataFrame, timeframe: str) -> Dict[str, pd.Series]:
        """
        Get all bands for a specific timeframe.
        
        Args:
            data: OHLCV data
            timeframe: Timeframe identifier
            
        Returns:
            Dictionary with all band series
        """
        if not self._is_calculated:
            self.calculate(data)
        
        return {
            'upper_band_1': self._results['upper_band_1'],
            'upper_band_2': self._results['upper_band_2'], 
            'lower_band_1': self._results['lower_band_1'],
            'lower_band_2': self._results['lower_band_2'],
            'cloud_upper': self._results['cloud_upper_boundary'],
            'cloud_lower': self._results['cloud_lower_boundary'],
            'fast_ema': self._results['fast_ema'],
            'slow_ema': self._results['slow_ema']
        }
    
    def __str__(self) -> str:
        return f"DualDeviationCloud(fast={self.params['ema_fast_length']}, slow={self.params['ema_slow_length']})"