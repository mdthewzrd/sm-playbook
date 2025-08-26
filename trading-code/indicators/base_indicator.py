"""
Base Indicator Class for SM Playbook Trading System

This module provides the foundation for all technical indicators
used in the BMAT methodology.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseIndicator(ABC):
    """
    Abstract base class for all technical indicators.
    
    This class defines the interface and common functionality
    that all indicators must implement.
    """
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        """
        Initialize the base indicator.
        
        Args:
            name: Name of the indicator
            params: Dictionary of parameters for the indicator
        """
        self.name = name
        self.params = params or {}
        self._data = None
        self._results = None
        self._is_calculated = False
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the indicator values.
        
        Args:
            data: OHLCV data as pandas DataFrame
            
        Returns:
            DataFrame with indicator values
        """
        pass
    
    @abstractmethod
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on indicator values.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            DataFrame with signal columns (buy, sell, strength)
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data format and completeness.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not isinstance(data, pd.DataFrame):
            logger.error(f"{self.name}: Input must be a pandas DataFrame")
            return False
            
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            logger.error(f"{self.name}: Missing required columns: {missing_cols}")
            return False
            
        if data.empty:
            logger.error(f"{self.name}: Input data is empty")
            return False
            
        if data.isnull().any().any():
            logger.warning(f"{self.name}: Input data contains null values")
            
        return True
    
    def get_minimum_periods(self) -> int:
        """
        Get the minimum number of periods required for calculation.
        
        Returns:
            Minimum periods needed
        """
        return self.params.get('min_periods', 1)
    
    def update(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Update indicator with new data.
        
        Args:
            new_data: New OHLCV data
            
        Returns:
            Updated indicator values
        """
        if not self.validate_data(new_data):
            raise ValueError("Invalid data provided for update")
            
        return self.calculate(new_data)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current state of the indicator.
        
        Returns:
            Dictionary containing indicator state
        """
        return {
            'name': self.name,
            'params': self.params,
            'is_calculated': self._is_calculated,
            'last_update': getattr(self, '_last_update', None)
        }
    
    def reset(self):
        """Reset indicator state."""
        self._data = None
        self._results = None
        self._is_calculated = False
        
    def __str__(self) -> str:
        return f"{self.name}({self.params})"
    
    def __repr__(self) -> str:
        return f"BaseIndicator(name='{self.name}', params={self.params})"


class IndicatorRegistry:
    """
    Registry for managing indicator instances and configurations.
    """
    
    def __init__(self):
        self._indicators = {}
        self._configs = {}
    
    def register(self, indicator: BaseIndicator, config: Dict[str, Any] = None):
        """
        Register an indicator instance.
        
        Args:
            indicator: Indicator instance to register
            config: Configuration for the indicator
        """
        self._indicators[indicator.name] = indicator
        self._configs[indicator.name] = config or {}
        logger.info(f"Registered indicator: {indicator.name}")
    
    def get(self, name: str) -> Optional[BaseIndicator]:
        """
        Get registered indicator by name.
        
        Args:
            name: Name of the indicator
            
        Returns:
            Indicator instance or None if not found
        """
        return self._indicators.get(name)
    
    def list_indicators(self) -> List[str]:
        """
        Get list of registered indicator names.
        
        Returns:
            List of indicator names
        """
        return list(self._indicators.keys())
    
    def remove(self, name: str):
        """
        Remove indicator from registry.
        
        Args:
            name: Name of indicator to remove
        """
        if name in self._indicators:
            del self._indicators[name]
            del self._configs[name]
            logger.info(f"Removed indicator: {name}")


# Global indicator registry
indicator_registry = IndicatorRegistry()