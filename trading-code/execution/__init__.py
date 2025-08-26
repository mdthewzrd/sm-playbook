"""
SM Playbook Execution Engine
BMAT (BMad Trading) Methodology Implementation

This module provides trade execution capabilities with comprehensive
risk management for the SM Playbook trading system.
"""

from .execution_engine import ExecutionEngine
from .risk_manager import RiskManager
from .position_manager import PositionManager
from .order_manager import OrderManager
from .signal_processor import SignalProcessor

__all__ = [
    'ExecutionEngine',
    'RiskManager',
    'PositionManager', 
    'OrderManager',
    'SignalProcessor'
]

__version__ = '1.0.0'