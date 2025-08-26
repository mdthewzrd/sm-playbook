"""
SM Playbook Backtesting Framework
BMAT (BMad Trading) Methodology Implementation

This module provides comprehensive backtesting capabilities for
the SM Playbook trading system.
"""

from .backtest_engine import BacktestEngine
from .performance_analyzer import PerformanceAnalyzer
from .trade_simulator import TradeSimulator
from .portfolio_manager import PortfolioManager
from .risk_manager import RiskManager

__all__ = [
    'BacktestEngine',
    'PerformanceAnalyzer', 
    'TradeSimulator',
    'PortfolioManager',
    'RiskManager'
]

__version__ = '1.0.0'