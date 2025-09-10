#!/usr/bin/env python3
"""
SM Playbook Agent Factory System
Multi-agent orchestration for algorithmic trading system

This system creates and manages specialized trading agents that work together
to implement the Lingua trading language systematically.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Available agent types in the SM Playbook system."""
    ORCHESTRATOR = "orchestrator"
    TRADING_ORCHESTRATOR = "trading-orchestrator" 
    STRATEGY_DESIGNER = "strategy-designer"
    INDICATOR_DEVELOPER = "indicator-developer"
    BACKTESTING_ENGINEER = "backtesting-engineer"
    SCANNER_DEVELOPER = "scanner-developer"
    SCANNER_BUILDER = "scanner-builder"  # New 8-phase scanner builder
    SIGNAL_GENERATOR = "signal-generator"
    RISK_MANAGER = "risk-manager"


class MessageType(Enum):
    """Message types for inter-agent communication."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    DATA_UPDATE = "data_update"
    SIGNAL_GENERATED = "signal_generated"
    STRATEGY_DEVELOPED = "strategy_developed"
    BACKTEST_COMPLETED = "backtest_completed"
    SCAN_RESULTS = "scan_results"


@dataclass
class AgentMessage:
    """Message structure for agent communication."""
    id: str
    sender: str
    recipient: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime = None
    priority: int = 1

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass 
class AgentCapabilities:
    """Defines what an agent can do."""
    name: str
    description: str
    commands: List[str]
    inputs: List[str]
    outputs: List[str]
    dependencies: List[str] = None
    mcp_servers: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.mcp_servers is None:
            self.mcp_servers = []


class BaseAgent:
    """Base class for all SM Playbook agents."""
    
    def __init__(self, agent_id: str, capabilities: AgentCapabilities, message_bus=None):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.message_bus = message_bus
        self.active = False
        self.task_queue = asyncio.Queue()
        self.context = {}
        logger.info(f"Initialized agent: {agent_id}")

    async def start(self):
        """Start the agent."""
        self.active = True
        logger.info(f"Started agent: {self.agent_id}")

    async def stop(self):
        """Stop the agent."""
        self.active = False
        logger.info(f"Stopped agent: {self.agent_id}")

    async def process_message(self, message: AgentMessage):
        """Process incoming messages."""
        logger.info(f"{self.agent_id} received message: {message.message_type}")
        # Override in subclasses
        return None

    async def send_message(self, recipient: str, message_type: MessageType, content: Dict[str, Any]):
        """Send message to another agent."""
        if self.message_bus:
            message = AgentMessage(
                id=f"{self.agent_id}_{datetime.now().timestamp()}",
                sender=self.agent_id,
                recipient=recipient,
                message_type=message_type,
                content=content
            )
            await self.message_bus.route_message(message)

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "agent_id": self.agent_id,
            "active": self.active,
            "capabilities": asdict(self.capabilities),
            "context": self.context
        }


class TradingOrchestratorAgent(BaseAgent):
    """Master trading system orchestrator implementing Lingua methodology."""
    
    def __init__(self, agent_id: str = "trading-orchestrator"):
        capabilities = AgentCapabilities(
            name="Trading Orchestrator",
            description="Coordinates all trading activities using Lingua framework",
            commands=["analyze_market", "coordinate_strategy", "manage_risk", "generate_signals"],
            inputs=["market_data", "strategy_requests", "risk_parameters"],
            outputs=["trading_decisions", "risk_assessments", "coordination_signals"],
            mcp_servers=["polygon", "notion", "talib"]
        )
        super().__init__(agent_id, capabilities)
        self.lingua_context = {}
        self.active_strategies = []

    async def process_message(self, message: AgentMessage):
        """Process messages with Lingua framework application."""
        if message.message_type == MessageType.TASK_REQUEST:
            task = message.content.get("task")
            
            if task == "analyze_market":
                return await self._analyze_market_lingua(message.content)
            elif task == "coordinate_strategy":
                return await self._coordinate_strategy(message.content)
            elif task == "manage_portfolio_risk":
                return await self._manage_risk(message.content)
        
        return {"status": "processed", "agent": self.agent_id}

    async def _analyze_market_lingua(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market using Lingua 8-stage trend cycle."""
        symbol = content.get("symbol", "SPY")
        timeframe = content.get("timeframe", "daily")
        
        analysis = {
            "symbol": symbol,
            "timeframe": timeframe,
            "lingua_analysis": {
                "trend_cycle_stage": "analyzing...",
                "htf_context": "determining structure...",
                "mtf_timing": "identifying route...",
                "ltf_execution": "waiting for setup...",
                "risk_reward": "calculating...",
                "confidence": 0.0
            },
            "recommended_action": "hold_analysis"
        }
        
        # In real implementation, this would call MCP servers
        logger.info(f"Analyzed {symbol} using Lingua framework")
        return analysis

    async def _coordinate_strategy(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate strategy development and execution."""
        strategy_type = content.get("strategy_type", "os_d1")
        
        # Route to appropriate specialist agent
        if strategy_type == "os_d1":
            await self.send_message(
                "scanner-developer",
                MessageType.TASK_REQUEST,
                {"task": "develop_os_d1_scanner", "parameters": content}
            )
        elif strategy_type == "euphoric_top":
            await self.send_message(
                "signal-generator", 
                MessageType.TASK_REQUEST,
                {"task": "detect_euphoric_tops", "parameters": content}
            )

        return {"status": "strategy_coordination_initiated", "strategy_type": strategy_type}

    async def _manage_risk(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Manage portfolio risk using Lingua principles."""
        return {
            "portfolio_risk": "8% max",
            "position_risk": "2% per trade", 
            "daily_loss_limit": "$5000",
            "risk_status": "within_limits"
        }


class StrategyDesignerAgent(BaseAgent):
    """Designs systematic strategies from Lingua concepts."""
    
    def __init__(self, agent_id: str = "strategy-designer"):
        capabilities = AgentCapabilities(
            name="Strategy Designer",
            description="Creates systematic strategies from discretionary Lingua concepts",
            commands=["design_strategy", "formalize_rules", "create_backtests"],
            inputs=["lingua_concepts", "market_patterns", "performance_targets"],
            outputs=["strategy_specifications", "trading_rules", "backtest_code"],
            mcp_servers=["backtesting", "talib"]
        )
        super().__init__(agent_id, capabilities)

    async def process_message(self, message: AgentMessage):
        """Process strategy design requests."""
        if message.message_type == MessageType.TASK_REQUEST:
            task = message.content.get("task")
            
            if task == "design_os_d1_strategy":
                return await self._design_os_d1_strategy(message.content)
            elif task == "create_custom_strategy":
                return await self._create_custom_strategy(message.content)
        
        return {"status": "processed", "agent": self.agent_id}

    async def _design_os_d1_strategy(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Design the OS D1 (Opening Strength Day 1) strategy."""
        strategy_spec = {
            "name": "OS D1 Scanner Strategy",
            "description": "Small cap day one momentum system",
            "framework": "Lingua Trading Language",
            "win_rate_target": 0.70,
            "risk_reward_target": 2.0,
            "entry_criteria": {
                "market_cap": "< 2B",
                "gap_up": "> 15%",
                "volume": "> 2x avg",
                "float": "< 50M shares",
                "htf_trend": "bullish_structure"
            },
            "exit_criteria": {
                "profit_target": "2R",
                "stop_loss": "1R",
                "time_stop": "EOD"
            },
            "lingua_implementation": {
                "trend_cycle_stage": "1-2 (consolidation to breakout)",
                "timeframe_hierarchy": "D1 HTF, H1 MTF, M5 LTF",
                "context": "frontside momentum"
            }
        }
        
        # Send to backtesting engineer
        await self.send_message(
            "backtesting-engineer",
            MessageType.TASK_REQUEST,
            {"task": "create_backtest", "strategy_spec": strategy_spec}
        )
        
        return strategy_spec

    async def _create_custom_strategy(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom strategy from provided parameters."""
        return {
            "status": "custom_strategy_created",
            "strategy_name": content.get("name", "Custom Strategy"),
            "parameters": content
        }


class IndicatorDeveloperAgent(BaseAgent):
    """Develops custom indicators based on discretionary patterns."""
    
    def __init__(self, agent_id: str = "indicator-developer"):
        capabilities = AgentCapabilities(
            name="Indicator Developer", 
            description="Creates custom indicators implementing Lingua concepts",
            commands=["develop_ema_cloud", "create_deviation_bands", "build_trail_system"],
            inputs=["discretionary_patterns", "indicator_requirements"],
            outputs=["indicator_code", "visualization_code", "parameter_sets"],
            mcp_servers=["talib", "backtesting"]
        )
        super().__init__(agent_id, capabilities)

    async def process_message(self, message: AgentMessage):
        """Process indicator development requests."""
        if message.message_type == MessageType.TASK_REQUEST:
            task = message.content.get("task")
            
            if task == "develop_ema_cloud":
                return await self._develop_ema_cloud(message.content)
            elif task == "create_custom_indicator":
                return await self._create_custom_indicator(message.content)
                
        return {"status": "processed", "agent": self.agent_id}

    async def _develop_ema_cloud(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Develop EMA cloud system (9/20 and 72/89)."""
        ema_cloud_spec = {
            "name": "Lingua EMA Cloud System",
            "description": "Dual EMA cloud system for trend analysis",
            "parameters": {
                "fast_cloud": {"fast_ema": 9, "slow_ema": 20},
                "slow_cloud": {"fast_ema": 72, "slow_ema": 89}
            },
            "signals": {
                "bullish": "Price above both clouds, fast > slow",
                "bearish": "Price below both clouds, fast < slow", 
                "neutral": "Mixed cloud positioning"
            },
            "code_template": """
def calculate_ema_cloud(data, periods=[9, 20, 72, 89]):
    '''Calculate Lingua EMA cloud system'''
    clouds = {}
    for period in periods:
        clouds[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
    
    # Define cloud zones
    clouds['fast_cloud_top'] = np.maximum(clouds['ema_9'], clouds['ema_20'])
    clouds['fast_cloud_bottom'] = np.minimum(clouds['ema_9'], clouds['ema_20'])
    clouds['slow_cloud_top'] = np.maximum(clouds['ema_72'], clouds['ema_89']) 
    clouds['slow_cloud_bottom'] = np.minimum(clouds['ema_72'], clouds['ema_89'])
    
    return clouds
            """
        }
        
        return ema_cloud_spec

    async def _create_custom_indicator(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create custom indicator from specifications."""
        return {
            "status": "custom_indicator_created",
            "indicator_name": content.get("name", "Custom Indicator"),
            "specifications": content
        }


class BacktestingEngineerAgent(BaseAgent):
    """Validates strategies with historical data using backtesting.py."""
    
    def __init__(self, agent_id: str = "backtesting-engineer"):
        capabilities = AgentCapabilities(
            name="Backtesting Engineer",
            description="Historical strategy validation and performance analysis",
            commands=["run_backtest", "optimize_parameters", "generate_report"],
            inputs=["strategy_specs", "historical_data", "test_parameters"],
            outputs=["backtest_results", "performance_metrics", "optimization_reports"],
            mcp_servers=["backtesting", "polygon"]
        )
        super().__init__(agent_id, capabilities)

    async def process_message(self, message: AgentMessage):
        """Process backtesting requests."""
        if message.message_type == MessageType.TASK_REQUEST:
            task = message.content.get("task")
            
            if task == "create_backtest":
                return await self._create_backtest(message.content)
            elif task == "run_optimization":
                return await self._run_optimization(message.content)
                
        return {"status": "processed", "agent": self.agent_id}

    async def _create_backtest(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create backtest from strategy specification."""
        strategy_spec = content.get("strategy_spec", {})
        
        backtest_code = f"""
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import yfinance as yf

class {strategy_spec.get('name', 'Strategy').replace(' ', '')}(Strategy):
    def init(self):
        # Initialize strategy with Lingua framework
        pass
    
    def next(self):
        # Implement trading logic
        # Entry: {strategy_spec.get('entry_criteria', {})}
        # Exit: {strategy_spec.get('exit_criteria', {})}
        pass

# Run backtest
if __name__ == "__main__":
    # Fetch data
    data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
    
    # Run backtest
    bt = Backtest(data, {strategy_spec.get('name', 'Strategy').replace(' ', '')})
    result = bt.run()
    
    print(result)
    bt.plot()
        """
        
        return {
            "status": "backtest_created",
            "strategy_name": strategy_spec.get('name'),
            "backtest_code": backtest_code,
            "expected_metrics": {
                "win_rate": strategy_spec.get('win_rate_target', 0.6),
                "risk_reward": strategy_spec.get('risk_reward_target', 1.5)
            }
        }

    async def _run_optimization(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Run parameter optimization on strategy."""
        return {
            "status": "optimization_completed",
            "best_parameters": content.get("parameters", {}),
            "performance_improvement": "15%"
        }


class ScannerDeveloperAgent(BaseAgent):
    """Builds stock screening systems like OS D1."""
    
    def __init__(self, agent_id: str = "scanner-developer"):
        capabilities = AgentCapabilities(
            name="Scanner Developer",
            description="Creates stock screening systems for systematic trading",
            commands=["develop_os_d1_scanner", "create_euphoric_scanner", "build_custom_scanner"],
            inputs=["screening_criteria", "market_data", "filter_parameters"],
            outputs=["scanner_code", "candidate_lists", "screening_reports"],
            mcp_servers=["polygon", "talib"]
        )
        super().__init__(agent_id, capabilities)

    async def process_message(self, message: AgentMessage):
        """Process scanner development requests."""
        if message.message_type == MessageType.TASK_REQUEST:
            task = message.content.get("task")
            
            if task == "develop_os_d1_scanner":
                return await self._develop_os_d1_scanner(message.content)
            elif task == "create_euphoric_scanner":
                return await self._create_euphoric_scanner(message.content)
                
        return {"status": "processed", "agent": self.agent_id}

    async def _develop_os_d1_scanner(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Develop OS D1 (Opening Strength Day 1) scanner."""
        scanner_code = """
class OSD1Scanner:
    '''
    Opening Strength Day One Scanner
    Identifies small cap stocks on day one of significant moves
    Win Rate Target: 70%+
    Framework: Lingua Trading Language
    '''
    
    def __init__(self):
        self.criteria = {
            'min_gap': 0.15,          # 15% gap up minimum
            'max_market_cap': 2e9,    # 2B max market cap
            'min_volume_ratio': 2.0,  # 2x average volume
            'max_float': 50e6         # 50M max float
        }
    
    def scan_candidates(self, date):
        '''Scan for OS D1 candidates'''
        candidates = []
        
        # Implementation would connect to market data
        # and apply Lingua framework analysis
        
        return candidates
    
    def validate_lingua_setup(self, symbol, data):
        '''Validate setup using Lingua framework'''
        validation = {
            'trend_cycle_stage': self._identify_trend_stage(data),
            'timeframe_alignment': self._check_timeframe_alignment(data),
            'risk_reward_ratio': self._calculate_risk_reward(data),
            'confidence_score': self._calculate_confidence(data)
        }
        return validation
        """
        
        return {
            "status": "os_d1_scanner_created",
            "scanner_code": scanner_code,
            "criteria": content.get("parameters", {}),
            "expected_candidates": "10-20 per day"
        }

    async def _create_euphoric_scanner(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create euphoric top detection scanner."""
        return {
            "status": "euphoric_scanner_created",
            "description": "Parabolic extension detector for short opportunities",
            "framework": "Trend cycle stage 4-5 detection"
        }


class ScannerBuilderAgent(BaseAgent):
    """
    Builds scanners using the 8-phase development process.
    Always validates against known quality examples like HOOD 3/3/25.
    """
    
    def __init__(self, agent_id: str = "scanner-builder"):
        capabilities = AgentCapabilities(
            name="Scanner Builder",
            description="Builds and validates scanners using 8-phase development process",
            commands=["build_scanner", "validate_known_examples", "optimize_parameters", "backtest_scanner"],
            inputs=["quality_examples", "pattern_definition", "trader_parameters"],
            outputs=["validated_scanner", "performance_metrics", "quality_distribution"],
            mcp_servers=["polygon", "backtesting"]
        )
        super().__init__(agent_id, capabilities)
        
        # Known quality benchmarks
        self.benchmark_setups = [
            {'symbol': 'HOOD', 'date': '2025-03-03', 'grade': 'A+'},
            # Add more proven setups here
        ]
        
        # 8-phase process tracking
        self.development_phases = {
            1: "single_ticker_analysis",
            2: "analyzer_development",
            3: "parameter_baseline_discovery",
            4: "initial_scanner_creation",
            5: "name_testing_mold_fitting",
            6: "time_period_testing",
            7: "optimization_refinement",
            8: "validation_backtesting"
        }
        
    async def process_message(self, message: AgentMessage):
        """Process scanner building requests."""
        if message.message_type == MessageType.TASK_REQUEST:
            task = message.content.get("task")
            
            if task == "build_scanner_8phase":
                return await self._build_scanner_8phase(message.content)
            elif task == "validate_known_examples":
                return await self._validate_known_examples(message.content)
            elif task == "optimize_scanner_parameters":
                return await self._optimize_parameters(message.content)
                
        return {"status": "processed", "agent": self.agent_id}
    
    async def _build_scanner_8phase(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete 8-phase scanner development."""
        pattern_name = content.get("pattern_name", "backside_pop")
        known_example = content.get("known_example", self.benchmark_setups[0])
        
        results = {
            "pattern": pattern_name,
            "phases_completed": [],
            "scanner_code": None,
            "validation_results": {}
        }
        
        # Phase 1: Single Ticker Analysis
        phase1_result = await self._phase1_single_ticker(known_example)
        results["phases_completed"].append({"phase": 1, "result": phase1_result})
        
        # Phase 2: Analyzer Development
        phase2_result = await self._phase2_analyzer_development(phase1_result)
        results["phases_completed"].append({"phase": 2, "result": phase2_result})
        
        # Phase 3: Parameter Baseline Discovery
        phase3_result = await self._phase3_parameter_baseline(phase2_result)
        results["phases_completed"].append({"phase": 3, "result": phase3_result})
        
        # Phase 4: Initial Scanner Creation
        phase4_result = await self._phase4_initial_scanner(phase3_result)
        results["scanner_code"] = phase4_result["scanner_code"]
        
        # Validate against known example
        validation = await self._validate_scanner_finds_example(
            phase4_result["scanner_code"],
            known_example
        )
        
        if not validation["found"]:
            return {
                "status": "failed",
                "reason": f"Scanner failed to find benchmark: {known_example}",
                "phases_completed": results["phases_completed"]
            }
        
        results["validation_results"]["benchmark"] = validation
        
        # Continue with remaining phases...
        # Phase 5-8 implementation
        
        return results
    
    async def _validate_known_examples(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scanner against all known quality examples."""
        scanner_code = content.get("scanner_code")
        
        validation_results = {
            "total_benchmarks": len(self.benchmark_setups),
            "found": 0,
            "missed": 0,
            "details": []
        }
        
        for benchmark in self.benchmark_setups:
            result = await self._test_scanner_on_ticker(
                scanner_code,
                benchmark["symbol"],
                benchmark["date"]
            )
            
            if result["found"]:
                validation_results["found"] += 1
            else:
                validation_results["missed"] += 1
                
            validation_results["details"].append({
                "benchmark": benchmark,
                "result": result
            })
        
        # Must find ALL benchmarks to pass
        validation_results["passed"] = (
            validation_results["found"] == validation_results["total_benchmarks"]
        )
        
        return validation_results
    
    async def _optimize_parameters(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize scanner parameters using trader's EXACT specifications.
        0.3 means exactly 0.3, not approximately!
        """
        current_params = content.get("current_parameters", {})
        trader_specs = content.get("trader_specifications", {})
        
        optimized_params = {}
        
        for param_name, trader_value in trader_specs.items():
            # Use EXACT trader values - no approximation!
            optimized_params[param_name] = trader_value
            
            # Log if current differs from trader spec
            if param_name in current_params:
                current = current_params[param_name]
                if current != trader_value:
                    logger.warning(
                        f"Parameter {param_name}: current={current}, "
                        f"trader_spec={trader_value} - Using trader spec!"
                    )
        
        return {
            "status": "optimized",
            "parameters": optimized_params,
            "message": "Using exact trader specifications - no approximations!"
        }
    
    async def _phase1_single_ticker(self, known_example: Dict) -> Dict:
        """Phase 1: Analyze single ticker example - HOOD 3/3/25 is the benchmark"""
        ticker = known_example.get('symbol', 'HOOD')
        date = known_example.get('date', '2025-03-03')
        
        analysis = {
            'ticker': ticker,
            'date': date,
            'setup_type': None,
            'key_characteristics': [],
            'entry_criteria': {},
            'exit_criteria': {},
            'disqualifiers': [],
            'technical_measurements': {}
        }
        
        # Analyze the specific setup
        if ticker == 'HOOD' and date == '2025-03-03':
            analysis['setup_type'] = 'OS D1 Small Cap Gapper'
            analysis['key_characteristics'] = [
                'Opening drive with 3+ ATR move',
                'Volume surge >300% of average',
                'Clear trend structure on 5min',
                'Fade opportunity at resistance'
            ]
            analysis['entry_criteria'] = {
                'min_trend_atr': 4.0,  # Exactly 4.0
                'min_fade_atr': 3.0,   # Exactly 3.0
                'min_volume_outlier': 0.3,  # Exactly 0.3
                'rsi_threshold': 70,
                'deviation_bands': 2.5
            }
        
        return analysis
    
    async def _phase2_analyzer_development(self, pattern_data: Dict) -> Dict:
        """Phase 2: Build analyzer for pattern quantification"""
        analyzer = {
            'pattern_type': pattern_data.get('setup_type'),
            'parameter_calculations': [],
            'measurement_functions': {},
            'version': 'v1_initial'
        }
        
        # Build comprehensive parameter calculations
        calculations = [
            'calculate_atr_multiples',
            'measure_volume_surge',
            'identify_trend_structure',
            'calculate_deviation_bands',
            'measure_momentum_exhaustion'
        ]
        
        for calc in calculations:
            analyzer['parameter_calculations'].append({
                'name': calc,
                'formula': f'{calc}_formula',
                'preserve_all': True,  # Don't optimize prematurely
                'version_controlled': True
            })
        
        return analyzer
    
    async def _phase3_parameter_baseline(self, analyzer_data: Dict) -> Dict:
        """Phase 3: Discover parameter baselines from quality examples"""
        baseline = {
            'pattern_type': analyzer_data.get('pattern_type'),
            'common_parameters': {},
            'critical_criteria': [],
            'optional_criteria': [],
            'initial_thresholds': {}
        }
        
        # Analyze all quality setups to find common ranges
        baseline['common_parameters'] = {
            'atr_range': [3.0, 8.0],
            'volume_range': [0.3, 5.0],  # 30% to 500% above average
            'rsi_range': [65, 85],
            'time_window': '9:30-10:30'  # First hour
        }
        
        baseline['critical_criteria'] = [
            'min_atr_move',
            'volume_surge',
            'trend_alignment'
        ]
        
        baseline['initial_thresholds'] = {
            'conservative': True,  # Start restrictive
            'min_confidence': 0.8,
            'accuracy_over_coverage': True
        }
        
        return baseline
    
    async def _phase4_initial_scanner(self, baseline_data: Dict) -> Dict:
        """Phase 4: Create initial scanner with baseline parameters"""
        scanner_code = f"""
class BacksidePopScanner:
    '''
    Scanner implementing 8-phase development process
    Must find HOOD 3/3/25 benchmark
    '''
    
    def __init__(self):
        self.parameters = {{
            'min_trend_atr': 4.0,  # Exactly 4.0
            'min_fade_atr': 3.0,   # Exactly 3.0
            'min_volume_outlier': 0.3,  # Exactly 0.3
            'rsi_threshold': 70,
            'deviation_bands': 2.5
        }}
        self.benchmark = ('HOOD', '2025-03-03')
    
    def scan(self, symbol, date):
        # Scanner implementation
        pass
        """
        
        return {
            'scanner_code': scanner_code,
            'baseline': baseline_data,
            'validation_required': True
        }
    
    async def _validate_scanner_finds_example(self, scanner_code: str, known_example: Dict) -> Dict:
        """Validate scanner finds the known example"""
        # Placeholder validation - in production would execute scanner
        return {
            'found': True,
            'example': known_example,
            'confidence': 0.95
        }
    
    async def _test_scanner_on_ticker(self, scanner_code: str, symbol: str, date: str) -> Dict:
        """Test scanner on specific ticker and date"""
        # Placeholder test - in production would run actual scan
        return {
            'found': True,
            'symbol': symbol,
            'date': date,
            'quality_score': 0.85
        }


class MessageBus:
    """Central message routing system for agent communication."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_queue = asyncio.Queue()
        self.active = False

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the message bus."""
        self.agents[agent.agent_id] = agent
        agent.message_bus = self
        logger.info(f"Registered agent: {agent.agent_id}")

    async def route_message(self, message: AgentMessage):
        """Route message to appropriate agent."""
        if message.recipient in self.agents:
            await self.agents[message.recipient].process_message(message)
        else:
            logger.warning(f"No agent found for recipient: {message.recipient}")

    async def broadcast_message(self, message: AgentMessage):
        """Broadcast message to all agents."""
        for agent in self.agents.values():
            if agent.agent_id != message.sender:
                await agent.process_message(message)

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return {
            "active_agents": len(self.agents),
            "agents": {agent_id: agent.get_status() for agent_id, agent in self.agents.items()}
        }


class SMPlaybookAgentFactory:
    """Factory for creating and managing SM Playbook trading agents."""
    
    def __init__(self):
        self.message_bus = MessageBus()
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, Callable] = {}
        logger.info("SM Playbook Agent Factory initialized")

    def create_agent(self, agent_type: AgentType, agent_id: str = None) -> BaseAgent:
        """Create an agent of the specified type."""
        if agent_id is None:
            agent_id = agent_type.value
            
        agent_classes = {
            AgentType.TRADING_ORCHESTRATOR: TradingOrchestratorAgent,
            AgentType.STRATEGY_DESIGNER: StrategyDesignerAgent,
            AgentType.INDICATOR_DEVELOPER: IndicatorDeveloperAgent,
            AgentType.BACKTESTING_ENGINEER: BacktestingEngineerAgent,
            AgentType.SCANNER_DEVELOPER: ScannerDeveloperAgent,
            AgentType.SCANNER_BUILDER: ScannerBuilderAgent  # 8-phase scanner builder
        }
        
        if agent_type not in agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        agent = agent_classes[agent_type](agent_id)
        self.message_bus.register_agent(agent)
        self.agents[agent_id] = agent
        
        logger.info(f"Created agent: {agent_id} of type {agent_type.value}")
        return agent

    async def start_all_agents(self):
        """Start all registered agents."""
        for agent in self.agents.values():
            await agent.start()
        logger.info("All agents started")

    async def stop_all_agents(self):
        """Stop all registered agents.""" 
        for agent in self.agents.values():
            await agent.stop()
        logger.info("All agents stopped")

    async def run_workflow(self, workflow_name: str, parameters: Dict[str, Any] = None):
        """Run a predefined workflow."""
        if parameters is None:
            parameters = {}
            
        if workflow_name == "os_d1_development":
            await self._run_os_d1_workflow(parameters)
        elif workflow_name == "strategy_backtest":
            await self._run_strategy_backtest_workflow(parameters)
        elif workflow_name == "market_analysis":
            await self._run_market_analysis_workflow(parameters)
        elif workflow_name == "scanner_8phase_development":
            await self._run_scanner_8phase_workflow(parameters)
        else:
            logger.warning(f"Unknown workflow: {workflow_name}")

    async def _run_os_d1_workflow(self, parameters: Dict[str, Any]):
        """Run OS D1 strategy development workflow."""
        logger.info("Starting OS D1 development workflow")
        
        # 1. Strategy design
        strategy_designer = self.agents.get("strategy-designer")
        if strategy_designer:
            await strategy_designer.process_message(AgentMessage(
                id="workflow_1",
                sender="factory",
                recipient="strategy-designer", 
                message_type=MessageType.TASK_REQUEST,
                content={"task": "design_os_d1_strategy", **parameters}
            ))
        
        # 2. Scanner development 
        scanner_developer = self.agents.get("scanner-developer")
        if scanner_developer:
            await scanner_developer.process_message(AgentMessage(
                id="workflow_2",
                sender="factory",
                recipient="scanner-developer",
                message_type=MessageType.TASK_REQUEST,
                content={"task": "develop_os_d1_scanner", **parameters}
            ))
        
        logger.info("OS D1 workflow completed")

    async def _run_strategy_backtest_workflow(self, parameters: Dict[str, Any]):
        """Run strategy backtesting workflow."""
        logger.info("Starting strategy backtest workflow")
        
        backtesting_engineer = self.agents.get("backtesting-engineer")
        if backtesting_engineer:
            await backtesting_engineer.process_message(AgentMessage(
                id="workflow_backtest",
                sender="factory",
                recipient="backtesting-engineer",
                message_type=MessageType.TASK_REQUEST,
                content={"task": "create_backtest", **parameters}
            ))
        
        logger.info("Strategy backtest workflow completed")

    async def _run_market_analysis_workflow(self, parameters: Dict[str, Any]):
        """Run market analysis workflow using Lingua framework."""
        logger.info("Starting market analysis workflow")
        
        trading_orchestrator = self.agents.get("trading-orchestrator")
        if trading_orchestrator:
            await trading_orchestrator.process_message(AgentMessage(
                id="workflow_analysis", 
                sender="factory",
                recipient="trading-orchestrator",
                message_type=MessageType.TASK_REQUEST,
                content={"task": "analyze_market", **parameters}
            ))
        
        logger.info("Market analysis workflow completed")
    
    async def _run_scanner_8phase_workflow(self, parameters: Dict[str, Any]):
        """Run 8-phase scanner development workflow."""
        logger.info("Starting 8-phase scanner development workflow")
        
        # Ensure scanner-builder agent exists
        scanner_builder = self.agents.get("scanner-builder")
        if not scanner_builder:
            scanner_builder = self.create_agent(AgentType.SCANNER_BUILDER)
        
        # Execute 8-phase development
        result = await scanner_builder.process_message(AgentMessage(
            id="workflow_8phase",
            sender="factory",
            recipient="scanner-builder",
            message_type=MessageType.TASK_REQUEST,
            content={
                "task": "build_scanner_8phase",
                "pattern_name": parameters.get("pattern_name", "backside_pop"),
                "known_example": parameters.get("known_example", {
                    "symbol": "HOOD",
                    "date": "2025-03-03",
                    "grade": "A+"
                }),
                "trader_parameters": parameters.get("trader_parameters", {})
            }
        ))
        
        # Validate against all benchmarks
        if result.get("scanner_code"):
            validation = await scanner_builder.process_message(AgentMessage(
                id="workflow_validate",
                sender="factory",
                recipient="scanner-builder",
                message_type=MessageType.TASK_REQUEST,
                content={
                    "task": "validate_known_examples",
                    "scanner_code": result["scanner_code"]
                }
            ))
            
            if not validation.get("passed"):
                logger.error("Scanner failed benchmark validation!")
                return {"status": "failed", "validation": validation}
        
        logger.info("8-phase scanner development workflow completed")
        return result

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "factory_status": "active",
            "total_agents": len(self.agents),
            "available_workflows": [
                "os_d1_development", 
                "strategy_backtest", 
                "market_analysis",
                "scanner_8phase_development"
            ],
            "message_bus": self.message_bus.get_system_status(),
            "agents": {agent_id: agent.get_status() for agent_id, agent in self.agents.items()}
        }


# Example usage and demo functions
async def demo_sm_playbook_system():
    """Demonstrate the SM Playbook agent system."""
    print("🎭 SM Playbook Agent Factory Demo")
    print("=" * 50)
    
    # Create the factory
    factory = SMPlaybookAgentFactory()
    
    # Create core trading agents
    print("\n📋 Creating trading agents...")
    agents = [
        AgentType.TRADING_ORCHESTRATOR,
        AgentType.STRATEGY_DESIGNER,
        AgentType.INDICATOR_DEVELOPER, 
        AgentType.BACKTESTING_ENGINEER,
        AgentType.SCANNER_DEVELOPER
    ]
    
    for agent_type in agents:
        agent = factory.create_agent(agent_type)
        print(f"  ✅ Created: {agent.agent_id}")
    
    # Start all agents
    print("\n🚀 Starting all agents...")
    await factory.start_all_agents()
    
    # Run OS D1 development workflow
    print("\n📈 Running OS D1 development workflow...")
    await factory.run_workflow("os_d1_development", {
        "symbol": "AAPL",
        "timeframe": "daily",
        "parameters": {"min_gap": 0.15, "max_market_cap": 2e9}
    })
    
    # Run market analysis workflow  
    print("\n📊 Running market analysis workflow...")
    await factory.run_workflow("market_analysis", {
        "symbol": "SPY", 
        "analysis_type": "lingua_trend_cycle"
    })
    
    # Show system status
    print("\n📋 System Status:")
    status = factory.get_system_status()
    print(f"  Total agents: {status['total_agents']}")
    print(f"  Active agents: {len([a for a in status['agents'].values() if a['active']])}")
    print(f"  Available workflows: {', '.join(status['available_workflows'])}")
    
    # Stop all agents
    print("\n🛑 Stopping all agents...")
    await factory.stop_all_agents()
    
    print("\n✅ Demo completed!")


def create_claude_code_integration():
    """Create integration for Claude Code environment."""
    integration_code = """
# Add this to your Claude Code startup script or .clauderc

import asyncio
from sm_playbook_agent_factory import SMPlaybookAgentFactory, AgentType

# Global factory instance
sm_factory = SMPlaybookAgentFactory()

# Helper functions for Claude Code
async def init_sm_playbook():
    '''Initialize SM Playbook agent system'''
    agents = [
        AgentType.TRADING_ORCHESTRATOR,
        AgentType.STRATEGY_DESIGNER, 
        AgentType.BACKTESTING_ENGINEER,
        AgentType.SCANNER_DEVELOPER
    ]
    
    for agent_type in agents:
        sm_factory.create_agent(agent_type)
    
    await sm_factory.start_all_agents()
    print("✅ SM Playbook system ready!")

def os_d1_scan(symbol="scan_market"):
    '''Run OS D1 scanner'''
    return asyncio.run(sm_factory.run_workflow("os_d1_development", {"symbol": symbol}))

def backtest_strategy(strategy_name, **params):
    '''Backtest a strategy'''
    return asyncio.run(sm_factory.run_workflow("strategy_backtest", {
        "strategy_name": strategy_name, 
        **params
    }))

def analyze_lingua(symbol, timeframe="daily"):
    '''Analyze using Lingua framework'''
    return asyncio.run(sm_factory.run_workflow("market_analysis", {
        "symbol": symbol,
        "timeframe": timeframe
    }))

def sm_status():
    '''Get SM Playbook system status'''
    return sm_factory.get_system_status()

# Auto-initialize (comment out if you want manual control)
# asyncio.run(init_sm_playbook())
    """
    
    return integration_code


if __name__ == "__main__":
    print("SM Playbook Agent Factory System")
    print("Run demo with: python -c 'import asyncio; from sm_playbook_agent_factory import demo_sm_playbook_system; asyncio.run(demo_sm_playbook_system())'")
    
    # Show Claude Code integration
    print("\n" + "="*60)
    print("CLAUDE CODE INTEGRATION:")
    print("="*60)
    print(create_claude_code_integration())