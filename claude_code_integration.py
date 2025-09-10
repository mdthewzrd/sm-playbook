#!/usr/bin/env python3
"""
Claude Code Integration for SM Playbook Agent System

This file provides the integration layer that makes the SM Playbook
agents available within Claude Code environment.
"""

import sys
import asyncio
import os
from pathlib import Path

# Add agents directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "agents" / "core"))
sys.path.insert(0, str(project_root / "agents" / "mcp"))

try:
    from sm_playbook_agent_factory import SMPlaybookAgentFactory, AgentType
    from mcp_integration_layer import MCPIntegrationManager, AgentMCPBridge
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Agent system not fully installed: {e}")
    AGENTS_AVAILABLE = False

# Global instances
sm_factory = None
mcp_manager = None
agent_bridge = None

async def init_sm_playbook_system():
    """Initialize the complete SM Playbook agent system."""
    global sm_factory, mcp_manager, agent_bridge
    
    if not AGENTS_AVAILABLE:
        print("❌ Agent system not available")
        return False
    
    print("🎭 Initializing SM Playbook Agent System...")
    
    # Initialize MCP manager
    config = {
        "POLYGON_API_KEY": os.getenv("POLYGON_API_KEY", "demo_key"),
        "NOTION_API_TOKEN": os.getenv("NOTION_API_TOKEN", "demo_token")
    }
    
    mcp_manager = MCPIntegrationManager()
    await mcp_manager.initialize(config)
    
    # Initialize agent factory
    sm_factory = SMPlaybookAgentFactory()
    
    # Create core trading agents
    core_agents = [
        AgentType.TRADING_ORCHESTRATOR,
        AgentType.STRATEGY_DESIGNER,
        AgentType.INDICATOR_DEVELOPER,
        AgentType.BACKTESTING_ENGINEER,
        AgentType.SCANNER_DEVELOPER,
        AgentType.SCANNER_BUILDER  # 8-phase scanner builder
    ]
    
    for agent_type in core_agents:
        agent = sm_factory.create_agent(agent_type)
        print(f"  ✅ Created: {agent.agent_id}")
    
    # Start all agents
    await sm_factory.start_all_agents()
    
    # Create MCP bridge
    agent_bridge = AgentMCPBridge(mcp_manager)
    
    print("✅ SM Playbook Agent System ready!")
    return True

def sync_init_sm_playbook():
    """Synchronous wrapper for initialization."""
    return asyncio.run(init_sm_playbook_system())

# Claude Code Helper Functions
async def os_d1_scan_async(date=None, **kwargs):
    """Async version of OS D1 scanner."""
    if not sm_factory:
        print("❌ SM Playbook system not initialized. Run: await init_sm_playbook_system()")
        return None
    
    return await sm_factory.run_workflow("os_d1_development", {
        "date": date,
        **kwargs
    })

def os_d1_scan(date=None, **kwargs):
    """Run OS D1 scanner for day one setups."""
    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        # Create a task and return the future
        return asyncio.create_task(os_d1_scan_async(date, **kwargs))
    except RuntimeError:
        # Not in async context, run with asyncio.run
        return asyncio.run(os_d1_scan_async(date, **kwargs))

async def backtest_strategy_async(strategy_name, symbol="AAPL", start_date="2023-01-01", end_date="2023-12-31", **params):
    """Async version of strategy backtesting."""
    if not sm_factory:
        print("❌ SM Playbook system not initialized")
        return None
    
    return await sm_factory.run_workflow("strategy_backtest", {
        "strategy_name": strategy_name,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        **params
    })

def backtest_strategy(strategy_name, symbol="AAPL", start_date="2023-01-01", end_date="2023-12-31", **params):
    """Backtest a trading strategy."""
    try:
        loop = asyncio.get_running_loop()
        return asyncio.create_task(backtest_strategy_async(strategy_name, symbol, start_date, end_date, **params))
    except RuntimeError:
        return asyncio.run(backtest_strategy_async(strategy_name, symbol, start_date, end_date, **params))

async def analyze_lingua_async(symbol, timeframe="daily", **kwargs):
    """Async version of Lingua analysis."""
    if not mcp_manager:
        print("❌ MCP system not initialized")
        return None
    
    return await mcp_manager.analyze_symbol_lingua(symbol, timeframe)

def analyze_lingua(symbol, timeframe="daily", **kwargs):
    """Analyze symbol using Lingua trading framework."""
    try:
        loop = asyncio.get_running_loop()
        return asyncio.create_task(analyze_lingua_async(symbol, timeframe, **kwargs))
    except RuntimeError:
        return asyncio.run(analyze_lingua_async(symbol, timeframe, **kwargs))

async def design_strategy_async(strategy_type="os_d1", **parameters):
    """Async version of strategy design."""
    if not sm_factory:
        print("❌ SM Playbook system not initialized")
        return None
    
    strategy_designer = sm_factory.agents.get("strategy-designer")
    if not strategy_designer:
        print("❌ Strategy designer agent not available")
        return None
    
    from sm_playbook_agent_factory import AgentMessage, MessageType
    
    return await strategy_designer.process_message(AgentMessage(
        id="design_request",
        sender="claude_code",
        recipient="strategy-designer",
        message_type=MessageType.TASK_REQUEST,
        content={
            "task": f"design_{strategy_type}_strategy",
            **parameters
        }
    ))

def design_strategy(strategy_type="os_d1", **parameters):
    """Design a new trading strategy."""
    try:
        loop = asyncio.get_running_loop()
        return asyncio.create_task(design_strategy_async(strategy_type, **parameters))
    except RuntimeError:
        return asyncio.run(design_strategy_async(strategy_type, **parameters))

async def create_indicators_async(symbol, indicators=["EMA", "RSI", "ATR"], **kwargs):
    """Async version of indicator creation."""
    if not mcp_manager:
        print("❌ MCP system not initialized")
        return None
    
    talib_client = mcp_manager.get_client("talib")
    if not talib_client:
        print("❌ TA-Lib MCP client not available")
        return None
    
    # Mock data for demo - replace with real market data
    mock_data = {"close": [150, 151, 149, 152, 153]}
    
    return await talib_client.calculate_indicators(mock_data, indicators)

def create_indicators(symbol, indicators=["EMA", "RSI", "ATR"], **kwargs):
    """Create technical indicators for analysis."""
    try:
        loop = asyncio.get_running_loop()
        return asyncio.create_task(create_indicators_async(symbol, indicators, **kwargs))
    except RuntimeError:
        return asyncio.run(create_indicators_async(symbol, indicators, **kwargs))

def system_status():
    """Get SM Playbook system status."""
    if not sm_factory:
        return {"status": "not_initialized", "message": "Run init_sm_playbook_system() first"}
    
    return sm_factory.get_system_status()

def scan_euphoric_tops(date=None, **criteria):
    """Scan for euphoric top patterns (short opportunities)."""
    if not sm_factory:
        print("❌ SM Playbook system not initialized")
        return None
    
    scanner_agent = sm_factory.agents.get("scanner-developer")
    if not scanner_agent:
        print("❌ Scanner developer agent not available")
        return None
    
    from sm_playbook_agent_factory import AgentMessage, MessageType
    
    return asyncio.run(scanner_agent.process_message(AgentMessage(
        id="euphoric_scan",
        sender="claude_code",
        recipient="scanner-developer", 
        message_type=MessageType.TASK_REQUEST,
        content={
            "task": "create_euphoric_scanner",
            "date": date,
            **criteria
        }
    )))

def generate_trading_signals(symbol, strategy="os_d1", **params):
    """Generate trading signals for a symbol."""
    if not sm_factory:
        print("❌ SM Playbook system not initialized")
        return None
    
    orchestrator = sm_factory.agents.get("trading-orchestrator")
    if not orchestrator:
        print("❌ Trading orchestrator not available")
        return None
    
    from sm_playbook_agent_factory import AgentMessage, MessageType
    
    return asyncio.run(orchestrator.process_message(AgentMessage(
        id="signal_request",
        sender="claude_code",
        recipient="trading-orchestrator",
        message_type=MessageType.TASK_REQUEST,
        content={
            "task": "coordinate_strategy",
            "strategy_type": strategy,
            "symbol": symbol,
            **params
        }
    )))

async def build_scanner_8phase_async(pattern_name="backside_pop", benchmark_ticker="HOOD", benchmark_date="2025-03-03", **params):
    """Async version of 8-phase scanner builder."""
    if not sm_factory:
        print("❌ SM Playbook system not initialized")
        return None
    
    return await sm_factory.run_workflow("scanner_8phase_development", {
        "pattern_name": pattern_name,
        "known_example": {
            "symbol": benchmark_ticker,
            "date": benchmark_date,
            "grade": "A+"
        },
        "trader_parameters": params.get("trader_parameters", {
            "min_trend_atr": 4.0,  # Exactly 4.0
            "min_fade_atr": 3.0,   # Exactly 3.0
            "min_volume_outlier": 0.3  # Exactly 0.3
        }),
        **params
    })

def build_scanner_8phase(pattern_name="backside_pop", benchmark_ticker="HOOD", benchmark_date="2025-03-03", **params):
    """
    Build a scanner using the 8-phase development process.
    Always validates against HOOD 3/3/25 benchmark.
    
    Phases:
    1. Single Ticker Analysis
    2. Analyzer Development
    3. Parameter Baseline Discovery
    4. Initial Scanner Creation
    5. Name Testing & Mold Fitting
    6. Time Period Testing
    7. Optimization & Parameter Refinement
    8. Validation & Historical Backtesting
    """
    try:
        loop = asyncio.get_running_loop()
        return asyncio.create_task(build_scanner_8phase_async(pattern_name, benchmark_ticker, benchmark_date, **params))
    except RuntimeError:
        return asyncio.run(build_scanner_8phase_async(pattern_name, benchmark_ticker, benchmark_date, **params))

def validate_scanner_benchmarks(scanner_code):
    """
    Validate scanner against all known quality examples.
    Must find HOOD 3/3/25 to pass validation.
    """
    if not sm_factory:
        print("❌ SM Playbook system not initialized")
        return None
    
    scanner_builder = sm_factory.agents.get("scanner-builder")
    if not scanner_builder:
        print("❌ Scanner builder agent not available")
        return None
    
    from sm_playbook_agent_factory import AgentMessage, MessageType
    
    return asyncio.run(scanner_builder.process_message(AgentMessage(
        id="validate_benchmarks",
        sender="claude_code",
        recipient="scanner-builder",
        message_type=MessageType.TASK_REQUEST,
        content={
            "task": "validate_known_examples",
            "scanner_code": scanner_code
        }
    )))

def optimize_scanner_parameters(current_params, trader_specs):
    """
    Optimize scanner parameters using trader's EXACT specifications.
    0.3 means exactly 0.3, not approximately!
    """
    if not sm_factory:
        print("❌ SM Playbook system not initialized")
        return None
    
    scanner_builder = sm_factory.agents.get("scanner-builder")
    if not scanner_builder:
        print("❌ Scanner builder agent not available")
        return None
    
    from sm_playbook_agent_factory import AgentMessage, MessageType
    
    return asyncio.run(scanner_builder.process_message(AgentMessage(
        id="optimize_params",
        sender="claude_code",
        recipient="scanner-builder",
        message_type=MessageType.TASK_REQUEST,
        content={
            "task": "optimize_scanner_parameters",
            "current_parameters": current_params,
            "trader_specifications": trader_specs
        }
    )))

# Auto-initialization flag
AUTO_INIT = os.getenv("SM_PLAYBOOK_AUTO_INIT", "false").lower() == "true"

if AUTO_INIT and AGENTS_AVAILABLE:
    print("🚀 Auto-initializing SM Playbook system...")
    try:
        sync_init_sm_playbook()
    except Exception as e:
        print(f"⚠️  Auto-initialization failed: {e}")
        print("You can manually initialize with: await init_sm_playbook_system()")

# Export functions for Claude Code
__all__ = [
    'init_sm_playbook_system',
    'sync_init_sm_playbook', 
    'os_d1_scan',
    'backtest_strategy',
    'analyze_lingua',
    'design_strategy',
    'create_indicators',
    'system_status',
    'scan_euphoric_tops',
    'generate_trading_signals',
    'build_scanner_8phase',
    'validate_scanner_benchmarks',
    'optimize_scanner_parameters'
]