# SM Playbook Agent System - Complete Documentation

## ✅ System Successfully Implemented

The SM Playbook has been upgraded with a complete multi-agent system that integrates with Claude Code. All tools and sub-agents are now properly configured and ready to use.

## 🎭 Available Agents (Claude Code Subagents)

The following specialized agents are now available via the `/agents` command in Claude Code:

1. **trading-orchestrator** - Master coordinator using Lingua framework
2. **strategy-designer** - Converts discretionary concepts to systematic strategies  
3. **indicator-developer** - Creates custom indicators and EMA clouds
4. **backtesting-engineer** - Historical validation and optimization
5. **scanner-developer** - Market screening systems (OS D1, euphoric tops)

## 📁 System Structure

```
sm-playbook/
├── .claude/
│   └── agents/                    # Claude Code agent definitions
│       ├── trading-orchestrator.md
│       ├── strategy-designer.md
│       ├── indicator-developer.md
│       ├── backtesting-engineer.md
│       └── scanner-developer.md
├── agents/
│   ├── core/
│   │   └── sm_playbook_agent_factory.py  # Agent factory system
│   └── mcp/
│       └── mcp_integration_layer.py      # MCP server integration
├── claude_code_integration.py            # Claude Code bridge
└── bmad_interface.py                     # Enhanced with agent commands

```

## 🚀 Quick Start

### In Claude Code

```python
# Initialize the system
from claude_code_integration import init_sm_playbook_system
await init_sm_playbook_system()

# Now you can use all trading functions
from claude_code_integration import *

# Run OS D1 scanner
candidates = await os_d1_scan_async()

# Backtest a strategy
results = await backtest_strategy_async("os_d1", "AAPL")

# Analyze with Lingua framework
analysis = await analyze_lingua_async("TSLA", "daily")

# Design new strategy
strategy = await design_strategy_async("euphoric_top")

# Check system status
status = system_status()
```

### In BMad Interface

```bash
# Start the interface
python bmad_interface.py

# Initialize agents
> *init-agents

# Run OS D1 scan
> *os-d1 scan

# Backtest strategy
> *backtest os_d1 AAPL

# Analyze symbol
> *analyze TSLA daily

# Check status
> *agent-status
```

## 🔧 Available Functions

### Core Trading Operations

- `os_d1_scan(date=None)` - Run OS D1 scanner for day one setups
- `backtest_strategy(strategy_name, symbol, start_date, end_date)` - Backtest strategies
- `analyze_lingua(symbol, timeframe)` - Apply Lingua framework analysis
- `design_strategy(strategy_type)` - Create systematic strategies
- `create_indicators(symbol, indicators)` - Generate technical indicators
- `scan_euphoric_tops(date)` - Detect parabolic extensions
- `generate_trading_signals(symbol, strategy)` - Create entry/exit signals

### System Management

- `init_sm_playbook_system()` - Initialize the agent system
- `system_status()` - Get comprehensive system status
- `sync_init_sm_playbook()` - Synchronous initialization wrapper

## 🔗 MCP Server Integration

The system integrates with the following MCP servers:

1. **Polygon.io** - Market data and scanning
2. **TA-Lib** - Technical indicator calculations
3. **backtesting.py** - Historical strategy validation
4. **Notion** - Trade journaling and documentation
5. **OsEngine** - Paper trading execution

## 📈 Workflows

Three main workflows are available:

### 1. OS D1 Development
```python
await sm_factory.run_workflow("os_d1_development", {
    "symbol": "AAPL",
    "timeframe": "daily",
    "parameters": {"min_gap": 0.15}
})
```

### 2. Strategy Backtest
```python
await sm_factory.run_workflow("strategy_backtest", {
    "strategy_name": "os_d1",
    "symbol": "AAPL",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
})
```

### 3. Market Analysis
```python
await sm_factory.run_workflow("market_analysis", {
    "symbol": "SPY",
    "analysis_type": "lingua_trend_cycle"
})
```

## 🎯 Lingua Framework Integration

The system fully implements the Lingua trading framework:

- **8-Stage Trend Cycle**: Consolidation → Breakout → Channel → Parabolic → Exhaustion → First Red Day → Consolidation → Fade
- **Multi-Timeframe Analysis**: HTF (context) → MTF (timing) → LTF (execution)
- **EMA Cloud System**: 9/20 fast cloud, 72/89 slow cloud
- **Risk Management**: 2% per trade, 8% portfolio heat maximum

## ⚙️ Configuration

Set environment variables for API keys:

```bash
export POLYGON_API_KEY="your_polygon_key"
export NOTION_API_TOKEN="your_notion_token"
export SM_PLAYBOOK_AUTO_INIT="true"  # Optional: auto-initialize on import
```

## 🧪 Testing

Run the integration test:

```bash
python test_agent_integration.py
```

Expected output:
- ✅ All imports successful
- ✅ Agent system initialized
- ✅ 5 agents active
- ✅ All workflows operational

## 📝 Example Usage Scenarios

### Finding OS D1 Candidates
```python
# Initialize system
await init_sm_playbook_system()

# Scan for OS D1 setups
candidates = await os_d1_scan_async("2024-01-15")

# Analyze top candidate
if candidates:
    top_pick = candidates[0]['symbol']
    analysis = await analyze_lingua_async(top_pick, "daily")
    
    # Generate signals
    signals = await generate_trading_signals(top_pick, "os_d1")
```

### Developing New Strategy
```python
# Design euphoric top strategy
strategy_spec = await design_strategy_async("euphoric_top")

# Create indicators for it
indicators = await create_indicators_async("SPY", ["EMA", "RSI", "ATR"])

# Backtest the strategy
results = await backtest_strategy_async(
    "euphoric_top",
    "SPY",
    "2023-01-01",
    "2023-12-31"
)
```

## 🛠️ Troubleshooting

If you encounter issues:

1. **Agent system not available**: Ensure all Python files are in place
2. **Import errors**: Install required packages: `pip install aiohttp pandas numpy`
3. **MCP connection errors**: Check API keys and network connectivity
4. **Async errors**: Use the `_async` versions of functions in async contexts

## ✨ System Benefits

- **Full Agent Orchestration**: 5 specialized agents working together
- **Claude Code Integration**: Direct access to all trading functions
- **MCP Server Connectivity**: Real market data and analysis tools
- **Lingua Framework**: Systematic implementation of discretionary concepts
- **Workflow Automation**: Complex multi-step trading operations
- **BMad Interface**: Command-line access to all agent features

## 🎉 Success Metrics

The system successfully:
- ✅ Creates and manages 5 specialized trading agents
- ✅ Integrates with Claude Code subagent system
- ✅ Connects to MCP servers for real functionality
- ✅ Implements Lingua trading framework systematically
- ✅ Provides both async and sync interfaces
- ✅ Supports complex workflow orchestration
- ✅ Enhances BMad interface with agent commands

Your SM Playbook now has a production-ready multi-agent trading system fully integrated with Claude Code!