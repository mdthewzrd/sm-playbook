# BMAD Knowledge Pack - Enhanced

## Quick Commands
- **Golden path**: `make demo-e2e` → writes `reports/strategy_report.md` and `trading-journal/journal.csv`
- **Engine**: backtesting.py (offline, cached synthetic data)
- **Commands**: demo, backtest, report, journal
- **Guardrails**: offline by default; no live orders; short outputs

## Core Knowledge Integration

### Lingua Trading Language
- **Primary Framework**: Complete discretionary methodology in `docs/knowledge/core/lingua_trading_language.md`
- **Trend Cycles**: 8-stage systematic approach to market movement
- **Execution**: HTF/MTF/LTF timeframe hierarchy for all trades
- **High EV Spots**: Systematic identification via custom indicators

### Systematic Strategies
- **OS D1 Setup**: Flagship small cap day one system with 70%+ win rates
- **Custom Indicators**: EMA clouds, deviation bands, trail systems
- **Backtesting Data**: Extensive historical validation across multiple years
- **Implementation**: Fully coded scanners and execution systems

### Data Infrastructure
- **Primary Source**: Polygon.io for all market data (required MCP server)
- **Backtesting**: backtesting.py engine (required MCP server)  
- **Technical Analysis**: TA-Lib for indicators (required MCP server)
- **Execution**: OsEngine for order management (required MCP server)
- **Documentation**: Notion for trade journaling (required MCP server)

### Key Files and Locations
- **Knowledge Base**: `docs/knowledge/` - Complete trading methodology
- **Strategies**: `trading-code/scanners/` - Implementation files
- **Indicators**: `trading-code/indicators/` - Custom indicator code
- **MCP Integration**: `mcp-integration/` - System architecture
- **Results**: `reports/` - Backtesting and analysis outputs

## Agent Capabilities

### Trading-Specific Agents
- **strategy-designer**: Builds new systematic strategies from Lingua concepts
- **indicator-developer**: Creates custom indicators based on discretionary patterns
- **backtesting-engineer**: Validates strategies with historical data
- **scanner-developer**: Builds stock screening systems

### Enhanced Commands
- **Lingua Analysis**: `lingua analyze [symbol] [timeframe]` - Apply full Lingua framework
- **OS D1 Scan**: `os-d1 scan [date]` - Run small cap day one scanner
- **Indicator Calculate**: `indicators apply [symbol] [period]` - Calculate custom indicators
- **Strategy Backtest**: `backtest run [strategy] [period]` - Historical validation

## Integration Points

### MCP Architecture
All trading knowledge integrates with MCP components:
- **Models**: Align with Lingua concepts (trend, structure, context)
- **Controllers**: Implement Lingua trading rules and risk management
- **Processors**: Use Lingua indicators for signal generation
- **Clients**: Connect to required data sources (Polygon, TA-Lib, etc.)

### Workflow Integration
```
Discretionary Analysis (Lingua) → Systematic Rules (Code) → 
Backtesting (Validation) → Execution (Automation) → 
Monitoring (Performance) → Journal (Documentation)
```
