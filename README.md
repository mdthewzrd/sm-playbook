# 🎭 SM Playbook - BMad Trading System

**Breakthrough Method of Agile AI-driven Development for Trading (BMAT)**

A comprehensive trading system built using the BMad methodology, featuring specialized AI agents, automated workflows, and advanced risk management.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Run the complete system setup
python setup_bmad_system.py

# Set up Python virtual environment
./setup_venv.sh

# Copy environment template and configure
cp .env.template .env
# Edit .env with your API keys
```

### 2. Start Trading System
```bash
# Full startup (with data updates)
./start_bmad.sh

# Quick start with trading orchestrator
./quick_start.sh

# Direct interface access
python bmad_interface.py
```

### 3. Basic Commands
```bash
BMad> *help                    # Show all commands
BMad> *agent trading-orchestrator    # Activate trading agent
BMad> *workflow strategy-development # Start strategy workflow
BMad> *status                  # Show system status
```

## 📁 Project Structure

```
sm-playbook/
├── .bmad-core/                 # BMad core system
│   ├── agents/                 # AI trading agents
│   ├── workflows/              # Trading workflows
│   ├── data/                   # Knowledge base & configs
│   └── agent-teams/            # Agent team definitions
├── trading-code/               # Core trading algorithms
│   ├── indicators/             # Technical indicators
│   ├── backtesting/            # Backtesting framework
│   ├── execution/              # Trade execution engine
│   └── monitoring/             # Performance monitoring
├── config/                     # System configuration
├── scripts/                    # Integration scripts
├── pipelines/                  # Data processing pipelines
├── playbook/                   # Trading strategies (your existing)
├── trading-data/               # Market data (your existing)
├── backtest-results/           # Test results (your existing)
├── trading-journal/            # Trading journal (your existing)
├── trading-logs/               # System logs (your existing)
└── bmad_interface.py           # Main system interface
```

## 🤖 Available Agents

### Core Agents
- **`bmad-orchestrator`** - Master system coordinator
- **`trading-orchestrator`** - Trading system manager & playbook curator

### Specialized Agents
- **`strategy-designer`** - Formalizes trading strategies from concepts
- **`backtesting-engineer`** - Validates strategies through comprehensive testing
- **`execution-engineer`** - Manages trade execution and risk controls
- **`indicator-developer`** - Develops custom technical indicators

## 🔄 Available Workflows

1. **`strategy-development`** - Complete strategy development lifecycle
   - Formalize trading concepts → Implement indicators → Backtest → Validate → Add to playbook

2. **`signal-generation`** - Generate and validate trading signals
   - Market analysis → Signal generation → Risk validation → Execution preparation

3. **`market-analysis`** - Comprehensive market condition analysis
   - Data collection → Technical analysis → Regime identification → Insights generation

4. **`performance-review`** - Strategy optimization and performance review
   - Data collection → Performance analysis → Optimization → Implementation

## 🛠 Key Features

### Trading Capabilities
- **Custom Indicator Framework** - Lingua ATR Bands, RSI Gradient, Multi-timeframe Momentum
- **Advanced Backtesting** - Comprehensive performance analysis with statistical validation
- **Risk Management** - Portfolio-level risk controls with real-time monitoring
- **Trade Execution** - Paper and live trading with comprehensive order management

### BMad Methodology
- **Agent-Based Architecture** - Specialized AI agents for different trading aspects
- **Workflow Automation** - Structured processes for strategy development and execution
- **Knowledge Integration** - Comprehensive trading knowledge base and pattern library
- **Command Interface** - Natural language commands with * prefix system

### Data Integration
- **Market Data** - Yahoo Finance, Alpha Vantage, and other sources
- **Broker Integration** - Alpaca, Interactive Brokers support
- **Historical Analysis** - Comprehensive backtesting with multiple data sources
- **Real-time Monitoring** - Live performance tracking and alerting

## 📊 Usage Examples

### Develop a New Strategy
```bash
BMad> *agent strategy-designer
BMad[strategy-designer]> *formalize mean-reversion-rsi
BMad[strategy-designer]> *pattern-define oversold-bounce
BMad[strategy-designer]> *exit
BMad> *workflow strategy-development
```

### Generate Trading Signals
```bash
BMad> *agent trading-orchestrator
BMad[trading-orchestrator]> *signal-generate SPY 1D
BMad[trading-orchestrator]> *playbook-review
BMad[trading-orchestrator]> *workflow-start signal-generation
```

### Analyze Performance
```bash
BMad> *workflow performance-review
# Or directly with backtesting
BMad> *agent backtesting-engineer
BMad[backtesting-engineer]> *analyze-performance my-strategy
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Market Data APIs
ALPHA_VANTAGE_API_KEY=your_key
YAHOO_FINANCE_API_KEY=your_key

# Broker APIs  
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Risk Management
MAX_PORTFOLIO_RISK=0.08
MAX_POSITION_RISK=0.02
MAX_DAILY_LOSS=5000
```

### System Configuration (config/bmad_config.yaml)
```yaml
execution:
  default_mode: paper
  risk_limits:
    max_portfolio_risk: 0.08
    max_position_risk: 0.02
    max_daily_loss: 5000

data:
  market_data_path: trading-data
  playbook_path: playbook
  results_path: backtest-results
```

## 🔍 Monitoring & Reporting

### Real-time Monitoring
- Portfolio performance tracking
- Risk metric monitoring
- Trade execution monitoring
- System health monitoring

### Automated Reports
- Daily trading reports
- Weekly performance summaries
- Strategy analysis reports
- Risk assessment reports

### Alerting System
- Performance threshold alerts
- Risk limit breach notifications
- System status alerts
- Custom alert configurations

## 🛡️ Risk Management

### Portfolio Level
- Maximum portfolio risk (8% default)
- Daily loss limits ($5,000 default)
- Drawdown monitoring and alerts
- Correlation exposure limits

### Position Level
- Position size limits (2% risk per trade)
- Stop loss management
- Position concentration limits
- Sector exposure limits

### Execution Level
- Pre-trade risk checks
- Real-time risk monitoring
- Emergency position closure
- Order validation and limits

## 📈 Technical Indicators

### Standard Indicators
- Moving Averages (SMA, EMA, WMA)
- RSI, MACD, Bollinger Bands
- ATR, Stochastic, ADX

### Custom Lingua Indicators
- **Lingua ATR Bands** - Volatility-adjusted bands using ATR
- **RSI Gradient** - Rate of change of RSI for momentum detection
- **Multi-timeframe Momentum** - Composite momentum across timeframes
- **Volume Profile** - Volume distribution analysis

## 🔄 Data Pipeline

### Market Data Sources
- Yahoo Finance (free)
- Alpha Vantage (API key required)
- Interactive Brokers (professional)
- Alpaca Markets (commission-free)

### Data Processing
- Automated data updates
- Historical data management
- Real-time data streaming
- Data quality validation

## 🚨 Troubleshooting

### Common Issues

1. **Virtual Environment Issues**
   ```bash
   # Recreate virtual environment
   rm -rf bmad_env
   ./setup_venv.sh
   ```

2. **Import Errors**
   ```bash
   # Ensure you're in the virtual environment
   source bmad_env/bin/activate
   # Reinstall requirements
   pip install -r requirements.txt
   ```

3. **Market Data Issues**
   ```bash
   # Update market data manually
   python scripts/update_market_data.py
   # Check API keys in .env file
   ```

4. **Permission Errors**
   ```bash
   # Make scripts executable
   chmod +x *.sh
   chmod +x scripts/*.py
   ```

### Log Files
- **System logs**: `trading-logs/bmad.log`
- **Trading logs**: `trading-logs/trading.log`
- **Error logs**: `trading-logs/error.log`

## 🔗 Integration

### Existing Directory Integration
The system automatically integrates with your existing directories:
- `playbook/` - Your trading strategies
- `trading-data/` - Historical market data
- `backtest-results/` - Backtesting results
- `trading-journal/` - Trading journal entries
- `trading-logs/` - System and trading logs

### Broker Integration
- **Paper Trading**: Recommended for testing (Alpaca Paper Trading)
- **Live Trading**: Production trading (requires funded account)
- **Backtesting**: Historical simulation using your data

## 📚 Learning Resources

### BMad Methodology
- Agent-based development principles
- Workflow automation patterns
- Knowledge base management
- Command interface usage

### Trading System Components
- Strategy development process
- Risk management principles
- Performance analysis methods
- Trade execution best practices

## 🤝 Contributing

This is your personal trading system. Customize it by:
1. Adding new trading strategies to `playbook/`
2. Creating custom indicators in `trading-code/indicators/`
3. Modifying risk parameters in `config/`
4. Adding new data sources to `scripts/`

## 📞 Support

### Getting Help
```bash
# In BMad interface
BMad> *help
BMad> *kb-mode          # Load knowledge base
BMad> *chat-mode        # Natural language help

# System status
BMad> *status
```

### Log Analysis
```bash
# View recent logs
tail -f trading-logs/bmad.log
tail -f trading-logs/trading.log
```

## 🎯 Next Steps

1. **Configure APIs** - Add your market data and broker API keys
2. **Import Strategies** - Move existing strategies to playbook format
3. **Paper Trading** - Test with paper trading before live
4. **Customize Risk** - Adjust risk parameters for your comfort level
5. **Monitor Performance** - Set up alerts and reporting schedules

---

## 🏆 Happy Trading!

Your SM Playbook BMad Trading System is now ready. The system combines the power of AI agents with systematic trading methodology to help you develop, test, and execute trading strategies effectively.

**Remember**: Always start with paper trading and thoroughly test strategies before deploying real capital.

---

*Built with BMad™ - Breakthrough Method of Agile AI-driven Development*