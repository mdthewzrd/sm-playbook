# MCP Integration System for SM Playbook

A comprehensive Model Context Protocol (MCP) integration system that connects your SM Playbook BMad Trading System with multiple MCP servers for enhanced trading capabilities.

## 🚀 Quick Start

### Installation

```typescript
import { setupDevelopmentEnvironment, examples } from './mcp-integration';

// Quick development setup
const system = await setupDevelopmentEnvironment();

// Run basic usage example
await examples.basicUsage();
```

### Environment Variables

```bash
# Notion API Token
NOTION_API_TOKEN=your_notion_api_token_here

# Polygon.io API Key
POLYGON_API_KEY=your_polygon_api_key_here

# Optional: Custom MCP server configurations
# (The system uses your existing MCP server configurations)
```

## 🏗️ Architecture Overview

The MCP Integration System consists of several key components:

### Core Components

1. **MCPManager** - Central management for all MCP server connections
2. **MCP Clients** - Specialized clients for each MCP server
3. **BMadAgentIntegrator** - Integration layer connecting MCP clients to BMad agents
4. **MCPTestFramework** - Comprehensive testing suite
5. **MCPSystem** - Main system orchestrator

### MCP Clients

#### 🗂️ NotionClient
- Interface with Notion API for strategy documentation
- Create and update strategy pages
- Search and retrieve trading strategies
- Sync backtest results automatically

#### 📊 BacktestingClient  
- Interface with backtesting.py for strategy testing
- Run comprehensive backtest simulations
- Performance metrics calculation
- Strategy optimization and comparison

#### 📈 TALibClient
- Technical analysis using TA-Lib indicators
- Custom indicator development
- EMA cloud analysis (9/20 and 72/89)
- Pattern recognition and signal generation

#### ⚡ OsEngineClient
- Trade execution and order management
- Position tracking and portfolio management
- Risk management and monitoring
- Integration with multiple exchanges

#### 📡 PolygonClient
- Real-time and historical market data
- Market scanner functionality
- WebSocket support for live data
- News and fundamental data access

## 🔧 Configuration

### Development Configuration

```typescript
import { setupDevelopmentEnvironment } from './mcp-integration';

const system = await setupDevelopmentEnvironment();
// Automatically configured for development with:
// - Enhanced logging
// - Test execution on startup
// - HTML test reports
// - 60-second health checks
```

### Production Configuration

```typescript
import { setupProductionEnvironment } from './mcp-integration';

const system = await setupProductionEnvironment();
// Optimized for production with:
// - Live trading mode
// - Tighter risk limits
// - Higher concurrency
// - Minimal logging
```

### Custom Configuration

```typescript
import { createMCPSystem } from './mcp-integration';

const system = await createMCPSystem({
  manager: {
    healthCheckInterval: 30000,
    maxRetryAttempts: 3,
    enableMetrics: true
  },
  clients: {
    osengine: {
      tradingMode: 'live',
      enableAutoExecution: true,
      riskLimits: {
        maxDrawdown: 0.08,
        maxPositions: 3
      }
    }
  },
  testing: {
    enabled: true,
    runOnStartup: false
  }
});
```

## 🤖 BMad Agent Integration

The system integrates with all your BMad agents:

### Trading Orchestrator
```typescript
const response = await integrator.processAgentRequest({
  action: 'generate-signals',
  parameters: {
    symbol: 'AAPL',
    timeframe: '1Day',
    autoExecute: false
  },
  context: {
    agentId: 'trading-orchestrator',
    sessionId: 'session-1',
    parameters: {},
    mcpClients: {}
  }
});
```

### Strategy Designer
```typescript
const response = await integrator.processAgentRequest({
  action: 'design-strategy',
  parameters: {
    strategyName: 'RSI Mean Reversion',
    indicators: ['RSI', 'EMA'],
    entryRules: ['RSI < 30'],
    exitRules: ['RSI > 70'],
    validateWithBacktest: true
  },
  context: {
    agentId: 'strategy-designer',
    sessionId: 'session-1',
    parameters: {},
    mcpClients: {}
  }
});
```

### Backtesting Engineer
```typescript
const response = await integrator.processAgentRequest({
  action: 'run-backtest',
  parameters: {
    strategy: strategyDefinition,
    symbol: 'SPY',
    timeframe: '1Day',
    initialCapital: 100000
  },
  context: {
    agentId: 'backtesting-engineer',
    sessionId: 'session-1',
    parameters: {},
    mcpClients: {}
  }
});
```

### Execution Engineer
```typescript
const response = await integrator.processAgentRequest({
  action: 'execute-trades',
  parameters: {
    signals: tradingSignals,
    riskManagement: {
      maxPositions: 5,
      positionSize: 0.02
    }
  },
  context: {
    agentId: 'execution-engineer',
    sessionId: 'session-1',
    parameters: {},
    mcpClients: {}
  }
});
```

### Indicator Developer
```typescript
const response = await integrator.processAgentRequest({
  action: 'develop-indicator',
  parameters: {
    indicatorName: 'Custom RSI Gradient',
    formula: 'RSI_GRADIENT',
    parameters: { period: 14, smoothing: 3 }
  },
  context: {
    agentId: 'indicator-developer',
    sessionId: 'session-1',
    parameters: {},
    mcpClients: {}
  }
});
```

## 📈 Usage Examples

### Complete Strategy Development Workflow

```typescript
import { setupDevelopmentEnvironment } from './mcp-integration';

async function strategyWorkflow() {
  const system = await setupDevelopmentEnvironment();
  const integrator = system.getIntegrator();

  // 1. Design Strategy
  const strategy = await integrator.processAgentRequest({
    action: 'design-strategy',
    parameters: {
      strategyName: 'EMA Cloud Strategy',
      description: 'EMA 9/20 cloud with RSI filter',
      indicators: ['EMA_9', 'EMA_20', 'RSI'],
      entryRules: ['EMA_9 > EMA_20', 'RSI < 70'],
      exitRules: ['EMA_9 < EMA_20', 'RSI > 30']
    },
    context: { agentId: 'strategy-designer', sessionId: 'workflow', parameters: {}, mcpClients: {} }
  });

  // 2. Run Comprehensive Backtest
  const backtest = await integrator.processAgentRequest({
    action: 'run-backtest',
    parameters: {
      strategy: strategy.data.strategy,
      symbol: 'SPY',
      timeframe: '1Day',
      initialCapital: 100000,
      startDate: '2023-01-01',
      endDate: '2024-01-01'
    },
    context: { agentId: 'backtesting-engineer', sessionId: 'workflow', parameters: {}, mcpClients: {} }
  });

  // 3. Optimize Parameters
  const optimization = await integrator.processAgentRequest({
    action: 'optimize-parameters',
    parameters: {
      strategy: strategy.data.strategy,
      parameterRanges: {
        rsi_period: { min: 10, max: 20, step: 2 },
        ema_fast: { min: 5, max: 15, step: 2 },
        ema_slow: { min: 15, max: 25, step: 2 }
      },
      optimizationMetric: 'sharpe'
    },
    context: { agentId: 'backtesting-engineer', sessionId: 'workflow', parameters: {}, mcpClients: {} }
  });

  // 4. Generate Trading Signals
  const signals = await integrator.processAgentRequest({
    action: 'generate-signals',
    parameters: {
      symbol: 'SPY',
      strategy: optimization.data.bestStrategy,
      timeframe: '1Day'
    },
    context: { agentId: 'trading-orchestrator', sessionId: 'workflow', parameters: {}, mcpClients: {} }
  });

  console.log('Strategy Development Complete:', {
    strategy: strategy.success,
    backtest: backtest.success,
    optimization: optimization.success,
    signals: signals.success
  });

  await system.shutdown();
}
```

### Real-time Trading Pipeline

```typescript
async function realTimeTradingPipeline() {
  const system = await setupProductionEnvironment();
  const integrator = system.getIntegrator();

  // Monitor and execute trades in real-time
  setInterval(async () => {
    try {
      // 1. Generate fresh signals
      const signals = await integrator.processAgentRequest({
        action: 'generate-signals',
        parameters: {
          symbol: 'SPY',
          timeframe: '5m',
          limit: 20,
          autoExecute: true
        },
        context: { agentId: 'trading-orchestrator', sessionId: 'live-trading', parameters: {}, mcpClients: {} }
      });

      // 2. Manage existing positions
      const positions = await integrator.processAgentRequest({
        action: 'monitor-positions',
        parameters: {
          updateStopLoss: true,
          trailStops: true
        },
        context: { agentId: 'execution-engineer', sessionId: 'live-trading', parameters: {}, mcpClients: {} }
      });

      // 3. Risk management check
      const risk = await integrator.processAgentRequest({
        action: 'manage-risk',
        parameters: {
          maxDrawdown: 0.05,
          maxPositions: 3
        },
        context: { agentId: 'execution-engineer', sessionId: 'live-trading', parameters: {}, mcpClients: {} }
      });

      console.log('Trading Pipeline Status:', {
        signals: signals.success,
        positions: positions.success,
        risk: risk.success
      });

    } catch (error) {
      console.error('Trading pipeline error:', error);
    }
  }, 60000); // Run every minute
}
```

## 🧪 Testing

### Running Tests

```typescript
import { setupDevelopmentEnvironment } from './mcp-integration';

const system = await setupDevelopmentEnvironment();
const testFramework = system.getTestFramework();

// Run all tests
const results = await testFramework.runAllTests();

// Generate test report
const report = testFramework.generateReport('html');
console.log('Test Report Generated:', report);
```

### Test Categories

1. **Unit Tests** - Individual MCP client functionality
2. **Integration Tests** - BMad agent integration
3. **Performance Tests** - Load and performance testing  
4. **End-to-End Tests** - Complete workflow testing

## 📊 Monitoring & Diagnostics

### System Status

```typescript
const system = await setupDevelopmentEnvironment();

// Get current status
const status = system.getStatus();
console.log('System Status:', {
  initialized: status.initialized,
  clients: status.clientStatuses,
  errors: status.errors.length
});

// Run full diagnostics
const diagnostics = await system.runDiagnostics();
console.log('System Health:', diagnostics);
```

### Health Monitoring

The system automatically monitors:
- MCP client connectivity
- Response times and performance
- Error rates and failures
- Resource utilization
- Trading system health

## 🔒 Security & Risk Management

### Built-in Protections

1. **Trading Mode Controls** - Paper vs Live trading modes
2. **Risk Limits** - Position size, drawdown, and exposure limits
3. **Connection Security** - Secure MCP server connections
4. **API Key Management** - Environment-based key management
5. **Error Handling** - Comprehensive error recovery

### Risk Configuration

```typescript
const riskLimits = {
  stopLoss: 0.02,        // 2% stop loss
  takeProfit: 0.06,      // 6% take profit  
  positionSize: 0.05,    // 5% position size
  maxDrawdown: 0.15,     // 15% max drawdown
  maxPositions: 10       // Max 10 positions
};
```

## 🚨 Troubleshooting

### Common Issues

#### Connection Errors
```bash
# Check MCP server status
npm list -g | grep mcp

# Restart MCP servers
npm install -g @notionhq/notion-mcp-server
```

#### API Key Issues
```bash
# Verify environment variables
echo $NOTION_API_TOKEN
echo $POLYGON_API_KEY
```

#### Performance Issues
```typescript
// Check system diagnostics
const diagnostics = await system.runDiagnostics();
console.log('Performance Metrics:', diagnostics.performance);
```

### Debug Mode

```typescript
const system = await createMCPSystem({
  logging: {
    level: 'debug',
    enableFileLogging: true
  },
  testing: {
    enabled: true,
    runOnStartup: true
  }
});
```

## 🔮 Advanced Features

### Custom Workflows

```typescript
const workflow = {
  workflowId: 'custom-strategy-optimization',
  steps: [
    {
      id: 'fetch-data',
      agentId: 'trading-orchestrator',
      action: 'fetch-market-data',
      parameters: { symbol: 'AAPL', timeframe: '1Day' },
      dependencies: [],
      timeout: 30000
    },
    {
      id: 'analyze-indicators',
      agentId: 'indicator-developer',
      action: 'calculate-indicators',
      parameters: { indicators: ['RSI', 'MACD', 'EMA'] },
      dependencies: ['fetch-data'],
      timeout: 15000
    },
    {
      id: 'generate-strategy',
      agentId: 'strategy-designer',
      action: 'generate-from-analysis',
      parameters: {},
      dependencies: ['analyze-indicators'],
      timeout: 45000
    }
  ],
  currentStep: 0,
  metadata: {},
  results: {}
};

const results = await integrator.executeWorkflow(workflow);
```

### Custom Indicators

```typescript
const customIndicator = await integrator.processAgentRequest({
  action: 'develop-indicator',
  parameters: {
    indicatorName: 'Lingua ATR Bands',
    formula: '(HIGH + LOW + CLOSE) / 3 + ATR(period) * multiplier',
    dependencies: ['ATR'],
    parameters: {
      period: 14,
      multiplier: 2.0
    }
  },
  context: { agentId: 'indicator-developer', sessionId: 'custom', parameters: {}, mcpClients: {} }
});
```

## 📚 API Reference

### MCPSystem Methods

- `initialize()` - Initialize the system
- `shutdown()` - Gracefully shutdown  
- `getStatus()` - Get system status
- `runDiagnostics()` - Run system diagnostics
- `reloadConfiguration()` - Reload configuration

### BMadAgentIntegrator Methods

- `processAgentRequest()` - Process agent requests
- `executeWorkflow()` - Execute multi-step workflows

### Client-Specific Methods

Each client provides specialized methods for their domain:

- **NotionClient**: `createStrategyPage()`, `searchStrategies()`, `addBacktestResults()`
- **BacktestingClient**: `runBacktest()`, `optimizeStrategy()`, `compareStrategies()`
- **TALibClient**: `calculateRSI()`, `calculateEMA()`, `recognizePatterns()`
- **OsEngineClient**: `placeOrder()`, `getPositions()`, `calculateRiskMetrics()`
- **PolygonClient**: `getHistoricalData()`, `getQuote()`, `scanMarket()`

## 🤝 Contributing

This MCP integration system is designed to be extensible. You can:

1. Add new MCP clients for additional services
2. Extend agent capabilities with new actions
3. Create custom workflows for your trading strategies
4. Add new test cases and monitoring capabilities

## 📄 License

This integration system is part of your SM Playbook BMad Trading System and follows the same licensing terms.

---

**Happy Trading! 📈🚀**

Your MCP integration system is now ready to supercharge your BMad trading framework with the power of multiple specialized AI services.