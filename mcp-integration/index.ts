/**
 * MCP Integration System - Main Entry Point
 * 
 * Complete MCP integration system for SM Playbook BMad Trading System
 * 
 * This module provides the main entry point and exports for the entire
 * MCP integration system, including all clients, managers, and utilities.
 */

// Core system
export { MCPSystem, createMCPSystem, MCPConfigurations } from './config/mcp-config';
export { MCPManager, MCPManagerConfig } from './core/mcp-manager';

// Client exports
export { NotionClient, NotionClientConfig } from './clients/notion-client';
export { BacktestingClient, BacktestingClientConfig } from './clients/backtesting-client';
export { TALibClient, TALibClientConfig } from './clients/talib-client';
export { OsEngineClient, OsEngineClientConfig } from './clients/osengine-client';
export { PolygonClient, PolygonClientConfig } from './clients/polygon-client';

// Integration layer
export { BMadAgentIntegrator, AgentIntegrationConfig } from './agents/bmad-agent-integrator';

// Testing framework
export { MCPTestFramework } from './testing/mcp-test-framework';

// Types and interfaces
export * from './types';

// Utility functions for quick setup
import { MCPSystem, createMCPSystem, MCPConfigurations } from './config/mcp-config';
import { MCPSystemConfig } from './config/mcp-config';

/**
 * Quick setup function for development environment
 */
export async function setupDevelopmentEnvironment(): Promise<MCPSystem> {
  console.log('🚀 Setting up MCP system for development...');
  
  const config: Partial<MCPSystemConfig> = {
    ...MCPConfigurations.development,
    clients: {
      notion: {
        name: 'notion',
        command: 'npx',
        args: ['-y', '@notionhq/notion-mcp-server'],
        env: {
          NOTION_API_TOKEN: process.env.NOTION_API_TOKEN || ''
        }
      },
      backtesting: {
        name: 'backtesting',
        command: 'npx',
        args: ['mcp-remote', 'https://gitmcp.io/kernc/backtesting.py'],
        defaultCommission: 0.001,
        defaultCash: 100000,
        exclusiveOrders: true,
        tradeOnClose: false
      },
      talib: {
        name: 'talib',
        command: 'npx',
        args: ['mcp-remote', 'https://gitmcp.io/TA-Lib/ta-lib-python'],
        defaultLookbackPeriod: 20,
        enablePatternRecognition: true,
        cachingEnabled: true
      },
      osengine: {
        name: 'osengine',
        command: 'npx',
        args: ['mcp-remote', 'https://gitmcp.io/AlexWan/OsEngine'],
        tradingMode: 'paper',
        defaultExchange: 'binance',
        riskLimits: {
          stopLoss: 0.02,
          takeProfit: 0.06,
          positionSize: 0.05,
          maxDrawdown: 0.15,
          maxPositions: 10
        },
        enableAutoExecution: false
      },
      polygon: {
        name: 'polygon',
        command: 'npx',
        args: ['mcp-remote', 'https://gitmcp.io/polygon-io/mcp_polygon'],
        apiKey: process.env.POLYGON_API_KEY || '',
        enableWebSocket: true,
        enableNewsData: true,
        defaultTimeframe: '1Day',
        maxConcurrentRequests: 10
      }
    }
  };

  return await createMCPSystem(config);
}

/**
 * Quick setup function for production environment
 */
export async function setupProductionEnvironment(): Promise<MCPSystem> {
  console.log('🚀 Setting up MCP system for production...');
  
  const config: Partial<MCPSystemConfig> = {
    ...MCPConfigurations.production,
    clients: {
      notion: {
        name: 'notion',
        command: 'npx',
        args: ['-y', '@notionhq/notion-mcp-server'],
        env: {
          NOTION_API_TOKEN: process.env.NOTION_API_TOKEN || ''
        }
      },
      backtesting: {
        name: 'backtesting',
        command: 'npx',
        args: ['mcp-remote', 'https://gitmcp.io/kernc/backtesting.py'],
        defaultCommission: 0.0005, // Lower commission for production
        defaultCash: 100000,
        exclusiveOrders: true,
        tradeOnClose: false
      },
      talib: {
        name: 'talib',
        command: 'npx',
        args: ['mcp-remote', 'https://gitmcp.io/TA-Lib/ta-lib-python'],
        defaultLookbackPeriod: 20,
        enablePatternRecognition: true,
        cachingEnabled: true
      },
      osengine: {
        name: 'osengine',
        command: 'npx',
        args: ['mcp-remote', 'https://gitmcp.io/AlexWan/OsEngine'],
        tradingMode: 'live', // Live trading for production
        defaultExchange: 'binance',
        riskLimits: {
          stopLoss: 0.015, // Tighter risk limits for production
          takeProfit: 0.04,
          positionSize: 0.02,
          maxDrawdown: 0.10,
          maxPositions: 5
        },
        enableAutoExecution: true
      },
      polygon: {
        name: 'polygon',
        command: 'npx',
        args: ['mcp-remote', 'https://gitmcp.io/polygon-io/mcp_polygon'],
        apiKey: process.env.POLYGON_API_KEY || '',
        enableWebSocket: true,
        enableNewsData: true,
        defaultTimeframe: '1Day',
        maxConcurrentRequests: 20 // Higher concurrency for production
      }
    }
  };

  return await createMCPSystem(config);
}

/**
 * Quick setup function for testing environment
 */
export async function setupTestEnvironment(): Promise<MCPSystem> {
  console.log('🧪 Setting up MCP system for testing...');
  
  const config: Partial<MCPSystemConfig> = {
    ...MCPConfigurations.testing,
    clients: {
      notion: {
        name: 'notion',
        command: 'npx',
        args: ['-y', '@notionhq/notion-mcp-server'],
        env: {
          NOTION_API_TOKEN: 'test-token'
        }
      },
      backtesting: {
        name: 'backtesting',
        command: 'npx',
        args: ['mcp-remote', 'https://gitmcp.io/kernc/backtesting.py'],
        defaultCommission: 0,
        defaultCash: 10000,
        exclusiveOrders: true,
        tradeOnClose: false
      },
      talib: {
        name: 'talib',
        command: 'npx',
        args: ['mcp-remote', 'https://gitmcp.io/TA-Lib/ta-lib-python'],
        defaultLookbackPeriod: 10,
        enablePatternRecognition: false,
        cachingEnabled: false
      },
      osengine: {
        name: 'osengine',
        command: 'npx',
        args: ['mcp-remote', 'https://gitmcp.io/AlexWan/OsEngine'],
        tradingMode: 'paper',
        defaultExchange: 'test',
        riskLimits: {
          stopLoss: 0.05,
          takeProfit: 0.10,
          positionSize: 0.10,
          maxDrawdown: 0.20,
          maxPositions: 2
        },
        enableAutoExecution: false
      },
      polygon: {
        name: 'polygon',
        command: 'npx',
        args: ['mcp-remote', 'https://gitmcp.io/polygon-io/mcp_polygon'],
        apiKey: 'test-api-key',
        enableWebSocket: false,
        enableNewsData: false,
        defaultTimeframe: '1Day',
        maxConcurrentRequests: 1
      }
    }
  };

  return await createMCPSystem(config);
}

/**
 * Example usage and demo functions
 */
export const examples = {
  /**
   * Basic usage example
   */
  async basicUsage(): Promise<void> {
    console.log('📖 Basic MCP Integration Usage Example');
    
    // Initialize system
    const system = await setupDevelopmentEnvironment();
    
    // Get integrator for agent operations
    const integrator = system.getIntegrator();
    
    // Example: Generate trading signals
    const signalRequest = {
      action: 'generate-signals',
      parameters: {
        symbol: 'AAPL',
        timeframe: '1Day',
        limit: 100
      },
      context: {
        agentId: 'trading-orchestrator',
        sessionId: 'example-session',
        parameters: {},
        mcpClients: {}
      }
    };
    
    const response = await integrator.processAgentRequest(signalRequest);
    console.log('📊 Signal Response:', response.success ? 'Success' : 'Failed');
    
    // Cleanup
    await system.shutdown();
  },

  /**
   * Strategy development workflow example
   */
  async strategyDevelopmentWorkflow(): Promise<void> {
    console.log('📖 Strategy Development Workflow Example');
    
    const system = await setupDevelopmentEnvironment();
    const integrator = system.getIntegrator();
    
    // Step 1: Design strategy
    console.log('Step 1: Designing strategy...');
    const designResponse = await integrator.processAgentRequest({
      action: 'design-strategy',
      parameters: {
        strategyName: 'EMA Crossover Strategy',
        description: 'Simple EMA crossover strategy',
        indicators: ['EMA_9', 'EMA_21'],
        entryRules: ['EMA_9 > EMA_21'],
        exitRules: ['EMA_9 < EMA_21'],
        validateWithBacktest: true,
        testSymbol: 'SPY'
      },
      context: {
        agentId: 'strategy-designer',
        sessionId: 'workflow-session',
        parameters: {},
        mcpClients: {}
      }
    });
    
    console.log('✅ Strategy designed:', designResponse.success);
    
    // Step 2: Run comprehensive backtest
    if (designResponse.success && designResponse.data?.strategy) {
      console.log('Step 2: Running comprehensive backtest...');
      const backtestResponse = await integrator.processAgentRequest({
        action: 'run-backtest',
        parameters: {
          strategy: {
            name: 'EMA Crossover Strategy',
            code: 'generated_code_here',
            parameters: {},
            indicators: ['EMA_9', 'EMA_21']
          },
          symbol: 'SPY',
          timeframe: '1Day',
          initialCapital: 100000,
          notionPageId: designResponse.data.strategy.id
        },
        context: {
          agentId: 'backtesting-engineer',
          sessionId: 'workflow-session',
          parameters: {},
          mcpClients: {}
        }
      });
      
      console.log('✅ Backtest completed:', backtestResponse.success);
      
      if (backtestResponse.success) {
        console.log('📈 Backtest Results Summary:');
        console.log(`  - Total Return: ${(backtestResponse.data?.backtestResult?.totalReturn * 100).toFixed(2)}%`);
        console.log(`  - Sharpe Ratio: ${backtestResponse.data?.backtestResult?.sharpeRatio?.toFixed(2)}`);
        console.log(`  - Max Drawdown: ${(backtestResponse.data?.backtestResult?.maxDrawdown * 100).toFixed(2)}%`);
      }
    }
    
    await system.shutdown();
  },

  /**
   * System monitoring example
   */
  async systemMonitoring(): Promise<void> {
    console.log('📖 System Monitoring Example');
    
    const system = await setupDevelopmentEnvironment();
    
    // Get system status
    const status = system.getStatus();
    console.log('📊 System Status:', {
      initialized: status.initialized,
      clientCount: Object.keys(status.clientStatuses).length,
      healthyClients: Object.values(status.clientStatuses).filter(s => s === 'connected').length,
      errors: status.errors.length
    });
    
    // Run diagnostics
    const diagnostics = await system.runDiagnostics();
    console.log('🔍 Diagnostics:', {
      systemHealth: diagnostics.system.initialized ? 'Healthy' : 'Unhealthy',
      recommendations: diagnostics.recommendations.length,
      performanceMetrics: Object.keys(diagnostics.performance).length
    });
    
    // Run tests
    const testFramework = system.getTestFramework();
    const testResults = await testFramework.runAllTests();
    console.log('🧪 Test Results:', Object.keys(testResults).map(suiteId => ({
      suite: suiteId,
      passed: testResults[suiteId].passedTests,
      total: testResults[suiteId].totalTests
    })));
    
    await system.shutdown();
  }
};

// Default export
export default {
  MCPSystem,
  createMCPSystem,
  setupDevelopmentEnvironment,
  setupProductionEnvironment,
  setupTestEnvironment,
  examples
};