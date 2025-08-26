/**
 * MCP Configuration and Initialization System
 * 
 * This module provides comprehensive configuration management and system initialization
 * for the MCP integration with SM Playbook BMad Trading System.
 */

import { EventEmitter } from 'events';
import {
  MCPServerConfig,
  BaseMCPClient,
  MCPError
} from '../types';
import { MCPManager, MCPManagerConfig } from '../core/mcp-manager';
import { NotionClient, NotionClientConfig } from '../clients/notion-client';
import { BacktestingClient, BacktestingClientConfig } from '../clients/backtesting-client';
import { TALibClient, TALibClientConfig } from '../clients/talib-client';
import { OsEngineClient, OsEngineClientConfig } from '../clients/osengine-client';
import { PolygonClient, PolygonClientConfig } from '../clients/polygon-client';
import { BMadAgentIntegrator, AgentIntegrationConfig } from '../agents/bmad-agent-integrator';
import { MCPTestFramework } from '../testing/mcp-test-framework';

export interface MCPSystemConfig {
  manager: MCPManagerConfig;
  clients: {
    notion: NotionClientConfig;
    backtesting: BacktestingClientConfig;
    talib: TALibClientConfig;
    osengine: OsEngineClientConfig;
    polygon: PolygonClientConfig;
  };
  integration: AgentIntegrationConfig;
  testing: {
    enabled: boolean;
    runOnStartup: boolean;
    generateReports: boolean;
    reportFormat: 'json' | 'html' | 'markdown';
  };
  logging: {
    level: 'debug' | 'info' | 'warn' | 'error';
    enableFileLogging: boolean;
    logDirectory: string;
  };
}

export interface SystemStatus {
  initialized: boolean;
  startTime: Date;
  clientStatuses: Record<string, 'connected' | 'disconnected' | 'error'>;
  lastHealthCheck: Date;
  errors: string[];
}

export class MCPSystem extends EventEmitter {
  private config: MCPSystemConfig;
  private manager: MCPManager;
  private integrator: BMadAgentIntegrator;
  private testFramework: MCPTestFramework;
  private status: SystemStatus;
  private initialized: boolean = false;

  constructor(config: Partial<MCPSystemConfig> = {}) {
    super();
    
    this.config = this.mergeWithDefaults(config);
    this.status = {
      initialized: false,
      startTime: new Date(),
      clientStatuses: {},
      lastHealthCheck: new Date(),
      errors: []
    };

    this.manager = new MCPManager(this.config.manager);
    this.integrator = new BMadAgentIntegrator(this.manager, this.config.integration);
    this.testFramework = new MCPTestFramework(this.manager, this.integrator);

    this.setupEventHandlers();
  }

  /**
   * Initialize the complete MCP system
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      throw new MCPError('System already initialized', 'mcp-system');
    }

    try {
      this.emit('systemInitializing');
      console.log('🚀 Initializing MCP Integration System...');

      // Initialize clients in order
      await this.initializeClients();

      // Start health monitoring
      await this.startHealthMonitoring();

      // Run tests if enabled
      if (this.config.testing.enabled && this.config.testing.runOnStartup) {
        await this.runInitialTests();
      }

      this.initialized = true;
      this.status.initialized = true;
      this.status.startTime = new Date();

      this.emit('systemInitialized');
      console.log('✅ MCP Integration System initialized successfully');
    } catch (error) {
      this.status.errors.push(error instanceof Error ? error.message : String(error));
      this.emit('systemInitializationError', error);
      throw error;
    }
  }

  /**
   * Shutdown the system gracefully
   */
  async shutdown(): Promise<void> {
    if (!this.initialized) {
      return;
    }

    try {
      this.emit('systemShuttingDown');
      console.log('🔄 Shutting down MCP Integration System...');

      // Shutdown manager (which will shutdown all clients)
      await this.manager.shutdown();

      this.initialized = false;
      this.status.initialized = false;

      this.emit('systemShutdown');
      console.log('✅ MCP Integration System shut down successfully');
    } catch (error) {
      this.emit('systemShutdownError', error);
      console.error('❌ Error during system shutdown:', error);
      throw error;
    }
  }

  /**
   * Get current system status
   */
  getStatus(): SystemStatus {
    return {
      ...this.status,
      clientStatuses: { ...this.status.clientStatuses }
    };
  }

  /**
   * Get MCP manager instance
   */
  getManager(): MCPManager {
    return this.manager;
  }

  /**
   * Get agent integrator instance
   */
  getIntegrator(): BMadAgentIntegrator {
    return this.integrator;
  }

  /**
   * Get test framework instance
   */
  getTestFramework(): MCPTestFramework {
    return this.testFramework;
  }

  /**
   * Reload configuration
   */
  async reloadConfiguration(newConfig: Partial<MCPSystemConfig>): Promise<void> {
    console.log('🔄 Reloading MCP system configuration...');
    
    // Merge new config with existing
    this.config = this.mergeWithDefaults(newConfig);
    
    // Restart system with new configuration
    if (this.initialized) {
      await this.shutdown();
      await this.initialize();
    }

    this.emit('configurationReloaded', this.config);
    console.log('✅ Configuration reloaded successfully');
  }

  /**
   * Run system diagnostics
   */
  async runDiagnostics(): Promise<{
    system: SystemStatus;
    clients: Record<string, any>;
    performance: Record<string, number>;
    recommendations: string[];
  }> {
    console.log('🔍 Running system diagnostics...');

    const diagnostics = {
      system: this.getStatus(),
      clients: this.manager.getConnectionStatuses(),
      performance: {},
      recommendations: []
    };

    // Check client health
    const healthChecks = await this.manager.performHealthCheck();
    let unhealthyClients = 0;

    for (const [clientId, health] of Object.entries(healthChecks)) {
      if (health.status !== 'healthy') {
        unhealthyClients++;
        diagnostics.recommendations.push(`Check ${clientId} client connection`);
      }
      
      if (health.responseTime && health.responseTime > 5000) {
        diagnostics.recommendations.push(`${clientId} client response time is high (${health.responseTime}ms)`);
      }
    }

    // Performance metrics
    const metrics = this.manager.getAllMetrics();
    for (const [clientId, metric] of Object.entries(metrics)) {
      diagnostics.performance[`${clientId}_avg_response_time`] = metric.averageResponseTime;
      diagnostics.performance[`${clientId}_success_rate`] = 
        metric.totalRequests > 0 ? metric.successfulRequests / metric.totalRequests : 0;
    }

    // System-level recommendations
    if (unhealthyClients > 0) {
      diagnostics.recommendations.push(`${unhealthyClients} client(s) are unhealthy`);
    }

    if (this.status.errors.length > 0) {
      diagnostics.recommendations.push(`${this.status.errors.length} system error(s) need attention`);
    }

    console.log('✅ Diagnostics completed');
    return diagnostics;
  }

  /**
   * Update system configuration
   */
  updateConfig(updates: Partial<MCPSystemConfig>): void {
    this.config = { ...this.config, ...updates };
    this.emit('configurationUpdated', this.config);
  }

  /**
   * Get current configuration
   */
  getConfiguration(): MCPSystemConfig {
    return { ...this.config };
  }

  private async initializeClients(): Promise<void> {
    console.log('📡 Initializing MCP clients...');

    const clientConfigs = [
      { id: 'notion', class: NotionClient, config: this.config.clients.notion },
      { id: 'backtesting', class: BacktestingClient, config: this.config.clients.backtesting },
      { id: 'talib', class: TALibClient, config: this.config.clients.talib },
      { id: 'osengine', class: OsEngineClient, config: this.config.clients.osengine },
      { id: 'polygon', class: PolygonClient, config: this.config.clients.polygon }
    ];

    for (const { id, class: ClientClass, config } of clientConfigs) {
      try {
        console.log(`  📋 Initializing ${id} client...`);
        const client = new ClientClass(config);
        await this.manager.registerClient(client);
        this.status.clientStatuses[id] = 'connected';
        console.log(`  ✅ ${id} client initialized`);
      } catch (error) {
        this.status.clientStatuses[id] = 'error';
        this.status.errors.push(`Failed to initialize ${id} client: ${error}`);
        console.warn(`  ⚠️ Failed to initialize ${id} client:`, error);
        
        // Continue with other clients rather than failing completely
      }
    }
  }

  private async startHealthMonitoring(): Promise<void> {
    console.log('🏥 Starting health monitoring...');
    
    // Initial health check
    await this.performHealthCheck();
    
    // Schedule periodic health checks
    setInterval(async () => {
      await this.performHealthCheck();
    }, this.config.manager.healthCheckInterval || 30000);
  }

  private async performHealthCheck(): Promise<void> {
    try {
      const healthChecks = await this.manager.performHealthCheck();
      
      for (const [clientId, health] of Object.entries(healthChecks)) {
        const prevStatus = this.status.clientStatuses[clientId];
        const newStatus = health.status === 'healthy' ? 'connected' : 'error';
        
        if (prevStatus !== newStatus) {
          this.status.clientStatuses[clientId] = newStatus;
          this.emit('clientStatusChanged', clientId, newStatus, health);
        }
      }
      
      this.status.lastHealthCheck = new Date();
    } catch (error) {
      console.error('Health check failed:', error);
    }
  }

  private async runInitialTests(): Promise<void> {
    console.log('🧪 Running initial tests...');
    
    try {
      const results = await this.testFramework.runAllTests();
      
      if (this.config.testing.generateReports) {
        const report = this.testFramework.generateReport(this.config.testing.reportFormat);
        console.log('📊 Test report generated');
        
        // In a real implementation, save the report to file
        // fs.writeFileSync(`./test-report.${this.config.testing.reportFormat}`, report);
      }
      
      this.emit('initialTestsCompleted', results);
    } catch (error) {
      console.error('Initial tests failed:', error);
      this.status.errors.push(`Initial tests failed: ${error}`);
    }
  }

  private setupEventHandlers(): void {
    // Manager events
    this.manager.on('clientRegistered', (clientId) => {
      console.log(`📱 Client ${clientId} registered`);
    });

    this.manager.on('clientError', (clientId, error) => {
      console.warn(`⚠️ Client ${clientId} error:`, error);
      this.status.errors.push(`${clientId}: ${error}`);
    });

    this.manager.on('clientConnected', (clientId) => {
      console.log(`🔗 Client ${clientId} connected`);
      this.status.clientStatuses[clientId] = 'connected';
    });

    this.manager.on('clientUnhealthy', (clientId, health) => {
      console.warn(`🏥 Client ${clientId} unhealthy:`, health.error);
      this.status.clientStatuses[clientId] = 'error';
    });

    // Integrator events
    this.integrator.on('workflowStarted', (workflowId) => {
      console.log(`🚀 Workflow ${workflowId} started`);
    });

    this.integrator.on('workflowCompleted', (workflowId, results) => {
      console.log(`✅ Workflow ${workflowId} completed`);
    });

    this.integrator.on('workflowError', (workflowId, error) => {
      console.error(`❌ Workflow ${workflowId} failed:`, error);
    });

    // Test framework events
    this.testFramework.on('testingStarted', () => {
      console.log('🧪 Testing started');
    });

    this.testFramework.on('testingCompleted', (results) => {
      const totalTests = Object.values(results).reduce((sum, report) => sum + report.totalTests, 0);
      const passedTests = Object.values(results).reduce((sum, report) => sum + report.passedTests, 0);
      console.log(`✅ Testing completed: ${passedTests}/${totalTests} tests passed`);
    });
  }

  private mergeWithDefaults(config: Partial<MCPSystemConfig>): MCPSystemConfig {
    return {
      manager: {
        healthCheckInterval: 30000,
        maxRetryAttempts: 3,
        retryBackoff: 1000,
        connectionTimeout: 10000,
        enableMetrics: true,
        ...config.manager
      },
      clients: {
        notion: {
          name: 'notion',
          command: 'npx',
          args: ['-y', '@notionhq/notion-mcp-server'],
          env: {
            NOTION_API_TOKEN: process.env.NOTION_API_TOKEN || ''
          },
          ...config.clients?.notion
        },
        backtesting: {
          name: 'backtesting',
          command: 'npx',
          args: ['mcp-remote', 'https://gitmcp.io/kernc/backtesting.py'],
          defaultCommission: 0.001,
          defaultCash: 100000,
          exclusiveOrders: true,
          tradeOnClose: false,
          ...config.clients?.backtesting
        },
        talib: {
          name: 'talib',
          command: 'npx',
          args: ['mcp-remote', 'https://gitmcp.io/TA-Lib/ta-lib-python'],
          defaultLookbackPeriod: 20,
          enablePatternRecognition: true,
          cachingEnabled: true,
          ...config.clients?.talib
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
          enableAutoExecution: false,
          ...config.clients?.osengine
        },
        polygon: {
          name: 'polygon',
          command: 'npx',
          args: ['mcp-remote', 'https://gitmcp.io/polygon-io/mcp_polygon'],
          apiKey: process.env.POLYGON_API_KEY || '',
          enableWebSocket: true,
          enableNewsData: true,
          defaultTimeframe: '1Day',
          maxConcurrentRequests: 10,
          ...config.clients?.polygon
        }
      },
      integration: {
        enabledAgents: [
          'trading-orchestrator',
          'strategy-designer',
          'backtesting-engineer',
          'execution-engineer',
          'indicator-developer'
        ],
        defaultTimeframes: ['1m', '5m', '15m', '1h', '4h', '1d'],
        maxConcurrentOperations: 10,
        enableResultCaching: true,
        autoSyncNotionResults: true,
        ...config.integration
      },
      testing: {
        enabled: true,
        runOnStartup: false,
        generateReports: true,
        reportFormat: 'markdown',
        ...config.testing
      },
      logging: {
        level: 'info',
        enableFileLogging: true,
        logDirectory: './logs',
        ...config.logging
      }
    };
  }
}

/**
 * Factory function to create and initialize MCP system
 */
export async function createMCPSystem(config: Partial<MCPSystemConfig> = {}): Promise<MCPSystem> {
  const system = new MCPSystem(config);
  await system.initialize();
  return system;
}

/**
 * Default configuration for common setups
 */
export const MCPConfigurations = {
  development: {
    manager: {
      healthCheckInterval: 60000,
      enableMetrics: true
    },
    testing: {
      enabled: true,
      runOnStartup: true,
      generateReports: true,
      reportFormat: 'html' as const
    },
    logging: {
      level: 'debug' as const,
      enableFileLogging: true
    }
  },

  production: {
    manager: {
      healthCheckInterval: 30000,
      maxRetryAttempts: 5,
      enableMetrics: true
    },
    testing: {
      enabled: false,
      runOnStartup: false,
      generateReports: false
    },
    logging: {
      level: 'info' as const,
      enableFileLogging: true
    }
  },

  testing: {
    manager: {
      healthCheckInterval: 5000,
      enableMetrics: false
    },
    testing: {
      enabled: true,
      runOnStartup: true,
      generateReports: true,
      reportFormat: 'json' as const
    },
    logging: {
      level: 'warn' as const,
      enableFileLogging: false
    }
  }
};

export default MCPSystem;