/**
 * MCP Test Framework - Comprehensive testing suite for MCP clients and integration
 * 
 * This framework provides testing capabilities for:
 * - Individual MCP client functionality
 * - Integration layer testing
 * - End-to-end workflow testing
 * - Performance and load testing
 */

import { EventEmitter } from 'events';
import {
  BaseMCPClient,
  MCPHealthCheck,
  MCPResponse,
  AgentRequest,
  AgentResponse,
  Strategy,
  BacktestResult,
  OHLCV,
  MCPError
} from '../types';
import { MCPManager } from '../core/mcp-manager';
import { BMadAgentIntegrator } from '../agents/bmad-agent-integrator';

export interface TestCase {
  id: string;
  name: string;
  description: string;
  category: 'unit' | 'integration' | 'e2e' | 'performance';
  setup?: () => Promise<void>;
  execute: () => Promise<TestResult>;
  cleanup?: () => Promise<void>;
  timeout: number;
  dependencies: string[];
  tags: string[];
}

export interface TestResult {
  success: boolean;
  duration: number;
  data?: any;
  error?: string;
  metrics?: Record<string, number>;
  logs: string[];
}

export interface TestSuite {
  id: string;
  name: string;
  description: string;
  testCases: TestCase[];
  beforeAll?: () => Promise<void>;
  afterAll?: () => Promise<void>;
}

export interface TestReport {
  suiteId: string;
  startTime: Date;
  endTime: Date;
  totalTests: number;
  passedTests: number;
  failedTests: number;
  skippedTests: number;
  results: Record<string, TestResult>;
  performance: {
    averageResponseTime: number;
    slowestTest: string;
    fastestTest: string;
  };
  coverage: {
    clientsCovered: string[];
    methodsCovered: string[];
    coveragePercentage: number;
  };
}

export interface MockDataConfig {
  historicalData: OHLCV[];
  strategies: Strategy[];
  backtestResults: BacktestResult[];
  marketQuotes: Record<string, any>;
}

export class MCPTestFramework extends EventEmitter {
  private mcpManager: MCPManager;
  private integrator: BMadAgentIntegrator;
  private testSuites: Map<string, TestSuite> = new Map();
  private mockData: MockDataConfig;
  private testResults: Map<string, TestReport> = new Map();

  constructor(mcpManager: MCPManager, integrator: BMadAgentIntegrator) {
    super();
    this.mcpManager = mcpManager;
    this.integrator = integrator;
    this.mockData = this.generateMockData();
    this.setupTestSuites();
  }

  /**
   * Run all test suites
   */
  async runAllTests(): Promise<Record<string, TestReport>> {
    const reports: Record<string, TestReport> = {};
    
    this.emit('testingStarted');

    for (const [suiteId, suite] of this.testSuites) {
      try {
        const report = await this.runTestSuite(suite);
        reports[suiteId] = report;
        this.testResults.set(suiteId, report);
      } catch (error) {
        console.error(`Failed to run test suite ${suiteId}:`, error);
      }
    }

    this.emit('testingCompleted', reports);
    return reports;
  }

  /**
   * Run specific test suite
   */
  async runTestSuite(suite: TestSuite): Promise<TestReport> {
    const startTime = new Date();
    const results: Record<string, TestResult> = {};
    
    this.emit('testSuiteStarted', suite.id);

    try {
      // Run beforeAll hook
      if (suite.beforeAll) {
        await suite.beforeAll();
      }

      // Run test cases
      for (const testCase of suite.testCases) {
        try {
          const result = await this.runTestCase(testCase);
          results[testCase.id] = result;
          
          this.emit('testCaseCompleted', suite.id, testCase.id, result);
        } catch (error) {
          results[testCase.id] = {
            success: false,
            duration: 0,
            error: error instanceof Error ? error.message : String(error),
            logs: [`Test case failed: ${error}`]
          };
        }
      }

      // Run afterAll hook
      if (suite.afterAll) {
        await suite.afterAll();
      }
    } finally {
      const endTime = new Date();
      const report = this.generateTestReport(suite, startTime, endTime, results);
      
      this.emit('testSuiteCompleted', suite.id, report);
      return report;
    }
  }

  /**
   * Run individual test case
   */
  async runTestCase(testCase: TestCase): Promise<TestResult> {
    const startTime = Date.now();
    const logs: string[] = [];

    try {
      // Setup
      if (testCase.setup) {
        logs.push('Running setup...');
        await testCase.setup();
      }

      // Execute with timeout
      logs.push('Executing test...');
      const result = await Promise.race([
        testCase.execute(),
        new Promise<TestResult>((_, reject) => 
          setTimeout(() => reject(new Error('Test timeout')), testCase.timeout)
        )
      ]);

      // Cleanup
      if (testCase.cleanup) {
        logs.push('Running cleanup...');
        await testCase.cleanup();
      }

      const duration = Date.now() - startTime;
      return {
        ...result,
        duration,
        logs: [...logs, ...result.logs]
      };
    } catch (error) {
      const duration = Date.now() - startTime;
      return {
        success: false,
        duration,
        error: error instanceof Error ? error.message : String(error),
        logs: [...logs, `Test failed: ${error}`]
      };
    }
  }

  /**
   * Test MCP client health checks
   */
  async testClientHealthChecks(): Promise<TestResult> {
    const logs: string[] = [];
    const metrics: Record<string, number> = {};
    let allHealthy = true;

    const clientIds = this.mcpManager.getRegisteredClients();
    
    for (const clientId of clientIds) {
      const client = this.mcpManager.getClient(clientId);
      if (client) {
        try {
          const startTime = Date.now();
          const healthCheck = await client.healthCheck();
          const responseTime = Date.now() - startTime;
          
          metrics[`${clientId}_response_time`] = responseTime;
          
          if (healthCheck.status === 'healthy') {
            logs.push(`✓ ${clientId}: Healthy (${responseTime}ms)`);
          } else {
            logs.push(`✗ ${clientId}: Unhealthy - ${healthCheck.error}`);
            allHealthy = false;
          }
        } catch (error) {
          logs.push(`✗ ${clientId}: Error - ${error}`);
          allHealthy = false;
        }
      }
    }

    return {
      success: allHealthy,
      duration: 0,
      metrics,
      logs,
      data: { healthyClients: clientIds.length - Object.keys(metrics).filter(k => k.includes('error')).length }
    };
  }

  /**
   * Test agent integration functionality
   */
  async testAgentIntegration(): Promise<TestResult> {
    const logs: string[] = [];
    const results: any[] = [];

    // Test trading orchestrator
    try {
      logs.push('Testing trading orchestrator...');
      const request: AgentRequest = {
        action: 'generate-signals',
        parameters: {
          symbol: 'AAPL',
          timeframe: '1Day',
          limit: 50
        },
        context: {
          agentId: 'trading-orchestrator',
          sessionId: 'test-session',
          parameters: {},
          mcpClients: {}
        }
      };

      const response = await this.integrator.processAgentRequest(request);
      results.push({ agent: 'trading-orchestrator', success: response.success });
      logs.push(`✓ Trading orchestrator: ${response.success ? 'Success' : 'Failed'}`);
    } catch (error) {
      results.push({ agent: 'trading-orchestrator', success: false, error });
      logs.push(`✗ Trading orchestrator: ${error}`);
    }

    // Test strategy designer
    try {
      logs.push('Testing strategy designer...');
      const request: AgentRequest = {
        action: 'design-strategy',
        parameters: {
          strategyName: 'Test RSI Strategy',
          description: 'Simple RSI-based strategy for testing',
          indicators: ['RSI'],
          entryRules: ['RSI < 30'],
          exitRules: ['RSI > 70']
        },
        context: {
          agentId: 'strategy-designer',
          sessionId: 'test-session',
          parameters: {},
          mcpClients: {}
        }
      };

      const response = await this.integrator.processAgentRequest(request);
      results.push({ agent: 'strategy-designer', success: response.success });
      logs.push(`✓ Strategy designer: ${response.success ? 'Success' : 'Failed'}`);
    } catch (error) {
      results.push({ agent: 'strategy-designer', success: false, error });
      logs.push(`✗ Strategy designer: ${error}`);
    }

    const successCount = results.filter(r => r.success).length;
    const success = successCount === results.length;

    return {
      success,
      duration: 0,
      logs,
      data: { results, successRate: successCount / results.length }
    };
  }

  /**
   * Performance test for MCP operations
   */
  async performanceTest(): Promise<TestResult> {
    const logs: string[] = [];
    const metrics: Record<string, number> = {};
    const operations = 100;
    const concurrency = 5;

    logs.push(`Running performance test with ${operations} operations, concurrency: ${concurrency}`);

    // Test historical data retrieval performance
    const polygonClient = this.mcpManager.getClient('polygon');
    if (polygonClient) {
      const startTime = Date.now();
      const promises = [];

      for (let i = 0; i < operations; i++) {
        if (i % concurrency === 0 && i > 0) {
          await Promise.all(promises);
          promises.length = 0;
        }

        promises.push(
          polygonClient.request({
            method: 'market-data/historical',
            params: {
              symbol: 'AAPL',
              timeframe: '1Day',
              limit: 100
            }
          })
        );
      }

      await Promise.all(promises);
      const duration = Date.now() - startTime;
      const opsPerSecond = (operations * 1000) / duration;

      metrics.polygon_ops_per_second = opsPerSecond;
      metrics.polygon_avg_response_time = duration / operations;
      
      logs.push(`Polygon performance: ${opsPerSecond.toFixed(2)} ops/sec, avg: ${(duration/operations).toFixed(2)}ms`);
    }

    // Test indicator calculations performance
    const talibClient = this.mcpManager.getClient('talib');
    if (talibClient) {
      const startTime = Date.now();
      const testData = Array.from({ length: 1000 }, () => Math.random() * 100 + 50);

      for (let i = 0; i < 50; i++) {
        await talibClient.request({
          method: 'indicators/RSI',
          params: { data: testData, period: 14 }
        });
      }

      const duration = Date.now() - startTime;
      metrics.talib_indicators_per_second = (50 * 1000) / duration;
      
      logs.push(`TA-Lib performance: ${((50 * 1000) / duration).toFixed(2)} calculations/sec`);
    }

    return {
      success: true,
      duration: 0,
      metrics,
      logs,
      data: { performanceMetrics: metrics }
    };
  }

  /**
   * End-to-end workflow test
   */
  async e2eWorkflowTest(): Promise<TestResult> {
    const logs: string[] = [];
    
    try {
      logs.push('Starting end-to-end workflow test...');

      // Step 1: Generate signals
      logs.push('Step 1: Generating trading signals...');
      const signalRequest: AgentRequest = {
        action: 'generate-signals',
        parameters: {
          symbol: 'AAPL',
          timeframe: '1Day',
          autoExecute: false
        },
        context: {
          agentId: 'trading-orchestrator',
          sessionId: 'e2e-test',
          parameters: {},
          mcpClients: {}
        }
      };

      const signalResponse = await this.integrator.processAgentRequest(signalRequest);
      if (!signalResponse.success) {
        throw new Error(`Signal generation failed: ${signalResponse.error}`);
      }
      logs.push('✓ Signals generated successfully');

      // Step 2: Design strategy based on signals
      logs.push('Step 2: Designing strategy...');
      const strategyRequest: AgentRequest = {
        action: 'design-strategy',
        parameters: {
          strategyName: 'E2E Test Strategy',
          description: 'Generated from e2e test',
          indicators: ['RSI', 'EMA'],
          validateWithBacktest: false
        },
        context: {
          agentId: 'strategy-designer',
          sessionId: 'e2e-test',
          parameters: {},
          mcpClients: {}
        }
      };

      const strategyResponse = await this.integrator.processAgentRequest(strategyRequest);
      if (!strategyResponse.success) {
        throw new Error(`Strategy design failed: ${strategyResponse.error}`);
      }
      logs.push('✓ Strategy designed successfully');

      // Step 3: Run backtest
      logs.push('Step 3: Running backtest...');
      const backtestRequest: AgentRequest = {
        action: 'run-backtest',
        parameters: {
          strategy: {
            name: 'E2E Test Strategy',
            code: 'mock_strategy_code',
            parameters: {},
            indicators: ['RSI', 'EMA']
          },
          symbol: 'AAPL',
          timeframe: '1Day',
          initialCapital: 100000
        },
        context: {
          agentId: 'backtesting-engineer',
          sessionId: 'e2e-test',
          parameters: {},
          mcpClients: {}
        }
      };

      const backtestResponse = await this.integrator.processAgentRequest(backtestRequest);
      if (!backtestResponse.success) {
        throw new Error(`Backtest failed: ${backtestResponse.error}`);
      }
      logs.push('✓ Backtest completed successfully');

      logs.push('End-to-end workflow test completed successfully');

      return {
        success: true,
        duration: 0,
        logs,
        data: {
          signalData: signalResponse.data,
          strategyData: strategyResponse.data,
          backtestData: backtestResponse.data
        }
      };
    } catch (error) {
      logs.push(`✗ E2E test failed: ${error}`);
      return {
        success: false,
        duration: 0,
        error: error instanceof Error ? error.message : String(error),
        logs
      };
    }
  }

  private setupTestSuites(): void {
    // Unit tests suite
    this.testSuites.set('unit-tests', {
      id: 'unit-tests',
      name: 'Unit Tests',
      description: 'Individual MCP client functionality tests',
      testCases: [
        {
          id: 'health-checks',
          name: 'Health Checks',
          description: 'Test all MCP client health checks',
          category: 'unit',
          execute: () => this.testClientHealthChecks(),
          timeout: 30000,
          dependencies: [],
          tags: ['health', 'connectivity']
        }
      ]
    });

    // Integration tests suite
    this.testSuites.set('integration-tests', {
      id: 'integration-tests',
      name: 'Integration Tests',
      description: 'BMad agent integration tests',
      testCases: [
        {
          id: 'agent-integration',
          name: 'Agent Integration',
          description: 'Test BMad agent integration functionality',
          category: 'integration',
          execute: () => this.testAgentIntegration(),
          timeout: 60000,
          dependencies: ['health-checks'],
          tags: ['agents', 'integration']
        }
      ]
    });

    // Performance tests suite
    this.testSuites.set('performance-tests', {
      id: 'performance-tests',
      name: 'Performance Tests',
      description: 'Performance and load testing',
      testCases: [
        {
          id: 'performance-test',
          name: 'Performance Test',
          description: 'Test MCP operation performance under load',
          category: 'performance',
          execute: () => this.performanceTest(),
          timeout: 120000,
          dependencies: [],
          tags: ['performance', 'load']
        }
      ]
    });

    // End-to-end tests suite
    this.testSuites.set('e2e-tests', {
      id: 'e2e-tests',
      name: 'End-to-End Tests',
      description: 'Complete workflow testing',
      testCases: [
        {
          id: 'e2e-workflow',
          name: 'E2E Workflow',
          description: 'Test complete trading workflow',
          category: 'e2e',
          execute: () => this.e2eWorkflowTest(),
          timeout: 180000,
          dependencies: ['health-checks', 'agent-integration'],
          tags: ['e2e', 'workflow']
        }
      ]
    });
  }

  private generateTestReport(
    suite: TestSuite,
    startTime: Date,
    endTime: Date,
    results: Record<string, TestResult>
  ): TestReport {
    const testResults = Object.values(results);
    const passedTests = testResults.filter(r => r.success).length;
    const failedTests = testResults.filter(r => !r.success).length;
    
    const responseTimes = testResults.map(r => r.duration).filter(d => d > 0);
    const averageResponseTime = responseTimes.length > 0 
      ? responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length 
      : 0;

    let slowestTest = '';
    let fastestTest = '';
    let maxTime = 0;
    let minTime = Infinity;

    for (const [testId, result] of Object.entries(results)) {
      if (result.duration > maxTime) {
        maxTime = result.duration;
        slowestTest = testId;
      }
      if (result.duration < minTime && result.duration > 0) {
        minTime = result.duration;
        fastestTest = testId;
      }
    }

    return {
      suiteId: suite.id,
      startTime,
      endTime,
      totalTests: suite.testCases.length,
      passedTests,
      failedTests,
      skippedTests: 0,
      results,
      performance: {
        averageResponseTime,
        slowestTest,
        fastestTest
      },
      coverage: {
        clientsCovered: this.mcpManager.getRegisteredClients(),
        methodsCovered: [], // Would be populated with actual method coverage
        coveragePercentage: 0 // Would be calculated based on actual coverage
      }
    };
  }

  private generateMockData(): MockDataConfig {
    // Generate mock historical data
    const historicalData: OHLCV[] = [];
    const basePrice = 100;
    
    for (let i = 0; i < 252; i++) { // 1 year of daily data
      const date = new Date(Date.now() - (252 - i) * 24 * 60 * 60 * 1000);
      const variation = (Math.random() - 0.5) * 10;
      const open = basePrice + variation;
      const close = open + (Math.random() - 0.5) * 5;
      const high = Math.max(open, close) + Math.random() * 2;
      const low = Math.min(open, close) - Math.random() * 2;

      historicalData.push({
        timestamp: date,
        open,
        high,
        low,
        close,
        volume: Math.floor(Math.random() * 1000000) + 100000
      });
    }

    return {
      historicalData,
      strategies: [],
      backtestResults: [],
      marketQuotes: {
        'AAPL': { price: 150.25, change: 2.50, changePercent: 1.69 },
        'GOOGL': { price: 2500.75, change: -15.25, changePercent: -0.60 }
      }
    };
  }

  /**
   * Generate test report in multiple formats
   */
  generateReport(format: 'json' | 'html' | 'markdown' = 'json'): string {
    const allResults = Object.fromEntries(this.testResults.entries());
    
    switch (format) {
      case 'json':
        return JSON.stringify(allResults, null, 2);
      
      case 'markdown':
        return this.generateMarkdownReport(allResults);
      
      case 'html':
        return this.generateHTMLReport(allResults);
      
      default:
        return JSON.stringify(allResults, null, 2);
    }
  }

  private generateMarkdownReport(results: Record<string, TestReport>): string {
    let markdown = '# MCP Integration Test Report\n\n';
    markdown += `Generated: ${new Date().toISOString()}\n\n`;

    for (const [suiteId, report] of Object.entries(results)) {
      markdown += `## ${suiteId}\n\n`;
      markdown += `- **Total Tests**: ${report.totalTests}\n`;
      markdown += `- **Passed**: ${report.passedTests}\n`;
      markdown += `- **Failed**: ${report.failedTests}\n`;
      markdown += `- **Success Rate**: ${((report.passedTests / report.totalTests) * 100).toFixed(1)}%\n`;
      markdown += `- **Duration**: ${report.endTime.getTime() - report.startTime.getTime()}ms\n\n`;

      markdown += '### Test Results\n\n';
      for (const [testId, result] of Object.entries(report.results)) {
        const status = result.success ? '✅' : '❌';
        markdown += `- ${status} **${testId}**: ${result.success ? 'PASSED' : 'FAILED'} (${result.duration}ms)\n`;
        if (result.error) {
          markdown += `  - Error: ${result.error}\n`;
        }
      }
      markdown += '\n';
    }

    return markdown;
  }

  private generateHTMLReport(results: Record<string, TestReport>): string {
    return `
<!DOCTYPE html>
<html>
<head>
    <title>MCP Integration Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .suite { margin-bottom: 30px; }
        .passed { color: green; }
        .failed { color: red; }
        .metrics { background: #f5f5f5; padding: 10px; margin: 10px 0; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>MCP Integration Test Report</h1>
    <p>Generated: ${new Date().toISOString()}</p>
    
    ${Object.entries(results).map(([suiteId, report]) => `
        <div class="suite">
            <h2>${suiteId}</h2>
            <div class="metrics">
                <p><strong>Total Tests:</strong> ${report.totalTests}</p>
                <p><strong>Passed:</strong> <span class="passed">${report.passedTests}</span></p>
                <p><strong>Failed:</strong> <span class="failed">${report.failedTests}</span></p>
                <p><strong>Success Rate:</strong> ${((report.passedTests / report.totalTests) * 100).toFixed(1)}%</p>
            </div>
            
            <table>
                <thead>
                    <tr><th>Test</th><th>Status</th><th>Duration</th><th>Error</th></tr>
                </thead>
                <tbody>
                    ${Object.entries(report.results).map(([testId, result]) => `
                        <tr>
                            <td>${testId}</td>
                            <td class="${result.success ? 'passed' : 'failed'}">
                                ${result.success ? 'PASSED' : 'FAILED'}
                            </td>
                            <td>${result.duration}ms</td>
                            <td>${result.error || '-'}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `).join('')}
</body>
</html>
    `;
  }
}