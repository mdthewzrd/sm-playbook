/**
 * BMadAgentIntegrator - Integration layer connecting MCP clients to BMad agents
 * 
 * This module provides the integration layer that connects all MCP clients
 * to the BMad agent system, enabling seamless communication and orchestration.
 */

import { EventEmitter } from 'events';
import {
  AgentContext,
  AgentRequest,
  AgentResponse,
  Strategy,
  BacktestResult,
  OHLCV,
  NotionStrategy,
  MCPError
} from '../types';
import { MCPManager } from '../core/mcp-manager';
import { NotionClient } from '../clients/notion-client';
import { BacktestingClient } from '../clients/backtesting-client';
import { TALibClient } from '../clients/talib-client';
import { OsEngineClient } from '../clients/osengine-client';
import { PolygonClient } from '../clients/polygon-client';

export interface AgentIntegrationConfig {
  enabledAgents: string[];
  defaultTimeframes: string[];
  maxConcurrentOperations: number;
  enableResultCaching: boolean;
  autoSyncNotionResults: boolean;
}

export interface WorkflowContext {
  workflowId: string;
  steps: WorkflowStep[];
  currentStep: number;
  metadata: Record<string, any>;
  results: Record<string, any>;
}

export interface WorkflowStep {
  id: string;
  agentId: string;
  action: string;
  parameters: Record<string, any>;
  dependencies: string[];
  timeout: number;
}

export class BMadAgentIntegrator extends EventEmitter {
  private mcpManager: MCPManager;
  private config: AgentIntegrationConfig;
  private activeWorkflows: Map<string, WorkflowContext> = new Map();
  private agentCapabilities: Map<string, string[]> = new Map();

  constructor(mcpManager: MCPManager, config: Partial<AgentIntegrationConfig> = {}) {
    super();
    
    this.mcpManager = mcpManager;
    this.config = {
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
      ...config
    };

    this.initializeAgentCapabilities();
  }

  /**
   * Process agent request with appropriate MCP clients
   */
  async processAgentRequest<T = any>(request: AgentRequest<T>): Promise<AgentResponse> {
    try {
      this.validateAgentRequest(request);

      const agentId = request.context.agentId;
      const capabilities = this.agentCapabilities.get(agentId);

      if (!capabilities || !capabilities.includes(request.action)) {
        throw new MCPError(
          `Agent ${agentId} does not support action: ${request.action}`,
          'bmad-integrator'
        );
      }

      // Route request to appropriate handler based on agent
      switch (agentId) {
        case 'trading-orchestrator':
          return await this.handleTradingOrchestratorRequest(request);
        
        case 'strategy-designer':
          return await this.handleStrategyDesignerRequest(request);
        
        case 'backtesting-engineer':
          return await this.handleBacktestingEngineerRequest(request);
        
        case 'execution-engineer':
          return await this.handleExecutionEngineerRequest(request);
        
        case 'indicator-developer':
          return await this.handleIndicatorDeveloperRequest(request);
        
        default:
          throw new MCPError(`Unknown agent: ${agentId}`, 'bmad-integrator');
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Execute a complete workflow across multiple agents
   */
  async executeWorkflow(workflow: WorkflowContext): Promise<Record<string, any>> {
    const workflowId = workflow.workflowId;
    this.activeWorkflows.set(workflowId, workflow);

    try {
      this.emit('workflowStarted', workflowId);

      const results: Record<string, any> = {};

      for (let i = 0; i < workflow.steps.length; i++) {
        const step = workflow.steps[i];
        
        // Check dependencies
        if (!this.areDependenciesMet(step, results)) {
          throw new MCPError(
            `Dependencies not met for step ${step.id}`,
            'bmad-integrator'
          );
        }

        workflow.currentStep = i;
        this.emit('workflowStepStarted', workflowId, step.id);

        // Execute step
        const stepResult = await this.executeWorkflowStep(step, results);
        results[step.id] = stepResult;

        this.emit('workflowStepCompleted', workflowId, step.id, stepResult);
      }

      workflow.results = results;
      this.emit('workflowCompleted', workflowId, results);

      return results;
    } catch (error) {
      this.emit('workflowError', workflowId, error);
      throw error;
    } finally {
      this.activeWorkflows.delete(workflowId);
    }
  }

  /**
   * Handle Trading Orchestrator requests
   */
  private async handleTradingOrchestratorRequest(request: AgentRequest): Promise<AgentResponse> {
    const { action, parameters } = request;
    
    switch (action) {
      case 'generate-signals':
        return await this.generateTradingSignals(parameters);
      
      case 'review-portfolio':
        return await this.reviewPortfolio(parameters);
      
      case 'orchestrate-strategy':
        return await this.orchestrateStrategy(parameters);
      
      case 'sync-playbook':
        return await this.syncPlaybook(parameters);
      
      default:
        throw new MCPError(`Unknown trading orchestrator action: ${action}`, 'trading-orchestrator');
    }
  }

  /**
   * Handle Strategy Designer requests
   */
  private async handleStrategyDesignerRequest(request: AgentRequest): Promise<AgentResponse> {
    const { action, parameters } = request;
    
    switch (action) {
      case 'design-strategy':
        return await this.designStrategy(parameters);
      
      case 'formalize-concept':
        return await this.formalizeConcept(parameters);
      
      case 'create-indicators':
        return await this.createCustomIndicators(parameters);
      
      case 'validate-logic':
        return await this.validateStrategyLogic(parameters);
      
      default:
        throw new MCPError(`Unknown strategy designer action: ${action}`, 'strategy-designer');
    }
  }

  /**
   * Handle Backtesting Engineer requests
   */
  private async handleBacktestingEngineerRequest(request: AgentRequest): Promise<AgentResponse> {
    const { action, parameters } = request;
    
    switch (action) {
      case 'run-backtest':
        return await this.runComprehensiveBacktest(parameters);
      
      case 'optimize-parameters':
        return await this.optimizeStrategyParameters(parameters);
      
      case 'analyze-performance':
        return await this.analyzePerformance(parameters);
      
      case 'compare-strategies':
        return await this.compareStrategies(parameters);
      
      case 'monte-carlo-simulation':
        return await this.runMonteCarloAnalysis(parameters);
      
      default:
        throw new MCPError(`Unknown backtesting engineer action: ${action}`, 'backtesting-engineer');
    }
  }

  /**
   * Handle Execution Engineer requests
   */
  private async handleExecutionEngineerRequest(request: AgentRequest): Promise<AgentResponse> {
    const { action, parameters } = request;
    
    switch (action) {
      case 'execute-trades':
        return await this.executeTrades(parameters);
      
      case 'manage-risk':
        return await this.manageRisk(parameters);
      
      case 'monitor-positions':
        return await this.monitorPositions(parameters);
      
      case 'update-orders':
        return await this.updateOrders(parameters);
      
      default:
        throw new MCPError(`Unknown execution engineer action: ${action}`, 'execution-engineer');
    }
  }

  /**
   * Handle Indicator Developer requests
   */
  private async handleIndicatorDeveloperRequest(request: AgentRequest): Promise<AgentResponse> {
    const { action, parameters } = request;
    
    switch (action) {
      case 'develop-indicator':
        return await this.developIndicator(parameters);
      
      case 'test-indicator':
        return await this.testIndicator(parameters);
      
      case 'optimize-indicator':
        return await this.optimizeIndicator(parameters);
      
      case 'calculate-indicators':
        return await this.calculateIndicators(parameters);
      
      default:
        throw new MCPError(`Unknown indicator developer action: ${action}`, 'indicator-developer');
    }
  }

  /**
   * Generate trading signals using multiple data sources and indicators
   */
  private async generateTradingSignals(parameters: any): Promise<AgentResponse> {
    const polygonClient = this.mcpManager.getClient<PolygonClient>('polygon');
    const talibClient = this.mcpManager.getClient<TALibClient>('talib');
    const osEngineClient = this.mcpManager.getClient<OsEngineClient>('osengine');

    if (!polygonClient || !talibClient || !osEngineClient) {
      throw new MCPError('Required clients not available', 'bmad-integrator');
    }

    // Get market data
    const marketData = await polygonClient.getHistoricalData({
      symbol: parameters.symbol,
      timeframe: parameters.timeframe || '1Day',
      limit: parameters.limit || 100
    });

    // Calculate technical indicators
    const closePrices = marketData.map(d => d.close);
    const [rsi, emaFast, emaSlow] = await Promise.all([
      talibClient.calculateRSI(closePrices, 14),
      talibClient.calculateEMA(closePrices, 9),
      talibClient.calculateEMA(closePrices, 21)
    ]);

    // Generate signals using technical analysis
    const signals = await talibClient.generateSignals(marketData, [
      { name: 'RSI', parameters: { period: 14 }, inputFields: ['close'], outputFields: ['rsi'] },
      { name: 'EMA_CROSS', parameters: { fast: 9, slow: 21 }, inputFields: ['close'], outputFields: ['signal'] }
    ]);

    // Execute signals if auto-execution is enabled
    if (parameters.autoExecute && signals.buySignals.length > 0) {
      const latestSignal = signals.buySignals[signals.buySignals.length - 1];
      await osEngineClient.executeTradeSignal({
        symbol: parameters.symbol,
        action: 'buy',
        quantity: parameters.quantity || 1,
        confidence: latestSignal.confidence,
        strategy: 'technical-analysis',
        timestamp: new Date()
      });
    }

    return {
      success: true,
      data: {
        signals,
        marketData: marketData.slice(-10), // Last 10 bars
        indicators: { rsi, emaFast, emaSlow },
        analysis: {
          trend: signals.summary.marketTrend,
          strength: signals.summary.signalStrength,
          recommendation: signals.buySignals.length > signals.sellSignals.length ? 'BUY' : 'SELL'
        }
      },
      recommendations: [
        'Monitor RSI for oversold/overbought conditions',
        'Watch for EMA crossovers for trend confirmation',
        'Consider volume confirmation for signal strength'
      ]
    };
  }

  /**
   * Design a new trading strategy
   */
  private async designStrategy(parameters: any): Promise<AgentResponse> {
    const notionClient = this.mcpManager.getClient<NotionClient>('notion');
    const backtestingClient = this.mcpManager.getClient<BacktestingClient>('backtesting');

    if (!notionClient || !backtestingClient) {
      throw new MCPError('Required clients not available', 'bmad-integrator');
    }

    // Create strategy page in Notion
    const strategyPage = await notionClient.createStrategyPage({
      title: parameters.strategyName,
      description: parameters.description,
      parameters: parameters.strategyParameters,
      status: 'draft',
      tags: parameters.tags || ['custom', 'bmad-generated']
    });

    // Generate strategy template code
    const strategyCode = await backtestingClient.generateStrategyTemplate(
      parameters.strategyName,
      parameters.indicators || ['RSI', 'EMA'],
      parameters.entryRules || ['RSI < 30', 'EMA_CROSS_UP'],
      parameters.exitRules || ['RSI > 70', 'EMA_CROSS_DOWN']
    );

    // Perform initial validation backtest
    let backtestResult: BacktestResult | null = null;
    if (parameters.validateWithBacktest) {
      const strategyDefinition = {
        name: parameters.strategyName,
        code: strategyCode,
        parameters: parameters.strategyParameters,
        indicators: parameters.indicators || ['RSI', 'EMA']
      };

      // Get sample data for validation
      const polygonClient = this.mcpManager.getClient<PolygonClient>('polygon');
      const sampleData = await polygonClient?.getHistoricalData({
        symbol: parameters.testSymbol || 'SPY',
        timeframe: '1Day',
        limit: 252 // 1 year of data
      });

      if (sampleData) {
        backtestResult = await backtestingClient.runCustomStrategyBacktest(
          strategyDefinition,
          sampleData
        );

        // Update Notion page with backtest results
        if (this.config.autoSyncNotionResults) {
          await notionClient.addBacktestResults(strategyPage.id, [backtestResult]);
        }
      }
    }

    return {
      success: true,
      data: {
        strategy: strategyPage,
        code: strategyCode,
        backtestResult,
        validation: {
          hasBacktest: !!backtestResult,
          sharpeRatio: backtestResult?.sharpeRatio,
          totalReturn: backtestResult?.totalReturn
        }
      },
      recommendations: [
        'Review generated strategy code for accuracy',
        'Perform comprehensive backtesting before live deployment',
        'Consider parameter optimization',
        'Test strategy on multiple timeframes and symbols'
      ],
      nextActions: [
        'run-comprehensive-backtest',
        'optimize-parameters',
        'validate-on-multiple-symbols'
      ]
    };
  }

  /**
   * Run comprehensive backtest with full analysis
   */
  private async runComprehensiveBacktest(parameters: any): Promise<AgentResponse> {
    const backtestingClient = this.mcpManager.getClient<BacktestingClient>('backtesting');
    const polygonClient = this.mcpManager.getClient<PolygonClient>('polygon');
    const notionClient = this.mcpManager.getClient<NotionClient>('notion');

    if (!backtestingClient || !polygonClient) {
      throw new MCPError('Required clients not available', 'bmad-integrator');
    }

    // Get historical data
    const historicalData = await polygonClient.getHistoricalData({
      symbol: parameters.symbol,
      timeframe: parameters.timeframe || '1Day',
      from: parameters.startDate ? new Date(parameters.startDate) : new Date(Date.now() - 365 * 24 * 60 * 60 * 1000),
      to: parameters.endDate ? new Date(parameters.endDate) : new Date(),
      limit: parameters.limit || 1000
    });

    // Run backtest
    const backtestResult = await backtestingClient.runCustomStrategyBacktest(
      parameters.strategy,
      historicalData,
      {
        cash: parameters.initialCapital || 100000,
        commission: parameters.commission || 0.001
      }
    );

    // Generate visualization
    const visualization = await backtestingClient.generateVisualization(backtestResult);

    // Calculate detailed performance metrics
    const performanceMetrics = await backtestingClient.calculatePerformanceMetrics(
      backtestResult.trades,
      backtestResult.equity,
      parameters.initialCapital || 100000
    );

    // Update Notion if strategy page is provided
    if (parameters.notionPageId && notionClient) {
      await notionClient.addBacktestResults(parameters.notionPageId, [backtestResult]);
    }

    return {
      success: true,
      data: {
        backtestResult,
        performanceMetrics,
        visualization,
        analysis: {
          profitability: performanceMetrics.totalReturn > 0 ? 'Profitable' : 'Unprofitable',
          riskAdjustedReturn: performanceMetrics.sharpeRatio,
          consistency: performanceMetrics.winRate,
          recommendation: this.generateBacktestRecommendation(performanceMetrics)
        }
      },
      recommendations: this.generateBacktestRecommendations(performanceMetrics),
      nextActions: [
        'optimize-parameters',
        'test-different-timeframes',
        'run-monte-carlo-simulation',
        'compare-with-benchmark'
      ]
    };
  }

  /**
   * Execute trades based on signals
   */
  private async executeTrades(parameters: any): Promise<AgentResponse> {
    const osEngineClient = this.mcpManager.getClient<OsEngineClient>('osengine');
    
    if (!osEngineClient) {
      throw new MCPError('OsEngine client not available', 'bmad-integrator');
    }

    const results: any[] = [];

    // Execute each trade signal
    for (const signal of parameters.signals || []) {
      try {
        const order = await osEngineClient.executeTradeSignal(signal);
        if (order) {
          results.push({
            signal,
            order,
            status: 'executed'
          });
        } else {
          results.push({
            signal,
            status: 'skipped',
            reason: 'Signal did not meet execution criteria'
          });
        }
      } catch (error) {
        results.push({
          signal,
          status: 'failed',
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Get updated portfolio status
    const portfolioStatus = await osEngineClient.getPortfolioStatus();
    const riskMetrics = await osEngineClient.calculateRiskMetrics();

    return {
      success: true,
      data: {
        executionResults: results,
        portfolioStatus,
        riskMetrics,
        summary: {
          executedCount: results.filter(r => r.status === 'executed').length,
          skippedCount: results.filter(r => r.status === 'skipped').length,
          failedCount: results.filter(r => r.status === 'failed').length
        }
      },
      recommendations: [
        'Monitor executed positions closely',
        'Review failed executions for improvements',
        'Ensure risk limits are appropriate'
      ]
    };
  }

  private initializeAgentCapabilities(): void {
    this.agentCapabilities.set('trading-orchestrator', [
      'generate-signals',
      'review-portfolio',
      'orchestrate-strategy',
      'sync-playbook'
    ]);

    this.agentCapabilities.set('strategy-designer', [
      'design-strategy',
      'formalize-concept',
      'create-indicators',
      'validate-logic'
    ]);

    this.agentCapabilities.set('backtesting-engineer', [
      'run-backtest',
      'optimize-parameters',
      'analyze-performance',
      'compare-strategies',
      'monte-carlo-simulation'
    ]);

    this.agentCapabilities.set('execution-engineer', [
      'execute-trades',
      'manage-risk',
      'monitor-positions',
      'update-orders'
    ]);

    this.agentCapabilities.set('indicator-developer', [
      'develop-indicator',
      'test-indicator',
      'optimize-indicator',
      'calculate-indicators'
    ]);
  }

  private validateAgentRequest(request: AgentRequest): void {
    if (!request.context?.agentId) {
      throw new MCPError('Agent ID is required', 'bmad-integrator');
    }

    if (!request.action) {
      throw new MCPError('Action is required', 'bmad-integrator');
    }

    if (!this.config.enabledAgents.includes(request.context.agentId)) {
      throw new MCPError(`Agent ${request.context.agentId} is not enabled`, 'bmad-integrator');
    }
  }

  private areDependenciesMet(step: WorkflowStep, results: Record<string, any>): boolean {
    return step.dependencies.every(dep => results.hasOwnProperty(dep));
  }

  private async executeWorkflowStep(step: WorkflowStep, previousResults: Record<string, any>): Promise<any> {
    const request: AgentRequest = {
      action: step.action,
      parameters: {
        ...step.parameters,
        ...previousResults // Include previous step results
      },
      context: {
        agentId: step.agentId,
        sessionId: `workflow_${Date.now()}`,
        parameters: step.parameters,
        mcpClients: {}
      }
    };

    const response = await this.processAgentRequest(request);
    
    if (!response.success) {
      throw new MCPError(`Workflow step ${step.id} failed: ${response.error}`, 'bmad-integrator');
    }

    return response.data;
  }

  private generateBacktestRecommendation(metrics: any): string {
    if (metrics.sharpeRatio > 1.5 && metrics.totalReturn > 0.15) {
      return 'Excellent strategy - consider live deployment';
    } else if (metrics.sharpeRatio > 1.0 && metrics.totalReturn > 0.08) {
      return 'Good strategy - optimize parameters before deployment';
    } else if (metrics.sharpeRatio > 0.5) {
      return 'Moderate strategy - needs significant improvements';
    } else {
      return 'Poor strategy - requires major redesign';
    }
  }

  private generateBacktestRecommendations(metrics: any): string[] {
    const recommendations: string[] = [];
    
    if (metrics.maxDrawdown > 0.20) {
      recommendations.push('Consider implementing stronger stop-loss rules');
    }
    
    if (metrics.winRate < 0.4) {
      recommendations.push('Review entry criteria to improve win rate');
    }
    
    if (metrics.sharpeRatio < 1.0) {
      recommendations.push('Focus on risk-adjusted returns optimization');
    }
    
    if (metrics.totalTrades < 30) {
      recommendations.push('Test on longer time periods for statistical significance');
    }

    return recommendations;
  }

  // Placeholder methods for other operations - implement as needed
  private async reviewPortfolio(parameters: any): Promise<AgentResponse> {
    // Implementation for portfolio review
    return { success: true, data: {} };
  }

  private async orchestrateStrategy(parameters: any): Promise<AgentResponse> {
    // Implementation for strategy orchestration
    return { success: true, data: {} };
  }

  private async syncPlaybook(parameters: any): Promise<AgentResponse> {
    // Implementation for playbook synchronization
    return { success: true, data: {} };
  }

  private async formalizeConcept(parameters: any): Promise<AgentResponse> {
    // Implementation for concept formalization
    return { success: true, data: {} };
  }

  private async createCustomIndicators(parameters: any): Promise<AgentResponse> {
    // Implementation for custom indicator creation
    return { success: true, data: {} };
  }

  private async validateStrategyLogic(parameters: any): Promise<AgentResponse> {
    // Implementation for strategy logic validation
    return { success: true, data: {} };
  }

  private async optimizeStrategyParameters(parameters: any): Promise<AgentResponse> {
    // Implementation for parameter optimization
    return { success: true, data: {} };
  }

  private async analyzePerformance(parameters: any): Promise<AgentResponse> {
    // Implementation for performance analysis
    return { success: true, data: {} };
  }

  private async compareStrategies(parameters: any): Promise<AgentResponse> {
    // Implementation for strategy comparison
    return { success: true, data: {} };
  }

  private async runMonteCarloAnalysis(parameters: any): Promise<AgentResponse> {
    // Implementation for Monte Carlo analysis
    return { success: true, data: {} };
  }

  private async manageRisk(parameters: any): Promise<AgentResponse> {
    // Implementation for risk management
    return { success: true, data: {} };
  }

  private async monitorPositions(parameters: any): Promise<AgentResponse> {
    // Implementation for position monitoring
    return { success: true, data: {} };
  }

  private async updateOrders(parameters: any): Promise<AgentResponse> {
    // Implementation for order updates
    return { success: true, data: {} };
  }

  private async developIndicator(parameters: any): Promise<AgentResponse> {
    // Implementation for indicator development
    return { success: true, data: {} };
  }

  private async testIndicator(parameters: any): Promise<AgentResponse> {
    // Implementation for indicator testing
    return { success: true, data: {} };
  }

  private async optimizeIndicator(parameters: any): Promise<AgentResponse> {
    // Implementation for indicator optimization
    return { success: true, data: {} };
  }

  private async calculateIndicators(parameters: any): Promise<AgentResponse> {
    // Implementation for indicator calculations
    return { success: true, data: {} };
  }
}