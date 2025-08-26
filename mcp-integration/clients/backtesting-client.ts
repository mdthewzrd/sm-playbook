/**
 * BacktestingClient - Interface with backtesting.py for comprehensive strategy testing
 * 
 * This client provides integration with the backtesting.py library for:
 * - Running backtest simulations using custom strategies
 * - Performance metrics calculation and analysis
 * - Visualization of backtest results
 * - Portfolio and risk analysis
 */

import {
  BaseMCPClient,
  MCPServerConfig,
  MCPHealthCheck,
  MCPRequest,
  MCPResponse,
  Strategy,
  BacktestRequest,
  BacktestResult,
  Trade,
  OHLCV,
  MCPError,
  MCPValidationError
} from '../types';

export interface BacktestingClientConfig extends MCPServerConfig {
  defaultCommission: number;
  defaultCash: number;
  exclusiveOrders: boolean;
  tradeOnClose: boolean;
}

export interface CustomStrategyDefinition {
  name: string;
  code: string; // Python code for the strategy
  parameters: Record<string, any>;
  indicators: string[];
}

export interface BacktestConfiguration {
  cash: number;
  commission: number;
  margin: number;
  tradeOnClose: boolean;
  hedging: boolean;
  exclusiveOrders: boolean;
}

export interface PerformanceMetrics {
  // Returns
  totalReturn: number;
  annualizedReturn: number;
  volatility: number;
  
  // Risk metrics
  sharpeRatio: number;
  sortino: number;
  calmar: number;
  maxDrawdown: number;
  maxDrawdownDuration: number;
  
  // Trade statistics
  winRate: number;
  lossRate: number;
  bestTrade: number;
  worstTrade: number;
  avgWinningTrade: number;
  avgLosingTrade: number;
  totalTrades: number;
  
  // Additional metrics
  profitFactor: number;
  expectancy: number;
  sqn: number; // System Quality Number
}

export interface BacktestVisualization {
  equityCurve: OHLCV[];
  drawdownCurve: OHLCV[];
  returnsDistribution: number[];
  monthlyReturns: Record<string, number>;
  tradeAnalysis: {
    winningTrades: Trade[];
    losingTrades: Trade[];
    profitByMonth: Record<string, number>;
  };
}

export interface OptimizationRequest {
  strategy: CustomStrategyDefinition;
  data: OHLCV[];
  parameterRanges: Record<string, { min: number; max: number; step: number }>;
  optimizationMetric: 'sharpe' | 'return' | 'sqn' | 'calmar';
  maxIterations?: number;
}

export interface OptimizationResult {
  bestParameters: Record<string, any>;
  bestScore: number;
  allResults: Array<{
    parameters: Record<string, any>;
    score: number;
    metrics: PerformanceMetrics;
  }>;
  convergenceData: number[];
}

export class BacktestingClient extends BaseMCPClient {
  private clientConfig: BacktestingClientConfig;

  constructor(config: BacktestingClientConfig) {
    super('backtesting', config);
    this.clientConfig = {
      defaultCommission: 0.001,
      defaultCash: 100000,
      exclusiveOrders: true,
      tradeOnClose: false,
      ...config
    };
  }

  async connect(): Promise<void> {
    try {
      // Test connection by checking if backtesting.py is available
      const response = await this.request<void, string>({
        method: 'system/version'
      });

      if (response.success) {
        this.connectionStatus = {
          connected: true,
          lastHeartbeat: new Date(),
          retryCount: 0
        };
      } else {
        throw new MCPError(`Connection failed: ${response.error}`, this.serverId);
      }
    } catch (error) {
      this.connectionStatus = {
        connected: false,
        error: error instanceof Error ? error.message : String(error),
        retryCount: this.connectionStatus.retryCount + 1
      };
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    this.connectionStatus = {
      connected: false,
      retryCount: 0
    };
  }

  async healthCheck(): Promise<MCPHealthCheck> {
    const startTime = Date.now();
    
    try {
      const response = await this.request<void, string>({
        method: 'system/health',
        timeout: 5000
      });

      const responseTime = Date.now() - startTime;

      return {
        serverId: this.serverId,
        status: response.success ? 'healthy' : 'unhealthy',
        responseTime,
        lastCheck: new Date(),
        error: response.error
      };
    } catch (error) {
      return {
        serverId: this.serverId,
        status: 'unhealthy',
        responseTime: Date.now() - startTime,
        lastCheck: new Date(),
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  async request<T, R>(request: MCPRequest<T>): Promise<MCPResponse<R>> {
    const startTime = Date.now();
    
    try {
      const result = await this.makeBacktestApiCall(request.method, request.params);
      
      return {
        success: true,
        data: result as R,
        timestamp: new Date(),
        responseTime: Date.now() - startTime
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        timestamp: new Date(),
        responseTime: Date.now() - startTime
      };
    }
  }

  /**
   * Run a comprehensive backtest with the given strategy and data
   */
  async runBacktest(request: BacktestRequest): Promise<BacktestResult> {
    this.validateBacktestRequest(request);

    const response = await this.request<BacktestRequest, BacktestResult>({
      method: 'backtest/run',
      params: request,
      timeout: 60000 // 60 seconds for backtests
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Backtest failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Run backtest with custom strategy code
   */
  async runCustomStrategyBacktest(
    strategyDefinition: CustomStrategyDefinition,
    data: OHLCV[],
    config?: Partial<BacktestConfiguration>
  ): Promise<BacktestResult> {
    const backtestConfig: BacktestConfiguration = {
      cash: this.clientConfig.defaultCash,
      commission: this.clientConfig.defaultCommission,
      margin: 1.0,
      tradeOnClose: this.clientConfig.tradeOnClose,
      hedging: false,
      exclusiveOrders: this.clientConfig.exclusiveOrders,
      ...config
    };

    const response = await this.request<{
      strategy: CustomStrategyDefinition;
      data: OHLCV[];
      config: BacktestConfiguration;
    }, BacktestResult>({
      method: 'backtest/custom-strategy',
      params: {
        strategy: strategyDefinition,
        data,
        config: backtestConfig
      },
      timeout: 120000 // 2 minutes for custom strategies
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Custom strategy backtest failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Calculate detailed performance metrics
   */
  async calculatePerformanceMetrics(
    trades: Trade[],
    equity: OHLCV[],
    initialCapital: number
  ): Promise<PerformanceMetrics> {
    const response = await this.request<{
      trades: Trade[];
      equity: OHLCV[];
      initialCapital: number;
    }, PerformanceMetrics>({
      method: 'analysis/performance-metrics',
      params: { trades, equity, initialCapital }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Performance metrics calculation failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Generate visualization data for backtest results
   */
  async generateVisualization(result: BacktestResult): Promise<BacktestVisualization> {
    const response = await this.request<BacktestResult, BacktestVisualization>({
      method: 'visualization/generate',
      params: result
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Visualization generation failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Optimize strategy parameters
   */
  async optimizeStrategy(request: OptimizationRequest): Promise<OptimizationResult> {
    this.validateOptimizationRequest(request);

    const response = await this.request<OptimizationRequest, OptimizationResult>({
      method: 'optimization/run',
      params: request,
      timeout: 300000 // 5 minutes for optimization
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Strategy optimization failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Run Monte Carlo simulation on strategy
   */
  async runMonteCarloSimulation(
    strategy: CustomStrategyDefinition,
    data: OHLCV[],
    iterations: number = 1000
  ): Promise<{
    meanReturn: number;
    medianReturn: number;
    standardDeviation: number;
    confidenceIntervals: Record<string, { lower: number; upper: number }>;
    worstCaseScenario: BacktestResult;
    bestCaseScenario: BacktestResult;
    allResults: BacktestResult[];
  }> {
    const response = await this.request<{
      strategy: CustomStrategyDefinition;
      data: OHLCV[];
      iterations: number;
    }, any>({
      method: 'monte-carlo/run',
      params: { strategy, data, iterations },
      timeout: 600000 // 10 minutes for Monte Carlo
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Monte Carlo simulation failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Compare multiple strategies side by side
   */
  async compareStrategies(
    strategies: CustomStrategyDefinition[],
    data: OHLCV[],
    config?: Partial<BacktestConfiguration>
  ): Promise<{
    results: BacktestResult[];
    comparison: {
      bestByReturn: string;
      bestBySharpe: string;
      bestByDrawdown: string;
      correlationMatrix: number[][];
      rankingByMetric: Record<string, string[]>;
    };
  }> {
    const response = await this.request<{
      strategies: CustomStrategyDefinition[];
      data: OHLCV[];
      config?: Partial<BacktestConfiguration>;
    }, any>({
      method: 'comparison/run',
      params: { strategies, data, config },
      timeout: 300000 // 5 minutes for comparison
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Strategy comparison failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Perform walk-forward analysis
   */
  async walkForwardAnalysis(
    strategy: CustomStrategyDefinition,
    data: OHLCV[],
    optimizationWindow: number,
    testingWindow: number
  ): Promise<{
    periods: Array<{
      optimizationStart: Date;
      optimizationEnd: Date;
      testingStart: Date;
      testingEnd: Date;
      optimizedParameters: Record<string, any>;
      testingResult: BacktestResult;
    }>;
    overallMetrics: PerformanceMetrics;
    stability: number; // Parameter stability across periods
  }> {
    const response = await this.request<{
      strategy: CustomStrategyDefinition;
      data: OHLCV[];
      optimizationWindow: number;
      testingWindow: number;
    }, any>({
      method: 'walk-forward/run',
      params: { strategy, data, optimizationWindow, testingWindow },
      timeout: 900000 // 15 minutes for walk-forward analysis
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Walk-forward analysis failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Generate strategy template code
   */
  async generateStrategyTemplate(
    strategyName: string,
    indicators: string[],
    entryRules: string[],
    exitRules: string[]
  ): Promise<string> {
    const response = await this.request<{
      name: string;
      indicators: string[];
      entryRules: string[];
      exitRules: string[];
    }, string>({
      method: 'template/generate',
      params: {
        name: strategyName,
        indicators,
        entryRules,
        exitRules
      }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Template generation failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  private validateBacktestRequest(request: BacktestRequest): void {
    if (!request.strategy) {
      throw new MCPValidationError(this.serverId, 'Strategy is required');
    }

    if (!request.symbol) {
      throw new MCPValidationError(this.serverId, 'Symbol is required');
    }

    if (request.initialCapital <= 0) {
      throw new MCPValidationError(this.serverId, 'Initial capital must be positive');
    }

    if (request.startDate >= request.endDate) {
      throw new MCPValidationError(this.serverId, 'Start date must be before end date');
    }
  }

  private validateOptimizationRequest(request: OptimizationRequest): void {
    if (!request.strategy) {
      throw new MCPValidationError(this.serverId, 'Strategy is required for optimization');
    }

    if (!request.data || request.data.length === 0) {
      throw new MCPValidationError(this.serverId, 'Data is required for optimization');
    }

    if (!request.parameterRanges || Object.keys(request.parameterRanges).length === 0) {
      throw new MCPValidationError(this.serverId, 'Parameter ranges are required');
    }

    // Validate parameter ranges
    for (const [param, range] of Object.entries(request.parameterRanges)) {
      if (range.min >= range.max) {
        throw new MCPValidationError(
          this.serverId,
          `Invalid range for parameter ${param}: min must be less than max`
        );
      }

      if (range.step <= 0) {
        throw new MCPValidationError(
          this.serverId,
          `Invalid step for parameter ${param}: must be positive`
        );
      }
    }
  }

  private async makeBacktestApiCall(method: string, params?: any): Promise<any> {
    // Mock implementation - in reality this would communicate with the MCP server
    switch (method) {
      case 'system/version':
        return 'backtesting.py v0.3.3';

      case 'system/health':
        return 'OK';

      case 'backtest/run':
        return this.mockBacktestResult();

      case 'backtest/custom-strategy':
        return this.mockBacktestResult();

      case 'analysis/performance-metrics':
        return this.mockPerformanceMetrics();

      case 'visualization/generate':
        return this.mockVisualization();

      case 'optimization/run':
        return this.mockOptimizationResult();

      case 'template/generate':
        return this.mockStrategyTemplate(params);

      default:
        throw new MCPError(`Unknown method: ${method}`, this.serverId);
    }
  }

  private mockBacktestResult(): BacktestResult {
    return {
      totalReturn: 0.15,
      annualizedReturn: 0.12,
      sharpeRatio: 1.45,
      maxDrawdown: -0.08,
      winRate: 0.65,
      totalTrades: 150,
      equity: [],
      trades: [],
      metrics: {
        volatility: 0.18,
        sortino: 1.67,
        calmar: 1.5,
        profitFactor: 1.8
      }
    };
  }

  private mockPerformanceMetrics(): PerformanceMetrics {
    return {
      totalReturn: 0.15,
      annualizedReturn: 0.12,
      volatility: 0.18,
      sharpeRatio: 1.45,
      sortino: 1.67,
      calmar: 1.5,
      maxDrawdown: -0.08,
      maxDrawdownDuration: 45,
      winRate: 0.65,
      lossRate: 0.35,
      bestTrade: 0.05,
      worstTrade: -0.03,
      avgWinningTrade: 0.02,
      avgLosingTrade: -0.015,
      totalTrades: 150,
      profitFactor: 1.8,
      expectancy: 0.001,
      sqn: 2.1
    };
  }

  private mockVisualization(): BacktestVisualization {
    return {
      equityCurve: [],
      drawdownCurve: [],
      returnsDistribution: [],
      monthlyReturns: {},
      tradeAnalysis: {
        winningTrades: [],
        losingTrades: [],
        profitByMonth: {}
      }
    };
  }

  private mockOptimizationResult(): OptimizationResult {
    return {
      bestParameters: { rsi_period: 14, threshold: 30 },
      bestScore: 1.45,
      allResults: [],
      convergenceData: []
    };
  }

  private mockStrategyTemplate(params: any): string {
    return `
from backtesting import Strategy
import pandas as pd

class ${params.name}(Strategy):
    def init(self):
        # Initialize indicators
        ${params.indicators.map((ind: string) => `# self.${ind} = self.I(...)`).join('\n        ')}
        
    def next(self):
        # Entry rules
        ${params.entryRules.map((rule: string) => `# ${rule}`).join('\n        ')}
        
        # Exit rules  
        ${params.exitRules.map((rule: string) => `# ${rule}`).join('\n        ')}
`;
  }
}