/**
 * TALibClient - Interface with TA-Lib for comprehensive technical analysis
 * 
 * This client provides integration with TA-Lib library for:
 * - Calculating technical indicators (EMA, RSI, MACD, etc.)
 * - Custom indicator parameters and batch calculations
 * - Special support for EMA clouds (72/89 and 9/20)
 * - Pattern recognition and signal generation
 */

import {
  BaseMCPClient,
  MCPServerConfig,
  MCPHealthCheck,
  MCPRequest,
  MCPResponse,
  TechnicalIndicatorRequest,
  TechnicalIndicatorResult,
  OHLCV,
  MCPError,
  MCPValidationError
} from '../types';

export interface TALibClientConfig extends MCPServerConfig {
  defaultLookbackPeriod: number;
  enablePatternRecognition: boolean;
  cachingEnabled: boolean;
}

export interface IndicatorConfig {
  name: string;
  parameters: Record<string, any>;
  inputFields: ('open' | 'high' | 'low' | 'close' | 'volume')[];
  outputFields: string[];
}

export interface EMACloudConfig {
  fastPeriod: number;
  slowPeriod: number;
  signalThreshold: number;
}

export interface BatchIndicatorRequest {
  data: OHLCV[];
  indicators: IndicatorConfig[];
  alignOutput: boolean;
}

export interface BatchIndicatorResult {
  indicators: Record<string, TechnicalIndicatorResult>;
  timestamp: Date[];
  metadata: {
    dataPoints: number;
    calculationTime: number;
    warnings: string[];
  };
}

export interface PatternRecognitionRequest {
  data: OHLCV[];
  patterns: string[];
  minConfidence: number;
}

export interface PatternRecognitionResult {
  patterns: Array<{
    name: string;
    confidence: number;
    startIndex: number;
    endIndex: number;
    direction: 'bullish' | 'bearish' | 'neutral';
    reliability: 'high' | 'medium' | 'low';
  }>;
  summary: {
    bullishPatterns: number;
    bearishPatterns: number;
    neutralPatterns: number;
    averageConfidence: number;
  };
}

export interface SignalAnalysis {
  buySignals: Array<{
    timestamp: Date;
    price: number;
    indicators: Record<string, number>;
    confidence: number;
    reason: string;
  }>;
  sellSignals: Array<{
    timestamp: Date;
    price: number;
    indicators: Record<string, number>;
    confidence: number;
    reason: string;
  }>;
  summary: {
    totalBuySignals: number;
    totalSellSignals: number;
    signalStrength: number;
    marketTrend: 'bullish' | 'bearish' | 'sideways';
  };
}

export interface CustomIndicatorDefinition {
  name: string;
  formula: string;
  dependencies: string[];
  parameters: Record<string, any>;
  validation: {
    minDataPoints: number;
    requiredFields: string[];
  };
}

export class TALibClient extends BaseMCPClient {
  private clientConfig: TALibClientConfig;
  private indicatorCache: Map<string, TechnicalIndicatorResult> = new Map();

  constructor(config: TALibClientConfig) {
    super('talib', config);
    this.clientConfig = {
      defaultLookbackPeriod: 20,
      enablePatternRecognition: true,
      cachingEnabled: true,
      ...config
    };
  }

  async connect(): Promise<void> {
    try {
      const response = await this.request<void, string[]>({
        method: 'system/functions'
      });

      if (response.success && response.data && response.data.length > 0) {
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
    this.indicatorCache.clear();
    this.connectionStatus = {
      connected: false,
      retryCount: 0
    };
  }

  async healthCheck(): Promise<MCPHealthCheck> {
    const startTime = Date.now();
    
    try {
      const response = await this.request<void, string[]>({
        method: 'system/functions',
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
      const result = await this.makeTALibApiCall(request.method, request.params);
      
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
   * Calculate Simple Moving Average (SMA)
   */
  async calculateSMA(data: number[], period: number): Promise<TechnicalIndicatorResult> {
    this.validateIndicatorInput(data, period);

    const cacheKey = `SMA_${period}_${this.hashArray(data)}`;
    if (this.clientConfig.cachingEnabled && this.indicatorCache.has(cacheKey)) {
      return this.indicatorCache.get(cacheKey)!;
    }

    const response = await this.request<{ data: number[]; period: number }, TechnicalIndicatorResult>({
      method: 'indicators/SMA',
      params: { data, period }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`SMA calculation failed: ${response.error}`, this.serverId);
    }

    if (this.clientConfig.cachingEnabled) {
      this.indicatorCache.set(cacheKey, response.data);
    }

    return response.data;
  }

  /**
   * Calculate Exponential Moving Average (EMA)
   */
  async calculateEMA(data: number[], period: number): Promise<TechnicalIndicatorResult> {
    this.validateIndicatorInput(data, period);

    const cacheKey = `EMA_${period}_${this.hashArray(data)}`;
    if (this.clientConfig.cachingEnabled && this.indicatorCache.has(cacheKey)) {
      return this.indicatorCache.get(cacheKey)!;
    }

    const response = await this.request<{ data: number[]; period: number }, TechnicalIndicatorResult>({
      method: 'indicators/EMA',
      params: { data, period }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`EMA calculation failed: ${response.error}`, this.serverId);
    }

    if (this.clientConfig.cachingEnabled) {
      this.indicatorCache.set(cacheKey, response.data);
    }

    return response.data;
  }

  /**
   * Calculate EMA Cloud (special support for 72/89 and 9/20)
   */
  async calculateEMACloud(
    data: number[],
    config: EMACloudConfig
  ): Promise<{
    fastEMA: TechnicalIndicatorResult;
    slowEMA: TechnicalIndicatorResult;
    cloudDirection: ('bullish' | 'bearish' | 'neutral')[];
    crossoverSignals: Array<{
      index: number;
      type: 'golden' | 'death';
      strength: number;
    }>;
  }> {
    const [fastEMA, slowEMA] = await Promise.all([
      this.calculateEMA(data, config.fastPeriod),
      this.calculateEMA(data, config.slowPeriod)
    ]);

    const cloudDirection: ('bullish' | 'bearish' | 'neutral')[] = [];
    const crossoverSignals: Array<{ index: number; type: 'golden' | 'death'; strength: number }> = [];

    for (let i = 0; i < Math.min(fastEMA.values.length, slowEMA.values.length); i++) {
      const fastValue = fastEMA.values[i];
      const slowValue = slowEMA.values[i];
      const diff = Math.abs(fastValue - slowValue);

      if (fastValue > slowValue) {
        cloudDirection.push(diff > config.signalThreshold ? 'bullish' : 'neutral');
      } else if (fastValue < slowValue) {
        cloudDirection.push(diff > config.signalThreshold ? 'bearish' : 'neutral');
      } else {
        cloudDirection.push('neutral');
      }

      // Detect crossovers
      if (i > 0) {
        const prevFast = fastEMA.values[i - 1];
        const prevSlow = slowEMA.values[i - 1];

        if (prevFast <= prevSlow && fastValue > slowValue) {
          crossoverSignals.push({
            index: i,
            type: 'golden',
            strength: Math.min((fastValue - slowValue) / slowValue, 1)
          });
        } else if (prevFast >= prevSlow && fastValue < slowValue) {
          crossoverSignals.push({
            index: i,
            type: 'death',
            strength: Math.min((slowValue - fastValue) / fastValue, 1)
          });
        }
      }
    }

    return { fastEMA, slowEMA, cloudDirection, crossoverSignals };
  }

  /**
   * Calculate Relative Strength Index (RSI)
   */
  async calculateRSI(data: number[], period: number = 14): Promise<TechnicalIndicatorResult> {
    this.validateIndicatorInput(data, period);

    const response = await this.request<{ data: number[]; period: number }, TechnicalIndicatorResult>({
      method: 'indicators/RSI',
      params: { data, period }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`RSI calculation failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Calculate MACD (Moving Average Convergence Divergence)
   */
  async calculateMACD(
    data: number[],
    fastPeriod: number = 12,
    slowPeriod: number = 26,
    signalPeriod: number = 9
  ): Promise<{
    macd: TechnicalIndicatorResult;
    signal: TechnicalIndicatorResult;
    histogram: TechnicalIndicatorResult;
  }> {
    const response = await this.request<{
      data: number[];
      fastPeriod: number;
      slowPeriod: number;
      signalPeriod: number;
    }, any>({
      method: 'indicators/MACD',
      params: { data, fastPeriod, slowPeriod, signalPeriod }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`MACD calculation failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Calculate Bollinger Bands
   */
  async calculateBollingerBands(
    data: number[],
    period: number = 20,
    standardDeviations: number = 2
  ): Promise<{
    upper: TechnicalIndicatorResult;
    middle: TechnicalIndicatorResult;
    lower: TechnicalIndicatorResult;
    bandWidth: TechnicalIndicatorResult;
    percentB: TechnicalIndicatorResult;
  }> {
    const response = await this.request<{
      data: number[];
      period: number;
      standardDeviations: number;
    }, any>({
      method: 'indicators/BBANDS',
      params: { data, period, standardDeviations }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Bollinger Bands calculation failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Calculate multiple indicators in batch for performance
   */
  async calculateBatchIndicators(request: BatchIndicatorRequest): Promise<BatchIndicatorResult> {
    this.validateBatchRequest(request);

    const response = await this.request<BatchIndicatorRequest, BatchIndicatorResult>({
      method: 'indicators/batch',
      params: request,
      timeout: 30000 // 30 seconds for batch processing
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Batch indicator calculation failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Perform pattern recognition analysis
   */
  async recognizePatterns(request: PatternRecognitionRequest): Promise<PatternRecognitionResult> {
    if (!this.clientConfig.enablePatternRecognition) {
      throw new MCPError('Pattern recognition is disabled', this.serverId);
    }

    this.validatePatternRequest(request);

    const response = await this.request<PatternRecognitionRequest, PatternRecognitionResult>({
      method: 'patterns/recognize',
      params: request,
      timeout: 15000 // 15 seconds for pattern recognition
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Pattern recognition failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Generate trading signals based on multiple indicators
   */
  async generateSignals(
    data: OHLCV[],
    indicatorConfigs: IndicatorConfig[]
  ): Promise<SignalAnalysis> {
    const response = await this.request<{
      data: OHLCV[];
      indicators: IndicatorConfig[];
    }, SignalAnalysis>({
      method: 'signals/generate',
      params: { data, indicators: indicatorConfigs },
      timeout: 20000
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Signal generation failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Create custom indicator with formula
   */
  async createCustomIndicator(definition: CustomIndicatorDefinition): Promise<string> {
    this.validateCustomIndicator(definition);

    const response = await this.request<CustomIndicatorDefinition, string>({
      method: 'indicators/custom/create',
      params: definition
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Custom indicator creation failed: ${response.error}`, this.serverId);
    }

    return response.data; // Returns the indicator ID
  }

  /**
   * Calculate custom indicator
   */
  async calculateCustomIndicator(
    indicatorId: string,
    data: OHLCV[],
    parameters?: Record<string, any>
  ): Promise<TechnicalIndicatorResult> {
    const response = await this.request<{
      indicatorId: string;
      data: OHLCV[];
      parameters?: Record<string, any>;
    }, TechnicalIndicatorResult>({
      method: 'indicators/custom/calculate',
      params: { indicatorId, data, parameters }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Custom indicator calculation failed: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get list of available TA-Lib functions
   */
  async getAvailableFunctions(): Promise<string[]> {
    const response = await this.request<void, string[]>({
      method: 'system/functions'
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get available functions: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get function documentation
   */
  async getFunctionInfo(functionName: string): Promise<{
    name: string;
    group: string;
    description: string;
    parameters: Array<{
      name: string;
      type: string;
      default?: any;
      description: string;
    }>;
    inputs: string[];
    outputs: string[];
  }> {
    const response = await this.request<{ functionName: string }, any>({
      method: 'system/function-info',
      params: { functionName }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get function info: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Clear indicator cache
   */
  clearCache(): void {
    this.indicatorCache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): {
    size: number;
    hitRate: number;
    memoryUsage: number;
  } {
    // Mock implementation - in reality would track actual cache statistics
    return {
      size: this.indicatorCache.size,
      hitRate: 0.85, // Mock 85% hit rate
      memoryUsage: this.indicatorCache.size * 1024 // Mock memory usage
    };
  }

  private validateIndicatorInput(data: number[], period: number): void {
    if (!Array.isArray(data) || data.length === 0) {
      throw new MCPValidationError(this.serverId, 'Data array is required and cannot be empty');
    }

    if (period <= 0) {
      throw new MCPValidationError(this.serverId, 'Period must be positive');
    }

    if (data.length < period) {
      throw new MCPValidationError(
        this.serverId,
        `Insufficient data: need at least ${period} data points, got ${data.length}`
      );
    }
  }

  private validateBatchRequest(request: BatchIndicatorRequest): void {
    if (!request.data || request.data.length === 0) {
      throw new MCPValidationError(this.serverId, 'Data is required for batch calculation');
    }

    if (!request.indicators || request.indicators.length === 0) {
      throw new MCPValidationError(this.serverId, 'At least one indicator is required');
    }

    for (const indicator of request.indicators) {
      if (!indicator.name) {
        throw new MCPValidationError(this.serverId, 'Indicator name is required');
      }
    }
  }

  private validatePatternRequest(request: PatternRecognitionRequest): void {
    if (!request.data || request.data.length === 0) {
      throw new MCPValidationError(this.serverId, 'Data is required for pattern recognition');
    }

    if (!request.patterns || request.patterns.length === 0) {
      throw new MCPValidationError(this.serverId, 'At least one pattern is required');
    }

    if (request.minConfidence < 0 || request.minConfidence > 1) {
      throw new MCPValidationError(this.serverId, 'Confidence must be between 0 and 1');
    }
  }

  private validateCustomIndicator(definition: CustomIndicatorDefinition): void {
    if (!definition.name) {
      throw new MCPValidationError(this.serverId, 'Custom indicator name is required');
    }

    if (!definition.formula) {
      throw new MCPValidationError(this.serverId, 'Custom indicator formula is required');
    }

    if (definition.validation.minDataPoints <= 0) {
      throw new MCPValidationError(this.serverId, 'Minimum data points must be positive');
    }
  }

  private hashArray(data: number[]): string {
    // Simple hash for caching - in production would use a proper hash function
    return data.slice(0, 10).join(',') + data.length;
  }

  private async makeTALibApiCall(method: string, params?: any): Promise<any> {
    // Mock implementation - in reality this would communicate with the MCP server
    switch (method) {
      case 'system/functions':
        return [
          'SMA', 'EMA', 'RSI', 'MACD', 'BBANDS', 'STOCH', 'ADX', 'ATR',
          'CCI', 'WILLR', 'MOM', 'ROC', 'STOCHRSI', 'AROON', 'SAR'
        ];

      case 'system/function-info':
        return this.mockFunctionInfo(params.functionName);

      case 'indicators/SMA':
        return this.mockIndicatorResult(params.data, params.period, 'SMA');

      case 'indicators/EMA':
        return this.mockIndicatorResult(params.data, params.period, 'EMA');

      case 'indicators/RSI':
        return this.mockIndicatorResult(params.data, params.period, 'RSI');

      case 'indicators/MACD':
        return this.mockMACDResult();

      case 'indicators/BBANDS':
        return this.mockBollingerBands();

      case 'indicators/batch':
        return this.mockBatchResult();

      case 'patterns/recognize':
        return this.mockPatternResult();

      case 'signals/generate':
        return this.mockSignalAnalysis();

      default:
        throw new MCPError(`Unknown method: ${method}`, this.serverId);
    }
  }

  private mockIndicatorResult(data: number[], period: number, type: string): TechnicalIndicatorResult {
    const values = data.slice(period - 1).map((_, i) => 
      type === 'RSI' ? Math.random() * 100 : data[i + period - 1] * (0.95 + Math.random() * 0.1)
    );

    return {
      values,
      metadata: {
        period,
        type,
        dataPoints: values.length
      }
    };
  }

  private mockMACDResult(): any {
    return {
      macd: { values: [], metadata: {} },
      signal: { values: [], metadata: {} },
      histogram: { values: [], metadata: {} }
    };
  }

  private mockBollingerBands(): any {
    return {
      upper: { values: [], metadata: {} },
      middle: { values: [], metadata: {} },
      lower: { values: [], metadata: {} },
      bandWidth: { values: [], metadata: {} },
      percentB: { values: [], metadata: {} }
    };
  }

  private mockBatchResult(): BatchIndicatorResult {
    return {
      indicators: {},
      timestamp: [],
      metadata: {
        dataPoints: 100,
        calculationTime: 150,
        warnings: []
      }
    };
  }

  private mockPatternResult(): PatternRecognitionResult {
    return {
      patterns: [
        {
          name: 'Hammer',
          confidence: 0.85,
          startIndex: 45,
          endIndex: 45,
          direction: 'bullish',
          reliability: 'high'
        }
      ],
      summary: {
        bullishPatterns: 1,
        bearishPatterns: 0,
        neutralPatterns: 0,
        averageConfidence: 0.85
      }
    };
  }

  private mockSignalAnalysis(): SignalAnalysis {
    return {
      buySignals: [],
      sellSignals: [],
      summary: {
        totalBuySignals: 0,
        totalSellSignals: 0,
        signalStrength: 0,
        marketTrend: 'sideways'
      }
    };
  }

  private mockFunctionInfo(functionName: string): any {
    return {
      name: functionName,
      group: 'Overlap Studies',
      description: `${functionName} indicator`,
      parameters: [],
      inputs: ['close'],
      outputs: [functionName.toLowerCase()]
    };
  }
}