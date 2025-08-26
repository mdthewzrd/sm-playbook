import { CandleData } from '../models/market_data';
import { 
  IndicatorValue, 
  EMACloudConfig, 
  ATRBandsConfig, 
  RSIGradientConfig,
  IndicatorCalculator 
} from '../models/indicator';
import { Timeframe } from '../types/common';

export interface IndicatorProcessingConfig {
  emaCloud: {
    enabled: boolean;
    configs: EMACloudConfig[];
  };
  atrBands: {
    enabled: boolean;
    config: ATRBandsConfig;
  };
  rsiGradient: {
    enabled: boolean;
    config: RSIGradientConfig;
  };
  momentum: {
    enabled: boolean;
    periods: number[];
  };
  volume: {
    enabled: boolean;
    period: number;
  };
  volatility: {
    enabled: boolean;
    period: number;
  };
  customIndicators: {
    enabled: boolean;
    definitions: CustomIndicatorDefinition[];
  };
}

export interface CustomIndicatorDefinition {
  name: string;
  formula: string;
  parameters: Record<string, number>;
  dependencies: string[];
}

export interface IndicatorResult {
  name: string;
  timeframe: Timeframe;
  values: IndicatorValue[];
  metadata: {
    calculatedAt: number;
    dataPoints: number;
    config: any;
    performance: {
      calculationTimeMs: number;
      memoryUsageKB: number;
    };
  };
}

export interface ProcessingBatch {
  id: string;
  symbol: string;
  timeframe: Timeframe;
  indicators: IndicatorResult[];
  status: 'pending' | 'processing' | 'completed' | 'failed';
  startTime: number;
  endTime?: number;
  errors: string[];
}

export class IndicatorProcessor {
  private calculator: IndicatorCalculator;
  private config: IndicatorProcessingConfig;
  private cache: Map<string, IndicatorResult>;
  private activeBatches: Map<string, ProcessingBatch>;

  constructor(config?: Partial<IndicatorProcessingConfig>) {
    this.calculator = new IndicatorCalculator();
    this.cache = new Map();
    this.activeBatches = new Map();
    
    this.config = {
      emaCloud: {
        enabled: true,
        configs: [
          { fastPeriod: 9, slowPeriod: 20 },
          { fastPeriod: 72, slowPeriod: 89 }
        ]
      },
      atrBands: {
        enabled: true,
        config: { period: 14, multiplier: 2.0, smoothing: 1 }
      },
      rsiGradient: {
        enabled: true,
        config: { period: 14, smoothing: 3 }
      },
      momentum: {
        enabled: true,
        periods: [14, 21, 50]
      },
      volume: {
        enabled: true,
        period: 20
      },
      volatility: {
        enabled: true,
        period: 20
      },
      customIndicators: {
        enabled: false,
        definitions: []
      },
      ...config
    };
  }

  async processIndicators(
    marketData: CandleData[],
    symbol: string,
    timeframe: Timeframe
  ): Promise<ProcessingBatch> {
    const batchId = `${symbol}_${timeframe}_${Date.now()}`;
    const batch: ProcessingBatch = {
      id: batchId,
      symbol,
      timeframe,
      indicators: [],
      status: 'processing',
      startTime: Date.now(),
      errors: []
    };

    this.activeBatches.set(batchId, batch);

    try {
      if (this.config.emaCloud.enabled) {
        for (const config of this.config.emaCloud.configs) {
          const result = await this.calculateEMACloud(marketData, symbol, timeframe, config);
          batch.indicators.push(result);
        }
      }

      if (this.config.atrBands.enabled) {
        const result = await this.calculateATRBands(marketData, symbol, timeframe, this.config.atrBands.config);
        batch.indicators.push(result);
      }

      if (this.config.rsiGradient.enabled) {
        const result = await this.calculateRSIGradient(marketData, symbol, timeframe, this.config.rsiGradient.config);
        batch.indicators.push(result);
      }

      if (this.config.momentum.enabled) {
        for (const period of this.config.momentum.periods) {
          const result = await this.calculateMomentum(marketData, symbol, timeframe, period);
          batch.indicators.push(result);
        }
      }

      if (this.config.volume.enabled) {
        const result = await this.calculateVolumeProfile(marketData, symbol, timeframe, this.config.volume.period);
        batch.indicators.push(result);
      }

      if (this.config.volatility.enabled) {
        const result = await this.calculateVolatility(marketData, symbol, timeframe, this.config.volatility.period);
        batch.indicators.push(result);
      }

      if (this.config.customIndicators.enabled) {
        for (const definition of this.config.customIndicators.definitions) {
          const result = await this.calculateCustomIndicator(marketData, symbol, timeframe, definition);
          batch.indicators.push(result);
        }
      }

      batch.status = 'completed';
      batch.endTime = Date.now();
      
    } catch (error) {
      batch.status = 'failed';
      batch.errors.push(`Processing failed: ${error.message}`);
      batch.endTime = Date.now();
    }

    return batch;
  }

  private async calculateEMACloud(
    data: CandleData[],
    symbol: string,
    timeframe: Timeframe,
    config: EMACloudConfig
  ): Promise<IndicatorResult> {
    const startTime = Date.now();
    const cacheKey = `emacloud_${symbol}_${timeframe}_${config.fastPeriod}_${config.slowPeriod}`;
    
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    const values = this.calculator.calculateEMACloud(data, config);
    const calculationTime = Date.now() - startTime;

    const result: IndicatorResult = {
      name: `EMA Cloud ${config.fastPeriod}/${config.slowPeriod}`,
      timeframe,
      values,
      metadata: {
        calculatedAt: Date.now(),
        dataPoints: values.length,
        config,
        performance: {
          calculationTimeMs: calculationTime,
          memoryUsageKB: this.estimateMemoryUsage(values)
        }
      }
    };

    this.cache.set(cacheKey, result);
    return result;
  }

  private async calculateATRBands(
    data: CandleData[],
    symbol: string,
    timeframe: Timeframe,
    config: ATRBandsConfig
  ): Promise<IndicatorResult> {
    const startTime = Date.now();
    const cacheKey = `atrbands_${symbol}_${timeframe}_${config.period}_${config.multiplier}`;
    
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    const values = this.calculator.calculateATRBands(data, config);
    const calculationTime = Date.now() - startTime;

    const result: IndicatorResult = {
      name: `ATR Bands ${config.period}`,
      timeframe,
      values,
      metadata: {
        calculatedAt: Date.now(),
        dataPoints: values.length,
        config,
        performance: {
          calculationTimeMs: calculationTime,
          memoryUsageKB: this.estimateMemoryUsage(values)
        }
      }
    };

    this.cache.set(cacheKey, result);
    return result;
  }

  private async calculateRSIGradient(
    data: CandleData[],
    symbol: string,
    timeframe: Timeframe,
    config: RSIGradientConfig
  ): Promise<IndicatorResult> {
    const startTime = Date.now();
    const cacheKey = `rsigradient_${symbol}_${timeframe}_${config.period}_${config.smoothing}`;
    
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    const values = this.calculator.calculateRSIGradient(data, config);
    const calculationTime = Date.now() - startTime;

    const result: IndicatorResult = {
      name: `RSI Gradient ${config.period}`,
      timeframe,
      values,
      metadata: {
        calculatedAt: Date.now(),
        dataPoints: values.length,
        config,
        performance: {
          calculationTimeMs: calculationTime,
          memoryUsageKB: this.estimateMemoryUsage(values)
        }
      }
    };

    this.cache.set(cacheKey, result);
    return result;
  }

  private async calculateMomentum(
    data: CandleData[],
    symbol: string,
    timeframe: Timeframe,
    period: number
  ): Promise<IndicatorResult> {
    const startTime = Date.now();
    const cacheKey = `momentum_${symbol}_${timeframe}_${period}`;
    
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    const values = this.calculator.calculateMomentum(data, period);
    const calculationTime = Date.now() - startTime;

    const result: IndicatorResult = {
      name: `Momentum ${period}`,
      timeframe,
      values,
      metadata: {
        calculatedAt: Date.now(),
        dataPoints: values.length,
        config: { period },
        performance: {
          calculationTimeMs: calculationTime,
          memoryUsageKB: this.estimateMemoryUsage(values)
        }
      }
    };

    this.cache.set(cacheKey, result);
    return result;
  }

  private async calculateVolumeProfile(
    data: CandleData[],
    symbol: string,
    timeframe: Timeframe,
    period: number
  ): Promise<IndicatorResult> {
    const startTime = Date.now();
    const cacheKey = `volume_${symbol}_${timeframe}_${period}`;
    
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    const values = this.calculator.calculateVolumeProfile(data, period);
    const calculationTime = Date.now() - startTime;

    const result: IndicatorResult = {
      name: `Volume Profile ${period}`,
      timeframe,
      values,
      metadata: {
        calculatedAt: Date.now(),
        dataPoints: values.length,
        config: { period },
        performance: {
          calculationTimeMs: calculationTime,
          memoryUsageKB: this.estimateMemoryUsage(values)
        }
      }
    };

    this.cache.set(cacheKey, result);
    return result;
  }

  private async calculateVolatility(
    data: CandleData[],
    symbol: string,
    timeframe: Timeframe,
    period: number
  ): Promise<IndicatorResult> {
    const startTime = Date.now();
    const cacheKey = `volatility_${symbol}_${timeframe}_${period}`;
    
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    const values = this.calculator.calculateVolatility(data, period);
    const calculationTime = Date.now() - startTime;

    const result: IndicatorResult = {
      name: `Volatility ${period}`,
      timeframe,
      values,
      metadata: {
        calculatedAt: Date.now(),
        dataPoints: values.length,
        config: { period },
        performance: {
          calculationTimeMs: calculationTime,
          memoryUsageKB: this.estimateMemoryUsage(values)
        }
      }
    };

    this.cache.set(cacheKey, result);
    return result;
  }

  private async calculateCustomIndicator(
    data: CandleData[],
    symbol: string,
    timeframe: Timeframe,
    definition: CustomIndicatorDefinition
  ): Promise<IndicatorResult> {
    const startTime = Date.now();
    const cacheKey = `custom_${symbol}_${timeframe}_${definition.name}`;
    
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    try {
      const values = await this.evaluateCustomFormula(data, definition);
      const calculationTime = Date.now() - startTime;

      const result: IndicatorResult = {
        name: definition.name,
        timeframe,
        values,
        metadata: {
          calculatedAt: Date.now(),
          dataPoints: values.length,
          config: definition,
          performance: {
            calculationTimeMs: calculationTime,
            memoryUsageKB: this.estimateMemoryUsage(values)
          }
        }
      };

      this.cache.set(cacheKey, result);
      return result;
    } catch (error) {
      throw new Error(`Custom indicator ${definition.name} failed: ${error.message}`);
    }
  }

  private async evaluateCustomFormula(
    data: CandleData[],
    definition: CustomIndicatorDefinition
  ): Promise<IndicatorValue[]> {
    const values: IndicatorValue[] = [];
    const dependencies = new Map<string, IndicatorValue[]>();
    
    for (const depName of definition.dependencies) {
      switch (depName) {
        case 'sma':
          dependencies.set('sma', this.calculator.calculateSMA(data, definition.parameters.period || 14));
          break;
        case 'ema':
          dependencies.set('ema', this.calculator.calculateEMA(data, definition.parameters.period || 14));
          break;
        case 'rsi':
          dependencies.set('rsi', this.calculator.calculateRSI(data, definition.parameters.period || 14));
          break;
        case 'atr':
          dependencies.set('atr', this.calculator.calculateATR(data, definition.parameters.period || 14));
          break;
      }
    }

    const maxLength = Math.min(data.length, Math.max(...Array.from(dependencies.values()).map(dep => dep.length)));
    
    for (let i = 0; i < maxLength; i++) {
      try {
        const context = {
          price: data[i].close,
          high: data[i].high,
          low: data[i].low,
          open: data[i].open,
          volume: data[i].volume,
          ...definition.parameters
        };

        for (const [name, depValues] of dependencies) {
          if (depValues[i]) {
            context[name] = depValues[i].value;
          }
        }

        const result = this.evaluateFormula(definition.formula, context);
        
        values.push({
          timestamp: data[i].timestamp,
          value: result
        });
      } catch (error) {
        values.push({
          timestamp: data[i].timestamp,
          value: 0
        });
      }
    }

    return values;
  }

  private evaluateFormula(formula: string, context: Record<string, number>): number {
    const sanitizedFormula = formula.replace(/[^a-zA-Z0-9+\-*/.()_\s]/g, '');
    
    try {
      const func = new Function(...Object.keys(context), `return ${sanitizedFormula};`);
      const result = func(...Object.values(context));
      return isFinite(result) ? result : 0;
    } catch (error) {
      return 0;
    }
  }

  private estimateMemoryUsage(values: IndicatorValue[]): number {
    return Math.round((values.length * (8 + 8 + 50)) / 1024); // Rough estimate in KB
  }

  getBatchStatus(batchId: string): ProcessingBatch | null {
    return this.activeBatches.get(batchId) || null;
  }

  clearCache(): void {
    this.cache.clear();
  }

  getCacheStats(): { size: number; keys: string[] } {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys())
    };
  }

  updateConfiguration(config: Partial<IndicatorProcessingConfig>): void {
    this.config = { ...this.config, ...config };
    this.clearCache(); // Clear cache when configuration changes
  }

  getConfiguration(): IndicatorProcessingConfig {
    return { ...this.config };
  }
}