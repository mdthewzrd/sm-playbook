import { MarketData, CandleData, DataQuality } from '../models/market_data';
import { Timeframe } from '../types/common';

export interface DataProcessingOptions {
  normalize: boolean;
  fillGaps: boolean;
  removeOutliers: boolean;
  validateQuality: boolean;
  timezone?: string;
}

export interface DataTransformResult {
  data: CandleData[];
  quality: DataQuality;
  transformations: string[];
  errors: string[];
}

export class DataProcessor {
  private static readonly DEFAULT_OPTIONS: DataProcessingOptions = {
    normalize: true,
    fillGaps: true,
    removeOutliers: true,
    validateQuality: true,
    timezone: 'America/New_York'
  };

  constructor(private options: DataProcessingOptions = DataProcessor.DEFAULT_OPTIONS) {}

  async processMarketData(
    rawData: CandleData[],
    symbol: string,
    timeframe: Timeframe
  ): Promise<DataTransformResult> {
    const transformations: string[] = [];
    const errors: string[] = [];
    let processedData = [...rawData];

    try {
      if (this.options.validateQuality) {
        const validation = this.validateDataQuality(processedData, symbol, timeframe);
        if (!validation.isValid) {
          errors.push(...validation.issues);
        }
      }

      if (this.options.fillGaps) {
        const filled = this.fillDataGaps(processedData, timeframe);
        processedData = filled.data;
        transformations.push(`Filled ${filled.gapsFilled} gaps`);
      }

      if (this.options.removeOutliers) {
        const cleaned = this.removeOutliers(processedData);
        processedData = cleaned.data;
        transformations.push(`Removed ${cleaned.outliersRemoved} outliers`);
      }

      if (this.options.normalize) {
        processedData = this.normalizeData(processedData);
        transformations.push('Data normalized');
      }

      const quality = this.calculateDataQuality(processedData);

      return {
        data: processedData,
        quality,
        transformations,
        errors
      };
    } catch (error) {
      errors.push(`Processing failed: ${error.message}`);
      return {
        data: rawData,
        quality: { score: 0, issues: ['Processing failed'] },
        transformations,
        errors
      };
    }
  }

  async aggregateTimeframes(
    data: CandleData[],
    fromTimeframe: Timeframe,
    toTimeframe: Timeframe
  ): Promise<CandleData[]> {
    if (fromTimeframe === toTimeframe) return data;

    const multiplier = this.getTimeframeMultiplier(fromTimeframe, toTimeframe);
    if (multiplier < 1) {
      throw new Error(`Cannot aggregate from ${fromTimeframe} to ${toTimeframe}: invalid direction`);
    }

    const aggregated: CandleData[] = [];
    const sortedData = data.sort((a, b) => a.timestamp - b.timestamp);

    for (let i = 0; i < sortedData.length; i += multiplier) {
      const chunk = sortedData.slice(i, i + multiplier);
      if (chunk.length === 0) continue;

      const aggregatedCandle = this.aggregateCandles(chunk);
      aggregated.push(aggregatedCandle);
    }

    return aggregated;
  }

  private validateDataQuality(data: CandleData[], symbol: string, timeframe: Timeframe): { isValid: boolean; issues: string[] } {
    const issues: string[] = [];

    if (data.length === 0) {
      issues.push('Empty dataset');
      return { isValid: false, issues };
    }

    const sorted = data.every((candle, index) => 
      index === 0 || candle.timestamp >= data[index - 1].timestamp
    );
    if (!sorted) {
      issues.push('Data not in chronological order');
    }

    const invalidPrices = data.filter(candle => 
      candle.open <= 0 || candle.high <= 0 || candle.low <= 0 || candle.close <= 0 ||
      candle.high < Math.max(candle.open, candle.close) ||
      candle.low > Math.min(candle.open, candle.close)
    );
    if (invalidPrices.length > 0) {
      issues.push(`${invalidPrices.length} candles with invalid OHLC relationships`);
    }

    const zeroVolume = data.filter(candle => candle.volume === 0).length;
    if (zeroVolume > data.length * 0.1) {
      issues.push(`High zero volume ratio: ${(zeroVolume / data.length * 100).toFixed(1)}%`);
    }

    return { isValid: issues.length === 0, issues };
  }

  private fillDataGaps(data: CandleData[], timeframe: Timeframe): { data: CandleData[]; gapsFilled: number } {
    if (data.length < 2) return { data, gapsFilled: 0 };

    const timeframeMs = this.getTimeframeMs(timeframe);
    const filled: CandleData[] = [data[0]];
    let gapsFilled = 0;

    for (let i = 1; i < data.length; i++) {
      const prev = data[i - 1];
      const curr = data[i];
      const expectedTime = prev.timestamp + timeframeMs;
      
      if (curr.timestamp > expectedTime + timeframeMs / 2) {
        const gapCandle: CandleData = {
          timestamp: expectedTime,
          open: prev.close,
          high: prev.close,
          low: prev.close,
          close: prev.close,
          volume: 0
        };
        filled.push(gapCandle);
        gapsFilled++;
      }
      
      filled.push(curr);
    }

    return { data: filled, gapsFilled };
  }

  private removeOutliers(data: CandleData[]): { data: CandleData[]; outliersRemoved: number } {
    if (data.length < 10) return { data, outliersRemoved: 0 };

    const prices = data.map(candle => (candle.high + candle.low) / 2);
    const q1 = this.quantile(prices, 0.25);
    const q3 = this.quantile(prices, 0.75);
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;

    const filtered = data.filter(candle => {
      const midPrice = (candle.high + candle.low) / 2;
      return midPrice >= lowerBound && midPrice <= upperBound;
    });

    return {
      data: filtered,
      outliersRemoved: data.length - filtered.length
    };
  }

  private normalizeData(data: CandleData[]): CandleData[] {
    if (data.length === 0) return data;

    const maxVolume = Math.max(...data.map(c => c.volume));
    if (maxVolume === 0) return data;

    return data.map(candle => ({
      ...candle,
      volume: candle.volume / maxVolume
    }));
  }

  private calculateDataQuality(data: CandleData[]): DataQuality {
    if (data.length === 0) {
      return { score: 0, issues: ['Empty dataset'] };
    }

    let score = 100;
    const issues: string[] = [];

    const zeroVolumeRatio = data.filter(c => c.volume === 0).length / data.length;
    if (zeroVolumeRatio > 0.1) {
      score -= zeroVolumeRatio * 30;
      issues.push(`High zero volume ratio: ${(zeroVolumeRatio * 100).toFixed(1)}%`);
    }

    const gapCount = this.countGaps(data);
    if (gapCount > 0) {
      const gapRatio = gapCount / data.length;
      score -= gapRatio * 20;
      issues.push(`${gapCount} gaps detected`);
    }

    const volatility = this.calculateVolatility(data);
    if (volatility > 0.1) {
      score -= 10;
      issues.push('High volatility detected');
    }

    return {
      score: Math.max(0, Math.min(100, score)),
      issues
    };
  }

  private countGaps(data: CandleData[]): number {
    if (data.length < 2) return 0;

    let gaps = 0;
    for (let i = 1; i < data.length; i++) {
      const timeDiff = data[i].timestamp - data[i - 1].timestamp;
      if (timeDiff > 300000) { // 5 minutes
        gaps++;
      }
    }
    return gaps;
  }

  private calculateVolatility(data: CandleData[]): number {
    if (data.length < 2) return 0;

    const returns = [];
    for (let i = 1; i < data.length; i++) {
      const ret = (data[i].close - data[i - 1].close) / data[i - 1].close;
      returns.push(ret);
    }

    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance);
  }

  private aggregateCandles(candles: CandleData[]): CandleData {
    if (candles.length === 0) {
      throw new Error('Cannot aggregate empty candle array');
    }

    if (candles.length === 1) {
      return candles[0];
    }

    return {
      timestamp: candles[0].timestamp,
      open: candles[0].open,
      high: Math.max(...candles.map(c => c.high)),
      low: Math.min(...candles.map(c => c.low)),
      close: candles[candles.length - 1].close,
      volume: candles.reduce((sum, c) => sum + c.volume, 0)
    };
  }

  private getTimeframeMultiplier(from: Timeframe, to: Timeframe): number {
    const timeframes = { '5m': 5, '15m': 15, '1h': 60, '1d': 1440 };
    return timeframes[to] / timeframes[from];
  }

  private getTimeframeMs(timeframe: Timeframe): number {
    const timeframes = { '5m': 300000, '15m': 900000, '1h': 3600000, '1d': 86400000 };
    return timeframes[timeframe];
  }

  private quantile(arr: number[], q: number): number {
    const sorted = arr.slice().sort((a, b) => a - b);
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    
    if (sorted[base + 1] !== undefined) {
      return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
    } else {
      return sorted[base];
    }
  }
}