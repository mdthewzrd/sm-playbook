/**
 * Data Controller
 * 
 * Manages market data acquisition, processing, and distribution across timeframes
 * Coordinates data from multiple sources and handles data quality validation
 */

import { 
  MarketDataFrame, 
  MultiTimeframeData, 
  Timeframe, 
  OHLCV, 
  MarketDataValidator, 
  MarketDataUtils,
  Quote,
  Tick,
  MarketStatus
} from '../models/market_data';

export interface DataSource {
  id: string;
  name: string;
  priority: number; // Higher number = higher priority
  latency: number; // Average latency in ms
  reliability: number; // 0-1 reliability score
  supportedSymbols: string[];
  supportedTimeframes: Timeframe[];
  costPerRequest: number;
  rateLimits: {
    requestsPerMinute: number;
    dailyLimit: number;
  };
}

export interface DataRequest {
  symbols: string[];
  timeframes: Timeframe[];
  startDate?: Date;
  endDate?: Date;
  limit?: number;
  includeExtendedHours?: boolean;
  adjustForSplits?: boolean;
  adjustForDividends?: boolean;
  priority: 'low' | 'medium' | 'high' | 'urgent';
}

export interface DataSubscription {
  id: string;
  symbols: string[];
  timeframes: Timeframe[];
  callback: (data: MarketDataFrame) => void;
  isActive: boolean;
  lastUpdate: Date;
  errorCount: number;
  dataSource: string;
}

export interface DataQualityReport {
  symbol: string;
  timeframe: Timeframe;
  totalBars: number;
  validBars: number;
  gapsDetected: number;
  duplicatesRemoved: number;
  outliersDetected: number;
  qualityScore: number; // 0-100
  issues: string[];
  recommendations: string[];
}

export interface DataCacheConfig {
  enableCaching: boolean;
  maxCacheSize: number; // In MB
  defaultTTL: number; // Cache TTL in minutes
  compressionEnabled: boolean;
  persistToDisk: boolean;
}

export class DataController {
  private dataSources: Map<string, DataSource> = new Map();
  private subscriptions: Map<string, DataSubscription> = new Map();
  private dataCache: Map<string, { data: MarketDataFrame; expires: Date }> = new Map();
  private realtimeData: Map<string, MultiTimeframeData> = new Map();
  private cacheConfig: DataCacheConfig;
  private requestQueue: DataRequest[] = [];
  private processingQueue: boolean = false;

  constructor(cacheConfig: DataCacheConfig) {
    this.cacheConfig = cacheConfig;
    this.startDataProcessor();
  }

  /**
   * Register a data source
   */
  registerDataSource(source: DataSource): void {
    this.dataSources.set(source.id, source);
    console.log(`Data source ${source.name} registered`);
  }

  /**
   * Request historical market data
   */
  async getHistoricalData(request: DataRequest): Promise<Map<string, Map<Timeframe, MarketDataFrame>>> {
    const result = new Map<string, Map<Timeframe, MarketDataFrame>>();

    // Validate request
    this.validateDataRequest(request);

    // Add to request queue based on priority
    this.addToQueue(request);

    try {
      // Process request for each symbol and timeframe
      for (const symbol of request.symbols) {
        const symbolData = new Map<Timeframe, MarketDataFrame>();

        for (const timeframe of request.timeframes) {
          // Check cache first
          const cachedData = this.getCachedData(symbol, timeframe, request);
          if (cachedData) {
            symbolData.set(timeframe, cachedData);
            continue;
          }

          // Fetch from data source
          const dataFrame = await this.fetchFromBestSource(symbol, timeframe, request);
          
          // Validate and clean data
          const cleanedData = await this.processDataFrame(dataFrame);
          
          // Cache the data
          if (this.cacheConfig.enableCaching) {
            this.cacheData(symbol, timeframe, cleanedData);
          }

          symbolData.set(timeframe, cleanedData);
        }

        result.set(symbol, symbolData);
      }

      return result;
    } catch (error) {
      console.error('Error fetching historical data:', error);
      throw error;
    }
  }

  /**
   * Subscribe to real-time data updates
   */
  async subscribeRealtime(
    symbols: string[],
    timeframes: Timeframe[],
    callback: (data: MarketDataFrame) => void
  ): Promise<string> {
    const subscriptionId = `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const subscription: DataSubscription = {
      id: subscriptionId,
      symbols,
      timeframes,
      callback,
      isActive: true,
      lastUpdate: new Date(),
      errorCount: 0,
      dataSource: this.getBestDataSourceForRealtime(symbols, timeframes)
    };

    this.subscriptions.set(subscriptionId, subscription);

    // Initialize real-time data structures
    symbols.forEach(symbol => {
      if (!this.realtimeData.has(symbol)) {
        this.realtimeData.set(symbol, {
          symbol,
          frames: new Map(),
          lastUpdate: new Date(),
          isComplete: false
        });
      }
    });

    // Start real-time data feed (would integrate with actual data providers)
    await this.startRealtimeFeed(subscription);

    return subscriptionId;
  }

  /**
   * Unsubscribe from real-time data
   */
  async unsubscribe(subscriptionId: string): Promise<boolean> {
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription) return false;

    subscription.isActive = false;
    this.subscriptions.delete(subscriptionId);

    // Stop feed if no more subscriptions for this data
    await this.stopRealtimeFeedIfNeeded(subscription);

    return true;
  }

  /**
   * Get current market data for symbols
   */
  getCurrentData(symbols: string[], timeframes?: Timeframe[]): Map<string, MultiTimeframeData> {
    const result = new Map<string, MultiTimeframeData>();

    for (const symbol of symbols) {
      const symbolData = this.realtimeData.get(symbol);
      if (symbolData) {
        if (timeframes) {
          // Filter by requested timeframes
          const filteredFrames = new Map<Timeframe, MarketDataFrame>();
          timeframes.forEach(tf => {
            const frame = symbolData.frames.get(tf);
            if (frame) {
              filteredFrames.set(tf, frame);
            }
          });
          
          result.set(symbol, {
            ...symbolData,
            frames: filteredFrames
          });
        } else {
          result.set(symbol, symbolData);
        }
      }
    }

    return result;
  }

  /**
   * Get latest quotes for symbols
   */
  async getQuotes(symbols: string[]): Promise<Map<string, Quote>> {
    const quotes = new Map<string, Quote>();

    for (const symbol of symbols) {
      try {
        const quote = await this.fetchQuote(symbol);
        if (quote) {
          quotes.set(symbol, quote);
        }
      } catch (error) {
        console.error(`Error fetching quote for ${symbol}:`, error);
      }
    }

    return quotes;
  }

  /**
   * Update market data with new tick
   */
  async updateWithTick(tick: Tick): Promise<void> {
    // This would be called by real-time data feeds
    // Process tick into OHLCV bars for different timeframes
    
    const symbol = this.extractSymbolFromTick(tick);
    let symbolData = this.realtimeData.get(symbol);
    
    if (!symbolData) {
      symbolData = {
        symbol,
        frames: new Map(),
        lastUpdate: new Date(),
        isComplete: false
      };
      this.realtimeData.set(symbol, symbolData);
    }

    // Update each timeframe
    for (const timeframe of Object.values(Timeframe)) {
      await this.updateTimeframeWithTick(symbolData, timeframe, tick);
    }

    symbolData.lastUpdate = tick.timestamp;

    // Notify subscribers
    this.notifySubscribers(symbol, symbolData);
  }

  /**
   * Validate data quality
   */
  async validateDataQuality(
    symbol: string,
    timeframe: Timeframe,
    data: OHLCV[]
  ): Promise<DataQualityReport> {
    const report: DataQualityReport = {
      symbol,
      timeframe,
      totalBars: data.length,
      validBars: 0,
      gapsDetected: 0,
      duplicatesRemoved: 0,
      outliersDetected: 0,
      qualityScore: 0,
      issues: [],
      recommendations: []
    };

    if (data.length === 0) {
      report.qualityScore = 0;
      report.issues.push('No data available');
      return report;
    }

    // Validate individual bars
    let validCount = 0;
    for (const bar of data) {
      if (MarketDataValidator.validateOHLCV(bar)) {
        validCount++;
      } else {
        report.issues.push(`Invalid OHLCV data at ${bar.timestamp}`);
      }
    }
    report.validBars = validCount;

    // Detect gaps
    const expectedInterval = this.getTimeframeMilliseconds(timeframe);
    let gapCount = 0;
    
    for (let i = 1; i < data.length; i++) {
      const timeDiff = data[i].timestamp.getTime() - data[i - 1].timestamp.getTime();
      if (timeDiff > expectedInterval * 1.5) {
        gapCount++;
      }
    }
    report.gapsDetected = gapCount;

    // Detect outliers
    let outlierCount = 0;
    const prices = data.map(d => d.close);
    const median = this.calculateMedian(prices);
    const mad = this.calculateMAD(prices, median);
    
    for (const price of prices) {
      if (Math.abs(price - median) > mad * 3) {
        outlierCount++;
      }
    }
    report.outliersDetected = outlierCount;

    // Calculate quality score
    const validityScore = (report.validBars / report.totalBars) * 40;
    const gapScore = Math.max(0, 30 - (report.gapsDetected / report.totalBars) * 100);
    const outlierScore = Math.max(0, 30 - (report.outliersDetected / report.totalBars) * 100);
    
    report.qualityScore = validityScore + gapScore + outlierScore;

    // Generate recommendations
    if (report.qualityScore < 70) {
      report.recommendations.push('Consider using alternative data source');
    }
    if (report.gapsDetected > report.totalBars * 0.05) {
      report.recommendations.push('Fill gaps with interpolated data');
    }
    if (report.outliersDetected > 0) {
      report.recommendations.push('Review and potentially filter outliers');
    }

    return report;
  }

  /**
   * Convert data between timeframes
   */
  async convertTimeframe(
    sourceData: MarketDataFrame,
    targetTimeframe: Timeframe
  ): Promise<MarketDataFrame> {
    try {
      return MarketDataUtils.convertTimeframe(sourceData, targetTimeframe);
    } catch (error) {
      console.error('Error converting timeframe:', error);
      throw error;
    }
  }

  /**
   * Get market status for symbols
   */
  async getMarketStatus(symbols: string[]): Promise<Map<string, MarketStatus>> {
    const statuses = new Map<string, MarketStatus>();

    // This would integrate with actual market data providers
    // For now, return mock status
    for (const symbol of symbols) {
      statuses.set(symbol, {
        exchange: 'NYSE', // Would determine actual exchange
        status: 'open',
        session: {
          name: 'Regular Trading',
          timezone: 'America/New_York',
          openTime: '09:30',
          closeTime: '16:00',
          isActive: true
        },
        nextSessionChange: new Date(),
        timestamp: new Date()
      });
    }

    return statuses;
  }

  /**
   * Clear data cache
   */
  clearCache(symbol?: string, timeframe?: Timeframe): void {
    if (symbol && timeframe) {
      const cacheKey = `${symbol}_${timeframe}`;
      this.dataCache.delete(cacheKey);
    } else if (symbol) {
      // Clear all timeframes for symbol
      for (const key of this.dataCache.keys()) {
        if (key.startsWith(`${symbol}_`)) {
          this.dataCache.delete(key);
        }
      }
    } else {
      // Clear entire cache
      this.dataCache.clear();
    }
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): {
    size: number;
    hitRate: number;
    memoryUsage: number;
    entries: number;
  } {
    // Calculate cache memory usage (simplified)
    let memoryUsage = 0;
    for (const [_, entry] of this.dataCache) {
      memoryUsage += entry.data.data.length * 100; // Rough estimate
    }

    return {
      size: this.dataCache.size,
      hitRate: 0.75, // Would track actual hit rate
      memoryUsage,
      entries: this.dataCache.size
    };
  }

  private validateDataRequest(request: DataRequest): void {
    if (!request.symbols || request.symbols.length === 0) {
      throw new Error('At least one symbol is required');
    }

    if (!request.timeframes || request.timeframes.length === 0) {
      throw new Error('At least one timeframe is required');
    }

    if (request.startDate && request.endDate && request.startDate >= request.endDate) {
      throw new Error('Start date must be before end date');
    }
  }

  private addToQueue(request: DataRequest): void {
    // Insert based on priority
    const priorityOrder = { urgent: 0, high: 1, medium: 2, low: 3 };
    const requestPriority = priorityOrder[request.priority];
    
    let insertIndex = this.requestQueue.length;
    for (let i = 0; i < this.requestQueue.length; i++) {
      if (priorityOrder[this.requestQueue[i].priority] > requestPriority) {
        insertIndex = i;
        break;
      }
    }
    
    this.requestQueue.splice(insertIndex, 0, request);
  }

  private getCachedData(symbol: string, timeframe: Timeframe, request: DataRequest): MarketDataFrame | null {
    if (!this.cacheConfig.enableCaching) return null;

    const cacheKey = `${symbol}_${timeframe}`;
    const cached = this.dataCache.get(cacheKey);
    
    if (!cached || cached.expires < new Date()) {
      return null;
    }

    // Check if cached data covers the requested range
    if (request.startDate && request.endDate) {
      const dataStart = cached.data.startTime;
      const dataEnd = cached.data.endTime;
      
      if (dataStart > request.startDate || dataEnd < request.endDate) {
        return null; // Cached data doesn't cover full range
      }
    }

    return cached.data;
  }

  private cacheData(symbol: string, timeframe: Timeframe, data: MarketDataFrame): void {
    if (!this.cacheConfig.enableCaching) return;

    const cacheKey = `${symbol}_${timeframe}`;
    const expires = new Date(Date.now() + this.cacheConfig.defaultTTL * 60 * 1000);
    
    this.dataCache.set(cacheKey, { data, expires });

    // Clean up expired entries
    this.cleanExpiredCache();
  }

  private cleanExpiredCache(): void {
    const now = new Date();
    for (const [key, entry] of this.dataCache) {
      if (entry.expires < now) {
        this.dataCache.delete(key);
      }
    }
  }

  private async fetchFromBestSource(
    symbol: string,
    timeframe: Timeframe,
    request: DataRequest
  ): Promise<MarketDataFrame> {
    // Select best data source based on priority, reliability, and symbol support
    const availableSources = Array.from(this.dataSources.values())
      .filter(source => 
        source.supportedSymbols.includes(symbol) &&
        source.supportedTimeframes.includes(timeframe)
      )
      .sort((a, b) => b.priority * b.reliability - a.priority * a.reliability);

    if (availableSources.length === 0) {
      throw new Error(`No data source available for ${symbol} ${timeframe}`);
    }

    // Try sources in order until successful
    for (const source of availableSources) {
      try {
        return await this.fetchFromSource(source, symbol, timeframe, request);
      } catch (error) {
        console.warn(`Failed to fetch from ${source.name}:`, error);
        continue;
      }
    }

    throw new Error(`All data sources failed for ${symbol} ${timeframe}`);
  }

  private async fetchFromSource(
    source: DataSource,
    symbol: string,
    timeframe: Timeframe,
    request: DataRequest
  ): Promise<MarketDataFrame> {
    // This would integrate with actual data provider APIs
    // For now, return mock data
    
    const mockData: OHLCV[] = [];
    const basePrice = 100;
    const startTime = request.startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
    const endTime = request.endDate || new Date();
    
    let currentTime = new Date(startTime);
    const interval = this.getTimeframeMilliseconds(timeframe);
    
    while (currentTime <= endTime) {
      const variation = (Math.random() - 0.5) * 4;
      const open = basePrice + variation;
      const close = open + (Math.random() - 0.5) * 2;
      const high = Math.max(open, close) + Math.random();
      const low = Math.min(open, close) - Math.random();
      
      mockData.push({
        timestamp: new Date(currentTime),
        open,
        high,
        low,
        close,
        volume: Math.floor(Math.random() * 100000) + 10000
      });
      
      currentTime = new Date(currentTime.getTime() + interval);
    }

    return {
      symbol,
      timeframe,
      data: mockData,
      startTime: mockData[0]?.timestamp || startTime,
      endTime: mockData[mockData.length - 1]?.timestamp || endTime,
      count: mockData.length,
      metadata: {
        source: source.id,
        quality: 'high',
        gaps: 0,
        adjustedForSplits: request.adjustForSplits || false,
        adjustedForDividends: request.adjustForDividends || false
      }
    };
  }

  private async processDataFrame(dataFrame: MarketDataFrame): Promise<MarketDataFrame> {
    // Validate and clean the data
    const validation = MarketDataValidator.validateDataFrame(dataFrame);
    
    if (!validation.isValid) {
      console.warn(`Data quality issues for ${dataFrame.symbol}:`, validation.errors);
    }

    // Remove duplicates
    const uniqueData = this.removeDuplicates(dataFrame.data);
    
    // Sort by timestamp
    uniqueData.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    return {
      ...dataFrame,
      data: uniqueData,
      count: uniqueData.length,
      startTime: uniqueData[0]?.timestamp || dataFrame.startTime,
      endTime: uniqueData[uniqueData.length - 1]?.timestamp || dataFrame.endTime
    };
  }

  private removeDuplicates(data: OHLCV[]): OHLCV[] {
    const seen = new Set<number>();
    return data.filter(bar => {
      const timestamp = bar.timestamp.getTime();
      if (seen.has(timestamp)) {
        return false;
      }
      seen.add(timestamp);
      return true;
    });
  }

  private getBestDataSourceForRealtime(symbols: string[], timeframes: Timeframe[]): string {
    // Select data source with lowest latency for real-time data
    const availableSources = Array.from(this.dataSources.values())
      .filter(source => 
        symbols.some(symbol => source.supportedSymbols.includes(symbol)) &&
        timeframes.some(tf => source.supportedTimeframes.includes(tf))
      )
      .sort((a, b) => a.latency - b.latency);

    return availableSources[0]?.id || 'default';
  }

  private async startRealtimeFeed(subscription: DataSubscription): Promise<void> {
    // This would start actual real-time data feed
    console.log(`Starting real-time feed for ${subscription.symbols.join(', ')}`);
  }

  private async stopRealtimeFeedIfNeeded(subscription: DataSubscription): Promise<void> {
    // Check if any other subscriptions need this data
    const stillNeeded = Array.from(this.subscriptions.values()).some(sub =>
      sub.isActive && 
      sub.dataSource === subscription.dataSource &&
      sub.symbols.some(symbol => subscription.symbols.includes(symbol))
    );

    if (!stillNeeded) {
      console.log(`Stopping real-time feed for ${subscription.dataSource}`);
    }
  }

  private async fetchQuote(symbol: string): Promise<Quote | null> {
    // This would fetch current quote from data provider
    // Return mock quote for now
    return {
      timestamp: new Date(),
      symbol,
      bid: 99.95,
      ask: 100.05,
      bidSize: 100,
      askSize: 200,
      lastPrice: 100.00,
      lastSize: 150,
      spread: 0.10,
      midPrice: 100.00
    };
  }

  private extractSymbolFromTick(tick: Tick): string {
    // Extract symbol from tick data - this would depend on tick format
    return 'MOCK_SYMBOL';
  }

  private async updateTimeframeWithTick(
    symbolData: MultiTimeframeData,
    timeframe: Timeframe,
    tick: Tick
  ): Promise<void> {
    // Update OHLCV bar for the timeframe with new tick
    // This is complex logic that aggregates ticks into bars
    
    let frame = symbolData.frames.get(timeframe);
    if (!frame) {
      frame = {
        symbol: symbolData.symbol,
        timeframe,
        data: [],
        startTime: tick.timestamp,
        endTime: tick.timestamp,
        count: 0,
        metadata: {
          source: 'realtime',
          quality: 'high',
          gaps: 0,
          adjustedForSplits: false,
          adjustedForDividends: false
        }
      };
      symbolData.frames.set(timeframe, frame);
    }

    // Logic to aggregate tick into appropriate time bar
    // This is simplified - real implementation would be more complex
    const barIndex = this.getBarIndexForTick(frame, tick, timeframe);
    // Update or create bar at barIndex...
  }

  private getBarIndexForTick(frame: MarketDataFrame, tick: Tick, timeframe: Timeframe): number {
    // Calculate which bar this tick belongs to
    const interval = this.getTimeframeMilliseconds(timeframe);
    // Complex logic to determine bar alignment
    return frame.data.length - 1; // Simplified
  }

  private notifySubscribers(symbol: string, symbolData: MultiTimeframeData): void {
    const relevantSubscriptions = Array.from(this.subscriptions.values())
      .filter(sub => sub.isActive && sub.symbols.includes(symbol));

    for (const subscription of relevantSubscriptions) {
      for (const timeframe of subscription.timeframes) {
        const frame = symbolData.frames.get(timeframe);
        if (frame) {
          try {
            subscription.callback(frame);
            subscription.lastUpdate = new Date();
          } catch (error) {
            subscription.errorCount++;
            console.error(`Error in subscription callback for ${subscription.id}:`, error);
          }
        }
      }
    }
  }

  private getTimeframeMilliseconds(timeframe: Timeframe): number {
    switch (timeframe) {
      case Timeframe.FIVE_MINUTE: return 5 * 60 * 1000;
      case Timeframe.FIFTEEN_MINUTE: return 15 * 60 * 1000;
      case Timeframe.ONE_HOUR: return 60 * 60 * 1000;
      case Timeframe.ONE_DAY: return 24 * 60 * 60 * 1000;
      case Timeframe.ONE_WEEK: return 7 * 24 * 60 * 60 * 1000;
      default: return 60 * 1000;
    }
  }

  private calculateMedian(numbers: number[]): number {
    const sorted = [...numbers].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 
      ? (sorted[middle - 1] + sorted[middle]) / 2
      : sorted[middle];
  }

  private calculateMAD(numbers: number[], median: number): number {
    const deviations = numbers.map(num => Math.abs(num - median));
    return this.calculateMedian(deviations);
  }

  private startDataProcessor(): void {
    // Process queued requests
    setInterval(async () => {
      if (!this.processingQueue && this.requestQueue.length > 0) {
        this.processingQueue = true;
        const request = this.requestQueue.shift();
        if (request) {
          try {
            await this.getHistoricalData(request);
          } catch (error) {
            console.error('Error processing queued request:', error);
          }
        }
        this.processingQueue = false;
      }
    }, 100);
  }
}