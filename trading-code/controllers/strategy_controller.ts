/**
 * Strategy Controller
 * 
 * Orchestrates strategy execution, signal generation, and strategy lifecycle management
 * Coordinates between indicators, market data, and trade execution
 */

import { Strategy, StrategyEngine, TradingSignal, SignalType, StrategyStatus } from '../models/strategy';
import { IndicatorType, IndicatorValue, IndicatorManager } from '../models/indicator';
import { Position, PositionStatus } from '../models/position';
import { MarketDataFrame, Timeframe } from '../models/market_data';

export interface StrategyExecutionContext {
  strategy: Strategy;
  marketData: Map<Timeframe, MarketDataFrame>;
  indicators: IndicatorManager;
  currentPositions: Position[];
  portfolioValue: number;
  availableCash: number;
  timestamp: Date;
}

export interface StrategyPerformanceMetrics {
  totalSignals: number;
  executedSignals: number;
  successfulSignals: number;
  avgSignalConfidence: number;
  signalLatency: number;
  strategyAccuracy: number;
  riskAdjustedReturn: number;
}

export interface BacktestConfiguration {
  startDate: Date;
  endDate: Date;
  initialCapital: number;
  commission: number;
  slippage: number;
  benchmark?: string;
  warmupPeriod: number;
}

export interface BacktestResult {
  strategy: Strategy;
  config: BacktestConfiguration;
  performance: {
    totalReturn: number;
    annualizedReturn: number;
    volatility: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    profitFactor: number;
    totalTrades: number;
  };
  trades: Array<{
    entrySignal: TradingSignal;
    exitSignal?: TradingSignal;
    pnl: number;
    duration: number;
    success: boolean;
  }>;
  equityCurve: Array<{
    timestamp: Date;
    value: number;
    drawdown: number;
  }>;
  signals: TradingSignal[];
  metrics: StrategyPerformanceMetrics;
}

export class StrategyController {
  private strategies: Map<string, Strategy> = new Map();
  private engines: Map<string, StrategyEngine> = new Map();
  private executionHistory: Map<string, TradingSignal[]> = new Map();
  private performanceCache: Map<string, StrategyPerformanceMetrics> = new Map();

  constructor() {}

  /**
   * Register a new strategy for execution
   */
  async registerStrategy(strategy: Strategy): Promise<void> {
    // Validate strategy before registration
    const validation = this.validateStrategy(strategy);
    if (!validation.isValid) {
      throw new Error(`Strategy validation failed: ${validation.errors.join(', ')}`);
    }

    // Create strategy engine
    const engine = new StrategyEngine(strategy);
    
    // Register strategy and engine
    this.strategies.set(strategy.id, strategy);
    this.engines.set(strategy.id, engine);
    this.executionHistory.set(strategy.id, []);

    console.log(`Strategy ${strategy.metadata.name} registered successfully`);
  }

  /**
   * Update strategy configuration
   */
  async updateStrategy(strategyId: string, updates: Partial<Strategy>): Promise<boolean> {
    const strategy = this.strategies.get(strategyId);
    if (!strategy) {
      throw new Error(`Strategy ${strategyId} not found`);
    }

    const updatedStrategy = { ...strategy, ...updates };
    updatedStrategy.metadata.lastModified = new Date();

    // Validate updated strategy
    const validation = this.validateStrategy(updatedStrategy);
    if (!validation.isValid) {
      throw new Error(`Strategy update validation failed: ${validation.errors.join(', ')}`);
    }

    // Update strategy and recreate engine if needed
    this.strategies.set(strategyId, updatedStrategy);
    if (updates.rules || updates.riskParameters) {
      const newEngine = new StrategyEngine(updatedStrategy);
      this.engines.set(strategyId, newEngine);
    }

    return true;
  }

  /**
   * Execute strategy and generate signals
   */
  async executeStrategy(
    strategyId: string,
    context: StrategyExecutionContext
  ): Promise<TradingSignal[]> {
    const strategy = this.strategies.get(strategyId);
    const engine = this.engines.get(strategyId);

    if (!strategy || !engine) {
      throw new Error(`Strategy ${strategyId} not found or not properly initialized`);
    }

    if (strategy.status !== StrategyStatus.LIVE) {
      return []; // Don't execute inactive strategies
    }

    try {
      // Update engine with latest indicator values
      this.updateEngineIndicators(engine, context.indicators);

      // Check cooldown period
      if (!this.checkCooldownPeriod(strategy, context.timestamp)) {
        return [];
      }

      // Generate signals
      const signals = await this.generateSignalsForSymbols(
        engine,
        strategy,
        context
      );

      // Filter and validate signals
      const validSignals = this.filterValidSignals(signals, strategy, context);

      // Store execution history
      const history = this.executionHistory.get(strategyId) || [];
      history.push(...validSignals);
      this.executionHistory.set(strategyId, history);

      // Update strategy performance metrics
      await this.updatePerformanceMetrics(strategyId);

      return validSignals;
    } catch (error) {
      console.error(`Error executing strategy ${strategyId}:`, error);
      return [];
    }
  }

  /**
   * Run comprehensive backtest for strategy
   */
  async runBacktest(
    strategyId: string,
    config: BacktestConfiguration,
    marketData: Map<string, Map<Timeframe, MarketDataFrame>>
  ): Promise<BacktestResult> {
    const strategy = this.strategies.get(strategyId);
    if (!strategy) {
      throw new Error(`Strategy ${strategyId} not found`);
    }

    const results: BacktestResult = {
      strategy,
      config,
      performance: {
        totalReturn: 0,
        annualizedReturn: 0,
        volatility: 0,
        sharpeRatio: 0,
        maxDrawdown: 0,
        winRate: 0,
        profitFactor: 0,
        totalTrades: 0
      },
      trades: [],
      equityCurve: [],
      signals: [],
      metrics: {
        totalSignals: 0,
        executedSignals: 0,
        successfulSignals: 0,
        avgSignalConfidence: 0,
        signalLatency: 0,
        strategyAccuracy: 0,
        riskAdjustedReturn: 0
      }
    };

    // Initialize backtest state
    let currentCapital = config.initialCapital;
    let positions: Position[] = [];
    let currentDate = new Date(config.startDate);
    const endDate = new Date(config.endDate);

    // Create strategy engine for backtest
    const backtestEngine = new StrategyEngine(strategy);
    const indicatorManager = new IndicatorManager();

    try {
      // Run backtest simulation
      while (currentDate <= endDate) {
        // Get market data for current date
        const dayData = this.getMarketDataForDate(marketData, currentDate, strategy.symbols);
        
        if (dayData.size === 0) {
          currentDate = new Date(currentDate.getTime() + 24 * 60 * 60 * 1000);
          continue;
        }

        // Update indicators with current data
        await this.updateIndicatorsForBacktest(indicatorManager, dayData, currentDate);

        // Create execution context
        const context: StrategyExecutionContext = {
          strategy,
          marketData: dayData.get(strategy.symbols[0]) || new Map(),
          indicators: indicatorManager,
          currentPositions: positions,
          portfolioValue: currentCapital,
          availableCash: currentCapital,
          timestamp: currentDate
        };

        // Execute strategy
        const signals = await this.executeStrategy(strategy.id, context);
        results.signals.push(...signals);

        // Simulate trade execution
        const { newPositions, capitalChange } = await this.simulateTradeExecution(
          signals,
          positions,
          currentCapital,
          config,
          dayData
        );

        positions = newPositions;
        currentCapital += capitalChange;

        // Record equity curve point
        results.equityCurve.push({
          timestamp: new Date(currentDate),
          value: currentCapital,
          drawdown: this.calculateDrawdown(currentCapital, config.initialCapital, results.equityCurve)
        });

        // Move to next day
        currentDate = new Date(currentDate.getTime() + 24 * 60 * 60 * 1000);
      }

      // Calculate final performance metrics
      results.performance = this.calculateBacktestPerformance(results, config);
      results.trades = this.generateTradeHistory(results.signals, positions);
      results.metrics = this.calculateStrategyMetrics(results.signals, results.trades);

      return results;
    } catch (error) {
      console.error(`Backtest failed for strategy ${strategyId}:`, error);
      throw error;
    }
  }

  /**
   * Get strategy performance metrics
   */
  getPerformanceMetrics(strategyId: string): StrategyPerformanceMetrics | undefined {
    return this.performanceCache.get(strategyId);
  }

  /**
   * Get all registered strategies
   */
  getAllStrategies(): Strategy[] {
    return Array.from(this.strategies.values());
  }

  /**
   * Get strategy by ID
   */
  getStrategy(strategyId: string): Strategy | undefined {
    return this.strategies.get(strategyId);
  }

  /**
   * Get strategy execution history
   */
  getExecutionHistory(strategyId: string): TradingSignal[] {
    return this.executionHistory.get(strategyId) || [];
  }

  /**
   * Activate/deactivate strategy
   */
  setStrategyStatus(strategyId: string, status: StrategyStatus): boolean {
    const strategy = this.strategies.get(strategyId);
    if (!strategy) return false;

    strategy.status = status;
    strategy.metadata.lastModified = new Date();
    return true;
  }

  /**
   * Remove strategy from controller
   */
  removeStrategy(strategyId: string): boolean {
    const removed = this.strategies.delete(strategyId) &&
                   this.engines.delete(strategyId) &&
                   this.executionHistory.delete(strategyId) &&
                   this.performanceCache.delete(strategyId);
    
    return removed;
  }

  private validateStrategy(strategy: Strategy): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!strategy.metadata.name) {
      errors.push('Strategy name is required');
    }

    if (!strategy.symbols || strategy.symbols.length === 0) {
      errors.push('At least one symbol is required');
    }

    if (!strategy.timeframes || strategy.timeframes.length === 0) {
      errors.push('At least one timeframe is required');
    }

    if (!strategy.rules.entryLong.length && !strategy.rules.entryShort.length) {
      errors.push('At least one entry rule is required');
    }

    if (strategy.riskParameters.maxPositionSize <= 0 || strategy.riskParameters.maxPositionSize > 1) {
      errors.push('Max position size must be between 0 and 1');
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  private updateEngineIndicators(engine: StrategyEngine, indicators: IndicatorManager): void {
    const allIndicators = indicators.getAllIndicators();
    
    for (const indicatorSeries of allIndicators) {
      engine.updateIndicatorValues(indicatorSeries.indicator, indicatorSeries.values);
    }
  }

  private checkCooldownPeriod(strategy: Strategy, currentTime: Date): boolean {
    if (!strategy.lastSignal) return true;

    // Check each rule set for cooldown
    const allRuleSets = [
      ...strategy.rules.entryLong,
      ...strategy.rules.entryShort,
      ...strategy.rules.exitLong,
      ...strategy.rules.exitShort
    ];

    for (const ruleSet of allRuleSets) {
      if (ruleSet.cooldownPeriod) {
        const cooldownEnd = new Date(strategy.lastSignal.getTime() + ruleSet.cooldownPeriod * 60 * 1000);
        if (currentTime < cooldownEnd) {
          return false;
        }
      }
    }

    return true;
  }

  private async generateSignalsForSymbols(
    engine: StrategyEngine,
    strategy: Strategy,
    context: StrategyExecutionContext
  ): Promise<TradingSignal[]> {
    const signals: TradingSignal[] = [];

    for (const symbol of strategy.symbols) {
      // Get current price from market data
      const currentPrice = this.getCurrentPrice(symbol, context.marketData);
      if (!currentPrice) continue;

      // Generate signals for this symbol
      const symbolSignals = engine.generateSignals(symbol, currentPrice, context.timestamp);
      signals.push(...symbolSignals);
    }

    return signals;
  }

  private getCurrentPrice(symbol: string, marketData: Map<Timeframe, MarketDataFrame>): number | null {
    // Try to get price from the shortest timeframe first
    for (const timeframe of [Timeframe.FIVE_MINUTE, Timeframe.FIFTEEN_MINUTE, Timeframe.ONE_HOUR, Timeframe.ONE_DAY]) {
      const frame = marketData.get(timeframe);
      if (frame && frame.data.length > 0) {
        return frame.data[frame.data.length - 1].close;
      }
    }
    return null;
  }

  private filterValidSignals(
    signals: TradingSignal[],
    strategy: Strategy,
    context: StrategyExecutionContext
  ): TradingSignal[] {
    return signals.filter(signal => {
      // Check minimum confidence
      if (signal.confidence < 0.5) return false;

      // Check if we have enough capital
      if (signal.type === SignalType.ENTRY_LONG || signal.type === SignalType.ENTRY_SHORT) {
        const positionValue = signal.price * (signal.quantity || 0);
        if (positionValue > context.availableCash * 0.95) return false;
      }

      // Check maximum positions
      if (context.currentPositions.length >= strategy.maxConcurrentPositions) {
        if (signal.type === SignalType.ENTRY_LONG || signal.type === SignalType.ENTRY_SHORT) {
          return false;
        }
      }

      // Check risk limits
      if (signal.risk.portfolioRisk > strategy.riskParameters.maxDrawdown) {
        return false;
      }

      return true;
    });
  }

  private async updatePerformanceMetrics(strategyId: string): Promise<void> {
    const history = this.executionHistory.get(strategyId) || [];
    if (history.length === 0) return;

    const totalSignals = history.length;
    const executedSignals = history.filter(s => s.quantity && s.quantity > 0).length;
    const avgConfidence = history.reduce((sum, s) => sum + s.confidence, 0) / totalSignals;

    // Calculate other metrics based on historical performance
    const metrics: StrategyPerformanceMetrics = {
      totalSignals,
      executedSignals,
      successfulSignals: 0, // Would need actual trade outcomes
      avgSignalConfidence: avgConfidence,
      signalLatency: 0, // Would measure from signal generation to execution
      strategyAccuracy: 0, // Would need backtesting results
      riskAdjustedReturn: 0 // Would calculate from performance data
    };

    this.performanceCache.set(strategyId, metrics);
  }

  private getMarketDataForDate(
    marketData: Map<string, Map<Timeframe, MarketDataFrame>>,
    date: Date,
    symbols: string[]
  ): Map<string, Map<Timeframe, MarketDataFrame>> {
    const dayData = new Map<string, Map<Timeframe, MarketDataFrame>>();

    for (const symbol of symbols) {
      const symbolData = marketData.get(symbol);
      if (symbolData) {
        const filteredData = new Map<Timeframe, MarketDataFrame>();
        
        for (const [timeframe, frame] of symbolData) {
          const filteredFrame = {
            ...frame,
            data: frame.data.filter(bar => 
              bar.timestamp.toDateString() === date.toDateString()
            )
          };
          
          if (filteredFrame.data.length > 0) {
            filteredData.set(timeframe, filteredFrame);
          }
        }
        
        if (filteredData.size > 0) {
          dayData.set(symbol, filteredData);
        }
      }
    }

    return dayData;
  }

  private async updateIndicatorsForBacktest(
    indicatorManager: IndicatorManager,
    marketData: Map<string, Map<Timeframe, MarketDataFrame>>,
    currentDate: Date
  ): Promise<void> {
    // This would update indicators based on market data
    // Implementation would depend on specific indicator calculations
    // For now, we'll add placeholder indicator values
    
    for (const [symbol, timeframeData] of marketData) {
      for (const [timeframe, frame] of timeframeData) {
        if (frame.data.length > 0) {
          const latestPrice = frame.data[frame.data.length - 1].close;
          
          // Add mock indicator values
          const indicatorValue: IndicatorValue = {
            timestamp: currentDate,
            value: latestPrice,
            confidence: 0.8,
            signal: 'neutral'
          };
          
          // This is simplified - real implementation would calculate actual indicators
        }
      }
    }
  }

  private async simulateTradeExecution(
    signals: TradingSignal[],
    positions: Position[],
    currentCapital: number,
    config: BacktestConfiguration,
    marketData: Map<string, Map<Timeframe, MarketDataFrame>>
  ): Promise<{ newPositions: Position[]; capitalChange: number }> {
    // Simplified trade execution simulation
    // In practice, this would be much more sophisticated
    
    let capitalChange = 0;
    const newPositions = [...positions];

    for (const signal of signals) {
      if (signal.type === SignalType.ENTRY_LONG) {
        // Simulate buying
        const cost = signal.price * (signal.quantity || 0);
        const commission = cost * config.commission;
        capitalChange -= (cost + commission);
        
        // Add position (simplified)
        // Real implementation would create proper Position objects
      }
    }

    return { newPositions, capitalChange };
  }

  private calculateDrawdown(
    currentValue: number,
    initialCapital: number,
    equityCurve: Array<{ timestamp: Date; value: number; drawdown: number }>
  ): number {
    if (equityCurve.length === 0) return 0;
    
    const peakValue = Math.max(...equityCurve.map(point => point.value), initialCapital);
    return (peakValue - currentValue) / peakValue;
  }

  private calculateBacktestPerformance(
    results: BacktestResult,
    config: BacktestConfiguration
  ): BacktestResult['performance'] {
    const equityCurve = results.equityCurve;
    if (equityCurve.length === 0) {
      return {
        totalReturn: 0,
        annualizedReturn: 0,
        volatility: 0,
        sharpeRatio: 0,
        maxDrawdown: 0,
        winRate: 0,
        profitFactor: 0,
        totalTrades: 0
      };
    }

    const initialValue = config.initialCapital;
    const finalValue = equityCurve[equityCurve.length - 1].value;
    const totalReturn = (finalValue - initialValue) / initialValue;

    const days = (config.endDate.getTime() - config.startDate.getTime()) / (1000 * 60 * 60 * 24);
    const annualizedReturn = Math.pow(1 + totalReturn, 365 / days) - 1;

    const maxDrawdown = Math.max(...equityCurve.map(point => point.drawdown));

    // Calculate volatility from daily returns
    const dailyReturns: number[] = [];
    for (let i = 1; i < equityCurve.length; i++) {
      const dailyReturn = (equityCurve[i].value - equityCurve[i - 1].value) / equityCurve[i - 1].value;
      dailyReturns.push(dailyReturn);
    }

    const avgDailyReturn = dailyReturns.reduce((sum, r) => sum + r, 0) / dailyReturns.length;
    const variance = dailyReturns.reduce((sum, r) => sum + Math.pow(r - avgDailyReturn, 2), 0) / (dailyReturns.length - 1);
    const volatility = Math.sqrt(variance * 252); // Annualized

    const sharpeRatio = volatility > 0 ? annualizedReturn / volatility : 0;

    return {
      totalReturn,
      annualizedReturn,
      volatility,
      sharpeRatio,
      maxDrawdown,
      winRate: 0, // Would calculate from actual trades
      profitFactor: 0, // Would calculate from actual trades
      totalTrades: results.signals.length
    };
  }

  private generateTradeHistory(
    signals: TradingSignal[],
    positions: Position[]
  ): BacktestResult['trades'] {
    // Simplified trade history generation
    // In practice, this would match entry/exit signals and calculate actual trade outcomes
    return [];
  }

  private calculateStrategyMetrics(
    signals: TradingSignal[],
    trades: BacktestResult['trades']
  ): StrategyPerformanceMetrics {
    return {
      totalSignals: signals.length,
      executedSignals: signals.length, // Simplified
      successfulSignals: trades.filter(t => t.success).length,
      avgSignalConfidence: signals.reduce((sum, s) => sum + s.confidence, 0) / signals.length,
      signalLatency: 0, // Would measure actual latency
      strategyAccuracy: trades.length > 0 ? trades.filter(t => t.success).length / trades.length : 0,
      riskAdjustedReturn: 0 // Would calculate from performance data
    };
  }
}