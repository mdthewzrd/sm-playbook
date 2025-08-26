import { CandleData } from '../models/market_data';
import { TradingSignal, Strategy, StrategyConfig } from '../models/strategy';
import { Position, Order, OrderType } from '../models/position';
import { Portfolio, PortfolioMetrics } from '../models/portfolio';
import { Timeframe } from '../types/common';

export interface BacktestConfig {
  initialCapital: number;
  commission: number;
  slippage: number;
  maxPositions: number;
  riskPerTrade: number;
  startDate: Date;
  endDate: Date;
  benchmark?: string;
  timeframe: Timeframe;
}

export interface BacktestResult {
  id: string;
  strategy: string;
  symbol: string;
  config: BacktestConfig;
  performance: PerformanceMetrics;
  trades: TradeResult[];
  portfolio: PortfolioMetrics;
  drawdown: DrawdownAnalysis;
  riskMetrics: RiskMetrics;
  executionTime: number;
  status: 'completed' | 'failed';
  errors: string[];
}

export interface PerformanceMetrics {
  totalReturn: number;
  annualizedReturn: number;
  volatility: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  avgWin: number;
  avgLoss: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
}

export interface TradeResult {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  entryTime: number;
  exitTime: number;
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  pnl: number;
  pnlPercent: number;
  commission: number;
  duration: number;
  signal: TradingSignal;
  exitReason: 'signal' | 'stop_loss' | 'take_profit' | 'time_limit';
}

export interface DrawdownAnalysis {
  maxDrawdown: number;
  maxDrawdownDate: number;
  currentDrawdown: number;
  drawdownPeriods: DrawdownPeriod[];
  recoveryTime: number;
  underwaterCurve: number[];
}

export interface DrawdownPeriod {
  startDate: number;
  endDate: number;
  peak: number;
  trough: number;
  drawdown: number;
  duration: number;
  recovery: number;
}

export interface RiskMetrics {
  valueAtRisk95: number;
  valueAtRisk99: number;
  conditionalVaR95: number;
  conditionalVaR99: number;
  beta: number;
  alpha: number;
  informationRatio: number;
  calmarRatio: number;
  sterlingRatio: number;
}

export class BacktestProcessor {
  private activeBacktests: Map<string, BacktestResult>;

  constructor() {
    this.activeBacktests = new Map();
  }

  async runBacktest(
    strategy: Strategy,
    marketData: CandleData[],
    signals: TradingSignal[],
    config: BacktestConfig
  ): Promise<BacktestResult> {
    const backtestId = `${strategy.id}_${Date.now()}`;
    const startTime = Date.now();

    const result: BacktestResult = {
      id: backtestId,
      strategy: strategy.name,
      symbol: strategy.symbol,
      config,
      performance: {} as PerformanceMetrics,
      trades: [],
      portfolio: {} as PortfolioMetrics,
      drawdown: {} as DrawdownAnalysis,
      riskMetrics: {} as RiskMetrics,
      executionTime: 0,
      status: 'completed',
      errors: []
    };

    this.activeBacktests.set(backtestId, result);

    try {
      const portfolio = this.initializePortfolio(config);
      const positions: Map<string, Position> = new Map();
      const trades: TradeResult[] = [];
      const equityCurve: number[] = [];
      const returns: number[] = [];

      const filteredSignals = this.filterSignalsByDateRange(signals, config);
      const dataMap = this.createDataMap(marketData);

      for (const signal of filteredSignals) {
        const candle = dataMap.get(signal.timestamp);
        if (!candle) continue;

        const existingPosition = positions.get(signal.symbol);
        
        if (signal.type === 'buy' && !existingPosition) {
          const trade = await this.executeEntry(signal, candle, portfolio, config);
          if (trade) {
            positions.set(signal.symbol, trade.position);
            trades.push(trade.tradeResult);
          }
        } else if (signal.type === 'sell' && existingPosition) {
          const trade = await this.executeExit(signal, candle, existingPosition, portfolio, config);
          if (trade) {
            positions.delete(signal.symbol);
            trades.push(trade);
          }
        }

        const equity = this.calculateEquity(portfolio, positions, candle.close);
        equityCurve.push(equity);
        
        if (equityCurve.length > 1) {
          const ret = (equity - equityCurve[equityCurve.length - 2]) / equityCurve[equityCurve.length - 2];
          returns.push(ret);
        }
      }

      result.trades = trades;
      result.performance = this.calculatePerformanceMetrics(trades, equityCurve, returns, config);
      result.drawdown = this.calculateDrawdownAnalysis(equityCurve);
      result.riskMetrics = this.calculateRiskMetrics(returns, config);
      result.portfolio = portfolio.getMetrics();
      result.executionTime = Date.now() - startTime;

    } catch (error) {
      result.status = 'failed';
      result.errors.push(`Backtest failed: ${error.message}`);
      result.executionTime = Date.now() - startTime;
    }

    return result;
  }

  private initializePortfolio(config: BacktestConfig): Portfolio {
    return new Portfolio(
      'backtest',
      config.initialCapital,
      {
        maxPositions: config.maxPositions,
        maxRiskPerTrade: config.riskPerTrade,
        maxPortfolioRisk: 0.25,
        maxDrawdown: 0.2,
        maxLeverage: 1.0
      }
    );
  }

  private filterSignalsByDateRange(signals: TradingSignal[], config: BacktestConfig): TradingSignal[] {
    const startTime = config.startDate.getTime();
    const endTime = config.endDate.getTime();
    
    return signals.filter(signal => 
      signal.timestamp >= startTime && signal.timestamp <= endTime
    );
  }

  private createDataMap(marketData: CandleData[]): Map<number, CandleData> {
    const dataMap = new Map<number, CandleData>();
    marketData.forEach(candle => {
      dataMap.set(candle.timestamp, candle);
    });
    return dataMap;
  }

  private async executeEntry(
    signal: TradingSignal,
    candle: CandleData,
    portfolio: Portfolio,
    config: BacktestConfig
  ): Promise<{ position: Position; tradeResult: TradeResult } | null> {
    try {
      const price = this.applySlippage(candle.close, config.slippage, 'buy');
      const riskAmount = portfolio.getAvailableCapital() * config.riskPerTrade;
      const quantity = Math.floor(riskAmount / price);
      
      if (quantity === 0 || quantity * price > portfolio.getAvailableCapital()) {
        return null;
      }

      const order: Order = {
        id: `order_${Date.now()}`,
        symbol: signal.symbol,
        type: 'market' as OrderType,
        side: 'buy',
        quantity,
        price,
        timestamp: signal.timestamp,
        status: 'filled'
      };

      const position = new Position(
        `pos_${signal.symbol}_${Date.now()}`,
        signal.symbol,
        'long',
        quantity,
        price,
        signal.timestamp
      );

      position.addOrder(order);
      portfolio.addPosition(position);

      const commission = quantity * price * config.commission;
      const tradeResult: TradeResult = {
        id: `trade_${Date.now()}`,
        symbol: signal.symbol,
        side: 'buy',
        entryTime: signal.timestamp,
        exitTime: 0,
        entryPrice: price,
        exitPrice: 0,
        quantity,
        pnl: 0,
        pnlPercent: 0,
        commission,
        duration: 0,
        signal,
        exitReason: 'signal'
      };

      return { position, tradeResult };
    } catch (error) {
      return null;
    }
  }

  private async executeExit(
    signal: TradingSignal,
    candle: CandleData,
    position: Position,
    portfolio: Portfolio,
    config: BacktestConfig
  ): Promise<TradeResult | null> {
    try {
      const exitPrice = this.applySlippage(candle.close, config.slippage, 'sell');
      const exitOrder: Order = {
        id: `order_${Date.now()}`,
        symbol: signal.symbol,
        type: 'market' as OrderType,
        side: 'sell',
        quantity: position.getQuantity(),
        price: exitPrice,
        timestamp: signal.timestamp,
        status: 'filled'
      };

      position.addOrder(exitOrder);
      portfolio.closePosition(position.id);

      const entryPrice = position.getAveragePrice();
      const pnl = (exitPrice - entryPrice) * position.getQuantity();
      const pnlPercent = (exitPrice - entryPrice) / entryPrice;
      const commission = position.getQuantity() * exitPrice * config.commission;
      const duration = signal.timestamp - position.getOpenTime();

      const entryTrade = this.findEntryTrade(position.id);
      const tradeResult: TradeResult = {
        id: `trade_${Date.now()}`,
        symbol: signal.symbol,
        side: 'sell',
        entryTime: position.getOpenTime(),
        exitTime: signal.timestamp,
        entryPrice,
        exitPrice,
        quantity: position.getQuantity(),
        pnl: pnl - commission,
        pnlPercent,
        commission,
        duration,
        signal,
        exitReason: 'signal'
      };

      return tradeResult;
    } catch (error) {
      return null;
    }
  }

  private applySlippage(price: number, slippageBps: number, side: 'buy' | 'sell'): number {
    const slippageMultiplier = slippageBps / 10000;
    return side === 'buy' 
      ? price * (1 + slippageMultiplier)
      : price * (1 - slippageMultiplier);
  }

  private calculateEquity(portfolio: Portfolio, positions: Map<string, Position>, currentPrice: number): number {
    let equity = portfolio.getCash();
    
    for (const position of positions.values()) {
      const marketValue = position.getQuantity() * currentPrice;
      equity += marketValue;
    }
    
    return equity;
  }

  private findEntryTrade(positionId: string): TradeResult | null {
    for (const backtest of this.activeBacktests.values()) {
      const trade = backtest.trades.find(t => t.id === positionId && t.side === 'buy');
      if (trade) return trade;
    }
    return null;
  }

  private calculatePerformanceMetrics(
    trades: TradeResult[],
    equityCurve: number[],
    returns: number[],
    config: BacktestConfig
  ): PerformanceMetrics {
    if (trades.length === 0 || equityCurve.length === 0) {
      return this.getEmptyPerformanceMetrics();
    }

    const totalReturn = (equityCurve[equityCurve.length - 1] - config.initialCapital) / config.initialCapital;
    const winningTrades = trades.filter(t => t.pnl > 0);
    const losingTrades = trades.filter(t => t.pnl < 0);
    
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const volatility = Math.sqrt(returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length);
    
    const downside = returns.filter(ret => ret < 0);
    const downsideVolatility = downside.length > 0 
      ? Math.sqrt(downside.reduce((sum, ret) => sum + Math.pow(ret, 2), 0) / downside.length)
      : 0;

    return {
      totalReturn,
      annualizedReturn: Math.pow(1 + totalReturn, 365 / this.getTradingDays(config)) - 1,
      volatility: volatility * Math.sqrt(252),
      sharpeRatio: volatility > 0 ? (avgReturn * Math.sqrt(252)) / (volatility * Math.sqrt(252)) : 0,
      sortinoRatio: downsideVolatility > 0 ? (avgReturn * Math.sqrt(252)) / (downsideVolatility * Math.sqrt(252)) : 0,
      maxDrawdown: this.calculateMaxDrawdown(equityCurve),
      winRate: trades.length > 0 ? winningTrades.length / trades.length : 0,
      profitFactor: this.calculateProfitFactor(trades),
      avgWin: winningTrades.length > 0 ? winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length : 0,
      avgLoss: losingTrades.length > 0 ? losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length : 0,
      totalTrades: trades.length,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length
    };
  }

  private calculateDrawdownAnalysis(equityCurve: number[]): DrawdownAnalysis {
    if (equityCurve.length === 0) {
      return {
        maxDrawdown: 0,
        maxDrawdownDate: 0,
        currentDrawdown: 0,
        drawdownPeriods: [],
        recoveryTime: 0,
        underwaterCurve: []
      };
    }

    const peaks: number[] = [];
    const drawdowns: number[] = [];
    let runningMax = equityCurve[0];

    for (let i = 0; i < equityCurve.length; i++) {
      runningMax = Math.max(runningMax, equityCurve[i]);
      peaks.push(runningMax);
      drawdowns.push((equityCurve[i] - runningMax) / runningMax);
    }

    const maxDrawdown = Math.min(...drawdowns);
    const maxDrawdownIndex = drawdowns.indexOf(maxDrawdown);

    return {
      maxDrawdown: Math.abs(maxDrawdown),
      maxDrawdownDate: maxDrawdownIndex,
      currentDrawdown: Math.abs(drawdowns[drawdowns.length - 1]),
      drawdownPeriods: this.identifyDrawdownPeriods(drawdowns),
      recoveryTime: this.calculateRecoveryTime(drawdowns, maxDrawdownIndex),
      underwaterCurve: drawdowns
    };
  }

  private calculateRiskMetrics(returns: number[], config: BacktestConfig): RiskMetrics {
    if (returns.length === 0) {
      return {
        valueAtRisk95: 0,
        valueAtRisk99: 0,
        conditionalVaR95: 0,
        conditionalVaR99: 0,
        beta: 0,
        alpha: 0,
        informationRatio: 0,
        calmarRatio: 0,
        sterlingRatio: 0
      };
    }

    const sortedReturns = returns.slice().sort((a, b) => a - b);
    const var95Index = Math.floor(returns.length * 0.05);
    const var99Index = Math.floor(returns.length * 0.01);

    return {
      valueAtRisk95: Math.abs(sortedReturns[var95Index] || 0),
      valueAtRisk99: Math.abs(sortedReturns[var99Index] || 0),
      conditionalVaR95: Math.abs(sortedReturns.slice(0, var95Index).reduce((sum, ret) => sum + ret, 0) / var95Index || 0),
      conditionalVaR99: Math.abs(sortedReturns.slice(0, var99Index).reduce((sum, ret) => sum + ret, 0) / var99Index || 0),
      beta: 1.0, // Placeholder - would need benchmark data
      alpha: 0.0, // Placeholder - would need benchmark data
      informationRatio: 0.0, // Placeholder - would need benchmark data
      calmarRatio: 0.0, // Placeholder - would need annualized return and max drawdown
      sterlingRatio: 0.0 // Placeholder - would need average drawdown
    };
  }

  private getEmptyPerformanceMetrics(): PerformanceMetrics {
    return {
      totalReturn: 0,
      annualizedReturn: 0,
      volatility: 0,
      sharpeRatio: 0,
      sortinoRatio: 0,
      maxDrawdown: 0,
      winRate: 0,
      profitFactor: 0,
      avgWin: 0,
      avgLoss: 0,
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0
    };
  }

  private calculateMaxDrawdown(equityCurve: number[]): number {
    let maxDrawdown = 0;
    let peak = equityCurve[0];

    for (const equity of equityCurve) {
      peak = Math.max(peak, equity);
      const drawdown = (peak - equity) / peak;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }

    return maxDrawdown;
  }

  private calculateProfitFactor(trades: TradeResult[]): number {
    const grossProfit = trades.filter(t => t.pnl > 0).reduce((sum, t) => sum + t.pnl, 0);
    const grossLoss = Math.abs(trades.filter(t => t.pnl < 0).reduce((sum, t) => sum + t.pnl, 0));
    
    return grossLoss > 0 ? grossProfit / grossLoss : 0;
  }

  private identifyDrawdownPeriods(drawdowns: number[]): DrawdownPeriod[] {
    const periods: DrawdownPeriod[] = [];
    let inDrawdown = false;
    let startIndex = 0;
    let peak = 0;

    for (let i = 0; i < drawdowns.length; i++) {
      if (!inDrawdown && drawdowns[i] < 0) {
        inDrawdown = true;
        startIndex = i;
        peak = 0;
      } else if (inDrawdown && drawdowns[i] === 0) {
        const trough = Math.min(...drawdowns.slice(startIndex, i));
        periods.push({
          startDate: startIndex,
          endDate: i,
          peak,
          trough,
          drawdown: Math.abs(trough),
          duration: i - startIndex,
          recovery: i
        });
        inDrawdown = false;
      }
    }

    return periods;
  }

  private calculateRecoveryTime(drawdowns: number[], maxDrawdownIndex: number): number {
    for (let i = maxDrawdownIndex; i < drawdowns.length; i++) {
      if (drawdowns[i] === 0) {
        return i - maxDrawdownIndex;
      }
    }
    return drawdowns.length - maxDrawdownIndex;
  }

  private getTradingDays(config: BacktestConfig): number {
    const timeDiff = config.endDate.getTime() - config.startDate.getTime();
    return Math.ceil(timeDiff / (1000 * 60 * 60 * 24));
  }

  getBacktestResult(backtestId: string): BacktestResult | null {
    return this.activeBacktests.get(backtestId) || null;
  }

  getAllBacktestResults(): BacktestResult[] {
    return Array.from(this.activeBacktests.values());
  }

  clearBacktestHistory(): void {
    this.activeBacktests.clear();
  }
}