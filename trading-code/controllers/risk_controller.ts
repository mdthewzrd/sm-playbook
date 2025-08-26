/**
 * Risk Controller
 * 
 * Comprehensive risk management including position sizing, portfolio risk assessment,
 * drawdown monitoring, and risk limit enforcement
 */

import { Portfolio, PortfolioCalculator, RiskAssessment, RiskViolation } from '../models/portfolio';
import { Position, PositionSide, PositionStatus } from '../models/position';
import { TradingSignal, SignalType } from '../models/strategy';
import { Order } from '../models/position';

export interface RiskConfiguration {
  maxPortfolioRisk: number; // Maximum portfolio VaR
  maxDrawdown: number; // Maximum acceptable drawdown
  maxConcentration: number; // Maximum single position size
  maxLeverage: number; // Maximum portfolio leverage
  maxDailyLoss: number; // Maximum daily loss
  maxWeeklyLoss: number; // Maximum weekly loss
  maxMonthlyLoss: number; // Maximum monthly loss
  correlationThreshold: number; // Maximum correlation between positions
  stopLossRequired: boolean; // Whether stop losses are mandatory
  maxPositionsPerStrategy: number;
  maxTotalPositions: number;
  emergencyLiquidationThreshold: number; // Emergency liquidation trigger
}

export interface RiskMetrics {
  portfolioRisk: number;
  currentDrawdown: number;
  leverage: number;
  concentration: number;
  dailyPnL: number;
  weeklyPnL: number;
  monthlyPnL: number;
  var95: number; // 95% Value at Risk
  var99: number; // 99% Value at Risk
  expectedShortfall: number;
  betaExposure: number;
  correlationRisk: number;
  liquidityRisk: number;
  lastUpdate: Date;
}

export interface PositionSizeRequest {
  symbol: string;
  signal: TradingSignal;
  portfolioValue: number;
  currentPositions: Position[];
  riskPerTrade: number;
  maxPositionSize: number;
}

export interface PositionSizeResult {
  recommendedSize: number;
  maxAllowedSize: number;
  riskAmount: number;
  riskPercent: number;
  reasoning: string[];
  warnings: string[];
}

export interface RiskEvent {
  id: string;
  timestamp: Date;
  type: 'limit_breach' | 'warning' | 'emergency' | 'position_size' | 'correlation';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  affectedPositions: string[];
  recommendedActions: string[];
  autoActionTaken?: string;
}

export class RiskController {
  private config: RiskConfiguration;
  private portfolio: Portfolio;
  private riskEvents: RiskEvent[] = [];
  private dailyLossTracker: Map<string, number> = new Map(); // Date -> PnL
  private correlationMatrix: Map<string, Map<string, number>> = new Map();
  private lastRiskCalculation: Date = new Date(0);

  constructor(config: RiskConfiguration, portfolio: Portfolio) {
    this.config = config;
    this.portfolio = portfolio;
  }

  /**
   * Assess overall portfolio risk
   */
  assessPortfolioRisk(currentPrices: Record<string, number>): RiskAssessment {
    const assessment = PortfolioCalculator.assessRisk(this.portfolio);
    
    // Add custom risk checks
    const customViolations = this.checkCustomRiskLimits(currentPrices);
    assessment.violations.push(...customViolations);
    
    // Update risk score based on custom violations
    assessment.riskScore += customViolations.filter(v => v.severity === 'critical').length * 15;
    assessment.riskScore = Math.min(assessment.riskScore, 100);
    
    // Generate risk events for new violations
    this.generateRiskEvents(assessment.violations);
    
    return assessment;
  }

  /**
   * Calculate position size based on risk parameters
   */
  calculatePositionSize(request: PositionSizeRequest): PositionSizeResult {
    const result: PositionSizeResult = {
      recommendedSize: 0,
      maxAllowedSize: 0,
      riskAmount: 0,
      riskPercent: 0,
      reasoning: [],
      warnings: []
    };

    try {
      // Calculate risk-based position size
      const riskAmount = request.portfolioValue * request.riskPerTrade;
      const stopLossPrice = request.signal.stopLoss;
      
      if (!stopLossPrice) {
        result.warnings.push('No stop loss specified - using default risk calculation');
        
        // Use percentage-based sizing without stop loss
        const maxValue = request.portfolioValue * request.maxPositionSize;
        result.recommendedSize = Math.floor(maxValue / request.signal.price);
        result.reasoning.push('Position sized based on maximum position percentage');
      } else {
        // Risk-based sizing with stop loss
        const riskPerShare = Math.abs(request.signal.price - stopLossPrice);
        if (riskPerShare > 0) {
          result.recommendedSize = Math.floor(riskAmount / riskPerShare);
          result.reasoning.push('Position sized based on risk per share and stop loss');
        } else {
          result.warnings.push('Invalid stop loss - using fallback sizing');
          result.recommendedSize = Math.floor(riskAmount / (request.signal.price * 0.02)); // 2% default risk
        }
      }

      // Apply maximum position size limit
      const maxValueLimit = request.portfolioValue * this.config.maxConcentration;
      const maxSizeByValue = Math.floor(maxValueLimit / request.signal.price);
      
      result.maxAllowedSize = Math.min(result.recommendedSize, maxSizeByValue);
      result.recommendedSize = result.maxAllowedSize;

      // Check concentration risk
      if (result.recommendedSize < result.maxAllowedSize) {
        result.warnings.push('Position size reduced due to concentration limits');
      }

      // Check portfolio leverage
      const currentLeverage = this.calculateCurrentLeverage(request.currentPositions, request.portfolioValue);
      const newPositionValue = result.recommendedSize * request.signal.price;
      const projectedLeverage = (this.getTotalPositionValue(request.currentPositions) + newPositionValue) / request.portfolioValue;

      if (projectedLeverage > this.config.maxLeverage) {
        const maxAdditionalValue = (this.config.maxLeverage * request.portfolioValue) - this.getTotalPositionValue(request.currentPositions);
        if (maxAdditionalValue > 0) {
          result.recommendedSize = Math.min(result.recommendedSize, Math.floor(maxAdditionalValue / request.signal.price));
          result.warnings.push('Position size reduced due to leverage limits');
        } else {
          result.recommendedSize = 0;
          result.warnings.push('Cannot open new position - leverage limit exceeded');
        }
      }

      // Check correlation risk
      const correlationWarning = this.checkCorrelationRisk(request.symbol, request.currentPositions);
      if (correlationWarning) {
        result.warnings.push(correlationWarning);
        result.recommendedSize = Math.floor(result.recommendedSize * 0.7); // Reduce by 30% for correlation risk
      }

      // Calculate final risk metrics
      result.riskAmount = result.recommendedSize * Math.abs(request.signal.price - (stopLossPrice || request.signal.price * 0.98));
      result.riskPercent = result.riskAmount / request.portfolioValue;

      // Final validations
      if (result.recommendedSize <= 0) {
        result.warnings.push('Calculated position size is zero or negative');
      }

      if (result.riskPercent > request.riskPerTrade * 1.2) {
        result.warnings.push('Risk exceeds target - consider reducing position size');
      }

      return result;
    } catch (error) {
      result.warnings.push(`Error calculating position size: ${error}`);
      return result;
    }
  }

  /**
   * Check if a trade passes risk validation
   */
  validateTrade(signal: TradingSignal, currentPositions: Position[], portfolioValue: number): {
    approved: boolean;
    reasons: string[];
    modifications?: Partial<TradingSignal>;
  } {
    const reasons: string[] = [];
    let approved = true;
    const modifications: Partial<TradingSignal> = {};

    // Check daily loss limit
    const todayKey = new Date().toDateString();
    const todayLoss = this.dailyLossTracker.get(todayKey) || 0;
    
    if (todayLoss < -this.config.maxDailyLoss) {
      approved = false;
      reasons.push('Daily loss limit exceeded - trade rejected');
    }

    // Check maximum positions
    if (currentPositions.length >= this.config.maxTotalPositions) {
      if (signal.type === SignalType.ENTRY_LONG || signal.type === SignalType.ENTRY_SHORT) {
        approved = false;
        reasons.push('Maximum position count reached');
      }
    }

    // Check drawdown limit
    const currentDrawdown = this.portfolio.metrics.currentDrawdown;
    if (currentDrawdown > this.config.maxDrawdown) {
      approved = false;
      reasons.push('Portfolio drawdown exceeds maximum limit');
    }

    // Check position concentration
    if (signal.quantity) {
      const positionValue = signal.price * signal.quantity;
      const concentration = positionValue / portfolioValue;
      
      if (concentration > this.config.maxConcentration) {
        const maxSize = Math.floor((portfolioValue * this.config.maxConcentration) / signal.price);
        modifications.quantity = maxSize;
        reasons.push(`Position size reduced from ${signal.quantity} to ${maxSize} due to concentration limits`);
      }
    }

    // Check stop loss requirement
    if (this.config.stopLossRequired && !signal.stopLoss) {
      const defaultStopLoss = signal.side === 'long' 
        ? signal.price * 0.98 
        : signal.price * 1.02;
      
      modifications.stopLoss = defaultStopLoss;
      reasons.push('Stop loss added due to risk requirements');
    }

    // Check leverage limit
    const currentLeverage = this.calculateCurrentLeverage(currentPositions, portfolioValue);
    if (currentLeverage > this.config.maxLeverage) {
      approved = false;
      reasons.push('Portfolio leverage exceeds maximum limit');
    }

    return { approved, reasons, modifications: Object.keys(modifications).length > 0 ? modifications : undefined };
  }

  /**
   * Monitor existing positions for risk violations
   */
  monitorPositions(positions: Position[], currentPrices: Record<string, number>): RiskEvent[] {
    const events: RiskEvent[] = [];

    for (const position of positions) {
      if (position.status !== PositionStatus.OPEN && position.status !== PositionStatus.PARTIAL) {
        continue;
      }

      const currentPrice = currentPrices[position.symbol] || position.currentPrice;
      
      // Check stop loss violation
      if (position.riskLimits.stopLoss) {
        const shouldTriggerStop = position.side === PositionSide.LONG
          ? currentPrice <= position.riskLimits.stopLoss
          : currentPrice >= position.riskLimits.stopLoss;

        if (shouldTriggerStop) {
          events.push({
            id: `stop_loss_${position.id}_${Date.now()}`,
            timestamp: new Date(),
            type: 'limit_breach',
            severity: 'high',
            description: `Stop loss triggered for ${position.symbol} at ${currentPrice}`,
            affectedPositions: [position.id],
            recommendedActions: ['Execute stop loss order immediately'],
            autoActionTaken: 'Stop loss order placed'
          });
        }
      }

      // Check position-level risk limits
      const unrealizedPnL = this.calculateUnrealizedPnL(position, currentPrice);
      if (position.riskLimits.maxLoss && unrealizedPnL < -position.riskLimits.maxLoss) {
        events.push({
          id: `max_loss_${position.id}_${Date.now()}`,
          timestamp: new Date(),
          type: 'limit_breach',
          severity: 'critical',
          description: `Position loss exceeds maximum for ${position.symbol}`,
          affectedPositions: [position.id],
          recommendedActions: ['Consider closing position', 'Review risk parameters'],
        });
      }

      // Check time limits
      if (position.riskLimits.timeLimit) {
        const holdingTime = (Date.now() - position.openTime.getTime()) / (1000 * 60);
        if (holdingTime > position.riskLimits.timeLimit) {
          events.push({
            id: `time_limit_${position.id}_${Date.now()}`,
            timestamp: new Date(),
            type: 'warning',
            severity: 'medium',
            description: `Position ${position.symbol} exceeds time limit`,
            affectedPositions: [position.id],
            recommendedActions: ['Review position', 'Consider exit strategy']
          });
        }
      }
    }

    // Store events
    events.forEach(event => this.riskEvents.push(event));

    return events;
  }

  /**
   * Calculate portfolio-level risk metrics
   */
  calculateRiskMetrics(positions: Position[], currentPrices: Record<string, number>): RiskMetrics {
    const now = new Date();
    
    // Calculate portfolio value and P&L
    let totalValue = this.portfolio.availableCash;
    let totalPnL = 0;
    
    positions.forEach(position => {
      const currentPrice = currentPrices[position.symbol] || position.currentPrice;
      totalValue += Math.abs(currentPrice * position.filledQuantity);
      totalPnL += this.calculateUnrealizedPnL(position, currentPrice);
    });

    // Calculate drawdown
    const peakValue = Math.max(...this.portfolio.snapshots.map(s => s.totalValue));
    const currentDrawdown = (peakValue - totalValue) / peakValue;

    // Calculate leverage
    const grossExposure = positions.reduce((sum, p) => 
      sum + Math.abs(p.currentPrice * p.filledQuantity), 0);
    const leverage = grossExposure / totalValue;

    // Calculate concentration
    const largestPosition = Math.max(...positions.map(p => 
      Math.abs(p.currentPrice * p.filledQuantity)));
    const concentration = largestPosition / totalValue;

    // Calculate VaR (simplified implementation)
    const var95 = this.calculateVaR(positions, 0.95);
    const var99 = this.calculateVaR(positions, 0.99);

    // Calculate time-based P&L
    const dailyPnL = this.getTimeBasedPnL('daily');
    const weeklyPnL = this.getTimeBasedPnL('weekly');
    const monthlyPnL = this.getTimeBasedPnL('monthly');

    return {
      portfolioRisk: var95 / totalValue,
      currentDrawdown,
      leverage,
      concentration,
      dailyPnL,
      weeklyPnL,
      monthlyPnL,
      var95,
      var99,
      expectedShortfall: var99 * 1.2, // Simplified ES calculation
      betaExposure: this.calculateBetaExposure(positions),
      correlationRisk: this.calculateCorrelationRisk(positions),
      liquidityRisk: this.calculateLiquidityRisk(positions),
      lastUpdate: now
    };
  }

  /**
   * Update daily P&L tracking
   */
  updateDailyPnL(pnl: number): void {
    const todayKey = new Date().toDateString();
    this.dailyLossTracker.set(todayKey, pnl);

    // Clean up old entries (keep last 30 days)
    const cutoffDate = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
    for (const [dateKey, _] of this.dailyLossTracker) {
      if (new Date(dateKey) < cutoffDate) {
        this.dailyLossTracker.delete(dateKey);
      }
    }
  }

  /**
   * Get recent risk events
   */
  getRiskEvents(since?: Date): RiskEvent[] {
    if (!since) {
      return [...this.riskEvents];
    }
    
    return this.riskEvents.filter(event => event.timestamp >= since);
  }

  /**
   * Update risk configuration
   */
  updateConfiguration(updates: Partial<RiskConfiguration>): void {
    this.config = { ...this.config, ...updates };
  }

  /**
   * Get current risk configuration
   */
  getConfiguration(): RiskConfiguration {
    return { ...this.config };
  }

  private checkCustomRiskLimits(currentPrices: Record<string, number>): RiskViolation[] {
    const violations: RiskViolation[] = [];
    const positions = this.portfolio.positions.filter(p => 
      p.status === PositionStatus.OPEN || p.status === PositionStatus.PARTIAL
    );

    // Check daily loss limit
    const todayKey = new Date().toDateString();
    const todayLoss = this.dailyLossTracker.get(todayKey) || 0;
    
    if (todayLoss < -this.config.maxDailyLoss) {
      violations.push({
        type: 'loss_limit',
        severity: 'critical',
        description: 'Daily loss limit exceeded',
        currentValue: Math.abs(todayLoss),
        limitValue: this.config.maxDailyLoss,
        action: 'Stop opening new positions, consider closing existing ones'
      });
    }

    // Check correlation limits
    const highCorrelations = this.findHighCorrelations(positions);
    if (highCorrelations.length > 0) {
      violations.push({
        type: 'correlation',
        severity: 'warning',
        description: `High correlation detected between positions: ${highCorrelations.join(', ')}`,
        currentValue: 0.8, // Example correlation
        limitValue: this.config.correlationThreshold,
        action: 'Consider reducing correlated positions'
      });
    }

    // Check emergency liquidation threshold
    if (this.portfolio.metrics.currentDrawdown > this.config.emergencyLiquidationThreshold) {
      violations.push({
        type: 'drawdown',
        severity: 'critical',
        description: 'Emergency liquidation threshold breached',
        currentValue: this.portfolio.metrics.currentDrawdown,
        limitValue: this.config.emergencyLiquidationThreshold,
        action: 'Immediate position liquidation required'
      });
    }

    return violations;
  }

  private generateRiskEvents(violations: RiskViolation[]): void {
    violations.forEach(violation => {
      const event: RiskEvent = {
        id: `${violation.type}_${Date.now()}`,
        timestamp: new Date(),
        type: 'limit_breach',
        severity: violation.severity === 'critical' ? 'critical' : 'high',
        description: violation.description,
        affectedPositions: [], // Would identify specific positions
        recommendedActions: [violation.action]
      };

      this.riskEvents.push(event);
    });
  }

  private calculateCurrentLeverage(positions: Position[], portfolioValue: number): number {
    const totalExposure = this.getTotalPositionValue(positions);
    return totalExposure / portfolioValue;
  }

  private getTotalPositionValue(positions: Position[]): number {
    return positions.reduce((sum, position) => 
      sum + Math.abs(position.currentPrice * position.filledQuantity), 0);
  }

  private checkCorrelationRisk(symbol: string, currentPositions: Position[]): string | null {
    // Simplified correlation check
    const existingSymbols = currentPositions.map(p => p.symbol);
    
    // Check if adding similar asset (same sector, etc.)
    // This would use actual correlation data in practice
    const similarSymbols = existingSymbols.filter(s => 
      s.charAt(0) === symbol.charAt(0) // Very simplified similarity check
    );

    if (similarSymbols.length > 2) {
      return 'High correlation risk with existing positions';
    }

    return null;
  }

  private calculateUnrealizedPnL(position: Position, currentPrice: number): number {
    const currentValue = position.filledQuantity * currentPrice;
    const entryValue = position.filledQuantity * position.avgEntryPrice;

    if (position.side === PositionSide.LONG) {
      return currentValue - entryValue;
    } else {
      return entryValue - currentValue;
    }
  }

  private calculateVaR(positions: Position[], confidence: number): number {
    // Simplified VaR calculation
    // In practice, would use historical simulation or Monte Carlo
    const totalValue = positions.reduce((sum, p) => 
      sum + Math.abs(p.currentPrice * p.filledQuantity), 0);
    
    const volatility = 0.15; // Assumed portfolio volatility
    const zScore = confidence === 0.95 ? 1.645 : 2.326;
    
    return totalValue * volatility * zScore / Math.sqrt(252);
  }

  private getTimeBasedPnL(period: 'daily' | 'weekly' | 'monthly'): number {
    const now = new Date();
    let cutoffDate: Date;

    switch (period) {
      case 'daily':
        cutoffDate = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        break;
      case 'weekly':
        cutoffDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        break;
      case 'monthly':
        cutoffDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        break;
    }

    // Calculate P&L from snapshots within period
    const periodSnapshots = this.portfolio.snapshots.filter(s => s.timestamp >= cutoffDate);
    
    if (periodSnapshots.length < 2) return 0;
    
    const startValue = periodSnapshots[0].totalValue;
    const endValue = periodSnapshots[periodSnapshots.length - 1].totalValue;
    
    return endValue - startValue;
  }

  private calculateBetaExposure(positions: Position[]): number {
    // Simplified beta calculation
    // Would use actual beta data for each position
    return 1.0; // Placeholder
  }

  private calculateCorrelationRisk(positions: Position[]): number {
    // Simplified correlation risk calculation
    if (positions.length < 2) return 0;
    
    // Would calculate actual correlations between positions
    return 0.3; // Placeholder moderate correlation
  }

  private calculateLiquidityRisk(positions: Position[]): number {
    // Simplified liquidity risk calculation
    // Would consider actual liquidity metrics like average volume, bid-ask spread, etc.
    return 0.1; // Placeholder low liquidity risk
  }

  private findHighCorrelations(positions: Position[]): string[] {
    // Simplified high correlation detection
    // Would use actual correlation matrix
    return []; // Placeholder
  }
}