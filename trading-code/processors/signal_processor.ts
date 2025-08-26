import { CandleData } from '../models/market_data';
import { 
  IndicatorValue, 
  EMACloudConfig, 
  ATRBandsConfig, 
  RSIGradientConfig,
  IndicatorCalculator 
} from '../models/indicator';
import { TradingSignal, SignalType, SignalStrength } from '../models/strategy';
import { Timeframe } from '../types/common';

export interface SignalProcessingConfig {
  enableEMACloud: boolean;
  enableATRBands: boolean;
  enableRSIGradient: boolean;
  enableMomentum: boolean;
  signalStrengthThreshold: number;
  confirmationRequired: boolean;
  multiTimeframeAnalysis: boolean;
}

export interface SignalContext {
  symbol: string;
  timeframe: Timeframe;
  timestamp: number;
  price: number;
  indicators: Map<string, IndicatorValue>;
  historicalSignals: TradingSignal[];
}

export interface ProcessedSignal {
  signal: TradingSignal;
  confidence: number;
  components: SignalComponent[];
  reasoning: string;
  riskLevel: 'low' | 'medium' | 'high';
}

export interface SignalComponent {
  name: string;
  value: number;
  weight: number;
  contribution: number;
  type: 'bullish' | 'bearish' | 'neutral';
}

export class SignalProcessor {
  private calculator: IndicatorCalculator;
  private config: SignalProcessingConfig;

  constructor(config?: Partial<SignalProcessingConfig>) {
    this.calculator = new IndicatorCalculator();
    this.config = {
      enableEMACloud: true,
      enableATRBands: true,
      enableRSIGradient: true,
      enableMomentum: true,
      signalStrengthThreshold: 0.6,
      confirmationRequired: true,
      multiTimeframeAnalysis: true,
      ...config
    };
  }

  async processSignals(
    marketData: CandleData[],
    symbol: string,
    timeframe: Timeframe
  ): Promise<ProcessedSignal[]> {
    if (marketData.length < 100) {
      throw new Error('Insufficient data for signal processing (minimum 100 candles)');
    }

    const signals: ProcessedSignal[] = [];
    const indicators = await this.calculateAllIndicators(marketData);

    for (let i = 89; i < marketData.length; i++) {
      const context: SignalContext = {
        symbol,
        timeframe,
        timestamp: marketData[i].timestamp,
        price: marketData[i].close,
        indicators: this.getIndicatorsAtIndex(indicators, i),
        historicalSignals: signals.map(s => s.signal)
      };

      const processedSignal = await this.generateSignal(context, marketData.slice(0, i + 1));
      if (processedSignal) {
        signals.push(processedSignal);
      }
    }

    return this.filterSignals(signals);
  }

  private async calculateAllIndicators(data: CandleData[]): Promise<Map<string, IndicatorValue[]>> {
    const indicators = new Map<string, IndicatorValue[]>();

    if (this.config.enableEMACloud) {
      const emaCloud72_89 = this.calculator.calculateEMACloud(data, { fastPeriod: 72, slowPeriod: 89 });
      const emaCloud9_20 = this.calculator.calculateEMACloud(data, { fastPeriod: 9, slowPeriod: 20 });
      indicators.set('emaCloud_72_89', emaCloud72_89);
      indicators.set('emaCloud_9_20', emaCloud9_20);
    }

    if (this.config.enableATRBands) {
      const atrBands = this.calculator.calculateATRBands(data, { period: 14, multiplier: 2.0 });
      indicators.set('atrBands', atrBands);
    }

    if (this.config.enableRSIGradient) {
      const rsiGradient = this.calculator.calculateRSIGradient(data, { period: 14, smoothing: 3 });
      indicators.set('rsiGradient', rsiGradient);
    }

    if (this.config.enableMomentum) {
      const momentum = this.calculator.calculateMomentum(data, 14);
      indicators.set('momentum', momentum);
    }

    const volume = this.calculator.calculateVolumeProfile(data, 20);
    const volatility = this.calculator.calculateVolatility(data, 20);
    
    indicators.set('volume', volume);
    indicators.set('volatility', volatility);

    return indicators;
  }

  private getIndicatorsAtIndex(indicators: Map<string, IndicatorValue[]>, index: number): Map<string, IndicatorValue> {
    const result = new Map<string, IndicatorValue>();
    
    for (const [name, values] of indicators) {
      if (values[index]) {
        result.set(name, values[index]);
      }
    }
    
    return result;
  }

  private async generateSignal(
    context: SignalContext,
    historicalData: CandleData[]
  ): Promise<ProcessedSignal | null> {
    const components: SignalComponent[] = [];
    let totalScore = 0;
    let totalWeight = 0;

    if (this.config.enableEMACloud) {
      const emaComponent = this.analyzeEMACloud(context);
      if (emaComponent) {
        components.push(emaComponent);
        totalScore += emaComponent.contribution;
        totalWeight += emaComponent.weight;
      }
    }

    if (this.config.enableATRBands) {
      const atrComponent = this.analyzeATRBands(context);
      if (atrComponent) {
        components.push(atrComponent);
        totalScore += atrComponent.contribution;
        totalWeight += atrComponent.weight;
      }
    }

    if (this.config.enableRSIGradient) {
      const rsiComponent = this.analyzeRSIGradient(context);
      if (rsiComponent) {
        components.push(rsiComponent);
        totalScore += rsiComponent.contribution;
        totalWeight += rsiComponent.weight;
      }
    }

    if (this.config.enableMomentum) {
      const momentumComponent = this.analyzeMomentum(context);
      if (momentumComponent) {
        components.push(momentumComponent);
        totalScore += momentumComponent.contribution;
        totalWeight += momentumComponent.weight;
      }
    }

    const volumeComponent = this.analyzeVolume(context, historicalData);
    if (volumeComponent) {
      components.push(volumeComponent);
      totalScore += volumeComponent.contribution;
      totalWeight += volumeComponent.weight;
    }

    if (totalWeight === 0 || components.length === 0) {
      return null;
    }

    const confidence = totalScore / totalWeight;
    
    if (Math.abs(confidence) < this.config.signalStrengthThreshold) {
      return null;
    }

    const signalType: SignalType = confidence > 0 ? 'buy' : 'sell';
    const strength = this.calculateSignalStrength(Math.abs(confidence));
    
    const signal: TradingSignal = {
      id: `${context.symbol}_${context.timeframe}_${context.timestamp}`,
      symbol: context.symbol,
      type: signalType,
      strength,
      price: context.price,
      timestamp: context.timestamp,
      timeframe: context.timeframe,
      indicators: Object.fromEntries(context.indicators),
      metadata: {
        confidence,
        components: components.length,
        source: 'signal_processor'
      }
    };

    const reasoning = this.generateReasoning(components, confidence);
    const riskLevel = this.assessRiskLevel(components, confidence);

    return {
      signal,
      confidence: Math.abs(confidence),
      components,
      reasoning,
      riskLevel
    };
  }

  private analyzeEMACloud(context: SignalContext): SignalComponent | null {
    const emaCloud72_89 = context.indicators.get('emaCloud_72_89');
    const emaCloud9_20 = context.indicators.get('emaCloud_9_20');
    
    if (!emaCloud72_89 || !emaCloud9_20) return null;

    const longTermBias = emaCloud72_89.value > 0 ? 1 : -1;
    const shortTermBias = emaCloud9_20.value > 0 ? 1 : -1;
    
    const alignment = longTermBias === shortTermBias ? 1 : 0;
    const strength = Math.abs(emaCloud9_20.value) / 100;
    
    const value = longTermBias * shortTermBias * alignment * strength;

    return {
      name: 'EMA Cloud',
      value,
      weight: 0.4,
      contribution: value * 0.4,
      type: value > 0 ? 'bullish' : value < 0 ? 'bearish' : 'neutral'
    };
  }

  private analyzeATRBands(context: SignalContext): SignalComponent | null {
    const atrBands = context.indicators.get('atrBands');
    if (!atrBands || typeof atrBands.value !== 'object') return null;

    const bands = atrBands.value as { upper: number; middle: number; lower: number; position: number };
    const position = bands.position;
    
    let value = 0;
    if (position > 0.8) {
      value = -0.7; // Near upper band, potential reversal
    } else if (position < 0.2) {
      value = 0.7; // Near lower band, potential reversal
    } else if (position > 0.6) {
      value = 0.3; // Above middle, bullish momentum
    } else if (position < 0.4) {
      value = -0.3; // Below middle, bearish momentum
    }

    return {
      name: 'ATR Bands',
      value,
      weight: 0.25,
      contribution: value * 0.25,
      type: value > 0 ? 'bullish' : value < 0 ? 'bearish' : 'neutral'
    };
  }

  private analyzeRSIGradient(context: SignalContext): SignalComponent | null {
    const rsiGradient = context.indicators.get('rsiGradient');
    if (!rsiGradient) return null;

    const gradient = rsiGradient.value;
    const normalizedGradient = Math.tanh(gradient / 10);
    
    return {
      name: 'RSI Gradient',
      value: normalizedGradient,
      weight: 0.2,
      contribution: normalizedGradient * 0.2,
      type: normalizedGradient > 0 ? 'bullish' : normalizedGradient < 0 ? 'bearish' : 'neutral'
    };
  }

  private analyzeMomentum(context: SignalContext): SignalComponent | null {
    const momentum = context.indicators.get('momentum');
    if (!momentum) return null;

    const normalizedMomentum = Math.tanh(momentum.value / 100);
    
    return {
      name: 'Momentum',
      value: normalizedMomentum,
      weight: 0.1,
      contribution: normalizedMomentum * 0.1,
      type: normalizedMomentum > 0 ? 'bullish' : normalizedMomentum < 0 ? 'bearish' : 'neutral'
    };
  }

  private analyzeVolume(context: SignalContext, historicalData: CandleData[]): SignalComponent | null {
    const volume = context.indicators.get('volume');
    if (!volume || historicalData.length < 20) return null;

    const currentVolume = historicalData[historicalData.length - 1].volume;
    const avgVolume = historicalData.slice(-20).reduce((sum, candle) => sum + candle.volume, 0) / 20;
    
    const volumeRatio = currentVolume / avgVolume;
    let value = 0;

    if (volumeRatio > 2) {
      value = 0.5; // High volume confirmation
    } else if (volumeRatio > 1.5) {
      value = 0.3; // Above average volume
    } else if (volumeRatio < 0.5) {
      value = -0.2; // Low volume warning
    }

    return {
      name: 'Volume',
      value,
      weight: 0.05,
      contribution: value * 0.05,
      type: value > 0 ? 'bullish' : value < 0 ? 'bearish' : 'neutral'
    };
  }

  private calculateSignalStrength(confidence: number): SignalStrength {
    if (confidence >= 0.8) return 'strong';
    if (confidence >= 0.65) return 'medium';
    return 'weak';
  }

  private generateReasoning(components: SignalComponent[], confidence: number): string {
    const bullishComponents = components.filter(c => c.type === 'bullish');
    const bearishComponents = components.filter(c => c.type === 'bearish');
    const direction = confidence > 0 ? 'bullish' : 'bearish';
    
    let reasoning = `${direction.toUpperCase()} signal (${(Math.abs(confidence) * 100).toFixed(1)}% confidence) based on:\n`;
    
    const relevantComponents = confidence > 0 ? bullishComponents : bearishComponents;
    relevantComponents.forEach(component => {
      const contributionPct = (Math.abs(component.contribution) * 100).toFixed(1);
      reasoning += `• ${component.name}: ${contributionPct}% contribution\n`;
    });

    if (relevantComponents.length === 0) {
      reasoning += '• Mixed signals with slight bias';
    }

    return reasoning.trim();
  }

  private assessRiskLevel(components: SignalComponent[], confidence: number): 'low' | 'medium' | 'high' {
    const conflictingSignals = components.filter(c => c.type !== (confidence > 0 ? 'bullish' : 'bearish')).length;
    const signalStrength = Math.abs(confidence);
    
    if (conflictingSignals >= components.length / 2) return 'high';
    if (signalStrength < 0.7) return 'medium';
    return 'low';
  }

  private filterSignals(signals: ProcessedSignal[]): ProcessedSignal[] {
    if (!this.config.confirmationRequired) return signals;

    return signals.filter((signal, index) => {
      if (index === 0) return true;
      
      const previousSignal = signals[index - 1];
      const timeDiff = signal.signal.timestamp - previousSignal.signal.timestamp;
      
      if (timeDiff < 300000) { // 5 minutes
        return signal.signal.type === previousSignal.signal.type;
      }
      
      return true;
    });
  }

  updateConfiguration(config: Partial<SignalProcessingConfig>): void {
    this.config = { ...this.config, ...config };
  }

  getConfiguration(): SignalProcessingConfig {
    return { ...this.config };
  }
}