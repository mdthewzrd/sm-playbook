# Risk Management Parameters

## Overview

This document defines the comprehensive risk management framework for the SM Playbook trading system, including position sizing, stop loss placement, and portfolio-level risk controls.

## Account Risk Limits

### Daily Risk Limits
- **Maximum Daily Loss**: 3% of account value
- **Maximum Daily Trades**: 5 new positions
- **Daily Risk Budget**: Allocated across active strategies

### Weekly Risk Limits
- **Maximum Weekly Loss**: 5% of account value
- **Maximum Weekly Trades**: 15 new positions
- **Weekly Drawdown Alert**: 3% triggers review

### Monthly Risk Limits
- **Maximum Monthly Loss**: 10% of account value
- **Monthly Performance Target**: 2-5% positive return
- **Monthly Review Threshold**: -5% triggers strategy audit

## Position Sizing Framework

### Base Position Size Calculation

```
Position Size = (Account Value × Risk Per Trade) / (Entry Price - Stop Loss Price)
```

### Risk Per Trade Guidelines
- **Conservative**: 0.5% - 1% per trade
- **Standard**: 1% - 1.5% per trade
- **Aggressive**: 1.5% - 2% per trade
- **Maximum**: Never exceed 2% per trade

### Volatility Adjustment

Position sizes are adjusted based on market volatility:

- **Low Volatility (ATR < 20th percentile)**: Increase size by 25%
- **Normal Volatility (ATR 20-80th percentile)**: Standard size
- **High Volatility (ATR > 80th percentile)**: Decrease size by 25%

## Stop Loss Methods

### Technical Stops
- **Support/Resistance**: Place stops beyond key levels
- **Pattern Stops**: Based on pattern invalidation points
- **Swing Stops**: Beyond recent swing highs/lows

### Volatility-Based Stops
- **ATR Multiple**: 1.5x to 2.5x ATR from entry
- **Bollinger Band Stops**: Beyond opposite band
- **Standard Deviation**: 2 standard deviations from mean

### Time-Based Stops
- **Maximum Hold Period**: 5-10 trading days
- **Intraday Stops**: Close all positions at market close
- **Weekend Risk**: Reduce positions before weekends

## Portfolio-Level Risk Controls

### Concentration Limits
- **Single Position**: Maximum 5% of portfolio
- **Sector Concentration**: Maximum 20% in single sector
- **Strategy Concentration**: Maximum 30% in single strategy
- **Correlation Limit**: Maximum 3 highly correlated positions

### Heat Management
- **Portfolio Heat**: Sum of all position risks
- **Maximum Heat**: 8% of account value
- **Heat Reduction**: Required when exceeding 6%

### Drawdown Controls
- **Maximum Drawdown**: 15% from peak equity
- **Drawdown Alert**: 8% triggers position review
- **Drawdown Action**: 12% triggers strategy pause

## Dynamic Risk Adjustment

### Market Regime Adjustments

#### Trending Markets
- **Risk Increase**: Up to 25% larger positions
- **Stop Distance**: Wider stops (2.5x ATR)
- **Hold Period**: Extended to capture trends

#### Ranging Markets
- **Risk Neutral**: Standard position sizes
- **Stop Distance**: Tighter stops (1.5x ATR)
- **Hold Period**: Shorter, quicker profits

#### Volatile Markets
- **Risk Reduction**: 25-50% smaller positions
- **Stop Distance**: Very tight stops (1x ATR)
- **Hold Period**: Minimal exposure time

### Performance-Based Adjustments

#### Winning Streaks
- **Gradual Increase**: Add 10% to position size after 3 wins
- **Maximum Increase**: 50% above base size
- **Reset Trigger**: Any loss resets to base size

#### Losing Streaks
- **Gradual Decrease**: Reduce 20% after 2 consecutive losses
- **Maximum Reduction**: 50% below base size
- **Recovery Trigger**: 2 consecutive wins restore base size

## Emergency Procedures

### Account Drawdown Response
- **5% Drawdown**: Review all open positions
- **8% Drawdown**: Close weakest positions
- **10% Drawdown**: Reduce all position sizes by 50%
- **12% Drawdown**: Close all positions, pause trading

### Market Crisis Response
- **Volatility Spike**: Immediately reduce position sizes
- **Gap Risk**: Close positions with overnight exposure
- **Black Swan**: Emergency liquidation of all positions
- **System Failure**: Manual intervention procedures

## Risk Monitoring and Reporting

### Daily Risk Reports
- Current portfolio heat and exposure
- Individual position risk and P&L
- Daily risk budget utilization
- Stop loss and target distances

### Weekly Risk Reviews
- Portfolio performance vs. benchmarks
- Risk-adjusted return metrics
- Strategy performance attribution
- Risk limit compliance audit

### Monthly Risk Analysis
- Comprehensive risk metrics calculation
- Stress testing and scenario analysis
- Risk parameter optimization
- Strategy risk profile updates