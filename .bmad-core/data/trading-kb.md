# Trading Knowledge Base

## Overview

This knowledge base contains essential information about the Lingua trading system, patterns, indicators, and methodologies used in the trading workflow.

## Lingua Trading System

Lingua is a systematic approach to trading that uses precise language to define patterns, setups, and rules. It focuses on creating reproducible, testable trading methodologies.

### Key Concepts

- **Setup**: A specific market condition or pattern that provides a trading opportunity
- **Entry Condition**: Precise rules for initiating a trade
- **Exit Condition**: Rules for closing a position (profit target or stop loss)
- **Risk Parameters**: Guidelines for position sizing and risk management
- **Pattern**: A recognizable market formation with statistical edge
- **Timeframe**: The time interval used for analysis (daily, hourly, etc.)

## Common Trading Patterns

### Mean Reversion

- **Concept**: Assets tend to return to their mean or average price over time
- **Indicators**: Bollinger Bands, RSI, Stochastics, ATR
- **Entry**: When price reaches extreme levels (oversold/overbought)
- **Exit**: When price returns to mean or predefined target

### Trend Following

- **Concept**: Assets that are trending tend to continue in that direction
- **Indicators**: Moving Averages, ADX, MACD
- **Entry**: When price confirms trend direction
- **Exit**: When trend shows signs of reversal or target reached

### Breakout

- **Concept**: Price movement beyond established range leads to continuation
- **Indicators**: Support/Resistance, Donchian Channels, ATR
- **Entry**: When price breaks above resistance or below support
- **Exit**: When price reaches target extension or shows reversal

## Indicator Reference

### Core Indicators

- **Moving Averages**: Average price over specified period
  - Simple (SMA), Exponential (EMA), Weighted (WMA)
  
- **Relative Strength Index (RSI)**: Momentum oscillator
  - Range: 0-100
  - Oversold: <30, Overbought: >70
  
- **Bollinger Bands**: Volatility-based envelope
  - Components: Middle (SMA), Upper (SMA+2σ), Lower (SMA-2σ)
  
- **Average True Range (ATR)**: Volatility measure
  - Used for: Stop placement, position sizing

### Custom Indicators

- **Lingua ATR Bands**: Modified Bollinger Bands using ATR for width
- **RSI Gradient**: Rate of change of RSI over time
- **Volume Profile**: Distribution of volume across price levels
- **Multi-timeframe Momentum**: Composite momentum across timeframes

## Risk Management Framework

### Position Sizing

- **Fixed Risk**: Risk consistent percentage of capital per trade
- **Volatility-based**: Adjust position size based on market volatility
- **Account-scaled**: Scale position size with account growth

### Stop Loss Methods

- **Technical**: Based on support/resistance or pattern invalidation
- **Volatility-based**: Multiple of ATR from entry
- **Time-based**: Exit if setup doesn't resolve within timeframe

## Performance Metrics

### Key Metrics

- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits divided by gross losses
- **Sharpe Ratio**: Return adjusted for risk (volatility)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Recovery Factor**: Net profit divided by maximum drawdown

### Benchmark Thresholds

- **Acceptable Strategy**: Sharpe >1.0, Win Rate >45%, Profit Factor >1.5
- **Good Strategy**: Sharpe >1.5, Win Rate >50%, Profit Factor >2.0
- **Excellent Strategy**: Sharpe >2.0, Win Rate >55%, Profit Factor >2.5

## Market Regimes

### Classification

- **Trending**: Strong directional movement, low volatility
- **Ranging**: Sideways movement within bounds
- **Volatile**: Large price swings, high uncertainty
- **Reversal**: Change in primary trend direction

### Regime Indicators

- **ADX**: <20 (range), >25 (trend)
- **ATR Percentile**: Current vs. historical volatility
- **Correlation Matrix**: Asset correlations across markets
- **Regime Model**: Composite classification system