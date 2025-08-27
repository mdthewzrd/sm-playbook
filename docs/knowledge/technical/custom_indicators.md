# Custom Indicators - Lingua Trading System

## Overview

The Lingua trading system uses a comprehensive set of custom indicators to systematically identify trend cycles, means, extremes, and execution points. These indicators form the "words" of our trading language, providing systematic ways to see market structure and trading opportunities.

## Core Indicator Categories

### 1. Means (EMA Clouds)
### 2. Extremes (Deviation Bands)  
### 3. Trail (Trend Following)
### 4. Supporting Indicators

---

## 1. Means - EMA Cloud System

### Purpose
Identify mean reversion levels and trend direction across multiple timeframes.

### Timeframe Structure

#### 15m Chart EMAs
- 15m 50 EMA
- 15m 72/89 cloud (primary mean)
- 15m 111 EMA  
- 15m 222 EMA
- 30m 72/89 cloud
- 1hr 72/89

#### Hourly Chart EMAs
- 1hr 72/89
- 2hr 72/89  
- 4hr 72/89

### Implementation: RahulLines Cloud

```pinescript
//@version=4
study("RahulLines Cloud", overlay=true)

sl = input(72, "Smaller length")
hl = input(89, "Higher length")

res = input(title="JLines - Time Frame 1", type=input.resolution, defval="1")
res1 = input(title="JLines - Time Frame 2", type=input.resolution, defval="3")

enable515 = input(false,"5/15 EMA")
res2 = input(title="5 /15 EMA", type=input.resolution, defval="5")

tickprice1 = security(syminfo.tickerid, res, close)
tickprice2 = security(syminfo.tickerid, res, close)

ema1_72 = security(syminfo.tickerid, res,ema(close,sl))
ema1_89 = security(syminfo.tickerid, res,ema(close,hl))
ema2_72 = security(syminfo.tickerid, res1,ema(close,sl))
ema2_89 = security(syminfo.tickerid, res1,ema(close,hl))

ema3_5 = security(syminfo.tickerid, res2,ema(close,5))
ema3_15 = security(syminfo.tickerid, res2,ema(close,15))

p1_1 = plot(ema1_72,"TimeFrame 1- SL",color=color.blue, style=plot.style_line, display=display.none)
p1_2 = plot(ema1_89,"TimeFrame 1 - HL",color=color.blue, style=plot.style_line, display=display.none)
p2_1 = plot(ema2_72, "TimeFrame 2 - SL",color=color.yellow, style=plot.style_line, display=display.none)
p2_2 = plot(ema2_89,"TimeFrame 2 - HL",color=color.yellow, style=plot.style_line, display=display.none)
p3_1 = plot(enable515?ema3_5:na, "Lade Day Fade - 5 EMA",color=color.yellow, style=plot.style_line, display=display.none)
p3_2 = plot(enable515?ema3_15:na,"Lade Day Fade  - 15 EMA",color=color.yellow, style=plot.style_line, display=display.none)

fill(p1_1, p1_2, color=ema1_72>ema1_89?color.green:color.red, transp=30, title="Background 1") 
fill(p2_1, p2_2, color=ema2_72>ema2_89?color.green:color.red,transp=90,title="Background 2") 
fill(p3_1, p3_2, color=enable515?ema3_5>ema3_15?color.blue:color.red:na,transp=50,title="Lade Day Fade ") 

dl = input(true, title="Show daily Open")

dopen = security(syminfo.tickerid, "D", open, lookahead=barmerge.lookahead_on)
plot(dl and dopen ? dopen : na, title="Day open", color=#FF9800, style=plot.style_circles, linewidth = 2, transp=0)
```

### Usage
- **15m 72/89 cloud**: Primary mean for MTF analysis
- **Color coding**: Green when 72 EMA > 89 EMA (bullish), Red when opposite
- **Multiple timeframes**: Creates wide net for mean identification
- **Daily open**: Reference level for intraday trading

---

## 2. Extremes - Dual Deviation Band System

### Purpose  
Identify extreme deviation points for entries and route identification.

### Two Versions

#### A. Main Deviation Band (72/89 Based)
Used for setup identification and route start/end points.

#### B. Execution Deviation Band (9/20 Based)  
Used for precise entry timing and execution.

### Implementation: Dual Deviation Cloud

```pinescript
//@version=5
indicator("Dual Deviation Cloud", overlay=true)

// User Inputs
deviationAbove1 = input.float(10.0, "First Positive Deviation Multiplier")
deviationAbove2 = input.float(8.0, "Second Positive Deviation Multiplier")
deviationBelow1 = input.float(10.0, "First Negative Deviation Multiplier")
deviationBelow2 = input.float(8.0, "Second Negative Deviation Multiplier")
length72EMA = input.int(72, "Length for the 72 EMA")
length89EMA = input.int(89, "Length for the 89 EMA")

// Colors for the clouds
colorAbove = color.new(color.red, 60)
colorBelow = color.new(color.green, 60)

// Calculate EMAs
ema72 = ta.ema(close, length72EMA)
ema89 = ta.ema(close, length89EMA)

// Correct ATR calculation
atr72 = ta.sma(ta.tr(true), length72EMA)
atr89 = ta.sma(ta.tr(true), length89EMA)

// Deviation lines
deviationAboveLine1 = ema72 + (deviationAbove1 * atr72)
deviationAboveLine2 = ema72 + (deviationAbove2 * atr72)
deviationBelowLine1 = ema89 - (deviationBelow1 * atr89)
deviationBelowLine2 = ema89 - (deviationBelow2 * atr89)

// Plot the clouds
plot(deviationAboveLine1, color=colorAbove, title="Deviation Above Line 1")
plot(deviationAboveLine2, color=colorAbove, title="Deviation Above Line 2")
plot(deviationBelowLine1, color=colorBelow, title="Deviation Below Line 1")
plot(deviationBelowLine2, color=colorBelow, title="Deviation Below Line 2")

fill(plot1=plot(deviationAboveLine1, display=display.none),
     plot2=plot(deviationAboveLine2, display=display.none),
     color=colorAbove)

fill(plot1=plot(deviationBelowLine1, display=display.none),
     plot2=plot(deviationBelowLine2, display=display.none),
     color=colorBelow)
```

### Parameters by Use Case

#### Short Setup Parameters (Main)
- **Upper Deviation 1**: 10.0
- **Upper Deviation 2**: 8.0  
- **Lower Deviation 1**: 10.0
- **Lower Deviation 2**: 8.0
- **EMA Length**: 72/89

#### Long Setup Parameters
- **Upper Deviation 1**: 8.0
- **Upper Deviation 2**: 6.0
- **Lower Deviation 1**: 8.0  
- **Lower Deviation 2**: 6.0
- **EMA Length**: 72/89

#### Execution Parameters (9/20 Based)
- **Upper Deviation 1**: 4.0
- **Upper Deviation 2**: 3.0
- **Lower Deviation 1**: 4.0
- **Lower Deviation 2**: 3.0  
- **EMA Length**: 9/20

### Usage
- **Red upper bands**: Excellent at picking tops in trends
- **Green lower bands**: Bounce and support levels
- **Entry timing**: Use execution version for precise fills
- **Route identification**: Use main version for bigger picture

---

## 3. Trail System - 9/20 Deviation Bands

### Purpose
Help identify trend trail when actual trend drawing isn't clear. Used for trade management and trend confirmation.

### Implementation
Uses the same Dual Deviation Cloud code but with 9/20 EMA parameters instead of 72/89.

### Key Parameters
- **EMA Length**: 9/20
- **Deviation Multipliers**: Lower values (3.0-4.0) for tighter bands
- **Color Logic**: Red when trend ending, Green when trend continuing

### Usage
- **Trend confirmation**: When we flip red, trend ends on close outside top of red cloud
- **Trail management**: Helps maintain positions when trend isn't clearly drawn
- **Execution timing**: Provides systematic approach to trend following

---

## 4. Supporting Indicators

### VWAP (Volume Weighted Average Price)
- **Purpose**: Confirmation after entry, looking for add opportunities
- **Implementation**: Standard VWAP calculation
- **Usage**: Price above VWAP = bullish context, below = bearish context

### Volume  
- **Purpose**: Liquidity confirmation and move validation
- **Implementation**: Standard volume bars with average overlays
- **Usage**: High relative volume confirms "in play" status

### Previous Day Close (PDC)
- **Purpose**: Key reference level for gap analysis
- **Implementation**: `security(syminfo.tickerid, "D", close[1])`
- **Usage**: Gap magnitude and reversion reference

### Historical Gap Stats
- **Purpose**: A+ setup qualification for small cap day one setups  
- **Implementation**: Database lookup of historical gap performance
- **Usage**: Filters for highest probability setups

---

## Python Implementation Examples

### EMA Cloud Calculation
```python
import pandas as pd
import talib as ta

def calculate_ema_cloud(df, short_period=72, long_period=89):
    """Calculate EMA cloud for mean identification"""
    df[f'ema_{short_period}'] = ta.EMA(df['close'], timeperiod=short_period)
    df[f'ema_{long_period}'] = ta.EMA(df['close'], timeperiod=long_period)
    
    # Cloud color logic
    df['cloud_bullish'] = df[f'ema_{short_period}'] > df[f'ema_{long_period}']
    
    return df

def calculate_deviation_bands(df, ema_short=72, ema_long=89, 
                            upper_mult1=10.0, upper_mult2=8.0,
                            lower_mult1=10.0, lower_mult2=8.0):
    """Calculate deviation bands for extreme identification"""
    
    # Calculate EMAs
    df[f'ema_{ema_short}'] = ta.EMA(df['close'], timeperiod=ema_short)
    df[f'ema_{ema_long}'] = ta.EMA(df['close'], timeperiod=ema_long)
    
    # Calculate ATR
    df[f'atr_{ema_short}'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=ema_short)
    df[f'atr_{ema_long}'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=ema_long)
    
    # Upper deviation bands
    df['upper_dev_1'] = df[f'ema_{ema_short}'] + (upper_mult1 * df[f'atr_{ema_short}'])
    df['upper_dev_2'] = df[f'ema_{ema_short}'] + (upper_mult2 * df[f'atr_{ema_short}'])
    
    # Lower deviation bands  
    df['lower_dev_1'] = df[f'ema_{ema_long}'] - (lower_mult1 * df[f'atr_{ema_long}'])
    df['lower_dev_2'] = df[f'ema_{ema_long}'] - (lower_mult2 * df[f'atr_{ema_long}'])
    
    return df
```

### Integration with Polygon Data
```python
def apply_lingua_indicators(df):
    """Apply complete Lingua indicator suite to OHLCV data"""
    
    # Means - Multiple EMA clouds
    df = calculate_ema_cloud(df, 72, 89)  # Primary cloud
    df = calculate_ema_cloud(df, 9, 20)   # Execution cloud
    
    # Add single EMAs
    for period in [50, 111, 222]:
        df[f'ema_{period}'] = ta.EMA(df['close'], timeperiod=period)
    
    # Extremes - Deviation bands
    df = calculate_deviation_bands(df)  # Main bands
    df = calculate_deviation_bands(df, 9, 20, 4.0, 3.0, 4.0, 3.0)  # Execution bands
    
    # Supporting indicators
    df['vwap'] = calculate_vwap(df)
    df['volume_avg'] = ta.SMA(df['volume'], timeperiod=20)
    
    return df

def calculate_vwap(df):
    """Calculate Volume Weighted Average Price"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap
```

---

## Integration with MCP Architecture

### Data Flow
1. **Polygon Client**: Fetches OHLCV data
2. **Indicator Processor**: Applies Lingua indicators  
3. **Signal Processor**: Identifies setup conditions
4. **Strategy Controller**: Makes trading decisions
5. **Execution Engine**: Places orders via OsEngine

### Backtesting Integration
```python
from backtesting import Strategy
import sys
sys.path.append('../../mcp-integration')
from processors.indicator_processor import IndicatorProcessor

class LinguaStrategy(Strategy):
    def init(self):
        # Apply Lingua indicators
        self.data = apply_lingua_indicators(self.data)
        
        # Setup signals
        self.upper_dev = self.data['upper_dev_1']
        self.lower_dev = self.data['lower_dev_1'] 
        self.ema_cloud = self.data['cloud_bullish']
    
    def next(self):
        # Implement Lingua trading logic
        if self.position.size == 0:
            # Entry logic using indicators
            if self.data.close[-1] > self.upper_dev[-1]:
                self.sell()  # Short at extreme deviation
```

---

## Indicator Configuration Files

### Default Parameters
```yaml
# lingua_indicators.yaml
indicators:
  ema_clouds:
    primary:
      short: 72
      long: 89
      timeframes: ['15m', '1h', '2h', '4h']
    execution:
      short: 9
      long: 20
      timeframes: ['2m', '5m']
  
  deviation_bands:
    main:
      upper_1: 10.0
      upper_2: 8.0
      lower_1: 10.0
      lower_2: 8.0
      base_ema: [72, 89]
    
    execution:
      upper_1: 4.0
      upper_2: 3.0
      lower_1: 4.0
      lower_2: 3.0
      base_ema: [9, 20]
  
  supporting:
    vwap: true
    volume_average: 20
    previous_day_close: true
```

---

## Performance and Optimization

### Calculation Efficiency
- **Vectorized Operations**: All calculations use pandas/numpy vectorization
- **Cached Indicators**: Results cached to avoid recalculation
- **Minimal Lookback**: Only calculate necessary periods

### Memory Management
- **Rolling Windows**: Use fixed-size rolling windows for real-time data
- **Data Cleanup**: Remove unnecessary columns after calculation
- **Chunked Processing**: Process large datasets in chunks

### Real-time Updates
- **Incremental Updates**: Only recalculate latest bars
- **WebSocket Integration**: Real-time price updates trigger indicator updates
- **Multi-threading**: Separate threads for indicator calculation and display

---

*These custom indicators form the systematic backbone of the Lingua trading language, providing objective ways to identify the subjective patterns that drive profitable trading opportunities.*