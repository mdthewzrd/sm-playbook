# OS D1 Small Cap Day One Setup

## Strategy Overview

The OS D1 (Opening Stage Day One) is a systematic small cap day trading strategy that focuses on heavily gapped stocks on their first day of significant movement. This strategy leverages the high volatility and predictable patterns that emerge when small cap stocks experience major gap ups with significant volume.

## Core Concept

The OS D1 strategy identifies three main high Expected Value (EV) entry points:
1. **FBO (Failed Breakout)**
2. **Extension** 
3. **Dev Band Pop (Deviation Band Pop)**

These entries are classified based on opening stage context to tailor the trading approach for maximum profitability.

## Opening Stage Classification

At 9:25 AM, we classify the opening stage to understand setup context:

### 1. Frontside
- **Duration**: Until 5m trigger
- **Characteristics**: Lower grade setup (41% A+ or better, 16% C grade)
- **Performance**: -1.43R process
- **Best Opportunities**: Opening FBO (75% A+ or better)
- **Approach**: Less aggressive, focus on opening FBO primarily

### 2. High and Tight  
- **Duration**: Until close below 5m 20 EMA or 15m trigger
- **Characteristics**: Higher grade (56.8% A+ or better, 22.7% C)
- **Best Opportunities**: Opening FBO (80% A+ better), Opening EXT (75% A+ better)
- **Valid Entries**: Opening FBO, EXT, morning FBO/EXT, 5m DB push

### 3. Backside Pop
- **Duration**: Until multiple closes below 15m 20 EMA or 15m 9/20 cross
- **Characteristics**: High grade (60% A+ or better, 9% C)
- **Best Opportunities**: Opening FBO (72.8% A+ better), Opening EXT (67% A+ better)
- **Approach**: Open-centric setup (10:30 cutoff)

### 4. Deep Backside
- **Duration**: Until 5m BDB hit pre-pop
- **Requirements**: 
  - Confirm liquid PM extension and fade
  - Check if 5m bull dev band hit on fade
  - Valid pop level: 15m 9/20 bear dev band + 60% fib from high to PM low

### 5. Reverted D1
- **Trigger**: After 5m BDB hit
- **Special Rule**: Wait for first 5m top to clear out (backside reverted contingency)

## High EV Entry Spots

### Extensions
- **Best Timing**: Non-frontside, open or morning
- **Quality**: High & tight or below (not frontside extensions)
- **Pattern**: Parabolic extension from initial consolidation
- **Key Statistics**:
  - 1 BB top: 83% A+ or better
  - 2 BB tops: 58% A+ better (100% green on B grade)
  - 3-4 BB tops: 28% A+, 42% B grade

### FBO (Failed Breakout)
- **Overall Performance**: 70.6% A+ better across all setups
- **Opening FBO**: 77% A+ better
- **Morning FBO**: 55% A+ better  
- **Combined Morning + Open**: 67% A+ better
- **Context Performance**:
  - High & tight or Frontside: 71% A+ better
  - Backside/Deep backside: 68% A+ better

### Dev Band Pop
- **Trigger**: 1m fail candle at deviation band
- **Performance**: Varies by context (detailed in opening stage sections)

## Entry Process Framework

### Risk Management Rules
- **Maximum Loss**: 3R per setup
- **Maximum Starters**: 2 per setup  
- **Maximum 5m BB**: 2 per setup
- **Time Cutoff**: 10:30 AM (except A+ setups with blowouts)
- **Post 2 × 0.25R Cuts**: Wait for trigger to re-enter

### Entry Structure
All setups follow the same pyramiding structure:

| **Entry Type** | **Size** | **Stop** | **Description** |
|----------------|----------|----------|-----------------|
| **Starter** | 0.25R | Context-specific | Initial position |
| **Pre-Trigger** | 0.25R | 1c over highs | Confirmation add |  
| **Trigger** | 1R | 1c over highs | Main position |

### Context-Specific Entry Rules

#### Frontside Entries
| **High EV Spot** | **Starter** | **Pre Trig** | **Trig** |
|------------------|-------------|--------------|----------|
| Opening FBO | 2m FBO vs 10% stop | 2m BB vs 1c over highs | 5m BB vs 1c over highs |
| Dev Band Pop | 1m fail vs PMH | 2m BB vs 1c over highs | 5m BB vs 1c over highs |
| Opening EXT | N/A | 2m BB vs 10% stop | 5m BB vs 1c over highs |

#### High Range Post Fail
| **High EV Spot** | **Starter** | **Pre Trig** | **Trig** |
|------------------|-------------|--------------|----------|
| Opening FBO | 2m FBO vs 10% stop | 2m BB vs 1c over highs | 5m BB vs 1c over highs |
| Dev Band Pop | 1m fail vs PMH | 2m BB vs 1c over highs | 5m BB vs 1c over highs |
| Opening EXT | 2m BB vs 10% stop | 2m Trig vs 1c over highs | 5m BB vs 1c over highs |
| Morning FBO | 2m FBO vs 10% stop | 2m BB vs 1c over highs | 5m BB vs 1c over highs |
| Morning EXT | N/A | 2m BB vs 10% stop | 5m BB vs 1c over highs |

#### Backside Pop  
| **High EV Spot** | **Starter** | **Pre Trig** | **Trig** |
|------------------|-------------|--------------|----------|
| Opening FBO | 2m FBO vs 10% stop | 2m BB vs 1c over highs | 5m BB vs 1c over highs |
| Dev Band Pop | 1m fail vs PMH | 2m BB vs 1c over highs | 5m BB vs 1c over highs |
| Opening EXT | 2m BB vs 10% stop | 2m Trig vs 1c over highs | 5m BB vs 1c over highs |
| Morning EXT | N/A | 2m BB vs 10% stop | 5m BB vs 1c over highs |

#### Deep Backside & Reverted
Same structure as Backside Pop, with additional validation:
- Confirm liquid PM extension and fade
- Check 5m bull dev band hit status
- Valid pop requires: 15m 9/20 bear dev band + 60% fib level
- If 5m bull dev band hit: Use backside reverted contingency

## Pyramiding Strategy

### Signal and Trigger
- **Pyramid Signal**: 5m DB push
- **Pyramid Trigger**: 2m BB

### Pyramid Levels

#### Small Pyramid (≤15% from highs to VWAP)
- **Trigger**: 5m VWAP or 2m 9/20 cross
- **Priority**: Always prioritize swing highs
- **Note**: Consolidating over VWAP is normal

#### Medium Pyramid (≤30% from highs to VWAP)  
- **Trigger**: 5m trig post 5m VWAP confirmation or 2m pivot
- **Priority**: Always prioritize swing highs
- **Room**: Can still clear out

#### Consolidation Build (≤50% from highs to VWAP)
- **Trigger**: 2m 9/20 cross + 2 checkpoints above
- **Use**: Can build in consolidation

#### Aggressive Pyramid
- **Trigger**: Consolidation breakdown vs last swing high near VWAP
- **Approach**: Should be done now, get aggressive

#### Non-breakdown Adds
- **Trigger**: 5m 9/20 flip or 5m DB pushes
- **Target**: vs 2 swing high ago (2m minimum, ideally 5m)
- **Use**: When no breakdown but getting add opportunities

### Cutoff Point
**Stop pyramiding once we hit the 5m J-line**

## Cover Strategy

### Philosophy
Capture the main 5m trend using systematic approach.

### Main Cover Signal
- **Primary**: 5m BDB (Bear Deviation Band)
- **Secondary**: 5m 200 EMA

### Cover Triggers
Once main signal hit, use three-tier covering:

| **Cover Trigger** | **Action** |
|-------------------|------------|
| 2m BB | Cover 1/3 |
| 5m BB | Cover 1/3 |  
| 15m BB | Cover 1/3 |

### Special Cases

#### Very Steep Move
- Use 1m BB and 2m BB for 50% cover

#### Long Tightening Downtrend  
- 5m BB/2m BB hasn't happened for extended period
- Cover more or get fully flat on break
- Better trend + deeper move = less waiting for curl

### Cover Guidelines
- Need true backside and 5m structure for trend trail
- If not clear trend, use 5m DB push with 5m trail
- Use 5m, 10m, 15m 9/20 guides to start trail
- VWAP continuation and 2m BDB hit as guides
- **Critical**: If 5m dev band hasn't been tested with 5m fail, can't trust lowering the high

## A+ Setup Criteria

### Gap Statistics Requirements
- Recent dilution history
- Baggies (heavily shorted stocks)
- Parabolic characteristics

### Fundamental Criteria  
- Recent amended dilution
- Strong cash need
- Positive historical diluter

### Technical Requirements
- Give high range or over PMH entry
- Work early in the morning (first 15 minutes)

## Scanner Implementation

The OS D1 setup uses a systematic scanner to identify qualified candidates:

### Core Filters
```python
# Trigger Day Criteria
df['trig_day'] = ((df['pm_high'] / df['prev_close'] - 1>= .5) & 
                (df['gap'] >= 0.5) & 
                (df['open'] / df['prev_high'] - 1>= .3) & 
                (df['pm_vol'] >= 5000000) & 
                (df['prev_close'] >= 0.75)).astype(int)

# EMA Validation
# Must be <= 80% of 200 EMA with minimum 200 periods of data
if c <= ema200*0.8 and len(daily_data_a) >= 200:
    df_trig_day.loc[i, 'ema_valid'] = 1

# D2 Filter (avoid day-2 fades)  
df['d2'] = ((df['prev_close'] / df['prev_close_1'] - 1>= .3) & 
            (df['prev_volume'] >= 10000000)).astype(int)
```

### Market Cap Validation
- Uses Polygon API for real-time market cap calculation
- Ensures liquidity requirements are met
- Filters for appropriate size for day trading

## Implementation Notes

### Data Requirements
- **Primary Source**: Polygon.io for all market data
- **Timeframes**: 1m, 2m, 5m, 15m, 1h for execution
- **Daily Data**: For EMA validation and context

### Integration Points
- **Backtesting**: backtesting.py engine for historical validation
- **Indicators**: TA-Lib for technical calculations  
- **Risk Management**: Automated position sizing and stop management
- **Execution**: OsEngine integration for order management

### Key Performance Metrics
- **Overall Win Rate**: 60-75% depending on context
- **Risk/Reward**: Target 2-5R per successful trade
- **Maximum Drawdown**: 3R per setup
- **Time Efficiency**: Most profitable trades occur within first hour

## Files and Code References

### Scanner Files
- `os_d1_complete_scanner.py`: Main scanner implementation
- `os_d1_momentum_scanner.py`: Momentum-focused version
- `os_d1_trading_system.py`: Complete trading system

### Analysis Files  
- `os_d1_complete_backtest_results.csv`: Historical performance data
- `os_d1_chart_validation.py`: Chart pattern validation
- `os_d1_trade_validation_results.csv`: Trade execution analysis

### Execution Files
- `os_d1_exact_strategy.py`: Precise entry/exit logic
- `os_d1_final_strategy.py`: Production-ready implementation

---

*The OS D1 setup represents one of the most systematized and profitable components of the Lingua trading language, with extensive backtesting data and proven performance metrics across multiple market conditions.*