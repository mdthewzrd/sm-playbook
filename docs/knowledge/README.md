# SM Playbook Knowledge Base

## Overview

This knowledge base contains the complete discretionary trading methodology, systematic strategies, and technical implementations that form the foundation of the SM Playbook algorithmic trading system. All knowledge is organized around the **Lingua Trading Language** - a comprehensive framework for systematically trading price action patterns.

## 🎯 Quick Start

### Essential Reading Order
1. **[Lingua Trading Language](core/lingua_trading_language.md)** - Start here: Complete trading methodology
2. **[OS D1 Setup](strategies/os_d1_small_cap_day_one_setup.md)** - Primary systematic strategy
3. **[Custom Indicators](technical/custom_indicators.md)** - Technical implementation guide

### Core Philosophy
- Everything is fractal across timeframes
- Trend cycles drive all market movement  
- Systematic indicators identify discretionary patterns
- High EV spots provide consistent edge
- Full algorithmic approach scales infinitely

## 📚 Knowledge Base Structure

### 🎯 Core Concepts (`core/`)
Foundational trading methodology and principles.

#### [Lingua Trading Language](core/lingua_trading_language.md)
**The complete discretionary trading framework** - This is the master document that contains:

#### [Lingua Trading Language - Complete Visual Guide](core/lingua_trading_language_complete.md)
**Full methodology with 64 annotated chart examples** - Visual learning version with:
- **Trend Cycles**: The 8-stage cycle that drives all market movement
- **Market Structure**: How trends interact across timeframes  
- **Daily Context**: Classification system for different setup types
- **Timeframe Hierarchy**: HTF/MTF/LTF execution framework
- **High EV Spots**: Systematic identification of profitable opportunities
- **Business Strategy**: Path to full algorithmic implementation

*This document represents 4+ years of trading experience distilled into a systematic approach.*

### ⚡ Strategies (`strategies/`)
Documented and systematic trading setups.

#### [OS D1 Small Cap Day One Setup](strategies/os_d1_small_cap_day_one_setup.md)
**The flagship systematic strategy** with extensive backtesting data:
- **Performance**: 60-75% win rate depending on context
- **Risk/Reward**: 2-5R per successful trade
- **Entry Types**: FBO, Extensions, Dev Band Pops
- **Opening Stages**: Frontside, High & Tight, Backside Pop, Deep Backside
- **Complete Framework**: Entry, pyramiding, covering, and risk management

#### Available Code Files
- `OS D1 Setup.md` - Original strategy documentation
- Multiple scanner implementations (`.py` files)
- Backtesting results and validation data

### 🔧 Technical Implementation (`technical/`)
Code, indicators, and system architecture.

#### [Custom Indicators](technical/custom_indicators.md)
**Complete indicator system** for Lingua trading language:
- **Means**: EMA cloud system (72/89, 9/20) with multi-timeframe analysis
- **Extremes**: Dual deviation bands for entry/exit timing  
- **Trail**: Trend following system for trade management
- **Integration**: Python/Pine Script implementations with MCP architecture

#### Implementation Files
- `analyze+.py`, `exec.py` - Analysis and execution tools
- Multiple scanner files - Various strategy implementations  
- Chart generation tools - Visualization and validation

### 📊 Reference Data (`reference/`)
All additional code, scans, and implementation examples.

#### Code Examples
- **A+ Scans**: `A+ daily para 1.0.py`, `half A+ scan.py`
- **Analysis Tools**: `fboanalyzereal.py`, `frontsidebucket.py`
- **Chart Tools**: `chartsave.py`, `d1chartsave.py`, `lc chart save.py`
- **Specialized Scans**: `mdr swing scan.py`, `og scans.py`, `produgscan.py`

#### Supporting Documentation
- `code dump.md` - Collection of code snippets and implementations
- Various specialized scanner files and analysis tools

## 🔗 Integration Points

### MCP Architecture Integration
All knowledge integrates seamlessly with the SM Playbook MCP architecture:

- **Models**: `trading-code/models/` - Data structures align with Lingua concepts
- **Controllers**: `trading-code/controllers/` - Business logic implements Lingua rules
- **Processors**: `trading-code/processors/` - Signal processing uses Lingua indicators  
- **Clients**: `trading-code/clients/` - Data sources (Polygon, TA-Lib) support Lingua calculations

### Data Pipeline
```
Polygon.io → Lingua Indicators → Signal Processing → Strategy Execution → OsEngine
```

### Backtesting Framework
```
Historical Data → Lingua Analysis → backtesting.py → Performance Validation → Strategy Optimization
```

## 🎯 Key Concepts Reference

### Lingua Trend Cycle (8 Stages)
1. **Consolidation** → 2. **Breakout** → 3. **Uptrend** → 4. **Extreme Deviation** → 5. **Euphoric Top** → 6. **Trend Break** → 7. **Backside** → 8. **Backside Reverted** → Repeat

### Timeframe Framework
- **HTF (High)**: Daily, 4hr - Setup ID, bias, key levels
- **MTF (Mid)**: 1hr, 15m - Route start/end, main trend breaks  
- **LTF (Low)**: 5m, 2m - Execution timing, trade management

### High EV Spots
- **Euphoric Tops**: Mean reversion at trend extremes
- **Trend Breaks**: Transition points with momentum
- **Dev Band Extremes**: Systematic over-extension identification
- **FBO (Failed Breakouts)**: High probability reversal patterns

### Daily Context Categories
- **Frontside**: At all-time highs  
- **Backside**: Below previous swing high
- **IPO**: Initial public offering context

## 📈 Performance Metrics

### OS D1 Strategy Results
- **Overall Win Rate**: 70.6% (FBO entries across all contexts)
- **A+ Setups**: 80%+ win rate in optimal contexts
- **Risk Management**: Maximum 3R loss per setup
- **Time Efficiency**: Most profitable trades within first hour

### System Capabilities
- **Asset Coverage**: Any liquid instrument worldwide
- **Timeframe Scalability**: 1m to daily analysis
- **Automation Ready**: Fully systematized for algorithmic execution
- **Backtesting**: Extensive historical validation across multiple years

## 🚀 Implementation Roadmap

### Phase 1: Foundation (Current)
- ✅ Complete Lingua documentation
- ✅ OS D1 systematic implementation  
- ✅ Custom indicator system
- ✅ MCP architecture integration

### Phase 2: Expansion (Next)
- **G2G S1 Strategy**: Gap-to-go systematic approach
- **SC DMR**: Small cap daily momentum reversal
- **LC Strategies**: Large cap trading systems
- **Multi-Asset**: Crypto, forex, futures adaptation

### Phase 3: Scale (Future) 
- **Fleet of Algorithms**: Multiple concurrent strategies
- **Machine Learning**: Pattern recognition enhancement
- **Portfolio Optimization**: Risk-adjusted allocation
- **Product Development**: Systematic strategy marketplace

## 🔍 Search and Navigation

### Find Information By:
- **Setup Type**: Search for "FBO", "Extension", "Dev Band Pop"
- **Timeframe**: Search for "15m", "5m", "HTF", "MTF", "LTF"  
- **Stage**: Search for "Uptrend", "Backside", "Euphoric Top"
- **Code**: Search for ".py", "scanner", "backtest"
- **Performance**: Search for "win rate", "R", "EV"

### Cross-References
- **Lingua → OS D1**: How core methodology applies to specific strategy
- **Indicators → Implementation**: From theory to code
- **Strategy → Backtesting**: Performance validation
- **Discretionary → Systematic**: Converting manual patterns to algorithms

## 📞 Support and Updates

### Key Files to Monitor
- **Core Changes**: `lingua_trading_language.md`
- **Strategy Updates**: `os_d1_small_cap_day_one_setup.md`
- **Technical Changes**: `custom_indicators.md`
- **New Implementations**: Watch `/strategies/` folder

### Integration Support
- **BMAD Framework**: Automatic agent-driven development
- **MCP Architecture**: Seamless component integration
- **Backtesting Engine**: Continuous validation
- **Documentation**: Auto-generated from code comments

## 🎯 Success Metrics

### Knowledge Base Goals
- **Completeness**: 100% of discretionary knowledge systematized
- **Accessibility**: Any concept findable within 30 seconds
- **Implementation**: Direct code examples for every concept
- **Validation**: Backtesting data for every strategy

### Trading System Goals  
- **Performance**: >1.0 Sharpe ratio across all strategies
- **Scale**: $1M+/month capacity
- **Reliability**: >99.5% system uptime
- **Accuracy**: >99.9% data quality

---

**Welcome to the SM Playbook Knowledge Base - Where discretionary expertise meets systematic execution.**

*This knowledge base represents the complete systematization of 4+ years of profitable trading experience, designed to scale from individual strategies to a full algorithmic trading operation generating $1M+ monthly revenue.*