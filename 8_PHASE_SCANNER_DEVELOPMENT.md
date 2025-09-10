# 8-Phase Scanner Development Process

## Overview
The 8-Phase Scanner Development Process is a systematic approach to building high-quality trading scanners that prioritize quality over quantity. This process ensures that every scanner is validated against known quality examples and produces consistent, reliable results.

## Core Principles
- **Quality Over Quantity**: Focus on finding the best setups, not the most setups
- **Exact Parameter Adherence**: 0.3 means exactly 0.3, not approximately
- **Benchmark Validation**: Always validate against known quality examples (e.g., HOOD 3/3/25)
- **Systematic Progression**: Each phase builds upon the previous one

## The 8 Phases

### Phase 1: Single Ticker Analysis
- Start with one known quality example (e.g., HOOD on 3/3/25)
- Analyze the complete setup including:
  - Pre-market action and key levels
  - Opening range dynamics
  - Volume patterns and relative volume
  - Price action characteristics
  - Technical indicators at the time
- Document all observations and patterns

### Phase 2: Analyzer Development
- Create analysis tools to measure the patterns identified in Phase 1
- Develop metrics for:
  - Gap measurements (percentage and dollar)
  - Volume analysis (relative volume, surge detection)
  - Price action patterns (range breaks, momentum)
  - Technical indicator readings
- Test analyzer on the single ticker to ensure accuracy

### Phase 3: Parameter Baseline Discovery
- Use the analyzer to establish baseline parameters
- Document exact values from the quality example:
  - Gap percentage: exact value (e.g., 15.2%)
  - Relative volume: exact multiplier (e.g., 3.5x)
  - Price levels: exact support/resistance
  - Indicator readings: exact values
- These become the foundation for the scanner

### Phase 4: Initial Scanner Creation
- Build scanner using exact baseline parameters
- No approximations or rounding
- Test scanner to ensure it identifies the original example
- Initial scanner should be highly restrictive

### Phase 5: Name Testing & Mold Fitting
- Test scanner on additional known quality examples
- Verify pattern consistency across different tickers
- Adjust parameters only if necessary to capture quality setups
- Document any parameter adjustments and reasoning

### Phase 6: Time Period Testing
- Test scanner across different market conditions:
  - Bull markets
  - Bear markets
  - Choppy/sideways markets
  - Different volatility regimes
- Ensure scanner maintains quality standards
- Document performance across periods

### Phase 7: Optimization & Parameter Refinement
- Fine-tune parameters based on testing results
- Always prioritize quality over quantity
- Use exact trader specifications - no approximations
- Validate that original benchmarks still pass

### Phase 8: Validation & Historical Backtesting
- Comprehensive backtesting across extended periods
- Validate against all known quality benchmarks
- Calculate performance metrics:
  - Win rate
  - Risk/reward ratios
  - Maximum favorable/adverse excursions
  - Distribution of results
- Final quality assurance before deployment

## Critical Benchmarks

### HOOD 3/3/25 - A+ Setup
- **Pre-market**: Gap up with strong volume
- **Opening**: Clear break above resistance
- **Pattern**: Backside pop configuration
- **Volume**: 3.5x relative volume surge
- **Result**: Clean momentum continuation

### Additional Quality Examples
- AAPL specific dates with A+ setups
- TSLA momentum days
- Small cap runners with specific characteristics

## Implementation in SM Playbook

### Claude Code Integration
```python
# Using the scanner builder agent
from claude_code_integration import build_scanner_8phase

# Build scanner with exact parameters
scanner = build_scanner_8phase(
    pattern_name="backside_pop",
    benchmark_ticker="HOOD",
    benchmark_date="2025-03-03",
    gap_percentage=15.2,  # Exact value, not approximate
    relative_volume=3.5,   # Exact multiplier
    price_above_vwap=True
)
```

### Agent Commands
- `*build-scanner [pattern_name]` - Start 8-phase development
- `*validate-scanner [ticker] [date]` - Validate against benchmark
- `*optimize-scanner` - Run phase 7 optimization
- `*backtest-scanner [start_date] [end_date]` - Phase 8 validation

### MCP Integration
The scanner builder integrates with:
- **Polygon.io**: Real-time and historical data
- **TA-Lib**: Technical indicator calculations
- **backtesting.py**: Historical validation
- **Notion**: Documentation and trade journaling

## Quality Standards

### Acceptance Criteria
- Scanner must identify all known A+ setups
- False positive rate < 20%
- Average setup quality score > 8/10
- Consistent performance across market conditions

### Parameter Exactness Rules
1. Never round parameters unless explicitly instructed
2. Use exact values from trader specifications
3. Document why each parameter has its specific value
4. Validate that exact parameters produce expected results

## Web App Implementation

### UI Components
1. **Phase Progress Tracker**: Visual representation of current phase
2. **Parameter Editor**: Exact value input with validation
3. **Benchmark Validator**: Test against known examples
4. **Backtesting Dashboard**: Historical performance metrics
5. **Quality Score Display**: Real-time quality assessment

### API Endpoints
```
POST /api/scanner/build
  - Initiates 8-phase development
  
GET /api/scanner/validate/{ticker}/{date}
  - Validates against specific benchmark
  
POST /api/scanner/optimize
  - Runs optimization with exact parameters
  
POST /api/scanner/backtest
  - Performs historical validation
```

## Common Pitfalls to Avoid

1. **Approximating Parameters**: Never use "about" or "approximately"
2. **Quantity Chasing**: Don't loosen parameters just to find more setups
3. **Skipping Phases**: Each phase provides critical validation
4. **Ignoring Benchmarks**: Always validate against HOOD 3/3/25 and others
5. **Premature Optimization**: Complete all phases before final optimization

## Continuous Improvement

### Post-Deployment Monitoring
- Track scanner performance in live markets
- Compare actual results to backtested expectations
- Document new A+ setups for future validation
- Refine parameters based on real-world performance

### Version Control
- Document all parameter changes
- Maintain changelog of scanner versions
- Keep benchmark validation results for each version
- Enable rollback to previous versions if needed

## Integration with Lingua Framework

The 8-phase process aligns with Lingua's multi-timeframe analysis:
- **HTF (Higher Time Frame)**: Market regime identification
- **MTF (Medium Time Frame)**: Pattern recognition
- **LTF (Lower Time Frame)**: Entry optimization

Each phase incorporates Lingua concepts:
- Stage analysis (8-stage trend cycle)
- Risk management (R-multiple framework)
- Technical analysis (deviation bands, EMA clouds)

## Conclusion

The 8-Phase Scanner Development Process ensures that every scanner in the SM Playbook system is built to the highest standards, validated against real-world examples, and optimized for quality over quantity. By following this systematic approach and maintaining exact parameter adherence, we create scanners that consistently identify high-probability trading opportunities.

Remember: **0.3 means exactly 0.3, not approximately!**