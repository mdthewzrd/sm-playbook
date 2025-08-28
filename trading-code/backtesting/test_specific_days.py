#!/usr/bin/env python3

"""Quick test for specific days (15th, 19th) to debug the strategy"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'indicators'))

from ibit_gtz_short_backtest import IBITGTZStrategy
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_specific_days():
    """Test the 15th and 19th specifically"""
    
    config_path = "/Users/michaeldurante/sm-playbook/config/backtests/IBIT_GTZ_Short_v1_Test.json"
    strategy = IBITGTZStrategy(config_path)
    
    # Download data
    strategy.download_data()
    strategy.calculate_indicators()
    
    # Focus on specific dates
    test_dates = ['2025-08-15', '2025-08-19']
    
    for test_date in test_dates:
        logger.info(f"\n=== ANALYZING {test_date} ===")
        
        # Get data for this date
        m5_data = strategy.indicators['5m']
        h1_data = strategy.indicators['1h']
        
        # Filter to this date
        date_mask = m5_data.index.date == pd.to_datetime(test_date).date()
        day_data = m5_data[date_mask]
        
        if day_data.empty:
            logger.info(f"No data for {test_date}")
            continue
            
        logger.info(f"Bars available: {len(day_data)}")
        
        # Check regime at market open
        market_open = day_data.index[day_data.index.time >= pd.to_datetime('09:30').time()][0]
        h1_times = h1_data.index[h1_data.index <= market_open]
        if len(h1_times) > 0:
            h1_current = h1_data.loc[h1_times[-1]]
            regime_short = h1_current['fast_ema'] < h1_current['slow_ema']
            logger.info(f"9:30 AM Regime: EMA9={h1_current['fast_ema']:.2f}, EMA20={h1_current['slow_ema']:.2f}, Short Bias={regime_short}")
            
        # Check if price ever went above 1H EMA9 during 8-12 window
        setup_window = day_data[(day_data.index.time >= pd.to_datetime('08:00').time()) & 
                               (day_data.index.time <= pd.to_datetime('12:00').time())]
        
        if not setup_window.empty:
            max_high = setup_window['high'].max()
            if len(h1_times) > 0:
                h1_ema9 = h1_current['fast_ema']
                route_start_triggered = max_high > h1_ema9
                logger.info(f"Setup window high: {max_high:.2f}, 1H EMA9: {h1_ema9:.2f}, Route Start: {route_start_triggered}")
                
        # Show key levels and potential entry points
        logger.info(f"Day range: ${day_data['low'].min():.2f} - ${day_data['high'].max():.2f}")
        
        # Check EMA breaks during regular session
        session_data = day_data[(day_data.index.time >= pd.to_datetime('09:30').time()) & 
                               (day_data.index.time <= pd.to_datetime('16:00').time())]
        
        if not session_data.empty:
            ema9_breaks = session_data[session_data['low'] < session_data['fast_ema']]
            ema20_breaks = session_data[session_data['low'] < session_data['slow_ema']]
            
            logger.info(f"EMA9 breaks: {len(ema9_breaks)} times")
            logger.info(f"EMA20 breaks: {len(ema20_breaks)} times")
            
            if len(ema9_breaks) > 0:
                first_break = ema9_breaks.index[0]
                logger.info(f"First EMA9 break at: {first_break.strftime('%H:%M')} @ ${ema9_breaks.loc[first_break]['close']:.2f}")
                
        print("-" * 50)

if __name__ == "__main__":
    test_specific_days()