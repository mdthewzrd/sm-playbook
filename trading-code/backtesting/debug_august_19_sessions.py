#!/usr/bin/env python3

"""Debug August 19th, 2025 Scalping Sessions - Detailed Analysis"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'indicators'))

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, time, timedelta
from polygon import RESTClient
from dual_deviation_cloud import DualDeviationCloud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_august_19_sessions():
    """Detailed analysis of August 19th, 2025 scalping sessions"""
    
    # Initialize data client
    api_key = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
    polygon_client = RESTClient(api_key)
    
    # Initialize indicators
    indicator_calc = DualDeviationCloud({
        'ema_fast_length': 9,
        'ema_slow_length': 20,
        'positive_dev_1': 1.0,
        'positive_dev_2': 0.5,
        'negative_dev_1': 2.0,
        'negative_dev_2': 2.4
    })
    
    symbol = "IBIT"
    target_date = "2025-08-19"
    
    # Download extended data for proper EMA calculation
    logger.info(f"=== AUGUST 19TH, 2025 SESSION ANALYSIS ===")
    
    # Get 1H data (with history for accurate EMAs)
    aggs_1h = []
    for agg in polygon_client.get_aggs(
        ticker=symbol,
        multiplier=60,
        timespan='minute',
        from_='2025-08-16',  # Extended range for EMA accuracy
        to='2025-08-19',
        adjusted=True,
        sort='asc',
        limit=50000
    ):
        aggs_1h.append({
            'timestamp': pd.Timestamp(agg.timestamp, unit='ms', tz='America/New_York'),
            'open': agg.open,
            'high': agg.high,
            'low': agg.low,
            'close': agg.close,
            'volume': agg.volume
        })
    
    # Get 5M data
    aggs_5m = []
    for agg in polygon_client.get_aggs(
        ticker=symbol,
        multiplier=5,
        timespan='minute',
        from_='2025-08-16',
        to='2025-08-19',
        adjusted=True,
        sort='asc',
        limit=50000
    ):
        aggs_5m.append({
            'timestamp': pd.Timestamp(agg.timestamp, unit='ms', tz='America/New_York'),
            'open': agg.open,
            'high': agg.high,
            'low': agg.low,
            'close': agg.close,
            'volume': agg.volume
        })
    
    # Get 2M data for scalping analysis
    aggs_2m = []
    for agg in polygon_client.get_aggs(
        ticker=symbol,
        multiplier=2,
        timespan='minute',
        from_='2025-08-16',
        to='2025-08-19',
        adjusted=True,
        sort='asc',
        limit=50000
    ):
        aggs_2m.append({
            'timestamp': pd.Timestamp(agg.timestamp, unit='ms', tz='America/New_York'),
            'open': agg.open,
            'high': agg.high,
            'low': agg.low,
            'close': agg.close,
            'volume': agg.volume
        })
    
    # Convert to DataFrames
    df_1h = pd.DataFrame(aggs_1h)
    df_1h.set_index('timestamp', inplace=True)
    df_1h.sort_index(inplace=True)
    
    df_5m = pd.DataFrame(aggs_5m)
    df_5m.set_index('timestamp', inplace=True)
    df_5m.sort_index(inplace=True)
    
    df_2m = pd.DataFrame(aggs_2m)
    df_2m.set_index('timestamp', inplace=True)
    df_2m.sort_index(inplace=True)
    
    # Calculate indicators
    indicators_1h = indicator_calc.calculate(df_1h)
    indicators_5m = indicator_calc.calculate(df_5m)
    indicators_2m = indicator_calc.calculate(df_2m)
    
    # Filter to August 19th only
    aug_19 = pd.Timestamp('2025-08-19').date()
    
    aug_19_1h = indicators_1h[indicators_1h.index.date == aug_19]
    aug_19_5m = indicators_5m[indicators_5m.index.date == aug_19]
    aug_19_2m = indicators_2m[indicators_2m.index.date == aug_19]
    
    logger.info(f"\nAugust 19th Data Summary:")
    logger.info(f"1H bars: {len(aug_19_1h)}")
    logger.info(f"5M bars: {len(aug_19_5m)}")
    logger.info(f"2M bars: {len(aug_19_2m)}")
    
    # === STEP 1: ROUTE START ANALYSIS ===
    logger.info(f"\n=== STEP 1: ROUTE START ANALYSIS ===")
    
    # Check 1H short bias at market open
    if len(aug_19_1h) > 0:
        first_1h = aug_19_1h.iloc[0]
        logger.info(f"First 1H bar: {aug_19_1h.index[0].strftime('%H:%M')}")
        logger.info(f"  EMA9: {first_1h['fast_ema']:.2f}")
        logger.info(f"  EMA20: {first_1h['slow_ema']:.2f}")
        logger.info(f"  Short Bias: {first_1h['fast_ema'] < first_1h['slow_ema']}")
    
    # Find Route Start (5M high > 1H EMA9 in GTZ window)
    route_start_found = False
    route_start_time = None
    
    for timestamp, row in aug_19_5m.iterrows():
        # GTZ window check (8 AM - 12 PM)
        if not (time(8, 0) <= timestamp.time() <= time(12, 0)):
            continue
            
        # Get corresponding 1H EMA9
        h1_times = indicators_1h.index[indicators_1h.index <= timestamp]
        if len(h1_times) == 0:
            continue
            
        current_h1 = indicators_1h.loc[h1_times[-1]]
        h1_ema9 = current_h1['fast_ema']
        short_bias = current_h1['fast_ema'] < current_h1['slow_ema']
        
        if not short_bias:
            continue
            
        # Check Route Start
        if row['high'] > h1_ema9 and not route_start_found:
            route_start_found = True
            route_start_time = timestamp
            logger.info(f"\n🚀 ROUTE START at {timestamp.strftime('%H:%M')}")
            logger.info(f"  5M High: {row['high']:.2f} > 1H EMA9: {h1_ema9:.2f}")
            logger.info(f"  1H Short Bias: {short_bias}")
            break
    
    if not route_start_found:
        logger.info("❌ No Route Start found on August 19th")
        return
    
    # === STEP 2: 5M EMA CROSS ANALYSIS ===
    logger.info(f"\n=== STEP 2: 5M EMA CROSS ANALYSIS ===")
    
    # Find 5M bearish cross after Route Start
    bearish_cross_found = False
    bearish_cross_time = None
    
    aug_19_5m_after_route = aug_19_5m[aug_19_5m.index > route_start_time]
    
    for i in range(1, len(aug_19_5m_after_route)):
        current = aug_19_5m_after_route.iloc[i]
        previous = aug_19_5m_after_route.iloc[i-1]
        timestamp = aug_19_5m_after_route.index[i]
        
        # Bearish cross: EMA9 was above/equal EMA20, now below
        if (previous['fast_ema'] >= previous['slow_ema'] and 
            current['fast_ema'] < current['slow_ema'] and 
            not bearish_cross_found):
            
            bearish_cross_found = True
            bearish_cross_time = timestamp
            logger.info(f"\n📉 5M BEARISH CROSS at {timestamp.strftime('%H:%M')}")
            logger.info(f"  Previous: EMA9={previous['fast_ema']:.2f}, EMA20={previous['slow_ema']:.2f}")
            logger.info(f"  Current:  EMA9={current['fast_ema']:.2f}, EMA20={current['slow_ema']:.2f}")
            logger.info(f"  🎯 SCALPING SESSION STARTED")
            break
    
    if not bearish_cross_found:
        logger.info("❌ No 5M bearish cross found after Route Start")
        return
    
    # === STEP 3: SCALPING SESSION ANALYSIS ===
    logger.info(f"\n=== STEP 3: SCALPING SESSION ANALYSIS ===")
    
    # Track scalping session
    session_active = True
    session_end_time = None
    scalp_opportunities = []
    
    # Start from bearish cross time
    aug_19_5m_session = aug_19_5m[aug_19_5m.index >= bearish_cross_time]
    aug_19_2m_session = aug_19_2m[aug_19_2m.index >= bearish_cross_time]
    
    # Find session end (5M bullish cross)
    for i in range(1, len(aug_19_5m_session)):
        current = aug_19_5m_session.iloc[i]
        previous = aug_19_5m_session.iloc[i-1]
        timestamp = aug_19_5m_session.index[i]
        
        # Check for bullish cross (session end)
        if (previous['fast_ema'] < previous['slow_ema'] and 
            current['fast_ema'] > current['slow_ema']):
            
            session_end_time = timestamp
            logger.info(f"\n📈 5M BULLISH CROSS at {timestamp.strftime('%H:%M')}")
            logger.info(f"  🛑 SCALPING SESSION ENDED")
            break
    
    if session_end_time is None:
        session_end_time = aug_19_5m.index[-1]  # End of day
        logger.info(f"Session continued until end of day: {session_end_time.strftime('%H:%M')}")
    
    # === STEP 4: 2M DEVIATION BAND SCALP ANALYSIS ===
    logger.info(f"\n=== STEP 4: 2M DEVIATION BAND SCALPING ANALYSIS ===")
    
    session_2m_data = aug_19_2m[(aug_19_2m.index >= bearish_cross_time) & 
                                (aug_19_2m.index < session_end_time)]
    
    logger.info(f"Scalping Session Duration: {bearish_cross_time.strftime('%H:%M')} to {session_end_time.strftime('%H:%M')}")
    logger.info(f"2M bars in session: {len(session_2m_data)}")
    
    # Find scalping opportunities
    in_trade = False
    entry_time = None
    entry_price = None
    
    for timestamp, row in session_2m_data.iterrows():
        if not in_trade:
            # Look for entry (pop into upper_band_1)
            if row['high'] >= row['upper_band_1']:
                in_trade = True
                entry_time = timestamp
                entry_price = row['close']
                logger.info(f"\n💥 SCALP ENTRY at {timestamp.strftime('%H:%M')}")
                logger.info(f"  High: {row['high']:.2f} >= Upper Band 1: {row['upper_band_1']:.2f}")
                logger.info(f"  Entry Price: ${entry_price:.2f}")
        else:
            # Look for exit (hit lower_band_1)
            if row['low'] <= row['lower_band_1']:
                exit_price = row['close']
                pnl = (entry_price - exit_price) * 1000
                
                scalp_opportunities.append({
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'duration_minutes': (timestamp - entry_time).total_seconds() / 60
                })
                
                logger.info(f"🎯 SCALP EXIT at {timestamp.strftime('%H:%M')}")
                logger.info(f"  Low: {row['low']:.2f} <= Lower Band 1: {row['lower_band_1']:.2f}")
                logger.info(f"  Exit Price: ${exit_price:.2f}")
                logger.info(f"  P&L: ${pnl:.2f}")
                logger.info(f"  Duration: {(timestamp - entry_time).total_seconds() / 60:.1f} minutes")
                
                in_trade = False
                entry_time = None
                entry_price = None
    
    # Handle any remaining open trade at session end
    if in_trade:
        exit_price = session_2m_data.iloc[-1]['close']
        pnl = (entry_price - exit_price) * 1000
        
        scalp_opportunities.append({
            'entry_time': entry_time,
            'exit_time': session_end_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'duration_minutes': (session_end_time - entry_time).total_seconds() / 60
        })
        
        logger.info(f"\n🛑 SESSION END CLOSE at {session_end_time.strftime('%H:%M')}")
        logger.info(f"  Exit Price: ${exit_price:.2f}")
        logger.info(f"  P&L: ${pnl:.2f}")
    
    # === SUMMARY ===
    logger.info(f"\n=== AUGUST 19TH SCALPING SESSION SUMMARY ===")
    logger.info(f"Route Start: {route_start_time.strftime('%H:%M')}")
    logger.info(f"5M Bearish Cross: {bearish_cross_time.strftime('%H:%M')}")
    logger.info(f"Session End: {session_end_time.strftime('%H:%M')}")
    logger.info(f"Total Scalp Opportunities: {len(scalp_opportunities)}")
    
    total_pnl = sum(trade['pnl'] for trade in scalp_opportunities)
    winning_trades = [t for t in scalp_opportunities if t['pnl'] > 0]
    avg_duration = np.mean([t['duration_minutes'] for t in scalp_opportunities]) if scalp_opportunities else 0
    
    logger.info(f"Total P&L: ${total_pnl:.2f}")
    logger.info(f"Win Rate: {len(winning_trades)/len(scalp_opportunities)*100:.1f}%" if scalp_opportunities else "N/A")
    logger.info(f"Average Duration: {avg_duration:.1f} minutes")
    
    # Show price action during session
    logger.info(f"\n=== SESSION PRICE ACTION ===")
    session_start_price = session_2m_data.iloc[0]['close'] if len(session_2m_data) > 0 else 0
    session_end_price = session_2m_data.iloc[-1]['close'] if len(session_2m_data) > 0 else 0
    price_move = session_end_price - session_start_price
    
    logger.info(f"Session Start Price: ${session_start_price:.2f}")
    logger.info(f"Session End Price: ${session_end_price:.2f}")
    logger.info(f"Total Price Move: ${price_move:.2f}")
    
    return scalp_opportunities

if __name__ == "__main__":
    debug_august_19_sessions()