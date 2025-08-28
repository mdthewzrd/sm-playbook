#!/usr/bin/env python3

"""Debug August 19th to see why no immediate entry after Route Start"""

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

def debug_august_19th():
    config_path = "/Users/michaeldurante/sm-playbook/config/backtests/IBIT_GTZ_Short_v1_Test.json"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
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
    
    symbol = config['backtest_configuration']['simulation_environment']['symbol']
    
    # Download data for August 19th and previous days to get proper EMA calculation
    aggs_1h = []
    for agg in polygon_client.get_aggs(
        ticker=symbol,
        multiplier=60,
        timespan='minute',
        from_='2025-08-16',  # Get more historical data for accurate EMAs
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
    
    aggs_5m = []
    for agg in polygon_client.get_aggs(
        ticker=symbol,
        multiplier=5,
        timespan='minute',
        from_='2025-08-16',  # Get more historical data for accurate EMAs
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
    
    # Convert to DataFrames
    df_1h = pd.DataFrame(aggs_1h)
    df_1h.set_index('timestamp', inplace=True)
    df_1h.sort_index(inplace=True)
    
    df_5m = pd.DataFrame(aggs_5m)
    df_5m.set_index('timestamp', inplace=True)
    df_5m.sort_index(inplace=True)
    
    # Calculate indicators
    indicators_1h = indicator_calc.calculate(df_1h)
    indicators_5m = indicator_calc.calculate(df_5m)
    
    logger.info("=== AUGUST 19TH DEBUG ===")
    logger.info(f"5M data points: {len(df_5m)}")
    logger.info(f"Time range: {df_5m.index[0]} to {df_5m.index[-1]}")
    
    # Check 9:00 AM Route Start
    route_start_time = pd.Timestamp('2025-08-19 09:00:00', tz='America/New_York')
    
    # Show several 1H bars around this time
    logger.info(f"\nAll 1H bars on August 19th:")
    aug_19_data = indicators_1h[indicators_1h.index.date == pd.Timestamp('2025-08-19').date()]
    for timestamp, row in aug_19_data.iterrows():
        logger.info(f"{timestamp.strftime('%H:%M')}: OHLC={row['open']:.2f},{row['high']:.2f},{row['low']:.2f},{row['close']:.2f} "
                   f"EMA9={row['fast_ema']:.2f}, EMA20={row['slow_ema']:.2f}, Short Bias={row['fast_ema'] < row['slow_ema']}")
    
    # Also show some bars from previous days for context
    logger.info(f"\n1H bars from August 16-18 (last few bars each day):")
    for date_str in ['2025-08-16', '2025-08-17', '2025-08-18']:
        date_data = indicators_1h[indicators_1h.index.date == pd.Timestamp(date_str).date()]
        if len(date_data) > 0:
            # Show last bar of each day
            last_bar = date_data.iloc[-1]
            timestamp = date_data.index[-1]
            logger.info(f"{timestamp.strftime('%m/%d %H:%M')}: Close={last_bar['close']:.2f}, "
                       f"EMA9={last_bar['fast_ema']:.2f}, EMA20={last_bar['slow_ema']:.2f}, Short Bias={last_bar['fast_ema'] < last_bar['slow_ema']}")
    
    # Get 1H data at Route Start
    h1_times = indicators_1h.index[indicators_1h.index <= route_start_time]
    if len(h1_times) > 0:
        current_h1 = indicators_1h.loc[h1_times[-1]]
        logger.info(f"\nUsing 1H bar at {h1_times[-1]} for Route Start check:")
        logger.info(f"  EMA9: {current_h1['fast_ema']:.2f}")
        logger.info(f"  EMA20: {current_h1['slow_ema']:.2f}")
        logger.info(f"  Short bias: {current_h1['fast_ema'] < current_h1['slow_ema']}")
        
        # Also check a few bars before
        if len(h1_times) >= 3:
            for i in range(max(0, len(h1_times)-3), len(h1_times)):
                bar_time = h1_times[i]
                bar_data = indicators_1h.loc[bar_time]
                logger.info(f"  {bar_time.strftime('%m/%d %H:%M')}: EMA9={bar_data['fast_ema']:.2f}, EMA20={bar_data['slow_ema']:.2f}")
    
    # Check 5M bars around Route Start
    logger.info(f"\n5M bars around Route Start:")
    route_start_window = df_5m[(df_5m.index >= pd.Timestamp('2025-08-19 08:55:00', tz='America/New_York')) &
                               (df_5m.index <= pd.Timestamp('2025-08-19 09:35:00', tz='America/New_York'))]
    
    for timestamp, row in route_start_window.iterrows():
        # Check if this bar should trigger Route Start
        h1_ema9 = current_h1['fast_ema'] if len(h1_times) > 0 else 0
        route_trigger = row['high'] > h1_ema9
        time_in_window = time(8, 0) <= timestamp.time() <= time(12, 0)
        
        logger.info(f"{timestamp.strftime('%H:%M')}: High={row['high']:.2f}, Close={row['close']:.2f}, "
                   f"Route Trigger={route_trigger}, Time Window={time_in_window}")
        
        if route_trigger and time_in_window:
            logger.info(f"  *** ROUTE START TRIGGERED - Should enter SHORT at {row['close']:.2f} ***")

if __name__ == "__main__":
    debug_august_19th()