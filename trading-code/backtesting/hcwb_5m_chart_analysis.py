#!/usr/bin/env python3
"""
HCWB 5-Minute Chart Analysis - Find the actual 5m bar breaks
Look at the 5m chart to identify correct break points
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, time

plt.style.use('dark_background')

def fetch_intraday_data(ticker, date, api_key):
    """Fetch intraday data from Polygon API"""
    trade_date = pd.to_datetime(date).strftime('%Y-%m-%d')
    
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{trade_date}/{date}'
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                df = pd.DataFrame(data['results'])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_convert('US/Eastern')
                df = df.set_index('timestamp')
                
                # Full trading session: 4:00 AM - 8:00 PM ET
                session_start = time(4, 0)
                session_end = time(20, 0) 
                df = df[(df.index.time >= session_start) & (df.index.time <= session_end)]
                
                df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
                return df
                
    except Exception as e:
        print(f"Error: {e}")
    
    return pd.DataFrame()

def resample_to_5min(df):
    """Resample to 5-minute bars"""
    return df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

def analyze_5m_chart():
    """Analyze the actual 5-minute chart to find real break points"""
    
    ticker = 'HCWB'
    date = '2025-08-25'
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    
    print(f"📊 ANALYZING ACTUAL 5-MINUTE CHART FOR {ticker}")
    print(f"   Finding real 5m bar breaks from the 5m chart")
    
    # Fetch 1-minute data and create 5m bars
    df_1m = fetch_intraday_data(ticker, date, api_key)
    df_5m = resample_to_5min(df_1m)
    
    # Focus on market session
    market_5m = df_5m[df_5m.index.time >= time(9, 30)]
    
    print(f"✅ Got {len(market_5m)} 5-minute bars for market session")
    print(f"\n📋 5-MINUTE BARS FROM 9:30 AM ONWARD:")
    print(f"{'Time':<8} {'Open':<6} {'High':<6} {'Low':<6} {'Close':<6} {'Comments'}")
    print("-" * 60)
    
    # Display first 20 bars to see the pattern
    hod = market_5m['high'].max()
    running_high = 0
    break_candidates = []
    
    for i, (timestamp, bar) in enumerate(market_5m.head(20).iterrows()):
        time_str = timestamp.strftime('%H:%M')
        is_new_high = bar['high'] > running_high
        is_hod = bar['high'] == hod
        
        comments = []
        if is_new_high:
            comments.append("NEW HIGH")
            running_high = bar['high']
            break_candidates.append({
                'time': timestamp,
                'high': bar['high'],
                'low': bar['low'],
                'is_hod': is_hod
            })
        if is_hod:
            comments.append("HOD")
        
        comment_str = " | ".join(comments) if comments else ""
        
        print(f"{time_str:<8} {bar['open']:<6.2f} {bar['high']:<6.2f} {bar['low']:<6.2f} {bar['close']:<6.2f} {comment_str}")
    
    print(f"\n🔍 IDENTIFIED 5M BAR BREAK CANDIDATES:")
    print(f"   (Bars that set new session highs)")
    
    for i, candidate in enumerate(break_candidates[:5], 1):  # Show first 5
        time_str = candidate['time'].strftime('%H:%M')
        hod_tag = " (HOD)" if candidate['is_hod'] else ""
        print(f"   {i}. {time_str}: High ${candidate['high']:.2f}, Low ${candidate['low']:.2f}{hod_tag}")
    
    # Now look for post-HOD bars
    hod_time = None
    for timestamp, bar in market_5m.iterrows():
        if bar['high'] == hod:
            hod_time = timestamp
            break
    
    if hod_time:
        print(f"\n📊 HIGH OF DAY: ${hod:.2f} at {hod_time.strftime('%H:%M')}")
        
        # Show bars after HOD
        post_hod_bars = market_5m[market_5m.index > hod_time].head(10)
        
        print(f"\n📋 5-MINUTE BARS AFTER HOD:")
        print(f"{'Time':<8} {'Open':<6} {'High':<6} {'Low':<6} {'Close':<6} {'Comments'}")
        print("-" * 60)
        
        prev_bar = None
        for timestamp, bar in post_hod_bars.iterrows():
            time_str = timestamp.strftime('%H:%M')
            
            comments = []
            if prev_bar is not None:
                if bar['low'] < prev_bar['low']:
                    comments.append("BREAKS BELOW PREV LOW")
            
            comment_str = " | ".join(comments) if comments else ""
            print(f"{time_str:<8} {bar['open']:<6.2f} {bar['high']:<6.2f} {bar['low']:<6.2f} {bar['close']:<6.2f} {comment_str}")
            
            prev_bar = bar
    
    print(f"\n🔑 QUESTION: Which of these 5m bars represents the actual break points?")
    print(f"   Please identify the correct 5m bar breaks from this chart data.")
    
    return market_5m, break_candidates

if __name__ == '__main__':
    df_5m, candidates = analyze_5m_chart()