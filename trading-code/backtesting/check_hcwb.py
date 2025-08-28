#!/usr/bin/env python3
"""
Check HCWB data for 8/25/2024 - OS D1 analysis
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, time
import matplotlib.pyplot as plt

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

def get_previous_day_close(ticker, date, api_key):
    """Get previous trading day close"""
    prev_date = pd.to_datetime(date) - pd.Timedelta(days=1)
    while prev_date.weekday() > 4:  # Skip weekends
        prev_date -= pd.Timedelta(days=1)
    
    prev_date_str = prev_date.strftime('%Y-%m-%d')
    
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{prev_date_str}/{prev_date_str}'
    params = {
        'adjusted': 'true',
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                return data['results'][0]['c']
    except Exception as e:
        print(f"Error getting previous close: {e}")
    
    return None

def analyze_hcwb():
    """Analyze HCWB for OS D1 setup on 8/25/2024"""
    
    ticker = 'HCWB'
    date = '2025-08-25'  # 8/25/2025
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    
    print(f"📊 Analyzing {ticker} for OS D1 Setup on {date}")
    
    # Fetch data
    df = fetch_intraday_data(ticker, date, api_key)
    if df.empty:
        print("❌ No intraday data available")
        return
    
    # Get previous close
    prev_close = get_previous_day_close(ticker, date, api_key)
    if prev_close is None:
        print("❌ Could not get previous close")
        return
    
    print(f"✅ Got {len(df)} data points")
    print(f"   Time range: {df.index[0].strftime('%H:%M')} - {df.index[-1].strftime('%H:%M')} ET")
    print(f"   Previous close: ${prev_close:.2f}")
    
    # Analyze pre-market data
    pm_data = df[df.index.time < time(9, 30)]
    if pm_data.empty:
        print("❌ No pre-market data")
        return
    
    pm_high = pm_data['high'].max()
    pm_low = pm_data['low'].min()
    
    # Market open data
    market_data = df[df.index.time >= time(9, 30)]
    if market_data.empty:
        print("❌ No market data")
        return
    
    market_open = market_data.iloc[0]['close']
    
    # Calculate key metrics
    gap_pct = ((market_open - prev_close) / prev_close) * 100
    pm_high_pct = ((pm_high - prev_close) / prev_close) * 100
    
    print(f"\n🎯 HCWB OS D1 Analysis:")
    print(f"   Previous Close: ${prev_close:.2f}")
    print(f"   PM High: ${pm_high:.2f} ({pm_high_pct:+.1f}%)")
    print(f"   PM Low: ${pm_low:.2f}")
    print(f"   Market Open: ${market_open:.2f}")
    print(f"   Gap: {gap_pct:+.1f}%")
    
    # Check OS D1 qualification
    print(f"\n📋 OS D1 QUALIFICATION CHECK:")
    gap_qualified = gap_pct >= 50
    
    print(f"   Gap ≥50%: {gap_qualified} ({gap_pct:+.1f}%)")
    print(f"   PM High: {pm_high_pct:+.1f}% (no minimum requirement)")
    
    if gap_qualified:
        print(f"✅ {ticker} QUALIFIES for OS D1 setup!")
        
        # Show price action through day
        high_of_day = df['high'].max()
        low_of_day = df['low'].min()
        close_price = df.iloc[-1]['close']
        
        print(f"\n📈 PRICE ACTION:")
        print(f"   High of Day: ${high_of_day:.2f}")
        print(f"   Low of Day: ${low_of_day:.2f}")
        print(f"   Close: ${close_price:.2f}")
        
        # Check if it failed PM high (SHORT setup)
        if close_price < pm_high:
            print(f"🔥 POTENTIAL SHORT SETUP - Failed to hold PM high")
        
    else:
        print(f"❌ {ticker} does NOT qualify for OS D1 setup")
    
    return df, prev_close, pm_high, market_open

if __name__ == '__main__':
    analyze_hcwb()