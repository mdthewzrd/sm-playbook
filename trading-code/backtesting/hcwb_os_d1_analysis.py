#!/usr/bin/env python3
"""
HCWB OS D1 Analysis - 5-minute bar break stop and extension pattern
Shows the first loss followed by re-entry on extension
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

def resample_to_5min(df):
    """Resample to 5-minute bars"""
    return df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

def analyze_hcwb_os_d1():
    """Analyze HCWB 5-minute bar break and extension pattern"""
    
    ticker = 'HCWB'
    date = '2025-08-25'
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    
    print(f"📊 HCWB OS D1 - 5min Bar Break & Extension Analysis")
    print(f"   Date: {date}")
    
    # Fetch 1-minute data
    df_1m = fetch_intraday_data(ticker, date, api_key)
    if df_1m.empty:
        print("❌ No data available")
        return
    
    # Create 5-minute bars
    df_5m = resample_to_5min(df_1m)
    
    # Get previous close (Friday before)
    prev_close = 3.45  # From earlier analysis
    
    # Pre-market analysis
    pm_data = df_1m[df_1m.index.time < time(9, 30)]
    pm_high = pm_data['high'].max()
    
    # Market session data
    market_data_1m = df_1m[df_1m.index.time >= time(9, 30)]
    market_data_5m = df_5m[df_5m.index.time >= time(9, 30)]
    
    print(f"✅ Data loaded - {len(df_1m)} 1m bars, {len(df_5m)} 5m bars")
    print(f"   PM High: ${pm_high:.2f}")
    print(f"   Previous Close: ${prev_close:.2f}")
    
    # OS D1 SHORT process simulation
    print(f"\n⚡ OS D1 SHORT PROCESS - First Loss, Then Extension:")
    
    # Initial SHORT entry simulation
    entry_time_1 = "09:35:00"
    entry_price_1 = 5.30  # Near market open
    stop_price_1 = pm_high + 0.05  # 5c over PM high
    
    print(f"\n🔸 FIRST ENTRY (Initial SHORT):")
    print(f"   Entry: {entry_time_1} @ ${entry_price_1:.2f}")
    print(f"   Stop: ${stop_price_1:.2f} (5c over PM high)")
    
    # Find when first stop gets hit
    entry_time_dt_1 = pd.to_datetime(f"2025-08-25 {entry_time_1}").tz_localize('US/Eastern')
    future_data_1 = market_data_1m[market_data_1m.index > entry_time_dt_1]
    
    first_stop_hit = False
    first_stop_time = None
    
    for idx, row in future_data_1.iterrows():
        if row['high'] >= stop_price_1:
            first_stop_hit = True
            first_stop_time = idx
            print(f"   🔴 FIRST STOP HIT at {idx.strftime('%H:%M')} - High: ${row['high']:.2f}")
            pnl_1 = entry_price_1 - stop_price_1  # SHORT P&L
            print(f"   📉 Loss: ${pnl_1:+.3f}/share (-1R)")
            break
    
    if not first_stop_hit:
        print(f"   ✅ First stop NOT hit")
        return
    
    # Now look for the extension entry AFTER the first loss
    print(f"\n🔸 EXTENSION ENTRY (After first loss):")
    
    # Find 5-minute bar break after first stop
    extension_data_5m = market_data_5m[market_data_5m.index > first_stop_time]
    
    if len(extension_data_5m) < 3:
        print(f"   ❌ Not enough data for extension analysis")
        return
    
    # Look for 5-minute bar that breaks above previous high
    extension_entry = None
    extension_time = None
    
    # Check for bar that extends above the first stop level
    for i in range(len(extension_data_5m)):
        bar = extension_data_5m.iloc[i]
        
        # If this 5m bar breaks above our stop level significantly
        if bar['high'] > stop_price_1 + 0.20:  # Extension above first stop
            extension_entry = bar['high'] - 0.10  # Enter slightly below the high
            extension_time = bar.name
            print(f"   🟡 EXTENSION SPOTTED at {extension_time.strftime('%H:%M')}")
            print(f"   📈 5m Bar High: ${bar['high']:.2f} (extension above ${stop_price_1:.2f})")
            print(f"   🎯 RE-ENTRY: ${extension_entry:.2f}")
            break
    
    if extension_entry is None:
        print(f"   ❌ No clear extension pattern found")
        return
    
    # Simulate re-entry SHORT trade on extension
    stop_price_2 = extension_entry + 0.30  # Stop above extension high
    
    print(f"\n🔸 RE-ENTRY SHORT (On Extension):")
    print(f"   Entry: {extension_time.strftime('%H:%M')} @ ${extension_entry:.2f}")
    print(f"   Stop: ${stop_price_2:.2f} (above extension high)")
    
    # Check if second stop gets hit
    future_data_2 = market_data_1m[market_data_1m.index > extension_time]
    
    second_stop_hit = False
    exit_price_2 = None
    exit_time_2 = None
    
    for idx, row in future_data_2.iterrows():
        if row['high'] >= stop_price_2:
            second_stop_hit = True
            exit_price_2 = stop_price_2
            exit_time_2 = idx
            print(f"   🔴 SECOND STOP HIT at {idx.strftime('%H:%M')} - High: ${row['high']:.2f}")
            break
    
    # If no second stop, look for profit exit
    if not second_stop_hit:
        print(f"   ✅ Second stop NOT hit - position held")
        
        # Look for profitable exit (price decline for SHORT)
        exit_window = future_data_2.iloc[:60] if len(future_data_2) >= 60 else future_data_2
        
        if not exit_window.empty:
            lowest_bar = exit_window.loc[exit_window['low'].idxmin()]
            exit_price_2 = lowest_bar['low']
            exit_time_2 = lowest_bar.name
            print(f"   🟢 PROFIT EXIT at {exit_time_2.strftime('%H:%M')} @ ${exit_price_2:.2f}")
    
    # Calculate second trade P&L
    if exit_price_2:
        pnl_2 = extension_entry - exit_price_2  # SHORT P&L
        status_2 = "🔴 STOPPED" if second_stop_hit else "🟢 PROFIT"
        
        print(f"   Second Trade: ${pnl_2:+.3f}/share {status_2}")
    
    # Now look for SECOND 5-minute bar break (the one that works)
    print(f"\n🔸 SECOND 5-MINUTE BAR BREAK (After 2 starters stopped):")
    
    # Continue looking for another 5m bar break after second stop
    if second_stop_hit and exit_time_2:
        third_entry_data_5m = market_data_5m[market_data_5m.index > exit_time_2]
        
        # Look for another significant 5m bar break
        third_entry = None
        third_time = None
        
        # Look for 5m bar that breaks BELOW previous 5m candle's low (SHORT entry signal)
        print(f"   🔍 Looking for 5m bar break BELOW previous candle low...")
        
        for i in range(1, len(third_entry_data_5m)):  # Start from 1 to compare with previous
            current_bar = third_entry_data_5m.iloc[i]
            previous_bar = third_entry_data_5m.iloc[i-1]
            
            # 5m bar break: current bar's low breaks below previous bar's low
            if current_bar['low'] < previous_bar['low']:
                third_entry = previous_bar['low'] - 0.01  # Enter on break below previous low
                third_time = current_bar.name
                print(f"   🟡 5M BAR BREAK BELOW at {third_time.strftime('%H:%M')}")
                print(f"   📉 Previous 5m Low: ${previous_bar['low']:.2f}")
                print(f"   📉 Current 5m Low: ${current_bar['low']:.2f} (breaks below)")
                print(f"   🎯 SHORT ENTRY: ${third_entry:.2f} (on break)")
                break
        
        if third_entry:
            stop_price_3 = third_entry + 0.25  # Stop above third entry
            
            print(f"   Entry: {third_time.strftime('%H:%M')} @ ${third_entry:.2f}")
            print(f"   Stop: ${stop_price_3:.2f}")
            
            # Check third trade outcome
            future_data_3 = market_data_1m[market_data_1m.index > third_time]
            
            third_stop_hit = False
            exit_price_3 = None
            exit_time_3 = None
            
            for idx, row in future_data_3.iterrows():
                if row['high'] >= stop_price_3:
                    third_stop_hit = True
                    exit_price_3 = stop_price_3
                    exit_time_3 = idx
                    print(f"   🔴 THIRD STOP HIT at {idx.strftime('%H:%M')} - High: ${row['high']:.2f}")
                    break
            
            # If no third stop, look for profit
            if not third_stop_hit:
                print(f"   ✅ Third stop NOT hit - position held")
                
                # Look for profitable exit
                exit_window_3 = future_data_3.iloc[:120] if len(future_data_3) >= 120 else future_data_3
                
                if not exit_window_3.empty:
                    lowest_bar_3 = exit_window_3.loc[exit_window_3['low'].idxmin()]
                    exit_price_3 = lowest_bar_3['low']
                    exit_time_3 = lowest_bar_3.name
                    print(f"   🟢 PROFIT EXIT at {exit_time_3.strftime('%H:%M')} @ ${exit_price_3:.2f}")
            
            # Calculate third trade P&L
            if exit_price_3:
                pnl_3 = third_entry - exit_price_3  # SHORT P&L
                status_3 = "🔴 STOPPED" if third_stop_hit else "🟢 PROFIT"
                
                print(f"\n💰 COMPLETE OS D1 RESULTS:")
                print(f"   Starter 1: ${pnl_1:+.3f}/share (-1R) 🔴")
                print(f"   Starter 2: ${pnl_2:+.3f}/share 🔴") 
                print(f"   5m Break 2: ${pnl_3:+.3f}/share {status_3}")
                print(f"   TOTAL P&L: ${(pnl_1 + pnl_2 + pnl_3):+.3f}/share")
                
                print(f"\n🎯 PARTIAL OS D1 RESULTS:")
                print(f"   ✓ Both starters stopped (-1R each)")
                print(f"   ✓ First 5m bar break stopped") 
                print(f"   ✓ Second 5m bar break: {'STOPPED' if third_stop_hit else 'WORKED!'}")
                
                # If third trade also stopped, look for more 5m bar breaks
                if third_stop_hit:
                    print(f"\n🔸 CONTINUING 5M BAR BREAK SEARCH (After 3 stops)...")
                    
                    # Look for additional 5m bar breaks
                    remaining_data_5m = market_data_5m[market_data_5m.index > exit_time_3]
                    
                    for i in range(1, len(remaining_data_5m)):
                        current_bar = remaining_data_5m.iloc[i]
                        previous_bar = remaining_data_5m.iloc[i-1]
                        
                        # Look for 5m bar break below previous low
                        if current_bar['low'] < previous_bar['low']:
                            fourth_entry = previous_bar['low'] - 0.01
                            fourth_time = current_bar.name
                            
                            print(f"   🟡 ANOTHER 5M BAR BREAK at {fourth_time.strftime('%H:%M')}")
                            print(f"   📉 Previous 5m Low: ${previous_bar['low']:.2f}")
                            print(f"   📉 Current 5m Low: ${current_bar['low']:.2f}")
                            print(f"   🎯 FOURTH ENTRY: ${fourth_entry:.2f}")
                            
                            # Check if this one works
                            stop_price_4 = fourth_entry + 0.25
                            future_data_4 = market_data_1m[market_data_1m.index > fourth_time]
                            
                            fourth_stop_hit = False
                            exit_price_4 = None
                            
                            for idx, row in future_data_4.iterrows():
                                if row['high'] >= stop_price_4:
                                    fourth_stop_hit = True
                                    exit_price_4 = stop_price_4
                                    print(f"   🔴 FOURTH STOP HIT at {idx.strftime('%H:%M')}")
                                    break
                            
                            if not fourth_stop_hit:
                                # Look for profit exit
                                exit_window_4 = future_data_4.iloc[:120] if len(future_data_4) >= 120 else future_data_4
                                
                                if not exit_window_4.empty:
                                    lowest_bar_4 = exit_window_4.loc[exit_window_4['low'].idxmin()]
                                    exit_price_4 = lowest_bar_4['low']
                                    exit_time_4 = lowest_bar_4.name
                                    print(f"   🟢 WINNING EXIT at {exit_time_4.strftime('%H:%M')} @ ${exit_price_4:.2f}")
                                    
                                    pnl_4 = fourth_entry - exit_price_4
                                    total_pnl_all = pnl_1 + pnl_2 + pnl_3 + pnl_4
                                    
                                    print(f"\n💰 FINAL COMPLETE RESULTS:")
                                    print(f"   Starter 1: ${pnl_1:+.3f}/share")
                                    print(f"   Starter 2: ${pnl_2:+.3f}/share")
                                    print(f"   5m Break 1: ${pnl_3:+.3f}/share")
                                    print(f"   5m Break 2: ${pnl_4:+.3f}/share ✅ WINNER")
                                    print(f"   TOTAL: ${total_pnl_all:+.3f}/share")
                                    
                                    return
                            
                            break  # Only check first valid break
                
                return
    
    print(f"\n💰 PARTIAL RESULTS (2 trades):")
    print(f"   First Trade: ${pnl_1:+.3f}/share (-1R)")
    if exit_price_2:
        print(f"   Second Trade: ${pnl_2:+.3f}/share {status_2}")
        print(f"   Combined P&L: ${(pnl_1 + pnl_2):+.3f}/share")

if __name__ == '__main__':
    analyze_hcwb_os_d1()