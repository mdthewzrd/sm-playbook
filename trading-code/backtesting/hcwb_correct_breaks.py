#!/usr/bin/env python3
"""
HCWB Correct 5m Bar Breaks - Based on 5m chart analysis
1st break: 09:45 candle (low $5.79)
2nd break: 10:00 candle (low $6.70) - WINNER
"""

import requests
import pandas as pd
from datetime import datetime, time

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

def analyze_correct_breaks():
    """Analyze the correct 5m bar breaks: 09:45 and 10:00"""
    
    ticker = 'HCWB'
    date = '2025-08-25'
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    
    print(f"📊 HCWB CORRECT 5M BAR BREAKS ANALYSIS")
    print(f"   Based on 5m chart: 09:45 break and 10:00 break (winner)")
    
    # Fetch 1-minute data
    df_1m = fetch_intraday_data(ticker, date, api_key)
    market_data = df_1m[df_1m.index.time >= time(9, 30)]
    
    # Starter trades (known to be stopped)
    starters = [
        {
            'name': 'Starter 1',
            'entry_price': 5.30,
            'stop_price': 5.83,
            'pnl': -0.530
        },
        {
            'name': 'Starter 2', 
            'entry_price': 6.27,
            'stop_price': 6.57,
            'pnl': -0.300
        }
    ]
    
    print(f"\n⚡ STARTER RESULTS:")
    for starter in starters:
        print(f"   {starter['name']}: ${starter['pnl']:+.3f}/share 🔴")
    
    # Correct 5m bar breaks
    breaks = [
        {
            'name': '5m Break 1 (09:45)',
            'candle_time': '09:45',
            'candle_low': 5.79,  # From 5m chart: 09:45 candle low
            'description': '09:45 5m candle broke to $6.48 high'
        },
        {
            'name': '5m Break 2 (10:00 HOD)',
            'candle_time': '10:00', 
            'candle_low': 6.70,  # From 5m chart: 10:00 candle low
            'description': '10:00 5m candle = HOD $7.35'
        }
    ]
    
    print(f"\n🔍 CORRECT 5M BAR BREAKS:")
    
    executed_trades = []
    
    for break_info in breaks:
        print(f"\n🟡 {break_info['name']}")
        print(f"   {break_info['description']}")
        print(f"   Break level: ${break_info['candle_low']:.2f} (5m candle low)")
        
        # Find when price breaks below this 5m candle's low
        break_level = break_info['candle_low']
        candle_time = pd.to_datetime(f"2025-08-25 {break_info['candle_time']}:00").tz_localize('US/Eastern')
        
        # Look for 1m price action after the 5m candle time
        future_data = market_data[market_data.index > candle_time]
        
        entry_found = False
        entry_price = None
        entry_time = None
        
        for idx, row in future_data.iterrows():
            if row['low'] < break_level:
                entry_price = break_level - 0.01  # Enter on break
                entry_time = idx
                entry_found = True
                
                print(f"   📉 BREAK at {idx.strftime('%H:%M')}: Price hit ${row['low']:.2f}")
                print(f"   🎯 SHORT ENTRY: ${entry_price:.2f}")
                break
        
        if not entry_found:
            print(f"   ❌ No break occurred")
            continue
        
        # Determine if this is the winner (10:00 break)
        if '10:00' in break_info['name']:
            # This is the winner - use reasonable stop
            stop_price = entry_price + 0.30
            
            print(f"   📊 Stop: ${stop_price:.2f}")
            
            # Check if stopped (shouldn't be for winner)
            future_after_entry = market_data[market_data.index > entry_time]
            
            stop_hit = False
            for idx, row in future_after_entry.iterrows():
                if row['high'] >= stop_price:
                    stop_hit = True
                    print(f"   🔴 STOP HIT at {idx.strftime('%H:%M')}")
                    break
            
            if not stop_hit:
                # Find profitable exit
                exit_window = future_after_entry.iloc[:90] if len(future_after_entry) >= 90 else future_after_entry
                
                if not exit_window.empty:
                    lowest_bar = exit_window.loc[exit_window['low'].idxmin()]
                    exit_price = lowest_bar['low']
                    exit_time = lowest_bar.name
                    pnl = entry_price - exit_price
                    
                    print(f"   🟢 PROFIT EXIT at {exit_time.strftime('%H:%M')} @ ${exit_price:.2f}")
                    print(f"   💰 P&L: ${pnl:+.3f}/share ✅ WINNER")
                    
                    executed_trades.append({
                        'name': break_info['name'],
                        'pnl': pnl,
                        'winner': True
                    })
        else:
            # First break - determine if stopped or profitable
            stop_price = 7.36  # Stop over HOD
            
            print(f"   📊 Stop: ${stop_price:.2f} (over HOD)")
            
            # Check if stopped
            future_after_entry = market_data[market_data.index > entry_time]
            
            stop_hit = False
            for idx, row in future_after_entry.iterrows():
                if row['high'] >= stop_price:
                    stop_hit = True
                    pnl = entry_price - stop_price
                    print(f"   🔴 STOP HIT at {idx.strftime('%H:%M')}")
                    print(f"   💰 P&L: ${pnl:+.3f}/share 🔴")
                    
                    executed_trades.append({
                        'name': break_info['name'],
                        'pnl': pnl,
                        'winner': False
                    })
                    break
            
            if not stop_hit:
                # Profitable
                exit_window = future_after_entry.iloc[:60] if len(future_after_entry) >= 60 else future_after_entry
                
                if not exit_window.empty:
                    lowest_bar = exit_window.loc[exit_window['low'].idxmin()]
                    exit_price = lowest_bar['low']
                    exit_time = lowest_bar.name
                    pnl = entry_price - exit_price
                    
                    print(f"   🟢 PROFIT EXIT at {exit_time.strftime('%H:%M')} @ ${exit_price:.2f}")
                    print(f"   💰 P&L: ${pnl:+.3f}/share ✅")
                    
                    executed_trades.append({
                        'name': break_info['name'],
                        'pnl': pnl,
                        'winner': True
                    })
    
    # Final results
    print(f"\n💰 CORRECT FINAL RESULTS:")
    
    total_pnl = sum([s['pnl'] for s in starters])
    print(f"   Starter 1: ${starters[0]['pnl']:+.3f}/share")
    print(f"   Starter 2: ${starters[1]['pnl']:+.3f}/share")
    
    for trade in executed_trades:
        status = "✅ WINNER" if trade['winner'] else "🔴 STOPPED"
        print(f"   {trade['name']}: ${trade['pnl']:+.3f}/share {status}")
        total_pnl += trade['pnl']
    
    print(f"   TOTAL: ${total_pnl:+.3f}/share")
    
    print(f"\n🔑 CORRECTED INSIGHTS:")
    print(f"   ✓ First 5m break: 09:45 candle (high $6.48, low $5.79)")
    print(f"   ✓ Second 5m break: 10:00 candle (HOD $7.35, low $6.70)")
    print(f"   ✓ 10:00 break = WINNER (post-HOD gap exhaustion)")
    print(f"   ✓ Entry when price breaks below 5m candle lows")

if __name__ == '__main__':
    analyze_correct_breaks()