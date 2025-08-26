#!/usr/bin/env python3
"""
Debug script to check why specific tickers from your list are not being detected
"""

import pandas as pd
import asyncio
from datetime import datetime
import sys
import os

# Add the scanner directory to the path
sys.path.append(os.path.dirname(__file__))

from os_d1_pure_scanner import OS_D1_Scanner

async def debug_specific_setup(ticker, date):
    """Debug a specific ticker/date combination"""
    
    print(f"\n🔍 DEBUGGING: {ticker} on {date}")
    print("=" * 50)
    
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    scanner = OS_D1_Scanner(api_key)
    
    try:
        # Get previous trading day
        prev_date = scanner.get_previous_trading_day(date, 1)
        prev_date_str = prev_date.strftime('%Y-%m-%d')
        
        print(f"📅 Scan Date: {date}")
        print(f"📅 Previous Date: {prev_date_str}")
        
        # Fetch market data for both days
        import aiohttp
        async with aiohttp.ClientSession() as session:
            current_data = await scanner.fetch_grouped_daily_data(session, date)
            prev_data = await scanner.fetch_grouped_daily_data(session, prev_date_str)
        
        # Check if ticker exists in the data
        current_ticker_data = current_data[current_data['ticker'] == ticker]
        prev_ticker_data = prev_data[prev_data['ticker'] == ticker]
        
        if current_ticker_data.empty:
            print(f"❌ {ticker} not found in current day data ({len(current_data)} tickers total)")
            return
            
        if prev_ticker_data.empty:
            print(f"❌ {ticker} not found in previous day data ({len(prev_data)} tickers total)")
            return
        
        print(f"✅ Found {ticker} in both days' data")
        
        # Get the specific data
        current_row = current_ticker_data.iloc[0]
        prev_row = prev_ticker_data.iloc[0]
        
        print(f"\n📊 Current Day Data:")
        print(f"   Open: ${current_row['o']:.2f}")
        print(f"   High: ${current_row['h']:.2f}")
        print(f"   Low: ${current_row['l']:.2f}")
        print(f"   Close: ${current_row['c']:.2f}")
        print(f"   Volume: {current_row['v']:,}")
        
        print(f"\n📊 Previous Day Data:")
        print(f"   Close: ${prev_row['c']:.2f}")
        print(f"   High: ${prev_row['h']:.2f}")
        print(f"   Volume: {prev_row['v']:,}")
        
        # Calculate trigger day criteria manually
        pm_high = current_row.get('pm_high', current_row['h'])
        pm_vol = current_row.get('pm_vol', current_row['v'])
        gap = current_row['o'] - prev_row['c']
        
        pm_high_ratio = pm_high / prev_row['c'] - 1 if prev_row['c'] > 0 else 0
        gap_check = gap >= 0.5
        open_prev_high_ratio = current_row['o'] / prev_row['h'] - 1 if prev_row['h'] > 0 else 0
        pm_vol_check = pm_vol >= 5000000
        prev_close_check = prev_row['c'] >= 0.75
        
        print(f"\n🎯 Trigger Day Criteria Check:")
        print(f"   PM High Ratio: {pm_high_ratio:.1%} (need ≥50%) {'✅' if pm_high_ratio >= 0.5 else '❌'}")
        print(f"   Gap Amount: ${gap:.2f} (need ≥$0.50) {'✅' if gap_check else '❌'}")
        print(f"   Open/Prev High: {open_prev_high_ratio:.1%} (need ≥30%) {'✅' if open_prev_high_ratio >= 0.3 else '❌'}")
        print(f"   PM Volume: {pm_vol:,} (need ≥5M) {'✅' if pm_vol_check else '❌'}")
        print(f"   Prev Close: ${prev_row['c']:.2f} (need ≥$0.75) {'✅' if prev_close_check else '❌'}")
        
        # Check if it passes trigger day
        trig_day = (
            (pm_high_ratio >= 0.5) and
            gap_check and
            (open_prev_high_ratio >= 0.3) and
            pm_vol_check and
            prev_close_check
        )
        
        print(f"\n🎯 Trigger Day Result: {'✅ PASS' if trig_day else '❌ FAIL'}")
        
        if not trig_day:
            print(f"   Reason for failure:")
            if pm_high_ratio < 0.5:
                print(f"   - PM High ratio too low ({pm_high_ratio:.1%} < 50%)")
            if not gap_check:
                print(f"   - Gap too small (${gap:.2f} < $0.50)")
            if open_prev_high_ratio < 0.3:
                print(f"   - Open/prev high too low ({open_prev_high_ratio:.1%} < 30%)")
            if not pm_vol_check:
                print(f"   - Volume too low ({pm_vol:,} < 5M)")
            if not prev_close_check:
                print(f"   - Previous close too low (${prev_row['c']:.2f} < $0.75)")
        else:
            # Check EMA validation
            print(f"\n🧮 Checking EMA Validation...")
            
            # Get historical data for EMA200
            start_date = pd.to_datetime(date) - pd.Timedelta(days=500)
            start_date_str = start_date.strftime('%Y-%m-%d')
            prev_date_str = (pd.to_datetime(date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            daily_data = scanner.fetch_daily_data(ticker, start_date_str, prev_date_str, adjusted="true")
            
            if not daily_data.empty:
                daily_data = scanner.adjust_daily(daily_data)
                
                if len(daily_data) >= 200:
                    current_price = current_row['c']
                    ema200 = daily_data['ema200'].iloc[-1]
                    ema_threshold = ema200 * 0.8
                    ema_valid = current_price <= ema_threshold
                    
                    print(f"   Current Price: ${current_price:.2f}")
                    print(f"   EMA200: ${ema200:.2f}")
                    print(f"   80% of EMA200: ${ema_threshold:.2f}")
                    print(f"   EMA Valid: {'✅ PASS' if ema_valid else '❌ FAIL'}")
                    
                    if not ema_valid:
                        print(f"   Price too high: ${current_price:.2f} > ${ema_threshold:.2f}")
                else:
                    print(f"   ⚠️ Insufficient historical data ({len(daily_data)} < 200 days)")
            else:
                print(f"   ❌ No historical data available")
        
    except Exception as e:
        print(f"❌ Error debugging {ticker}: {e}")
        import traceback
        traceback.print_exc()

async def debug_sample_missing_setups():
    """Debug a sample of missing setups from your list"""
    
    # Sample from your list to debug
    missing_setups = [
        ('BTCM', '2025-07-10'),
        ('SEPN', '2025-05-14'),
        ('LOBO', '2025-05-13'),
        ('CTMX', '2025-05-12'),
        ('PRTG', '2025-03-28'),
        ('FMTO', '2025-02-25'),
        ('SLDB', '2025-02-18'),  # This one I did scan on 2/18 but didn't find
        ('TRIB', '2025-01-28'),
        ('INM', '2025-01-21'),
        ('CURR', '2025-01-08')
    ]
    
    print("🚨 DEBUGGING MISSING OS D1 SETUPS")
    print("=" * 60)
    print(f"Checking {len(missing_setups)} setups that should have been found...")
    
    for ticker, date in missing_setups:
        await debug_specific_setup(ticker, date)
        print("\n" + "-" * 60)

if __name__ == '__main__':
    asyncio.run(debug_sample_missing_setups())