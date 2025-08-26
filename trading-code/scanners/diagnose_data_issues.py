#!/usr/bin/env python3
"""
Diagnose data coverage and pre-market data issues
"""

import pandas as pd
import asyncio
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(__file__))
from os_d1_pure_scanner import OS_D1_Scanner

async def diagnose_data_coverage():
    """Check data coverage and pre-market data availability"""
    
    print("🔍 DIAGNOSING DATA COVERAGE ISSUES")
    print("=" * 60)
    
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    scanner = OS_D1_Scanner(api_key)
    
    # Test a date we know should have setups
    test_date = '2025-01-08'  # CURR was on this date
    
    print(f"📅 Testing data for {test_date}")
    
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            print("📡 Fetching grouped daily data...")
            current_data = await scanner.fetch_grouped_daily_data(session, test_date)
            
        print(f"📊 Total tickers returned: {len(current_data):,}")
        
        # Check data structure
        print(f"\n🔍 Data Structure Analysis:")
        print(f"Columns available: {list(current_data.columns)}")
        
        # Check for pre-market data fields
        pm_fields = ['pm_high', 'pm_vol', 'pm_open', 'pm_close']
        available_pm_fields = [field for field in pm_fields if field in current_data.columns]
        
        print(f"Pre-market fields found: {available_pm_fields}")
        
        if not available_pm_fields:
            print("⚠️ NO PRE-MARKET FIELDS FOUND!")
            print("   This explains why we're missing setups - using regular high/volume instead")
        
        # Check specific ticker that should be there
        curr_data = current_data[current_data['ticker'] == 'CURR']
        if not curr_data.empty:
            print(f"\n✅ Found CURR data:")
            row = curr_data.iloc[0]
            print(f"   Open: ${row['o']:.2f}")
            print(f"   High: ${row['h']:.2f}")
            print(f"   Close: ${row['c']:.2f}")
            print(f"   Volume: {row['v']:,}")
            
            # Check for PM fields
            for field in pm_fields:
                if field in row and pd.notna(row[field]):
                    print(f"   {field}: {row[field]}")
                else:
                    print(f"   {field}: NOT AVAILABLE")
        else:
            print(f"❌ CURR not found in data for {test_date}")
            
        # Check volume distribution
        print(f"\n📊 Volume Distribution:")
        high_vol_count = (current_data['v'] >= 5000000).sum()
        print(f"   Tickers with volume ≥ 5M: {high_vol_count:,}")
        print(f"   Percentage: {high_vol_count/len(current_data):.1%}")
        
        # Check price distribution  
        price_ranges = [
            (0.75, 5, "Low ($0.75-$5)"),
            (5, 20, "Mid ($5-$20)"), 
            (20, 100, "High ($20-$100)"),
            (100, float('inf'), "Very High (>$100)")
        ]
        
        print(f"\n📊 Price Distribution:")
        for min_price, max_price, label in price_ranges:
            count = ((current_data['c'] >= min_price) & (current_data['c'] < max_price)).sum()
            print(f"   {label}: {count:,} tickers")
        
        # Sample some high-volume, gapping stocks
        print(f"\n🔍 Sample High-Volume Gapping Stocks:")
        
        # Get previous day data
        prev_date = scanner.get_previous_trading_day(test_date, 1).strftime('%Y-%m-%d')
        async with aiohttp.ClientSession() as session:
            prev_data = await scanner.fetch_grouped_daily_data(session, prev_date)
        
        # Calculate gaps for stocks with high volume
        high_vol_stocks = current_data[current_data['v'] >= 10000000].copy()
        
        results = []
        for _, row in high_vol_stocks.head(20).iterrows():  # Check top 20
            ticker = row['ticker']
            prev_row = prev_data[prev_data['ticker'] == ticker]
            
            if not prev_row.empty:
                prev_close = prev_row.iloc[0]['c']
                gap = row['o'] - prev_close
                gap_pct = (gap / prev_close) * 100 if prev_close > 0 else 0
                
                if abs(gap_pct) >= 10:  # Significant gap
                    results.append({
                        'ticker': ticker,
                        'gap_pct': gap_pct,
                        'gap_amount': gap,
                        'volume': row['v'],
                        'close': row['c']
                    })
        
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('gap_pct', ascending=False)
            print(f"Found {len(results_df)} stocks with significant gaps:")
            for _, row in results_df.head(10).iterrows():
                print(f"   {row['ticker']}: Gap {row['gap_pct']:+.1f}% (${row['gap_amount']:+.2f}), Vol {row['volume']:,.0f}")
        else:
            print("   No significant gaps found in sample")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def check_pre_market_data_source():
    """Check if we can get pre-market data from a different API endpoint"""
    
    print(f"\n🔍 CHECKING ALTERNATIVE PRE-MARKET DATA SOURCES")
    print("=" * 60)
    
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    
    # Try different Polygon endpoints for pre-market data
    test_ticker = 'CURR'
    test_date = '2025-01-08'
    
    import requests
    
    # Try aggregates endpoint with pre-market
    print(f"📡 Testing pre-market data for {test_ticker} on {test_date}")
    
    # Regular daily aggregates
    url = f'https://api.polygon.io/v2/aggs/ticker/{test_ticker}/range/1/day/{test_date}/{test_date}'
    params = {'adjusted': 'true', 'apiKey': api_key}
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'results' in data and data['results']:
            result = data['results'][0]
            print(f"✅ Daily data found:")
            print(f"   Open: ${result['o']:.2f}")
            print(f"   High: ${result['h']:.2f}")
            print(f"   Volume: {result['v']:,}")
    
    # Try pre-market specific endpoint
    # Note: This might not exist or require special permissions
    try:
        pm_url = f'https://api.polygon.io/v1/open-close/{test_ticker}/{test_date}'
        pm_params = {'apiKey': api_key}
        
        pm_response = requests.get(pm_url, params=pm_params)
        if pm_response.status_code == 200:
            pm_data = pm_response.json()
            print(f"✅ Pre-market data found:")
            print(f"   Pre-market data: {pm_data}")
        else:
            print(f"⚠️ Pre-market endpoint returned: {pm_response.status_code}")
    except Exception as e:
        print(f"⚠️ Pre-market endpoint error: {e}")
    
    # Check if we can get more detailed data
    try:
        trades_url = f'https://api.polygon.io/v3/trades/{test_ticker}'
        trades_params = {
            'timestamp.gte': f'{test_date}T04:00:00.000Z',  # 4 AM
            'timestamp.lt': f'{test_date}T09:30:00.000Z',   # Market open
            'limit': 1,
            'apiKey': api_key
        }
        
        trades_response = requests.get(trades_url, params=trades_params)
        if trades_response.status_code == 200:
            trades_data = trades_response.json()
            if 'results' in trades_data and trades_data['results']:
                print(f"✅ Pre-market trades found: {len(trades_data['results'])} trades")
            else:
                print(f"⚠️ No pre-market trades found")
        else:
            print(f"⚠️ Trades endpoint returned: {trades_response.status_code}")
    except Exception as e:
        print(f"⚠️ Trades endpoint error: {e}")

if __name__ == '__main__':
    asyncio.run(diagnose_data_coverage())
    asyncio.run(check_pre_market_data_source())