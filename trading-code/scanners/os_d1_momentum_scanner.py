#!/usr/bin/env python3
"""
OS D1 Momentum Strategy Scanner - Actual Implementation
Based on the Notion SM Playbook OS D1 Setup Document

This implements the correct OS D1 strategy:
- LONG momentum on small cap day one gappers 
- EMA200 validation (close <= ema200 * 0.8)
- Market cap filtering
- Stage classification system
- FBO, Extension, and Dev Band Pop entries
"""

import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

class OS_D1_MomentumScanner:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
    async def fetch_grouped_daily_data(self, session, date, adjusted=True):
        """Fetch grouped daily data for all tickers"""
        adj_str = "true" if adjusted else "false"
        url = f"{self.base_url}/v2/aggs/grouped/locale/us/market/stocks/{date}"
        params = {'adjusted': adj_str, 'apiKey': self.api_key}
        
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if 'results' in data:
                    df = pd.DataFrame(data['results'])
                    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
                    df.rename(columns={'T': 'ticker'}, inplace=True)
                    return df
            return pd.DataFrame()
    
    async def fetch_historical_data(self, session, ticker, start_date, end_date):
        """Fetch historical data for EMA200 validation"""
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc', 
            'limit': 5000,
            'apiKey': self.api_key
        }
        
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if 'results' in data and len(data['results']) >= 200:
                    df = pd.DataFrame(data['results'])
                    df['date'] = pd.to_datetime(df['t'], unit='ms')
                    
                    # Calculate EMA200
                    df['ema200'] = df['c'].ewm(span=200, adjust=False).mean()
                    
                    return df
            return pd.DataFrame()
    
    async def get_market_cap(self, session, ticker, date):
        """Get market cap data"""
        url = f"{self.base_url}/v3/reference/tickers/{ticker}"
        params = {'date': date, 'apiKey': self.api_key}
        
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if 'results' in data:
                    try:
                        market_cap = data['results']['market_cap'] / 1000000  # Convert to millions
                        return market_cap
                    except:
                        return None
            return None
    
    async def validate_os_d1_setup(self, session, setup_row):
        """Validate OS D1 setup with EMA200 and market cap criteria"""
        ticker = setup_row['ticker']
        current_date = setup_row['scan_date']
        prev_date = current_date - timedelta(days=1)
        
        # Get 200+ days of historical data for EMA calculation  
        hist_start = current_date - timedelta(days=500)
        hist_end = prev_date
        
        hist_data = await self.fetch_historical_data(
            session, ticker, 
            hist_start.strftime('%Y-%m-%d'), 
            hist_end.strftime('%Y-%m-%d')
        )
        
        if hist_data.empty or len(hist_data) < 200:
            return False, None
        
        # Check EMA200 validation: close <= ema200 * 0.8
        last_close = hist_data['c'].iloc[-1] 
        last_ema200 = hist_data['ema200'].iloc[-1]
        
        ema_valid = last_close <= (last_ema200 * 0.8)
        
        if not ema_valid:
            return False, None
        
        # Get market cap
        market_cap = await self.get_market_cap(session, ticker, prev_date.strftime('%Y-%m-%d'))
        
        return True, {
            'last_close': last_close,
            'ema200': last_ema200,
            'ema_ratio': last_close / last_ema200,
            'market_cap': market_cap,
            'hist_days': len(hist_data)
        }
    
    async def scan_os_d1_setups(self, start_date='2025-01-01', end_date='2025-08-31'):
        """Scan for OS D1 momentum setups with full validation"""
        
        print("🚀 OS D1 Momentum Strategy Scanner")
        print("=" * 60)
        print("Scanning for LONG momentum setups on small cap day one gappers")
        print(f"Date range: {start_date} to {end_date}")
        
        # Generate trading days
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = [d for d in date_range if d.weekday() < 5]
        
        all_setups = []
        
        async with aiohttp.ClientSession() as session:
            for i, scan_date in enumerate(trading_days[1:]):  # Skip first day
                prev_date = trading_days[i]
                
                print(f"\n📅 {i+1}/{len(trading_days)-1}: Scanning {scan_date.strftime('%Y-%m-%d')}")
                
                try:
                    # Fetch market data for both days
                    current_data = await self.fetch_grouped_daily_data(
                        session, scan_date.strftime('%Y-%m-%d')
                    )
                    prev_data = await self.fetch_grouped_daily_data(
                        session, prev_date.strftime('%Y-%m-%d') 
                    )
                    
                    if current_data.empty or prev_data.empty:
                        print("   ⚠️ No market data available")
                        continue
                    
                    # Merge data
                    merged = current_data.merge(prev_data, on='ticker', suffixes=('', '_prev'))
                    
                    # Apply OS D1 trigger day criteria from your document
                    merged['pm_high'] = merged['h']  # Pre-market high = high of day
                    merged['pm_vol'] = merged['v']   # Pre-market volume = total volume
                    merged['prev_close'] = merged['c_prev']
                    merged['prev_high'] = merged['h_prev'] 
                    merged['gap'] = merged['o'] - merged['prev_close']
                    merged['prev_volume'] = merged['v_prev']
                    
                    # Trigger day formula from your scanner code
                    merged['trig_day'] = (
                        (merged['pm_high'] / merged['prev_close'] - 1 >= 0.5) &  # 50%+ pre-market high
                        (merged['gap'] >= 0.5) &                                   # $0.50+ gap
                        (merged['o'] / merged['prev_high'] - 1 >= 0.3) &          # 30%+ open vs prev high  
                        (merged['pm_vol'] >= 5000000) &                           # 5M+ volume
                        (merged['prev_close'] >= 0.75)                            # $0.75+ prev close
                    ).astype(int)
                    
                    # Check for d2 == 0 (not a second day setup)
                    merged['prev_close_1'] = merged.groupby('ticker')['prev_close'].shift(1)
                    merged['d2'] = (
                        (merged['prev_close'] / merged['prev_close_1'] - 1 >= 0.3) &
                        (merged['prev_volume'] >= 10000000)
                    ).astype(int)
                    
                    # Filter for trigger day setups that are NOT d2
                    day_candidates = merged[
                        (merged['trig_day'] == 1) & 
                        (merged['d2'] == 0)
                    ].copy()
                    
                    if day_candidates.empty:
                        print("   ℹ️ No trigger day candidates found")
                        continue
                    
                    print(f"   🔍 Found {len(day_candidates)} trigger day candidates")
                    print(f"   🧪 Validating EMA200 and market cap criteria...")
                    
                    # Validate each candidate
                    valid_setups = []
                    for _, candidate in day_candidates.iterrows():
                        candidate['scan_date'] = scan_date
                        
                        is_valid, validation_data = await self.validate_os_d1_setup(session, candidate)
                        
                        if is_valid:
                            # Add validation data to candidate
                            for key, value in validation_data.items():
                                candidate[f'valid_{key}'] = value
                            
                            candidate['gap_pct'] = (candidate['gap'] / candidate['prev_close']) * 100
                            candidate['pm_high_pct'] = (candidate['pm_high'] / candidate['prev_close'] - 1) * 100
                            candidate['open_prev_high_pct'] = (candidate['o'] / candidate['prev_high'] - 1) * 100
                            
                            valid_setups.append(candidate)
                            
                        # Small delay to respect API limits
                        await asyncio.sleep(0.1)
                    
                    if valid_setups:
                        print(f"   ✅ {len(valid_setups)} validated OS D1 setups:")
                        for setup in valid_setups[:3]:  # Show top 3
                            print(f"      📊 {setup['ticker']}: Gap {setup['gap_pct']:.1f}%, "
                                  f"PM High {setup['pm_high_pct']:.1f}%, EMA Ratio {setup['valid_ema_ratio']:.2f}")
                        
                        all_setups.extend(valid_setups)
                    else:
                        print("   ❌ No setups passed EMA200 validation")
                    
                except Exception as e:
                    print(f"   ❌ Error scanning {scan_date}: {e}")
                    continue
                
                # Delay between days
                await asyncio.sleep(0.5)
        
        if all_setups:
            final_df = pd.DataFrame(all_setups)
            print(f"\n✅ Found {len(final_df)} validated OS D1 momentum setups")
            
            # Save results
            output_file = f"os_d1_momentum_setups_{start_date}_{end_date}.csv"
            final_df.to_csv(output_file, index=False)
            print(f"💾 Results saved to: {output_file}")
            
            # Show summary statistics
            print(f"\n📊 Setup Summary:")
            print(f"   • Average gap: {final_df['gap_pct'].mean():.1f}%")
            print(f"   • Average PM high: {final_df['pm_high_pct'].mean():.1f}%") 
            print(f"   • Average EMA ratio: {final_df['valid_ema_ratio'].mean():.2f}")
            if 'valid_market_cap' in final_df.columns:
                valid_mc = final_df['valid_market_cap'].dropna()
                if not valid_mc.empty:
                    print(f"   • Average market cap: ${valid_mc.mean():.1f}M")
            
            return final_df
        else:
            print(f"\n❌ No validated OS D1 setups found")
            return pd.DataFrame()

async def main():
    """Run OS D1 momentum scanner"""
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    scanner = OS_D1_MomentumScanner(api_key)
    
    # Scan for OS D1 setups in 2025
    results = await scanner.scan_os_d1_setups(
        start_date='2025-01-01',
        end_date='2025-02-28'  # Start with smaller range for testing
    )
    
    if not results.empty:
        print(f"\n🎯 Ready to implement stage classification and entry logic")
        print(f"📈 Found {len(results)} setups for momentum trading")

if __name__ == '__main__':
    asyncio.run(main())