#!/usr/bin/env python3
"""
Complete OS D1 Scanner - Combines LC Framework with Entry/Exit Logic
Uses your exact parameters for trigger day detection and EMA validation
"""

import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
import pandas_market_calendars as mcal
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

class OS_D1_Complete_Scanner:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.nyse = mcal.get_calendar('NYSE')
        
    def get_previous_trading_day(self, current_date, n=1):
        """Get the nth previous trading day"""
        start_date = pd.to_datetime(current_date) - pd.Timedelta(days=500)
        end_date = pd.to_datetime(current_date) + pd.Timedelta(days=30)
        schedule = self.nyse.schedule(start_date=start_date, end_date=end_date)
        trading_days = schedule.index
        
        try:
            idx = trading_days.get_loc(pd.to_datetime(current_date))
            return trading_days[max(0, idx - n)]
        except:
            return pd.to_datetime(current_date) - pd.Timedelta(days=n)
    
    def fetch_daily_data(self, ticker, start_date, end_date, adjusted="true"):
        """Fetch daily data from Polygon API"""
        url = f'{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}'
        params = {
            'adjusted': adjusted,
            'sort': 'asc',
            'limit': 5000,
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    return pd.DataFrame(data['results'])
            else:
                print(f"Error fetching data for {ticker}: {response.status_code}")
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
        
        return pd.DataFrame()
    
    def adjust_daily(self, df):
        """Apply daily adjustments and calculate EMAs"""
        if df.empty:
            return df
            
        df = df.copy()
        df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
        df['pdc'] = df['c'].shift(1)  # Previous day close
        
        # Calculate EMAs
        df['ema200'] = df['c'].ewm(span=200, adjust=False).mean().fillna(0)
        
        return df
    
    def get_market_cap(self, ticker, date):
        """Get market cap data from Polygon API"""
        try:
            mc_url = f'{self.base_url}/v3/reference/tickers/{ticker}'
            params = {
                'date': date,
                'apiKey': self.api_key
            }
            
            response = requests.get(mc_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    try:
                        market_cap = data['results'].get('market_cap', 0)
                        return market_cap / 1000000  # Convert to millions
                    except:
                        # Try alternative calculation with outstanding shares
                        try:
                            os_shares = data['results'].get('weighted_shares_outstanding', 0)
                            if os_shares > 0:
                                # Get current price
                                price_data = self.fetch_daily_data(ticker, date, date, adjusted="false")
                                if not price_data.empty:
                                    close_price = price_data['c'].iloc[-1]
                                    return (os_shares * close_price) / 1000000
                        except:
                            pass
            return None
        except Exception as e:
            print(f"Error getting market cap for {ticker}: {e}")
            return None
    
    async def fetch_grouped_daily_data(self, session, date, adjusted=True):
        """Fetch grouped daily data for all stocks on a given date"""
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
    
    def calculate_trigger_day_criteria(self, current_data, prev_data):
        """Calculate trigger day criteria based on your exact parameters"""
        results = []
        
        for _, row in current_data.iterrows():
            ticker = row['ticker']
            
            # Find previous day data for this ticker
            prev_row = prev_data[prev_data['ticker'] == ticker]
            if prev_row.empty:
                continue
                
            prev_row = prev_row.iloc[0]
            
            # Your exact trigger day criteria
            try:
                # Calculate ratios
                pm_high_ratio = row.get('pm_high', row['h']) / prev_row['c'] if prev_row['c'] > 0 else 0
                gap_ratio = (row['o'] - prev_row['c']) / prev_row['c'] if prev_row['c'] > 0 else 0
                open_prev_high_ratio = row['o'] / prev_row['h'] if prev_row['h'] > 0 else 0
                
                # Check trigger day conditions
                trig_day = (
                    (pm_high_ratio >= 1.5) and  # pm_high/prev_close >= 150%
                    (gap_ratio >= 0.5) and      # gap >= 50%
                    (open_prev_high_ratio >= 1.3) and  # open/prev_high >= 130%
                    (row.get('pm_vol', row['v']) >= 5000000) and  # pm_vol >= 5M
                    (prev_row['c'] >= 0.75)      # prev_close >= $0.75
                )
                
                if trig_day:
                    results.append({
                        'ticker': ticker,
                        'date': row['date'],
                        'open': row['o'],
                        'high': row['h'],
                        'low': row['l'],
                        'close': row['c'],
                        'volume': row['v'],
                        'prev_close': prev_row['c'],
                        'prev_high': prev_row['h'],
                        'prev_volume': prev_row['v'],
                        'pm_high_ratio': pm_high_ratio,
                        'gap_ratio': gap_ratio,
                        'open_prev_high_ratio': open_prev_high_ratio,
                        'trig_day': 1
                    })
                    
            except Exception as e:
                print(f"Error calculating trigger day for {ticker}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def validate_ema_criteria(self, trigger_candidates):
        """Validate EMA criteria for trigger day candidates"""
        validated_results = []
        
        for _, row in trigger_candidates.iterrows():
            ticker = row['ticker']
            date = row['date']
            
            try:
                # Get historical data for EMA calculation (200+ days)
                start_date = pd.to_datetime(date) - pd.Timedelta(days=500)
                start_date = start_date.strftime('%Y-%m-%d')
                prev_date = (pd.to_datetime(date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                
                print(f"Validating EMA for {ticker} on {date}")
                
                # Fetch historical data
                daily_data = self.fetch_daily_data(ticker, start_date, prev_date, adjusted="true")
                
                if not daily_data.empty:
                    daily_data = self.adjust_daily(daily_data)
                    
                    # Check if we have enough data for EMA200
                    if len(daily_data) >= 200:
                        current_price = row['close']
                        ema200 = daily_data['ema200'].iloc[-1]
                        
                        # EMA validation: current price <= EMA200 * 0.8 (80% below EMA200)
                        ema_valid = current_price <= (ema200 * 0.8)
                        
                        if ema_valid:
                            # Get market cap
                            market_cap = self.get_market_cap(ticker, prev_date)
                            
                            result = row.to_dict()
                            result.update({
                                'ema200': ema200,
                                'ema_valid': 1,
                                'ema_ratio': current_price / ema200 if ema200 > 0 else 0,
                                'market_cap_millions': market_cap,
                                'historical_days': len(daily_data)
                            })
                            
                            validated_results.append(result)
                            print(f"✅ {ticker}: EMA Valid (Price: ${current_price:.2f}, EMA200: ${ema200:.2f})")
                        else:
                            print(f"❌ {ticker}: EMA Invalid (Price: ${current_price:.2f}, EMA200: ${ema200:.2f})")
                    else:
                        print(f"⚠️ {ticker}: Insufficient data ({len(daily_data)} days)")
                else:
                    print(f"❌ {ticker}: No historical data available")
                    
            except Exception as e:
                print(f"Error validating EMA for {ticker}: {e}")
                continue
        
        return pd.DataFrame(validated_results)
    
    async def scan_os_d1_complete(self, scan_date):
        """Complete OS D1 scan with trigger day and EMA validation"""
        
        print(f"🚀 Running Complete OS D1 Scan for {scan_date}")
        print("=" * 60)
        
        try:
            # Get previous trading day
            prev_date = self.get_previous_trading_day(scan_date, 1)
            prev_date_str = prev_date.strftime('%Y-%m-%d')
            
            print(f"📅 Current Date: {scan_date}")
            print(f"📅 Previous Date: {prev_date_str}")
            
            # Fetch current and previous day data
            async with aiohttp.ClientSession() as session:
                print("📡 Fetching current day data...")
                current_data = await self.fetch_grouped_daily_data(session, scan_date)
                
                print("📡 Fetching previous day data...")
                prev_data = await self.fetch_grouped_daily_data(session, prev_date_str)
                
            if current_data.empty or prev_data.empty:
                print("❌ No market data available for the specified dates")
                return pd.DataFrame()
                
            print(f"📊 Current day: {len(current_data)} tickers")
            print(f"📊 Previous day: {len(prev_data)} tickers")
            
            # Step 1: Find trigger day candidates
            print("\n🎯 Step 1: Identifying Trigger Day Candidates...")
            trigger_candidates = self.calculate_trigger_day_criteria(current_data, prev_data)
            
            print(f"📈 Found {len(trigger_candidates)} trigger day candidates")
            
            if trigger_candidates.empty:
                print("ℹ️ No trigger day setups found for this date")
                return pd.DataFrame()
            
            # Show trigger day candidates
            print(f"\n🎯 Trigger Day Candidates:")
            for _, row in trigger_candidates.iterrows():
                print(f"  • {row['ticker']}: Gap {row['gap_ratio']:.1%}, PM High {row['pm_high_ratio']:.1%}")
            
            # Step 2: Validate EMA criteria
            print(f"\n🧮 Step 2: Validating EMA Criteria...")
            final_results = self.validate_ema_criteria(trigger_candidates)
            
            print(f"\n✅ Final OS D1 Setups: {len(final_results)}")
            
            return final_results
            
        except Exception as e:
            print(f"❌ Scan failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

async def main():
    """Test the complete OS D1 scanner"""
    
    api_key = 'Fm7brz4s23eSocDErnL68cE7wspz2K1I'
    scanner = OS_D1_Complete_Scanner(api_key)
    
    # Use a recent trading date
    scan_date = '2024-08-20'  # Use a date that likely has data
    
    results = await scanner.scan_os_d1_complete(scan_date)
    
    if not results.empty:
        print(f"\n{'='*60}")
        print("🎯 OS D1 SETUP RESULTS")
        print(f"{'='*60}")
        
        for _, row in results.iterrows():
            print(f"\n📊 {row['ticker']} ({scan_date}):")
            print(f"  • Price: ${row['close']:.2f} (Gap: {row['gap_ratio']:.1%})")
            print(f"  • EMA200: ${row['ema200']:.2f} (Ratio: {row['ema_ratio']:.1%})")
            print(f"  • Volume: {row['volume']:,}")
            if row.get('market_cap_millions'):
                print(f"  • Market Cap: ${row['market_cap_millions']:.0f}M")
        
        # Export results
        output_file = f"os_d1_complete_results_{scan_date}.csv"
        results.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
    else:
        print("\nℹ️ No OS D1 setups found for this date")

if __name__ == '__main__':
    asyncio.run(main())