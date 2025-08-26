"""
OS D1 Long Continuation (LC) Scanner
Based on exact logic from Notion SM Playbook
Adapted for Polygon.io API integration
"""

import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings("ignore")

class OS_D1_LC_Scanner:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.nyse = mcal.get_calendar('NYSE')
        
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
            else:
                print(f"Failed to fetch data for {date}: {response.status}")
                return pd.DataFrame()
                
    def adjust_daily(self, df):
        """Apply all daily adjustments matching your exact logic"""
        # Previous day close
        df['pdc'] = df['c'].shift(1)
        
        # True Range and ATR calculation
        df['high_low'] = df['h'] - df['l']
        df['high_pdc'] = abs(df['h'] - df['pdc'])
        df['low_pdc'] = abs(df['l'] - df['pdc'])
        df['true_range'] = df[['high_low', 'high_pdc', 'low_pdc']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        # Historical data shifts (h1, h2, h3, c1, c2, etc.)
        for i in range(1, 4):
            df[f'h{i}'] = df['h'].shift(i)
            if i <= 2:
                df[f'c{i}'] = df['c'].shift(i)
                df[f'o{i}'] = df['o'].shift(i)
                df[f'l{i}'] = df['l'].shift(i)
                df[f'v{i}'] = df['v'].shift(i)
        
        # Dollar volume calculations
        df['dol_v'] = (df['c'] * df['v'])
        df['dol_v1'] = df['dol_v'].shift(1)
        df['dol_v2'] = df['dol_v'].shift(2)
        
        # Close range calculations
        df['close_range'] = (df['c'] - df['l']) / (df['h'] - df['l'])
        df['close_range1'] = df['close_range'].shift(1)
        
        # Gap calculations (normalized by ATR)
        df['gap_atr'] = ((df['o'] - df['pdc']) / df['atr'])
        df['gap_atr1'] = ((df['o1'] - df['c2']) / df['atr'])
        df['gap_pdh_atr'] = ((df['o'] - df['h1']) / df['atr'])
        
        # High change calculations
        df['high_chg'] = (df['h'] - df['o'])
        df['high_chg_atr'] = ((df['h'] - df['o']) / df['atr'])
        df['high_chg_atr1'] = ((df['h1'] - df['o1']) / df['atr'])
        df['high_chg_from_pdc_atr'] = ((df['h'] - df['c1']) / df['atr'])
        df['high_chg_from_pdc_atr1'] = ((df['h1'] - df['c2']) / df['atr'])
        
        # Percentage change
        df['pct_change'] = round(((df['c'] / df['c1']) - 1) * 100, 2)
        
        # EMA calculations
        df['ema9'] = df['c'].ewm(span=9, adjust=False).mean().fillna(0)
        df['ema20'] = df['c'].ewm(span=20, adjust=False).mean().fillna(0)
        df['ema50'] = df['c'].ewm(span=50, adjust=False).mean().fillna(0)
        df['ema200'] = df['c'].ewm(span=200, adjust=False).mean().fillna(0)
        
        # Distance from EMAs (normalized by ATR)
        df['dist_h_9ema'] = (df['h'] - df['ema9'])
        df['dist_h_20ema'] = (df['h'] - df['ema20'])
        df['dist_h_50ema'] = (df['h'] - df['ema50'])
        df['dist_h_200ema'] = (df['h'] - df['ema200'])
        
        df['dist_h_9ema1'] = df['dist_h_9ema'].shift(1)
        df['dist_h_20ema1'] = df['dist_h_20ema'].shift(1)
        df['dist_h_50ema1'] = df['dist_h_50ema'].shift(1)
        df['dist_h_200ema1'] = df['dist_h_200ema'].shift(1)
        
        df['dist_h_9ema_atr'] = df['dist_h_9ema'] / df['atr']
        df['dist_h_20ema_atr'] = df['dist_h_20ema'] / df['atr']
        df['dist_h_50ema_atr'] = df['dist_h_50ema'] / df['atr']
        df['dist_h_200ema_atr'] = df['dist_h_200ema'] / df['atr']
        
        df['dist_h_9ema_atr1'] = df['dist_h_9ema1'] / df['atr']
        df['dist_h_20ema_atr1'] = df['dist_h_20ema1'] / df['atr']
        df['dist_h_50ema_atr1'] = df['dist_h_50ema1'] / df['atr']
        df['dist_h_200ema_atr1'] = df['dist_h_200ema1'] / df['atr']
        
        # Rolling highs and lows
        for window in [5, 20, 50, 100, 250]:
            df[f'lowest_low_{window}'] = df['l'].rolling(window=window, min_periods=1).min()
            df[f'highest_high_{window}'] = df['h'].rolling(window=window, min_periods=1).max()
        
        # Historical highest highs with shifts
        df['highest_high_100_1'] = df['highest_high_100'].shift(1)
        df['highest_high_100_4'] = df['highest_high_100'].shift(4)
        df['highest_high_250_1'] = df['highest_high_250'].shift(1)
        df['highest_high_20_2'] = df['highest_high_20'].shift(2)
        df['highest_high_50_1'] = df['highest_high_50'].shift(1)
        df['highest_high_50_4'] = df['highest_high_50'].shift(4)
        
        # Distance calculations
        df['h_dist_to_lowest_low_20_atr'] = ((df['h'] - df['lowest_low_20']) / df['atr'])
        df['h_dist_to_lowest_low_5_atr'] = ((df['h'] - df['lowest_low_5']) / df['atr'])
        
        return df
    
    def check_high_lvl_filter_lc(self, df):
        """Apply your exact LC filtering logic from Notion"""
        
        # LC Frontside D3 Extended 1
        df['lc_frontside_d3_extended_1'] = (
            (df['h'] >= df['h1']) & (df['h1'] >= df['h2']) &
            (df['high_chg_atr1'] >= 0.5) & (df['gap_atr1'] >= 0.2) &
            (df['close_range1'] >= 0.6) & (df['c1'] >= df['o1']) &
            (df['dist_h_9ema_atr1'] >= 1.5) & (df['dist_h_20ema_atr1'] >= 3) &
            (df['high_chg_atr'] >= 0.7) & (df['gap_atr'] >= 0.2) &
            (df['close_range'] >= 0.6) & (df['dist_h_9ema_atr'] >= 2) &
            (df['dist_h_50ema_atr'] >= 4) & (df['v_ua'] >= 10000000) &
            (df['dol_v'] >= 500000000) & (df['c_ua'] >= 20) &
            ((df['h'] >= df['highest_high_250']) &
             (df['ema9'] >= df['ema20']) & (df['ema20'] >= df['ema50']) &
             (df['ema50'] >= df['ema200']))
        ).astype(int)
        
        # LC Backside D3 Extended 1
        df['lc_backside_d3_extended_1'] = (
            (df['h'] >= df['h1']) & (df['h1'] >= df['h2']) &
            (df['high_chg_atr1'] >= 0.5) & (df['gap_atr1'] >= 0.2) &
            (df['close_range1'] >= 0.6) & (df['c1'] >= df['o1']) &
            (df['dist_h_9ema_atr1'] >= 1.5) & (df['dist_h_20ema_atr1'] >= 3) &
            (df['high_chg_atr'] >= 0.7) & (df['gap_atr'] >= 0.2) &
            (df['close_range'] >= 0.6) & (df['dist_h_9ema_atr'] >= 2) &
            (df['dist_h_50ema_atr'] >= 4) & (df['v_ua'] >= 10000000) &
            (df['dol_v'] >= 500000000) & (df['c_ua'] >= 20) &
            ((df['ema9'] < df['ema20']) | (df['ema20'] < df['ema50']) |
             (df['ema50'] < df['ema200']) | (df['h'] < df['highest_high_250']))
        ).astype(int)
        
        # LC FBO (Failed Breakout)
        df['lc_fbo'] = (
            (((df['high_chg_atr'] >= 0.5) | (df['high_chg_from_pdc_atr'] >= 0.5)) &
             (df['h'] >= df['h1']) & (df['close_range'] >= 0.3) &
             (df['c'] >= df['o']) & (df['v_ua'] >= 10000000) &
             (df['dol_v'] >= 500000000) & (df['c_ua'] >= 2000000000) &
             ((df['h_dist_to_lowest_low_20_atr'] >= 4) | (df['h_dist_to_lowest_low_5_atr'] >= 2)) &
             (df['h'] >= df['highest_high_50_4'] - df['atr'] * 1) &
             (df['h'] <= df['highest_high_50_4'] + df['atr'] * 1) &
             (df['highest_high_50_4'] >= df['highest_high_100_4']) &
             (df['h1'] < df['highest_high_50_4']) &
             (df['h2'] < df['highest_high_50_4']) &
             (df['ema9'] >= df['ema20']) & (df['ema20'] >= df['ema50']) &
             (df['ema50'] >= df['ema200']))
        ).astype(int)
        
        # Add other LC patterns following your exact logic...
        # (I can add more patterns if needed)
        
        # Filter for rows that match any LC criteria
        columns_to_check = ['lc_frontside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_fbo']
        df_filtered = df[df[columns_to_check].any(axis=1)]
        
        return df_filtered
    
    def compute_indicators_grouped(self, df):
        """Apply indicators grouped by ticker (matching your logic)"""
        df = df.sort_values(by=['ticker', 'date'])
        
        # Group by ticker and apply adjustments
        df_grouped = df.groupby('ticker').apply(lambda group: self.adjust_daily(group))
        df_grouped.reset_index(drop=True, inplace=True)
        
        return df_grouped
    
    async def scan_dates(self, start_date, end_date):
        """Main scanning function matching your exact workflow"""
        
        print(f"🔍 Starting OS D1 LC scan from {start_date} to {end_date}")
        
        # Get trading days
        start_date_extended = (pd.to_datetime(start_date) - pd.DateOffset(days=400)).strftime('%Y-%m-%d')
        dates = [date.strftime('%Y-%m-%d') for date in 
                self.nyse.valid_days(start_date=start_date_extended, end_date=end_date)]
        
        print(f"Fetching data for {len(dates)} trading days...")
        
        # Fetch adjusted and unadjusted data
        async with aiohttp.ClientSession() as session:
            # Adjusted data
            adjusted_tasks = [self.fetch_grouped_daily_data(session, date, True) for date in dates]
            adjusted_results = await asyncio.gather(*adjusted_tasks, return_exceptions=True)
            
            # Unadjusted data  
            unadjusted_tasks = [self.fetch_grouped_daily_data(session, date, False) for date in dates]
            unadjusted_results = await asyncio.gather(*unadjusted_tasks, return_exceptions=True)
        
        # Process results
        df_adjusted = pd.concat([r for r in adjusted_results if isinstance(r, pd.DataFrame) and not r.empty], 
                               ignore_index=True)
        df_unadjusted = pd.concat([r for r in unadjusted_results if isinstance(r, pd.DataFrame) and not r.empty], 
                                 ignore_index=True)
        
        # Add '_ua' suffix to unadjusted columns (except date/ticker)
        df_unadjusted.rename(columns={col: col + '_ua' if col not in ['date', 'ticker'] else col 
                                     for col in df_unadjusted.columns}, inplace=True)
        
        # Merge adjusted and unadjusted data
        df = pd.merge(df_adjusted, df_unadjusted, on=['date', 'ticker'], how='inner')
        df = df.drop(columns=['vw', 't', 'n', 'vw_ua', 't_ua', 'n_ua'], errors='ignore')
        df = df.sort_values(by='date')
        
        print(f"Processing {len(df)} total records...")
        
        # Apply pre-conditions filter (matching your logic exactly)
        df['date'] = pd.to_datetime(df['date'])
        df['close_range'] = (df['c'] - df['l']) / (df['h'] - df['l'])
        df['dol_v'] = (df['c'] * df['v'])
        df['pre_conditions'] = (
            (df['c_ua'] >= 5) &
            (df['v_ua'] >= 10000000) &
            (df['dol_v'] >= 500000000) &
            (df['c'] > df['o']) &
            (df['close_range'] >= 0.3)
        )
        
        # Filter tickers that meet conditions at least once
        ticker_meet_conditions = df.groupby('ticker')['pre_conditions'].any()
        filtered_tickers = ticker_meet_conditions[ticker_meet_conditions].index
        df = df[df['ticker'].isin(filtered_tickers)]
        
        print(f"After pre-filtering: {len(df)} records from {len(filtered_tickers)} tickers")
        
        # Apply indicators grouped by ticker
        df = self.compute_indicators_grouped(df)
        df = df[df['pre_conditions'] == True]
        
        print(f"After indicator calculation: {len(df)} records")
        
        # Apply LC filters
        df_lc = self.check_high_lvl_filter_lc(df)
        
        # Filter for the requested date range
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        df_lc = df_lc[(df_lc['date'] >= start_date_dt) & (df_lc['date'] <= end_date_dt)]
        
        print(f"✅ Found {len(df_lc)} LC setups in date range")
        
        return df_lc
    
    def export_results(self, df, filename="os_d1_lc_scan_results.csv"):
        """Export results to CSV"""
        if not df.empty:
            # Select key columns for export
            export_columns = ['date', 'ticker', 'c', 'h', 'l', 'o', 'v', 'atr', 'gap_atr', 'high_chg_atr',
                            'close_range', 'dist_h_9ema_atr', 'dist_h_20ema_atr', 'dist_h_50ema_atr',
                            'lc_frontside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_fbo']
            
            df_export = df[[col for col in export_columns if col in df.columns]]
            df_export.to_csv(filename, index=False)
            print(f"📁 Results exported to {filename}")
            return df_export
        else:
            print("⚠️ No results to export")
            return df

# Example usage
async def main():
    # Replace with your Polygon API key
    API_KEY = "4r6MZNWLy2ucmhVI7fY8MrvXfXTSmxpy"
    
    scanner = OS_D1_LC_Scanner(API_KEY)
    
    # Scan recent dates
    start_date = "2025-01-10"
    end_date = "2025-01-17"
    
    results = await scanner.scan_dates(start_date, end_date)
    
    if not results.empty:
        print("\n🎯 LC SCAN RESULTS:")
        print("=" * 50)
        
        # Show summary by setup type
        setup_columns = [col for col in results.columns if col.startswith('lc_')]
        for col in setup_columns:
            count = results[col].sum()
            if count > 0:
                print(f"{col}: {count} setups")
                tickers = results[results[col] == 1]['ticker'].tolist()
                print(f"  Tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
        
        # Export results
        scanner.export_results(results)
        
        return results
    else:
        print("No LC setups found in the specified date range")
        return pd.DataFrame()

if __name__ == "__main__":
    results = asyncio.run(main())