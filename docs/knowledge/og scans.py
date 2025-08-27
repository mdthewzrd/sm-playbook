import pandas as pd
import requests
import time
import numpy as np
import pandas_market_calendars as mcal
import aiohttp
import asyncio
import pandas as pd
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate
import webbrowser
import plotly.graph_objects as go
import sys
from concurrent.futures import ThreadPoolExecutor
import dask
from dask.distributed import Client, as_completed
import dask.dataframe as dd
import datetime
import logging
import backoff

nyse = mcal.get_calendar('NYSE')
executor = ThreadPoolExecutor() 

import warnings
warnings.filterwarnings("ignore")

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

DATE = "2025-01-17"


# Replace with your Polygon API Key
API_KEY = '4r6MZNWLy2ucmhVI7fY8MrvXfXTSmxpy'
BASE_URL = "https://api.polygon.io"



def adjust_daily(df):
    # df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df['date'] = df['date'].dt.date          
    df['pdc'] = df['c'].shift(1)
    df['high_low'] = df['h'] - df['l']  # High - Low
    df['high_pdc'] = abs(df['h'] - df['pdc'])  # High - Previous Day Close
    df['low_pdc'] = abs(df['l'] - df['pdc'])  # Low - Previous Day Close
    # True Range (TR) is the max of high-low, high-pdc, low-pdc
    df['true_range'] = df[['high_low', 'high_pdc', 'low_pdc']].max(axis=1)
    # Calculate the ATR (using a 14-day rolling window)
    df['atr'] = df['true_range'].rolling(window=14).mean()
    # Drop intermediate columns for clean output
    df.drop(['high_low', 'high_pdc', 'low_pdc'], axis=1, inplace=True)

         
    df['h1'] = df['h'].shift(1)
    df['h2'] = df['h'].shift(2)

    df['h3'] = df['h'].shift(3)

    df['c1'] = df['c'].shift(1)
    df['c2'] = df['c'].shift(2)

    df['o1'] = df['o'].shift(1)
    df['o2'] = df['o'].shift(2)
    
    df['l1'] = df['l'].shift(1)
    df['l2'] = df['l'].shift(2)

    df['v1'] = df['v'].shift(1)
    df['v2'] = df['v'].shift(2)
    
    df['dol_v'] = (df['c'] * df['v'])
    df['dol_v1'] = df['dol_v'].shift(1)
    df['dol_v2'] = df['dol_v'].shift(2)


    df['close_range'] = (df['c'] - df['l'])/(df['h'] - df['l'])
    df['close_range1'] = df['close_range'].shift(1)

    df['gap_atr'] = ((df['o'] - df['pdc'])/df['atr'])
    df['gap_atr1'] = ((df['o1'] - df['c2'])/df['atr'])

    df['gap_pdh_atr'] = ((df['o'] - df['h1'])/df['atr'])
    
    df['high_chg'] = (df['h'] - df['o'])
    df['high_chg_atr'] = ((df['h'] - df['o'])/df['atr'])
    # df['high_chg_atr'] = round(df['high_chg_atr'], 2)
    df['high_chg_atr1'] = ((df['h1'] - df['o1'])/df['atr'])

    df['high_chg_from_pdc_atr'] = ((df['h'] - df['c1'])/df['atr'])
    df['high_chg_from_pdc_atr1'] = ((df['h1'] - df['c2'])/df['atr'])

    df['pct_change'] = round(((df['c'] / df['c1']) - 1)*100, 2)
    
    df['ema9'] = df['c'].ewm(span=9, adjust=False).mean().fillna(0)
    df['ema20'] = df['c'].ewm(span=20, adjust=False).mean().fillna(0)
    df['ema50'] = df['c'].ewm(span=50, adjust=False).mean().fillna(0)
    df['ema200'] = df['c'].ewm(span=200, adjust=False).mean().fillna(0)

    df['ema20_2'] = df['ema20'].shift(2)
    
    df['dist_h_9ema'] = (df['h'] - df['ema9'])
    df['dist_h_20ema'] = (df['h'] - df['ema20'])
    df['dist_h_50ema'] = (df['h'] - df['ema50'])
    df['dist_h_200ema'] = (df['h'] - df['ema200'])

    df['dist_h_9ema1'] = df['dist_h_9ema'].shift(1)
    df['dist_h_20ema1'] = df['dist_h_20ema'].shift(1)
    df['dist_h_50ema1'] = df['dist_h_50ema'].shift(1)
    df['dist_h_200ema1'] = df['dist_h_200ema'].shift(1)

    df['dist_h_9ema_atr'] = df['dist_h_9ema'] /df['atr']
    df['dist_h_20ema_atr'] = df['dist_h_20ema'] /df['atr']
    df['dist_h_50ema_atr'] = df['dist_h_50ema'] /df['atr']
    df['dist_h_200ema_atr'] = df['dist_h_200ema'] /df['atr']

    df['dist_h_9ema_atr1'] = df['dist_h_9ema1'] /df['atr']
    df['dist_h_20ema_atr1'] = df['dist_h_20ema1'] /df['atr']
    df['dist_h_50ema_atr1'] = df['dist_h_50ema1'] /df['atr']
    df['dist_h_200ema_atr1'] = df['dist_h_200ema1'] /df['atr']

    df['dist_h_9ema2'] = df['dist_h_9ema'].shift(2)
    df['dist_h_9ema3'] = df['dist_h_9ema'].shift(3)
    df['dist_h_9ema4'] = df['dist_h_9ema'].shift(4)

    df['dist_h_20ema2'] = df['dist_h_20ema'].shift(2)
    df['dist_h_20ema3'] = df['dist_h_20ema'].shift(3)
    df['dist_h_20ema4'] = df['dist_h_20ema'].shift(4)
    df['dist_h_20ema5'] = df['dist_h_20ema'].shift(5)

    df['dist_h_9ema_atr2'] = df['dist_h_9ema2'] /df['atr']
    df['dist_h_9ema_atr3'] = df['dist_h_9ema3'] /df['atr']
    df['dist_h_9ema_atr4'] = df['dist_h_9ema4'] /df['atr']
    
    df['dist_h_20ema_atr2'] = df['dist_h_20ema2'] /df['atr']
    df['dist_h_20ema_atr3'] = df['dist_h_20ema3'] /df['atr']
    df['dist_h_20ema_atr4'] = df['dist_h_20ema4'] /df['atr']

    df['lowest_low_20'] = df['l'].rolling(window=20, min_periods=1).min()
    df['lowest_low_20_2'] = df['lowest_low_20'].shift(2)
    df['h_dist_to_lowest_low_20_atr'] = ((df['h'] - df['lowest_low_20'])/df['atr'])

    df['lowest_low_30'] = df['l'].rolling(window=30, min_periods=1).min()
    df['lowest_low_30_1'] = df['lowest_low_30'].shift(1)

    df['h_dist_to_lowest_low_30'] = (df['h'] - df['lowest_low_30'])

    df['lowest_low_5'] = df['l'].rolling(window=5, min_periods=1).min()
    df['h_dist_to_lowest_low_5_atr'] = ((df['h'] - df['lowest_low_5'])/df['atr'])

    df['highest_high_100'] = df['h'].rolling(window=100).max()
    df['highest_high_100_1'] = df['highest_high_100'].shift(1)
    
    df['highest_high_250'] = df['h'].rolling(window=250, min_periods=1).max()
    df['highest_high_250_1'] = df['highest_high_250'].shift(1)

    df['highest_high_5'] = df['h'].rolling(window=5, min_periods=1).max()

    df['highest_high_50'] = df['h'].rolling(window=50, min_periods=1).max()
    df['highest_high_50_1'] = df['highest_high_50'].shift(1)
    df['highest_high_50_4'] = df['highest_high_50'].shift(4)

    df['highest_high_20'] = df['h'].rolling(window=20, min_periods=1).max()
    df['highest_high_20_2'] = df['highest_high_20'].shift(2)

    df['highest_high_100'] = df['h'].rolling(window=100, min_periods=1).max()
    df['highest_high_100_4'] = df['highest_high_100'].shift(4)

    return df
def adjust_intraday(df):
    #df=pd.DataFrame(results)
    df['date_time']=pd.to_datetime(df['t']*1000000).dt.tz_localize('UTC')
    df['date_time']=df['date_time'].dt.tz_convert('US/Eastern')

    #df['Date Time'] = pd.to_datetime(df['Date Time'])

    # format the datetime objects to "yyyy-mm-dd hh:mm:ss" format
    df['date_time'] = df['date_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['date_time'] = pd.to_datetime(df['date_time'])

    # df=df.set_index(['date_time']).asfreq('1min')
    # df.v = df.v.fillna(0)
    # df[['c']] = df[['c']].ffill()
    # df['h'].fillna(df['c'], inplace=True)
    # df['l'].fillna(df['c'], inplace=True)
    # df['o'].fillna(df['c'], inplace=True)
    # df=df.between_time('04:00', '20:00')
    # df = df.reset_index(level=0)

    df['time'] = pd.to_datetime(df['date_time']).dt.time
    df['date'] = pd.to_datetime(df['date_time']).dt.date

    # daily_v_sum = df.groupby(df['date_time'].dt.date)['v'].sum()
    # valid_dates = daily_v_sum[daily_v_sum > 0].index
    # df = df[df['date_time'].dt.date.isin(valid_dates)]
    # # df = df.reset_index(level=0)
    # df = df.reset_index(drop=True)

    

    df['time_int'] = df['date_time'].dt.hour * 100 + df['date_time'].dt.minute
    df['date_int'] = df['date_time'].dt.strftime('%Y%m%d').astype(int)

    
    
    df['date_time_int'] = df['date_int'].astype(str) + '.' + df['time_int'].astype(str)
    df['date_time_int'] = df['date_time_int'].astype(float)

    
    df['v_sum'] = df.groupby('date')['v'].cumsum()
    
    df['hod_all'] = df.groupby(df['date'])['h'].cummax().fillna(0)

    

    return df

def check_high_lvl_filter_lc(df):
    # df = df1.iloc[-1]
    # df = df1.tail(1)

    df['lc_frontside_d3_extended_1'] = ((df['h'] >= df['h1']) & 
                        (df['h1'] >= df['h2']) & 

                        (df['high_chg_atr1'] >= 0.5) & 
                        (df['gap_atr1'] >= 0.2) & 
                        (df['close_range1'] >= 0.6) &                  
                        (df['c1'] >= df['o1']) & 
                        (df['dist_h_9ema_atr1'] >= 1.5) & 
                        (df['dist_h_20ema_atr1'] >= 3) & 

                        (df['high_chg_atr'] >= 0.7) & 
                        (df['gap_atr'] >= 0.2) & 
                        (df['close_range'] >= 0.6) & 
                        (df['dist_h_9ema_atr'] >= 2) & 
                        (df['dist_h_50ema_atr'] >= 4) & 
                        (df['v_ua'] >= 10000000) & 
                        (df['dol_v'] >= 500000000) & 
                        (df['c_ua'] >= 20) & 

                        ((df['h'] >= df['highest_high_250']) &
                        (df['ema9'] >= df['ema20']) & 
                        (df['ema20'] >= df['ema50']) & 
                        (df['ema50'] >= df['ema200']))
                        
                        ).astype(int)
    df['lc_backside_d3_extended_1'] = ((df['h'] >= df['h1']) & 
                        (df['h1'] >= df['h2']) & 

                        (df['high_chg_atr1'] >= 0.5) & 
                        (df['gap_atr1'] >= 0.2) & 
                        (df['close_range1'] >= 0.6) &                  
                        (df['c1'] >= df['o1']) & 
                        (df['dist_h_9ema_atr1'] >= 1.5) & 
                        (df['dist_h_20ema_atr1'] >= 3) & 

                        (df['high_chg_atr'] >= 0.7) & 
                        (df['gap_atr'] >= 0.2) & 
                        (df['close_range'] >= 0.6) & 
                        (df['dist_h_9ema_atr'] >= 2) & 
                        (df['dist_h_50ema_atr'] >= 4) & 
                        (df['v_ua'] >= 10000000) & 
                        (df['dol_v'] >= 500000000) & 
                        (df['c_ua'] >= 20) & 

                        ((df['ema9'] < df['ema20']) | 
                        (df['ema20'] < df['ema50']) | 
                        (df['ema50'] < df['ema200']) |
                        (df['h'] < df['highest_high_250']))
                                                
                        ).astype(int)
    
    df['lc_frontside_d3_extended_2'] = ((df['h'] >= df['h1']) & 
                        (df['h1'] >= df['h2']) & 

                        (df['high_chg_atr1'] >= 0.3) & 
                        (df['close_range1'] >= 0.5) &                  
                        (df['c1'] >= df['o1']) & 
                        (df['dist_h_9ema_atr1'] >= 1) &                     
                        (df['high_chg1'] >= df['high_chg']*0.4) & 

                        (df['high_chg_atr'] >= 0.6) & 
                        (df['gap_atr'] >= 0.2) & 
                        (df['close_range'] >= 0.4) & 
                        (df['dist_h_9ema_atr'] >= 1.5) & 
                        (df['dist_h_20ema_atr'] >= 3) & 
                        (df['v_ua'] >= 10000000) & 
                        (df['dol_v'] >= 500000000) & 
                        (df['c_ua'] >= 20) & 

                        ((df['h'] >= df['highest_high_250']) &
                        (df['ema9'] >= df['ema20']) & 
                        (df['ema20'] >= df['ema50']) & 
                        (df['ema50'] >= df['ema200']))
                        
                        ).astype(int)  
    df['lc_backside_d3_extended_2'] = ((df['h'] >= df['h1']) & 
                        (df['h1'] >= df['h2']) & 

                        (df['high_chg_atr1'] >= 0.3) & 
                        (df['close_range1'] >= 0.5) &                  
                        (df['c1'] >= df['o1']) & 
                        (df['dist_h_9ema_atr1'] >= 1) &                     
                        (df['high_chg1'] >= df['high_chg']*0.4) & 

                        (df['high_chg_atr'] >= 0.6) & 
                        (df['gap_atr'] >= 0.2) & 
                        (df['close_range'] >= 0.4) & 
                        (df['dist_h_9ema_atr'] >= 1.5) & 
                        (df['dist_h_20ema_atr'] >= 3) & 
                        (df['v_ua'] >= 10000000) & 
                        (df['dol_v'] >= 500000000) & 
                        (df['c_ua'] >= 20) & 

                        ((df['ema9'] < df['ema20']) | 
                        (df['ema20'] < df['ema50']) | 
                        (df['ema50'] < df['ema200']) |
                        (df['h'] < df['highest_high_250']))
                        
                        ).astype(int)
    
    df['lc_frontside_d4_para'] = ((df['h'] >= df['h1']) & 
                        (df['h1'] >= df['h2']) & 
                        
                        (df['dist_h_9ema_atr2'] >= 1) & 
                        (df['dist_h_9ema_atr3'] >= 1) & 
                        (df['dist_h_9ema_atr4'] >= 1) & 
                        
                        (df['c2'] >= df['o2']) & 
                        (df['dist_h_20ema_atr2'] >= 1.5) & 

                        (df['high_chg_atr1'] >= 0.3) & 
                        (df['gap_atr1'] >= 0) & 
                        (df['close_range1'] >= 0.3) &                  
                        (df['c1'] >= df['o1']) & 
                        (df['dist_h_9ema_atr1'] >= 1.5) & 
                        (df['dist_h_20ema_atr1'] >= 3) & 

                        (df['high_chg_atr'] >= 0.5) & 
                        (df['gap_atr'] >= 0.25) & 
                        (df['close_range'] >= 0.3) &                   
                        (df['c'] >= df['o']) & 
                        (df['dist_h_9ema_atr'] >= 2) & 
                        (df['dist_h_50ema_atr'] >= 4) & 
                        (df['v_ua'] >= 10000000) & 
                        (df['dol_v'] >= 500000000) & 
                        (df['c_ua'] >= 20) & 

                        ((df['h'] >= df['highest_high_250']) &
                        (df['ema9'] >= df['ema20']) & 
                        (df['ema20'] >= df['ema50']) & 
                        (df['ema50'] >= df['ema200']))
                        
                        ).astype(int)       
    df['lc_backside_d4_para'] = ((df['h'] >= df['h1']) & 
                        (df['h1'] >= df['h2']) & 
                        
                        (df['dist_h_9ema_atr2'] >= 1) & 
                        (df['dist_h_9ema_atr3'] >= 1) & 
                        (df['dist_h_9ema_atr4'] >= 1) & 
                        
                        (df['c2'] >= df['o2']) & 
                        (df['dist_h_20ema_atr2'] >= 1.5) & 

                        (df['high_chg_atr1'] >= 0.3) & 
                        (df['gap_atr1'] >= 0) & 
                        (df['close_range1'] >= 0.3) &                  
                        (df['c1'] >= df['o1']) & 
                        (df['dist_h_9ema_atr1'] >= 1.5) & 
                        (df['dist_h_20ema_atr1'] >= 3) & 

                        (df['high_chg_atr'] >= 0.5) & 
                        (df['gap_atr'] >= 0.25) & 
                        (df['close_range'] >= 0.3) &                   
                        (df['c'] >= df['o']) & 
                        (df['dist_h_9ema_atr'] >= 2) & 
                        (df['dist_h_50ema_atr'] >= 4) & 
                        (df['v_ua'] >= 10000000) & 
                        (df['dol_v'] >= 500000000) & 
                        (df['c_ua'] >= 20) & 

                        ((df['ema9'] < df['ema20']) | 
                        (df['ema20'] < df['ema50']) | 
                        (df['ema50'] < df['ema200']) |
                        (df['h'] < df['highest_high_250']))
                        
                        ).astype(int)

    df['lc_frontside_d3_uptrend'] = ((df['h'] >= df['h1']) & 
                        (df['h1'] >= df['h2']) & 

                        (df['high_chg_atr1'] >= 0.3) & 
                        (df['close_range1'] >= 0.5) &                  
                        (df['c1'] >= df['o1']) & 
                        (df['dist_h_9ema_atr1'] >= 2) & 
                        (df['dist_h_20ema_atr1'] >= 3) & 

                        (df['high_chg_atr'] >= 0.5) & 
                        (df['gap_atr'] >= -0.2) & 
                        (df['close_range'] >= 0.5) & 
                        (df['dist_h_9ema_atr'] >= 2) & 
                        (df['dist_h_20ema_atr'] >= 3) & 
                        (df['dist_h_200ema_atr'] >= 7) & 
                        (df['v_ua'] >= 10000000) & 
                        (df['dol_v'] >= 500000000) & 
                        (df['c_ua'] >= 20) & 

                        ((df['h'] >= df['highest_high_250']) &
                        (df['ema9'] >= df['ema20']) & 
                        (df['ema20'] >= df['ema50']) & 
                        (df['ema50'] >= df['ema200']))
                        
                        ).astype(int)         
    df['lc_backside_d3'] = ((df['h'] >= df['h1']) & 
                            (df['h1'] >= df['h2']) & 

                            (df['high_chg_atr1'] >= 1) & 
                            (df['close_range1'] >= 0.5) &                  
                            (df['c1'] >= df['o1']) & 
                            (df['dist_h_9ema_atr1'] >= 1) & 
                            (df['dist_h_20ema_atr1'] >= 2) & 

                            (df['high_chg_atr'] >= 1) & 
                            (df['close_range'] >= 0.5) & 
                            (df['dist_h_9ema_atr'] >= 1) & 
                            (df['dist_h_20ema_atr'] >= 2) & 
                            (df['v_ua'] >= 10000000) & 
                            (df['dol_v'] >= 500000000) & 
                            (df['c_ua'] >= 20) & 

                            ((df['ema9'] < df['ema20']) | 
                            (df['ema20'] < df['ema50']) | 
                            (df['ema50'] < df['ema200']) |
                            (df['h'] < df['highest_high_250']))
                                    
                        ).astype(int)

    
    df['lc_frontside_d2_uptrend'] = ((df['high_chg_atr'] >= 0.75) & 
                            (df['close_range'] >= 0.7) &           
                            (df['c'] >= df['o']) & 
                            (df['dist_h_9ema_atr'] >= 1.5) & 
                            (df['dist_h_20ema_atr'] >= 3) & 
                            (df['v_ua'] >= 10000000) & 
                            (df['dol_v'] >= 500000000) & 
                            (df['c_ua'] >= 2000000000) & 
                            (df['h_dist_to_lowest_low_20_atr'] >= 5) & 
                            (df['h'] >= df['highest_high_20']) &

                            ((df['h'] >= df['highest_high_250']) &
                            (df['ema9'] >= df['ema20']) & 
                            (df['ema20'] >= df['ema50']) & 
                            (df['ema50'] >= df['ema200']))
                            
                            ).astype(int)        
    df['lc_frontside_d2'] = ((df['high_chg_atr'] >= 1.5) & 
                            (df['close_range'] >= 0.5) &                   
                            (df['c'] >= df['o']) & 
                            (df['v_ua'] >= 10000000) & 
                            (df['dol_v'] >= 500000000) & 
                            (df['c_ua'] >= 20) & 

                            ((df['h'] >= df['highest_high_250']) &
                            (df['ema9'] >= df['ema20']) & 
                            (df['ema20'] >= df['ema50']) & 
                            (df['ema50'] >= df['ema200']))

                            ).astype(int)
    df['lc_backside_d2'] = ((df['high_chg_atr'] >= 1.5) & 
                            (df['close_range'] >= 0.5) &           
                            (df['c'] >= df['o']) & 
                            (df['v_ua'] >= 10000000) & 
                            (df['dol_v'] >= 500000000) & 
                            (df['c_ua'] >= 20) & 
                            (df['h'] >= df['highest_high_5']) &

                            ((df['ema9'] < df['ema20']) | 
                            (df['ema20'] < df['ema50']) | 
                            (df['ema50'] < df['ema200']) |
                            (df['h'] < df['highest_high_250']))
                            
                            ).astype(int)
    
    
    df['lc_fbo'] = (((df['high_chg_atr'] >= 0.5) | (df['high_chg_from_pdc_atr'] >= 0.5)) & 
                    (df['h'] >= df['h1']) & 
                    (df['close_range'] >= 0.3) &           
                    (df['c'] >= df['o']) & 
                    (df['v_ua'] >= 10000000) & 
                    (df['dol_v'] >= 500000000) & 
                    (df['c_ua'] >= 2000000000) & 
                    ((df['h_dist_to_lowest_low_20_atr'] >= 4) | (df['h_dist_to_lowest_low_5_atr'] >= 2)) & 
                    (df['h'] >= df['highest_high_50_4'] - df['atr']*1) & 
                    (df['h'] <= df['highest_high_50_4'] + df['atr']*1) & 
                    
                    (df['highest_high_50_4'] >= df['highest_high_100_4']) & 

                    (df['h1'] < df['highest_high_50_4']) & 
                    (df['h2'] < df['highest_high_50_4']) & 
                    (df['h3'] < df['highest_high_50_4']) & 

                    (df['ema9'] >= df['ema20']) & 
                    (df['ema20'] >= df['ema50']) & 
                    (df['ema50'] >= df['ema200'])
                    
                    ).astype(int)


   

    columns_to_check = ['lc_frontside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_frontside_d3_extended_2', 'lc_backside_d3_extended_2', 'lc_frontside_d4_para', 'lc_backside_d4_para',
     'lc_frontside_d3_uptrend', 'lc_backside_d3', 'lc_frontside_d2_uptrend', 'lc_frontside_d2', 'lc_backside_d2', 'lc_fbo']


    df2 = df[df[columns_to_check].any(axis=1)]
    return df2 



async def fetch_intraday_data(session, ticker, start_date, end_date):
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}'
    params = {'apiKey': API_KEY}
    async with session.get(url, params=params) as response:
        if response.status == 200:
            data = await response.json()
            return pd.DataFrame(data['results'])
        else:
            print(f"Failed to fetch data for {ticker}: {response.status}")
            return pd.DataFrame()
        

def get_min_price_lc(df):
    ### LC Min Price
    df['lc_frontside_d3_extended_1_min_price'] = round((df['c'] + df['atr']*.25), 2)
    df['lc_backside_d3_extended_1_min_price'] = round(np.maximum(
            df['c'] + df['atr']*1, df['h'] + df['atr']*0.75
        ), 2)

    df['lc_frontside_d3_extended_2_min_price'] = round(np.maximum(
                    df['c'] + df['atr']*0.75,
                    np.maximum(df['h'], df['ema50'] + df['atr']*5)
                    ), 2)   
    df['lc_backside_d3_extended_2_min_price'] = round(np.maximum(
                    df['c'] + df['atr']*1,
                    np.maximum(df['h'] + df['atr']*0.75, df['ema50'] + df['atr']*5)
                    ), 2)   
    
    
    df['lc_frontside_d4_para_min_price'] = round((df['c'] + df['atr']*0.25), 2)
    # df['lc_backside_d4_para_min_price'] = round((df['c'] + df['atr']*0.25), 2)
    df['lc_backside_d4_para_min_price'] = round(np.maximum(
            df['c'] + df['atr']*1, df['h'] + df['atr']*0.75
        ), 2)



    df['lc_frontside_d3_uptrend_min_price'] = round((df['c'] + df['atr']*0.2), 2)
    df['lc_backside_d3_min_price'] = round(np.maximum(
            df['c'] + df['atr']*1, df['h'] + df['atr']*0.75
        ), 2)
    
    df['lc_frontside_d2_uptrend_min_price'] = round(np.maximum(
            df['c'] + df['atr']*0.5, df['h']
        ), 2)
    
    df['lc_frontside_d2_min_price'] = round((df['c'] + df['atr']*2), 2)
    df['lc_backside_d2_min_price'] = round((df['c'] + df['atr']*2), 2)
    # df['lc_backside_d2_min_price'] = round(np.maximum(
    #         df['c'] + df['atr']*1, df['h'] + df['atr']*0.75
    #     ), 2)


    df['lc_fbo_min_price'] = round(np.maximum(
            df['c'] + df['atr']*0.5, df['h']
        ), 2)
        


    df['lc_frontside_d3_extended_1_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
    df['lc_backside_d3_extended_1_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
    df['lc_frontside_d3_extended_2_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
    df['lc_backside_d3_extended_2_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
    df['lc_frontside_d4_para_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
    df['lc_backside_d4_para_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
    df['lc_frontside_d3_uptrend_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
    df['lc_backside_d3_min_price'] = round((df['c'] + df['d1_range']*.3), 2)
    df['lc_frontside_d2_uptrend_min_price'] = round((df['c'] + df['d1_range']*.5), 2)
    df['lc_frontside_d2_min_price'] = round((df['c'] + df['d1_range']*.5), 2)
    df['lc_frontside_d2_min_price'] = round((df['c'] + df['d1_range']*.5), 2)
    df['lc_backside_d2_min_price'] = round((df['c'] + df['d1_range']*.5), 2)

    

    columns_to_check = ['lc_frontside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_frontside_d3_extended_2', 'lc_backside_d3_extended_2', 'lc_frontside_d4_para', 'lc_backside_d4_para',
     'lc_frontside_d3_uptrend', 'lc_backside_d3', 'lc_frontside_d2_uptrend', 'lc_frontside_d2', 'lc_backside_d2', 'lc_fbo']

    df['lowest_min_price'] = df.apply(lambda row: min([row[col + '_min_price'] for col in columns_to_check if row[col] == 1]), axis=1)

    for col in columns_to_check:
        min_price_col = f"{col}_min_price"
        min_pct_col = f"{col}_min_pct"
        df[min_pct_col] = round((df[min_price_col] / df['c'] - 1) * 100, 2)


    return df



async def fetch_intial_stock_list(session, date, adj):
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}?adjusted={adj}&apiKey={API_KEY}"
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            if 'results' in data:
                df = pd.DataFrame(data['results'])
                # df = df[(df['v_ua'] >= 2000000) & (df['c'] >= df['o']) & (df['c'] * df['v'] >= 20000000) & (((df['c'] * df['l']) / (df['h'] * df['l'])) >= 0.3)]
                df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
                df.rename(columns={'T': 'ticker'}, inplace=True)
                # print(url)
                return df




def process_stock_data_grouped(df):
    # Group by 'ticker' and apply the calculations to each group
    df_grouped = df.groupby('ticker').apply(lambda group: compute_indicators(group))
    # Reset index to undo the multi-index caused by groupby
    df_grouped.reset_index(drop=True, inplace=True)
    return df_grouped
def compute_indicators(group):
    # Make sure to sort by date to respect chronological order
    group = group.sort_values(by='date')

    # Calculating various financial metrics and indicators
    group['pdc'] = group['c'].shift(1)
    group['high_low'] = group['h'] - group['l']
    group['high_pdc'] = abs(group['h'] - group['pdc'])
    group['low_pdc'] = abs(group['l'] - group['pdc'])
    group['true_range'] = group[['high_low', 'high_pdc', 'low_pdc']].max(axis=1)
    group['atr'] = group['true_range'].rolling(window=14).mean()

    
    group['d1_range'] = abs(group['h'] - group['l'])
    
    # Shifting high, close, open, low, and volume
    for i in range(1, 4):  # Example for up to 3 days shift
        group[f'h{i}'] = group['h'].shift(i).fillna(0)
        if i <= 2:  # Up to 2 days shift for close, open, low, volume
            group[f'c{i}'] = group['c'].shift(i).fillna(0)
            group[f'o{i}'] = group['o'].shift(i).fillna(0)
            group[f'l{i}'] = group['l'].shift(i).fillna(0)
            group[f'v{i}'] = group['v'].shift(i).fillna(0)

    # Dollar volume and other calculations
    # group['dol_v'] = group['c'] * group['v']
    group['dol_v1'] = group['dol_v'].shift(1)
    group['dol_v2'] = group['dol_v'].shift(2)
    
    # group['close_range'] = (group['c'] - group['l']) / (group['h'] - group['l'])
    group['close_range1'] = group['close_range'].shift(1)
    
    # Gap metrics related to ATR
    group['gap_atr'] = ((group['o'] - group['pdc']) / group['atr'])
    group['gap_atr1'] = ((group['o1'] - group['c2']) / group['atr'])
    group['gap_pdh_atr'] = ((group['o'] - group['h1']) / group['atr'])

    # High change metrics normalized by ATR
    group['high_chg'] = (group['h'] - group['o'])
    group['high_chg_atr'] = (group['high_chg'] / group['atr'])
    group['high_chg_atr1'] = ((group['h1'] - group['o1']) / group['atr'])

    # High change from previous day close normalized by ATR
    group['high_chg_from_pdc_atr'] = ((group['h'] - group['c1']) / group['atr'])
    group['high_chg_from_pdc_atr1'] = ((group['h1'] - group['c2']) / group['atr'])

    # Percentage change in close price from the previous day
    group['pct_change'] = ((group['c'] / group['c1'] - 1) * 100).round(2)



    # Calculate EMAs and fill NAs.
    for period in [9, 20, 50, 200]:
        group[f'ema{period}'] = group['c'].ewm(span=period, adjust=False).mean().fillna(0)
        
        # Calculate distances from EMAs for the high price.
        group[f'dist_h_{period}ema'] = group['h'] - group[f'ema{period}']
        
        # Normalize these distances by the ATR.
        group[f'dist_h_{period}ema_atr'] = group[f'dist_h_{period}ema'] / group['atr']
        
        # Apply shifts to the calculated distances and normalize again by ATR.
        for dist in range(1, 5):
            group[f'dist_h_{period}ema{dist}'] = group[f'dist_h_{period}ema'].shift(dist)
            group[f'dist_h_{period}ema_atr{dist}'] = group[f'dist_h_{period}ema{dist}'] / group['atr']


    # Rolling maximums and minimums
    for window in [5, 20, 50, 100, 250]:
        group[f'lowest_low_{window}'] = group['l'].rolling(window=window, min_periods=1).min()
        group[f'highest_high_{window}'] = group['h'].rolling(window=window, min_periods=1).max()
        if window in [20, 50, 100, 250]:  # Shifting previous highs for selected windows
            for dist in range(1, 5):
                if window == 20 or window == 50:
                    group[f'highest_high_{window}_{dist}'] = group[f'highest_high_{window}'].shift(dist)
    
    
    group['lowest_low_30'] = group['l'].rolling(window=30, min_periods=1).min()
    group['lowest_low_30_1'] = group['lowest_low_30'].shift(1)


    group['highest_high_100_1'] = group['highest_high_100'].shift(1)
    group['highest_high_100_4'] = group['highest_high_100'].shift(4)
    group['highest_high_250_1'] = group['highest_high_250'].shift(1)
    group['lowest_low_20_2'] = group['lowest_low_20'].shift(2)


    group['lowest_low_20_ua'] = group['l_ua'].rolling(window=20, min_periods=1).min()

    group['h_dist_to_lowest_low_30'] = (group['h'] - group['lowest_low_30']) / group['atr']

    group['h_dist_to_lowest_low_20_atr'] = (group['h'] - group['lowest_low_20']) / group['atr']
    group['h_dist_to_lowest_low_5_atr'] = (group['h'] - group['lowest_low_5']) / group['atr']
    
    group['ema20_2'] = group['ema20'].shift(2)

    # Drop intermediate columns
    columns_to_drop = ['high_low', 'high_pdc', 'low_pdc']
    group.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    return group

def compute_indicators1(df):
    # Sorting by 'ticker' and 'date' to respect chronological order for each ticker
    df = df.sort_values(by=['ticker', 'date'])
    
    # Calculating previous day's close
    df['pdc'] = df.groupby('ticker')['c'].shift(1)

    # Calculating ranges and true range
    df['high_low'] = df['h'] - df['l']
    df['high_pdc'] = (df['h'] - df['pdc']).abs()
    df['low_pdc'] = (df['l'] - df['pdc']).abs()
    df['true_range'] = df[['high_low', 'high_pdc', 'low_pdc']].max(axis=1)
    df['atr'] = df.groupby('ticker')['true_range'].transform(lambda x: x.rolling(window=14).mean())

    
    df['d1_range'] = abs(df['h'] - df['l'])

    # Shifting values for high, close, open, low, volume
    for i in range(1, 4):
        df[f'h{i}'] = df.groupby('ticker')['h'].shift(i).fillna(0)
        if i <= 2:  # Limiting to 2 days shift for close, open, low, volume
            df[f'c{i}'] = df.groupby('ticker')['c'].shift(i).fillna(0)
            df[f'o{i}'] = df.groupby('ticker')['o'].shift(i).fillna(0)
            df[f'l{i}'] = df.groupby('ticker')['l'].shift(i).fillna(0)
            df[f'v{i}'] = df.groupby('ticker')['v'].shift(i).fillna(0)

    # Dollar volume calculations and shifts
    df['dol_v'] = df['c'] * df['v']
    df['dol_v1'] = df.groupby('ticker')['dol_v'].shift(1)
    df['dol_v2'] = df.groupby('ticker')['dol_v'].shift(2)

    # Close range calculations and shifts
    df['close_range'] = (df['c'] - df['l']) / (df['h'] - df['o'])
    df['close_range1'] = df.groupby('ticker')['close_range'].shift(1)

    # Gap metrics related to ATR
    df['gap_atr'] = ((df['o'] - df['pdc']) / df['atr'])
    df['gap_atr1'] = ((df['o1'] - df['c2']) / df['atr'])
    df['gap_pdh_atr'] = ((df['o'] - df['h1']) / df['atr'])

    # High change metrics normalized by ATR
    df['high_chg'] = df['h'] - df['o']
    df['high_chg1'] = df['h1'] - df['o1']
    df['high_chg_atr'] = df['high_chg'] / df['atr']
    df['high_chg_atr1'] = ((df['h1'] - df['o1']) / df['atr'])

    # High change from previous day close normalized by ATR
    df['high_chg_from_pdc_atr'] = ((df['h'] - df['c1']) / df['atr'])
    df['high_chg_from_pdc_atr1'] = ((df['h1'] - df['c2']) / df['atr'])

    # Percentage change in close price from the previous day
    df['pct_change'] = ((df['c'] / df['c1'] - 1) * 100).round(2)

    # Calculating EMAs
    for period in [9, 20, 50, 200]:
        df[f'ema{period}'] = df.groupby('ticker')['c'].transform(lambda x: x.ewm(span=period, adjust=False).mean().fillna(0))
        df[f'dist_h_{period}ema'] = df['h'] - df[f'ema{period}']
        df[f'dist_h_{period}ema_atr'] = df[f'dist_h_{period}ema'] / df['atr']

        # Apply shifts to the calculated distances and normalize again by ATR
        for dist in range(1, 5):
            df[f'dist_h_{period}ema{dist}'] = df.groupby('ticker')[f'dist_h_{period}ema'].shift(dist)
            df[f'dist_h_{period}ema_atr{dist}'] = df[f'dist_h_{period}ema{dist}'] / df['atr']

    # Calculate rolling maximums and minimums
    for window in [5, 20, 50, 100, 250]:
        df[f'lowest_low_{window}'] = df.groupby('ticker')['l'].transform(lambda x: x.rolling(window=window, min_periods=1).min())
        df[f'highest_high_{window}'] = df.groupby('ticker')['h'].transform(lambda x: x.rolling(window=window, min_periods=1).max())

        # Shifting previous highs for selected windows
        for dist in range(1, 5):
            df[f'highest_high_{window}_{dist}'] = df.groupby('ticker')[f'highest_high_{window}'].shift(dist)

    # Calculate rolling minimums for the low prices with shifts
    df['lowest_low_30'] = df.groupby('ticker')['l'].transform(lambda x: x.rolling(window=30, min_periods=1).min())
    df['lowest_low_30_1'] = df.groupby('ticker')['lowest_low_30'].shift(1)

    # Calculate rolling maximums for high prices with multiple shifts
    df['highest_high_100_1'] = df.groupby('ticker')['highest_high_100'].shift(1)
    df['highest_high_100_4'] = df.groupby('ticker')['highest_high_100'].shift(4)
    df['highest_high_250_1'] = df.groupby('ticker')['highest_high_250'].shift(1)
    df['lowest_low_20_2'] = df.groupby('ticker')['lowest_low_20'].shift(2)

    # Assuming l_ua is a predefined column in your DataFrame
    df['lowest_low_20_ua'] = df.groupby('ticker')['l_ua'].transform(lambda x: x.rolling(window=20, min_periods=1).min())

    # Calculate distances from the lowest lows normalized by ATR
    df['h_dist_to_lowest_low_30'] = (df['h'] - df['lowest_low_30'])
    df['h_dist_to_lowest_low_30_atr'] = (df['h'] - df['lowest_low_30']) / df['atr']
    df['h_dist_to_lowest_low_20_atr'] = (df['h'] - df['lowest_low_20']) / df['atr']
    df['h_dist_to_lowest_low_5_atr'] = (df['h'] - df['lowest_low_5']) / df['atr']

    # Shifting EMAs
    df['ema20_2'] = df.groupby('ticker')['ema20'].shift(2)


    df['v_ua1'] = df.groupby('ticker')['v_ua'].shift(1)
    df['v_ua2'] = df.groupby('ticker')['v_ua'].shift(2)


    # Drop intermediate columns
    columns_to_drop = ['high_low', 'high_pdc', 'low_pdc']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    return df

def calculate_trading_days(date):
    # Define a range of trading days (30 days before and after the given date)
    start_date = date - pd.Timedelta(days=30)
    end_date = date + pd.Timedelta(days=30)

    # Get the trading schedule for the range
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index

    # Ensure the date exists in the trading days
    if date not in trading_days:
        return pd.NaT, pd.NaT  # Return NaT if the date is not a trading day

    # Find the location of the given date in the trading days
    idx = trading_days.get_loc(date)

    # Calculate the next trading day and the fourth previous trading day
    date_plus_1 = trading_days[idx + 1] if idx + 1 < len(trading_days) else pd.NaT
    date_minus_4 = trading_days[idx - 4] if idx - 4 >= 0 else pd.NaT

    return date_plus_1, date_minus_4
def get_offsets(date):
    if date not in trading_days_map:
        return pd.NaT, pd.NaT
    idx = trading_days_map[date]
    date_plus_1 = trading_days_list[idx + 1] if idx + 1 < len(trading_days_list) else pd.NaT
    date_minus_4 = trading_days_list[idx - 4] if idx - 4 >= 0 else pd.NaT
    date_minus_30 = trading_days_list[idx - 30] if idx - 30 >= 0 else pd.NaT
    return date_plus_1, date_minus_4, date_minus_30





def fetch_intraday_data(ticker, start_date, end_date):
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/30/minute/{start_date}/{end_date}?adjusted=true'
    params = {'apiKey': API_KEY}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                return pd.DataFrame(data['results'])
            else:
                print(f"No results for {ticker}")
        else:
            print(f"Error fetching data for {ticker}: {response.status_code}")
    except Exception as e:
        print(f"Exception for {ticker}: {e}")
    return pd.DataFrame()



def process_lc_row(row):
    """Process a single row for intraday data and calculations."""
    try:
        ticker = row['ticker']
        date = row['date']

        date_minus_4 = row['date_minus_4']
        date_plus_1 = row['date_plus_1']

        # Fetch and adjust intraday data
        intraday_data = fetch_intraday_data(ticker, date_minus_4, date_plus_1)
        intraday_data = adjust_intraday(intraday_data)

        date_plus_1_formatted = datetime.datetime.strptime(date_plus_1, '%Y-%m-%d').date()

        intraday_data_before = intraday_data[intraday_data['date'] != date_plus_1_formatted]
        
        resp_v = intraday_data_before[intraday_data_before['time_int'] == 900].set_index('date')['v_sum']
        resp_o = intraday_data_before[intraday_data_before['time_int'] == 930].set_index('date')['o']
        resp_v, resp_o = resp_v.align(resp_o, fill_value=0)
        pm_dol_vol_all = resp_o * resp_v
        avg_pm_dol_vol = pm_dol_vol_all.mean()

        # Add results to row
        row['avg_5d_pm_dol_vol'] = avg_pm_dol_vol
        if avg_pm_dol_vol >= 10000000:
            row['valid_pm_liq'] = 1
        else:
            row['valid_pm_liq'] = 0
        # row['valid_pm_liq'] = 1 if avg_pm_dol_vol >= 10000000 else 0

        
        intraday_data_after = intraday_data[intraday_data['date'] == date_plus_1_formatted]      
                    

        pm_df = intraday_data_after[intraday_data_after['time_int'] <= 900].set_index('time')
        open_df = intraday_data_after[intraday_data_after['time_int'] <= 930].set_index('time')

        if not pm_df.empty:
            resp_v = pm_df['v_sum'][-1]
            resp_o = open_df['o'][-1]
            pm_dol_vol_next_day = resp_o * resp_v
            pmh_next_day = pm_df['hod_all'][-1]
        else:
            pm_dol_vol_next_day = 0
            pmh_next_day = 0
            resp_o = 0
        
        row['pmh_next_day'] = pmh_next_day
        row['open_next_day'] = resp_o
        row['pm_dol_vol_next_day'] = pm_dol_vol_next_day

    except Exception as e:
        print(f"Error processing LC Row {row['ticker']} on {row['date']}: {e}")
        row['avg_5d_pm_dol_vol'] = None
        row['valid_pm_liq'] = None
        row['pmh_next_day'] = 0
        row['open_next_day'] = 0
        row['pm_dol_vol_next_day'] = 0

    return row


def dates_before_after(df):
    global trading_days_map, trading_days_list

    start_date = df['date'].min() - pd.Timedelta(days=60)
    end_date = df['date'].max() + pd.Timedelta(days=30)
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = pd.Series(schedule.index, index=schedule.index)

    # Map trading days for faster lookup
    trading_days_list = trading_days.index.to_list()
    trading_days_map = {day: idx for idx, day in enumerate(trading_days_list)}

    results = df_lc['date'].map(get_offsets)
    df_lc[['date_plus_1', 'date_minus_4', 'date_minus_30']] = pd.DataFrame(results.tolist(), index=df_lc.index)

    # Format dates as strings if needed
    df_lc['date_plus_1'] = df_lc['date_plus_1'].dt.strftime('%Y-%m-%d')
    df_lc['date_minus_4'] = df_lc['date_minus_4'].dt.strftime('%Y-%m-%d')
    df_lc['date_minus_30'] = df_lc['date_minus_30'].dt.strftime('%Y-%m-%d')


    return df


def check_next_day_valid_lc(df):
    # Ensure minimum price columns exist for each check column

    columns_to_check = ['lc_backside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_frontside_d3_extended_2', 'lc_backside_d3_extended_2', 'lc_frontside_d4_para', 'lc_backside_d4_para',
     'lc_frontside_d3_uptrend', 'lc_backside_d3', 'lc_frontside_d2_uptrend', 'lc_frontside_d2', 'lc_backside_d2', 'lc_fbo']
    
    for col in columns_to_check:
        min_price_col = col + '_min_price'
        if col in df.columns and min_price_col in df.columns:
            # Vectorized condition checks
            condition = (df[col] == 1) & (df['open_next_day'] >= df[min_price_col]) & (df['pm_dol_vol_next_day'] >= 10000000)
            df[col] = condition.astype(int)  # Update the column based on conditions

    df = df[~(df[columns_to_check] == 0).all(axis=1)]

    return df


def check_lc_pm_liquidity(df):
    df.loc[
        ((df['lc_frontside_d3_extended_1'] == 1) | (df['lc_backside_d3_extended_1'] == 1) | (df['lc_frontside_d3_extended_2'] == 1) | (df['lc_backside_d3_extended_2'] == 1) | (df['lc_frontside_d4_para'] == 1) | (df['lc_backside_d4_para'] == 1) | 
         (df['lc_frontside_d3_uptrend'] == 1) | (df['lc_backside_d3'] == 1) | (df['lc_frontside_d2_uptrend'] == 1) | (df['lc_fbo'] == 1)) & 
        (df['valid_pm_liq'] != 1),
        ['lc_frontside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_frontside_d3_extended_2', 'lc_backside_d3_extended_2', 'lc_frontside_d4_para', 'lc_backside_d4_para', 
        'lc_frontside_d3_uptrend', 'lc_backside_d3', 'lc_frontside_d2_uptrend', 'lc_fbo']
        ] = 0
    
    df.loc[
        ((df['lc_frontside_d2'] == 1) | (df['lc_backside_d2'] == 1)) & 
        (df['valid_pm_liq'] == 0),
        ['lc_frontside_d2', 'lc_backside_d2']
        ] = 0
    


    columns_to_check = ['lc_frontside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_frontside_d3_extended_2', 'lc_backside_d3_extended_2', 'lc_frontside_d4_para', 'lc_backside_d4_para',
     'lc_frontside_d3_uptrend', 'lc_backside_d3', 'lc_frontside_d2_uptrend', 'lc_frontside_d2', 'lc_backside_d2', 'lc_fbo']
    
    # Drop rows where all specified columns are 0
    df = df[~(df[columns_to_check] == 0).all(axis=1)]
    
    # df_lc = df_lc[df_lc['valid_pm_liq'] == 1]

    return df

def process_dataframe(func, data):
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        processed_rows = list(executor.map(func, data))
    return pd.DataFrame(processed_rows)


async def main():
    global df_lc, df_sc
    ### Get Main List
    all_results = []
    adj = "true"
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_intial_stock_list(session, date, adj) for date in DATES]
        # results = await asyncio.gather(*tasks)
        # all_results = [result for result in results if result is not None]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # all_results = [result for result in results if isinstance(result, pd.DataFrame)]


        all_results = []
        retry_tasks = []

        # Check results and prepare for retry if necessary
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # print(f"Retrying failed task for date: {DATES[i]} due to {result}")
                retry_tasks.append(fetch_intial_stock_list(session, DATES[i], adj))
            else:
                all_results.append(result)

        # Retry failed tasks
        if retry_tasks:
            retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
            # Merge retry results, assuming they are in the same order as retry_tasks
            for retry_result in retry_results:
                if not isinstance(retry_result, Exception):
                    all_results.append(retry_result)
                else:
                    print(f"Failed after retry: {retry_result}")



    df_a = pd.concat(all_results, ignore_index=True)
    all_results = []
    adj = "false"
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_intial_stock_list(session, date, adj) for date in DATES]
        # results = await asyncio.gather(*tasks)
        # all_results = [result for result in results if result is not None]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # all_results = [result for result in results if isinstance(result, pd.DataFrame)]

        
        all_results = []
        retry_tasks = []

        # Check results and prepare for retry if necessary
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # print(f"Retrying failed task for date: {DATES[i]} due to {result}")
                retry_tasks.append(fetch_intial_stock_list(session, DATES[i], adj))
            else:
                all_results.append(result)

        # Retry failed tasks
        if retry_tasks:
            retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
            # Merge retry results, assuming they are in the same order as retry_tasks
            for retry_result in retry_results:
                if not isinstance(retry_result, Exception):
                    all_results.append(retry_result)
                else:
                    print(f"Failed after retry: {retry_result}")


    df_ua = pd.concat(all_results, ignore_index=True)
    df_ua.rename(columns={col: col + '_ua' if col not in ['date', 'ticker'] else col for col in df_ua.columns}, inplace=True)

    

    print("done 1")

    df = pd.merge(df_a, df_ua, on=['date', 'ticker'], how='inner')
    
    df = df.drop(columns=['vw', 't', 'n', 'vw_ua', 't_ua', 'n_ua'])
    df = df.sort_values(by='date')

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

    ticker_meet_conditions = df.groupby('ticker')['pre_conditions'].any()
    filtered_tickers = ticker_meet_conditions[ticker_meet_conditions].index
    df = df[df['ticker'].isin(filtered_tickers)]

    print(df)
    print("done 2")

    

    df = df.select_dtypes(include=['floating']).round(2).join(df.select_dtypes(exclude=['floating']))



    # df = process_stock_data_grouped(df)    
    df = compute_indicators1(df)
    df = df.sort_values(by='date')
    df = df[df['pre_conditions'] == True]

    

    df = df.select_dtypes(include=['floating']).round(2).join(df.select_dtypes(exclude=['floating']))



    # df = df[(df['date'] >= START_DATE_DT) & (df['date'] <= END_DATE_DT)]
    # df = df.reset_index(drop=True)

    print(df)
    print("done 3")


    df_lc = check_high_lvl_filter_lc(df)

    df = dates_before_after(df)

    print("done 4")

    
    # df = df[(df['date'] >= start_date_70_days_before) & (df['date'] <= END_DATE_DT)]
    # df = df.reset_index(drop=True)



    rows_lc = df_lc.to_dict(orient='records')

    # Use ProcessPoolExecutor to process both dataframes concurrently
    with ProcessPoolExecutor() as executor:
        future_lc = executor.submit(process_dataframe, process_lc_row, rows_lc)

        df_lc = future_lc.result()

    print("done 5")

    # Continue with further processing
    df_lc = get_min_price_lc(df_lc)
    df_lc = check_next_day_valid_lc(df_lc)
    df_lc = check_lc_pm_liquidity(df_lc)

    print("done 6")
    



    df_lc = df_lc[(df_lc['date'] >= START_DATE_DT) & (df_lc['date'] <= END_DATE_DT)]
    df_lc = df_lc.reset_index(drop=True)



    # Output the final dataframes
    print(df_lc)
    df_lc.to_csv("lc_backtest.csv")




        




        
if __name__ == "__main__":
    START_DATE = '2025-05-01'  # Specify the start date
    END_DATE = '2025-06-05'  

    START_DATE_DT = pd.to_datetime(START_DATE)
    END_DATE_DT = pd.to_datetime(END_DATE)

    start_date_70_days_before = pd.Timestamp(START_DATE) - pd.DateOffset(days=70)
    start_date_70_days_before = pd.to_datetime(start_date_70_days_before)

    start_date_300_days_before = pd.Timestamp(START_DATE) - pd.DateOffset(days=400)
    start_date_300_days_before = str(start_date_300_days_before)[:10]
                                                                          
    schedule = nyse.schedule(start_date=start_date_300_days_before, end_date=END_DATE)
    DATES = nyse.valid_days(start_date=start_date_300_days_before, end_date=END_DATE)
    DATES = [date.strftime('%Y-%m-%d') for date in nyse.valid_days(start_date=start_date_300_days_before, end_date=END_DATE)]


    asyncio.run(main())
    # asyncio.run(main())

    print("Getting Stocks")

