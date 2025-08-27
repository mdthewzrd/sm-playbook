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
import subprocess

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



def widen_columns(content, width=4):
    """
    Adds padding to the content of each column to widen the gap between columns.
    """
    return "\n".join(f"{line}{'    ' * width}" for line in content.split("\n"))
def add_vertical_dividers(dataframes):
    """
    Adds vertical dividers (as a pipe `|` separator) between sub-tables (dataframes) in a category.
    """
    divider = "\n" + "|" + ("-" * 48) + "|\n"  # Customize divider length if needed
    return divider.join(dataframes)
def show_interactive_tables(table1_rows, table2_rows, headers):
    # Convert rows into pandas DataFrame
    df1 = pd.DataFrame(table1_rows, columns=headers)
    df2 = pd.DataFrame(table2_rows, columns=headers)
    
    # Create Plotly tables
    fig = go.Figure(data=[go.Table(
        header=dict(values=headers),
        cells=dict(values=[df1[col] for col in headers])
    )])
    
    fig.update_layout(title="First Scan Table")
    
    # Save to HTML
    fig.write_html("interactive_table_1.html")
    
    # Now creating second table
    fig2 = go.Figure(data=[go.Table(
        header=dict(values=headers),
        cells=dict(values=[df2[col] for col in headers])
    )])

    fig2.update_layout(title="Second Scan Table")
    
    # Save second table
    fig2.write_html("interactive_table_2.html")
    
    # Combine them in one HTML file
    with open("combined_interactive_tables.html", "w") as f:
        f.write("<h1>First Scan Table</h1>")
        with open("interactive_table_1.html", "r") as t1:
            f.write(t1.read())
        
        f.write("<h1>Second Scan Table</h1>")
        with open("interactive_table_2.html", "r") as t2:
            f.write(t2.read())
    
    # Open the combined HTML in the default web browser
    webbrowser.open('file://' + "combined_interactive_tables.html")
def show_combined_scans(table1, table2):
    # Generate the first table
    # table1 = tabulate.tabulate(table1_rows, headers=headers, tablefmt="grid")
    
    # # Generate the second table
    # table2 = tabulate.tabulate(table2_rows, headers=headers, tablefmt="grid")

    # Combine both tables into one HTML content
    html_content = f"""
    <html>
    <head><title>Combined Scan Tables</title></head>
    <body>
        <h1>LC Scan</h1>
        <pre>{table1}</pre>
        <h1>SC Scan</h1>
        <pre>{table2}</pre>
    </body>
    </html>
    """
    
    # Save the combined tables to an HTML file
    html_file = "scans.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # time.sleep(2)
    # Open the HTML file in the default web browser
    webbrowser.open(f'file://{html_file}')
def show_scans(df, trigger_columns):
    # Initialize a dictionary to store the results for each trigger
    results = {}

    # Iterate over trigger columns
    for trigger in trigger_columns:
        trigger_min_price_col = f"{trigger}_min_price"
        trigger_min_pct_col = f"{trigger}_min_pct"

        # Check if the required columns exist in the DataFrame
        if trigger in df.columns and trigger_min_price_col in df.columns and trigger_min_pct_col in df.columns and "lc_" in trigger and "d2_high_volume" not in trigger:
            # Filter rows where the trigger is 1
            filtered_df = df[df[trigger] == 1]
            
            # Select and rename columns
            result_df = filtered_df[['date', 'ticker', 'pct_change', 'high_chg_atr', 'v', 'dol_v', 'avg_5d_pm_dol_vol', 'atr', trigger_min_price_col, trigger_min_pct_col]].rename(
                columns={
                    'date': 'Date',
                    'ticker': 'Ticker',
                    'v': 'Volume (m)',
                    'dol_v': '$ Volume (m)',
                    'pct_change': 'Gain %',
                    'high_chg_atr': 'ATR Expansion',
                    'atr': 'ATR',
                    'avg_5d_pm_dol_vol': 'Avg. PM $ Vol (m)',
                    trigger_min_price_col: 'Min Gap',
                    trigger_min_pct_col: 'Min Gap %'
                }
            )

            
            result_df['Avg. PM $ Vol (m)'] = result_df['Avg. PM $ Vol (m)']/1000000
            result_df['Volume (m)'] = result_df['Volume (m)']/1000000
            result_df['$ Volume (m)'] = result_df['$ Volume (m)']/1000000

            result_df = result_df.round(2)

            # Add '%' suffix to the 'Gain %' and 'Min Gap %' columns
            result_df['Gain %'] = result_df['Gain %'].astype(str) + '%'
            result_df['ATR Expansion'] = result_df['ATR Expansion'].astype(str) + 'x'
            result_df['Min Gap %'] = result_df['Min Gap %'].astype(str) + '%'
            result_df['Avg. PM $ Vol (m)'] = result_df['Avg. PM $ Vol (m)'].astype(str) + 'm'
            result_df['Volume (m)'] = result_df['Volume (m)'].astype(str) + 'm'
            result_df['$ Volume (m)'] = result_df['$ Volume (m)'].astype(str) + 'm'

            # Store the result
            results[trigger] = result_df

        if trigger in df.columns and trigger_min_price_col in df.columns and trigger_min_pct_col in df.columns and ("lc_" not in trigger or "d2_high_volume" in trigger):
            # Filter rows where the trigger is 1
            filtered_df = df[df[trigger] == 1]
            
            # Select and rename columns
            result_df = filtered_df[['date', 'ticker', 'pct_change', 'high_chg_atr', 'v', 'dol_v', 'atr', trigger_min_price_col, trigger_min_pct_col]].rename(
                columns={
                    'date': 'Date',
                    'ticker': 'Ticker',
                    'v': 'Volume (m)',
                    'dol_v': '$ Volume (m)',
                    'pct_change': 'Gain %',
                    'high_chg_atr': 'ATR Expansion',
                    'atr': 'ATR',
                    trigger_min_price_col: 'Min Gap',
                    trigger_min_pct_col: 'Min Gap %'
                }
            )

            result_df['Volume (m)'] = result_df['Volume (m)']/1000000
            result_df['$ Volume (m)'] = result_df['$ Volume (m)']/1000000

            result_df = result_df.round(2)

            # Add '%' suffix to the 'Gain %' and 'Min Gap %' columns
            result_df['Gain %'] = result_df['Gain %'].astype(str) + '%'
            result_df['ATR Expansion'] = result_df['ATR Expansion'].astype(str) + 'x'
            result_df['Min Gap %'] = result_df['Min Gap %'].astype(str) + '%'
            result_df['Volume (m)'] = result_df['Volume (m)'].astype(str) + 'm'
            result_df['$ Volume (m)'] = result_df['$ Volume (m)'].astype(str) + 'm'

            # Store the result
            results[trigger] = result_df


    categories = {
        "LC FRONTSIDE": [],
        "LC BACKSIDE": [],
        "SC FRONTSIDE": [],
        "SC BACKSIDE": []
    }

    # Categorize results
    for trigger, result_df in results.items():
        

        if len(result_df) > 0:
            if "lc_" in trigger:
                if "frontside_" in trigger  or "fbo" in trigger:#  or "lc_d2_high_volume" in trigger:
                    categories["LC FRONTSIDE"].append((trigger, result_df))
                elif "backside_" in trigger or "t30" in trigger:
                    categories["LC BACKSIDE"].append((trigger, result_df))
            elif "sc_" in trigger:
                if "frontside_" in trigger:
                    categories["SC FRONTSIDE"].append((trigger, result_df))
                elif "backside_" in trigger or "t30" in trigger:
                    categories["SC BACKSIDE"].append((trigger, result_df))

    # Print results under categorized headings
    for category, triggers in categories.items():
        if triggers:
            print(f"{category}:")
            print("=====================")# * len(category))
            print()
            for trigger, result_df in triggers:
                formatted_trigger = trigger.replace('_', ' ').upper()
                result_df = result_df.reset_index(drop=True)
                print(f"{formatted_trigger}:")
                print("-------------------------")
                print(result_df)
                print()
            print()
            print()

    
    category_tables = []
    for category, triggers in categories.items():
        if triggers:
            category_data = []
            for trigger, result_df in triggers:
                formatted_trigger = trigger.replace('_', ' ').upper()
                result_df = result_df.reset_index(drop=True)
                
            #     category_data.append(f"{formatted_trigger}:\n{result_df.to_string(index=False)}")
            # # category_tables.append((category, "\n\n".join(category_data)))
            # category_tables.append((category, add_vertical_dividers(category_data)))

                formatted_table = tabulate(result_df, headers="keys", tablefmt="fancy_grid", showindex=False)
                
                # Add the trigger title
                category_data.append(f"{formatted_trigger}:\n{formatted_table}")
            # Combine all tables for this category
            category_tables.append((category, "\n\n".join(category_data)))

            #     formatted_table = result_df.to_string(index=False)
            #     padded_table = widen_columns(f"{formatted_trigger}:\n{formatted_table}")
            #     category_data.append(padded_table)
            # category_tables.append((category, "\n\n".join(category_data)))
    # Display categories side by side

    # headers = [table[0] for table in category_tables]
    # rows = zip(*[table[1].split("\n\n") for table in category_tables])

    # table = tabulate(rows, headers=headers, tablefmt="grid")

    headers = [table[0] for table in category_tables]

    # Process rows carefully
    rows = []
    max_rows = max(len(table[1].split("\n\n")) for table in category_tables)  # Get max rows for alignment

    for i in range(max_rows):
        row = []
        for table in category_tables:
            split_rows = table[1].split("\n\n")
            # Add row data or empty string if this table has fewer rows
            row.append(split_rows[i] if i < len(split_rows) else "")
        rows.append(row)

    # Debug final rows structure
    # print("Debug: rows =", rows)

    # Generate the final tabulated table
    table = tabulate(rows, headers=headers, tablefmt="grid")
    # print(table)



    # html_table = f"<pre>{table}</pre>"  # Wrap it in <pre> tags for formatting
    # html_file = "table.html"

    # with open(html_file, "w", encoding="utf-8") as f:
    #     f.write(html_table)

    # # Open the HTML file in the default web browser
    # webbrowser.open(f'file://{html_file}')

    return table

    # print(tabulate(rows, headers=headers, tablefmt="grid"))


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

def check_high_lvl_filter_sc(df):

    # df = df1.tail(1)

    df['d2_mdr'] = ((df['change_percent'] >= .2) & 
                       (df['c_ua'] >= 1) & 
                       (df['v_ua'] >= 10000000) & 
                       (df['20d_pct_change'] >= 3) & 
                       (df['h'] > df['high_1']) & 
                       (df['c'] > df['o']) & 
                       (df['had_30pct_move_on_10m_volume_last_20rows'] == True) &
                       (df['h'] >= df['highest_high_20'])#&
                    #    (df['total_dol_vol_20'] >= 2000000000)
                       ).astype(int)
    

    columns_to_check = ['d2_mdr']

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
        

def get_min_price_sc(df):

    ### SC Min Price

    df['sc_frontside_d3_extended_1_min_price'] = round((df['c'] + df['atr']*1), 2)
    df['sc_backside_d3_extended_1_min_price'] = round((df['c'] + df['atr']*1), 2)

    df['sc_frontside_d3_extended_2_min_price'] = round(np.maximum(
            df['c'] + df['atr']*1, df['ema50'] + df['atr']*5
        ), 2)
    df['sc_backside_d3_extended_2_min_price'] = round(np.maximum(
            df['c'] + df['atr']*1, df['ema50'] + df['atr']*5
        ), 2)
        
    df['sc_frontside_d3_para_min_price'] = round((df['c'] + df['atr']*0.5), 2)
    df['sc_backside_d3_para_min_price'] = round((df['c'] + df['atr']*0.5), 2)


    df['sc_frontside_d2_min_price'] = round(np.maximum(
            df['h'] + df['atr']*1, df['h'] + df['high_chg']*0.3
        ), 2)
    df['sc_backside_d2_min_price'] = round(np.maximum(
            df['h'] + df['atr']*1, df['h'] + df['high_chg']*0.3
        ), 2)
        
    df['sc_frontside_d2_trap_extended_min_price'] = round((df['c'] + df['atr']*1), 2)
    df['sc_backside_d2_trap_extended_min_price'] = round((df['c'] + df['atr']*1), 2)

    df['sc_frontside_d2_extended_min_price'] = round((df['c'] + df['atr']*1), 2)
    df['sc_backside_d2_extended_min_price'] = round((df['c'] + df['atr']*1), 2)

    df['sc_frontside_2nd_leg_min_price'] = round(np.maximum(
            df['h'] + df['atr']*0.5, df['c'] + df['atr']*1
        ), 2)
    df['sc_backside_2nd_leg_min_price'] = round(np.maximum(
            df['h'] + df['atr']*0.5, df['c'] + df['atr']*1
        ), 2)

    
    df['sc_frontside_d2_high_volume_min_price'] = round((df['c'] + df['atr']*3), 2)
    df['sc_backside_d2_high_volume_min_price'] = round((df['c'] + df['atr']*3), 2)
    
    columns_to_check = ['sc_frontside_d3_extended_1', 'sc_backside_d3_extended_1', 'sc_frontside_d3_extended_2', 'sc_backside_d3_extended_2', 'sc_frontside_d3_para', 'sc_backside_d3_para', 
                        'sc_frontside_d2', 'sc_backside_d2', 'sc_frontside_d2_trap_extended', 'sc_backside_d2_trap_extended', 'sc_frontside_d2_extended', 'sc_backside_d2_extended', 
                        'sc_frontside_2nd_leg', 'sc_backside_2nd_leg', 'sc_frontside_d2_high_volume', 'sc_backside_d2_high_volume']

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

def get_prev_high(x):
    highs = x['h'].to_numpy()
    cond = x['meets_condition'].to_numpy()
    result = np.full(len(x), np.nan)

    last_high = np.nan
    for i in range(len(x)):
        result[i] = last_high
        if cond[i]:  # only update AFTER assigning
            last_high = highs[i]
    return pd.Series(result, index=x.index)

def compute_indicators1(df):
    # Sorting by 'ticker' and 'date' to respect chronological order for each ticker
    df = df.sort_values(by=['ticker', 'date'])
    
    # Calculating previous day's close
    df['pdc'] = df.groupby('ticker')['c'].shift(1)

    df['high_1'] = df.groupby('ticker')['h'].shift(1)

    df['change_percent'] = df['c'] / df['pdc'] - 1

    # df['low_20'] = df.groupby('ticker')['l'].transform(lambda x: x.rolling(20).min())
    df['low_20'] = df.groupby('ticker')['l'].transform(lambda x: x.rolling(window=20, min_periods=1).min())

    df['20d_pct_change'] = df['h']/df['low_20'] - 1
    df['high_1'] = df.groupby('ticker')['h'].shift(1)


    df['dol_vol'] = df['v'] * df['c']
    df['meets_condition'] = (df['change_percent'] >= .2) & (df['dol_vol'] >= 100000000) & (df['c'] > df['o']) & (df['h'] > df['high_1'])

    df['had_30pct_move_on_10m_volume_last_20rows'] = (
    df.groupby('ticker')['meets_condition']
      .transform(lambda x: x.shift(1).rolling(window=20, min_periods=1).max().fillna(False))
      .astype(bool)
    )

    df['last_condition_high'] = df.groupby('ticker', group_keys=False).apply(get_prev_high)
    
    df['highest_high_20'] = df.groupby('ticker')['h'].transform(lambda x: x.rolling(20, min_periods=1).max())


    # df['highest_high_20'] = (
    #     df.groupby('ticker')['highest_high_20_1']
    #     .shift(1)
    # )

    df['total_dol_vol_20'] = (
        df.groupby('ticker')['dol_vol']
        .transform(lambda x: x.rolling(window=20, min_periods=1).sum())
    )

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
    date_plus_30 = trading_days_list[idx + 30] if idx + 30 < len(trading_days_list) else pd.NaT
    date_minus_1 = trading_days_list[idx - 1] if idx - 1 >= 0 else pd.NaT
    return date_plus_1, date_minus_4, date_minus_30, date_minus_1, date_plus_30





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


def process_sc_row(row):
    """Process a single row for intraday data and calculations."""
    try:
        ticker = row['ticker']
        date = row['date']

        date_plus_1 = row['date_plus_1']

        intraday_data = fetch_intraday_data(ticker, date_plus_1, date_plus_1)
        intraday_data_after = adjust_intraday(intraday_data)

                    

        pm_df = intraday_data_after[intraday_data_after['time_int'] <= 900].set_index('time')
        open_df = intraday_data_after[intraday_data_after['time_int'] <= 930].set_index('time')

        if not pm_df.empty:
            resp_v = pm_df['v_sum'][-1]
            resp_o = open_df['o'][-1]
            pm_dol_vol_next_day = resp_o * resp_v
            pmh_next_day = pm_df['hod_all'][-1]
        else:
            resp_o = 0
            pm_dol_vol_next_day = 0
            pmh_next_day = 0
        
        row['next_open'] = resp_o
        row['pmh_next_day'] = pmh_next_day
        row['pm_dol_vol_next_day'] = pm_dol_vol_next_day

    except Exception as e:
        print(f"Error processing SC Row {row['ticker']} on {row['date']}: {e}")
        row['next_open'] = 0
        row['pmh_next_day'] = 0
        row['pm_dol_vol_next_day'] = 0

    return row

def dates_before_after(df):
    global trading_days_map, trading_days_list

    start_date = df['date'].min() - pd.Timedelta(days=60)
    end_date = df['date'].max() + pd.Timedelta(days=60)
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = pd.Series(schedule.index, index=schedule.index)

    # Map trading days for faster lookup
    trading_days_list = trading_days.index.to_list()
    trading_days_map = {day: idx for idx, day in enumerate(trading_days_list)}

    results = df_sc['date'].map(get_offsets)
    df_sc[['date_plus_1', 'date_minus_4', 'date_minus_30', 'date_minus_1', 'date_plus_30']] = pd.DataFrame(results.tolist(), index=df_sc.index)

    # Format dates as strings if needed
    df_sc['date_plus_1'] = df_sc['date_plus_1'].dt.strftime('%Y-%m-%d')
    df_sc['date_minus_4'] = df_sc['date_minus_4'].dt.strftime('%Y-%m-%d')
    df_sc['date_minus_30'] = df_sc['date_minus_30'].dt.strftime('%Y-%m-%d')
    df_sc['date_minus_1'] = df_sc['date_minus_1'].dt.strftime('%Y-%m-%d')
    df_sc['date_plus_30'] = df_sc['date_plus_30'].dt.strftime('%Y-%m-%d')

    return df

def check_next_day_valid_sc(df):
    # Ensure minimum price columns exist for each check column
    columns_to_check = ['sc_frontside_d3_extended_1', 'sc_backside_d3_extended_1', 'sc_frontside_d3_extended_2', 'sc_backside_d3_extended_2', 'sc_frontside_d3_para', 'sc_backside_d3_para', 
                        'sc_frontside_d2', 'sc_backside_d2', 'sc_frontside_d2_trap_extended', 'sc_backside_d2_trap_extended', 'sc_frontside_d2_extended', 'sc_backside_d2_extended', 
                        'sc_frontside_2nd_leg', 'sc_backside_2nd_leg', 'sc_frontside_d2_high_volume', 'sc_backside_d2_high_volume']
    
    for col in columns_to_check:
        min_price_col = col + '_min_price'
        if col in df.columns and min_price_col in df.columns:
            # Vectorized condition checks
            condition = (df[col] == 1) & (df['pmh_next_day'] >= df[min_price_col]) & (df['pm_dol_vol_next_day'] >= 10000000)
            df[col] = condition.astype(int)  # Update the column based on conditions

    df = df[~(df[columns_to_check] == 0).all(axis=1)]

    return df

async def fetch_stock_type(session, ticker, API_KEY):
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={API_KEY}"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return ticker, data.get('results', {}).get('type', 'Unknown')
            else:
                return ticker, 'API Error'
    except Exception as e:
        return ticker, f'Error: {e}'

async def get_all_stock_types(tickers, API_KEY):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_stock_type(session, ticker, API_KEY) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results


async def update_df_with_stock_types(df, API_KEY):
    tickers = df['ticker'].tolist()
    results = await get_all_stock_types(tickers, API_KEY)
    
    # Convert results to dictionary
    type_map = {ticker: stock_type for ticker, stock_type in results}
    df['stock_type'] = df['ticker'].map(type_map)
    return df

def process_dataframe(func, data):
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        processed_rows = list(executor.map(func, data))
    return pd.DataFrame(processed_rows)


async def main():
    global df_sc
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
        (df['c_ua'] >= 1) &
        (df['v_ua'] >= 10000000) &
        (df['dol_v'] >= 100000000) &
        (df['c'] > df['o'])
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


    df_sc = check_high_lvl_filter_sc(df)
    df = dates_before_after(df)

    print("done 4")

    
    # df = df[(df['date'] >= start_date_70_days_before) & (df['date'] <= END_DATE_DT)]
    # df = df.reset_index(drop=True)




    df_sc = df_sc[(df_sc['date'] >= START_DATE_DT) & (df_sc['date'] <= END_DATE_DT)]
    df_sc = df_sc.reset_index(drop=True)


    df_sc = await update_df_with_stock_types(df_sc, API_KEY)

    df_sc = df_sc[~df_sc['stock_type'].isin(["ETF", "ETS", "ETV"])]


    df_sc = df_sc.reset_index(drop=True)

    # df_sc = df_sc.rename(columns={
    # 'date': 'date_minus_1',
    # 'date_plus_1': 'date'
    # })

    # Reorder columns
    cols = df_sc.columns.tolist()
    # Move 'ticker' and 'date' to the front
    new_order = ['ticker', 'date'] + [col for col in cols if col not in ['ticker', 'date']]
    df_sc = df_sc[new_order]


    print(df_sc)
    df_sc.to_csv("d2_mdr_backtest.csv")

    
    # save_data = "D:/TRADING/Backtesting/SC D2/save d2 data.py"

    # subprocess.run(['python', save_data], check=True)


    print("done")




        
if __name__ == "__main__":
    START_DATE = '2025-08-01'  # Specify the start date
    END_DATE = '2025-08-13'  

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


    '''
    
    trigger_columns = ['lc_frontside_d4_para', 'lc_backside_d4_para', 'lc_frontside_d3_extended_1', 'lc_backside_d3_extended_1', 'lc_frontside_d3_extended_2', 'lc_backside_d3_extended_2', 
                        'lc_frontside_d3_uptrend', 'lc_backside_d3', 'lc_frontside_d2_uptrend', 'lc_backside_d2', 'lc_frontside_d2_high_volume', 'lc_backside_d2_high_volume', 'lc_fbo', 'lc_t30_d2', 'lc_t30_d3']
    table_lc = show_scans(df_lc, trigger_columns)

    # print(table_lc)
    

    trigger_columns = ['sc_frontside_d3_para', 'sc_backside_d3_para', 'sc_frontside_d3_extended_1', 'sc_backside_d3_extended_1', 'sc_frontside_d3_extended_2', 'sc_backside_d3_extended_2', 
                        'sc_frontside_d2_trap_extended', 'sc_backside_d2_trap_extended', 'sc_frontside_2nd_leg', 'sc_backside_2nd_leg', 'sc_frontside_d2_extended', 'sc_backside_d2_extended', 
                        'sc_frontside_d2_high_volume', 'sc_backside_d2_high_volume', 'sc_frontside_d2', 'sc_backside_d2', 'sc_t30_d2']
    table_sc = show_scans(df_sc, trigger_columns)

    show_combined_scans(table_lc, table_sc)
                        

    #'''
    # df_sc_in_play = df_sc[['date', 'ticker', 'sc_frontside_d3_extended_1', 'sc_frontside_d3_extended_2', 'sc_frontside_d3_para', 'sc_d2_gap', 'sc_trap_blowout', 'sc_d2_mdr', 'sc_2nd_leg']]
    # df_lc_in_play = df_lc1[['date', 'ticker', 'lc_frontside_d3_extended_1', 'lc_frontside_d3_extended_2', 'lc_frontside_d4_para', 'd2_big_day_volume', 'lc_frontside_d3_uptrend', 'lc_frontside_d2_uptrend', 'lc_fbo', 'lc_t30_d2', 'lc_t30_d3', 'valid_t30']]


    # print(df_sc_in_play)
    # print(df_lc_in_play)

