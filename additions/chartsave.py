import matplotlib
matplotlib.use("Qt5Agg")

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
from mplfinance.original_flavor import candlestick_ohlc
import pandas_market_calendars as mcal
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo

plt.style.use('dark_background')

def generate_and_save_chart(TICKER, TARGET_DATE, API_KEY):
    TICKER = TICKER.upper()
    today = pd.Timestamp(datetime.now().date())

    def fetch_polygon_agg(ticker, multiplier, timespan, from_date, to_date, api_key):
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
        response = requests.get(url, params=params)
        data = response.json()
        if "results" not in data:
            raise ValueError("Polygon API error or no data found")
        df = pd.DataFrame(data["results"])
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("US/Eastern")
        df.set_index("timestamp", inplace=True)
        df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def add_bands(df, ema_short, ema_long, du1, du2, dl1, dl2, prefix=""):
        df[f"{prefix}emaS"] = df["Close"].ewm(span=ema_short, min_periods=0).mean()
        df[f"{prefix}emaL"] = df["Close"].ewm(span=ema_long, min_periods=0).mean()
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs()
        ], axis=1).max(axis=1)
        df[f"{prefix}ATR_S"] = tr.rolling(ema_short, min_periods=1).mean()
        df[f"{prefix}ATR_L"] = tr.rolling(ema_long, min_periods=1).mean()
        df[f"{prefix}UPP1"] = df[f"{prefix}emaS"] + du1 * df[f"{prefix}ATR_S"]
        df[f"{prefix}UPP2"] = df[f"{prefix}emaS"] + du2 * df[f"{prefix}ATR_S"]
        df[f"{prefix}LOW1"] = df[f"{prefix}emaL"] - dl1 * df[f"{prefix}ATR_L"]
        df[f"{prefix}LOW2"] = df[f"{prefix}emaL"] - dl2 * df[f"{prefix}ATR_L"]

    def compress_market_time(df, target_date):
        df = df.copy()
        df = df[((df.index.time >= pd.to_datetime("04:00").time()) & (df.index.time < pd.to_datetime("09:30").time())) |
                ((df.index.time >= pd.to_datetime("09:30").time()) & (df.index.time <= pd.to_datetime("20:00").time()))]
        df = df[df.index.dayofweek < 5]
        df = df[df.index.date == target_date.date()]
        df["timestamp_est"] = df.index.tz_convert("US/Eastern")
        return df

    def remove_fake_wicks(df):
        return df[~((df["High"] == df["Low"]) & (df["Open"] == df["High"]) & (df["Close"] == df["High"]))]

    start_target = pd.to_datetime(TARGET_DATE)
    prev_day = start_target - timedelta(days=5)
    next_day = start_target + timedelta(days=1)
    daily_start = (TARGET_DATE - pd.Timedelta(days=50)).strftime("%Y-%m-%d")
    daily_end = TARGET_DATE.strftime("%Y-%m-%d")

    df_2m_raw = remove_fake_wicks(fetch_polygon_agg(TICKER, 2, "minute", prev_day.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d"), API_KEY))
    df_5m_raw = remove_fake_wicks(fetch_polygon_agg(TICKER, 5, "minute", prev_day.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d"), API_KEY))
    df_15m_raw = remove_fake_wicks(fetch_polygon_agg(TICKER, 15, "minute", prev_day.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d"), API_KEY))
    df_daily = remove_fake_wicks(fetch_polygon_agg(TICKER, 1, "day", daily_start, daily_end, API_KEY))
    add_bands(df_15m_raw, 72, 89, 6.9, 9.6, 4.2, 5.5, prefix="l_")
    add_bands(df_5m_raw, 72, 89, 6.9, 9.6, 4.2, 5.5, prefix="l_")
    add_bands(df_15m_raw, 9, 20, 1.0, 0.5, 2.0, 2.5, prefix="s_")
    add_bands(df_5m_raw, 9, 20, 1.0, 0.5, 2.0, 2.5, prefix="s_")
    add_bands(df_2m_raw, 9, 20, 1.0, 0.5, 2.0, 2.5, prefix="q_")
    df_daily["ema9"] = df_daily["Close"].ewm(span=9, min_periods=0).mean()
    df_daily["ema20"] = df_daily["Close"].ewm(span=20, min_periods=0).mean()
    #df_5m_raw["EMA_161"] = df_5m_raw["Close"].ewm(span=161, adjust=False, min_periods=0).mean()
    #df_5m_raw["EMA_222"] = df_5m_raw["Close"].ewm(span=222, adjust=False, min_periods=0).mean()

    df_2m = compress_market_time(df_2m_raw, TARGET_DATE)
    df_5m = compress_market_time(df_5m_raw, TARGET_DATE)
    df_15m = compress_market_time(df_15m_raw, TARGET_DATE)

    #df_5m["EMA_161"] = df_5m_raw.loc[df_5m.index, "EMA_161"].values
    #df_5m["EMA_222"] = df_5m_raw.loc[df_5m.index, "EMA_222"].values

    for df in [df_5m, df_2m, df_15m]:
        df["date_only"] = df.index.date
        df["cum_vol"] = df.groupby("date_only")["Volume"].cumsum()
        df["cum_vol_price"] = (df["Close"] * df["Volume"]).groupby(df["date_only"]).cumsum()
        df["VWAP"] = df["cum_vol_price"] / df["cum_vol"]

    df_daily = df_daily.copy()
    df_daily = df_daily[df_daily.index.date <= TARGET_DATE.date()]  # Keep only dates up to and including target
    df_daily["timestamp_est"] = df_daily.index.tz_convert("US/Eastern")

    output_dir = os.path.expanduser("~/Desktop/lc setups")
    os.makedirs(output_dir, exist_ok=True)

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8), constrained_layout=True)

    visible_high = df_5m["High"].max()
    visible_low = df_5m["Low"].min()
    ax1.set_ylim(visible_low * 0.995, visible_high * 1.005)
    start_time = pd.Timestamp(f"{TARGET_DATE.date()} 07:00:00", tz="US/Eastern")
    end_time = pd.Timestamp(f"{TARGET_DATE.date()} 17:00:00", tz="US/Eastern")
    ax1.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))


    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=ZoneInfo("US/Eastern")))
    for i in range(len(df_5m) - 1):
        t1 = df_5m.index[i]
        if pd.to_datetime("04:00").time() <= t1.time() < pd.to_datetime("09:30").time():
            ax1.axvspan(mdates.date2num(df_5m["timestamp_est"].iloc[i]), mdates.date2num(df_5m["timestamp_est"].iloc[i + 1]), color="#444444", alpha=0.6)
        if pd.to_datetime("16:00").time() <= t1.time() < pd.to_datetime("20:00").time():
            ax1.axvspan(mdates.date2num(df_5m["timestamp_est"].iloc[i]), mdates.date2num(df_5m["timestamp_est"].iloc[i + 1]), color="#333333", alpha=0.5)

    candlestick_ohlc(ax1, list(zip(mdates.date2num(df_5m["timestamp_est"]), df_5m["Open"], df_5m["High"], df_5m["Low"], df_5m["Close"])),
                     width=0.0017, colorup="white", colordown="red")
    for col1, col2, color, alpha in [
        ("s_UPP1", "s_UPP2", "pink", 0.3),
        ("s_LOW1", "s_LOW2", "cyan", 0.3),
        ("l_UPP1", "l_UPP2", "orange", 0.25),
        ("l_LOW1", "l_LOW2", "lime", 0.25)
    ]:

        ax1.fill_between(mdates.date2num(df_5m["timestamp_est"]), df_5m[col1], df_5m[col2], color=color, alpha=alpha)
    #ax1.fill_between(mdates.date2num(df_5m["timestamp_est"]), df_5m["EMA_161"], df_5m["EMA_222"],
     #                color="gold", alpha=0.3, label="J-Line (EMA 161/222)")

    for col, style, label, color in [("s_emaS", "--", "EMA9", "yellow"), ("s_emaL", "--", "EMA20", "lightskyblue"), ("VWAP", "--", "VWAP", "purple")]:
        ax1.plot(mdates.date2num(df_5m["timestamp_est"]), df_5m[col], linestyle=style, label=label, color=color)
    ax1.set_title(f"{TICKER} {TARGET_DATE.date()} 5M")
    ax1.legend()

    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=ZoneInfo("US/Eastern")))

    start_time = pd.Timestamp(f"{TARGET_DATE.date()} 07:00:00", tz="US/Eastern")
    end_time = pd.Timestamp(f"{TARGET_DATE.date()} 16:00:00", tz="US/Eastern")
    ax2.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))


    for i in range(len(df_2m) - 1):
        t1 = df_2m.index[i]
        if pd.to_datetime("04:00").time() <= t1.time() < pd.to_datetime("09:30").time():
            ax2.axvspan(mdates.date2num(df_2m["timestamp_est"].iloc[i]), mdates.date2num(df_2m["timestamp_est"].iloc[i + 1]), color="#444444", alpha=0.6)
        if pd.to_datetime("16:00").time() <= t1.time() < pd.to_datetime("20:00").time():
            ax2.axvspan(mdates.date2num(df_2m["timestamp_est"].iloc[i]), mdates.date2num(df_2m["timestamp_est"].iloc[i + 1]), color="#333333", alpha=0.5)

    candlestick_ohlc(ax2, list(zip(mdates.date2num(df_2m["timestamp_est"]), df_2m["Open"], df_2m["High"], df_2m["Low"], df_2m["Close"])),
                     width=0.0007, colorup="white", colordown="red")
    for col1, col2, color, alpha in [
        ("q_UPP1", "q_UPP2", "red", 0.4),
        ("q_LOW1", "q_LOW2", "lightgreen", 0.4)
    ]:
        ax2.fill_between(mdates.date2num(df_2m["timestamp_est"]), df_2m[col1], df_2m[col2], color=color, alpha=alpha)
    for col, style, label, color in [("q_emaS", "--", "EMA9", "yellow"), ("q_emaL", "--", "EMA20", "lightskyblue"), ("VWAP", "--", "VWAP", "purple")]:
        ax2.plot(mdates.date2num(df_2m["timestamp_est"]), df_2m[col], linestyle=style, label=label, color=color)
    ax2.set_title(f"{TICKER} {TARGET_DATE.date()} 2M")
    ax2.legend()

    fig1.savefig(os.path.join(output_dir, f"{TICKER}_{TARGET_DATE.date()}_1.png"))
    plt.close(fig1)

    # Skipping daily and 15m chart here intentionally to stay within length limit. Let me know to continue.
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(22, 8), constrained_layout=True)

    ax3.xaxis_date()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    candlestick_ohlc(ax3, list(zip(mdates.date2num(df_daily["timestamp_est"]), df_daily["Open"], df_daily["High"], df_daily["Low"], df_daily["Close"])),
                     width=0.5, colorup="white", colordown="red")
    ax3.plot(mdates.date2num(df_daily["timestamp_est"]), df_daily["ema9"], linestyle="--", color="yellow", label="EMA9")
    ax3.plot(mdates.date2num(df_daily["timestamp_est"]), df_daily["ema20"], linestyle="--", color="lightskyblue", label="EMA20")
    ax3.set_title("Daily Chart")
    ax3.legend()

    visible_high_15m = df_15m["High"].max()
    visible_low_15m = df_15m["Low"].min()
    ax4.set_ylim(visible_low_15m * 0.995, visible_high_15m * 1.005)

    ax4.xaxis_date()
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=ZoneInfo("US/Eastern")))

    for i in range(len(df_15m) - 1):
        t1 = df_15m.index[i]
        if pd.to_datetime("04:00").time() <= t1.time() < pd.to_datetime("09:30").time():
            ax4.axvspan(mdates.date2num(df_15m["timestamp_est"].iloc[i]), mdates.date2num(df_15m["timestamp_est"].iloc[i + 1]), color="#444444", alpha=0.6)
        if pd.to_datetime("16:00").time() <= t1.time() < pd.to_datetime("20:00").time():
            ax4.axvspan(mdates.date2num(df_15m["timestamp_est"].iloc[i]), mdates.date2num(df_15m["timestamp_est"].iloc[i + 1]), color="#333333", alpha=0.5)

    candlestick_ohlc(ax4, list(zip(mdates.date2num(df_15m["timestamp_est"]), df_15m["Open"], df_15m["High"], df_15m["Low"], df_15m["Close"])),
                     width=0.005, colorup="white", colordown="red")
    for col1, col2, color, alpha in [
        ("s_UPP1", "s_UPP2", "pink", 0.3),
        ("s_LOW1", "s_LOW2", "cyan", 0.3),
        ("l_UPP1", "l_UPP2", "orange", 0.25),
        ("l_LOW1", "l_LOW2", "lime", 0.25)
    ]:
        ax4.fill_between(mdates.date2num(df_15m["timestamp_est"]), df_15m[col1], df_15m[col2], color=color, alpha=alpha)
    for col, style, label, color in [("s_emaS", "--", "EMA9", "yellow"), ("s_emaL", "--", "EMA20", "lightskyblue"), ("VWAP", "--", "VWAP", "purple")]:
        ax4.plot(mdates.date2num(df_15m["timestamp_est"]), df_15m[col], linestyle=style, label=label, color=color)
    ax4.set_title(f"{TICKER} {TARGET_DATE.date()} 15M")
    ax4.legend()

    fig2.savefig(os.path.join(output_dir, f"{TICKER}_{TARGET_DATE.date()}_2.png"))
    plt.close(fig2)

API_KEY = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
ticker_dates = [
    ("MSTR", "2024-10-14"),
    ("ORCL", "2024-12-09"),
    ("COIN", "2024-12-05"),
    ("MSTR", "2024-09-30"),
    ("NVDA", "2024-07-11"),
    ("DELL", "2024-06-20"),
    ("CLSK", "2024-03-11"),

]

for ticker, date_str in ticker_dates:
    try:
        generate_and_save_chart(ticker, pd.to_datetime(date_str), API_KEY)
        print(f"Saved: {ticker} on {date_str}")
    except Exception as e:
        print(f"Failed: {ticker} on {date_str} -> {e}")
    