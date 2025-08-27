import matplotlib
matplotlib.use("Qt5Agg")

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MaxNLocator
from zoneinfo import ZoneInfo

plt.style.use('dark_background')

# ──────────────────────────────────────────────────────────────────────────────
def idx_formatter_factory(timestamps, fmt):
    def _fmt(x, pos):
        i = int(round(x))
        if i < 0 or i >= len(timestamps):
            return ""
        ts = timestamps[i]
        return ts.strftime(fmt)
    return _fmt

def plot_candles_no_gaps(ax, df, width=0.8, timefmt='%H:%M', shade_prepost=False):
    """Plot candles using integer indices to remove calendar gaps.
       df must include columns: Open, High, Low, Close, timestamp_est.
    """
    if df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return None

    x = np.arange(len(df), dtype=float)  # integer positions
    tuples = list(zip(x, df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values))
    candlestick_ohlc(ax, tuples, width=width, colorup="white", colordown="red")

    # Shading only where bars exist
    if shade_prepost and len(df) > 1:
        for i in range(len(df)-1):
            t = df.index[i].time()
            if (pd.to_datetime("04:00").time() <= t < pd.to_datetime("09:30").time()):
                ax.axvspan(x[i], x[i+1], color="#444444", alpha=0.6)
            if (pd.to_datetime("16:00").time() <= t < pd.to_datetime("20:00").time()):
                ax.axvspan(x[i], x[i+1], color="#333333", alpha=0.5)

    # Adaptive tick formatting — FIX: use .dt.to_pydatetime() for Series
    ts_list = df["timestamp_est"].dt.to_pydatetime().tolist()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(idx_formatter_factory(ts_list, timefmt)))

    # Tight y-lims
    y_max = df["High"].max()
    y_min = df["Low"].min()
    ax.set_ylim(y_min * 0.995, y_max * 1.005)

    return x

# ──────────────────────────────────────────────────────────────────────────────
def generate_and_save_chart(TICKER, TARGET_DATE, API_KEY):
    TICKER = TICKER.upper()

    def fetch_polygon_agg(ticker, multiplier, timespan, from_date, to_date, api_key):
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "results" not in data:
            raise ValueError(f"Polygon API error or no data found: {data}")
        df = pd.DataFrame(data["results"])
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("US/Eastern")
        df = df.sort_values("timestamp")
        df.set_index("timestamp", inplace=True)
        df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def add_bands(df, ema_short, ema_long, du1, du2, dl1, dl2, prefix=""):
        if df.empty:
            return
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
        # Keep only 04:00–20:00 Eastern on the target date, weekdays only
        df = df[((df.index.time >= pd.to_datetime("04:00").time()) & (df.index.time < pd.to_datetime("09:30").time())) | 
                ((df.index.time >= pd.to_datetime("09:30").time()) & (df.index.time <= pd.to_datetime("20:00").time()))]
        df = df[df.index.dayofweek < 5]
        df = df[df.index.date == target_date.date()]
        df["timestamp_est"] = df.index.tz_convert("US/Eastern")
        return df

    def filter_session_hours(df):
        """For HOURLY: remove weekends and hours outside 04:00–20:00 ET."""
        if df.empty:
            return df
        df = df.copy()
        df = df[df.index.dayofweek < 5]
        start_t = pd.to_datetime("04:00").time()
        end_t   = pd.to_datetime("20:00").time()
        mask = (df.index.time >= start_t) & (df.index.time <= end_t)
        df = df[mask]
        df["timestamp_est"] = df.index.tz_convert("US/Eastern")
        return df

    def remove_fake_wicks(df):
        if df.empty:
            return df
        return df[~((df["High"] == df["Low"]) & (df["Open"] == df["High"]) & (df["Close"] == df["High"]))]

    start_target = pd.to_datetime(TARGET_DATE)
    prev_day = start_target - timedelta(days=5)
    next_day = start_target + timedelta(days=1)
    daily_start = (start_target - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    daily_end = start_target.strftime("%Y-%m-%d")

    # Hourly window: 6 days before to target day
    hour_start = (start_target - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    hour_end   = (start_target + pd.Timedelta(days=0)).strftime("%Y-%m-%d")

    # Fetch data
    df_5m_raw   = remove_fake_wicks(fetch_polygon_agg(TICKER, 5,  "minute", prev_day.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d"), API_KEY))
    df_15m_raw  = remove_fake_wicks(fetch_polygon_agg(TICKER, 15, "minute", prev_day.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d"), API_KEY))
    df_hour_raw = remove_fake_wicks(fetch_polygon_agg(TICKER, 1,  "hour",   hour_start, hour_end, API_KEY))
    df_daily    = remove_fake_wicks(fetch_polygon_agg(TICKER, 1,  "day",    daily_start, daily_end, API_KEY))

    # Indicators
    for d in (df_15m_raw, df_5m_raw, df_hour_raw):
        add_bands(d, 72, 89, 6.9, 9.6, 4.2, 5.5, prefix="l_")
        add_bands(d, 9, 20, 1.0, 0.5, 2.0, 2.5, prefix="s_")

    # Daily EMA
    if not df_daily.empty:
        df_daily["ema9"]  = df_daily["Close"].ewm(span=9,  min_periods=0).mean()
        df_daily["ema20"] = df_daily["Close"].ewm(span=20, min_periods=0).mean()

    # Intraday compress for target date + VWAP
    df_5m   = compress_market_time(df_5m_raw,   start_target)
    df_15m  = compress_market_time(df_15m_raw,  start_target)

    # Hourly — filter to trading session to *remove all non-trading gaps*
    df_hour = filter_session_hours(df_hour_raw)

    # Ensure timestamp_est for all
    for d in [df_5m, df_15m, df_daily]:
        if not d.empty:
            d["timestamp_est"] = d.index.tz_convert("US/Eastern")

    # VWAPs (per day) for intraday + hourly
    for d in [df_5m, df_15m, df_hour]:
        if d.empty:
            continue
        d["date_only"] = d.index.date
        d["cum_vol"] = d.groupby("date_only")["Volume"].cumsum()
        d["cum_vol_price"] = (d["Close"] * d["Volume"]).groupby(d["date_only"]).cumsum()
        d["VWAP"] = d["cum_vol_price"] / d["cum_vol"]

    # Daily up to target
    if not df_daily.empty:
        df_daily = df_daily[df_daily.index.date <= start_target.date()].copy()

    # Robust output dir
    default_dir = os.path.expanduser("~/Desktop/lc setups")
    try:
        os.makedirs(default_dir, exist_ok=True)
        output_dir = default_dir
    except Exception as e:
        print(f"Warning: couldn't create Desktop path ({e}). Using current directory.")
        output_dir = os.getcwd()

    # Helper to save & close
    def _save_fig(fig, path):
        try:
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print("Saved:", path)
        finally:
            plt.close(fig)

    # ── DAILY (own file)
    figD, axD = plt.subplots(figsize=(12, 7))
    xD = plot_candles_no_gaps(axD, df_daily, width=0.6, timefmt='%Y-%m-%d', shade_prepost=False)
    if xD is not None and not df_daily.empty:
        axD.plot(np.arange(len(df_daily)), df_daily["ema9"],  linestyle="--", label="EMA9")
        axD.plot(np.arange(len(df_daily)), df_daily["ema20"], linestyle="--", label="EMA20")
        axD.set_title(f"{TICKER} Daily")
        axD.legend()
    fD = os.path.join(output_dir, f"{TICKER}_{start_target.date()}_daily.png")
    _save_fig(figD, fD)

    # ── 1H (own file)
    figH, axH = plt.subplots(figsize=(12, 7))
    xH = plot_candles_no_gaps(axH, df_hour, width=0.6, timefmt='%m-%d %H:%M', shade_prepost=True)
    if xH is not None and not df_hour.empty:
        for col1, col2, alpha in [
            ("s_UPP1", "s_UPP2", 0.3),
            ("s_LOW1", "s_LOW2", 0.3),
            ("l_UPP1", "l_UPP2", 0.25),
            ("l_LOW1", "l_LOW2", 0.25),
        ]:
            axH.fill_between(np.arange(len(df_hour)), df_hour[col1], df_hour[col2], alpha=alpha)
        for series, style, label in [
            ("s_emaS", "--", "EMA9"),
            ("s_emaL", "--", "EMA20"),
            ("VWAP",  "--", "VWAP"),
        ]:
            axH.plot(np.arange(len(df_hour)), df_hour[series], linestyle=style, label=label)
        axH.set_title(f"{TICKER} 1H (session-only, no gaps)")
        axH.legend()
    fH = os.path.join(output_dir, f"{TICKER}_{start_target.date()}_1h.png")
    _save_fig(figH, fH)

    # ── 15M (own file)
    fig15, ax15 = plt.subplots(figsize=(12, 7))
    x15 = plot_candles_no_gaps(ax15, df_15m, width=0.6, timefmt='%H:%M', shade_prepost=True)
    if x15 is not None and not df_15m.empty:
        for col1, col2, alpha in [
            ("s_UPP1", "s_UPP2", 0.3),
            ("s_LOW1", "s_LOW2", 0.3),
            ("l_UPP1", "l_UPP2", 0.25),
            ("l_LOW1", "l_LOW2", 0.25),
        ]:
            ax15.fill_between(np.arange(len(df_15m)), df_15m[col1], df_15m[col2], alpha=alpha)
        for series, style, label in [
            ("s_emaS", "--", "EMA9"),
            ("s_emaL", "--", "EMA20"),
            ("VWAP",  "--", "VWAP"),
        ]:
            ax15.plot(np.arange(len(df_15m)), df_15m[series], linestyle=style, label=label)
        ax15.set_title(f"{TICKER} {start_target.date()} 15M")
        ax15.legend()
    f15 = os.path.join(output_dir, f"{TICKER}_{start_target.date()}_15m.png")
    _save_fig(fig15, f15)

    # ── 5M (own file)
    fig5, ax5 = plt.subplots(figsize=(12, 7))
    x5 = plot_candles_no_gaps(ax5, df_5m, width=0.6, timefmt='%H:%M', shade_prepost=True)
    if x5 is not None and not df_5m.empty:
        for col1, col2, alpha in [
            ("s_UPP1", "s_UPP2", 0.3),
            ("s_LOW1", "s_LOW2", 0.3),
            ("l_UPP1", "l_UPP2", 0.25),
            ("l_LOW1", "l_LOW2", 0.25),
        ]:
            ax5.fill_between(np.arange(len(df_5m)), df_5m[col1], df_5m[col2], alpha=alpha)
        for series, style, label in [
            ("s_emaS", "--", "EMA9"),
            ("s_emaL", "--", "EMA20"),
            ("VWAP",  "--", "VWAP"),
        ]:
            ax5.plot(np.arange(len(df_5m)), df_5m[series], linestyle=style, label=label)
        ax5.set_title(f"{TICKER} {start_target.date()} 5M")
        ax5.legend()
    f5 = os.path.join(output_dir, f"{TICKER}_{start_target.date()}_5m.png")
    _save_fig(fig5, f5)

# ──────────────────────────────────────────────────────────────────────────────
API_KEY = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
ticker_dates = [
    ("NVDA", "2024-06-20"),
    ("NVDA", "2024-03-08"),
    ("NVDA", "2025-07-31"),
    ("NVDA", "2024-07-11"),
    ("MSTR", "2024-11-21"),
    ("PLTR", "2024-12-09"),
]

if __name__ == "__main__":
    for ticker, date_str in ticker_dates:
        try:
            generate_and_save_chart(ticker, pd.to_datetime(date_str), API_KEY)
        except Exception as e:
            print(f"Failed: {ticker} on {date_str} -> {e}")
