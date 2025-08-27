# analyze_dates_intraday_dev.py
# Analyze specific (ticker, date) pairs:
#   • Highest-high from D-1 16:00 ET → D0 16:00 ET on 1h, 15m, 5m
#   • Deviation metrics:
#       - Dev72Mult = (HH - EMA_72) / ATR_72  (ATR72 multiple, matches your cloud logic)
#       - HH_vs_EMA72_%  = (HH - EMA_72)/EMA_72 * 100
#       - HH_vs_EMA9_%   = (HH - EMA_9)/EMA_9   * 100  (1h & 15m only)
# Prints a tidy table for the provided events.

import pandas as pd, numpy as np, requests
from datetime import datetime, timedelta

API_KEY  = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
BASE_URL = "https://api.polygon.io"
ET_TZ    = "US/Eastern"

# ───────── fetch helpers ─────────
def fetch_aggregates(ticker: str, multiplier: int, timespan: str, start: str, end: str) -> pd.DataFrame:
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
    r   = requests.get(url, params={"apiKey": API_KEY, "adjusted": "true", "sort": "asc", "limit": 50000})
    r.raise_for_status()
    rows = r.json().get("results", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if timespan == "day":
        df = (
            df.assign(Date=lambda d: pd.to_datetime(d["t"], unit="ms"))
              .rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
              .set_index("Date")[["Open","High","Low","Close","Volume"]]
              .sort_index()
        )
    else:
        df = (
            df.assign(ts=lambda d: pd.to_datetime(d["t"], unit="ms", utc=True).dt.tz_convert(ET_TZ))
              .rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
              .set_index("ts")[["Open","High","Low","Close","Volume"]]
              .sort_index()
        )
    return df

# ───────── indicators ─────────
def add_intraday_indicators(df: pd.DataFrame, want_ema9: bool = True) -> pd.DataFrame:
    """Add EMA_72, ATR_72 (true range rolling mean), and optionally EMA_9 to intraday df."""
    if df.empty: 
        return df
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR_72"] = tr.rolling(72, min_periods=30).mean()
    df["EMA_72"] = df["Close"].ewm(span=72, adjust=False).mean()
    if want_ema9:
        df["EMA_9"]  = df["Close"].ewm(span=9, adjust=False).mean()
    return df

def highest_high_window_metrics(df: pd.DataFrame, d0_et: pd.Timestamp, want_ema9: bool) -> dict:
    """Window: D-1 16:00 ET → D0 16:00 ET, return metrics for the bar with the highest High."""
    if df.empty:
        return {"HH": np.nan, "HH_Time_ET": None, "Dev72Mult": np.nan, "HH_vs_EMA72_%": np.nan, "HH_vs_EMA9_%": np.nan}
    # Ensure indicators
    df = add_intraday_indicators(df, want_ema9=want_ema9)

    w_start = (d0_et - timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0)
    w_end   = d0_et.replace(hour=16, minute=0, second=0, microsecond=0)
    win = df.loc[(df.index >= w_start) & (df.index <= w_end)]
    if win.empty:
        return {"HH": np.nan, "HH_Time_ET": None, "Dev72Mult": np.nan, "HH_vs_EMA72_%": np.nan, "HH_vs_EMA9_%": np.nan}

    idx = win["High"].idxmax()
    bar = win.loc[idx]
    hh  = float(bar["High"])
    ema72 = bar.get("EMA_72", np.nan)
    atr72 = bar.get("ATR_72", np.nan)
    ema9  = bar.get("EMA_9",  np.nan) if want_ema9 else np.nan

    dev_mult = (hh - ema72) / atr72 if (pd.notna(ema72) and pd.notna(atr72) and atr72 != 0) else np.nan
    dev72_pct = (hh - ema72) / ema72 * 100 if (pd.notna(ema72) and ema72 != 0) else np.nan
    dev9_pct  = (hh - ema9)  / ema9  * 100 if (want_ema9 and pd.notna(ema9) and ema9 != 0) else np.nan

    return {
        "HH": hh,
        "HH_Time_ET": idx.strftime("%Y-%m-%d %H:%M %Z"),
        "Dev72Mult": dev_mult,
        "HH_vs_EMA72_%": dev72_pct,
        "HH_vs_EMA9_%": dev9_pct,
    }

# ───────── analyze driver ─────────
def analyze_events(evts: list[tuple[str,str]], lookback_days: int = 20) -> pd.DataFrame:
    """For each (ticker, 'YYYY-MM-DD') return 1 row of intraday HH deviation metrics on 1h/15m/5m."""
    rows = []
    for tk, date_str in evts:
        try:
            d0 = pd.Timestamp(date_str)
        except Exception:
            # tolerate M/D/YY formats by letting pandas parse then normalize to date
            d0 = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(d0):
            continue
        d0_et = pd.Timestamp(d0).tz_localize(ET_TZ)

        start = (d0_et - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end   = (d0_et + timedelta(days=1)).strftime("%Y-%m-%d")

        # fetch intraday
        h1  = fetch_aggregates(tk, 1, "hour", start, end)
        m15 = fetch_aggregates(tk, 15, "minute", start, end)
        m5  = fetch_aggregates(tk, 5, "minute", start, end)

        # filter to weekdays and 04:00–20:00 ET (keeps PRE/AH relevant but trims dead hours)
        def session_filter(df):
            if df.empty: return df
            df = df[df.index.dayofweek < 5]
            st = pd.to_datetime("04:00").time()
            en = pd.to_datetime("20:00").time()
            return df[(df.index.time >= st) & (df.index.time <= en)].copy()

        h1  = session_filter(h1)
        m15 = session_filter(m15)
        m5  = session_filter(m5)

        # compute HH window metrics
        h1m  = highest_high_window_metrics(h1,  d0_et, want_ema9=True)
        m15m = highest_high_window_metrics(m15, d0_et, want_ema9=True)
        m5m  = highest_high_window_metrics(m5,  d0_et, want_ema9=False)

        row = {
            "Ticker": tk.upper(),
            "D0": d0.strftime("%Y-%m-%d"),
            "HH_1H": h1m["HH"],
            "HH_1H_Time_ET": h1m["HH_Time_ET"],
            "Dev72x_1H": h1m["Dev72Mult"],
            "HH_vs_EMA72%_1H": h1m["HH_vs_EMA72_%"],
            "HH_vs_EMA9%_1H": h1m["HH_vs_EMA9_%"],
            "HH_15m": m15m["HH"],
            "HH_15m_Time_ET": m15m["HH_Time_ET"],
            "Dev72x_15m": m15m["Dev72Mult"],
            "HH_vs_EMA72%_15m": m15m["HH_vs_EMA72_%"],
            "HH_vs_EMA9%_15m": m15m["HH_vs_EMA9_%"],
            "HH_5m": m5m["HH"],
            "HH_5m_Time_ET": m5m["HH_Time_ET"],
            "Dev72x_5m": m5m["Dev72Mult"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # neat formatting
    num_cols = [c for c in df.columns if c.startswith("Dev72x") or "HH_vs_" in c]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    for c in num_cols + ["HH_1H","HH_15m","HH_5m"]:
        if c in df.columns:
            df[c] = df[c].astype(float).round(2)

    return df

if __name__ == "__main__":
    # Example events (edit freely)
    events = [
        ("TCOM", "2024-09-20"),
        ("DJT",  "2024-10-29"),
        ("GME",  "2024-05-14"),
        ("AMC",  "2024-05-14"),
        ("NVDA", "2024-03-08"),
        ("NVDA", "2024-03-20"),
        ("MSTR", "2024-11-21"),
        ("MRNA", "2024-12-01"),
    ]

    df = analyze_events(events, lookback_days=20)
    if df.empty:
        print("No data returned. Check symbols/dates or API limits.")
    else:
        pd.set_option("display.max_columns", None, "display.width", 0)
        print("\nIntraday highest-high deviation metrics (D-1 16:00 → D0 16:00):\n")
        print(df.to_string(index=False))

