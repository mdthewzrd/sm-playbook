# daily_para_backside_lite_scan_with_fade_grading.py
# Daily-only "A+ para, backside" scan — lite mold + intraday fade metrics + grading.
# Adds configurable fade requirement (percent or ATR) and prints TOTALS across the entire scan.

import pandas as pd, numpy as np, requests, os, math
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ───────── config ─────────
session  = requests.Session()
API_KEY  = os.environ.get("POLYGON_API_KEY", "Fm7brz4s23eSocDErnL68cE7wspz2K1I")
BASE_URL = "https://api.polygon.io"
MAX_WORKERS = 6
INTRA_MAX_WORKERS = 4

PRINT_FROM = "2020-01-01"
PRINT_TO   = None

ENABLE_GRADING = True
FADE_TO = "LOW1"            # "LOW1" or "LOW2"
MAKE_PLOTS = False
ARTIFACTS_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "lc_fade_metrics")

# printing toggles
PRINT_PER_NAME = False       # you said you don't care about per-name totals
PRINT_TOP_GRADED = True      # keep the top-rows view

# ── Fade requirement ──────────────────────────────────────────────────────────
FADE_RULE = {
    "mode": "pct",          # "pct" -> use min_pct;  "atr" -> use min_atr
    "min_pct": 3,         # require ≥ 2% fade
    "min_atr": 0.50,        # or ≥ 0.5 ATR
    "use_daily_atr": True,  # if mode="atr": True = daily ATR_prev(14); False = 5m s_ATR_L at start
}

# ───────── knobs (daily) ─────────
P = {
    "price_min"        : 8.0,
    "adv20_min_usd"    : 30_000_000,

    "abs_lookback_days": 1000,
    "abs_exclude_days" : 10,
    "pos_abs_max"      : 0.90,

    "trigger_mode"     : "D1_or_D2",
    "atr_mult"         : 1.10,
    "vol_mult"         : 1.00,

    "d1_vol_mult_min"  : 1.5,         # e.g., 1.25 (None disables)
    "d1_volume_min"    : 20_000_000,   # absolute floor (shares)

    "slope5d_min"      : 6,
    "high_ema9_mult"   : 1.10,

    "gap_div_atr_min"   : 1.25,
    "open_over_ema9_min": 1.10,
    "d1_green_atr_min"  : 1,
    "require_open_gt_prev_high": True,

    "enforce_d1_above_d2": True,
}

# ───────── universe ─────────
SYMBOLS = [
    'MSTR','SMCI','DJT','BABA','TCOM','AMC','SOXL','MRVL','TGT','DOCU','ZM','DIS',
    'NFLX','SNAP','RBLX','META','SE','NVDA','AAPL','MSFT','GOOGL','AMZN','TSLA',
    'AMD','INTC','BA','PYPL','QCOM','ORCL','KO','PEP','ABBV','JNJ','CRM','BAC',
    'JPM','WMT','CVX','XOM','COP','RTX','SPGI','GS','HD','LOW','COST','UNH','NKE',
    'LMT','HON','CAT','LIN','ADBE','AVGO','TXN','ACN','UPS','BLK','PM','ELV','VRTX',
    'ZTS','NOW','ISRG','PLD','MS','MDT','WM','GE','IBM','BKNG','FDX','ADP','EQIX',
    'DHR','SNPS','REGN','SYK','TMO','CVS','INTU','SCHW','CI','APD','SO','MMC','ICE',
    'FIS','ADI','CSX','LRCX','GILD','RIVN','PLTR','SNOW','SPY','QQQ','IWM','RIOT',
    'MARA','COIN','MRNA','CELH','UPST','AFRM','DKNG'
]

# ───────── fetch (daily) ─────────
def fetch_daily(tkr: str, start: str, end: str) -> pd.DataFrame:
    url = f"{BASE_URL}/v2/aggs/ticker/{tkr}/range/1/day/{start}/{end}"
    r   = session.get(url, params={"apiKey": API_KEY, "adjusted":"true", "sort":"asc", "limit":50000})
    r.raise_for_status()
    rows = r.json().get("results", [])
    if not rows: return pd.DataFrame()
    return (pd.DataFrame(rows)
            .assign(Date=lambda d: pd.to_datetime(d["t"], unit="ms", utc=True))
            .rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
            .set_index("Date")[["Open","High","Low","Close","Volume"]]
            .sort_index())

# ───────── metrics (daily lite) ─────────
def add_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    m = df.copy()
    try: m.index = m.index.tz_localize(None)
    except Exception: pass

    m["EMA_9"]  = m["Close"].ewm(span=9 , adjust=False).mean()
    m["EMA_20"] = m["Close"].ewm(span=20, adjust=False).mean()

    hi_lo   = m["High"] - m["Low"]
    hi_prev = (m["High"] - m["Close"].shift(1)).abs()
    lo_prev = (m["Low"]  - m["Close"].shift(1)).abs()
    m["TR"]      = pd.concat([hi_lo, hi_prev, lo_prev], axis=1).max(axis=1)
    m["ATR_raw"] = m["TR"].rolling(14, min_periods=14).mean()
    m["ATR"]     = m["ATR_raw"].shift(1)

    m["VOL_AVG"]     = m["Volume"].rolling(14, min_periods=14).mean().shift(1)
    m["Prev_Volume"] = m["Volume"].shift(1)
    m["ADV20_$"]     = (m["Close"] * m["Volume"]).rolling(20, min_periods=20).mean().shift(1)

    m["Slope_9_5d"]  = (m["EMA_9"] - m["EMA_9"].shift(5)) / m["EMA_9"].shift(5) * 100
    m["High_over_EMA9_div_ATR"] = (m["High"] - m["EMA_9"]) / m["ATR"]

    m["Gap_abs"]       = (m["Open"] - m["Close"].shift(1)).abs()
    m["Gap_over_ATR"]  = m["Gap_abs"] / m["ATR"]
    m["Open_over_EMA9"]= m["Open"] / m["EMA_9"]

    m["Body_over_ATR"] = (m["Close"] - m["Open"]) / m["ATR"]

    m["Prev_Close"] = m["Close"].shift(1)
    m["Prev_Open"]  = m["Open"].shift(1)
    m["Prev_High"]  = m["High"].shift(1)
    return m

# ───────── helpers (daily) ─────────
def abs_top_window(df: pd.DataFrame, d0: pd.Timestamp, lookback_days: int, exclude_days: int):
    if df.empty: return (np.nan, np.nan)
    cutoff = d0 - pd.Timedelta(days=exclude_days)
    wstart = cutoff - pd.Timedelta(days=lookback_days)
    win = df[(df.index > wstart) & (df.index <= cutoff)]
    if win.empty: return (np.nan, np.nan)
    return float(win["Low"].min()), float(win["High"].max())

def pos_between(val, lo, hi):
    if any(pd.isna(t) for t in (val, lo, hi)) or hi <= lo: return np.nan
    return max(0.0, min(1.0, float((val - lo) / (hi - lo))))

def _mold_on_row(rx: pd.Series) -> bool:
    if pd.isna(rx.get("Prev_Close")) or pd.isna(rx.get("ADV20_$")):
        return False
    if rx["Prev_Close"] < P["price_min"] or rx["ADV20_$"] < P["adv20_min_usd"]:
        return False
    vol_avg = rx["VOL_AVG"]
    if pd.isna(vol_avg) or vol_avg <= 0: return False
    vol_sig = max(rx["Volume"]/vol_avg, rx["Prev_Volume"]/vol_avg)
    checks = [
        (rx["TR"] / rx["ATR"]) >= P["atr_mult"],
        vol_sig                 >= P["vol_mult"],
        rx["Slope_9_5d"]        >= P["slope5d_min"],
        rx["High_over_EMA9_div_ATR"] >= P["high_ema9_mult"],
    ]
    return all(bool(x) and np.isfinite(x) for x in checks)

# ───────── scan one symbol (daily) ─────────
def scan_symbol(sym: str, start: str, end: str) -> pd.DataFrame:
    df = fetch_daily(sym, start, end)
    if df.empty: return pd.DataFrame()
    m  = add_daily_metrics(df)

    rows = []
    for i in range(2, len(m)):
        d0 = m.index[i]
        r0 = m.iloc[i]       # D0
        r1 = m.iloc[i-1]     # D-1
        r2 = m.iloc[i-2]     # D-2

        lo_abs, hi_abs = abs_top_window(m, d0, P["abs_lookback_days"], P["abs_exclude_days"])
        pos_abs_prev = pos_between(r1["Close"], lo_abs, hi_abs)
        if not (pd.notna(pos_abs_prev) and pos_abs_prev <= P["pos_abs_max"]):
            continue

        trigger_ok = False; trig_row = None; trig_tag = "-"
        if P["trigger_mode"] == "D1_only":
            if _mold_on_row(r1): trigger_ok, trig_row, trig_tag = True, r1, "D-1"
        else:
            if _mold_on_row(r1): trigger_ok, trig_row, trig_tag = True, r1, "D-1"
            elif _mold_on_row(r2): trigger_ok, trig_row, trig_tag = True, r2, "D-2"
        if not trigger_ok:
            continue

        if not (pd.notna(r1["Body_over_ATR"]) and r1["Body_over_ATR"] >= P["d1_green_atr_min"]):
            continue

        if P["d1_volume_min"] is not None:
            if not (pd.notna(r1["Volume"]) and r1["Volume"] >= P["d1_volume_min"]):
                continue

        if P["d1_vol_mult_min"] is not None:
            if not (pd.notna(r1["VOL_AVG"]) and r1["VOL_AVG"] > 0 and (r1["Volume"]/r1["VOL_AVG"]) >= P["d1_vol_mult_min"]):
                continue

        if P["enforce_d1_above_d2"]:
            if not (pd.notna(r1["High"]) and pd.notna(r2["High"]) and r1["High"] > r2["High"]
                    and pd.notna(r1["Close"]) and pd.notna(r2["Close"]) and r1["Close"] > r2["Close"]):
                continue

        if pd.isna(r0["Gap_over_ATR"]) or r0["Gap_over_ATR"] < P["gap_div_atr_min"]:
            continue
        if P["require_open_gt_prev_high"] and not (r0["Open"] > r1["High"]):
            continue
        if pd.isna(r0["Open_over_EMA9"]) or r0["Open_over_EMA9"] < P["open_over_ema9_min"]:
            continue

        d1_vol_mult = (r1["Volume"]/r1["VOL_AVG"]) if (pd.notna(r1["VOL_AVG"]) and r1["VOL_AVG"]>0) else np.nan
        volsig_max  = (max(r1["Volume"]/r1["VOL_AVG"], r2["Volume"]/r2["VOL_AVG"])
                       if (pd.notna(r1["VOL_AVG"]) and r1["VOL_AVG"]>0 and pd.notna(r2["VOL_AVG"]) and r2["VOL_AVG"]>0)
                       else np.nan)

        rows.append({
            "Ticker": sym,
            "Date": d0.strftime("%Y-%m-%d"),
            "Trigger": trig_tag,
            "PosAbs_1000d": round(float(pos_abs_prev), 3),
            "D1_Body/ATR": round(float(r1["Body_over_ATR"]), 2),
            "D1Vol(shares)": int(r1["Volume"]) if pd.notna(r1["Volume"]) else np.nan,
            "D1Vol/Avg": round(float(d1_vol_mult), 2) if pd.notna(d1_vol_mult) else np.nan,
            "VolSig(max D-1,D-2)/Avg": round(float(volsig_max), 2) if pd.notna(volsig_max) else np.nan,
            "Gap/ATR": round(float(r0["Gap_over_ATR"]), 2),
            "Open>PrevHigh": bool(r0["Open"] > r1["High"]),
            "Open/EMA9": round(float(r0["Open_over_EMA9"]), 2),
            "D1>H(D-2)": bool(r1["High"] > r2["High"]),
            "D1Close>D2Close": bool(r1["Close"] > r2["Close"]),
            "Slope9_5d": round(float(r0["Slope_9_5d"]), 2) if pd.notna(r0["Slope_9_5d"]) else np.nan,
            "High-EMA9/ATR(trigger)": round(float(trig_row["High_over_EMA9_div_ATR"]), 2),
            "ADV20_$": round(float(r0["ADV20_$"])) if pd.notna(r0["ADV20_$"]) else np.nan,
        })

    return pd.DataFrame(rows)

# ============================================================================
#                   Intraday metrics + scoring (integrated)
# ============================================================================

ET = "US/Eastern"

def fetch_polygon_agg(ticker: str, multiplier: int, timespan: str, from_date: str, to_date: str) -> pd.DataFrame:
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": API_KEY}
    r = session.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    if "results" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["results"])
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(ET)
    df = df.sort_values("timestamp").set_index("timestamp")
    return df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})[["Open","High","Low","Close","Volume"]]

def remove_fake_wicks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[~((df["High"] == df["Low"]) & (df["Open"] == df["High"]) & (df["Close"] == df["High"]))]

def add_bands(df: pd.DataFrame, ema_short: int, ema_long: int, du1: float, du2: float, dl1: float, dl2: float, prefix: str = "") -> None:
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

def vwap_by_day(df: pd.DataFrame) -> None:
    if df.empty:
        return
    df["date_only"] = df.index.date
    df["cum_vol"] = df.groupby("date_only")["Volume"].cumsum()
    df["cum_pv"] = (df["Close"] * df["Volume"]).groupby(df["date_only"]).cumsum()
    df["VWAP"] = df["cum_pv"] / df["cum_vol"].replace(0, np.nan)

def _compress_time_window(df: pd.DataFrame, date: pd.Timestamp, start_t: str, end_t: str) -> pd.DataFrame:
    if df.empty:
        return df
    s_t = pd.to_datetime(start_t).time()
    e_t = pd.to_datetime(end_t).time()
    out = df[(df.index.dayofweek < 5) & (df.index.date == date.date())]
    mask = (out.index.time >= s_t) & (out.index.time <= e_t)
    return out.loc[mask].copy()

def _first_touch_after(df: pd.DataFrame, start_ts: pd.Timestamp, column: str, side: str) -> pd.Timestamp | None:
    df2 = df.loc[df.index >= start_ts]
    if df2.empty or column not in df2:
        return None
    cond = (df2["High"] >= df2[column]) if side == "above" else (df2["Low"] <= df2[column])
    return cond.idxmax() if cond.any() else None

# daily ATR_prev(14) for ATR-normalized fade
def fetch_daily_atr_prev(ticker: str, date_ts: pd.Timestamp) -> float | float:
    d0 = pd.Timestamp(date_ts.date())
    start = (d0 - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end   = d0.strftime("%Y-%m-%d")
    df = fetch_daily(ticker, start, end)
    if df.empty or d0 not in df.index:
        return float("nan")
    m = add_daily_metrics(df)
    val = m.loc[d0, "ATR"] if "ATR" in m.columns else float("nan")
    try:
        return float(val)
    except Exception:
        return float("nan")

# --- metrics for one (ticker, date) ---
def intraday_metrics_for_day(ticker: str, date_str: str) -> dict:
    try:
        date = pd.Timestamp(date_str).tz_localize(None)
        date = pd.Timestamp(date).tz_localize(ET)
    except Exception:
        return {"ticker": ticker, "date": date_str, "error": "bad_date"}

    prev = (date - timedelta(days=5)).strftime("%Y-%m-%d")
    nextd = (date + timedelta(days=1)).strftime("%Y-%m-%d")

    df15 = remove_fake_wicks(fetch_polygon_agg(ticker, 15, "minute", prev, nextd))
    df5  = remove_fake_wicks(fetch_polygon_agg(ticker, 5,  "minute", prev, nextd))
    if df15.empty or df5.empty:
        return {"ticker": ticker, "date": date.date().isoformat(), "error": "no_data"}

    # Bands & VWAP
    add_bands(df15, 72, 89, 6.9, 9.6, 4.2, 5.5, prefix="l_")
    add_bands(df15, 9, 20, 1.0, 0.5, 2.0, 2.5, prefix="s_")
    add_bands(df5,  9, 20, 1.0, 0.5, 2.0, 2.5, prefix="s_")
    vwap_by_day(df15); vwap_by_day(df5)

    morning15 = _compress_time_window(df15, date, "07:00", "11:00")
    session15 = _compress_time_window(df15, date, "04:00", "20:00")
    session5  = _compress_time_window(df5,  date, "04:00", "20:00")
    if morning15.empty or session15.empty:
        return {"ticker": ticker, "date": date.date().isoformat(), "error": "no_session"}

    hh_morning_ts = morning15["High"].idxmax()
    hh_morning_px = float(morning15.loc[hh_morning_ts, "High"])
    hh_full_ts = session15["High"].idxmax(); hh_full_px = float(session15.loc[hh_full_ts, "High"])
    high_set_by_11 = bool(hh_full_ts <= morning15.index.max())

    ema9 = session15["s_emaS"]; ema20 = session15["s_emaL"]
    ema9_val = float(ema9.loc[hh_morning_ts]) if hh_morning_ts in ema9.index else math.nan
    ema20_val = float(ema20.loc[hh_morning_ts]) if hh_morning_ts in ema20.index else math.nan
    ema9_slope = float(ema9.diff().loc[hh_morning_ts]) if hh_morning_ts in ema9.index else math.nan
    ema20_slope = float(ema20.diff().loc[hh_morning_ts]) if hh_morning_ts in ema9.index else math.nan
    bearish_15m_9_20 = bool((ema9_val < ema20_val) and (ema9_slope <= 0))

    ema72 = session15["l_emaS"]; ema72_val = float(ema72.loc[hh_morning_ts]) if hh_morning_ts in ema72.index else math.nan
    dist_to_ema72_bps = float((hh_morning_px - ema72_val) / (ema72_val if ema72_val != 0 else np.nan) * 1e4) \
                        if np.isfinite(ema72_val) else float("nan")
    touch_ema72_ts = _first_touch_after(session15, hh_morning_ts, "l_emaS", side="below")

    l_upp1_touch_ts = _first_touch_after(morning15, morning15.index.min(), "l_UPP1", side="above")
    l_upp2_touch_ts = _first_touch_after(morning15, morning15.index.min(), "l_UPP2", side="above")

    start_ts = hh_morning_ts
    if l_upp1_touch_ts is not None and l_upp1_touch_ts <= start_ts:
        start_ts = l_upp1_touch_ts
        start_px = float(session15.loc[start_ts, "High"]) if start_ts in session15.index else hh_morning_px
    else:
        start_px = hh_morning_px

    col_target = f"s_{FADE_TO}"
    hit_ts = _first_touch_after(session5, start_ts, col_target, side="below")
    if hit_ts is not None:
        min_after_start = float(session5.loc[hit_ts, "Low"]) if hit_ts in session5.index else float("nan")
        fade_return = (start_px - min_after_start) / start_px if np.isfinite(min_after_start) else float("nan")
        ttf_minutes = int((hit_ts - start_ts).total_seconds() // 60)
        fade_hit = True
    else:
        after = session5.loc[session5.index >= start_ts]
        if after.empty:
            fade_return = float("nan"); ttf_minutes = None
        else:
            min_px = float(after["Low"].min()); fade_return = (start_px - min_px) / start_px
            ttf_minutes = None
        fade_hit = False

    # ATR-normalized fade & rule pass/fail
    if FADE_RULE["mode"] == "atr":
        if FADE_RULE.get("use_daily_atr", True):
            atr_base = fetch_daily_atr_prev(ticker, date)
        else:
            try:
                atr_base = float(session5.loc[start_ts, "s_ATR_L"])
            except Exception:
                atr_base = float("nan")
        fade_return_atr = (start_px - (min_after_start if hit_ts is not None else
                           float(session5.loc[session5.index >= start_ts, "Low"].min())))
        if np.isfinite(atr_base) and atr_base > 0 and np.isfinite(fade_return_atr):
            fade_return_atr = fade_return_atr / atr_base
        else:
            fade_return_atr = float("nan")
        threshold_value = float(FADE_RULE["min_atr"])
        fade_meets_threshold = (np.isfinite(fade_return_atr) and (fade_return_atr >= threshold_value))
    else:
        fade_return_atr = float("nan")
        threshold_value = float(FADE_RULE["min_pct"])
        fade_meets_threshold = (np.isfinite(fade_return) and ((fade_return*100.0) >= threshold_value))

    vwap_val = float(session15.loc[hh_morning_ts, "VWAP"]) if hh_morning_ts in session15.index else math.nan
    above_vwap_at_hh = bool(hh_morning_px >= vwap_val) if np.isfinite(vwap_val) else False

    return {
        "ticker": ticker,
        "date": date.date().isoformat(),
        "high_set_by_11": high_set_by_11,
        "hh_morning_ts": hh_morning_ts,
        "hh_morning_px": round(hh_morning_px, 4),
        "hh_full_ts": hh_full_ts,
        "hh_full_px": round(hh_full_px, 4),
        "bearish_15m_9_20": bearish_15m_9_20,
        "dist_to_ema72_bps": round(dist_to_ema72_bps, 1) if np.isfinite(dist_to_ema72_bps) else np.nan,
        "reverted_to_ema72": touch_ema72_ts is not None,
        "time_to_ema72_min": int((touch_ema72_ts - hh_morning_ts).total_seconds() // 60) if touch_ema72_ts else None,
        "touched_15m_l_upp1": l_upp1_touch_ts is not None,
        "touched_15m_l_upp2": l_upp2_touch_ts is not None,
        f"fade_hit_5m_{FADE_TO.lower()}": bool(fade_hit),
        "fade_return_pct": round(float(fade_return*100), 2) if np.isfinite(fade_return) else np.nan,
        "fade_return_atr": round(float(fade_return_atr), 3) if np.isfinite(fade_return_atr) else np.nan,
        "fade_threshold_used": threshold_value,
        "fade_meets_threshold": bool(fade_meets_threshold),
        "time_to_fade_min": ttf_minutes,
        "above_vwap_at_hh": above_vwap_at_hh,
        "error": None,
    }

# --- scoring ---
FEATURES = [
    "high_set_by_11",
    "bearish_15m_9_20",
    "dist_to_ema72_bps",
    "touched_15m_l_upp1",
    "touched_15m_l_upp2",
    f"fade_hit_5m_{FADE_TO.lower()}",
    "fade_meets_threshold",
    "fade_return_pct",
    "time_to_fade_min",
    "above_vwap_at_hh",
    "Gap/ATR",
    "Open/EMA9",
    "PosAbs_1000d",
    "D1_Body/ATR",
    "D1Vol/Avg",
    "VolSig(max D-1,D-2)/Avg",
]

WEIGHTS = {
    "high_set_by_11": 0.6,
    "bearish_15m_9_20": 0.8,
    "dist_to_ema72_bps": 1.2,
    "touched_15m_l_upp1": 0.6,
    "touched_15m_l_upp2": 0.9,
    f"fade_hit_5m_{FADE_TO.lower()}": 1.0,
    "fade_meets_threshold": 2.2,
    "fade_return_pct": 1.6,
    "time_to_fade_min": 1.0,
    "above_vwap_at_hh": 0.5,
    "Gap/ATR": 0.8,
    "Open/EMA9": 0.8,
    "PosAbs_1000d": -0.6,
    "D1_Body/ATR": 0.6,
    "D1Vol/Avg": 0.6,
    "VolSig(max D-1,D-2)/Avg": 0.6,
}

TRANSFORMS = {
    "bool": lambda v, a, b: 0.0 if (pd.isna(v) or not bool(v)) else 1.0,
    "cap": lambda v, lo, hi: 0.0 if (v is None or not np.isfinite(v)) else float(np.clip((v - lo)/(hi - lo), 0.0, 1.0)),
    "invcap": lambda v, hi, lo: 0.0 if (v is None or not np.isfinite(v)) else float(np.clip((hi - v)/(hi - lo), 0.0, 1.0)),
}

TRANS_CFG = {
    "high_set_by_11": ("bool", (0,0)),
    "bearish_15m_9_20": ("bool", (0,0)),
    "touched_15m_l_upp1": ("bool", (0,0)),
    "touched_15m_l_upp2": ("bool", (0,0)),
    f"fade_hit_5m_{FADE_TO.lower()}": ("bool", (0,0)),
    "fade_meets_threshold": ("bool", (0,0)),
    "above_vwap_at_hh": ("bool", (0,0)),
    "dist_to_ema72_bps": ("cap", (50, 300)),
    "fade_return_pct": ("cap", (0.1, 4.0)),
    "Gap/ATR": ("cap", (0.5, 2.0)),
    "Open/EMA9": ("cap", (1.00, 1.08)),
    "D1_Body/ATR": ("cap", (0.2, 1.2)),
    "D1Vol/Avg": ("cap", (0.8, 3.0)),
    "VolSig(max D-1,D-2)/Avg": ("cap", (1.0, 3.0)),
    "time_to_fade_min": ("invcap", (120, 10)),
    "PosAbs_1000d": ("invcap", (0.85, 0.4)),
}

def _grade_row(row: pd.Series) -> tuple[float, dict]:
    score = 0.0
    contrib = {}
    for f in FEATURES:
        val = row.get(f, np.nan)
        kind, params = TRANS_CFG.get(f, ("cap", (0.0,1.0)))
        tr = TRANSFORMS[kind]
        x = tr(val, *params)
        c = WEIGHTS.get(f, 0.0) * x
        contrib[f] = c
        score += c
    return score, contrib

def grade_scan_hits(scan_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scan_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    hits = [(str(t), str(d)) for t, d in zip(scan_df["Ticker"], scan_df["Date"])]

    rows = []
    with ThreadPoolExecutor(max_workers=INTRA_MAX_WORKERS) as exe:
        futs = {exe.submit(intraday_metrics_for_day, t, d): (t,d) for t,d in hits}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as e:
                t,d = futs[fut]
                rows.append({"ticker": t, "date": d, "error": str(e)})

    intra = pd.DataFrame(rows).rename(columns={"ticker":"Ticker","date":"Date"})
    merged = pd.merge(scan_df, intra, on=["Ticker","Date"], how="left", copy=False)

    grades = []
    for _, row in merged.iterrows():
        score, contrib = _grade_row(row)
        g = {"grade": score}
        for f in FEATURES:
            g[f"contrib::{f}"] = contrib.get(f, 0.0)
        grades.append(g)
    grades_df = pd.DataFrame(grades, index=merged.index)

    per_hit = pd.concat([merged, grades_df], axis=1).sort_values(["Date","grade"], ascending=[False, False])

    # keep per-name DF for optional CSV/printing
    per_name = per_hit.groupby("Ticker").agg(
        grade_mean=("grade","mean"),
        grade_median=("grade","median"),
        n_hits=("grade","count"),
        fade_hit_rate=(f"fade_hit_5m_{FADE_TO.lower()}", lambda x: float(pd.Series(x).mean(skipna=True))),
        mean_fade_return_pct=("fade_return_pct", lambda x: float(pd.Series(x).mean(skipna=True))),
        pct_high_set_by_11=("high_set_by_11", lambda x: float(pd.Series(x).mean(skipna=True))),
        pct_bearish_15m_9_20=("bearish_15m_9_20", lambda x: float(pd.Series(x).mean(skipna=True))),
    ).reset_index().sort_values("grade_mean", ascending=False)

    return per_hit, per_name

# ───────── TOTALS helpers ─────────
def _cap01(x, lo, hi):
    try: return float(np.clip((x - lo)/(hi - lo), 0.0, 1.0))
    except Exception: return 0.0

def _invcap01(x, hi, lo):
    try: return float(np.clip((hi - x)/(hi - lo), 0.0, 1.0))
    except Exception: return 0.0

def compute_totals(df: pd.DataFrame, fade_col: str) -> dict:
    """Compute overall totals across the provided per-hit DataFrame."""
    if df.empty:
        return {"n_hits":0}

    ph = df.copy()
    # coerce numeric
    for c in ["fade_return_pct","fade_return_atr","time_to_fade_min","dist_to_ema72_bps",
              "Gap/ATR","Open/EMA9","PosAbs_1000d","D1_Body/ATR","D1Vol/Avg","VolSig(max D-1,D-2)/Avg","grade"]:
        if c in ph.columns:
            ph[c] = pd.to_numeric(ph[c], errors="coerce")

    # boolean rates
    for c in ["high_set_by_11","bearish_15m_9_20","touched_15m_l_upp1","touched_15m_l_upp2","fade_meets_threshold", fade_col]:
        if c in ph.columns:
            ph[c] = ph[c].astype("float")  # True=1.0, False=0.0, NaN stays NaN

    n_hits   = int(len(ph))
    n_names  = int(ph["Ticker"].nunique()) if "Ticker" in ph.columns else 0
    fade_hit_rate = float(ph[fade_col].mean(skipna=True)) if fade_col in ph.columns else float("nan")
    meet_thresh   = float(ph["fade_meets_threshold"].mean(skipna=True)) if "fade_meets_threshold" in ph.columns else float("nan")
    mean_fade_pct = float(ph["fade_return_pct"].mean(skipna=True)) if "fade_return_pct" in ph.columns else float("nan")
    med_fade_pct  = float(ph["fade_return_pct"].median(skipna=True)) if "fade_return_pct" in ph.columns else float("nan")
    mean_fade_atr = float(ph["fade_return_atr"].mean(skipna=True)) if "fade_return_atr" in ph.columns else float("nan")
    med_fade_atr  = float(ph["fade_return_atr"].median(skipna=True)) if "fade_return_atr" in ph.columns else float("nan")
    avg_ttf       = float(ph["time_to_fade_min"].mean(skipna=True)) if "time_to_fade_min" in ph.columns else float("nan")
    med_ttf       = float(ph["time_to_fade_min"].median(skipna=True)) if "time_to_fade_min" in ph.columns else float("nan")
    pct_high_11   = float(ph["high_set_by_11"].mean(skipna=True)) if "high_set_by_11" in ph.columns else float("nan")
    pct_bear_920  = float(ph["bearish_15m_9_20"].mean(skipna=True)) if "bearish_15m_9_20" in ph.columns else float("nan")
    pct_touch_u1  = float(ph["touched_15m_l_upp1"].mean(skipna=True)) if "touched_15m_l_upp1" in ph.columns else float("nan")
    pct_touch_u2  = float(ph["touched_15m_l_upp2"].mean(skipna=True)) if "touched_15m_l_upp2" in ph.columns else float("nan")
    avg_dist_ema72= float(ph["dist_to_ema72_bps"].mean(skipna=True)) if "dist_to_ema72_bps" in ph.columns else float("nan")

    # overall scan quality (0..100) emphasizing depth + speed + consistency
    q = (
        0.45 * _cap01(fade_hit_rate,         0.35, 0.80) +
        0.30 * _cap01(mean_fade_pct,         0.50, 4.00) +
        0.10 * _cap01(meet_thresh,           0.40, 0.90) +
        0.10 * _invcap01(avg_ttf,            120, 15) +
        0.05 * _cap01(pct_high_11,           0.30, 0.90)
    )
    quality_score = round(100 * q, 1)

    return {
        "n_hits": n_hits,
        "n_names": n_names,
        "fade_hit_rate": fade_hit_rate,
        "meet_threshold_rate": meet_thresh,
        "mean_fade_return_pct": mean_fade_pct,
        "median_fade_return_pct": med_fade_pct,
        "mean_fade_return_atr": mean_fade_atr,
        "median_fade_return_atr": med_fade_atr,
        "avg_time_to_fade_min": avg_ttf,
        "median_time_to_fade_min": med_ttf,
        "pct_high_set_by_11": pct_high_11,
        "pct_bearish_15m_9_20": pct_bear_920,
        "pct_touch_l_upp1": pct_touch_u1,
        "pct_touch_l_upp2": pct_touch_u2,
        "avg_dist_to_ema72_bps": avg_dist_ema72,
        "scan_quality_score": quality_score,
    }

# ============================================================================
#                                main
# ============================================================================
if __name__ == "__main__":
    fetch_start = "2018-01-01"
    fetch_end   = datetime.today().strftime("%Y-%m-%d")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futs = {exe.submit(scan_symbol, s, fetch_start, fetch_end): s for s in SYMBOLS}
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None and not df.empty:
                results.append(df)

    if not results:
        print("No hits. Consider relaxing high_ema9_mult / gap_div_atr_min / d1_volume_min.")
        raise SystemExit(0)

    out = pd.concat(results, ignore_index=True)
    if PRINT_FROM: out = out[pd.to_datetime(out["Date"]) >= pd.to_datetime(PRINT_FROM)]
    if PRINT_TO:   out = out[pd.to_datetime(out["Date"]) <= pd.to_datetime(PRINT_TO)]
    out = out.sort_values(["Date","Ticker"], ascending=[False, True])

    pd.set_option("display.max_columns", None, "display.width", 0)
    print("\nBackside A+ (lite) — trade-day hits:\n")
    print(out.to_string(index=False))

    if ENABLE_GRADING:
        print("\nRunning intraday fade metrics + grading…\n")
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        per_hit, per_name = grade_scan_hits(out)

        # Save (still useful to have)
        per_hit_csv = os.path.join(ARTIFACTS_DIR, "graded_per_hit.csv")
        per_hit.to_csv(per_hit_csv, index=False)
        print(f"Saved per-hit:   {per_hit_csv}")

        if PRINT_TOP_GRADED:
            print("\nTop graded hits:\n")
            cols_show = [
                "Date","Ticker","grade","Gap/ATR","Open/EMA9","PosAbs_1000d","D1_Body/ATR","D1Vol/Avg",
                "VolSig(max D-1,D-2)/Avg","high_set_by_11","bearish_15m_9_20","dist_to_ema72_bps",
                "touched_15m_l_upp1","touched_15m_l_upp2",f"fade_hit_5m_{FADE_TO.lower()}",
                "fade_return_pct","fade_return_atr","fade_threshold_used","fade_meets_threshold",
                "time_to_fade_min","above_vwap_at_hh","error"
            ]
            cols_show = [c for c in cols_show if c in per_hit.columns]
            print(per_hit[cols_show].head(50).to_string(index=False))

        # ── TOTALS: ALL hits ───────────────────────────────────────────────────
        fade_col = f"fade_hit_5m_{FADE_TO.lower()}"
        totals_all = compute_totals(per_hit, fade_col)
        totals_all_df = pd.DataFrame([totals_all])
        totals_all_csv = os.path.join(ARTIFACTS_DIR, "totals_overall.csv")
        totals_all_df.to_csv(totals_all_csv, index=False)

        print("\n=== Totals — ALL scan hits ===")
        for k,v in totals_all.items():
            if isinstance(v, float):
                if "rate" in k or "pct_" in k:
                    print(f"{k}: {v:.2%}")
                elif "score" in k:
                    print(f"{k}: {v:.1f}")
                else:
                    print(f"{k}: {v:.3f}")
            else:
                print(f"{k}: {v}")
        print(f"(saved {totals_all_csv})")

        # ── TOTALS: LATEST DATE ONLY (useful watchlist summary) ───────────────
        latest_dt = pd.to_datetime(per_hit["Date"]).max()
        latest_hits = per_hit[pd.to_datetime(per_hit["Date"]) == latest_dt]
        if not latest_hits.empty:
            totals_latest = compute_totals(latest_hits, fade_col)
            totals_latest_df = pd.DataFrame([{"latest_date": latest_dt.date(), **totals_latest}])
            totals_latest_csv = os.path.join(ARTIFACTS_DIR, "totals_latest_day.csv")
            totals_latest_df.to_csv(totals_latest_csv, index=False)

            print(f"\n=== Totals — LATEST DATE only ({latest_dt.date()}) ===")
            for k,v in totals_latest.items():
                if isinstance(v, float):
                    if "rate" in k or "pct_" in k:
                        print(f"{k}: {v:.2%}")
                    elif "score" in k:
                        print(f"{k}: {v:.1f}")
                    else:
                        print(f"{k}: {v:.3f}")
                else:
                    print(f"{k}: {v}")
            print(f"(saved {totals_latest_csv})")

        # Optional: save per-name CSV silently
        names_csv = os.path.join(ARTIFACTS_DIR, "graded_per_name.csv")
        per_name.to_csv(names_csv, index=False)
        if PRINT_PER_NAME:
            print("\nTop names by average grade:")
            print(per_name.head(30).to_string(index=False))
