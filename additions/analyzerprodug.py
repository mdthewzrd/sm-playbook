# analyzer_uptrend_gap.py — v3.1
# Daily Uptrend Gap (EMA-distance channel + tolerant hourly dev-band confirm)
from __future__ import annotations

import os, math, argparse, time, random
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_KEY  = os.environ.get("POLYGON_API_KEY", "Fm7brz4s23eSocDErnL68cE7wspz2K1I")
BASE_URL = "https://api.polygon.io"
ET       = "US/Eastern"

# ───────── resilient HTTP ─────────
def _init_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=6, connect=6, read=6,
        backoff_factor=0.8,
        status_forcelist=[429,500,502,503,504],
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    ad = HTTPAdapter(max_retries=retries, pool_connections=64, pool_maxsize=64)
    s.mount("https://", ad); s.mount("http://", ad)
    s.headers.update({"User-Agent":"uptrend-gap-v3.1/1.0"})
    return s
session = _init_session()

def _robust_get(url: str, *, params: dict, timeout: tuple=(5,30), attempts: int=2):
    last = None
    for k in range(attempts+1):
        try:
            r = session.get(url, params=params, timeout=timeout); r.raise_for_status(); return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last = e
            if k == attempts: raise
            time.sleep((1.5**k) + random.random()*0.3)
        except requests.HTTPError:
            raise
    raise last if last else RuntimeError("HTTP failure")

# ───────── params ─────────
P_DEFAULT: Dict[str, Any] = {
    # Liquidity
    "price_min": 20.0,
    "adv20_min_usd": 250_000_000,

    # Channel via EMA-distance (trend start by 4h last red before D-1 open)
    "trend_seed_4h_lookback_days": 40,
    "upper_channel_top_frac": 0.333,          # top third ⇒ ≥66.7th pct
    "use_atr_norm_for_channel": True,         # use (value/ATR_prev)

    # D-1 strength
    "d1_green_required": True,
    "d1_return_min_pct": 2.0,
    "d1_body_atr_min": 0.40,
    "d1_vol_mult_min": 1.10,

    # Gap on D0
    "gap_over_atr_min": 0.50,
    "gap_min_pct": 1.0,
    "open_over_ema9_min": 1.02,

    # Hourly dev band confirm
    "check_hourly": True,
    "hourly_window_days_back": 6,
    "require_hourly_touch": True,                 # D-1 16:00 → D0 11:00
    "require_hourly_near": True,                  # 07:00 → 11:00
    "hourly_touch_tolerance_bps": 25.0,           # touch if within 25 bps of l_UPP1/2
    "hourly_near_lupp1_bps": 60.0,                # near-to-band tolerance (bps)
    "hourly_retest_hh_bps": 60.0,                 # retest of pre-open HH within (bps) also passes

    # Slopes (report only)
    "report_slopes": True,

    # IO
    "target_as": "d0",        # ← default to D0 (exact session required)
    "style": "stacked",
}

# ───────── fetchers & indicators ─────────
def fetch_daily(tkr: str, start: str, end: str) -> pd.DataFrame:
    url = f"{BASE_URL}/v2/aggs/ticker/{tkr}/range/1/day/{start}/{end}"
    r   = _robust_get(url, params={"apiKey": API_KEY, "adjusted":"true","sort":"asc","limit":50000})
    rows = r.json().get("results", [])
    if not rows: return pd.DataFrame()
    return (pd.DataFrame(rows)
            .assign(Date=lambda d: pd.to_datetime(d["t"], unit="ms", utc=True))
            .rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
            .set_index("Date")[["Open","High","Low","Close","Volume"]]).sort_index()

def fetch_polygon_agg(ticker: str, mult: int, span: str, fr: str, to: str) -> pd.DataFrame:
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{fr}/{to}"
    r = _robust_get(url, params={"adjusted":"true","sort":"asc","limit":50000,"apiKey":API_KEY})
    data = r.json()
    if "results" not in data: return pd.DataFrame()
    df = pd.DataFrame(data["results"])
    if df.empty: return df
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(ET)
    df = df.sort_values("ts").set_index("ts")
    return df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})[
        ["Open","High","Low","Close","Volume"]
    ]

def remove_fake_wicks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    return df[~((df["High"] == df["Low"]) & (df["Open"] == df["High"]) & (df["Close"] == df["High"]))]

# --- wick softener for hourly ATR (guards “mega wick” on split/holiday days)
def soften_1h_wicks_for_atr(df: pd.DataFrame, span_for_rng: int = 72, cap_mult: float = 6.0) -> None:
    """Create sanitized High/Low columns (_sanHigh/_sanLow) used only for ATR calc.
       Caps wicks to body ± cap_mult * EMA(range)."""
    if df.empty: return
    rng_ema = (df["High"] - df["Low"]).ewm(span=span_for_rng, min_periods=1).mean()
    body_hi = np.maximum(df["Open"], df["Close"])
    body_lo = np.minimum(df["Open"], df["Close"])
    df["_sanHigh"] = np.minimum(df["High"], body_hi + cap_mult * rng_ema)
    df["_sanLow"]  = np.maximum(df["Low"],  body_lo - cap_mult * rng_ema)

def add_bands(df: pd.DataFrame, ema_short: int, ema_long: int, du1: float, du2: float, dl1: float, dl2: float, prefix=""):
    if df.empty: return
    df[f"{prefix}emaS"] = df["Close"].ewm(span=ema_short, min_periods=0).mean()
    df[f"{prefix}emaL"] = df["Close"].ewm(span=ema_long, min_periods=0).mean()

    # Use sanitized highs/lows for ATR if present
    H = df.get("_sanHigh", df["High"])
    L = df.get("_sanLow",  df["Low"])
    prevC = df["Close"].shift()

    tr = pd.concat([
        H - L,
        (H - prevC).abs(),
        (L - prevC).abs()
    ], axis=1).max(axis=1)

    df[f"{prefix}ATR_S"] = tr.rolling(ema_short, min_periods=1).mean()
    df[f"{prefix}ATR_L"] = tr.rolling(ema_long,  min_periods=1).mean()
    df[f"{prefix}UPP1"]  = df[f"{prefix}emaS"] + du1 * df[f"{prefix}ATR_S"]
    df[f"{prefix}UPP2"]  = df[f"{prefix}emaS"] + du2 * df[f"{prefix}ATR_S"]
    df[f"{prefix}LOW1"]  = df[f"{prefix}emaL"] - dl1 * df[f"{prefix}ATR_L"]
    df[f"{prefix}LOW2"]  = df[f"{prefix}emaL"] - dl2 * df[f"{prefix}ATR_L"]

def add_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    m = df.copy()
    try: m.index = m.index.tz_localize(None)
    except Exception: pass
    m["EMA_9"]  = m["Close"].ewm(span=9,  adjust=False).mean()
    m["EMA_20"] = m["Close"].ewm(span=20, adjust=False).mean()
    m["EMA_50"] = m["Close"].ewm(span=50, adjust=False).mean()

    # ATR_prev(14)
    hi_lo   = m["High"] - m["Low"]
    hi_prev = (m["High"] - m["Close"].shift(1)).abs()
    lo_prev = (m["Low"]  - m["Close"].shift(1)).abs()
    m["TR"]      = pd.concat([hi_lo, hi_prev, lo_prev], axis=1).max(axis=1)
    m["ATR_raw"] = m["TR"].rolling(14, min_periods=14).mean()
    m["ATR"]     = m["ATR_raw"].shift(1)

    m["VOL_AVG"] = m["Volume"].rolling(14, min_periods=14).mean().shift(1)
    m["Gap_abs"]      = (m["Open"] - m["Close"].shift(1)).abs()
    m["Gap_over_ATR"] = m["Gap_abs"] / m["ATR"]
    m["Body_over_ATR"]= (m["Close"] - m["Open"]) / m["ATR"]

    m["Prev_Close"] = m["Close"].shift(1)

    # distances vs EMA20/EMA9 (ATR-normed)
    m["High_minus_EMA20"] = (m["High"] - m["EMA_20"])
    m["Open_minus_EMA20"] = (m["Open"] - m["EMA_20"])
    m["High_minus_EMA20_over_ATR"] = m["High_minus_EMA20"] / m["ATR"]
    m["Open_minus_EMA20_over_ATR"] = m["Open_minus_EMA20"] / m["ATR"]
    m["High_minus_EMA9_over_ATR"]  = (m["High"] - m["EMA_9"]) / m["ATR"]

    # slopes (reporting)
    m["Slope_EMA9_5d_%"]  = (m["EMA_9"]  - m["EMA_9"].shift(5))  / m["EMA_9"].shift(5)  * 100
    m["Slope_EMA20_5d_%"] = (m["EMA_20"] - m["EMA_20"].shift(5)) / m["EMA_20"].shift(5) * 100
    return m

# ───────── helpers ─────────
def _pct(a: float, b: float) -> float:
    try: return (a/b - 1.0) * 100.0 if (np.isfinite(a) and np.isfinite(b) and b!=0) else float("nan")
    except Exception: return float("nan")

def _lr_slope_bps_per_day(series: pd.Series) -> float:
    y = pd.to_numeric(pd.Series(series).dropna(), errors="coerce")
    if len(y) < 5: return float("nan")
    x = np.arange(len(y), dtype=float)
    m, _ = np.polyfit(x, y.values, 1)
    base = float(y.iloc[-1]) if len(y) else float("nan")
    if not (np.isfinite(m) and np.isfinite(base) and base>0): return float("nan")
    return (m / base) * 1e4

def _percentile_of_value(x: float, hist: pd.Series) -> float:
    hist = pd.to_numeric(pd.Series(hist).dropna(), errors="coerce")
    if hist.empty or not np.isfinite(x): return float("nan")
    return float((hist < x).mean() * 100.0)

def _bps_dist(a: float | pd.Series, b: float | pd.Series) -> pd.Series | float:
    """Absolute distance in basis points between a and b (|a/b - 1|*1e4)."""
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        return (np.abs((pd.Series(a)/pd.Series(b)) - 1.0) * 1e4).astype(float)
    if not (np.isfinite(a) and np.isfinite(b) and b != 0): return float("nan")
    return abs((a/b - 1.0) * 1e4)

# ───────── main analyzer ─────────
def analyze_uptrend_gap(ticker: str, date_str: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    P = P_DEFAULT.copy()
    if overrides:
        for k,v in overrides.items(): P[k] = v

    tgt = pd.to_datetime(date_str).tz_localize(None)
    d_start = (tgt - pd.Timedelta(days=260)).strftime("%Y-%m-%d")
    d_end   = (tgt + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    d = fetch_daily(ticker, d_start, d_end)
    if d.empty: return {"ticker": ticker, "date": date_str, "error": "no_daily_data"}
    m = add_daily_metrics(d)

    # map target to index; choose D0 — EXACT match only (no pad across holidays)
    target = pd.Timestamp(tgt.date())
    try:
        idx_t = m.index.get_loc(target)  # exact session required
    except KeyError:
        return {"ticker": ticker, "date": date_str, "error": f"no_exact_session_for_{target.date()}"}
    i = idx_t  # D0
    if i < 1 or i >= len(m): return {"ticker": ticker, "date": date_str, "error": "need valid D0 with prior day"}

    r0, r1 = m.iloc[i], m.iloc[i-1]

    # ── liquidity
    adv20 = float((m["Close"]*m["Volume"]).rolling(20, min_periods=20).mean().shift(1).iloc[i-1])
    liquidity_ok = (float(r1.get("Prev_Close", np.nan)) >= P["price_min"]) and (adv20 >= P["adv20_min_usd"])

    # ── channel via EMA-distance (trend seed = last 4h red before D-1 open)
    t0 = (pd.Timestamp(m.index[i-1].date()) - pd.Timedelta(days=P["trend_seed_4h_lookback_days"]))
    t1 = pd.Timestamp(m.index[i-1].date())
    h4 = remove_fake_wicks(fetch_polygon_agg(ticker, 4, "hour", t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d")))
    last_red_ts = None
    if not h4.empty:
        d1_open_ts = pd.Timestamp(f"{m.index[i-1].date()} 09:30:00", tz=ET)
        h4 = h4[(h4.index < d1_open_ts) & (h4.index.dayofweek < 5)]
        reds = h4[h4["Close"] < h4["Open"]]
        if not reds.empty: last_red_ts = reds.index.max()
    trend_seed_day = (last_red_ts.date() if last_red_ts is not None else (m.index[i-1] - pd.Timedelta(days=60)).date())

    chan = m[(m.index.date >= trend_seed_day) & (m.index < m.index[i])]
    if bool(P.get("use_atr_norm_for_channel", True)):
        dist_hist = chan["High_minus_EMA20_over_ATR"]
        d1_dist   = float(r1["High_minus_EMA20_over_ATR"])
        d0_open_d = float(r0["Open_minus_EMA20_over_ATR"])
    else:
        dist_hist = chan["High_minus_EMA20"]
        d1_dist   = float(r1["High_minus_EMA20"])
        d0_open_d = float(r0["Open_minus_EMA20"])
    top_third_pct = 100.0 * (1.0 - float(P["upper_channel_top_frac"]))   # 66.7
    hist_clean = pd.to_numeric(dist_hist, errors="coerce").dropna()
    thr_val = float(np.nanpercentile(hist_clean, top_third_pct)) if not hist_clean.empty else float("nan")
    d1_pctile = _percentile_of_value(d1_dist, dist_hist)
    d0_pctile = _percentile_of_value(d0_open_d, dist_hist)
    in_top_third_d1 = (np.isfinite(d1_dist) and np.isfinite(thr_val) and d1_dist >= thr_val)
    in_top_third_d0 = (np.isfinite(d0_open_d) and np.isfinite(thr_val) and d0_open_d >= thr_val)

    # similarity z-score (info)
    if not hist_clean.empty:
        mu, sd = float(hist_clean.mean()), float(hist_clean.std(ddof=0))
        d1_z = (d1_dist - mu) / sd if (np.isfinite(sd) and sd>0) else float("nan")
    else:
        d1_z = float("nan")

    lr_bps = _lr_slope_bps_per_day(chan["Close"]) if bool(P.get("report_slopes", True)) else float("nan")

    # ── D-1 strength
    d1_ret_pct   = _pct(r1["Close"], r1["Prev_Close"]) if pd.notna(r1["Prev_Close"]) else float("nan")
    d1_green     = bool(r1["Close"] > r1["Open"])
    d1_body_atr  = float(r1["Body_over_ATR"])
    vol_avg_prev = float(m["Volume"].rolling(14, min_periods=14).mean().shift(1).iloc[i-1])
    d1_vol_mult  = (float(r1["Volume"]) / vol_avg_prev) if (np.isfinite(vol_avg_prev) and vol_avg_prev>0) else float("nan")
    d1_ok = (
        ((not P["d1_green_required"]) or d1_green) and
        (np.isfinite(d1_ret_pct) and d1_ret_pct >= P["d1_return_min_pct"]) and
        (np.isfinite(d1_body_atr) and d1_body_atr >= P["d1_body_atr_min"]) and
        (np.isfinite(d1_vol_mult) and d1_vol_mult >= P["d1_vol_mult_min"])
    )

    # ── D0 gap
    gap_over_atr   = float(r0["Gap_over_ATR"])
    gap_pct        = _pct(r0["Open"], r1["Close"])
    open_over_ema9 = float(r0["Open"] / r0["EMA_9"]) if pd.notna(r0.get("EMA_9", np.nan)) else float("nan")
    gap_ok = (
        (np.isfinite(gap_over_atr) and gap_over_atr >= P["gap_over_atr_min"]) and
        (np.isfinite(gap_pct) and gap_pct >= P["gap_min_pct"]) and
        (np.isfinite(open_over_ema9) and open_over_ema9 >= P["open_over_ema9_min"])
    )

    # ── Hourly dev-band confirm (tolerant)
    touched_between_close_and_11 = None
    near_7_to_11 = None
    retest_hh_7_to_11 = None
    hourly_ema9_slope_pct = None
    first_touch_ts = None
    first_touch_band = None
    near_min_bps = None
    preopen_hh_px = None

    if bool(P.get("check_hourly", True)):
        start_h = (pd.Timestamp(m.index[i-1].date()) - pd.Timedelta(days=P["hourly_window_days_back"])).strftime("%Y-%m-%d")
        end_h   = pd.Timestamp(m.index[i].date()).strftime("%Y-%m-%d")
        h1 = remove_fake_wicks(fetch_polygon_agg(ticker, 1, "hour", start_h, end_h))
        if not h1.empty:
            # soften mega-wicks BEFORE band calc
            soften_1h_wicks_for_atr(h1, span_for_rng=72, cap_mult=6.0)

            s_t = pd.to_datetime("04:00").time(); e_t = pd.to_datetime("20:00").time()
            h1 = h1[(h1.index.dayofweek < 5) & (h1.index.time >= s_t) & (h1.index.time <= e_t)].copy()
            add_bands(h1, 72, 89, 6.9, 9.6, 6.9, 6.3, prefix="l_")  # your TV inputs
            h1["s_emaS"] = h1["Close"].ewm(span=9, min_periods=0).mean()

            d1_1600 = pd.Timestamp(f"{m.index[i-1].date()} 16:00:00", tz=ET)
            d0_0700 = pd.Timestamp(f"{m.index[i].date()} 07:00:00", tz=ET)
            d0_0930 = pd.Timestamp(f"{m.index[i].date()} 09:30:00", tz=ET)
            d0_1100 = pd.Timestamp(f"{m.index[i].date()} 11:00:00", tz=ET)

            tol = float(P["hourly_touch_tolerance_bps"]) / 1e4

            win_touch = h1[(h1.index >= d1_1600) & (h1.index <= d0_1100)]
            if not win_touch.empty:
                mask1 = (win_touch["High"] >= win_touch["l_UPP1"] * (1.0 - tol))
                mask2 = (win_touch["High"] >= win_touch["l_UPP2"] * (1.0 - tol))
                any_touch = bool(mask1.any() or mask2.any())
                touched_between_close_and_11 = any_touch
                if any_touch:
                    ts1 = mask1[mask1].index.min() if mask1.any() else None
                    ts2 = mask2[mask2].index.min() if mask2.any() else None
                    if ts1 is not None and (ts2 is None or ts1 <= ts2):
                        first_touch_ts, first_touch_band = ts1.isoformat(), "UPP1"
                    elif ts2 is not None:
                        first_touch_ts, first_touch_band = ts2.isoformat(), "UPP2"

            pre = h1[(h1.index >= d1_1600) & (h1.index < d0_0930)]
            preopen_hh_px = float(pre["High"].max()) if not pre.empty else None

            morn = h1[(h1.index >= d0_0700) & (h1.index <= d0_1100)]
            if not morn.empty:
                near_bps_series = _bps_dist(morn["High"], morn["l_UPP1"])
                near_min_bps = float(np.nanmin(near_bps_series)) if np.isfinite(near_bps_series).any() else None
                near_7_to_11 = bool((near_min_bps is not None) and (near_min_bps <= float(P["hourly_near_lupp1_bps"])))
                if preopen_hh_px is not None and np.isfinite(preopen_hh_px):
                    hh_bps_series = _bps_dist(morn["High"], preopen_hh_px)
                    retest_hh_7_to_11 = bool(np.isfinite(hh_bps_series).any() and np.nanmin(hh_bps_series) <= float(P["hourly_retest_hh_bps"]))
                ema9_a = float(morn["s_emaS"].iloc[0]); ema9_b = float(morn["s_emaS"].iloc[-1])
                hourly_ema9_slope_pct = _pct(ema9_b, ema9_a) if np.isfinite(ema9_a) and np.isfinite(ema9_b) else float("nan")

    # hourly decision
    hourly_ok = True
    if bool(P.get("check_hourly", True)):
        cond_touch = (touched_between_close_and_11 if P.get("require_hourly_touch", True) else True)
        cond_near  = ((near_7_to_11 or retest_hh_7_to_11) if P.get("require_hourly_near", True) else True)
        hourly_ok  = bool(cond_touch and cond_near)

    # final daily channel decision (upper third)
    channel_ok = bool(in_top_third_d1 and in_top_third_d0)

    qualifies = bool(liquidity_ok and channel_ok and d1_ok and gap_ok and hourly_ok)

    row = {
        "Ticker": ticker,
        "Target_Date": m.index[i].date().isoformat(),

        # Liquidity
        "Prev_Close": round(float(r1["Prev_Close"]),2) if pd.notna(r1["Prev_Close"]) else None,
        "ADV20_$": round(adv20),

        # Channel via EMA-distance
        "TrendSeed_4h_last_red": last_red_ts.isoformat() if last_red_ts is not None else None,
        "TrendSeed_Day": str(trend_seed_day),
        "Channel_metric": "High−EMA20" + ("/ATR" if P.get("use_atr_norm_for_channel", True) else ""),
        "TopThird_Threshold": round(float(thr_val), 4) if np.isfinite(thr_val) else None,
        "D1_High_EMA20_Dist": round(float(d1_dist), 4) if np.isfinite(d1_dist) else None,
        "D0_Open_EMA20_Dist": round(float(d0_open_d), 4) if np.isfinite(d0_open_d) else None,
        "D1_Dist_pctile": round(float(d1_pctile), 1) if np.isfinite(d1_pctile) else None,
        "D0_Open_Dist_pctile": round(float(d0_pctile), 1) if np.isfinite(d0_pctile) else None,
        "D1_in_TopThird": bool(in_top_third_d1),
        "D0_in_TopThird": bool(in_top_third_d0),
        "D1_Dist_z": round(float(d1_z), 2) if np.isfinite(d1_z) else None,

        # D-1 strength
        "D1_Return_%": round(float(d1_ret_pct), 2) if np.isfinite(d1_ret_pct) else None,
        "D1_Body/ATR": round(float(d1_body_atr), 2) if np.isfinite(d1_body_atr) else None,
        "D1_Vol/Avg": round(float(d1_vol_mult), 2) if np.isfinite(d1_vol_mult) else None,

        # Gap
        "Gap_%": round(float(gap_pct), 2) if np.isfinite(gap_pct) else None,
        "Gap/ATR": round(float(gap_over_atr), 2) if np.isfinite(gap_over_atr) else None,
        "Open/EMA9": round(float(open_over_ema9), 3) if np.isfinite(open_over_ema9) else None,

        # Hourly confirmation (with diagnostics)
        "Hour_touch_lUPP1_or_2_D1close_to_11": touched_between_close_and_11 if touched_between_close_and_11 is not None else None,
        "Hour_first_touch_ts": first_touch_ts,
        "Hour_first_touch_band": first_touch_band,
        "Hour_near_min_bps_7to11": round(float(near_min_bps),0) if near_min_bps is not None else None,
        "Hour_near_lUPP1_7to11": near_7_to_11 if near_7_to_11 is not None else None,
        "Hour_retest_preopen_HH_px": round(float(preopen_hh_px),2) if preopen_hh_px is not None else None,
        "Hour_retest_preopen_HH_7to11": retest_hh_7_to_11 if retest_hh_7_to_11 is not None else None,
        "Hourly_EMA9_slope_7to11_%": round(float(hourly_ema9_slope_pct), 2) if np.isfinite(hourly_ema9_slope_pct) else None,

        # Slopes (daily info)
        "LR_Close_slope_bps_per_day": round(float(lr_bps), 2) if np.isfinite(lr_bps) else None,
        "Slope_EMA9_5d_%": round(float(m["Slope_EMA9_5d_%"].iloc[i-1]), 2) if pd.notna(m["Slope_EMA9_5d_%"].iloc[i-1]) else None,
        "Slope_EMA20_5d_%": round(float(m["Slope_EMA20_5d_%"].iloc[i-1]), 2) if pd.notna(m["Slope_EMA20_5d_%"].iloc[i-1]) else None,

        # OK flags
        "Liquidity_OK": bool(liquidity_ok),
        "Channel_TopThird_OK": bool(channel_ok),
        "D1_Strong_OK": bool(d1_ok),
        "Gap_OK": bool(gap_ok),
        "Hourly_OK": bool(hourly_ok),

        # Final
        "Qualifies_DailyUptrendGap": bool(qualifies),
    }
    return {"ticker": ticker, "date_input": pd.Timestamp(tgt.date()).isoformat(),
            "evaluated_session": row["Target_Date"], "table": row, "P_used": P, "error": None}

# ───────── printing / batch ─────────
def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)): return "-"
    if isinstance(v, bool): return "✓" if v else "✗"
    if isinstance(v, (int, np.integer)): return str(int(v))
    if isinstance(v, float): return f"{v:.2f}"
    return str(v)

def print_stacked(res: Dict[str, Any]) -> None:
    t = res["table"]; sym = t["Ticker"]; dt = t["Target_Date"]
    print(f"\n{sym}  {dt} — Daily Uptrend Gap")
    print("-"*84)
    left = [
        ("Prev_Close", t.get("Prev_Close")), ("ADV20_$", t.get("ADV20_$")), ("Liquidity_OK", t.get("Liquidity_OK")),
        ("TrendSeed_Day", t.get("TrendSeed_Day")), ("Channel_metric", t.get("Channel_metric")),
        ("TopThird_Threshold", t.get("TopThird_Threshold")),
        ("D1_High_EMA20_Dist", t.get("D1_High_EMA20_Dist")), ("D0_Open_EMA20_Dist", t.get("D0_Open_EMA20_Dist")),
        ("D1_Dist_pctile", t.get("D1_Dist_pctile")), ("D0_Open_Dist_pctile", t.get("D0_Open_Dist_pctile")),
        ("D1_in_TopThird", t.get("D1_in_TopThird")), ("D0_in_TopThird", t.get("D0_in_TopThird")),
        ("D1_Dist_z", t.get("D1_Dist_z")), ("Channel_TopThird_OK", t.get("Channel_TopThird_OK")),
        ("D1_Return_%", t.get("D1_Return_%")), ("D1_Body/ATR", t.get("D1_Body/ATR")), ("D1_Vol/Avg", t.get("D1_Vol/Avg")),
        ("Gap_%", t.get("Gap_%")), ("Gap/ATR", t.get("Gap/ATR")), ("Open/EMA9", t.get("Open/EMA9")), ("Gap_OK", t.get("Gap_OK")),
        ("Hour_touch_lUPP1_or_2_D1close_to_11", t.get("Hour_touch_lUPP1_or_2_D1close_to_11")),
        ("Hour_first_touch_ts", t.get("Hour_first_touch_ts")),
        ("Hour_first_touch_band", t.get("Hour_first_touch_band")),
        ("Hour_near_min_bps_7to11", t.get("Hour_near_min_bps_7to11")),
        ("Hour_near_lUPP1_7to11", t.get("Hour_near_lUPP1_7to11")),
        ("Hour_retest_preopen_HH_px", t.get("Hour_retest_preopen_HH_px")),
        ("Hour_retest_preopen_HH_7to11", t.get("Hour_retest_preopen_HH_7to11")),
        ("Hourly_EMA9_slope_7to11_%", t.get("Hourly_EMA9_slope_7to11_%")),
        ("Hourly_OK", t.get("Hourly_OK")),
        ("LR_Close_slope_bps_per_day", t.get("LR_Close_slope_bps_per_day")),
        ("Slope_EMA9_5d_%", t.get("Slope_EMA9_5d_%")),
        ("Slope_EMA20_5d_%", t.get("Slope_EMA20_5d_%")),
        ("Qualifies_DailyUptrendGap", t.get("Qualifies_DailyUptrendGap")),
    ]
    for k, v in left:
        print(f"{k:<36}: {_fmt(v)}")

def analyze_batch(
    cases: List[Tuple[str, str]],
    overrides: Dict[str, Any] | None = None,
    csv_path: str | None = None,
    style: str | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t, d in cases:
        res = analyze_uptrend_gap(t, d, overrides)
        if res.get("error"):
            print(f"{t} {d}: ERROR {res['error']}")
            continue
        if (style or P_DEFAULT["style"]) == "stacked":
            print_stacked(res)
        if "table" in res:
            rows.append(res["table"])

    df = pd.DataFrame(rows)
    if not df.empty and (style or P_DEFAULT["style"]) == "table":
        pd.set_option("display.max_columns", None, "display.width", 0)
        print(df.to_string(index=False))

    if csv_path and not df.empty:
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV -> {csv_path}")

    return df

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Daily Uptrend Gap Analyzer (EMA-distance channel + hourly dev-band confirm)"
    )
    ap.add_argument("--ticker", type=str, default=None)
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (evaluates this exact session as D0)")
    ap.add_argument("--batch", type=str, default=None, help="Comma list 'SYM:YYYY-MM-DD,...'")
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--style", type=str, choices=["stacked", "table"], default=P_DEFAULT["style"])
    ap.add_argument("--target_as", type=str, choices=["d0", "d1"], default=P_DEFAULT["target_as"])
    ap.add_argument("--overrides", type=str, default=None, help="Comma 'key=value' overrides")

    args = ap.parse_args()

    overrides: Dict[str, Any] = {"target_as": args.target_as, "style": args.style}
    if args.overrides:
        for kv in args.overrides.split(","):
            if not kv.strip() or "=" not in kv: continue
            k, v = kv.split("=", 1); k = k.strip(); v = v.strip()
            try:
                if v.lower() in ("true", "false"): overrides[k] = (v.lower() == "true")
                elif "." in v:                    overrides[k] = float(v)
                else:                             overrides[k] = int(v)
            except Exception:
                overrides[k] = v

    cases: List[Tuple[str, str]] = []
    if args.ticker and args.date:
        cases.append((args.ticker.upper(), pd.to_datetime(args.date).date().isoformat()))
    if args.batch:
        for pair in args.batch.split(","):
            if ":" not in pair: continue
            t, d = pair.split(":", 1)
            cases.append((t.strip().upper(), pd.to_datetime(d.strip()).date().isoformat()))

    if not cases:
        # defaults for quick check
        cases = [
            ("NVDA", "2024-06-20"),
            ("NVDA", "2025-07-31"),
            ("NVDA", "2025-07-29"),
            ("MSTR", "2024-10-14"),
        ]

    analyze_batch(cases, overrides=overrides, csv_path=args.csv, style=args.style)
