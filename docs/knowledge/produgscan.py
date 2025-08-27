# scan_uptrend_gap_fast_graded.py  (fixed)
from __future__ import annotations

import os, math, time, random, argparse, warnings
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_KEY  = os.environ.get("POLYGON_API_KEY", "Fm7brz4s23eSocDErnL68cE7wspz2K1I")
BASE_URL = "https://api.polygon.io"
ET       = "US/Eastern"

MAX_WORKERS = 8
USE_CACHE   = True
ARTIFACTS_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "lc_uptrend_gap")
CACHE_DIR     = os.path.join(ARTIFACTS_DIR, "cache", "daily")
os.makedirs(CACHE_DIR, exist_ok=True)
warnings.filterwarnings("ignore", message=r".*Series\.fillna with 'method'.*")

SYMBOLS = [
    'NVDA','AAPL','MSFT','AMZN','META','TSLA','AMD','AVGO','GOOGL','SPY','QQQ'
]

# thresholds: slightly under NVDA 2024-06-20; rounded where sensible
P: Dict[str, Any] = {
    "price_min": 20.0,
    "adv20_min_usd": 250_000_000,

    "trend_seed_4h_lookback_days": 40,
    "upper_channel_top_frac": 0.333,
    "use_atr_norm_for_channel": True,

    "d1_green_required": True,
    "d1_return_min_pct": 3.50,
    "d1_body_atr_min": 0.40,
    "d1_vol_mult_min": 0.70,

    "gap_over_atr_min": 0.40,
    "gap_min_pct": 3.00,
    "open_over_ema9_min": 1.05,

    "check_hourly": True,
    "hourly_window_days_back": 6,
    "require_hourly_touch": True,
    "require_hourly_near": False,
    "hourly_touch_tolerance_bps": 25.0,
    "hourly_near_lupp1_bps": 60.0,
    "hourly_retest_hh_bps": 60.0,

    # 1h sanitization (split/holiday protection)
    "sanitize_intraday": True,
    "wick_range_win": 200,
    "wick_range_mult_cap": 6.0,
    "wick_pct_cap": 12.0,
    "gap_tr_bar_minutes": 60,
    "gap_tr_factor": 2.0,
}

# ───────── HTTP ─────────
def _init_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=6, connect=6, read=6, backoff_factor=0.8,
                    status_forcelist=[429,500,502,503,504],
                    allowed_methods=frozenset(["GET"]),
                    respect_retry_after_header=True)
    ad = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
    s.mount("https://", ad); s.mount("http://", ad)
    s.headers.update({"User-Agent":"uptrend-gap-fast/graded-2.1"})
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

# ───────── daily fetch + cache ─────────
def _cache_path(sym: str) -> str:
    return os.path.join(CACHE_DIR, f"{sym.upper()}.parquet")

def _fetch_daily_http(tkr: str, start: str, end: str) -> pd.DataFrame:
    url = f"{BASE_URL}/v2/aggs/ticker/{tkr}/range/1/day/{start}/{end}"
    r   = _robust_get(url, params={"apiKey": API_KEY, "adjusted":"true", "sort":"asc", "limit":50000})
    rows = r.json().get("results", [])
    if not rows: return pd.DataFrame()
    return (pd.DataFrame(rows)
            .assign(Date=lambda d: pd.to_datetime(d["t"], unit="ms", utc=True))
            .rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
            .set_index("Date")[["Open","High","Low","Close","Volume"]]
            .sort_index())

def fetch_daily(tkr: str, start: str, end: str) -> pd.DataFrame:
    if not USE_CACHE: return _fetch_daily_http(tkr, start, end)
    path = _cache_path(tkr)
    need_start = pd.to_datetime(start).tz_localize("UTC")
    need_end   = pd.to_datetime(end).tz_localize("UTC")

    if os.path.exists(path):
        try: df_all = pd.read_parquet(path)
        except Exception: df_all = pd.DataFrame()
    else:
        df_all = pd.DataFrame()

    to_fetch: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if df_all.empty:
        to_fetch.append((need_start, need_end))
    else:
        have_start = df_all.index.min(); have_end = df_all.index.max()
        if need_start < have_start: to_fetch.append((need_start, have_start - pd.Timedelta(days=1)))
        if need_end > have_end:     to_fetch.append((have_end + pd.Timedelta(days=1), need_end))

    parts = [df_all]
    for a,b in to_fetch:
        if a <= b: parts.append(_fetch_daily_http(tkr, a.date().isoformat(), b.date().isoformat()))
    if parts:
        df_all = pd.concat(parts) if len(parts) > 1 else parts[0]
        if not df_all.empty:
            df_all = df_all[~df_all.index.duplicated(keep="last")].sort_index()
            try: df_all.to_parquet(path)
            except Exception: pass
    if df_all.empty: return df_all
    return df_all.loc[(df_all.index >= need_start) & (df_all.index <= need_end)]

# ───────── intraday fetch ─────────
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

# ───────── daily metrics ─────────
def add_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    m = df.copy()
    try: m.index = m.index.tz_localize(None)
    except Exception: pass

    m["EMA_9"]  = m["Close"].ewm(span=9,  adjust=False).mean()
    m["EMA_20"] = m["Close"].ewm(span=20, adjust=False).mean()
    m["EMA_50"] = m["Close"].ewm(span=50, adjust=False).mean()

    hi_lo   = m["High"] - m["Low"]
    hi_prev = (m["High"] - m["Close"].shift(1)).abs()
    lo_prev = (m["Low"]  - m["Close"].shift(1)).abs()
    m["TR"]      = pd.concat([hi_lo, hi_prev, lo_prev], axis=1).max(axis=1)
    m["ATR_raw"] = m["TR"].rolling(14, min_periods=14).mean()
    m["ATR"]     = m["ATR_raw"].shift(1)

    m["VOL_AVG"] = m["Volume"].rolling(14, min_periods=14).mean().shift(1)
    m["ADV20_$"] = (m["Close"]*m["Volume"]).rolling(20, min_periods=20).mean().shift(1)
    m["Prev_Close"] = m["Close"].shift(1)

    m["Gap_abs"]        = (m["Open"] - m["Close"].shift(1)).abs()
    m["Gap_over_ATR"]   = m["Gap_abs"] / m["ATR"]
    m["Body_over_ATR"]  = (m["Close"] - m["Open"]) / m["ATR"]
    m["Open_over_EMA9"] = m["Open"] / m["EMA_9"]

    m["High_minus_EMA20"] = (m["High"] - m["EMA_20"])
    m["Open_minus_EMA20"] = (m["Open"] - m["EMA_20"])
    m["High_minus_EMA20_over_ATR"] = m["High_minus_EMA20"] / m["ATR"]
    m["Open_minus_EMA20_over_ATR"] = m["Open_minus_EMA20"] / m["ATR"]
    return m

# ───────── utils ─────────
def _pct(a: float, b: float) -> float:
    try: return (a/b - 1.0) * 100.0 if (np.isfinite(a) and np.isfinite(b) and b!=0) else float("nan")
    except Exception: return float("nan")

def _bps_dist(a: float | pd.Series, b: float | pd.Series) -> pd.Series | float:
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        return (np.abs((pd.Series(a)/pd.Series(b)) - 1.0) * 1e4).astype(float)
    if not (np.isfinite(a) and np.isfinite(b) and b != 0): return float("nan")
    return abs((a/b - 1.0) * 1e4)

# ───────── daily prefilter (fixed green requirement) ─────────
def prefilter_daily_candidates(m: pd.DataFrame) -> np.ndarray:
    n = len(m)
    if n < 2: return np.array([], dtype=int)
    C  = m["Close"].to_numpy(float); O = m["Open"].to_numpy(float)
    ATR = m["ATR"].to_numpy(float); VOL = m["Volume"].to_numpy(float)
    VOL_AVG = m["VOL_AVG"].to_numpy(float); EMA9 = m["EMA_9"].to_numpy(float)
    ADV20 = m["ADV20_$"].to_numpy(float); PrevClose = m["Prev_Close"].to_numpy(float)

    liq = (PrevClose >= P["price_min"]) & (ADV20 >= P["adv20_min_usd"])

    d1_ret = (C / PrevClose - 1.0) * 100.0
    d1_green = (C > O)
    body_atr = ((C - O) / ATR)
    vol_mult = (VOL / VOL_AVG)

    green_req = d1_green if P["d1_green_required"] else np.ones_like(d1_green, dtype=bool)

    d1_ok = green_req & \
            np.isfinite(d1_ret) & (d1_ret >= P["d1_return_min_pct"]) & \
            np.isfinite(body_atr) & (body_atr >= P["d1_body_atr_min"]) & \
            np.isfinite(vol_mult) & (vol_mult >= P["d1_vol_mult_min"])

    gap_over_atr = m["Gap_over_ATR"].to_numpy(float)
    gap_pct = (O / PrevClose - 1.0) * 100.0
    open_over_ema9 = (O / EMA9)
    gap_ok = np.isfinite(gap_over_atr) & (gap_over_atr >= P["gap_over_atr_min"]) & \
             np.isfinite(gap_pct) & (gap_pct >= P["gap_min_pct"]) & \
             np.isfinite(open_over_ema9) & (open_over_ema9 >= P["open_over_ema9_min"])

    mask = np.zeros(n, dtype=bool)
    mask[1:] = liq[:-1] & d1_ok[:-1] & gap_ok[1:]
    return np.where(mask)[0]

# ───────── 4h seed ─────────
def compute_trend_seed_days_for_dates(sym: str, d1_dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, pd.Timestamp]:
    if not d1_dates: return {}
    d1_dates = sorted(d1_dates)
    start4 = (pd.Timestamp(d1_dates[0].date()) - pd.Timedelta(days=P["trend_seed_4h_lookback_days"])).strftime("%Y-%m-%d")
    end4   = pd.Timestamp(d1_dates[-1].date()).strftime("%Y-%m-%d")
    h4 = remove_fake_wicks(fetch_polygon_agg(sym, 4, "hour", start4, end4))
    if h4.empty: return {d: (d - pd.Timedelta(days=60)) for d in d1_dates}
    reds = h4[h4["Close"] < h4["Open"]].copy()
    if reds.empty: return {d: (d - pd.Timedelta(days=60)) for d in d1_dates}
    reds = reds.reset_index().sort_values("ts")
    q = pd.DataFrame({"q": [pd.Timestamp(f"{d.date()} 09:30:00", tz=ET) for d in d1_dates]})
    merged = pd.merge_asof(q.sort_values("q"), reds[["ts"]].rename(columns={"ts":"q"}), on="q",
                           direction="backward", tolerance=pd.Timedelta("365D"))
    out: Dict[pd.Timestamp, pd.Timestamp] = {}
    for d, last_red_ts in zip(d1_dates, merged["q"].to_list()):
        out[d] = (d - pd.Timedelta(days=60)) if pd.isna(last_red_ts) else pd.Timestamp(last_red_ts.date())
    return out

# ───────── 1h sanitize + bands (fixed first_bar computation) ─────────
def _sanitize_1h_extremes(h1: pd.DataFrame) -> pd.DataFrame:
    if h1.empty or not P.get("sanitize_intraday", True): return h1
    df = h1.copy()
    body_hi = np.maximum(df["Open"], df["Close"])
    body_lo = np.minimum(df["Open"], df["Close"])
    rng = (df["High"] - df["Low"]).rolling(P["wick_range_win"], min_periods=50).median()
    cap = np.maximum(rng * P["wick_range_mult_cap"], body_hi * (P["wick_pct_cap"]/100.0))
    df["High_san"] = np.minimum(df["High"], body_hi + cap.ffill().fillna(0.0))
    df["Low_san"]  = np.maximum(df["Low"],  body_lo - cap.ffill().fillna(0.0))
    return df

def _add_bands_1h_sanitized(h1: pd.DataFrame, ema_short=72, ema_long=89, du1=6.9, du2=9.6, dl1=6.9, dl2=6.3, prefix="l_"):
    if h1.empty: return
    df = h1
    if "High_san" not in df.columns or "Low_san" not in df.columns:
        df["High_san"] = df["High"]; df["Low_san"] = df["Low"]

    hi_lo = df["High_san"] - df["Low_san"]
    close_shift = df["Close"].shift(1)
    hi_prev = (df["High_san"] - close_shift).abs()
    lo_prev = (df["Low_san"]  - close_shift).abs()
    tr_raw = pd.concat([hi_lo, hi_prev, lo_prev], axis=1).max(axis=1)

    dt = df.index.to_series().diff()
    first_bar = (dt > pd.Timedelta(minutes=P["gap_tr_bar_minutes"]))
    first_bar.iloc[0] = True

    tr = tr_raw.copy()
    tr.iloc[0] = hi_lo.iloc[0]
    tr[first_bar] = np.minimum(tr[first_bar], hi_lo[first_bar] * P["gap_tr_factor"])

    df[f"{prefix}emaS"] = df["Close"].ewm(span=ema_short, min_periods=0).mean()
    df[f"{prefix}emaL"] = df["Close"].ewm(span=ema_long,  min_periods=0).mean()
    df[f"{prefix}ATR_S"] = tr.rolling(ema_short, min_periods=1).mean()
    df[f"{prefix}ATR_L"] = tr.rolling(ema_long,  min_periods=1).mean()
    df[f"{prefix}UPP1"]  = df[f"{prefix}emaS"] + du1 * df[f"{prefix}ATR_S"]
    df[f"{prefix}UPP2"]  = df[f"{prefix}emaS"] + du2 * df[f"{prefix}ATR_S"]
    df[f"{prefix}LOW1"]  = df[f"{prefix}emaL"] - dl1 * df[f"{prefix}ATR_L"]
    df[f"{prefix}LOW2"]  = df[f"{prefix}emaL"] - dl2 * df[f"{prefix}ATR_L"]

# ───────── hourly confirm ─────────
def hourly_eval_for_dates(sym: str, dates_d0: List[pd.Timestamp]) -> Dict[pd.Timestamp, dict]:
    if not dates_d0: return {}
    start = (min(dates_d0) - pd.Timedelta(days=P["hourly_window_days_back"])).strftime("%Y-%m-%d")
    end   = max(dates_d0).strftime("%Y-%m-%d")
    h1 = remove_fake_wicks(fetch_polygon_agg(sym, 1, "hour", start, end))
    if h1.empty: return {d: {"Hourly_OK": False} for d in dates_d0}

    s_t = pd.to_datetime("04:00").time(); e_t = pd.to_datetime("20:00").time()
    h1 = h1[(h1.index.dayofweek < 5) & (h1.index.time >= s_t) & (h1.index.time <= e_t)].copy()
    h1 = _sanitize_1h_extremes(h1)
    _add_bands_1h_sanitized(h1, 72, 89, 6.9, 9.6, 6.9, 6.3, prefix="l_")
    h1["s_emaS"] = h1["Close"].ewm(span=9, min_periods=0).mean()

    tol = float(P["hourly_touch_tolerance_bps"]) / 1e4
    out: Dict[pd.Timestamp, dict] = {}

    for d0 in dates_d0:
        d1c = pd.Timestamp(f"{(d0 - pd.Timedelta(days=1)).date()} 16:00:00", tz=ET)
        d0_0700 = pd.Timestamp(f"{d0.date()} 07:00:00", tz=ET)
        d0_0930 = pd.Timestamp(f"{d0.date()} 09:30:00", tz=ET)
        d0_1100 = pd.Timestamp(f"{d0.date()} 11:00:00", tz=ET)

        win_touch = h1[(h1.index >= d1c) & (h1.index <= d0_1100)]
        touch = False; tts = None; tband = None
        if not win_touch.empty:
            m1 = (win_touch["High"] >= win_touch["l_UPP1"] * (1.0 - tol))
            m2 = (win_touch["High"] >= win_touch["l_UPP2"] * (1.0 - tol))
            touch = bool(m1.any() or m2.any())
            if touch:
                ts1 = m1[m1].index.min() if m1.any() else None
                ts2 = m2[m2].index.min() if m2.any() else None
                if ts1 is not None and (ts2 is None or ts1 <= ts2):
                    tts, tband = ts1.isoformat(), "UPP1"
                elif ts2 is not None:
                    tts, tband = ts2.isoformat(), "UPP2"

        morn = h1[(h1.index >= d0_0700) & (h1.index <= d0_1100)]
        near_min_bps = np.nan; ema9_slope = np.nan
        if not morn.empty:
            near_bps = _bps_dist(morn["High"], morn["l_UPP1"])
            near_min_bps = float(np.nanmin(near_bps)) if np.isfinite(near_bps).any() else np.nan
            ema9_a = float(morn["s_emaS"].iloc[0]); ema9_b = float(morn["s_emaS"].iloc[-1])
            ema9_slope = _pct(ema9_b, ema9_a) if np.isfinite(ema9_a) and np.isfinite(ema9_b) else np.nan

        cond_touch = (touch if P["require_hourly_touch"] else True)
        ok = bool(cond_touch)

        out[d0] = {
            "Hourly_OK": ok,
            "Hour_touch": touch,
            "Hour_first_touch_ts": tts,
            "Hour_first_touch_band": tband,
            "Hour_near_min_bps_7to11": round(float(near_min_bps),1) if np.isfinite(near_min_bps) else np.nan,
            "Hourly_EMA9_slope_7to11_%": round(float(ema9_slope),2) if np.isfinite(ema9_slope) else np.nan,
        }
    return out

# ───────── scan one symbol ─────────
def scan_symbol(sym: str, fr: str, to: str) -> pd.DataFrame:
    d = fetch_daily(sym, fr, to)
    if d.empty: return pd.DataFrame()
    m = add_daily_metrics(d)

    idxs = prefilter_daily_candidates(m)
    if idxs.size == 0: return pd.DataFrame()

    d0_dates = [pd.Timestamp(m.index[i].date()) for i in idxs]
    d1_dates = [pd.Timestamp(m.index[i-1].date()) for i in idxs]

    seed_map = compute_trend_seed_days_for_dates(sym, d1_dates)

    rows = []
    for i, d0, d1 in zip(idxs, d0_dates, d1_dates):
        seed_day = seed_map.get(pd.Timestamp(d1), d1 - pd.Timedelta(days=60))
        ch_ok, thr, d1_d, d0_d, d1_pct, d0_pct = channel_top_third_ok(m, int(i), pd.Timestamp(seed_day))
        if not ch_ok: continue

        r0, r1 = m.iloc[i], m.iloc[i-1]
        rows.append({
            "Ticker": sym, "Date": d0.date().isoformat(),
            "TrendSeed_Day": str(pd.Timestamp(seed_day).date()),
            "TopThird_Threshold": float(thr),
            "D1_Dist_pctile": float(d1_pct), "D0_Open_Dist_pctile": float(d0_pct),
            "Gap/ATR": float(r0["Gap_over_ATR"]), "Open/EMA9": float(r0["Open_over_EMA9"]),
            "D1_Return_%": _pct(r1["Close"], r1["Prev_Close"]),
            "D1_Body/ATR": float(r1["Body_over_ATR"]),
            "D1_Vol/Avg": float(r1["Volume"] / r1["VOL_AVG"]) if r1["VOL_AVG"] else np.nan,
        })

    df = pd.DataFrame(rows)
    if df.empty or not P.get("check_hourly", True): return df

    d0_ts = [pd.Timestamp(x) for x in pd.to_datetime(df["Date"]).to_list()]
    hourly_map = hourly_eval_for_dates(sym, d0_ts)

    out_rows = []
    for _, rr in df.iterrows():
        d0 = pd.Timestamp(rr["Date"])
        h  = hourly_map.get(d0, {"Hourly_OK": False})
        if not h.get("Hourly_OK", False): continue
        out_rows.append({**rr.to_dict(), **h})
    return pd.DataFrame(out_rows)

# ───────── grading ─────────
FEATURES = [
    "Hour_touch","Hourly_EMA9_slope_7to11_%","Hour_near_min_bps_7to11",
    "Gap/ATR","Open/EMA9","D1_Return_%","D1_Body/ATR","D1_Vol/Avg",
    "D1_Dist_pctile","D0_Open_Dist_pctile"
]
WEIGHTS = {
    "Hour_touch": 1.2,
    "Hourly_EMA9_slope_7to11_%": 0.8,
    "Hour_near_min_bps_7to11": 0.8,
    "Gap/ATR": 0.8, "Open/EMA9": 0.6,
    "D1_Return_%": 0.8, "D1_Body/ATR": 0.6, "D1_Vol/Avg": 0.6,
    "D1_Dist_pctile": 0.9, "D0_Open_Dist_pctile": 1.0,
}
def _cap(v, lo, hi):
    return 0.0 if (v is None or not np.isfinite(v)) else float(np.clip((v - lo)/(hi - lo), 0.0, 1.0))
def _invcap(v, hi, lo):
    return 0.0 if (v is None or not np.isfinite(v)) else float(np.clip((hi - v)/(hi - lo), 0.0, 1.0))
TRANS = {
    "Hour_touch": lambda v: 1.0 if bool(v) else 0.0,
    "Hourly_EMA9_slope_7to11_%": lambda v: _cap(v if np.isfinite(v) else 0.0, 0.0, 2.5),
    "Hour_near_min_bps_7to11":   lambda v: _invcap(v, 60.0, 0.0),
    "Gap/ATR":       lambda v: _cap(v, 0.40, 1.50),
    "Open/EMA9":     lambda v: _cap(v, 1.05, 1.15),
    "D1_Return_%":   lambda v: _cap(v, 3.5, 8.0),
    "D1_Body/ATR":   lambda v: _cap(v, 0.40, 1.20),
    "D1_Vol/Avg":    lambda v: _cap(v, 0.70, 3.00),
    "D1_Dist_pctile":lambda v: _cap(v, 66.7, 95.0),
    "D0_Open_Dist_pctile": lambda v: _cap(v, 66.7, 99.0),
}
def grade_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    scores = []
    for _, row in df.iterrows():
        s = 0.0
        for f in FEATURES:
            s += WEIGHTS[f] * TRANS[f](row.get(f, np.nan))
        scores.append(s)
    out = df.copy()
    out["grade"] = scores
    return out.sort_values(["Date","grade"], ascending=[False, False])

# ───────── CLI ─────────
def main():
    ap = argparse.ArgumentParser(description="Daily Uptrend Gap — fast graded scan")
    ap.add_argument("--from", dest="dfrom", type=str, default="2019-07-01")
    ap.add_argument("--to", dest="dto", type=str,
                    default=(pd.Timestamp.now(tz=ET).normalize() - pd.Timedelta(days=1)).date().isoformat())
    ap.add_argument("--symbols", type=str, default=",".join(SYMBOLS))
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--csv", type=str, default=os.path.join(ARTIFACTS_DIR, "uptrend_gap_hits_graded.csv"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    print(f"\nDaily Uptrend Gap — GRADED\nRange: {args.dfrom} → {args.dto} | Symbols: {len(symbols)}\n")

    results: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futs = {exe.submit(scan_symbol, s, args.dfrom, args.dto): s for s in symbols}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    results.append(df)
                    print(f"  ✓ {s}: {len(df)} hits pre-grade")
            except Exception as e:
                print(f"  ✗ {s}: {e}")

    if not results:
        print("\nNo qualifying hits.")
        return

    all_hits = pd.concat(results, ignore_index=True)
    graded = grade_rows(all_hits)
    graded.to_csv(args.csv, index=False)

    top_cols = ["Date","Ticker","grade"]
    combos_cols = ["Date","Ticker","grade","Hour_first_touch_band","Hour_first_touch_ts","Gap/ATR","Open/EMA9"]

    print("\nTop (by grade):\n")
    print(graded[top_cols].head(50).to_string(index=False))

    print("\nName–Date combos (key params):\n")
    print(graded[combos_cols].head(80).to_string(index=False))

    print(f"\nSaved CSV → {args.csv}")

if __name__ == "__main__":
    main()
