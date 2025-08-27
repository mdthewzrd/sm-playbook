# scan_bucket_A_frontside_AH_or_PRE_touch.py
# Bucket A — Front-side blowoff into 72-band in AH (prev 16–20) or PRE (04–10:15).
# Trade at the touch (AH/PRE) or at the open if first touch was earlier and we’re still extended.
# Includes: daily-strength filters, liquidity guards, and EMA9 “angle” grades at the HH
# for 15m / 30m / 1H / 2H. Prints a compact table.

import pandas as pd, numpy as np, requests
from datetime import datetime, timedelta, time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ───────── config ─────────
session  = requests.Session()
API_KEY  = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
BASE_URL = "https://api.polygon.io"
ET_TZ    = "US/Eastern"

# Deviation threshold & proximity
DEV_THR   = 6.9
OPEN_PROX = 0.5          # 09:30 open needs Dev72 >= DEV_THR - OPEN_PROX

LOOKBACK_CAL_DAYS = 25   # intraday fetch window

WINDOWS = {
    "AH":   ((16,0),(20,0)),   # previous day (ET)
    "PRE1": ((4,0),(7,0)),
    "PRE2": ((7,0),(10,15)),   # GTZ
    "REG":  ((9,30),(16,0)),
}

# Liquidity thresholds
LIQ_THR = {
    "min_price": 5.0,
    "adv20_dollar_min": 50_000_000,
    "reg10_med_min":    2_000_000,   # $/bar 1H REG median over ~10 days
    "pre10_med_min":      300_000,   # $/bar 15m PRE median over ~10 days
    "d0_pre_sum_min":     400_000,   # $ sum in PRE (04:00–09:30) on D0
}

# EMA9 “angle” windows (bars ending at HH)
SLOPE_BARS = {"15m": 3, "30m": 3, "1h": 3, "2h": 2}
BAR_MINS   = {"15m": 15, "30m": 30, "1h": 60, "2h": 120}

# ───────── fetch helpers ─────────
def fetch_daily(tkr: str, start: str, end: str) -> pd.DataFrame:
    url = f"{BASE_URL}/v2/aggs/ticker/{tkr}/range/1/day/{start}/{end}"
    r   = session.get(url, params={"apiKey": API_KEY, "adjusted": "true", "sort": "asc", "limit": 50000})
    r.raise_for_status()
    rows = r.json().get("results", [])
    if not rows: return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .assign(Date=lambda d: pd.to_datetime(d["t"], unit="ms"))
        .rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
        .set_index("Date")[["Open","High","Low","Close","Volume"]]
        .sort_index()
    )

def fetch_intraday(tkr: str, mult: int, span: str, start: str, end: str) -> pd.DataFrame:
    url = f"{BASE_URL}/v2/aggs/ticker/{tkr}/range/{mult}/{span}/{start}/{end}"
    r   = session.get(url, params={"apiKey": API_KEY, "adjusted": "true", "sort": "asc", "limit": 50000})
    r.raise_for_status()
    rows = r.json().get("results", [])
    if not rows: return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .assign(ts=lambda d: pd.to_datetime(d["t"], unit="ms", utc=True).dt.tz_convert(ET_TZ))
        .rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
        .set_index("ts")[["Open","High","Low","Close","Volume"]]
        .sort_index()
    )

# ───────── daily strength ─────────
def add_daily_strength_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    for w in (5, 15):
        df[f"Slope_9_{w}d"] = (df["EMA_9"].shift(1) - df["EMA_9"].shift(w+1)) / df["EMA_9"].shift(w+1) * 100
    rng = (df["Close"] - df["Low"]) / (df["High"] - df["Low"]).replace(0, np.nan) * 100
    df["Upper_70_Range_prev"] = rng.shift(1)
    df["Vol_avg20"] = df["Volume"].rolling(20, min_periods=20).mean()
    df["ADV20_$"]   = (df["Close"] * df["Volume"]).rolling(20, min_periods=20).mean()
    df["Vol_over_AVG_prev"] = df["Volume"].shift(1) / df["Vol_avg20"].shift(1)
    df["Pct_1d_prev"] = df["Close"].pct_change().shift(1) * 100
    df["Close_over_EMA9_prev"] = df["Close"].shift(1) / df["EMA_9"].shift(1)
    df["Prev_Close"] = df["Close"].shift(1)
    return df

def passes_daily_strength(df: pd.DataFrame, d0: pd.Timestamp) -> bool:
    try:
        row = df.loc[d0]
    except KeyError:
        return False
    return bool(
        (row["Pct_1d_prev"] >= 4.0) and
        (row["Upper_70_Range_prev"] >= 70) and
        (row["Vol_over_AVG_prev"] >= 1.5) and
        (row["Slope_9_5d"] >= 6) and
        (row["Slope_9_15d"] >= 15) and
        (row["Close_over_EMA9_prev"] >= 1.01)
    )

# ───────── intraday cloud + liquidity ─────────
def add_cloud_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR_72"] = tr.rolling(72, min_periods=30).mean()
    df["EMA_72"] = df["Close"].ewm(span=72, adjust=False).mean()
    df["Dev72Mult_High"] = (df["High"] - df["EMA_72"]) / df["ATR_72"].replace(0, np.nan)
    df["Dev72Mult_Open"] = (df["Open"] - df["EMA_72"]) / df["ATR_72"].replace(0, np.nan)
    df["DollarVol"] = df["Close"] * df["Volume"]
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()   # for angle calc
    df["tod"]  = df.index.tz_convert(ET_TZ).time
    return df

def _ts_local(d: datetime.date, hh: int, mm: int) -> pd.Timestamp:
    return pd.Timestamp.combine(d, time(hh, mm)).tz_localize(ET_TZ)

def window_flags(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp, dev_thr=DEV_THR) -> dict:
    w = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    if w.empty:
        return {"max_dev": np.nan, "first_touchTS": None}
    hit = w[w["Dev72Mult_High"] >= dev_thr]
    first = hit.index[0] if not hit.empty else None
    return {"max_dev": float(w["Dev72Mult_High"].max()), "first_touchTS": first}

def earliest_touch(*touches):
    ts = [t["first_touchTS"] for t in touches if t["first_touchTS"] is not None]
    if not ts: return None
    return min(ts)

def intraday_liquidity(df: pd.DataFrame, d0_et: pd.Timestamp) -> dict:
    if df.empty:
        return {"REG_med_$": np.nan, "PRE_med_$": np.nan, "D0_pre_sum_$": np.nan}
    def in_tod(t, start, end): return (t >= start) and (t <= end)
    start_date = (d0_et - timedelta(days=15)).date()
    mask_hist = (df.index.date < d0_et.date()) & (df.index.date >= start_date)

    reg_start, reg_end = WINDOWS["REG"][0], WINDOWS["REG"][1]
    reg_start, reg_end = time(*reg_start), time(*reg_end)
    pre_start, pre_end = WINDOWS["PRE2"][0], WINDOWS["PRE2"][1]
    pre_start, pre_end = time(*pre_start), time(*pre_end)

    reg = df[mask_hist & df["tod"].apply(lambda t: in_tod(t, reg_start, reg_end))]
    pre = df[mask_hist & df["tod"].apply(lambda t: in_tod(t, pre_start, pre_end))]

    reg_med = float(reg["DollarVol"].median()) if not reg.empty else np.nan
    pre_med = float(pre["DollarVol"].median()) if not pre.empty else np.nan

    d0_pre = df[(df.index.date == d0_et.date()) & df["tod"].apply(lambda t: time(4,0) <= t <= time(9,30))]
    d0_pre_sum = float(d0_pre["DollarVol"].sum()) if not d0_pre.empty else np.nan

    return {"REG_med_$": reg_med, "PRE_med_$": pre_med, "D0_pre_sum_$": d0_pre_sum}

# ───────── HH + EMA9 angle helpers ─────────
def _nearest_le_idx(idx, ts):
    if ts in idx: return ts
    pos = idx.searchsorted(ts, side="right") - 1
    if pos < 0: return None
    return idx[pos]

def highest_high_segment(df: pd.DataFrame, seg_start: pd.Timestamp, seg_end: pd.Timestamp) -> dict:
    w = df.loc[(df.index >= seg_start) & (df.index <= seg_end)]
    if w.empty:
        return {"HH": np.nan, "HH_TS": None, "HH_Dev": np.nan}
    i = w["High"].idxmax()
    return {
        "HH": float(w.at[i, "High"]),
        "HH_TS": i,
        "HH_Dev": float(w.at[i, "Dev72Mult_High"]) if pd.notna(w.at[i, "Dev72Mult_High"]) else np.nan,
    }

def ema9_angle_at_ts(df: pd.DataFrame, ts: pd.Timestamp, bars: int, bar_minutes: int) -> dict:
    """EMA9 slope & angle ending at *ts*:
       - pct_per_hr and angle(deg) = atan(pct_per_hr)
       - dev_per_hr (ATR72-normalized) and angle(deg)
    """
    if df.empty or ts is None or bars <= 0:
        return {"pct_per_hr": np.nan, "dev_per_hr": np.nan,
                "angle_pct_deg_hr": np.nan, "angle_dev_deg_hr": np.nan}

    ts_i = _nearest_le_idx(df.index, ts)
    if ts_i is None:
        return {"pct_per_hr": np.nan, "dev_per_hr": np.nan,
                "angle_pct_deg_hr": np.nan, "angle_dev_deg_hr": np.nan}

    loc = df.index.get_loc(ts_i)
    if isinstance(loc, slice):
        return {"pct_per_hr": np.nan, "dev_per_hr": np.nan,
                "angle_pct_deg_hr": np.nan, "angle_dev_deg_hr": np.nan}

    prev = loc - bars
    if prev < 0:
        return {"pct_per_hr": np.nan, "dev_per_hr": np.nan,
                "angle_pct_deg_hr": np.nan, "angle_dev_deg_hr": np.nan}

    ema_now = df["EMA_9"].iloc[loc]
    ema_prev= df["EMA_9"].iloc[prev]
    if pd.isna(ema_now) or pd.isna(ema_prev) or ema_prev == 0:
        return {"pct_per_hr": np.nan, "dev_per_hr": np.nan,
                "angle_pct_deg_hr": np.nan, "angle_dev_deg_hr": np.nan}

    # % slope per bar → per hour
    pct_per_bar = (ema_now - ema_prev) / ema_prev / bars * 100.0
    pct_per_hr  = pct_per_bar * (60.0 / max(1, bar_minutes))

    # deviation-normalized slope
    atr_mean = df["ATR_72"].iloc[prev:loc+1].mean()
    dev_per_bar = ((ema_now - ema_prev) / max(1e-12, atr_mean)) / bars
    dev_per_hr  = dev_per_bar * (60.0 / max(1, bar_minutes))

    angle_pct_deg_hr = float(np.degrees(np.arctan(pct_per_hr)))
    angle_dev_deg_hr = float(np.degrees(np.arctan(dev_per_hr)))

    return {"pct_per_hr": float(pct_per_hr), "dev_per_hr": float(dev_per_hr),
            "angle_pct_deg_hr": angle_pct_deg_hr, "angle_dev_deg_hr": angle_dev_deg_hr}

# ───────── main scan ─────────
def scan_bucket_A(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    daily = fetch_daily(symbol, start_date, end_date)
    if daily.empty: return pd.DataFrame()
    daily = add_daily_strength_metrics(daily)

    out = []
    for d0 in daily.index[1:]:
        if not passes_daily_strength(daily, d0):
            continue
        row_d = daily.loc[d0]
        if (row_d["Prev_Close"] < LIQ_THR["min_price"]) or (row_d["ADV20_$"] < LIQ_THR["adv20_dollar_min"]):
            continue

        d0_et = pd.Timestamp(d0).tz_localize(ET_TZ)
        start = (d0_et - timedelta(days=LOOKBACK_CAL_DAYS)).strftime("%Y-%m-%d")
        end   = (d0_et + timedelta(days=1)).strftime("%Y-%m-%d")

        # intraday frames
        h1  = add_cloud_metrics(fetch_intraday(symbol, 1,  "hour",   start, end))
        m15 = add_cloud_metrics(fetch_intraday(symbol, 15, "minute", start, end))
        m30 = add_cloud_metrics(fetch_intraday(symbol, 30, "minute", start, end))
        h2  = add_cloud_metrics(fetch_intraday(symbol, 2,  "hour",   start, end))
        if h1.empty or m15.empty:
            continue

        prev_date = (d0_et - timedelta(days=1)).date()
        ah_s, ah_e   = _ts_local(prev_date, *WINDOWS["AH"][0]),   _ts_local(prev_date, *WINDOWS["AH"][1])
        pre1_s, pre1_e = _ts_local(d0_et.date(), *WINDOWS["PRE1"][0]), _ts_local(d0_et.date(), *WINDOWS["PRE1"][1])
        pre2_s, pre2_e = _ts_local(d0_et.date(), *WINDOWS["PRE2"][0]), _ts_local(d0_et.date(), *WINDOWS["PRE2"][1])

        # touches
        h1_ah   = window_flags(h1,  ah_s, ah_e)
        m15_ah  = window_flags(m15, ah_s, ah_e)
        h1_pre1 = window_flags(h1,  pre1_s, pre1_e)
        m15_pre1= window_flags(m15, pre1_s, pre1_e)
        h1_pre2 = window_flags(h1,  pre2_s, pre2_e)
        m15_pre2= window_flags(m15, pre2_s, pre2_e)

        # open proximity
        t0930 = _ts_local(d0_et.date(), 9, 30)
        r0930 = m15.loc[t0930] if t0930 in m15.index else None
        open_dev = None
        open_prox_ok = False
        if r0930 is not None and not pd.isna(r0930["Dev72Mult_Open"]):
            open_dev = float(r0930["Dev72Mult_Open"])
            open_prox_ok = (open_dev >= (DEV_THR - OPEN_PROX))

        # validity
        touch_in_pre2 = (h1_pre2["first_touchTS"] is not None) or (m15_pre2["first_touchTS"] is not None)
        touch_earlier = any([d["first_touchTS"] is not None for d in (h1_ah, m15_ah, h1_pre1, m15_pre1)])
        valid = touch_in_pre2 or (touch_earlier and (touch_in_pre2 or open_prox_ok))

        # liquidity gates
        liq1h = intraday_liquidity(h1,  d0_et)
        liq15 = intraday_liquidity(m15, d0_et)
        liq_ok = (
            (liq1h["REG_med_$"] >= LIQ_THR["reg10_med_min"]) and
            (liq15["PRE_med_$"] >= LIQ_THR["pre10_med_min"]) and
            (liq15["D0_pre_sum_$"] >= LIQ_THR["d0_pre_sum_min"])
        )
        if not (valid and liq_ok):
            continue

        # HH window (prev 16:00 → D0 10:15)
        seg_start = ah_s
        seg_end   = pre2_e

        hh_1h  = highest_high_segment(h1,  seg_start, seg_end)
        hh_15  = highest_high_segment(m15, seg_start, seg_end)
        hh_30  = highest_high_segment(m30, seg_start, seg_end) if not m30.empty else {"HH":np.nan,"HH_TS":None,"HH_Dev":np.nan}
        hh_2h  = highest_high_segment(h2,  seg_start, seg_end) if not h2.empty else {"HH":np.nan,"HH_TS":None,"HH_Dev":np.nan}

        # angles
        a_1h  = ema9_angle_at_ts(h1,  hh_1h["HH_TS"],  SLOPE_BARS["1h"],  BAR_MINS["1h"])  if hh_1h["HH_TS"]  is not None else {}
        a_15  = ema9_angle_at_ts(m15, hh_15["HH_TS"],  SLOPE_BARS["15m"], BAR_MINS["15m"]) if hh_15["HH_TS"]  is not None else {}
        a_30  = ema9_angle_at_ts(m30, hh_30["HH_TS"],  SLOPE_BARS["30m"], BAR_MINS["30m"]) if hh_30["HH_TS"]  is not None else {}
        a_2h  = ema9_angle_at_ts(h2,  hh_2h["HH_TS"],  SLOPE_BARS["2h"],  BAR_MINS["2h"])  if hh_2h["HH_TS"]  is not None else {}

        fmt = "%m-%d %H:%M"  # compact timestamps
        row = {
            "Ticker": symbol,
            "Date": d0.strftime("%Y-%m-%d"),
            # D-1 context
            "D-1%": round(float(row_d["Pct_1d_prev"]), 2),
            "D-1 Upper%": round(float(row_d["Upper_70_Range_prev"]), 1),
            "D-1 Vol/Avg": round(float(row_d["Vol_over_AVG_prev"]), 2),
            "Slope9_5d": round(float(row_d["Slope_9_5d"]), 2),
            "Slope9_15d": round(float(row_d["Slope_9_15d"]), 2),
            # touches
            "AH FirstTouch":   h1_ah["first_touchTS"].strftime(fmt)    if h1_ah["first_touchTS"]   is not None else (m15_ah["first_touchTS"].strftime(fmt)   if m15_ah["first_touchTS"]   is not None else None),
            "PRE1 FirstTouch": h1_pre1["first_touchTS"].strftime(fmt)  if h1_pre1["first_touchTS"] is not None else (m15_pre1["first_touchTS"].strftime(fmt) if m15_pre1["first_touchTS"] is not None else None),
            "PRE2 FirstTouch": h1_pre2["first_touchTS"].strftime(fmt)  if h1_pre2["first_touchTS"] is not None else (m15_pre2["first_touchTS"].strftime(fmt) if m15_pre2["first_touchTS"] is not None else None),
            "OpenDev@09:30": round(open_dev, 2) if open_dev is not None else np.nan,
            "OpenProxOK": bool(open_prox_ok),
            # liquidity
            "ADV20_$": round(float(row_d["ADV20_$"]), 0),
            "REG_MedDV$_10d": round(liq1h["REG_med_$"], 0) if pd.notna(liq1h["REG_med_$"]) else np.nan,
            "PRE_MedDV$_10d": round(liq15["PRE_med_$"], 0) if pd.notna(liq15["PRE_med_$"]) else np.nan,
            "D0_PRE_SumDV$_toOpen": round(liq15["D0_pre_sum_$"], 0) if pd.notna(liq15["D0_pre_sum_$"]) else np.nan,
            # HH + Dev72 at HH
            "HH_1H": hh_1h["HH"], "HH_1H_dev": hh_1h["HH_Dev"], "HH_1H_time": hh_1h["HH_TS"].strftime(fmt) if hh_1h["HH_TS"] else None,
            "HH_15m": hh_15["HH"], "HH_15m_dev": hh_15["HH_Dev"], "HH_15m_time": hh_15["HH_TS"].strftime(fmt) if hh_15["HH_TS"] else None,
            "HH_30m": hh_30["HH"], "HH_30m_dev": hh_30["HH_Dev"], "HH_30m_time": hh_30["HH_TS"].strftime(fmt) if hh_30["HH_TS"] else None,
            "HH_2H": hh_2h["HH"], "HH_2H_dev": hh_2h["HH_Dev"], "HH_2H_time": hh_2h["HH_TS"].strftime(fmt) if hh_2h["HH_TS"] else None,
            # EMA9 angles (hour-normalized)
            "EMA9_PctSlope_1H_per_hr": a_1h.get("pct_per_hr", np.nan),
            "EMA9_AnglePct_1H_deg_hr": a_1h.get("angle_pct_deg_hr", np.nan),
            "EMA9_DevSlope_1H_per_hr": a_1h.get("dev_per_hr", np.nan),
            "EMA9_AngleDev_1H_deg_hr": a_1h.get("angle_dev_deg_hr", np.nan),
            "EMA9_PctSlope_15m_per_hr": a_15.get("pct_per_hr", np.nan),
            "EMA9_AnglePct_15m_deg_hr": a_15.get("angle_pct_deg_hr", np.nan),
            "EMA9_DevSlope_15m_per_hr": a_15.get("dev_per_hr", np.nan),
            "EMA9_AngleDev_15m_deg_hr": a_15.get("angle_dev_deg_hr", np.nan),
            "EMA9_PctSlope_30m_per_hr": a_30.get("pct_per_hr", np.nan),
            "EMA9_AnglePct_30m_deg_hr": a_30.get("angle_pct_deg_hr", np.nan),
            "EMA9_DevSlope_30m_per_hr": a_30.get("dev_per_hr", np.nan),
            "EMA9_AngleDev_30m_deg_hr": a_30.get("angle_dev_deg_hr", np.nan),
            "EMA9_PctSlope_2H_per_hr": a_2h.get("pct_per_hr", np.nan),
            "EMA9_AnglePct_2H_deg_hr": a_2h.get("angle_pct_deg_hr", np.nan),
            "EMA9_DevSlope_2H_per_hr": a_2h.get("dev_per_hr", np.nan),
            "EMA9_AngleDev_2H_deg_hr": a_2h.get("angle_dev_deg_hr", np.nan),
        }
        out.append(row)

    return pd.DataFrame(out)

# ───────── compact printing helpers ─────────
ROUND = 2  # decimals for floats

SHORT_NAMES = {
    "D-1%":"D1%", "D-1 Upper%":"D1U%", "D-1 Vol/Avg":"D1V",
    "Slope9_5d":"s9_5d", "Slope9_15d":"s9_15d",
    "AH FirstTouch":"AH", "PRE1 FirstTouch":"PRE1", "PRE2 FirstTouch":"PRE2",
    "OpenDev@09:30":"ODev", "OpenProxOK":"OProx",
    "ADV20_$":"ADV20$", "REG_MedDV$_10d":"H1$Med", "PRE_MedDV$_10d":"P15$Med",
    "D0_PRE_SumDV$_toOpen":"PRE$Sum",
    "HH_1H":"HH1H", "HH_1H_dev":"Dev1H", "HH_1H_time":"T1H",
    "HH_15m":"HH15", "HH_15m_dev":"Dev15", "HH_15m_time":"T15",
    "HH_30m":"HH30", "HH_30m_dev":"Dev30", "HH_30m_time":"T30",
    "HH_2H":"HH2H", "HH_2H_dev":"Dev2H", "HH_2H_time":"T2H",
    "EMA9_PctSlope_1H_per_hr":"Pct1H", "EMA9_AnglePct_1H_deg_hr":"Ang%1H",
    "EMA9_DevSlope_1H_per_hr":"DevS1H", "EMA9_AngleDev_1H_deg_hr":"AngD1H",
    "EMA9_PctSlope_15m_per_hr":"Pct15", "EMA9_AnglePct_15m_deg_hr":"Ang%15",
    "EMA9_DevSlope_15m_per_hr":"DevS15", "EMA9_AngleDev_15m_deg_hr":"AngD15",
    "EMA9_PctSlope_30m_per_hr":"Pct30", "EMA9_AnglePct_30m_deg_hr":"Ang%30",
    "EMA9_DevSlope_30m_per_hr":"DevS30", "EMA9_AngleDev_30m_deg_hr":"AngD30",
    "EMA9_PctSlope_2H_per_hr":"Pct2H", "EMA9_AnglePct_2H_deg_hr":"Ang%2H",
    "EMA9_DevSlope_2H_per_hr":"DevS2H", "EMA9_AngleDev_2H_deg_hr":"AngD2H",
}

SHORT_ORDER = [
    "Ticker","Date","D1%","D1U%","D1V","s9_5d","s9_15d",
    "AH","PRE1","PRE2","ODev","OProx",
    "ADV20$","H1$Med","P15$Med","PRE$Sum",
    "HH1H","Dev1H","T1H","Pct1H","Ang%1H","DevS1H","AngD1H",
    "HH15","Dev15","T15","Pct15","Ang%15","DevS15","AngD15",
    "HH30","Dev30","T30","Pct30","Ang%30","DevS30","AngD30",
    "HH2H","Dev2H","T2H","Pct2H","Ang%2H","DevS2H","AngD2H",
]

def compact_for_print(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # shorten any time-like columns (already compact, but keep just in case)
    for c in [c for c in d.columns if c.endswith("_time") or c in ("AH FirstTouch","PRE1 FirstTouch","PRE2 FirstTouch")]:
        d[c] = d[c].astype(str)
    # rename & round
    d = d.rename(columns=SHORT_NAMES)
    num_cols = d.select_dtypes(include=[float, int]).columns
    d[num_cols] = d[num_cols].round(ROUND)
    # order
    cols = [c for c in SHORT_ORDER if c in d.columns]
    other = [c for c in d.columns if c not in cols]
    return d[cols + other]

# ───────── run CLI ─────────
if __name__ == "__main__":
    symbols = ['MSTR','SMCI','DJT','BABA','TCOM','AMC','SOXL','MRVL','TGT','DOCU','ZM','DIS','NFLX','RKT','SNAP','RBLX','META','SE','NVDA','AAPL','MSFT','GOOGL','AMZN','TSLA','AMD','INTC','BA','PYPL','QCOM','ORCL','T','CSCO','VZ','KO','PEP','MRK','PFE','ABBV','JNJ','CRM','BAC','C','JPM','WMT','CVX','XOM','COP','RTX','SPGI','GS','HD','LOW','COST','UNH','NEE','NKE','LMT','HON','CAT','MMM','LIN','ADBE','AVGO','TXN','ACN','UPS','BLK','PM','MO','ELV','VRTX','ZTS','NOW','ISRG','PLD','MS','MDT','WM','GE','IBM','BKNG','FDX','ADP','EQIX','DHR','SNPS','REGN','SYK','TMO','CVS','INTU','SCHW','CI','APD','SO','MMC','ICE','FIS','ADI','CSX','LRCX','GILD','RIVN','LCID','PLTR','SNOW','SPY','QQQ','DIA','IWM','TQQQ','SQQQ','ARKK','LABU','TECL','UVXY','XLE','XLK','XLF','IBB','KWEB','TAN','XOP','EEM','HYG','EFA','USO','GLD','SLV','BITO','RIOT','MARA','COIN','SQ','AFRM','DKNG','SHOP','UPST','CLF','AA','F','GM','ROKU','WBD','WBA','PARA','PINS','LYFT','BYND','RDDT','GME','VKTX','APLD','KGEI','INOD','LMB','AMR','PMTS','SAVA','CELH','ESOA','IVT','MOD','SKYE','AR','VIXY','TECS','LABD','SPXS','SPXL','DRV','TZA','FAZ','WEBS','PSQ','SDOW','MSTU','MSTZ','NFLU','BTCL','BTCZ','ETU','ETQ','FAS','TNA','NUGT','TSLL','NVDU','AMZU','MSFU','UVIX','CRCL','SBET','MRNA','TIGR']
    start   = "2023-01-01"
    end     = datetime.today().strftime("%Y-%m-%d")
    MAX_WORKERS = 5

    def _worker(sym):
        try:
            return scan_bucket_A(sym, start, end)
        except Exception:
            return None

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futs = {exe.submit(_worker, s): s for s in symbols}
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None and hasattr(df, "empty") and not df.empty:
                results.append(df)

    if results:
        out = pd.concat(results, ignore_index=True).sort_values(["Date","Ticker"], ascending=[False, True])
        out = compact_for_print(out)
        pd.set_option("display.max_columns", None, "display.max_colwidth", 14)
        print("\nBucket A — compact view (AH/PRE 72-band touch + liquidity + HH & EMA9 angles):\n")
        print(out.to_string(index=False))
    else:
        print("No hits for given symbols/date range.")
