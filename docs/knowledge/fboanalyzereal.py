# fbo_analyze_levels_clean.py

import os, pandas as pd, numpy as np, requests
from datetime import timedelta

API_KEY  = os.getenv("POLYGON_API_KEY", "Fm7brz4s23eSocDErnL68cE7wspz2K1I")
BASE_URL = "https://api.polygon.io"
S = requests.Session(); S.headers.update({"User-Agent": "fbo-analyze/2.2"})

P = {
    "price_min": 8.0,
    "adv20_min_usd": 15_000_000,

    "lookback_days_for_level": 1000,
    "exclude_recent_days": 5,
    "min_trade_days_between": 5,

    "touch_tol_pct": 5.0,
    "gap_min_pct": 2.0,
    "min_low_pullback_min_pct": 3.0,
    "require_close_below_level": True,
    "require_intraday_break": False,
    "break_tol_pct": 0.0,

    "prev_volume_min": 15_000_000,
    "prev_vol_over_avg_min": 1.1,

    "slope2d_min": 5.0,
    "slope3d_min": 7.0,
    "slope5d_min": 8.0,

    "require_prev_green": True,
    "d1_min_atr_body": 0.6,

    "sig_require_pivot": True,
    "sig_level_pos_abs_min": 0.55,
    "sig_pivot_percentile_min": 0.65,
    "sig_prom_atr_min": 1.0,
    "pivot_left": 1,
    "pivot_right": 1,
    "prom_left_days": 5,
    "prom_right_days": 5,

    # NEW: split/fake-wick guards
    "clean_spikes": True,
    "spike_atr_k": 8.0,          # cap High vs max(Open,Close)+k*ATR, Low vs min(Open,Close)-k*ATR
    "spike_max_up_pct": 4.00,    # cap High vs PrevClose * (1+X)
    "spike_max_dn_pct": 4.00,    # cap Low  vs PrevClose * (1-X)
}

def fetch_daily(tkr, start, end):
    url = f"{BASE_URL}/v2/aggs/ticker/{tkr}/range/1/day/{start}/{end}"
    r = S.get(url, params={"apiKey": API_KEY, "adjusted":"true", "sort":"asc", "limit":50000}, timeout=30)
    r.raise_for_status()
    rows = r.json().get("results", [])
    if not rows: return pd.DataFrame()
    df = (pd.DataFrame(rows)
            .assign(Date=lambda d: pd.to_datetime(d["t"], unit="ms", utc=True)
                                     .dt.tz_convert("America/New_York")
                                     .dt.normalize()
                                     .dt.tz_localize(None))
            .rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
            .set_index("Date")[["Open","High","Low","Close","Volume"]]
            .sort_index())
    return df

def add_metrics(df):
    m = df.copy()
    tr = pd.concat([(m["High"]-m["Low"]),
                    (m["High"]-m["Close"].shift(1)).abs(),
                    (m["Low"] -m["Close"].shift(1)).abs()], axis=1).max(axis=1)
    m["TR"] = tr
    m["ATR_30"] = tr.rolling(30, min_periods=30).mean().shift(1)
    m["VOL_AVG_30"] = m["Volume"].rolling(30, min_periods=30).mean().shift(1)
    m["ADV20_$"] = (m["Close"] * m["Volume"]).rolling(20, min_periods=20).mean()

    m["Prev_Open"]   = m["Open"].shift(1)
    m["Prev_Close"]  = m["Close"].shift(1)
    m["Prev_High"]   = m["High"].shift(1)
    m["Prev_Volume"] = m["Volume"].shift(1)

    # Slope metrics evaluated at D-1
    for k in (2,3,5):
        m[f"Slope{k}d_%"] = (m["Prev_Close"] - m["Close"].shift(k+1)) / m["Close"].shift(k+1) * 100

    # --- spike cleaning for High/Low (creates HighF/LowF) ---
    if P["clean_spikes"]:
        atr = m["ATR_30"]
        maxOC = np.maximum(m["Open"], m["Close"])
        minOC = np.minimum(m["Open"], m["Close"])

        cap_hi_atr = maxOC + P["spike_atr_k"] * atr
        cap_lo_atr = minOC - P["spike_atr_k"] * atr

        up_cap = m["Prev_Close"] * (1.0 + P["spike_max_up_pct"])
        dn_cap = m["Prev_Close"] * (1.0 - P["spike_max_dn_pct"])

        hi_cap = pd.concat([cap_hi_atr, up_cap], axis=1).min(axis=1)
        lo_cap = pd.concat([cap_lo_atr, dn_cap], axis=1).max(axis=1)

        m["HighF"] = np.minimum(m["High"], hi_cap)
        m["LowF"]  = np.maximum(m["Low"],  lo_cap)
    else:
        m["HighF"] = m["High"]
        m["LowF"]  = m["Low"]

    return m

def last_occurrence_of_max(s: pd.Series):
    if s.empty: return None
    mx = s.max(); hits = s[s == mx]
    return None if hits.empty else hits.index[-1]

def trading_days_between(idx, a, b):
    return int(((idx > a) & (idx < b)).sum())

def _pivot_high_dates(df: pd.DataFrame, left: int, right: int, hi_col: str="HighF"):
    h = df[hi_col].values; idx = df.index; n = len(df); out = []
    for i in range(left, n - right):
        lv = np.nanmax(h[i-left:i]) if left  > 0 else -np.inf
        rv = np.nanmax(h[i+1:i+1+right]) if right > 0 else -np.inf
        if np.isfinite(h[i]) and h[i] >= lv and h[i] > rv:
            out.append(idx[i])
    return out

def _local_valley_min(df: pd.DataFrame, center: pd.Timestamp, left_days: int, right_days: int, lo_col: str="LowF"):
    lwin = df.loc[(df.index >= center - pd.Timedelta(days=left_days)) & (df.index <= center)]
    rwin = df.loc[(df.index >= center) & (df.index <= center + pd.Timedelta(days=right_days))]
    lmin = float(lwin[lo_col].min()) if not lwin.empty else np.nan
    rmin = float(rwin[lo_col].min()) if not rwin.empty else np.nan
    return lmin, rmin

def significance_block(m: pd.DataFrame, level_date: pd.Timestamp, d0: pd.Timestamp, p: dict,
                       hi_col: str="HighF", lo_col: str="LowF") -> dict:
    cut = d0 - pd.Timedelta(days=p["exclude_recent_days"])
    wstart = cut - pd.Timedelta(days=p["lookback_days_for_level"])
    win = m[(m.index > wstart) & (m.index <= cut)]
    if win.empty or level_date not in win.index:
        return dict(LevelPosAbs=np.nan, LevelPctOfTop=np.nan, LevelPivotProm_ATR=np.nan,
                    LevelPivotPercentile=np.nan, IsPivotHigh=False, AbsTop=np.nan, AbsTop_Date=None)

    abs_top = float(win[hi_col].max()); abs_low = float(win[lo_col].min())
    level_px = float(m.at[level_date, hi_col])
    atr_at_lvl = float(m.at[level_date, "ATR_30"]) if "ATR_30" in m.columns else np.nan

    pos_abs = (level_px - abs_low) / (abs_top - abs_low) if abs_top > abs_low else np.nan
    pct_of_top = level_px / abs_top if abs_top > 0 else np.nan

    piv_dates = _pivot_high_dates(win, p["pivot_left"], p["pivot_right"], hi_col=hi_col)
    is_pivot = level_date in piv_dates
    perc = float((win.loc[piv_dates, hi_col] <= level_px).mean()) if piv_dates else np.nan

    lmin, rmin = _local_valley_min(win, level_date, p["prom_left_days"], p["prom_right_days"], lo_col=lo_col)
    ref_valley = np.nanmax([lmin, rmin])
    prom = level_px - ref_valley if np.isfinite(ref_valley) else np.nan
    prom_atr = (prom / atr_at_lvl) if (np.isfinite(prom) and atr_at_lvl and not np.isnan(atr_at_lvl)) else np.nan

    return {
        "LevelPosAbs": None if pd.isna(pos_abs) else round(pos_abs, 3),
        "LevelPctOfTop": None if pd.isna(pct_of_top) else round(pct_of_top, 3),
        "LevelPivotProm_ATR": None if pd.isna(prom_atr) else round(prom_atr, 2),
        "LevelPivotPercentile": None if pd.isna(perc) else round(perc, 3),
        "IsPivotHigh": bool(is_pivot),
        "AbsTop": round(abs_top, 4),
        "AbsTop_Date": win[hi_col].idxmax().strftime("%Y-%m-%d"),
    }

def choose_fbo_level(m: pd.DataFrame, d0: pd.Timestamp, p: dict, hi_col: str="HighF"):
    i = m.index.get_loc(d0)
    pre = m.iloc[:i]
    core = pre.iloc[:-p["min_trade_days_between"]] if len(pre) > p["min_trade_days_between"] else pre.iloc[:0]
    if core.empty: return None

    pivs = _pivot_high_dates(core, p["pivot_left"], p["pivot_right"], hi_col=hi_col)
    if not pivs: return None

    o, h, pc = map(float, m.loc[d0, ["Open","High","Prev_Close"]])
    tol  = p["touch_tol_pct"] / 100.0
    btol = p["break_tol_pct"] / 100.0

    cand = []
    for dt in pivs:
        price = float(m.at[dt, hi_col])
        sig = significance_block(m, dt, d0, p, hi_col=hi_col, lo_col="LowF")
        if p["sig_require_pivot"] and not sig["IsPivotHigh"]: continue
        if (p["sig_level_pos_abs_min"] is not None) and (sig["LevelPosAbs"] is not None) and (sig["LevelPosAbs"] < p["sig_level_pos_abs_min"]): continue
        if (p["sig_pivot_percentile_min"] is not None) and (sig["LevelPivotPercentile"] is not None) and (sig["LevelPivotPercentile"] < p["sig_pivot_percentile_min"]): continue
        if (p["sig_prom_atr_min"] is not None) and (sig["LevelPivotProm_ATR"] is not None) and (sig["LevelPivotProm_ATR"] < p["sig_prom_atr_min"]): continue

        open_touch = (o >= price * (1 - tol))
        high_touch = (h >= price * (1 - tol))
        intr_brk   = (h >= price * (1 + btol))
        prev_touch = (pc >= price * (1 - tol))
        cand.append([dt, price,
                     abs(o-price)/price*100.0,
                     abs(h-price)/price*100.0,
                     abs(pc-price)/price*100.0,
                     open_touch, high_touch, intr_brk, prev_touch, sig])

    if not cand: return None
    c = pd.DataFrame(cand, columns=["Date","Price","OpenDist_%","HighDist_%","PrevCloseDist_%","OpenTouch","HighTouch","IntradayBreak","PrevTouch","Sig"])

    pick_by, filt = "OpenTouch", c[c["OpenTouch"]].sort_values("OpenDist_%")
    if filt.empty:
        pick_by, filt = "HighTouch", c[c["HighTouch"]].sort_values("HighDist_%")
    if filt.empty:
        pick_by, filt = "PrevTouch", c[c["PrevTouch"]].sort_values("PrevCloseDist_%")
    if filt.empty:
        pick_by, filt = "NearestPivot", c.sort_values(["OpenDist_%","HighDist_%"])

    top = filt.iloc[0]
    return {
        "Level_Date": top["Date"],
        "Level_Price": float(top["Price"]),
        "Pick_By": pick_by,
        "OpenDist_%": float(top["OpenDist_%"]),
        "HighDist_%": float(top["HighDist_%"]),
        "PrevCloseDist_%": float(top["PrevCloseDist_%"]),
        "Sig": top["Sig"]
    }

def analyze_one(sym: str, target_str: str, p: dict) -> pd.DataFrame:
    t0 = pd.Timestamp(pd.to_datetime(target_str).date())
    start = (t0 - timedelta(days=max(140, p["lookback_days_for_level"]) + 40)).strftime("%Y-%m-%d")
    end   = (t0 + timedelta(days=5)).strftime("%Y-%m-%d")

    df = fetch_daily(sym, start, end)
    if df.empty: raise SystemExit("No data from Polygon.")
    if t0 not in df.index:
        print("Available:", df.index.min().date(), "→", df.index.max().date())
        near = df.index[df.index.get_indexer([t0], method="nearest")][0]
        print("Nearest:", near.date())
        raise SystemExit("Exact target date missing.")

    m = add_metrics(df)

    choice = choose_fbo_level(m, t0, p, hi_col="HighF")
    if not choice: raise SystemExit("No significant pivot level found after cleaning.")
    level_day = choice["Level_Date"]; level_px = choice["Level_Price"]

    i  = m.index.get_loc(t0)
    r0 = m.iloc[i]; d1 = m.iloc[i-1]

    between = m[(m.index > level_day) & (m.index < t0)]
    min_low = float(between["LowF"].min()) if not between.empty else np.nan
    pullback_pct = ((level_px - min_low) / level_px * 100.0) if pd.notna(min_low) else np.nan

    o,h,l,c = map(float, r0[["Open","High","Low","Close"]])
    pc = float(r0["Prev_Close"])
    gap_pct = (o - pc) / pc * 100.0 if pc else np.nan
    touch_dist_pct = (o - level_px) / level_px * 100.0

    prev_vol = float(r0["Prev_Volume"]) if pd.notna(r0["Prev_Volume"]) else np.nan
    vol_avg  = float(r0["VOL_AVG_30"])  if pd.notna(r0["VOL_AVG_30"])  else np.nan
    prev_vol_over = (prev_vol / vol_avg) if (pd.notna(prev_vol) and pd.notna(vol_avg) and vol_avg) else np.nan

    d1_green = bool(d1["Close"] > d1["Open"])
    d1_body_atr = ((d1["Close"] - d1["Open"]) / d1["ATR_30"]) if pd.notna(d1["ATR_30"]) else np.nan

    sig = choice["Sig"]
    out = {
        "Ticker": sym,
        "High_Date": level_day.strftime("%Y-%m-%d"),
        "Target_Date": t0.strftime("%Y-%m-%d"),
        "LevelPick_By": choice["Pick_By"],
        "OpenDist_toLevel_%": round(choice["OpenDist_%"],2),
        "HighDist_toLevel_%": round(choice["HighDist_%"],2),
        "PrevCloseDist_toLevel_%": round(choice["PrevCloseDist_%"],2),

        "Days_From_Set": trading_days_between(m.index, level_day, t0),
        "Level": round(level_px,4),
        "Open": round(o,4), "High": round(h,4), "Low": round(l,4), "Close": round(c,4),

        "Gap_%": None if pd.isna(gap_pct) else round(gap_pct,2),
        "Touch_Dist_%": None if pd.isna(touch_dist_pct) else round(touch_dist_pct,2),
        "Between_MinLow": None if pd.isna(min_low) else round(min_low,4),
        "PullbackDown_%": None if pd.isna(pullback_pct) else round(pullback_pct,2),

        "Prev_Close": None if pd.isna(r0["Prev_Close"]) else round(float(r0["Prev_Close"]),4),
        "Prev_Volume": None if pd.isna(prev_vol) else int(prev_vol),
        "Prev_Vol/Avg": None if pd.isna(prev_vol_over) else round(prev_vol_over,2),
        "ADV20_$": None if pd.isna(r0["ADV20_$"]) else int(r0["ADV20_$"]),

        "D1_Green": d1_green,
        "D1_Body/ATR": None if pd.isna(d1_body_atr) else round(d1_body_atr,2),

        "Slope2d_%": None if pd.isna(r0["Slope2d_%"]) else round(float(r0["Slope2d_%"]),2),
        "Slope3d_%": None if pd.isna(r0["Slope3d_%"]) else round(float(r0["Slope3d_%"]),2),
        "Slope5d_%": None if pd.isna(r0["Slope5d_%"]) else round(float(r0["Slope5d_%"]),2),

        "LevelPosAbs": sig["LevelPosAbs"],
        "LevelPctOfTop": sig["LevelPctOfTop"],
        "LevelPivotProm_ATR": sig["LevelPivotProm_ATR"],
        "LevelPivotPercentile": sig["LevelPivotPercentile"],
        "IsPivotHigh": sig["IsPivotHigh"],
        "AbsTop": sig["AbsTop"],
        "AbsTop_Date": sig["AbsTop_Date"],
    }
    cols = list(out.keys())
    return pd.DataFrame([[out[c] for c in cols]], columns=cols)

if __name__ == "__main__":
    df = analyze_one("NVDA", "2025-01-07", P)
    pd.set_option("display.max_columns", None, "display.width", 0)
    print(df.to_string(index=False))
