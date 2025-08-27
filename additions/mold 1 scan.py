# short_fbo_scan.py
# Daily short FBO scan (Polygon)
# - Uses "significant high" selection (pivot + prominence + near-top filter)
# - D0 gap/touch checks; D-1 volume/slopes checks
# - Spike cleanup to ignore split/misprint highs
# - Progress + per-symbol rejection reasons (toggle with P["debug_reasons"])

import os, warnings, time
import pandas as pd, numpy as np, requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=FutureWarning)

API_KEY  = os.getenv("POLYGON_API_KEY", "Fm7brz4s23eSocDErnL68cE7wspz2K1I")
BASE_URL = "https://api.polygon.io"
S = requests.Session()
S.headers.update({"User-Agent": "short-fbo-scan-v6"})

P = {
    # universe / liquidity
    "price_min": 8.0,
    "adv20_min_usd": 15_000_000,

    # windows
    "lookback_days_for_level": 1000,
    "exclude_recent_days": 10,       # used in significance window only
    "min_trade_days_between": 15,    # trading days from level-set → D0

    # D0 (target day) gates
    "gap_min_pct": 0.1,              # D0 open vs D-1 close (NVDA 1/07 ~0.16%)
    "touch_tol_pct": 1.0,            # open within 1% below level counts as a touch
    "require_intraday_break": False, # if True: require High ≥ level*(1+break_tol/100)
    "break_tol_pct": 0.0,
    "require_close_below_level": False,

    # behavior between level and D0
    "min_low_pullback_min_pct": 5.0, # (level - minLow)/level * 100

    # D-1 filters (volume/slopes)
    "prev_volume_min": 20_000_000,
    "prev_vol_over_avg_min": 1.2,    # D-1 Volume / 30d average (shifted)
    "slope2d_min": 5.0,
    "slope3d_min": 7.0,
    "slope5d_min": 8.0,
    "require_prev_green": True,      # D-1 close > open
    "d1_min_atr_body": 0.2,          # (D-1 close - open) / ATR30_prev

    # “Significant high” gating for level selection
    "sig_require_pivot": True,
    "sig_level_pos_abs_min": 0.55,   # pos in [absLow..absTop] of last 1000d excl last 10d
    "sig_pivot_percentile_min": 0.55,# pivot rank among pivots in that window (by price)
    "sig_prom_atr_min": 1.0,         # prominence vs local valleys, in ATRs
    "pivot_left": 1, "pivot_right": 1,
    "prom_left_days": 5, "prom_right_days": 5,
    # optional: require the level to be within X% of the window’s absolute top
    "sig_within_top_pct_max": 3.0,   # e.g., within 3% of top

    # Spike cleanup (helps ignore split / fat-finger highs)
    "clean_spikes": True,
    "spike_atr_k": 8.0,              # cap High/Low at OC +/- k*ATR30_raw
    "spike_max_up_pct": 3.0,         # cap High <= PrevClose*(1+300%)
    "spike_max_dn_pct": 3.0,         # cap Low  >= PrevClose*(1-300%)

    # run controls
    "max_workers": 6,
    "start_date": "2024-01-01",
    "end_date": datetime.today().strftime("%Y-%m-%d"),

    # diagnostics
    "debug_reasons": True,           # print per-symbol rejection counts
    "verbose_every": 1,              # progress ping every N symbols
}

# ─────────────────────────── data ───────────────────────────

def fetch_daily(tkr: str, start: str, end: str) -> pd.DataFrame:
    url = f"{BASE_URL}/v2/aggs/ticker/{tkr}/range/1/day/{start}/{end}"
    r = S.get(url, params={"apiKey": API_KEY, "adjusted":"true", "sort":"asc", "limit":50000}, timeout=30)
    r.raise_for_status()
    rows = r.json().get("results", [])
    if not rows: return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .assign(Date=lambda d: pd.to_datetime(d["t"], unit="ms", utc=True)
                                  .dt.tz_convert("America/New_York")
                                  .dt.normalize()
                                  .dt.tz_localize(None))
        .rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
        .set_index("Date")[["Open","High","Low","Close","Volume"]]
        .sort_index()
    )

def _raw_atr30(df: pd.DataFrame) -> pd.Series:
    hi_lo = df["High"] - df["Low"]
    hi_pc = (df["High"] - df["Close"].shift(1)).abs()
    lo_pc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr = pd.concat([hi_lo, hi_pc, lo_pc], axis=1).max(axis=1)
    return tr.rolling(30, min_periods=30).mean()  # unshifted

def clean_spikes(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    if df.empty or not p.get("clean_spikes", True):
        return df.copy()
    m = df.copy()
    atr = _raw_atr30(m)
    prev_close = m["Close"].shift(1)
    oc_max = m[["Open","Close"]].max(axis=1)
    oc_min = m[["Open","Close"]].min(axis=1)

    hi_cap = pd.concat([oc_max + p["spike_atr_k"]*atr, prev_close*(1+p["spike_max_up_pct"])], axis=1).min(axis=1)
    lo_cap = pd.concat([oc_min - p["spike_atr_k"]*atr, prev_close*(1-p["spike_max_dn_pct"])], axis=1).max(axis=1)

    m["High"] = np.minimum(m["High"], hi_cap)
    m["Low"]  = np.maximum(m["Low"],  lo_cap)
    # candle integrity
    m["High"] = np.maximum(m["High"], m[["Open","Close","Low"]].max(axis=1))
    m["Low"]  = np.minimum(m["Low"],  m[["Open","Close","High"]].min(axis=1))
    return m

# ───────────────────────── metrics ─────────────────────────

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    m = df.copy()
    hi_lo = m["High"] - m["Low"]
    hi_pc = (m["High"] - m["Close"].shift(1)).abs()
    lo_pc = (m["Low"]  - m["Close"].shift(1)).abs()
    tr = pd.concat([hi_lo, hi_pc, lo_pc], axis=1).max(axis=1)
    m["TR"] = tr
    m["ATR30_prev"] = tr.rolling(30, min_periods=30).mean().shift(1)
    m["ADV20_$"] = (m["Close"] * m["Volume"]).rolling(20, min_periods=20).mean()
    m["VOL_AVG"] = m["Volume"].rolling(30, min_periods=30).mean().shift(1)

    m["Prev_Open"]   = m["Open"].shift(1)
    m["Prev_Close"]  = m["Close"].shift(1)
    m["Prev_High"]   = m["High"].shift(1)
    m["Prev_Volume"] = m["Volume"].shift(1)

    # slopes measured to D-1 (Prev_Close)
    for k in (2,3,5):
        m[f"Slope{k}d_%"] = (m["Prev_Close"] - m["Close"].shift(k+1)) / m["Close"].shift(k+1) * 100
    return m

# ───────────────────────── helpers ─────────────────────────

def last_occurrence_of_max(series: pd.Series):
    if series.empty: return None
    mx = series.max()
    s = series[series == mx]
    return None if s.empty else s.index[-1]

def trading_days_between(idx: pd.DatetimeIndex, a: pd.Timestamp, b: pd.Timestamp) -> int:
    return int(((idx > a) & (idx < b)).sum())

def _pivot_high_dates(df: pd.DataFrame, left: int, right: int):
    h, idx = df["High"].values, df.index
    out, n = [], len(df)
    for i in range(left, n - right):
        lv = h[i-left:i].max() if left  > 0 else -np.inf
        rv = h[i+1:i+1+right].max() if right > 0 else -np.inf
        if np.isfinite(h[i]) and h[i] >= lv and h[i] > rv:
            out.append(idx[i])
    return out

def _local_valley_min(df: pd.DataFrame, center: pd.Timestamp, left_days: int, right_days: int):
    lwin = df.loc[(df.index >= center - pd.Timedelta(days=left_days)) & (df.index <= center)]
    rwin = df.loc[(df.index >= center) & (df.index <= center + pd.Timedelta(days=right_days))]
    lmin = float(lwin["Low"].min()) if not lwin.empty else np.nan
    rmin = float(rwin["Low"].min()) if not rwin.empty else np.nan
    return lmin, rmin

def significance_metrics(m: pd.DataFrame, level_date: pd.Timestamp, d0: pd.Timestamp, p: dict) -> dict:
    cut = d0 - pd.Timedelta(days=p["exclude_recent_days"])
    wstart = cut - pd.Timedelta(days=p["lookback_days_for_level"])
    win = m[(m.index > wstart) & (m.index <= cut)]
    if win.empty or level_date not in win.index:
        return {"LevelPosAbs": np.nan, "LevelPctOfTop": np.nan,
                "LevelPivotProm_ATR": np.nan, "LevelPivotPercentile": np.nan,
                "IsPivotHigh": False, "AbsTop": np.nan, "AbsTop_Date": None}
    abs_top = float(win["High"].max()); abs_low = float(win["Low"].min())
    level_px = float(win.at[level_date, "High"])
    pos_abs = (level_px - abs_low) / (abs_top - abs_low) if abs_top > abs_low else np.nan
    pct_of_top = level_px / abs_top if abs_top > 0 else np.nan

    piv_dates = _pivot_high_dates(win, p["pivot_left"], p["pivot_right"])
    is_pivot = level_date in piv_dates
    pct_rank = float((win.loc[piv_dates, "High"] <= level_px).mean()) if piv_dates else np.nan

    lmin, rmin = _local_valley_min(win, level_date, p["prom_left_days"], p["prom_right_days"])
    ref_valley = np.nanmax([lmin, rmin])
    atr_at_lvl = float(win.at[level_date, "ATR30_prev"]) if "ATR30_prev" in win.columns else np.nan
    prom_atr = ((level_px - ref_valley) / atr_at_lvl) if (np.isfinite(ref_valley) and atr_at_lvl) else np.nan

    return {
        "LevelPosAbs": round(pos_abs, 3) if pd.notna(pos_abs) else np.nan,
        "LevelPctOfTop": round(pct_of_top, 3) if pd.notna(pct_of_top) else np.nan,
        "LevelPivotProm_ATR": round(prom_atr, 2) if pd.notna(prom_atr) else np.nan,
        "LevelPivotPercentile": round(pct_rank, 3) if pd.notna(pct_rank) else np.nan,
        "IsPivotHigh": bool(is_pivot),
        "AbsTop": round(abs_top, 4) if pd.notna(abs_top) else np.nan,
        "AbsTop_Date": win["High"].idxmax().strftime("%Y-%m-%d") if not win.empty else None,
    }

# pick most-recent significant pivot (not merely the absolute max)
def pick_level_date(m: pd.DataFrame, d0: pd.Timestamp, p: dict) -> pd.Timestamp | None:
    i0 = m.index.get_loc(d0)
    lookback_start = d0 - timedelta(days=p["lookback_days_for_level"])
    prior = m[(m.index >= lookback_start) & (m.index <= m.index[i0-1])]
    if prior.empty:
        return None

    tail = prior.index[-p["min_trade_days_between"]:] if len(prior) >= p["min_trade_days_between"] else prior.index[:0]
    core = prior.drop(index=tail, errors="ignore")
    if core.empty: return None

    pivs = _pivot_high_dates(core, p["pivot_left"], p["pivot_right"])
    candidates = pivs if pivs else list(core.index)

    keep = []
    for dt in candidates:
        lvl = float(m.at[dt, "High"])
        sig = significance_metrics(m, dt, d0, p)
        if p["sig_require_pivot"] and not sig["IsPivotHigh"]:
            continue
        if pd.notna(sig["LevelPosAbs"]) and sig["LevelPosAbs"] < p["sig_level_pos_abs_min"]:
            continue
        if pd.notna(sig["LevelPivotPercentile"]) and sig["LevelPivotPercentile"] < p["sig_pivot_percentile_min"]:
            continue
        if pd.notna(sig["LevelPivotProm_ATR"]) and sig["LevelPivotProm_ATR"] < p["sig_prom_atr_min"]:
            continue
        if pd.notna(sig["AbsTop"]) and p.get("sig_within_top_pct_max") is not None:
            if lvl < sig["AbsTop"] * (1 - p["sig_within_top_pct_max"]/100.0):
                continue
        keep.append(dt)

    if keep:
        return max(keep)
    if pivs:
        return last_occurrence_of_max(core.loc[pivs, "High"])
    return last_occurrence_of_max(core["High"])

# ───────────────────────── scanner ─────────────────────────

def scan_symbol(sym: str, start: str, end: str, p: dict) -> pd.DataFrame:
    raw = fetch_daily(sym, start, end)
    if raw.empty or len(raw) < 80:
        if p["debug_reasons"]:
            print(f"{sym}: no data / too short", flush=True)
        return pd.DataFrame()

    df = clean_spikes(raw, p)
    m  = compute_metrics(df)
    hits = []
    why = {"liq":0,"days_between":0,"gap":0,"touch":0,"break":0,"close":0,
           "pullback":0,"d1_green":0,"d1_body":0,"vol":0,"volavg":0,"slope":0,"sig":0}

    for d0 in m.index[60:]:
        r = m.loc[d0]

        # liquidity guards
        if r["Prev_Close"] < p["price_min"] or r["ADV20_$"] < p["adv20_min_usd"]:
            why["liq"] += 1; continue

        level_date = pick_level_date(m, d0, p)
        if level_date is None:
            why["sig"] += 1; continue
        days_between = trading_days_between(m.index, level_date, d0)
        if days_between < p["min_trade_days_between"]:
            why["days_between"] += 1; continue

        level = float(m.at[level_date, "High"])
        o,h,l,c = map(float, (r["Open"], r["High"], r["Low"], r["Close"]))
        pc = float(r["Prev_Close"])

        gap_pct = (o - pc) / pc * 100.0 if pc else np.nan
        if pd.notna(gap_pct) and gap_pct < p["gap_min_pct"]:
            why["gap"] += 1; continue

        if o < level * (1 - p["touch_tol_pct"]/100.0):
            why["touch"] += 1; continue
        if p["require_intraday_break"] and h < level*(1 + p["break_tol_pct"]/100.0):
            why["break"] += 1; continue
        if p["require_close_below_level"] and not (c < level):
            why["close"] += 1; continue

        between = m[(m.index > level_date) & (m.index < d0)]
        if between.empty:
            why["pullback"] += 1; continue
        min_low = float(between["Low"].min())
        pull_dn = (level - min_low) / level * 100.0 if level else np.nan
        if pd.notna(pull_dn) and pull_dn < p["min_low_pullback_min_pct"]:
            why["pullback"] += 1; continue

        # D-1 state
        if p["require_prev_green"] and not (r["Prev_Close"] > r["Prev_Open"]):
            why["d1_green"] += 1; continue
        d1_body_atr = (r["Prev_Close"] - r["Prev_Open"]) / r["ATR30_prev"] if r["ATR30_prev"] else np.nan
        if pd.notna(d1_body_atr) and d1_body_atr < p["d1_min_atr_body"]:
            why["d1_body"] += 1; continue
        if pd.notna(r["Prev_Volume"]) and r["Prev_Volume"] < p["prev_volume_min"]:
            why["vol"] += 1; continue
        vol_over = (r["Prev_Volume"] / r["VOL_AVG"]) if r["VOL_AVG"] else np.nan
        if pd.notna(vol_over) and vol_over < p["prev_vol_over_avg_min"]:
            why["volavg"] += 1; continue

        ok_slopes = True
        for k, th in (("Slope2d_%", p["slope2d_min"]), ("Slope3d_%", p["slope3d_min"]), ("Slope5d_%", p["slope5d_min"])):
            if k in r and pd.notna(r[k]) and r[k] < th:
                ok_slopes = False
                break
        if not ok_slopes:
            why["slope"] += 1; continue

        # significance (re-check in case of dynamic window)
        sig = significance_metrics(m, level_date, d0, p)
        if p["sig_require_pivot"] and not sig["IsPivotHigh"]:
            why["sig"] += 1; continue
        if pd.notna(sig["LevelPosAbs"]) and sig["LevelPosAbs"] < p["sig_level_pos_abs_min"]:
            why["sig"] += 1; continue
        if pd.notna(sig["LevelPivotPercentile"]) and sig["LevelPivotPercentile"] < p["sig_pivot_percentile_min"]:
            why["sig"] += 1; continue
        if pd.notna(sig["LevelPivotProm_ATR"]) and sig["LevelPivotProm_ATR"] < p["sig_prom_atr_min"]:
            why["sig"] += 1; continue

        hits.append({
            "Ticker": sym,
            "High_Date": level_date.strftime("%Y-%m-%d"),
            "Target_Date": d0.strftime("%Y-%m-%d"),
            "Days_From_Set": days_between,
            "Level": round(level,4),
            "Open": round(o,4), "High": round(h,4), "Low": round(l,4), "Close": round(c,4),
            "Gap_%": round(gap_pct,2) if pd.notna(gap_pct) else None,
            "Touch_Dist_%": round(((o - level)/level)*100.0, 2),
            "Between_MinLow": round(min_low,4),
            "PullbackDown_%": round(pull_dn,2),
            "Prev_Volume": int(r["Prev_Volume"]) if pd.notna(r["Prev_Volume"]) else None,
            "Prev_Vol/Avg": round(vol_over,2) if pd.notna(vol_over) else None,
            "ADV20_$": int(r["ADV20_$"]) if pd.notna(r["ADV20_$"]) else None,
            "D1_Green": bool(r["Prev_Close"] > r["Prev_Open"]),
            "D1_Body/ATR": None if pd.isna(d1_body_atr) else round(d1_body_atr,2),
            "Slope2d_%": None if pd.isna(r["Slope2d_%"]) else round(r["Slope2d_%"],2),
            "Slope3d_%": None if pd.isna(r["Slope3d_%"]) else round(r["Slope3d_%"],2),
            "Slope5d_%": None if pd.isna(r["Slope5d_%"]) else round(r["Slope5d_%"],2),
            "LevelPosAbs": sig["LevelPosAbs"],
            "LevelPctOfTop": sig["LevelPctOfTop"],
            "LevelPivotProm_ATR": sig["LevelPivotProm_ATR"],
            "LevelPivotPercentile": sig["LevelPivotPercentile"],
            "IsPivotHigh": sig["IsPivotHigh"],
        })

    if not hits and P["debug_reasons"]:
        print(f"{sym}: no hits — {why}", flush=True)
    return pd.DataFrame(hits)

def fetch_and_scan(sym: str, start: str, end: str, p: dict):
    try:
        return scan_symbol(sym, start, end, p)
    except Exception as e:
        if p["debug_reasons"]:
            print(f"{sym}: ERROR {e}", flush=True)
        return pd.DataFrame()

# ───────────────────────── main ─────────────────────────

if __name__ == "__main__":
    symbols = [
        # start with a small set; expand after you see matches
        "NVDA","ORCL","COIN","MARA","MSTR","TSLA","AAPL","AMZN","MSFT","META",
        "AMD","SMCI","RIOT","QQQ","SPY","IWM","RIVN","PLTR","DKNG","SHOP"
    ]

    t0 = time.time()
    print(f"Scanning {len(symbols)} symbols {P['start_date']} → {P['end_date']} with {P['max_workers']} workers…", flush=True)

    results, done = [], 0
    with ThreadPoolExecutor(max_workers=P["max_workers"]) as exe:
        futs = {exe.submit(fetch_and_scan, s, P["start_date"], P["end_date"], P): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                df = fut.result()
            except Exception as e:
                df = pd.DataFrame()
                if P["debug_reasons"]:
                    print(f"{sym}: ERROR {e}", flush=True)
            done += 1
            if P["verbose_every"] and (done % P["verbose_every"] == 0):
                print(f"[{done}/{len(symbols)}] {sym}  hits={0 if df is None or df.empty else len(df)}", flush=True)
            if df is not None and not df.empty:
                results.append(df)

    print(f"Finished in {time.time()-t0:0.1f}s", flush=True)

    if results:
        out = (pd.concat(results, ignore_index=True)
                 .sort_values(["Target_Date","Ticker"], ascending=[False, True]))
        pd.set_option("display.max_columns", None, "display.width", 0)
        print("\nMATCHES:\n", flush=True)
        print(out.to_string(index=False))
    else:
        print("\nNo short FBO hits for given symbols/date range.", flush=True)
