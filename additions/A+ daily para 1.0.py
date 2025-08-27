# daily_para_scan_daily_only.py
# A+ daily parabola mold — DAILY ONLY (no intraday, no dev bands)

import pandas as pd, numpy as np, requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------- config --------
session  = requests.Session()
API_KEY  = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
BASE_URL = "https://api.polygon.io"
MAX_WORKERS = 5

# -------- knobs (daily only) --------
P = {
    # hard liquidity / price
    "price_min"        : 8.0,
    "adv20_min_usd"    : 30_000_000,

    # daily mold thresholds (OG)
    "atr_mult"               : 2.5,  # TR_prev / ATR
    "vol_mult"               : 2.0,  # Volume/VOL_AVG and Prev_Volume/VOL_AVG
    "slope3d_min"            : 10,
    "slope5d_min"            : 20,
    "slope15d_min"           : 50,
    "high_ema9_mult"         : 3.5,  # (High-EMA9)/ATR
    "high_ema20_mult"        : 5.0,  # (High-EMA20)/ATR
    "pct7d_low_div_atr_min"  : 0.5,
    "pct14d_low_div_atr_min" : 1.5,
    "gap_div_atr_min"        : 0.5,
    "open_over_ema9_min"     : 1.0,
    "atr_pct_change_min"     : 9.0,
    "prev_close_min"         : 10.0,
    "pct2d_div_atr_min"      : 2.0,  # (Prev_Close - Close_D3)/ATR
    "pct3d_div_atr_min"      : 2.5,  # (Prev_Close - Close_D4)/ATR

    # optional toggles
    "require_green_prev"        : False,  # Prev_Close > Prev_Open
    "require_open_gt_prev_high" : False,  # Open > Prev_High
}

# -------- data fetch --------
def fetch_daily(tkr, start, end):
    url = f"{BASE_URL}/v2/aggs/ticker/{tkr}/range/1/day/{start}/{end}"
    r   = session.get(url, params={"apiKey": API_KEY, "adjusted":"true", "sort":"asc", "limit":50000})
    r.raise_for_status()
    rows = r.json().get("results", [])
    if not rows: return pd.DataFrame()
    return (pd.DataFrame(rows)
            .assign(Date=lambda d: pd.to_datetime(d["t"], unit="ms"))
            .rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
            .set_index("Date")[["Open","High","Low","Close","Volume"]]
            .sort_index())

# -------- daily metrics --------
def add_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()

    # EMAs
    out["EMA_9"]  = out["Close"].ewm(span=9 , adjust=False).mean()
    out["EMA_20"] = out["Close"].ewm(span=20, adjust=False).mean()

    # TR / ATR(14)
    hi_lo   = out["High"] - out["Low"]
    hi_prev = (out["High"] - out["Close"].shift(1)).abs()
    lo_prev = (out["Low"]  - out["Close"].shift(1)).abs()
    out["TR"]      = pd.concat([hi_lo, hi_prev, lo_prev], axis=1).max(axis=1)
    out["TR_prev"] = out["TR"].shift(1)
    out["ATR_raw"] = out["TR"].rolling(14, min_periods=14).mean()
    out["ATR"]     = out["ATR_raw"].shift(1)
    out["ATR_Pct_Change"] = out["ATR_raw"].pct_change().shift(1) * 100

    # Volume / ADV
    out["VOL_AVG"]     = out["Volume"].rolling(14, min_periods=14).mean().shift(1)
    out["Prev_Volume"] = out["Volume"].shift(1)
    out["ADV20_$"]     = (out["Close"] * out["Volume"]).rolling(20, min_periods=20).mean().shift(1)

    # Slopes on EMA9
    for w in (3,5,15):
        out[f"Slope_9_{w}d"] = (out["EMA_9"] - out["EMA_9"].shift(w)) / out["EMA_9"].shift(w) * 100

    # Gap / div-ATR features
    out["Gap"]          = (out["Open"] - out["Close"].shift(1)).abs()
    out["Gap_over_ATR"] = out["Gap"] / out["ATR"]
    out["High_over_EMA9_div_ATR"]  = (out["High"] - out["EMA_9"])  / out["ATR"]
    out["High_over_EMA20_div_ATR"] = (out["High"] - out["EMA_20"]) / out["ATR"]
    out["Open_over_EMA9"]          = out["Open"] / out["EMA_9"]

    # % from recent lows (7,14)
    low7  = out["Low"].rolling(7 , min_periods=7 ).min()
    low14 = out["Low"].rolling(14, min_periods=14).min()
    out["Pct_7d_low_div_ATR"]  = ((out["Close"] - low7)  / low7 ) / out["ATR"] * 100
    out["Pct_14d_low_div_ATR"] = ((out["Close"] - low14) / low14) / out["ATR"] * 100

    # Multi-day refs
    out["Prev_Close"] = out["Close"].shift(1)
    out["Prev_Open"]  = out["Open"].shift(1)
    out["Prev_High"]  = out["High"].shift(1)
    out["Close_D3"]   = out["Close"].shift(3)
    out["Close_D4"]   = out["Close"].shift(4)
    out["Move2d_div_ATR"] = (out["Prev_Close"] - out["Close_D3"]) / out["ATR"]
    out["Move3d_div_ATR"] = (out["Prev_Close"] - out["Close_D4"]) / out["ATR"]

    # Convenience ratios
    out["Range_over_ATR_prev"] = out["TR_prev"]    / out["ATR"]
    out["Vol_over_AVG"]        = out["Volume"]     / out["VOL_AVG"]
    out["PrevVol_over_AVG"]    = out["Prev_Volume"]/ out["VOL_AVG"]
    return out

# -------- filter --------
def passes_daily(row) -> bool:
    # hard gates
    if pd.isna(row["Prev_Close"]) or pd.isna(row["ADV20_$"]):
        return False
    if (row["Prev_Close"] < P["price_min"]) or (row["ADV20_$"] < P["adv20_min_usd"]):
        return False

    checks = [
        row["Range_over_ATR_prev"] >= P["atr_mult"],
        row["Vol_over_AVG"]       >= P["vol_mult"],
        row["PrevVol_over_AVG"]   >= P["vol_mult"],
        row["Slope_9_3d"]         >= P["slope3d_min"],
        row["Slope_9_5d"]         >= P["slope5d_min"],
        row["Slope_9_15d"]        >= P["slope15d_min"],
        row["High_over_EMA9_div_ATR"]  >= P["high_ema9_mult"],
        row["High_over_EMA20_div_ATR"] >= P["high_ema20_mult"],
        row["Pct_7d_low_div_ATR"]      >= P["pct7d_low_div_atr_min"],
        row["Pct_14d_low_div_ATR"]     >= P["pct14d_low_div_atr_min"],
        row["Gap_over_ATR"]            >= P["gap_div_atr_min"],
        row["Open_over_EMA9"]          >= P["open_over_ema9_min"],
        row["ATR_Pct_Change"]          >= P["atr_pct_change_min"],
        row["Prev_Close"]              >  P["prev_close_min"],
        row["Move2d_div_ATR"]          >= P["pct2d_div_atr_min"],
        row["Move3d_div_ATR"]          >= P["pct3d_div_atr_min"],
    ]
    if P["require_green_prev"] and not (row["Prev_Close"] > row["Prev_Open"]):
        return False
    if P["require_open_gt_prev_high"] and not (row["Open"] > row["Prev_High"]):
        return False
    return all(bool(x) for x in checks)

# -------- worker --------
def scan_symbol(sym, start, end):
    df = fetch_daily(sym, start, end)
    if df.empty: return pd.DataFrame()
    m  = add_daily_metrics(df)
    hits = []
    for d, row in m.iloc[1:].iterrows():
        if passes_daily(row):
            hits.append({
                "Ticker": sym, "Date": d.strftime("%Y-%m-%d"),
                "Prev_Close": round(float(row["Prev_Close"]),2),
                "ADV20_$": float(row["ADV20_$"]),
                "Slope_9_3d": round(float(row["Slope_9_3d"]),2),
                "Slope_9_5d": round(float(row["Slope_9_5d"]),2),
                "Slope_9_15d": round(float(row["Slope_9_15d"]),2),
                "High_EMA9_ATR": round(float(row["High_over_EMA9_div_ATR"]),2),
                "High_EMA20_ATR": round(float(row["High_over_EMA20_div_ATR"]),2),
                "Gap_ATR": round(float(row["Gap_over_ATR"]),2),
                "Open_over_EMA9": round(float(row["Open_over_EMA9"]),2),
                "ATR_pct_change": round(float(row["ATR_Pct_Change"]),2),
                "Move2d_ATR": round(float(row["Move2d_div_ATR"]),2),
                "Move3d_ATR": round(float(row["Move3d_div_ATR"]),2),
            })
    return pd.DataFrame(hits)

# -------- main --------
if __name__ == "__main__":
    symbols = ['MSTR','SMCI','DJT','BABA','TCOM','AMC','SOXL','MRVL','TGT','DOCU',
               'ZM','DIS','NFLX','SNAP','RBLX','META','SE','NVDA','AAPL','MSFT',
               'GOOGL','AMZN','TSLA','AMD','INTC','BA','PYPL','QCOM','ORCL','KO',
               'PEP','ABBV','JNJ','CRM','BAC','JPM','WMT','CVX','XOM','COP','RTX',
               'SPGI','GS','HD','LOW','COST','UNH','NKE','LMT','HON','CAT','LIN',
               'ADBE','AVGO','TXN','ACN','UPS','BLK','PM','ELV','VRTX','ZTS','NOW',
               'ISRG','PLD','MS','MDT','WM','GE','IBM','BKNG','FDX','ADP','EQIX',
               'DHR','SNPS','REGN','SYK','TMO','CVS','INTU','SCHW','CI','APD','SO',
               'MMC','ICE','FIS','ADI','CSX','LRCX','GILD','RIVN','PLTR','SNOW',
               'SPY','QQQ','IWM','RIOT','MARA','COIN','MRNA','CELH','UPST','AFRM','DKNG']

    start = "2023-01-01"
    end   = datetime.today().strftime("%Y-%m-%d")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futs = {exe.submit(scan_symbol, s, start, end): s for s in symbols}
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None and not df.empty:
                results.append(df)

    if results:
        out = (pd.concat(results, ignore_index=True)
                 .sort_values(["Date","Ticker"], ascending=[False, True]))
        pd.set_option("display.max_columns", None, "display.width", 0)
        print("\nA+ DAILY (no intraday) candidates:\n")
        print(out.to_string(index=False))
        # out.to_csv("daily_para_hits.csv", index=False)
    else:
        print("No hits.")
