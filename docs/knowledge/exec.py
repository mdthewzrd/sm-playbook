import matplotlib
matplotlib.use("Qt5Agg")

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mplfinance.original_flavor import candlestick_ohlc

plt.style.use('dark_background')

# === CONFIG ===
API_KEY = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
TICKER = "SRFM"
TARGET_DATE = pd.Timestamp("2025-06-25")
END_DATE = (TARGET_DATE + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

# === DATA FETCH ===
def fetch_polygon_agg(ticker, multiplier, timespan, from_date, to_date, api_key):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "results" not in data:
        raise ValueError("Polygon API error or no data found")
    df = pd.DataFrame(data["results"])
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("US/Eastern")
    df.set_index("timestamp", inplace=True)
    df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]]

# === BAND LOGIC ===
def add_bands(df, ema_short, ema_long, du1, du2, dl1, dl2, prefix=""):
    df[f"{prefix}emaS"] = df["Close"].ewm(span=ema_short).mean()
    df[f"{prefix}emaL"] = df["Close"].ewm(span=ema_long).mean()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df[f"{prefix}ATR_S"] = tr.rolling(ema_short).mean()
    df[f"{prefix}ATR_L"] = tr.rolling(ema_long).mean()
    df[f"{prefix}UPP1"] = df[f"{prefix}emaS"] + du1 * df[f"{prefix}ATR_S"]
    df[f"{prefix}UPP2"] = df[f"{prefix}emaS"] + du2 * df[f"{prefix}ATR_S"]
    df[f"{prefix}LOW1"] = df[f"{prefix}emaL"] - dl1 * df[f"{prefix}ATR_L"]
    df[f"{prefix}LOW2"] = df[f"{prefix}emaL"] - dl2 * df[f"{prefix}ATR_L"]

# === TIME FILTERING ===
def compress_market_time(df):
    df = df.copy()
    df = df[((df.index.time >= pd.to_datetime("04:00").time()) & (df.index.time < pd.to_datetime("08:00").time())) |
            ((df.index.time >= pd.to_datetime("09:30").time()) & (df.index.time <= pd.to_datetime("20:00").time()))]
    df = df[df.index.dayofweek < 5]
    df["compressed_time"] = pd.Series(range(len(df)), index=df.index)
    df["datetime"] = df.index.tz_convert("US/Eastern")
    return df

# === OHLC FOR MPL ===
def to_ohlc(df, xcol):
    return list(zip(df[xcol], df["Open"], df["High"], df["Low"], df["Close"]))

# === FETCH ===
start = (TARGET_DATE - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
full_15m = fetch_polygon_agg(TICKER, 15, "minute", start, END_DATE, API_KEY)
full_5m = fetch_polygon_agg(TICKER, 5, "minute", start, END_DATE, API_KEY)
full_2m = fetch_polygon_agg(TICKER, 2, "minute", start, END_DATE, API_KEY)

# === BANDS ===
add_bands(full_15m, 72, 89, 6.6, 9.6, 6.6, 9.6)
add_bands(full_15m, 9, 20, 0.1, 0.5, 2.0, 2.5, prefix="s_")
add_bands(full_5m, 72, 89, 7.9, 6.0, 7.9, 6.0)
add_bands(full_5m, 9, 20, 1, 0.5, 2.0, 2.5, prefix="s_")
add_bands(full_2m, 9, 20, 3.3, 2.4, 2.5, 3.3, prefix="r_")
add_bands(full_2m, 9, 20, 1, 0.5, 2.0, 2.5, prefix="q_")

# === VWAP Calculation (per day grouping) ===
for df in [full_5m, full_2m]:
    df["date_only"] = df.index.date
    df["cum_vol"] = df.groupby("date_only")["Volume"].cumsum()
    df["cum_vol_price"] = df.groupby("date_only").apply(lambda x: (x["Close"] * x["Volume"]).cumsum()).reset_index(level=0, drop=True)
    df["VWAP"] = df["cum_vol_price"] / df["cum_vol"]


# === COMPRESS ===
df_15m = compress_market_time(full_15m)
df_5m = compress_market_time(full_5m)
df_2m = compress_market_time(full_2m)

# === ENTRY LOGIC ===
band = df_15m.between_time("07:00", "12:00")
hit15 = band[(band["High"] >= band["UPP1"]) & (band["Open"] < band["UPP1"])].head(1)
entries15 = [(hit15["compressed_time"].iloc[0], hit15["High"].iloc[0])] if not hit15.empty else []

entries2 = []
entries5 = []

if not hit15.empty:
    upp1_price = hit15["UPP1"].iloc[0]

    # Find first 2M candle where High >= UPP1
    for i in range(len(df_2m)):
        row = df_2m.iloc[i]
        if row["High"] >= upp1_price:
            t_hit = row.name  # timestamp when 2M actually hits UPP1
            df2_after = df_2m.loc[t_hit:]
            break
    else:
        df2_after = pd.DataFrame()  # fallback if not found

    # Proceed with bar break detection only after confirmed hit
    for i in range(2, len(df2_after)):
        curr = df2_after.iloc[i]
        prev = df2_after.iloc[i - 1]
        if curr["High"] == curr["Low"] == curr["Open"] == curr["Close"]:
            continue
        if curr["Low"] < prev["Low"]:
            entries2.append((curr["compressed_time"], prev["Low"]))
            break

        df5_after = df_5m.loc[t_hit:]
        prev_low = None
        for i in range(len(df5_after)):
            curr = df5_after.iloc[i]

            if curr["High"] == curr["Low"] == curr["Open"] == curr["Close"]:
                continue

            if prev_low is not None and curr["Low"] < prev_low:
                # map to 2M chart
                t5 = curr.name
                df2_after_5m = df_2m.loc[t5:]
                for k in range(len(df2_after_5m)):
                    mapped = df2_after_5m.iloc[k]
                    entries2.append((mapped["compressed_time"], prev_low))  # plot 5M break on 2M chart
                    break
                break  # stop after first 5M break is handled

            prev_low = curr["Low"]


        # fallback in case no break happened — plot the first bar anyway
        if not entries5 and len(df5_after) > 0:
            first = df5_after.iloc[0]
            entries5.append((first["compressed_time"], first["Low"]))
# === THIRD ENTRY: VWAP break -> Dev band pop (after) -> 2M bar break ===
entry3 = None

if not hit15.empty and not df_5m.empty and "VWAP" in df_5m.columns:
    # Step 1: First 5M VWAP close break
    vwap_triggered = df_5m.loc[t_hit:]
    vwap_triggered = vwap_triggered[vwap_triggered["Close"] < vwap_triggered["VWAP"]]

    if not vwap_triggered.empty:
        vwap_trigger_time = vwap_triggered.index[0]

        # Step 2: From NEXT 5M candle onward, look for dev band pop
        post_vwap_after = df_5m[df_5m.index > vwap_trigger_time]
        dev_hit = post_vwap_after[post_vwap_after["High"] >= post_vwap_after["s_UPP2"]]

        if not dev_hit.empty:
            dev_hit_time = dev_hit.index[0]

            # Step 3: From dev_hit_time, find first 2M bar break
            df2_after = df_2m.loc[dev_hit_time:]
            for i in range(2, len(df2_after)):
                curr = df2_after.iloc[i]
                prev = df2_after.iloc[i - 1]
                if curr["High"] == curr["Low"] == curr["Open"] == curr["Close"]:
                    continue
                if curr["Low"] < prev["Low"]:
                    entry3 = (curr["compressed_time"], prev["Low"])  # plot 2M break
                    break

# === FOURTH ENTRY: Post-VWAP 5M bar low break -> Dev band pop -> 2M bar break ===
entry4 = None

if not hit15.empty and not df_5m.empty and "VWAP" in df_5m.columns and "s_UPP2" in df_5m.columns:
    if not vwap_triggered.empty:
        # Start from the bar *after* the VWAP close
        vwap_idx = df_5m.index.get_loc(vwap_trigger_time)
        df5_post_vwap = df_5m.iloc[vwap_idx + 1:]

        trigger_idx = None
        for i in range(1, len(df5_post_vwap)):
            curr = df5_post_vwap.iloc[i]
            prev = df5_post_vwap.iloc[i - 1]
            if curr["Close"] < prev["Low"]:
                trigger_idx = df5_post_vwap.index[i]
                break

        if trigger_idx:
            # Look for dev band hit AFTER trigger
            df5_after_trigger = df_5m[df_5m.index > trigger_idx]
            dev_hit_5m = df5_after_trigger[df5_after_trigger["High"] >= df5_after_trigger["s_UPP2"]]

            if not dev_hit_5m.empty:
                dev_hit_time = dev_hit_5m.index[0]

                # Look for 2M bar break entry — fix timing by starting from index 1
                df2_after = df_2m.loc[dev_hit_time:]
                for j in range(1, len(df2_after)):
                    curr = df2_after.iloc[j]
                    prev = df2_after.iloc[j - 1]
                    if curr["High"] == curr["Low"] == curr["Open"] == curr["Close"]:
                        continue
                    if curr["Low"] < prev["Low"]:
                        entry4 = (curr["compressed_time"], prev["Low"])
                        break



# === PLOT ===
ohlc_15 = to_ohlc(df_15m, "compressed_time")
ohlc_5 = to_ohlc(df_5m, "compressed_time")
ohlc_2 = to_ohlc(df_2m, "compressed_time")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(26, 8), sharey=False, sharex=False, constrained_layout=True)

for ax, ohlc, df, title in zip(
    [ax1, ax2, ax3], [ohlc_15, ohlc_5, ohlc_2], [df_15m, df_5m, df_2m], ["15M Chart", "5M Chart", "2M Chart"]):

    for i in range(len(df) - 1):
        hour = df.index[i].time()
        if (hour >= pd.to_datetime("04:00").time() and hour < pd.to_datetime("09:30").time()) or \
           (hour >= pd.to_datetime("16:00").time() and hour < pd.to_datetime("20:00").time()):
            ax.axvspan(df["compressed_time"].iloc[i], df["compressed_time"].iloc[i + 1], color="#222222", alpha=0.85)

    candlestick_ohlc(ax, ohlc, width=0.5, colorup="white", colordown="red")

    for col1, col2, color, alpha in [
        ("UPP1", "UPP2", "lightcoral", 0.5),
        ("LOW1", "LOW2", "lightgreen", 0.5),
        ("s_UPP1", "s_UPP2", "pink", 0.3),
        ("s_LOW1", "s_LOW2", "cyan", 0.3)
    ]:
        if col1 in df.columns and col2 in df.columns:
            ax.fill_between(df["compressed_time"], df[col1], df[col2], color=color, alpha=alpha)

        # Plot EMA and VWAP lines
    for col, style, label, color in [
        ("s_emaS", "--", "EMA9", "yellow"),
        ("s_emaL", "--", "EMA20", "lightskyblue"),
        ("VWAP", "--", "VWAP", "purple")  # <- ADD THIS LINE
    ]:
        if col in df.columns:
            ax.plot(df["compressed_time"], df[col], linestyle=style, label=label, color=color)

    if title == "2M Chart":
        ax.fill_between(df["compressed_time"], df["r_UPP1"], df["r_UPP2"], color="lightcoral", alpha=0.6)
        ax.fill_between(df["compressed_time"], df["r_LOW1"], df["r_LOW2"], color="green", alpha=0.6)
        ax.fill_between(df["compressed_time"], df["q_UPP1"], df["q_UPP2"], color="red", alpha=0.4)
        ax.fill_between(df["compressed_time"], df["q_LOW1"], df["q_LOW2"], color="lightgreen", alpha=0.4)
        for t, p in entries2:
            ax.plot(t, p, marker="v", color="red", markersize=10)
        if entry3:
            t, p = entry3
            ax.plot(t, p, marker="v", color="#FF4500", markersize=10)
        if entry4:
            t, p = entry4
            ax.plot(t, p, marker="v", color="#FF4500", markersize=10)

    #if title == "5M Chart":
     #   for t, p in entries5:
      #      ax.plot(t, p, marker="v", color="red", markersize=10)


    if title == "15M Chart":
        for t, p in entries15:
            ax.plot(t, p, marker="v", color="gold", markersize=10)

    ax.set_title(title)
    tick_idx = np.linspace(0, len(df) - 1, 10, dtype=int)
    ax.set_xticks(df["compressed_time"].iloc[tick_idx])
    labels = [ts.strftime("%b %d\n%H:%M") for ts in df.index[tick_idx]]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.xaxis.grid(False)

plt.show()
