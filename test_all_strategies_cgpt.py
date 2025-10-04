#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# ==================== CONFIGURATION ====================
CAPITAL                 = 100_000
EXCHANGE_LOT_SIZE       = 50        # e.g., NIFTY options = 50
POINT_VALUE             = 1.0       # ₹ per point per contract (usually 1 for index options)
CONTRACTS               = 1         # number of contracts per trade (position size)
SLIPPAGE_POINTS         = 0.25      # round-trip slippage in points applied on entry+exit
BROKERAGE_PER_LEG       = 20.0      # ₹ per order leg (entry or exit), per order (not per contract)
TAXES_MULTIPLIER        = 1.0       # leave 1.0 if fees already inclusive; else multiply brokerage by this

TARGET_POINTS           = 10
STOPLOSS_POINTS         = 5

INPUT_CSV               = "NIFTY28OCT2524800CE_history.csv"

# ==================== CSV LOADER (ROBUST) ====================

def load_ohlc(path: str) -> pd.DataFrame:
    """
    Loads common NSE/NFO intraday schema robustly and returns canonical columns:
    ['timestamp','open','high','low','close','volume','oi'] with timestamp_index.
    Handles:
      A) timestamp, open, high, low, close, volume, oi
      B) timestamp, close, high, low, oi, open, volume
    Raises clear error otherwise.
    """
    df = pd.read_csv(path, parse_dates=[0])
    # If the file has unnamed columns (0..N), try to coerce to B
    if df.columns.tolist() == list(range(len(df.columns))):
        if len(df.columns) == 7:
            df.columns = ["timestamp","close","high","low","oi","open","volume"]
        else:
            raise ValueError(f"Unrecognized CSV schema (positional): {df.columns.tolist()}")
    else:
        df.columns = [str(c).strip().lower() for c in df.columns]

    cols = set(df.columns)
    if cols == {"timestamp","open","high","low","close","volume","oi"}:
        pass  # already canonical A
    elif cols == {"timestamp","close","high","low","oi","open","volume"}:
        pass  # already canonical B
    else:
        # try to reorder to B if lengths match
        if len(df.columns) == 7:
            df.columns = ["timestamp","close","high","low","oi","open","volume"]
        else:
            raise ValueError(f"Unrecognized CSV schema: {df.columns.tolist()}")

    df = df[["timestamp","open","high","low","close","volume","oi"]].copy()
    df.set_index("timestamp", inplace=True)

    # Basic sanity guards (catch swapped OI/close issues)
    # Close and Open should be roughly same scale across most rows
    scale = (df["close"] / df["open"]).replace([np.inf, -np.inf], np.nan).dropna()
    if not (scale.quantile(0.01) > 0.5 and scale.quantile(0.99) < 2.0):
        raise AssertionError("close/open scale looks wrong. Check CSV column mapping.")

    # Price sanity
    if not df[["open","high","low","close"]].applymap(lambda x: x >= 0).all().all():
        raise AssertionError("Negative price detected.")
    if not (df["high"] >= df["low"]).all():
        raise AssertionError("Found high < low rows; bad data?")

    return df

df = load_ohlc(INPUT_CSV)

# ==================== TECHNICAL INDICATORS ====================
df["ema_5"]  = df["close"].ewm(span=5, adjust=False).mean()
df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

df["tr"] = np.maximum(
    df["high"] - df["low"],
    np.maximum(
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    )
)
df["atr"]          = df["tr"].rolling(window=14).mean()
df["avg_volume"]   = df["volume"].rolling(window=20).mean()
df["candle_range"] = df["high"] - df["low"]
df["avg_range"]    = df["candle_range"].rolling(window=10).mean()

# ==================== STRATEGY SIGNALS ====================

def basic_signal(i, df_):
    """Original breakout logic: prior high/low break."""
    prev = df_.iloc[i - 1]
    curr = df_.iloc[i]
    if curr["high"] > prev["high"]:
        return "LONG"
    elif curr["low"] < prev["low"]:
        return "SHORT"
    return None

def filtered_breakout_signal(i, df_):
    """Breakout + volume/range filters + trend alignment."""
    if i < 20:
        return None
    prev = df_.iloc[i - 1]
    curr = df_.iloc[i]
    volume_ok = curr["volume"] > df_.iloc[i]["avg_volume"] * 0.8
    range_ok  = curr["candle_range"] > df_.iloc[i]["avg_range"] * 0.5
    if not (volume_ok and range_ok):
        return None
    if curr["high"] > prev["high"] and df_.iloc[i]["ema_5"] > df_.iloc[i]["ema_20"]:
        return "LONG"
    elif curr["low"] < prev["low"] and df_.iloc[i]["ema_5"] < df_.iloc[i]["ema_20"]:
        return "SHORT"
    return None

def trend_following_signal(i, df_):
    """Momentum in trend direction with EMA5 filter."""
    if i < 20:
        return None
    prev = df_.iloc[i - 1]
    curr = df_.iloc[i]
    if df_.iloc[i]["ema_5"] > df_.iloc[i]["ema_20"] and curr["close"] > df_.iloc[i]["ema_5"]:
        if curr["high"] > prev["high"]:
            return "LONG"
    elif df_.iloc[i]["ema_5"] < df_.iloc[i]["ema_20"] and curr["close"] < df_.iloc[i]["ema_5"]:
        if curr["low"] < prev["low"]:
            return "SHORT"
    return None

def mean_reversion_signal(i, df_):
    """Fade pullbacks within the trend."""
    if i < 20:
        return None
    curr = df_.iloc[i]
    ema5 = df_.iloc[i]["ema_5"]
    ema20 = df_.iloc[i]["ema_20"]
    distance_pct = abs(curr["close"] - ema5) / ema5 * 100 if ema5 else 0
    if distance_pct > 0.5:
        if ema5 > ema20 and curr["close"] < ema5:
            return "LONG"
        elif ema5 < ema20 and curr["close"] > ema5:
            return "SHORT"
    return None

# ==================== BACKTEST ENGINE ====================

def per_trade_fees():
    """Return entry+exit fee (₹) for 1 round trip (flat, not per contract)."""
    return 2 * BROKERAGE_PER_LEG * TAXES_MULTIPLIER

def apply_slippage(entry_price, exit_price, side):
    """
    Applies simple round-trip slippage in points relative to SIDE.
    - LONG: entry worse (higher), exit worse (lower)
    - SHORT: entry worse (lower), exit worse (higher)
    """
    if side == "LONG":
        return entry_price + SLIPPAGE_POINTS, exit_price - SLIPPAGE_POINTS
    else:
        return entry_price - SLIPPAGE_POINTS, exit_price + SLIPPAGE_POINTS

def conservative_fill_on_bar(side, entry, tp, sl, o, h, l, c):
    """
    Determine exit price on the next bar (with gaps & level hits).
    - If gap beyond TP/SL, fill at open.
    - If both levels hit within bar range (min<=level<=max), assume adverse (stop) hits first.
    - Else if only one level hit, fill at that level.
    - Else exit at close.
    Returns (exit_price, exit_reason).
    """
    bar_min, bar_max = l, h

    # 1) Gap logic: if open beyond level, fill at open
    if side == "LONG":
        if o >= tp:
            return o, "Target Gap Fill"
        if o <= sl:
            return o, "Stop Gap Fill"
    else:  # SHORT
        if o <= sl:
            return o, "Target Gap Fill"  # For SHORT, target is entry - TARGET (named 'sl' variable below? no, we pass 'tp' and 'sl' precomputed)
        if o >= tp:
            return o, "Stop Gap Fill"

    # 2) In-bar touches
    tp_hit = bar_min <= tp <= bar_max
    sl_hit = bar_min <= sl <= bar_max

    if tp_hit and sl_hit:
        # Conservative: adverse first
        if side == "LONG":
            return sl, "Stop Hit (Conservative)"
        else:
            return tp, "Stop Hit (Conservative)"  # For SHORT, adverse is 'tp' since tp>entry
    if tp_hit:
        return tp, "Target Hit"
    if sl_hit:
        return sl, "Stoploss Hit"

    # 3) Neither level hit: exit at close
    return c, "End of Candle"

def run_backtest(strategy_name, get_signal, df_):
    results = []
    in_position = False
    side = None
    entry_price = None
    entry_time = None
    capital = CAPITAL

    lot_multiplier = EXCHANGE_LOT_SIZE * POINT_VALUE * CONTRACTS
    fees_roundtrip = per_trade_fees()

    for i in range(1, len(df_) - 1):
        curr = df_.iloc[i]
        ts = curr.name

        # EXIT if in position (evaluate on next candle)
        if in_position:
            nxt = df_.iloc[i + 1]
            # Precompute levels
            if side == "LONG":
                tp = entry_price + TARGET_POINTS
                sl = entry_price - STOPLOSS_POINTS
            else:  # SHORT
                # For SHORT, "tp" is below entry; "sl" is above entry
                tp = entry_price - TARGET_POINTS
                sl = entry_price + STOPLOSS_POINTS

            exit_price, exit_reason = conservative_fill_on_bar(
                side=side,
                entry=entry_price,
                tp=tp,
                sl=sl,
                o=nxt["open"], h=nxt["high"], l=nxt["low"], c=nxt["close"]
            )

            # Apply slippage (round trip) AFTER determining raw exit
            adj_entry, adj_exit = apply_slippage(entry_price, exit_price, side)

            if side == "LONG":
                pnl_points = adj_exit - adj_entry
            else:
                pnl_points = adj_entry - adj_exit

            pnl_rupees = pnl_points * lot_multiplier
            pnl_rupees -= fees_roundtrip  # brokerage/taxes

            capital += pnl_rupees

            results.append({
                "entry_time": entry_time,
                "exit_time": nxt.name,
                "action": side,
                "entry_price": adj_entry,
                "exit_price": adj_exit,
                "pnl_points": pnl_points,
                "pnl_rupees": pnl_rupees,
                "capital": capital,
                "exit_reason": exit_reason
            })

            in_position = False
            side = None
            entry_price = None
            entry_time = None

        # ENTRY if flat: signal on current bar, enter next bar open
        if not in_position:
            signal = get_signal(i, df_)
            if signal and i < len(df_) - 1:
                nxt = df_.iloc[i + 1]
                # Entry at next open with half slippage (we add full round trip at exit function)
                raw_entry = nxt["open"]
                # We'll add the other half of slippage on exit; to keep symmetry we add half now:
                if signal == "LONG":
                    entry_price = raw_entry + (SLIPPAGE_POINTS / 2.0)
                else:
                    entry_price = raw_entry - (SLIPPAGE_POINTS / 2.0)

                in_position = True
                side = signal
                entry_time = nxt.name

    return pd.DataFrame(results)

# ==================== RUN ALL STRATEGIES ====================

strategies = {
    "basic": basic_signal,
    "filtered_breakout": filtered_breakout_signal,
    "trend_following": trend_following_signal,
    "mean_reversion": mean_reversion_signal
}

comparison_results = []

print("🚀 Testing all strategies...\n")
print("=" * 80)

for name, func in strategies.items():
    print(f"\n📊 Running: {name}")
    print("-" * 80)
    out = run_backtest(name, func, df)

    if len(out) > 0:
        out.to_csv(f"{name}_results.csv", index=False)

        total_trades = len(out)
        winning = (out["pnl_rupees"] > 0).sum()
        losing  = (out["pnl_rupees"] < 0).sum()
        win_rate = (winning / total_trades * 100) if total_trades else 0.0

        net_pnl = out["pnl_rupees"].sum()
        final_capital = CAPITAL + net_pnl
        roi = (net_pnl / CAPITAL * 100.0) if CAPITAL else 0.0

        avg_win  = out.loc[out["pnl_rupees"] > 0, "pnl_rupees"].mean() if winning else 0.0
        avg_loss = out.loc[out["pnl_rupees"] < 0, "pnl_rupees"].mean() if losing else 0.0
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

        out["cumulative_pnl"] = out["pnl_rupees"].cumsum()
        out["peak"] = out["cumulative_pnl"].cummax()
        out["drawdown"] = out["cumulative_pnl"] - out["peak"]
        max_dd = out["drawdown"].min() if len(out) else 0.0

        max_profit = out["pnl_rupees"].max() if len(out) else 0.0
        max_loss   = out["pnl_rupees"].min() if len(out) else 0.0

        comparison_results.append({
            "Strategy": name,
            "Total Trades": total_trades,
            "Win Rate (%)": round(win_rate, 2),
            "Net P&L (₹)": round(net_pnl, 2),
            "ROI (%)": round(roi, 2),
            "Final Capital (₹)": round(final_capital, 2),
            "Avg Win (₹)": round(avg_win, 2),
            "Avg Loss (₹)": round(avg_loss, 2),
            "Risk:Reward": round(rr, 2),
            "Max Drawdown (₹)": round(max_dd, 2),
            "Max Win (₹)": round(max_profit, 2),
            "Max Loss (₹)": round(max_loss, 2),
        })

        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Net P&L: ₹{net_pnl:,.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"Final Capital: ₹{final_capital:,.2f}")
        print(f"Max Drawdown: ₹{max_dd:,.2f}")

    else:
        print("⚠️  No trades generated")
        comparison_results.append({
            "Strategy": name,
            "Total Trades": 0,
            "Win Rate (%)": 0.0,
            "Net P&L (₹)": 0.0,
            "ROI (%)": 0.0,
            "Final Capital (₹)": CAPITAL,
            "Avg Win (₹)": 0.0,
            "Avg Loss (₹)": 0.0,
            "Risk:Reward": 0.0,
            "Max Drawdown (₹)": 0.0,
            "Max Win (₹)": 0.0,
            "Max Loss (₹)": 0.0,
        })

# ==================== COMPARISON TABLE ====================

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv("strategy_comparison.csv", index=False)

print("\n" + "=" * 80)
print("🏆 STRATEGY COMPARISON")
print("=" * 80)
print(comparison_df.to_string(index=False))
print("=" * 80)

# Bests
if len(comparison_df) > 0:
    best_roi = comparison_df.iloc[comparison_df["ROI (%)"].idxmax()]
    best_wr  = comparison_df.iloc[comparison_df["Win Rate (%)"].idxmax()]
    best_rr  = comparison_df.iloc[comparison_df["Risk:Reward"].idxmax()]
    print(f"\n🥇 Best ROI: {best_roi['Strategy']} ({best_roi['ROI (%)']:.2f}%)")
    print(f"🎯 Best Win Rate: {best_wr['Strategy']} ({best_wr['Win Rate (%)']:.2f}%)")
    print(f"💰 Best Risk:Reward: {best_rr['Strategy']} (1:{best_rr['Risk:Reward']:.2f})")

print("\n✅ All results saved!")
print("   - Individual CSVs: basic_results.csv, filtered_breakout_results.csv, trend_following_results.csv, mean_reversion_results.csv")
print("   - Comparison: strategy_comparison.csv")
