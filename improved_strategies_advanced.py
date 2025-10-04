#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import time

# ==================== CONFIGURATION ====================
CAPITAL                 = 100_000
EXCHANGE_LOT_SIZE       = 50        # NIFTY options lot size
POINT_VALUE             = 1.0
CONTRACTS               = 1
SLIPPAGE_POINTS         = 0.25
BROKERAGE_PER_LEG       = 20.0
TAXES_MULTIPLIER        = 1.0

TARGET_POINTS           = 10
STOPLOSS_POINTS         = 5

INPUT_CSV               = "NIFTY28OCT2524800CE_history.csv"

# ==================== CSV LOADER ====================

def load_ohlc(path: str) -> pd.DataFrame:
    """Load and validate OHLC data"""
    df = pd.read_csv(path, parse_dates=[0])
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Reorder to standard format
    df = df[["timestamp","open","high","low","close","volume","oi"]].copy()
    df.set_index("timestamp", inplace=True)

    return df

df = load_ohlc(INPUT_CSV)

# ==================== TECHNICAL INDICATORS ====================
df["ema_5"]  = df["close"].ewm(span=5, adjust=False).mean()
df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

df["tr"] = np.maximum(
    df["high"] - df["low"],
    np.maximum(
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    )
)
df["atr"] = df["tr"].rolling(window=14).mean()
df["avg_volume"] = df["volume"].rolling(window=20).mean()
df["candle_range"] = df["high"] - df["low"]
df["avg_range"] = df["candle_range"].rolling(window=10).mean()

# Add time-based features
df["hour"] = df.index.hour
df["minute"] = df.index.minute
df["time_of_day"] = df["hour"] + df["minute"] / 60.0

# Calculate opening range (first 30 minutes of each day)
df["date"] = df.index.date
df["session_start"] = df.groupby("date").cumcount() == 0
df["is_first_30min"] = (df["hour"] == 9) & (df["minute"] < 45)

# Rolling high/low for opening range
def calculate_opening_range(group):
    first_30 = group[group["is_first_30min"]]
    if len(first_30) > 0:
        or_high = first_30["high"].max()
        or_low = first_30["low"].min()
        group["or_high"] = or_high
        group["or_low"] = or_low
    return group

df = df.groupby("date", group_keys=False).apply(calculate_opening_range)
df["or_high"] = df["or_high"].fillna(method="ffill")
df["or_low"] = df["or_low"].fillna(method="ffill")

# ==================== ADVANCED STRATEGY SIGNALS ====================

def opening_range_breakout_signal(i, df_):
    """
    Strategy 1: Opening Range Breakout
    - Wait for first 30 min to establish range
    - Trade breakouts with volume confirmation
    - Only between 9:45 AM - 2:00 PM
    """
    if i < 20:
        return None

    curr = df_.iloc[i]

    # Time filter: After opening range, before last hour
    if curr["time_of_day"] < 9.75 or curr["time_of_day"] > 14.0:
        return None

    # Volume confirmation
    if curr["volume"] < df_.iloc[i]["avg_volume"] * 1.2:
        return None

    # Breakout logic
    if curr["high"] > df_.iloc[i]["or_high"] and curr["close"] > df_.iloc[i]["or_high"]:
        return "LONG"
    elif curr["low"] < df_.iloc[i]["or_low"] and curr["close"] < df_.iloc[i]["or_low"]:
        return "SHORT"

    return None


def volume_spike_momentum_signal(i, df_):
    """
    Strategy 2: Volume Spike Momentum
    - Enter only on 2x+ volume spikes
    - Must have strong momentum (large candle)
    - Trend alignment with EMA
    """
    if i < 20:
        return None

    curr = df_.iloc[i]
    prev = df_.iloc[i - 1]

    # Massive volume spike
    if curr["volume"] < df_.iloc[i]["avg_volume"] * 2.0:
        return None

    # Strong momentum candle
    if curr["candle_range"] < df_.iloc[i]["avg_range"] * 1.5:
        return None

    # Trend alignment
    ema5 = df_.iloc[i]["ema_5"]
    ema20 = df_.iloc[i]["ema_20"]

    # Bullish: price above EMA, strong up candle
    if curr["close"] > ema5 > ema20 and curr["close"] > curr["open"]:
        if curr["high"] > prev["high"]:
            return "LONG"

    # Bearish: price below EMA, strong down candle
    elif curr["close"] < ema5 < ema20 and curr["close"] < curr["open"]:
        if curr["low"] < prev["low"]:
            return "SHORT"

    return None


def atr_based_breakout_signal(i, df_):
    """
    Strategy 3: ATR-Based Dynamic Strategy
    - Uses ATR for adaptive stop/target (in backtest engine)
    - Only trades in trending markets
    - Filters out low volatility periods
    """
    if i < 50:
        return None

    curr = df_.iloc[i]
    prev = df_.iloc[i - 1]

    # Skip low volatility periods
    if df_.iloc[i]["atr"] < df_.iloc[i]["avg_range"] * 0.8:
        return None

    # Strong trend required
    ema5 = df_.iloc[i]["ema_5"]
    ema20 = df_.iloc[i]["ema_20"]
    ema50 = df_.iloc[i]["ema_50"]

    # Uptrend: all EMAs aligned
    if ema5 > ema20 > ema50 and curr["close"] > ema5:
        if curr["high"] > prev["high"]:
            return "LONG"

    # Downtrend: all EMAs aligned
    elif ema5 < ema20 < ema50 and curr["close"] < ema5:
        if curr["low"] < prev["low"]:
            return "SHORT"

    return None


def time_of_day_filtered_signal(i, df_):
    """
    Strategy 4: Time-of-Day Filter
    - Only trades during high-probability hours
    - Avoids: First 15 min (whipsaw), lunch (11:30-12:30), last 30 min
    - Focus: 9:30-11:15 and 13:00-14:45
    """
    if i < 20:
        return None

    curr = df_.iloc[i]
    prev = df_.iloc[i - 1]

    # Time windows: Morning (9.5-11.25) and Afternoon (13.0-14.75)
    tod = curr["time_of_day"]
    morning_session = 9.5 <= tod <= 11.25
    afternoon_session = 13.0 <= tod <= 14.75

    if not (morning_session or afternoon_session):
        return None

    # Volume and range filters
    if curr["volume"] < df_.iloc[i]["avg_volume"] * 0.9:
        return None

    if curr["candle_range"] < df_.iloc[i]["avg_range"] * 0.7:
        return None

    # Trend breakout
    ema5 = df_.iloc[i]["ema_5"]
    ema20 = df_.iloc[i]["ema_20"]

    if curr["high"] > prev["high"] and ema5 > ema20:
        return "LONG"
    elif curr["low"] < prev["low"] and ema5 < ema20:
        return "SHORT"

    return None


def consolidation_breakout_signal(i, df_):
    """
    Strategy 5: Consolidation Breakout
    - Waits for tight consolidation (low ATR)
    - Then trades explosive breakouts
    - Quality over quantity
    """
    if i < 30:
        return None

    curr = df_.iloc[i]

    # Look back at last 5 candles for consolidation
    last_5 = df_.iloc[i-5:i]
    consolidation_range = last_5["high"].max() - last_5["low"].min()
    avg_atr = df_.iloc[i]["atr"]

    # Consolidation: range < 1.5 * ATR
    is_consolidating = consolidation_range < avg_atr * 1.5

    if not is_consolidating:
        return None

    # Breakout with volume
    if curr["volume"] < df_.iloc[i]["avg_volume"] * 1.5:
        return None

    # Strong breakout candle
    if curr["candle_range"] < avg_atr * 1.2:
        return None

    # Direction
    cons_high = last_5["high"].max()
    cons_low = last_5["low"].min()

    if curr["close"] > cons_high and curr["close"] > curr["open"]:
        return "LONG"
    elif curr["close"] < cons_low and curr["close"] < curr["open"]:
        return "SHORT"

    return None


def multi_timeframe_trend_signal(i, df_):
    """
    Strategy 6: Multi-Timeframe Trend
    - Multiple EMA confirmation (5, 20, 50)
    - Price action confirmation
    - Higher timeframe trend alignment
    """
    if i < 50:
        return None

    curr = df_.iloc[i]
    prev = df_.iloc[i - 1]

    ema5 = df_.iloc[i]["ema_5"]
    ema20 = df_.iloc[i]["ema_20"]
    ema50 = df_.iloc[i]["ema_50"]

    # Check trend strength: distance between EMAs
    ema_spread_up = (ema5 - ema50) / ema50 * 100 if ema50 > 0 else 0
    ema_spread_down = (ema50 - ema5) / ema50 * 100 if ema50 > 0 else 0

    # Volume confirmation
    if curr["volume"] < df_.iloc[i]["avg_volume"] * 1.0:
        return None

    # Strong uptrend: EMAs aligned + good spread + price above all
    if ema5 > ema20 > ema50 and ema_spread_up > 0.5:
        if curr["close"] > ema5 and curr["high"] > prev["high"]:
            # Additional confirmation: bullish candle
            if curr["close"] > curr["open"]:
                return "LONG"

    # Strong downtrend: EMAs aligned + good spread + price below all
    elif ema5 < ema20 < ema50 and ema_spread_down > 0.5:
        if curr["close"] < ema5 and curr["low"] < prev["low"]:
            # Additional confirmation: bearish candle
            if curr["close"] < curr["open"]:
                return "SHORT"

    return None


# ==================== BACKTEST ENGINE ====================

def per_trade_fees():
    return 2 * BROKERAGE_PER_LEG * TAXES_MULTIPLIER

def apply_slippage(entry_price, exit_price, side):
    if side == "LONG":
        return entry_price + SLIPPAGE_POINTS, exit_price - SLIPPAGE_POINTS
    else:
        return entry_price - SLIPPAGE_POINTS, exit_price + SLIPPAGE_POINTS

def conservative_fill_on_bar(side, entry, tp, sl, o, h, l, c):
    """Determine exit price on next bar with conservative assumptions"""
    bar_min, bar_max = l, h

    # Gap fills
    if side == "LONG":
        if o >= tp:
            return o, "Target Gap Fill"
        if o <= sl:
            return o, "Stop Gap Fill"
    else:
        if o <= tp:
            return o, "Target Gap Fill"
        if o >= sl:
            return o, "Stop Gap Fill"

    # In-bar touches
    tp_hit = bar_min <= tp <= bar_max
    sl_hit = bar_min <= sl <= bar_max

    if tp_hit and sl_hit:
        # Conservative: assume stop hits first
        return sl, "Stop Hit (Conservative)"
    if tp_hit:
        return tp, "Target Hit"
    if sl_hit:
        return sl, "Stoploss Hit"

    return c, "End of Candle"

def run_backtest(strategy_name, get_signal, df_, use_atr_stops=False):
    """Run backtest with optional ATR-based stops"""
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

        # EXIT if in position
        if in_position:
            nxt = df_.iloc[i + 1]

            # Dynamic stops based on ATR
            if use_atr_stops and "atr" in df_.columns:
                atr_value = df_.iloc[i]["atr"]
                target_mult = 1.5  # 1.5x ATR target
                stop_mult = 1.0    # 1x ATR stop

                if side == "LONG":
                    tp = entry_price + (atr_value * target_mult)
                    sl = entry_price - (atr_value * stop_mult)
                else:
                    tp = entry_price - (atr_value * target_mult)
                    sl = entry_price + (atr_value * stop_mult)
            else:
                # Fixed stops
                if side == "LONG":
                    tp = entry_price + TARGET_POINTS
                    sl = entry_price - STOPLOSS_POINTS
                else:
                    tp = entry_price - TARGET_POINTS
                    sl = entry_price + STOPLOSS_POINTS

            exit_price, exit_reason = conservative_fill_on_bar(
                side=side, entry=entry_price, tp=tp, sl=sl,
                o=nxt["open"], h=nxt["high"], l=nxt["low"], c=nxt["close"]
            )

            adj_entry, adj_exit = apply_slippage(entry_price, exit_price, side)

            if side == "LONG":
                pnl_points = adj_exit - adj_entry
            else:
                pnl_points = adj_entry - adj_exit

            pnl_rupees = pnl_points * lot_multiplier
            pnl_rupees -= fees_roundtrip

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

        # ENTRY if flat
        if not in_position:
            signal = get_signal(i, df_)
            if signal and i < len(df_) - 1:
                nxt = df_.iloc[i + 1]
                raw_entry = nxt["open"]

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
    "opening_range_breakout": (opening_range_breakout_signal, False),
    "volume_spike_momentum": (volume_spike_momentum_signal, False),
    "atr_based_breakout": (atr_based_breakout_signal, True),  # Uses ATR stops
    "time_of_day_filtered": (time_of_day_filtered_signal, False),
    "consolidation_breakout": (consolidation_breakout_signal, False),
    "multi_timeframe_trend": (multi_timeframe_trend_signal, False),
}

comparison_results = []

print("🚀 Testing ADVANCED strategies...\n")
print("=" * 80)

for name, (func, use_atr) in strategies.items():
    print(f"\n📊 Running: {name}")
    print("-" * 80)
    out = run_backtest(name, func, df, use_atr_stops=use_atr)

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
comparison_df.to_csv("advanced_strategy_comparison.csv", index=False)

print("\n" + "=" * 80)
print("🏆 ADVANCED STRATEGY COMPARISON")
print("=" * 80)
print(comparison_df.to_string(index=False))
print("=" * 80)

if len(comparison_df) > 0 and comparison_df["Total Trades"].sum() > 0:
    # Only consider strategies with positive ROI
    profitable = comparison_df[comparison_df["ROI (%)"] > 0]

    if len(profitable) > 0:
        best_roi = profitable.iloc[profitable["ROI (%)"].idxmax()]
        best_wr  = profitable.iloc[profitable["Win Rate (%)"].idxmax()]
        best_rr  = profitable.iloc[profitable["Risk:Reward"].idxmax()]

        print(f"\n🥇 Best ROI: {best_roi['Strategy']} ({best_roi['ROI (%)']:.2f}%)")
        print(f"🎯 Best Win Rate: {best_wr['Strategy']} ({best_wr['Win Rate (%)']:.2f}%)")
        print(f"💰 Best Risk:Reward: {best_rr['Strategy']} (1:{best_rr['Risk:Reward']:.2f})")
    else:
        print("\n⚠️  No profitable strategies found")

print("\n✅ All advanced strategy results saved!")
print("   - Individual CSVs: opening_range_breakout_results.csv, volume_spike_momentum_results.csv, etc.")
print("   - Comparison: advanced_strategy_comparison.csv")
