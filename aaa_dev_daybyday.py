#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scalp-with-Trend Backtest (single-bar hold)
-------------------------------------------
• Data: 5m-style OHLC from a CSV like:
    timestamp,close,high,low,oi,open,volume
    2025-09-01 09:15:00+05:30,412.0,413.5,383.0,62100,383.0,7725
    ...

• Signal:
  LONG  when current bar breaks prev HIGH AND uptrend (EMA5 > EMA20)
  SHORT when current bar breaks prev LOW  AND downtrend (EMA5 < EMA20)
  (Entry placed at NEXT bar open, exit only on THAT next bar)

• Exit on next bar only (single-bar hold):
  - Check TP/SL using high/low of the exit bar.
  - If both TP and SL lie within the exit bar range, assume WORST CASE:
      → SL triggers first (conservative).

• Risk/Reward defaults:
  TARGET_POINTS = 6, STOPLOSS_POINTS = 4  (R:R = 1.5)
  QTY_PER_POINT = 150  (e.g., 2 lots × 75)

• Filters:
  - ATR regime: require ATR >= ATR_MIN_POINTS
  - Time windows (IST): [(09:20, 11:00), (13:45, 15:05)]
  - Daily loss cap in ₹ (stop trading for that day once breached)

Outputs:
  - Prints summary
  - Saves trades to 'scalp_with_trend_results.csv'
"""

import pandas as pd
import numpy as np
from datetime import time

# ==================== CONFIGURATION ====================

INPUT_CSV = "NIFTY25NOV2525800PE_history.csv"

# Capital tracking (not used for sizing here; fixed qty per point)
STARTING_CAPITAL = 200_000

# Position sizing: rupees per point of move (e.g., 2 lots × 75 = 150)
QTY_PER_POINT = 150

# Targets & Stops (points/rupees)
TARGET_POINTS = 10         # e.g., ₹6
STOPLOSS_POINTS = 2       # e.g., ₹4 (R:R = 1.5)

# Trailing Target settings
ENABLE_TRAILING_TARGET = True
TRAILING_TARGET_TRIGGER = 3    # points of profit to activate trailing target
TRAILING_TARGET_OFFSET = 2     # trail this many points below high (LONG) / above low (SHORT)

# Trailing Stoploss settings
ENABLE_TRAILING_STOPLOSS = True
TRAILING_SL_TRIGGER = 3        # points of profit to activate trailing SL
TRAILING_SL_OFFSET = 1         # trail this many points below high (LONG) / above low (SHORT)

# Trend filter EMAs

EMA_FAST = 5
EMA_SLOW = 20

# ATR filter (on points)
ATR_WINDOW = 14
ATR_MIN_POINTS = 2.0      # require at least this much average range

# Trading session windows (IST)
SESSION_WINDOWS = [(time(10, 20), time(11, 0)),
                   (time(11, 1), time(14, 29))]

# Daily loss cap (absolute ₹). Stop trading for the day after breaching.
DAILY_LOSS_CAP = -1000   # tweak as needed

# If True, when both TP & SL are inside the exit bar range, assume SL hits first.
CONSERVATIVE_BOTH_TOUCHED_SL_FIRST = True

# ==================== LOAD & PREP DATA ====================

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=['timestamp'])
    except Exception:
        # Fallback if header may be absent or different
        df = pd.read_csv(path, parse_dates=[0])
        df.columns = ['timestamp', 'close', 'high', 'low', 'oi', 'open', 'volume']

    # Standardize order & index
    cols = ['timestamp', 'close', 'high', 'low', 'oi', 'open', 'volume']
    df = df[cols].copy()
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    return df

df = load_data(INPUT_CSV)

# Indicators
df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

# True Range & ATR (simple rolling mean)
tr1 = df['high'] - df['low']
tr2 = (df['high'] - df['close'].shift(1)).abs()
tr3 = (df['low'] - df['close'].shift(1)).abs()
df['tr'] = np.maximum(tr1, np.maximum(tr2, tr3))
df['atr'] = df['tr'].rolling(window=ATR_WINDOW).mean()

# Helpers
def in_session(ts) -> bool:
    t = ts.time()
    for start, end in SESSION_WINDOWS:
        if start <= t <= end:
            return True
    return False

def trend_up(i: int) -> bool:
    row = df.iloc[i]
    return row['ema_fast'] > row['ema_slow']

def trend_down(i: int) -> bool:
    row = df.iloc[i]
    return row['ema_fast'] < row['ema_slow']

def atr_ok(i: int) -> bool:
    return df.iloc[i]['atr'] >= ATR_MIN_POINTS

# ==================== SIGNALS ====================

def scalp_signal(i: int) -> str | None:
    """
    Momentum-aligned breakout/breakdown:
      - LONG  if curr.high > prev.high and trend_up and ATR OK
      - SHORT if curr.low  < prev.low  and trend_down and ATR OK
    """
    if i < 1:
        return None

    prev = df.iloc[i - 1]
    curr = df.iloc[i]

    if not in_session(curr.name):
        return None

    if not atr_ok(i):
        return None

    # Long breakout with uptrend
    if (curr['high'] > prev['high']) and trend_up(i):
        return 'LONG'

    # Short breakdown with downtrend
    if (curr['low'] < prev['low']) and trend_down(i):
        return 'SHORT'

    return None

# ==================== EXIT LOGIC (SINGLE-BAR HOLD) ====================

def resolve_exit_on_bar(position: str, entry_price: float, bar: pd.Series):
    """
    Given a position and the single exit bar (next bar), decide exit price and reason.
    Conservative rule: if both TP & SL are inside the bar, assume SL triggers first.

    Supports trailing target and trailing stop-loss based on the bar's high/low.
    """
    high = bar['high']
    low = bar['low']

    if position == 'LONG':
        # Initial levels
        tp = entry_price + TARGET_POINTS
        sl = entry_price - STOPLOSS_POINTS

        # Calculate unrealized profit at the high of the bar
        profit_at_high = high - entry_price

        # Apply trailing target if enabled and trigger reached
        trailing_target_active = False
        if ENABLE_TRAILING_TARGET and profit_at_high >= TRAILING_TARGET_TRIGGER:
            trailing_tp = high - TRAILING_TARGET_OFFSET
            if trailing_tp > tp:  # Only trail up, never down
                tp = trailing_tp
                trailing_target_active = True

        # Apply trailing stoploss if enabled and trigger reached
        trailing_sl_active = False
        if ENABLE_TRAILING_STOPLOSS and profit_at_high >= TRAILING_SL_TRIGGER:
            trailing_sl_level = high - TRAILING_SL_OFFSET
            if trailing_sl_level > sl:  # Only trail up, never down
                sl = trailing_sl_level
                trailing_sl_active = True

        # Check what got hit
        hit_tp = high >= tp
        hit_sl = low <= sl

        if hit_tp and hit_sl and CONSERVATIVE_BOTH_TOUCHED_SL_FIRST:
            reason = "Stoploss Hit (Trailing)" if trailing_sl_active else "Stoploss Hit"
            return sl, reason
        elif hit_tp:
            reason = "Target Hit (Trailing)" if trailing_target_active else "Target Hit"
            return tp, reason
        elif hit_sl:
            reason = "Stoploss Hit (Trailing)" if trailing_sl_active else "Stoploss Hit"
            return sl, reason
        else:
            return float(bar['close']), "End of Candle"

    elif position == 'SHORT':
        # Initial levels
        tp = entry_price - TARGET_POINTS
        sl = entry_price + STOPLOSS_POINTS

        # Calculate unrealized profit at the low of the bar
        profit_at_low = entry_price - low

        # Apply trailing target if enabled and trigger reached
        trailing_target_active = False
        if ENABLE_TRAILING_TARGET and profit_at_low >= TRAILING_TARGET_TRIGGER:
            trailing_tp = low + TRAILING_TARGET_OFFSET
            if trailing_tp < tp:  # Only trail down, never up
                tp = trailing_tp
                trailing_target_active = True

        # Apply trailing stoploss if enabled and trigger reached
        trailing_sl_active = False
        if ENABLE_TRAILING_STOPLOSS and profit_at_low >= TRAILING_SL_TRIGGER:
            trailing_sl_level = low + TRAILING_SL_OFFSET
            if trailing_sl_level < sl:  # Only trail down, never up
                sl = trailing_sl_level
                trailing_sl_active = True

        # Check what got hit
        hit_tp = low <= tp
        hit_sl = high >= sl

        if hit_tp and hit_sl and CONSERVATIVE_BOTH_TOUCHED_SL_FIRST:
            reason = "Stoploss Hit (Trailing)" if trailing_sl_active else "Stoploss Hit"
            return sl, reason
        elif hit_tp:
            reason = "Target Hit (Trailing)" if trailing_target_active else "Target Hit"
            return tp, reason
        elif hit_sl:
            reason = "Stoploss Hit (Trailing)" if trailing_sl_active else "Stoploss Hit"
            return sl, reason
        else:
            return float(bar['close']), "End of Candle"

    else:
        raise ValueError("Unknown position type")

# ==================== BACKTEST ENGINE ====================

def run_backtest() -> pd.DataFrame:
    results = []

    in_position = False
    position = None
    entry_price = None
    entry_time = None

    equity = STARTING_CAPITAL

    # Track daily P&L to enforce daily loss cap
    daily_pnl = {}  # date -> cumulative pnl (₹)
    current_date = None
    day_hard_stopped = set()  # dates where loss cap breached

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        ts = row.name
        d = ts.date()

        # New day: reset in_position (optional), reset flags if date changes
        if current_date is None or d != current_date:
            current_date = d
            # ensure a container for the date
            daily_pnl.setdefault(d, 0.0)

        # If day already hard-stopped, skip trading entries, but still exit if holding
        day_stopped = d in day_hard_stopped

        # Exit on next bar only
        if in_position:
            next_bar = df.iloc[i + 1]
            exit_price, exit_reason = resolve_exit_on_bar(position, entry_price, next_bar)

            if position == 'LONG':
                pnl_points = exit_price - entry_price
            else:  # SHORT
                pnl_points = entry_price - exit_price

            pnl_rupees = pnl_points * QTY_PER_POINT
            equity += pnl_rupees

            # update daily pnl (counts on the exit bar date)
            ed = next_bar.name.date()
            daily_pnl.setdefault(ed, 0.0)
            daily_pnl[ed] += pnl_rupees

            # if daily cap breached, mark date as stopped
            if daily_pnl[ed] <= DAILY_LOSS_CAP:
                day_hard_stopped.add(ed)

            results.append({
                'entry_time': entry_time,
                'exit_time' : next_bar.name,
                'side'      : position,
                'entry'     : entry_price,
                'exit'      : exit_price,
                'pnl_points': pnl_points,
                'pnl_rupees': pnl_rupees,
                'equity'    : equity,
                'exit_reason': exit_reason
            })

            # flat now
            in_position = False
            position = None
            entry_price = None
            entry_time = None

        # Entry after generating/closing any prior position
        if not in_position and not day_stopped:
            sig = scalp_signal(i)
            if sig:
                # Enter at NEXT bar open
                nb = df.iloc[i + 1]
                in_position = True
                position = sig
                entry_price = float(nb['open'])
                entry_time = nb.name

    return pd.DataFrame(results)

# ==================== RUN & REPORT ====================

def main():
    print("🚀 Running Scalp-with-Trend (single-bar hold) ...\n")
    print("=" * 90)
    print(f"Target: {TARGET_POINTS} | Stop: {STOPLOSS_POINTS}  (R:R = {TARGET_POINTS/STOPLOSS_POINTS:.2f})")
    print(f"Qty per point: {QTY_PER_POINT}  | Start Capital: ₹{STARTING_CAPITAL:,}")
    print(f"ATR≥{ATR_MIN_POINTS}, EMA{EMA_FAST}/{EMA_SLOW} trend, Sessions: {[(s.strftime('%H:%M'), e.strftime('%H:%M')) for s,e in SESSION_WINDOWS]}")
    print(f"Daily loss cap: ₹{DAILY_LOSS_CAP}")
    print(f"Trailing Target: {'ON' if ENABLE_TRAILING_TARGET else 'OFF'} (Trigger: {TRAILING_TARGET_TRIGGER}, Offset: {TRAILING_TARGET_OFFSET})")
    print(f"Trailing SL: {'ON' if ENABLE_TRAILING_STOPLOSS else 'OFF'} (Trigger: {TRAILING_SL_TRIGGER}, Offset: {TRAILING_SL_OFFSET})")
    print("=" * 90)

    trades = run_backtest()

    if trades.empty:
        print("⚠️  No trades generated.")
        return

    # Save
    out_csv = "scalp_with_trend_results.csv"
    trades.to_csv(out_csv, index=False)

    # Metrics
    total = len(trades)
    wins = (trades['pnl_rupees'] > 0).sum()
    losses = (trades['pnl_rupees'] < 0).sum()
    flat = (trades['pnl_rupees'] == 0).sum()
    win_rate = (wins / total * 100) if total else 0.0

    net_pnl = trades['pnl_rupees'].sum()
    final_eq = STARTING_CAPITAL + net_pnl
    roi = (net_pnl / STARTING_CAPITAL * 100) if STARTING_CAPITAL else 0.0

    avg_win = trades.loc[trades['pnl_rupees'] > 0, 'pnl_rupees'].mean() if wins else 0.0
    avg_loss = trades.loc[trades['pnl_rupees'] < 0, 'pnl_rupees'].mean() if losses else 0.0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    trades['cum'] = trades['pnl_rupees'].cumsum()
    trades['peak'] = trades['cum'].cummax()
    trades['dd'] = trades['cum'] - trades['peak']
    max_dd = trades['dd'].min() if not trades['dd'].empty else 0.0

    reason_counts = trades['exit_reason'].value_counts()

    print("\n📋 Last 10 Trades:")
    cols = ['entry_time','side','entry','exit','pnl_rupees','exit_reason']
    print(trades[cols].tail(10))

    print("\n" + "=" * 90)
    print("📊 BACKTEST SUMMARY")
    print("=" * 90)
    print(f"Initial Capital : ₹{STARTING_CAPITAL:,.2f}")
    print(f"Final Capital   : ₹{final_eq:,.2f}")
    print(f"Net P&L         : ₹{net_pnl:,.2f}")
    print(f"ROI             : {roi:.2f}%")

    print(f"\nTotal Trades    : {total}")
    print(f"Wins            : {wins} ({win_rate:.2f}%)")
    print(f"Losses          : {losses}")
    print(f"Breakeven       : {flat}")

    print(f"\nAvg Win         : ₹{avg_win:,.2f}")
    print(f"Avg Loss        : ₹{avg_loss:,.2f}")
    print(f"Actual R:R      : {rr:.2f}")

    print(f"\nMax Drawdown    : ₹{max_dd:,.2f}")

    print("\n📈 Exit Reason Breakdown:")
    for r, c in reason_counts.items():
        print(f"  {r}: {c} ({c/total*100:.1f}%)")

    print("\n✅ Saved:", out_csv)

    # Quick sanity: required win-rate for given TP/SL
    breakeven_wr = STOPLOSS_POINTS / (TARGET_POINTS + STOPLOSS_POINTS) * 100
    print(f"\nℹ️  Math check: With TP={TARGET_POINTS} & SL={STOPLOSS_POINTS}, "
          f"breakeven win-rate ≈ {breakeven_wr:.1f}%")

if __name__ == "__main__":
    main()
