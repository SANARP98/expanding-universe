#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scalp-with-Trend Backtest (multi-bar hold until TP/SL, intraday square-off)
---------------------------------------------------------------------------
• Entry:
  LONG  when current bar breaks prev HIGH AND uptrend (EMA5 > EMA20) & ATR OK & in session
  SHORT when current bar breaks prev LOW  AND downtrend (EMA5 < EMA20) & ATR OK & in session
  Entry placed at NEXT bar OPEN.

• Exit:
  Hold the position across bars until TP or SL hit.
  If both TP & SL lie within the SAME bar range, the exit follows EXIT_BAR_PATH:
     "color": green bar → open→low→high→close, red bar → open→high→low→close
     "bull" : open→low→high→close
     "bear" : open→high→low→close
     "worst": conservative (SL first if both are inside the bar)

• Intraday only:
  ENABLE_EOD_SQUARE_OFF = True forces square-off before market close (SQUARE_OFF_TIME),
  and also on the last bar of a day (half-days/holidays safe).

• Risk/Reward defaults:
  TARGET_POINTS = 10, STOPLOSS_POINTS = 2  (R:R = 5.0)
  QTY_PER_POINT = 150  (e.g., 2 lots × 75)

• Filters:
  - ATR regime: require ATR >= ATR_MIN_POINTS
  - Time windows (IST): [(09:20, 11:00), (11:15, 15:05)]
  - Daily loss cap in ₹ (stop new entries for that day once breached)

• Costs:
  - BROKERAGE_PER_TRADE per leg (entry+exit)
  - SLIPPAGE_POINTS per leg (entry+exit)

Outputs:
  - Prints summary
  - Saves trades to 'scalp_with_trend_results.csv'
"""

import pandas as pd
import numpy as np
from datetime import time
import argparse

# ==================== CONFIGURATION ====================

INPUT_CSV = "NIFTY28OCT2524800CE_history.csv"

STARTING_CAPITAL = 100_000
QTY_PER_POINT = 150

TARGET_POINTS   = 10.0
STOPLOSS_POINTS = 2.0

# Trend filter EMAs
EMA_FAST = 5
EMA_SLOW = 20

# ATR filter
ATR_WINDOW = 14
ATR_MIN_POINTS = 2.0

# Trading session windows (IST)
SESSION_WINDOWS = [(time(9, 20), time(11, 0)),
                   (time(11, 15), time(15, 5))]

# Daily loss cap (net, per calendar day). When breached, no new entries that day.
DAILY_LOSS_CAP = -1000.0

# Intra-bar path model for deciding which level hits first if both inside the bar
# "color" | "bull" | "bear" | "worst"
EXIT_BAR_PATH = "color"

# Costs
BROKERAGE_PER_TRADE = 20.0   # per leg
SLIPPAGE_POINTS     = 0.10   # per leg

# Reconfirm trend on the signal bar before taking entry (conservative)
CONFIRM_TREND_AT_ENTRY = True

# Intraday square-off: never carry positions overnight
ENABLE_EOD_SQUARE_OFF = True
SQUARE_OFF_TIME = time(15, 25)   # exit before market close (IST)

# ==================== DATA & INDICATORS ====================

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=['timestamp'])
    except Exception:
        df = pd.read_csv(path, parse_dates=[0])
        df.columns = ['timestamp', 'close', 'high', 'low', 'oi', 'open', 'volume']
    cols = ['timestamp', 'close', 'high', 'low', 'oi', 'open', 'volume']
    df = df[cols].copy()
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    return df

df = load_data(INPUT_CSV)

# Indicators
df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

tr1 = df['high'] - df['low']
tr2 = (df['high'] - df['close'].shift(1)).abs()
tr3 = (df['low'] - df['close'].shift(1)).abs()
df['tr']  = np.maximum(tr1, np.maximum(tr2, tr3))
df['atr'] = df['tr'].rolling(window=ATR_WINDOW).mean()

# ==================== HELPERS ====================

def in_session(ts) -> bool:
    t = ts.time()
    for start, end in SESSION_WINDOWS:
        if start <= t <= end:
            return True
    return False

def trend_up(i: int) -> bool:
    r = df.iloc[i]
    return r['ema_fast'] > r['ema_slow']

def trend_down(i: int) -> bool:
    r = df.iloc[i]
    return r['ema_fast'] < r['ema_slow']

def atr_ok(i: int) -> bool:
    return df.iloc[i]['atr'] >= ATR_MIN_POINTS

def is_last_bar_of_day(i: int) -> bool:
    """True if the next bar belongs to a different calendar date (or no next bar)."""
    if i + 1 >= len(df):
        return True
    return df.index[i + 1].date() != df.index[i].date()

def past_square_off_time(ts) -> bool:
    """True if current bar time is at/after the configured cut-off."""
    return ts.time() >= SQUARE_OFF_TIME

def scalp_signal(i: int) -> str | None:
    """
    Momentum-aligned breakout/breakdown signal on bar i.
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

def bar_path_tuple(bar: pd.Series):
    """
    Return a path tuple describing intrabar sequence:
    ("open","low","high","close") or ("open","high","low","close") or ("worst",)
    """
    o = float(bar.get('open', np.nan))
    c = float(bar.get('close', np.nan))
    if EXIT_BAR_PATH == "bull":
        return ("open", "low", "high", "close")
    if EXIT_BAR_PATH == "bear":
        return ("open", "high", "low", "close")
    if EXIT_BAR_PATH == "color":
        # green bar → bull path; red bar → bear path
        if not np.isnan(o) and not np.isnan(c) and c >= o:
            return ("open", "low", "high", "close")
        else:
            return ("open", "high", "low", "close")
    return ("worst",)

def decide_exit_this_bar(position: str, entry_price: float, bar: pd.Series,
                         tp: float, sl: float):
    """
    Given a position and a bar, see if TP or SL is hit in THIS bar.
    Returns: (exited: bool, exit_price: float | None, reason: str | None)
    If neither hits, returns (False, None, None).
    """
    high = float(bar['high'])
    low  = float(bar['low'])

    hit_tp = False
    hit_sl = False
    if position == 'LONG':
        hit_tp = high >= tp
        hit_sl = low  <= sl
    else:  # SHORT
        hit_tp = low  <= tp
        hit_sl = high >= sl

    if not hit_tp and not hit_sl:
        return False, None, None

    path = bar_path_tuple(bar)

    # Conservative mode
    if path == ("worst",):
        if hit_tp and hit_sl:
            return True, sl, "Stoploss Hit"
        elif hit_tp:
            return True, tp, "Target Hit"
        else:
            return True, sl, "Stoploss Hit"

    # Path-based ordering
    if position == 'LONG':
        if path == ("open", "low", "high", "close"):
            if low <= sl:    return True, sl, "Stoploss Hit"
            if high >= tp:   return True, tp, "Target Hit"
        else:  # ("open","high","low","close")
            if high >= tp:   return True, tp, "Target Hit"
            if low  <= sl:   return True, sl, "Stoploss Hit"
    else:  # SHORT
        if path == ("open", "low", "high", "close"):
            if low <= tp:    return True, tp, "Target Hit"
            if high >= sl:   return True, sl, "Stoploss Hit"
        else:  # ("open","high","low","close")
            if high >= sl:   return True, sl, "Stoploss Hit"
            if low  <= tp:   return True, tp, "Target Hit"

    # Fallback conservative
    if hit_tp and hit_sl:
        return True, sl, "Stoploss Hit"
    elif hit_tp:
        return True, tp, "Target Hit"
    else:
        return True, sl, "Stoploss Hit"

# ==================== BACKTEST ENGINE ====================

def run_backtest() -> pd.DataFrame:
    results = []
    equity = STARTING_CAPITAL

    in_position   = False
    side          = None
    entry_price   = None
    entry_time    = None
    tp_level      = None
    sl_level      = None

    # Daily PnL tracking
    daily_pnl = {}          # date -> pnl
    loss_stopped_days = set()

    i = 1
    n = len(df)

    while i < n:
        row = df.iloc[i]
        ts  = row.name
        d   = ts.date()

        daily_pnl.setdefault(d, 0.0)
        day_stopped = d in loss_stopped_days

        # ========= EXIT if in position =========
        if in_position:
            # Try TP/SL on the current bar
            exited, exit_px, reason = decide_exit_this_bar(side, entry_price, row, tp_level, sl_level)

            if exited:
                pnl_points = (exit_px - entry_price) if side == 'LONG' else (entry_price - exit_px)

                gross_rupees   = pnl_points * QTY_PER_POINT
                slippage_rupee = SLIPPAGE_POINTS * QTY_PER_POINT * 2  # entry + exit
                fees_rupee     = 2 * BROKERAGE_PER_TRADE               # entry + exit
                costs_rupees   = slippage_rupee + fees_rupee
                pnl_rupees     = gross_rupees - costs_rupees
                equity        += pnl_rupees

                ed = row.name.date()
                daily_pnl.setdefault(ed, 0.0)
                daily_pnl[ed] += pnl_rupees
                if daily_pnl[ed] <= DAILY_LOSS_CAP:
                    loss_stopped_days.add(ed)

                results.append({
                    'entry_time'  : entry_time,
                    'exit_time'   : row.name,
                    'side'        : side,
                    'entry'       : entry_price,
                    'exit'        : exit_px,
                    'pnl_points'  : pnl_points,
                    'gross_rupees': gross_rupees,
                    'costs_rupees': costs_rupees,
                    'pnl_rupees'  : pnl_rupees,
                    'equity'      : equity,
                    'exit_reason' : reason
                })

                # Flat now
                in_position = False
                side = None
                entry_price = None
                entry_time  = None
                tp_level    = None
                sl_level    = None

                i += 1
                continue
            else:
                # No TP/SL this bar → check for end-of-day square-off
                if ENABLE_EOD_SQUARE_OFF and (is_last_bar_of_day(i) or past_square_off_time(row.name)):
                    forced_exit_px = float(row['close'])
                    pnl_points = (forced_exit_px - entry_price) if side == 'LONG' else (entry_price - forced_exit_px)

                    gross_rupees   = pnl_points * QTY_PER_POINT
                    slippage_rupee = SLIPPAGE_POINTS * QTY_PER_POINT * 2
                    fees_rupee     = 2 * BROKERAGE_PER_TRADE
                    costs_rupees   = slippage_rupee + fees_rupee
                    pnl_rupees     = gross_rupees - costs_rupees
                    equity        += pnl_rupees

                    ed = row.name.date()
                    daily_pnl.setdefault(ed, 0.0)
                    daily_pnl[ed] += pnl_rupees
                    if daily_pnl[ed] <= DAILY_LOSS_CAP:
                        loss_stopped_days.add(ed)

                    results.append({
                        'entry_time'  : entry_time,
                        'exit_time'   : row.name,
                        'side'        : side,
                        'entry'       : entry_price,
                        'exit'        : forced_exit_px,
                        'pnl_points'  : pnl_points,
                        'gross_rupees': gross_rupees,
                        'costs_rupees': costs_rupees,
                        'pnl_rupees'  : pnl_rupees,
                        'equity'      : equity,
                        'exit_reason' : 'Square-off EOD'
                    })

                    # Flat now
                    in_position = False
                    side = None
                    entry_price = None
                    entry_time  = None
                    tp_level    = None
                    sl_level    = None

                    i += 1
                    continue

                # Still holding, move to next bar
                i += 1
                continue

        # ========= ENTRY if flat and day not stopped =========
        if not in_position and not day_stopped:
            sig = scalp_signal(i)
            if sig:
                if CONFIRM_TREND_AT_ENTRY:
                    if sig == 'LONG' and not trend_up(i):
                        i += 1; continue
                    if sig == 'SHORT' and not trend_down(i):
                        i += 1; continue

                # Enter at NEXT bar open
                if i + 1 < n:
                    nb = df.iloc[i + 1]
                    in_position = True
                    side        = sig
                    entry_price = float(nb['open'])
                    entry_time  = nb.name

                    if side == 'LONG':
                        tp_level = entry_price + TARGET_POINTS
                        sl_level = entry_price - STOPLOSS_POINTS
                    else:
                        tp_level = entry_price - TARGET_POINTS
                        sl_level = entry_price + STOPLOSS_POINTS

                    # Start evaluating from the bar AFTER the entry bar
                    i += 2
                    continue

        # If no entry/exit, just move forward
        i += 1

    return pd.DataFrame(results)

# ==================== REPORT ====================

def main(config=None):
    # Optional runtime overrides
    if config:
        global INPUT_CSV, STARTING_CAPITAL, QTY_PER_POINT
        global TARGET_POINTS, STOPLOSS_POINTS, EMA_FAST, EMA_SLOW
        global ATR_WINDOW, ATR_MIN_POINTS, SESSION_WINDOWS, DAILY_LOSS_CAP
        global EXIT_BAR_PATH, BROKERAGE_PER_TRADE, SLIPPAGE_POINTS, CONFIRM_TREND_AT_ENTRY, df
        global ENABLE_EOD_SQUARE_OFF, SQUARE_OFF_TIME

        INPUT_CSV             = config.get('input_csv', INPUT_CSV)
        STARTING_CAPITAL      = config.get('starting_capital', STARTING_CAPITAL)
        QTY_PER_POINT         = config.get('qty_per_point', QTY_PER_POINT)
        TARGET_POINTS         = config.get('target_points', TARGET_POINTS)
        STOPLOSS_POINTS       = config.get('stoploss_points', STOPLOSS_POINTS)
        EMA_FAST              = config.get('ema_fast', EMA_FAST)
        EMA_SLOW              = config.get('ema_slow', EMA_SLOW)
        ATR_WINDOW            = config.get('atr_window', ATR_WINDOW)
        ATR_MIN_POINTS        = config.get('atr_min_points', ATR_MIN_POINTS)
        DAILY_LOSS_CAP        = config.get('daily_loss_cap', DAILY_LOSS_CAP)
        EXIT_BAR_PATH         = config.get('exit_bar_path', EXIT_BAR_PATH)
        BROKERAGE_PER_TRADE   = config.get('brokerage_per_trade', BROKERAGE_PER_TRADE)
        SLIPPAGE_POINTS       = config.get('slippage_points', SLIPPAGE_POINTS)
        CONFIRM_TREND_AT_ENTRY= config.get('confirm_trend_at_entry', CONFIRM_TREND_AT_ENTRY)
        ENABLE_EOD_SQUARE_OFF = config.get('enable_eod_square_off', ENABLE_EOD_SQUARE_OFF)

        if 'square_off_time' in config:
            hh, mm = map(int, str(config['square_off_time']).split(':'))
            SQUARE_OFF_TIME = time(hh, mm)

        if 'session_windows' in config:
            SESSION_WINDOWS.clear()
            for sw in config['session_windows']:
                h1, m1 = map(int, sw['start'].split(':'))
                h2, m2 = map(int, sw['end'].split(':'))
                SESSION_WINDOWS.append((time(h1, m1), time(h2, m2)))

        # Reload data (if CSV changed) and recompute indicators
        df = load_data(INPUT_CSV)
        df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        df['tr']  = np.maximum(tr1, np.maximum(tr2, tr3))
        df['atr'] = df['tr'].rolling(window=ATR_WINDOW).mean()

    print("🚀 Running Scalp-with-Trend (multi-bar hold; intraday square-off) ...\n")
    print("=" * 96)
    print(f"TP={TARGET_POINTS} | SL={STOPLOSS_POINTS} (R:R={TARGET_POINTS/STOPLOSS_POINTS:.2f}) | Qty/pt={QTY_PER_POINT}")
    print(f"ATR≥{ATR_MIN_POINTS}, EMA{EMA_FAST}/{EMA_SLOW}, Sessions={[(s.strftime('%H:%M'), e.strftime('%H:%M')) for s,e in SESSION_WINDOWS]}")
    print(f"Exit bar path: {EXIT_BAR_PATH} | Confirm trend at entry: {CONFIRM_TREND_AT_ENTRY}")
    print(f"Costs -> Brokerage/leg: ₹{BROKERAGE_PER_TRADE}, Slippage/leg: {SLIPPAGE_POINTS} pts")
    print(f"EOD Square-off: {ENABLE_EOD_SQUARE_OFF} at {SQUARE_OFF_TIME.strftime('%H:%M')} | Daily loss cap: ₹{DAILY_LOSS_CAP}")
    print("=" * 96)

    trades = run_backtest()

    if trades.empty:
        print("⚠️  No trades generated.")
        return

    out_csv = "scalp_with_trend_results.csv"
    trades.to_csv(out_csv, index=False)

    total   = len(trades)
    wins    = (trades['pnl_rupees'] > 0).sum()
    losses  = (trades['pnl_rupees'] < 0).sum()
    flats   = (trades['pnl_rupees'] == 0).sum()
    winrate = wins / total * 100 if total else 0.0

    total_gross = trades['gross_rupees'].sum()
    total_costs = trades['costs_rupees'].sum()
    net_pnl     = trades['pnl_rupees'].sum()
    final_eq    = STARTING_CAPITAL + net_pnl
    roi         = net_pnl / STARTING_CAPITAL * 100 if STARTING_CAPITAL else 0.0

    trades['cum']  = trades['pnl_rupees'].cumsum()
    trades['peak'] = trades['cum'].cummax()
    trades['dd']   = trades['cum'] - trades['peak']
    max_dd         = trades['dd'].min()

    avg_win  = trades.loc[trades['pnl_rupees'] > 0, 'pnl_rupees'].mean() if wins else 0.0
    avg_loss = trades.loc[trades['pnl_rupees'] < 0, 'pnl_rupees'].mean() if losses else 0.0
    rr = abs(avg_win / avg_loss) if avg_loss else 0.0

    reasons = trades['exit_reason'].value_counts()

    print("\n📋 Last 10 Trades:")
    cols = ['entry_time','side','entry','exit','gross_rupees','costs_rupees','pnl_rupees','exit_reason']
    print(trades[cols].tail(10))

    print("\n" + "=" * 96)
    print("📊 BACKTEST SUMMARY")
    print("=" * 96)
    print(f"Initial Capital : ₹{STARTING_CAPITAL:,.2f}")
    print(f"Final Capital   : ₹{final_eq:,.2f}")
    print(f"Gross P&L       : ₹{total_gross:,.2f}")
    print(f"Total Costs     : ₹{total_costs:,.2f}  (avg/trade: ₹{(total_costs/total):.2f})")
    print(f"Net P&L         : ₹{net_pnl:,.2f}")
    print(f"ROI             : {roi:.2f}%")

    print(f"\nTotal Trades    : {total}")
    print(f"Wins            : {wins} ({winrate:.2f}%)")
    print(f"Losses          : {losses}")
    print(f"Breakeven       : {flats}")

    print(f"\nAvg Win         : ₹{avg_win:,.2f}")
    print(f"Avg Loss        : ₹{avg_loss:,.2f}")
    print(f"Actual R:R      : {rr:.2f}")
    print(f"Max Drawdown    : ₹{max_dd:,.2f}")

    print("\n📈 Exit Reason Breakdown:")
    for r, c in reasons.items():
        print(f"  {r}: {c} ({c/total*100:.1f}%)")

    print("\n✅ Saved:", out_csv)

    be_wr = STOPLOSS_POINTS / (TARGET_POINTS + STOPLOSS_POINTS) * 100
    print(f"\nℹ️  Math: With TP={TARGET_POINTS}, SL={STOPLOSS_POINTS}, breakeven win-rate ≈ {be_wr:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scalp-with-Trend Backtest (multi-bar hold, intraday square-off)')
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--starting_capital', type=float)
    parser.add_argument('--qty_per_point', type=int)
    parser.add_argument('--target_points', type=float)
    parser.add_argument('--stoploss_points', type=float)
    parser.add_argument('--ema_fast', type=int)
    parser.add_argument('--ema_slow', type=int)
    parser.add_argument('--atr_window', type=int)
    parser.add_argument('--atr_min_points', type=float)
    parser.add_argument('--daily_loss_cap', type=float)
    parser.add_argument('--exit_bar_path', type=str, choices=['color','bull','bear','worst'])
    parser.add_argument('--brokerage_per_trade', type=float)
    parser.add_argument('--slippage_points', type=float)
    parser.add_argument('--confirm_trend_at_entry', type=lambda x: x.lower()=='true')
    parser.add_argument('--enable_eod_square_off', type=lambda x: x.lower()=='true')
    parser.add_argument('--square_off_time', type=str, help='HH:MM (IST)')

    args = parser.parse_args()
    cfg = {k: v for k, v in vars(args).items() if v is not None}
    if 'square_off_time' in cfg:
        # will be parsed in main()
        pass
    main(cfg if cfg else None)
