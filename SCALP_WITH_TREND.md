# Scalp-with-Trend (Single‑Bar Hold) — Strategy Spec

> Version: 1.0 • Author: You • Modeled by backtest script `scalp_with_trend.py`  
> Scope: Intraday, 5‑minute OHLC, **enter at next bar’s open**, **exit on that same bar** (true single‑bar hold).

---

## 1) Objective

Take quick, momentum‑aligned scalps in the direction of the short‑term trend. Entries are triggered by a **breakout/breakdown** of the previous bar’s extremes **in trend**, with **strict one‑bar holding time** to avoid overexposure and reduce overnight/long‑tail risks.

---

## 2) Data Requirements

- **Timeframe**: 5‑minute bars (or similar intraday granularity)
- **Columns (CSV)**:
  ```csv
  timestamp,close,high,low,oi,open,volume
  2025-09-01 09:15:00+05:30,412.0,413.5,383.0,62100,383.0,7725
  ```
- **Timezone**: IST in `timestamp` (e.g., `+05:30` suffix acceptable)

> Note: Columns are standardized inside the script as:  
> `['timestamp','close','high','low','oi','open','volume']`

---

## 3) Indicators & Filters

- **Trend filter**: Exponential Moving Averages on **close**  
  - `EMA_FAST = 5`  
  - `EMA_SLOW = 20`  
  - Uptrend: `EMA_FAST > EMA_SLOW`  
  - Downtrend: `EMA_FAST < EMA_SLOW`
- **ATR regime filter** (range sufficiency):  
  - True Range per bar: `max(high-low, |high-prev_close|, |low-prev_close|)`  
  - **ATR**: simple rolling mean of TR with `ATR_WINDOW = 14`  
  - Require: `ATR >= ATR_MIN_POINTS` (default **2.0** points)
- **Session windows (IST)**:  
  - Default: `[(09:20–11:00), (13:45–15:05)]`  
  - Signals must occur inside a session; entry/exit bar must also lie within a session.

---

## 4) Signal Logic

Let bar index be `i` (current), `i-1` (previous).

### LONG signal
- `high[i] > high[i-1]` **and**
- trend up at `i`: `EMA5[i] > EMA20[i]` **and**
- `ATR[i] >= ATR_MIN_POINTS` **and**
- `timestamp[i]` in a session window

### SHORT signal
- `low[i] < low[i-1]` **and**
- trend down at `i`: `EMA5[i] < EMA20[i]` **and**
- `ATR[i] >= ATR_MIN_POINTS` **and**
- `timestamp[i]` in a session window

---

## 5) Entry & Exit (Single‑Bar Hold)

- **Entry timing**: On the **next bar** (`i+1`) **open**.
- **Exit timing**: Exit on that **same bar (`i+1`)** using its **high/low/close**.  
- **Targets/Stops (points)**:
  - `TARGET_POINTS` (default **10**)
  - `STOPLOSS_POINTS` (default **2**)
- **Both‑touched rule** (conservative): If both TP & SL fall inside the exit bar range, assume **SL hits first**.
- **Trailing on single bar**: Disabled by default (`DISABLE_TRAILING_ON_SINGLE_BAR = True`).  
  - If enabled, intra‑bar **path assumption** is required (see §6).

---

## 6) Intra‑Bar Path Modeling (for optional trailing)

Since exits occur using **only one bar (i+1)** without tick‑level path, an assumption can be chosen:

- `EXIT_BAR_PATH = "color"` (default):  
  - **Green bar** → path `open → low → high → close`  
  - **Red bar** → path `open → high → low → close`
- Other modes:
  - `"bull"`: always `open → low → high → close`
  - `"bear"`: always `open → high → low → close`
  - `"worst"`: skip path modeling; rely solely on conservative both‑touched → SL‑first
- **Recommendation**: Keep `DISABLE_TRAILING_ON_SINGLE_BAR = True` unless you explicitly want modeled trailing in a one‑bar exit.

---

## 7) Costs & Position Sizing

- **Position sizing**: Fixed rupees per point, `QTY_PER_POINT` (default **150**)  
  Example: 2 lots × 75 multiplier = 150 ₹ per point.
- **Costs** (per trade, both legs):
  - **Brokerage**: `BROKERAGE_PER_TRADE` per leg (default **₹20** → ₹40 roundtrip)
  - **Slippage**: `SLIPPAGE_POINTS` per leg (default **0.10** points)  
    Roundtrip slippage ₹ = `SLIPPAGE_POINTS × QTY_PER_POINT × 2`

> **Net P&L** = `(Exit − Entry) × QTY_PER_POINT` (sign‑adjusted for SHORT) **− (Brokerage + Slippage)`

---

## 8) Risk Controls

- **Daily Loss Cap** (`DAILY_LOSS_CAP`, default **−₹1000**):  
  - After a day’s cumulative P&L ≤ cap, **no new entries** are taken that day.
- (Optional ideas not enforced by default): cooldown after loss, max trades/day, ATR‑based dynamic stops.

---

## 9) Defaults (Recommended Starting Point)

```python
INPUT_CSV = "NIFTYxx_history.csv"
STARTING_CAPITAL = 100_000
QTY_PER_POINT = 150

TARGET_POINTS = 10
STOPLOSS_POINTS = 2

ENABLE_TRAILING_TARGET = True
TRAILING_TARGET_TRIGGER = 3
TRAILING_TARGET_OFFSET = 2

ENABLE_TRAILING_STOPLOSS = True
TRAILING_SL_TRIGGER = 3
TRAILING_SL_OFFSET = 1

EXIT_BAR_PATH = "color"
DISABLE_TRAILING_ON_SINGLE_BAR = True
CONFIRM_TREND_AT_ENTRY = True

EMA_FAST = 5
EMA_SLOW = 20

ATR_WINDOW = 14
ATR_MIN_POINTS = 2.0

SESSION_WINDOWS = [(time(9,20), time(11,0)), (time(13,45), time(15,5))]

DAILY_LOSS_CAP = -1000
CONSERVATIVE_BOTH_TOUCHED_SL_FIRST = True

BROKERAGE_PER_TRADE = 20.0
SLIPPAGE_POINTS = 0.10
```

---

## 10) Backtest Mechanics

- Iterate bars `i = 1 … N-2` (need `i+1` for entry/exit).
- On signal at `i`, **reconfirm** trend at `i` if `CONFIRM_TREND_AT_ENTRY` is `True` (conservative guard).
- Ensure the **entry/exit bar (`i+1`)** is inside session (recommended).
- **Enter** at `open[i+1]` and immediately **resolve exit** on bar `i+1`:
  - Compute TP/SL from `entry_price`.
  - Use exit bar `high/low/close` and conservative rule to determine exit.
  - Subtract costs.
  - Update equity and daily P&L; enforce daily loss cap.
- **Record** trade with: entry_time, exit_time, side, prices, P&L (gross/costs/net), reason.

---

## 11) Metrics & Reporting

- **Trade list** saved to: `scalp_with_trend_results.csv`
- Console summary includes:
  - Total trades, wins/losses/breakeven, **win‑rate**
  - **Gross P&L**, **Total Costs**, **Net P&L**, **ROI**
  - **Average win/loss**, **Actual R:R**
  - **Max Drawdown** (from cumulative net P&L)
  - Exit reason distribution
- **Breakeven win‑rate** (for given TP/SL):  
  \[ SL / (TP + SL) \] × 100  
  Example with TP=10, SL=2 → **16.7%**

---

## 12) CLI Usage

```bash
python scalp_with_trend.py   --input_csv NIFTY28OCT2524800CE_history.csv   --starting_capital 100000   --qty_per_point 150   --target_points 10   --stoploss_points 2   --atr_min_points 2   --exit_bar_path color   --disable_trailing_on_single_bar true   --confirm_trend_at_entry true   --brokerage_per_trade 20   --slippage_points 0.10
```

Optional overrides:
- `--ema_fast 5 --ema_slow 20`
- `--atr_window 14`
- `--daily_loss_cap -1500`
- Custom sessions (JSON in your integration) or modify defaults in code.

---

## 13) Assumptions & Limitations

- **OHLC‑only** evaluation within the exit bar; no tick‑path simulation.  
  Conservative rule handles TP/SL conflict as **SL‑first**.
- **Single‑bar hold**: no multi‑bar trailing or management.  
- **Costs model** is simplified; exchange‑specific fees, taxes, and impact costs may vary.
- **Gap risk**: Entries at `i+1` **open** can be far from `i` close; use ATR/filters if needed.

---

## 14) Extensions (Optional)

- Add **gap filters**: skip if `|open[i+1]-close[i]| > k × ATR[i]`.
- **Dynamic SL/TP** via ATR multiples.
- **Cooldown** after loss or max trades/day.
- **Portfolio mode** across instruments with capital/risk budgeting.
- **Transaction tax pack** per broker/exchange.

---

## 15) Pseudocode

```text
for i in 1..N-2:
  if not in_session(time[i]): continue
  if ATR[i] < ATR_MIN: continue

  if long_signal(i) or short_signal(i):
    if CONFIRM_TREND_AT_ENTRY and not trend_ok(i): continue
    if not in_session(time[i+1]): continue  # recommended

    entry = open[i+1]
    (exit, reason) = resolve_on_bar(position, entry, bar[i+1])
    pnl_points = sign(position) * (exit - entry)
    gross = pnl_points * QTY_PER_POINT
    costs = brokerage*2 + slippage*QTY_PER_POINT*2
    net = gross - costs
    update_equity_and_daily(net)
    record_trade(...)
```

---

## 16) Change Log

- **v1.0**: Initial spec for single‑bar hold variant with trend/ATR filters, conservative exit, optional path modeling, and costs.
