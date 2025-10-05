# Scalp-with-Trend — Multi‑Bar Hold (Intraday Square‑Off) — Strategy Spec

> Version: 2.0 (Multi‑Bar Hold) • Author: You • Modeled by `scalp_with_trend_multibar.py`  
> Scope: Intraday, 5‑minute OHLC. **Enter at next bar’s open**; **hold across bars until TP/SL**; **force square‑off before market close**.

---

## 1) Objective

Capture short, momentum‑aligned intraday swings with tight risk. Entries are triggered by a breakout/breakdown **in the direction of the EMA trend** and only when the **ATR regime** indicates sufficient range. Unlike the single‑bar variant, the trade is **held across multiple bars** until **take profit (TP)** or **stop loss (SL)** is hit, or **square‑off** occurs near the end of the session.


---

## 2) Data Requirements

- **Timeframe**: 5‑minute bars (or similar intraday bar size).  
- **CSV Columns (exact names expected / standardized):**
  ```csv
  timestamp,close,high,low,oi,open,volume
  2025-09-01 09:15:00+05:30,412.0,413.5,383.0,62100,383.0,7725
  ```
- **Timezone**: Timestamps in IST (offset `+05:30` acceptable).  
- **Index**: Script reindexes to `timestamp` (datetime), sorted ascending.


---

## 3) Indicators & Filters

### Trend
- **EMAs** on **close**:
  - `EMA_FAST = 5`
  - `EMA_SLOW = 20`
- **Uptrend**: `EMA5 > EMA20`  
- **Downtrend**: `EMA5 < EMA20`

### ATR Regime
- **True Range (TR)** per bar:  
  `TR = max(high - low, |high - prev_close|, |low - prev_close|)`
- **ATR**: simple moving average of TR with `ATR_WINDOW = 14`.
- **Requirement**: `ATR >= ATR_MIN_POINTS` (default **2.0** points).

### Trading Sessions (IST)
- Default session windows (inclusive):  
  - `(09:20, 11:00)` and `(11:15, 15:05)`  
- Signals must occur inside a session window. (Entry is placed on the next bar’s open.)


---

## 4) Signal Logic (on bar *i*)

Let `i` be the **current** bar, `i-1` the **previous** bar.

**LONG signal if**:
1) `high[i] > high[i-1]` (breakout),  
2) `EMA5[i] > EMA20[i]` (uptrend),  
3) `ATR[i] >= ATR_MIN_POINTS`,  
4) `timestamp[i]` is **within a session**.

**SHORT signal if**:
1) `low[i] < low[i-1]` (breakdown),  
2) `EMA5[i] < EMA20[i]` (downtrend),  
3) `ATR[i] >= ATR_MIN_POINTS`,  
4) `timestamp[i]` is **within a session**.

> **Optional conservative check** (enabled by default): `CONFIRM_TREND_AT_ENTRY = True` re‑checks the trend on the signal bar before taking the trade.


---

## 5) Entry Timing

- When a signal is detected on bar `i`, **enter at the next bar’s open** (`i+1.open`).  
- **Price levels at entry**:  
  - For **LONG**:  
    `TP = entry_price + TARGET_POINTS`  
    `SL = entry_price - STOPLOSS_POINTS`  
  - For **SHORT**:  
    `TP = entry_price - TARGET_POINTS`  
    `SL = entry_price + STOPLOSS_POINTS`


---

## 6) Exit Logic (Multi‑Bar Hold)

After entry, the engine evaluates **each subsequent bar** (`j = i+1, i+2, …`) until the position closes. On each bar:

1) **Check TP/SL hits using that bar’s high/low:**
   - LONG: `hit_tp = high[j] >= TP`, `hit_sl = low[j] <= SL`  
   - SHORT: `hit_tp = low[j] <= TP`, `hit_sl = high[j] >= SL`

2) **If neither hit**: continue to next bar unless **intraday square‑off** rules apply (see §7).

3) **If both hit inside the same bar**: resolve which one hits first using **Intra‑Bar Path** (§6.1).

4) **Exit price** equals the exact TP or SL level that is deemed to have hit first per the path rule.

### 6.1 Intra‑Bar Path (TP vs SL priority within a bar)

Because OHLC bars lack tick‑level sequencing, choose an assumption via `EXIT_BAR_PATH`:

- `"color"` (**default**)  
  - **Green bar** (`close >= open`): path `open → low → high → close`  
  - **Red bar** (`close < open`):  path `open → high → low → close`
- `"bull"`: always `open → low → high → close`
- `"bear"`: always `open → high → low → close`
- `"worst"`: conservative fallback (if both TP and SL lie inside the bar, **SL is assumed to hit first**).

This path determines **which level is considered to have been reached first** when both are within the bar’s range.


---

## 7) Intraday Only — Square‑Off

- `ENABLE_EOD_SQUARE_OFF = True` forces **no overnight positions**.  
- `SQUARE_OFF_TIME = 15:25` (default IST). When a held position is evaluated on/after the cut‑off time (or on the last bar of the day), it is exited **at that bar’s close** with reason `"Square-off EOD"`.

> This also safely handles short sessions/half‑days: the engine checks the **last bar of each date** and squares off if still in position.


---

## 8) Risk/Reward & Costs

### Fixed Targets & Stops
- Defaults: `TARGET_POINTS = 10.0`, `STOPLOSS_POINTS = 2.0` ⇒ **R:R = 5.0**.

### Position Sizing
- Fixed rupees per point: `QTY_PER_POINT = 150` (e.g., 2 lots × 75 multiplier).

### Costs (per roundtrip)
- **Brokerage**: `BROKERAGE_PER_TRADE` **per leg** (entry and exit). Default `₹20` per leg.
- **Slippage**: `SLIPPAGE_POINTS` **per leg**. Default `0.10` points per leg.  
- **Total costs (₹)** per trade: `2 × BROKERAGE_PER_TRADE + 2 × SLIPPAGE_POINTS × QTY_PER_POINT`.

### P&L
- **Gross (₹)**: `(exit − entry) × QTY_PER_POINT` (sign‑adjusted for shorts).  
- **Net (₹)**: `gross − total_costs`.  
- Costs are applied **when the trade closes** (TP/SL/EOD).


---

## 9) Daily Loss Cap

- `DAILY_LOSS_CAP` (default `−₹1000`) — tracked on a **per‑calendar‑day** basis using realized net P&L.  
- Once a day’s cumulative P&L ≤ cap, **no new entries** are allowed on that day (existing positions can still be squared off or hit TP/SL).


---

## 10) Engine Mechanics & State

### State Variables
- `in_position` (bool), `side` (`LONG`/`SHORT`), `entry_price`, `entry_time`  
- `tp_level`, `sl_level`  
- `equity`, `daily_pnl[date]`, `loss_stopped_days`

### Control Flow (high level)
1) Iterate bars with an index `i` (while‑loop).  
2) **If flat** and **day not stopped**: compute `scalp_signal(i)`.  
   - If signal: (optionally) `CONFIRM_TREND_AT_ENTRY`; then **enter at `open[i+1]`** and set `TP/SL`.  
   - Jump to evaluate from bar `i+2` onward (next bar after entry bar).  
3) **If in position**: on each bar, call `decide_exit_this_bar()` with `EXIT_BAR_PATH`:
   - If TP/SL hits: **exit**, book P&L, record trade.  
   - Else, if EOD rule triggers: **square‑off at close**, record trade.  
   - Else: **continue** to next bar.
4) Track **daily P&L** and **equity** after each exit; enforce daily loss cap for new entries.

### Pseudocode
```text
while i < N:
  bar = df[i]
  d = date(bar)

  if in_position:
    hit, px, reason = decide_exit_this_bar(side, entry_price, bar, TP, SL)
    if hit:
      book_pnl_and_record(px, reason)
      flat()
      i += 1; continue
    if ENABLE_EOD_SQUARE_OFF and (is_last_bar_of_day(i) or past_square_off_time(time[i])):
      book_pnl_and_record(close[i], "Square-off EOD")
      flat()
      i += 1; continue
    i += 1; continue

  # flat
  if d not loss_stopped_days:
    sig = scalp_signal(i)
    if sig:
      if CONFIRM_TREND_AT_ENTRY and not trend_ok(i): i += 1; continue
      enter_at_open(i+1); set_TP_SL()
      i += 2; continue

  i += 1
```


---

## 11) Assumptions & Limitations

- **OHLC path ambiguity** is resolved by the chosen `EXIT_BAR_PATH`; there is no tick data.  
- **No trailing** logic in this variant; exits are strictly TP/SL or EOD square‑off.  
- **Gap risk** at entry (open can gap beyond/near levels). Consider adding gap filters if needed.  
- **Costs model** is simplified; actual brokerage/taxes/impact may differ by broker & instrument.  
- **Session enforcement**: Signals are only taken when the **signal bar** is in‑session; ensure your data’s session boundaries match your venue.


---

## 12) Defaults (Quick Reference)

```python
INPUT_CSV = "NIFTY28OCT2524800CE_history.csv"

STARTING_CAPITAL = 100_000
QTY_PER_POINT    = 150

TARGET_POINTS    = 10.0
STOPLOSS_POINTS  = 2.0

EMA_FAST = 5
EMA_SLOW = 20

ATR_WINDOW      = 14
ATR_MIN_POINTS  = 2.0

SESSION_WINDOWS = [(time(9,20), time(11,0)),
                   (time(11,15), time(15,5))]

DAILY_LOSS_CAP  = -1000.0
EXIT_BAR_PATH   = "color"

BROKERAGE_PER_TRADE = 20.0  # per leg
SLIPPAGE_POINTS     = 0.10  # per leg

CONFIRM_TREND_AT_ENTRY = True

ENABLE_EOD_SQUARE_OFF  = True
SQUARE_OFF_TIME        = time(15, 25)
```


---

## 13) CLI Usage

```bash
python scalp_with_trend_multibar.py \
  --input_csv NIFTY28OCT2524800CE_history.csv \
  --starting_capital 100000 \
  --qty_per_point 150 \
  --target_points 10 \
  --stoploss_points 2 \
  --ema_fast 5 --ema_slow 20 \
  --atr_window 14 --atr_min_points 2 \
  --exit_bar_path color \
  --brokerage_per_trade 20 \
  --slippage_points 0.10 \
  --confirm_trend_at_entry true \
  --enable_eod_square_off true \
  --square_off_time 15:25
```

**Optional overrides:**
- `--daily_loss_cap -1500`
- Custom `--session_windows` via your own config parsing (or edit defaults).
- Try `--exit_bar_path bull|bear|worst` to test sequencing assumptions.


---

## 14) Reporting & Metrics

- Trades saved to `scalp_with_trend_results.csv` with:
  - `entry_time, exit_time, side, entry, exit`
  - `pnl_points, gross_rupees, costs_rupees, pnl_rupees, equity`
  - `exit_reason` (e.g., `"Target Hit"`, `"Stoploss Hit"`, `"Square-off EOD"`)
- Console summary:
  - **Gross P&L, Total Costs, Net P&L, ROI**
  - **Trades/Wins/Losses/Win‑rate**, **Avg Win/Loss**, **Actual R:R**
  - **Max Drawdown** (from cumulative net P&L)
  - **Exit Reason Breakdown**

**Breakeven win‑rate** given fixed TP/SL:  
`SL / (TP + SL) × 100` → with TP=10, SL=2 ⇒ **16.7%**.


---

## 15) Change Log vs Single‑Bar Variant

- **Holding period**: from single‑bar exit to **multi‑bar until TP/SL**.  
- **EOD square‑off**: added (`Square‑off EOD`).  
- **Exit sequencing**: `EXIT_BAR_PATH` now applied **on every holding bar**.  
- **Trailing**: **removed** in multi‑bar variant (simplifies sequencing).  
- **Session windows**: defaults `(09:20–11:00)` & `(11:15–15:05)`.  
- **Engine**: refactored into a state machine with while‑loop progression.

---

## 16) Extension Ideas

- **Gap filter at entry**: skip if `|open[i+1] − close[i]| > k × ATR[i]`.  
- **ATR‑based dynamic TP/SL**: e.g., `TP = k1 × ATR`, `SL = k2 × ATR`.  
- **Cooldown/max trades per day**.  
- **Portfolio mode** across contracts with risk budgeting.  
- **Broker/tax packs** per venue (STT, exchange fees, GST, stamp, SEBI, etc.).
