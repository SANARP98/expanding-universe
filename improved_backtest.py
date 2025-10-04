import pandas as pd
import numpy as np

# 🔁 Improved Backtest with Capital Management & Better Filters

# ==================== CONFIGURATION ====================
CAPITAL = 100000
LOT_SIZE = 2  # Number of lots per trade
TARGET = 10
STOPLOSS = 5

# Strategy Selection (choose one)
STRATEGY = "filtered_breakout"  # Options: "basic", "filtered_breakout", "trend_following", "mean_reversion"

# ==================== LOAD DATA ====================
df = pd.read_csv("NIFTY28OCT2524800CE_history.csv", parse_dates=[0])
df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
df.set_index('timestamp', inplace=True)

# ==================== TECHNICAL INDICATORS ====================
# Moving Averages for trend filter
df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

# ATR for volatility filter
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
df['atr'] = df['tr'].rolling(window=14).mean()

# Volume filter
df['avg_volume'] = df['volume'].rolling(window=20).mean()

# Range of candle
df['candle_range'] = df['high'] - df['low']
df['avg_range'] = df['candle_range'].rolling(window=10).mean()

# ==================== STRATEGY FUNCTIONS ====================

def basic_signal(i, df):
    """Original breakout logic"""
    prev = df.iloc[i - 1]
    curr = df.iloc[i]

    if curr['high'] > prev['high']:
        return 'LONG'
    elif curr['low'] < prev['low']:
        return 'SHORT'
    return None


def filtered_breakout_signal(i, df):
    """Breakout with volume & trend filters"""
    if i < 20:  # Need data for indicators
        return None

    prev = df.iloc[i - 1]
    curr = df.iloc[i]

    # Volume must be above average
    volume_ok = curr['volume'] > df.iloc[i]['avg_volume'] * 0.8

    # Candle range should be significant
    range_ok = curr['candle_range'] > df.iloc[i]['avg_range'] * 0.5

    if not (volume_ok and range_ok):
        return None

    # LONG: breakout + uptrend
    if curr['high'] > prev['high'] and df.iloc[i]['ema_5'] > df.iloc[i]['ema_20']:
        return 'LONG'

    # SHORT: breakdown + downtrend
    elif curr['low'] < prev['low'] and df.iloc[i]['ema_5'] < df.iloc[i]['ema_20']:
        return 'SHORT'

    return None


def trend_following_signal(i, df):
    """Only trade in direction of trend"""
    if i < 20:
        return None

    prev = df.iloc[i - 1]
    curr = df.iloc[i]

    # Strong uptrend
    if df.iloc[i]['ema_5'] > df.iloc[i]['ema_20'] and curr['close'] > df.iloc[i]['ema_5']:
        if curr['high'] > prev['high']:
            return 'LONG'

    # Strong downtrend
    elif df.iloc[i]['ema_5'] < df.iloc[i]['ema_20'] and curr['close'] < df.iloc[i]['ema_5']:
        if curr['low'] < prev['low']:
            return 'SHORT'

    return None


def mean_reversion_signal(i, df):
    """Trade pullbacks in trend"""
    if i < 20:
        return None

    curr = df.iloc[i]

    # Price far from EMA = potential reversion
    distance_from_ema = abs(curr['close'] - df.iloc[i]['ema_5']) / df.iloc[i]['ema_5'] * 100

    # Only trade if stretched > 0.5%
    if distance_from_ema > 0.5:
        # Buy dips in uptrend
        if df.iloc[i]['ema_5'] > df.iloc[i]['ema_20'] and curr['close'] < df.iloc[i]['ema_5']:
            return 'LONG'
        # Sell rallies in downtrend
        elif df.iloc[i]['ema_5'] < df.iloc[i]['ema_20'] and curr['close'] > df.iloc[i]['ema_5']:
            return 'SHORT'

    return None


# ==================== SELECT STRATEGY ====================
strategy_map = {
    "basic": basic_signal,
    "filtered_breakout": filtered_breakout_signal,
    "trend_following": trend_following_signal,
    "mean_reversion": mean_reversion_signal
}

get_signal = strategy_map[STRATEGY]

# ==================== BACKTEST ENGINE ====================
results = []
in_position = False
position_type = None
entry_price = None
entry_time = None
current_capital = CAPITAL

for i in range(1, len(df) - 1):
    curr = df.iloc[i]
    ts = curr.name

    # Exit logic
    if in_position:
        next_candle = df.iloc[i + 1]
        exit_price = None
        exit_reason = None
        pnl_points = 0

        if position_type == 'LONG':
            if next_candle['high'] >= entry_price + TARGET:
                exit_price = entry_price + TARGET
                pnl_points = TARGET
                exit_reason = "Target Hit"
            elif next_candle['low'] <= entry_price - STOPLOSS:
                exit_price = entry_price - STOPLOSS
                pnl_points = -STOPLOSS
                exit_reason = "Stoploss Hit"
            else:
                exit_price = next_candle['close']
                pnl_points = exit_price - entry_price
                exit_reason = "End of Candle"

        elif position_type == 'SHORT':
            if next_candle['low'] <= entry_price - TARGET:
                exit_price = entry_price - TARGET
                pnl_points = TARGET
                exit_reason = "Target Hit"
            elif next_candle['high'] >= entry_price + STOPLOSS:
                exit_price = entry_price + STOPLOSS
                pnl_points = -STOPLOSS
                exit_reason = "Stoploss Hit"
            else:
                exit_price = next_candle['close']
                pnl_points = entry_price - exit_price
                exit_reason = "End of Candle"

        # Calculate P&L in rupees (assuming 1 point = ₹1 per lot)
        pnl_rupees = pnl_points * LOT_SIZE
        current_capital += pnl_rupees

        results.append({
            'entry_time': entry_time,
            'exit_time': next_candle.name,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_points': pnl_points,
            'pnl_rupees': pnl_rupees,
            'capital': current_capital,
            'exit_reason': exit_reason,
            'action': position_type
        })

        in_position = False
        position_type = None
        entry_price = None
        entry_time = None

    # Entry logic
    if not in_position:
        signal = get_signal(i, df)

        if signal and i < len(df) - 1:
            next_candle = df.iloc[i + 1]
            in_position = True
            position_type = signal
            entry_price = next_candle['open']
            entry_time = next_candle.name

# ==================== RESULTS & ANALYSIS ====================
result_df = pd.DataFrame(results)

if len(result_df) > 0:
    result_df.to_csv(f"{STRATEGY}_backtest_result.csv", index=False)

    total_trades = len(result_df)
    winning = len(result_df[result_df['pnl_rupees'] > 0])
    losing = len(result_df[result_df['pnl_rupees'] < 0])
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0

    net_pnl_rupees = result_df['pnl_rupees'].sum()
    final_capital = CAPITAL + net_pnl_rupees
    roi = (net_pnl_rupees / CAPITAL) * 100

    avg_win = result_df[result_df['pnl_rupees'] > 0]['pnl_rupees'].mean() if winning > 0 else 0
    avg_loss = result_df[result_df['pnl_rupees'] < 0]['pnl_rupees'].mean() if losing > 0 else 0
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    max_profit = result_df['pnl_rupees'].max()
    max_loss = result_df['pnl_rupees'].min()

    # Drawdown calculation
    result_df['cumulative_pnl'] = result_df['pnl_rupees'].cumsum()
    result_df['peak'] = result_df['cumulative_pnl'].cummax()
    result_df['drawdown'] = result_df['cumulative_pnl'] - result_df['peak']
    max_drawdown = result_df['drawdown'].min()

    print(f"✅ Backtest completed using '{STRATEGY}' strategy")
    print(f"Results saved to {STRATEGY}_backtest_result.csv\n")

    print("📋 Last 10 Trades:")
    print(result_df[['entry_time', 'action', 'pnl_rupees', 'exit_reason', 'capital']].tail(10))

    print("\n" + "="*60)
    print("📊 BACKTEST SUMMARY")
    print("="*60)
    print(f"Strategy: {STRATEGY}")
    print(f"Initial Capital: ₹{CAPITAL:,.2f}")
    print(f"Final Capital: ₹{final_capital:,.2f}")
    print(f"Net P&L: ₹{net_pnl_rupees:,.2f}")
    print(f"ROI: {roi:.2f}%")
    print(f"\nTotal Trades: {total_trades}")
    print(f"Winning Trades: {winning} ({win_rate:.2f}%)")
    print(f"Losing Trades: {losing}")
    print(f"\nAverage Win: ₹{avg_win:,.2f}")
    print(f"Average Loss: ₹{avg_loss:,.2f}")
    print(f"Risk:Reward Ratio: 1:{risk_reward:.2f}")
    print(f"\nMax Single Win: ₹{max_profit:,.2f}")
    print(f"Max Single Loss: ₹{max_loss:,.2f}")
    print(f"Max Drawdown: ₹{max_drawdown:,.2f}")
    print("="*60)
else:
    print(f"⚠️  No trades generated for '{STRATEGY}' strategy")
    print("Try adjusting filters or using a different strategy")
