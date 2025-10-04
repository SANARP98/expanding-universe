import pandas as pd
import numpy as np

# 🔁 Test ALL Strategies and Compare Results

# ==================== CONFIGURATION ====================
CAPITAL = 100000
LOT_SIZE = 2
TARGET = 10
STOPLOSS = 5

# ==================== LOAD DATA ====================
df = pd.read_csv("NIFTY28OCT2524800CE_history.csv", parse_dates=[0])
df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
df.set_index('timestamp', inplace=True)

# ==================== TECHNICAL INDICATORS ====================
df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
df['atr'] = df['tr'].rolling(window=14).mean()
df['avg_volume'] = df['volume'].rolling(window=20).mean()
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
    if i < 20:
        return None

    prev = df.iloc[i - 1]
    curr = df.iloc[i]

    volume_ok = curr['volume'] > df.iloc[i]['avg_volume'] * 0.8
    range_ok = curr['candle_range'] > df.iloc[i]['avg_range'] * 0.5

    if not (volume_ok and range_ok):
        return None

    if curr['high'] > prev['high'] and df.iloc[i]['ema_5'] > df.iloc[i]['ema_20']:
        return 'LONG'
    elif curr['low'] < prev['low'] and df.iloc[i]['ema_5'] < df.iloc[i]['ema_20']:
        return 'SHORT'

    return None


def trend_following_signal(i, df):
    """Only trade in direction of trend"""
    if i < 20:
        return None

    prev = df.iloc[i - 1]
    curr = df.iloc[i]

    if df.iloc[i]['ema_5'] > df.iloc[i]['ema_20'] and curr['close'] > df.iloc[i]['ema_5']:
        if curr['high'] > prev['high']:
            return 'LONG'
    elif df.iloc[i]['ema_5'] < df.iloc[i]['ema_20'] and curr['close'] < df.iloc[i]['ema_5']:
        if curr['low'] < prev['low']:
            return 'SHORT'

    return None


def mean_reversion_signal(i, df):
    """Trade pullbacks in trend"""
    if i < 20:
        return None

    curr = df.iloc[i]
    distance_from_ema = abs(curr['close'] - df.iloc[i]['ema_5']) / df.iloc[i]['ema_5'] * 100

    if distance_from_ema > 0.5:
        if df.iloc[i]['ema_5'] > df.iloc[i]['ema_20'] and curr['close'] < df.iloc[i]['ema_5']:
            return 'LONG'
        elif df.iloc[i]['ema_5'] < df.iloc[i]['ema_20'] and curr['close'] > df.iloc[i]['ema_5']:
            return 'SHORT'

    return None


# ==================== BACKTEST ENGINE ====================

def run_backtest(strategy_name, get_signal, df):
    """Run backtest for a given strategy"""
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
print("="*80)

for strategy_name, strategy_func in strategies.items():
    print(f"\n📊 Running: {strategy_name}")
    print("-"*80)

    result_df = run_backtest(strategy_name, strategy_func, df)

    if len(result_df) > 0:
        # Save individual results
        result_df.to_csv(f"{strategy_name}_results.csv", index=False)

        # Calculate metrics
        total_trades = len(result_df)
        winning = len(result_df[result_df['pnl_rupees'] > 0])
        losing = len(result_df[result_df['pnl_rupees'] < 0])
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0

        net_pnl = result_df['pnl_rupees'].sum()
        final_capital = CAPITAL + net_pnl
        roi = (net_pnl / CAPITAL) * 100

        avg_win = result_df[result_df['pnl_rupees'] > 0]['pnl_rupees'].mean() if winning > 0 else 0
        avg_loss = result_df[result_df['pnl_rupees'] < 0]['pnl_rupees'].mean() if losing > 0 else 0
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        max_profit = result_df['pnl_rupees'].max()
        max_loss = result_df['pnl_rupees'].min()

        # Drawdown
        result_df['cumulative_pnl'] = result_df['pnl_rupees'].cumsum()
        result_df['peak'] = result_df['cumulative_pnl'].cummax()
        result_df['drawdown'] = result_df['cumulative_pnl'] - result_df['peak']
        max_drawdown = result_df['drawdown'].min()

        # Store comparison data
        comparison_results.append({
            'Strategy': strategy_name,
            'Total Trades': total_trades,
            'Win Rate (%)': round(win_rate, 2),
            'Net P&L (₹)': round(net_pnl, 2),
            'ROI (%)': round(roi, 2),
            'Final Capital (₹)': round(final_capital, 2),
            'Avg Win (₹)': round(avg_win, 2),
            'Avg Loss (₹)': round(avg_loss, 2),
            'Risk:Reward': round(risk_reward, 2),
            'Max Drawdown (₹)': round(max_drawdown, 2),
            'Max Win (₹)': round(max_profit, 2),
            'Max Loss (₹)': round(max_loss, 2)
        })

        # Print summary
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Net P&L: ₹{net_pnl:,.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"Final Capital: ₹{final_capital:,.2f}")
        print(f"Max Drawdown: ₹{max_drawdown:,.2f}")

    else:
        print(f"⚠️  No trades generated")
        comparison_results.append({
            'Strategy': strategy_name,
            'Total Trades': 0,
            'Win Rate (%)': 0,
            'Net P&L (₹)': 0,
            'ROI (%)': 0,
            'Final Capital (₹)': CAPITAL,
            'Avg Win (₹)': 0,
            'Avg Loss (₹)': 0,
            'Risk:Reward': 0,
            'Max Drawdown (₹)': 0,
            'Max Win (₹)': 0,
            'Max Loss (₹)': 0
        })

# ==================== COMPARISON TABLE ====================

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv("strategy_comparison.csv", index=False)

print("\n" + "="*80)
print("🏆 STRATEGY COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

# Find best strategy
best_roi = comparison_df.loc[comparison_df['ROI (%)'].idxmax()]
best_winrate = comparison_df.loc[comparison_df['Win Rate (%)'].idxmax()]
best_rr = comparison_df.loc[comparison_df['Risk:Reward'].idxmax()]

print(f"\n🥇 Best ROI: {best_roi['Strategy']} ({best_roi['ROI (%)']:.2f}%)")
print(f"🎯 Best Win Rate: {best_winrate['Strategy']} ({best_winrate['Win Rate (%)']:.2f}%)")
print(f"💰 Best Risk:Reward: {best_rr['Strategy']} (1:{best_rr['Risk:Reward']:.2f})")

print(f"\n✅ All results saved!")
print(f"   - Individual CSVs: basic_results.csv, filtered_breakout_results.csv, etc.")
print(f"   - Comparison: strategy_comparison.csv")
