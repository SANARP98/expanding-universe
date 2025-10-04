import pandas as pd

# 🔁 OpenAlgo Python Bot is running.

# Load historical data
df = pd.read_csv("NIFTY28OCT2524800CE_history.csv", parse_dates=[0])
df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
df.set_index('timestamp', inplace=True)

results = []
TARGET = 10
STOPLOSS = 5

for i in range(1, len(df)):
    prev = df.iloc[i - 1]
    curr = df.iloc[i]
    ts = curr.name

    # --- LONG ENTRY ---
    if curr['high'] > prev['high']:
        entry_price = curr['high']
        exit_price = curr['close']  # default to close
        exit_time = ts
        pnl = 0
        exit_reason = "Same Candle Exit"
        action = "BUY"

        if curr['high'] >= entry_price + TARGET:
            exit_price = entry_price + TARGET
            pnl = TARGET
            exit_reason = "Target Hit"
        elif curr['low'] <= entry_price - STOPLOSS:
            exit_price = entry_price - STOPLOSS
            pnl = -STOPLOSS
            exit_reason = "Stoploss Hit"
        else:
            pnl = exit_price - entry_price

        results.append({
            'entry_time': ts,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'action': action
        })

    # --- SHORT ENTRY ---
    if curr['low'] < prev['low']:
        entry_price = curr['low']
        exit_price = curr['close']
        exit_time = ts
        pnl = 0
        exit_reason = "Same Candle Exit"
        action = "SELL"

        if curr['low'] <= entry_price - TARGET:
            exit_price = entry_price - TARGET
            pnl = TARGET
            exit_reason = "Target Hit"
        elif curr['high'] >= entry_price + STOPLOSS:
            exit_price = entry_price + STOPLOSS
            pnl = -STOPLOSS
            exit_reason = "Stoploss Hit"
        else:
            pnl = entry_price - exit_price  # short PnL logic

        results.append({
            'entry_time': ts,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'action': action
        })

# Save to CSV
result_df = pd.DataFrame(results)
result_df.to_csv("scalping_backtest_result.csv", index=False)

# Cumulative Stats
total_trades = len(result_df)
winning = len(result_df[result_df['pnl'] > 0])
losing = len(result_df[result_df['pnl'] < 0])
net_pnl = result_df['pnl'].sum()

print("✅ Backtest completed. Results saved to scalping_backtest_result.csv")
print(result_df[['entry_time', 'action', 'pnl', 'exit_reason']].tail())
print("\n📊 Backtest Summary:")
print(f"Total Trades: {total_trades}")
print(f"Winning Trades: {winning}")
print(f"Losing Trades: {losing}")
print(f"Net PnL: {net_pnl:.2f} points")
