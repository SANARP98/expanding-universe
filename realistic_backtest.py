import pandas as pd

# 🔁 Realistic Backtest with proper entry/exit logic

# Load historical data
df = pd.read_csv("NIFTY28OCT2524800CE_history.csv", parse_dates=[0])
df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
df.set_index('timestamp', inplace=True)

results = []
TARGET = 10
STOPLOSS = 5

# Position tracking
in_position = False
position_type = None  # 'LONG' or 'SHORT'
entry_price = None
entry_time = None

for i in range(1, len(df)):
    prev = df.iloc[i - 1]
    curr = df.iloc[i]
    ts = curr.name

    # If we're in a position, check for exit first
    if in_position:
        exit_price = None
        exit_reason = None
        pnl = 0

        if position_type == 'LONG':
            # Check target hit
            if curr['high'] >= entry_price + TARGET:
                exit_price = entry_price + TARGET
                pnl = TARGET
                exit_reason = "Target Hit"
            # Check stoploss hit
            elif curr['low'] <= entry_price - STOPLOSS:
                exit_price = entry_price - STOPLOSS
                pnl = -STOPLOSS
                exit_reason = "Stoploss Hit"
            # Exit at close if no target/SL hit
            else:
                exit_price = curr['close']
                pnl = exit_price - entry_price
                exit_reason = "End of Candle"

        elif position_type == 'SHORT':
            # Check target hit
            if curr['low'] <= entry_price - TARGET:
                exit_price = entry_price - TARGET
                pnl = TARGET
                exit_reason = "Target Hit"
            # Check stoploss hit
            elif curr['high'] >= entry_price + STOPLOSS:
                exit_price = entry_price + STOPLOSS
                pnl = -STOPLOSS
                exit_reason = "Stoploss Hit"
            # Exit at close if no target/SL hit
            else:
                exit_price = curr['close']
                pnl = entry_price - exit_price
                exit_reason = "End of Candle"

        # Record the trade
        results.append({
            'entry_time': entry_time,
            'exit_time': ts,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'action': position_type
        })

        # Reset position
        in_position = False
        position_type = None
        entry_price = None
        entry_time = None

    # If not in position, look for entry signals on NEXT candle
    # Entry is triggered when current candle breaks previous high/low
    # We enter at the OPEN of the NEXT candle (realistic)
    if not in_position and i < len(df) - 1:  # Make sure there's a next candle
        next_candle = df.iloc[i + 1]

        # LONG signal: current high > previous high
        if curr['high'] > prev['high']:
            in_position = True
            position_type = 'LONG'
            entry_price = next_candle['open']  # Enter at next candle's open
            entry_time = next_candle.name

        # SHORT signal: current low < previous low (only if no LONG signal)
        elif curr['low'] < prev['low']:
            in_position = True
            position_type = 'SHORT'
            entry_price = next_candle['open']  # Enter at next candle's open
            entry_time = next_candle.name

# Close any remaining position at end of backtest
if in_position:
    last_candle = df.iloc[-1]
    exit_price = last_candle['close']

    if position_type == 'LONG':
        pnl = exit_price - entry_price
    else:  # SHORT
        pnl = entry_price - exit_price

    results.append({
        'entry_time': entry_time,
        'exit_time': last_candle.name,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pnl': pnl,
        'exit_reason': 'End of Backtest',
        'action': position_type
    })

# Save to CSV
result_df = pd.DataFrame(results)
result_df.to_csv("realistic_backtest_result.csv", index=False)

# Cumulative Stats
total_trades = len(result_df)
winning = len(result_df[result_df['pnl'] > 0])
losing = len(result_df[result_df['pnl'] < 0])
breakeven = len(result_df[result_df['pnl'] == 0])
net_pnl = result_df['pnl'].sum()

# Additional stats
if winning > 0:
    avg_win = result_df[result_df['pnl'] > 0]['pnl'].mean()
else:
    avg_win = 0

if losing > 0:
    avg_loss = result_df[result_df['pnl'] < 0]['pnl'].mean()
else:
    avg_loss = 0

win_rate = (winning / total_trades * 100) if total_trades > 0 else 0

print("✅ Realistic Backtest completed. Results saved to realistic_backtest_result.csv")
print("\n📋 Last 10 Trades:")
print(result_df[['entry_time', 'action', 'entry_price', 'exit_price', 'pnl', 'exit_reason']].tail(10))
print("\n📊 Backtest Summary:")
print(f"Total Trades: {total_trades}")
print(f"Winning Trades: {winning} ({win_rate:.2f}%)")
print(f"Losing Trades: {losing}")
print(f"Breakeven Trades: {breakeven}")
print(f"Net PnL: {net_pnl:.2f} points")
print(f"Average Win: {avg_win:.2f} points")
print(f"Average Loss: {avg_loss:.2f} points")

if avg_loss != 0:
    risk_reward = abs(avg_win / avg_loss)
    print(f"Risk:Reward Ratio: 1:{risk_reward:.2f}")
