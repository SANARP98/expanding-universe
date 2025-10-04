from openalgo import ta

def sma_crossover_strategy(df, order_manager):
    df['SMA_10'] = ta.sma(df['close'], 10)
    df['SMA_20'] = ta.sma(df['close'], 20)

    position = None

    for i in range(1, len(df)):
        # Crossover condition
        if df['SMA_10'].iloc[i-1] < df['SMA_20'].iloc[i-1] and df['SMA_10'].iloc[i] > df['SMA_20'].iloc[i]:
            if position != "LONG":
                order_manager.place_order(
                    symbol="RELIANCE",
                    action="BUY",
                    qty=10,
                    price=df['close'].iloc[i],
                    timestamp=df.index[i]
                )
                position = "LONG"

        # Crossunder condition
        elif df['SMA_10'].iloc[i-1] > df['SMA_20'].iloc[i-1] and df['SMA_10'].iloc[i] < df['SMA_20'].iloc[i]:
            if position == "LONG":
                order_manager.place_order(
                    symbol="RELIANCE",
                    action="SELL",
                    qty=10,
                    price=df['close'].iloc[i],
                    timestamp=df.index[i]
                )
                position = None
