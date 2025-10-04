from core.data_provider import LiveDataProvider, BacktestDataProvider
from core.order_manager import LiveOrderManager, SimulatedOrderManager
from core.strategy_runner import run_strategy
from config import MODE, BACKTEST_CONFIG

print("🔁 OpenAlgo Python Bot is running.")

if MODE == "LIVE":
    data_provider = LiveDataProvider()
    order_manager = LiveOrderManager()
    run_strategy(data_provider, order_manager)
else:
    data_provider = BacktestDataProvider(
        symbol=BACKTEST_CONFIG['symbol'],
        exchange=BACKTEST_CONFIG['exchange'],
        interval=BACKTEST_CONFIG['interval'],
        start_date=BACKTEST_CONFIG['start_date'],
        end_date=BACKTEST_CONFIG['end_date']
    )
    order_manager = SimulatedOrderManager()
    run_strategy(data_provider, order_manager)
    order_manager.save_to_csv("results/trades.csv")
