from strategies.my_strategy import sma_crossover_strategy

def run_strategy(data_provider, order_manager):
    df = data_provider.get_data()
    sma_crossover_strategy(df, order_manager)
