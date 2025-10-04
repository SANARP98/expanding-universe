from openalgo import api
from config import BACKTEST_CONFIG

client = api(api_key="your_api_key", host="http://127.0.0.1:5000")

class LiveDataProvider:
    def get_data(self):
        # You can implement live mode later
        raise NotImplementedError("Live mode not implemented in this example")

class BacktestDataProvider:
    def __init__(self, symbol, exchange, interval, start_date, end_date):
        self.symbol = symbol
        self.exchange = exchange
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date

    def get_data(self):
        return client.history(
            symbol=self.symbol,
            exchange=self.exchange,
            interval=self.interval,
            start_date=self.start_date,
            end_date=self.end_date
        )
