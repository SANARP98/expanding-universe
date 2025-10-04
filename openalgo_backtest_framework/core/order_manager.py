import pandas as pd
from openalgo import api

client = api(api_key="your_api_key", host="http://127.0.0.1:5000")

class LiveOrderManager:
    def place_order(self, symbol, action, qty):
        return client.placeorder(
            strategy="SMA_Crossover_Live",
            symbol=symbol,
            exchange="NSE",
            action=action,
            price_type="MARKET",
            product="MIS",
            quantity=qty
        )

class SimulatedOrderManager:
    def __init__(self):
        self.orders = []

    def place_order(self, symbol, action, qty, price, timestamp):
        self.orders.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "action": action,
            "qty": qty,
            "price": price
        })

    def save_to_csv(self, filename):
        df = pd.DataFrame(self.orders)
        df.to_csv(filename, index=False)
        print(f"✅ Simulated orders saved to {filename}")
