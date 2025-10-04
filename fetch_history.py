import pandas as pd
from openalgo import api
from datetime import datetime
from dotenv import load_dotenv
import os

# 🔁 OpenAlgo Python Bot is running.

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_HOST = os.getenv("OPENALGO_API_HOST")

# Initialize OpenAlgo client
client = api(api_key=API_KEY, host=API_HOST)

# --------- INPUT CONFIG -------------
symbol = "NIFTY25NOV2525800PE"     # Format: [BASE][YY][MMM][DD]FUT
exchange = "NFO"                   # NFO = NSE Futures & Options
interval = "5m"                    # Options: "1m", "5m", "15m", "1h", "D"
start_date = "2025-07-01"          # Format: YYYY-MM-DD
end_date = "2025-10-04"
output_csv = f"{symbol}_history.csv"
# ------------------------------------

# Fetch Historical Data
df = client.history(
    symbol=symbol,
    exchange=exchange,
    interval=interval,
    start_date=start_date,
    end_date=end_date
)

# Convert index to datetime if not already
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

# Save to CSV
df.to_csv(output_csv)
print(f"✅ Historical data saved to {output_csv}")
