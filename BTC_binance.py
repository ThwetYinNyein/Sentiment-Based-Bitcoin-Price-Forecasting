# Import necessary libraries
import pandas as pd
import requests
from datetime import datetime, timedelta

# Define the Binance API endpoint
BASE_URL = "https://api.binance.com/api/v3/klines"

# Define the cryptocurrencies and their symbols
CRYPTOS = {
    #"BCH": "BCHUSDT",
    #"DOGE": "DOGEUSDT",
    #"USDT": "USDTUSDT",
    #"SOL": "SOLUSDT"
    #"ETH": "ETHUSDT"
    "BTC": "BTCUSDT"
}

# Define the time range
START_DATE = "2021-01-01"
END_DATE = "2023-01-01"

# Function to fetch minute-level data from Binance
def fetch_minute_data(symbol, start_time, end_time):
    data = []
    current_time = start_time

    while current_time < end_time:
        # Convert time to milliseconds
        start_timestamp = int(current_time.timestamp() * 1000)
        end_timestamp = int((current_time + timedelta(minutes=1000)).timestamp() * 1000)

        # Fetch data from Binance API
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": start_timestamp,
            "endTime": end_timestamp,
            "limit": 1000
        }
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            klines = response.json()
            for kline in klines:
                data.append({
                    "time": datetime.fromtimestamp(kline[0] / 1000),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5])
                })
        else:
            print(f"Error fetching data for {symbol}: {response.status_code}")
            break

        # Move to the next time window
        current_time += timedelta(minutes=1000)

    return pd.DataFrame(data)

# Main function to fetch data for all cryptocurrencies
def fetch_all_data():
    start_time = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_time = datetime.strptime(END_DATE, "%Y-%m-%d")

    for crypto, symbol in CRYPTOS.items():
        print(f"Fetching data for {crypto}...")
        df = fetch_minute_data(symbol, start_time, end_time)
        df.to_csv(f"{crypto}_minute_data_2021_2023.csv", index=False)
        print(f"Data for {crypto} saved to {crypto}_minute_data_2021_2023.csv")

# Run the script
fetch_all_data()