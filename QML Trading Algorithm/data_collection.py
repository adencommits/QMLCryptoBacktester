import os
import pandas as pd
import requests

"""
This script is responsible for fetching historical cryptocurrency data from the CryptoCompare API. It 
retrieves daily price information for specified cryptocurrencies and saves the data to CSV files. The 
data includes opening, closing, high, and low prices, as well as volume information.
"""

API_KEY = '3d4f5b6fb0a9e3a69bae401585e5fbc4550e74fd176f40a24c43e5570e5869ba'
BASE_URL = "https://min-api.cryptocompare.com/data/v2/histoday"


def fetch_data(coin, limit=400, aggregate=1):
    params = {
        'fsym': coin,
        'tsym': 'USD',
        'limit': limit - 1,  # Subtracting 1 to ensure fetching data for 1000 days
        'aggregate': aggregate,
        'api_key': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()['Data']['Data']
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        # Remove 'conversionSymbol' and 'conversionType' columns if present
        if 'conversionSymbol' in df.columns:
            df.drop(columns=['conversionSymbol', 'conversionType'], inplace=True)
        return df
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return pd.DataFrame()


def save_data(df, filename):
    path = os.path.join('data', filename)
    df.to_csv(path)
    print(f"Data saved to {path}")


def main():
    coins = ['BTC', 'ETH']
    if not os.path.exists('data'):
        os.makedirs('data')
    for coin in coins:
        print(f"Fetching data for {coin}")
        df = fetch_data(coin)
        if not df.empty:
            save_data(df, f"{coin}_daily_data.csv")


if __name__ == "__main__":
    main()
