import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


def load_data(filepath):
    print(f"Loading data from {filepath}")
    data = pd.read_csv(filepath, index_col='time', parse_dates=True)
    print(f"Loaded data with {data.shape[0]} rows.")
    return data


def clean_and_feature_engineer_for_backtesting(data):
    print("Cleaning data and calculating features for backtesting...")
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Ensure all NaNs are forward-filled to avoid issues with rolling calculations
    data.ffill(inplace=True)
    data.bfill(inplace=True)  # Back-fill to handle NaNs at the start

    # Drop 'conversionSymbol' and 'conversionType' columns if present
    columns_to_drop = ['conversionSymbol', 'conversionType']
    data.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    print("Calculating technical indicators...")
    data['sma'] = data['close'].rolling(window=10, min_periods=1).mean()
    data['ema'] = data['close'].ewm(span=10, adjust=False).mean()
    data['macd'] = data['close'].ewm(span=12, adjust=False).mean() - data['close'].ewm(span=26, adjust=False).mean()
    change = data['close'].diff()
    gain = change.clip(lower=0).rolling(window=14, min_periods=1).mean()
    loss = -change.clip(upper=0).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Apply changes directly to avoid FutureWarning about chained assignment
    data.loc[:, 'rsi'] = data['rsi'].fillna(0)  # Safe and future-proof filling of NaNs

    # Normalize only the technical indicators, not the close price
    scaler = MinMaxScaler()
    features_to_scale = ['sma', 'ema', 'macd', 'rsi']
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale].dropna())

    # Define target variable
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

    print("Feature engineering complete for backtesting.")
    return data


def save_preprocessed_data_for_backtesting(data, filename):
    directory = 'data/preprocessed_backtesting'
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, filename)
    data.to_csv(path)
    print(f"Data saved to {path}")


def main():
    coins = ['BTC', 'ETH']
    for coin in coins:
        filepath = f'data/{coin}_daily_data.csv'
        data = load_data(filepath)
        data = clean_and_feature_engineer_for_backtesting(data)
        save_preprocessed_data_for_backtesting(data, f"{coin}_preprocessed_backtesting.csv")


if __name__ == "__main__":
    main()
