import os
import pandas as pd
import numpy as np
import requests
import joblib
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from qiskit_aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

API_KEY = '3d4f5b6fb0a9e3a69bae401585e5fbc4550e74fd176f40a24c43e5570e5869ba'
BASE_URL = "https://min-api.cryptocompare.com/data/v2/histominute"
TRADES_FILE = 'trades.csv'
TRANSACTION_FEE_PERCENT = 0.1  # Example transaction fee of 0.1%


class CustomFidelityQuantumKernel(FidelityQuantumKernel):
    def __init__(self, feature_map):
        super().__init__()
        self.feature_map = feature_map

    @property
    def feature_map(self):
        return self._feature_map

    @feature_map.setter
    def feature_map(self, value):
        self._feature_map = value


def fetch_real_time_data(coin, limit=50, aggregate=1):
    print("Fetching real-time data...")
    params = {
        'fsym': coin,
        'tsym': 'USD',
        'limit': limit - 1,  # Fetch past 50 minutes of data
        'aggregate': aggregate,
        'api_key': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()['Data']['Data']
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        print("Real-time data fetched successfully.")
        return df
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return pd.DataFrame()


def clean_and_feature_engineer(data):
    print("Cleaning data and calculating features...")
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Ensure all NaNs are forward-filled to avoid issues with rolling calculations
    data.ffill(inplace=True)
    data.bfill(inplace=True)  # Back-fill to handle NaNs at the start

    # Calculate technical indicators
    data['sma'] = data['close'].rolling(window=10, min_periods=1).mean()
    data['ema'] = data['close'].ewm(span=10, adjust=False).mean()
    data['macd'] = data['close'].ewm(span=12, adjust=False).mean() - data['close'].ewm(span=26, adjust=False).mean()
    change = data['close'].diff()
    gain = change.clip(lower=0).rolling(window=14, min_periods=1).mean()
    loss = -change.clip(upper=0).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Apply changes directly to avoid FutureWarning about chained assignment
    data['rsi'] = data['rsi'].fillna(0)  # Safe and future-proof filling of NaNs

    # Normalize only the technical indicators
    scaler = MinMaxScaler()
    features_to_scale = ['sma', 'ema', 'macd', 'rsi']
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale].dropna())

    print("Feature engineering complete.")
    return data


def load_quantum_svm_model(filepath):
    print("Loading Quantum SVM model...")
    model = joblib.load(filepath)
    feature_map = ZZFeatureMap(feature_dimension=10, reps=2, entanglement='linear')
    if isinstance(model, QSVC) and model.quantum_kernel is None:
        simulator = AerSimulator()
        kernel = CustomFidelityQuantumKernel(feature_map=feature_map)
        kernel.quantum_instance = simulator
        model.quantum_kernel = kernel
    print("Quantum SVM model loaded successfully.")
    return model, feature_map


def generate_signals(data, model):
    print("Generating trading signals...")
    feature_columns = ['high', 'low', 'open', 'volumefrom', 'volumeto', 'close', 'sma', 'ema', 'macd', 'rsi']
    if len(feature_columns) != model.quantum_kernel.feature_map.feature_dimension:
        print(
            f"Mismatch in feature dimensions: {len(feature_columns)} features provided, but model expects {model.quantum_kernel.feature_map.feature_dimension} features.")
        raise ValueError(
            f"Mismatch in feature dimensions: {len(feature_columns)} features provided, but model expects {model.quantum_kernel.feature_map.feature_dimension} features.")
    signals = model.predict(data[feature_columns].values)
    print("Trading signals generated.")
    return signals


def log_trade(trades, action, time, price, quantity, usd_balance, btc_balance):
    trades.append({
        'action': action,
        'time': time,
        'price': price,
        'quantity': quantity,
        'usd_balance': usd_balance,
        'btc_balance': btc_balance
    })
    df = pd.DataFrame(trades)
    if os.path.isfile(TRADES_FILE):
        df.to_csv(TRADES_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(TRADES_FILE, index=False)
    print(
        f"Trade logged: {action} {quantity} BTC at {price} on {time} | USD Balance: {usd_balance} | BTC Balance: {btc_balance}")


def calculate_profit_loss(initial_usd_balance, final_usd_balance, final_btc_balance, btc_price):
    total_assets_initial = initial_usd_balance
    total_assets_final = final_usd_balance + (final_btc_balance * btc_price)
    return total_assets_final - total_assets_initial


def main():
    coin = 'BTC'
    quantity = 1  # Default trade quantity

    # Input initial USD balance
    initial_usd_balance = float(input("Enter initial USD balance: "))
    usd_balance = initial_usd_balance
    btc_balance = 0

    model_filepath = 'models/BTC_QuantumSVM_model.pkl'
    qsvc, _ = load_quantum_svm_model(model_filepath)

    trades = []
    holding = False
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=1)  # Stop after 1 hour

    try:
        while datetime.now() < end_time:
            # Fetch data at the start of the minute
            start_of_minute = datetime.now()
            data = fetch_real_time_data(coin, limit=50)  # Fetch the past 50 minutes of data
            if not data.empty:
                print(f"Fetched data for {len(data)} minutes.")
                data = clean_and_feature_engineer(data)

                # Print technical indicators for debugging
                print("Technical Indicators:")
                print(data[['sma', 'ema', 'macd', 'rsi']].tail())

                try:
                    signals = generate_signals(data, qsvc)
                    latest_signal = signals[-1]
                    current_time = data.index[-1]
                    current_price = data['close'].iloc[-1]

                    # Log signal for debugging
                    print(f"Latest Signal: {latest_signal}")

                    if latest_signal == 1 and usd_balance >= current_price * quantity:
                        # Execute buy trade
                        total_cost = current_price * quantity * (1 + TRANSACTION_FEE_PERCENT / 100)
                        if usd_balance >= total_cost:
                            usd_balance -= total_cost
                            btc_balance += quantity
                            log_trade(trades, 'buy', current_time, current_price, quantity, usd_balance, btc_balance)
                            print(f"Buy trade made: {quantity} BTC at {current_price}")
                        else:
                            print("Insufficient USD balance to execute buy trade.")
                    elif latest_signal == 0 and btc_balance >= quantity:
                        # Execute sell trade
                        total_proceeds = current_price * quantity * (1 - TRANSACTION_FEE_PERCENT / 100)
                        usd_balance += total_proceeds
                        btc_balance -= quantity
                        log_trade(trades, 'sell', current_time, current_price, quantity, usd_balance, btc_balance)
                        print(f"Sell trade made: {quantity} BTC at {current_price}")
                    else:
                        log_trade(trades, 'no_trade', current_time, current_price, 0, usd_balance, btc_balance)
                        print("No trade made.")

                except ValueError as e:
                    print(f"Error during signal generation: {e}")

            # Ensure the loop runs exactly once per minute
            time_to_next_minute = 60 - (datetime.now() - start_of_minute).seconds
            if time_to_next_minute > 0:
                time.sleep(time_to_next_minute)

        # Calculate and display P/L
        final_btc_price = fetch_real_time_data(coin, limit=1).iloc[-1]['close']
        profit_loss = calculate_profit_loss(initial_usd_balance, usd_balance, btc_balance, final_btc_price)
        print(f"Final P/L: {profit_loss} USD")

    except KeyboardInterrupt:
        print("Trading bot stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Trading bot stopped after running for 1 hour.")


if __name__ == "__main__":
    main()
