import pandas as pd
import joblib
from qiskit_aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit.circuit.library import ZFeatureMap


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


def calculate_pl_for_trade(buy_price, sell_price, quantity, transaction_cost):
    return (sell_price - buy_price) * quantity - transaction_cost


def calculate_pl(trades):
    return sum(trade['pl'] for trade in trades if 'sell_price' in trade)


def load_historical_data(filepath):
    print("Checkpoint 1: Loading historical data")
    data = pd.read_csv(filepath, index_col='time', parse_dates=True)
    return data


def load_quantum_svm_model(filepath):
    print("Checkpoint 2: Loading Quantum SVM model")
    model = joblib.load(filepath)
    feature_map = ZZFeatureMap(feature_dimension=10, reps=2, entanglement='linear')
    if isinstance(model, QSVC) and model.quantum_kernel is None:
        simulator = AerSimulator()
        kernel = CustomFidelityQuantumKernel(feature_map=feature_map)
        kernel.quantum_instance = simulator
        model.quantum_kernel = kernel
    return model, feature_map


def generate_signals(data, model):
    print("Checkpoint 3: Starting signal generation")
    signals = []
    feature_columns = ['high', 'low', 'open', 'volumefrom', 'volumeto', 'close', 'sma', 'ema', 'macd', 'rsi']
    for index, row in data.iterrows():
        print(f"Checkpoint 3.{index}: Generating signal for {index}")
        feature_vector = row[feature_columns].values.reshape(1, -1)
        signal = model.predict(feature_vector)[0]
        signals.append(signal)
    print("Checkpoint 3: Signal generation completed")
    return pd.Series(signals, index=data.index)


def execute_trades(data):
    print("Checkpoint 4: Executing trades")
    trades = []
    holding = False
    for time, row in data.iterrows():
        if row['signal'] == 1 and not holding:
            trades.append({'action': 'buy', 'time': time, 'price': row['close'], 'quantity': 1})
            holding = True
        elif row['signal'] == 0 and holding:
            trades.append({'action': 'sell', 'time': time, 'price': row['close'], 'quantity': 1})
            holding = False
    return trades


def main():
    historical_data_filepath = 'data/preprocessed_backtesting/BTC_preprocessed_backtesting.csv'
    quantum_svm_model_filepath = 'models/BTC_QuantumSVM_model.pkl'

    historical_data = load_historical_data(historical_data_filepath)
    qsvc, _ = load_quantum_svm_model(quantum_svm_model_filepath)
    historical_data['signal'] = generate_signals(historical_data, qsvc)

    trades = execute_trades(historical_data)
    trades_filepath = 'past_trades.csv'
    with open(trades_filepath, 'w') as file:
        file.write('action,time,price,quantity\n')
        for trade in trades:
            file.write(f"{trade['action']},{trade['time']},{trade['price']},{trade['quantity']}\n")

    print("Checkpoint 5: Backtesting completed. Trades saved.")
    quantum_trades = [trade for trade in trades if trade['action'] == 'sell']
    quantum_total_pl = calculate_pl(quantum_trades)
    print(f"Total P/L for Quantum Strategy: {quantum_total_pl}")


if __name__ == "__main__":
    main()