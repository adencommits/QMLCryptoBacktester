import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import pytz
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from qiskit_aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


def current_time():
    est = pytz.timezone('America/New_York')
    return datetime.now(est).strftime('%Y-%m-%d %H:%M:%S %Z')


def load_data(filepath):
    print(f"{current_time()} - Loading data from {filepath}")
    data = pd.read_csv(filepath, index_col='time', parse_dates=True)
    if 'target' not in data.columns:
        print(f"{current_time()} - Error: 'target' column not found. Ensure data preprocessing includes feature labeling.")
        return None

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    print(f"{current_time()} - Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def prepare_data(data, test_size=0.2, random_state=42):
    print(f"{current_time()} - Preparing data...")
    X = data.drop('target', axis=1).values.astype(np.float32)
    y = data['target'].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def train_classical_models(X_train, y_train):
    print(f"{current_time()} - Training classical models...")
    models = {
        'SVM': SVC(probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{current_time()} - {name} model trained.")
    return models


def train_quantum_model(X_train, y_train, feature_dim):
    print(f"{current_time()} - Initializing quantum simulation...")
    simulator = AerSimulator()
    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement='linear')
    fidelity_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=None, enforce_psd=True)
    qsvc = QSVC(quantum_kernel=fidelity_kernel)
    qsvc.fit(X_train, y_train)
    print(f"{current_time()} - Quantum SVM trained.")
    return qsvc


def train_lstm_model(X_train, y_train):
    print(f"{current_time()} - Training LSTM model...")
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    model = Sequential([
        Input(shape=(1, X_train.shape[2])),
        LSTM(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=2)
    return model


def evaluate_models(models, X_test, y_test):
    print(f"{current_time()} - Evaluating models...")
    results = {}
    for name, model in models.items():
        if name == 'LSTM':
            X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            y_pred = (model.predict(X_test_reshaped) > 0.5).astype(int).flatten()
        else:
            y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        results[name] = {'accuracy': accuracy, 'report': report}
        print(f"{current_time()} - Model: {name}, Accuracy: {accuracy:.2f}")
    return results


def main():
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    coins = ['BTC', 'ETH']  # Cryptocurrencies to process
    feature_dim = 3  # Feature dimension for quantum model
    for coin in coins:
        filepath = f'data/preprocessed/{coin}_preprocessed.csv'
        data = load_data(filepath)
        if data is not None:
            X_train, X_test, y_train, y_test = prepare_data(data)
            classical_models = train_classical_models(X_train, y_train)
            quantum_model = train_quantum_model(X_train, y_train, feature_dim)
            lstm_model = train_lstm_model(X_train, y_train)
            classical_models['QuantumSVM'] = quantum_model
            classical_models['LSTM'] = lstm_model
            results = evaluate_models(classical_models, X_test, y_test)
            for model_name, model in classical_models.items():
                model_path = os.path.join(models_dir, f'{coin}_{model_name}.h5') if 'LSTM' in model_name else os.path.join(models_dir, f'{coin}_{model_name}_model.pkl')
                if 'LSTM' in model_name:
                    model.save(model_path)  # Save Keras model correctly
                else:
                    joblib.dump(model, model_path)  # Save sklearn models
                print(f"{current_time()} - {model_name} model saved to {model_path}.")
        else:
            print(f"{current_time()} - Data loading failed for {coin}.")


if __name__ == "__main__":
    main()
