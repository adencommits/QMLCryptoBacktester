import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report
from keras.models import load_model
from qiskit_machine_learning.algorithms import QSVC
from qiskit_aer import AerSimulator

"""
The model_evaluation.py script is designed to evaluate the performance of various machine learning models 
that have been trained and saved in the models directory. The models include classical models like Support 
Vector Machines (SVM) and Random Forest, a quantum model (QuantumSVM), and a deep learning model (LSTM).  

The script begins by loading the training and testing data from the specified CSV files. The data is cleaned 
and prepared for prediction by removing any non-numeric columns and applying one-hot encoding if necessary.  

For each model in the models directory, the script loads the model and evaluates its performance on the test 
data. The evaluation metrics include accuracy and a classification report, which provides detailed performance 
metrics such as precision, recall, and F1-score for each class.  

For LSTM models, the test data is reshaped 
to match the input shape expected by the model, and the model is recompiled before evaluation. For QuantumSVM 
models, a new AerSimulator is created and the model is retrained on the training data before evaluation.  

The optimal outcome for this script would be high accuracy scores and high values for precision, recall, 
and F1-score for each class in the classification report. This would indicate that the models are performing 
well on the test data and are able to accurately predict the target variable.  

However, it's important to note that high performance on the test data does not necessarily guarantee high 
performance on new, unseen data. The models should be evaluated on a variety of datasets and under different 
conditions to ensure their robustness and generalizability.
"""

def load_data(filepath):
    print(f"Loading data from {filepath}")
    data = pd.read_csv(filepath, index_col='time', parse_dates=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    return data


def prepare_data(data):
    print("Preparing data for prediction...")
    non_numeric_columns = data.select_dtypes(include=['object']).columns
    if not non_numeric_columns.empty:
        print(f"Non-numeric columns found: {non_numeric_columns}. Applying one-hot encoding.")
        data = pd.get_dummies(data, columns=non_numeric_columns)
    X = data.drop('target', axis=1).values.astype(np.float32)
    y = data['target'].values
    return X, y


def evaluate_model(model, X, y, X_train, y_train, model_name="Model"):
    print(f"Evaluating {model_name}...")
    try:
        if 'LSTM' in model_name:
            X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for LSTM input
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Recompile model, optional
            y_pred = (model.predict(X) > 0.5).astype(int).flatten()
        elif 'QuantumSVM' in model_name:
            simulator = AerSimulator()  # Create a new AerSimulator for QSVC
            qsvc = QSVC(quantum_kernel=None)  # Initialize QSVC without a quantum kernel
            qsvc.quantum_kernel.quantum_instance = simulator  # Set the quantum instance
            qsvc.fit(X_train, y_train)  # Train the QSVC model
            y_pred = qsvc.predict(X)  # Use the predict method
        else:
            y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, zero_division=0)
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print(f"{model_name} Classification Report:\n{report}")
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")


def main():
    models_dir = 'models'
    train_data_filepath = 'data/preprocessed/BTC_preprocessed.csv'
    test_data_filepath = 'data/preprocessed/BTC_preprocessed.csv'

    # Load training data
    train_data = load_data(train_data_filepath)
    X_train, y_train = prepare_data(train_data)

    # Load test data
    test_data = load_data(test_data_filepath)
    X_test, y_test = prepare_data(test_data)

    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') or f.endswith('.h5')]
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        model_name = model_file.split('.')[0]
        try:
            if model_file.endswith('.h5'):
                model = load_model(model_path)
            else:
                model = joblib.load(model_path)
            evaluate_model(model, X_test, y_test, X_train, y_train, model_name=model_name)
        except Exception as e:
            print(f"Error loading or evaluating {model_name}: {e}")


if __name__ == "__main__":
    main()
