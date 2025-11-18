import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_extractor import SQLFeatureExtractor

def train_real_ml_model():
    print("Training a real ML model for SQL query optimization using Neural Networks...")

    # Load execution data
    execution_file = 'data/queries_with_execution_times.json'
    if not os.path.exists(execution_file):
        print(f"Execution data file not found: {execution_file}")
        print("Run execute_queries.py first to collect data.")
        return

    with open(execution_file, 'r') as f:
        execution_data = json.load(f)

    # Extract features and targets
    feature_extractor = SQLFeatureExtractor()
    features_list = []
    execution_times = []

    for item in execution_data:
        if item.get('execution_success') and item.get('actual_execution_time_ms'):
            query = item['query_sql']
            exec_time = item['actual_execution_time_ms']

            try:
                features = feature_extractor.extract_features(query)
                features_list.append(list(features.values()))
                execution_times.append(exec_time)
            except Exception as e:
                print(f"Skipping query due to feature extraction error: {e}")
                continue

    if len(features_list) < 50:
        print("Not enough data for training. Need at least 50 samples.")
        return

    X = np.array(features_list)
    y = np.array(execution_times)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build neural network model
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Regression output
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    print("Training neural network...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    )

    # Evaluate
    y_pred = model.predict(X_test_scaled).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"Mean Absolute Error: {mae:.2f} ms")
    print(f"R² Score: {r2:.3f}")

    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    model.save('models/real_ml_model.h5')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_extractor, 'models/feature_extractor.pkl')

    print("Real ML model trained and saved!")
    print("Algorithm used: Deep Neural Network Regression")

if __name__ == '__main__':
    train_real_ml_model()