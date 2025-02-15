"""
Created on 15/02/2025

@author: Aryan

Filename: Asset_Selection.py

Relative Path: backend/MachineLearning/Supervised_Learning/Asset_Selection.py
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

# Machine learning and deep learning libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json  # type: ignore
from tensorflow.keras.layers import Dense, LSTM  # type: ignore

from arch import arch_model

STEPS_FEATURES = "backend/processed_data/steps_features.csv"
ORDERBOOK_FEATURES = "backend/processed_data/orderbook_features.csv"


def save_models(xgb_model, rf_model, nn_model, save_path="backend/models/"):
    """
    Saves trained models (XGBoost, Random Forest, Neural Network) in JSON format.

    Parameters:
      xgb_model: Trained XGBoost model
      rf_model: Trained Random Forest model
      nn_model: Trained Neural Network model
      save_path (str): Path to save models
    """
    os.makedirs(save_path, exist_ok=True)

    # Save XGBoost model as JSON
    xgb_model_json = xgb_model.get_booster().save_raw(raw_format="json")
    # Save the raw JSON string after decoding into a Python string
    with open(os.path.join(save_path, "xgb_model.json"), "w") as f:
        json.dump(xgb_model_json.decode("utf-8"), f)

    # Save Random Forest model using joblib
    joblib.dump(rf_model, os.path.join(save_path, "rf_model.pkl"))

    # Save Neural Network architecture & weights separately
    nn_model_json = nn_model.to_json()
    with open(os.path.join(save_path, "nn_model.json"), "w") as f:
        f.write(nn_model_json)
    # Updated filename to end with '.weights.h5'
    nn_model.save_weights(os.path.join(save_path, "nn_model.weights.h5"))

    print(f"✅ Models saved successfully in {save_path}")


def load_models(load_path="backend/models/"):
    """
    Loads trained models (XGBoost, Random Forest, Neural Network) from saved files.

    Parameters:
      load_path (str): Path from which to load models

    Returns:
      xgb_model: Loaded XGBoost model
      rf_model: Loaded Random Forest model
      nn_model: Loaded Neural Network model
    """
    # Load XGBoost model
    with open(os.path.join(load_path, "xgb_model.json"), "r") as f:
        xgb_model_json = json.load(f)
    xgb_model = XGBRegressor()
    # load_model accepts bytes, so we encode our loaded JSON string
    xgb_model.load_model(xgb_model_json.encode("utf-8"))

    # Load Random Forest model
    rf_model = joblib.load(os.path.join(load_path, "rf_model.pkl"))

    # Load Neural Network model
    with open(os.path.join(load_path, "nn_model.json"), "r") as f:
        nn_model_json = f.read()
    nn_model = model_from_json(nn_model_json)
    nn_model.load_weights(os.path.join(load_path, "nn_model.weights.h5"))

    print(f"✅ Models loaded successfully from {load_path}")
    return xgb_model, rf_model, nn_model


def create_sequences(X, y, window_size=10):
    """
    Helper function to convert tabular data into sequences for LSTM training.
    """
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i + window_size)])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)


def select_asset(new_data, xgb_model, rf_model, nn_model):
    """
    Select the best asset from new_data based on model predictions.

    Parameters:
      new_data (pd.DataFrame): DataFrame containing features for each asset.
        Expected columns: ['price', 'moving_avg_5', 'moving_avg_10',
                           'momentum', 'trade_frequency', 'avg_trade_size']
      xgb_model: Trained XGBoost model for predicting expected returns.
      rf_model: Trained Random Forest model for predicting market volatility.
      nn_model: Trained Neural Network model for predicting optimal position sizing.

    Returns:
      best_asset (pd.Series): The row from new_data corresponding to the selected asset.
      new_data['composite_score'] (pd.Series): The computed composite scores for each asset.

    Approach:
      - Predict expected returns using the XGBoost model.
      - Predict volatility using the Random Forest model.
      - Predict optimal position sizing using the Neural Network.
      - Compute a composite score (e.g., risk-adjusted return = predicted_return / (predicted_volatility + 1e-8)).
    """
    features = ['price', 'moving_avg_5', 'moving_avg_10',
                'momentum', 'trade_frequency', 'avg_trade_size']
    X_new = new_data[features].values

    # Predict metrics for each asset
    predicted_return = xgb_model.predict(X_new)        # Expected return
    predicted_volatility = rf_model.predict(X_new)       # Volatility
    predicted_optimal = nn_model.predict(
        X_new).flatten()  # Optimal position sizing

    # Compute a composite score. Here, risk-adjusted return is used:
    composite_score = predicted_return / (predicted_volatility + 1e-8)

    # Append predictions and composite score to new_data
    new_data = new_data.copy()
    new_data['predicted_return'] = predicted_return
    new_data['predicted_volatility'] = predicted_volatility
    new_data['predicted_optimal'] = predicted_optimal
    new_data['composite_score'] = composite_score

    # Select the asset with the highest composite score
    best_asset = new_data.loc[new_data['composite_score'].idxmax()]

    return best_asset, new_data['composite_score']


def train_asset_selection_models():
    """
    Train models for supervised asset selection:
      - Expected Returns: LSTM and XGBoost regression models.
      - Market Volatility: GARCH model and Random Forest.
      - Optimal Position Sizing: Neural Network.

    After training, this function demonstrates how to use the trained models
    for asset selection using new data, and then saves the models.
    """
    # ---------------------------
    # 1. Load and Prepare Data
    # ---------------------------
    steps_df = pd.read_csv(STEPS_FEATURES)
    orderbook_df = pd.read_csv(ORDERBOOK_FEATURES)
    # Optionally merge orderbook data if needed (e.g., on timestamp)

    # Create target columns
    steps_df['return'] = steps_df['price'].pct_change().shift(-1)
    steps_df['volatility_target'] = steps_df['volatility']
    steps_df['optimal_position'] = steps_df['return'] / \
        (steps_df['volatility_target']**2 + 1e-8)
    steps_df.dropna(subset=['return', 'volatility_target',
                    'optimal_position'], inplace=True)

    features = ['price', 'moving_avg_5', 'moving_avg_10',
                'momentum', 'trade_frequency', 'avg_trade_size']
    X = steps_df[features].values

    y_return = steps_df['return'].values
    y_vol = steps_df['volatility_target'].values
    y_optimal = steps_df['optimal_position'].values

    X_train, X_test, y_train_return, y_test_return = train_test_split(
        X, y_return, test_size=0.2, random_state=42)
    _, _, y_train_vol, y_test_vol = train_test_split(
        X, y_vol, test_size=0.2, random_state=42)
    _, _, y_train_optimal, y_test_optimal = train_test_split(
        X, y_optimal, test_size=0.2, random_state=42)

    # ---------------------------
    # 2. Expected Returns Models
    # ---------------------------
    print("\n--- Training Expected Returns Models ---")
    window_size = 10
    X_train_seq, y_train_seq = create_sequences(
        X_train, y_train_return, window_size)
    X_test_seq, y_test_seq = create_sequences(
        X_test, y_test_return, window_size)

    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(
            X_train_seq.shape[1], X_train_seq.shape[2])),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    print("Training LSTM for expected returns...")
    lstm_model.fit(X_train_seq, y_train_seq,
                   epochs=10, batch_size=32, verbose=1)
    lstm_pred = lstm_model.predict(X_test_seq)
    lstm_mse = mean_squared_error(y_test_seq, lstm_pred)
    print(f"LSTM Test MSE: {lstm_mse:.6f}")

    print("\nTraining XGBoost for expected returns...")
    xgb_model = XGBRegressor(objective='reg:squarederror',
                             n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train_return)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mse = mean_squared_error(y_test_return, xgb_pred)
    print(f"XGBoost Test MSE: {xgb_mse:.6f}")

    # ---------------------------
    # 3. Market Volatility Models
    # ---------------------------
    print("\n--- Training Market Volatility Models ---")
    print("Fitting GARCH(1,1) model on training returns...")
    garch_model = arch_model(
        y_train_return, vol='Garch', p=1, q=1, dist='normal')
    garch_fit = garch_model.fit(disp='off')
    garch_forecast = garch_fit.forecast(horizon=1)
    garch_pred_vol = np.sqrt(garch_forecast.variance.values[-1, :])
    garch_pred = np.full_like(y_test_vol, garch_pred_vol.mean())
    garch_mse = mean_squared_error(y_test_vol, garch_pred)
    print(f"GARCH Model Test MSE (volatility): {garch_mse:.6f}")

    print("\nTraining Random Forest for volatility prediction...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train_vol)
    rf_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test_vol, rf_pred)
    print(f"Random Forest Test MSE (volatility): {rf_mse:.6f}")

    # ---------------------------
    # 4. Optimal Position Sizing Model
    # ---------------------------
    print("\n--- Training Optimal Position Sizing Model ---")
    nn_model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    nn_model.compile(optimizer='adam', loss='mse')
    print("Training Neural Network for optimal position sizing...")
    nn_model.fit(X_train, y_train_optimal, epochs=10, batch_size=32, verbose=1)
    nn_pred = nn_model.predict(X_test)
    nn_mse = mean_squared_error(y_test_optimal, nn_pred)
    print(f"Neural Network Test MSE (position sizing): {nn_mse:.6f}")

    # ---------------------------
    # 5. Summarize Model Performance
    # ---------------------------
    print("\n✅ Model training complete!")
    print("\nModel Performance Summary:")
    print(f"Expected Returns - LSTM MSE: {lstm_mse:.6f}")
    print(f"Expected Returns - XGBoost MSE: {xgb_mse:.6f}")
    print(f"Market Volatility - GARCH MSE: {garch_mse:.6f}")
    print(f"Market Volatility - Random Forest MSE: {rf_mse:.6f}")
    print(f"Optimal Position Sizing - NN MSE: {nn_mse:.6f}")

    # ---------------------------
    # 6. Asset Selection Demo
    # ---------------------------
    print("\n--- Asset Selection Demo ---")
    demo_data = pd.DataFrame({
        'price': [100.5, 101.0, 99.8],
        'moving_avg_5': [100.2, 100.7, 100.0],
        'moving_avg_10': [100.0, 100.4, 99.9],
        'momentum': [0.5, 0.3, -0.2],
        'trade_frequency': [50, 45, 55],
        'avg_trade_size': [200, 210, 190]
    })

    best_asset, asset_scores = select_asset(
        demo_data, xgb_model, rf_model, nn_model)
    print("\nAsset Selection Results:")
    print("Best Asset:")
    print(best_asset)
    print("\nAll Asset Scores:")
    print(asset_scores)

    # ---------------------------
    # 7. Save Models
    # ---------------------------
    save_models(xgb_model, rf_model, nn_model)


if __name__ == "__main__":
    train_asset_selection_models()
