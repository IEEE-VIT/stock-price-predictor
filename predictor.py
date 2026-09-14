"""
Time-series stock predictor with CLI argument parsing.

Reads a CSV file with columns: Date, Close
Creates lag features plus SMA/EMA technical indicators to predict
future closing prices.
"""

import argparse
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# -----------------------
# Command-line arguments
# -----------------------
parser = argparse.ArgumentParser(
    description="Time-series stock price predictor"
)

parser.add_argument(
    "--csv",
    type=str,
    default="AAPL.csv",
    help="Path to the input CSV file (default: AAPL.csv)"
)

parser.add_argument(
    "--days",
    type=int,
    default=5,
    help="Number of future days to predict (default: 5)"
)

args = parser.parse_args()


# -----------------------
# Config
# -----------------------
CSV_PATH = args.csv
DATE_COL = "Date"
TARGET_COL = "Close"
LAGS = 5
SMA_WINDOWS = (7, 21)
EMA_SPANS = (7, 21)
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
PREDICT_N_DAYS = args.days


# -----------------------
# Validate arguments
# -----------------------
if PREDICT_N_DAYS <= 0:
    raise ValueError("The number of prediction days must be greater than 0.")



def main(save_metrics=False):
    # Load and prepare data.
    df = pd.read_csv(CSV_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[[DATE_COL, TARGET_COL]].dropna().reset_index(drop=True)

    for lag in range(1, LAGS + 1):
        df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)

    # Technical indicators: simple and exponential moving averages.
    for window in SMA_WINDOWS:
        df[f"SMA_{window}"] = df[TARGET_COL].rolling(window=window).mean()
    for span in EMA_SPANS:
        df[f"EMA_{span}"] = df[TARGET_COL].ewm(
            span=span, adjust=False, min_periods=span
        ).mean()

    # Drop initial rows with NaNs from lag shifts and rolling windows.
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        max_window = max(LAGS, *SMA_WINDOWS, *EMA_SPANS)
        raise ValueError(
            "Not enough rows remain after computing technical indicators and "
            f"dropping NaN rows. Provide a CSV with more than {max_window} rows "
            "of historical data."
        )

    feature_cols = (
        [f"lag_{lag}" for lag in range(1, LAGS + 1)]
        + [f"SMA_{window}" for window in SMA_WINDOWS]
        + [f"EMA_{span}" for span in EMA_SPANS]
    )
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    dates = df[DATE_COL].copy()

    # Keep the split time-based so future observations never enter training.
    split_idx = int(len(df) * TRAIN_RATIO)
    X_train, X_test = X.iloc[:split_idx].values, X.iloc[split_idx:].values
    y_train, y_test = y.iloc[:split_idx].values, y.iloc[split_idx:].values
    dates_train, dates_test = dates.iloc[:split_idx], dates.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred_test = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R2: {r2:.4f}")

    if save_metrics:
        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2) if np.isfinite(r2) else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_file": CSV_PATH,
        }
        with open("metrics.json", "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)
            metrics_file.write("\n")

    # Forecast next N days iteratively, updating lag and indicator
    # features with each new prediction.
    lag_cols = [f"lag_{lag}" for lag in range(1, LAGS + 1)]
    current_lags = df.iloc[-1][lag_cols].values.astype(float)
    close_history = df[TARGET_COL].tolist()
    ema_alphas = {span: 2 / (span + 1) for span in EMA_SPANS}
    ema_state = {span: df[f"EMA_{span}"].iloc[-1] for span in EMA_SPANS}

    future_preds = []
    for _ in range(PREDICT_N_DAYS):
        sma_values = [np.mean(close_history[-window:]) for window in SMA_WINDOWS]
        ema_values = [ema_state[span] for span in EMA_SPANS]
        features = np.concatenate([current_lags, sma_values, ema_values])
        scaled = scaler.transform(features.reshape(1, -1))
        pred = model.predict(scaled)[0]
        future_preds.append(pred)

        current_lags = np.roll(current_lags, 1)
        current_lags[0] = pred
        for span in EMA_SPANS:
            ema_state[span] = (
                ema_alphas[span] * pred + (1 - ema_alphas[span]) * ema_state[span]
            )
        close_history.append(pred)

    last_date = df[DATE_COL].iloc[-1]
    future_dates = [last_date +
                    pd.Timedelta(days=i + 1) for i in range(PREDICT_N_DAYS)]

    print("\nPredictions for next", PREDICT_N_DAYS, "days:")
    for d, p in zip(future_dates, future_preds):
        print(f"{d.date()}: {p:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(dates_train, y_train, label="Train (actual)", linewidth=1)
    plt.plot(dates_test, y_test, label="Test (actual)", linewidth=1)
    plt.plot(dates_test, y_pred_test, label="Test (predicted)",
             linestyle="--", linewidth=1)
    plt.plot(future_dates, future_preds,
             label="Future predictions", marker="o", linestyle="-")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("AAPL - Time-series prediction (lag features, time-based split)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict AAPL stock prices.")
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Save test RMSE, MAE, and R² metrics to metrics.json.",
    )
    args = parser.parse_args()
    main(save_metrics=args.save_metrics)
