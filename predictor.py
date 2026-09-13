# predictor.py
"""
Time-series stock predictor with proper train/test split and evaluation.
Reads a CSV with at least columns: Date, Close
Creates lag features (past n days' Close) to predict next-day Close.
"""

import argparse

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

DEFAULT_CSV_PATH = "AAPL.csv"
DATE_COL = "Date"
TARGET_COL = "Close"
LAGS = 5
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
DEFAULT_PREDICT_N_DAYS = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Time-series stock close predictor using lag features and linear regression."
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV_PATH,
        help=f"Path to the input dataset (default: {DEFAULT_CSV_PATH}).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_PREDICT_N_DAYS,
        help=f"Number of future days to forecast (default: {DEFAULT_PREDICT_N_DAYS}).",
    )
    parser.add_argument(
        "--save-plot",
        metavar="FILENAME",
        default=None,
        help="Save the plot to this path instead of opening an interactive window.",
    )
    return parser.parse_args(argv)


def run(csv_path, predict_n_days, save_plot=None):
    if predict_n_days < 1:
        raise ValueError("--days must be a positive integer")

    df = pd.read_csv(csv_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    df = df[[DATE_COL, TARGET_COL]].dropna().reset_index(drop=True)

    for lag in range(1, LAGS + 1):
        df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)

    df = df.dropna().reset_index(drop=True)

    feature_cols = [f"lag_{lag}" for lag in range(1, LAGS + 1)]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    dates = df[DATE_COL].copy()

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
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")

    last_known = df.iloc[-1][feature_cols].values.astype(float)
    future_preds = []
    current_lags = last_known.copy()

    for _ in range(predict_n_days):
        scaled = scaler.transform(current_lags.reshape(1, -1))
        pred = model.predict(scaled)[0]
        future_preds.append(pred)
        current_lags = np.roll(current_lags, 1)
        current_lags[0] = pred

    last_date = df[DATE_COL].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(predict_n_days)]

    print("\nPredictions for next", predict_n_days, "days:")
    for d, p in zip(future_dates, future_preds):
        print(f"{d.date()}: {p:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(dates_train, y_train, label="Train (actual)", linewidth=1)
    plt.plot(dates_test, y_test, label="Test (actual)", linewidth=1)
    plt.plot(dates_test, y_pred_test, label="Test (predicted)", linestyle="--", linewidth=1)
    plt.plot(future_dates, future_preds, label="Future predictions", marker="o", linestyle="-")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("Time-series prediction (lag features, time-based split)")
    plt.legend()
    plt.tight_layout()
    if save_plot:
        plt.savefig(save_plot)
        print(f"\nSaved plot to {save_plot}")
    else:
        plt.show()


def main(argv=None):
    args = parse_args(argv)
    run(csv_path=args.csv, predict_n_days=args.days, save_plot=args.save_plot)


if __name__ == "__main__":
    main()
