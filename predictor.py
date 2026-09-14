# predictor.py
"""
Time-series stock predictor with proper train/test split and evaluation.
Reads AAPL.csv with at least columns: Date, Close.
Creates lag features (past n days' Close) to predict next-day Close.
"""

import argparse

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
CSV_PATH = "AAPL.csv"       # input CSV file path
DATE_COL = "Date"
TARGET_COL = "Close"
LAGS = 5                    # use past LAGS days as features
TRAIN_RATIO = 0.8           # time-based train/test split
RANDOM_SEED = 42            # not used for linear regression but good practice
PREDICT_N_DAYS = 5          # forecast horizon


def build_parser():
    """Create the CLI parser for the stock predictor."""
    parser = argparse.ArgumentParser(
        description="Train a lag-based stock price predictor and forecast future prices."
    )
    parser.add_argument(
        "--csv",
        default=CSV_PATH,
        help="Path to the CSV file containing Date and Close columns.",
    )
    parser.add_argument(
        "--lags",
        type=int,
        default=LAGS,
        help="Number of past closing prices to use as features.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help="Fraction of the data used for training.",
    )
    parser.add_argument(
        "--predict-days",
        type=int,
        default=PREDICT_N_DAYS,
        help="Number of future days to forecast.",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=True,
        help="Display the prediction plot (default).",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip displaying the prediction plot.",
    )
    return parser


def run_prediction(csv_path=CSV_PATH, lags=LAGS, train_ratio=TRAIN_RATIO, predict_days=PREDICT_N_DAYS, plot=True):
    """Load stock data, train the model, and print forecast results."""
    df = pd.read_csv(csv_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # ensure target exists and drop rows missing the target
    df = df[[DATE_COL, TARGET_COL]].dropna().reset_index(drop=True)

    # create lag features: Close_t-1 ... Close_t-lags
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)

    # We want to predict CLOSE at time t using lag_1..lag_lags (i.e., previous days)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [f"lag_{lag}" for lag in range(1, lags + 1)]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    dates = df[DATE_COL].copy()

    split_idx = int(len(df) * train_ratio)
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

    for _ in range(predict_days):
        scaled = scaler.transform(current_lags.reshape(1, -1))
        pred = model.predict(scaled)[0]
        future_preds.append(pred)

        current_lags = np.roll(current_lags, 1)
        current_lags[0] = pred

    last_date = df[DATE_COL].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(predict_days)]

    print("\nPredictions for next", predict_days, "days:")
    for d, p in zip(future_dates, future_preds):
        print(f"{d.date()}: {p:.2f}")

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(dates_train, y_train, label="Train (actual)", linewidth=1)
        plt.plot(dates_test, y_test, label="Test (actual)", linewidth=1)
        plt.plot(dates_test, y_pred_test, label="Test (predicted)", linestyle="--", linewidth=1)
        plt.plot(future_dates, future_preds, label="Future predictions", marker="o", linestyle="-")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.title("AAPL - Time-series prediction (lag features, time-based split)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "mse": mse,
        "mae": mae,
        "future_predictions": future_preds,
        "future_dates": future_dates,
    }


def main(args=None):
    """CLI entry point."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)
    run_prediction(
        csv_path=parsed_args.csv,
        lags=parsed_args.lags,
        train_ratio=parsed_args.train_ratio,
        predict_days=parsed_args.predict_days,
        plot=parsed_args.plot,
    )


if __name__ == "__main__":
    main()