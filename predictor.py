# predictor.py
"""
Time-series stock predictor with proper train/test split and evaluation.
Reads AAPL.csv with at least columns: Date, Close
Creates lag features (past n days' Close) to predict next-day Close.
"""

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

# -----------------------
# Load & prepare data
# -----------------------
df = pd.read_csv(CSV_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

# ensure target exists and drop rows missing the target
df = df[[DATE_COL, TARGET_COL]].dropna().reset_index(drop=True)

# create lag features: Close_t-1 ... Close_t-LAGS
for lag in range(1, LAGS + 1):
    df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)

# We want to predict CLOSE at time t using lag_1..lag_LAGS (i.e., previous days)
df = df.dropna().reset_index(drop=True)

# features and target
feature_cols = [f"lag_{lag}" for lag in range(1, LAGS + 1)]
X = df[feature_cols].copy()
y = df[TARGET_COL].copy()
dates = df[DATE_COL].copy()

# -----------------------
# Train/test (time-based split)
# -----------------------
split_idx = int(len(df) * TRAIN_RATIO)
X_train, X_test = X.iloc[:split_idx].values, X.iloc[split_idx:].values
y_train, y_test = y.iloc[:split_idx].values, y.iloc[split_idx:].values
dates_train, dates_test = dates.iloc[:split_idx], dates.iloc[split_idx:]

# -----------------------
# Scale features
# -----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# Fit model
# -----------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -----------------------
# Evaluate on test set
# -----------------------
y_pred_test = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")

# -----------------------
# Forecast next N days (iterative using last observed lags)
# -----------------------
# Start from the last available row's lag features (most recent LAGS closes)
last_known = df.iloc[-1][feature_cols].values.astype(float)  # shape (LAGS,)
future_preds = []
current_lags = last_known.copy()

for i in range(PREDICT_N_DAYS):
    # scale using same scaler — note scaler expects shape (n_samples, n_features)
    scaled = scaler.transform(current_lags.reshape(1, -1))
    pred = model.predict(scaled)[0]
    future_preds.append(pred)

    # shift lags: drop oldest, insert this prediction at lag_1 position
    current_lags = np.roll(current_lags, 1)
    current_lags[0] = pred  # newest becomes lag_1 for next iteration

# prepare dates for the forecast
last_date = df[DATE_COL].iloc[-1]
future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(PREDICT_N_DAYS)]

print("\nPredictions for next", PREDICT_N_DAYS, "days:")
for d, p in zip(future_dates, future_preds):
    print(f"{d.date()}: {p:.2f}")

# -----------------------
# Plot results (train/test + future forecasts)
# -----------------------
plt.figure(figsize=(12, 6))
# plot training data (true closes)
plt.plot(dates_train, y_train, label="Train (actual)", linewidth=1)
# plot test (true)
plt.plot(dates_test, y_test, label="Test (actual)", linewidth=1)
# plot model predictions on test set (align with dates_test)
plt.plot(dates_test, y_pred_test, label="Test (predicted)", linestyle="--", linewidth=1)
# plot future predictions (extend plot)
plt.plot(future_dates, future_preds, label="Future predictions", marker="o", linestyle="-")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("AAPL - Time-series prediction (lag features, time-based split)")
plt.legend()
plt.tight_layout()
plt.show()