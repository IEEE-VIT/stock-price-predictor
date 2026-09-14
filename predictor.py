# predictor.py
"""
Time-series stock predictor with proper train/test split and evaluation.
Reads AAPL.csv with at least columns: Date, Close
Creates lag features (past n days' Close) to predict next-day Close.

Usage:
    Train a new model version (fits scaler + estimator, saves both to disk):
        python predictor.py --train

    Predict using the most recently trained model version (no refitting):
        python predictor.py --predict

    Predict using a specific saved version:
        python predictor.py --predict --model-version v003
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# -----------------------
# Config
# -----------------------
CSV_PATH = "AAPL.csv"          # input CSV file path
DATE_COL = "Date"
TARGET_COL = "Close"
LAGS = 5                       # use past LAGS days as features
TRAIN_RATIO = 0.8              # time-based train/test split
PREDICT_N_DAYS = 5             # forecast horizon
MODEL_DIR = "models"           # base directory for versioned artifacts

MODEL_FILENAME = "model.joblib"
SCALER_FILENAME = "scaler.joblib"
METADATA_FILENAME = "metadata.json"

VERSION_PATTERN = re.compile(r"^v(\d{3,})$")


class SchemaValidationError(Exception):
    """Raised when loaded artifacts don't match the schema the code expects."""


# -----------------------
# Data preparation (shared by train & predict)
# -----------------------
def feature_cols_for(lags: int) -> list:
    return [f"lag_{lag}" for lag in range(1, lags + 1)]


def load_and_engineer(csv_path: str, lags: int) -> pd.DataFrame:
    """Load the CSV, sort by date, and create lag_1..lag_LAGS feature columns."""
    df = pd.read_csv(csv_path)
    if DATE_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"CSV must contain '{DATE_COL}' and '{TARGET_COL}' columns.")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[[DATE_COL, TARGET_COL]].dropna().reset_index(drop=True)

    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)

    df = df.dropna().reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, feature_cols: list, train_ratio: float):
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    dates = df[DATE_COL].copy()

    split_idx = int(len(df) * train_ratio)
    X_train, X_test = X.iloc[:split_idx].values, X.iloc[split_idx:].values
    y_train, y_test = y.iloc[:split_idx].values, y.iloc[split_idx:].values
    dates_train, dates_test = dates.iloc[:split_idx], dates.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, dates_train, dates_test


# -----------------------
# Versioned artifact directory helpers
# -----------------------
def _existing_versions(base_dir: Path) -> list:
    if not base_dir.exists():
        return []
    versions = []
    for child in base_dir.iterdir():
        if child.is_dir():
            m = VERSION_PATTERN.match(child.name)
            if m:
                versions.append(int(m.group(1)))
    return sorted(versions)


def next_version_dir(base_dir: Path) -> Path:
    versions = _existing_versions(base_dir)
    next_num = (versions[-1] + 1) if versions else 1
    version_dir = base_dir / f"v{next_num:03d}"
    version_dir.mkdir(parents=True, exist_ok=False)
    return version_dir


def latest_version_dir(base_dir: Path) -> Path:
    versions = _existing_versions(base_dir)
    if not versions:
        raise FileNotFoundError(
            f"No trained model versions found under '{base_dir}'. Run with --train first."
        )
    return base_dir / f"v{versions[-1]:03d}"


def resolve_version_dir(base_dir: Path, requested: Optional[str]) -> Path:
    if requested is None:
        return latest_version_dir(base_dir)
    version_dir = base_dir / requested
    if not version_dir.is_dir():
        raise FileNotFoundError(f"Model version '{requested}' not found under '{base_dir}'.")
    return version_dir


# -----------------------
# Artifact validation
# -----------------------
def validate_artifacts(model, scaler, metadata: dict, expected_feature_cols: list):
    """
    Verify that a loaded model/scaler pair matches the feature schema and
    output shape the current code expects, before they're used for inference.
    Raises SchemaValidationError with a specific reason on any mismatch.
    """
    saved_cols = metadata.get("feature_cols")
    if saved_cols != expected_feature_cols:
        raise SchemaValidationError(
            f"Feature schema mismatch: model was trained on {saved_cols}, "
            f"but current config (LAGS={LAGS}) expects {expected_feature_cols}."
        )

    n_expected = len(expected_feature_cols)

    scaler_n_features = getattr(scaler, "n_features_in_", None)
    if scaler_n_features != n_expected:
        raise SchemaValidationError(
            f"Scaler expects {scaler_n_features} features, but current schema has {n_expected}."
        )

    model_n_features = getattr(model, "n_features_in_", None)
    if model_n_features != n_expected:
        raise SchemaValidationError(
            f"Model expects {model_n_features} features, but current schema has {n_expected}."
        )

    # Dry-run inference with a dummy row to confirm the loaded pair actually
    # produces a single scalar prediction per row (output shape check).
    dummy = np.zeros((1, n_expected))
    try:
        scaled_dummy = scaler.transform(dummy)
        dry_run_pred = model.predict(scaled_dummy)
    except Exception as e:
        raise SchemaValidationError(f"Dry-run inference with loaded artifacts failed: {e}") from e

    if np.asarray(dry_run_pred).shape != (1,):
        raise SchemaValidationError(
            f"Unexpected model output shape {np.asarray(dry_run_pred).shape}; expected (1,)."
        )


# -----------------------
# --train mode
# -----------------------
def run_train(args):
    lags = args.lags
    feature_cols = feature_cols_for(lags)

    df = load_and_engineer(args.csv, lags)
    if len(df) < 10:
        print(f"Warning: only {len(df)} usable rows after lag engineering; "
              f"metrics below are not statistically meaningful.")

    X_train, X_test, y_train, y_test, dates_train, dates_test = time_split(
        df, feature_cols, args.train_ratio
    )
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Train/test split produced an empty split (train={len(X_train)}, "
            f"test={len(X_test)}). Provide more data or adjust --train-ratio."
        )

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

    base_dir = Path(args.model_dir)
    version_dir = next_version_dir(base_dir)

    # Synchronous, blocking serialization — no background thread/async queue.
    # Each version gets its own directory so earlier artifacts are never
    # overwritten by a later training run.
    joblib.dump(model, version_dir / MODEL_FILENAME)
    joblib.dump(scaler, version_dir / SCALER_FILENAME)

    metadata = {
        "feature_cols": feature_cols,
        "lags": lags,
        "date_col": DATE_COL,
        "target_col": TARGET_COL,
        "train_ratio": args.train_ratio,
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "test_mse": float(mse),
        "test_mae": float(mae),
        "sklearn_version": sklearn.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(Path(args.csv).resolve()),
    }
    with open(version_dir / METADATA_FILENAME, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved trained artifacts to: {version_dir}/")
    print(f"  - {MODEL_FILENAME}")
    print(f"  - {SCALER_FILENAME}")
    print(f"  - {METADATA_FILENAME}")

    if args.plot:
        _plot_train(dates_train, dates_test, y_train, y_test,
                    model.predict(X_train_scaled), y_pred_test, version_dir)


def _plot_train(dates_train, dates_test, y_train, y_test, y_pred_train, y_pred_test, version_dir):
    import matplotlib.pyplot as plt

    train_residuals = y_train - y_pred_train
    test_residuals = y_test - y_pred_test

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(dates_train, y_train, label="Train (actual)", linewidth=1, color="blue", alpha=0.7)
    ax1.plot(dates_train, y_pred_train, label="Train (predicted)", linestyle="--", linewidth=1, color="cyan")
    ax1.plot(dates_test, y_test, label="Test (actual)", linewidth=1, color="green", alpha=0.7)
    ax1.plot(dates_test, y_pred_test, label="Test (predicted)", linestyle="--", linewidth=1, color="orange")
    ax1.set_ylabel("Close Price")
    ax1.set_title("AAPL - Time-Series Actual vs. Predicted (Training Run)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(dates_train, train_residuals, label="Train Residuals", linewidth=1, color="purple", alpha=0.7)
    ax2.plot(dates_test, test_residuals, label="Test Residuals", linewidth=1, color="red", alpha=0.7)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Residual Error (Actual - Pred)")
    ax2.set_title("Model Residual Errors Over Time")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = version_dir / "train_report.png"
    plt.savefig(out_path)
    print(f"  - train_report.png")
    plt.close(fig)


# -----------------------
# --predict mode
# -----------------------
def run_predict(args):
    lags = args.lags
    feature_cols = feature_cols_for(lags)
    base_dir = Path(args.model_dir)

    version_dir = resolve_version_dir(base_dir, args.model_version)

    model_path = version_dir / MODEL_FILENAME
    scaler_path = version_dir / SCALER_FILENAME
    metadata_path = version_dir / METADATA_FILENAME
    for p in (model_path, scaler_path, metadata_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected artifact missing: {p}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Fail loudly before doing anything with a stale/incompatible model.
    validate_artifacts(model, scaler, metadata, feature_cols)
    print(f"Loaded model version '{version_dir.name}' (trained {metadata.get('created_at')}), "
          f"schema OK ({len(feature_cols)} features).")

    df = load_and_engineer(args.csv, lags)
    if df.empty:
        raise ValueError("No usable rows after lag engineering; check --csv input.")

    last_known = df.iloc[-1][feature_cols].values.astype(float)
    current_lags = last_known.copy()
    future_preds = []

    for _ in range(args.predict_days):
        scaled = scaler.transform(current_lags.reshape(1, -1))
        pred = model.predict(scaled)[0]
        future_preds.append(pred)

        current_lags = np.roll(current_lags, 1)
        current_lags[0] = pred

    last_date = df[DATE_COL].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(args.predict_days)]

    print(f"\nPredictions for next {args.predict_days} days:")
    for d, p in zip(future_dates, future_preds):
        print(f"{d.date()}: {p:.2f}")

    if args.plot:
        _plot_predict(df, future_dates, future_preds, version_dir)

    return future_dates, future_preds


def _plot_predict(df, future_dates, future_preds, version_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df[DATE_COL], df[TARGET_COL], label="Historical (actual)", linewidth=1, color="blue")
    ax.plot(future_dates, future_preds, label="Future predictions", marker="o", linestyle="-", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.set_title("AAPL - Forecast from Loaded Model")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = version_dir / "predict_report.png"
    plt.savefig(out_path)
    print(f"Saved plot to: {out_path}")
    plt.close(fig)



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or run a lag-feature stock price predictor.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Fit scaler + model and save a new versioned artifact set.")
    mode.add_argument("--predict", action="store_true", help="Load a saved model/scaler and forecast, without refitting.")

    parser.add_argument("--csv", default=CSV_PATH, help=f"Path to input CSV (default: {CSV_PATH}).")
    parser.add_argument("--lags", type=int, default=LAGS, help=f"Number of lag features (default: {LAGS}).")
    parser.add_argument("--model-dir", default=MODEL_DIR, help=f"Base directory for versioned artifacts (default: {MODEL_DIR}).")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO, help=f"Time-based train split ratio (default: {TRAIN_RATIO}).")
    parser.add_argument("--model-version", default=None, help="Specific version dir (e.g. v003) to load for --predict. Defaults to the latest.")
    parser.add_argument("--predict-days", type=int, default=PREDICT_N_DAYS, help=f"Forecast horizon in days (default: {PREDICT_N_DAYS}).")
    parser.add_argument("--plot", action="store_true", help="Save a PNG report alongside the artifacts/predictions instead of just printing.")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        if args.train:
            run_train(args)
        else:
            run_predict(args)
    except (SchemaValidationError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()