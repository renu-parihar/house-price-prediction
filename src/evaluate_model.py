"""Evaluate the trained model on a hold-out validation split.

Usage (from project root):
    python -m src.evaluate_model
"""

from __future__ import annotations

import os

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .preprocess import load_train_data

RANDOM_STATE = 42
MODEL_PATH = "models/ridge_model.pkl"
TARGET_COL = "SalePrice"


def main() -> None:
    # Ensure model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Train it first with: python -m src.train_model"
        )

    print("Loading training data...")
    df = load_train_data()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in training data.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Same split strategy as used during training
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("Evaluating model...")
    y_pred = model.predict(X_val)

    # Older scikit-learn versions don't support squared=False, so we compute RMSE manually
    mse = mean_squared_error(y_val, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_val, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")


if __name__ == "__main__":
    main()
