

from __future__ import annotations

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocess import load_train_data


RANDOM_STATE = 42
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ridge_model.pkl")
TARGET_COL = "SalePrice"


def build_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, ColumnTransformer]:
    """Build a preprocessing + RidgeCV model pipeline."""

    # Separate numeric and categorical feature names
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove target from numeric_features if present
    if TARGET_COL in numeric_features:
        numeric_features.remove(TARGET_COL)

    # Pipelines for each type
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # RidgeCV with a range of alphas
    alphas = np.logspace(-3, 3, 13)
    model = RidgeCV(alphas=alphas, scoring="neg_root_mean_squared_error")

    # Full pipeline
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipe, preprocessor


def main() -> None:
    print("Loading training data...")
    df = load_train_data()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in training data.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("Building pipeline...")
    pipe, _ = build_pipeline(df)

    print("Training RidgeCV model...")
    pipe.fit(X_train, y_train)

    # Make sure models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the whole pipeline (preprocessing + model)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    # Quick validation score (R²)
    score = pipe.score(X_val, y_val)
    print(f"Validation R²: {score:.4f}")


if __name__ == "__main__":
    main()
