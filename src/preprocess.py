"""Utility functions for loading and preprocessing the Ames Housing / Kaggle House Prices dataset."""

from __future__ import annotations

from typing import Tuple, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def load_train_data(path: str = "data/train.csv") -> pd.DataFrame:
    """Load the Kaggle training data."""
    return pd.read_csv(path)


def load_test_data(path: str = "data/test.csv") -> pd.DataFrame:
    """Load the Kaggle test data."""
    return pd.read_csv(path)


def build_preprocessor(
    df: pd.DataFrame,
    target_col: str = "SalePrice",
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer that:
    - imputes numeric features with the median
    - imputes categorical features with the most frequent value
    (encoding and scaling are handled in the training script)
    """
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    if target_col in numeric_features:
        numeric_features.remove(target_col)

    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = SimpleImputer(strategy="most_frequent")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features
