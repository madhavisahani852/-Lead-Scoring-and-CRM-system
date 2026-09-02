import pandas as pd
from typing import Tuple, List
from ml.config.feature_config import (
    TARGET_COL,
    NUMERIC_COLS,
    NOMINAL_COLS,
    ORDINAL_COLS,
)


def validate_input_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates if an input DataFrame contains the necessary feature columns.
    Returns (is_valid, list_of_missing_columns).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    required_features = list(NUMERIC_COLS) + list(NOMINAL_COLS) + list(ORDINAL_COLS)
    missing_cols = [col for col in required_features if col not in df.columns]

    is_valid = len(missing_cols) == 0
    return is_valid, missing_cols


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepares resolved leads for supervised ML training.
    Filters out unresolved leads (target = NaN).
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"The dataset must contain the target column '{TARGET_COL}'.")

    # Keep resolved records only
    resolved = df[df[TARGET_COL].notna()].copy()
    if len(resolved) == 0:
        raise ValueError("No resolved records (non-null target) found in dataset.")

    resolved[TARGET_COL] = resolved[TARGET_COL].astype(int)

    unique_targets = sorted(resolved[TARGET_COL].unique().tolist())
    if unique_targets != [0, 1]:
        raise ValueError(f"Target column must contain binary values [0, 1]. Found: {unique_targets}")

    # Check required feature columns
    is_valid, missing_cols = validate_input_dataframe(resolved)
    if not is_valid:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    X = resolved.drop(columns=[TARGET_COL], errors="ignore").copy()
    y = resolved[TARGET_COL].copy()

    return X, y
