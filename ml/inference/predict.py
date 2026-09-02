import sys
from pathlib import Path

# Ensure project root is in sys.path for standalone script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Union, List, Dict, Any
import joblib
import pandas as pd
import numpy as np

from ml.config.feature_config import (
    MODELS_DIR,
    NUMERIC_COLS,
    NOMINAL_COLS,
    ORDINAL_COLS,
    DROP_COLS,
)


def get_default_model_path() -> Path:
    """
    Finds and returns the best available trained model path.
    Prefers random_forest.joblib or xgboost.joblib, falls back to logistic_regression_baseline.joblib.
    """
    candidates = [
        MODELS_DIR / "random_forest.joblib",
        MODELS_DIR / "xgboost.joblib",
        MODELS_DIR / "logistic_regression_baseline.joblib",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No trained model artifact found in {MODELS_DIR}. Run training pipeline first.")


def load_trained_pipeline(model_path: Union[str, Path] = None):
    """
    Loads a fitted scikit-learn pipeline (preprocessor + model).
    """
    if model_path is None:
        model_path = get_default_model_path()
    return joblib.load(model_path)


def predict_batch(
    leads_df: pd.DataFrame,
    model_path: Union[str, Path] = None
) -> pd.DataFrame:
    """
    Generates conversion probabilities and lead scores (0-100) for a DataFrame of leads.
    Safely handles missing input columns.
    """
    pipeline = load_trained_pipeline(model_path)

    X_in = leads_df.copy()
    if 'target' in X_in.columns:
        X_in = X_in.drop(columns=['target'], errors='ignore')

    # Ensure all expected feature columns exist in X_in
    expected_cols = list(NUMERIC_COLS) + list(NOMINAL_COLS) + list(ORDINAL_COLS) + list(DROP_COLS)
    for col in expected_cols:
        if col not in X_in.columns:
            X_in[col] = np.nan

    probabilities = pipeline.predict_proba(X_in)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    lead_scores = np.round(probabilities * 100).astype(int)

    result_df = leads_df.copy()
    result_df['conversion_probability'] = np.round(probabilities, 4)
    result_df['lead_score'] = lead_scores
    result_df['predicted_conversion'] = predictions

    # Priority Tier assignment
    conditions = [
        result_df['lead_score'] >= 75,
        result_df['lead_score'] >= 40,
    ]
    choices = ['Hot', 'Warm']
    result_df['lead_priority'] = np.select(conditions, choices, default='Cold')

    return result_df
