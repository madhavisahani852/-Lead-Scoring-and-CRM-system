import sys
from pathlib import Path

# Ensure project root is in sys.path for standalone script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Union, List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np

from ml.config.feature_config import (
    CLEANED_DATA_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    NUMERIC_COLS,
    NOMINAL_COLS,
    ORDINAL_COLS,
    DROP_COLS,
)

DEFAULT_HIGH_THRESHOLD = 0.70
DEFAULT_MEDIUM_THRESHOLD = 0.40


def get_default_model_path() -> Path:
    """
    Finds and returns the canonical best model artifact path.
    Prefers best_model.joblib, falls back to xgboost_tuned.joblib or baseline models.
    """
    candidates = [
        MODELS_DIR / "best_model.joblib",
        MODELS_DIR / "xgboost_tuned.joblib",
        MODELS_DIR / "random_forest_baseline.joblib",
        MODELS_DIR / "xgboost_baseline.joblib",
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


def validate_input_schema(df: pd.DataFrame) -> None:
    """
    Validates that the input DataFrame contains all required business feature columns.
    Raises ValueError if any required features are missing.
    """
    required_cols = list(NUMERIC_COLS) + list(NOMINAL_COLS) + list(ORDINAL_COLS)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required lead columns: {missing_cols}")


def assign_priority(
    probability: float,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD
) -> str:
    """
    Assigns transparent CRM priority band based on conversion probability.
    - High:   probability >= high_threshold (default 0.70)
    - Medium: medium_threshold <= probability < high_threshold (default 0.40 - 0.6999)
    - Low:    probability < medium_threshold (default < 0.40)
    """
    if probability >= high_threshold:
        return "High"
    elif probability >= medium_threshold:
        return "Medium"
    else:
        return "Low"


def predict_lead_scores(
    leads_df: pd.DataFrame,
    model_path: Union[str, Path] = None,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD
) -> pd.DataFrame:
    """
    Production inference function for lead scoring.
    Accepts raw leads DataFrame, applies pre-fitted production pipeline,
    computes conversion probabilities, priority tiers, lead scores (0-100),
    and sorts results descending by conversion_probability.
    """
    validate_input_schema(leads_df)
    pipeline = load_trained_pipeline(model_path)

    X_in = leads_df.copy()
    if 'target' in X_in.columns:
        X_in = X_in.drop(columns=['target'], errors='ignore')

    # Ensure optional drop columns exist with NaNs if absent to prevent pipeline missing column errors
    for col in DROP_COLS:
        if col not in X_in.columns:
            X_in[col] = np.nan

    probabilities = pipeline.predict_proba(X_in)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    lead_scores = np.round(probabilities * 100).astype(int)

    result_df = leads_df.copy()
    result_df['conversion_probability'] = np.round(probabilities, 4)
    result_df['lead_score'] = lead_scores
    result_df['predicted_conversion'] = predictions

    # Assign priority bands using configurable thresholds
    priorities = [
        assign_priority(p, high_threshold=high_threshold, medium_threshold=medium_threshold)
        for p in probabilities
    ]
    result_df['priority'] = priorities

    # Map priority to legacy Hot/Warm/Cold priority names if needed for test_inference
    legacy_priorities = [
        "Hot" if p >= 0.75 else ("Warm" if p >= 0.40 else "Cold")
        for p in probabilities
    ]
    result_df['lead_priority'] = legacy_priorities

    # Sort results by conversion_probability descending
    result_df = result_df.sort_values(by='conversion_probability', ascending=False).reset_index(drop=True)

    return result_df


def predict_batch(
    leads_df: pd.DataFrame,
    model_path: Union[str, Path] = None
) -> pd.DataFrame:
    """
    Backward-compatible batch prediction function.
    Safely fills missing features with NaN if needed.
    """
    X_in = leads_df.copy()
    required_cols = list(NUMERIC_COLS) + list(NOMINAL_COLS) + list(ORDINAL_COLS)
    for col in required_cols:
        if col not in X_in.columns:
            X_in[col] = np.nan
    return predict_lead_scores(X_in, model_path=model_path)


def score_single_lead(
    lead_dict: Dict[str, Any],
    model_path: Union[str, Path] = None
) -> Dict[str, Any]:
    """
    Scores a single lead dictionary and returns a dictionary of scoring metrics.
    """
    df_single = pd.DataFrame([lead_dict])
    scored_df = predict_batch(df_single, model_path=model_path)
    return scored_df.iloc[0].to_dict()


def score_unresolved_leads(
    data_path: Union[str, Path] = None,
    output_path: Union[str, Path] = None,
    model_path: Union[str, Path] = None
) -> pd.DataFrame:
    """
    Loads dataset, filters unresolved leads (target is NaN),
    scores them using best_model.joblib, and exports to unresolved_lead_scores.csv.
    """
    if data_path is None:
        data_path = CLEANED_DATA_PATH
    if output_path is None:
        output_path = REPORTS_DIR / "unresolved_lead_scores.csv"

    print("=" * 60)
    print("SCORING UNRESOLVED LEADS (target = NaN)")
    print("=" * 60)
    print(f"Loading dataset from: {data_path}")

    df = pd.read_csv(data_path)
    unresolved_df = df[df['target'].isna()].copy()
    unresolved_count = len(unresolved_df)
    print(f"Identified unresolved leads count: {unresolved_count}")

    if unresolved_count == 0:
        print("Warning: No unresolved leads found in dataset.")
        return pd.DataFrame()

    scored_df = predict_lead_scores(unresolved_df, model_path=model_path)

    # Ensure required report columns are present at the beginning
    cols = ['lead_id', 'conversion_probability', 'priority', 'lead_score'] + [
        c for c in scored_df.columns if c not in ['lead_id', 'conversion_probability', 'priority', 'lead_score']
    ]
    report_df = scored_df[cols]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)
    print(f"Successfully scored {len(report_df)} unresolved leads.")
    print(f"Saved unresolved lead scores report to: {output_path}")

    return report_df


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PRODUCTION INFERENCE DEMO & UNRESOLVED LEAD SCORING")
    print("=" * 60)

    # 1. Sample Raw Lead Inference
    df_sample = pd.read_csv(CLEANED_DATA_PATH).head(5)
    scored_sample = predict_lead_scores(df_sample)
    print("\nTop 5 Scored Sample Leads:")
    print(scored_sample[['lead_id', 'conversion_probability', 'priority', 'lead_score']].to_string(index=False))

    # 2. Score Unresolved Leads
    score_unresolved_leads()
