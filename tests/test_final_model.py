import pytest
import joblib
import json
import pandas as pd
import numpy as np
from ml.config import CLEANED_DATA_PATH, ARTIFACTS_DIR, MODELS_DIR, METRICS_DIR, REPORTS_DIR
from ml.evaluation.final_model_selection import run_final_model_selection


@pytest.fixture
def clean_data_file():
    if not CLEANED_DATA_PATH.exists():
        pytest.skip(f"Cleaned dataset missing at {CLEANED_DATA_PATH}")
    return CLEANED_DATA_PATH


def test_final_model_selection_artifacts(clean_data_file):
    results = run_final_model_selection()

    best_model_path = MODELS_DIR / "best_model.joblib"
    metadata_path = MODELS_DIR / "model_metadata.json"
    comparison_json_path = METRICS_DIR / "final_model_comparison.json"
    comparison_report_path = REPORTS_DIR / "final_model_comparison.md"
    model_card_path = REPORTS_DIR / "model_card.md"

    # Verify canonical final files exist
    assert best_model_path.exists(), "best_model.joblib missing"
    assert metadata_path.exists(), "model_metadata.json missing"
    assert comparison_json_path.exists(), "final_model_comparison.json missing"
    assert comparison_report_path.exists(), "final_model_comparison.md missing"
    assert model_card_path.exists(), "model_card.md missing"

    # Verify best_model.joblib pipeline loading and prediction
    pipeline = joblib.load(best_model_path)
    assert hasattr(pipeline, "predict_proba"), "Pipeline missing predict_proba"
    assert hasattr(pipeline, "named_steps"), "Pipeline missing named_steps"
    assert "preprocessor" in pipeline.named_steps, "Preprocessor step missing"
    assert "classifier" in pipeline.named_steps, "Classifier step missing"

    # Verify probability generation on sample data
    df_sample = pd.read_csv(clean_data_file).head(15)
    probs = pipeline.predict_proba(df_sample)[:, 1]

    assert len(probs) == 15
    assert not np.isnan(probs).any(), "NaN in probabilities"
    assert np.isfinite(probs).all(), "Inf in probabilities"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities out of range [0, 1]"

    # Verify metadata contents
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["model_name"] == "Tuned XGBoost"
    assert meta["resolved_records"] == 1017
    assert meta["test_size"] == 204
    assert "selection_reason" in meta
