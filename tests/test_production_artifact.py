import pytest
import joblib
import pandas as pd
import numpy as np
from ml.config import CLEANED_DATA_PATH, MODELS_DIR, REPORTS_DIR
from ml.inference.predict import (
    get_default_model_path,
    load_trained_pipeline,
    predict_lead_scores,
    predict_batch,
    score_unresolved_leads,
    assign_priority,
    validate_input_schema,
)


@pytest.fixture
def clean_data():
    if not CLEANED_DATA_PATH.exists():
        pytest.skip(f"Cleaned dataset missing at {CLEANED_DATA_PATH}")
    return pd.read_csv(CLEANED_DATA_PATH)


def test_production_artifact_exists_and_loads():
    model_path = MODELS_DIR / "best_model.joblib"
    assert model_path.exists(), "best_model.joblib does not exist"

    pipeline = joblib.load(model_path)
    assert hasattr(pipeline, "predict_proba"), "Pipeline missing predict_proba"
    assert hasattr(pipeline, "named_steps"), "Pipeline missing named_steps"
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_predict_lead_scores_output_format(clean_data):
    df_sample = clean_data.head(25)
    scored_df = predict_lead_scores(df_sample)

    assert len(scored_df) == 25
    assert "lead_id" in scored_df.columns
    assert "conversion_probability" in scored_df.columns
    assert "priority" in scored_df.columns
    assert "lead_score" in scored_df.columns

    probs = scored_df["conversion_probability"].values
    assert not np.isnan(probs).any()
    assert np.isfinite(probs).all()
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Verify descending sort order
    prob_list = scored_df["conversion_probability"].tolist()
    assert prob_list == sorted(prob_list, reverse=True)


def test_priority_assignment_logic():
    assert assign_priority(0.90, high_threshold=0.70, medium_threshold=0.40) == "High"
    assert assign_priority(0.70, high_threshold=0.70, medium_threshold=0.40) == "High"
    assert assign_priority(0.65, high_threshold=0.70, medium_threshold=0.40) == "Medium"
    assert assign_priority(0.40, high_threshold=0.70, medium_threshold=0.40) == "Medium"
    assert assign_priority(0.20, high_threshold=0.70, medium_threshold=0.40) == "Low"


def test_input_schema_validation():
    invalid_df = pd.DataFrame({"some_random_column": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing required lead columns"):
        predict_lead_scores(invalid_df)


def test_unresolved_lead_scoring(clean_data):
    scored_unresolved = score_unresolved_leads()
    assert len(scored_unresolved) == 183
    assert (REPORTS_DIR / "unresolved_lead_scores.csv").exists()

    probs = scored_unresolved["conversion_probability"].values
    assert not np.isnan(probs).any()
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
