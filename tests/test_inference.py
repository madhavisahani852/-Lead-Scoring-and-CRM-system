import pytest
import pandas as pd
from ml.inference import score_single_lead, predict_batch, get_default_model_path
from ml.config import CLEANED_DATA_PATH


@pytest.fixture
def sample_lead_dict():
    return {
        "lead_id": "L_TEST",
        "company_size": 150,
        "industry": "Tech",
        "lead_source": "Website",
        "product_interest": "Enterprise",
        "budget_range": "10k-50k",
        "website_visits": 8,
        "page_views": 20,
        "pricing_page_visits": 2,
        "demo_requested": "Yes",
        "email_opens": 5,
        "form_completions": 1,
        "content_downloads": 2,
        "previous_interactions": 3,
        "response_time_hours": 5.0,
        "num_calls": 2,
        "num_meetings": 1,
    }


def test_single_lead_scoring(sample_lead_dict):
    result = score_single_lead(sample_lead_dict)
    assert "lead_score" in result
    assert 0 <= result["lead_score"] <= 100
    assert 0.0 <= result["conversion_probability"] <= 1.0
    assert result["predicted_conversion"] in [0, 1]
    assert result["lead_priority"] in ["Hot", "Warm", "Cold"]


def test_batch_lead_prediction():
    if not CLEANED_DATA_PATH.exists():
        pytest.skip(f"Cleaned dataset missing at {CLEANED_DATA_PATH}")

    df = pd.read_csv(CLEANED_DATA_PATH).head(10)
    scored_df = predict_batch(df)

    assert "lead_score" in scored_df.columns
    assert "conversion_probability" in scored_df.columns
    assert "lead_priority" in scored_df.columns
    assert (scored_df["lead_score"] >= 0).all() and (scored_df["lead_score"] <= 100).all()


def test_incomplete_lead_scoring(sample_lead_dict):
    incomplete_lead = sample_lead_dict.copy()
    incomplete_lead.pop("company_size")
    incomplete_lead.pop("website_visits")

    result = score_single_lead(incomplete_lead)
    assert 0 <= result["lead_score"] <= 100
