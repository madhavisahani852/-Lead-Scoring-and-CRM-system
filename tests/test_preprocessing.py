import pytest
import pandas as pd
import numpy as np

from ml.config import (
    DROP_COLS,
    NUMERIC_COLS,
    NOMINAL_COLS,
    ORDINAL_COLS,
    CLEANED_DATA_PATH,
)
from ml.preprocessing import (
    build_preprocessing_pipeline,
    OrdinalMapper,
    FeatureEngineer,
    validate_input_dataframe,
    prepare_training_data,
)


@pytest.fixture
def mock_leads_data():
    """Provides a small mock dataframe mirroring the cleaned dataset schema."""
    data = {
        'lead_id': ['L001', 'L002', 'L003', 'L004'],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'diana@example.com'],
        'email_valid': [True, True, True, False],
        'phone': ['1234567890', '0987654321', '1112223333', '4445556666'],
        'company': ['Acme Corp', 'Beta LLC', 'Gamma Inc', 'Delta Co'],
        'industry': ['SaaS', 'Finance', 'Retail', 'SaaS'],
        'company_size': [50, None, 200, 10],
        'job_title': ['Manager', 'Director', 'Analyst', 'CEO'],
        'location': ['New York', 'London', 'Berlin', 'Tokyo'],
        'lead_source': ['Website', 'Website', 'Organic Search', 'Referral'],
        'campaign': ['Spring_2026', 'Summer_2026', 'Fall_2026', 'Winter_2026'],
        'product_interest': ['Add-on Module', 'Core Platform', 'Enterprise', 'Core Platform'],
        'budget_range': ['<10k', '10k-50k', '50k-1L', 'Unknown'],
        'website_visits': [5, 12, None, 3],
        'page_views': [15, 30, 4, None],
        'pricing_page_visits': [1, 3, 0, 5],
        'demo_requested': ['Yes', 'No', 'Unknown', 'Yes'],
        'email_opens': [4, 8, 2, None],
        'form_completions': [1, 2, 0, 1],
        'content_downloads': [0, 1, 3, 2],
        'previous_interactions': [2, 5, 1, 0],
        'response_time_hours': [12.5, 4.0, None, 24.0],
        'num_calls': [2, 4, 1, 3],
        'num_meetings': [1, 0, 2, 1],
        'created_date': ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04'],
        'last_activity_date': ['2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08'],
        'is_resolved': [True, True, False, True],
        'target': [1, 0, None, 1]
    }
    return pd.DataFrame(data)


def test_ordinal_mapper(mock_leads_data):
    mappings = {'budget_range': {'Unknown': 0, '<10k': 1, '10k-50k': 2, '50k-1L': 3}}
    mapper = OrdinalMapper(mappings=mappings)
    transformed = mapper.fit_transform(mock_leads_data)
    assert transformed['budget_range'].tolist() == [1, 2, 3, 0]


def test_feature_engineer(mock_leads_data):
    fe = FeatureEngineer()
    engineered = fe.fit_transform(mock_leads_data)
    assert 'total_engagement_score' in engineered.columns
    assert 'pricing_intent_ratio' in engineered.columns
    assert 'days_since_last_activity' in engineered.columns


def test_preprocessing_pipeline_imputation(mock_leads_data):
    """Ensures no NaN values leak through to the estimator."""
    pipeline = build_preprocessing_pipeline()
    X_mock = mock_leads_data.drop(columns=['target', 'is_resolved'], errors='ignore')
    transformed_data = pipeline.fit_transform(X_mock)
    
    assert isinstance(transformed_data, np.ndarray)
    assert not np.isnan(transformed_data).any()


def test_dropped_columns(mock_leads_data):
    X_mock = mock_leads_data.drop(columns=['target', 'is_resolved'], errors='ignore')
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X_mock)
    
    feature_names = pipeline.get_feature_names_out()
    for col in DROP_COLS:
        assert not any(feature.split('__')[-1] == col for feature in feature_names), \
            f"Column '{col}' leaked into features!"


def test_validation_helpers(mock_leads_data):
    is_valid, missing = validate_input_dataframe(mock_leads_data)
    assert is_valid
    assert len(missing) == 0

    X, y = prepare_training_data(mock_leads_data)
    assert len(X) == 3  # row 3 had target=NaN
    assert len(y) == 3


def test_real_cleaned_dataset_if_available():
    if CLEANED_DATA_PATH.exists():
        df = pd.read_csv(CLEANED_DATA_PATH)
        X, y = prepare_training_data(df)
        pipeline = build_preprocessing_pipeline()
        transformed = pipeline.fit_transform(X)

        assert not np.isnan(transformed).any()
        assert np.isfinite(transformed).all()
        assert transformed.shape[0] == len(X)
