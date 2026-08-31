import pytest
import pandas as pd
import numpy as np
from preprocessing import build_preprocessing_pipeline, OrdinalMapper
from feature_config import DROP_COLS, NUMERIC_COLS, NOMINAL_COLS, ORDINAL_COLS

@pytest.fixture
def mock_leads_data():
    """
    Generates synthetic data mirroring the cleaned_leads.csv schema 
    and missingness profiles mapped in quality_metrics.json.
    """
    return pd.DataFrame({
        # Dropped Columns (Noise/Leakage identifiers)
        'lead_id': ['L0001', 'L0002', 'L0003', 'L0004'],
        'name': ['Rahul Sharma', 'Priya Verma', np.nan, 'Amit Patel'],
        'email': ['rahul@example.com', 'priya@example.com', 'amit@example.com', np.nan],
        'phone': ['+91-9876543210', np.nan, '+91-9123456780', '+91-9988776655'],
        'company': ['ABC Tech', 'XYZ Solutions', 'Nova Labs', 'Bright Systems'],
        'job_title': ['CEO', 'Manager', 'Director', 'VP'],
        'location': ['Pune', 'Mumbai', 'Delhi', 'Bangalore'],
        'campaign': ['CMP-123', np.nan, np.nan, 'CMP-456'],
        'email_valid': [True, True, True, False],
        'is_resolved': [True, True, False, True],
        
        # Numeric Columns (Testing median imputation)
        'company_size': [50, np.nan, 250, 10],
        'website_visits': [5, 0, np.nan, 12],
        'page_views': [10, 0, 15, np.nan],
        'pricing_page_visits': [1, 0, 2, 0],
        'email_opens': [2, 1, 0, 5],
        'form_completions': [1, 0, 0, 2],
        'content_downloads': [0, 0, 1, 3],
        'previous_interactions': [3, 1, 0, 5],
        'response_time_hours': [1.5, np.nan, 24.0, 0.5],
        'num_calls': [1, 0, 0, 3],
        'num_meetings': [0, 0, 1, 2],
        
        # Nominal Columns (Testing constant imputation & OneHot)
        'industry': ['SaaS', 'Finance', np.nan, 'Retail'],
        'lead_source': ['Website', 'Organic Search', 'Unknown', 'Paid Ads'],
        'product_interest': ['Starter Plan', 'Growth Plan', 'Enterprise Plan', 'Add-on Module'],
        'demo_requested': ['Yes', 'No', 'Unknown', np.nan],
        
        # Ordinal Columns (Testing custom mapper)
        'budget_range': ['10k-50k', 'Unknown', '50k-1L', '<10k'],
        
        # Target (Passthrough for scoring/training split)
        'target': [1.0, 0.0, np.nan, 1.0]
    })

def test_ordinal_mapper(mock_leads_data):
    """Validates that non-standard strings like 'Unknown' do not break the ordinal scale."""
    mappings = {
        'budget_range': {'Unknown': 0, '<10k': 1, '10k-50k': 2, '50k-1L': 3}
    }
    mapper = OrdinalMapper(mappings=mappings)
    transformed = mapper.fit_transform(mock_leads_data)
    
    # Expected: '10k-50k' -> 2, 'Unknown' -> 0, '50k-1L' -> 3, '<10k' -> 1
    assert transformed['budget_range'].tolist() == [2, 0, 3, 1]

def test_preprocessing_pipeline_imputation(mock_leads_data):
    """Ensures no NaN values leak through to the estimator (XGBoost requirement)."""
    pipeline = build_preprocessing_pipeline()
    transformed_data = pipeline.fit_transform(mock_leads_data)
    
    assert isinstance(transformed_data, np.ndarray)
    
    # The target column is passed through and retains a NaN in row 3 (unresolved lead)
    # We slice out the features (everything except the last column) to check imputation
    features_only = transformed_data[:, :-1]
    assert not np.isnan(features_only).any()

def test_dropped_columns(mock_leads_data):
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(mock_leads_data)
    
    feature_names = pipeline.get_feature_names_out()
    
    for col in DROP_COLS:
        # Split by '__' to remove transformer prefixes and check for exact matches
        # This prevents 'email' from falsely triggering on 'num__email_opens'
        assert not any(feature.split('__')[-1] == col for feature in feature_names), \
            f"Column '{col}' leaked into features!"