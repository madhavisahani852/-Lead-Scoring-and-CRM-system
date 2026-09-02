import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from ml.config.feature_config import (
    NUMERIC_COLS,
    NOMINAL_COLS,
    ORDINAL_COLS,
    ORDINAL_MAPPINGS,
    DROP_COLS,
)


class OrdinalMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mappings=None):
        self.mappings = mappings or ORDINAL_MAPPINGS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        for col, mapping in self.mappings.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].map(mapping).fillna(0)
        return X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(list(self.mappings.keys()))
        return np.array(input_features)


class DaysSinceLastActivity(BaseEstimator, TransformerMixin):
    def __init__(self, as_of_date=None):
        self.as_of_date = pd.to_datetime(as_of_date) if as_of_date else pd.Timestamp.now()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        if 'last_activity_date' in X_out.columns:
            last_act = pd.to_datetime(X_out['last_activity_date'], errors='coerce')
            X_out['days_since_last_activity'] = (self.as_of_date - last_act).dt.days.fillna(999)
        return X_out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Computes domain-specific engineered features:
    - total_engagement_score
    - pricing_intent_ratio
    - days_since_last_activity
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = pd.DataFrame(X).copy()
        
        # 1. Total Engagement Score
        num_cols = ['website_visits', 'page_views', 'pricing_page_visits', 'email_opens', 'content_downloads']
        for col in num_cols:
            if col not in X_copy.columns:
                X_copy[col] = 0

        engagement_sum = (
            pd.to_numeric(X_copy['website_visits'], errors='coerce').fillna(0) +
            pd.to_numeric(X_copy['page_views'], errors='coerce').fillna(0) +
            pd.to_numeric(X_copy['pricing_page_visits'], errors='coerce').fillna(0) +
            pd.to_numeric(X_copy['email_opens'], errors='coerce').fillna(0) +
            pd.to_numeric(X_copy['content_downloads'], errors='coerce').fillna(0)
        )
        
        if 'demo_requested' in X_copy.columns:
            demo_num = (X_copy['demo_requested'].astype(str).str.lower() == 'yes').astype(int)
            engagement_sum += demo_num

        X_copy['total_engagement_score'] = engagement_sum
        
        # 2. Pricing Intent Ratio
        visits = pd.to_numeric(X_copy.get('website_visits', pd.Series(np.zeros(len(X_copy)))), errors='coerce').fillna(0)
        pricing_visits = pd.to_numeric(X_copy.get('pricing_page_visits', pd.Series(np.zeros(len(X_copy)))), errors='coerce').fillna(0)
        
        X_copy['pricing_intent_ratio'] = np.where(visits > 0, pricing_visits / visits, 0.0)
        
        if 'last_activity_date' in X_copy.columns:
            X_copy['last_activity_date'] = pd.to_datetime(X_copy['last_activity_date'], errors='coerce')
            current_date = pd.Timestamp.now()
            X_copy['days_since_last_activity'] = (current_date - X_copy['last_activity_date']).dt.days.fillna(999)
        else:
            X_copy['days_since_last_activity'] = 999
            
        return X_copy


def build_preprocessing_pipeline():
    """
    Builds and returns the scikit-learn ColumnTransformer preprocessing pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    ordinal_transformer = Pipeline(steps=[
        ('mapper', OrdinalMapper(mappings=ORDINAL_MAPPINGS)),
        ('scaler', StandardScaler()) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_COLS),
            ('nom', nominal_transformer, NOMINAL_COLS),
            ('ord', ordinal_transformer, ORDINAL_COLS),
            ('drop', 'drop', DROP_COLS)
        ],
        remainder='drop'
    )
    
    return preprocessor
