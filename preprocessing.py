from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from feature_config import NUMERIC_COLS, NOMINAL_COLS, ORDINAL_COLS, ORDINAL_MAPPINGS, DROP_COLS

class OrdinalMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mappings):
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
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
        X_out = X.copy()
        if 'last_activity_date' in X_out.columns:
            last_act = pd.to_datetime(X_out['last_activity_date'])
            X_out['days_since_last_activity'] = (self.as_of_date - last_act).dt.days
        return X_out

def build_preprocessing_pipeline():
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