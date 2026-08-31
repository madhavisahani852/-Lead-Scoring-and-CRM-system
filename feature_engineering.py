import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # 1. Total Engagement Score
        engagement_cols = [
            'website_visits', 'page_views', 'pricing_page_visits', 'demo_requests', 
            'email_interactions', 'content_downloads'
        ]
        
        for col in engagement_cols:
            if col not in X_copy.columns:
                X_copy[col] = 0
                
        X_copy['total_engagement_score'] = X_copy[engagement_cols].fillna(0).sum(axis=1)
        
        # 2. Pricing Intent Ratio
        visits = X_copy.get('website_visits', pd.Series(np.zeros(len(X_copy))))
        pricing_visits = X_copy.get('pricing_page_visits', pd.Series(np.zeros(len(X_copy))))
        
        visits = visits.fillna(0)
        pricing_visits = pricing_visits.fillna(0)
        
        X_copy['pricing_intent_ratio'] = np.where(visits > 0, pricing_visits / visits, 0.0)
        
        if 'last_activity_date' in X_copy.columns:
            X_copy['last_activity_date'] = pd.to_datetime(X_copy['last_activity_date'], errors='coerce')
            current_date = pd.Timestamp.now()
            
            # Calculate delta in days
            X_copy['days_since_last_activity'] = (current_date - X_copy['last_activity_date']).dt.days
            
            X_copy['days_since_last_activity'] = X_copy['days_since_last_activity'].fillna(999)
        else:
            X_copy['days_since_last_activity'] = 999
            
        return X_copy
