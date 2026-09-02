from pathlib import Path

# Centralized Path Resolutions
CONFIG_DIR = Path(__file__).resolve().parent
ML_DIR = CONFIG_DIR.parent
PROJECT_ROOT = ML_DIR.parent

DATA_DIR = ML_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Backward compatible dataset location lookup
RAW_DATA_PATH = RAW_DATA_DIR / "raw_leads.csv" if (RAW_DATA_DIR / "raw_leads.csv").exists() else PROJECT_ROOT / "raw_leads.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_leads.csv" if (PROCESSED_DATA_DIR / "cleaned_leads.csv").exists() else PROJECT_ROOT / "cleaned_leads.csv"

ARTIFACTS_DIR = ML_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Feature Definitions
TARGET_COL = 'target'

DROP_COLS = [
    'lead_id', 
    'name', 
    'email', 
    'phone', 
    'company', 
    'campaign', 
    'job_title', 
    'location', 
    'created_date', 
    'last_activity_date'
]

NUMERIC_COLS = [
    'company_size', 'website_visits', 'page_views', 
    'pricing_page_visits', 'email_opens', 'form_completions', 
    'content_downloads', 'previous_interactions', 
    'response_time_hours', 'num_calls', 'num_meetings'
]

NOMINAL_COLS = [
    'industry', 'lead_source', 'product_interest', 'demo_requested'
]

ORDINAL_COLS = ['budget_range']

DATETIME_COLS = ['created_date', 'last_activity_date']

ORDINAL_MAPPINGS = {
    'budget_range': {
        'Unknown': 0,
        '<10k': 1,
        '10k-50k': 2,
        '50k-1L': 3,
        '1L-5L': 4,
        '5L+': 5
    }
}
