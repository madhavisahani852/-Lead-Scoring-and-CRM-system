TARGET_COL = 'target'

DROP_COLS = [
    'lead_id', 'name', 'email', 'phone', 'company', 
    'job_title', 'location', 'campaign', 'email_valid',
    'is_resolved' # Used for filtering, not a feature
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