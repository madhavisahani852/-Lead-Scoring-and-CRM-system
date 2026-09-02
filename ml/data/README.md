# Data Management

This directory manages dataset storage for the Lead Scoring & CRM Machine Learning pipeline.

## Directory Layout

- `raw/`: Unmodified, original CSV datasets (e.g., `raw_leads.csv`). Treated as read-only source files.
- `processed/`: Cleaned, validated, and normalized datasets ready for model training and feature engineering (e.g., `cleaned_leads.csv`).

## Data Policy

1. **Raw preservation**: Never modify files inside `raw/`.
2. **Backward Compatibility**: `cleaned_leads.csv` at project root is supported as a fallback path so legacy scripts and root commands run seamlessly.
