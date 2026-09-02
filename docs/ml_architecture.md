# ML Architecture & System Ownership Documentation

## End-to-End Pipeline Overview

```
   Raw Dataset (ml/data/raw/ & root)
                  │
                  ▼
   Cleaning & Preprocessing (ml/preprocessing/)
                  │
                  ▼
   Feature Engineering (ml/config/ & ml/preprocessing/)
                  │
                  ▼
   Model Training (ml/training/)
                  │
                  ▼
   Model Evaluation (ml/evaluation/)
                  │
                  ▼
   Artifact Storage (ml/artifacts/models/, metrics/, reports/)
                  │
                  ▼
   Inference & Scoring (ml/inference/)
                  │
                  ▼
   Backend & API Integration (backend/)
```

---

## Component Ownership Breakdown

| Stage | Responsible Directory | Key Module / Entry Point | Description |
|---|---|---|---|
| **Configuration** | `ml/config/` | `feature_config.py` | Feature definitions, column types, target definition, path resolution. |
| **Data Management** | `ml/data/` | `raw/`, `processed/` | Raw data landing zone and preprocessed dataset management. |
| **Preprocessing** | `ml/preprocessing/` | `preprocessing.py`, `validation.py` | Data cleaning pipelines, ordinal mapping, imputations, schema validation. |
| **Training** | `ml/training/` | `train_models.py`, `train_baseline.py`, `train_random_forest.py`, `train_xgboost.py` | Training routines for Baseline (Logistic Regression), Random Forest, and XGBoost. |
| **Evaluation** | `ml/evaluation/` | `classification.py`, `ranking.py`, `calibration.py`, `compare_models.py` | Metrics calculation, Precision@K ranking, probability calibration, model comparison. |
| **Artifacts** | `ml/artifacts/` | `models/`, `metrics/`, `reports/` | Versioned model binaries (`.joblib`), evaluation summaries (`.json`), comparison reports (`.md`). |
| **Inference** | `ml/inference/` | `predict.py`, `score_lead.py` | Batch lead prediction, single lead scoring (0-100 scale), priority classification (Hot/Warm/Cold). |
| **Notebooks** | `ml/notebooks/` | `01_data_exploration.ipynb`, `02_feature_analysis.ipynb`, `03_model_comparison.ipynb` | Research, EDA, feature analysis, and comparison visualization. |
| **Tests** | `tests/` | `test_preprocessing.py`, `test_training.py`, `test_inference.py` | Automated unit and integration test suite. |
