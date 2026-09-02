# Machine Learning Subsystem & Team Ownership Guide

Welcome to the ML engineering subsystem for the **Lead Scoring & CRM Intelligence Tool**. This directory contains modular, production-ready machine learning pipelines for data preprocessing, training, model evaluation, and real-time inference.

---

## Directory Ownership Model

| Directory | Responsibility / Scope |
|---|---|
| `config/` | ML configuration, path constants, and feature definitions (`feature_config.py`). |
| `data/` | Dataset management (`raw/` unmodified datasets, `processed/` normalized training datasets). |
| `preprocessing/` | Data cleaning, missing-value imputation, ordinal mapping, and feature engineering transformers. |
| `training/` | Model training scripts (`train_baseline.py`, `train_random_forest.py`, `train_xgboost.py`, `train_models.py`). |
| `evaluation/` | Model evaluation routines (classification, ranking, calibration, and cross-model comparison). |
| `inference/` | Real-time and batch prediction engine (`predict.py`, `score_lead.py`). |
| `artifacts/` | Model binary storage (`models/`), metric summaries (`metrics/`), and markdown reports (`reports/`). |
| `notebooks/` | Jupyter notebooks for EDA, feature research, and performance visualization. |
| `tests/` | Automated unit and integration test suite located at project root (`tests/`). |
| `docs/` | Comprehensive technical documentation located at project root (`docs/`). |

---

## Development Rule

Follow these strict placement rules when adding new code to maintain code quality:

- **Research / Experimentation** → `notebooks/`
- **Reusable ML Logic & Transformers** → `ml/preprocessing/` or `ml/config/`
- **Training Pipelines** → `ml/training/`
- **Evaluation & Metrics** → `ml/evaluation/`
- **Prediction / Scoring** → `ml/inference/`
- **Automated Tests** → `tests/`
- **Technical Documentation** → `docs/`

---

## Quick Start ML Commands

### 1. Run Baseline Model Training
```bash
python ml/training/train_baseline.py
```

### 2. Run Full Multi-Model Training Pipeline
```bash
python ml/training/train_models.py
```

### 3. Run Lead Scoring Inference
```bash
python ml/inference/score_lead.py
```

### 4. Run Automated Tests
```bash
pytest tests/
```
