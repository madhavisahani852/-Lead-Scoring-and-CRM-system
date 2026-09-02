# Model Documentation

This document summarizes the machine learning models implemented in the Lead Scoring & CRM Intelligence pipeline.

---

## 1. Executive Summary & Model Overview

The Lead Scoring system evaluates incoming CRM leads and predicts their probability of converting into paying customers (`target = 1`). Leads are scored on a scale from **0 to 100** and categorized into priority tiers (**Hot**, **Warm**, **Cold**).

Three models are trained and evaluated in the pipeline:

1. **Baseline Model**: Logistic Regression
2. **Random Forest Classifier**: Ensemble of decision trees with bagging
3. **XGBoost / Gradient Boosting**: Sequential gradient boosting with regularized trees

---

## 2. Models & Training Procedures

### Data Splitting Strategy

- **Development Set (80%)**: Split into:
  - **Training Set (80% of Dev = 64% Total)**: Model parameter estimation.
  - **Validation Set (20% of Dev = 16% Total)**: Hyperparameter selection and threshold tuning.
- **Test Set (20% Total)**: Held-out evaluation to report unbiased final performance.
- Stratified sampling is enforced across all splits based on `target`.

### Preprocessing & Feature Pipeline

All models consume input data transformed via `build_preprocessing_pipeline()`:
- **Numeric Features**: Median Imputation + StandardScaler scaling.
- **Nominal Features**: Missing value constant filling ("Unknown") + One-Hot Encoding.
- **Ordinal Features**: Explicit ordinal integer mapping + StandardScaler scaling.
- **Dropped Columns**: Identifiers and raw metadata dropped to prevent data leakage.

---

## 3. Evaluated Models

### Model 1: Logistic Regression (Baseline)
- **Class**: `sklearn.linear_model.LogisticRegression`
- **Hyperparameters**: `max_iter=1000`, `random_state=42`
- **Role**: Baseline benchmark model. Lightweight, interpretable linear decision boundary.

### Model 2: Random Forest Classifier
- **Class**: `sklearn.ensemble.RandomForestClassifier`
- **Hyperparameters**: `n_estimators=100`, `max_depth=10`, `random_state=42`
- **Role**: Handles non-linear feature interactions and non-monotonic relationships.

### Model 3: XGBoost / Gradient Boosting
- **Class**: `xgboost.XGBClassifier` (or `sklearn.ensemble.HistGradientBoostingClassifier` fallback)
- **Hyperparameters**: `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `random_state=42`
- **Role**: High-capacity boosted ensemble optimizing log-loss with fine-grained probability outputs.

---

## 4. Model Artifact Locations

All trained models and evaluation outputs are persisted under `ml/artifacts/`:

- **Models**:
  - `ml/artifacts/models/logistic_regression_baseline.joblib`
  - `ml/artifacts/models/random_forest.joblib`
  - `ml/artifacts/models/xgboost.joblib`
- **Metrics**:
  - `ml/artifacts/metrics/baseline_metrics.json`
  - `ml/artifacts/metrics/random_forest_metrics.json`
  - `ml/artifacts/metrics/xgboost_metrics.json`
  - `ml/artifacts/metrics/model_comparison.json`
- **Reports**:
  - `ml/artifacts/reports/model_comparison.md`

---

## 5. Inference & Lead Scoring Engine

The inference module (`ml/inference/score_lead.py` and `predict_batch`) calculates:

$$\text{Lead Score} = \text{round}(\text{Conversion Probability} \times 100)$$

### Priority Tier Assignment:

- **Hot Lead**: `Lead Score ≥ 75` (Immediate sales call within 2 hours)
- **Warm Lead**: `40 ≤ Lead Score < 75` (Nurturing email campaign)
- **Cold Lead**: `Lead Score < 40` (Automated low-touch follow-up)

---

## 6. Known Limitations & Future Work

1. **Unresolved Leads**: Leads with status pending (`target = NaN`) are excluded during training and evaluated exclusively via inference.
2. **Concept Drift**: Retraining recommended on a monthly schedule as marketing campaigns shift.
