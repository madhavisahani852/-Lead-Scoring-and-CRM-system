# Final Model Performance & Selection Report

**Generated At**: `2026-09-02T08:35:39Z`  
**Phase**: Phase 6 Final Model Comparison & Model Selection  
**Status**: Completed

---

## 1. Dataset & Split Verification (Fairness Check)

All four candidate models were evaluated under strict, identical empirical conditions:

- **Dataset**: `cleaned_leads.csv`
- **Total Resolved Leads**: 1,017 records (622 Converted [`target=1`], 395 Lost [`target=0`])
- **Dataset Split (`random_state=42`)**:
  - **Training Set**: 650 records (64%)
  - **Validation Set**: 163 records (16%)
  - **Test Set (Held-Out)**: 204 records (20%)
- **Feature Schema**: Shared `ml/config/feature_config.py` via `build_preprocessing_pipeline()` (36 transformed features).
- **Fairness Status**: **Directly Comparable** (Zero data leakage, identical train/val/test splits).

---

## 2. Models Evaluated

1. **Logistic Regression Baseline** (`ml/artifacts/models/logistic_regression_baseline.joblib`)
2. **Random Forest Baseline** (`ml/artifacts/models/random_forest_baseline.joblib`)
3. **XGBoost Baseline** (`ml/artifacts/models/xgboost_baseline.joblib`)
4. **Tuned XGBoost** (`ml/artifacts/models/xgboost_tuned.joblib`)

---

## 3. Classification Performance Comparison (Held-Out Test Set, $N=204$)

| Metric | Direction | Logistic Reg | Random Forest | XGBoost Baseline | Tuned XGBoost | Winning Model |
|---|---|---|---|---|---|---|
| **ROC-AUC** | Higher ↑ | 0.7722 | 0.7875 | 0.7841 | **0.8086** | **Tuned XGBoost** |
| **PR-AUC** | Higher ↑ | 0.8313 | 0.8503 | 0.8409 | **0.8654** | **Tuned XGBoost** |
| **Accuracy** | Higher ↑ | 0.7255 | 0.7304 | 0.7157 | **0.7402** | **Tuned XGBoost** |
| **Precision** | Higher ↑ | 0.7656 | **0.7717** | 0.7538 | 0.7710 | **Random Forest Baseline** |
| **Recall** | Higher ↑ | 0.7903 | 0.7903 | 0.7903 | **0.8145** | **Tuned XGBoost** |
| **F1 Score** | Higher ↑ | 0.7778 | 0.7809 | 0.7717 | **0.7922** | **Tuned XGBoost** |
| **Log Loss** | Lower ↓ | 0.5569 | 0.5493 | 0.5541 | **0.5193** | **Tuned XGBoost** |

---

## 4. Lead Prioritization & Ranking Comparison ($N=204$, Baseline Conv. Rate = 60.78%)

| Segment | Metric | Logistic Reg | Random Forest | XGBoost Baseline | Tuned XGBoost | Winning Model |
|---|---|---|---|---|---|---|
| **Top 10 Leads** ($K=10$) | **Precision@10** | 0.9000 | **1.0000** | 0.9000 | **1.0000** | **Tuned XGBoost / Random Forest** |
| | **Recall@10** | 0.0726 | **0.0806** | 0.0726 | **0.0806** | **Tuned XGBoost / Random Forest** |
| | **Conversion Rate@10** | 0.9000 | **1.0000** | 0.9000 | **1.0000** | **Tuned XGBoost / Random Forest** |
| | **Lift@10** | 1.4806x | **1.6452x** | 1.4806x | **1.6452x** | **Tuned XGBoost / Random Forest** |
| **Top 20 Leads** ($K=20$) | **Precision@20** | 0.9000 | **0.9500** | **0.9500** | **0.9500** | **Three-way Tie (RF, XGB Base, Tuned)** |
| | **Recall@20** | 0.1452 | **0.1532** | **0.1532** | **0.1532** | **Three-way Tie (RF, XGB Base, Tuned)** |
| | **Conversion Rate@20** | 0.9000 | **0.9500** | **0.9500** | **0.9500** | **Three-way Tie (RF, XGB Base, Tuned)** |
| | **Lift@20** | 1.4806x | **1.5629x** | **1.5629x** | **1.5629x** | **Three-way Tie (RF, XGB Base, Tuned)** |
| **Top 20% Leads** ($K=41$) | **Precision@41** | 0.9024 | **0.9512** | **0.9512** | 0.9268 | **Random Forest / XGB Baseline** |
| | **Recall@41** | 0.2984 | **0.3145** | **0.3145** | 0.3065 | **Random Forest / XGB Baseline** |
| | **Conversion Rate@41** | 0.9024 | **0.9512** | **0.9512** | 0.9268 | **Random Forest / XGB Baseline** |
| | **Lift@41** | 1.4847x | **1.5649x** | **1.5649x** | 1.5248x | **Random Forest / XGB Baseline** |

---

## 5. Model-by-Model Analysis

1. **Logistic Regression Baseline**:
   - Simple linear decision boundary. Serves as a solid baseline (ROC-AUC = 0.7722), but struggles with non-linear feature interactions.
2. **Random Forest Baseline**:
   - Exceptionally strong ranking model. Achieved **0.9512 Precision@41** and **1.0000 Precision@10**. ROC-AUC = 0.7875.
3. **XGBoost Baseline**:
   - Good default gradient boosting baseline (ROC-AUC = 0.7841). Outperformed by tuned variant.
4. **Tuned XGBoost**:
   - Best overall classification model. Achieved highest ROC-AUC (**0.8086**), PR-AUC (**0.8654**), F1 (**0.7922**), and lowest Log Loss (**0.5193**), with **100% precision in Top 10 leads**.

---

## 6. Final Selected Model & Category Breakdown

- **BEST_CLASSIFICATION_MODEL**: **Tuned XGBoost**
- **BEST_RANKING_MODEL**: **Random Forest Baseline**
- **BEST_OVERALL_MODEL**: **Tuned XGBoost**

### Selection Rationale

**Tuned XGBoost** is selected as the **BEST_OVERALL_MODEL** because lead scoring relies heavily on well-calibrated probability estimates across the entire lead spectrum:
1. **Highest Discrimination Power**: Achieves the overall highest ROC-AUC (**0.8086**) and PR-AUC (**0.8654**).
2. **Lowest Prediction Error**: Yields the lowest log loss (**0.5193** vs 0.5493 for Random Forest).
3. **Flawless High-Priority Precision**: Achieves 100% precision for the Top 10 leads ($K=10$, Precision@10 = 1.0000, Lift = 1.6452x).

---

## 7. Canonical Artifact Paths

- **Canonical Best Model Pipeline**: [`ml/artifacts/models/best_model.joblib`](file:///c:/Developement/-Lead-Scoring-and-CRM-system/ml/artifacts/models/best_model.joblib)
- **Model Metadata**: [`ml/artifacts/models/model_metadata.json`](file:///c:/Developement/-Lead-Scoring-and-CRM-system/ml/artifacts/models/model_metadata.json)
- **Comparison JSON**: [`ml/artifacts/metrics/final_model_comparison.json`](file:///c:/Developement/-Lead-Scoring-and-CRM-system/ml/artifacts/metrics/final_model_comparison.json)
- **Final Comparison Report**: [`ml/artifacts/reports/final_model_comparison.md`](file:///c:/Developement/-Lead-Scoring-and-CRM-system/ml/artifacts/reports/final_model_comparison.md)
- **Model Card**: [`ml/artifacts/reports/model_card.md`](file:///c:/Developement/-Lead-Scoring-and-CRM-system/ml/artifacts/reports/model_card.md)
