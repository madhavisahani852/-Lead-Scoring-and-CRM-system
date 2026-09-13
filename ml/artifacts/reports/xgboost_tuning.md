# XGBoost Hyperparameter Tuning Report

**Generated At**: `2026-09-06T18:13:24Z`  
**Phase**: Phase 5 XGBoost Hyperparameter Tuning  
**Status**: Completed

---

## 1. Dataset & Split Information

- **Dataset Path**: `cleaned_leads.csv`
- **Total Resolved Records**: 1017 (100% binary outcome: `target` in [0, 1])
- **Reproducible Dataset Split (`random_state=42`)**:
  - **Training Set**: 650 records (64% of resolved dataset)
  - **Validation Set**: 163 records (16% of resolved dataset)
  - **Test Set (Held-Out)**: 204 records (20% of resolved dataset)
- **Transformed Feature Count**: 36 features (via `build_preprocessing_pipeline()`)

---

## 2. Tuning Methodology & Search Space

- **Method**: `RandomizedSearchCV` (scikit-learn)
- **CV Folds**: 3-fold cross-validation
- **Search Iterations**: 20 parameter combinations
- **Scoring Metric**: `roc_auc`
- **Data Leakage Safeguards**: Search was executed **exclusively on the training dataset (650 records)**. Preprocessing transformers were fit strictly inside CV folds. The test set remained completely untouched throughout parameter selection.

### Hyperparameter Search Space

```python
{
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [1, 1.5, 2, 5]
}
```

---

## 3. Best Hyperparameters & CV Score

- **Best Cross-Validation ROC-AUC**: **0.7741**
- **Optimal Hyperparameter Configuration**:

```json
{
  "subsample": 0.7,
  "reg_lambda": 1.5,
  "reg_alpha": 0.01,
  "n_estimators": 100,
  "min_child_weight": 1,
  "max_depth": 2,
  "learning_rate": 0.1,
  "gamma": 0,
  "colsample_bytree": 0.8
}
```

---

## 4. Evaluation Results

### Validation Set Results ($N=163$)

- **ROC-AUC**: 0.8281
- **PR-AUC**: 0.8717
- **Accuracy**: 0.7607
- **Precision**: 0.7632
- **Recall**: 0.8788
- **F1 Score**: 0.8169
- **Log Loss**: 0.5021

### Test Set Results ($N=204$)

- **ROC-AUC**: 0.7932
- **PR-AUC**: 0.8534
- **Accuracy**: 0.7157
- **Precision**: 0.7578
- **Recall**: 0.7823
- **F1 Score**: 0.7698
- **Log Loss**: 0.5324

---

## 5. Ranking & Lead Prioritization Performance

| Segment | Precision@K | Recall@K | Conversion Rate | Lift@K | Conversions Found |
|---|---|---|---|---|---|
| **Top 10 Leads** ($K=10$) | 1.0000 | 0.0806 | 1.0000 | 1.6452x | 10 |
| **Top 20 Leads** ($K=20$) | 0.9000 | 0.1452 | 0.9000 | 1.4806x | 18 |
| **Top 20% Leads** ($K=41$) | 0.9268 | 0.3065 | 0.9268 | 1.5248x | 38 |

---

## 6. Comparison with XGBoost Baseline

Comparison of tuned XGBoost vs untuned XGBoost baseline on held-out test data:

| Metric | XGBoost Baseline | XGBoost Tuned | Difference (Tuned - Baseline) | Status |
|---|---|---|---|---|
| **ROC-AUC** | 0.7869 | 0.7932 | +0.0064 | Improved ↑ |
| **PR-AUC** | 0.8388 | 0.8534 | +0.0146 | Improved ↑ |
| **F1 Score** | 0.7812 | 0.7698 | -0.0114 | Worsened/Equal |
| **Log Loss** | 0.5479 | 0.5324 | -0.0155 | Improved ↓ |

---

## 7. Observations & Safeguards

1. **Hyperparameter Selection Integrity**: Cross-validation tuning strictly operated on `X_train`. The test set was evaluated exactly once after pipeline refitting.
2. **Model Selection Deferred**: Final production model selection across Logistic Regression, Random Forest, XGBoost Baseline, and XGBoost Tuned will take place in the next phase.
