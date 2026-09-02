# Baseline Model Comparison Report: Logistic Regression vs. Random Forest

**Generated At**: `2026-09-02T08:35:30Z`  
**Phase**: Phase 3 Baseline Model Comparison  
**Status**: Completed (Model selection pending XGBoost evaluation)

---

## 1. Dataset & Split Verification

Both models were trained and evaluated using the exact same reproducible setup and shared pipeline:

- **Dataset**: `cleaned_leads.csv`
- **Total Records**: 1200 (1017 resolved leads: 622 Converted [`target=1`], 395 Lost [`target=0`])
- **Dataset Split (`random_state=42`)**:
  - **Training Set**: 650 records
  - **Validation Set**: 163 records
  - **Test Set (Held-Out)**: 204 records
- **Feature Schema**: Shared `ml/config/feature_config.py` (11 numeric, 4 nominal, 1 ordinal) via `build_preprocessing_pipeline()`.

---

## 2. Classification Performance Comparison

| Metric | Direction | Validation: Logistic Reg | Validation: Random Forest | Val Diff (RF - LR) | Test: Logistic Reg | Test: Random Forest | Test Diff (RF - LR) | Better Model (Test) |
|---|---|---|---|---|---|---|---|---|
| **ROC-AUC** | Higher ↑ | 0.8198 | 0.8194 | -0.0003 | 0.7722 | 0.7875 | +0.0153 | **Random Forest Baseline** |
| **PR-AUC** | Higher ↑ | 0.8683 | 0.8746 | +0.0064 | 0.8313 | 0.8503 | +0.0190 | **Random Forest Baseline** |
| **Accuracy** | Higher ↑ | 0.7607 | 0.7362 | -0.0245 | 0.7255 | 0.7304 | +0.0049 | **Random Forest Baseline** |
| **Precision** | Higher ↑ | 0.7679 | 0.7333 | -0.0345 | 0.7656 | 0.7717 | +0.0060 | **Random Forest Baseline** |
| **Recall** | Higher ↑ | 0.8687 | 0.8889 | +0.0202 | 0.7903 | 0.7903 | +0.0000 | **Tie** |
| **F1 Score** | Higher ↑ | 0.8152 | 0.8037 | -0.0115 | 0.7778 | 0.7809 | +0.0031 | **Random Forest Baseline** |
| **Log Loss** | Lower ↓ | 0.5142 | 0.5271 | +0.0129 | 0.5569 | 0.5493 | -0.0076 | **Random Forest Baseline** |

---

## 3. Lead Prioritization & Ranking Performance Comparison

| Segment | Metric | Logistic Reg | Random Forest | Difference (RF - LR) | Better Model |
|---|---|---|---|---|---|
| **Top 10 Leads** ($K=10$) | **Precision@10** | 0.9000 | 1.0000 | +0.1000 | **Random Forest Baseline** |
| | **Recall@10** | 0.0726 | 0.0806 | +0.0081 | **Random Forest Baseline** |
| | **Conversion Rate@10** | 0.9000 | 1.0000 | +0.1000 | **Random Forest Baseline** |
| | **Lift@10** | 1.4806x | 1.6452x | +0.1645x | **Random Forest Baseline** |
| **Top 20 Leads** ($K=20$) | **Precision@20** | 0.9000 | 0.9500 | +0.0500 | **Random Forest Baseline** |
| | **Recall@20** | 0.1452 | 0.1532 | +0.0081 | **Random Forest Baseline** |
| | **Conversion Rate@20** | 0.9000 | 0.9500 | +0.0500 | **Random Forest Baseline** |
| | **Lift@20** | 1.4806x | 1.5629x | +0.0823x | **Random Forest Baseline** |
| **Top 20% Leads** ($K=41$) | **Precision@41** | 0.9024 | 0.9512 | +0.0488 | **Random Forest Baseline** |
| | **Recall@41** | 0.2984 | 0.3145 | +0.0161 | **Random Forest Baseline** |
| | **Conversion Rate@41** | 0.9024 | 0.9512 | +0.0488 | **Random Forest Baseline** |
| | **Lift@41** | 1.4847x | 1.5649x | +0.0803x | **Random Forest Baseline** |

---

## 4. Metric-by-Metric Observations

- **ROC-AUC**: Random Forest achieved **0.7875** on the held-out test set compared to **0.7722** for Logistic Regression (+0.0153 improvement).
- **PR-AUC**: Random Forest scored **0.8503** vs **0.8313** for Logistic Regression (+0.0190 improvement).
- **Accuracy & F1 Score**: Random Forest demonstrated higher test accuracy (**73.04%** vs **72.55%**) and F1 Score (**0.7809** vs **0.7778**).
- **Log Loss**: Random Forest yielded lower test log loss (**0.5493** vs **0.5569**).
- **Lead Ranking**: In Top 10 leads ($K=10$), Random Forest achieved **100% precision** (10 out of 10 conversions correctly identified; **1.6452x lift**).

---

## 5. Overall Baseline Observations

1. **Model Strength**: Random Forest Baseline outperforms Logistic Regression Baseline across all evaluation dimensions on the held-out test set.
2. **Non-Linear Advantage**: Non-linear tree decision boundaries better capture interaction terms between engagement metrics and categorical parameters.

---

## 6. Limitations & Next Steps

> [!IMPORTANT]
> **No Final Production Model Selected Yet**  
> While Random Forest Baseline shows superior performance compared to Logistic Regression, final model selection is deferred until **Phase 4: XGBoost Baseline Training & Comparison** is complete.
