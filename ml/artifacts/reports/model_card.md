# Model Card: Lead Scoring & CRM Intelligence Tool

**Model Name**: Tuned XGBoost Pipeline (`best_model.joblib`)  
**Version**: 1.0.0  
**Generated At**: `2026-09-02T09:34:21Z`  
**Status**: Final Selected Production Candidate  

---

## 1. Model Details

- **Developer**: Lead Scoring & CRM ML Engineering Team
- **Model Architecture**: Scikit-Learn Pipeline combining `ColumnTransformer` (preprocessing) and `XGBClassifier` (gradient boosted decision trees).
- **Hyperparameters**: `n_estimators=100`, `max_depth=2`, `learning_rate=0.1`, `subsample=0.7`, `colsample_bytree=0.8`, `min_child_weight=1`, `gamma=0`, `reg_alpha=0.01`, `reg_lambda=1.5`.

---

## 2. Intended Use & Scope

### Appropriate Use Cases
- **Lead Prioritization**: Predict binary conversion probability (0.0 to 1.0) and map to a 0–100 Lead Score for sales representatives.
- **Queue Ranking**: Sort inbound leads by conversion probability to maximize sales productivity.
- **CRM Integration**: Feed automated lead scoring tiers (Hot [>= 75], Warm [40–74], Cold [<40]) into sales pipelines.

### Inappropriate Use Cases
- **Fully Automated Decisions**: Discarding or ignoring leads without human review based solely on low scores.
- **Out-of-Domain Prediction**: Applying the model to enterprise B2B sales or non-CRM datasets without retraining.

---

## 3. Problem & Target Definition

- **Problem Type**: Binary Classification & Probability Ranking.
- **Target Variable (`target`)**:
  - `1`: **Converted** (Lead became a paying customer).
  - `0`: **Lost** (Lead explicitly closed without converting).

> [!IMPORTANT]
> **Handling of Unresolved Leads**:  
> Leads with status `"New"`, `"Contacted"`, or `"Qualified"` (`target = NaN`) are **NOT** training negatives. They represent open, unresolved opportunities and are strictly reserved as **scoring-only targets**.

---

## 4. Input Features (36 Transformed Features)

- **Numeric Features (11)**: `page_views`, `time_on_site`, `email_opens`, `email_clicks`, `form_submissions`, `webinar_attended`, `downloads`, `calls_made`, `demo_requested`, `lead_age_days`, `company_size`.
- **Nominal Features (4)**: `lead_source`, `industry`, `country`, `job_role`.
- **Ordinal Feature (1)**: `budget_range`.

---

## 5. Evaluation Methodology & Safeguards

- **Dataset**: `cleaned_leads.csv` (1,017 resolved leads).
- **Train/Val/Test Split (`random_state=42`)**: 650 Train / 163 Validation / 204 Test.
- **Data Leakage Safeguards**: Preprocessing parameters fitted strictly on training split; hyperparameter cross-validation conducted strictly on training split.

---

## 6. Performance Summary (Held-Out Test Set, $N=204$)

- **ROC-AUC**: `0.8086`
- **PR-AUC**: `0.8654`
- **Accuracy**: `74.02%`
- **Precision**: `77.10%`
- **Recall**: `81.45%`
- **F1 Score**: `0.7922`
- **Log Loss**: `0.5193`
- **Precision@10**: `1.0000` (100% precision in Top 10 leads, Lift = `1.6452x`)

---

## 7. Known Limitations & Risks

1. **Human Oversight Required**: Model predictions reflect historical conversion patterns and should support sales prioritization rather than replace human judgment.
2. **Concept Drift**: Performance may degrade if market conditions or acquisition channels shift over time.
3. **Data Completeness**: Incomplete interaction logs may artificially lower a lead's predicted probability.

---

## 8. Artifact Locations

- **Pipeline Model**: [`ml/artifacts/models/best_model.joblib`](file:///c:/Developement/-Lead-Scoring-and-CRM-system/ml/artifacts/models/best_model.joblib)
- **Model Metadata**: [`ml/artifacts/models/model_metadata.json`](file:///c:/Developement/-Lead-Scoring-and-CRM-system/ml/artifacts/models/model_metadata.json)
- **Final Comparison Report**: [`ml/artifacts/reports/final_model_comparison.md`](file:///c:/Developement/-Lead-Scoring-and-CRM-system/ml/artifacts/reports/final_model_comparison.md)
