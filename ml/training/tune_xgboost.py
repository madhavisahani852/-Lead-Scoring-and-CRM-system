import sys
from pathlib import Path

# Ensure project root is in sys.path for standalone script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime, timezone
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from ml.config.feature_config import (
    CLEANED_DATA_PATH,
    MODELS_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    NUMERIC_COLS,
    NOMINAL_COLS,
    ORDINAL_COLS,
)
from ml.preprocessing import (
    prepare_training_data,
    build_preprocessing_pipeline,
)
from ml.evaluation import (
    evaluate_classification,
    print_classification_metrics,
    evaluate_ranking,
    evaluate_calibration,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
CLASSIFICATION_THRESHOLD = 0.50


def tune_xgboost_model(data_path=None):
    if data_path is None:
        data_path = CLEANED_DATA_PATH

    print("=" * 60)
    print("XGBOOST HYPERPARAMETER TUNING (PHASE 5)")
    print("=" * 60)
    print(f"Loading dataset from: {data_path}")

    df = pd.read_csv(data_path)
    X, y = prepare_training_data(df)

    # Train / Test split (80% development, 20% test)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Train / Validation split (80% train, 20% validation of dev set)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=y_dev
    )

    print(f"Resolved records:    {len(y)}")
    print(f"Training set size:   {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size:       {len(X_test)}")

    # Preprocessing feature count check
    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out()
    transformed_feature_count = len(feature_names)
    print(f"Transformed feature count: {transformed_feature_count}")
    assert transformed_feature_count == 36, f"Expected 36 features, found {transformed_feature_count}"

    # Search distribution space
    param_distributions = {
        "classifier__n_estimators": [100, 200, 300, 400, 500],
        "classifier__max_depth": [2, 3, 4, 5, 6],
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "classifier__min_child_weight": [1, 3, 5],
        "classifier__gamma": [0, 0.1, 0.3],
        "classifier__reg_alpha": [0, 0.01, 0.1],
        "classifier__reg_lambda": [1, 1.5, 2, 5],
    }

    base_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("classifier", XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1
        )),
    ])

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="roc_auc",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True
    )

    print("\nRunning RandomizedSearchCV on Training Data ONLY (650 records, 3 folds)...")
    search.fit(X_train, y_train)
    print("Hyperparameter search completed.")

    best_cv_score = float(search.best_score_)
    raw_best_params = search.best_params_
    best_params = {
        k.replace("classifier__", ""): v for k, v in raw_best_params.items()
    }

    print(f"\nBest CV ROC-AUC Score: {best_cv_score:.4f}")
    print("Best Hyperparameters:")
    print(json.dumps(best_params, indent=2))

    best_pipeline = search.best_estimator_

    # Validation predictions & evaluation
    val_probs = best_pipeline.predict_proba(X_val)[:, 1]
    assert not np.isnan(val_probs).any(), "NaN detected in validation probabilities"
    assert np.isfinite(val_probs).all(), "Inf detected in validation probabilities"
    assert (val_probs >= 0.0).all() and (val_probs <= 1.0).all(), "Validation probabilities out of bounds"

    val_preds = (val_probs >= CLASSIFICATION_THRESHOLD).astype(int)
    val_metrics = evaluate_classification(y_val, val_probs, val_preds)
    print_classification_metrics("XGBOOST TUNED VALIDATION RESULTS", val_metrics)

    # Test predictions & evaluation (Test set untouched during tuning)
    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    assert not np.isnan(test_probs).any(), "NaN detected in test probabilities"
    assert np.isfinite(test_probs).all(), "Inf detected in test probabilities"
    assert (test_probs >= 0.0).all() and (test_probs <= 1.0).all(), "Test probabilities out of bounds"

    test_preds = (test_probs >= CLASSIFICATION_THRESHOLD).astype(int)
    test_metrics = evaluate_classification(y_test, test_probs, test_preds)
    print_classification_metrics("XGBOOST TUNED TEST RESULTS", test_metrics)

    # Ranking metrics for K=10, 20, and ~20% of test set (K=41)
    k_20_pct = int(round(0.20 * len(y_test)))
    ranking_k_values = [10, 20, k_20_pct]
    ranking_metrics = evaluate_ranking(y_test, test_probs, k_values=ranking_k_values)
    calibration_metrics = evaluate_calibration(y_test, test_probs)

    print("\n" + "=" * 60)
    print("XGBOOST TUNED RANKING RESULTS")
    print("=" * 60)
    for key, result in ranking_metrics.items():
        print(f"Top {result['k']} leads:")
        print(f"  Precision@K:    {result['precision_at_k']:.4f}")
        print(f"  Recall@K:       {result['recall_at_k']:.4f}")
        print(f"  ConversionRate: {result['conversion_rate']:.4f}")
        print(f"  Lift@K:         {result['lift_at_k']:.4f}")
        print(f"  Conversions:    {result['conversions_found']}")

    all_metrics = {
        "model_name": "XGBoost Tuned",
        "model_type": "XGBClassifier",
        "random_state": RANDOM_STATE,
        "dataset_path": str(data_path),
        "total_resolved_records": len(y),
        "dataset_summary": {
            "total_records": len(df),
            "resolved_records": len(y),
            "unresolved_records": len(df) - len(y),
            "training_records": len(X_train),
            "validation_records": len(X_val),
            "test_records": len(X_test),
        },
        "transformed_feature_count": transformed_feature_count,
        "tuning_method": "RandomizedSearchCV",
        "cv_folds": 3,
        "n_iter": 20,
        "scoring_metric": "roc_auc",
        "best_cv_score": best_cv_score,
        "best_hyperparameters": best_params,
        "validation": val_metrics,
        "test": test_metrics,
        "ranking": ranking_metrics,
        "calibration": calibration_metrics,
    }

    # Ensure output directories exist automatically
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save fitted pipeline to ml/artifacts/models/xgboost_tuned.joblib
    model_path = MODELS_DIR / "xgboost_tuned.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\nSaved tuned model pipeline to: {model_path}")

    # Save metrics to ml/artifacts/metrics/xgboost_tuning_metrics.json
    metrics_path = METRICS_DIR / "xgboost_tuning_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved tuning metrics to: {metrics_path}")

    # Generate tuning report markdown
    baseline_metrics_path = METRICS_DIR / "xgboost_metrics.json"
    baseline = {}
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)

    generate_tuning_report(all_metrics, baseline)

    print("\n" + "=" * 60)
    print("XGBOOST TUNING & EVALUATION COMPLETE")
    print("=" * 60)

    return all_metrics


def generate_tuning_report(tuned: Dict[str, Any], baseline: Dict[str, Any]) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report_path = REPORTS_DIR / "xgboost_tuning.md"

    b_test = baseline.get("test", {})
    t_test = tuned.get("test", {})

    b_roc = b_test.get("roc_auc", 0.0)
    t_roc = t_test.get("roc_auc", 0.0)
    roc_diff = t_roc - b_roc

    b_pr = b_test.get("pr_auc", 0.0)
    t_pr = t_test.get("pr_auc", 0.0)
    pr_diff = t_pr - b_pr

    b_f1 = b_test.get("f1", 0.0)
    t_f1 = t_test.get("f1", 0.0)
    f1_diff = t_f1 - b_f1

    b_loss = b_test.get("log_loss", 0.0)
    t_loss = t_test.get("log_loss", 0.0)
    loss_diff = t_loss - b_loss

    report_content = f"""# XGBoost Hyperparameter Tuning Report

**Generated At**: `{timestamp}`  
**Phase**: Phase 5 XGBoost Hyperparameter Tuning  
**Status**: Completed

---

## 1. Dataset & Split Information

- **Dataset Path**: `cleaned_leads.csv`
- **Total Resolved Records**: {tuned['total_resolved_records']} (100% binary outcome: `target` in [0, 1])
- **Reproducible Dataset Split (`random_state=42`)**:
  - **Training Set**: {tuned['dataset_summary']['training_records']} records (64% of resolved dataset)
  - **Validation Set**: {tuned['dataset_summary']['validation_records']} records (16% of resolved dataset)
  - **Test Set (Held-Out)**: {tuned['dataset_summary']['test_records']} records (20% of resolved dataset)
- **Transformed Feature Count**: {tuned['transformed_feature_count']} features (via `build_preprocessing_pipeline()`)

---

## 2. Tuning Methodology & Search Space

- **Method**: `RandomizedSearchCV` (scikit-learn)
- **CV Folds**: 3-fold cross-validation
- **Search Iterations**: 20 parameter combinations
- **Scoring Metric**: `roc_auc`
- **Data Leakage Safeguards**: Search was executed **exclusively on the training dataset (650 records)**. Preprocessing transformers were fit strictly inside CV folds. The test set remained completely untouched throughout parameter selection.

### Hyperparameter Search Space

```python
{{
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [1, 1.5, 2, 5]
}}
```

---

## 3. Best Hyperparameters & CV Score

- **Best Cross-Validation ROC-AUC**: **{tuned['best_cv_score']:.4f}**
- **Optimal Hyperparameter Configuration**:

```json
{json.dumps(tuned['best_hyperparameters'], indent=2)}
```

---

## 4. Evaluation Results

### Validation Set Results ($N=163$)

- **ROC-AUC**: {tuned['validation']['roc_auc']:.4f}
- **PR-AUC**: {tuned['validation']['pr_auc']:.4f}
- **Accuracy**: {tuned['validation']['accuracy']:.4f}
- **Precision**: {tuned['validation']['precision']:.4f}
- **Recall**: {tuned['validation']['recall']:.4f}
- **F1 Score**: {tuned['validation']['f1']:.4f}
- **Log Loss**: {tuned['validation']['log_loss']:.4f}

### Test Set Results ($N=204$)

- **ROC-AUC**: {tuned['test']['roc_auc']:.4f}
- **PR-AUC**: {tuned['test']['pr_auc']:.4f}
- **Accuracy**: {tuned['test']['accuracy']:.4f}
- **Precision**: {tuned['test']['precision']:.4f}
- **Recall**: {tuned['test']['recall']:.4f}
- **F1 Score**: {tuned['test']['f1']:.4f}
- **Log Loss**: {tuned['test']['log_loss']:.4f}

---

## 5. Ranking & Lead Prioritization Performance

| Segment | Precision@K | Recall@K | Conversion Rate | Lift@K | Conversions Found |
|---|---|---|---|---|---|
| **Top 10 Leads** ($K=10$) | {tuned['ranking']['top_10']['precision_at_k']:.4f} | {tuned['ranking']['top_10']['recall_at_k']:.4f} | {tuned['ranking']['top_10']['conversion_rate']:.4f} | {tuned['ranking']['top_10']['lift_at_k']:.4f}x | {tuned['ranking']['top_10']['conversions_found']} |
| **Top 20 Leads** ($K=20$) | {tuned['ranking']['top_20']['precision_at_k']:.4f} | {tuned['ranking']['top_20']['recall_at_k']:.4f} | {tuned['ranking']['top_20']['conversion_rate']:.4f} | {tuned['ranking']['top_20']['lift_at_k']:.4f}x | {tuned['ranking']['top_20']['conversions_found']} |
| **Top 20% Leads** ($K=41$) | {tuned['ranking']['top_41']['precision_at_k']:.4f} | {tuned['ranking']['top_41']['recall_at_k']:.4f} | {tuned['ranking']['top_41']['conversion_rate']:.4f} | {tuned['ranking']['top_41']['lift_at_k']:.4f}x | {tuned['ranking']['top_41']['conversions_found']} |

---

## 6. Comparison with XGBoost Baseline

Comparison of tuned XGBoost vs untuned XGBoost baseline on held-out test data:

| Metric | XGBoost Baseline | XGBoost Tuned | Difference (Tuned - Baseline) | Status |
|---|---|---|---|---|
| **ROC-AUC** | {b_roc:.4f} | {t_roc:.4f} | {roc_diff:+.4f} | {'Improved ↑' if roc_diff > 0 else 'Worsened/Equal'} |
| **PR-AUC** | {b_pr:.4f} | {t_pr:.4f} | {pr_diff:+.4f} | {'Improved ↑' if pr_diff > 0 else 'Worsened/Equal'} |
| **F1 Score** | {b_f1:.4f} | {t_f1:.4f} | {f1_diff:+.4f} | {'Improved ↑' if f1_diff > 0 else 'Worsened/Equal'} |
| **Log Loss** | {b_loss:.4f} | {t_loss:.4f} | {loss_diff:+.4f} | {'Improved ↓' if loss_diff < 0 else 'Worsened/Equal'} |

---

## 7. Observations & Safeguards

1. **Hyperparameter Selection Integrity**: Cross-validation tuning strictly operated on `X_train`. The test set was evaluated exactly once after pipeline refitting.
2. **Model Selection Deferred**: Final production model selection across Logistic Regression, Random Forest, XGBoost Baseline, and XGBoost Tuned will take place in the next phase.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved tuning report to: {report_path}")


if __name__ == "__main__":
    tune_xgboost_model()
