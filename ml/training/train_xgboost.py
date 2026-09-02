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
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.config.feature_config import (
    CLEANED_DATA_PATH,
    MODELS_DIR,
    METRICS_DIR,
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


def train_xgboost_model(data_path=None):
    if data_path is None:
        data_path = CLEANED_DATA_PATH

    print("=" * 60)
    print("XGBOOST BASELINE MODEL TRAINING")
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

    # Use shared preprocessing pipeline from preprocessing.py
    preprocessor = build_preprocessing_pipeline()

    # Preprocessing feature count check
    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out()
    transformed_feature_count = len(feature_names)
    print(f"Transformed feature count: {transformed_feature_count}")
    assert transformed_feature_count == 36, f"Expected 36 features, found {transformed_feature_count}"

    # Model configuration
    xgb_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    classifier = XGBClassifier(**xgb_params)

    # Build pipeline: preprocessor + XGBoost
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("classifier", classifier),
    ])

    print("\nTraining XGBoost baseline model...")
    pipeline.fit(X_train, y_train)
    print("Training completed.")

    # Validation predictions & evaluation
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    assert not np.isnan(val_probs).any(), "NaN detected in validation probabilities"
    assert np.isfinite(val_probs).all(), "Inf detected in validation probabilities"
    assert (val_probs >= 0.0).all() and (val_probs <= 1.0).all(), "Validation probabilities out of bounds"

    val_preds = (val_probs >= CLASSIFICATION_THRESHOLD).astype(int)
    val_metrics = evaluate_classification(y_val, val_probs, val_preds)
    print_classification_metrics("XGBOOST BASELINE VALIDATION RESULTS", val_metrics)

    # Test predictions & evaluation
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    assert not np.isnan(test_probs).any(), "NaN detected in test probabilities"
    assert np.isfinite(test_probs).all(), "Inf detected in test probabilities"
    assert (test_probs >= 0.0).all() and (test_probs <= 1.0).all(), "Test probabilities out of bounds"

    test_preds = (test_probs >= CLASSIFICATION_THRESHOLD).astype(int)
    test_metrics = evaluate_classification(y_test, test_probs, test_preds)
    print_classification_metrics("XGBOOST BASELINE TEST RESULTS", test_metrics)

    # Ranking metrics for K=10, 20, and ~20% of test set (K=41)
    k_20_pct = int(round(0.20 * len(y_test)))
    ranking_k_values = [10, 20, k_20_pct]
    ranking_metrics = evaluate_ranking(y_test, test_probs, k_values=ranking_k_values)
    calibration_metrics = evaluate_calibration(y_test, test_probs)

    print("\n" + "=" * 60)
    print("XGBOOST BASELINE RANKING RESULTS")
    print("=" * 60)
    for key, result in ranking_metrics.items():
        print(f"Top {result['k']} leads:")
        print(f"  Precision@K:    {result['precision_at_k']:.4f}")
        print(f"  Recall@K:       {result['recall_at_k']:.4f}")
        print(f"  ConversionRate: {result['conversion_rate']:.4f}")
        print(f"  Lift@K:         {result['lift_at_k']:.4f}")
        print(f"  Conversions:    {result['conversions_found']}")

    all_metrics = {
        "model_name": "XGBoost Baseline",
        "model_type": "XGBClassifier",
        "random_seed": RANDOM_STATE,
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
        "class_distribution": {
            "train_0": int((y_train == 0).sum()),
            "train_1": int((y_train == 1).sum()),
            "val_0": int((y_val == 0).sum()),
            "val_1": int((y_val == 1).sum()),
            "test_0": int((y_test == 0).sum()),
            "test_1": int((y_test == 1).sum()),
        },
        "transformed_feature_count": transformed_feature_count,
        "features_summary": {
            "numeric_columns": len(NUMERIC_COLS),
            "nominal_columns": len(NOMINAL_COLS),
            "ordinal_columns": len(ORDINAL_COLS),
        },
        "model_configuration": xgb_params,
        "validation": val_metrics,
        "test": test_metrics,
        "ranking": ranking_metrics,
        "calibration": calibration_metrics,
    }

    # Ensure output directories exist automatically
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Save fitted pipeline ONLY to canonical location
    model_path = MODELS_DIR / "xgboost_baseline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved model pipeline to: {model_path}")

    # Save metrics ONLY to canonical location
    metrics_path = METRICS_DIR / "xgboost_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    print("\n" + "=" * 60)
    print("XGBOOST BASELINE TRAINING & EVALUATION COMPLETE")
    print("=" * 60)

    return all_metrics


if __name__ == "__main__":
    train_xgboost_model()
