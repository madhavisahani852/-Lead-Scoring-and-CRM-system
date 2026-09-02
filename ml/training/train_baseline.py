import sys
from pathlib import Path

# Ensure project root is in sys.path for standalone script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.config.feature_config import (
    CLEANED_DATA_PATH,
    ARTIFACTS_DIR,
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


def train_baseline_model(data_path=None):
    if data_path is None:
        data_path = CLEANED_DATA_PATH

    print("=" * 60)
    print("BASELINE MODEL TRAINING (Logistic Regression)")
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

    print(f"Resolved records: {len(y)}")
    print(f"Training set size:   {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size:       {len(X_test)}")

    # Use team's shared preprocessing pipeline from preprocessing.py
    preprocessor = build_preprocessing_pipeline()
    classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    print("\nTraining Logistic Regression baseline...")
    pipeline.fit(X_train, y_train)
    print("Training completed.")

    # Validation predictions & evaluation
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= CLASSIFICATION_THRESHOLD).astype(int)
    val_metrics = evaluate_classification(y_val, val_probs, val_preds)
    print_classification_metrics("LOGISTIC REGRESSION VALIDATION RESULTS", val_metrics)

    # Test predictions & evaluation
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= CLASSIFICATION_THRESHOLD).astype(int)
    test_metrics = evaluate_classification(y_test, test_probs, test_preds)
    print_classification_metrics("LOGISTIC REGRESSION TEST RESULTS", test_metrics)

    # Ranking metrics for K=10, 20, and ~20% of test set
    k_20_pct = int(round(0.20 * len(y_test)))
    ranking_k_values = [10, 20, k_20_pct]
    ranking_metrics = evaluate_ranking(y_test, test_probs, k_values=ranking_k_values)
    calibration_metrics = evaluate_calibration(y_test, test_probs)

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION RANKING RESULTS")
    print("=" * 60)
    for key, result in ranking_metrics.items():
        print(f"Top {result['k']} leads:")
        print(f"  Precision@K:    {result['precision_at_k']:.4f}")
        print(f"  Recall@K:       {result['recall_at_k']:.4f}")
        print(f"  ConversionRate: {result['conversion_rate']:.4f}")
        print(f"  Lift@K:         {result['lift_at_k']:.4f}")
        print(f"  Conversions:    {result['conversions_found']}")

    all_metrics = {
        "model": "Logistic Regression Baseline",
        "random_state": RANDOM_STATE,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "dataset_summary": {
            "total_records": len(df),
            "resolved_records": len(y),
            "unresolved_records": len(df) - len(y),
            "training_records": len(X_train),
            "validation_records": len(X_val),
            "test_records": len(X_test),
        },
        "features_summary": {
            "numeric_columns": len(NUMERIC_COLS),
            "nominal_columns": len(NOMINAL_COLS),
            "ordinal_columns": len(ORDINAL_COLS),
        },
        "validation": val_metrics,
        "test": test_metrics,
        "ranking": ranking_metrics,
        "calibration": calibration_metrics,
    }

    # Ensure output directories exist
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Save fitted pipeline to both ml/artifacts/ and ml/artifacts/models/
    model_paths = [
        ARTIFACTS_DIR / "logistic_regression_baseline.joblib",
        MODELS_DIR / "logistic_regression_baseline.joblib",
    ]
    for m_path in model_paths:
        joblib.dump(pipeline, m_path)
        print(f"Saved model pipeline to: {m_path}")

    # Save complete metrics to both ml/artifacts/ and ml/artifacts/metrics/
    metrics_paths = [
        ARTIFACTS_DIR / "baseline_metrics.json",
        METRICS_DIR / "baseline_metrics.json",
    ]
    for metrics_path in metrics_paths:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")

    print("\n" + "=" * 60)
    print("BASELINE TRAINING & EVALUATION COMPLETE")
    print("=" * 60)

    return all_metrics


if __name__ == "__main__":
    train_baseline_model()
