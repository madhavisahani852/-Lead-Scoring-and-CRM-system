import sys
from pathlib import Path

# Ensure project root is in sys.path for standalone script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.config.feature_config import (
    CLEANED_DATA_PATH,
    MODELS_DIR,
    METRICS_DIR,
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
    print("XGBOOST / GRADIENT BOOSTING MODEL TRAINING")
    print("=" * 60)
    print(f"Loading data from: {data_path}")

    df = pd.read_csv(data_path)
    X, y = prepare_training_data(df)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=y_dev
    )

    preprocessor = build_preprocessing_pipeline()

    try:
        from xgboost import XGBClassifier
        classifier = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        )
        model_name = "XGBoost"
    except ImportError:
        print("Note: xgboost package not found. Falling back to sklearn.ensemble.HistGradientBoostingClassifier.")
        from sklearn.ensemble import HistGradientBoostingClassifier
        classifier = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )
        model_name = "XGBoost (HistGradientBoosting Fallback)"

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train, y_train)

    val_probs = pipeline.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= CLASSIFICATION_THRESHOLD).astype(int)
    val_metrics = evaluate_classification(y_val, val_probs, val_preds)

    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= CLASSIFICATION_THRESHOLD).astype(int)
    test_metrics = evaluate_classification(y_test, test_probs, test_preds)

    ranking_metrics = evaluate_ranking(y_test, test_probs, k_values=[10, 20, 50])
    calibration_metrics = evaluate_calibration(y_test, test_probs)

    print_classification_metrics(f"{model_name.upper()} TEST RESULTS", test_metrics)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model_file = MODELS_DIR / "xgboost.joblib"
    joblib.dump(pipeline, model_file)
    print(f"Model saved to: {model_file}")

    all_metrics = {
        "model": model_name,
        "random_state": RANDOM_STATE,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "validation": val_metrics,
        "test": test_metrics,
        "ranking": ranking_metrics,
        "calibration": calibration_metrics,
    }

    metrics_file = METRICS_DIR / "xgboost_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_file}")
    return all_metrics


if __name__ == "__main__":
    train_xgboost_model()
