import pytest
import joblib
import json
import pandas as pd
from ml.config import CLEANED_DATA_PATH, ARTIFACTS_DIR, MODELS_DIR, METRICS_DIR
from ml.training.train_baseline import train_baseline_model
from ml.training.train_random_forest import train_random_forest_model
from ml.training.train_xgboost import train_xgboost_model
from ml.training.tune_xgboost import tune_xgboost_model


@pytest.fixture
def clean_data_file():
    if not CLEANED_DATA_PATH.exists():
        pytest.skip(f"Cleaned dataset missing at {CLEANED_DATA_PATH}")
    return CLEANED_DATA_PATH


def test_train_baseline_execution(clean_data_file):
    metrics = train_baseline_model(data_path=clean_data_file)
    assert "test" in metrics
    assert "roc_auc" in metrics["test"]
    assert (ARTIFACTS_DIR / "logistic_regression_baseline.joblib").exists()
    assert (MODELS_DIR / "logistic_regression_baseline.joblib").exists()

    # Verify model reload
    model = joblib.load(MODELS_DIR / "logistic_regression_baseline.joblib")
    assert hasattr(model, "predict_proba")


def test_train_random_forest_execution(clean_data_file):
    metrics = train_random_forest_model(data_path=clean_data_file)
    assert "test" in metrics
    assert "roc_auc" in metrics["test"]
    assert "ranking" in metrics
    assert "top_10" in metrics["ranking"]
    assert "lift_at_k" in metrics["ranking"]["top_10"]

    # Verify canonical artifact creation
    assert (MODELS_DIR / "random_forest_baseline.joblib").exists()
    assert (METRICS_DIR / "random_forest_metrics.json").exists()

    # Verify model reload from canonical path
    model = joblib.load(MODELS_DIR / "random_forest_baseline.joblib")
    assert hasattr(model, "predict_proba")


def test_train_xgboost_execution(clean_data_file):
    metrics = train_xgboost_model(data_path=clean_data_file)
    assert "test" in metrics
    assert "roc_auc" in metrics["test"]
    assert "ranking" in metrics
    assert "top_10" in metrics["ranking"]

    # Verify canonical artifact creation
    assert (MODELS_DIR / "xgboost_baseline.joblib").exists()
    assert (METRICS_DIR / "xgboost_metrics.json").exists()

    # Verify model reload
    model = joblib.load(MODELS_DIR / "xgboost_baseline.joblib")
    assert hasattr(model, "predict_proba")


def test_tune_xgboost_execution(clean_data_file):
    metrics = tune_xgboost_model(data_path=clean_data_file)
    assert "test" in metrics
    assert "best_cv_score" in metrics
    assert "best_hyperparameters" in metrics

    # Verify tuned artifact creation
    assert (MODELS_DIR / "xgboost_tuned.joblib").exists()
    assert (METRICS_DIR / "xgboost_tuning_metrics.json").exists()
    assert (ARTIFACTS_DIR / "reports" / "xgboost_tuning.md").exists()

    # Verify model reload
    model = joblib.load(MODELS_DIR / "xgboost_tuned.joblib")
    assert hasattr(model, "predict_proba")
