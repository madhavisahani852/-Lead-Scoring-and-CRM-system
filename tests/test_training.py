import pytest
import joblib
import pandas as pd
from ml.config import CLEANED_DATA_PATH, MODELS_DIR, METRICS_DIR
from ml.training.train_baseline import train_baseline_model
from ml.training.train_random_forest import train_random_forest_model
from ml.training.train_xgboost import train_xgboost_model


@pytest.fixture
def clean_data_file():
    if not CLEANED_DATA_PATH.exists():
        pytest.skip(f"Cleaned dataset missing at {CLEANED_DATA_PATH}")
    return CLEANED_DATA_PATH


def test_train_baseline_execution(clean_data_file):
    metrics = train_baseline_model(data_path=clean_data_file)
    assert "test" in metrics
    assert "roc_auc" in metrics["test"]
    assert (MODELS_DIR / "logistic_regression_baseline.joblib").exists()

    # Verify model reload
    model = joblib.load(MODELS_DIR / "logistic_regression_baseline.joblib")
    assert hasattr(model, "predict_proba")


def test_train_random_forest_execution(clean_data_file):
    metrics = train_random_forest_model(data_path=clean_data_file)
    assert "test" in metrics
    assert "roc_auc" in metrics["test"]
    assert (MODELS_DIR / "random_forest.joblib").exists()

    model = joblib.load(MODELS_DIR / "random_forest.joblib")
    assert hasattr(model, "predict_proba")


def test_train_xgboost_execution(clean_data_file):
    metrics = train_xgboost_model(data_path=clean_data_file)
    assert "test" in metrics
    assert "roc_auc" in metrics["test"]
    assert (MODELS_DIR / "xgboost.joblib").exists()

    model = joblib.load(MODELS_DIR / "xgboost.joblib")
    assert hasattr(model, "predict_proba")
