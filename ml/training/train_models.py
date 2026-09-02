import sys
from pathlib import Path

# Ensure project root is in sys.path for standalone script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.train_baseline import train_baseline_model
from ml.training.train_random_forest import train_random_forest_model
from ml.training.train_xgboost import train_xgboost_model
from ml.evaluation.compare_models import compare_model_results


def run_full_training_pipeline(data_path=None):
    """
    Main entry point for training all ML models (Baseline, Random Forest, XGBoost),
    evaluating performance, and generating comparison artifacts.
    """
    print("\n" + "#" * 70)
    print("STARTING FULL LEAD SCORING ML TRAINING PIPELINE")
    print("#" * 70 + "\n")

    baseline_metrics = train_baseline_model(data_path=data_path)
    rf_metrics = train_random_forest_model(data_path=data_path)
    xgb_metrics = train_xgboost_model(data_path=data_path)

    all_metrics = [baseline_metrics, rf_metrics, xgb_metrics]

    comparison_summary = compare_model_results(all_metrics)

    print("\n" + "#" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print(f"Best Model Selected: {comparison_summary.get('best_model')}")
    print(f"Best Test ROC-AUC:   {comparison_summary.get('best_roc_auc'):.4f}")
    print("#" * 70 + "\n")

    return comparison_summary


if __name__ == "__main__":
    run_full_training_pipeline()
