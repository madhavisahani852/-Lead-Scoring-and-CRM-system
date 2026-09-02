import sys
from pathlib import Path

# Ensure project root is in sys.path for standalone script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Dict, Any

from ml.config.feature_config import MODELS_DIR
from ml.training.train_baseline import train_baseline_model
from ml.training.train_random_forest import train_random_forest_model
from ml.evaluation.compare_models import generate_baseline_comparison_report
from ml.training.train_xgboost import train_xgboost_model
from ml.training.tune_xgboost import tune_xgboost_model
from ml.evaluation.final_model_selection import run_final_model_selection


def run_full_training_pipeline(data_path=None) -> Dict[str, Any]:
    """
    Master ML Pipeline Orchestrator for the Lead Scoring & CRM system.

    Executes all phases in strict sequential order:
      - Phase 1: Logistic Regression Baseline
      - Phase 2: Random Forest Baseline
      - Phase 3: Baseline Model Comparison
      - Phase 4: XGBoost Baseline
      - Phase 5: XGBoost Hyperparameter Tuning
      - Phase 6: Final Model Comparison & Model Selection (Authoritative)

    Reads the final selected model from Phase 6 metadata (model_metadata.json).
    """
    print("\n" + "#" * 70)
    print("STARTING FULL LEAD SCORING ML TRAINING PIPELINE")
    print("#" * 70 + "\n")

    try:
        # PHASE 1: LOGISTIC REGRESSION BASELINE
        print("\n" + "=" * 60)
        print("PHASE 1 — LOGISTIC REGRESSION BASELINE")
        print("=" * 60)
        train_baseline_model(data_path=data_path)

        # PHASE 2: RANDOM FOREST BASELINE
        print("\n" + "=" * 60)
        print("PHASE 2 — RANDOM FOREST BASELINE")
        print("=" * 60)
        train_random_forest_model(data_path=data_path)

        # PHASE 3: BASELINE COMPARISON
        print("\n" + "=" * 60)
        print("PHASE 3 — BASELINE MODEL COMPARISON")
        print("=" * 60)
        generate_baseline_comparison_report()

        # PHASE 4: XGBOOST BASELINE
        print("\n" + "=" * 60)
        print("PHASE 4 — XGBOOST BASELINE")
        print("=" * 60)
        train_xgboost_model(data_path=data_path)

        # PHASE 5: XGBOOST HYPERPARAMETER TUNING
        print("\n" + "=" * 60)
        print("PHASE 5 — XGBOOST HYPERPARAMETER TUNING")
        print("=" * 60)
        tune_xgboost_model(data_path=data_path)

        # PHASE 6: FINAL MODEL COMPARISON & SELECTION
        print("\n" + "=" * 60)
        print("PHASE 6 — FINAL MODEL COMPARISON & SELECTION")
        print("=" * 60)
        final_selection_results = run_final_model_selection()

    except Exception as e:
        print("\n" + "!" * 70)
        print("MASTER ML PIPELINE FAILED DURING EXECUTION")
        print(f"Error: {e}")
        print("!" * 70 + "\n")
        raise e

    # Read authoritative metadata from Phase 6 output
    metadata_path = MODELS_DIR / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Phase 6 metadata missing at {metadata_path}. Master pipeline aborted.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    final_model_name = metadata.get("model_name", "Unknown")
    test_metrics = metadata.get("test_metrics", {})
    test_roc_auc = test_metrics.get("roc_auc", 0.0)
    test_pr_auc = test_metrics.get("pr_auc", 0.0)
    test_f1 = test_metrics.get("f1", 0.0)

    print("\n" + "#" * 70)
    print("FULL LEAD SCORING ML PIPELINE COMPLETE")
    print("#" * 70)
    print(f"Final Model Selected: {final_model_name}")
    print(f"Final Test ROC-AUC:   {test_roc_auc:.4f}")
    print(f"Final Test PR-AUC:    {test_pr_auc:.4f}")
    print(f"Final Test F1:        {test_f1:.4f}")
    print("#" * 70 + "\n")

    return final_selection_results


if __name__ == "__main__":
    run_full_training_pipeline()
