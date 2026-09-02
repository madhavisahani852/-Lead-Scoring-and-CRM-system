import sys
from pathlib import Path

# Ensure project root is in sys.path for standalone script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import numpy as np
from ml.config.feature_config import CLEANED_DATA_PATH, MODELS_DIR, REPORTS_DIR
from ml.inference.predict import predict_lead_scores, score_unresolved_leads


def validate_production_artifact() -> bool:
    """
    Validates production readiness of best_model.joblib.
    Exits with code 0 if all checks pass, or code 1 if any check fails.
    """
    print("=" * 50)
    print("PRODUCTION ARTIFACT VALIDATION")
    print("=" * 50)

    model_path = MODELS_DIR / "best_model.joblib"
    results = {}

    # Check 1: File existence
    results["Artifact existence"] = model_path.exists()

    if not model_path.exists():
        print(f"FAIL: {model_path} does not exist!")
        sys.exit(1)

    # Check 2: Model loading
    try:
        pipeline = joblib.load(model_path)
        results["Model loading"] = True
    except Exception as e:
        print(f"FAIL: Could not load {model_path}: {e}")
        sys.exit(1)

    # Check 3: Support predict_proba()
    results["predict_proba support"] = hasattr(pipeline, "predict_proba")

    # Check 4: Input schema & Check 8: Prediction count
    try:
        df_sample = pd.read_csv(CLEANED_DATA_PATH).head(50)
        scored_df = predict_lead_scores(df_sample, model_path=model_path)
        results["Input schema"] = True
        results["Preprocessing pipeline"] = True
        results["Expected prediction count"] = len(scored_df) == 50
    except Exception as e:
        print(f"FAIL: Preprocessing/Prediction error: {e}")
        sys.exit(1)

    # Check 5: Probabilities numeric & Check 6: Range & Check 7: No NaN
    probs = scored_df["conversion_probability"].values
    results["Numeric probabilities"] = np.issubdtype(probs.dtype, np.number)
    results["Probability range (0..1)"] = bool((probs >= 0.0).all() and (probs <= 1.0).all())
    results["NaN check"] = not np.isnan(probs).any() and np.isfinite(probs).all()

    # Check 9: Unresolved lead scoring
    try:
        unresolved_report = score_unresolved_leads(model_path=model_path)
        results["Unresolved lead scoring"] = len(unresolved_report) > 0
    except Exception as e:
        print(f"FAIL: Unresolved lead scoring error: {e}")
        sys.exit(1)

    # Check 10: Ranking sorting
    sorted_probs = scored_df["conversion_probability"].tolist()
    results["Probability ranking"] = sorted_probs == sorted(sorted_probs, reverse=True)

    # Check 11: Retraining not required (Pipeline complete)
    results["Standalone pipeline (no refit needed)"] = hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps

    print("\n--- VALIDATION SUMMARY ---")
    all_passed = True
    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{check_name:<35}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("Status: PRODUCTION READY")
        print("=" * 50)
        return True
    else:
        print("Status: VALIDATION FAILED")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    validate_production_artifact()
