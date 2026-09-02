from typing import Dict, Any
import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


def evaluate_calibration(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Evaluates probability calibration using Brier score and calibration curve bins.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    brier_score = float(brier_score_loss(y_true, probabilities))

    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=n_bins, strategy='uniform')

    return {
        "brier_score": brier_score,
        "calibration_curve": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
    }
