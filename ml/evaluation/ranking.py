from typing import Dict, Any, List
import pandas as pd
import numpy as np


def evaluate_ranking(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    k_values: List[int] = None,
) -> Dict[str, Any]:
    """
    Evaluates how well the model ranks positive/converting leads near the top of the prediction list.
    Computes Precision@K, Recall@K, Conversion Rate@K, and Lift@K.
    """
    ranking_df = pd.DataFrame({
        "actual": np.asarray(y_true),
        "probability": np.asarray(probabilities),
    })

    ranking_df = ranking_df.sort_values("probability", ascending=False).reset_index(drop=True)

    total_positives = int(ranking_df["actual"].sum())
    total_rows = len(ranking_df)
    overall_conversion_rate = total_positives / total_rows if total_rows > 0 else 0.0

    if k_values is None:
        k_20_percent = int(round(0.20 * total_rows))
        k_values = [10, 20, k_20_percent]

    results = {}
    for k in k_values:
        k_eff = min(k, total_rows)
        top_k = ranking_df.iloc[:k_eff]

        positives_in_top_k = int(top_k["actual"].sum())
        precision_at_k = positives_in_top_k / k_eff if k_eff > 0 else 0.0
        recall_at_k = positives_in_top_k / total_positives if total_positives > 0 else 0.0
        conversion_rate_at_k = precision_at_k
        lift_at_k = conversion_rate_at_k / overall_conversion_rate if overall_conversion_rate > 0 else 0.0

        results[f"top_{k_eff}"] = {
            "k": k_eff,
            "precision_at_k": float(precision_at_k),
            "recall_at_k": float(recall_at_k),
            "conversion_rate": float(conversion_rate_at_k),
            "lift_at_k": float(lift_at_k),
            "conversions_found": positives_in_top_k,
        }

    return results
