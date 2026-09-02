import json
from typing import Dict, Any, List
from ml.config.feature_config import METRICS_DIR, REPORTS_DIR


def compare_model_results(model_metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compares metrics across multiple models and identifies the best performing model based on ROC-AUC.
    Generates model_comparison.md in ml/artifacts/reports/ and model_comparison.json in ml/artifacts/metrics/.
    """
    if not model_metrics_list:
        return {}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    sorted_models = sorted(
        model_metrics_list,
        key=lambda m: m.get("test", {}).get("roc_auc", 0.0),
        reverse=True
    )

    best_model = sorted_models[0]

    comparison_summary = {
        "best_model": best_model.get("model", "Unknown"),
        "best_roc_auc": best_model.get("test", {}).get("roc_auc", 0.0),
        "models_compared": len(sorted_models),
        "comparison": [
            {
                "model": m.get("model"),
                "test_roc_auc": m.get("test", {}).get("roc_auc"),
                "test_accuracy": m.get("test", {}).get("accuracy"),
                "test_precision": m.get("test", {}).get("precision"),
                "test_recall": m.get("test", {}).get("recall"),
                "test_f1": m.get("test", {}).get("f1"),
                "test_log_loss": m.get("test", {}).get("log_loss"),
                "brier_score": m.get("calibration", {}).get("brier_score"),
            }
            for m in sorted_models
        ]
    }

    # Write model_comparison.json
    json_path = METRICS_DIR / "model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    # Write model_comparison.md report
    md_path = REPORTS_DIR / "model_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Performance Comparison Report\n\n")
        f.write(f"**Best Performing Model**: {comparison_summary['best_model']} (ROC-AUC: {comparison_summary['best_roc_auc']:.4f})\n\n")
        f.write("## Performance Metrics Table\n\n")
        f.write("| Model | ROC-AUC | Accuracy | Precision | Recall | F1 Score | Log Loss | Brier Score |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for item in comparison_summary["comparison"]:
            brier_str = f"{item['brier_score']:.4f}" if item['brier_score'] is not None else "N/A"
            f.write(f"| {item['model']} | {item['test_roc_auc']:.4f} | {item['test_accuracy']:.4f} | {item['test_precision']:.4f} | {item['test_recall']:.4f} | {item['test_f1']:.4f} | {item['test_log_loss']:.4f} | {brier_str} |\n")

    return comparison_summary
