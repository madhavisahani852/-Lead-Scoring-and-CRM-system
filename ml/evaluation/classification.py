from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    precision_recall_curve,
    auc,
)


def evaluate_classification(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes standard binary classification evaluation metrics.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = np.asarray(predictions)

    metrics = {}
    metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    metrics["accuracy"] = float(accuracy_score(y_true, predictions))
    metrics["precision"] = float(precision_score(y_true, predictions, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, predictions, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, predictions, zero_division=0))
    metrics["log_loss"] = float(log_loss(y_true, probabilities))

    # Precision-Recall AUC
    p_curve, r_curve, _ = precision_recall_curve(y_true, probabilities)
    metrics["pr_auc"] = float(auc(r_curve, p_curve))

    # Confusion Matrix
    cm = confusion_matrix(y_true, predictions)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def print_classification_metrics(title: str, metrics: Dict[str, Any]) -> None:
    """
    Pretty prints classification metrics.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics.get('pr_auc', 0.0):.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Log Loss:  {metrics['log_loss']:.4f}")
