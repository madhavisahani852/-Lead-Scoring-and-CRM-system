"""
ml/training/retrain_service.py
Automated Retraining Workflow for XGBoost Lead Scoring Model.

Workflow:
Historical CRM outcomes -> Validate training data -> Feature engineering ->
Train XGBoost -> Evaluate model (ROC-AUC, Precision, Recall, F1) ->
Compare with current production model ->
If better -> Create new model version (promote)
If worse  -> Keep current production model (reject)
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier

from ml.config.feature_config import (
    CLEANED_DATA_PATH,
    MODELS_DIR,
    NUMERIC_COLS,
    NOMINAL_COLS,
    ORDINAL_COLS,
    DROP_COLS,
)
from ml.preprocessing.preprocessing import build_preprocessing_pipeline
from backend import supabase_db


def collect_resolved_training_data() -> pd.DataFrame:
    """
    Gathers historical leads with ground-truth outcomes (converted / Won / Lost).
    Merges live CRM outcomes with cleaned baseline records.
    """
    # 1. Baseline cleaned leads with target
    df_base = pd.DataFrame()
    if CLEANED_DATA_PATH.exists():
        df_base = pd.read_csv(CLEANED_DATA_PATH)
        df_base = df_base.dropna(subset=['target']).copy()

    # 2. Live resolved leads from CRM
    crm_leads = supabase_db.get_all_leads()
    crm_resolved = []
    for l in crm_leads:
        if l.get("converted") is not None:
            l_dict = l.copy()
            l_dict["target"] = 1 if l["converted"] else 0
            crm_resolved.append(l_dict)

    if crm_resolved:
        df_crm = pd.DataFrame(crm_resolved)
        combined = pd.concat([df_base, df_crm], ignore_index=True)
        # Deduplicate by email/id
        combined = combined.drop_duplicates(subset=['email'], keep='last')
        return combined

    return df_base


def execute_retraining_workflow() -> Dict[str, Any]:
    """
    Executes the end-to-end model evaluation and conditional promotion workflow.
    """
    data = collect_resolved_training_data()
    total_records = len(data)

    if total_records < 30 or data['target'].nunique() < 2:
        return {
            "success": False,
            "error": f"Insufficient training data with both outcomes. Current resolved records: {total_records}.",
            "promoted": False,
        }

    # Ensure all required columns are present
    feature_cols = list(NUMERIC_COLS) + list(NOMINAL_COLS) + list(ORDINAL_COLS) + list(DROP_COLS)
    for c in feature_cols:
        if c not in data.columns:
            data[c] = np.nan

    X = data[feature_cols].copy()
    y = data['target'].astype(int).values

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Build and fit preprocessing pipeline
    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Train Candidate XGBoost Classifier
    candidate_clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    candidate_clf.fit(X_train_proc, y_train)

    # Evaluate Candidate Model
    y_prob = candidate_clf.predict_proba(X_test_proc)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    cand_roc_auc = float(roc_auc_score(y_test, y_prob))
    cand_precision = float(precision_score(y_test, y_pred, zero_division=0))
    cand_recall = float(recall_score(y_test, y_pred, zero_division=0))
    cand_f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cand_accuracy = float(accuracy_score(y_test, y_pred))

    # Retrieve Current Production Baseline
    prod_info = supabase_db.get_production_model_info()
    prod_roc_auc = float(prod_info["test_metrics"]["roc_auc"])

    # Determine Version String
    existing_versions = supabase_db.get_model_versions()
    version_num = len(existing_versions) + 1
    new_version_tag = f"v{version_num}.0"

    # Evaluate promotion rule
    # Promoted ONLY IF candidate ROC-AUC > production baseline
    is_promoted = cand_roc_auc > prod_roc_auc

    candidate_version_record = {
        "version": new_version_tag,
        "algorithm": "XGBoost Classifier (Retrained)",
        "training_records": total_records,
        "roc_auc": round(cand_roc_auc, 4),
        "precision": round(cand_precision, 4),
        "recall": round(cand_recall, 4),
        "f1": round(cand_f1, 4),
        "status": "Production" if is_promoted else "Candidate (Not Promoted)",
        "selection_reason": (
            f"Candidate achieved higher ROC-AUC ({round(cand_roc_auc, 4)} vs {round(prod_roc_auc, 4)})"
            if is_promoted else
            f"Candidate ROC-AUC ({round(cand_roc_auc, 4)}) did not exceed production model ({round(prod_roc_auc, 4)}). Current production model preserved."
        ),
        "created_at": datetime.now().isoformat(),
    }

    # Register in version registry
    supabase_db.register_model_version(candidate_version_record)

    return {
        "success": True,
        "promoted": is_promoted,
        "candidate_version": new_version_tag,
        "candidate_metrics": {
            "roc_auc": round(cand_roc_auc, 4),
            "precision": round(cand_precision, 4),
            "recall": round(cand_recall, 4),
            "f1": round(cand_f1, 4),
            "accuracy": round(cand_accuracy, 4),
        },
        "production_metrics": prod_info["test_metrics"],
        "training_records": total_records,
        "message": (
            f"Candidate model {new_version_tag} promoted to Production (ROC-AUC {round(cand_roc_auc, 4)} > {round(prod_roc_auc, 4)})."
            if is_promoted else
            f"Retraining evaluated: Candidate ROC-AUC ({round(cand_roc_auc, 4)}) <= Production ({round(prod_roc_auc, 4)}). Production model preserved."
        ),
    }
