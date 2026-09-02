# Model Performance Comparison Report

**Best Performing Model**: Random Forest (ROC-AUC: 0.7875)

## Performance Metrics Table

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 Score | Log Loss | Brier Score |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.7875 | 0.7304 | 0.7717 | 0.7903 | 0.7809 | 0.5493 | 0.1851 |
| Logistic Regression Baseline | 0.7722 | 0.7255 | 0.7656 | 0.7903 | 0.7778 | 0.5569 | 0.1878 |
| XGBoost (HistGradientBoosting Fallback) | 0.7628 | 0.6961 | 0.7313 | 0.7903 | 0.7597 | 0.5847 | 0.1974 |
