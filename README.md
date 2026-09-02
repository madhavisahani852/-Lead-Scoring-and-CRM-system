# Lead Scoring & CRM Intelligence Tool

A modular, team-friendly machine learning repository for scoring CRM sales leads, predicting customer conversion probabilities, and prioritizing sales pipeline activity.

---

## Project Structure

```text
Lead-Scoring-and-CRM-system/
│
├── cleaned_leads.csv               # Processed dataset for model training
├── raw_leads.csv                   # Raw historical CRM export dataset
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── ml/                             # Machine Learning Subsystem
│   ├── README.md                   # Team development guidelines & ownership model
│   ├── config/                     # Feature definitions & path resolution
│   ├── data/                       # Raw & processed data management
│   ├── preprocessing/              # Data cleaning & feature engineering transformers
│   ├── training/                   # Model training entry points (Baseline, RF, XGBoost)
│   ├── evaluation/                 # Metrics, ranking, calibration & model comparison
│   ├── inference/                  # Batch prediction & single lead scoring engine
│   ├── artifacts/                  # Model binaries, metric JSONs, & markdown reports
│   └── notebooks/                  # EDA & experimentation notebooks
│
├── backend/                        # Backend API service
├── frontend/                       # Web user interface
├── tests/                          # Automated pytest suite (preprocessing, training, inference)
└── docs/                           # System documentation & feature dictionary
```

---

## Getting Started

### 1. Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Model Training

Train the Logistic Regression baseline model:

```bash
python ml/training/train_baseline.py
```

Train all models (Baseline, Random Forest, XGBoost) and generate comparison reports:

```bash
python ml/training/train_models.py
```

### 3. Lead Scoring & Inference

Score a sample lead via CLI inference:

```bash
python ml/inference/score_lead.py
```

### 4. Running Automated Tests

Execute the full pytest test suite:

```bash
pytest tests/
```

---

## Documentation Links

- [ML System Architecture & Ownership](docs/ml_architecture.md)
- [Feature Dictionary](docs/feature_dictionary.md)
- [Model Documentation](docs/model_documentation.md)
- [ML Team Guidelines](ml/README.md)
