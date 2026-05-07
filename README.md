# Multi-Model Registry with Automated Promotion

Train 3 competing models (Random Forest, XGBoost, LightGBM) → run challenger vs champion evaluation → auto-promote the winner to production via MLflow Model Registry.

## Tech Stack
- **MLflow** – experiment tracking & model registry
- **XGBoost / LightGBM / scikit-learn** – model training
- **DVC** – data versioning
- **Airflow** – pipeline orchestration
- **pytest** – testing

## Project Structure
```
manu/
├── data/                   # DVC-tracked data
│   └── .gitkeep
├── src/
│   ├── preprocess.py       # Data loading & feature engineering
│   ├── train.py            # Train all 3 models, log to MLflow
│   ├── evaluate.py         # Champion vs challenger evaluation
│   └── promote.py          # Auto-promote best model to Production
├── dags/
│   └── mlops_pipeline.py   # Airflow DAG
├── tests/
│   ├── test_preprocess.py
│   └── test_evaluate.py
├── dvc.yaml                # DVC pipeline stages
├── params.yaml             # Hyperparameters
├── requirements.txt
└── README.md
```

## Quick Start
```bash
pip install -r requirements.txt
dvc pull                    # pull data
mlflow server &             # start tracking server
python src/train.py         # train & register models
python src/evaluate.py      # evaluate & auto-promote
```

## Airflow
```bash
airflow dags trigger mlops_champion_challenger
```
