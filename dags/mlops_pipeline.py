"""
Airflow DAG: Champion/Challenger MLOps Pipeline
Orchestrates: preprocess -> train (3 models) -> evaluate -> promote
"""

import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="mlops_champion_challenger",
    default_args=default_args,
    description="Train 3 models, evaluate champion vs challenger, auto-promote",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "champion-challenger", "model-registry"],
) as dag:

    # Stage 1: Pull latest data with DVC
    pull_data = BashOperator(
        task_id="pull_data",
        bash_command="cd /opt/airflow/project && dvc pull",
    )

    # Stage 2: Preprocess
    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="cd /opt/airflow/project && python src/preprocess.py",
    )

    # Stage 3: Train all models
    train_models = BashOperator(
        task_id="train_models",
        bash_command="cd /opt/airflow/project && python src/train.py",
        env={"MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
    )

    # Stage 4: Evaluate and auto-promote
    evaluate_promote = BashOperator(
        task_id="evaluate_and_promote",
        bash_command="cd /opt/airflow/project && python src/evaluate.py",
        env={"MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
    )

    # Stage 5: Notify (optional)
    notify_success = BashOperator(
        task_id="notify_success",
        bash_command='echo "Pipeline completed. Check MLflow for results."',
    )

    # DAG dependency chain
    pull_data >> preprocess >> train_models >> evaluate_promote >> notify_success
