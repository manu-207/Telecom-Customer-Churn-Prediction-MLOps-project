"""
Model training module.
Trains 3 competing models (RF, XGBoost, LightGBM),
logs all runs to MLflow, and registers each model version.
"""

import os
import yaml
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_processed_data() -> tuple:
    """Load preprocessed train/test splits."""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }
    if y_proba is not None:
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        elif y_proba.ndim == 2:
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted"
            )
    return metrics


def train_random_forest(X_train, y_train, X_test, y_test, params: dict) -> dict:
    """Train Random Forest and log to MLflow."""
    with mlflow.start_run(run_name="RandomForest_Challenger") as run:
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"{MODEL_NAME}_rf",
        )

        print(f"  RF  -> F1: {metrics['f1_score']:.4f} | Acc: {metrics['accuracy']:.4f}")
        return {"run_id": run.info.run_id, "metrics": metrics, "model_type": "rf"}


def train_xgboost(X_train, y_train, X_test, y_test, params: dict) -> dict:
    """Train XGBoost and log to MLflow."""
    with mlflow.start_run(run_name="XGBoost_Challenger") as run:
        model = XGBClassifier(**params, eval_metric="logloss", use_label_encoder=False)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"{MODEL_NAME}_xgb",
        )

        print(f"  XGB -> F1: {metrics['f1_score']:.4f} | Acc: {metrics['accuracy']:.4f}")
        return {"run_id": run.info.run_id, "metrics": metrics, "model_type": "xgb"}


def train_lightgbm(X_train, y_train, X_test, y_test, params: dict) -> dict:
    """Train LightGBM and log to MLflow."""
    with mlflow.start_run(run_name="LightGBM_Challenger") as run:
        model = LGBMClassifier(**params, verbose=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
        )

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"{MODEL_NAME}_lgbm",
        )

        print(f"  LGBM-> F1: {metrics['f1_score']:.4f} | Acc: {metrics['accuracy']:.4f}")
        return {"run_id": run.info.run_id, "metrics": metrics, "model_type": "lgbm"}


def main():
    """Train all 3 models and register in MLflow."""
    params = load_params()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow experiment: {EXPERIMENT_NAME}")

    X_train, X_test, y_train, y_test = load_processed_data()
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

    print("\nTraining models...")
    results = []

    results.append(
        train_random_forest(X_train, y_train, X_test, y_test, params["random_forest"])
    )
    results.append(
        train_xgboost(X_train, y_train, X_test, y_test, params["xgboost"])
    )
    results.append(
        train_lightgbm(X_train, y_train, X_test, y_test, params["lightgbm"])
    )

    # Save results summary
    os.makedirs("reports", exist_ok=True)
    import json
    with open("reports/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nAll models trained and registered in MLflow!")
    return results


if __name__ == "__main__":
    main()
