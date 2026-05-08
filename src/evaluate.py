"""
Champion vs Challenger evaluation module.
Compares all registered model versions, selects the best challenger,
and triggers auto-promotion if it beats the current champion.
"""

import os
import json
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
REGISTERED_MODELS = [
    f"{MODEL_NAME}_rf",
    f"{MODEL_NAME}_xgb",
    f"{MODEL_NAME}_lgbm",
]


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def compute_weighted_score(metrics: dict, weights: dict) -> float:
    """Compute weighted composite score from multiple metrics."""
    score = 0.0
    for metric_name, weight in weights.items():
        if metric_name in metrics:
            score += metrics[metric_name] * weight
    return score


def get_champion_score(client: MlflowClient, weights: dict) -> tuple:
    """
    Get the current production (champion) model's score.
    Returns (score, model_name, version) or (None, None, None) if no champion.
    """
    for model_name in REGISTERED_MODELS:
        try:
            # Search for versions with 'Production' alias
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                version = versions[0]
                run = client.get_run(version.run_id)
                metrics = run.data.metrics
                score = compute_weighted_score(metrics, weights)
                return score, model_name, version.version
        except Exception:
            continue
    return None, None, None


def get_best_challenger(client: MlflowClient, weights: dict) -> tuple:
    """
    Find the best challenger across all registered models.
    Returns (score, model_name, version, metrics).
    """
    best_score = -1
    best_info = (None, None, None, None)

    for model_name in REGISTERED_MODELS:
        try:
            versions = client.get_latest_versions(model_name, stages=["None", "Staging"])
            for version in versions:
                run = client.get_run(version.run_id)
                metrics = run.data.metrics
                score = compute_weighted_score(metrics, weights)

                if score > best_score:
                    best_score = score
                    best_info = (score, model_name, version.version, metrics)
        except Exception:
            continue

    return best_info


def evaluate_and_promote() -> dict:
    """
    Main evaluation logic:
    1. Find current champion (Production model)
    2. Find best challenger (latest non-Production versions)
    3. Compare and auto-promote if challenger wins
    """
    params = load_params()
    promotion_params = params["promotion"]
    weights = promotion_params["metrics_weights"]
    min_improvement = promotion_params["min_improvement"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    print("=" * 60)
    print("CHAMPION vs CHALLENGER EVALUATION")
    print("=" * 60)

    # Get champion
    champion_score, champion_model, champion_version = get_champion_score(
        client, weights
    )

    if champion_score is not None:
        print(f"\nCurrent Champion: {champion_model} v{champion_version}")
        print(f"  Weighted Score: {champion_score:.4f}")
    else:
        print("\nNo current champion (first run).")
        champion_score = 0.0

    # Get best challenger
    challenger_score, challenger_model, challenger_version, challenger_metrics = (
        get_best_challenger(client, weights)
    )

    if challenger_score is None:
        print("\nNo challengers found. Nothing to evaluate.")
        return {"promoted": False, "reason": "no_challengers"}

    print(f"\nBest Challenger: {challenger_model} v{challenger_version}")
    print(f"  Weighted Score: {challenger_score:.4f}")
    if challenger_metrics:
        for k, v in challenger_metrics.items():
            print(f"    {k}: {v:.4f}")

    # Compare
    improvement = challenger_score - champion_score
    print(f"\nImprovement: {improvement:.4f} (threshold: {min_improvement})")

    result = {
        "champion_score": champion_score,
        "challenger_score": challenger_score,
        "challenger_model": challenger_model,
        "challenger_version": challenger_version,
        "improvement": improvement,
    }

    if improvement >= min_improvement:
        print(f"\n✓ Challenger WINS! Promoting {challenger_model} v{challenger_version}")
        from src.promote import promote_model
        promote_model(client, challenger_model, challenger_version)
        result["promoted"] = True
        result["reason"] = "challenger_beats_champion"
    else:
        print(f"\n✗ Champion holds. Challenger stays in Staging/Archived.")
        # Move challenger to Staging
        client.transition_model_version_stage(
            name=challenger_model,
            version=challenger_version,
            stage="Staging",
        )
        result["promoted"] = False
        result["reason"] = "insufficient_improvement"

    # Save evaluation report
    with open("reports/metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 60)
    return result


def main():
    evaluate_and_promote()


if __name__ == "__main__":
    main()
