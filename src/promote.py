"""
Model promotion module.
Handles transitioning model versions through MLflow Registry stages:
  None -> Staging -> Production -> Archived
"""

from mlflow.tracking import MlflowClient


def promote_model(
    client: MlflowClient,
    model_name: str,
    version: str,
) -> None:
    """
    Promote a model version to Production.
    Archives the previous Production version if one exists.
    """
    # Archive current production version(s)
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for pv in prod_versions:
            print(f"  Archiving previous champion: {model_name} v{pv.version}")
            client.transition_model_version_stage(
                name=model_name,
                version=pv.version,
                stage="Archived",
            )
    except Exception:
        pass  # No existing production version

    # Also archive production versions of OTHER model types
    from src.evaluate import REGISTERED_MODELS
    for other_model in REGISTERED_MODELS:
        if other_model == model_name:
            continue
        try:
            prod_versions = client.get_latest_versions(
                other_model, stages=["Production"]
            )
            for pv in prod_versions:
                print(f"  Archiving: {other_model} v{pv.version}")
                client.transition_model_version_stage(
                    name=other_model,
                    version=pv.version,
                    stage="Archived",
                )
        except Exception:
            continue

    # Promote challenger to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
    )
    print(f"  ✓ Promoted {model_name} v{version} to Production!")
