"""
ChurnGuard - Prediction API
Serves the current Production champion model from MLflow Registry.
Provides endpoints for:
  - /predict       : single prediction
  - /predict_batch : batch predictions
  - /health        : health check
  - /model_info    : current production model details
"""

import os
import logging
from flask import Flask, request, jsonify
import pandas as pd
import mlflow
import mlflow.pyfunc

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME_PREFIX = "multi_model_classifier"
REGISTERED_MODELS = [
    f"{MODEL_NAME_PREFIX}_rf",
    f"{MODEL_NAME_PREFIX}_xgb",
    f"{MODEL_NAME_PREFIX}_lgbm",
]
PORT = int(os.getenv("API_PORT", 5001))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_production_model():
    """
    Load the current Production model from MLflow Registry.
    Searches across all registered model names for one in Production stage.
    """
    client = mlflow.tracking.MlflowClient()

    for model_name in REGISTERED_MODELS:
        try:
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                version = versions[0]
                model_uri = f"models:/{model_name}/Production"
                model = mlflow.pyfunc.load_model(model_uri)
                logger.info(
                    f"Loaded model: {model_name} v{version.version}"
                )
                return model, model_name, version.version
        except Exception as e:
            logger.warning(f"Could not load {model_name}: {e}")
            continue

    logger.error("No production model found in registry!")
    return None, None, None


# Global model cache
_model_cache = {"model": None, "name": None, "version": None}


def get_model():
    """Get cached model or load from registry."""
    if _model_cache["model"] is None:
        model, name, version = load_production_model()
        _model_cache["model"] = model
        _model_cache["name"] = name
        _model_cache["version"] = version
    return _model_cache["model"], _model_cache["name"], _model_cache["version"]


def reload_model():
    """Force reload model from registry (after promotion)."""
    _model_cache["model"] = None
    _model_cache["name"] = None
    _model_cache["version"] = None
    return get_model()


# ─── API Endpoints ────────────────────────────────────────────────────────────


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    model, name, version = get_model()
    status = "healthy" if model is not None else "no_model_loaded"
    return jsonify({
        "status": status,
        "model_name": name,
        "model_version": version,
        "mlflow_uri": MLFLOW_TRACKING_URI,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single prediction endpoint.

    Request body (JSON):
    {
        "tenure": 12,
        "monthly_charges": 79.5,
        "total_charges": 954.0,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "tech_support": "No",
        "num_support_tickets": 4,
        "avg_monthly_usage_gb": 65.3,
        "late_payments": 2,
        "age": 35
    }
    """
    try:
        model, name, version = get_model()
        if model is None:
            return jsonify({"error": "No production model available"}), 503

        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert to DataFrame
        df = pd.DataFrame([data])
        prediction = model.predict(df)

        return jsonify({
            "prediction": int(prediction[0]),
            "churn_label": "Yes" if prediction[0] == 1 else "No",
            "model_name": name,
            "model_version": version,
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction endpoint.

    Request body (JSON):
    {
        "instances": [
            {"tenure": 12, "monthly_charges": 79.5, ...},
            {"tenure": 45, "monthly_charges": 35.2, ...}
        ]
    }
    """
    try:
        model, name, version = get_model()
        if model is None:
            return jsonify({"error": "No production model available"}), 503

        data = request.get_json()
        if not data or "instances" not in data:
            return jsonify({"error": "Provide 'instances' array in body"}), 400

        df = pd.DataFrame(data["instances"])
        predictions = model.predict(df)

        results = []
        for pred in predictions:
            results.append({
                "prediction": int(pred),
                "churn_label": "Yes" if pred == 1 else "No",
            })

        return jsonify({
            "predictions": results,
            "count": len(results),
            "model_name": name,
            "model_version": version,
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/model_info", methods=["GET"])
def model_info():
    """Get current production model details."""
    model, name, version = get_model()
    if model is None:
        return jsonify({"error": "No production model loaded"}), 503

    client = mlflow.tracking.MlflowClient()
    model_version = client.get_model_version(name, version)
    run = client.get_run(model_version.run_id)

    return jsonify({
        "model_name": name,
        "model_version": version,
        "stage": model_version.current_stage,
        "run_id": model_version.run_id,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "created_at": str(model_version.creation_timestamp),
    })


@app.route("/reload", methods=["POST"])
def reload():
    """Force reload model from registry (call after promotion)."""
    model, name, version = reload_model()
    if model is not None:
        return jsonify({"status": "reloaded", "model_name": name, "version": version})
    return jsonify({"status": "failed", "error": "No production model found"}), 503


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"Starting ChurnGuard API on port {PORT}")
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    # Try to load model on startup
    model, name, version = get_model()
    if model:
        logger.info(f"Model loaded: {name} v{version}")
    else:
        logger.warning("No production model found. API will return 503 until a model is promoted.")

    app.run(host="0.0.0.0", port=PORT, debug=False)
