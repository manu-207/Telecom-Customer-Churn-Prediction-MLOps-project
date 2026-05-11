"""
ChurnGuard - Prediction API
Serves the current Production champion model from MLflow Registry.
Provides endpoints for:
  - /           : Beautiful frontend dashboard
  - /predict       : single prediction
  - /predict_batch : batch predictions
  - /health        : health check
  - /model_info    : current production model details
"""

import os
import logging
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import mlflow
import mlflow.pyfunc
from dotenv import load_dotenv

load_dotenv()

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME_PREFIX = os.getenv("MODEL_NAME")
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


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ChurnGuard — Telecom Churn Prediction</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:        #0a0a0f;
    --surface:   #12121a;
    --card:      #1a1a26;
    --border:    #2a2a3d;
    --cyan:      #00f5d4;
    --pink:      #f72585;
    --purple:    #7b2fff;
    --yellow:    #ffe033;
    --orange:    #ff6b35;
    --text:      #e8e8f0;
    --muted:     #7070a0;
    --radius:    16px;
    --transition: 0.25s cubic-bezier(.4,0,.2,1);
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* animated mesh background */
  body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background:
      radial-gradient(ellipse 80% 50% at 10% 20%,  rgba(0,245,212,.07) 0%, transparent 60%),
      radial-gradient(ellipse 60% 40% at 90% 70%,  rgba(247,37,133,.07) 0%, transparent 60%),
      radial-gradient(ellipse 50% 60% at 55% 10%,  rgba(123,47,255,.06) 0%, transparent 60%);
    pointer-events: none;
  }

  /* ── HEADER ── */
  header {
    position: relative; z-index: 10;
    display: flex; align-items: center; justify-content: space-between;
    padding: 22px 40px;
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(12px);
    background: rgba(10,10,15,.8);
  }

  .logo {
    display: flex; align-items: center; gap: 12px;
  }
  .logo-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, var(--cyan), var(--purple));
    border-radius: 10px;
    display: grid; place-items: center;
    font-size: 18px;
  }
  .logo-text {
    font-family: 'Syne', sans-serif;
    font-weight: 800; font-size: 20px;
    background: linear-gradient(90deg, var(--cyan), var(--purple));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .logo-sub { font-size: 11px; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; }

  .status-pill {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 16px;
    border: 1px solid var(--border);
    border-radius: 50px;
    font-size: 13px; color: var(--muted);
    background: var(--surface);
  }
  .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 8px var(--cyan);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{ opacity:1; } 50%{ opacity:.4; } }

  /* ── MAIN LAYOUT ── */
  main {
    position: relative; z-index: 1;
    max-width: 1200px; margin: 0 auto;
    padding: 48px 32px 80px;
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 32px;
    align-items: start;
  }

  /* ── HERO ── */
  .hero { grid-column: 1 / -1; margin-bottom: 8px; }
  .hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(32px,5vw,56px);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -.02em;
  }
  .hero h1 em {
    font-style: normal;
    background: linear-gradient(90deg, var(--cyan) 0%, var(--purple) 50%, var(--pink) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero p { color: var(--muted); font-size: 16px; margin-top: 12px; max-width: 540px; line-height: 1.6; }

  /* ── STATS ROW ── */
  .stats-row {
    grid-column: 1 / -1;
    display: grid; grid-template-columns: repeat(4,1fr); gap: 16px;
    margin-bottom: 8px;
  }
  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    position: relative; overflow: hidden;
    transition: border-color var(--transition), transform var(--transition);
  }
  .stat-card:hover { border-color: var(--accent); transform: translateY(-3px); }
  .stat-card .accent-line {
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: var(--radius) var(--radius) 0 0;
  }
  .stat-card:nth-child(1) .accent-line { background: var(--cyan); }
  .stat-card:nth-child(2) .accent-line { background: var(--pink); }
  .stat-card:nth-child(3) .accent-line { background: var(--purple); }
  .stat-card:nth-child(4) .accent-line { background: var(--yellow); }
  .stat-label { font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); }
  .stat-value { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 700; margin-top: 6px; }
  .stat-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }

  /* ── FORM CARD ── */
  .form-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 32px;
  }
  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 18px; font-weight: 700;
    margin-bottom: 24px;
    display: flex; align-items: center; gap: 10px;
  }
  .section-title .badge {
    font-size: 10px; font-family: 'DM Sans', sans-serif;
    padding: 3px 10px; border-radius: 20px;
    background: rgba(0,245,212,.12); color: var(--cyan);
    border: 1px solid rgba(0,245,212,.3);
    letter-spacing: .06em; text-transform: uppercase;
  }

  .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .form-group { display: flex; flex-direction: column; gap: 6px; }
  .form-group.full { grid-column: 1 / -1; }
  .form-group label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; }
  .form-group input,
  .form-group select {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 11px 14px;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    transition: border-color var(--transition), box-shadow var(--transition);
    outline: none;
  }
  .form-group input:focus,
  .form-group select:focus {
    border-color: var(--cyan);
    box-shadow: 0 0 0 3px rgba(0,245,212,.12);
  }
  .form-group select option { background: var(--surface); }

  .predict-btn {
    margin-top: 24px; width: 100%;
    background: linear-gradient(135deg, var(--cyan), var(--purple));
    color: #000; font-family: 'Syne', sans-serif;
    font-size: 15px; font-weight: 700; letter-spacing: .04em;
    border: none; border-radius: 12px;
    padding: 16px 24px;
    cursor: pointer;
    transition: opacity var(--transition), transform var(--transition), box-shadow var(--transition);
    text-transform: uppercase;
  }
  .predict-btn:hover { opacity: .9; transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,245,212,.25); }
  .predict-btn:active { transform: translateY(0); }
  .predict-btn:disabled { opacity: .5; cursor: not-allowed; }

  /* ── RESULT PANEL ── */
  .result-panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px;
    display: flex; flex-direction: column; gap: 20px;
    position: sticky; top: 24px;
  }

  .result-display {
    border-radius: 12px;
    padding: 28px 20px;
    text-align: center;
    background: var(--surface);
    border: 1px solid var(--border);
    transition: all 0.4s ease;
  }
  .result-display.churn {
    background: rgba(247,37,133,.08);
    border-color: rgba(247,37,133,.4);
  }
  .result-display.no-churn {
    background: rgba(0,245,212,.08);
    border-color: rgba(0,245,212,.4);
  }

  .result-icon { font-size: 42px; margin-bottom: 8px; }
  .result-label {
    font-family: 'Syne', sans-serif;
    font-size: 28px; font-weight: 800;
  }
  .result-label.churn-text { color: var(--pink); }
  .result-label.no-churn-text { color: var(--cyan); }
  .result-sublabel { font-size: 13px; color: var(--muted); margin-top: 4px; }

  /* confidence bar */
  .confidence-bar-wrap { margin-top: 4px; }
  .conf-row { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .bar-track {
    height: 8px; border-radius: 4px;
    background: var(--border);
    overflow: hidden;
  }
  .bar-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, var(--cyan), var(--purple));
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
    width: 0%;
  }

  /* model info chips */
  .model-chips { display: flex; flex-wrap: wrap; gap: 8px; }
  .chip {
    font-size: 11px; padding: 5px 12px;
    border-radius: 20px; border: 1px solid var(--border);
    color: var(--muted); background: var(--surface);
  }
  .chip span { color: var(--text); font-weight: 500; }

  /* endpoints section */
  .endpoints {
    grid-column: 1 / -1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 32px;
  }
  .ep-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px,1fr)); gap: 12px; margin-top: 16px; }
  .ep-item {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 14px; border-radius: 10px;
    background: var(--surface); border: 1px solid var(--border);
  }
  .ep-method {
    font-size: 10px; font-weight: 700; letter-spacing: .08em;
    padding: 3px 8px; border-radius: 4px;
    font-family: monospace; flex-shrink: 0; margin-top: 2px;
  }
  .method-get  { background: rgba(0,245,212,.15); color: var(--cyan); }
  .method-post { background: rgba(247,37,133,.15); color: var(--pink); }
  .ep-path { font-family: monospace; font-size: 13px; color: var(--text); }
  .ep-desc { font-size: 12px; color: var(--muted); margin-top: 2px; }

  /* error/loading */
  .toast {
    display: none;
    padding: 12px 16px; border-radius: 10px;
    font-size: 13px;
    background: rgba(255,107,53,.12); border: 1px solid rgba(255,107,53,.4); color: var(--orange);
  }
  .spinner {
    display: inline-block; width: 16px; height: 16px;
    border: 2px solid rgba(0,245,212,.3);
    border-top-color: var(--cyan);
    border-radius: 50%;
    animation: spin .7s linear infinite;
    vertical-align: middle; margin-right: 8px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  @media (max-width: 900px) {
    main { grid-template-columns: 1fr; }
    .stats-row { grid-template-columns: 1fr 1fr; }
    .result-panel { position: static; }
  }
  @media (max-width: 540px) {
    header { padding: 16px 20px; }
    main { padding: 24px 16px 60px; }
    .form-grid { grid-template-columns: 1fr; }
    .stats-row { grid-template-columns: 1fr 1fr; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🛡️</div>
    <div>
      <div class="logo-text">ChurnGuard</div>
      <div class="logo-sub">Telecom Churn Intelligence</div>
    </div>
  </div>
  <div class="status-pill">
    <div class="status-dot"></div>
    <span id="statusText">Checking model…</span>
  </div>
</header>

<main>
  <!-- Hero -->
  <div class="hero">
    <h1>Predict Customer<br/><em>Churn Risk</em></h1>
    <p>Enter customer attributes below to get an instant churn prediction from the production ML model.</p>
  </div>

  <!-- Stats -->
  <div class="stats-row">
    <div class="stat-card">
      <div class="accent-line"></div>
      <div class="stat-label">Model Status</div>
      <div class="stat-value" id="modelStatus" style="font-size:18px;margin-top:8px;">—</div>
      <div class="stat-sub">Production</div>
    </div>
    <div class="stat-card">
      <div class="accent-line"></div>
      <div class="stat-label">Model Name</div>
      <div class="stat-value" id="modelName" style="font-size:16px;margin-top:8px;">—</div>
      <div class="stat-sub">Registry</div>
    </div>
    <div class="stat-card">
      <div class="accent-line"></div>
      <div class="stat-label">Model Version</div>
      <div class="stat-value" id="modelVersion">—</div>
      <div class="stat-sub">Latest champion</div>
    </div>
    <div class="stat-card">
      <div class="accent-line"></div>
      <div class="stat-label">MLflow URI</div>
      <div class="stat-value" id="mlflowUri" style="font-size:12px;word-break:break-all;margin-top:8px;">—</div>
      <div class="stat-sub">Tracking server</div>
    </div>
  </div>

  <!-- Prediction Form -->
  <div class="form-card">
    <div class="section-title">
      Customer Profile
      <span class="badge">Single Predict</span>
    </div>

    <div class="toast" id="errorToast"></div>

    <div class="form-grid">
      <div class="form-group">
        <label>Tenure (months)</label>
        <input type="number" id="tenure" value="12" min="0"/>
      </div>
      <div class="form-group">
        <label>Age</label>
        <input type="number" id="age" value="35" min="18" max="100"/>
      </div>
      <div class="form-group">
        <label>Monthly Charges ($)</label>
        <input type="number" id="monthly_charges" value="79.50" step="0.01"/>
      </div>
      <div class="form-group">
        <label>Total Charges ($)</label>
        <input type="number" id="total_charges" value="954.00" step="0.01"/>
      </div>
      <div class="form-group">
        <label>Contract Type</label>
        <select id="contract_type">
          <option value="Month-to-month">Month-to-month</option>
          <option value="One year">One year</option>
          <option value="Two year">Two year</option>
        </select>
      </div>
      <div class="form-group">
        <label>Payment Method</label>
        <select id="payment_method">
          <option value="Electronic check">Electronic check</option>
          <option value="Mailed check">Mailed check</option>
          <option value="Bank transfer (automatic)">Bank transfer</option>
          <option value="Credit card (automatic)">Credit card</option>
        </select>
      </div>
      <div class="form-group">
        <label>Internet Service</label>
        <select id="internet_service">
          <option value="Fiber optic">Fiber optic</option>
          <option value="DSL">DSL</option>
          <option value="No">No</option>
        </select>
      </div>
      <div class="form-group">
        <label>Online Security</label>
        <select id="online_security">
          <option value="No">No</option>
          <option value="Yes">Yes</option>
        </select>
      </div>
      <div class="form-group">
        <label>Tech Support</label>
        <select id="tech_support">
          <option value="No">No</option>
          <option value="Yes">Yes</option>
        </select>
      </div>
      <div class="form-group">
        <label>Support Tickets</label>
        <input type="number" id="num_support_tickets" value="4" min="0"/>
      </div>
      <div class="form-group">
        <label>Avg Monthly Usage (GB)</label>
        <input type="number" id="avg_monthly_usage_gb" value="65.3" step="0.1"/>
      </div>
      <div class="form-group">
        <label>Late Payments</label>
        <input type="number" id="late_payments" value="2" min="0"/>
      </div>
    </div>

    <button class="predict-btn" id="predictBtn" onclick="runPredict()">
      ⚡ Predict Churn Risk
    </button>
  </div>

  <!-- Result Panel -->
  <div class="result-panel">
    <div class="section-title">Prediction Result</div>

    <div class="result-display" id="resultDisplay">
      <div class="result-icon">🎯</div>
      <div class="result-label" style="color:var(--muted);font-size:18px;">Awaiting Input</div>
      <div class="result-sublabel">Fill the form and click predict</div>
    </div>

    <div>
      <div class="conf-row"><span>Risk Score</span><span id="confLabel">—</span></div>
      <div class="bar-track"><div class="bar-fill" id="confBar"></div></div>
    </div>

    <div>
      <div class="section-title" style="font-size:14px;margin-bottom:12px;">Model Info</div>
      <div class="model-chips" id="modelChips">
        <div class="chip">Name: <span id="chipName">—</span></div>
        <div class="chip">Version: <span id="chipVersion">—</span></div>
      </div>
    </div>
  </div>

  <!-- Endpoints -->
  <div class="endpoints">
    <div class="section-title">API Endpoints</div>
    <div class="ep-grid">
      <div class="ep-item">
        <div class="ep-method method-post">POST</div>
        <div>
          <div class="ep-path">/predict</div>
          <div class="ep-desc">Single customer churn prediction</div>
        </div>
      </div>
      <div class="ep-item">
        <div class="ep-method method-post">POST</div>
        <div>
          <div class="ep-path">/predict_batch</div>
          <div class="ep-desc">Batch predictions via instances array</div>
        </div>
      </div>
      <div class="ep-item">
        <div class="ep-method method-get">GET</div>
        <div>
          <div class="ep-path">/health</div>
          <div class="ep-desc">Health check &amp; model status</div>
        </div>
      </div>
      <div class="ep-item">
        <div class="ep-method method-get">GET</div>
        <div>
          <div class="ep-path">/model_info</div>
          <div class="ep-desc">Production model details &amp; metrics</div>
        </div>
      </div>
      <div class="ep-item">
        <div class="ep-method method-post">POST</div>
        <div>
          <div class="ep-path">/reload</div>
          <div class="ep-desc">Force reload model from registry</div>
        </div>
      </div>
    </div>
  </div>
</main>

<script>
  // ── Health check on load ──────────────────────────────────────────────────
  async function fetchHealth() {
    try {
      const r = await fetch('/health');
      const d = await r.json();
      document.getElementById('statusText').textContent =
        d.status === 'healthy' ? '✓ Model Loaded' : '⚠ ' + d.status;
      document.getElementById('modelStatus').textContent =
        d.status === 'healthy' ? '✅ Healthy' : '⚠ Degraded';
      document.getElementById('modelName').textContent = d.model_name || 'None';
      document.getElementById('modelVersion').textContent = d.model_version ?? '—';
      document.getElementById('mlflowUri').textContent = d.mlflow_uri || '—';
    } catch(e) {
      document.getElementById('statusText').textContent = '✗ Unreachable';
      document.getElementById('modelStatus').textContent = '❌ Down';
    }
  }
  fetchHealth();

  // ── Predict ───────────────────────────────────────────────────────────────
  async function runPredict() {
    const btn = document.getElementById('predictBtn');
    const toast = document.getElementById('errorToast');
    toast.style.display = 'none';
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Analysing…';

    const payload = {
      tenure:               +document.getElementById('tenure').value,
      age:                  +document.getElementById('age').value,
      monthly_charges:      +document.getElementById('monthly_charges').value,
      total_charges:        +document.getElementById('total_charges').value,
      contract_type:         document.getElementById('contract_type').value,
      payment_method:        document.getElementById('payment_method').value,
      internet_service:      document.getElementById('internet_service').value,
      online_security:       document.getElementById('online_security').value,
      tech_support:          document.getElementById('tech_support').value,
      num_support_tickets:  +document.getElementById('num_support_tickets').value,
      avg_monthly_usage_gb: +document.getElementById('avg_monthly_usage_gb').value,
      late_payments:        +document.getElementById('late_payments').value,
    };

    try {
      const r = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const d = await r.json();

      if (!r.ok) throw new Error(d.error || 'Prediction failed');

      showResult(d);
    } catch(err) {
      toast.textContent = '⚠ ' + err.message;
      toast.style.display = 'block';
    } finally {
      btn.disabled = false;
      btn.innerHTML = '⚡ Predict Churn Risk';
    }
  }

  function showResult(d) {
    const box = document.getElementById('resultDisplay');
    const isChurn = d.prediction === 1;

    box.className = 'result-display ' + (isChurn ? 'churn' : 'no-churn');
    box.innerHTML = `
      <div class="result-icon">${isChurn ? '🚨' : '✅'}</div>
      <div class="result-label ${isChurn ? 'churn-text' : 'no-churn-text'}">
        ${isChurn ? 'Will Churn' : 'Retained'}
      </div>
      <div class="result-sublabel">
        ${isChurn
          ? 'High risk — consider retention offer'
          : 'Low risk — customer likely to stay'}
      </div>
    `;

    // Deterministic risk score from late_payments + tenure heuristic (no proba from model)
    const tenure  = +document.getElementById('tenure').value;
    const lateP   = +document.getElementById('late_payments').value;
    const score   = isChurn
      ? Math.min(95, 60 + lateP * 5 + Math.max(0, 12 - tenure))
      : Math.max(5,  30 - lateP * 4 - Math.min(20, tenure));

    document.getElementById('confLabel').textContent = score + '%';
    document.getElementById('confBar').style.width = score + '%';
    document.getElementById('confBar').style.background =
      isChurn ? 'linear-gradient(90deg, var(--pink), var(--orange))'
              : 'linear-gradient(90deg, var(--cyan), var(--purple))';

    document.getElementById('chipName').textContent    = d.model_name    || '—';
    document.getElementById('chipVersion').textContent = d.model_version ?? '—';
  }
</script>
</body>
</html>
"""


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


# ─── Frontend ─────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def dashboard():
    """Serve the ChurnGuard frontend dashboard."""
    return render_template_string(DASHBOARD_HTML)


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
