name: MLOps CI Pipeline with my mlops project
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
permissions:
  contents: write
jobs:
  # ─────────────────────────────────────────────────────────────
  # JOB 1 — Run pytest
  # ─────────────────────────────────────────────────────────────
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Cache pip dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: ${{ runner.os }}-pip-
      - name: Install dependencies
        run: pip install -r requirements.txt pytest flask pandas
      - name: Run tests
        run: pytest tests/ -v
        env:
          MLFLOW_TRACKING_URI: "file:///tmp/mlruns"

  # ─────────────────────────────────────────────────────────────
  # JOB 2 — Train → Monitor → Build Docker → Deploy to ECS
  # ─────────────────────────────────────────────────────────────
  train-evaluate:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    env:
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      EXPERIMENT_NAME: champion_challenger_experiment
      MODEL_NAME: multi_model_classifier
      AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      AWS_DEFAULT_REGION: ap-south-1
      PYTHONPATH: ${{ github.workspace }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Cache pip dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: ${{ runner.os }}-pip-

      - name: Install dependencies
        run: pip install -r requirements.txt pytest flask pandas boto3

      # ── Create required directories ───────────────────────────────────────
      - name: Create required directories
        run: |
          mkdir -p reports
          mkdir -p data/processed
          mkdir -p models

      # ── Pull only raw data from DVC (not models — those get rebuilt) ──────
      - name: Pull raw data from DVC
        run: dvc pull data/raw/dataset.csv --force

      # ── Run full pipeline (always rebuilds models/ from scratch) ──────────
      - name: Run DVC Pipeline
        run: dvc repro --force

      # ── Push all new outputs (data + models) back to S3 ──────────────────
      - name: Push outputs to DVC remote
        run: dvc push

      # ── Commit updated dvc.lock so next run has correct hashes ───────────
      - name: Commit dvc.lock if changed
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add dvc.lock
          git diff --staged --quiet || git commit -m "ci: update dvc.lock [skip ci]"
          git push
