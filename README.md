# Preempt Analytics — Predictive Maintenance Capstone

Capstone project 2 of the AI Engineering bootcamp at neuefische.
Predictive maintenance with a focus on MLOps — by Nate and Ivo.

---

## What this project does

Predicts equipment failure and failure mode from real-time sensor readings before the machine stops working. Built on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (UCI). The model is as much a deliverable as the pipeline that runs it.

**Research question:**
_To what extent can a machine learning model predict equipment failure and failure mode from real-time sensor readings, and what kind of MLOps pipeline is optimal to support that?_

---

## What the model predicts

CNC-type industrial machine with five distinct failure modes:

| Code | Failure                  | Trigger                                       |
| ---- | ------------------------ | --------------------------------------------- |
| TWF  | Tool Wear Failure        | Tool exceeds wear threshold                   |
| HDF  | Heat Dissipation Failure | Temperature differential too low at low RPM   |
| PWF  | Power Failure            | Torque × speed outside safe power band        |
| OSF  | Overstrain Failure       | Tool wear × torque exceeds strain limit       |
| RNF  | Random Failure           | Unpredictable — ~0.1% of cases               |

---

## Architecture — two loops

```
Inference loop:
  sensor_simulator.py ──POST /predict──► api.py ──MLflow @production──► prediction
                                                                            │
                                                                            ▼
                                                                      simulation.db

Retrain loop:
  simulation.db ──► detect_drift.py ──► export_simulation_to_parquet.py ──► DagsHub (DVC)
                                                                            │
                                                              retrain.trigger updated
                                                                            │
                                                                            ▼
                                                              GitHub Actions retrain workflow
                                                                            │
                                                              dvc repro ──► promote_model.py
                                                                            │
                                                                    new @production alias
```

The two loops are intentionally decoupled. The API serves the current `@production` model continuously. The retrain loop fires only when Evidently AI detects a meaningful distribution shift — a data push without drift does not trigger retraining.

---

## Project structure

```
├── data/
│   ├── ai4i2020.parquet            # DVC-tracked training dataset (grows with each export)
│   ├── ai4i2020.csv                # Original UCI dataset — kept for inspection only, not used by any script
│   └── ai4i2020_baseline.csv      # Frozen drift reference — never modified
├── src/
│   ├── feature_transformation.py  # Single source of truth for all feature engineering
│   ├── modeling_pipeline.py       # DVC training stages — 12 model families
│   ├── sensor_simulator.py        # Generates readings, calls API, stores in SQLite
│   └── api.py                     # FastAPI serving layer — loads @production from MLflow
├── scripts/
│   ├── detect_drift.py            # Evidently AI: simulation.db vs baseline CSV
│   ├── export_simulation_to_parquet.py # ETL: simulation.db → AI4I Parquet format → DagsHub
│   └── promote_model.py           # Two-gate promotion: improvement + floor → @production
├── .github/workflows/
│   └── retrain.yml                # Triggered by retrain.trigger; retrains XGBoost + promotes
├── retrain.trigger                # Sentinel file — updated on drift; GitHub Actions watches this
├── dvc.yaml                       # DVC pipeline definition — 12 training stages
├── params.yaml                    # Hyperparameters for all model families
└── simulation.db                  # Local SQLite — gitignored; holds live sensor readings
```

---

## Training data — two files, two purposes

The `data/` directory contains two files for the same dataset and one important distinction between them:

| File | Format | Purpose | Modified by scripts? |
|------|--------|---------|----------------------|
| `ai4i2020.parquet` | Parquet (binary) | Active training dataset — grows with each retrain cycle | Yes — appended by `export_simulation_to_parquet.py` |
| `ai4i2020.csv` | CSV (text) | Human-readable copy of the original UCI dataset | No — kept for inspection only |
| `ai4i2020_baseline.csv` | CSV (text) | Frozen drift reference | No — never modified |

**Why Parquet for the pipeline?** Parquet is a compressed, columnar format — roughly 10× smaller than the equivalent CSV and significantly faster for pandas to load. All scripts (`modeling_pipeline.py`, `export_simulation_to_parquet.py`) read and write `ai4i2020.parquet`. DVC tracks this file.

**Why keep the CSV?** The original data from the UCI repository arrived as a CSV. It is kept as-is so you can open it in Excel or a text editor and inspect the raw values without needing any special tooling. It is not tracked by DVC and is not read by any script — it is reference material only.

---

## Prerequisites

- Python 3.11
- Git and DVC (`pip install dvc`)
- A [DagsHub](https://dagshub.com) account — used for DVC remote storage and MLflow tracking
- GitHub Secrets configured for the retrain workflow (see [GitHub Actions setup](#github-actions-setup))

---

## First-time setup

```bash
# 1. Clone and install
git clone https://github.com/Preempt-analytics/predictive-maintenance-capstone
cd predictive-maintenance-capstone
pip install -r requirements.txt

# 2. Configure DVC to reach DagsHub (stored locally, never committed)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_DAGSHUB_USERNAME
dvc remote modify origin --local password YOUR_DAGSHUB_TOKEN

# 3. Pull training data and frozen baseline
dvc pull data/ai4i2020.parquet data/ai4i2020_baseline.csv

# 4. Set MLflow tracking credentials (add to your .env or shell profile)
export MLFLOW_TRACKING_URI=https://dagshub.com/YOUR_USERNAME/predictive-maintenance-capstone.mlflow
export MLFLOW_TRACKING_USERNAME=YOUR_DAGSHUB_USERNAME
export MLFLOW_TRACKING_PASSWORD=YOUR_DAGSHUB_TOKEN

# 5. Create the simulation database file (required before running Docker)
# docker-compose.yml bind-mounts ./simulation.db into every container. If the
# file does not exist on the host before `docker compose up`, Docker creates a
# directory with that name instead — which breaks SQLite. One touch fixes this.
touch simulation.db
```

---

## Running the project

### Step 1 — Start the API (Terminal 1, keep running)

```bash
uvicorn src.api:app --reload
```

The API loads the `@production` model from MLflow at startup. Verify it is ready:

```bash
curl http://localhost:8000/health
```

Expected response: `{"status": "ok", "model_loaded": true, ...}`

### Step 2 — Run the simulator (Terminal 2)

**Basic run — generate readings only:**

```bash
python src/sensor_simulator.py --n-readings 1000 --mode normal
```

**Full automated pipeline — simulate, detect drift, export, and trigger retraining:**

```bash
python src/sensor_simulator.py --n-readings 1000 --mode normal --detect-drift --export-on-drift
```

What `--detect-drift --export-on-drift` does automatically after the simulation finishes:

1. Compares the new readings against `data/ai4i2020_baseline.csv` using Evidently AI
2. **Drift detected** → exports data as Parquet, pushes to DagsHub, updates `retrain.trigger` → GitHub Actions fires the retrain workflow
3. **No drift** → exports data to DagsHub for accumulation → no retraining triggered

**Simulation modes:**

| Flag | Failure rate | When to use |
|------|-------------|-------------|
| `--mode normal` | Stable 3.4% | Routine data collection; meaningful for drift detection |
| `--mode gradual-drift` | 3.4% → 25% | Simulates equipment ageing over time |
| `--mode sudden-spike` | 3.4% then 40% | Stress-testing the retrain pipeline; demos |

> All modes support `--detect-drift` and `--export-on-drift`. Non-normal modes will note that drift is expected and continue — this is intentional for testing the full retrain loop.

**Other useful flags:**

```bash
# Reset the database before a clean run
python src/sensor_simulator.py --reset --n-readings 1000 --mode normal

# Slow it down for a live demo
python src/sensor_simulator.py --n-readings 200 --mode normal --interval 1.0
```

---

## Running individual steps manually

### Drift detection

```bash
# Compare simulation.db against the frozen baseline
python scripts/detect_drift.py

# Check only recent readings (since a specific timestamp)
python scripts/detect_drift.py --since "2026-05-29T00:00:00"
```

The HTML report is saved to `reports/drift_report.html` — open it in a browser for per-feature histograms.

### Export simulation data

```bash
# Preview — show counts and column layout, write nothing
python scripts/export_simulation_to_parquet.py --dry-run

# Export and push to DagsHub (data accumulation only, no retrain)
python scripts/export_simulation_to_parquet.py --push

# Export, push, and trigger the GitHub Actions retrain workflow
python scripts/export_simulation_to_parquet.py --push --retrain
```

### Reload the API after model promotion

After GitHub Actions promotes a new model, the running API still serves the old version. Reload it:

```bash
curl -X POST http://localhost:8000/model/reload
# or restart the server: uvicorn src.api:app --reload
```

### Check or run promotion gates manually

```bash
# Dry run — shows what WOULD happen without moving the alias
python scripts/promote_model.py --model-name predictive-maintenance-binary
python scripts/promote_model.py --model-name predictive-maintenance-multiclass

# Auto-promote if gates pass (this is what GitHub Actions runs)
python scripts/promote_model.py --model-name predictive-maintenance-binary --auto --min-f1 0.85
python scripts/promote_model.py --model-name predictive-maintenance-multiclass --auto --min-f1 0.60
```

---

## GitHub Actions setup

The retrain workflow (`.github/workflows/retrain.yml`) runs on GitHub-hosted Ubuntu runners. It requires five repository secrets:

| Secret | Value |
|--------|-------|
| `DAGSHUB_USERNAME` | Your DagsHub username |
| `DAGSHUB_TOKEN` | Your DagsHub access token |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/USERNAME/REPO.mlflow` |
| `API_URL` | *(optional)* Base URL of a deployed API — enables automatic model reload |

Add them at: **GitHub repo → Settings → Secrets and variables → Actions → New repository secret**

**What the workflow does (in order):**

1. Checks out the repo and installs dependencies
2. Pulls training data from DagsHub via DVC
3. Retrains only the two XGBoost models (`train_xgb_binary`, `train_xgb_multiclass`)
4. Runs `promote_model.py` — promotes the new version only if it beats `@production` AND clears the minimum F1 floor
5. Reloads the serving API at `API_URL` if a model was promoted (skipped if `API_URL` is not set)

**What triggers it:**
The workflow watches `retrain.trigger`, not `data/ai4i2020.parquet.dvc`. Only a push that updates `retrain.trigger` (i.e., drift was detected) fires the workflow. Data-accumulation pushes without drift leave `retrain.trigger` unchanged — no workflow runs.

**Manual trigger:** Actions tab → "Retrain on new data" → "Run workflow" → select `main`

---

## Using DagsHub as a single source of truth

| What | Where in DagsHub |
|------|-----------------|
| MLflow experiments and runs | `dagshub.com/USERNAME/REPO` → Experiments tab |
| Model registry and `@production` alias | Experiments tab → Models |
| DVC-tracked dataset versions | Files tab → `data/ai4i2020.parquet` → History |
| GitHub Actions CI results | Connect repo via DagsHub repo Settings → Integrations |

To surface GitHub Actions results in DagsHub: go to your DagsHub repo → **Settings → Integrations → GitHub Actions**. Once connected, each workflow run appears alongside the corresponding MLflow experiment.

---

## Team

| Name | GitHub |
|------|--------|
| Nate | @x |
| Ivo  | @y |

neuefische AI Engineering Bootcamp · Cohort 2026
