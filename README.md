# Predictive-Maintenance-Capstone

Capstone project 2 of the AI Engineering bootcamp at neuefische about predictive maintenance with a focus on MLOps by Nate and Ivo

## What this project does

This project predicts equipment failure and failure mode from real-time sensor readings — before the machine stops working.

We train a classification model on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (UCI), then wrap it in a production-grade MLOps pipeline. The model is as much a deliverable as the pipeline that runs it.

**Research question:**  
_To what extent can a machine learning model predict equipment failure and failure mode from real-time sensor readings, and what kind of MLOps pipeline would be optimal to support that?_

---

## Why it matters

Unplanned equipment failure in manufacturing costs thousands of euros per hour in downtime. Most factories either fix machines after they break or replace parts on a fixed schedule regardless of actual condition — both approaches waste time and money.

A model that reads live sensor data and flags imminent failure changes that. Maintenance becomes intelligently planned, not reactive.

---

## What the model predicts

The AI4I dataset simulates a CNC-type industrial machine with five distinct failure modes:

| Code | Failure                  | Trigger                                     |
| ---- | ------------------------ | ------------------------------------------- |
| TWF  | Tool Wear Failure        | Tool exceeds wear threshold                 |
| HDF  | Heat Dissipation Failure | Temperature differential too low at low RPM |
| PWF  | Power Failure            | Torque × speed outside safe power band      |
| OSF  | Overstrain Failure       | Tool wear × torque exceeds strain limit     |
| RNF  | Random Failure           | Unpredictable — ~0.1% of cases              |

---

## The MLOps pipeline

The pipeline is the core deliverable of this project — not just the model accuracy.

Data (DVC) → Experiments (MLflow) → Serving (FastAPI + Docker) → CI/CD (GitHub Actions) → Monitoring (Prometheus + Grafana)

| Tool                 | Role                                   |
| -------------------- | -------------------------------------- |
| MLflow               | Experiment tracking and model registry |
| FastAPI              | REST endpoint for live predictions     |
| Pydantic             | Input validation on the API layer      |
| Docker               | Containerise the full pipeline         |
| Pytest               | Test data pipeline and API             |
| GitHub Actions       | Automated test and build on push       |
| DVC                  | Dataset and artifact versioning        |
| Prometheus + Grafana | Serving metrics and dashboard          |

---

## How to run it

```bash
# Clone the repo
git clone https://github.com/yourteam/predictive-maintenance-capstone
cd predictive-maintenance-capstone

# Pull data
dvc pull

# Build and start the pipeline
docker compose up --build

# API available at
http://localhost:8000

# MLflow UI available at
http://localhost:5000
Project structure

├── data/               # Raw and processed data (DVC-tracked)
├── notebooks/          # Exploratory analysis
├── src/
│   ├── train.py        # Model training + MLflow logging
│   ├── predict.py      # Inference logic
│   └── api.py          # FastAPI app
├── tests/              # Pytest test suite
├── docker-compose.yml
├── .github/workflows/  # GitHub Actions CI/CD
└── README.md

Team    Preempt Analytics
Name:	GitHub:

Nate	@x
Ivo 	@y
neuefische AI Engineering Bootcamp · Cohort 2026



---
```
