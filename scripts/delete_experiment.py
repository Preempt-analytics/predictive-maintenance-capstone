"""
MLflow Experiment Cleanup Utility
==================================
Permanently deletes a soft-deleted MLflow experiment by name so the name
can be reused. Reads tracking credentials from the project .env file.

Usage:
    python delete_experiment.py <experiment_name>

Example:
    python delete_experiment.py predictive-maintenance/logreg/multiclass

Note: MLflow soft-deletes experiments by default. This script permanently
removes them, which is required before re-creating an experiment with the
same name.
"""

import os
import sys
from dotenv import load_dotenv
import mlflow

if len(sys.argv) != 2:
    print("Usage: python delete_experiment.py <experiment_name>")
    print("Example: python delete_experiment.py predictive-maintenance/logreg/multiclass")
    sys.exit(1)

experiment_name = sys.argv[1]

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
if experiment:
    client.restore_experiment(experiment.experiment_id)
    client.delete_experiment(experiment.experiment_id)
    print(f"Permanently deleted: {experiment_name}")
else:
    print(f"Experiment not found: {experiment_name}")
