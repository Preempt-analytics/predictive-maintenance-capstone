"""
Predictive Maintenance — Modeling Pipeline
==========================================
Trains a failure classifier on the AI4I 2020 dataset and logs results to MLflow.

Supported experiments (pass via --experiment):
    xgb_binary,   xgb_multiclass
    rf_binary,    rf_multiclass
    logreg_binary, logreg_multiclass
    lgbm_binary,  lgbm_multiclass

Usage:
    python modeling_pipeline.py --experiment xgb_binary
    python modeling_pipeline.py --experiment xgb_binary --cml-run
    python modeling_pipeline.py --experiment rf_binary
    ...
    Additional experiments can be added to the EXPERIMENTS registry with a new config
    and run via the script .
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import click
import mlflow
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

load_dotenv()

DATA_PATH = Path("data/ai4i2020.csv")

COLUMN_RENAME = {
    "Type": "type",
    "Air temperature [K]": "air_temperature_kelvin",
    "Process temperature [K]": "process_temperature_kelvin",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_minutes",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

FEATURES = [
    "type",
    "air_temperature_kelvin",
    "process_temperature_kelvin",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_minutes",
    "power_kw",
    "temp_diff_kelvin",
    "mechanical_stress",
]


# ── Experiment registry ────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """All settings needed to run, log, and reproduce one experiment.

    The classifier_factory pattern keeps train_model() free of if/else branching:
    each config owns its classifier definition. All variation lives here, not
    scattered across functions.

    Args:
        experiment_name:       MLflow experiment path shown in the UI.
                               Convention: "project/model-family/target-type".
                               Created automatically if it does not yet exist.
        registered_model_name: Versioned entry in the MLflow model registry.
                               Enables staging → production lifecycle management.
        model_family:          Human-readable label logged as an MLflow tag
                               (e.g. "xgboost", "random_forest", "logreg").
        target:                DataFrame column to predict.
                               "machine_failure" (binary) or "failure_type" (multiclass).
        target_type:           "binary" or "multiclass".
                               Controls label engineering, metric selection, and ROC-AUC.
        metric_average:        Averaging strategy passed to sklearn scoring functions.
                               "binary" for two-class targets, "macro" for multiclass.
        classifier_factory:    Callable(imbalance_ratio: float) → unfitted classifier.
                               Owns all hyperparameters for this experiment.
                               Multiclass factories ignore imbalance_ratio (use lambda _).
        test_size:             Fraction of data held out for evaluation. Default 0.2.
        description:           Optional free-text summary for documentation.
        notes:                 Optional scratchpad — not logged to MLflow.
        tags:                  Extra key/value pairs merged into MLflow run tags.
    """
    experiment_name: str
    registered_model_name: str
    model_family: str
    target: str
    target_type: str       # "binary" or "multiclass"
    metric_average: str    # "binary" or "macro"
    classifier_factory: Callable
    test_size: float = 0.2
    description: str = ""
    notes: Optional[str] = None
    tags: dict = field(default_factory=dict)


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "xgb_binary": ExperimentConfig(
        experiment_name="predictive-maintenance/xgboost/binary",
        registered_model_name="xgboost-binary",
        model_family="xgboost",
        target="machine_failure",
        target_type="binary",
        metric_average="binary",
        # scale_pos_weight compensates for ~97:3 class imbalance without resampling
        classifier_factory=lambda r: xgb.XGBClassifier(
            n_estimators=200, scale_pos_weight=r,
            random_state=42, n_jobs=-1, eval_metric="logloss",
        ),
    ),

    "xgb_multiclass": ExperimentConfig(
        experiment_name="predictive-maintenance/xgboost/multiclass",
        registered_model_name="xgboost-multiclass",
        model_family="xgboost",
        target="failure_type",
        target_type="multiclass",
        metric_average="macro",
        classifier_factory=lambda _: xgb.XGBClassifier(
            n_estimators=200, objective="multi:softprob",
            random_state=42, n_jobs=-1, eval_metric="mlogloss",
        ),
    ),

    "lgbm_binary": ExperimentConfig(
        experiment_name="predictive-maintenance/lightgbm/binary",
        registered_model_name="lightgbm-binary",
        model_family="lightgbm",
        target="machine_failure",
        target_type="binary",
        metric_average="binary",
        classifier_factory=lambda r: lgb.LGBMClassifier(
            n_estimators=200,
            scale_pos_weight=r,
            random_state=42,
            n_jobs=-1,
        ),
    ),

    "lgbm_multiclass": ExperimentConfig(
        experiment_name="predictive-maintenance/lightgbm/multiclass",
        registered_model_name="lightgbm-multiclass",
        model_family="lightgbm",
        target="failure_type",
        target_type="multiclass",
        metric_average="macro",
        classifier_factory=lambda _: lgb.LGBMClassifier(
            n_estimators=200,
            objective="multiclass",
            random_state=42,
            n_jobs=-1,
        ),
    ),
    "rf_binary": ExperimentConfig(
        experiment_name="predictive-maintenance/random-forest/binary",
        registered_model_name="random-forest-binary",
        model_family="random_forest",
        target="machine_failure",
        target_type="binary",
        metric_average="binary",
        classifier_factory=lambda _: RandomForestClassifier(
            class_weight="balanced", n_estimators=100, random_state=42, n_jobs=-1,
        ),
    ),
    "rf_multiclass": ExperimentConfig(
        experiment_name="predictive-maintenance/random-forest/multiclass",
        registered_model_name="random-forest-multiclass",
        model_family="random_forest",
        target="failure_type",
        target_type="multiclass",
        metric_average="macro",
        classifier_factory=lambda _: RandomForestClassifier(
            class_weight="balanced", n_estimators=100, random_state=42, n_jobs=-1,
        ),
    ),
    "logreg_binary": ExperimentConfig(
        experiment_name="predictive-maintenance/logreg/binary",
        registered_model_name="logreg-binary",
        model_family="logreg",
        target="machine_failure",
        target_type="binary",
        metric_average="binary",
        classifier_factory=lambda _: LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42,
        ),
    ),
    "logreg_multiclass": ExperimentConfig(
        experiment_name="predictive-maintenance/logistic-regression/multiclass",
        registered_model_name="logreg-multiclass",
        model_family="logreg",
        target="failure_type",
        target_type="multiclass",
        metric_average="macro",
        classifier_factory=lambda _: LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42,
        ),
    ),
}


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Rename columns, engineer features, and slice to model inputs + target.

    Domain features added here (each justified by EDA):
    - power_kw:          torque × rpm → kW. Failures cluster at power extremes.
    - temp_diff_kelvin:  process − air temperature. HDF risk rises when diff < 8.6 K.
    - mechanical_stress: torque × tool wear. High torque on a worn tool is a compound hazard.

    For multiclass experiments the five binary failure columns (twf, hdf, pwf, osf, rnf)
    are collapsed into a single string label via resolve_label(); rows with no active
    flag become "none". Where multiple flags are set simultaneously, the first match
    wins (TWF > HDF > PWF > OSF > RNF) — an acceptable simplification for this dataset.

    DictVectorizer in the downstream pipeline handles one-hot encoding of 'type'
    automatically — no manual encoding needed here.

    Args:
        df:     Raw DataFrame loaded directly from ai4i2020.csv (original column names).
        config: Active experiment config. Determines the target column and whether
                the multiclass label column is built.

    Returns:
        DataFrame with columns FEATURES + [config.target]. All rows preserved.
    """
    df = df.copy().rename(columns=COLUMN_RENAME)

    df["power_kw"] = (df["torque_nm"] * df["rotational_speed_rpm"] * 2 * 3.14159 / 60) / 1000
    df["temp_diff_kelvin"] = df["process_temperature_kelvin"] - df["air_temperature_kelvin"]
    df["mechanical_stress"] = df["torque_nm"] * df["tool_wear_minutes"]

    if config.target_type == "multiclass":
        failure_cols = ["twf", "hdf", "pwf", "osf", "rnf"]
        def resolve_label(row):
            active = [c for c in failure_cols if row[c] == 1]
            return active[0] if active else "none"
        df["failure_type"] = df.apply(resolve_label, axis=1)

    return df[FEATURES + [config.target]]


# ── Classifier builder ─────────────────────────────────────────────────────────

def _build_classifier(config: ExperimentConfig, imbalance_ratio: float):
    """Instantiate the classifier defined in config.classifier_factory.

    Keeping this as a thin wrapper means train_model() stays free of any
    classifier-specific logic — all hyperparameter decisions live in EXPERIMENTS.

    Args:
        config:           Active experiment config.
        imbalance_ratio:  Ratio of negative to positive training samples (~28 for
                          this dataset). Passed to the factory lambda; multiclass
                          factories ignore it (declared as lambda _).

    Returns:
        Unfitted sklearn-compatible classifier instance.
    """
    return config.classifier_factory(imbalance_ratio)


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, config: ExperimentConfig):
    """Preprocess, split, train, and evaluate. Return (pipeline, metrics, params).

    Stratified split preserves the ~97:3 minority-class ratio in both folds —
    without stratify=y, the test set could end up with almost no failure cases,
    making evaluation metrics unreliable.

    ROC-AUC requires a single scalar probability score and is only computed for
    binary targets. Multiclass ROC-AUC would need one-vs-rest decomposition and
    is excluded here to keep cross-experiment comparison clean.

    f1_train is logged alongside f1_test to surface overfitting at a glance:
    a large train/test gap signals the model memorised training data.

    Args:
        df:     Raw DataFrame from ai4i2020.csv.
        config: Active experiment config. Drives split size, target column,
                classifier selection, and metric averaging strategy.

    Returns:
        pipeline (sklearn.Pipeline):
            Fitted DictVectorizer + classifier. Ready for mlflow.sklearn.log_model().
        metrics (dict[str, float]):
            f1_train, f1_test, precision_test, recall_test — always present.
            roc_auc_test — binary experiments only.
        params (dict[str, object]):
            Full classifier hyperparameters from get_params(), plus model_family,
            target_type, and test_size. Sufficient to reproduce this exact run.
    """
    df_processed = preprocess(df, config)
    y = df_processed[config.target]
    X = df_processed.drop(columns=[config.target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=config.test_size, stratify=y
    )

    # Guard against ZeroDivisionError: multiclass y_train holds string labels,
    # so integer comparisons would produce 0/0.
    if config.target_type == "binary":
        imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    else:
        imbalance_ratio = 1.0  # ignored by multiclass factories (lambda _)

    classifier = _build_classifier(config, imbalance_ratio)

    X_train_records = X_train.to_dict(orient="records")
    X_test_records = X_test.to_dict(orient="records")

    pipeline = make_pipeline(DictVectorizer(), classifier)
    pipeline.fit(X_train_records, y_train)

    y_pred_train = pipeline.predict(X_train_records)
    y_pred_test = pipeline.predict(X_test_records)

    y_prob_test = (
        pipeline.predict_proba(X_test_records)[:, 1]
        if config.target_type == "binary"
        else None
    )

    average = config.metric_average
    metrics: dict[str, float] = {
        "f1_train":       f1_score(y_train, y_pred_train, average=average),
        "f1_test":        f1_score(y_test, y_pred_test, average=average),
        "precision_test": precision_score(y_test, y_pred_test, average=average),
        "recall_test":    recall_score(y_test, y_pred_test, average=average),
    }
    if y_prob_test is not None:
        metrics["roc_auc_test"] = roc_auc_score(y_test, y_prob_test)

    params: dict[str, object] = {
        **classifier.get_params(),
        "model_family": config.model_family,
        "target_type": config.target_type,
        "test_size": config.test_size,
    }

    return pipeline, metrics, params


# ── MLflow ─────────────────────────────────────────────────────────────────────

def configure_mlflow(config: ExperimentConfig) -> None:
    """Point MLflow at the remote tracking server and activate the experiment.

    Reads MLFLOW_TRACKING_URI from the environment (loaded from .env).
    Credentials (username + password) are picked up automatically by MLflow
    from MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD in the environment.
    Creates the experiment on the server if it does not yet exist.

    Args:
        config: Active experiment config supplying the experiment name.

    Raises:
        EnvironmentError: If MLFLOW_TRACKING_URI is not set in the environment.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise EnvironmentError("MLFLOW_TRACKING_URI is not set in the environment.")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)


def log_model(
    pipeline, metrics: dict, params: dict, config: ExperimentConfig
) -> None:
    """Open a new MLflow run and log tags, params, metrics, and the model artifact.

    MLflow distinguishes three metadata types — keep them separate:
      tags    → who/what/why (free-form labels, not compared across runs)
      params  → inputs chosen before training (hyperparameters, split size)
      metrics → outputs produced by training (scores, counts)

    The model is both stored as a run artifact and registered under
    config.registered_model_name, enabling versioned lifecycle management
    (staging → production) in the MLflow registry.

    Args:
        pipeline: Fitted sklearn Pipeline produced by train_model().
        metrics:  Evaluation scores from train_model() — logged as MLflow metrics.
        params:   Hyperparameters from train_model() — logged as MLflow params.
        config:   Active experiment config. Supplies tags and the registered model name.
    """
    with mlflow.start_run():
        mlflow.set_tags({
            "model_family": config.model_family,
            "target_type": config.target_type,
            "target": config.target,
            "experiment_name": config.experiment_name,
            "developer": os.getenv("MLFLOW_TRACKING_USERNAME", "unknown"),
        })
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=config.registered_model_name,
        )


# ── CML report ─────────────────────────────────────────────────────────────────

def write_cml_metrics(metrics: dict) -> None:
    """Write key test metrics to metrics.txt for a CML pull-request comment.

    The CML GitHub Action picks up this file and attaches it as a comment on
    the pull request, letting reviewers see model performance without opening
    the MLflow UI. Includes test-set metrics only — f1_train is omitted since
    reviewers need generalisation performance, not evidence of fitting.

    Args:
        metrics: Dict produced by train_model(). Keys f1_test, precision_test,
                 and recall_test are always present. roc_auc_test is optional
                 (binary experiments only) and included when available.
    """
    lines = [
        "# Training Metrics",
        "",
        f"f1_test:        {metrics['f1_test']:.4f}",
        f"precision_test: {metrics['precision_test']:.4f}",
        f"recall_test:    {metrics['recall_test']:.4f}",
    ]
    if "roc_auc_test" in metrics:
        lines.append(f"roc_auc_test:   {metrics['roc_auc_test']:.4f}")
    Path("metrics.txt").write_text("\n".join(lines), encoding="utf-8")


# ── Entry point ────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--experiment",
    default="xgb_binary",
    type=click.Choice(list(EXPERIMENTS)),
    show_default=True,
    help="Which experiment config to use.",
)
@click.option(
    "--cml-run/--no-cml-run",
    default=False,
    help="Write metrics.txt for a CML pull request report.",
)
def main(experiment: str, cml_run: bool) -> None:
    """Train a predictive maintenance failure classifier."""
    config = EXPERIMENTS[experiment]

    df = pd.read_csv(DATA_PATH)
    configure_mlflow(config)

    pipeline, metrics, params = train_model(df, config)
    log_model(pipeline, metrics, params, config)

    if cml_run:
        write_cml_metrics(metrics)


if __name__ == "__main__":
    main()
