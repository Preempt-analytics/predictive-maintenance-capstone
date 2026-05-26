import os
from dataclasses import dataclass
from pathlib import Path

import click
import mlflow
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

load_dotenv()

DATA_PATH = Path("data/ai4i2020.csv")
WEAR_LIMITS = {"L": 200, "M": 220, "H": 240}

COLUMN_RENAME = {
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

FEATURES = [
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "temp_diff",
    "power",
    "overstrain",
    "rpm_torque_ratio",
    "wear_pct",
    "in_safe_power_band",
    "hdf_risk",
]


# ── Experiment registry ────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    experiment_name: str        # groups runs in the MLflow UI
    registered_model_name: str  # versioned entry in the model registry
    target: str                 # column name of what we are predicting
    target_type: str            # "binary" or "multiclass" — drives metric selection
    model_family: str           # human-readable label logged as a tag


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "xgb_binary": ExperimentConfig(
        experiment_name="predictive-maintenance/xgboost/binary",
        registered_model_name="predictive-maintenance-xgb-binary",
        target="machine_failure",
        target_type="binary",
        model_family="xgboost",
    ),
    "xgb_multiclass": ExperimentConfig(
        # Each failure type (TWF, HDF, PWF, OSF, RNF) becomes its own class.
        # Swap in a multiclass objective and per-class metrics when you implement this.
        experiment_name="predictive-maintenance/xgboost/multiclass",
        registered_model_name="predictive-maintenance-xgb-multiclass",
        target="failure_type",   # derived column — see preprocess() note below
        target_type="multiclass",
        model_family="xgboost",
    ),
}


# ── Data ───────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise click.ClickException(
            f"Missing {DATA_PATH}. Place the AI4I 2020 dataset at that path before training."
        )
    return pd.read_csv(DATA_PATH)


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    df = df.copy().rename(columns=COLUMN_RENAME)
    df["temp_diff"] = df["process_temperature_k"] - df["air_temperature_k"]
    df["power"] = df["torque_nm"] * df["rotational_speed_rpm"]
    df["overstrain"] = df["torque_nm"] * df["tool_wear_min"]
    df["rpm_torque_ratio"] = df["rotational_speed_rpm"] / df["torque_nm"]
    df["wear_pct"] = df.apply(
        lambda r: r["tool_wear_min"] / WEAR_LIMITS[r["type"]], axis=1
    )
    df["in_safe_power_band"] = df["power"].between(3500, 9000).astype(int)
    df["hdf_risk"] = (
        (df["temp_diff"] < 8.6) & (df["rotational_speed_rpm"] < 1380)
    ).astype(int)

    # For multiclass: collapse the five failure-type columns into one label column.
    # "none" when no failure type is flagged; the specific type otherwise.
    if config.target_type == "multiclass":
        failure_cols = ["twf", "hdf", "pwf", "osf", "rnf"]
        def resolve_label(row):
            active = [c for c in failure_cols if row[c] == 1]
            return active[0] if active else "none"
        df["failure_type"] = df.apply(resolve_label, axis=1)

    return df[FEATURES + [config.target]]


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, config: ExperimentConfig):
    """Return (pipeline, metrics, params)."""
    df_processed = preprocess(df, config)
    y = df_processed[config.target]
    X = df_processed.drop(columns=[config.target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2, stratify=y
    )

    X_train_records = X_train.to_dict(orient="records")
    X_test_records = X_test.to_dict(orient="records")

    if config.target_type == "binary":
        # Compensate for ~28:1 class imbalance without resampling
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        classifier = xgb.XGBClassifier(
            n_estimators=200,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
    else:
        # multiclass: XGBoost infers num_class from the labels
        scale_pos_weight = None
        classifier = xgb.XGBClassifier(
            n_estimators=200,
            objective="multi:softprob",
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        )

    pipeline = make_pipeline(DictVectorizer(), classifier)
    pipeline.fit(X_train_records, y_train)

    y_pred_train = pipeline.predict(X_train_records)
    y_pred_test = pipeline.predict(X_test_records)

    average = "binary" if config.target_type == "binary" else "macro"
    metrics: dict[str, float] = {
        "f1_train": f1_score(y_train, y_pred_train, average=average),
        "f1_test": f1_score(y_test, y_pred_test, average=average),
        "precision_test": precision_score(y_test, y_pred_test, average=average),
        "recall_test": recall_score(y_test, y_pred_test, average=average),
    }
    if config.target_type == "binary":
        y_prob_test = pipeline.predict_proba(X_test_records)[:, 1]
        metrics["roc_auc_test"] = roc_auc_score(y_test, y_prob_test)

    params: dict[str, object] = {
        "n_estimators": 200,
        "target_type": config.target_type,
    }
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = round(scale_pos_weight, 2)

    return pipeline, metrics, params


# ── MLflow ─────────────────────────────────────────────────────────────────────

def configure_mlflow(config: ExperimentConfig) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)


def log_model(
    pipeline, metrics: dict, params: dict, config: ExperimentConfig
) -> None:
    with mlflow.start_run():
        mlflow.set_tags(
            {
                "model_family": config.model_family,
                "target_type": config.target_type,
                "target": config.target,
                "developer": os.getenv("MLFLOW_TRACKING_USERNAME", "unknown"),
                "dataset": "AI4I 2020 predictive maintenance",
                "features": ",".join(FEATURES),
            }
        )
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=config.registered_model_name,
        )


# ── CML report ─────────────────────────────────────────────────────────────────

def write_cml_metrics(metrics: dict) -> None:
    lines = ["# Training Metrics", ""]
    for name, value in metrics.items():
        lines.append(f"- {name}: {value:.4f}")
    lines.append("")
    Path("metrics.txt").write_text("\n".join(lines), encoding="utf-8")


# ── Entry point ────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--experiment",
    default="xgb_binary",
    type=click.Choice(list(EXPERIMENTS)),
    show_default=True,
    help="Which experiment config to run.",
)
@click.option(
    "--cml-run/--no-cml-run",
    default=False,
    help="Write metrics.txt for a CML pull request report.",
)
def main(experiment: str, cml_run: bool) -> None:
    """Train a predictive maintenance failure classifier."""
    config = EXPERIMENTS[experiment]

    df = load_data()
    configure_mlflow(config)

    pipeline, metrics, params = train_model(df, config)
    log_model(pipeline, metrics, params, config)

    if cml_run:
        write_cml_metrics(metrics)


if __name__ == "__main__":
    main()
