import os
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
TARGET = "Machine failure"
MODEL_NAME = "predictive-maintenance-xgb"

# Wear limits per machine type, used to compute wear_pct
WEAR_LIMITS = {"L": 200, "M": 220, "H": 240}

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_diff",
    "power",
    "overstrain",
    "rpm_torque_ratio",
    "wear_pct",
    "in_safe_power_band",
    "hdf_risk",
]


def load_data():
    if not DATA_PATH.exists():
        raise click.ClickException(
            f"Missing {DATA_PATH}. Place the AI4I 2020 dataset at that path before training."
        )
    return pd.read_csv(DATA_PATH)


def preprocess(df):
    df = df.copy()
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    df["overstrain"] = df["Torque [Nm]"] * df["Tool wear [min]"]
    df["rpm_torque_ratio"] = df["Rotational speed [rpm]"] / df["Torque [Nm]"]
    df["wear_pct"] = df.apply(
        lambda r: r["Tool wear [min]"] / WEAR_LIMITS[r["Type"]], axis=1
    )
    # Binary flags derived from EDA-identified failure thresholds
    df["in_safe_power_band"] = df["power"].between(3500, 9000).astype(int)
    df["hdf_risk"] = (
        (df["temp_diff"] < 8.6) & (df["Rotational speed [rpm]"] < 1380)
    ).astype(int)
    return df[FEATURES + [TARGET]]


def train_model(df):
    df_processed = preprocess(df)
    y = df_processed[TARGET]
    X = df_processed.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2, stratify=y
    )

    # Compensate for class imbalance (~28:1 ratio) without resampling
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    X_train_records = X_train.to_dict(orient="records")
    X_test_records = X_test.to_dict(orient="records")

    pipeline = make_pipeline(
        DictVectorizer(),
        xgb.XGBClassifier(
            n_estimators=200,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
    )
    pipeline.fit(X_train_records, y_train)

    y_pred_train = pipeline.predict(X_train_records)
    y_pred_test = pipeline.predict(X_test_records)
    y_prob_test = pipeline.predict_proba(X_test_records)[:, 1]

    metrics = {
        "f1_train": f1_score(y_train, y_pred_train),
        "f1_test": f1_score(y_test, y_pred_test),
        "precision_test": precision_score(y_test, y_pred_test),
        "recall_test": recall_score(y_test, y_pred_test),
        "roc_auc_test": roc_auc_score(y_test, y_prob_test),
    }
    params = {
        "n_estimators": 200,
        "scale_pos_weight": round(scale_pos_weight, 2),
    }
    return pipeline, metrics, params


def configure_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("predictive-maintenance-xgb")


def log_model(pipeline, metrics, params):
    with mlflow.start_run():
        mlflow.set_tags(
            {
                "model": "xgboost classifier",
                "developer": os.getenv("MLFLOW_TRACKING_USERNAME", "unknown"),
                "dataset": "AI4I 2020 predictive maintenance",
                "features": ",".join(FEATURES),
                "target": TARGET,
            }
        )
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=MODEL_NAME,
        )


def write_cml_metrics(metrics):
    Path("metrics.txt").write_text(
        "\n".join(
            [
                "# Training Metrics",
                "",
                f"- F1 on the train set: {metrics['f1_train']:.4f}",
                f"- F1 on the test set: {metrics['f1_test']:.4f}",
                f"- Precision on the test set: {metrics['precision_test']:.4f}",
                f"- Recall on the test set: {metrics['recall_test']:.4f}",
                f"- ROC-AUC on the test set: {metrics['roc_auc_test']:.4f}",
                "",
            ]
        ),
        encoding="utf-8",
    )


@click.command()
@click.option(
    "--cml-run/--no-cml-run",
    default=False,
    help="Write metrics.txt for a CML pull request report.",
)
def main(cml_run):
    """Train a predictive maintenance failure classifier."""
    df = load_data()
    configure_mlflow()

    pipeline, metrics, params = train_model(df)
    log_model(pipeline, metrics, params)

    if cml_run:
        write_cml_metrics(metrics)


if __name__ == "__main__":
    main()
