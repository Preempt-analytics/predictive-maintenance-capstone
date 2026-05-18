"""
Predictive Maintenance — Training Template
==========================================
Fill in every section marked with a question or TODO.
The structure guides you toward a setup that scales across
multiple experiment types without copy-pasting.

Run with:
    python train_template.py --experiment xgb_binary
    python train_template.py --experiment xgb_binary --cml-run
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import click
from git import Optional
import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

load_dotenv()

DATA_PATH = Path("data/ai4i2020.csv")

# Import dataset using the relative path above. 
# Adjust this part if the data intake changes (e.g. if you switch to a database or an API instead of a CSV file).
df = pd.read_csv(DATA_PATH)

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


# ── 1. Experiment registry ─────────────────────────────────────────────────────
#
# Q: If you run XGBoost today and a Random Forest next week, what changes between
#    the two runs? Make a list. Now ask: which of those things belong in code,
#    and which belong in a config object that you can swap out?
#     
#    - Model type, - belongs to config object (e.g. "xgboost" vs "random_forest") 
#    - Hyperparameters, -  config object 
#    - Feature list,       
#    - Target type (binary vs multiclass),

# Q: MLflow groups runs under "experiments". If you have binary classification
#    runs and multiclass runs in the same experiment, what problem does that
#    create in the UI? What naming scheme would keep things navigable?
#    One convention: "{project}/{model_family}/{target_type}"
#    e.g. "predictive-maintenance/xgboost/binary"
#    e.g. "predictive-maintenance/GBoost/multiclass"
#
# Q: What is an experiment vs what is a run in MLflow? Which one do you want to link to a registered model in the registry, and why?
#
# Experiment: A blog post 
# Run: A heading in the blog post that breaks the article into sections.
#
# Registered model: Versioned model that can be linked to a run. You want to link a registered model to a run because it 
# allows you to track the performance and parameters of the model that was registered, making it easier to reproduce 
# and compare different versions of the model.
#
# Q: What is the difference between an experiment name and a registered model
#    name in MLflow? Do they have to match?
# 
# Experiment name: A label for grouping related runs in MLflow. It is used to organize 
# and categorize runs based on the experiment they belong to.
#
# Registered model on the hand is a model that is registered with MLflow complete with hyperparameters, eval. metrics and other config settings. 
# This enables greater reproducibility since variables remain the same. 
#
# Q: What fields does a config object need so that configure_mlflow() and
#    log_model() below require zero hardcoded strings?

@dataclass
class ExperimentConfig:
    experiment_name: str       # shown in the MLflow UI — think about namespacing
    registered_model_name: str # name under which the model is versioned in the registry
    model_family: str # name of the type of model
    target_type: str # binary vs multiclass
    description: str = ""                    # optional, defaults to empty string
    notes: Optional[str] = None              # optional, explicitly nullable
    tags: dict = field(default_factory=dict) # optional, mutable default


    # TODO: what other fields would make this config self-contained?
    #       Think about: model family label, target type, description...


# Q: How would you add a second experiment (e.g. multiclass or a different model
#    family) without changing any of the functions below?
#    What is the minimum edit required?

# TBD — adding external yaml config file similar to how .env works?

EXPERIMENTS: dict[str, ExperimentConfig] = {
    "xgb_binary": ExperimentConfig(
        # TODO: choose an experiment_name that follows a clear convention
        experiment_name="predictive-maintenance/xgboost/binary",
        # TODO: choose a registered_model_name
        registered_model_name="xgboost-binary",
        # TODO: fill in any additional fields you added above
        model_family="xgboost",
        target_type="binary"
    ),
    
    "xgb_multiclass": ExperimentConfig(
        # TODO: choose an experiment_name that follows a clear convention
        experiment_name="predictive-maintenance/xgboost/multiclass",
        # TODO: choose a registered_model_name
        registered_model_name="xgboost-multiclass",
        # TODO: fill in any additional fields you added above
        model_family="xgboost",
        target_type="multiclass"
    ),

    # TODO: add at least one more experiment config here
    #       e.g. "rf_binary", "xgb_multiclass", "logreg_binary"
}
 

# ── 2. Feature list ────────────────────────────────────────────────────────────
#
# Q: Which columns should be excluded entirely, and why?
#    (Does UDI carry any signal? What about Product ID?)
#
#  - We drop UDI and Product ID because they do not carry any predictive signal for machine failure. 
#
# Q: The EDA surfaced derived features: temp_diff, power, mechanical stress, hdf_risk…
#    Which ones would you include and on what grounds?
#
#   - Power (Torque x RPM) allows us to more easily detect how failures often occur at the edges and helps us determine a "safe" operational band
#   - Temperature difference combined with RPM (T2 - T1) allows us to better detect heat failure condition boundary
#   - Mechanical stress (Torque x Tool wear) In plain terms: a machine running on high torque under normal conditions is fine, but a machine running high torque and running on high tool wear is
#    under mechanical stress. This feature helps the model detect that combined hazardous condition.
#    
#    These are all derived features that could potentially carry predictive signal for machine failure.
#
#
#    What would you lose by using only the five raw sensor columns?
#    
#   - A model adept at pattern recognition might be able to learn the relationships between the
#     raw sensor columns and the target variable, but it would require more data and computational 
#     resources to do so. By engineering derived features based on domain knowledge, we can provide the 
#     model with more informative inputs that may lead to better performance and faster convergence during training.

# Define the features to be used for training the model 
# and relabel them to be more code-friendly (no spaces, lowercase, etc.)

FEATURES = [
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_minutes",
    "power",
    "temp_diff",
    "mechanical_stress"
]

BINARY_TARGET = "Machine failure"

MULTICLASS_TARGETS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


# ── 3. Preprocessing ───────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with FEATURES + TARGET only.

    Q: For each derived feature you add, point to the EDA chart that
       justifies it. If you can't, should it be here?
    A: addressed and justified in the EDA notebook

    Q: How does DictVectorizer handle a string column like "Type"?
       Do you need to encode it manually, or does the pipeline handle it?
    A: DictVectorizer will automatically handle string columns by converting them into a one-hot encoded format.
       This means that if you have a column like "Type" with categorical values, DictVectorizer will create 
       new binary columns for each unique category in the "Type" column, 
       allowing the model to process the categorical data without needing manual encoding beforehand.
    """
    df = df.copy()
    df = df.rename(columns=COLUMN_RENAME)

    # TODO: engineer any derived features you identified from the EDA
    df["power"] = df["torque_nm"] * df["rotational_speed_rpm"]
    df["temp_diff"] = df["process_temperature_k"] - df["air_temperature_k"]
    df["mechanical_stress"] = df["torque_nm"] * df["tool_wear_minutes"]

    target = "Machine failure"
    return df[FEATURES + [target]]


# ── 4. Model and training ──────────────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    """Preprocess, split, train, and evaluate.

    Returns (pipeline, metrics, params).

    Q: The dataset has a ~97:3 class split. What does a model predict
       if it learns to always say "no failure"? What accuracy would it get?
       Is accuracy the right metric at all?

    Q: XGBClassifier has scale_pos_weight. What value should it take?
       How would you compute it from the training labels, not hardcode it?

    Q: Should you stratify the train/test split? What breaks if you don't,
       given the class imbalance?

    Q: A false negative means a machine fails without warning.
       A false positive means an unnecessary maintenance stop.
       Which costs more in a real factory? How does your answer
       change which metric you optimise for?

    Q: Your pipeline needs at least a vectoriser and a classifier.
       DictVectorizer + XGBClassifier is one option.
       What would you use instead, and what would you gain or lose?
    """
    target = "Machine failure"
    df_processed = preprocess(df)
    y = df_processed[target]
    X = df_processed.drop(columns=[target])

    # TODO: split — consider test_size and whether to stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42  # add your arguments
    )

    X_train_records = X_train.to_dict(orient="records")
    X_test_records = X_test.to_dict(orient="records")

    # TODO: build the pipeline — what goes inside make_pipeline()?
    pipeline = make_pipeline()

    pipeline.fit(X_train_records, y_train)

    y_pred_train = pipeline.predict(X_train_records)
    y_pred_test = pipeline.predict(X_test_records)
    # TODO: if you need probability scores (e.g. for ROC-AUC), compute them here

    # TODO: which metrics did you decide matter most for this use-case?
    metrics: dict[str, float] = {}

    # TODO: which hyperparameters are worth tracking so you can reproduce this run?
    params: dict[str, object] = {}

    return pipeline, metrics, params


# ── 5. MLflow setup ────────────────────────────────────────────────────────────
#
# Q: The tracking URI, username, and password live in .env — not in code.
#    Why? What would happen if you committed them to the repo instead?
#
# Q: configure_mlflow() below receives a config object instead of hardcoding
#    the experiment name. What does that buy you when you have multiple
#    experiments? What would the alternative look like?
#
# Q: mlflow.set_experiment() creates the experiment if it does not exist.
#    Is that the behaviour you want in a team setting, or should new
#    experiments require an explicit approval step?

def configure_mlflow(config: ExperimentConfig) -> None:
    # TODO: read the tracking URI from the environment and set it on mlflow
    #       Hint: os.getenv(...) + mlflow.set_tracking_uri(...)

    # TODO: activate the experiment defined in the config
    #       Hint: mlflow.set_experiment(...)
    pass


# ── 6. MLflow logging ──────────────────────────────────────────────────────────
#
# Q: MLflow has three metadata slots: tags, params, metrics.
#    Before writing any code, sort your values into the right bucket:
#      - tags   → free-form labels that describe the run (who, what, why)
#      - params → inputs you chose before training (hyperparameters, feature list)
#      - metrics → numbers produced by training (scores, counts)
#    Why does the distinction matter? What breaks if you log a metric as a param?
#
# Q: What would a teammate need to see in the MLflow UI to understand this run
#    three months from now, without reading the code?
#
# Q: registered_model_name links a run to a versioned model in the registry.
#    What does "registering" a model give you that a plain logged artifact does not?
#
# Q: log_model() receives the config so it can read registered_model_name.
#    If you hardcoded that name here instead, what would break when you add
#    a second experiment config?

def log_model(
    pipeline, metrics: dict, params: dict, config: ExperimentConfig
) -> None:
    with mlflow.start_run():
        # TODO: set tags — what context about this run should be preserved?
        mlflow.set_tags({})

        # TODO: log params and metrics from the dicts returned by train_model()

        # TODO: log the pipeline as a sklearn model artifact and register it
        #       Hint: mlflow.sklearn.log_model(..., registered_model_name=...)
        pass


# ── 7. CML report ──────────────────────────────────────────────────────────────
def write_cml_metrics(metrics: dict) -> None:
    """Write a markdown summary to metrics.txt for a CML PR report.

    Q: Which metrics from your dict does a reviewer need to see
       to decide whether this model is better than the previous one?
       Which ones are noise at the PR review stage?
    """
    lines = ["# Training Metrics", ""]
    # TODO: add one f-string line per metric you want to surface
    Path("metrics.txt").write_text("\n".join(lines), encoding="utf-8")


# ── Entry point ────────────────────────────────────────────────────────────────
#
# Q: The --experiment flag selects a config from EXPERIMENTS by key.
#    What happens today if someone passes an unknown key?
#    How would you give them a helpful error message?
#
# Q: What other CLI flags might be useful as your experiment suite grows?
#    (Think: --n-estimators, --test-size, --dry-run…)
#    At what point does a config file become a better choice than CLI flags?

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

    pipeline, metrics, params = train_model(df)
    log_model(pipeline, metrics, params, config)

    if cml_run:
        write_cml_metrics(metrics)


if __name__ == "__main__":
    main()
