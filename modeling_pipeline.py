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
    # Standard training run
    python modeling_pipeline.py --experiment xgb_binary
    python modeling_pipeline.py --experiment lgbm_binary --cml-run

    # Hyperparameter tuning with Optuna (lgbm experiments only)
    python modeling_pipeline.py --experiment lgbm_binary --tune
    python modeling_pipeline.py --experiment lgbm_binary --tune --n-trials 50
    python modeling_pipeline.py --experiment lgbm_multiclass --tune --n-trials 30

    # Tuning + CML report in one run
    python modeling_pipeline.py --experiment lgbm_binary --tune --n-trials 30 --cml-run

To add a new experiment: add one entry to EXPERIMENTS. No function code changes needed.
To enable tuning for an experiment: add a param_space lambda to its ExperimentConfig.
Credentials must live in .env — see .env.example.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import click
import mlflow
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline

# Silence Optuna's per-trial logging — summary is printed at the end instead.
optuna.logging.set_verbosity(optuna.logging.WARNING)

load_dotenv()




DATA_PATH = Path("data/ai4i2020.csv")

# Maps original CSV column names to clean snake_case equivalents used throughout.
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

# Five raw sensor readings + three domain-derived features (engineered in preprocess).
# "type" (L/M/H machine variant) is a string; DictVectorizer one-hot encodes it automatically.
FEATURES = [
    "type",
    "air_temperature_kelvin",
    "process_temperature_kelvin",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_minutes",
    "power_kw",           # torque × rpm → watts, converted to kW
    "temp_diff_kelvin",   # process − air temperature
    "mechanical_stress",  # torque × tool wear (combined wear hazard)
]


# ── Experiment registry ────────────────────────────────────────────────────────
# Central config layer: all experiment-specific decisions live here.
# Adding a new experiment = one new dict entry; no function code changes needed.

@dataclass
class ExperimentConfig:
    """All settings needed to run, log, and reproduce one experiment.

    The classifier_factory pattern keeps train_model() free of if/else branching:
    each config owns its classifier definition. All variation is here, not
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
                               Controls label handling, metric selection, and ROC-AUC.
        metric_average:        Averaging strategy passed to sklearn scoring functions.
                               "binary" for two-class targets, "macro" for multiclass.
        classifier_factory:    Callable(imbalance_ratio: float) → unfitted classifier.
                               Owns all hyperparameters for this experiment.
                               Multiclass factories ignore imbalance_ratio (use lambda _).
        test_size:             Fraction of data held out for evaluation. Default 0.2.
        description:           Optional free-text summary for documentation.
        notes:                 Optional scratchpad — not logged to MLflow.
        tags:                  Extra key/value pairs merged into MLflow run tags.
        param_space:           Optional Callable(trial, imbalance_ratio) → dict of params.
                               When provided, --tune mode uses this to define the search
                               space Optuna samples from. If None, --tune is not supported
                               for this experiment.
    """
    experiment_name: str
    registered_model_name: str
    model_family: str
    target: str
    target_type: str       # "binary" or "multiclass"
    metric_average: str    # "binary" or "macro" — passed directly to sklearn metrics
    classifier_factory: Callable
    test_size: float = 0.2
    description: str = ""
    notes: Optional[str] = None
    tags: dict = field(default_factory=dict)
    # param_space is optional — only experiments that support tuning define it.
    # None means "this experiment has no search space; --tune will raise clearly."
    param_space: Optional[Callable] = None


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "xgb_binary": ExperimentConfig(
        experiment_name="predictive-maintenance/xgboost/binary",
        registered_model_name="xgboost-binary",
        model_family="xgboost",
        target="machine_failure",
        target_type="binary",
        metric_average="binary",
        # `r` = imbalance_ratio (~28). scale_pos_weight tells XGBoost to penalise
        # missed failures 28× more heavily — no resampling needed.
        classifier_factory=lambda r: xgb.XGBClassifier(
            n_estimators=200,
            scale_pos_weight=r,    # passed in from train_model; compensates ~97:3 split
            random_state=42,
            n_jobs=-1,             # use all CPU cores
            eval_metric="logloss",
        ),
    ),

    "xgb_multiclass": ExperimentConfig(
        experiment_name="predictive-maintenance/xgboost/multiclass",
        registered_model_name="xgboost-multiclass",
        model_family="xgboost",
        target="failure_type",
        target_type="multiclass",
        metric_average="macro",
        # `_` signals the factory intentionally ignores imbalance_ratio.
        classifier_factory=lambda _: xgb.XGBClassifier(
            n_estimators=200,
            objective="multi:softprob",  # outputs a probability per class
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
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
            max_depth=4,
        ),
        # param_space defines what Optuna is allowed to search.
        # trial.suggest_* methods tell Optuna the type and range of each parameter:
        #   suggest_int   → integer (e.g. tree depth)
        #   suggest_float → continuous value; log=True means search on log scale,
        #                   useful for learning_rate which spans 0.001 → 0.3
        # imbalance_ratio (r) is passed through directly — it's data-derived, not tunable.
        param_space=lambda trial, r: dict(
            n_estimators       = trial.suggest_int("n_estimators", 100, 500),
            max_depth          = trial.suggest_int("max_depth", 2, 8),
            learning_rate      = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples  = trial.suggest_int("min_child_samples", 10, 100),
            reg_lambda         = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            scale_pos_weight   = r,
            random_state       = 42,
            n_jobs             = -1,
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
            max_depth=3,
            min_child_samples=30,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
        ),
        param_space=lambda trial, _: dict(
            n_estimators       = trial.suggest_int("n_estimators", 100, 500),
            max_depth          = trial.suggest_int("max_depth", 2, 6),
            learning_rate      = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples  = trial.suggest_int("min_child_samples", 20, 150),
            reg_lambda         = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            objective          = "multiclass",
            random_state       = 42,
            n_jobs             = -1,
        ),
    ),
    "rf_binary": ExperimentConfig(
        experiment_name="predictive-maintenance/random-forest/binary",
        registered_model_name="random-forest-binary",
        model_family="random_forest",
        target="machine_failure",
        target_type="binary",
        metric_average="binary",
        # class_weight="balanced" is RF's equivalent of XGBoost's scale_pos_weight.
        classifier_factory=lambda _: RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
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
            class_weight="balanced",
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
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
            class_weight="balanced",
            max_iter=1000,   # default 100 rarely converges on this dataset
            random_state=42,
        ),
    ),
    "logreg_multiclass": ExperimentConfig(
        experiment_name="predictive-maintenance/logistic-regression/multiclass",
        registered_model_name="logreg-multiclass",
        model_family="logreg",
        target="failure_type",
        target_type="multiclass",
        metric_average="macro",
        # sklearn's LogisticRegression handles multiclass natively (one-vs-rest by default).
        classifier_factory=lambda _: LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    ),
}


# ── Preprocessing ──────────────────────────────────────────────────────────────
# Renames columns, engineers three domain features from EDA, and (for multiclass)
# collapses the five binary failure flags into one string label.
# Output contains only FEATURES + target — nothing else reaches the model.

def preprocess(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Rename columns, engineer features, and slice to model inputs + target.

    Domain features added here (each justified by EDA):
    - power_kw:          torque × rpm → kW. Failures cluster at power extremes.
    - temp_diff_kelvin:  process − air temperature. HDF risk rises when diff < 8.6 K.
    - mechanical_stress: torque × tool wear. High torque on a worn tool is a compound hazard.

    For multiclass experiments the five binary failure columns (twf, hdf, pwf, osf, rnf)
    are collapsed into a single string label; rows with no active flag become "none".

    Args:
        df:     Raw DataFrame loaded directly from ai4i2020.csv (original column names).
        config: Active experiment config. Determines the target column and whether
                the multiclass label column is built.

    Returns:
        DataFrame with columns FEATURES + [config.target]. Shape: (n_rows, 10).
        All rows from the input are preserved — no filtering is applied here.
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
# Thin wrapper that keeps train_model() free of any classifier-specific logic.
# All hyperparameter decisions live in the EXPERIMENTS registry.

def _build_classifier(config: ExperimentConfig, imbalance_ratio: float):
    """Instantiate the classifier defined in config.classifier_factory.

    Args:
        config:           Active experiment config.
        imbalance_ratio:  Ratio of negative to positive training samples (~28 for this
                          dataset). Passed to the factory lambda; multiclass factories
                          ignore it (declared as lambda _).

    Returns:
        Unfitted sklearn-compatible classifier instance.
    """
    return config.classifier_factory(imbalance_ratio)


# ── Training ───────────────────────────────────────────────────────────────────
# Orchestrates the full run: split → build pipeline → fit → score.
# The pipeline chains DictVectorizer (handles mixed types, one-hot encodes strings)
# into the classifier. Inputs are passed as record dicts, not numeric arrays.

def train_model(df: pd.DataFrame, config: ExperimentConfig):
    """Preprocess, split, train, and evaluate. Return pipeline, metrics, and params.

    Stratified split preserves the minority-class ratio in both folds — essential
    with a ~97:3 split to avoid an empty positive class in the test set.
    ROC-AUC is computed for binary targets only (requires scalar probability scores).
    f1_train is logged alongside f1_test to surface overfitting at a glance.

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
            target_type, and test_size. Logged to MLflow to reproduce this run.
    """
    df_processed = preprocess(df, config)
    y = df_processed[config.target]
    X = df_processed.drop(columns=[config.target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=config.test_size, stratify=y
    )

    # Guard: multiclass y_train contains string labels ("twf", "none", …).
    # Integer comparisons (== 0, == 1) return all-False → 0 / 0 → ZeroDivisionError.
    if config.target_type == "binary":
        imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    else:
        imbalance_ratio = 1.0  # ignored by multiclass factories (lambda _)

    classifier = _build_classifier(config, imbalance_ratio)

    # DictVectorizer expects a list of row dicts — convert here once.
    X_train_records = X_train.to_dict(orient="records")
    X_test_records = X_test.to_dict(orient="records")

    pipeline = make_pipeline(DictVectorizer(), classifier)
    pipeline.fit(X_train_records, y_train)

    y_pred_train = pipeline.predict(X_train_records)
    y_pred_test = pipeline.predict(X_test_records)

    y_prob_test = (
        pipeline.predict_proba(X_test_records)[:, 1]  # positive-class probability score
        if config.target_type == "binary"
        else None
    )

    average = config.metric_average  # "binary" or "macro" — stored per experiment
    metrics: dict[str, float] = {
        "f1_train":       f1_score(y_train, y_pred_train, average=average),
        "f1_test":        f1_score(y_test, y_pred_test, average=average),
        "precision_test": precision_score(y_test, y_pred_test, average=average),
        "recall_test":    recall_score(y_test, y_pred_test, average=average),
    }
    if y_prob_test is not None:
        metrics["roc_auc_test"] = roc_auc_score(y_test, y_prob_test)

    params: dict[str, object] = {
        **classifier.get_params(),        # full hyperparameter set from the classifier
        "model_family": config.model_family,
        "target_type":  config.target_type,
        "test_size":    config.test_size,
    }

    return pipeline, metrics, params


# ── Optuna hyperparameter tuning ───────────────────────────────────────────────
# Optuna works by running many "trials". Each trial:
#   1. Samples a set of hyperparameters from the param_space
#   2. Trains the model using cross-validation (not a full train/test split)
#   3. Returns a score — Optuna uses this to decide where to sample next
#
# Why cross-validation instead of a single train/test split?
#   A single split is noisy — the score depends on which rows ended up in test.
#   CV splits the training data into k folds, trains k times, and averages the
#   scores. This gives a more reliable signal per trial.
#
# Why not just grid search?
#   Grid search tries every combination — 5 values × 5 values × 5 values = 125 fits.
#   Optuna uses Bayesian optimisation (TPE sampler): after each trial it builds a
#   model of which regions of the search space produce good scores, and focuses
#   subsequent trials there. 30 Optuna trials typically beats 125 grid search fits.

def tune_model(
    df: pd.DataFrame,
    config: ExperimentConfig,
    n_trials: int = 30,
) -> dict:
    """Run Optuna hyperparameter search and return the best params found.

    Uses StratifiedKFold cross-validation as the objective so that the minority
    class is represented in every fold — important given the ~97:3 imbalance.
    The test set is never touched during tuning; it is only used in the final
    train_model() call after the best params are applied to classifier_factory.

    Args:
        df:       Raw DataFrame from ai4i2020.csv.
        config:   Active experiment config. Must have param_space defined.
        n_trials: Number of Optuna trials to run. More trials = better search
                  but longer runtime. 30 is a reasonable default for a laptop.

    Returns:
        Dict of best hyperparameters found. Caller applies these by updating
        config.classifier_factory before passing to train_model().

    Raises:
        ValueError: If config.param_space is None (experiment has no search space).
    """
    if config.param_space is None:
        raise ValueError(
            f"Experiment '{config.experiment_name}' has no param_space defined. "
            "Add a param_space lambda to its ExperimentConfig to enable tuning."
        )

    # Preprocess once — no point repeating feature engineering on every trial.
    df_processed = preprocess(df, config)
    y = df_processed[config.target]
    X = df_processed.drop(columns=[config.target])

    # Hold out the test set now and never touch it during tuning.
    # Tuning happens entirely within X_train — this prevents the test set from
    # influencing hyperparameter selection (which would be a form of data leakage).
    X_train, _, y_train, _ = train_test_split(
        X, y, random_state=42, test_size=config.test_size, stratify=y
    )

    # Compute imbalance ratio from training labels only (same as train_model).
    if config.target_type == "binary":
        imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    else:
        imbalance_ratio = 1.0

    # StratifiedKFold ensures each fold has the same class ratio as the full set.
    # 5 folds = each trial trains 5 models and averages their scores.
    # We use a manual loop (not cross_val_score) so we can pass eval_set to each
    # fold's fit() call — required for LightGBM early stopping.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # DictVectorizer is fit once on the full training set and reused across folds.
    # This is correct because the vectoriser only learns the feature schema (column
    # names and categories) — not any target-related statistics — so fitting it
    # outside the CV loop does not leak information.
    dv = DictVectorizer(sparse=False)
    X_train_records = X_train.to_dict(orient="records")
    X_train_vec = dv.fit_transform(X_train_records)

    def objective(trial: optuna.Trial) -> float:
        # Sample a candidate set of hyperparameters for this trial.
        # Optuna learns from previous trials which regions look promising.
        params = config.param_space(trial, imbalance_ratio)

        # early_stopping_rounds: if the validation score does not improve for
        # this many consecutive trees, LightGBM stops adding more trees early.
        # This directly prevents the model from memorising training data
        # (the f1_train=1.0 problem) within each trial.
        # n_estimators in params becomes the *maximum* number of trees —
        # early stopping may use far fewer.
        early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 20, 50)

        fold_scores = []

        for train_idx, val_idx in cv.split(X_train_vec, y_train):
            X_fold_train = X_train_vec[train_idx]
            X_fold_val   = X_train_vec[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val   = y_train.iloc[val_idx]

            classifier = lgb.LGBMClassifier(**params, verbose=-1)

            # eval_set gives LightGBM a validation fold to monitor during training.
            # Each new tree is evaluated on this fold — if it doesn't improve the
            # score for early_stopping_rounds consecutive rounds, training stops.
            classifier.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )

            y_pred = classifier.predict(X_fold_val)
            fold_scores.append(
                f1_score(y_fold_val, y_pred, average=config.metric_average)
            )

        # Mean across all 5 folds — more stable than a single split score.
        return sum(fold_scores) / len(fold_scores)

    # Create the study. "maximize" because higher F1 = better.
    # TPESampler is Optuna's default Bayesian sampler — not random search.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest CV f1_{config.metric_average}: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


# ── MLflow ─────────────────────────────────────────────────────────────────────
# Two-step process: configure the experiment server-side, then log the run.
# MLflow distinguishes three metadata types — keep them separate:
#   tags   → who/what/why (free-form labels, not compared across runs)
#   params → inputs chosen before training (hyperparameters, split size)
#   metrics → outputs produced by training (scores, counts)

def configure_mlflow(config: ExperimentConfig) -> None:
    """Point MLflow at the remote tracking server and activate the experiment.

    Reads MLFLOW_TRACKING_URI from the environment (loaded from .env).
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

    The model is both stored as a run artifact and registered under
    config.registered_model_name, enabling versioned lifecycle management
    (staging → production) in the MLflow registry.

    Args:
        pipeline: Fitted sklearn Pipeline produced by train_model().
        metrics:  Evaluation scores produced by train_model() — logged as metrics.
        params:   Hyperparameters produced by train_model() — logged as params.
        config:   Active experiment config. Supplies tags and the registered model name.
    """
    with mlflow.start_run():
        mlflow.set_tags({
            "model_family":    config.model_family,
            "target_type":     config.target_type,
            "target":          config.target,
            "experiment_name": config.experiment_name,
            "developer":       os.getenv("MLFLOW_TRACKING_USERNAME", "unknown"),
        })
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=config.registered_model_name,
        )


# ── CML report ─────────────────────────────────────────────────────────────────
# Writes a lightweight markdown file consumed by the CML GitHub Action.
# The Action attaches it as a comment on the pull request for quick review.

def write_cml_metrics(metrics: dict) -> None:
    """Write key test metrics to metrics.txt for a CML pull-request comment.

    Includes f1, precision, recall on the test set, and roc_auc (binary only).
    f1_train is intentionally omitted — reviewers need test performance,
    not evidence of fitting.

    Args:
        metrics: Dict produced by train_model(). Keys f1_test, precision_test,
                 and recall_test are always present. roc_auc_test is optional
                 (binary experiments only).
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
# Click validates --experiment against EXPERIMENTS keys automatically and prints
# the full list of valid choices on error — no extra validation needed.

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
@click.option(
    "--tune/--no-tune",
    default=False,
    help="Run Optuna hyperparameter search before training.",
)
@click.option(
    "--n-trials",
    default=30,
    show_default=True,
    help="Number of Optuna trials to run (only used with --tune).",
)
def main(experiment: str, cml_run: bool, tune: bool, n_trials: int) -> None:
    """Train a predictive maintenance failure classifier.

    With --tune: runs Optuna hyperparameter search first, then trains the final
    model with the best params found and logs it to MLflow as a normal run.
    Without --tune: trains once with the fixed params in EXPERIMENTS.
    """
    config = EXPERIMENTS[experiment]
    df = pd.read_csv(DATA_PATH)
    configure_mlflow(config)

    if tune:
        # ── Tuning path ───────────────────────────────────────────────────────
        # tune_model() searches for the best hyperparameters using CV on the
        # training set only. It returns a dict of the winning params.
        print(f"Running Optuna search ({n_trials} trials) for {experiment}...")
        best_params = tune_model(df, config, n_trials=n_trials)

        # Patch classifier_factory to use the best params Optuna found.
        # early_stopping_rounds is stripped out here — it was used during CV
        # (where each fold had an eval_set) but the final train_model() call
        # uses pipeline.fit() without eval_set, so passing it would crash.
        classifier_params = {
            k: v for k, v in best_params.items()
            if k != "early_stopping_rounds"
        }

        if config.model_family == "lightgbm":
            config.classifier_factory = lambda r: lgb.LGBMClassifier(
                **{**classifier_params, "scale_pos_weight": r if config.target_type == "binary" else 1.0}
            )
        elif config.model_family == "xgboost":
            config.classifier_factory = lambda r: xgb.XGBClassifier(
                **{**classifier_params, "scale_pos_weight": r if config.target_type == "binary" else 1.0}
            )
        # Note: RF and logreg don't have param_space defined so --tune will
        # raise a clear ValueError before reaching here.

    # ── Standard training path (always runs) ─────────────────────────────────
    # Whether or not we tuned, train_model() does the final fit on the full
    # train set and evaluates on the held-out test set.
    pipeline, metrics, params = train_model(df, config)

    if tune:
        # Tag the MLflow run so it's clearly identifiable as a tuned run.
        params["optuna_n_trials"] = n_trials
        params["tuned"] = True

    log_model(pipeline, metrics, params, config)

    if cml_run:
        write_cml_metrics(metrics)


if __name__ == "__main__":
    main()
