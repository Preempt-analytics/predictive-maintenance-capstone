"""
Predictive Maintenance — Modeling Pipeline
==========================================
Trains a failure classifier on the AI4I 2020 dataset and logs results to MLflow.

Supported experiments (pass via --experiment):
    xgb_binary,    xgb_multiclass
    rf_binary,     rf_multiclass
    logreg_binary, logreg_multiclass
    lgbm_binary,   lgbm_multiclass
    svm_binary,    svm_multiclass
    mlp_binary,    mlp_multiclass

Usage:
    # Standard training run
    python src/modeling_pipeline.py --experiment xgb_binary
    python src/modeling_pipeline.py --experiment lgbm_binary --cml-run

    # Hyperparameter tuning with Optuna
    python src/modeling_pipeline.py --experiment lgbm_binary --tune
    python src/modeling_pipeline.py --experiment lgbm_binary --tune --n-trials 50
    python src/modeling_pipeline.py --experiment svm_binary --tune --n-trials 50

    # Tuning + CML report in one run
    python src/modeling_pipeline.py --experiment lgbm_binary --tune --n-trials 30 --cml-run

To add a new experiment: add one entry to EXPERIMENTS. No function code changes needed.
To enable tuning for an experiment: add a param_space lambda to its ExperimentConfig.
SVM and other distance-based models require needs_scaling=True — StandardScaler is
inserted into the pipeline automatically when this flag is set.
Credentials must live in .env — see .env.example.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml
import click
import mlflow
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Silence Optuna's per-trial logging — summary is printed at the end instead.
optuna.logging.set_verbosity(optuna.logging.WARNING)

load_dotenv()

_p = yaml.safe_load(open("params.yaml"))["pipeline"]
TEST_SIZE     = _p["test_size"]
RANDOM_STATE  = _p["random_state"]
CV_FOLDS      = _p["cv_folds"]


# ── Data path and training window ─────────────────────────────────────────────
# Parquet is a columnar format: it is compressed (~10× smaller than CSV) and
# lets pandas load only the columns the model needs without reading the whole
# file. The training window caps how many rows are used per run. Once the
# dataset grows beyond 50,000 rows from repeated retrain cycles, df.tail()
# keeps training time flat by always taking the most recent observations —
# the rows that best reflect the current factory state. At a 3.4% failure rate,
# 50,000 rows still provides ~1,700 positive examples — well above what any of
# the classifiers here need.
DATA_PATH        = Path("data/ai4i2020.parquet")
TRAINING_WINDOW  = 50_000   # rows; only takes effect once the dataset exceeds this size

# ── Calibration ─────────────────────────────────────────────────────────────────
# Tree-based models tend to push predict_proba toward 0 or 1 more often than
# they're actually right — a confident-looking number that isn't trustworthy.
# CalibratedClassifierCV corrects this by holding out folds, fitting the raw
# classifier on the rest, and learning a correction curve from the held-out
# predictions. cv=3 (not sklearn's default of 5): the rarest class in this
# dataset (PWF) has only ~7 rows total, ~5 after the 80/20 split — 5 folds
# would leave under 1 sample per fold and crash; 3 leaves enough margin.
CALIBRATION_CV     = 3
CALIBRATION_METHOD = "sigmoid"   # Platt scaling — safer than isotonic on small/imbalanced classes

# ── xgboost / scikit-learn version mismatch workaround ──────────────────────────
# xgboost==2.0.3 predates sklearn 1.8's tag-based estimator typing, so
# sklearn.base.is_classifier(XGBClassifier()) incorrectly returns False —
# XGBClassifier reports no estimator_type tag at all. CalibratedClassifierCV
# calls is_classifier() internally to decide whether predict_proba is valid;
# misdetected as "not a classifier", it raises instead of calibrating.
# This patches the CLASS (not an instance) because CalibratedClassifierCV
# clones the estimator internally for each CV fold — an instance-level patch
# would not survive the clone. Remove this once xgboost is upgraded past the
# version that added proper __sklearn_tags__ support (2.1+).
_original_xgb_tags = xgb.XGBClassifier.__sklearn_tags__
def _patched_xgb_tags(self):
    tags = _original_xgb_tags(self)
    tags.estimator_type = "classifier"
    return tags
xgb.XGBClassifier.__sklearn_tags__ = _patched_xgb_tags

from feature_transformation import FEATURES, FAILURE_TYPE_TO_INT, engineer_features  # noqa: E402



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
        description:           Optional free-text summary for documentation.
        notes:                 Optional scratchpad — not logged to MLflow.
        tags:                  Extra key/value pairs merged into MLflow run tags.
        param_space:           Optional Callable(trial, imbalance_ratio) → dict of params.
                               When provided, --tune mode uses this to define the search
                               space Optuna samples from. If None, --tune is not supported
                               for this experiment.
        needs_scaling:         If True, StandardScaler is inserted between DictVectorizer
                               and the classifier in the pipeline. Required for distance-
                               and boundary-based models (SVM, KNN, MLP). Tree-based
                               models are scale-invariant — leave False for them.
    """
    experiment_name: str
    registered_model_name: str
    model_family: str
    target: str
    target_type: str       # "binary" or "multiclass"
    metric_average: str    # "binary" or "macro" — passed directly to sklearn metrics
    classifier_factory: Callable
    description: str = ""
    notes: Optional[str] = None
    tags: dict = field(default_factory=dict)
    # param_space is optional — only experiments that support tuning define it.
    # None means "this experiment has no search space; --tune will raise clearly."
    param_space: Optional[Callable] = None
    # needs_scaling controls whether StandardScaler is inserted into the pipeline.
    # Tree models are invariant to feature scale; distance-based models are not.
    needs_scaling: bool = False


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "xgb_binary": ExperimentConfig(
        experiment_name="predictive-maintenance/xgboost/binary",
        registered_model_name="predictive-maintenance-binary",
        model_family="xgboost",
        target="machine_failure",
        target_type="binary",
        metric_average="binary",
        classifier_factory=lambda r: xgb.XGBClassifier(
            n_estimators=200,
            scale_pos_weight=r,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        param_space=lambda trial, r: dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 500),
            max_depth         = trial.suggest_int("max_depth", 2, 8),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_weight  = trial.suggest_int("min_child_weight", 1, 20),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            scale_pos_weight  = r,
            random_state      = 42,
            n_jobs            = -1,
            eval_metric       = "logloss",
        ),
    ),

    "xgb_multiclass": ExperimentConfig(
        experiment_name="predictive-maintenance/xgboost/multiclass",
        registered_model_name="predictive-maintenance-multiclass",
        model_family="xgboost",
        target="failure_type",
        target_type="multiclass",
        metric_average="macro",
        classifier_factory=lambda _: xgb.XGBClassifier(
            n_estimators=200,
            objective="multi:softprob",
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        ),
        param_space=lambda trial, _: dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 500),
            max_depth         = trial.suggest_int("max_depth", 2, 6),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_weight  = trial.suggest_int("min_child_weight", 1, 20),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            objective         = "multi:softprob",
            random_state      = 42,
            n_jobs            = -1,
            eval_metric       = "mlogloss",
        ),
    ),

    "lgbm_binary": ExperimentConfig(
        experiment_name="predictive-maintenance/lightgbm/binary",
        registered_model_name="predictive-maintenance-binary",
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
        registered_model_name="predictive-maintenance-multiclass",
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
        registered_model_name="predictive-maintenance-binary",
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
        registered_model_name="predictive-maintenance-multiclass",
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
        registered_model_name="predictive-maintenance-binary",
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
        registered_model_name="predictive-maintenance-multiclass",
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

    # ── SVM ───────────────────────────────────────────────────────────────────────
    # SVC finds the maximum-margin hyperplane separating classes. The RBF kernel
    # maps features into a higher-dimensional space, enabling non-linear boundaries.
    #
    # Key difference from tree models: SVM is sensitive to feature scale, so
    # needs_scaling=True inserts StandardScaler into the pipeline automatically.
    #
    # C controls the bias-variance tradeoff:
    #   small C → wide margin, more misclassifications allowed (high bias)
    #   large C → narrow margin, tries to classify everything correctly (high variance)
    # gamma controls the RBF kernel width:
    #   "scale" → 1 / (n_features * X.var())  — adapts to data variance
    #   "auto"  → 1 / n_features               — simpler, ignores variance
    #
    # probability=True enables predict_proba (needed for ROC-AUC), but adds
    # overhead via Platt scaling — disabled during CV trials for speed.
    "svm_binary": ExperimentConfig(
        experiment_name="predictive-maintenance/svm/binary",
        registered_model_name="predictive-maintenance-binary",
        model_family="svm",
        target="machine_failure",
        target_type="binary",
        metric_average="binary",
        needs_scaling=True,
        classifier_factory=lambda _: SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        param_space=lambda trial, _: dict(
            C      = trial.suggest_float("C", 1e-2, 1e2, log=True),
            gamma  = trial.suggest_categorical("gamma", ["scale", "auto"]),
            kernel = "rbf",
        ),
    ),

    "svm_multiclass": ExperimentConfig(
        experiment_name="predictive-maintenance/svm/multiclass",
        registered_model_name="predictive-maintenance-multiclass",
        model_family="svm",
        target="failure_type",
        target_type="multiclass",
        metric_average="macro",
        needs_scaling=True,
        # SVC handles multiclass natively via one-vs-one (OvO) by default.
        # With k classes, OvO trains k*(k-1)/2 binary SVMs and takes a majority vote.
        classifier_factory=lambda _: SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        param_space=lambda trial, _: dict(
            C      = trial.suggest_float("C", 1e-2, 1e2, log=True),
            gamma  = trial.suggest_categorical("gamma", ["scale", "auto"]),
            kernel = "rbf",
        ),
    ),

    # ── MLP (Neural Network) ──────────────────────────────────────────────────────
    #
    # How a neural network works — plain language:
    #
    #   Input layer:   one node per feature (9 here). No computation — just passes
    #                  sensor readings into the network.
    #
    #   Hidden layers: each neuron takes a weighted sum of the previous layer's
    #                  outputs and passes it through an activation function.
    #                  The weights are what the network "learns" during training.
    #                  More neurons = more capacity to capture patterns.
    #                  More layers = more abstract patterns (layer 1 might learn
    #                  "high torque", layer 2 might learn "high torque AND worn tool").
    #
    #   Output layer:  one node per class. Softmax converts raw scores to
    #                  probabilities that sum to 1.
    #
    # Why activation functions matter:
    #   Without them, stacking layers is mathematically identical to one layer —
    #   a linear function of a linear function is still linear. Activations
    #   introduce non-linearity so the network can learn curved decision boundaries.
    #
    #   ReLU (Rectified Linear Unit) = max(0, x)
    #     Passes positive values through unchanged, kills negatives.
    #     Fast to compute. Default choice for most problems.
    #
    #   tanh = (e^x - e^-x) / (e^x + e^-x)
    #     Squashes output to [-1, 1]. Centred at zero (unlike ReLU).
    #     Can work better when features have negative values.
    #
    # How training works — backpropagation:
    #   1. Forward pass: run a batch of rows through the network, get predictions.
    #   2. Compute loss: how wrong were the predictions? (cross-entropy loss here)
    #   3. Backward pass: use the chain rule to compute how much each weight
    #      contributed to the error.
    #   4. Gradient descent: nudge every weight slightly in the direction that
    #      reduces the error. learning_rate_init controls the step size.
    #   Repeat until the val score stops improving (early_stopping=True).
    #
    # How MLP prevents overfitting:
    #   alpha (L2 regularization): adds a penalty to the loss for large weights.
    #     Large weights = the network is relying too heavily on specific features.
    #     Penalising them forces the network to spread evidence across many features.
    #     Small alpha → weak penalty, more expressive (overfit risk).
    #     Large alpha → strong penalty, smoother boundary (underfit risk).
    #   early_stopping: stops training when internal val score plateaus — prevents
    #     the network from continuing to memorise training rows after generalisation peaks.
    #
    # Why MLP needs scaling:
    #   Gradient descent moves all weights in proportion to their input values.
    #   A feature in [200, 400 K] causes gradient steps 200× larger than a feature
    #   in [0, 2 kW]. Without scaling, the optimiser oscillates and converges slowly
    #   or not at all. StandardScaler puts every feature on the same footing.
    #
    # Imbalance limitation:
    #   sklearn's MLPClassifier has NO class_weight parameter (unlike RF or LogReg).
    #   The network sees ~97 % "no failure" rows and may learn to predict that by
    #   default. Watch overfit_delta and recall_test closely — low recall on the
    #   positive class is the tell-tale sign that imbalance is hurting.

    "mlp_binary": ExperimentConfig(
        experiment_name="predictive-maintenance/mlp/binary",
        registered_model_name="predictive-maintenance-binary",
        model_family="mlp",
        target="machine_failure",
        target_type="binary",
        metric_average="binary",
        needs_scaling=True,
        # hidden_layer_sizes=(128, 64): two hidden layers, 128 neurons then 64.
        # Wider first layer captures more combinations of input features;
        # narrower second layer compresses them into higher-level patterns.
        classifier_factory=lambda _: MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-3,
            max_iter=300,
            early_stopping=True,
            random_state=42,
        ),
        param_space=lambda trial, _: dict(
            # Architecture: stored as "width_width" strings because Optuna's persistent
            # storage cannot serialise Python tuples. The objective converts them back
            # to tuples before passing to MLPClassifier.
            # "64"     → one hidden layer of 64 neurons  (shallow)
            # "128"    → one hidden layer of 128 neurons (shallow, wider)
            # "64_32"  → two layers: 64 then 32         (gets more abstract)
            # "128_64" → two layers: 128 then 64        (more capacity, more abstract)
            hidden_layer_sizes = trial.suggest_categorical(
                "hidden_layer_sizes", ["64", "128", "64_32", "128_64"]
            ),
            # Activation: relu is faster; tanh can win when features go negative.
            activation         = trial.suggest_categorical("activation", ["relu", "tanh"]),
            # alpha: L2 penalty weight. Log scale because useful range spans 4 orders
            # of magnitude (0.0001 to 0.1).
            alpha              = trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
            # learning_rate_init: step size for each gradient update.
            # Too large → overshoots the minimum. Too small → very slow convergence.
            learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            # max_iter is the upper bound on epochs. early_stopping will usually
            # stop training well before this limit is reached.
            max_iter           = 300,
        ),
    ),

    "mlp_multiclass": ExperimentConfig(
        experiment_name="predictive-maintenance/mlp/multiclass",
        registered_model_name="predictive-maintenance-multiclass",
        model_family="mlp",
        target="failure_type",
        target_type="multiclass",
        metric_average="macro",
        needs_scaling=True,
        # Same architecture as mlp_binary. MLPClassifier detects multiple classes
        # automatically and switches to a softmax output layer — no config change needed.
        classifier_factory=lambda _: MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-3,
            max_iter=300,
            early_stopping=True,
            random_state=42,
        ),
        param_space=lambda trial, _: dict(
            hidden_layer_sizes = trial.suggest_categorical(
                "hidden_layer_sizes", ["64", "128", "64_32", "128_64"]
            ),
            activation         = trial.suggest_categorical("activation", ["relu", "tanh"]),
            alpha              = trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
            learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            max_iter           = 300,
        ),
    ),
}


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Apply feature engineering and target construction to the raw DataFrame.

    Delegates column renaming and domain feature derivation to engineer_features(),
    which is shared with the simulator to guarantee identical transforms at inference.
    For multiclass targets, collapses the five binary failure flags into a single
    string label. Output is restricted to FEATURES + target — nothing else reaches
    the model.

    Args:
        df:     Raw DataFrame from ai4i2020.parquet.
        config: Active experiment config. target_type controls whether failure_type
                is constructed from the binary flag columns (multiclass only).

    Returns:
        DataFrame containing only FEATURES + config.target, ready for
        train_test_split and DictVectorizer.
    """
    
    # engineer_features() handles column renaming + all three derived features.
    # Defined in feature_transformation.py so the simulator uses the exact same transforms.
    df = engineer_features(df)

    if config.target_type == "multiclass":
        failure_cols = ["twf", "hdf", "pwf", "osf"]          # rnf excluded: never in data (no sensor signature)
        def resolve_label(row):
            active = [c for c in failure_cols if row[c] == 1]
            return FAILURE_TYPE_TO_INT[active[0] if active else "none"]
        df["failure_type"] = df.apply(resolve_label, axis=1)
    
    return df[FEATURES + [config.target]]

# ── Classifier builder ─────────────────────────────────────────────────────────
# Thin wrapper that keeps train_model() free of any classifier-specific logic.
# All hyperparameter decisions live in the EXPERIMENTS registry.

def _build_classifier(config: ExperimentConfig, imbalance_ratio: float):
    """Instantiate the classifier defined in config.classifier_factory, calibrated.

    Every classifier is wrapped in CalibratedClassifierCV so predict_proba
    output can be trusted as an actual probability, not just a confident-looking
    raw score — see the CALIBRATION_CV comment above for why cv=3.

    Args:
        config:           Active experiment config.
        imbalance_ratio:  Ratio of negative to positive training samples (~28 for this
                          dataset). Passed to the factory lambda; multiclass factories
                          ignore it (declared as lambda _).

    Returns:
        Unfitted CalibratedClassifierCV wrapping the configured classifier.
    """
    base_classifier = config.classifier_factory(imbalance_ratio)
    return CalibratedClassifierCV(base_classifier, method=CALIBRATION_METHOD, cv=CALIBRATION_CV)


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
        df:     Raw DataFrame from ai4i2020.parquet.
        config: Active experiment config. Drives split size, target column,
                classifier selection, and metric averaging strategy.

    Returns:
        pipeline (sklearn.Pipeline):
            Fitted DictVectorizer + classifier. Ready for mlflow.sklearn.log_model().
        metrics (dict[str, float]):
            f1_train, f1_test, precision_test, recall_test, brier_score —
            always present. roc_auc_test — binary experiments only.
        params (dict[str, object]):
            Full classifier hyperparameters from get_params(), plus model_family,
            target_type, and test_size. Logged to MLflow to reproduce this run.
    """
    df_processed = preprocess(df, config)
    y = df_processed[config.target]
    X = df_processed.drop(columns=[config.target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE, test_size=TEST_SIZE, stratify=y
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

    # StandardScaler is inserted for distance/boundary-based models (SVM, KNN, MLP).
    # Tree models ignore scale — omitting it keeps their pipelines leaner.
    # sparse=False is required when scaling: StandardScaler cannot center a sparse
    # matrix (subtracting the mean from a sparse array would make it dense anyway).
    # DictVectorizer defaults to sparse=True for memory efficiency on wide feature
    # spaces — safe to override here because our feature set is only ~11 columns.
    if config.needs_scaling:
        pipeline = make_pipeline(DictVectorizer(sparse=False), StandardScaler(), classifier)
    else:
        pipeline = make_pipeline(DictVectorizer(sparse=False), classifier)
    # LightGBM warns "X does not have valid feature names" when it receives a
    # numpy array. set_output("pandas") fixes this by making every transformer
    # output a named DataFrame. Restricted to LightGBM only: applying it
    # unconditionally causes XGBoost to be probed twice during fit (sklearn 1.8
    # internal behaviour), which corrupts the label encoder state and raises
    # "Invalid classes inferred from unique values of y".
    if config.model_family == "lightgbm":
        pipeline.set_output(transform="pandas")
    pipeline.fit(X_train_records, y_train)

    y_pred_train = pipeline.predict(X_train_records)
    y_pred_test = pipeline.predict(X_test_records)

    # Full probability matrix — needed for the Brier score regardless of
    # target_type. ROC-AUC (binary only) needs just the positive-class column.
    proba_test = pipeline.predict_proba(X_test_records)
    y_prob_test = proba_test[:, 1] if config.target_type == "binary" else None

    average = config.metric_average  # "binary" or "macro" — stored per experiment
    f1_train = f1_score(y_train, y_pred_train, average=average)
    f1_test  = f1_score(y_test,  y_pred_test,  average=average)

    # overfit_delta = how much better the model scores on training data than test data.
    # A healthy model stays below ~0.05. Above 0.10 suggests the model is memorising
    # training rows and may not generalise to new machine data.
    overfit_delta = f1_train - f1_test
    if overfit_delta > 0.10:
        print(
            f"  WARNING: overfit_delta={overfit_delta:.3f} "
            f"(f1_train={f1_train:.3f}, f1_test={f1_test:.3f}) — "
            "consider tightening depth/regularisation params."
        )

    # scale_by_half=True forces both binary and multiclass into the same [0, 1]
    # range. Without it, sklearn's "auto" default leaves multiclass in [0, 2] —
    # every multiclass run would look twice as miscalibrated as a binary run
    # for no reason other than scale, making cross-experiment comparison in
    # MLflow misleading.
    brier_score = brier_score_loss(y_test, proba_test, scale_by_half=True)

    metrics: dict[str, float] = {
        "f1_train":       f1_train,
        "f1_test":        f1_test,
        "overfit_delta":  overfit_delta,   # logged to MLflow so you can sort/filter by it
        "precision_test": precision_score(y_test, y_pred_test, average=average),
        "recall_test":    recall_score(y_test, y_pred_test, average=average),
        "brier_score":    brier_score,     # lower = better-calibrated probabilities; 0 = perfect, 0.25 = "always guess 50%"
    }
    if y_prob_test is not None:
        metrics["roc_auc_test"] = roc_auc_score(y_test, y_prob_test)

    params: dict[str, object] = {
        # "estimator" holds the raw nested classifier object itself (not a
        # string) — CalibratedClassifierCV.get_params(deep=True) duplicates
        # every one of its hyperparameters under "estimator__*" already, so
        # dropping the bare object avoids logging a noisy repr() to MLflow.
        k: v for k, v in classifier.get_params().items() if k != "estimator"
    }
    params.update({
        "model_family": config.model_family,
        "target_type":  config.target_type,
        "test_size":    TEST_SIZE,
    })

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
        df:       Raw DataFrame from ai4i2020.parquet.
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
        X, y, random_state=RANDOM_STATE, test_size=TEST_SIZE, stratify=y
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
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

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

        fold_scores = []

        for train_idx, val_idx in cv.split(X_train_vec, y_train):
            X_fold_train = X_train_vec[train_idx]
            X_fold_val   = X_train_vec[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val   = y_train.iloc[val_idx]

            # X_val_for_pred tracks which array to predict on — SVM rescales
            # inside the fold, so its val array differs from the default.
            X_val_for_pred = X_fold_val

            if config.model_family == "lightgbm":
                # early_stopping_rounds: if the validation score does not improve
                # for this many consecutive trees, LightGBM stops early.
                # This prevents memorising training data within each trial.
                # n_estimators in params becomes the *maximum* — early stopping
                # may use far fewer.
                early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 20, 50)
                classifier = lgb.LGBMClassifier(**params, verbose=-1)
                classifier.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
                )
            elif config.model_family == "svm":
                # SVM is sensitive to feature scale — StandardScaler is fit on the
                # fold's training split only (never on val) to avoid leakage.
                # probability=False skips Platt scaling during CV (only needed for
                # predict_proba/ROC-AUC, which the objective doesn't compute).
                scaler = StandardScaler()
                X_fold_train = scaler.fit_transform(X_fold_train)
                X_val_for_pred = scaler.transform(X_fold_val)
                classifier = SVC(
                    **params,
                    class_weight="balanced",
                    probability=False,
                    random_state=42,
                )
                classifier.fit(X_fold_train, y_fold_train)
            elif config.model_family == "mlp":
                # MLP (Multi-Layer Perceptron) — a.k.a. a neural network.
                #
                # Like SVM, gradients are computed in feature-space, so unscaled
                # inputs cause wildly unequal weight updates and slow convergence.
                # Same fix: fit the scaler on training fold only.
                scaler = StandardScaler()
                X_fold_train = scaler.fit_transform(X_fold_train)
                X_val_for_pred = scaler.transform(X_fold_val)

                # hidden_layer_sizes was stored as a string ("128_64") to satisfy
                # Optuna's serialisation rules. Convert back to the tuple that
                # MLPClassifier expects before building the classifier.
                mlp_params = {**params}
                mlp_params["hidden_layer_sizes"] = tuple(
                    int(x) for x in mlp_params["hidden_layer_sizes"].split("_")
                )

                # early_stopping=True tells sklearn's MLP to hold out 10 % of the
                # fold's training rows as an internal validation set.  Training stops
                # when that internal score stops improving — the same idea as LightGBM's
                # early stopping, just built into the classifier rather than a callback.
                # This keeps individual trials from overfitting within the CV loop.
                classifier = MLPClassifier(
                    **mlp_params,
                    early_stopping=True,
                    random_state=42,
                )
                classifier.fit(X_fold_train, y_fold_train)
            else:
                # XGBoost early stopping uses a different API — passed via fit params.
                # We use a fixed 30-round patience for XGBoost trials.
                classifier = xgb.XGBClassifier(**params, verbosity=0)
                classifier.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False,
                )

            y_pred = classifier.predict(X_val_for_pred)
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
    with mlflow.start_run(
        run_name=config.experiment_name,
        description=(
            f"{config.model_family} · {config.target_type} · "
            f"registered as {config.registered_model_name}"
        ),
        tags={
            "model_family":    config.model_family,
            "target_type":     config.target_type,
            "target":          config.target,
            "experiment_name": config.experiment_name,
            "developer":       os.getenv("MLFLOW_TRACKING_USERNAME", "unknown"),
        },
    ):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=config.registered_model_name,
        )

    # ── Registry metadata ──────────────────────────────────────────────────────
    # log_model() registers the model and creates a new version. We use
    # MlflowClient to write description and tags onto both levels:
    #
    #   Registered model  — the family entry (predictive-maintenance-binary).
    #                       Description and tags appear on the Models landing page.
    #   Model version     — the specific version just created by this run.
    #                       Description and tags appear on the version detail page
    #                       (the screenshot with "Description: None").
    #
    # This must run OUTSIDE the `with mlflow.start_run()` block because
    # MlflowClient operates on the registry directly, not on the active run.
    client  = mlflow.MlflowClient()
    version = model_info.registered_model_version

    # Registered model (family-level) — written once; subsequent runs overwrite.
    client.update_registered_model(
        name=config.registered_model_name,
        description=(
            f"Predictive maintenance classifier — {config.target_type} target. "
            f"Trained on AI4I 2020 dataset. "
            f"All model families ({', '.join(['xgboost', 'lightgbm', 'random_forest', 'logreg', 'svm', 'mlp'])}) "
            f"compete for the @production alias on this registry entry."
        ),
    )
    client.set_registered_model_tag(config.registered_model_name, "model_family", config.model_family)
    client.set_registered_model_tag(config.registered_model_name, "target_type",  config.target_type)
    client.set_registered_model_tag(config.registered_model_name, "project",      "predictive-maintenance-capstone")

    # Model version (this specific run) — unique per version.
    client.update_model_version(
        name=config.registered_model_name,
        version=version,
        description=(
            f"Model family : {config.model_family}\n"
            f"Target       : {config.target} ({config.target_type})\n"
            f"F1 (test)    : {metrics.get('test_f1', 'n/a')}\n"
            f"Developer    : {os.getenv('MLFLOW_TRACKING_USERNAME', 'unknown')}"
        ),
    )
    client.set_model_version_tag(config.registered_model_name, version, "model_family", config.model_family)
    client.set_model_version_tag(config.registered_model_name, version, "developer",    os.getenv("MLFLOW_TRACKING_USERNAME", "unknown"))


# ── CML report ─────────────────────────────────────────────────────────────────
# Writes a lightweight markdown file consumed by the CML GitHub Action.
# The Action attaches it as a comment on the pull request for quick review.

def write_cml_metrics(metrics: dict) -> None:
    """Write key test metrics to metrics.txt for a CML pull-request comment.

    Includes f1, precision, recall, and brier_score on the test set, plus
    roc_auc (binary only). f1_train is intentionally omitted — reviewers need
    test performance, not evidence of fitting.

    Args:
        metrics: Dict produced by train_model(). Keys f1_test, precision_test,
                 recall_test, and brier_score are always present. roc_auc_test
                 is optional (binary experiments only).
    """
    lines = [
        "# Training Metrics",
        "",
        f"f1_test:        {metrics['f1_test']:.4f}",
        f"precision_test: {metrics['precision_test']:.4f}",
        f"recall_test:    {metrics['recall_test']:.4f}",
        f"brier_score:    {metrics['brier_score']:.4f}  (lower is better; 0 = perfect)",
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
    df = pd.read_parquet(DATA_PATH)          # Parquet loads ~10× faster than CSV at the same row count
    df = df.tail(TRAINING_WINDOW)            # keep only the most recent N rows; no-op until dataset exceeds 50k
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
        elif config.model_family == "svm":
            # probability=True re-enabled for the final model so predict_proba
            # is available for ROC-AUC logging. It was disabled during CV trials
            # for speed (Platt scaling adds significant overhead per fold).
            config.classifier_factory = lambda _: SVC(
                **classifier_params,
                class_weight="balanced",
                probability=True,
                random_state=42,
            )
        elif config.model_family == "mlp":
            # Convert hidden_layer_sizes string back to tuple (same as in tune_model).
            mlp_params = {**classifier_params}
            mlp_params["hidden_layer_sizes"] = tuple(
                int(x) for x in mlp_params["hidden_layer_sizes"].split("_")
            )
            # early_stopping=True kept for the final model — MLP has no n_estimators
            # cap like tree models, so without it the network trains until max_iter
            # regardless of whether the val score is still improving.
            config.classifier_factory = lambda _: MLPClassifier(
                **mlp_params,
                early_stopping=True,
                random_state=42,
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
