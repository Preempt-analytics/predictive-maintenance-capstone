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
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from typing import Callable, Optional

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
    experiment_name: str        # shown in the MLflow UI — think about namespacing
    registered_model_name: str  # name under which the model is versioned in the registry
    model_family: str           # human-readable label logged as a tag
    target: str                 # column name of what we are predicting
    target_type: str            # "binary" or "multiclass" — drives metric selection
    metric_average: str         # "binary" or "macro" — passed to f1_score, precision_score, etc.
    classifier_factory: Callable  # receives imbalance_ratio, returns a fitted-ready classifier
    test_size: float = 0.2                   # optional, defaults to 20% test split
    description: str = ""                    # optional, defaults to empty string
    notes: Optional[str] = None              # optional, explicitly nullable
    tags: dict = field(default_factory=dict) # optional, mutable default


# Q: How would you add a second experiment (e.g. multiclass or a different model
#    family) without changing any of the functions below?
#    What is the minimum edit required?

# TBD — adding external yaml config file similar to how .env works?

EXPERIMENTS: dict[str, ExperimentConfig] = {
    "xgb_binary": ExperimentConfig(
        experiment_name="predictive-maintenance/xgboost/binary",
        registered_model_name="xgboost-binary",
        model_family="xgboost",
        target="machine_failure",
        target_type="binary",
        metric_average="binary",
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
        experiment_name="predictive-maintenance/logreg/multiclass",
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
    "air_temperature_kelvin",
    "process_temperature_kelvin",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_minutes",
    "power_kw",
    "temp_diff_kelvin",
    "mechanical_stress"
]

# ── 3. Preprocessing ───────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
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
    df["power_kw"] = (df["torque_nm"] * df["rotational_speed_rpm"] * 2 * 3.14159 / 60) / 1000
    df["temp_diff_kelvin"] = df["process_temperature_kelvin"] - df["air_temperature_kelvin"]
    df["mechanical_stress"] = df["torque_nm"] * df["tool_wear_minutes"]

    return df[FEATURES + [config.target]]


# ── 4. Classifier registry ─────────────────────────────────────────────────────
#
# Q: The finished train_model() uses an if/else to pick the classifier:
#
#       if config.target_type == "binary":
#           classifier = xgb.XGBClassifier(scale_pos_weight=...)
#       else:
#           classifier = xgb.XGBClassifier(objective="multi:softprob", ...)
#
#    What happens when you add a third model family — say, Random Forest?
#    You'd add another if/else branch inside train_model(). And a fourth?
#    Another branch. At what point does that become unmaintainable?
#
#  A: As we add more model families and target types, the number of if/else branches 
#       in train_model() would grow exponentially, making the code difficult to read and maintain.
#       To avoid this, we can implement a classifier registry that maps model families and target types 
#       to their corresponding classifier configurations.
#
# Q: You already solved this problem for experiments: instead of branching,
#    you look up a config from a registry dict. Can you apply the same idea
#    to classifiers? What would the key be? What would the value be?
#
#   A: We can create a CLASSIFIER_REGISTRY dictionary where the key is a tuple of (model_family, target_type) 
#   and the value is a function that returns the appropriate classifier instance based on the imbalance ratio. 
#   This way, we can easily look up the correct classifier configuration without needing multiple 
#   if/else branches in the train_model() function. 
# 
# Q: scale_pos_weight is XGBoost-specific. RandomForest handles imbalance
#    with class_weight="balanced". If the classifier factory owns that logic,
#    what does train_model() no longer need to know about?
#
#   A: If the classifier factory (_build_classifier) handles the logic for setting scale_pos_weight 
#   for XGBoost and class_weight for RandomForest, then train_model() no longer needs to know about 
#   the specific parameters required for each classifier to handle class imbalance.
# 
# Q: The factory below receives imbalance_ratio as an argument.
#    Why pass it in rather than hardcoding it? Who computes it, and when?

#  A: Passing imbalance_ratio as an argument to the factory allows for greater flexibility and 
#  adaptability of the classifier configuration. It allows the factory to compute the appropriate parameters 
# for handling class imbalance based on the actual distribution of the training data, rather than relying 
# on a hardcoded value that may not be accurate for different datasets or experiments. 
# The imbalance_ratio would be computed from the training labels (y_train) after the train/test split, 
# ensuring that it reflects the true class distribution of the training data at the time of model training. 
# This approach makes the code more robust and adaptable to changes in the dataset or target variable distribution.

def _build_classifier(config: ExperimentConfig, imbalance_ratio: float):
    # Option B: the factory lives in the config — no registry lookup needed here.
    # Each EXPERIMENTS entry owns its classifier definition.
    #
    # Q: What error would you get if classifier_factory is accidentally left out
    #    of a new ExperimentConfig entry? Is that a better or worse failure mode
    #    than a KeyError from a registry lookup?
    #
    # A: If classifier_factory is accidentally left out of a new ExperimentConfig entry, 
    # you would get an AttributeError when trying to call the missing factory function. 
    # This is a worse failure mode than a KeyError from a registry lookup because it may not be 
    # immediately clear that the issue is due to a missing classifier_factory in the config, 
    # and it could lead to confusion for someone who is not familiar with the codebase. 
    # A KeyError from a registry lookup would more clearly indicate that there is an issue 
    # with the configuration or the registry, making it easier to debug and fix the problem.

    return config.classifier_factory(imbalance_ratio)


# ── 5. Model and training ──────────────────────────────────────────────────────
def train_model(df: pd.DataFrame, config: ExperimentConfig):
    """Preprocess, split, train, and evaluate.

    Returns (pipeline, metrics, params).

    Q: The dataset has a ~97:3 class split. What does a model predict
       if it learns to always say "no failure"? What accuracy would it get?
       Is accuracy the right metric at all?

    A: A model that learns to always say "no failure" would predict that there is no machine failure in all cases and as such have no utility.
         It would get an accuracy of approximately 97% because it would be correct in 97% of the cases (the majority class), but it would fail
         to identify any of the actual failures (the minority class) which could be extrenmely costly in a real factory setting. Therefore, accuracy
         is not a reliable metric in this case.

    Q: XGBClassifier has scale_pos_weight. What value should it take?
       How would you compute it from the training labels, not hardcode it?

       A: The scale_pos_weight parameter in XGBClassifier is used to handle class imbalance by assigning a weight to the positive class.
            The value of scale_pos_weight can be computed as the ratio of the number of negative samples to the number of positive samples in the training labels.
            This can be calculated using the following formula:

            scale_pos_weight = (number of negative samples) / (number of positive samples)

            By computing this value from the training labels, you can ensure that the model is appropriately weighted to handle the class imbalance without hardcoding a specific value.

    Q: Should you stratify the train/test split? What breaks if you don't,
       given the class imbalance?

       A: Yes, you should stratify the train/test split to ensure that the class distribution in both the training and testing sets is
       representative of the overall dataset. If you don't stratify, you might end up with a training set that has a very different
       class distribution than the testing set, which can lead to a model that performs well on the training data but poorly on the
       testing data.

    Q: Which costs more in a real factory?
        A false negative means a machine fails without warning.
        A false positive means an unnecessary maintenance stop.

        A: In a real factory setting, a false negative (a machine failing without warning) typically costs more than a false positive (an unnecessary maintenance stop).
           A false negative can lead to unplanned downtime, damage to equipment, and potential safety hazards, which can result in significant financial losses and operational disruptions.
           On the other hand, while a false positive may lead to unnecessary maintenance and associated costs, it is generally less severe than the consequences of a false negative.
           Therefore, in this context, minimizing false negatives is often more critical to ensure the safety and efficiency of factory operations.

       Q: How does your answer change which metric you optimise for?

       A: Just like in MSE, where we penalise large errors more than small ones, here we want to penalise false negatives more than false positives.
         Therefore, we might choose to optimize for a metric that takes into account the cost of false negatives, such as F1-score or a
         custom cost-sensitive metric, rather than just accuracy.

         Another metrics such as recall (sensitivity) would also be important to optimize for, as it measures the ability of the model to
         correctly identify positive cases (failures), which is crucial in this context. In summary, we would choose a metric that reflects
           the higher cost of false negatives in order to ensure that our model is more focused on correctly identifying machine failures.

    Q: Your pipeline needs at least a vectoriser and a classifier.
       DictVectorizer + XGBClassifier is one option.
       What would you use instead, and what would you gain or lose?

    A: Instead of using DictVectorizer + XGBClassifier, I could use a different combination such as CountVectorizer + RandomForestClassifier.
       - CountVectorizer would be more suitable if we were dealing with text data, as it converts a collection of text documents to a matrix of token counts.
         However, since our dataset consists of numerical and categorical features, CountVectorizer may not be the best choice for this scenario.

    Q: The `average` argument to f1_score, precision_score, recall_score must be
       "binary" for binary classification and "macro" (or "weighted") for multiclass.
       Rather than writing another if/else here, where in the config could you store
       that value so train_model() just reads it?

    Q: params should let someone reproduce this exact run. What belongs there?
       Hint: the classifier object itself knows its hyperparameters —
       look up sklearn's get_params() method. What does that give you?
    """
    df_processed = preprocess(df, config)
    y = df_processed[config.target]
    X = df_processed.drop(columns=[config.target])

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=config.test_size, stratify=y
    )

   
    # Only meaningful for binary classification — multiclass factories ignore it (lambda _)
    # Guarded here to avoid ZeroDivisionError when y_train contains string labels
    if config.target_type == "binary":
        imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    else:
        imbalance_ratio = 1.0
    classifier = _build_classifier(config, imbalance_ratio)

    X_train_records = X_train.to_dict(orient="records")
    X_test_records = X_test.to_dict(orient="records")

    # TODO: build the pipeline — DictVectorizer vectorises the record dicts,
    #       then hands the numeric matrix to the classifier from _build_classifier
    pipeline = make_pipeline(DictVectorizer(), classifier)

    pipeline.fit(X_train_records, y_train)

    y_pred_train = pipeline.predict(X_train_records)
    y_pred_test = pipeline.predict(X_test_records)

    # TODO: for binary experiments only, compute probability scores for ROC-AUC
    #       Hint: pipeline.predict_proba(X_test_records)[:, 1]
    #       How would you guard this so it only runs when config.target_type == "binary"?

    if config.target_type == "binary":
        y_prob_test = pipeline.predict_proba(X_test_records)[:, 1]
    else:
        y_prob_test = None


    average = config.metric_average

    metrics: dict[str, float] = {
        "f1_train":       f1_score(y_train, y_pred_train, average=average),
        "f1_test":        f1_score(y_test, y_pred_test, average=average),
        "precision_test": precision_score(y_test, y_pred_test, average=average),
        "recall_test":    recall_score(y_test, y_pred_test, average=average),
    }

    if y_prob_test is not None:
        metrics["roc_auc_test"] = roc_auc_score(y_test, y_prob_test)

   
    # TODO: populate params — use classifier.get_params() plus the config fields above
    params: dict[str, object] = {**classifier.get_params(), **{
        "model_family": config.model_family,
        "target_type": config.target_type,
        "test_size": config.test_size
    }}

    return pipeline, metrics, params


# ── 5. MLflow setup ────────────────────────────────────────────────────────────
#
# Q: The tracking URI, username, and password live in .env — not in code.
#    Why? What would happen if you committed them to the repo instead?
#
#A: Storing the tracking URI, username, and password in a .env file instead of hardcoding them in the code is a best practice for several reasons:
# 1. Security: Hardcoding sensitive information like usernames and passwords in the code can lead to security vulnerabilities, especially if the code is shared or stored in a public repository. If someone gains access to the code, they would also have access to the sensitive information.
# 2. Flexibility: Using a .env file allows for easier configuration changes without modifying the code. This is particularly useful when deploying the application in different environments (development, staging, production) where the tracking URI and credentials may differ.
# 3. Version Control: Committing sensitive information to a repository can lead to accidental exposure

# Q: configure_mlflow() below receives a config object instead of hardcoding
#    the experiment name. What does that buy you when you have multiple
#    experiments? What would the alternative look like?

# A: By receiving a config object instead of hardcoding the experiment name, configure_mlflow() 
# becomes more flexible and reusable across multiple experiments. If you hardcode the experiment name,
#  it would limit the function to only work for that specific experiment, and you would need to create 
# separate functions or add conditional logic to handle different experiments. With a config object,
#  you can easily switch between experiments by simply passing a different config, making the code 
# cleaner and more maintainable. The alternative would involve hardcoding the experiment name
#  within the function, which would require additional modifications to accommodate multiple experiments.
#
# Q: mlflow.set_experiment() creates the experiment if it does not exist.
#    Is that the behaviour you want in a team setting, or should new
#    experiments require an explicit approval step?
#
# A: In a large corporate setting or if you are resource constrained, it might be preferable to have new 
# experiments require an explicit approval step rather than automatically creating them with 
# mlflow.set_experiment(), but for our purposes and in a smaller team setting, the convenience of 
# automatically creating experiments outweigh the need for an approval step.
#

def configure_mlflow(config: ExperimentConfig) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise EnvironmentError("MLFLOW_TRACKING_URI is not set in the environment.")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)



# ── 6. MLflow logging ──────────────────────────────────────────────────────────
#
# Q: MLflow has three metadata slots: tags, params, metrics.
#    Before writing any code, sort your values into the right bucket:
#      - tags   → free-form labels that describe the run (who, what, why)
#      - params → inputs you chose before training (hyperparameters, feature list)
#      - metrics → numbers produced by training (scores, counts)
#    Why does the distinction matter? What breaks if you log a metric as a param?
#
# A: The distinction between tags, params, and metrics in MLflow is important for organizing and interpreting the information
#   associated with a run. Tags are free-form labels that provide context about the run, such as who ran it, what it was for, 
#   and why it was done. Params are the inputs and configurations that were chosen before training, such as hyperparameters 
#   and feature lists. Metrics are the numerical results produced by the training process, such as accuracy scores or loss values.
#   If you log a metric as a param, it can lead to confusion and misinterpretation of the data. Metrics are meant to be tracked over time 
#   and compared across runs, while params are static inputs that define the conditions of the run. 
#   Logging a metric as a param would make it difficult to analyze the performance of different runs and could lead to incorrect 
#   conclusions about which configurations are more effective.
#
# Q: What would a teammate need to see in the MLflow UI to understand this run
#    three months from now, without reading the code?
#
# A: To understand this run three months from now without reading the code, a teammate would need to see:
#    - The experiment name
#    - The model family
#    - The target type
#    - The target variable
#    - The developer who ran the experiment
#    - The hyperparameters used for the model
#    - The evaluation metrics achieved by the model
#    - Any relevant tags that provide additional context about the run (e.g., "initial experiment", "tuned hyperparameters", etc.)
#
# Q: registered_model_name links a run to a versioned model in the registry.
#    What does "registering" a model give you that a plain logged artifact does not?
#
# A: Registering a model in the MLflow registry provides several benefits compared to just logging it as an artifact:
#    - Versioning: You can track different versions of the same model.
#    - Lifecycle Management: You can manage the model through different stages (staging, production, etc.).
#    - Deployment Integration: The registry integrates with deployment tools, making it easier to deploy models.
#    - Collaboration: Team members can easily access and use registered models.

#
# Q: log_model() receives the config so it can read registered_model_name.
#    If you hardcoded that name here instead, what would break when you add
#    a second experiment config?

def log_model(
    pipeline, metrics: dict, params: dict, config: ExperimentConfig
) -> None:
    with mlflow.start_run():
        mlflow.set_tags({
            "model_family": config.model_family,
            "target_type": config.target_type,
            "target": config.target,
            "experiment_name": config.experiment_name,
            "developer": os.getenv("MLFLOW_TRACKING_USERNAME", "unknown")

        })

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            pipeline, artifact_path="model",
            registered_model_name=config.registered_model_name
        )



# ── 7. CML report ──────────────────────────────────────────────────────────────
def write_cml_metrics(metrics: dict) -> None:
    """Write a markdown summary to metrics.txt for a CML PR report.

    Q: Which metrics from your dict does a reviewer need to see
       to decide whether this model is better than the previous one?
       Which ones are noise at the PR review stage?

    A: A reviewer would likely need to see the test metrics (e.g., f1_test, precision_test, recall_test, 
    and roc_auc_test if applicable) to evaluate the performance of the model on unseen data.
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
#
# Q: The --experiment flag selects a config from EXPERIMENTS by key.
#    What happens today if someone passes an unknown key?
#    How would you give them a helpful error message?
 
# A: If someone passes an unknown key to the --experiment flag, Click will automatically raise a BadParameter error 
# with a message indicating that the value is not valid and listing the valid choices. This is because we have defined 
# the type of the --experiment option as click.Choice(list(EXPERIMENTS)), which restricts the input to the keys of the 
# EXPERIMENTS dictionary and provides built-in validation and error handling for invalid inputs.
#
# Q: What other CLI flags might be useful as your experiment suite grows?
#    (Think: --n-estimators, --test-size, --dry-run…)
#    At what point does a config file become a better choice than CLI flags?
#
#   A: As the experiment suite grows, additional CLI flags that could be useful include:
#   - --n-estimators: to specify the number of trees in ensemble models like Random Forest or XGBoost.
#   - --test-size: to allow users to specify the proportion of the dataset to be used as the test set.
#   - --dry-run: to perform a trial run without actually training the model, which can be useful for testing the setup and configuration.
#   - --tags: to allow users to add custom tags to the MLflow run for better organization and filtering in the UI.
#   - --params: to allow users to specify additional hyperparameters for the model in a flexible way.

#   When we have larger teams potentially including non-technical stakeholders who need to run experiments since a yaml
#   or JSON config file can be more user-friendly and easier to manage than a long list of CLI flags, especially as the 
#   number of configurable options increases.
#   It is also safer to have the configuration in a seperate file to avoid unintentional edits to the production code.

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

    # FIXED: was train_model(df) — missing config meant the function had no access
    # to config.target, config.test_size, config.metric_average, or the classifier
    # factory. It would have raised an error at runtime.
    pipeline, metrics, params = train_model(df, config)
    log_model(pipeline, metrics, params, config)

    if cml_run:
        write_cml_metrics(metrics)


if __name__ == "__main__":
    main()
