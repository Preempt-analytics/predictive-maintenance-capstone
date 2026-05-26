"""
Feature Definitions and Engineering
=====================================
Single source of truth for column names and sensor-derived features.

Both the training pipeline (modeling_pipeline.py) and the inference layer
(sensor_simulator.py) import from here. Keeping transforms in one place
prevents training-serving skew — the most common silent failure in deployed
ML systems, where the model sees different feature values at inference time
than it was trained on.

Rule: any change to a feature formula here must be deployed to both
the training run and the serving layer at the same time.
"""

import pandas as pd

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

# Five raw sensor readings + three domain-derived features.
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


# ── Feature engineering ────────────────────────────────────────────────────────
# Single source of truth for all input transforms.
# Imported by both modeling_pipeline.py (training) and sensor_simulator.py (inference).
# Any change here must be deployed to both at the same time.

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and compute the three domain-derived features.

    This is the contract between training and inference. The model was
    trained on the output of this function — inference must pass through
    the same function or predictions will be wrong in ways that are hard
    to detect (training-serving skew).

    Domain features (each justified by EDA):
    - power_kw:          torque × rpm → kW. Failures cluster at power extremes.
    - temp_diff_kelvin:  process − air temperature. HDF risk rises when diff < 8.6 K.
    - mechanical_stress: torque × tool wear. High torque on a worn tool is a compound hazard.

    Args:
        df: Raw DataFrame with original sensor column names (from CSV or live feed).

    Returns:
        DataFrame with renamed columns and three added feature columns.
        Does not slice to FEATURES or add target columns — caller handles that.
    """
    df = df.copy().rename(columns=COLUMN_RENAME)

    df["power_kw"] = (df["torque_nm"] * df["rotational_speed_rpm"] * 2 * 3.14159 / 60) / 1000
    df["temp_diff_kelvin"] = df["process_temperature_kelvin"] - df["air_temperature_kelvin"]
    df["mechanical_stress"] = df["torque_nm"] * df["tool_wear_minutes"]

    return df
