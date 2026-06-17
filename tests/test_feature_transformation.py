# tests/test_feature_transformation.py
#
# ── Why this test exists ────────────────────────────────────────────────────
# FEATURES (the list the model is trained on) and engineer_features() (the
# function that builds those columns) live in the same file but are not
# checked against each other anywhere. If a column gets renamed or a feature
# formula gets removed without updating FEATURES, training and serving would
# silently disagree — Contract 1 in CLAUDE.md, the highest-risk contract in
# this project. This test catches that the moment it happens, in under a
# second, instead of waiting for a confusing prediction-quality bug later.

import pandas as pd
from feature_transformation import FEATURES, engineer_features


def test_engineer_features_produces_every_declared_feature():
    # One realistic raw row, using the exact original AI4I column names
    # (Contract 5) — the same names export_simulation_to_parquet.py writes
    # and modeling_pipeline.py reads from data/ai4i2020.parquet.
    raw_row = pd.DataFrame([{
        "Type":                     "M",
        "Air temperature [K]":      300.0,
        "Process temperature [K]":  310.0,
        "Rotational speed [rpm]":   1500,
        "Torque [Nm]":              40.0,
        "Tool wear [min]":          100,
    }])

    engineered = engineer_features(raw_row)

    missing = [col for col in FEATURES if col not in engineered.columns]
    assert not missing, (
        f"FEATURES expects {missing}, but engineer_features() did not produce "
        "them. Training and serving would now compute different inputs."
    )
