# tests/test_calibration.py
#
# ── Why this test exists ────────────────────────────────────────────────────
# Calibration was added because predict_proba from tree models tends to be
# overconfident — a number that looks like a probability without behaving
# like one. This test confirms the wrapping actually happened (the pipeline's
# final step is CalibratedClassifierCV, not the raw classifier) and that
# brier_score is computed and bounded in [0, 1] for every model family, not
# just the two in @production — a regression here would silently turn every
# failure_probability the API returns back into an uncalibrated guess.

import pytest
from sklearn.calibration import CalibratedClassifierCV

from modeling_pipeline import EXPERIMENTS, train_model
from test_modeling_pipeline_smoke import _synthetic_ai4i_dataframe


@pytest.mark.parametrize("experiment_name", list(EXPERIMENTS.keys()))
def test_pipeline_is_calibrated_and_reports_brier_score(experiment_name):
    config = EXPERIMENTS[experiment_name]
    df     = _synthetic_ai4i_dataframe()

    pipeline, metrics, _ = train_model(df, config)

    final_step = pipeline.steps[-1][1]
    assert isinstance(final_step, CalibratedClassifierCV), (
        f"{experiment_name}: final pipeline step is {type(final_step).__name__}, "
        "expected CalibratedClassifierCV — predict_proba would be uncalibrated."
    )

    assert "brier_score" in metrics
    # scale_by_half=True forces both binary and multiclass into [0, 1] — a
    # value outside that range means the scaling flag was dropped somewhere.
    assert 0.0 <= metrics["brier_score"] <= 1.0
