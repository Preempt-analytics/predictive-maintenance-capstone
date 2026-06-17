# tests/test_api.py
#
# ── Why this test exists ────────────────────────────────────────────────────
# /predict's response shape is a contract with real clients (PredictionResponse
# in api.py). A change to run_prediction() or the route handler could break
# that shape without anyone noticing until a client's parser fails. This test
# checks the contract directly, without needing a live MLflow connection or a
# real model — see "Skipping the real model" below for how that's done.

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
from feature_transformation import FAILURE_TYPE_CLASSES

# ── Skipping the real model ─────────────────────────────────────────────────
# api.lifespan() only loads a model from MLflow when the app actually starts,
# which happens if TestClient is used as "with TestClient(app) as client:".
# Using it WITHOUT the "with" block means lifespan never runs — app_state
# stays whatever we set it to by hand. That keeps this test fast, offline,
# and independent of MLflow credentials or network access.

VALID_READING = {
    "machine_type":                "M",
    "air_temperature_kelvin":      300.0,
    "process_temperature_kelvin":  310.0,
    "rotational_speed_rpm":        1500,
    "torque_nm":                   40.0,
    "tool_wear_minutes":           100,
}


class FakeBinaryModel:
    """Stands in for a loaded MLflow sklearn Pipeline — binary target."""

    def predict(self, record):
        return [0]                     # "no failure"

    def predict_proba(self, record):
        return [[0.9, 0.1]]             # [P(no failure), P(failure)]


class FakeMulticlassModel:
    """Stands in for a loaded MLflow sklearn Pipeline — multiclass target."""

    def predict(self, record):
        return [2]                     # index into FAILURE_TYPE_CLASSES

    def predict_proba(self, record):
        # One probability per class in FAILURE_TYPE_CLASSES, summing to 1.
        return [[0.05, 0.05, 0.8, 0.05, 0.05]]


@pytest.fixture
def client():
    return TestClient(api.app)         # no "with" — lifespan/MLflow never runs


def test_predict_binary_returns_expected_shape(client):
    api.app_state.clear()
    api.app_state.update({
        "model":          FakeBinaryModel(),
        "model_name":     "predictive-maintenance-binary",
        "model_version":  "7",
        "model_f1_score": 0.91,
        "model_loaded":   True,
        "is_multiclass":  False,
    })

    response = client.post("/predict", json=VALID_READING)

    assert response.status_code == 200
    body = response.json()
    assert body["machine_failure"] in (0, 1)
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["failure_type"] is None          # null for binary models
    assert body["model_name"] == "predictive-maintenance-binary"


def test_predict_multiclass_decodes_failure_type(client):
    api.app_state.clear()
    api.app_state.update({
        "model":          FakeMulticlassModel(),
        "model_name":     "predictive-maintenance-multiclass",
        "model_version":  "3",
        "model_f1_score": 0.65,
        "model_loaded":   True,
        "is_multiclass":  True,
    })

    response = client.post("/predict", json=VALID_READING)

    assert response.status_code == 200
    body = response.json()
    assert body["failure_type"] == FAILURE_TYPE_CLASSES[2]   # "osf"


def test_predict_returns_503_when_model_not_loaded(client):
    api.app_state.clear()
    api.app_state["model_loaded"] = False
    api.app_state["model_name"]   = "predictive-maintenance-binary"

    response = client.post("/predict", json=VALID_READING)

    assert response.status_code == 503
