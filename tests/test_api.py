# tests/test_api.py
#
# ── Why this test exists ────────────────────────────────────────────────────
# /predict's response shape is a contract with real clients (PredictionResponse
# in api.py). A change to run_prediction() or the route handler could break
# that shape without anyone noticing until a client's parser fails. This test
# checks the contract directly, without needing a live MLflow connection or a
# real model — see "Skipping the real model" below for how that's done.

import pytest
from fastapi.testclient import TestClient

import api
from feature_transformation import FAILURE_TYPE_CLASSES

# ── Skipping the real model ─────────────────────────────────────────────────
# api.lifespan() only loads models from MLflow when the app actually starts,
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
    """Fake binary model — always predicts no failure (0)."""

    def predict(self, record):
        return [0]                     # "no failure"

    def predict_proba(self, record):
        return [[0.9, 0.1]]             # [P(no failure), P(failure)]


class FakeFailureBinaryModel:
    """Fake binary model — always predicts failure (1).

    Used in tests that need the multiclass gate to open — failure_type
    is only populated when the binary model returns 1.
    """

    def predict(self, record):
        return [1]                     # "failure predicted"

    def predict_proba(self, record):
        return [[0.1, 0.9]]             # [P(no failure), P(failure)]


class FakeMulticlassModel:
    """Fake multiclass model — always predicts class index 2."""

    def predict(self, record):
        return [2]                     # index into FAILURE_TYPE_CLASSES

    def predict_proba(self, record):
        return [[0.05, 0.05, 0.8, 0.05, 0.05]]


@pytest.fixture
def client():
    return TestClient(api.app)         # no "with" — lifespan/MLflow never runs


def test_predict_binary_no_failure(client):
    # When binary predicts 0, failure_type must be null regardless of multiclass.
    api.app_state.clear()
    api.app_state.update({
        "binary_model":   FakeBinaryModel(),
        "binary_version": "17",
        "binary_f1":      0.91,
        "binary_loaded":  True,
        "multiclass_model":   FakeMulticlassModel(),
        "multiclass_version": "18",
        "multiclass_loaded":  True,
    })

    response = client.post("/predict", json=VALID_READING)

    assert response.status_code == 200
    body = response.json()
    assert body["machine_failure"] == 0
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["failure_type"] is None          # gate closed — multiclass never called
    assert body["model_name"] == "predictive-maintenance-binary"


def test_predict_failure_populates_failure_type(client):
    # When binary predicts failure (1), multiclass should identify the type.
    api.app_state.clear()
    api.app_state.update({
        "binary_model":   FakeFailureBinaryModel(),
        "binary_version": "17",
        "binary_f1":      0.91,
        "binary_loaded":  True,
        "multiclass_model":   FakeMulticlassModel(),
        "multiclass_version": "18",
        "multiclass_loaded":  True,
    })

    response = client.post("/predict", json=VALID_READING)

    assert response.status_code == 200
    body = response.json()
    assert body["machine_failure"] == 1
    assert body["failure_type"] == FAILURE_TYPE_CLASSES[2]   # multiclass predicted index 2


def test_predict_failure_without_multiclass(client):
    # If multiclass is not loaded, failure_type stays null even when binary predicts failure.
    api.app_state.clear()
    api.app_state.update({
        "binary_model":   FakeFailureBinaryModel(),
        "binary_version": "17",
        "binary_f1":      0.91,
        "binary_loaded":  True,
        "multiclass_loaded": False,     # multiclass not available
    })

    response = client.post("/predict", json=VALID_READING)

    assert response.status_code == 200
    body = response.json()
    assert body["machine_failure"] == 1
    assert body["failure_type"] is None          # graceful degradation


def test_predict_returns_503_when_binary_not_loaded(client):
    # Without the binary model (the gate), the API cannot predict at all.
    api.app_state.clear()
    api.app_state["binary_loaded"] = False

    response = client.post("/predict", json=VALID_READING)

    assert response.status_code == 503
