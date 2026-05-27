"""
FastAPI Serving Layer — Predictive Maintenance
===============================================
This module is the inference half of the two-loop architecture.

Two-loop recap
--------------
  Inference loop  — client → POST /predict → this API → Production model → JSON response
  Retraining loop — simulation.db → export_simulation_to_csv.py → dvc repro → new model

The API's only job is to answer: "Given these sensor readings, will this machine fail?"
It does not train, it does not store, it does not decide which model to use — MLflow's
@production alias handles that. Promoting a new model version in the MLflow UI is enough
to change what this server responds with on the next request.

Why FastAPI specifically?
-------------------------
Three reasons make FastAPI a natural fit for ML serving:

  1. Pydantic validation — request bodies are validated and typed automatically.
     Bad inputs (wrong units, missing fields) fail at the boundary, not silently
     inside the model where they're hard to diagnose.

  2. Auto-generated OpenAPI docs — visit /docs in a browser to get an interactive
     playground for your endpoints. No extra work required.

  3. Async-native — the server stays responsive during I/O (model loading, future
     database writes) without blocking other requests.

How to run
----------
  From the project root:

    uvicorn src.api:app --reload

  The --reload flag restarts the server automatically when you edit api.py.
  Remove it in production. Default port is 8000.

  Open http://127.0.0.1:8000/docs for the interactive Swagger UI.

Prerequisites
-------------
  1. A model tagged @production in the MLflow registry (same requirement as
     the simulator — if the simulator runs, the API will load).
  2. MLFLOW_TRACKING_URI set in .env (already done from training setup).
  3. pip install fastapi uvicorn  (already in requirements.txt).
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Import shared feature engineering ─────────────────────────────────────────
# Both training (modeling_pipeline.py) and inference share feature_transformation.py.
# Inserting src/ into sys.path here makes the import work whether you launch
# with `uvicorn src.api:app` from the project root or `python api.py` from src/.
# This is the same pattern the simulator uses — one source of truth for features.
sys.path.insert(0, str(Path(__file__).parent))
from feature_transformation import FEATURES, FAILURE_TYPE_CLASSES, engineer_features  # noqa: E402

load_dotenv()


# ── Environment ────────────────────────────────────────────────────────────────
# MODEL_NAME selects which prediction task the server handles.
# Override with an environment variable to switch between binary and multiclass:
#   MODEL_NAME=predictive-maintenance-multiclass uvicorn src.api:app
#
# Which model family is @production is decided in the MLflow UI — not here.
# Promoting a new family (e.g. LightGBM over XGBoost) = move the alias in the UI,
# then call POST /model/reload. Zero code or config changes required.

import os
MODEL_NAME = os.getenv("MODEL_NAME", "predictive-maintenance-binary")


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST AND RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════
#
# Pydantic models serve two purposes at once:
#   - They document your API contract (visible in /docs automatically).
#   - They validate inputs before they touch the model — bad values raise HTTP 422
#     with a clear error message rather than a silent wrong prediction.
#
# Field(...) marks a field as required (no default). Field(ge=0) enforces
# a constraint. The `description` string appears in the /docs UI.

class SensorReading(BaseModel):
    """One set of raw sensor readings sent by a client for prediction.

    Field names use snake_case (clean API contract). The internal helper
    reading_to_raw_dict() maps them back to the original CSV column names
    that engineer_features() expects — that translation stays hidden from callers.
    """
    machine_type: str = Field(
        ...,
        pattern="^[LMH]$",
        description="Machine variant: L (light), M (medium), or H (heavy).",
        examples=["M"],
    )
    air_temperature_kelvin: float = Field(
        ..., gt=270, lt=320,
        description="Ambient air temperature in Kelvin. Typical range: 295–305 K.",
        examples=[300.0],
    )
    process_temperature_kelvin: float = Field(
        ..., gt=270, lt=320,
        description="Process temperature in Kelvin. Usually ~10 K above air temperature.",
        examples=[310.0],
    )
    rotational_speed_rpm: float = Field(
        ..., gt=0, lt=3000,
        description="Spindle rotational speed in RPM. Typical range: 1200–1800.",
        examples=[1538],
    )
    torque_nm: float = Field(
        ..., ge=0, lt=100,
        description="Applied torque in Newton-metres. Typical range: 20–60 Nm.",
        examples=[40.0],
    )
    tool_wear_minutes: float = Field(
        ..., ge=0, le=240,
        description="Cumulative tool wear in minutes. Resets to 0 after tool replacement.",
        examples=[108],
    )


class PredictionResponse(BaseModel):
    """Prediction result returned for a single sensor reading.

    machine_failure mirrors the binary target used in training:
      0 = no failure predicted
      1 = failure predicted

    failure_type is null for binary models. For multiclass models it
    will be the predicted failure type string: hdf, twf, pwf, osf, rnf, or none.
    """
    machine_failure: int = Field(description="0 = normal, 1 = failure predicted.")
    failure_probability: float = Field(description="Model confidence in the failure prediction.")
    failure_type: str | None = Field(
        default=None,
        description="Predicted failure type (multiclass only). Null for binary models.",
    )
    model_name: str = Field(description="Registered model family that produced this prediction.")
    model_version: str | None = Field(
        default=None,
        description="Version number from the MLflow registry, if resolvable.",
    )


class BatchRequest(BaseModel):
    """Multiple sensor readings submitted in a single HTTP call.

    Use this endpoint when you have many readings buffered — it avoids
    the overhead of one HTTP round-trip per reading.
    """
    readings: list[SensorReading] = Field(
        ...,
        min_length=1,
        description="List of sensor readings. Must contain at least one reading.",
    )


class BatchResponse(BaseModel):
    """Predictions for every reading in a batch request, in the same order."""
    predictions: list[PredictionResponse]
    total_readings: int = Field(description="Number of readings processed.")
    total_failures_predicted: int = Field(description="Count of readings where machine_failure == 1.")


class HealthResponse(BaseModel):
    """API health status — check this before sending predictions."""
    status: str = Field(description="'ok' if the model is loaded, 'degraded' if not.")
    model_loaded: bool
    model_name: str | None
    model_version: str | None = None


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION STATE
# ══════════════════════════════════════════════════════════════════════════════
#
# FastAPI is stateless by default — each request handler is a fresh function
# call. Shared state (the loaded model) lives here in a module-level dict.
# The lifespan function below writes to it at startup; request handlers read
# from it on every call.
#
# Why a dict rather than individual globals?
#   A dict makes it easy to pass state around in tests and to clear it cleanly
#   on shutdown. Global variables are harder to reset between test runs.

app_state: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN — STARTUP AND SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════
#
# The lifespan context manager replaces the older @app.on_event("startup")
# pattern (deprecated in FastAPI 0.93+). Everything before `yield` runs once
# when the server starts; everything after `yield` runs once on shutdown.
#
# Loading the model here — not inside the /predict handler — means the
# expensive MLflow network call happens once, not on every request.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ────────────────────────────────────────────────────────────────
    uri = f"models:/{MODEL_NAME}@production"
    print(f"\n  Loading model: {uri} ...")

    try:
        model = mlflow.sklearn.load_model(uri)

        try:
            mv = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "production")
            resolved_version = mv.version
        except Exception:
            resolved_version = None

        app_state["model"]          = model
        app_state["model_name"]     = MODEL_NAME
        app_state["model_version"]  = resolved_version
        app_state["model_loaded"]   = True
        app_state["is_multiclass"]  = MODEL_NAME.endswith("-multiclass")
        print(f"  Model ready: {MODEL_NAME}@production (version {resolved_version})")

    except Exception as exc:
        # Intentional: the server starts in a degraded state rather than
        # refusing to start at all. /health reports the issue; /predict returns 503.
        # This lets ops teams investigate without having to restart the process.
        app_state["model_loaded"] = False
        app_state["model_name"]   = MODEL_NAME
        app_state["error"]        = str(exc)
        print(f"  WARNING: Could not load model from '{uri}'.")
        print(f"  Fix: open the MLflow UI, find your best run, set alias 'production'.")
        print(f"  Original error: {exc}")

    yield  # ← server is live; request handlers run during this pause

    # ── SHUTDOWN ───────────────────────────────────────────────────────────────
    app_state.clear()
    print("\n  Server shutdown — model unloaded.")


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Preempt Analytics — Predictive Maintenance API",
    description=(
        "Predicts machine failure from live sensor readings. "
        "The active model is controlled by the @production alias in MLflow — "
        "no code change required to promote a new version."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def reading_to_raw_dict(reading: SensorReading) -> dict:
    """Map the clean API field names back to the original CSV column names.

    engineer_features() was written to handle the original CSV format
    (column names with spaces and brackets like "Air temperature [K]").
    It has to stay that way — the training pipeline uses it too.

    This function is the translation layer. Keeping it separate means the
    API contract (snake_case) and the feature engineering contract (CSV names)
    can each evolve independently.
    """
    return {
        "Type":                    reading.machine_type,
        "Air temperature [K]":     reading.air_temperature_kelvin,
        "Process temperature [K]": reading.process_temperature_kelvin,
        "Rotational speed [rpm]":  reading.rotational_speed_rpm,
        "Torque [Nm]":             reading.torque_nm,
        "Tool wear [min]":         reading.tool_wear_minutes,
    }


def run_prediction(model, reading: SensorReading, is_multiclass: bool = False) -> tuple[int, str | None, float]:
    """Apply feature engineering and return a normalised prediction tuple.

    Keeping inference logic here (not inside the route handlers) means
    both /predict and /predict/batch go through exactly the same transform
    and model call. One change here updates both endpoints simultaneously.

    Args:
        model:          Fitted sklearn Pipeline from the MLflow registry.
        reading:        Validated Pydantic SensorReading from the request body.
        is_multiclass:  True when the loaded model targets failure_type (6 classes).

    Returns:
        (predicted_failure, failure_type, failure_probability)
        predicted_failure:   0 or 1
        failure_type:        None for binary; human-readable label for multiclass
        failure_probability: model confidence — probability of the predicted class
    """
    raw         = reading_to_raw_dict(reading)
    df_features = engineer_features(pd.DataFrame([raw]))
    record      = df_features[FEATURES].to_dict(orient="records")

    if is_multiclass:
        # Multiclass models predict an integer label (0–5). Decode it back to
        # the human-readable string using the same mapping used during training.
        pred_int     = int(model.predict(record)[0])
        failure_type = FAILURE_TYPE_CLASSES[pred_int]
        predicted    = 0 if failure_type == "none" else 1
        proba_row    = model.predict_proba(record)[0]
        failure_prob = float(proba_row[pred_int])
    else:
        predicted    = int(model.predict(record)[0])
        failure_prob = float(model.predict_proba(record)[0][1])
        failure_type = None

    return predicted, failure_type, failure_prob


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health() -> HealthResponse:
    """Return whether the API is running and a Production model is loaded.

    Clients should call this before sending predictions during startup
    or after a service interruption. A 200 response with status='degraded'
    means the API is reachable but the model failed to load.
    """
    return HealthResponse(
        status="ok" if app_state.get("model_loaded") else "degraded",
        model_loaded=app_state.get("model_loaded", False),
        model_name=app_state.get("model_name"),
        model_version=app_state.get("model_version"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(reading: SensorReading) -> PredictionResponse:
    """Predict whether a single machine will fail, given its current sensor readings.

    The body should match the sensor values at a single point in time for one
    machine. The model was trained on the AI4I 2020 dataset; values far outside
    the training distribution will produce unreliable predictions.

    Returns a failure probability alongside the binary prediction.
    Use the probability (not just the 0/1 flag) to set alert thresholds —
    a probability of 0.85 warrants a different response than 0.51.
    """
    if not app_state.get("model_loaded"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. Check /health for details. "
                "The server may still be starting up or the MLflow registry "
                "may not have a model tagged @production."
            ),
        )

    model                         = app_state["model"]
    is_multiclass                 = app_state.get("is_multiclass", False)
    predicted, failure_type, prob = run_prediction(model, reading, is_multiclass)

    return PredictionResponse(
        machine_failure=predicted,
        failure_probability=round(prob, 4),
        failure_type=failure_type,
        model_name=app_state["model_name"],
        model_version=app_state.get("model_version"),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Predictions"])
async def predict_batch(batch: BatchRequest) -> BatchResponse:
    """Predict failure for multiple sensor readings in a single request.

    Prefer this endpoint over calling /predict repeatedly when you have
    buffered readings — it avoids one HTTP round-trip per reading.
    """
    if not app_state.get("model_loaded"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. Check /health for details. "
                "The server may still be starting up or the MLflow registry "
                "may not have a model tagged @production."
            ),
        )

    model         = app_state["model"]
    is_multiclass = app_state.get("is_multiclass", False)

    predictions = []
    for reading in batch.readings:
        predicted, failure_type, prob = run_prediction(model, reading, is_multiclass)
        predictions.append(PredictionResponse(
            machine_failure=predicted,
            failure_probability=round(prob, 4),
            failure_type=failure_type,
            model_name=app_state["model_name"],
            model_version=app_state.get("model_version"),
        ))

    return BatchResponse(
        predictions=predictions,
        total_readings=len(batch.readings),
        total_failures_predicted=sum(p.machine_failure for p in predictions),
    )


@app.post("/model/reload", tags=["Operations"])
async def reload_model() -> dict:
    """Reload the @production model from MLflow without restarting the server.

    Useful after promoting a new model version in the MLflow UI — call this
    endpoint and the next prediction will use the updated model.
    """
    uri = f"models:/{MODEL_NAME}@production"
    try:
        model = mlflow.sklearn.load_model(uri)
        try:
            mv = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "production")
            version = mv.version
        except Exception:
            version = None
        app_state["model"]         = model
        app_state["model_loaded"]  = True
        app_state["model_version"] = version
        app_state["is_multiclass"] = MODEL_NAME.endswith("-multiclass")
        return {"status": "reloaded", "model": MODEL_NAME, "version": version}
    except Exception as exc:
        # Keep the old model in place so predictions can still be served.
        # The caller can check /health to see the stale version number.
        app_state["error"] = str(exc)
        raise HTTPException(status_code=503, detail=f"Reload failed: {exc}")
