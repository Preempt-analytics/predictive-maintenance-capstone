"""
FastAPI Serving Layer — Predictive Maintenance
===============================================
This module is the inference half of the two-loop architecture.

Two-loop recap
--------------
  Inference loop  — client → POST /predict → this API → both @production models → JSON response
  Retraining loop — simulation.db → export_simulation_to_parquet.py → dvc repro → new model

The API loads two models at startup:
  predictive-maintenance-binary      — answers "will this machine fail?" (0/1 + probability)
  predictive-maintenance-multiclass  — answers "which failure type?" (TWF / HDF / PWF / OSF / RNF)

They work as a gate-and-detail pair:
  Binary model fires first. If it predicts failure, the multiclass model identifies the type.
  If binary predicts no failure, the multiclass model is never called and failure_type is null.

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

    # to specify a different port
    
    python -m uvicorn src.api:app --reload --port 8001 

  The --reload flag restarts the server automatically when you edit api.py.
  Remove it in production. Default port is 8000.

  Open http://127.0.0.1:8000/docs for the interactive Swagger UI.

Prerequisites
-------------
  1. Both models tagged @production in the MLflow registry.
  2. MLFLOW_TRACKING_URI set in .env (already done from training setup).
  3. pip install fastapi uvicorn  (already in requirements.txt).
"""

import os
import sys
from contextlib import asynccontextmanager  # turns a generator function into a context manager (see lifespan below)
from pathlib import Path

import mlflow
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException  # HTTPException lets you send HTTP error responses (404, 503, etc.)
from pydantic import BaseModel, Field       # Pydantic handles request/response validation automatically

# ── XGBoost / scikit-learn 1.8 compatibility patch ────────────────────────────
# xgboost==2.0.3 predates sklearn 1.8's tag-based estimator typing.
# sklearn.base.is_classifier(XGBClassifier()) incorrectly returns False, so
# CalibratedClassifierCV's predict_proba raises "Got a regressor" when loading
# a calibrated XGBoost model. Patching the CLASS (not an instance) fixes every
# deserialized XGBClassifier — cloning inside CalibratedClassifierCV would not
# preserve an instance-level patch. Remove once xgboost >= 2.1 is in use.
_original_xgb_tags = xgb.XGBClassifier.__sklearn_tags__
def _patched_xgb_tags(self):
    tags = _original_xgb_tags(self)
    tags.estimator_type = "classifier"
    return tags
xgb.XGBClassifier.__sklearn_tags__ = _patched_xgb_tags

# ── Shared feature engineering ─────────────────────────────────────────────────
# sys.path.insert ensures Python can find feature_transformation.py whether you
# run from the project root (`uvicorn src.api:app`) or from inside src/.
# Without this, the import would fail with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).parent))
from feature_transformation import FEATURES, FAILURE_TYPE_CLASSES, engineer_features  # noqa: E402

load_dotenv()  # reads MLFLOW_TRACKING_URI and credentials from the .env file


# ── Model registry names (integration contract) ────────────────────────────────
# These names are fixed contracts shared with modeling_pipeline.py and the MLflow
# registry. Changing either name here requires the same change in modeling_pipeline.py
# and in the registry itself — see CLAUDE.md Contract 2.
BINARY_MODEL     = "predictive-maintenance-binary"
MULTICLASS_MODEL = "predictive-maintenance-multiclass"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DEFINE WHAT GOES IN AND WHAT COMES OUT (Pydantic schemas)
# ══════════════════════════════════════════════════════════════════════════════
#
# In FastAPI, you describe request and response shapes using Pydantic classes
# that inherit from BaseModel. FastAPI reads these classes and does three things
# automatically:
#   1. Validates every incoming request against the schema — wrong type or
#      missing field → HTTP 422 error with a clear message, before your code runs.
#   2. Serialises every outgoing response to JSON matching the schema.
#   3. Generates the interactive /docs page (no extra work needed).
#
# Think of BaseModel as a typed contract: "this endpoint accepts exactly these
# fields, with exactly these types and constraints."

class SensorReading(BaseModel):
    """The JSON body a client must send to /predict or /predict/batch."""

    # Field(...) — the three dots mean "this field is required; there is no default."
    # Named constraints: gt = greater than, lt = less than, ge = greater or equal, le = less or equal.
    # The description= string appears word-for-word in the /docs UI.
    # The examples= value pre-fills the /docs "Try it out" form.

    machine_type: str = Field(
        ...,
        pattern="^[LMH]$",                                         # regex: only the letters L, M, or H are valid
        description="Machine variant: L (light), M (medium), or H (heavy).",
        examples=["M"],
    )
    air_temperature_kelvin: float = Field(
        ..., gt=270, lt=320,                                        # rejects physically impossible values at the boundary
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
        ..., ge=0, le=260,                                            # training data max is 253; 260 gives a small safety margin
        description="Cumulative tool wear in minutes. Resets to 0 after tool replacement.",
        examples=[108],
    )


class PredictionResponse(BaseModel):
    """The JSON object the API sends back after every prediction."""

    # response_model= in the route decorator (below) tells FastAPI to use this
    # class to validate and serialise the return value. If your route function
    # accidentally returns an extra field, FastAPI strips it. If it's missing a
    # required field, FastAPI raises an error — catching bugs before the client sees them.

    machine_failure: int = Field(description="0 = normal, 1 = failure predicted.")
    failure_probability: float = Field(description="Model confidence in the failure prediction (binary model).")
    failure_type: str | None = Field(                               # str | None means the field can be a string OR null
        default=None,                                               # null when no failure is predicted; populated when binary predicts 1
        description="Predicted failure type from the multiclass model. Null when machine_failure=0.",
    )
    model_name: str = Field(description="Binary model name — the primary gate for this prediction.")
    model_version: str | None = Field(
        default=None,
        description="Binary model version from the MLflow registry.",
    )


class BatchRequest(BaseModel):
    """Body for /predict/batch — a list of readings submitted in one HTTP call."""

    readings: list[SensorReading] = Field(                         # list[SensorReading] = every item must pass SensorReading validation
        ...,
        min_length=1,                                              # rejects empty lists before your code runs
        description="List of sensor readings. Must contain at least one reading.",
    )


class BatchResponse(BaseModel):
    """What /predict/batch sends back: one prediction per input reading, plus totals."""

    predictions: list[PredictionResponse]
    total_readings: int = Field(description="Number of readings processed.")
    total_failures_predicted: int = Field(description="Count of readings where machine_failure == 1.")


class HealthResponse(BaseModel):
    """What /health sends back — tells callers whether the server is ready to predict."""

    # Binary model fields (primary — the API cannot serve predictions without this)
    status: str = Field(description="'ok' if binary model is loaded, 'degraded' if not.")
    model_loaded: bool
    model_name: str | None
    model_version: str | None = None
    model_f1_score: float | None = None

    # Multiclass model fields (secondary — failure_type will be null if this is not loaded)
    multiclass_loaded: bool = False
    multiclass_version: str | None = None
    multiclass_f1_score: float | None = None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SHARED STATE (the loaded models live here)
# ══════════════════════════════════════════════════════════════════════════════
#
# Problem: FastAPI calls your route functions fresh on every HTTP request.
# Local variables inside those functions don't persist between requests.
# You can't load the ML model inside /predict — that would add ~2 seconds to
# every single prediction as it downloads from MLflow each time.
#
# Solution: store both models in a module-level dict. Module-level variables
# persist for the lifetime of the running process. Every route function can
# read from app_state without reloading anything.

app_state: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — STARTUP AND SHUTDOWN (the lifespan function)
# ══════════════════════════════════════════════════════════════════════════════
#
# FastAPI needs a hook to run code once when the server starts (load both models)
# and once when it stops (clean up). The lifespan pattern is how FastAPI 0.93+
# handles this — it replaces the older @app.on_event("startup") decorator.
#
# @asynccontextmanager turns this generator function into a context manager.
# Everything before yield runs at startup; everything after yield runs at
# shutdown. The server serves requests during the yield.

def _load_one_model(uri: str, model_name: str, alias: str) -> tuple:
    """Download one @production model from MLflow and return (model, version, f1).

    Separated from lifespan() so reload can call the same logic without
    duplicating the download-and-resolve pattern.

    Returns (model, version, f1) on success, raises on failure.
    """
    model = mlflow.sklearn.load_model(uri)                         # slow step: downloads artifact from DagsHub
    try:
        client  = mlflow.MlflowClient()
        mv      = client.get_model_version_by_alias(model_name, alias)
        version = mv.version
        f1      = client.get_run(mv.run_id).data.metrics.get("f1_test")
    except Exception:
        version = None                                             # not critical — version stays null in responses
        f1      = None
    return model, version, f1


@asynccontextmanager
async def lifespan(app: FastAPI):           # FastAPI passes itself in; we don't use it, but the signature is required

    # ── STARTUP: load both models before the first request ────────────────────
    # Binary model is the gate — the API cannot serve predictions without it.
    # Multiclass model provides failure type detail — useful but not fatal if missing.
    # Both are attempted unconditionally so a multiclass failure doesn't block binary.

    binary_uri     = f"models:/{BINARY_MODEL}@production"
    multiclass_uri = f"models:/{MULTICLASS_MODEL}@production"

    print(f"\n  Loading model: {binary_uri} ...")
    try:
        model, version, f1 = _load_one_model(binary_uri, BINARY_MODEL, "production")
        app_state["binary_model"]   = model
        app_state["binary_version"] = version
        app_state["binary_f1"]      = f1
        app_state["binary_loaded"]  = True
        print(f"  Model ready: {BINARY_MODEL}@production (version {version})")
    except Exception as exc:
        app_state["binary_loaded"] = False
        app_state["binary_error"]  = str(exc)
        print(f"  WARNING: Could not load model from '{binary_uri}'.")
        print(f"  Fix: open the MLflow UI, find your best run, set alias 'production'.")
        print(f"  Original error: {exc}")

    print(f"\n  Loading model: {multiclass_uri} ...")
    try:
        model, version, f1 = _load_one_model(multiclass_uri, MULTICLASS_MODEL, "production")
        app_state["multiclass_model"]   = model
        app_state["multiclass_version"] = version
        app_state["multiclass_f1"]      = f1
        app_state["multiclass_loaded"]  = True
        print(f"  Model ready: {MULTICLASS_MODEL}@production (version {version})")
    except Exception as exc:
        app_state["multiclass_loaded"] = False
        app_state["multiclass_error"]  = str(exc)
        print(f"  WARNING: Could not load multiclass model — failure_type will be null.")
        print(f"  Original error: {exc}")

    yield  # ← the server is now live and handling requests; everything below runs on shutdown

    # ── SHUTDOWN: runs once after the last request ─────────────────────────────
    app_state.clear()
    print("\n  Server shutdown — models unloaded.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CREATE THE APP
# ══════════════════════════════════════════════════════════════════════════════
#
# FastAPI() creates the application object. `app` is the name uvicorn expects
# when you run `uvicorn src.api:app` — the part after the colon is this variable.
# lifespan= wires the startup/shutdown function defined above into the app.

app = FastAPI(
    title="Preempt Analytics — Predictive Maintenance API",
    description=(
        "Predicts machine failure and failure type from live sensor readings. "
        "Binary model answers 'will it fail?'; multiclass model answers 'which type?' "
        "Both are controlled by the @production alias in MLflow — "
        "no code change required to promote a new version."
    ),
    version="0.1.0",
    lifespan=lifespan,      # connect the startup/shutdown hook
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — INTERNAL HELPERS (not exposed as endpoints)
# ══════════════════════════════════════════════════════════════════════════════

def reading_to_raw_dict(reading: SensorReading) -> dict:
    """Translate clean API field names → original CSV column names.

    engineer_features() expects column names exactly as they appear in the
    original AI4I CSV (with spaces and brackets). The API uses snake_case.
    This function bridges the two naming conventions so neither contract
    has to change to accommodate the other.
    """
    return {
        "Type":                    reading.machine_type,
        "Air temperature [K]":     reading.air_temperature_kelvin,
        "Process temperature [K]": reading.process_temperature_kelvin,
        "Rotational speed [rpm]":  reading.rotational_speed_rpm,
        "Torque [Nm]":             reading.torque_nm,
        "Tool wear [min]":         reading.tool_wear_minutes,
    }


def run_prediction(
    binary_model,
    multiclass_model,
    reading: SensorReading,
) -> tuple[int, str | None, float]:
    """Run one sensor reading through the gate-and-detail prediction pair.

    The binary model fires first and acts as the gate:
      - If it predicts no failure (0), the multiclass model is never called.
        failure_type is null. One model inference per non-failure reading.
      - If it predicts failure (1), the multiclass model identifies the type.
        failure_type is populated with the most likely failure mode.

    This separation is intentional: the binary model is optimised for the
    fail/no-fail decision; the multiclass model is optimised for type attribution
    given that a failure is occurring. Running multiclass on every reading would
    ignore that specialisation.

    Returns: (predicted_failure, failure_type, failure_probability)
    """
    # Convert the Pydantic object → raw dict → single-row DataFrame → feature dict.
    # engineer_features() adds the three derived columns (power_kw, temp_diff, stress).
    raw    = reading_to_raw_dict(reading)
    df_eng = engineer_features(pd.DataFrame([raw]))

    # to_dict(orient="records") produces [{col: val, ...}] — a list with one dict
    # per row. The sklearn DictVectorizer inside the pipeline expects this format.
    record = df_eng[FEATURES].to_dict(orient="records")

    # Gate: binary model — fail (1) or no failure (0)
    predicted    = int(binary_model.predict(record)[0])
    failure_prob = float(binary_model.predict_proba(record)[0][1])  # P(failure)

    # Detail: multiclass model only runs when the gate opens
    failure_type = None
    if predicted == 1 and multiclass_model is not None:
        pred_int     = int(multiclass_model.predict(record)[0])
        failure_type = FAILURE_TYPE_CLASSES[pred_int]
        if failure_type == "none":
            failure_type = None  # multiclass uncertain — binary result stands, type unknown

    return predicted, failure_type, failure_prob


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — ROUTES (the endpoints callers actually hit)
# ══════════════════════════════════════════════════════════════════════════════
#
# How route decorators work:
#   @app.get("/health")  → this function runs when a client sends GET /health
#   @app.post("/predict") → this function runs when a client sends POST /predict
#
# response_model= does two things:
#   1. Validates the return value against the schema (catches bugs server-side).
#   2. Strips any extra fields so callers only see what the schema defines.
#
# async def — makes the handler non-blocking. While one request waits on I/O
# (e.g. a slow model call), the event loop serves other incoming requests.

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health() -> HealthResponse:
    """Is the server running and ready to predict?

    Call this before sending predictions, or after a restart, to confirm
    both models loaded successfully. status='degraded' means the binary model
    failed to load — predictions will return 503. If only multiclass failed,
    status is still 'ok' but failure_type will always be null in responses.
    """
    return HealthResponse(
        status           = "ok" if app_state.get("binary_loaded") else "degraded",
        model_loaded     = app_state.get("binary_loaded", False),
        model_name       = BINARY_MODEL,
        model_version    = app_state.get("binary_version"),
        model_f1_score   = app_state.get("binary_f1"),
        multiclass_loaded  = app_state.get("multiclass_loaded", False),
        multiclass_version = app_state.get("multiclass_version"),
        multiclass_f1_score = app_state.get("multiclass_f1"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(reading: SensorReading) -> PredictionResponse:
    """Predict whether a single machine is about to fail, and if so, which failure type.

    FastAPI automatically parses the JSON request body into a SensorReading
    object and validates every field before this function runs. If any field
    is missing or out of range, the caller gets HTTP 422 — your code never runs.

    The binary model fires first. If it predicts failure, the multiclass model
    identifies the failure type (TWF / HDF / PWF / OSF / RNF). If it predicts
    no failure, failure_type is null and the multiclass model is not called.

    Use failure_probability (not just machine_failure) to set alert thresholds.
    A probability of 0.85 and 0.51 both return machine_failure=1, but one
    warrants immediate shutdown while the other warrants a follow-up inspection.
    """
    # Guard clause: the binary model is required — without it we cannot predict.
    if not app_state.get("binary_loaded"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Binary model is not loaded. Check /health for details. "
                "The server may still be starting up or the MLflow registry "
                "may not have a model tagged @production."
            ),
        )

    binary_model     = app_state["binary_model"]
    multiclass_model = app_state.get("multiclass_model")  # None if multiclass failed to load

    predicted, failure_type, prob = run_prediction(binary_model, multiclass_model, reading)

    return PredictionResponse(
        machine_failure     = predicted,
        failure_probability = round(prob, 4),                       # 4 decimal places is precise enough for a probability
        failure_type        = failure_type,
        model_name          = BINARY_MODEL,
        model_version       = app_state.get("binary_version"),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Predictions"])
async def predict_batch(batch: BatchRequest) -> BatchResponse:
    """Predict failure for many machines in one HTTP call.

    Prefer this over calling /predict in a loop — each HTTP round-trip adds
    network overhead. Send all buffered readings at once, get all predictions back.
    The predictions list is in the same order as the input readings list.
    """
    if not app_state.get("binary_loaded"):
        raise HTTPException(status_code=503, detail="Binary model is not loaded. Check /health.")

    binary_model     = app_state["binary_model"]
    multiclass_model = app_state.get("multiclass_model")

    predictions = []
    for reading in batch.readings:                                  # process each reading with the same logic as /predict
        predicted, failure_type, prob = run_prediction(binary_model, multiclass_model, reading)
        predictions.append(PredictionResponse(
            machine_failure     = predicted,
            failure_probability = round(prob, 4),
            failure_type        = failure_type,
            model_name          = BINARY_MODEL,
            model_version       = app_state.get("binary_version"),
        ))

    return BatchResponse(
        predictions              = predictions,
        total_readings           = len(batch.readings),
        total_failures_predicted = sum(p.machine_failure for p in predictions),
    )


@app.post("/model/reload", tags=["Operations"])
async def reload_model() -> dict:
    """Hot-swap both production models without restarting the server.

    When you promote a new model version in the MLflow UI, call this endpoint
    and the very next prediction will use the updated models. No restart needed,
    no downtime, no in-flight requests interrupted.

    Each model is reloaded independently. If the new binary model fails to load,
    the old binary model keeps serving — predictions continue uninterrupted.
    If the new multiclass model fails, the old multiclass model is retained.
    """
    result = {}

    # ── Reload binary (primary gate) ──────────────────────────────────────────
    binary_uri = f"models:/{BINARY_MODEL}@production"
    try:
        model, version, f1 = _load_one_model(binary_uri, BINARY_MODEL, "production")
        app_state["binary_model"]   = model    # overwrite in place — subsequent requests immediately see the new model
        app_state["binary_loaded"]  = True
        app_state["binary_version"] = version
        app_state["binary_f1"]      = f1
        result["binary"] = {"status": "reloaded", "version": version}
    except Exception as exc:
        app_state["binary_error"] = str(exc)   # keep old model serving — do not clear binary_model
        result["binary"] = {"status": "failed", "error": str(exc)}

    # ── Reload multiclass (detail layer) ─────────────────────────────────────
    multiclass_uri = f"models:/{MULTICLASS_MODEL}@production"
    try:
        model, version, f1 = _load_one_model(multiclass_uri, MULTICLASS_MODEL, "production")
        app_state["multiclass_model"]   = model
        app_state["multiclass_loaded"]  = True
        app_state["multiclass_version"] = version
        app_state["multiclass_f1"]      = f1
        result["multiclass"] = {"status": "reloaded", "version": version}
    except Exception as exc:
        app_state["multiclass_error"] = str(exc)
        result["multiclass"] = {"status": "failed", "error": str(exc)}

    # Surface an HTTP error only if the binary model (the gate) failed.
    # A multiclass failure is logged in the result but does not block the response.
    if result["binary"]["status"] == "failed":
        raise HTTPException(
            status_code=503,
            detail=f"Binary model reload failed: {result['binary']['error']}",
        )

    return result
