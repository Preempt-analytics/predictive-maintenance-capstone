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

import os
import sys
from contextlib import asynccontextmanager  # turns a generator function into a context manager (see lifespan below)
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException  # HTTPException lets you send HTTP error responses (404, 503, etc.)
from pydantic import BaseModel, Field       # Pydantic handles request/response validation automatically

# ── Shared feature engineering ─────────────────────────────────────────────────
# sys.path.insert ensures Python can find feature_transformation.py whether you
# run from the project root (`uvicorn src.api:app`) or from inside src/.
# Without this, the import would fail with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).parent))
from feature_transformation import FEATURES, FAILURE_TYPE_CLASSES, engineer_features  # noqa: E402

load_dotenv()  # reads MLFLOW_TRACKING_URI and credentials from the .env file


# ── Which model to load ────────────────────────────────────────────────────────
# os.getenv("MODEL_NAME", "predictive-maintenance-binary") means:
#   → use the MODEL_NAME environment variable if set
#   → otherwise fall back to "predictive-maintenance-binary"
#
# To switch to multiclass without changing code:
#   MODEL_NAME=predictive-maintenance-multiclass uvicorn src.api:app
MODEL_NAME = os.getenv("MODEL_NAME", "predictive-maintenance-binary")


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
        ..., ge=0, le=240,
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
    failure_probability: float = Field(description="Model confidence in the failure prediction.")
    failure_type: str | None = Field(                               # str | None means the field can be a string OR null
        default=None,                                               # null for binary models; "hdf" / "twf" / etc. for multiclass
        description="Predicted failure type (multiclass only). Null for binary models.",
    )
    model_name: str = Field(description="Registered model family that produced this prediction.")
    model_version: str | None = Field(
        default=None,
        description="Version number from the MLflow registry, if resolvable.",
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

    status: str = Field(description="'ok' if the model is loaded, 'degraded' if not.")
    model_loaded: bool
    model_name: str | None
    model_version: str | None = None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SHARED STATE (the loaded model lives here)
# ══════════════════════════════════════════════════════════════════════════════
#
# Problem: FastAPI calls your route functions fresh on every HTTP request.
# Local variables inside those functions don't persist between requests.
# You can't load the ML model inside /predict — that would add ~2 seconds to
# every single prediction as it downloads from MLflow each time.
#
# Solution: store the model in a module-level dict. Module-level variables
# persist for the lifetime of the running process. Every route function can
# read from app_state without reloading anything.
#
# Why a dict and not separate global variables?
# A dict is easier to clear cleanly on shutdown and easier to reset in tests.

app_state: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — STARTUP AND SHUTDOWN (the lifespan function)
# ══════════════════════════════════════════════════════════════════════════════
#
# FastAPI needs a hook to run code once when the server starts (load the model)
# and once when it stops (clean up). The lifespan pattern is how FastAPI 0.93+
# handles this — it replaces the older @app.on_event("startup") decorator.
#
# @asynccontextmanager turns this generator function into a context manager.
# A generator function is one that contains `yield`. Everything before yield
# runs at startup; everything after yield runs at shutdown. The server serves
# requests during the yield — think of yield as "now open for business."

@asynccontextmanager
async def lifespan(app: FastAPI):           # FastAPI passes itself in; we don't use it, but the signature is required

    # ── STARTUP: runs once before the first request ────────────────────────────
    uri = f"models:/{MODEL_NAME}@production"
    print(f"\n  Loading model: {uri} ...")

    try:
        # mlflow.sklearn.load_model downloads the fitted sklearn Pipeline from
        # DagsHub and deserialises it into memory. This is the slow step (~2s).
        # Doing it here means every prediction request is fast (<10ms).
        model = mlflow.sklearn.load_model(uri)

        # Resolve the actual version number behind the @production alias.
        # This is a separate network call, so it gets its own try/except —
        # a failure here shouldn't prevent the model from serving predictions.
        try:
            mv = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "production")
            resolved_version = mv.version  # e.g. "3"
        except Exception:
            resolved_version = None         # version stays null in responses — not critical

        # Write everything into app_state so route handlers can read it.
        app_state["model"]          = model
        app_state["model_name"]     = MODEL_NAME
        app_state["model_version"]  = resolved_version
        app_state["model_loaded"]   = True
        app_state["is_multiclass"]  = MODEL_NAME.endswith("-multiclass")  # used in run_prediction below
        print(f"  Model ready: {MODEL_NAME}@production (version {resolved_version})")

    except Exception as exc:
        # The server starts even if the model fails to load. This is intentional:
        # /health will report "degraded" so ops can investigate, and /predict will
        # return HTTP 503 with a clear message. The alternative — crashing at startup
        # — makes the service harder to debug because logs disappear immediately.
        app_state["model_loaded"] = False
        app_state["model_name"]   = MODEL_NAME
        app_state["error"]        = str(exc)
        print(f"  WARNING: Could not load model from '{uri}'.")
        print(f"  Fix: open the MLflow UI, find your best run, set alias 'production'.")
        print(f"  Original error: {exc}")

    yield  # ← the server is now live and handling requests; everything below runs on shutdown

    # ── SHUTDOWN: runs once after the last request ─────────────────────────────
    app_state.clear()
    print("\n  Server shutdown — model unloaded.")


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
        "Predicts machine failure from live sensor readings. "
        "The active model is controlled by the @production alias in MLflow — "
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


def run_prediction(model, reading: SensorReading, is_multiclass: bool = False) -> tuple[int, str | None, float]:
    """Run one sensor reading through feature engineering and the model.

    Centralising this logic means /predict and /predict/batch always produce
    identical results — they can't drift apart if inference logic changes.

    Returns: (predicted_failure, failure_type, failure_probability)
    """
    # Convert the Pydantic object → raw dict → single-row DataFrame → feature dict.
    # engineer_features() adds the three derived columns (power_kw, temp_diff, stress).
    raw         = reading_to_raw_dict(reading)
    df_features = engineer_features(pd.DataFrame([raw]))

    # to_dict(orient="records") produces [{col: val, col: val, ...}] — a list with
    # one dict per row. The sklearn DictVectorizer inside the pipeline expects exactly
    # this format. A plain dict or a numpy array would raise a type error.
    record = df_features[FEATURES].to_dict(orient="records")

    if is_multiclass:
        # Multiclass models output an integer class index (0–5).
        # FAILURE_TYPE_CLASSES maps that index back to the human-readable label:
        # e.g. 0 → "none", 1 → "hdf", 2 → "twf", etc.
        pred_int     = int(model.predict(record)[0])
        failure_type = FAILURE_TYPE_CLASSES[pred_int]               # e.g. "hdf"
        predicted    = 0 if failure_type == "none" else 1           # 0 = no failure, 1 = any failure
        proba_row    = model.predict_proba(record)[0]               # probabilities for all 6 classes
        failure_prob = float(proba_row[pred_int])                   # probability of the predicted class
    else:
        # Binary models output 0 or 1. predict_proba returns [P(0), P(1)].
        # Index [1] is P(failure) — that's the number we surface to the caller.
        predicted    = int(model.predict(record)[0])
        failure_prob = float(model.predict_proba(record)[0][1])     # [0] = first (only) row, [1] = P(failure)
        failure_type = None

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
# tags= groups endpoints in the /docs UI — "Operations" and "Predictions"
# become collapsible sections in the Swagger interface.
#
# async def — makes the handler non-blocking. While one request waits on I/O
# (e.g. a slow model call), the event loop serves other incoming requests.
# For CPU-bound work (like model.predict) the gain is small, but it's the
# FastAPI convention and costs nothing.

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health() -> HealthResponse:
    """Is the server running and ready to predict?

    Call this before sending predictions, or after a restart, to confirm
    the model loaded successfully. HTTP 200 with status='degraded' means the
    server is reachable but the model failed to load — check the server logs.
    """
    return HealthResponse(
        status="ok" if app_state.get("model_loaded") else "degraded",
        model_loaded=app_state.get("model_loaded", False),         # .get() returns False if key is missing (safer than direct access)
        model_name=app_state.get("model_name"),
        model_version=app_state.get("model_version"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(reading: SensorReading) -> PredictionResponse:
    """Predict whether a single machine is about to fail.

    FastAPI automatically parses the JSON request body into a SensorReading
    object and validates every field before this function runs. If any field
    is missing or out of range, the caller gets HTTP 422 — your code never runs.

    Use failure_probability (not just machine_failure) to set alert thresholds.
    A probability of 0.85 and 0.51 both return machine_failure=1, but one
    warrants immediate shutdown while the other warrants a follow-up inspection.
    """
    # Guard clause: if the model didn't load at startup, refuse to predict.
    # HTTPException stops execution immediately and sends an HTTP error response.
    # status_code=503 means "Service Unavailable" — the server is up but not ready.
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

    # FastAPI validates this return value against PredictionResponse before sending.
    return PredictionResponse(
        machine_failure=predicted,
        failure_probability=round(prob, 4),                        # 4 decimal places is precise enough for a probability
        failure_type=failure_type,
        model_name=app_state["model_name"],
        model_version=app_state.get("model_version"),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Predictions"])
async def predict_batch(batch: BatchRequest) -> BatchResponse:
    """Predict failure for many machines in one HTTP call.

    Prefer this over calling /predict in a loop — each HTTP round-trip adds
    network overhead. Send all buffered readings at once, get all predictions back.
    The predictions list is in the same order as the input readings list.
    """
    if not app_state.get("model_loaded"):
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /health.")

    model         = app_state["model"]
    is_multiclass = app_state.get("is_multiclass", False)

    predictions = []
    for reading in batch.readings:                                  # process each reading with the same logic as /predict
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
        # Generator expression: iterates predictions, picks machine_failure from each, sums the 1s.
        total_failures_predicted=sum(p.machine_failure for p in predictions),
    )


@app.post("/model/reload", tags=["Operations"])
async def reload_model() -> dict:
    """Hot-swap the production model without restarting the server.

    When you promote a new model version in the MLflow UI, call this endpoint
    and the very next prediction will use the updated model. No restart needed,
    no downtime, no in-flight requests interrupted.
    """
    uri = f"models:/{MODEL_NAME}@production"
    try:
        model = mlflow.sklearn.load_model(uri)

        try:
            mv      = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "production")
            version = mv.version
        except Exception:
            version = None

        # Overwrite app_state in place — all subsequent requests immediately
        # see the new model. The old model object is garbage-collected by Python.
        app_state["model"]         = model
        app_state["model_loaded"]  = True
        app_state["model_version"] = version
        app_state["is_multiclass"] = MODEL_NAME.endswith("-multiclass")

        return {"status": "reloaded", "model": MODEL_NAME, "version": version}

    except Exception as exc:
        # Do NOT clear app_state — keep the old model serving predictions
        # while the caller investigates what went wrong with the new version.
        app_state["error"] = str(exc)
        raise HTTPException(status_code=503, detail=f"Reload failed: {exc}")
