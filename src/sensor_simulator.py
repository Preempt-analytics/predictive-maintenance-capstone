"""
Sensor Simulator — Predictive Maintenance
==========================================
Generates synthetic sensor readings that mimic the AI4I 2020 dataset,
sends each reading to the FastAPI serving layer for prediction, and stores
everything — raw readings, engineered features, and predictions — in a
local SQLite database.

Architecture (updated — simulator is now an API client)
--------------------------------------------------------
Previously the simulator loaded the ML model from MLflow itself and called
engineer_features() for inference. That meant two components both contained
inference logic. Now the responsibilities are clearly separated:

  THIS FILE (simulator)         API.PY (serving layer)
  ─────────────────────         ──────────────────────
  Generate sensor readings  →   Receive the reading via POST /predict
  Inject failures               Apply engineer_features()
  Track tool wear               Run model.predict()
  Send reading to API       ←   Return prediction + probability
  Store result in SQLite
  Print status line

In a real factory this maps to:

  Factory sensor             →  Serving layer (this API pattern)
  (generates readings)          (loads model, runs inference)

The simulator is the "factory sensor" half. The API is the "model" half.
HTTP is the interface between them — the same interface any real sensor
would use.

Why feature_transformation.py is still imported here
------------------------------------------------------
The SQLite database stores engineered features (power_kw, temp_diff_kelvin,
mechanical_stress) alongside raw sensor values, so that drift detection
can query them directly without recomputing. The API computes these
features internally for inference but does not return them in the response.

So the simulator computes them once — ONLY for writing to the database,
not for prediction. This is a deliberate split:

  Feature engineering for INFERENCE → api.py owns it
  Feature engineering for STORAGE   → simulator computes locally

If you ever add a new engineered feature to feature_transformation.py,
you need to update both api.py (for inference) AND this file (for storage).
That coupling is documented in CLAUDE.md Contract 1.

How to run
----------
  Step 1 — start the API (in a separate terminal):
    uvicorn src.api:app --reload

  Step 2 — start the simulator:
    python src/sensor_simulator.py --n-readings 500 --mode sudden-spike

  The simulator calls GET /health before starting the loop. If the API
  is not reachable, or if no @production model is loaded, it exits with
  a clear fix message — no ambiguous errors mid-simulation.

Simulation modes
----------------
  normal        — stable 3.4% failure rate (matches training distribution)
  gradual-drift — rate climbs from 3.4% to 25% (models equipment ageing)
  sudden-spike  — normal for first half, 40% for second half (best for demos)

Prerequisites
-------------
  1. API running with a @production model loaded.
     Start it: uvicorn src.api:app --reload
  2. No .env changes needed — the simulator no longer talks to MLflow directly.
"""

import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import httpx       # modern Python HTTP client; like `requests` but with better
                   # timeout and async support — standard choice in FastAPI projects
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# feature_transformation is imported ONLY to compute engineered features for
# storage in simulation.db (so drift detection can query them). It is NOT
# used for inference — that responsibility now belongs to api.py entirely.
# See "Why feature_transformation.py is still imported here" in the docstring.
from feature_transformation import engineer_features

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# SENSOR DISTRIBUTION CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
#
# These values come from running .describe() on data/ai4i2020.csv.
# Sampling from the same distributions as the training data keeps the
# simulator's readings in a range the model was actually trained on.
# Random numbers drawn far outside this range would produce meaningless predictions.

MACHINE_TYPES        = ["L", "M", "H"]
MACHINE_TYPE_WEIGHTS = [0.60, 0.30, 0.10]   # proportions from the original dataset

# Temperature constants — fitted from ai4i2020.csv describe()
# Process temperature is NOT sampled independently from air temperature.
# In the real dataset they correlate at 0.876 — both reflect the same factory
# thermal environment.  Sampling them independently inflates process_temp
# variance by 51% (simulator std 2.24K vs training 1.48K), causing false drift
# alerts when comparing simulation to training data.  A bivariate normal
# reproduces both marginal distributions and the correct joint covariance.
AIR_TEMP_MEAN     = 300.0       # mean air temperature (K)
PROCESS_TEMP_MEAN = 310.0       # mean process temperature (K)
TEMP_COV_MATRIX   = [           # 2×2 covariance: [[var_air, cov], [cov, var_proc]]
    [4.000, 2.601],             # var_air = 2.0² = 4.0;  cov = ρ×σ_air×σ_proc = 0.876×2.0×1.484 ≈ 2.601
    [2.601, 2.202],             # cov = 2.601;  var_proc = 1.484² ≈ 2.202
]

ROTATIONAL_SPEED_RPM = (1538.0, 179.0)
TORQUE_NM            = (39.9,   9.97)

TOOL_WEAR_MAX_MINUTES  = 253   # maximum wear before tool replacement (training data max; was 240)
TOOL_WEAR_STEP_MINUTES = 2     # wear added per reading, per machine

DEFAULT_N_MACHINES = 5         # spread readings across multiple machines in parallel


# ══════════════════════════════════════════════════════════════════════════════
# FAILURE INJECTION CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
#
# When the simulator decides to inject a failure, it shifts sensor values
# toward the failure zones identified in EDA — it doesn't force the label.
# The model must detect the failure from the shifted physics, just as a
# real model would. Three of the five AI4I failure modes are covered:
#
#   HDF (Heat Dissipation): shrink temp gap below 8.6 K
#   PWF (Power Failure):    low rpm + high torque → power out of safe range
#   OSF (Overstrain):       high torque × high wear → compound hazard
#   TWF (Tool Wear):        handled naturally by the wear lifecycle
#   RNF (Random):           not injected — by definition has no sensor signature

FAILURE_TORQUE_ADD_NM      = 18.0
FAILURE_RPM_SHIFT          = -350.0
FAILURE_TEMP_OFFSET_KELVIN = 6.0    # narrower gap than the normal ~10 K

BASE_FAILURE_RATE       = 0.034     # 3.4% — matches the training dataset failure rate
GRADUAL_DRIFT_PEAK_RATE = 0.25
SUDDEN_SPIKE_RATE       = 0.40

DB_PATH = Path("simulation.db")     # local SQLite file; gitignored


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE — init and write
# ══════════════════════════════════════════════════════════════════════════════

def init_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure the table exists.

    CREATE TABLE IF NOT EXISTS means this is safe to call on a database that
    already has data — it will not overwrite or truncate anything. Each
    simulation run appends new rows to the same table.

    Args:
        db_path: Path to the .db file. Created automatically if it doesn't exist.

    Returns:
        An open sqlite3.Connection ready for INSERT statements.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp                   TEXT    NOT NULL,
            reading_number              INTEGER NOT NULL,
            machine_id                  TEXT    NOT NULL,

            -- Raw sensor values — what the machine physically reported
            machine_type                TEXT    NOT NULL,
            air_temperature_kelvin      REAL    NOT NULL,
            process_temperature_kelvin  REAL    NOT NULL,
            rotational_speed_rpm        REAL    NOT NULL,
            torque_nm                   REAL    NOT NULL,
            tool_wear_minutes           REAL    NOT NULL,

            -- Engineered features — stored here so drift detection can query
            -- them without rerunning engineer_features() at query time
            power_kw                    REAL    NOT NULL,
            temp_diff_kelvin            REAL    NOT NULL,
            mechanical_stress           REAL    NOT NULL,

            -- Prediction returned by the API
            predicted_failure           INTEGER NOT NULL,   -- 0 = normal, 1 = failure
            predicted_failure_type      TEXT,               -- NULL for binary; "hdf" etc. for multiclass
            failure_probability         REAL    NOT NULL,

            -- Ground truth from the simulator (did we inject a failure?)
            injected_failure            INTEGER NOT NULL,

            -- Metadata about this simulation run
            mode                        TEXT    NOT NULL,
            target                      TEXT    NOT NULL,   -- "binary" or "multiclass", derived from API response
            effective_failure_rate      REAL    NOT NULL
        )
    """)
    conn.commit()
    return conn


def store_reading(conn: sqlite3.Connection, row: dict) -> None:
    """Insert one complete reading (sensors + engineered features + prediction) into SQLite.

    Named placeholders (:column_name) match keys in the row dict exactly.
    SQLite substitutes values safely — no SQL injection risk from sensor data.

    Args:
        conn: Open connection from init_db().
        row:  Dict whose keys match the column names above exactly.
    """
    conn.execute("""
        INSERT INTO sensor_readings (
            timestamp, reading_number, machine_id,
            machine_type, air_temperature_kelvin, process_temperature_kelvin,
            rotational_speed_rpm, torque_nm, tool_wear_minutes,
            power_kw, temp_diff_kelvin, mechanical_stress,
            predicted_failure, predicted_failure_type, failure_probability,
            injected_failure, mode, target, effective_failure_rate
        ) VALUES (
            :timestamp, :reading_number, :machine_id,
            :machine_type, :air_temperature_kelvin, :process_temperature_kelvin,
            :rotational_speed_rpm, :torque_nm, :tool_wear_minutes,
            :power_kw, :temp_diff_kelvin, :mechanical_stress,
            :predicted_failure, :predicted_failure_type, :failure_probability,
            :injected_failure, :mode, :target, :effective_failure_rate
        )
    """, row)
    conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# API CLIENT — health check and prediction call
# ══════════════════════════════════════════════════════════════════════════════
#
# These two functions are the only place in this file that talks to the network.
# Everything else is local: generating readings, computing features, writing to SQLite.

def check_api_health(api_url: str) -> None:
    """Confirm the API is reachable and a production model is loaded.

    Why check before starting the simulation loop?
    The loop runs for hundreds or thousands of readings. If the API is
    down, you want to know immediately — not after waiting through 500
    failed requests. This function fails fast so you can fix the problem
    and restart cleanly.

    Args:
        api_url: Base URL of the running FastAPI server, e.g. "http://127.0.0.1:8000".

    Raises:
        SystemExit: If the API is unreachable or the model is not loaded.
    """
    try:
        # timeout=5.0 means "give up after 5 seconds". Without a timeout,
        # httpx waits indefinitely if the server is slow or unresponsive.
        # A 5-second wait is generous for a server running on localhost.
        response = httpx.get(f"{api_url}/health", timeout=5.0)

        # raise_for_status() checks the HTTP status code and raises an
        # exception if it's 4xx (client error) or 5xx (server error).
        # Without this call, a 404 or 503 response would be silently accepted
        # and the next line would try to parse an error page as JSON.
        response.raise_for_status()

        data = response.json()  # {"status": "ok", "model_loaded": true, "model_name": "...", ...}

    except httpx.ConnectError:
        # ConnectError means the server isn't running at all (connection refused).
        print(f"\nERROR: Cannot connect to the API at {api_url}.")
        print("  The API must be running before the simulator starts.")
        print("  Start it with:  uvicorn src.api:app --reload")
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: API health check failed: {exc}")
        sys.exit(1)

    if not data.get("model_loaded"):
        # The server is up but no model was loaded at startup (startup failed).
        print(f"\nERROR: API is running but no @production model is loaded.")
        print("  Fix: open the MLflow UI → Models → set the @production alias.")
        print("  Then call POST /model/reload or restart the API.")
        sys.exit(1)

    # Confirm which model version is serving — useful for demo output
    print(f"  API ready   : {data.get('model_name')} v{data.get('model_version', '?')}")
    print(f"  API status  : {data.get('status')}")


def call_predict_api(api_url: str, raw: dict) -> tuple[int, str | None, float]:
    """Send one sensor reading to POST /predict and return the prediction.

    What this function does step by step:
      1. Translate raw CSV column names → snake_case field names the API expects.
         (The API's SensorReading schema uses snake_case; our raw dict uses CSV names.)
      2. POST the payload as JSON to the /predict endpoint.
      3. Parse the JSON response into Python values.
      4. Return a three-value tuple so the caller doesn't need to know the
         response format — it just gets (predicted, type, probability).

    Args:
        api_url: Base URL of the running FastAPI server.
        raw:     Dict with original CSV column names from generate_raw_reading().

    Returns:
        (predicted_failure, failure_type, failure_probability)
          predicted_failure:   0 = no failure, 1 = failure predicted
          failure_type:        None for binary models; "hdf"/"twf"/etc. for multiclass
          failure_probability: model confidence as a float in [0, 1]

    Raises:
        httpx.HTTPStatusError:  If the API returns a 4xx or 5xx status code.
        httpx.TimeoutException: If the API takes longer than 10 seconds to respond.
    """
    # The API's SensorReading schema expects snake_case field names.
    # Our raw dict uses the original CSV column names (with spaces and brackets).
    # This translation mirrors what reading_to_raw_dict() does inside api.py —
    # except we're going from CSV names to API names, not the other way around.
    payload = {
        "machine_type":               raw["Type"],
        "air_temperature_kelvin":     raw["Air temperature [K]"],
        "process_temperature_kelvin": raw["Process temperature [K]"],
        "rotational_speed_rpm":       raw["Rotational speed [rpm]"],
        "torque_nm":                  raw["Torque [Nm]"],
        "tool_wear_minutes":          raw["Tool wear [min]"],
    }

    # httpx.post(json=payload) does three things automatically:
    #   1. Converts the dict to a JSON string.
    #   2. Sets the Content-Type header to "application/json".
    #   3. Sends the HTTP POST request to the URL.
    # timeout=10.0: if the API takes longer than 10 seconds, raise an exception.
    # In a production loop, a hanging request without a timeout would freeze
    # the entire simulation indefinitely.
    response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
    response.raise_for_status()     # turn HTTP error codes into Python exceptions

    # response.json() decodes the response body from JSON → Python dict:
    # {
    #   "machine_failure": 1,
    #   "failure_probability": 0.8734,
    #   "failure_type": null,          ← null in JSON becomes None in Python
    #   "model_name": "predictive-maintenance-binary",
    #   "model_version": "3"
    # }
    data = response.json()

    return (
        int(data["machine_failure"]),       # 0 or 1
        data.get("failure_type"),           # None for binary; "hdf" etc. for multiclass
        float(data["failure_probability"]), # float in [0, 1]
    )


# ══════════════════════════════════════════════════════════════════════════════
# SENSOR READING GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_raw_reading(tool_wear_minutes: float, inject_failure: bool) -> dict:
    """Sample one sensor reading from the AI4I training distribution.

    Normal readings are drawn from Gaussian distributions fitted to the
    training data. Failure readings shift the same distributions toward
    the EDA-identified failure zones (HDF, PWF, OSF).

    The model receives these shifted values through the API and must
    detect the anomaly from the physics alone — the label is never passed.

    Args:
        tool_wear_minutes: Current wear for this machine. Accumulated
                           externally so each machine ages independently.
        inject_failure:    True → shift sensor values toward failure zones.
                           False → sample from normal operating distributions.

    Returns:
        Dict using original CSV column names. These names must match
        COLUMN_RENAME in feature_transformation.py exactly.
    """
    machine_type = random.choices(MACHINE_TYPES, weights=MACHINE_TYPE_WEIGHTS)[0]

    if inject_failure:
        # For failure injection, air temp is sampled normally; process temp is
        # then shifted downward to simulate HDF (Heat Dissipation Failure):
        # the gap between process and air temp drops below the 8.6 K threshold.
        air_temp     = np.random.normal(AIR_TEMP_MEAN, 2.0)
        process_temp = air_temp + np.random.normal(FAILURE_TEMP_OFFSET_KELVIN, 0.5)  # narrowed gap
        # PWF + OSF: lower rpm and raise torque simultaneously
        torque = max(0.0, np.random.normal(TORQUE_NM[0] + FAILURE_TORQUE_ADD_NM, TORQUE_NM[1] * 0.5))
        rpm    = max(500.0, np.random.normal(ROTATIONAL_SPEED_RPM[0] + FAILURE_RPM_SHIFT, ROTATIONAL_SPEED_RPM[1] * 0.5))
    else:
        # Joint sampling: preserves the 0.876 correlation from the training data.
        # Independent sampling would give process_temp std = 2.24K vs the true 1.48K.
        air_temp, process_temp = np.random.multivariate_normal(
            [AIR_TEMP_MEAN, PROCESS_TEMP_MEAN], TEMP_COV_MATRIX
        )
        torque = max(0.0, np.random.normal(*TORQUE_NM))
        rpm    = max(500.0, np.random.normal(*ROTATIONAL_SPEED_RPM))

    return {
        "Type":                    machine_type,
        "Air temperature [K]":     round(air_temp, 1),
        "Process temperature [K]": round(process_temp, 1),
        "Rotational speed [rpm]":  int(round(rpm)),
        "Torque [Nm]":             round(torque, 1),
        "Tool wear [min]":         int(round(tool_wear_minutes)),
    }


def compute_failure_rate(mode: str, reading_idx: int, n_readings: int) -> float:
    """Return the failure injection probability for this reading.

    This controls HOW OFTEN failures are injected — not whether any given
    reading is a failure. For each reading, the simulator draws a random
    number and compares it to this rate. The failure rate sets the probability
    of that draw succeeding.

    Args:
        mode:        "normal", "gradual-drift", or "sudden-spike".
        reading_idx: Zero-based index of the current reading.
        n_readings:  Total readings in this run (used to compute progress).

    Returns:
        Float in [0, 1]: the probability that this reading will be a failure.
    """
    if mode == "normal":
        return BASE_FAILURE_RATE    # flat 3.4% throughout — matches training distribution

    if mode == "gradual-drift":
        # progress goes from 0.0 (first reading) to 1.0 (last reading)
        # Rate interpolates linearly from BASE_FAILURE_RATE to GRADUAL_DRIFT_PEAK_RATE
        progress = reading_idx / max(n_readings - 1, 1)
        return BASE_FAILURE_RATE + progress * (GRADUAL_DRIFT_PEAK_RATE - BASE_FAILURE_RATE)

    if mode == "sudden-spike":
        # First half: normal rate. Second half: spike rate. No gradual transition.
        return SUDDEN_SPIKE_RATE if reading_idx >= (n_readings // 2) else BASE_FAILURE_RATE

    return BASE_FAILURE_RATE        # fallback for any unrecognised mode


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_reading(
    reading_number: int,
    machine_id: str,
    raw: dict,
    prob: float,
    predicted: int,
    predicted_type: str | None,
    injected: int,
    rate: float,
) -> None:
    """Print one reading as a compact, scannable status line.

    Binary output example:
      [0001 | 14:23:05]  machine_02  L  T=042.1Nm  W=045min  rpm=1538  → normal      p=0.03  (rate=3%)
      [0251 | 14:26:45]  machine_04  H  T=058.7Nm  W=187min  rpm=1188  → FAILURE ⚠   p=0.91  (rate=40%) [injected]

    Multiclass output example:
      [0251 | 14:26:45]  machine_04  H  T=058.7Nm  W=187min  rpm=1188  → HDF ⚠       p=0.87  (rate=40%) [injected]
    """
    ts   = datetime.now().strftime("%H:%M:%S")
    flag = "  [injected]" if injected else ""

    if predicted_type is not None:
        # Multiclass: show the specific failure type; capitalise and add warning if it's a failure
        label = f"{predicted_type.upper()} ⚠" if predicted == 1 else predicted_type
    else:
        # Binary: just show FAILURE or normal
        label = "FAILURE ⚠" if predicted == 1 else "normal   "

    print(
        f"[{reading_number:04d} | {ts}]  "
        f"{machine_id}  "
        f"{raw['Type']}  "
        f"T={raw['Torque [Nm]']:05.1f}Nm  "
        f"W={raw['Tool wear [min]']:03d}min  "
        f"rpm={raw['Rotational speed [rpm]']:04d}  "
        f"→ {label:<10}  p={prob:.2f}  "
        f"(rate={rate:.0%})"
        f"{flag}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(
    api_url: str,               # replaces `model` — we now call the API instead of predicting locally
    conn: sqlite3.Connection,
    mode: str,
    n_readings: int,
    interval: float,
    n_machines: int,
) -> None:
    """Run the simulation loop: generate → send to API → store → print → repeat.

    Multiple machines
    -----------------
    Each machine starts at a random wear stage so the distribution of
    tool_wear_minutes across all readings matches the training data from
    the start. A single machine cycling 0→240→0 would create an artificial
    ramp that doesn't reflect a real factory floor with parallel machines
    at different stages of their lifecycle.

    Each iteration:
      1.  Pick a random machine; get its current tool wear.
      2.  Compute the failure injection probability for this reading.
      3.  Decide whether to inject a failure (random draw vs. rate).
      4.  Generate raw sensor values (shifted if injecting).
      5.  Compute engineered features locally — for SQLite storage only.
      6.  Call the API: POST /predict → get prediction back.
      7.  Build the full database row and insert it.
      8.  Print a status line.
      9.  Advance this machine's wear; reset to 0 when it reaches the limit.
      10. Sleep for `interval` seconds (0 = as fast as possible).

    Args:
        api_url:    Base URL of the FastAPI server. Replaces the `model` argument
                    from the previous architecture — we call the API instead.
        conn:       Open SQLite connection from init_db().
        mode:       "normal", "gradual-drift", or "sudden-spike".
        n_readings: Total readings to generate before stopping.
        interval:   Seconds to wait between readings. 0 = fast mode.
        n_machines: Number of simulated machines running in parallel.
    """
    # Each machine starts at a random wear stage rather than 0.
    # This distributes readings across the full 0–240 wear range from
    # the very first reading, matching the training data distribution.
    machine_ids  = [f"machine_{i+1:02d}" for i in range(n_machines)]
    machine_wear = {m: random.uniform(0, TOOL_WEAR_MAX_MINUTES) for m in machine_ids}

    for i in range(n_readings):

        # ── Pick a machine and compute this reading's failure probability ───────
        machine_id = random.choice(machine_ids)
        tool_wear  = machine_wear[machine_id]
        rate           = compute_failure_rate(mode, i, n_readings)
        inject_failure = random.random() < rate  # True if random draw falls below the rate

        # ── Generate raw sensor values ─────────────────────────────────────────
        raw = generate_raw_reading(tool_wear, inject_failure)

        # ── Compute engineered features FOR STORAGE ONLY ───────────────────────
        # The API will compute these again internally for inference. We compute
        # them here separately so we can store them in simulation.db, where
        # drift detection scripts can query them without re-running engineering.
        # This is intentional duplication — two different purposes for the same values.
        df_features = engineer_features(pd.DataFrame([raw]))

        # ── Send to API and get prediction ─────────────────────────────────────
        # This is the core change: instead of calling model.predict() locally,
        # we POST the reading to the serving layer and receive the prediction
        # as a JSON response. The API handles feature engineering for inference.
        try:
            predicted, predicted_type, failure_prob = call_predict_api(api_url, raw)
        except httpx.TimeoutException:
            print(f"  [reading {i+1}] WARNING: API timeout — skipping this reading.")
            continue
        except httpx.HTTPStatusError as exc:
            print(f"  [reading {i+1}] WARNING: API error {exc.response.status_code} — skipping.")
            continue

        # ── Detect binary vs multiclass from the response ──────────────────────
        # The API determines which target type it's running — the simulator
        # no longer needs a --target flag. We infer it from the response:
        # failure_type is None for binary models, a string for multiclass.
        target = "binary" if predicted_type is None else "multiclass"

        # ── Store the full row in SQLite ───────────────────────────────────────
        row = {
            "timestamp":                  datetime.now(timezone.utc).isoformat(),
            "reading_number":             i + 1,
            "machine_id":                 machine_id,
            "machine_type":               raw["Type"],
            "air_temperature_kelvin":     raw["Air temperature [K]"],
            "process_temperature_kelvin": raw["Process temperature [K]"],
            "rotational_speed_rpm":       raw["Rotational speed [rpm]"],
            "torque_nm":                  raw["Torque [Nm]"],
            "tool_wear_minutes":          raw["Tool wear [min]"],
            # Engineered features — computed locally above, stored for drift detection
            "power_kw":                   float(df_features["power_kw"].iloc[0]),
            "temp_diff_kelvin":           float(df_features["temp_diff_kelvin"].iloc[0]),
            "mechanical_stress":          float(df_features["mechanical_stress"].iloc[0]),
            # Prediction from the API
            "predicted_failure":          predicted,
            "predicted_failure_type":     predicted_type,
            "failure_probability":        failure_prob,
            # Ground truth from the simulator
            "injected_failure":           int(inject_failure),
            # Metadata
            "mode":                       mode,
            "target":                     target,   # derived from API response
            "effective_failure_rate":     rate,
        }
        store_reading(conn, row)

        # ── Print status line ──────────────────────────────────────────────────
        print_reading(
            i + 1, machine_id, raw, failure_prob,
            predicted, predicted_type, int(inject_failure), rate,
        )

        # ── Advance tool wear; replace tool when limit is reached ──────────────
        machine_wear[machine_id] += TOOL_WEAR_STEP_MINUTES
        if machine_wear[machine_id] >= TOOL_WEAR_MAX_MINUTES:
            machine_wear[machine_id] = 0.0
            print(f"  ── {machine_id}: tool replaced, wear reset to 0 ──")

        if interval > 0:
            time.sleep(interval)


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
#
# @click.command() turns the main() function into a command-line program with
# named flags (--mode, --n-readings, etc.) and automatic --help output.
# Each @click.option() adds one flag. The decorated function receives the
# flag values as keyword arguments.

@click.command()
@click.option(
    "--mode",
    default="normal",
    type=click.Choice(["normal", "gradual-drift", "sudden-spike"]),
    show_default=True,
    help=(
        "normal: stable 3.4% failure rate. "
        "gradual-drift: rate climbs to 25%. "
        "sudden-spike: normal then 40% jump (best for demos)."
    ),
)
@click.option(
    "--n-readings", default=200, show_default=True,
    help="Total sensor readings to generate.",
)
@click.option(
    "--n-machines", default=DEFAULT_N_MACHINES, show_default=True,
    help="Number of parallel machines to simulate. Each has its own tool wear counter.",
)
@click.option(
    "--interval", default=0.0, show_default=True,
    help="Seconds between readings. 0 = as fast as possible. 1.0 = live demo pacing.",
)
@click.option(
    "--api-url", default="http://127.0.0.1:8000", show_default=True,
    help=(
        "Base URL of the running FastAPI server. "
        "Change this to point at a remote server or a non-default port."
    ),
)
def main(
    mode: str,
    n_readings: int,
    n_machines: int,
    interval: float,
    api_url: str,
) -> None:
    """Simulate sensor readings, send each to the prediction API, and store results.

    The API must be running before the simulator starts. Start it with:
      uvicorn src.api:app --reload
    """
    print("\nPredictive Maintenance — Sensor Simulator")
    print(f"  Mode       : {mode}")
    print(f"  Machines   : {n_machines}")
    print(f"  Readings   : {n_readings}")
    print(f"  Interval   : {'fast (no delay)' if interval == 0 else f'{interval}s per reading'}")
    print(f"  API        : {api_url}")
    print(f"  Storage    : {DB_PATH.resolve()}")
    print()

    # Verify the API is up and a model is loaded before opening the database
    # or starting the loop — fail fast rather than fail deep.
    check_api_health(api_url)

    conn = init_db(DB_PATH)
    print("  Database ready. Starting simulation...\n")

    try:
        run_simulation(api_url, conn, mode, n_readings, interval, n_machines)
    finally:
        conn.close()   # always close the DB connection, even if the loop crashes

    print(f"\nDone — {n_readings} readings stored in {DB_PATH}.")
    print("Next: run drift detection on simulation.db to check for feature distribution shift.")


if __name__ == "__main__":
    main()
