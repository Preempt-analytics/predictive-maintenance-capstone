"""
Sensor Simulator — Predictive Maintenance
==========================================
Generates synthetic sensor readings that mimic the AI4I 2020 dataset,
runs each reading through your Production model in the MLflow registry,
and stores everything in a local SQLite database.

Why this exists
---------------
The full architecture has two loops:
  Inference loop  — simulator → model → prediction   (this file)
  Retraining loop — SQLite snapshot → DVC → MLflow   (triggered later by drift)

This simulator replaces real factory sensors for the capstone. It samples
from the same statistical distributions the model was trained on, so the
model's predictions are meaningful rather than arbitrary.

Multiple machines
-----------------
Each simulated machine has its own tool wear counter, starting at a random
lifecycle stage. The simulator picks a random machine for every reading.
This mirrors a real factory: many machines running in parallel, each at a
different point in its tool lifecycle. It also keeps the distribution of
tool_wear_minutes close to the training data (0–240, spread across machines)
rather than the artificial 0→240→0 ramp of a single machine.

Three simulation modes control how failure rates change over time:
  normal        — stable 3.4 % failure rate, matching the training dataset
  gradual-drift — rate climbs from 3.4 % to 25 % across all readings
  sudden-spike  — normal for the first half, jumps to 40 % for the second half
                  (most compelling for a live demo)

Binary vs multiclass
--------------------
  --target binary      → loads xgboost-binary/Production by default.
                         Output: FAILURE / normal + probability score.
  --target multiclass  → loads xgboost-multiclass/Production by default.
                         Output: predicted failure type (hdf, twf, pwf, osf, rnf, none).

Use --model-name to override the family (e.g. lgbm-binary, svm-multiclass).
The Production tag always controls which version within that family runs.

Usage
-----
  # Fast mode — 500 readings, 5 machines, sudden-spike, binary prediction
  python src/sensor_simulator.py --n-readings 500 --mode sudden-spike

  # Multiclass — see which failure type the model predicts
  python src/sensor_simulator.py --target multiclass --n-readings 300

  # Live demo — one reading per second, 10 machines
  python src/sensor_simulator.py --n-machines 10 --interval 1.0 --mode sudden-spike

Prerequisites
-------------
  1. Tag a model as Production in the MLflow UI:
       MLflow UI → Models → find your best run → set stage to Production.
  2. MLFLOW_TRACKING_URI must be set in .env (already done from training setup).
"""

import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Both files live in src/ — Python adds src/ to sys.path when you run
# `python src/sensor_simulator.py`, so this import resolves automatically.
# Using engineer_features() here guarantees the simulator applies identical
# transforms to what the model saw during training — the core guard against
# training-serving skew.
from feature_transformation import FEATURES, engineer_features

load_dotenv()


# ── Training distribution statistics ──────────────────────────────────────────
# Derived from pandas .describe() on data/ai4i2020.csv.
# Sampling from these distributions keeps simulator inputs in the same range
# the model was trained on.

MACHINE_TYPES        = ["L", "M", "H"]
MACHINE_TYPE_WEIGHTS = [0.60, 0.30, 0.10]  # approximate proportions in training data

# (mean, std) tuples for each continuous sensor
AIR_TEMP_KELVIN            = (300.0,  2.0)
PROCESS_TEMP_OFFSET_KELVIN = (10.0,   1.0)   # process temp is always ~10 K above air
ROTATIONAL_SPEED_RPM       = (1538.0, 179.0)
TORQUE_NM                  = (39.9,   9.97)

TOOL_WEAR_MAX_MINUTES  = 240
TOOL_WEAR_STEP_MINUTES = 2    # minutes of wear added per reading per machine

DEFAULT_N_MACHINES = 5        # mirrors a small factory floor; spread wear across machines


# ── Failure injection settings ─────────────────────────────────────────────────
# When the simulator schedules a failure it shifts sensor values toward the
# failure zones identified in EDA. The model must detect the failure from
# the sensor physics — the label is not forced.
#
# Three of the five AI4I failure modes are covered by sensor shifting:
#   HDF (Heat Dissipation): temp_diff < 8.6 K  → shrink temp gap to ~6 K
#   PWF (Power Failure):    power out of range  → low rpm + high torque
#   OSF (Overstrain):       torque × wear high  → boost torque on worn tools
#   TWF (Tool Wear):        wear > 200 min      → handled naturally by tool lifecycle
#   RNF (Random):           no sensor pattern   → not injected (by definition random)

FAILURE_TORQUE_ADD_NM      = 18.0
FAILURE_RPM_SHIFT          = -350.0
FAILURE_TEMP_OFFSET_KELVIN = 6.0    # narrower than normal ~10 K gap


# ── Failure rates per mode ─────────────────────────────────────────────────────
BASE_FAILURE_RATE       = 0.034
GRADUAL_DRIFT_PEAK_RATE = 0.25
SUDDEN_SPIKE_RATE       = 0.40


# ── Storage ───────────────────────────────────────────────────────────────────
DB_PATH = Path("simulation.db")   # gitignored; stays local


# ── Database ──────────────────────────────────────────────────────────────────

def init_db(db_path: Path) -> sqlite3.Connection:
    """Create the SQLite database and sensor_readings table if absent.

    Safe to call on an existing database — CREATE TABLE IF NOT EXISTS means
    no data is lost on restart. Simulation runs append; nothing is overwritten.

    Args:
        db_path: Path to the SQLite file. Created automatically if absent.

    Returns:
        An open sqlite3.Connection ready for INSERT and SELECT operations.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp                   TEXT    NOT NULL,
            reading_number              INTEGER NOT NULL,
            machine_id                  TEXT    NOT NULL,   -- which simulated machine

            -- Raw sensor inputs (original units, before feature engineering)
            machine_type                TEXT    NOT NULL,
            air_temperature_kelvin      REAL    NOT NULL,
            process_temperature_kelvin  REAL    NOT NULL,
            rotational_speed_rpm        REAL    NOT NULL,
            torque_nm                   REAL    NOT NULL,
            tool_wear_minutes           REAL    NOT NULL,

            -- Engineered features (stored at write time so drift detection
            -- can query them directly without rerunning engineer_features())
            power_kw                    REAL    NOT NULL,
            temp_diff_kelvin            REAL    NOT NULL,
            mechanical_stress           REAL    NOT NULL,

            -- Model output
            -- predicted_failure:      0 = no failure, 1 = any failure
            -- predicted_failure_type: specific type for multiclass runs
            --                         (hdf / twf / pwf / osf / rnf / none)
            --                         NULL for binary runs
            predicted_failure           INTEGER NOT NULL,
            predicted_failure_type      TEXT,
            failure_probability         REAL    NOT NULL,

            -- Ground truth injected by the simulator
            injected_failure            INTEGER NOT NULL,

            -- Simulation metadata
            mode                        TEXT    NOT NULL,
            target                      TEXT    NOT NULL,   -- binary or multiclass
            effective_failure_rate      REAL    NOT NULL
        )
    """)
    conn.commit()
    return conn


def store_reading(conn: sqlite3.Connection, row: dict) -> None:
    """Insert one sensor reading into the sensor_readings table.

    Args:
        conn: Open database connection returned by init_db().
        row:  Dict whose keys match the column names in sensor_readings exactly.
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


# ── Model loading ──────────────────────────────────────────────────────────────

def resolve_model_name(target: str, model_name_override: str | None) -> str:
    """Return the registered model name to load, with sensible defaults.

    The model name determines the family (xgboost-binary, lgbm-multiclass, etc.).
    The Production tag determines which version within that family runs.
    Passing --model-name overrides both defaults.

    Args:
        target:              "binary" or "multiclass".
        model_name_override: Value of --model-name flag, or None if not passed.

    Returns:
        Registered model name string to use in the MLflow URI.
    """
    if model_name_override:
        return model_name_override
    return "xgboost-binary" if target == "binary" else "xgboost-multiclass"


def load_production_model(model_name: str):
    """Load the Production-tagged version of a registered model from MLflow.

    Two separate decisions live in the URI  models:/<name>@production:
      <name>      — the model family, set by --target or --model-name
      @production — the alias within that family, set in the MLflow UI via
                    "Promote model". Aliases replace Stages in MLflow 2.9+.

    Retag a better run as Production in the MLflow UI and this function
    picks it up automatically on the next run — no code change needed.

    Args:
        model_name: Registered model name, e.g. "xgboost-binary".

    Returns:
        A fitted sklearn Pipeline (DictVectorizer → classifier).

    Raises:
        SystemExit: Prints fix instructions and exits if no Production model exists.
    """
    uri = f"models:/{model_name}@production"
    try:
        model = mlflow.sklearn.load_model(uri)
        print(f"  Model loaded : {uri}")
        return model
    except Exception as exc:
        print(
            f"\nERROR: No Production model found at '{uri}'.\n\n"
            "To fix:\n"
            "  1. Open the MLflow UI.\n"
            "  2. Go to Models → find your best run for this experiment.\n"
            "  3. Click the version number → set Stage to 'Production'.\n"
            f"\nOriginal error: {exc}\n"
        )
        sys.exit(1)


# ── Sensor reading generation ──────────────────────────────────────────────────

def generate_raw_reading(tool_wear_minutes: float, inject_failure: bool) -> dict:
    """Sample one sensor reading from the AI4I training distribution.

    Normal readings are drawn from Gaussian distributions fitted to the
    training data. Failure readings shift the same distributions toward
    the EDA-identified failure zones (HDF, PWF, OSF).

    Args:
        tool_wear_minutes: Current wear for this machine. Passed in so wear
                           accumulates naturally per machine across readings.
        inject_failure:    True  → shift values toward failure-inducing extremes.
                           False → sample from normal operating distributions.

    Returns:
        Dict with original CSV column names matching COLUMN_RENAME in
        feature_transformation.py. Pass directly to engineer_features().
    """
    machine_type = random.choices(MACHINE_TYPES, weights=MACHINE_TYPE_WEIGHTS)[0]
    air_temp     = np.random.normal(*AIR_TEMP_KELVIN)

    if inject_failure:
        # HDF: shrink temp gap below the 8.6 K failure threshold
        process_temp = air_temp + np.random.normal(FAILURE_TEMP_OFFSET_KELVIN, 0.5)
        # PWF + OSF: low rpm, high torque → power spike and overstrain
        torque = max(0.0, np.random.normal(
            TORQUE_NM[0] + FAILURE_TORQUE_ADD_NM, TORQUE_NM[1] * 0.5
        ))
        rpm = max(500.0, np.random.normal(
            ROTATIONAL_SPEED_RPM[0] + FAILURE_RPM_SHIFT, ROTATIONAL_SPEED_RPM[1] * 0.5
        ))
    else:
        process_temp = air_temp + np.random.normal(*PROCESS_TEMP_OFFSET_KELVIN)
        torque       = max(0.0, np.random.normal(*TORQUE_NM))
        rpm          = max(500.0, np.random.normal(*ROTATIONAL_SPEED_RPM))

    return {
        "Type":                    machine_type,
        "Air temperature [K]":     round(air_temp, 1),
        "Process temperature [K]": round(process_temp, 1),
        "Rotational speed [rpm]":  int(round(rpm)),
        "Torque [Nm]":             round(torque, 1),
        "Tool wear [min]":         int(round(tool_wear_minutes)),
    }


def compute_failure_rate(mode: str, reading_idx: int, n_readings: int) -> float:
    """Return the failure injection probability for this reading index.

    Each mode produces a different failure rate trajectory:
      normal        → flat 3.4 % throughout
      gradual-drift → linear ramp from 3.4 % to 25 %
      sudden-spike  → 3.4 % for the first half, 40 % for the second half

    Args:
        mode:        Simulation mode string.
        reading_idx: Zero-based index of the current reading.
        n_readings:  Total readings in this run.

    Returns:
        Failure probability for this reading as a float between 0.0 and 1.0.
    """
    if mode == "normal":
        return BASE_FAILURE_RATE
    if mode == "gradual-drift":
        progress = reading_idx / max(n_readings - 1, 1)
        return BASE_FAILURE_RATE + progress * (GRADUAL_DRIFT_PEAK_RATE - BASE_FAILURE_RATE)
    if mode == "sudden-spike":
        return SUDDEN_SPIKE_RATE if reading_idx >= (n_readings // 2) else BASE_FAILURE_RATE
    return BASE_FAILURE_RATE


def predict(model, record: list[dict], target: str) -> tuple[int, str | None, float]:
    """Run one prediction and return a normalised result for both targets.

    Binary target:     predicted=0/1, failure_type=None, prob=P(failure)
    Multiclass target: predicted=0/1 (0 if "none"), failure_type=class string,
                       prob=probability of the predicted class

    Keeping prediction logic here rather than in the loop lets the loop stay
    identical for both targets — only this function changes between modes.

    Args:
        model:  Fitted sklearn Pipeline from load_production_model().
        record: Single-row list of feature dicts from engineer_features().
        target: "binary" or "multiclass".

    Returns:
        Tuple of (predicted_failure, predicted_failure_type, failure_probability).
    """
    if target == "binary":
        predicted    = int(model.predict(record)[0])
        failure_prob = float(model.predict_proba(record)[0][1])
        return predicted, None, failure_prob

    # Multiclass: predicted class is a string label ("hdf", "none", etc.)
    predicted_type = str(model.predict(record)[0])
    predicted      = 0 if predicted_type == "none" else 1
    failure_prob   = float(max(model.predict_proba(record)[0]))
    return predicted, predicted_type, failure_prob


# ── Console output ─────────────────────────────────────────────────────────────

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

    Binary example:
      [0001 | 14:23:05]  machine_2  L  T=042.1Nm  W=045min  → normal     p=0.03  (rate=3%)
      [0201 | 14:26:45]  machine_4  H  T=058.7Nm  W=187min  → FAILURE ⚠  p=0.91  (rate=40%) [injected]

    Multiclass example:
      [0001 | 14:23:05]  machine_2  L  T=042.1Nm  W=045min  → none       p=0.94  (rate=3%)
      [0201 | 14:26:45]  machine_4  H  T=058.7Nm  W=187min  → HDF ⚠      p=0.87  (rate=40%) [injected]

    Args:
        reading_number:  1-based counter.
        machine_id:      Which machine produced this reading.
        raw:             Raw sensor dict from generate_raw_reading().
        prob:            Failure probability (binary) or predicted-class probability (multiclass).
        predicted:       0 = no failure, 1 = failure.
        predicted_type:  Failure type string for multiclass runs; None for binary.
        injected:        1 if the simulator injected a failure, 0 otherwise.
        rate:            Effective failure rate at this reading.
    """
    ts   = datetime.now().strftime("%H:%M:%S")
    flag = "  [injected]" if injected else ""

    if predicted_type is not None:
        label = f"{predicted_type.upper()} ⚠" if predicted == 1 else predicted_type
    else:
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


# ── Simulation loop ────────────────────────────────────────────────────────────

def run_simulation(
    model,
    conn: sqlite3.Connection,
    mode: str,
    n_readings: int,
    interval: float,
    n_machines: int,
    target: str,
) -> None:
    """Run the main simulation loop until n_readings is reached.

    Multiple machines
    -----------------
    Each machine starts at a random wear stage (not always 0) so the
    distribution of tool_wear_minutes across readings matches the training
    data rather than always cycling 0→240 from the start.
    A random machine is selected for each reading — this is the key change
    that makes the output resemble readings arriving from many machines in
    parallel rather than one machine running sequentially.

    Each iteration:
      1. Pick a random machine; retrieve its current tool wear.
      2. Compute failure rate for this reading based on mode + position.
      3. Decide whether to inject a failure.
      4. Generate raw sensor values (shifted if injecting).
      5. Apply engineer_features() — identical to the training transform.
      6. Predict using the Production model.
      7. Store the full row in SQLite.
      8. Print a status line.
      9. Advance this machine's tool wear; reset on replacement.
      10. Sleep for `interval` seconds (0 = fast mode).

    Args:
        model:      Fitted sklearn Pipeline from load_production_model().
        conn:       Open SQLite connection from init_db().
        mode:       "normal", "gradual-drift", or "sudden-spike".
        n_readings: Total readings to generate.
        interval:   Seconds between readings. 0 = as fast as possible.
        n_machines: Number of parallel machines to simulate.
        target:     "binary" or "multiclass".
    """
    # Initialise each machine at a random lifecycle stage so readings
    # are spread across the full 0–240 wear range from the start.
    machine_ids  = [f"machine_{i+1:02d}" for i in range(n_machines)]
    machine_wear = {m: random.uniform(0, TOOL_WEAR_MAX_MINUTES) for m in machine_ids}

    for i in range(n_readings):
        machine_id = random.choice(machine_ids)
        tool_wear  = machine_wear[machine_id]

        rate           = compute_failure_rate(mode, i, n_readings)
        inject_failure = random.random() < rate

        # ── Generate and transform ─────────────────────────────────────────────
        raw         = generate_raw_reading(tool_wear, inject_failure)
        df_features = engineer_features(pd.DataFrame([raw]))
        record      = df_features[FEATURES].to_dict(orient="records")

        # ── Predict ────────────────────────────────────────────────────────────
        predicted, predicted_type, failure_prob = predict(model, record, target)

        # ── Store ──────────────────────────────────────────────────────────────
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
            "power_kw":                   float(df_features["power_kw"].iloc[0]),
            "temp_diff_kelvin":           float(df_features["temp_diff_kelvin"].iloc[0]),
            "mechanical_stress":          float(df_features["mechanical_stress"].iloc[0]),
            "predicted_failure":          predicted,
            "predicted_failure_type":     predicted_type,
            "failure_probability":        failure_prob,
            "injected_failure":           int(inject_failure),
            "mode":                       mode,
            "target":                     target,
            "effective_failure_rate":     rate,
        }
        store_reading(conn, row)

        # ── Print ──────────────────────────────────────────────────────────────
        print_reading(
            i + 1, machine_id, raw, failure_prob,
            predicted, predicted_type, int(inject_failure), rate
        )

        # ── Advance this machine's tool wear ───────────────────────────────────
        machine_wear[machine_id] += TOOL_WEAR_STEP_MINUTES
        if machine_wear[machine_id] >= TOOL_WEAR_MAX_MINUTES:
            machine_wear[machine_id] = 0.0
            print(f"  ── {machine_id}: tool replaced, wear reset to 0 ──")

        if interval > 0:
            time.sleep(interval)


# ── Entry point ────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--mode",
    default="normal",
    type=click.Choice(["normal", "gradual-drift", "sudden-spike"]),
    show_default=True,
    help=(
        "normal: stable 3.4 % failure rate.  "
        "gradual-drift: rate climbs to 25 %.  "
        "sudden-spike: normal then 40 % jump (best for demos)."
    ),
)
@click.option(
    "--target",
    default="binary",
    type=click.Choice(["binary", "multiclass"]),
    show_default=True,
    help=(
        "binary: predict failure / no failure. "
        "multiclass: predict which failure type (hdf, twf, pwf, osf, rnf, none)."
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
    help="Seconds between readings. 0 = fast mode. 1.0 = live demo pacing.",
)
@click.option(
    "--model-name", default=None,
    help=(
        "Override the registered model name (e.g. lgbm-binary, svm-multiclass). "
        "Defaults to xgboost-binary for --target binary, "
        "xgboost-multiclass for --target multiclass."
    ),
)
def main(
    mode: str,
    target: str,
    n_readings: int,
    n_machines: int,
    interval: float,
    model_name: str | None,
) -> None:
    """Simulate sensor readings from multiple machines and predict failures.

    Readings are stored in simulation.db (SQLite, gitignored).
    Use the accumulated readings as input to drift detection once enough
    data has been collected.
    """
    resolved_model = resolve_model_name(target, model_name)

    print("\nPredictive Maintenance — Sensor Simulator")
    print(f"  Mode       : {mode}")
    print(f"  Target     : {target}")
    print(f"  Machines   : {n_machines}")
    print(f"  Readings   : {n_readings}")
    print(f"  Interval   : {'fast (no delay)' if interval == 0 else f'{interval}s per reading'}")
    print(f"  Model      : {resolved_model} @ production (alias)")
    print(f"  Storage    : {DB_PATH.resolve()}")
    print()

    model = load_production_model(resolved_model)
    conn  = init_db(DB_PATH)
    print("  Database ready. Starting simulation...\n")

    try:
        run_simulation(model, conn, mode, n_readings, interval, n_machines, target)
    finally:
        conn.close()

    print(f"\nDone — {n_readings} readings stored in {DB_PATH}.")
    print("Next: run drift detection on simulation.db to check for feature distribution shift.")


if __name__ == "__main__":
    main()
