"""
scripts/detect_drift.py
========================
Detect whether live sensor readings have drifted away from the training
distribution using Evidently AI.

WHY THIS SCRIPT EXISTS
──────────────────────
A model is only as good as the match between what it was trained on and what
it sees in production.  If the factory floor changes — new machines, seasonal
temperature swings, changed maintenance schedules — the model's predictions
degrade silently.  You won't see an error; you'll just see quietly wrong
predictions until a technician notices a missed failure.

Drift detection is your early-warning system.  Run it daily (or hook it into
CI), and it tells you: "the sensor readings look different from training —
consider retraining before the model drifts out of accuracy."

HOW IT WORKS (four stages)
──────────────────────────
  Stage 1 — Load REFERENCE data  : the training CSV (what the model learned).
  Stage 2 — Load CURRENT data    : recent readings from simulation.db.
  Stage 3 — Run Evidently        : one statistical test per feature column.
  Stage 4 — Verdict              : PASS or FAIL based on your threshold.

KEY VOCABULARY (read once, then the code will make sense)
──────────────────────────────────────────────────────────
  Reference data    : historical baseline — the distribution the model knows.
  Current data      : recent live readings you want to compare against.
  Drift share       : fraction of features where the statistical test detected
                      a meaningful shift (e.g. 0.33 = 3 of 9 features drifted).
  KS test           : Kolmogorov-Smirnov test — the default for continuous
                      features (temperature, rpm, torque...).  Outputs a p-value:
                      p < 0.05 means "very unlikely these two samples come from
                      the same distribution" → drift detected.
  Chi-squared test  : used for categorical features (machine type L/M/H).
                      Tests whether the proportions of categories have changed.
  DRIFT_THRESHOLD   : the fraction of drifted features above which we raise an
                      alert.  Default is 0.25 — override with --threshold.

EVIDENTLY API (version 0.7.x) — what is different from older tutorials
──────────────────────────────────────────────────────────────────────────
  Older Evidently (< 0.7):  from evidently.report import Report
  This version (0.7.x):     from evidently import Report
                             from evidently.presets import DataDriftPreset

  report.run() now RETURNS a Snapshot object (it used to modify report in place).
  Use snapshot.save_html(), snapshot.dict() — NOT report.save_html().
"""

# ── Section 1: Imports ─────────────────────────────────────────────────────────
# Standard library
import argparse           # parse --flags from the command line
import pathlib            # Path objects are safer than raw strings for file paths
import sqlite3            # built-in SQLite adapter — no install required
import subprocess         # call export script when --export-on-drift is set
import sys                # sys.exit(1) signals FAIL to the calling shell / CI
import warnings           # suppress cosmetic RuntimeWarnings from scipy/Evidently

# Evidently calls scipy routines (vecdot, etc.) that emit RuntimeWarnings on
# Wasserstein distance computations when sample sizes are unequal. The results
# are correct; the warnings are noise from a scipy version mismatch in Evidently.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

# Third-party
import pandas as pd       # tabular data; Evidently expects DataFrames

# Evidently 0.7.x API.
# Report      : the container that holds one or more metric objects.
# DataDriftPreset : shorthand that adds per-column drift metrics for every feature.
#                   Produces the interactive histograms you see in the HTML report.
# DataSummaryPreset : adds data-quality checks and column-level stats to the same report.
# DriftedColumnsCount : a single aggregate metric: "N of M features drifted".
#                       We use this to extract a numeric drift_share for the verdict.
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.presets import DataSummaryPreset
from evidently.metrics import DriftedColumnsCount


# Our own project modules — feature definitions shared between training and inference.
# sys.path trick: lets Python find src/ without installing the package.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from feature_transformation import FEATURES, engineer_features   # 9 feature names + transforms


# ── Section 2: Configuration constants ────────────────────────────────────────
# Path defaults — all relative to the project root so the script works from any
# working directory (as long as you run it from the repo root or use --csv / --db).
_ROOT         = pathlib.Path(__file__).parent.parent
DATA_CSV      = _ROOT / "data" / "ai4i2020_baseline.csv"  # frozen original 10k rows — never appended to
SIMULATION_DB = _ROOT / "data" / "simulation.db"          # inside data/ so the existing Docker volume covers it
REPORT_DIR    = _ROOT / "reports"                         # where the HTML report is saved

# WHY a frozen baseline and not the growing ai4i2020.parquet?
# Every time export_simulation_to_parquet.py runs, new rows are appended to ai4i2020.parquet.
# If we used that file as the drift reference, the baseline would shift toward
# the current distribution after every retrain — making drift look smaller than
# it actually is over time (the ratchet effect).
# ai4i2020_baseline.csv is locked at project start and never changes, so drift
# is always measured against the same original ground truth.

# 0.33 ≈ 3/9 — alert if two or more features drift. Missing real drift is more costly
# than a false alarm in a safety-critical context (unplanned downtime, failed components),
# so we err on the side of sensitivity without triggering on a single noisy feature.
# Override at runtime with --threshold without editing this file.
DRIFT_THRESHOLD = 0.33

# Evidently's statistical tests (KS, Wasserstein) plateau in accuracy at around
# 5,000–10,000 rows — loading more does not improve detection quality but slows
# Evidently down linearly. This cap keeps each nightly check fast regardless of
# how many readings have accumulated in simulation.db since the last purge.
CURRENT_DATA_LIMIT = 10_000  # maximum rows read from simulation.db per drift check


# ── Section 3: Load reference data ────────────────────────────────────────────
def load_reference_data(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read the training CSV, run feature engineering, return the 9 feature columns.

    WHY engineer_features() here?
    The model was trained on the OUTPUT of engineer_features() — the renamed
    columns plus power_kw, temp_diff_kelvin, mechanical_stress.  Evidently must
    compare the same feature space that the model sees; otherwise you're comparing
    apples to oranges.  This is the same training-serving skew concern that
    feature_transformation.py was designed to prevent.

    WHY sample / dropna?
    We drop NaN rows so Evidently's statistical tests get clean input.
    A few missing values in a 10 000-row training set is normal; they're noise,
    not signal worth drifting on.
    """
    df = pd.read_csv(csv_path)          # load the original AI4I dataset
    df = engineer_features(df)          # rename columns + compute derived features
    df = df[FEATURES].dropna()          # keep only the 9 model features; drop NaN rows
    return df


# ── Section 4: Load current data ──────────────────────────────────────────────
def load_current_data(db_path: pathlib.Path, since: str | None = None) -> pd.DataFrame:
    """Read recent sensor readings from SQLite, return the same 9 feature columns.

    WHY skip engineer_features() here?
    The simulator already stores engineered features in simulation.db — it
    computed power_kw, temp_diff_kelvin, and mechanical_stress at write time
    using the same feature_transformation.py contract.  Re-running engineer_features()
    would double-apply the transform.  See sensor_simulator.py store_reading().

    WHY SELECT with column aliases?
    The DB column is machine_type; FEATURES expects 'type' (the renamed version).
    The SELECT alias renames it in one step, so the returned DataFrame matches
    the reference DataFrame column for column.
    """
    # ORDER BY timestamp DESC takes the newest rows first; LIMIT caps the total.
    # This means we always check the most recent factory readings — the ones that
    # reflect today's machine behaviour — rather than older data that may no longer
    # be representative. ISO-8601 strings sort correctly in SQLite lexicographic order.
    conn = sqlite3.connect(db_path)

    base_query = """
        SELECT
            machine_type            AS type,
            air_temperature_kelvin,
            process_temperature_kelvin,
            rotational_speed_rpm,
            torque_nm,
            tool_wear_minutes,
            power_kw,
            temp_diff_kelvin,
            mechanical_stress
        FROM sensor_readings
    """

    # Append WHERE (optional) then ORDER BY + LIMIT to both query paths.
    # Row order does not matter for KS or Wasserstein tests — statistics are
    # order-independent — so DESC ordering only affects which rows are kept.
    limit_suffix = f"ORDER BY timestamp DESC LIMIT {CURRENT_DATA_LIMIT}"
    if since is not None:
        query = base_query + f" WHERE timestamp > :since {limit_suffix}"
        df = pd.read_sql_query(query, conn, params={"since": since})
    else:
        query = base_query + f" {limit_suffix}"
        df = pd.read_sql_query(query, conn)
    conn.close()

    return df.dropna()


# ── Section 5: Run Evidently report ───────────────────────────────────────────
def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    report_path: pathlib.Path,
) -> dict:
    """Build the Evidently Report, run it, save HTML, return the results dict.

    HOW DataDriftPreset WORKS
    ─────────────────────────
    DataDriftPreset is a convenience wrapper.  It automatically adds one
    ValueDrift metric per column.  Each ValueDrift metric runs the most
    appropriate statistical test for the column type:
      • float / int columns  → Kolmogorov-Smirnov test
      • string / category    → chi-squared test

    HOW DriftedColumnsCount WORKS
    ─────────────────────────────
    After all the per-column ValueDrift tests run, DriftedColumnsCount
    counts how many columns had p_value < threshold (default 0.05) and
    returns {"count": N, "share": N/total}.

    We use the "share" value to compare against DRIFT_THRESHOLD in main().

    SNAPSHOT vs REPORT (Evidently 0.7.x)
    ─────────────────────────────────────
    In this version, report.run() RETURNS a Snapshot object instead of
    modifying report in place.  All result-access methods are on the snapshot:
      snapshot.save_html()  → save the interactive HTML dashboard
      snapshot.dict()       → get raw results as a Python dict for parsing
      snapshot.json()       → same but as a JSON string
    """
    report = Report(metrics=[
        DataDriftPreset(),       # per-column drift tests → drives the HTML histograms
        DataSummaryPreset(),     # data-quality checks and column-level stats
        DriftedColumnsCount(),   # aggregate: N drifted / M total → drives the verdict
    ])

    # report.run() executes all the statistical tests and returns a Snapshot.
    # Evidently automatically picks KS for numeric columns, chi-sq for categorical.
    snapshot = report.run(reference_data=reference, current_data=current)

    # Save the interactive HTML dashboard to disk.
    # Open this in a browser — you get side-by-side histograms for every feature,
    # colour-coded by whether drift was detected.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(report_path))

    # Return the results dict for programmatic parsing in extract_drift_summary().
    return snapshot.dict()


# ── Section 6: Parse results ───────────────────────────────────────────────────
def extract_drift_summary(result: dict) -> tuple[float, int, int]:
    """Pull (drift_share, drifted_count, total_count) out of the results dict.

    The results dict has this structure:
      {
        "metrics": [
          {
            "metric_name": "DriftedColumnsCount(drift_share=0.5)",
            "value": {"count": 3.0, "share": 0.3333}
          },
          {
            "metric_name": "ValueDrift(column=torque_nm, …)",
            "value": 0.042          # the p-value
          },
          ...
        ]
      }

    We search for the DriftedColumnsCount entry and extract its "share" field.
    The share is the fraction of features where p_value < 0.05 (Evidently default).
    """
    for metric in result["metrics"]:
        if "DriftedColumnsCount" in metric["metric_name"]:
            share  = float(metric["value"]["share"])
            count  = int(metric["value"]["count"])
            # Back-calculate total feature count from share and count.
            # Avoid ZeroDivisionError: if nothing drifted, share == 0.
            total = round(count / share) if share > 0 else len(FEATURES)
            return share, count, total

    # Fallback — should never happen with the Report setup above.
    return 0.0, 0, len(FEATURES)


def print_per_column_results(result: dict) -> None:
    """Print a compact table showing the drift verdict for each feature.

    Each row shows the feature name, the statistical test used, the metric value,
    and whether drift was detected.  This is a quick scan before you open the
    full HTML report.

    WHY 'metric value' and not always 'p-value'?
    Evidently automatically picks the best test based on sample size:
      Small samples (< ~1000) : KS test or chi-squared  (output: p-value; drift if < 0.05)
      Large samples            : Wasserstein distance    (output: distance; drift if > 0.1)
                                 Jensen-Shannon distance (output: distance; drift if > 0.1)
    The threshold shown next to each row is the one Evidently used internally.
    """
    print(f"\n  {'Feature':<28} {'Test':<30} {'Value':>10}  {'Thresh':>7}  Drift?")
    print("  " + "-" * 78)

    for metric in result["metrics"]:
        # Skip the summary metric -- we only want per-column ValueDrift entries.
        if not metric["metric_name"].startswith("ValueDrift"):
            continue

        cfg     = metric["config"]
        col     = cfg.get("column", "?")
        method  = cfg.get("method", "?")
        val     = float(metric["value"])    # distance or p-value depending on method
        thresh  = float(cfg.get("threshold", 0.05))

        # Determine drift direction: p-value tests drift when val < thresh;
        # distance-based tests drift when val > thresh.
        is_distance = "distance" in method.lower()
        drifted = (val > thresh) if is_distance else (val < thresh)
        label   = "YES !" if drifted else "no"

        print(f"  {col:<28} {method:<30} {val:>10.4f}  {thresh:>7.2f}  {label}")

    print()


# ── Section 7: CLI entrypoint ──────────────────────────────────────────────────
def main() -> None:
    """Parse CLI flags, run the four stages, print verdict, exit 0/1."""
    parser = argparse.ArgumentParser(
        description="Compare live sensor distributions to the training baseline (Evidently AI).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all simulation data against the training baseline:
  python scripts/detect_drift.py

  # Check only readings after a specific timestamp:
  python scripts/detect_drift.py --since "2026-05-28T00:00:00"

  # Override the drift threshold at runtime:
  python scripts/detect_drift.py --threshold 0.2

  # Write the HTML report to a custom path:
  python scripts/detect_drift.py --report reports/drift_morning.html
        """,
    )
    parser.add_argument(
        "--csv", default=str(DATA_CSV),
        help="Frozen reference CSV. Default: data/ai4i2020_baseline.csv (never use ai4i2020.csv here — it grows with each export and undermines long-term drift tracking).",
    )
    parser.add_argument(
        "--db", default=str(SIMULATION_DB),
        help="Path to simulation.db (current data). Default: data/simulation.db",
    )
    parser.add_argument(
        "--since", default=None,
        help="ISO-8601 timestamp — only include readings after this time. E.g. '2026-05-28T00:00:00'.",
    )
    parser.add_argument(
        "--threshold", type=float, default=DRIFT_THRESHOLD,
        help=f"Drift-share threshold that triggers a FAIL verdict (default: {DRIFT_THRESHOLD}).",
    )
    parser.add_argument(
        "--report", default=str(REPORT_DIR / "drift_report.html"),
        help="Output path for the Evidently HTML report.",
    )
    parser.add_argument(
        "--export-on-drift",
        action="store_true",
        default=False,
        help=(
            "Always export simulation data after detection. "
            "If drift was detected: export --push --retrain (updates retrain.trigger, fires GitHub Actions). "
            "If no drift: export --push only (CSV grows on DagsHub, no workflow triggered). "
            "Only safe when the simulator ran in --mode normal."
        ),
    )
    args = parser.parse_args()

    # ── Stage 1: Load data ────────────────────────────────────────────────────
    print("-" * 72)
    print("  Predictive Maintenance  - Drift Detection")
    print("-" * 72)

    print(f"\n[1/4] Reference data  : {args.csv}")
    reference = load_reference_data(pathlib.Path(args.csv))
    print(f"      Loaded {len(reference):,} rows  |  features: {list(reference.columns)}")

    since_label = f"since {args.since}" if args.since else "all rows"
    print(f"\n[2/4] Current data    : {args.db}  ({since_label})")
    current = load_current_data(pathlib.Path(args.db), since=args.since)
    print(f"      Loaded {len(current):,} rows")

    too_few = len(current) < 30
    if too_few:
        if sys.stdin.isatty():
            # Interactive terminal: ask before running statistically weak tests.
            response = input(
                f"\n  Only {len(current)} rows loaded — KS/Wasserstein tests are unreliable below 30.\n"
                f"  Continue anyway? [y/N] "
            ).strip().lower()
            if response != "y":
                print("  Aborted. Run more simulations first:")
                print("    python src/sensor_simulator.py --n-readings 200")
                sys.exit(0)
        else:
            # Non-interactive context (e.g. called via subprocess from the simulator).
            # input() would block indefinitely here — warn and continue instead.
            print(
                f"\n  WARNING: only {len(current)} rows — statistical tests unreliable below 30. "
                "Continuing (non-interactive)."
            )

    # ── Stage 2: Run Evidently ────────────────────────────────────────────────
    print(f"\n[3/4] Running Evidently...")
    result = run_drift_report(reference, current, pathlib.Path(args.report))
    print(f"      HTML report saved -> {args.report}")

    # ── Stage 3: Print per-column table ───────────────────────────────────────
    print_per_column_results(result)

    # ── Stage 4: Verdict ──────────────────────────────────────────────────────
    drift_share, drifted_count, total_count = extract_drift_summary(result)

    print(f"[4/4] Verdict")
    print(f"      Threshold  : {args.threshold:.0%} of features must drift to trigger alert")
    print(f"      Detected   : {drifted_count}/{total_count} features drifted ({drift_share:.0%})")

    export_script = pathlib.Path(__file__).parent / "export_simulation_to_parquet.py"

    if drift_share >= args.threshold:
        print(f"\n  *** DRIFT DETECTED — {drifted_count} feature(s) shifted significantly ***")
        if args.export_on_drift:
            # Drift confirmed: export CSV and update retrain.trigger in the same commit.
            # --retrain writes a timestamp to retrain.trigger, which GitHub Actions watches.
            print(f"  --export-on-drift set: exporting data and triggering retrain workflow...")
            subprocess.run(
                [sys.executable, str(export_script), "--push", "--retrain"], check=True
            )
        else:
            print(f"  Next steps:")
            print(f"    1. Open the HTML report and check which features changed.")
            print(f"    2. Export and retrain:")
            print(f"         Automated  : monitor.py handles this automatically — no action needed.")
            print(f"         Manual run : docker compose exec monitor python scripts/export_simulation_to_parquet.py --push --retrain")
    else:
        print(f"\n  PASS — distribution looks stable. No retraining triggered.")
        if args.export_on_drift:
            # No drift: export CSV to keep the training dataset growing, but do NOT update
            # retrain.trigger — the workflow stays silent because there is nothing to learn.
            print(f"  --export-on-drift set: exporting data for accumulation (no retrain trigger)...")
            subprocess.run(
                [sys.executable, str(export_script), "--push"], check=True   # no --retrain
            )

    if too_few:
        print(
            f"\n  WARNING: results above are based on only {len(current)} rows — "
            f"treat them as indicative only.\n"
            f"  Run more simulations for reliable statistics: "
            f"python src/sensor_simulator.py --n-readings 200"
        )

    sys.exit(1 if drift_share >= args.threshold else 0)


if __name__ == "__main__":
    main()
