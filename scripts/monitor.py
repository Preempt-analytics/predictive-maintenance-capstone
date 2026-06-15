# scripts/monitor.py
#
# USAGE
#   # Run directly (local dev)
#   python scripts/monitor.py
#
#   # Run inside Docker (started automatically by docker compose up)
#   docker compose up monitor
#
# WHAT THIS SCRIPT DOES
# Runs drift detection on a schedule. If drift is detected it delegates the
# entire export-and-push sequence to export_simulation_to_csv.py, which
# already contains the dvc add → dvc push → git commit → git push pipeline.
# This script is purely the scheduler and audit logger — it adds no new logic.
#
# WHY A PYTHON SCHEDULER AND NOT LINUX CRON
# Both work. The Python schedule library (already in requirements.txt) keeps
# the timing logic visible in code rather than a separate crontab file.
# It also runs in the foreground, so Docker can see the process and restart
# it if it crashes. A cron job runs silently in the background — Docker
# cannot monitor or restart individual cron jobs.
#
# THE SELF-MONITORING LOOP
#
#   monitor.py runs               (every night at 02:00)
#        |
#        v
#   detect_drift.py runs          (compares simulation.db vs baseline CSV)
#        |
#        +-- no drift --> log PASS to monitor_log.jsonl, sleep until next run
#        |
#        +-- drift detected
#              |
#              v
#         export_simulation_to_csv.py --purge --push --retrain
#              |   appends rows to CSV, clears DB, dvc add/push,
#              |   writes retrain.trigger, git commit + git push
#              v
#         GitHub Actions picks up the push      (retrain.yml triggers)
#              |
#              v
#         dvc repro + promote_model.py          (retrain + auto-promote if gates pass)
#              |
#              v
#         log result to monitor_log.jsonl


import json                                    # serialise log entries as single-line JSON
import subprocess
from datetime import datetime, timezone        # UTC timestamps for log entries
from pathlib import Path

import schedule    # schedule library — already in requirements.txt
import time

# ── Paths ─────────────────────────────────────────────────────────────────────
# ROOT resolves to the project directory regardless of where the script is called from.
# All subprocess calls use cwd=ROOT so relative paths inside those scripts work.
ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "reports" / "monitor_log.jsonl"  # one JSON line appended per run; never overwritten


# ── Run log ───────────────────────────────────────────────────────────────────
# Every scheduled run appends one JSON line to reports/monitor_log.jsonl.
# This gives a persistent, human-readable audit trail without any extra
# dependencies. Docker logs scroll away on restart; this file does not.
# Fields: timestamp (UTC ISO-8601), drift_detected, retrain_triggered.

def _append_log(drift_detected: bool, retrain_triggered: bool) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)   # create reports/ if absent
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),  # UTC so log is timezone-safe
        "drift_detected": drift_detected,
        "retrain_triggered": retrain_triggered,
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")   # one line per run; never truncates existing entries


# ── Drift check ───────────────────────────────────────────────────────────────
# This function runs on every scheduled tick. It calls detect_drift.py as a
# subprocess rather than importing it — keeping the two scripts independent and
# making it easy to test detect_drift.py on its own without the scheduler.

def check_drift() -> None:
    print("\n" + "—" * 60)
    print("  Drift check starting...")
    print("—" * 60)


    # ── Step 1: Run drift detection ───────────────────────────────────────────
    # detect_drift.py exits with code 0 (no drift) or 1 (drift detected).
    # We read the exit code to decide whether to trigger the export.
    result = subprocess.run(
        ["python", "scripts/detect_drift.py"],
        cwd=ROOT,
    )

    if result.returncode == 0:
        print("  No drift detected. Model distribution is stable.")
        print("  Next check scheduled per configured interval.")
        _append_log(drift_detected=False, retrain_triggered=False)   # record the PASS
        return

    # ── Step 2: Export simulation data ───────────────────────────────────────────
    # Drift confirmed. export_simulation_to_csv.py does one thing: append rows
    # from simulation.db to the training CSV and optionally purge exported rows.
    # The dvc/git orchestration is handled here in Steps 3 and 4 — not delegated
    # to the export script, which stays focused on its single responsibility.
    print("  Drift detected. Exporting simulation data...")
    export_result = subprocess.run(
        ["python", "scripts/export_simulation_to_csv.py", "--purge"],
        cwd=ROOT,
    )

    if export_result.returncode != 0:
        print("  ERROR: Export failed. Will retry on next scheduled run.")
        _append_log(drift_detected=True, retrain_triggered=False)
        return

    # ── Step 3: Update DVC ────────────────────────────────────────────────────
    # The training CSV has new rows. dvc add recomputes its content hash and
    # updates the .dvc pointer file. dvc push uploads the new CSV to DagsHub
    # so GitHub Actions can pull it during the retrain job.
    print("  Updating DVC pointer and pushing to DagsHub...")
    subprocess.run(["dvc", "add", "data/ai4i2020.csv"], cwd=ROOT, check=True)
    subprocess.run(["dvc", "push", "data/ai4i2020.csv"], cwd=ROOT, check=True)

    # ── Step 4: Commit and push to trigger GitHub Actions ────────────────────
    # retrain.yml watches for changes to retrain.trigger, not ai4i2020.csv.dvc.
    # Writing the current UTC timestamp gives git a real change to detect —
    # an empty touch would work too, but the timestamp makes the log readable.
    print("  Committing retrain trigger and pushing to GitHub...")
    (ROOT / "retrain.trigger").write_text(datetime.now(timezone.utc).isoformat())
    subprocess.run(["git", "add", "data/ai4i2020.csv.dvc", "retrain.trigger"], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", "auto: drift detected, retraining triggered"], cwd=ROOT, check=True)
    subprocess.run(["git", "push"], cwd=ROOT, check=True)

    print("  Retrain triggered. GitHub Actions will pick this up shortly.")
    _append_log(drift_detected=True, retrain_triggered=True)


# ── Schedule ──────────────────────────────────────────────────────────────────
# The schedule library uses a simple chained API:
#   schedule.every().day.at("02:00").do(check_drift)   ← production
#   schedule.every(5).minutes.do(check_drift)           ← demo / local dev
#
# Switch to the production line before deploying. The demo line runs frequently
# so you can verify the full pipeline end-to-end without waiting until 02:00.
#
# Production (uncomment when deploying):
# schedule.every().day.at("02:00").do(check_drift)
#
# Demo (comment out when deploying):
schedule.every(5).minutes.do(check_drift)


# ── Main loop ─────────────────────────────────────────────────────────────────
# schedule.run_pending() checks whether any scheduled jobs are due and runs them.
# It returns immediately — it does not block. idle_seconds() tells us exactly
# how long until the next job is due, so we sleep precisely that long rather
# than waking up every 60 seconds to find nothing to do.
if __name__ == "__main__":
    print("Preempt Analytics — Drift Monitor")
    print(f"  Log file : {LOG_PATH}")
    print("Scheduled checks configured. Running first check now...")

    # Run once immediately at startup so you can verify the pipeline works
    # without waiting for the first scheduled tick.
    check_drift()

    while True:
        schedule.run_pending()          # fire any jobs whose time has come
        idle = schedule.idle_seconds()  # seconds until the next scheduled job
        time.sleep(max(idle, 1))        # sleep exactly that long; floor at 1s to avoid busy-spin
