# Preempt Analytics — Predictive Maintenance Capstone
## Engineering Laws & Integration Contract

These laws govern every change made to this codebase — by humans and AI assistants alike.
They exist because this project has five tightly coupled components. A change that looks
local to one file can silently break another component that runs in a completely separate
process. The laws make those couplings explicit and require they be checked before committing.

---

## THE LAWS

### Zeroth Law — Intent Fidelity
Preserve the developer's intent. When a change is ambiguous or touches a shared
contract (see Integration Contracts below), surface the risk before executing.
Never make irreversible changes — overwriting a DVC-tracked dataset, renaming a
registered MLflow model, changing the `@production` alias — without stating the
downstream effect first.

### First Law — Outcome Integrity
Every change must leave the two-loop architecture intact:

```
Inference loop : client → POST /predict (api.py) → MLflow @production → response
Retraining loop: simulation.db → export_simulation_to_csv.py → dvc repro → new model
```

A change is not complete if either loop is broken, even if the modified file passes
its own tests. Correctness means the full pipeline works end-to-end, not just the
file that was edited.

### Second Law — Elegant Sufficiency
Use the simplest change that satisfies the First Law. Complexity must be justified
by a specific integration requirement. Do not add abstraction layers, new config
files, or new dependencies unless the First Law cannot be satisfied without them.

### Third Law — Compatibility & Longevity
Maintain contract stability across the five coupled components. Where a cleaner
internal design would require changing a shared contract (column names, model
registry names, SQLite schema, DVC paths), prefer the stable design unless the
contract change is explicitly planned and all dependent files are updated in the
same commit.

### Standing Protocol — Transparency
Before any change that touches an Integration Contract (listed below), state:
  1. Which contract is affected
  2. Which other files depend on that contract
  3. Whether those files are being updated in the same change

### Standing Protocol — Educational Comments

Every file written or edited must include two layers of comments.

**Layer 1 — Section header (one per logical block)**
Write a short prose paragraph above each section explaining WHY this block exists:
what problem it solves, what the reader needs to know before reading the code,
and any non-obvious constraint or decision. Lead with the most important sentence
(Redish: front-load). Keep it to 3–5 lines maximum — if you need more, the section
is too large.

```python
# ── Load reference data ───────────────────────────────────────────────────────
# We load the training CSV here, not simulation.db, because the model was trained
# on this distribution.  Evidently needs both sides to come from the same feature
# space — if we compared raw sensor columns to engineered features, every column
# would show drift regardless of whether anything actually changed.
```

**Layer 2 — Inline comment (one per meaningful line)**
Write a short phrase to the right of each non-obvious line explaining what it does
or why. Use plain words — never jargon the reader would have to look up (Krug:
don't make me think). Aim for 5–10 words. Skip lines where the code reads like
English already (`conn.close()`, `return df`).

```python
df = pd.read_csv(csv_path)       # load the 10k-row AI4I training dataset
df = engineer_features(df)       # rename columns + compute power_kw, temp_diff, stress
df = df[FEATURES].dropna()       # keep only the 9 model inputs; drop rows with gaps
```

**What NOT to comment:**
- Lines whose variable names already explain them (`model.fit(X_train, y_train)`)
- Restatements of the code in plain English that add no new information
- Implementation details that belong in the commit message, not the source file

**Standing Protocol — Commit & Push After Every Change
Every completed change — however small — must be committed and pushed immediately.
Do not batch changes across multiple edits before committing.

**Commit message format:**
```
<short imperative summary of what changed (max 72 chars)>

<one or two sentences on WHY: what problem this solves or what it enables>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

**What counts as a good commit message:**
- Summary uses an imperative verb: "add", "fix", "update", "remove" — not "added" or "adding"
- The body answers WHY, not WHAT (the diff shows what)
- Never reference the current task, issue number, or session — those rot as the
  codebase evolves. Write for a reader who has no context beyond git log.

**Scope:** one logical change = one commit. If two files change for the same reason,
they belong in the same commit. If they change for different reasons, split them.

### Meta-Law — Conflict Resolution
Laws are ordered. When they conflict, state the conflict, justify the resolution,
and resolve in hierarchy order.

---

## INTEGRATION CONTRACTS

These are the five shared interfaces where a change in one file breaks another.
Check every applicable contract before committing.

---

### Contract 1 — Feature Engineering (HIGHEST RISK)

**Owner:** `src/feature_transformation.py`
**Dependents:** `src/modeling_pipeline.py`, `src/sensor_simulator.py`, `src/api.py`

`feature_transformation.py` is the single source of truth for what the model sees.
It is imported by both the training pipeline and both inference layers. A change here
is simultaneously a training change and a serving change.

**What is locked:**

| Symbol | What it controls | If changed without updating dependents |
|--------|-----------------|----------------------------------------|
| `FEATURES` list | Which columns reach the model | Training and inference silently disagree — wrong predictions, no error |
| `COLUMN_RENAME` dict | Raw CSV name → internal name mapping | `sensor_simulator.py` generates wrong column names; API reads fail |
| `engineer_features()` formulas | `power_kw`, `temp_diff_kelvin`, `mechanical_stress` | Training-serving skew — model trained on different values than it predicts on |

**Before changing `feature_transformation.py`:** confirm the same change is applied
in the same commit to every file that imports it.

---

### Contract 2 — MLflow Registry Names

**Owner:** `src/modeling_pipeline.py` (writes)
**Dependents:** `src/sensor_simulator.py` (reads), `src/api.py` (reads)

The two registered model names are the runtime connection between training and serving:

```
predictive-maintenance-binary      # all binary experiments register here
predictive-maintenance-multiclass  # all multiclass experiments register here
```

These names appear in three files. Renaming in `modeling_pipeline.py` without updating
`sensor_simulator.py` and `api.py` means the simulator and API silently fail to load
any model at startup — no error until the first prediction request.

**`@production` alias:** both `sensor_simulator.py` and `api.py` load
`models:/{name}@production`. If no version carries this alias, both components fail
at runtime with a non-obvious MLflow error. After registering a new version, always
confirm the alias exists before running either component.

---

### Contract 3 — SQLite Schema

**Owner:** `src/sensor_simulator.py` (defines schema, writes rows)
**Dependents:** `scripts/export_simulation_to_csv.py` (reads by column name)

The `sensor_readings` table schema is defined once in `init_db()`. The export script
reads it by column name — it does not introspect the schema. Adding, removing, or
renaming a column in `sensor_simulator.py` without updating `export_simulation_to_csv.py`
causes the export to either silently produce wrong values or raise a KeyError.

**Columns the export script depends on (do not rename without updating the script):**
`machine_type`, `air_temperature_kelvin`, `process_temperature_kelvin`,
`rotational_speed_rpm`, `torque_nm`, `tool_wear_minutes`,
`process_temperature_kelvin`, `injected_failure`

---

### Contract 4 — DVC File Paths

**Owner:** `dvc.yaml`
**Dependents:** All 12 pipeline stages

`dvc.yaml` references exact file paths as `deps`. If any of the following files are
moved or renamed, every stage that lists them as a dep becomes permanently stale and
`dvc repro` will either error or silently skip retraining:

```
src/modeling_pipeline.py      — listed as dep in all 12 stages
src/feature_transformation.py — listed as dep in all 12 stages
data/ai4i2020.csv             — listed as dep in all 12 stages (DVC-tracked)
params.yaml                   — referenced for key-level invalidation
```

**Rule:** moving any of the above files requires updating `dvc.yaml` in the same commit
and running `dvc repro --dry` to confirm no stages are unexpectedly stale.

---

### Contract 5 — CSV Column Format

**Owner:** `data/ai4i2020.csv` (DVC-tracked)
**Dependents:** `src/modeling_pipeline.py`, `src/feature_transformation.py`,
               `scripts/export_simulation_to_csv.py`

The original AI4I 2020 column names flow through every component:

```
UDI, Product ID, Type, Air temperature [K], Process temperature [K],
Rotational speed [rpm], Torque [Nm], Tool wear [min],
Machine failure, TWF, HDF, PWF, OSF, RNF
```

`export_simulation_to_csv.py` writes these exact names when appending simulated data.
`feature_transformation.py`'s `COLUMN_RENAME` dict maps them to internal names.
Any deviation between what `export_simulation_to_csv.py` writes and what
`feature_transformation.py` expects produces silent wrong features at training time.

---

## PRE-CHANGE CHECKLIST

Before committing any change, run through the applicable rows:

| Change type | Check |
|-------------|-------|
| Edit `feature_transformation.py` | All three importers updated? Training re-run needed? |
| Edit MLflow model names | `sensor_simulator.py` and `api.py` updated? `@production` alias re-set? |
| Edit SQLite schema in `sensor_simulator.py` | `export_simulation_to_csv.py` column references updated? |
| Move or rename any file in `src/` | `dvc.yaml` deps updated? `dvc repro --dry` passes? |
| Edit `params.yaml` keys | `modeling_pipeline.py` reads updated? `dvc repro --dry` reflects correct invalidation? |
| Add a new feature to `FEATURES` | Is the feature computable from raw sensor values only? ETL export updated? |
| Bump a Python dependency | Does `dvc repro` still pass? Does the simulator still load the Production model? |

---

## DANGER ZONES

Changes in these areas have caused silent failures before. Approach with extra care:

- **`set_output(transform="pandas")`** in `modeling_pipeline.py` — apply to LightGBM
  only. Applying unconditionally corrupts XGBoost's label encoder (sklearn 1.8 issue).
- **`DictVectorizer(sparse=False)`** — required for models that use `StandardScaler`.
  Removing `sparse=False` breaks the scaler silently.
- **`@production` alias vs `/Production` stage** — MLflow 2.9+ uses aliases, not stages.
  Never use the `/Production` URI format. Always use `models:/{name}@production`.
- **`dvc add` on `data/ai4i2020.csv`** — this overwrites the `.dvc` pointer. Always
  run `--dry-run` on the export script first to verify the dataset looks correct.
- **`mlflow.start_run()` vs `MlflowClient()`** — run metadata (params, metrics, tags)
  goes inside `start_run`. Registry metadata (registered model description, version tags)
  goes via `MlflowClient` outside the `with` block. Mixing them causes incomplete writes
  if the run crashes mid-execution.

---

## COMPONENT QUICK-REFERENCE

| File | Role | Reads from | Writes to |
|------|------|-----------|-----------|
| `src/feature_transformation.py` | Feature contract | — | imported by 3 files |
| `src/modeling_pipeline.py` | Training | `data/ai4i2020.csv`, `params.yaml` | MLflow registry |
| `src/sensor_simulator.py` | Inference + data gen | MLflow `@production` | `simulation.db` |
| `src/api.py` | Serving | MLflow `@production` | HTTP responses |
| `scripts/export_simulation_to_csv.py` | ETL bridge | `simulation.db` | `data/ai4i2020.csv` |
| `dvc.yaml` | Pipeline definition | `src/*.py`, `data/*.csv` | DVC cache |
| `params.yaml` | Pipeline config | — | read by `dvc.yaml` stages |
