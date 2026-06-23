# Dockerfile
#
# WHY THIS FILE EXISTS
# A Dockerfile is a recipe for building a container image — a self-contained
# box that includes Python, your dependencies, and your source code.
# Anyone with Docker installed can reproduce your exact environment with one
# command, regardless of whether they are on Windows, Mac, or Linux.
#
# HOW DOCKER BUILDS WORK — LAYER CACHING
# Each instruction (FROM, COPY, RUN) creates a "layer". Docker caches layers
# and only rebuilds from the first line that changed. This is why system packages
# are installed first, requirements.txt is copied second, and source code last —
# things that change rarely are near the top; things that change often are near
# the bottom. Changing a .py file reuses every layer above it.
#
# ONE IMAGE, THREE SERVICES
# The API, the monitor, and the simulator all use this same image.
# docker-compose.yml starts the API with the default CMD below and overrides
# it per service (monitor: python scripts/monitor.py, simulator: its own
# entrypoint). One recipe, three uses.


# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim is a stripped Debian image (~130 MB vs ~900 MB for the full
# image). It has no C compilers, but our wheels are pre-compiled on PyPI so
# that is fine. If a package ever fails with "missing header", add:
#   RUN apt-get update && apt-get install -y gcc g++ build-essential
#
# Avoid python:3.11-alpine — it uses a different C library (musl instead of
# glibc) that breaks numpy, scikit-learn, and other scientific packages.
FROM python:3.11-slim


# ── System packages ───────────────────────────────────────────────────────────
# git is not included in the slim image but is required by the monitor service:
# when drift is detected, export_simulation_to_parquet.py runs git commit and
# git push to update retrain.trigger and fire the GitHub Actions workflow.
# --no-install-recommends keeps the layer small by skipping optional extras.
# The rm cleans up the apt cache so it is not frozen into the image layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*


# ── Working directory ─────────────────────────────────────────────────────────
# All subsequent commands run relative to /app inside the container.
# /app is the standard convention — use it unless you have a specific reason not to.
WORKDIR /app


# ── Dependencies ──────────────────────────────────────────────────────────────
# Copy requirements.txt first — before any source code — to exploit layer caching.
# If requirements.txt has not changed, Docker reuses the cached pip install layer
# even if your .py files changed. This saves minutes on every rebuild.
COPY requirements.txt .

# --no-cache-dir tells pip not to store the downloaded packages inside the image.
# There is no point caching them — the container will never run pip install again.
# Skipping the cache saves ~50–100 MB of image size for a project like this one.
RUN pip install --no-cache-dir -r requirements.txt


# ── Source code ───────────────────────────────────────────────────────────────
# Copy only the directories the running application actually needs.
# notebooks/ is excluded — it is not needed at runtime and would bloat the image.
COPY src/        ./src/
COPY scripts/    ./scripts/
COPY params.yaml .


# ── Data directory ────────────────────────────────────────────────────────────
# The data/ directory is NOT baked into the image — it is provided at runtime
# via a Docker volume (see docker-compose.yml: ./data:/app/data).
#
# Development: the volume maps your local data/ folder into the container.
#   ai4i2020_baseline.csv is pulled from DagsHub once via `dvc pull`.
#   simulation.db is created automatically on the first simulator run.
#
# Demo repo: data/ is committed directly (baseline CSV is small and frozen),
#   so cloning the demo repo gives the container everything it needs without
#   any dvc pull step. simulation.db is still created automatically.
#
# We create the directory here so the path exists at startup even before the
# volume is mounted — avoids a FileNotFoundError on first boot.
RUN mkdir -p data reports


# ── Python path ───────────────────────────────────────────────────────────────
# sensor_simulator.py and api.py import from feature_transformation directly:
#   from feature_transformation import engineer_features
# That import only works if Python can find feature_transformation.py on its path.
# Setting PYTHONPATH to include /app/src makes every module in src/ importable
# by name, the same way it works locally when you run from the project root.
ENV PYTHONPATH=/app/src


# ── Port ──────────────────────────────────────────────────────────────────────
# EXPOSE documents which port the container listens on. It does not actually
# open the port — that happens in docker-compose.yml under `ports:`.
# Think of EXPOSE as a label for humans and tooling, not a firewall rule.
EXPOSE 8000


# ── Default command ───────────────────────────────────────────────────────────
# CMD is what runs when `docker compose up` starts this container.
# The monitor and simulator services override this in docker-compose.yml.
#
# Why 0.0.0.0 and not 127.0.0.1?
# Inside a container, 127.0.0.1 means "this container only" — requests from
# outside the container (including your browser on the host) are refused.
# 0.0.0.0 means "accept connections on all interfaces", which is what allows
# the host machine to reach the API through the mapped port.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
