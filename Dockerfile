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
# and only rebuilds from the first line that changed. This is why requirements.txt
# is copied BEFORE the source code — if you only change a .py file, Docker
# reuses the cached pip install layer and the rebuild takes seconds, not minutes.
#
# ONE IMAGE, TWO SERVICES
# Both the API and the monitor use this same image. docker-compose.yml starts
# the API with the default CMD below, and overrides it for the monitor service
# with `command: python scripts/monitor.py`. One recipe, two uses.


# ── Base image ────────────────────────────────────────────────────────────────
# The base image is the starting point — it comes with an OS and Python
# pre-installed so you don't have to set those up yourself.
#
# TODO A — Understand the image tag choices:
#   python:3.11        → full Debian image, ~900 MB. Includes compilers and
#                        system libraries you probably don't need at runtime.
#   python:3.11-slim   → stripped Debian, ~130 MB. No compilers. Most pure-Python
#                        packages install fine; packages with C extensions (like
#                        numpy, scikit-learn) may need build tools added.
#   python:3.11-alpine → even smaller, but a different libc — often breaks
#                        scientific Python packages. Avoid for ML projects.
#
# We use slim. If a package fails to install with a "missing header" error,
# add:  RUN apt-get update && apt-get install -y gcc g++ build-essential
FROM python:3.11-slim


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
# We deliberately exclude notebooks/, archive/, and docs/ — they are not needed
# at runtime and would bloat the image unnecessarily.
COPY src/        ./src/
COPY scripts/    ./scripts/
COPY params.yaml .

# TODO B — Data: the training Parquet is DVC-tracked and lives on DagsHub, not in git.
# The container needs the data to exist at data/ai4i2020.parquet for the API to load
# a model and for drift detection to work.
#
# Two options — pick one:
#
#   Option 1 — Mount at runtime (recommended for development):
#     Add a volume in docker-compose.yml that maps your local ./data to /app/data.
#     The Parquet file is never baked into the image. Changes on the host are visible
#     immediately without rebuilding. See docker-compose.yml for where to add this.
#
#   Option 2 — Pull at build time (for a fully self-contained image):
#     Add these lines before CMD:
#       COPY .dvc/  ./.dvc/
#       RUN dvc pull data/ai4i2020.parquet data/ai4i2020_baseline.csv
#     Requires DVC credentials to be available at build time (risky — credentials
#     in build args can leak into image history). Only use for CI/CD pipelines
#     where you control the build environment.
#
# For now, we create the directory so the path exists at startup:
RUN mkdir -p data reports


# ── Python path ───────────────────────────────────────────────────────────────
# sensor_simulator.py imports feature_transformation directly:
#   from feature_transformation import engineer_features
# That import only works if Python can find feature_transformation.py on its path.
# Setting PYTHONPATH to include /app/src makes every module in src/ importable
# by name, the same way it works locally when you run from the project root.
ENV PYTHONPATH=/app/src


# ── Port ──────────────────────────────────────────────────────────────────────
# EXPOSE documents which port the container listens on. It does not actually
# open the port — that happens in docker-compose.yml under `ports:`.
# Think of EXPOSE as a label for humans and tooling, not a firewall rule.
#
# TODO C — What port does uvicorn listen on in this project?
# Hint: look at the CMD below. The port here should match.
EXPOSE 8000


# ── Default command ───────────────────────────────────────────────────────────
# CMD is what runs when `docker compose up` starts this container.
# The monitor service overrides this in docker-compose.yml with its own command.
#
# Why 0.0.0.0 and not 127.0.0.1?
# Inside a container, 127.0.0.1 means "this container only" — requests from
# outside the container (including your browser on the host) are refused.
# 0.0.0.0 means "accept connections on all interfaces", which is what allows
# the host machine to reach the API through the mapped port.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
