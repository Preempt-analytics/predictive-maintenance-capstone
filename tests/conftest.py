# tests/conftest.py
#
# ── Make src/ importable ─────────────────────────────────────────────────────
# Files inside src/ import each other directly by module name (e.g.
# "from feature_transformation import ...", not "from src.feature_transformation
# import ..."). src/ has no __init__.py, so it isn't a real Python package.
# Tests need the same sys.path setup api.py gives itself at runtime, or
# "import api" / "from feature_transformation import ..." would fail here.

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"  # project_root/src
sys.path.insert(0, str(SRC_DIR))                          # so tests import modules the same way api.py does
