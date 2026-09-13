#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3.11}
VENV=${VENV:-"$ROOT/.venv"}
"$PYTHON" -c 'import sys,platform; assert sys.version_info[:2] == (3,11), "Use Python 3.11"; assert platform.system() == "Linux" and platform.machine() == "x86_64", "CUDA lock targets Linux x86_64"'
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
# Preserve the observed runtime, not the incompatible upstream dependency ranges.
"$VENV/bin/python" -m pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r "$ROOT/requirements-linux-cu121.lock.txt"
"$VENV/bin/python" -m pip install --no-deps -e "$ROOT/LLaMA-Factory"
"$VENV/bin/python" -m pip install -r "$ROOT/requirements-test.txt"
CUDA_VISIBLE_DEVICES="" "$VENV/bin/python" "$ROOT/scripts/self_test.py"
printf '\nReady: %s\n' "$VENV/bin/python"
