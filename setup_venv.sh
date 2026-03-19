#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_venv.sh  —  Create virtual environment and install all dependencies
# Usage:  bash setup_venv.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VENV_DIR=".venv"
PYTHON=${PYTHON:-python3}

echo "==> Creating virtual environment in $VENV_DIR ..."
$PYTHON -m venv "$VENV_DIR"

echo "==> Activating venv ..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip (via the venv Python binary to avoid version mismatch) ..."
"$VENV_DIR/bin/python3.11" -m pip install --upgrade pip

echo "==> Installing project dependencies (requirements.txt) ..."
"$VENV_DIR/bin/python3.11" -m pip install -r requirements.txt

echo "==> Installing test + demo dependencies ..."
"$VENV_DIR/bin/python3.11" -m pip install pytest pytest-cov lap

echo ""
echo "✓  Virtual environment ready."
echo ""
echo "To activate:              source $VENV_DIR/bin/activate"
echo "To run tests:             python3.11 -m pytest"
echo "To run loitering demo:    python3.11 demo_loitering.py"
echo "To generate test images:  python3.11 tests/generate_test_images.py"
echo ""
echo "NOTE: Always use 'python3.11' (not 'python') inside this venv"
echo "      to ensure packages install and run in the same interpreter."
