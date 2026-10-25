#!/usr/bin/env bash
# Entry for the third-backbone feasibility probe (Plan r5).
# Read-only: no forward passes, no weight loading, no downloads, no installs.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "python: $(command -v python3)"
python3 -c 'import sys; print("sys.executable", sys.executable)'
python3 paper4/third_backbone_probe.py
