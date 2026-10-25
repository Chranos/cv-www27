#!/usr/bin/env bash
# Backbone probe entry point (no reading; plumbing + measured rates only).
#   P4_BACKBONE   = qwen2vl | llavaov | internvl3hf   (r6 second backbone: internvl3hf,
#                  which resolves its model directory from P4_MODEL_DIR)
#   P4_PLAN_VERSION = the Plan version this run binds
set -euo pipefail
cd "$(dirname "$0")/.."

: "${P4_BACKBONE:?set P4_BACKBONE to qwen2vl|llavaov}"
: "${P4_PLAN_VERSION:?set P4_PLAN_VERSION}"
: "${P4_PLAN_ID:?set P4_PLAN_ID to the Plan this run binds}"

export P4_PLAN_ID
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.pipcache"
export HF_HOME="$PWD/.hfhome"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" results

PY="${PYTHON:-python3}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -m py_compile paper4/backbone.py paper4/probe.py gen_common.py mvp_common.py
nvidia-smi --query-gpu=index,name,memory.free --format=csv || true
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

exec "$PY" -u paper4/probe.py
