#!/usr/bin/env bash
# Entry point for the four full-stage readings (Plan plan-paper-full-four-readings).
#   P4_READING = selection | single | distortion | donor
#   P4_MODE    = pilot | full
#   P4_PLAN_VERSION = the Plan version this run binds
#   P4_BACKBONE = qwen2vl | llavaov | internvl3hf (r6: internvl3hf, whose model directory
#                 comes from P4_MODEL_DIR -- the published device dataset
#                 internvl3-8b-hf-<manifest12>; both are passed in by the Task)
# Writes metrics.json and results/four_<reading>_samples.jsonl in the run directory.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${P4_READING:?set P4_READING to selection|single|distortion|donor}"
: "${P4_MODE:?set P4_MODE to pilot|full}"
: "${P4_PLAN_VERSION:?set P4_PLAN_VERSION}"
# The plan id must be passed explicitly: with a default, a run under a new plan silently
# wrote the OLD plan's id into metrics.run.plan_id (the sandbox binding was right, the
# self-reported field was not).
: "${P4_PLAN_ID:?set P4_PLAN_ID to the Plan this run binds}"
export P4_PLAN_ID
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.pipcache"
export HF_HOME="$PWD/.hfhome"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" results

PY="${PYTHON:-python3}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -m py_compile paper4/four.py paper4/backbone.py paper4/instrument.py \
    gen_common.py mvp_common.py twin5/lv_common.py twin5/floor_fresh.py twin5/qdraw.py \
    twin5/ddraw.py r4_common.py
nvidia-smi --query-gpu=index,name,memory.free --format=csv || true
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

exec "$PY" -u paper4/four.py
