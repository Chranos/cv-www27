#!/usr/bin/env bash
# Entry point for the exploratory, post-hoc additions of
# plan-exploratory-bolt-timing (stage=smoke, decides_claim=false):
#   P4_EXPLORATORY = bolt   -> exploratory_bolt.py  (official BOLT-ITS arm + uniform@16)
#   P4_EXPLORATORY = timing -> exploratory_timing.py (step timers over the existing arms)
# Env: P4_PLAN_ID / P4_PLAN_VERSION bind the run; P4_BOLT_PILOT_N>0 shrinks bolt to a
# pipeline pilot (first N questions in fixed qid order; no reading).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${P4_EXPLORATORY:?set P4_EXPLORATORY to bolt|timing}"
: "${P4_PLAN_ID:?set P4_PLAN_ID to the Plan this run binds}"
: "${P4_PLAN_VERSION:?set P4_PLAN_VERSION}"
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.pipcache"
export HF_HOME="$PWD/.hfhome"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" results

PY="${PYTHON:-python3}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -m py_compile paper4/exploratory_bolt.py paper4/exploratory_timing.py \
    paper4/four.py paper4/backbone.py paper4/instrument.py \
    gen_common.py mvp_common.py twin5/lv_common.py
nvidia-smi --query-gpu=index,name,memory.free --format=csv || true
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

if [ "$P4_EXPLORATORY" = "bolt" ]; then
  exec "$PY" -u paper4/exploratory_bolt.py
elif [ "$P4_EXPLORATORY" = "timing" ]; then
  exec "$PY" -u paper4/exploratory_timing.py
else
  echo "unknown P4_EXPLORATORY=$P4_EXPLORATORY" >&2
  exit 2
fi
