#!/usr/bin/env bash
# One Task, one Evidence: the r6 instrument pilot (Plan
# plan-four-readings-second-backbone-r6, run matrix rows 2 and 3).
#
#   * every reading runs stage=smoke, decides_claim=false, on P4_BACKBONE
#     (default internvl3hf, the r6 second backbone; the SAME script on
#     P4_BACKBONE=qwen2vl is the criterion's own positive control -- the first
#     backbone has never run this instrument check before);
#   * the v4 naming-step stimulus is UNCHANGED: channel A prints the option
#     texts only, the assistant turn is opened by a forced prefix, the first
#     generated token cannot be a bare letter (ids computed from the
#     vocabulary), a bare-letter reply gets exactly one escalation, and the
#     letter is never turned into option text; channel B is the test-retest
#     noise floor; the fault injection goes through the same comparison;
#   * r6 changes, all registered before this run: the corrected `fired`
#     boolean, the 21-unit denominator with the stable/order-specific split,
#     and mapping_ok = (n_pairs>0) AND all pairs evaluable AND zero
#     contradictions (the registered zero line; the r3 statistical line stays
#     reported beside it);
#   * the registered pilot structure is 6/6/3/8 measured units and 42
#     control pairs = 21 units x 2 option orders (selection 6+6, single 6+6,
#     distortion 3+3, donor 8 measured with the first 6 in the perm control).
#     Under the current four.py (Plan -r4 §15: the single cell's perm control
#     covers every MEASURED unit) the single cell reaches its registered 6
#     units only through the registered draw at smoke size, so this script
#     exports P4_QSELECT=draw and P4_QDRAW_SMOKE_N=6 FOR THE SINGLE READING
#     ONLY; selection/distortion/donor run exactly as in the v4 smoke.
#
# This is NOT a full run and cannot be: the pilot gate has to be recorded
# against this Plan version first.  Nothing here decides a claim.
set -uo pipefail
cd "$(dirname "$0")/.."

# The plan version is never hard-coded here: a run must stamp the version it is actually bound
# to.  task-f0d4a231c4 was bound to r3@v3 but wrote plan_version=1 into its own metrics from a
# stale constant in this file, so the Task now passes it in.
PLAN_ID=${P4_PLAN_ID:-plan-four-readings-second-backbone-r6}
PLAN_VERSION=${P4_PLAN_VERSION:-1}

export TMPDIR="$PWD/.tmp" PIP_CACHE_DIR="$PWD/.pipcache" HF_HOME="$PWD/.hfhome"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" results runs

PY="${PYTHON:-python3}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export P4_BACKBONE=${P4_BACKBONE:-internvl3hf} P4_MODE=pilot
export P4_PLAN_ID="$PLAN_ID" P4_PLAN_VERSION="$PLAN_VERSION"
# internvl3hf resolves its model directory from P4_MODEL_DIR (the published
# device dataset internvl3-8b-hf-<manifest12>); qwen2vl ignores it.
if [ -n "${P4_MODEL_DIR:-}" ]; then export P4_MODEL_DIR; fi

"$PY" -m py_compile paper4/four.py paper4/backbone.py paper4/instrument.py \
    paper4/aggregate_smoke.py gen_common.py mvp_common.py twin5/lv_common.py \
    twin5/floor_fresh.py twin5/qdraw.py twin5/ddraw.py r4_common.py || exit 1
nvidia-smi --query-gpu=index,name,memory.free --format=csv || true
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

RC=0
declare -A RCS
for R in selection single distortion donor; do
  echo "=== reading=$R mode=pilot backbone=$P4_BACKBONE plan=$PLAN_ID@$PLAN_VERSION ==="
  mkdir -p "runs/$R"
  if [ "$R" = "single" ]; then
    # registered 42-pair structure (see header): the single cell measures the
    # first 6 units of the registered draw, and its perm control covers them
    P4_READING="$R" P4_QSELECT=draw P4_QDRAW_SMOKE_N=6 \
      P4_METRICS="runs/$R/metrics.json" "$PY" -u paper4/four.py
  else
    P4_READING="$R" P4_METRICS="runs/$R/metrics.json" "$PY" -u paper4/four.py
  fi
  rc=$?
  RCS[$R]=$rc
  echo "exit $R = $rc"
  if [ "$rc" -ne 0 ]; then RC=$rc; fi
done

"$PY" - "$PLAN_ID" "$PLAN_VERSION" "${RCS[selection]}" "${RCS[single]}" \
    "${RCS[distortion]}" "${RCS[donor]}" <<'PYEOF'
import json
import sys

plan_id, plan_version, a, b, c, d = sys.argv[1:7]
codes = {"selection": int(a), "single": int(b), "distortion": int(c), "donor": int(d)}
with open("runs/_exit_codes.json", "w") as f:
    json.dump({"plan_id": plan_id, "plan_version": int(plan_version), "exit_codes": codes},
              f, indent=1, sort_keys=True)
print(json.dumps({"exit_codes": codes}))
PYEOF

"$PY" paper4/aggregate_smoke.py
ARC=$?
if [ "$ARC" -ne 0 ] && [ "$RC" -eq 0 ]; then RC=$ARC; fi
exit "$RC"
