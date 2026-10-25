#!/usr/bin/env bash
# Entry point: run the reviewer's recomputation script VERBATIM over the archived
# outputs of the four full runs (+ three archives), and record the script's hash
# before and after so the logic can be shown untouched.
#
# The script is review review-recompute-four-readings-script@v1, section 2.
# Its own invocation, from that review's section 3.2:
#     python3 recompute_four_readings.py --root . --out metrics.json
# Nothing in recompute_four_readings.py is edited here. If a file layout ever needs
# pointing at, it is done with the script's own --dir/--root flags and every such
# change is appended to paths_edited in script_hashes.json.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCRIPT="recompute/recompute_four_readings.py"
OUT="metrics.json"
HASHES="script_hashes.json"
STDOUT_LOG="recompute_stdout.txt"

# ---- pick an interpreter that actually has numpy and scipy (no installs, no network)
PY_CANDIDATES=(
  "${EXP_PYTHON:-}"
  "python3"
  "python"
)
PY=""
for cand in "${PY_CANDIDATES[@]}"; do
  [ -n "$cand" ] || continue
  command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ] || continue
  if "$cand" -c 'import numpy, scipy' >/dev/null 2>&1; then
    PY="$cand"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "no interpreter with numpy+scipy found" >&2
  exit 4
fi

# ---- hash before the run
SHA_BEFORE="$(sha256sum "$SCRIPT" | awk '{print $1}')"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

rm -f "$OUT"

# ---- the run: the reviewer's documented invocation, nothing added
"$PY" "$SCRIPT" --root "$ROOT" --out "$OUT" > "$STDOUT_LOG" 2>&1
RC=$?
cat "$STDOUT_LOG"

# ---- hash after the run
SHA_AFTER="$(sha256sum "$SCRIPT" | awk '{print $1}')"

"$PY" - "$HASHES" "$SCRIPT" "$SHA_BEFORE" "$SHA_AFTER" "$GIT_COMMIT" "$RC" <<'PYEOF'
import json, os, sys, platform
(out, script, before, after, commit, rc) = sys.argv[1:7]
import numpy, scipy
meta = {
    "script": script,
    "sha256_before_run": before,
    "sha256_after_run": after,
    "unchanged": before == after,
    "git_commit": commit,
    "invocation": "%s %s --root <repo-root> --out metrics.json" % (platform.python_implementation(), script),
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "exp_seed": os.environ.get("EXP_SEED"),
    "paths_edited": [],
    "paths_edited_note": ("the script's default layout (sources/<task_id>/...) is exactly "
                          "what read_from produced, so no path was changed; the run used only "
                          "--root/--out, both of which are the script's own flags"),
    "read_from": ["task-23cd232b2b", "task-88f6a1123d", "task-259e5c2b3a",
                  "task-07c9beea59", "task-ca0a24e61b", "task-67bfc74ca9",
                  "task-68cebbe16b"],
    "script_exit_code": int(rc),
    "metrics_written": os.path.exists("metrics.json"),
}
with open(out, "w") as f:
    json.dump(meta, f, indent=1)
print("wrote", out, "script_sha256=" + before, "unchanged=" + str(before == after))
PYEOF

if [ ! -s "$OUT" ]; then
  echo "the script did not write $OUT (exit $RC)" >&2
  exit 5
fi
exit 0
