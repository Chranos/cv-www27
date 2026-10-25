"""Locate the reviewer's recomputation script, or record that it is not on disk.

This runs NO model, performs NO statistics and imports no producer module.  Its only job is to
answer, on the record, whether the script the independent review says it wrote
(`review-four-readings-full-check@v2` §0/§10) exists in any place this run can read,
so that a missing input is a recorded outcome instead of an unverifiable claim.

If the script is found, its path, size and sha256 are recorded and the run stops there: running
it is a separate, deliberate step (the arithmetic must be the reviewer's, not the producer's).
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

T0 = time.time()
HOME = "/path/to/home"
REPO = os.getcwd()

# Names a reviewer-authored check script would plausibly carry, and the directories it could sit in.
REPO_DIRS = ["checks", "scripts", "tools", "audit", "review", "paper4"]
REPO_FILES = ["check_four_readings.py", "recompute_four_readings.py", "four_check.py",
              "paper4/check_four.py", "paper4/recompute.py", "paper4/recompute_four.py"]
DEVICE_ROOTS = [os.path.join(HOME, "staging"), os.path.join(HOME, "runs"), "/tmp"]
DEVICE_LIMIT = 4

# Identifier strings taken from the reviewer's own earlier script (task-db4a68fc29 body.script).
IDENTIFIERS = ["binom_cdf", "cp_interval", "cluster_boot", "reviewer"]

NAME_HINTS = ("recompute", "check_four", "four_check", "four_readings", "verify_four")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def repo_candidates():
    out = []
    for d in REPO_DIRS:
        for name in REPO_FILES:
            if not name.startswith(d + "/"):
                continue
        p = os.path.join(REPO, d)
        out.append({"path": d + ("/" if os.path.isdir(p) else ""),
                    "exists": os.path.exists(p),
                    "is_dir": os.path.isdir(p),
                    "entries": sorted(os.listdir(p))[:40] if os.path.isdir(p) else None})
    for name in REPO_FILES:
        p = os.path.join(REPO, name)
        out.append({"path": name, "exists": os.path.exists(p),
                    "bytes": os.path.getsize(p) if os.path.exists(p) else None})
    return out


def device_scan():
    found, seen = [], 0
    for root in DEVICE_ROOTS:
        if not os.path.isdir(root):
            continue
        base_depth = root.rstrip("/").count("/")
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            if dirpath.count("/") - base_depth >= DEVICE_LIMIT:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames
                           if d not in (".git", "data", "node_modules", "__pycache__")]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                low = fn.lower()
                if any(h in low for h in NAME_HINTS):
                    p = os.path.join(dirpath, fn)
                    try:
                        found.append({"path": p, "bytes": os.path.getsize(p),
                                      "sha256": sha256_file(p)})
                    except OSError as e:                                    # noqa: BLE001
                        found.append({"path": p, "error": str(e)})
                seen += 1
    return found, seen


def grep_identifiers(paths):
    hits = []
    for p in paths:
        try:
            with open(p, "r", errors="replace") as f:
                txt = f.read(400000)
        except OSError:
            continue
        for ident in IDENTIFIERS:
            if ident in txt:
                hits.append({"path": p, "identifier": ident})
    return hits


def main():
    t = time.time()
    repo = repo_candidates()
    dev_found, dev_seen = device_scan()
    # only scripts that are plausibly the reviewer's: skip the producer's own paper4/four.py
    cand = [d["path"] for d in dev_found if os.path.basename(d["path"]) != "four.py"]
    hits = grep_identifiers(cand)

    out = {
        "task": "locate the reviewer's recomputation script for the four readings",
        "verdict": "script_found" if cand else "blocked_script_not_found",
        "forwards": 0,
        "gpus_used": 0,
        "mechanism": "filesystem inspection only; no model, no statistics, no producer import",
        "review": {"artifact": "review-four-readings-full-check@v2",
                   "where_it_says_the_script_exists": "§0 '我没能做到的' and §10 '升级所需' item 1",
                   "closest_stored_script": {
                       "artifact": "evidence.task-db4a68fc29@v1",
                       "body_key": "script",
                       "scope": "the two smoke runs only (sources/task-214092dd1e, "
                                "sources/task-17a8080930); no single/distortion/110-row section",
                       "usable_for_full_batch": False}},
        "search": {"repo_root": REPO, "repo": repo,
                   "device_roots": DEVICE_ROOTS, "device_depth_limit": DEVICE_LIMIT,
                   "device_py_files_scanned": dev_seen,
                   "device_name_hits": dev_found,
                   "candidate_scripts": cand,
                   "identifier_grep": {"identifiers": IDENTIFIERS, "hits": hits}},
        "not_recomputed": ["all Clopper-Pearson intervals",
                           "all video-clustered bootstrap intervals (incl. s_boot=37 and 1/42)",
                           "all sign tests",
                           "the 110-question whole-pool aggregation",
                           "cell-by-cell identity comparison"],
        "ledger_correction": {"artifact": "plan-four-readings-ledger-status@v3",
                              "content": "reading (a) lands in a-mid (not a-reverse); the producer's "
                                         "reported a-reverse was wrong; k2 is an exact question-level "
                                         "tie and its delta 0.0000 is outside the pre-written range"},
        "to_unblock": "hand over the script text or a repo path; it is then run with "
                      "executor=ssh, no gpus, 0 forwards, adjusting only path plumbing",
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "elapsed_seconds": round(time.time() - T0, 2),
        "self_reported": True,
    }
    out["phases_seconds"] = {"search_s": round(time.time() - t, 2)}
    os.makedirs("results", exist_ok=True)
    with open("results/ledger_search.json", "w") as f:
        json.dump({"repo": repo, "device_name_hits": dev_found,
                   "identifier_grep": out["search"]["identifier_grep"]}, f,
                  indent=1, sort_keys=True)
    with open("metrics.json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
    print(json.dumps({"verdict": out["verdict"], "candidates": len(cand),
                      "py_scanned": dev_seen}, ensure_ascii=False))


if __name__ == "__main__":
    main()
