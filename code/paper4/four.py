"""Full-stage battery for the four readings the measurement paper rests on.

Plan: plan-paper-full-four-readings-r3 (version bound by P4_PLAN_VERSION).

READING=selection   claims (a)+(b): LongVideoBench x Qwen2-VL, 32-frame pool, B=16,
                    uniform@16 vs the option-contrast selector under three in-run noise
                    streams vs no_frames vs the zero-information option prior (computed from
                    annotations before the model is loaded, hashed, and reported as the first
                    column of every table).  Unit = video, questions clustered in videos.
READING=single      claim (c): LVBench 24 confirmation video units, RE-EXECUTED (first
                    measured in task-67bfc74ca9), uniform@16 vs single_frame_j_star vs
                    no_frames vs the option prior.  Reported as a bounded difference; no
                    equivalence claim, no unit is claimed untouched.
READING=distortion  claim (d1): LVBench old 20-video copy, 17 pairs, behavioural
                    destruction: clean j* vs distorted j* vs distorted non-evidence frame,
                    two operators (patchcopy, noise) x three in-run streams.  Family = 6
                    same-question comparisons (Holm); the bar is listed separately.
                    Question selection: P4_QSELECT=draw runs ddraw-v1, the registered
                    per-video uniform draw (plan-distortion-random-question); the
                    default minuid keeps the archived smallest-uid behaviour on record.
READING=donor       (d2) instrument diagnostic ONLY: carrier 3B, 23 in-force visible
                    questions, donor_floor_probe@16 vs no_frames vs the option prior.  No
                    finding, no delta, no decision chain; >4 visible exclusions raises
                    (exit 70) before any model step.
READING=guard_selftest  exercises the exclusion-budget crash path with the COUNT only.

MODE=pilot shrinks n for selection/distortion; the single reading always runs the 24
confirmation units so its identity control can be complete.

Nothing here is averaged across arms, nothing is hard-coded, and every forward is counted by
gen_common.Fwd.  stage=smoke for pilot runs, stage=full for reported runs; a pilot writes
decides_claim=false.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

import backbone as BK
import gen_common as G
import instrument as INS
import mvp_common as C
import twin5.lv_common as L
import twin5.floor_fresh as F
import twin5.qdraw as QD
import twin5.ddraw as DD

READING = os.environ.get("P4_READING", "").strip()
MODE = os.environ.get("P4_MODE", "pilot").strip()
PLAN_ID = os.environ.get("P4_PLAN_ID", "plan-paper-full-four-readings-r3")
PLAN_VERSION = int(os.environ.get("P4_PLAN_VERSION", "1"))
# Which weights.  The only thing a backbone swap changes; every arm, cell, pin and branch
# rule below is the code that already ran on the first backbone.
BACKBONE = os.environ.get("P4_BACKBONE", "qwen2vl").strip()
EXP_SEED = int(os.environ.get("EXP_SEED", "0"))
# Question selection for READING=single (qdraw-v1) and READING=distortion (ddraw-v1).
# "minuid" is the archived behaviour (smallest integer uid among the rule-selected questions)
# and remains the default, so every earlier binding of this script is unchanged.  "draw" is a
# per-video uniform draw over the frozen candidate set with a seed fixed in the Plan:
#   READING=single      qdraw-v1 of plan-single-frame-random-question (twin5/qdraw.py)
#   READING=distortion  ddraw-v1 of plan-distortion-random-question (twin5/ddraw.py),
#                       registered to replace that cell's archived smallest-uid rule; the
#                       archived deterministic reading (task-259e5c2b3a) stays on record
QSELECT = os.environ.get("P4_QSELECT", "minuid").strip()
QDRAW_SEED = int(os.environ.get("P4_QDRAW_SEED", str(QD.QDRAW_SEED)))
DDRAW_SEED = int(os.environ.get("P4_DDRAW_SEED", str(DD.DDRAW_SEED)))
# >0 measures only the first N registered units in fixed video-id order (pilot only): the draw
# and the structure checks still cover all 24, only the model steps shrink.
QDRAW_SMOKE_N = int(os.environ.get("P4_QDRAW_SMOKE_N", "0") or 0)
STAGE = "full" if MODE == "full" else "smoke"
DECIDES = bool(MODE == "full")

DATA = os.environ.get("DATA_ROOT", "/path/to/data")
LVB_DIR = os.path.join(DATA, "lvb-val-e383ac257332")
C3B_DIR = os.path.join(DATA, "mlvu-needle-plotqa-38e93f060520")
C3B_PIN_REL = "data/carrier3b_pin_r3.json"
C3B_PIN_SHA = "67563e778458ea50c2981103cd81cfe5af7efed00faef1949db5b83a76e7145c"
PINNED_QSET_SHA = "c1ee771fb2f963c9367ccb98d9b75c3770c55a2eb305abfbc212a2ef94050075"
LVB_FULL_Q = 110
LVB_FULL_V = 56
FRESH_CONFIRM_N = 24
BAR_FRAC = 0.35
SIGMA = 0.2
KS = (0, 1, 2)

# Default is the repository-root metrics.json the sandbox reads.  A smoke that runs several
# readings in one Task points each of them at runs/<reading>/metrics.json and aggregates.
METRICS = os.environ.get("P4_METRICS", "metrics.json")
T0 = time.time()
PHASE = {"identity_s": 0.0, "model_load_s": 0.0, "decode_s": 0.0, "forward_s": 0.0}
FWD = G.Fwd()
NOTES = []
# Per-unit vision-token record for the arm calls of the unit being measured (r3 §3.2).  It is
# filled by ask() and by the scoring calls from the very input_ids each call fed, and copied
# into the unit's sample row, so a full run carries a measured count per unit and per arm
# instead of one representative count per run.
TOKENS = {}


def log(*a):
    print(*a, flush=True)


def app_jsonl(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def jdump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)


def load_jsonl(path):
    if not os.path.exists(path):
        return None
    rows = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        keys = []
        if "qid" in o:
            keys.append(("qid", o["qid"]))
        if "video_id" in o and "pair" in o:
            keys.append(("vp", o["video_id"], int(o["pair"])))
        for k in keys:
            rows[k] = o
    return rows


def _row_keys(r):
    keys = [("qid", r.get("qid"))] if r.get("qid") is not None else []
    if r.get("video_id") is not None and r.get("pair") is not None:
        keys.append(("vp", r["video_id"], int(r["pair"])))
    return keys


def archive_identity(rows, rel, arm_map):
    """Positive control: the deterministic arms rewritten by this run must reproduce the
    archived letters cell by cell.  Rows are matched by qid when the archive has one, and by
    (video_id, pair) otherwise (the r6 carrier archive keys on the pair, not on a qid)."""
    arc = load_jsonl(rel)
    if arc is None:
        return {"available": False, "path": rel, "note": "archive not staged (read_from)"}
    n = 0
    per_arm = {}
    mism = []
    for r in rows:
        a = None
        for k in _row_keys(r):
            if k in arc:
                a = arc[k]
                break
        if a is None:
            continue
        n += 1
        letters = a.get("letters") or {}
        for mine, theirs in arm_map.items():
            want = letters.get(theirs, a.get(theirs))
            if want is None:
                continue
            got = r["letters"].get(mine)
            per_arm.setdefault(mine, [0, 0])
            per_arm[mine][1] += 1
            if got == want:
                per_arm[mine][0] += 1
            else:
                mism.append({"qid": r.get("qid"), "pair": r.get("pair"), "arm": mine,
                             "this": got, "archive": want})
    out = {"available": True, "path": rel, "n_overlap": n,
           "agreement": {k: {"agree": v[0], "n": v[1]} for k, v in per_arm.items()},
           "n_mismatch": len(mism), "mismatches": mism[:20]}
    if len(mism) > 20:
        # The printed list is capped at 20; record the cap so n_mismatch and the list length
        # cannot be read as contradicting each other (review §3, last line).
        out["list_truncated_at"] = 20
    return out


# ---------------------------------------------------------------- statistics

def bar_hits(n):
    return int(math.floor(BAR_FRAC * n))


def arm_entry(vals):
    """vals: list of 0/1 (None counts as wrong and is reported separately)."""
    n = len(vals)
    hits = int(sum(int(v) for v in vals if v is not None))
    return {"hits": hits, "n": n, "value": (float(hits) / n if n else None),
            "cp95": L.cp_ci(hits, n)}


def mcnemar(a, b):
    return C.mcnemar_exact(a, b)


def sign_test(diffs):
    from scipy.stats import binomtest
    pos = int(sum(1 for d in diffs if d > 0))
    neg = int(sum(1 for d in diffs if d < 0))
    if pos + neg == 0:
        return {"n_pos": 0, "n_neg": 0, "p": 1.0}
    return {"n_pos": pos, "n_neg": neg,
            "p": float(binomtest(min(pos, neg), pos + neg, 0.5).pvalue)}


def per_video_means(rows, key):
    by = {}
    for r in rows:
        by.setdefault(r["video_id"], []).append(int(r["correct"][key]))
    return {v: float(np.mean(x)) for v, x in by.items()}


def cluster_boot(rows, key_a, key_b, s_boot, n_boot=10000):
    """Video-clustered bootstrap of mean(correct_a - correct_b)."""
    by = {}
    for r in rows:
        by.setdefault(r["video_id"], []).append(
            (int(r["correct"][key_a]), int(r["correct"][key_b])))
    vids = sorted(by)
    if not vids:
        return [None, None]
    per = np.array([np.mean([a - b for a, b in by[v]]) for v in vids], float)
    rng = np.random.default_rng(int(s_boot))
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    means = per[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def compare(rows, key_a, key_b, s_boot):
    a = [int(r["correct"][key_a]) for r in rows]
    b = [int(r["correct"][key_b]) for r in rows]
    d = [x - y for x, y in zip(a, b)]
    va = per_video_means(rows, key_a)
    vb = per_video_means(rows, key_b)
    vd = [va[v] - vb[v] for v in sorted(va)]
    return {"mean_diff": float(np.mean(d)) if d else None,
            "cluster_boot95": cluster_boot(rows, key_a, key_b, s_boot),
            "mcnemar": mcnemar(a, b),
            "video_sign_test": sign_test(vd),
            "n_videos": len(vd)}


def constant_baseline(rows, gold_key="gold"):
    from collections import Counter
    cnt = Counter(r[gold_key] for r in rows)
    if not cnt:
        return None
    best_letter, best = max(sorted(cnt.items()), key=lambda kv: kv[1])
    return {"gold_letter_counts": dict(sorted(cnt.items())), "best_letter": best_letter,
            "best_letter_hits": int(best), "n": len(rows),
            "value": float(best) / len(rows)}


def extraction_block(rows, keys):
    n_none = 0
    bad = []
    for r in rows:
        for k in keys:
            v = r["letters"].get(k)
            if v is None:
                n_none += 1
                bad.append({"qid": r["qid"], "arm": k})
            elif v not in list("ABCDE"):
                bad.append({"qid": r["qid"], "arm": k, "letter": v})
    return {"n_none_letter_cells": n_none, "n_bad": len(bad), "examples": bad[:20],
            "rule": "mvp_common.parse_letter: first A-Ea-e uppercased; None counts as wrong "
                    "and is not dropped"}


def model_block(minfo, mident):
    out = dict(minfo)
    out["model_identity"] = mident
    return out


def holm_local(pairs, alpha=0.05):
    """Holm step-down over a pre-declared family.  pairs: list of (name, p)."""
    order = sorted([(float(p), str(n)) for n, p in pairs], key=lambda t: t[0])
    m = len(order)
    out = {}
    prev = True
    for i, (p, name) in enumerate(order):
        thr = alpha / float(m - i)
        reject = bool(prev and p <= thr)
        prev = reject
        out[name] = {"p": p, "threshold": thr, "reject": reject,
                     "p_holm": min(1.0, p * float(m - i))}
    return {"alpha": alpha, "family_size": m, "decisions": out}


def prior_block(golds, name, expect_letter=None, expect_hits=None):
    """The zero-information option prior, computed from annotations only and written to
    prerun/ BEFORE any model step, then hashed.

    Tie-break (frozen, never looks at a model output): the alphabetically first letter among
    the maxima.  Carrier 4 is a four-way tie by construction, so the rule has to be written
    down rather than left to whatever max() does.
    """
    from collections import Counter
    cnt = Counter(golds)
    best = max(sorted(cnt.items()), key=lambda kv: kv[1])[0] if cnt else None
    hits = int(cnt[best]) if best is not None else 0
    out = {"name": name, "n": len(golds),
           "gold_letter_counts": dict(sorted(cnt.items())),
           "prior_letter": best, "prior_hits": hits,
           "prior_value": (float(hits) / len(golds) if golds else None),
           "tie": bool(best is not None and
                       sum(1 for v in cnt.values() if v == cnt[best]) > 1),
           "tie_break_rule": "alphabetically first letter among the maxima",
           "computed_from": "gold annotation only; no model output, no forward pass",
           "before_model_load": True, "forwards": 0}
    if expect_letter is not None and best != expect_letter:
        raise SystemExit("prior calibration failed for %s: letter %s != %s"
                         % (name, best, expect_letter))
    if expect_hits is not None and hits != expect_hits:
        raise SystemExit("prior calibration failed for %s: %d hits != %d"
                         % (name, hits, expect_hits))
    rel = "prerun/const_prior_%s.json" % name
    jdump(out, rel)
    out["path"] = rel
    out["sha256"] = hashlib.sha256(open(rel, "rb").read()).hexdigest()
    return out


def enforce_destruction_budget(n_excluded, budget, where):
    """Pre-registered control: more than `budget` in-force visible questions excluded means
    crash_destruction_unverified and the cell produces NO readings.  This raises (exit 70);
    it is not a string in a metrics field."""
    if int(n_excluded) > int(budget):
        print("crash_destruction_unverified: %s excluded %d > budget %d"
              % (where, int(n_excluded), int(budget)), flush=True)
        raise SystemExit(70)


def guard_selftest():
    """Exercise the crash path in a subprocess with the COUNT only (5 > 4): no data is read
    or changed and the child produces no reading.  Records the child's exit code."""
    env = dict(os.environ)
    env["P4_READING"] = "guard_selftest"
    env["P4_MODE"] = "pilot"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, cwd=root,
                       capture_output=True, text=True)
    return {"injected_exclusions": 5, "budget": 4, "returncode": int(r.returncode),
            "raised": bool(r.returncode != 0),
            "stdout_tail": (r.stdout or "")[-200:],
            "stderr_tail": (r.stderr or "")[-200:]}


def prior_guard_selftest():
    """Prove the calibration check is live rather than decorative: an expectation the letters
    cannot meet must raise, in a subprocess, before any file is written.  No data is read and
    no model is loaded (the child exits inside prior_block)."""
    env = dict(os.environ)
    env["P4_READING"] = "prior_guard_selftest"
    env["P4_MODE"] = "pilot"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, cwd=root,
                       capture_output=True, text=True)
    return {"input_letters": ["A", "A", "B", "C"],
            "observed_letter": "A", "observed_hits": 2,
            "injected_letter": "D", "injected_hits": 9,
            "returncode": int(r.returncode), "raised": bool(r.returncode != 0),
            "stdout_tail": (r.stdout or "")[-200:],
            "stderr_tail": (r.stderr or "")[-200:]}


def load_model():
    """The only backbone-dependent step of the four readings: which weights are loaded and
    which per-frame pixel budget the processor gets (paper4/backbone.py).  Nothing else in
    run_selection / run_single / run_distortion / run_donor changes."""
    t = time.time()
    model, proc, dev, minfo = BK.load(BACKBONE, G.MAX_PIXELS)
    PHASE["model_load_s"] = time.time() - t
    log("backbone=%s family=%s arch=%s pixel_control=%s"
        % (BACKBONE, minfo["family"], minfo["architecture"], minfo["pixel_control"]))
    return model, proc, dev, minfo, BK.letter_ids(proc)


def ask(model, proc, pils, prompt, dev, arm):
    t = time.time()
    sink = C.TokenSink()
    try:
        raw, letter = G.answer_arm(model, proc, pils, prompt, dev, counter=FWD,
                                   kind="answer:" + arm, token_sink=sink)
    except Exception as e:                                            # noqa: BLE001
        NOTES.append("arm %s failed: %s" % (arm, str(e)[:200]))
        raw, letter = "ERR:" + str(e)[:200], None
    PHASE["forward_s"] += time.time() - t
    TOKENS[arm] = sink.summary()
    return letter, (raw or "")[:60]


def apply_guard(results, rows, arms_measured, controls, landing):
    """Write the instrument-failure block and the landing (r3 §3.1 item 5).

    The trigger is implemented, not worded: when `instrument_guard` fires, this cell reports
    `instrument_failure.triggered = true`, writes `landing = null` and marks the branch blocks
    withheld.  The computed blocks stay in the file for audit -- they are not a landing.
    """
    dists = INS.letter_distributions(rows, list(arms_measured))
    guard = INS.instrument_guard(controls, dists)
    results["instrument_guard"] = guard
    results["instrument_failure"] = {
        "triggered": bool(guard["triggered"]),
        "triggers": guard["triggers"],
        "consequence": "when triggered this cell reports instrument_failure and writes NO "
                       "landing; the computed blocks stay in the file for audit and are not a "
                       "landing",
        "rule": guard["rule"],
    }
    if guard["triggered"]:
        results["landing"] = None
        for k in ("branches", "bounded_difference"):
            if isinstance(results.get(k), dict):
                results[k]["withheld_due_to_instrument_failure"] = True
    else:
        results["landing"] = landing
    # A top-level marker, so the arm hits and the comparisons cannot be quoted as a reading
    # without it (review §4, "withheld marker only inside the bounded block"): when the guard
    # fires, every computed block in this file is an audit artefact.
    results["readings_withheld"] = {
        "withheld": bool(guard["triggered"]),
        "reason": (guard["triggers"] if guard["triggered"] else None),
        "consequence": ("landing = null; `arms`, `comparisons`, `bounded_difference` and "
                        "`constant_letter_baseline` are computed for audit only and are NOT a "
                        "reading" if guard["triggered"] else
                        "the landing is written; the blocks above are the reading"),
        "decides_claim": bool(DECIDES and not guard["triggered"]),
        "stage": STAGE,
    }
    return guard


def finish(results, samples_rel):
    results["run"]["elapsed_seconds"] = round(time.time() - T0, 1)
    results["run"]["phases_seconds"] = {k: round(float(v), 1) for k, v in PHASE.items()}
    results["forward_counts"] = FWD.report()
    results["notes"] = NOTES
    # ---- backbone bookkeeping (added for the second-backbone batch) ------------------
    bk = BK.get(BACKBONE)
    results["backbone"] = {"name": BACKBONE, "family": bk["family"],
                           "model_dir": bk["model_dir"],
                           "pixel_control": bk["pixel_control"],
                           "archive_identity_applies": bool(BACKBONE == "qwen2vl")}
    if isinstance(results.get("model"), dict):
        results["model"]["model_identity"] = BK.identity(BACKBONE)
        fp = results["model"]["model_identity"]
        # The pre-existing identity blocks name the first backbone's model directory and a
        # fixed manifest constant; on any other backbone those two fields would be wrong, so
        # they are rewritten here from the weights this run actually loaded.
        if isinstance(results.get("identity"), dict):
            results["identity"]["model_dir"] = bk["model_dir"]
            results["identity"]["model_manifest_sha256"] = fp["name_size_manifest_sha256"]
            results["identity"]["model_manifest_sha256_expected"] = None
            results["identity"]["model_identity_note"] = (
                "model_dir / model_manifest_sha256 were rewritten by finish() from the backbone "
                "this run loaded; the expected-constant field is nulled because it belonged to "
                "the first backbone")
    idv = results.get("identity_vs_archive")
    if isinstance(idv, dict) and BACKBONE != "qwen2vl":
        idv["applies_to_this_backbone"] = False
        idv["why"] = ("the archived letters belong to the first backbone; on a second backbone "
                      "this block is a cross-backbone divergence record -- NOT a pass condition, "
                      "NOT a criterion, and it is not used in any sentence. The instrument "
                      "control for this backbone is the annotation-only prior calibration plus "
                      "the P0 determinism control.")
    results["run"]["stage"] = STAGE
    results["run"]["decides_claim"] = DECIDES
    results["run"]["plan_id"] = PLAN_ID
    results["run"]["plan_version"] = PLAN_VERSION
    results["run"]["exp_seed"] = EXP_SEED
    results["run"]["reading"] = READING
    results["run"]["mode"] = MODE
    results["samples"] = samples_rel
    jdump(results, METRICS)
    log("%s/%s done elapsed=%.1fs forwards=%d"
        % (READING, MODE, time.time() - T0, FWD.report()["total"]))
    print(json.dumps({"verdict": results.get("verdict"), "reading": READING, "mode": MODE},
                     ensure_ascii=False))


# ================================================================ (a)+(b) selection

def run_selection():
    st = G.seed_streams(EXP_SEED)
    streams = G.run_seeds(EXP_SEED, ks=KS)
    pool = C.load_pool(LVB_DIR)
    groups = {}
    for q in pool:
        groups.setdefault(q["video_id"], []).append(q)
    qs = sorted(pool, key=lambda q: q["_qid"])
    arc_rel = "sources/task-ca0a24e61b/results/mvp2_samples.jsonl"
    pinned = load_jsonl(arc_rel) or {}
    pinned_qids = {k[1] for k in pinned if isinstance(k, tuple) and k[0] == "qid"}
    if MODE == "pilot":
        ordered = sorted(qs, key=lambda q: (q["_qid"] not in pinned_qids, q["_qid"]))
        keep = []
        for q in ordered:                 # first 6 questions over 6 distinct videos,
            if q["video_id"] not in {x["video_id"] for x in keep}:   # pinned set first
                keep.append(q)
            if len(keep) == 6:
                break
        qs = sorted(keep, key=lambda q: q["_qid"])
    ident = {"pool_questions": len(pool), "pool_videos": len(groups),
             "n_questions": len(qs), "n_videos": len({q["video_id"] for q in qs}),
             "question_set_sha256": C.question_set_sha(qs),
             "pinned_40_set_sha256": PINNED_QSET_SHA,
             "data": C.data_fingerprint(LVB_DIR),
             "note": "pool = every val record whose video file is on this device"}
    # Renamed (review §3): the old block put `matches_expected: null` beside
    # `expected_questions_full: 110` and `pool_questions: 110`, which reads as a pass, and in a
    # pilot the only sizes it carried were the shrunken set's.  A pilot now emits no expected_*
    # field at all, and the three numbers are named for what they are.
    structure = {"checked": False,
                 "n_measured": len(qs), "n_pool": len(pool),
                 "n_measured_videos": len({q["video_id"] for q in qs}),
                 "n_pool_videos": len(groups),
                 "checked_in_mode": MODE,
                 "note": "n_measured is what THIS run measured; n_pool is the carrier's whole "
                         "pool. Only a full run carries the structure pass condition, and only "
                         "then are expected_questions_full / expected_videos_full emitted."}
    if MODE == "full":
        structure["expected_questions_full"] = LVB_FULL_Q
        structure["expected_videos_full"] = LVB_FULL_V
        if not (len(pool) == LVB_FULL_Q and len(groups) == LVB_FULL_V):
            raise SystemExit("structure mismatch: pool %d q / %d v != %d / %d"
                             % (len(pool), len(groups), LVB_FULL_Q, LVB_FULL_V))
        structure["checked"] = True
    # Pre-run zero-information prior: computed from annotations and hashed BEFORE the model
    # is loaded, so it cannot be adjusted after seeing a single letter.
    if len(pinned_qids) == 40:
        g40 = [C.correct_letter(q) for q in pool if q["_qid"] in pinned_qids]
        prior40 = prior_block(g40, "selection_pinned40", expect_letter="C", expect_hits=10)
    else:
        prior40 = {"available": False, "n_pinned_qids": len(pinned_qids),
                   "note": "archive not staged; the 40-question calibration is a crash "
                           "condition in full mode"}
        if MODE == "full":
            raise SystemExit("pinned-40 calibration unavailable (n=%d)" % len(pinned_qids))
    prior = prior_block([C.correct_letter(q) for q in pool], "selection_pool110")
    model, proc, dev, minfo, ids = load_model()
    mident = L.model_identity()
    u16 = C.uniform_indices(G.K_CAND, G.B_MAIN)
    arm_keys = ["uniform@16", "no_frames"] + ["contrast@16_k%d" % k for k in KS]
    rows = []
    ctrl = INS.new_state()
    for qi, q in enumerate(qs):
        TOKENS.clear()
        t = time.time()
        arr, fps, nfr = C.decode_frames(q["_video_path"], G.K_CAND)
        pils = C.to_pil(arr, G.MAX_PIXELS)
        PHASE["decode_s"] += time.time() - t
        prompt = C.prompt_for(q)
        gold = C.correct_letter(q)
        t = time.time()
        sink = C.TokenSink()
        lo = C.option_logits(model, proc, pils, prompt, dev, ids, batch=4, token_sink=sink)
        FWD.count("score:clean@32", len(pils))
        TOKENS["score:clean@32"] = sink.summary()
        PHASE["forward_s"] += time.time() - t
        letters, raws, sel, ov = {}, {}, {}, {}
        letters["no_frames"], raws["no_frames"] = ask(model, proc, [], prompt, dev, "no_frames")
        letters["uniform@16"], raws["uniform@16"] = ask(
            model, proc, [pils[i] for i in u16], prompt, dev, "uniform@16")
        for s in streams:
            k = s["k"]
            darr = np.stack([G.distort_seeded(arr[i], SIGMA, s["s_noise"], qi, i,
                                              G.SIGMA_INDEX) for i in range(len(arr))])
            dpils = C.to_pil(darr, G.MAX_PIXELS)
            t = time.time()
            sink = C.TokenSink()
            ld = C.option_logits(model, proc, dpils, prompt, dev, ids, batch=4,
                                 token_sink=sink)
            FWD.count("score:distorted@32", len(dpils))
            TOKENS["score:distorted@32"] = sink.summary()
            PHASE["forward_s"] += time.time() - t
            tv, _l1 = C.contrast_scores(lo, ld)
            key = "contrast@16_k%d" % k
            sel[key] = G.top_b(tv, G.B_MAIN)
            ov[key] = len(set(sel[key]) & set(u16)) / float(G.B_MAIN)
            letters[key], raws[key] = ask(model, proc, [pils[i] for i in sel[key]],
                                          prompt, dev, key)
        letters["constant_prior"] = prior["prior_letter"]
        raws["constant_prior"] = "PRIOR:" + str(prior["prior_letter"])
        correct = {a: int(letters[a] == gold) for a in arm_keys + ["constant_prior"]}
        row = {"qid": q["_qid"], "video_id": q["video_id"], "gold": gold,
               "question_type": q.get("question_category"), "fps": fps, "n_frames": nfr,
               "letters": letters, "raw": raws, "correct": correct,
               "selected": sel, "overlap_vs_uniform": ov,
               "vision_tokens": dict(TOKENS),
               "duration_group": q.get("duration_group")}
        rows.append(row)
        app_jsonl("results/four_selection_samples.jsonl", row)
        # Instrument-control cases (review §1): the first unit supplies the three prompt
        # paths, and the first N_PERM_UNITS units supply the option-permutation control.
        if ctrl["paths"] is None:
            INS.set_paths(ctrl, frames16=[pils[i] for i in u16], frame1=pils[0],
                          prompt=prompt, arm16="uniform@16",
                          arm1="control-only 1-frame ask (this reading has no 1-frame arm)")
        INS.add_perm_unit(ctrl, q["_qid"], q["question"], q["candidates"], gold,
                          letters["uniform@16"], raws["uniform@16"],
                          [pils[i] for i in u16], "uniform@16")
        log("selection %2d/%d %-28s %s" % (qi + 1, len(qs), q["_qid"],
                                           {k: letters[k] for k in arm_keys}))
    all_arms = list(arm_keys) + ["constant_prior"]
    arms = {a: arm_entry([r["correct"][a] for r in rows]) for a in all_arms}
    comps = {}
    for s in streams:
        key = "contrast@16_k%d" % s["k"]
        comps["%s_vs_uniform@16" % key] = compare(rows, key, "uniform@16", st["s_boot"])
        comps["%s_vs_no_frames" % key] = compare(rows, key, "no_frames", st["s_boot"])
    comps["uniform@16_vs_no_frames"] = compare(rows, "uniform@16", "no_frames", st["s_boot"])
    comps["no_frames_vs_constant_prior"] = compare(rows, "no_frames", "constant_prior",
                                                   st["s_boot"])
    comps["uniform@16_vs_constant_prior"] = compare(rows, "uniform@16", "constant_prior",
                                                    st["s_boot"])
    holm_in = {("contrast@16_k%d_vs_uniform@16" % s["k"]):
               comps["contrast@16_k%d_vs_uniform@16" % s["k"]]["mcnemar"]["p"] for s in streams}
    fam2 = holm_local([("no_frames_vs_constant_prior",
                        comps["no_frames_vs_constant_prior"]["mcnemar"]["p"]),
                       ("no_frames_vs_uniform@16",
                        comps["uniform@16_vs_no_frames"]["mcnemar"]["p"]),
                       ("uniform@16_vs_constant_prior",
                        comps["uniform@16_vs_constant_prior"]["mcnemar"]["p"])])
    nf = comps["no_frames_vs_constant_prior"]
    uf = comps["uniform@16_vs_no_frames"]
    up = comps["uniform@16_vs_constant_prior"]
    sep_above = bool(fam2["decisions"]["no_frames_vs_constant_prior"]["reject"]
                     and nf["cluster_boot95"][0] is not None
                     and nf["cluster_boot95"][0] > 0)
    sep_below = bool(fam2["decisions"]["no_frames_vs_constant_prior"]["reject"]
                     and nf["cluster_boot95"][1] is not None
                     and nf["cluster_boot95"][1] < 0)
    headroom = bool(fam2["decisions"]["no_frames_vs_uniform@16"]["reject"]
                    and uf["cluster_boot95"][0] is not None and uf["cluster_boot95"][0] > 0
                    and fam2["decisions"]["uniform@16_vs_constant_prior"]["reject"]
                    and up["cluster_boot95"][0] is not None and up["cluster_boot95"][0] > 0)
    branch = "b1" if sep_above else ("b3" if sep_below else "b2")
    branches = {
        "b_branch": branch, "headroom": headroom,
        "b4_identification_failure": (not headroom),
        "rule": "b1 needs the Holm decision AND the video-level CI lower bound > 0; b3 needs "
                "the Holm decision AND the video-level CI upper bound < 0; otherwise b2. (a) is "
                "written as a selection finding only when headroom is true.",
        "b4_may_be_false_negative": "the in-sample max-letter prior is optimistic by roughly "
                                    "4-5 pp at n=110 with 5 classes; on the archived 40 "
                                    "questions uniform vs prior was 19 v 6 (p=0.0146), so the "
                                    "headroom margin is thin and b4 can be a false negative",
        "b_mid_branch": "delta<0 with a video-level interval containing 0 is neither a "
                        "reversal nor a confirmation; it is reported as 'same sign, not "
                        "separable at this resolution' and is NOT folded into either branch"}

    def subset_report(sub_rows):
        out = {"n_questions": len(sub_rows),
               "n_videos": len({r["video_id"] for r in sub_rows}),
               "arms": {a: arm_entry([r["correct"][a] for r in sub_rows]) for a in all_arms},
               "deltas": {}, "no_frames_vs_constant_prior": None}
        for s in streams:
            key = "contrast@16_k%d" % s["k"]
            out["deltas"][key] = compare(sub_rows, key, "uniform@16", st["s_boot"])
        out["no_frames_vs_constant_prior"] = compare(sub_rows, "no_frames", "constant_prior",
                                                     st["s_boot"])
        return out

    rows40 = [r for r in rows if r["qid"] in pinned_qids]
    rows70 = [r for r in rows if r["qid"] not in pinned_qids]
    subsets = {"pinned40": subset_report(rows40) if rows40 else None,
               "new70": subset_report(rows70) if rows70 else None,
               "note": "pinned40 = questions already measured by task-ca0a24e61b (36% of the "
                       "full pool); new70 = the rest. Both are reported; the new70 subset is "
                       "the independent observation and pinned40 is the same-question replication."}
    id_sel = archive_identity(rows, arc_rel, {"uniform@16": "pred_uniform"})
    id_sel["n_expected"] = len(rows40)
    if MODE == "full":
        n_ov = int(id_sel.get("n_overlap") or 0)
        if n_ov < len(rows40):
            raise SystemExit("selection identity overlap %d < %d pinned questions"
                             % (n_ov, len(rows40)))
    controls = INS.build(model, proc, dev, ids, ctrl, FWD, EXP_SEED)
    results = {
        "run": {},
        "claim_tested": {"a": "at a fixed 16-frame budget the training-free option-contrast "
                              "selector does not beat uniform sampling (stated as a selection "
                              "finding only where uniform separates from both no_frames and the "
                              "zero-information option prior)",
                         "b": "the share of questions the model still answers with every frame "
                              "removed, reported as a paired difference from the zero-information "
                              "option prior (not as an absolute rate)"},
        "independent_unit": "video (questions clustered inside videos; both levels reported)",
        "identity": ident, "structure": structure, "model": model_block(minfo, mident),
        "arms": arms,
        "pretest_prior": {"pool110": prior, "pinned40_calibration": prior40},
        "comparisons": comps,
        "holm_streams_vs_uniform": C.holm(list(holm_in.items())),
        "holm_family2_identification": fam2,
        "branches": branches,
        "subsets": subsets,
        "selection_overlap": {("contrast@16_k%d" % s["k"]):
                              {"mean": float(np.mean([r["overlap_vs_uniform"]["contrast@16_k%d"
                                                                              % s["k"]] for r in rows]))}
                              for s in streams},
        "constant_letter_baseline": constant_baseline(rows),
        "extraction": extraction_block(rows, arm_keys),
        "identity_vs_archive": id_sel,
        "instrument_controls": controls,
        "vision_tokens": controls["vision_tokens"],
        "n_measured": len(rows),
        "verdict": {"n_questions": len(rows), "n_videos": len({r["video_id"] for r in rows}),
                    "resolution_question_pp": (100.0 / len(rows) if rows else None),
                    "resolution_video_pp": (100.0 / len({r["video_id"] for r in rows})
                                            if rows else None)},
    }
    apply_guard(results, rows, all_arms, controls,
                {"reading": "selection", "b_branch": branch, "headroom": headroom,
                 "withheld_if_instrument_failure": True})
    finish(results, "results/four_selection_samples.jsonl")


# ================================================================ (c) single frame

def run_single():
    groups, skipped = F.eligible_by_video(F.FRESH_META, F.FRESH_DIR)
    design, confirmation, _shuffled = F.split_videos(groups)
    side = confirmation
    # Question selection.  Default (minuid) is the archived behaviour; draw is qdraw-v1 of
    # plan-single-frame-random-question and the only path that writes the candidate
    # sets and the draw record.
    if QSELECT == "draw":
        if QDRAW_SEED != QD.QDRAW_SEED:
            raise SystemExit("qdraw seed %d != registered %d" % (QDRAW_SEED, QD.QDRAW_SEED))
        units, shadow_golds, drawrec = QD.build(side, groups, log=log)
    else:
        units, _sk = F.build_units(groups, side)
        shadow_golds, drawrec = None, None
    ident = {"dataset": "lvbench-fresh-24209d43cbdd",
             "manifest_sha256": L.manifest_sha256(root=F.FRESH_DIR)[0],
             "manifest_expected": F.FRESH_MANIFEST_SHA,
             "annotation_sha256": L.sha256_file(F.FRESH_META),
             "annotation_expected": ANNOT_SHA_EXPECTED,
             "split_seed": F.SPLIT_SEED, "side": "confirmation",
             "n_units": len(units),
             "re_execution_of": "evidence.task-67bfc74ca9",
             "note": ("every confirmation video was measured before (task-67bfc74ca9, "
                      "results/fresh_readings.jsonl); with the registered question draw a unit "
                      "re-executes the archived question only where the draw returns the "
                      "archived one (always on the single-candidate videos, and on any "
                      "multi-candidate video that draws index 0). No unit is claimed untouched "
                      "as a video." if QSELECT == "draw" else
                      "every confirmation unit was measured before (task-67bfc74ca9, "
                      "results/fresh_readings.jsonl); this run re-executes them. No unit is "
                      "claimed untouched and no earlier-record claim is made.")}
    n_eligible = int(sum(len(c) for _v, c in groups))
    n_selected = int(sum(1 for _v, c in groups for (_u, _a, _b, q, _so) in c
                         if F.rule_select(q)))
    checks = {"manifest_matches": bool(ident["manifest_sha256"] == F.FRESH_MANIFEST_SHA),
              "annotation_matches": bool(ident["annotation_sha256"] == ANNOT_SHA_EXPECTED),
              "pool_videos": len(groups), "pool_videos_expected": 32,
              "pool_eligible_questions": n_eligible, "pool_eligible_expected": 134,
              "pool_selected_questions": n_selected, "pool_selected_expected": 55,
              "design_units": len(design), "design_units_expected": 8,
              "confirmation_units": len(confirmation), "confirmation_units_expected": 24,
              "side_size": bool(len(units) == FRESH_CONFIRM_N),
              "one_unit_per_video": bool(len({u["video_id"] for u in units}) == len(units)),
              "e_star_outside_i16u": bool(all(not u["e_star_in_i16u"] for u in units))}
    if QSELECT == "draw":
        _dist = {}
        for u in units:
            _dist[u["n_selected_in_video"]] = _dist.get(u["n_selected_in_video"], 0) + 1
        checks["draw_k_distribution"] = {str(k): v for k, v in sorted(_dist.items())}
        checks["draw_k_distribution_expected"] = {str(k): v for k, v in
                                                  sorted(QD.EXPECTED_K_DISTRIBUTION.items())}
        checks["draw_k_distribution_matches"] = bool(_dist == QD.EXPECTED_K_DISTRIBUTION)
        checks["draw_n_singleton"] = int(sum(1 for u in units
                                             if u["n_selected_in_video"] == 1))
        checks["draw_n_index0"] = int(sum(1 for u in units if u["draw_index"] == 0))
        checks["draw_recheck_from_disk_ok"] = bool(drawrec["recheck_from_disk_ok"])
        checks["draw_unit_uid_matches_record"] = bool(
            {u["video_id"]: int(u["uid"]) for u in units} ==
            {p["video_id"]: int(p["uid"]) for p in drawrec["per_video"]})
        # Falsification control written into the Plan before the draw: if every multi-candidate
        # video drew index 0, the draw is indistinguishable from the old smallest-uid rule.
        # Under the registered rule that has probability (1/2)^7 (1/3)^2 (1/4)^2 ~ 5.4e-5.
        checks["draw_moved_at_least_one"] = bool(checks["draw_n_index0"] >
                                                 checks["draw_n_singleton"])
    checks["all_structure_ok"] = bool(
        checks["manifest_matches"] and checks["annotation_matches"]
        and checks["pool_videos"] == 32 and checks["pool_eligible_questions"] == 134
        and checks["pool_selected_questions"] == 55 and checks["design_units"] == 8
        and checks["confirmation_units"] == 24 and checks["side_size"]
        and checks["one_unit_per_video"] and checks["e_star_outside_i16u"]
        and checks.get("draw_k_distribution_matches", True)
        and checks.get("draw_recheck_from_disk_ok", True)
        and checks.get("draw_unit_uid_matches_record", True)
        and checks.get("draw_moved_at_least_one", True))
    if not checks["all_structure_ok"]:
        raise SystemExit("single-frame structure check failed: %s"
                         % json.dumps(checks, default=str, ensure_ascii=False))
    # Pre-run, hashed, annotation-only: the zero-information option prior, computed before the
    # model is loaded so it cannot be adjusted after seeing a letter.
    if QSELECT == "draw":
        # Positive control B: the min-uid shadow set is rebuilt from the same annotations with
        # no decoding, and its prior must still be the archived constant (A, 9/24).  A mismatch
        # means the annotation read or the inclusion/SELECT rules moved, so the run stops before
        # any model step.
        prior_shadow = prior_block(shadow_golds, "single_minuid24_shadow",
                                   expect_letter="A", expect_hits=9)
        # The drawn set has its own gold distribution (the questions changed), so this prior
        # carries no expectation: the archived constant belongs to the min-uid unit set.
        prior = prior_block([u["gold"] for u in units], "single_drawn24")
    else:
        prior_shadow = None
        prior = prior_block([u["gold"] for u in units], "single_conf24",
                            expect_letter="A", expect_hits=9)
    # Smoke: measure only the first N registered units in fixed video-id order.  The draw and
    # the structure checks above already cover all 24; only the model steps shrink.
    measure_units = units
    if QSELECT == "draw" and QDRAW_SMOKE_N > 0:
        if MODE != "pilot":
            raise SystemExit("P4_QDRAW_SMOKE_N is only allowed with P4_MODE=pilot: a full run "
                             "measures all registered units")
        measure_units = units[:QDRAW_SMOKE_N]
    model, proc, dev, minfo, ids = load_model()
    mident = L.model_identity()
    rows = []
    arm_keys = ["uniform@16", "single_frame_j_star", "no_frames"]
    ctrl = INS.new_state()
    for i, u in enumerate(measure_units):
        TOKENS.clear()
        t = time.time()
        grid = L.decode_indices(u["video_path"], u["grid_idx"])
        ev = L.decode_indices(u["video_path"], [u["j_star"]])[0]
        gpils = C.to_pil(grid, G.MAX_PIXELS)
        evpil = C.to_pil(ev[None], G.MAX_PIXELS)[0]
        PHASE["decode_s"] += time.time() - t
        q = {"question": u["question"], "candidates": list(u["options"]),
             "correct_letter": u["gold"]}
        prompt = G.prompt_for(q)
        letters, raws = {}, {}
        letters["no_frames"], raws["no_frames"] = ask(model, proc, [], prompt, dev, "no_frames")
        letters["uniform@16"], raws["uniform@16"] = ask(
            model, proc, [gpils[j] for j in L.I16U], prompt, dev, "uniform@16")
        letters["single_frame_j_star"], raws["single_frame_j_star"] = ask(
            model, proc, [evpil], prompt, dev, "single_frame_j_star")
        letters["constant_prior"] = prior["prior_letter"]
        raws["constant_prior"] = "PRIOR:" + str(prior["prior_letter"])
        correct = {a: int(letters[a] == u["gold"]) for a in arm_keys + ["constant_prior"]}
        row = {"qid": u["qid"], "video_id": u["video_id"], "uid": u["uid"], "gold": u["gold"],
               "j_star": u["j_star"], "fps": u["fps"], "n_frames": u["n_frames"],
               "n_selected_in_video": u.get("n_selected_in_video"),
               "n_eligible_in_video": u.get("n_eligible_in_video"),
               "draw_index": u.get("draw_index"),
               "question_type": u["question_type"], "letters": letters, "raw": raws,
               "vision_tokens": dict(TOKENS),
               "correct": correct}
        rows.append(row)
        app_jsonl("results/four_single_samples.jsonl", row)
        # Instrument controls (review §1): three prompt paths once, and the permutation
        # control on this reading's own 1-frame arm -- the path that drives (c).
        if ctrl["paths"] is None:
            INS.set_paths(ctrl, frames16=[gpils[j] for j in L.I16U], frame1=evpil,
                          prompt=prompt, arm16="uniform@16",
                          arm1="single_frame_j_star")
        # Plan -r4 §15: the permutation control covers EVERY measured unit on this reading, so
        # the instrument guard is fed the registered control (24 units x 2 option orders = 48
        # pairs) instead of an unregistered 6-unit subset whose 12-pair verdict flips between
        # pass and not_discriminating.  Controls only: no measurement arm, no frame budget and
        # no criterion of the cell changes; the extra forwards are tagged control:*.
        INS.add_perm_unit(ctrl, u["qid"], u["question"], u["options"], u["gold"],
                          letters["single_frame_j_star"], raws["single_frame_j_star"],
                          [evpil], "single_frame_j_star", limit=len(measure_units))
        log("single %2d/%d %-18s %s" % (i + 1, len(measure_units), u["video_id"],
                                        {k: letters[k] for k in arm_keys}))
    all_arms = list(arm_keys) + ["constant_prior"]
    arms = {a: arm_entry([r["correct"][a] for r in rows]) for a in all_arms}
    st = G.seed_streams(EXP_SEED)
    comp = {"single_frame_j_star_vs_uniform@16":
            compare(rows, "single_frame_j_star", "uniform@16", st["s_boot"]),
            "uniform@16_vs_no_frames": compare(rows, "uniform@16", "no_frames", st["s_boot"]),
            "single_frame_j_star_vs_no_frames":
            compare(rows, "single_frame_j_star", "no_frames", st["s_boot"]),
            "single_frame_j_star_vs_constant_prior":
            compare(rows, "single_frame_j_star", "constant_prior", st["s_boot"]),
            "uniform@16_vs_constant_prior":
            compare(rows, "uniform@16", "constant_prior", st["s_boot"]),
            "no_frames_vs_constant_prior":
            compare(rows, "no_frames", "constant_prior", st["s_boot"])}
    # ---- bounded-difference block (no equivalence threshold) -----------------------------
    pair = comp["single_frame_j_star_vs_uniform@16"]
    n_u = len(rows)
    _a = [r["correct"]["single_frame_j_star"] for r in rows]
    _b = [r["correct"]["uniform@16"] for r in rows]
    n10 = int(sum(1 for x, y in zip(_a, _b) if x == 1 and y == 0))
    n01 = int(sum(1 for x, y in zip(_a, _b) if x == 0 and y == 1))
    m_dis = n01 + n10
    lo, hi = pair["cluster_boot95"][0], pair["cluster_boot95"][1]
    pt = pair["mean_diff"]
    half_units = (1.96 * math.sqrt(m_dis) / n_u) if n_u else None
    # Two exact bounds, each labelled for what it is (review §4): the old label called the
    # two-sided 95% value a "one-sided 95%" bound.  Both are reported, so the degenerate
    # bootstrap at m = 0 is never read as equivalence and the number never travels without
    # its label.
    _ub_two95 = (1.0 - 0.025 ** (1.0 / n_u)) if n_u else None
    _ub_one95 = (1.0 - 0.05 ** (1.0 / n_u)) if n_u else None
    bounded = {
        "point": pt,
        "point_units": (int(round(pt * n_u)) if pt is not None else None),
        "ci95": [lo, hi],
        "discordant_pairs": m_dis, "n10_single_only": n10, "n01_uniform_only": n01,
        "n_units": n_u,
        "unit_pp": (100.0 / n_u if n_u else None),
        "smallest_symmetric_halfwidth_units_at_this_m": (
            int(math.ceil(1.96 * math.sqrt(m_dis))) if m_dis > 0 else 0),
        "approx_halfwidth_units": half_units,
        "exact_upper_bound_if_m_zero": _ub_two95,
        "exact_upper_bound_if_m_zero_definition": "two-sided 95% Clopper-Pearson upper bound "
                                                  "(equivalently one-sided 97.5%)",
        "exact_upper_bound_one_sided95_if_m_zero": _ub_one95,
        "rules_out": ("differences outside [%.4f, %.4f] (i.e. beyond %s and %s units)"
                      % (lo, hi, (int(math.floor(lo * n_u)) if lo is not None else None),
                         (int(math.ceil(hi * n_u)) if hi is not None else None))),
        "does_not_rule_out": ("any difference inside [%.4f, %.4f], including zero and any band "
                              "narrower than the interval" % (lo, hi)),
        "equivalence_claimed": False,
        "statement": "bounded-difference statement only. No equivalence threshold, no pass/fail "
                     "rule, and no 'sixteen frames answer as much as one frame' sentence.",
        "identity_conflict_rule": "identity must cover every measured unit whose candidate set "
                                  "holds one question (n_overlap >= n_expected, 0 mismatch). "
                                  "It is a precondition, not a competing criterion: the reading "
                                  "has no threshold, so the two cannot exclude each other.",
        "power_note": "at n=24 one unit is 4.17 pp; with m discordant pairs the paired interval "
                      "is about +-1.96*sqrt(m)/24, so the narrowest non-degenerate symmetric "
                      "band is +-2 units at m=1 and +-4 units at m=3; a +-3-unit (0.125) band is "
                      "not attainable together with a 24/24 identity re-execution.",
        "bootstrap_degeneracy": ("percentile bootstrap collapses to [0,0] at m=0; the exact "
                                 "two-sided 95%% upper bound at 0/%d is %.4f (equivalently "
                                 "one-sided 97.5%%; the one-sided 95%% bound is %.4f), reported "
                                 "alongside so a degenerate interval is never read as "
                                 "equivalence" % (n_u, _ub_two95 or 0.0, _ub_one95 or 0.0))}
    id_single = archive_identity(
        rows, "sources/task-67bfc74ca9/results/fresh_readings.jsonl",
        {"uniform@16": "uniform@16", "single_frame_j_star": "single_frame_j_star"})
    if QSELECT == "draw":
        # Only units whose question is the archived one can be compared: a video with one
        # candidate necessarily keeps its question, and a multi-candidate video that happened to
        # draw index 0 does too.  Units whose question changed have no archived counterpart and
        # are excluded, not counted as mismatches.
        singl_qids = sorted({u["qid"] for u in measure_units
                             if u["n_selected_in_video"] == 1})
        id_single["n_expected"] = len(singl_qids)
        id_single["covered_rule"] = ("every measured unit with a single-candidate set must "
                                     "reproduce the archived question and its letters on all "
                                     "arms; a unit whose drawn uid differs from the archived "
                                     "min-uid uid is excluded, not counted as a mismatch")
        id_single["singleton_qids_measured"] = singl_qids
        if not id_single.get("available"):
            raise SystemExit("identity archive not staged: %s" % id_single.get("path"))
        if int(id_single.get("n_overlap") or 0) < len(singl_qids):
            raise SystemExit("identity overlap %s < %d singleton units: no identity control for "
                             "these units" % (id_single.get("n_overlap"), len(singl_qids)))
    else:
        id_single["n_expected"] = n_u
        if not id_single.get("available"):
            raise SystemExit("identity archive not staged: %s" % id_single.get("path"))
        if int(id_single.get("n_overlap") or 0) < n_u:
            raise SystemExit("identity overlap %s < %d units: no identity control for this cell"
                             % (id_single.get("n_overlap"), n_u))
    id_single["ok"] = bool(id_single["n_mismatch"] == 0)
    controls = INS.build(model, proc, dev, ids, ctrl, FWD, EXP_SEED)
    # ---- the option_mapping verdict and its fragility, recorded WITH the reading -----------
    # Plan -r4 §14: the control that gates this cell passed only narrowly, and the review
    # showed the verdict flips under another option-permutation seed; its scope is gross
    # mis-wiring only.  Those two facts belong to the result, not to a footnote, so they are
    # written here beside the arms as well as in the control block.
    _oc = (controls.get("option_control") or {})
    _mp = _oc.get("mapping") or {}
    _pl = _mp.get("pass_line") or {}
    _cov = _mp.get("coverage") or {}
    control_verdict = {
        "control": ("option_mapping on this cell's registered units: %d units x 2 option "
                    "orders = %d pairs" % (len(ctrl["perm_units"]),
                                           2 * len(ctrl["perm_units"]))),
        "status_exactly_as_the_registered_line_gives_it": _pl.get("status"),
        "n_units_in_control": int(len(ctrl["perm_units"])),
        "n_pairs": _mp.get("n_pairs"), "n_evaluable": _mp.get("n_evaluable"),
        "n_contradictions": _mp.get("n_contradictions"),
        "contradiction_counts_both_rules": {
            "corrected_exact_first": ((_mp.get("contradiction_counts") or {})
                                      .get("corrected_exact_first")),
            "old_rule_longest_match": ((_mp.get("contradiction_counts") or {})
                                       .get("old_rule_longest_match")),
            "note": "the corrected count is never printed alone: the count the superseded "
                    "longest-match parse gives on the SAME pairs is beside it (review §1.2 "
                    "condition 3)",
        },
        "n_unevaluable": _mp.get("n_unevaluable_charged"),
        "contradictions_plus_unevaluable": _mp.get("contradictions_plus_unevaluable"),
        "tolerance_b": _pl.get("tolerance_b"),
        "tolerance_b_strict": _pl.get("tolerance_b_strict"),
        "p_retest_upper95": _pl.get("p_retest_upper95"),
        "n_retest_both": _pl.get("n_retest_both"),
        "n_retest_disagree": _pl.get("n_retest_disagree"),
        "e_max_allowed": _cov.get("e_max_allowed"),
        "required_coverage_frac": _cov.get("required_coverage_frac"),
        "slack_b_minus_c": _cov.get("slack_b_minus_c"),
        "max_absorbable_unevaluable_pairs": _cov.get("max_absorbable_unevaluable_pairs"),
        "r2_rule_status": _mp.get("r2_rule_status"),
        "smallest_detectable_rate": _pl.get("smallest_detectable_rate"),
        "passes_only_because_of_noise": _pl.get("passes_only_because_of_noise"),
        "fragility_is_a_property_of_the_instrument": {
            "why_recorded": "the pilot-gate review established that this verdict flips under "
                            "other defensible choices on the same data; it is recorded with "
                            "the reading, not as a footnote",
            "same_pairs_other_conventions": _pl.get("sensitivity"),
            "review_reported": {
                "point_estimate_line": "with the observed naming noise as a point estimate "
                                       "(5/33) B = 11 and C + e = 13 > 11 -> incomplete",
                "missing_case_asymmetry": "charging the 13 pairs whose channel B named no "
                                          "option as noise disagreements gives 18/46, "
                                          "p_hi = 0.5232 -> not_discriminating",
                "other_option_permutation": "evidence.task-6e86f0d18c (seed 1; same "
                                            "commit, units and draw): 8/29, p_hi = 0.4429 "
                                            "-> not_discriminating",
                "source": "review-single-frame-random-question-pilot-gate-c1@v1 "
                          "§1.1 and §6; checks task-a6d13eb143, task-1fc972beb7, "
                          "task-6e86f0d18c",
            },
        },
        "scope": {
            "proves": _pl.get("can_prove"),
            "does_not_prove": _pl.get("cannot_prove"),
            "read_as": ("a pass excludes a gross mis-wiring only (smallest detectable "
                        "contradiction rate %s); it does NOT confirm that the letter<->option "
                        "mapping is correct, and it must not be reported as a verified "
                        "mapping or a sound instrument"
                        % _pl.get("smallest_detectable_rate")),
        },
    }
    results = {"run": {},
               "claim_tested": {"c": ("on the 24 confirmation video units, the paired difference "
                                      "between sixteen uniform frames and one frame is reported "
                                      "with its bounds; no equivalence is claimed and no unit is "
                                      "claimed untouched") if QSELECT == "draw" else
                                     "on the 24 confirmation video units first measured in "
                                     "task-67bfc74ca9 (this run re-executes the same units), the "
                                     "paired difference between sixteen uniform frames and one "
                                     "frame is reported with its bounds; no equivalence is "
                                     "claimed and no unit is claimed untouched"},
               "independent_unit": ("video (one question per video, drawn uniformly from the "
                                    "rule-selected candidate set)" if QSELECT == "draw"
                                    else "video (one rule-selected question per video)"),
               "question_selection": ("qdraw-v1: uniform draw over each video's rule-selected "
                                      "candidate set, seed registered in the Plan" if QSELECT
                                      == "draw" else "minuid: smallest integer uid among the "
                                      "rule-selected questions"),
               "identity": ident, "structure": checks, "skipped_videos": skipped,
               "model": model_block(minfo, mident), "arms": arms, "comparisons": comp,
               "pretest_prior": prior,
               "prior_minuid_shadow": prior_shadow,
               "registered_units": {"n_units": len(units), "n_measured": len(rows),
                                    "n_measured_qids": [u["qid"] for u in rows],
                                    "smoke_limit": int(QDRAW_SMOKE_N) if QDRAW_SMOKE_N else 0,
                                    "note": "the draw and the structure checks cover all "
                                            "registered units; a smoke measures only the first "
                                            "N in fixed video-id order and its readings are "
                                            "pipeline-only"},
               "unit_selection": (drawrec if QSELECT == "draw" else None),
               "bounded_difference": bounded,
               "constant_letter_baseline": constant_baseline(rows),
               "extraction": extraction_block(rows, arm_keys + ["constant_prior"]),
               "identity_vs_archive": id_single,
               "instrument_controls": controls,
               "instrument_control_verdict": control_verdict,
               "vision_tokens": controls["vision_tokens"],
               "n_measured": len(rows),
               "verdict": {"n_units": len(rows),
                           "resolution_pp": (100.0 / len(rows) if rows else None),
                           "n_units_registered": len(units),
                           "stage_note": ("smoke: %d of %d registered units measured; readings "
                                          "are pipeline-only" % (len(rows), len(units))
                                          if len(rows) != len(units) else
                                          "all registered units measured")}}
    apply_guard(results, rows, all_arms, controls,
                {"reading": "single", "bounded_difference_point": bounded["point"],
                 "ci95": bounded["ci95"], "discordant_pairs": m_dis,
                 "equivalence_claimed": False,
                 "withheld_if_instrument_failure": True})
    finish(results, "results/four_single_samples.jsonl")


# ================================================================ (d1) distortion

def non_evidence_slot(e):
    s = (int(e) + 3) % L.K
    for _ in range(L.K):
        if s not in L.I16U_SET:
            return s
        s = (s + 1) % L.K
    raise SystemExit("no non-evidence slot")


def run_distortion():
    st = G.seed_streams(EXP_SEED)
    ident = L.identity_check()
    # Question selection.  Default (minuid) is the archived behaviour; draw is ddraw-v1 of
    # plan-distortion-random-question: a per-video uniform draw over the compliant
    # candidate set, seed fixed in that Plan, with the min-uid shadow pin (the UNCHANGED
    # lv_common.build_pin) required to hash to the archived value before anything is
    # measured.
    shadow = None
    drawrec = None
    if QSELECT == "draw":
        if DDRAW_SEED != DD.DDRAW_SEED:
            raise SystemExit("ddraw seed %d != registered %d" % (DDRAW_SEED, DD.DDRAW_SEED))
        pin, shadow, drawrec = DD.build(log=log)
    elif QSELECT == "minuid":
        pin = L.build_pin()
    else:
        raise SystemExit("P4_QSELECT must be minuid or draw (got %r)" % QSELECT)
    chk = L.check_pin(pin)
    pairs = list(pin["pairs"])
    # Renamed (review §3): `n_pairs` used to be the size of the shrunken pilot set while the
    # same block listed 8 screen + 9 report pairs, so it read as the full pin.  Both numbers are
    # now named for what they are.
    n_pairs_in_pin = len(pairs)
    if MODE == "pilot":
        want = set(pin["screen_pairs"][:3])
        pairs = [p for p in pairs if int(p["pair"]) in want]
    structure = {
        "n_pairs_in_pin": n_pairs_in_pin, "n_pairs_expected": 17,
        "one_pair_per_video": bool(len({p["video_id"] for p in pin["pairs"]})
                                   == len(pin["pairs"])),
        "e_star_outside_i16u_all": chk["pass"],
        "videos_match_minuid_shadow": (None if shadow is None else
                                       [p["video_id"] for p in pin["pairs"]]
                                       == [p["video_id"] for p in shadow["pairs"]]),
        "split_matches_minuid_shadow": (None if shadow is None else
                                        pin["screen_pairs"] == shadow["screen_pairs"]
                                        and pin["report_pairs"] == shadow["report_pairs"]),
        "skipped_matches_minuid_shadow": (None if shadow is None else
                                          pin.get("skipped_videos")
                                          == shadow.get("skipped_videos")),
        "draw_recheck_from_disk_ok": (None if drawrec is None
                                      else drawrec["recheck_from_disk_ok"]),
        "draw_n_differ_from_minuid": (None if drawrec is None
                                      else drawrec["n_differ_from_minuid"]),
        "draw_n_multi_videos": (None if drawrec is None else drawrec["n_multi_videos"]),
        "draw_falsification_vacuous": (None if drawrec is None
                                       else drawrec["n_multi_videos"] == 0),
        "checked_in_mode": MODE,
        "note": "structure covers the WHOLE pin (17 pairs) in every mode; only the model "
                "steps shrink in a pilot",
    }
    structure["all_structure_ok"] = bool(
        structure["n_pairs_in_pin"] == 17 and structure["one_pair_per_video"]
        and structure["e_star_outside_i16u_all"]
        and structure["videos_match_minuid_shadow"] is not False
        and structure["split_matches_minuid_shadow"] is not False
        and structure["skipped_matches_minuid_shadow"] is not False
        and structure["draw_recheck_from_disk_ok"] is not False)
    if not structure["all_structure_ok"]:
        raise SystemExit("distortion structure check failed: %s"
                         % json.dumps(structure, default=str, ensure_ascii=False))
    model, proc, dev, minfo, ids = load_model()
    mident = L.model_identity()
    rows = []
    arm_keys = ["single_clean"]
    for form in ("patchcopy", "noise"):
        for k in KS:
            arm_keys.append("distorted_evidence_%s_k%d" % (form, k))
            arm_keys.append("distorted_nonevidence_%s_k%d" % (form, k))
    if MODE == "pilot":
        arm_keys += ["uniform@16", "no_frames"]
    ctrl = INS.new_state()
    for i, p in enumerate(pairs):
        TOKENS.clear()
        t = time.time()
        grid = L.decode_indices(p["video_path"], p["grid_idx_guarded"])
        ev = L.decode_indices(p["video_path"], [p["j_star"]])[0]
        ne = int(non_evidence_slot(p["e_star"]))
        ne_frame = grid[ne]
        PHASE["decode_s"] += time.time() - t
        gpils = C.to_pil(grid, G.MAX_PIXELS)
        evpil = C.to_pil(ev[None], G.MAX_PIXELS)[0]
        q = {"question": p["question"], "candidates": list(p["options"]),
             "correct_letter": p["gold"]}
        prompt = G.prompt_for(q)
        geom = L.geom_of(ev.shape[0], ev.shape[1])
        q_idx = int(p["pair"]) * 2              # the archived r6 carrier run's role index
        letters, raws = {}, {}
        letters["single_clean"], raws["single_clean"] = ask(
            model, proc, [evpil], prompt, dev, "single_clean")
        for form in ("patchcopy", "noise"):
            for k in KS:
                s_noise = G.run_seeds(EXP_SEED, ks=(k,))[0]["s_noise"]
                de, _m1 = L.operator_frame(form, ev, geom, k, q_idx, int(p["e_star"]), s_noise)
                dn, _m2 = L.operator_frame(form, ne_frame, geom, k, q_idx, ne, s_noise)
                a1 = "distorted_evidence_%s_k%d" % (form, k)
                a2 = "distorted_nonevidence_%s_k%d" % (form, k)
                letters[a1], raws[a1] = ask(model, proc,
                                             [C.to_pil(de[None], G.MAX_PIXELS)[0]], prompt,
                                             dev, a1)
                letters[a2], raws[a2] = ask(model, proc,
                                             [C.to_pil(dn[None], G.MAX_PIXELS)[0]], prompt,
                                             dev, a2)
        if MODE == "pilot":
            letters["uniform@16"], raws["uniform@16"] = ask(
                model, proc, [gpils[j] for j in L.I16U], prompt, dev, "uniform@16")
            letters["no_frames"], raws["no_frames"] = ask(
                model, proc, [], prompt, dev, "no_frames")
        correct = {a: int(letters[a] == p["gold"]) for a in arm_keys}
        row = {"qid": "%s#%d" % (p["video_id"], p["uid"]), "video_id": p["video_id"],
               "pair": int(p["pair"]), "uid": int(p["uid"]), "gold": p["gold"],
               "j_star": int(p["j_star"]),
               "e_star": int(p["e_star"]), "non_evidence_slot": ne, "fps": p["fps"],
               "n_candidates": p.get("n_candidates"), "draw_index": p.get("draw_index"),
               "uid_min": p.get("uid_min"),
               "changed_from_minuid": p.get("changed_from_minuid"),
               "letters": letters, "raw": raws, "vision_tokens": dict(TOKENS),
               "correct": correct}
        rows.append(row)
        app_jsonl("results/four_distortion_samples.jsonl", row)
        # Instrument controls (review §1).  This reading's driving arm is a 1-frame arm, so
        # the permutation control runs on the 1-frame prompt path; the 16-frame path is still
        # repeated by the determinism block (a control-only forward, counted as control:*).
        if ctrl["paths"] is None:
            INS.set_paths(ctrl, frames16=[gpils[j] for j in L.I16U], frame1=evpil,
                          prompt=prompt, arm16="control-only 16-frame ask (no such arm here)",
                          arm1="single_clean")
        INS.add_perm_unit(ctrl, "%s#%d" % (p["video_id"], p["uid"]), p["question"],
                          p["options"], p["gold"], letters["single_clean"],
                          raws["single_clean"], [evpil], "single_clean")
        log("distortion %2d/%d %-16s clean=%s ev_k0=%s ne_k0=%s"
            % (i + 1, len(pairs), p["video_id"], letters["single_clean"],
               letters["distorted_evidence_patchcopy_k0"],
               letters["distorted_nonevidence_patchcopy_k0"]))
    n = len(rows)
    arms = {a: arm_entry([r["correct"][a] for r in rows]) for a in arm_keys}
    bar = bar_hits(n)
    per_form = {}
    holm_in = {}
    for form in ("patchcopy", "noise"):
        per_form[form] = {}
        for k in KS:
            a1 = "distorted_evidence_%s_k%d" % (form, k)
            a2 = "distorted_nonevidence_%s_k%d" % (form, k)
            cmp_ = {"evidence_vs_nonevidence": compare(rows, a1, a2, st["s_boot"]),
                    "evidence_hits": arms[a1]["hits"],
                    "nonevidence_hits": arms[a2]["hits"],
                    "bar": bar,
                    "evidence_at_or_below_bar": bool(arms[a1]["hits"] <= bar),
                    "evidence_cp95_upper_below_bar_frac":
                        bool(arms[a1]["cp95"][1] is not None and arms[a1]["cp95"][1] < BAR_FRAC)}
            per_form[form][k] = cmp_
            holm_in["%s_k%d_evidence_vs_nonevidence" % (form, k)] = \
                cmp_["evidence_vs_nonevidence"]["mcnemar"]["p"]
    # ---- identity vs the archived carrier run (task-68cebbe16b) -------------------------
    # Under the registered draw only units whose drawn question IS the archived min-uid
    # question have an archived counterpart; changed units are excluded, never counted as
    # mismatches.  uniform@16 / no_frames are compared only where this reading measures
    # them (pilot mode); under the archived minuid path the map keeps its archived shape.
    arc_rel = "sources/task-68cebbe16b/results/twin5_P1_carrier_samples.jsonl"
    arm_map_measured = {"single_clean": "single_clean",
                        "distorted_evidence_patchcopy_k0":
                            "single_distorted_evidence_ours16_patchcopy_k0",
                        "distorted_evidence_patchcopy_k1":
                            "single_distorted_evidence_ours16_patchcopy_k1",
                        "distorted_evidence_patchcopy_k2":
                            "single_distorted_evidence_ours16_patchcopy_k2",
                        "distorted_evidence_noise_k0":
                            "single_distorted_evidence_ours16_noise_k0",
                        "distorted_evidence_noise_k1":
                            "single_distorted_evidence_ours16_noise_k1",
                        "distorted_evidence_noise_k2":
                            "single_distorted_evidence_ours16_noise_k2",
                        "distorted_nonevidence_patchcopy_k0":
                            "single_distorted_nonevidence_ours16_patchcopy_k0",
                        "distorted_nonevidence_patchcopy_k1":
                            "single_distorted_nonevidence_ours16_patchcopy_k1",
                        "distorted_nonevidence_patchcopy_k2":
                            "single_distorted_nonevidence_ours16_patchcopy_k2",
                        "distorted_nonevidence_noise_k0":
                            "single_distorted_nonevidence_ours16_noise_k0",
                        "distorted_nonevidence_noise_k1":
                            "single_distorted_nonevidence_ours16_noise_k1",
                        "distorted_nonevidence_noise_k2":
                            "single_distorted_nonevidence_ours16_noise_k2"}
    arm_map_archived = dict(arm_map_measured)
    arm_map_archived["uniform@16"] = "uniform@16"
    arm_map_archived["no_frames"] = "no_frames"
    if QSELECT == "draw":
        arm_map = dict(arm_map_measured)
        if MODE == "pilot":
            arm_map["uniform@16"] = "uniform@16"
            arm_map["no_frames"] = "no_frames"
        shadow_by_video = {p["video_id"]: p for p in shadow["pairs"]}

        def _unchanged(r):
            return int(r["uid"]) == int(shadow_by_video[r["video_id"]]["uid"])

        unchanged_rows = [r for r in rows if _unchanged(r)]
        changed_units = [{"qid": r["qid"], "video_id": r["video_id"], "pair": r["pair"],
                          "uid": r["uid"], "uid_min": r["uid_min"],
                          "n_candidates": r["n_candidates"],
                          "draw_index": r["draw_index"]}
                         for r in rows if not _unchanged(r)]
        arc = load_jsonl(arc_rel) or {}
        n_expected = sum(1 for r in unchanged_rows if any(k in arc for k in _row_keys(r)))
        idv = archive_identity(unchanged_rows, arc_rel, arm_map)
        idv["n_expected"] = n_expected
        idv["n_changed_units"] = len(changed_units)
        idv["changed_units"] = changed_units
        idv["covered_rule"] = ("only units whose drawn question is the archived min-uid "
                               "question have an archived counterpart; changed units are "
                               "excluded, not counted as mismatches; uniform@16/no_frames "
                               "are compared only in pilot mode, where this reading "
                               "measures them")
        per_arm_ok = all(v["n"] == idv["n_overlap"] for v in idv["agreement"].values())
        idv["per_arm_coverage_complete"] = bool(per_arm_ok)
        idv["ok"] = bool(idv["n_mismatch"] == 0 and per_arm_ok
                         and int(idv["n_overlap"] or 0) == n_expected)
        if MODE == "full" and int(idv.get("n_overlap") or 0) != n_expected:
            raise SystemExit("distortion identity overlap %s != %d unchanged units covered "
                             "by the archive" % (idv.get("n_overlap"), n_expected))
    else:
        idv = archive_identity(rows, arc_rel, arm_map_archived)
    controls = INS.build(model, proc, dev, ids, ctrl, FWD, EXP_SEED)
    results = {"run": {},
               "claim_tested": {"d1": "a pre-registered distortion never destroyed the evidence "
                                      "it was meant to remove (behavioural criterion)"},
               "independent_unit": "video (one twin pair per video)",
               "identity": ident, "pin": {"pin_sha256": pin["pin_sha256"],
                                          "n_pairs_measured_this_run": len(pairs),
                                          "n_pairs_in_pin": n_pairs_in_pin,
                                          "screen_pairs": pin["screen_pairs"],
                                          "report_pairs": pin["report_pairs"],
                                          "e_star_outside_i16u_all": chk["pass"],
                                          "minuid_shadow_pin_sha256":
                                              (shadow["pin_sha256"] if shadow else None),
                                          "minuid_shadow_matches_archive":
                                              (bool(shadow["pin_sha256"]
                                                    == DD.ARCHIVED_MINUID_PIN_SHA)
                                               if shadow else None),
                                          "note": ("under ddraw-v1 the pin holds the DRAWN "
                                                   "questions; the shadow is the unchanged "
                                                   "min-uid construction, required to hash "
                                                   "to the archived value"
                                                   if shadow else "archived min-uid pin")},
               "question_selection": ("ddraw-v1: per-video uniform draw over the compliant "
                                      "candidate set, seed 20261023 registered in "
                                      "plan-distortion-random-question (replaces the "
                                      "archived smallest-uid rule; that reading stays on "
                                      "record)" if QSELECT == "draw" else
                                      "minuid: smallest integer uid among the compliant "
                                      "questions (archived rule)"),
               "unit_selection": drawrec,
               "structure": structure,
               "model": model_block(minfo, mident), "arms": arms, "bar_hits": bar,
               "per_form": per_form, "holm_evidence_vs_nonevidence": C.holm(list(holm_in.items())),
               "extraction": extraction_block(rows, arm_keys),
               "identity_vs_archive": idv,
               "instrument_controls": controls,
               "vision_tokens": controls["vision_tokens"],
               "n_measured": n,
               "verdict": {"n_pairs": n, "resolution_pp": (100.0 / n if n else None)}}
    if MODE == "pilot":
        results["identity_arms"] = {a: arms[a] for a in ("uniform@16", "no_frames")}
    apply_guard(results, rows, arm_keys, controls,
                {"reading": "distortion", "n_pairs": n, "bar_hits": bar,
                 "family": "holm_evidence_vs_nonevidence",
                 "withheld_if_instrument_failure": True})
    finish(results, "results/four_distortion_samples.jsonl")


# ================================================================ (d2) donor

def run_donor():
    raw = open(C3B_PIN_REL, "rb").read()
    import hashlib
    sha = hashlib.sha256(raw).hexdigest()
    if sha != C3B_PIN_SHA:
        raise SystemExit("carrier3b pin sha256 %s != frozen %s" % (sha, C3B_PIN_SHA))
    pin = json.loads(raw)
    import r4_common as R4
    qs = R4.build_questions(pin, len(pin["pairs"]), C3B_DIR)
    excl = set(R4.STAGE1_EXCLUDED_QIDS)
    vis = [q for q in qs if q["_r4_role"] == "visible" and q["_qid"] not in excl]
    # Pre-run, hashed, annotation-only.  The registered carrier-4 calibration value is
    # EXECUTED here, not only worded: over the pin's 24 visible questions the gold letters are
    # A6 B6 C6 D6 by construction, so the zero-information prior is A with 6 hits (four-way
    # tie, alphabetically first).  prior_block raises SystemExit on any mismatch, and this call
    # sits before load_model(), so a mismatch produces no reading.
    pin24 = prior_block([q["correct_letter"] for q in qs if q["_r4_role"] == "visible"],
                        "donor_pin24", expect_letter="A", expect_hits=6)
    # The same calibration on the in-force set (23 questions after the stage-1 exclusions:
    # A6 B5 C6 D6, prior still A with 6 hits), computed before the pilot shrink and before any
    # forward pass.
    prior = prior_block([q["correct_letter"] for q in vis], "donor_inforce_visible",
                        expect_letter="A", expect_hits=6)
    # Renamed (review §3): `n_visible_in_force` used to carry the size of the shrunken pilot
    # set (8) while the block also reported the 24-pair pin and the 23-question in-force pool.
    # The in-force pool is now named separately from what this run measured.
    n_inforce_pool = len(vis)
    if MODE == "pilot":
        want = sorted({q["_r4_pair"] for q in vis})[:8]
        vis = [q for q in vis if q["_r4_pair"] in set(want)]
    cache = R4.PoolCache(cap=2)
    d_rows = R4.d_precheck(vis, pin, C3B_DIR, k=0, cache=cache)
    agg = R4.aggregate_d(d_rows, len(pin["pairs"]))
    n_excluded = int(sum(1 for r in d_rows if not r["pass"]))
    agg["n_excluded_recounted"] = n_excluded
    agg["budget"] = 4
    enforce_destruction_budget(n_excluded, 4, "donor visible")
    model, proc, dev, minfo, ids = load_model()
    mident = L.model_identity()
    u16 = C.uniform_indices(G.K_CAND, G.B_MAIN)
    rows = []
    arm_keys = ["donor_floor_probe@16", "no_frames"]
    accepted = {r["qid"]: r for r in d_rows if r["pass"]}
    ctrl = INS.new_state()
    for q in vis:
        if q["_qid"] not in accepted:
            continue
        TOKENS.clear()
        donor, di, dpath, _u = R4.donor_slot_array(q, pin, C3B_DIR, 0,
                                                   int(accepted[q["_qid"]]["accepted_offset"]),
                                                   cache)
        dpils = C.to_pil(donor, G.MAX_PIXELS)
        prompt = G.prompt_for(q)
        letters, raws = {}, {}
        letters["donor_floor_probe@16"], raws["donor_floor_probe@16"] = ask(
            model, proc, dpils, prompt, dev, "donor_floor_probe@16")
        letters["no_frames"], raws["no_frames"] = ask(model, proc, [], prompt, dev, "no_frames")
        gold = q["correct_letter"]
        letters["constant_prior"] = prior["prior_letter"]
        raws["constant_prior"] = "PRIOR:" + str(prior["prior_letter"])
        correct = {a: int(letters[a] == gold) for a in arm_keys + ["constant_prior"]}
        row = {"qid": q["_qid"], "video_id": q["video_id"], "pair": int(q["_r4_pair"]),
               "gold": gold, "donor_pair": int(di), "donor_video": os.path.basename(dpath),
               "letters": letters, "raw": raws, "vision_tokens": dict(TOKENS),
               "correct": correct}
        rows.append(row)
        app_jsonl("results/four_donor_samples.jsonl", row)
        # Instrument controls (review §1), on this reading's own 16-frame arm.
        if ctrl["paths"] is None:
            INS.set_paths(ctrl, frames16=dpils, frame1=dpils[0], prompt=prompt,
                          arm16="donor_floor_probe@16",
                          arm1="control-only 1-frame ask (this reading has no 1-frame arm)")
        INS.add_perm_unit(ctrl, q["_qid"], q["question"], q["candidates"], gold,
                          letters["donor_floor_probe@16"], raws["donor_floor_probe@16"],
                          dpils, "donor_floor_probe@16")
        log("donor %2d/%d %-40s donor=%s" % (len(rows), len(vis), q["_qid"], letters))
    arms = {a: arm_entry([r["correct"][a] for r in rows])
            for a in arm_keys + ["constant_prior"]}
    st = G.seed_streams(EXP_SEED)
    cmp_ = compare(rows, "donor_floor_probe@16", "no_frames", st["s_boot"])
    cmp_prior = compare(rows, "donor_floor_probe@16", "constant_prior", st["s_boot"])
    cmp_nf_prior = compare(rows, "no_frames", "constant_prior", st["s_boot"])
    controls = INS.build(model, proc, dev, ids, ctrl, FWD, EXP_SEED)
    results = {"run": {},
               "claim_tested": {"d2": "instrument diagnostic only: where the donor arm sits "
                                      "relative to the no-frames arm and to the zero-information "
                                      "option prior. Not a finding, not a method comparison, not "
                                      "in any delta or decision chain."},
               "independent_unit": "video (one twin pair per video; questions are measurement "
                                   "units)",
               "identity": {"pin_sha256": sha, "pin_expected": C3B_PIN_SHA,
                            "n_pairs": len(pin["pairs"]),
                            "n_units_measured_this_run": len(vis),
                            "n_inforce_pool": n_inforce_pool,
                            "stage1_excluded": sorted(excl),
                            "carrier_manifest_expected":
                                "38e93f0605209aa6192188a1c83e9baf6472a66cd2cc00e4dcd3fa677d24ee10"},
               "model": model_block(minfo, mident), "arms": arms, "comparison": cmp_,
               "comparison_vs_prior": cmp_prior, "no_frames_vs_prior": cmp_nf_prior,
               "pretest_prior": prior,
               "pretest_prior_calibration": {
                   "pin24_visible": pin24,
                   "in_force_visible": {"expected_letter": "A", "expected_hits": 6,
                                        "observed_letter": prior["prior_letter"],
                                        "observed_hits": prior["prior_hits"],
                                        "n": prior["n"]},
                   "selftest": prior_guard_selftest(),
                   "note": "the carrier-4 calibration value (24 visible questions A6 B6 C6 D6, "
                           "in-force 23 A6 B5 C6 D6) is checked by prior_block before "
                           "load_model(); a mismatch raises SystemExit and produces no reading"},
               "crash_guard": {"budget": 4, "n_excluded": n_excluded,
                               "enforced_by": "raise SystemExit(70) before any model step",
                               "selftest": guard_selftest()},
               "p_d": agg, "extraction": extraction_block(rows, arm_keys),
               "instrument_controls": controls,
               "vision_tokens": controls["vision_tokens"],
               "n_measured": len(rows),
               "verdict": {"n_units": len(rows),
                           "resolution_pp": (100.0 / len(rows) if rows else None)}}
    apply_guard(results, rows, arm_keys + ["constant_prior"], controls,
                {"reading": "donor", "role": "instrument diagnostic only: not a finding, not "
                                             "in any delta and not in any decision chain",
                 "withheld_if_instrument_failure": True})
    finish(results, "results/four_donor_samples.jsonl")


# ================================================================ option_mapping control at full n
# Plan plan-single-frame-random-question-r2 §5-§7, control C.  A control-only reading:
# the option<->letter mapping criterion on all 24 registered units, no measurement arm.
OPTIONCTRL_PAIRS_REL = "results/optionctrl_pairs.jsonl"
# The candidate file's sha256, as written by task-af72fbccf4 (pinned Evidence).  Recomputing the
# same draw in a new run must reproduce it byte for byte; that is a cross-run check on the
# sampling, not a reading.
OPTIONCTRL_CAND_SHA_SMOKE = "f84373174615142377802359341b503fb14e31f43704fd94b04524243ef7ace4"
QCTRL_ORIG_KIND = "control:letterseek_orig_single_frame_j_star"

# The C0 archive (read_from -> sources/<task>/<output path>).  Comparing the 48 pair records
# against it is the control's own test-retest across runs: it says how far a greedy decode on
# the same prompts re-executed on the same commit reproduces itself.  It gates nothing.
C0_PAIRS_REL = "sources/task-0cdb808f53/results/optionctrl_pairs.jsonl"


def compare_to_c0(pairs_rel):
    """Per-pair comparison of this run's 48 records against C0's, field by field."""
    if not os.path.exists(C0_PAIRS_REL):
        return {"available": False,
                "note": "the C0 archive copy was not present in this run (task.read_from); "
                        "nothing is inferred from its absence"}
    def _load(p):
        out = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    a, b = _load(C0_PAIRS_REL), _load(pairs_rel)
    ka = {(r.get("qid"), r.get("order")): r for r in a}
    kb = {(r.get("qid"), r.get("order")): r for r in b}
    fields = ["letter", "letter_points_at", "raw_naming", "raw_retest"]
    common = [k for k in kb if k in ka]
    diffs = []
    for k in common:
        for f in fields:
            if ka[k].get(f) != kb[k].get(f):
                diffs.append({"qid": k[0], "order": k[1], "field": f,
                              "c0": ka[k].get(f), "c1": kb[k].get(f)})
    return {
        "available": True, "c0_task": "task-0cdb808f53",
        "n_c0_pairs": len(a), "n_c1_pairs": len(b), "n_matched_keys": len(common),
        "n_c1_keys_absent_in_c0": int(len([k for k in kb if k not in ka])),
        "fields_compared": fields,
        "n_field_mismatches": len(diffs),
        "n_pairs_all_fields_identical": int(sum(
            1 for k in common if all(ka[k].get(f) == kb[k].get(f) for f in fields))),
        "mismatches": diffs[:50],
        "rule": "descriptive only. A mismatch says the model's own output was not bit-identical "
                "across runs; it is NOT evidence that the letter->option map changed and it "
                "changes no verdict. The two unevaluable pairs of C0 are the pairs of interest: "
                "if they are unevaluable again, the refusal rule and the coverage line are the "
                "only things that can make them judgeable.",
    }



def run_optionctrl():
    """The option_mapping control on the 24 registered units; zero measurement arms.

    Every model call here is a control:
      * one letter ask per unit in the cell's ORIGINAL option order, on the same 1-frame
        prompt and the same j* image the cell's single_frame_j_star arm sends, tagged
        control:* -- the cell reads that letter from the arm itself; this run reproduces the
        identical call so the original-order pairs exist at all, and records each prompt's
        sha256;
      * the permuted-order letter ask and the four naming asks that
        paper4/instrument.option_control makes per unit.
    The question set is rebuilt by qdraw-v1 from the same annotations and the registered
    seed, so the 24 qids are the cell's 24 qids and the candidate file must hash to the value
    the smoke run wrote.
    """
    if QSELECT != "draw":
        raise SystemExit("P4_READING=optionctrl requires P4_QSELECT=draw")
    if QDRAW_SEED != QD.QDRAW_SEED:
        raise SystemExit("qdraw seed %d != registered %d" % (QDRAW_SEED, QD.QDRAW_SEED))
    groups, skipped = F.eligible_by_video(F.FRESH_META, F.FRESH_DIR)
    _design, confirmation, _shuffled = F.split_videos(groups)
    units, shadow_golds, drawrec = QD.build(confirmation, groups, log=log)
    ident = {"dataset": "lvbench-fresh-24209d43cbdd",
             "manifest_sha256": L.manifest_sha256(root=F.FRESH_DIR)[0],
             "manifest_expected": F.FRESH_MANIFEST_SHA,
             "annotation_sha256": L.sha256_file(F.FRESH_META),
             "annotation_expected": ANNOT_SHA_EXPECTED,
             "split_seed": F.SPLIT_SEED, "side": "confirmation",
             "n_units": len(units),
             "note": "the 24 units of the single-frame cell, rebuilt by qdraw-v1; this run "
                     "measures the option_mapping control on them and runs no measurement arm"}
    dist = {}
    for u in units:
        dist[u["n_selected_in_video"]] = dist.get(u["n_selected_in_video"], 0) + 1
    cand_sha = drawrec["candidates_file"]["sha256"]
    checks = {
        "manifest_matches": bool(ident["manifest_sha256"] == F.FRESH_MANIFEST_SHA),
        "annotation_matches": bool(ident["annotation_sha256"] == ANNOT_SHA_EXPECTED),
        "pool_videos": len(groups), "pool_videos_expected": 32,
        "pool_eligible_questions": int(sum(len(c) for _v, c in groups)),
        "pool_eligible_expected": 134,
        "pool_selected_questions": int(sum(1 for _v, c in groups
                                           for (_u, _a, _b, q, _so) in c if F.rule_select(q))),
        "pool_selected_expected": 55,
        "confirmation_units": len(units), "confirmation_units_expected": 24,
        "design_units": len(_design), "design_units_expected": 8,
        "one_unit_per_video": bool(len({u["video_id"] for u in units}) == len(units)),
        "draw_k_distribution": {str(k): v for k, v in sorted(dist.items())},
        "draw_k_distribution_matches": bool(dist == QD.EXPECTED_K_DISTRIBUTION),
        "draw_recheck_from_disk_ok": bool(drawrec["recheck_from_disk_ok"]),
        "draw_unit_uid_matches_record": bool(
            {u["video_id"]: int(u["uid"]) for u in units} ==
            {p["video_id"]: int(p["uid"]) for p in drawrec["per_video"]}),
        "candidates_file_sha256": cand_sha,
        "candidates_sha_matches_smoke": bool(cand_sha == OPTIONCTRL_CAND_SHA_SMOKE),
        "draw_n_index0": int(sum(1 for u in units if u["draw_index"] == 0)),
        "draw_n_singleton": int(sum(1 for u in units if u["n_selected_in_video"] == 1)),
        "draw_moved_at_least_one": bool(any(u["draw_index"] != 0 for u in units)),
    }
    checks["all_structure_ok"] = bool(
        checks["manifest_matches"] and checks["annotation_matches"]
        and checks["pool_videos"] == 32 and checks["pool_eligible_questions"] == 134
        and checks["pool_selected_questions"] == 55
        and checks["confirmation_units"] == 24 and checks["design_units"] == 8
        and checks["one_unit_per_video"]
        and checks["draw_k_distribution_matches"] and checks["draw_recheck_from_disk_ok"]
        and checks["draw_unit_uid_matches_record"] and checks["draw_moved_at_least_one"])
    if not checks["all_structure_ok"]:
        raise SystemExit("optionctrl structure check failed: %s"
                         % json.dumps(checks, default=str, ensure_ascii=False))
    # Positive control B (annotation read): 0 forwards, before the model is loaded.
    prior_shadow = prior_block(shadow_golds, "single_minuid24_shadow",
                               expect_letter="A", expect_hits=9)
    prior = prior_block([u["gold"] for u in units], "single_drawn24")
    model, proc, dev, minfo, ids = load_model()
    mident = L.model_identity()
    if os.path.exists(OPTIONCTRL_PAIRS_REL):
        os.remove(OPTIONCTRL_PAIRS_REL)
    ctrl_units, rows = [], []
    for i, u in enumerate(units):
        t = time.time()
        ev = L.decode_indices(u["video_path"], [u["j_star"]])[0]
        evpil = C.to_pil(ev[None], G.MAX_PIXELS)[0]
        PHASE["decode_s"] += time.time() - t
        q = {"question": u["question"], "candidates": list(u["options"]),
             "correct_letter": u["gold"]}
        prompt = G.prompt_for(q)
        t = time.time()
        raw, letter = G.answer_arm(model, proc, [evpil], prompt, dev, counter=FWD,
                                   kind=QCTRL_ORIG_KIND)
        PHASE["forward_s"] += time.time() - t
        ctrl_units.append({"qid": u["qid"], "arm": "single_frame_j_star",
                           "question": u["question"], "options": list(u["options"]),
                           "gold": u["gold"], "pils": [evpil],
                           "letter": letter, "raw": raw})
        rows.append({"qid": u["qid"], "video_id": u["video_id"], "gold": u["gold"],
                     "j_star": int(u["j_star"]), "n_frames": int(u["n_frames"]),
                     "fps": float(u["fps"]), "draw_index": int(u["draw_index"]),
                     "n_selected_in_video": int(u["n_selected_in_video"]),
                     "letter_orig": letter, "raw_orig": raw,
                     "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
        log("optionctrl %2d/%d %-18s orig_letter=%s" % (i + 1, len(units), u["video_id"],
                                                        letter))
    t = time.time()
    oc = INS.option_control(model, proc, dev, ctrl_units, FWD, INS.perm_seed(EXP_SEED),
                            letter_ids=ids)
    PHASE["forward_s"] += time.time() - t
    mp = oc["mapping"]
    pl = mp["pass_line"]
    nr = oc["naming_retest"]
    # ---- per-pair record (declared output) + breakdowns -----------------------------------
    by_order, by_letter, c_u, d_u, per_unit = {}, {}, [], [], []
    for r in oc["units"]:
        c = 0
        d = 0
        for p in r["pairs"]:
            bad = bool(p["evaluable"] and not p["agrees"])
            ret = p.get("retest") or {}
            both = (p.get("named_text") is not None and ret.get("named_text") is not None)
            dis = bool(both and p["named_text"] != ret.get("named_text"))
            c += int(bad)
            d += int(dis)
            if bad:
                by_order[p["order"]] = by_order.get(p["order"], 0) + 1
                k = str(p["letter"])
                by_letter[k] = by_letter.get(k, 0) + 1
            app_jsonl(OPTIONCTRL_PAIRS_REL, {
                "qid": r["qid"], "video_id": r["qid"].split("#")[0],
                "order": p["order"], "permutation": r["permutation"],
                "letter": p["letter"], "letter_points_at": p["letter_points_at"],
                "naming_text": p["named_text"], "naming_parse": p["named_parse"],
                "retest_text": ret.get("named_text"), "retest_parse": ret.get("parse"),
                "raw_naming": p["raw_named"], "raw_retest": ret.get("raw"),
                "retest_disagrees": dis,
                "evaluable": bool(p["evaluable"]), "contradiction": bad,
                "naming_refusal": p["named_parse"] in ("refusal_matched_option",
                                                       "refusal_no_option"),
                "unevaluable_reason": (None if p["evaluable"]
                                       else (p["named_parse"] or "letter_missing"))})
        c_u.append(c)
        d_u.append(d)
        per_unit.append({"qid": r["qid"], "permutation": r["permutation"],
                         "n_contradictions": c, "n_retest_disagree": d,
                         "n_letter_orig": r["letter"], "n_letter_perm": r["letter_permuted"],
                         "named_orig": (r["pairs"][0] or {}).get("named_text"),
                         "named_perm": (r["pairs"][1] or {}).get("named_text")})
    diff = [c - d for c, d in zip(c_u, d_u)]
    n_pos = int(sum(1 for x in diff if x > 0))
    n_neg = int(sum(1 for x in diff if x < 0))
    n_zero = int(sum(1 for x in diff if x == 0))
    if n_pos + n_neg:
        from scipy.stats import binomtest
        sign_p = float(binomtest(min(n_pos, n_neg), n_pos + n_neg, 0.5).pvalue)
    else:
        sign_p = 1.0
    # unit-clustered bootstrap of (contradiction rate - retest rate); the two pairs of a unit
    # are not independent, so the unit is the resampling unit.
    per_unit_diff = [x / 2.0 for x in diff]
    rng = np.random.default_rng(int(G.seed_streams(EXP_SEED)["s_boot"]))
    n_u = len(per_unit_diff)
    boot = np.empty(10000)
    arr = np.asarray(per_unit_diff, float)
    for b in range(10000):
        idx = rng.integers(0, n_u, size=n_u)
        boot[b] = arr[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    summary = {
        "n_pairs": mp["n_pairs"], "n_evaluable": mp["n_evaluable"],
        "n_contradictions": mp["n_contradictions"],
        # ---- Plan r3 §9-§10: refusal accounting and the derived coverage line ------------
        "n_unevaluable": mp.get("n_unevaluable_charged"),
        "contradictions_plus_unevaluable": mp.get("contradictions_plus_unevaluable"),
        "r2_rule_status": mp.get("r2_rule_status"),
        "coverage": mp.get("coverage"),
        "n_refusal_like": mp.get("n_refusal_like"),
        "n_refusal_matched_option": mp.get("n_refusal_matched_option"),
        "n_refusal_no_option": mp.get("n_refusal_no_option"),
        "n_units_with_refusal_option": mp.get("n_units_with_refusal_option"),
        "n_recovered_pairs": mp.get("n_recovered_pairs"),
        "recovered_pairs": mp.get("recovered_pairs"),
        "unevaluable_pairs": mp.get("unevaluable_pairs"),
        "refusal_rule": mp.get("refusal_rule"),
        "contradiction_rate": pl["contradiction_rate"],
        "contradiction_rate_cp95": (L.cp_ci(mp["n_contradictions"], mp["n_evaluable"])
                                    if mp["n_evaluable"] else [None, None]),
        "n_retest_both": nr["n_both_parsed"], "n_retest_disagree": nr["n_disagree"],
        "retest_rate": nr["disagreement_rate"],
        "retest_rate_cp95": (L.cp_ci(nr["n_disagree"], nr["n_both_parsed"])
                             if nr["n_both_parsed"] else [None, None]),
        "p_retest_upper95": pl["p_retest_upper95"],
        "tolerance_b": pl["tolerance_b"],
        "tolerance_b_strict": pl["tolerance_b_strict"],
        "status": pl["status"], "reason": pl["reason"],
        "smallest_detectable_rate": pl["smallest_detectable_rate"],
        "naming_step_unstable": pl["naming_step_unstable"],
        "passes_only_because_of_noise": pl["passes_only_because_of_noise"],
        "can_prove": pl["can_prove"], "cannot_prove": pl["cannot_prove"],
        "rate_difference_contradiction_minus_retest": (
            (pl["contradiction_rate"] - nr["disagreement_rate"])
            if (pl["contradiction_rate"] is not None
                and nr["disagreement_rate"] is not None) else None),
        "rate_difference_unit_cluster_boot95": [float(lo), float(hi)],
        "unit_level_excess": {"n_units": n_u, "n_pos": n_pos, "n_neg": n_neg,
                              "n_zero": n_zero,
                              "sign_test_p": sign_p,
                              "definition": "c_u = contradictions in unit u's 2 orders, "
                                            "d_u = retest disagreements in the same 2 pairs; "
                                            "descriptive, it changes no verdict"},
        "contradictions_by_order": dict(sorted(by_order.items())),
        "contradictions_by_letter": dict(sorted(by_letter.items())),
        "per_unit": per_unit,
        "identical_repeat": nr["identical_repeat"],
        # NOTE: `option_control.fault_injection` (top level) is an int in the producer -- the
        # local name is rebound to the mapping-fault-signature count.  The injection block is
        # the one collected into `mapping`; read it there and leave the shared shape alone.
        "fault_injection_fired": bool(mp["fault_injection"]["fired"]),
        "fault_injection_verdict_changes": bool(mp["fault_injection"]["verdict_changes"]),
        "arithmetic_ok": bool(oc["arithmetic_ok"]),
        "candidates_sha_matches_smoke": bool(checks["candidates_sha_matches_smoke"]),
        "rule": "the registered pass line; see option_control.mapping.pass_line",
    }
    cmp_c0 = compare_to_c0(OPTIONCTRL_PAIRS_REL)
    results = {
        "run": {},
        "claim_tested": {"control": "the option<->letter mapping criterion at the cell's "
                                    "registered size (24 units x 2 option orders = 48 pairs): "
                                    "its contradiction rate against the naming step's own "
                                    "test-retest rate on the same pairs. A control "
                                    "measurement; it is not a reading of any arm."},
        "independent_unit": ("video (2 option orders per video; every interval is clustered at "
                             "the video level)"),
        "identity": ident, "structure": checks, "skipped_videos": skipped,
        "model": model_block(minfo, mident),
        "question_selection": ("qdraw-v1: uniform draw over each video's rule-selected "
                              "candidate set, seed registered in the Plan"),
        "unit_selection": drawrec,
        "registered_units": {"n_units": len(units), "n_measured": len(units),
                             "n_measured_qids": [u["qid"] for u in units],
                             "note": "all 24 registered units; this run measures only the "
                                     "option_mapping control"},
        "pretest_prior": prior, "prior_minuid_shadow": prior_shadow,
        "option_control": oc,
        "option_control_summary": summary,
        "reproduce_vs_c0": cmp_c0,
        "per_unit_letters": rows,
        "gate_relevance": {
            "decides": "whether option_mapping may be read as satisfied at the cell's "
                       "registered size",
            "measurement_arms_run": 0,
            "does_not_remeasure": "letter_decode, determinism and answer_visibility (the "
                                  "4-unit smoke measured 5/5, 3/3, and 6/6 answer plus 6/6 "
                                  "score; this version of the design does not touch them), "
                                  "and it does not evaluate the near-constant-letter trigger, "
                                  "which needs the measurement arms' letter distributions "
                                  "(n>=5) and can only be evaluated in the cell's full run",
            "original_order_letter": "produced by one control-kind ask with the identical "
                                     "prompt string, the same j* frame and the same greedy "
                                     "decode as the cell's single_frame_j_star arm; every "
                                     "unit's prompt sha256 is recorded in per_unit_letters",
            "stage_note": "stage=smoke at the sandbox level because this design's pilot gate "
                          "has not passed; it is not a full-stage run of the cell"},
        "landing": None,
        "n_measured": len(units),
        "verdict": {"n_units": len(units), "n_pairs": mp["n_pairs"],
                    "option_mapping_status": pl["status"],
                    "contradiction_rate": pl["contradiction_rate"],
                    "retest_rate": nr["disagreement_rate"],
                    "tolerance_b": pl["tolerance_b"],
                    "tolerance_b_strict": pl["tolerance_b_strict"],
                    "smallest_detectable_rate": pl["smallest_detectable_rate"],
                    "n_evaluable": mp["n_evaluable"],
                    "n_unevaluable": mp.get("n_unevaluable_charged"),
                    "contradictions_plus_unevaluable": mp.get(
                        "contradictions_plus_unevaluable"),
                    "r2_rule_status": mp.get("r2_rule_status"),
                    "e_max_allowed": (mp.get("coverage") or {}).get("e_max_allowed"),
                    "required_coverage_frac": (mp.get("coverage") or {}).get(
                        "required_coverage_frac"),
                    "coverage_sufficient": (mp.get("coverage") or {}).get("sufficient"),
                    "n_recovered_pairs": mp.get("n_recovered_pairs")},
    }
    finish(results, OPTIONCTRL_PAIRS_REL)


ANNOT_SHA_EXPECTED = F.ANNOT_SHA


def main():
    log("reading=%s mode=%s stage=%s plan=%s@%d seed=%d"
        % (READING, MODE, STAGE, PLAN_ID, PLAN_VERSION, EXP_SEED))
    if READING == "selection":
        run_selection()
    elif READING == "single":
        run_single()
    elif READING == "distortion":
        run_distortion()
    elif READING == "donor":
        run_donor()
    elif READING == "optionctrl":
        run_optionctrl()
    elif READING == "guard_selftest":
        enforce_destruction_budget(5, 4, "selftest")
        print("guard did not raise", flush=True)
        raise SystemExit(0)
    elif READING == "prior_guard_selftest":
        prior_block(["A", "A", "B", "C"], "prior_guard_selftest",
                    expect_letter="D", expect_hits=9)
        print("prior calibration guard did not raise", flush=True)
        raise SystemExit(0)
    else:
        raise SystemExit("set P4_READING to selection|single|distortion|donor")


if __name__ == "__main__":
    main()
