"""Registered question draw for the distortion cell (rule ddraw-v1).

Plan: plan-distortion-random-question (version bound by P4_PLAN_VERSION).

Why this module exists
----------------------
The distortion cell used to take, on each of its 17 pair videos, the compliant question with
the smallest integer uid (twin5/lv_common.py build_pin: cand.sort(key=lambda x: x[0]) then
cand[0]; pin rule string "one_question_per_video: smallest uid").  That deterministic rule
was registered in plan-paper-full-twin-lvbench-r5@v4 / -r6 section 2 and measured by
the archived full run evidence.task-259e5c2b3a; the filing notes disclose it as a
deterministic draw.  The person's delegate ruled integrity over speed: the cell's question
draw is re-registered as a stratified random draw -- the registered standard for per-video
question selection in this project (one question per video drawn uniformly over the video's
compliant candidate set; the qdraw-v1 form registered for the single-frame cell in
plan-single-frame-random-question) -- and the cell is re-run under it.  The archived
deterministic reading stays on record; nothing about it is deleted.

The rule, verbatim (this text is also embedded in the draw record)
-----------------------------------------------------------------
C(v) = the questions of video v admitted by the frozen inclusion rule of the distortion
carrier (twin5/lv_common.py build_pin, unchanged: time_reference parses as MM:SS-MM:SS;
0 <= end-start <= 3.0 s; end <= duration+1.0 s; the question splits into exactly four
non-empty options A-D; answer in {A,B,C,D}), ordered by ascending integer uid.  There is no
second SELECT layer in this cell: the inclusion rule is the whole rule.
k(v) ~ Uniform{0, ..., |C(v)|-1}:
    ss = numpy.random.SeedSequence([20261023, int(sha256(v.encode()).hexdigest()[:8], 16)])
    rng = numpy.random.default_rng(ss)
    k = int(rng.integers(0, len(C(v))))
chosen = C(v)[k].  No tie exists: integer uids are unique inside a video, and a duplicate is
a structural error that crashes the run rather than being deduplicated silently.  |C(v)| = 0
leaves the video a non-unit exactly as in the archived construction (the pool is the 20
local videos, 17 of which have at least one compliant question; the 17 unit videos and the
screen/report split must be identical to the archived min-uid pin).

Nothing here touches a model or decodes a frame: the whole draw is a function of the
annotation file and one seed, so it costs zero forward passes.  The pair arithmetic
(j*, grid, guard, i*, e*, split) is the archived construction verbatim; the shadow control
rebuilds the min-uid pin with the untouched lv_common.build_pin and requires its sha256 to
equal the archived value, and every video whose drawn question is the min-uid question must
have its structural fields byte-identical to the shadow's, so a divergence of this module's
copied arithmetic from the archived one cannot pass silently.
"""
import hashlib
import json
import os

import numpy as np

import twin5.lv_common as L

DDRAW_SEED = 20261023
RULE_VERSION = "ddraw-v1"
RULE_TEXT = (
    "C(v) = the questions of video v admitted by the frozen inclusion rule of the distortion "
    "carrier (lv_common.build_pin, unchanged: time_reference parses as MM:SS-MM:SS; "
    "0 <= end-start <= 3.0 s; end <= duration+1.0 s; the question splits into exactly four "
    "non-empty options A-D; answer in {A,B,C,D}), ordered by ascending integer uid. "
    "k(v) ~ Uniform{0,...,|C(v)|-1} with ss = numpy.random.SeedSequence([20261023, "
    "int(hashlib.sha256(v.encode()).hexdigest()[:8], 16)]); rng = "
    "numpy.random.default_rng(ss); k = int(rng.integers(0, len(C(v)))). chosen = C(v)[k]. "
    "No tie exists (integer uids are unique inside a video; a duplicate uid is a structural "
    "error and crashes the run, it is not deduplicated). |C(v)| = 0 leaves the video a "
    "non-unit exactly as in the archived construction; no video is dropped that the archived "
    "rule kept, no question is substituted, n stays 17, and the 17 unit videos and the "
    "screen 8 / report 9 split must be identical to the archived min-uid pin.")
CAND_REL = "results/distortion_ddraw_candidates.jsonl"
DRAW_REL = "results/distortion_ddraw.json"
# The min-uid pin the archived full run (evidence.task-259e5c2b3a) built and measured.
# The untouched lv_common.build_pin must reproduce it byte for byte; that is the shadow
# control proving the annotation read, the inclusion rule and the pair arithmetic did not
# move when the draw was added.
ARCHIVED_MINUID_PIN_SHA = "94e667e1f7efcdab17038ac9635e14889772d7471bfa06afa715288300360944"

# Structural fields that must be byte-identical to the shadow on every video whose drawn
# question is the min-uid question (the whole point: same arithmetic, different question only
# where the draw moved it).
_PAIR_STRUCT_FIELDS = ("j_star", "i_star", "e_star", "grid_idx", "grid_idx_guarded",
                       "fps", "n_frames", "video_sha256", "video_bytes")
_VIDEO_LEVEL_FIELDS = ("fps", "n_frames", "video_sha256", "video_bytes", "video_type")


def seed_sequence(video_id):
    h = int(hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.SeedSequence([DDRAW_SEED, h])


def draw_index(video_id, n_candidates):
    """The registered draw.  One independent stream per video, so a reader can recompute a
    single video without knowing the rest of the set (and without this file)."""
    rng = np.random.default_rng(seed_sequence(video_id))
    return int(rng.integers(0, int(n_candidates)))


def _candidates(qa_rows, dur):
    """The frozen inclusion rule, verbatim from lv_common.build_pin (same tolerances, same
    order of checks).  Returns [(uid, a, b, q, (stem, opts)), ...] sorted by ascending uid."""
    cand = []
    for q in qa_rows:
        p = L.parse_time_reference(q.get("time_reference", ""))
        if p is None:
            continue
        a, b = p
        if b < a or (b - a) > L.MAX_INTERVAL_S or b > dur + 1.0:
            continue
        o = L.parse_options(q.get("question", ""))
        if o is None:
            continue
        if q.get("answer") not in list("ABCD"):
            continue
        cand.append((int(q["uid"]), a, b, q, o))
    cand.sort(key=lambda x: x[0])
    return cand


def _sha_file(rel):
    try:
        return hashlib.sha256(open(rel, "rb").read()).hexdigest()
    except OSError:
        return None


def _jdump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def build(log=print):
    """The registered draw for all 17 pair videos, plus the min-uid shadow pin.

    Returns (pin, shadow, record):
      pin     the drawn pin (same dict shape as lv_common.build_pin's, plus the draw fields
              on each pair); the measurement reads only this;
      shadow  the untouched lv_common.build_pin() output; its sha256 must equal
              ARCHIVED_MINUID_PIN_SHA or the run crashes before any model step;
      record  the draw record written to DRAW_REL.

    0 forward passes, 0 frames decoded.
    """
    shadow = L.build_pin()
    if shadow["pin_sha256"] != ARCHIVED_MINUID_PIN_SHA:
        raise SystemExit(
            "ddraw: min-uid shadow pin sha256 %s != archived %s: the annotation read, the "
            "inclusion rule or the pair arithmetic drifted; no draw is trusted"
            % (shadow["pin_sha256"], ARCHIVED_MINUID_PIN_SHA))
    shadow_by_video = {p["video_id"]: p for p in shadow["pairs"]}

    meta = [json.loads(l) for l in open(L.META) if l.strip()]
    by_key = {r["key"]: r for r in meta}
    vids = sorted(f[:-4] for f in os.listdir(L.LVB_DIR) if f.endswith(".mp4"))
    cand_rows, per_video, pairs, skipped = [], [], [], []
    for vid in vids:
        r = by_key.get(vid)
        if r is None:
            skipped.append({"video": vid, "why": "no annotation row"})
            continue
        dur = float(r["video_info"]["duration_minutes"]) * 60.0
        cand = _candidates(r["qa"], dur)
        if not cand:
            skipped.append({"video": vid, "why": "no eligible question"})
            continue
        uids = [int(u) for (u, _a, _b, _q, _o) in cand]
        if len(set(uids)) != len(uids):
            raise SystemExit("ddraw: duplicate integer uid inside video %s: %s"
                             % (vid, sorted(uids)))
        k = draw_index(vid, len(cand))
        uid, a, b, q, (stem, opts) = cand[k]
        order = {int(u): i for i, (u, _a, _b, _q, _o) in enumerate(cand)}
        for (u_, _a2, _b2, q2, _so2) in cand:
            cand_rows.append({
                "video_id": vid, "uid": int(u_),
                "question_type": list(q2.get("question_type") or []),
                "time_reference": q2.get("time_reference"), "gold": q2.get("answer"),
                "order_index": order.get(int(u_)),
                "selected": bool(int(u_) == int(uid))})
        # ---- pair arithmetic: verbatim from lv_common.build_pin -----------------------
        vp = os.path.join(L.LVB_DIR, vid + ".mp4")
        fps, n = L.probe_video(vp)
        t_ev = 0.5 * (a + b)
        j_star = int(min(max(round(t_ev * fps), 0), n - 1))
        idx = [int(min(max(round((i + 0.5) * n / L.K), 0), n - 1)) for i in range(L.K)]
        n_slots_equal_j = int(sum(1 for x in idx if x == j_star))
        # keep j* only at e*: any grid slot whose frame is j* is nudged by one frame
        idx_guarded = list(idx)
        for s in range(L.K):
            if idx_guarded[s] == j_star:
                idx_guarded[s] = int(min(max(idx[s] + (1 if idx[s] + 1 < n else -1), 0),
                                         n - 1))
        i_star = L.nearest_slot_to_frame(j_star, idx_guarded)
        e_star = L.evidence_slot(i_star)
        pairs.append({
            "pair": len(pairs), "video_id": vid, "video_type": r.get("type"),
            "uid": uid, "time_reference": q.get("time_reference"), "t_ev_s": t_ev,
            "interval_s": b - a, "fps": fps, "n_frames": n,
            "j_star": int(j_star), "i_star": int(i_star), "e_star": int(e_star),
            "grid_idx": idx, "grid_idx_guarded": idx_guarded,
            "n_grid_slots_equal_j_star_before_guard": n_slots_equal_j,
            "question": stem, "options": opts, "gold": q["answer"],
            "question_type": q.get("question_type"),
            "video_path": vp,
            "video_bytes": os.path.getsize(vp),
            "video_sha256": L.sha256_file(vp),
            "n_candidates": len(cand), "draw_index": int(k),
            "uid_min": int(cand[0][0]), "changed_from_minuid": bool(int(k) != 0),
        })
        per_video.append({"video_id": vid, "n_candidates": len(cand), "k": int(k),
                          "uid": int(uid), "qid": "%s#%d" % (vid, int(uid)),
                          "uid_min": int(cand[0][0]),
                          "changed_from_minuid": bool(int(k) != 0)})
    pairs.sort(key=lambda p: p["video_id"])
    for i, p in enumerate(pairs):
        p["pair"] = i
    per_video.sort(key=lambda p: p["video_id"])
    cand_rows.sort(key=lambda r: (r["video_id"], r["uid"]))
    n = len(pairs)
    # the split is the archived pure function of the sorted video ids (lv_common: rng(11)):
    # same 17 videos in the same order must give the same screen 8 / report 9.
    perm = np.random.default_rng(L.SPLIT_SEED).permutation(n).tolist()
    screen_ids = sorted(int(i) for i in perm[:L.SCREEN_N])
    report_ids = sorted(int(i) for i in perm[L.SCREEN_N:])
    pin = {
        "carrier": "lvbench-24209d43cbdd",
        "manifest_sha256": L.MANIFEST_SHA,
        "rule": {"max_interval_s": L.MAX_INTERVAL_S,
                 "one_question_per_video":
                     "ddraw-v1: uniform draw over the compliant candidate set, seed 20261023 "
                     "registered in plan-distortion-random-question (replaces the "
                     "archived 'smallest uid' rule)",
                 "pool": "32 uniform midpoints, round((i+0.5)*n/32)",
                 "I16u": L.I16U, "e_star_rule": "nearest slot to i* not in I16u, +d first",
                 "evidence_frame": "round(midpoint_seconds * fps)",
                 "mark": "magenta border, frac=%.2f of min(H,W)" % L.MARK_FRAC,
                 "split": "np.random.default_rng(%d).permutation(n)" % L.SPLIT_SEED},
        "n_pairs": n, "screen_pairs": screen_ids, "report_pairs": report_ids,
        "skipped_videos": skipped, "pairs": pairs,
    }
    pin["pin_sha256"] = hashlib.sha256(
        json.dumps(pin, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    # ---- structural cross-checks against the shadow (crash, never warn) ----------------
    def _vids(pp):
        return [p["video_id"] for p in pp]

    if _vids(pairs) != _vids(shadow["pairs"]):
        raise SystemExit("ddraw: unit video set differs from the min-uid shadow pin")
    if screen_ids != shadow["screen_pairs"] or report_ids != shadow["report_pairs"]:
        raise SystemExit("ddraw: screen/report split differs from the min-uid shadow pin")
    if skipped != shadow.get("skipped_videos"):
        raise SystemExit("ddraw: skipped-video list differs from the min-uid shadow pin")
    struct_mismatch = []
    for p in pairs:
        s = shadow_by_video[p["video_id"]]
        for f in _VIDEO_LEVEL_FIELDS:
            if p[f] != s[f]:
                struct_mismatch.append({"video_id": p["video_id"], "field": f,
                                        "drawn": p[f], "shadow": s[f]})
        if not p["changed_from_minuid"]:
            if int(p["uid"]) != int(s["uid"]):
                struct_mismatch.append({"video_id": p["video_id"], "field": "uid",
                                        "drawn": p["uid"], "shadow": s["uid"]})
            for f in _PAIR_STRUCT_FIELDS + ("question", "options", "gold",
                                            "time_reference"):
                if p[f] != s[f]:
                    struct_mismatch.append({"video_id": p["video_id"], "field": f,
                                            "drawn": p[f], "shadow": s[f]})
    if struct_mismatch:
        raise SystemExit("ddraw: pair arithmetic diverged from lv_common.build_pin on %d "
                         "fields: %s" % (len(struct_mismatch), struct_mismatch[:5]))

    # ---- falsification control: the draw must be distinguishable from the old rule ------
    multi = [p for p in per_video if p["n_candidates"] > 1]
    p_all_index0 = 1.0
    for p in multi:
        p_all_index0 /= float(p["n_candidates"])
    if multi and all(p["k"] == 0 for p in multi):
        raise SystemExit(
            "ddraw: every multi-candidate video drew index 0 -- the draw is "
            "indistinguishable from the archived smallest-uid rule (a-priori probability "
            "%.3g); falsification control fired" % p_all_index0)

    _write_jsonl(CAND_REL, cand_rows)
    cand_sha = _sha_file(CAND_REL)

    # Recheck: read the candidate file back from disk and recompute every k from the rule
    # alone (video id + seed + ordered candidate list).
    back = {}
    for line in open(CAND_REL):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        back.setdefault(o["video_id"], []).append(o)
    recheck_bad = []
    for vid, rows_ in sorted(back.items()):
        rows_.sort(key=lambda r: r["uid"])
        kk = draw_index(vid, len(rows_))
        got = [r["uid"] for r in rows_ if r["selected"]]
        if len(got) != 1 or int(got[0]) != int(rows_[kk]["uid"]):
            recheck_bad.append({"video_id": vid, "k_recomputed": kk, "selected": got,
                                "expect": int(rows_[kk]["uid"])})
    if recheck_bad:
        raise SystemExit("ddraw: draw does not recompute from the written candidate file: %s"
                         % recheck_bad[:5])

    dist = {}
    for p in per_video:
        dist[p["n_candidates"]] = dist.get(p["n_candidates"], 0) + 1
    n_index0 = int(sum(1 for p in per_video if p["k"] == 0))
    n_single = int(sum(1 for p in per_video if p["n_candidates"] == 1))
    record = {
        "rule_version": RULE_VERSION, "rule_text": RULE_TEXT, "seed": DDRAW_SEED,
        "seed_fixed_not_exp_seed": True,
        "seed_formula": "SeedSequence([20261023, int(sha256(video_id)[:8], 16)]) -> "
                        "default_rng -> integers(0, |C(v)|)",
        "candidate_order": "ascending integer uid",
        "tie_rule": "none can occur: integer uids are unique inside a video; a duplicate uid "
                    "crashes the run",
        "empty_set_rule": "|C(v)| = 0 leaves the video a non-unit exactly as in the archived "
                          "construction (pool 20 local videos -> 17 pairs); n stays 17",
        "replaces": "one_question_per_video: smallest uid (lv_common.build_pin; registered in "
                    "twin-lvbench r5@v4/r6 s2; measured by evidence.task-259e5c2b3a; "
                    "that reading stays on record)",
        "n_videos": len(per_video),
        "k_distribution": {str(k): v for k, v in sorted(dist.items())},
        "n_singleton_videos": n_single, "n_index0": n_index0,
        "n_multi_videos": len(multi),
        "n_differ_from_minuid": int(len(per_video) - n_index0),
        "expected_n_differ": float(sum(1.0 - 1.0 / p["n_candidates"] for p in multi)),
        "p_all_multi_index0": p_all_index0,
        "per_video": per_video,
        "candidates_file": {"path": CAND_REL, "sha256": cand_sha,
                            "rows": len(cand_rows)},
        "drawn_pin_sha256": pin["pin_sha256"],
        "minuid_shadow_pin": {"pin_sha256": shadow["pin_sha256"],
                              "archived_expected": ARCHIVED_MINUID_PIN_SHA,
                              "matches_archive": bool(shadow["pin_sha256"]
                                                      == ARCHIVED_MINUID_PIN_SHA),
                              "built_by": "untouched lv_common.build_pin"},
        "structure_cross_check": "passed (video set, split, skipped, per-video fields; any "
                                 "divergence raises before any model step)",
        "selector_source_sha256": _sha_file("paper4/four.py"),
        "ddraw_source_sha256": _sha_file("twin5/ddraw.py"),
        "recheck_from_disk_ok": True,
        "forwards": 0,
        "ordering_note": "the rule and its seed were written in the Plan and committed before "
                         "this file existed; this record and the candidate file are written by "
                         "the run that first executed the committed selector",
    }
    _jdump(record, DRAW_REL)
    log("ddraw: %d videos, k distribution %s, index0 %d, differ-from-minuid %d "
        "(expected %.2f), shadow sha ok, recheck ok"
        % (len(per_video), record["k_distribution"], n_index0,
           record["n_differ_from_minuid"], record["expected_n_differ"]))
    return pin, shadow, record
