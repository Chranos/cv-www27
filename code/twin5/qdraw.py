"""Registered question draw for the single-frame cell (rule qdraw-v1).

Plan: plan-single-frame-random-question (version bound by P4_PLAN_VERSION).

Why this module exists
----------------------
The cell used to take, on each video, the rule-selected question with the smallest integer
uid (twin5/floor_fresh.py:176 sorts by uid, :235 keeps the SELECT-filtered list in that
order, :239 takes sel[0]).  A rule is not a sample, and on 11 of the 24 confirmation videos
the video had more than one rule-selected question, so the rule was doing the choosing.  No
candidate list was ever written down, so nobody could check which questions had been
eligible.

This module replaces the rule with the registered rule in the Plan: a uniform draw over each
video's rule-selected candidate set, with a seed fixed in the Plan before any draw existed,
and it writes the candidate sets out as a declared output so the draw can be recomputed after
the fact.

The rule, verbatim (this text is also embedded in the draw record)
-----------------------------------------------------------------
C(v) = the questions of video v admitted by the frozen inclusion rule and selected by the
frozen SELECT rule, ordered by ascending integer uid.  k(v) ~ Uniform{0, ..., |C(v)|-1}:
    ss = numpy.random.SeedSequence([20261021, int(sha256(v.encode())[:8], 16)])
    rng = numpy.random.default_rng(ss)
    k = int(rng.integers(0, len(C(v))))
chosen = C(v)[k].  No tie exists: integer uids are unique inside a video, and a duplicate is a
structural error that crashes the run rather than being deduplicated silently.  |C(v)| = 0
crashes the run: no video is dropped, no question is substituted, and n stays 24.

Nothing here touches a model, decodes a frame or writes a letter: the whole draw is a
function of the annotation file and one seed, so it costs zero forward passes.
"""
import hashlib
import json
import os

import numpy as np

import twin5.floor_fresh as F
import twin5.lv_common as L

QDRAW_SEED = 20261021
RULE_VERSION = "qdraw-v1"
RULE_TEXT = (
    "C(v) = the questions of video v admitted by the frozen inclusion rule and selected by the "
    "frozen SELECT rule, ordered by ascending integer uid. k(v) ~ Uniform{0,...,|C(v)|-1} with "
    "ss = numpy.random.SeedSequence([20261021, int(hashlib.sha256(v.encode()).hexdigest()[:8], "
    "16)]); rng = numpy.random.default_rng(ss); k = int(rng.integers(0, len(C(v)))). chosen = "
    "C(v)[k]. No tie exists (integer uids are unique inside a video; a duplicate uid is a "
    "structural error and crashes the run, it is not deduplicated). |C(v)| = 0 crashes the run: "
    "no video is dropped, no question is substituted, n stays 24.")
CAND_REL = "results/single_qdraw_candidates.jsonl"
DRAW_REL = "results/single_qdraw.json"
# Counts recorded by the archived run (task-88f6a1123d, n_selected_in_video per unit).
EXPECTED_K_DISTRIBUTION = {1: 13, 2: 7, 3: 2, 4: 2}


def seed_sequence(video_id):
    h = int(hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.SeedSequence([QDRAW_SEED, h])


def draw_index(video_id, n_candidates):
    """The registered draw.  One independent stream per video, so a reader can recompute a
    single video without knowing the rest of the set (and without this file)."""
    rng = np.random.default_rng(seed_sequence(video_id))
    return int(rng.integers(0, int(n_candidates)))


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


def build(side_ids, groups, log=print):
    """Candidate sets and the registered draw for every video of the side.

    Returns (units, shadow_golds, record):
      units        one unit per video, built by floor_fresh.structure_for from the DRAWN
                   question -- identical arithmetic to the archived cell, only a different
                   question can enter it;
      shadow_golds gold letters of the min-uid unit set (the old rule), computed from the same
                   annotations without decoding, for the annotation-read identity control;
      record       the draw record written to DRAW_REL.
    0 forward passes, 0 frames decoded.
    """
    side = set(side_ids)
    cand_rows, per_video, units, shadow_golds = [], [], [], []
    for vid, cand in groups:
        if vid not in side:
            continue
        uids = [int(u) for (u, _a, _b, _q, _so) in cand]
        if len(set(uids)) != len(uids):
            raise SystemExit("qdraw: duplicate integer uid inside video %s: %s"
                             % (vid, sorted(uids)))
        sel = [(uid, a, b, q, so) for (uid, a, b, q, so) in cand if F.rule_select(q)]
        if not sel:
            raise SystemExit("qdraw: video %s has an empty rule-selected candidate set; the "
                             "registered rule crashes here instead of dropping the video" % vid)
        order = {int(uid): i for i, (uid, _a, _b, _q, _so) in enumerate(sel)}
        k = draw_index(vid, len(sel))
        uid, a, b, q, so = sel[k]
        for (u_, _a2, _b2, q2, _so2) in cand:
            cand_rows.append({
                "video_id": vid, "uid": int(u_),
                "question_type": list(q2.get("question_type") or []),
                "time_reference": q2.get("time_reference"), "gold": q2.get("answer"),
                "rule_selected": bool(F.rule_select(q2)),
                "order_index": order.get(int(u_)),
                "selected": bool(int(u_) == int(uid))})
        vp = os.path.join(F.FRESH_DIR, vid + ".mp4")
        fps, n = L.probe_video(vp)
        sha = L.sha256_file(vp)
        nbytes = os.path.getsize(vp)
        # gold letters of the old rule (min uid) on the same video: annotation-only, used by
        # the annotation-read identity control before the model is loaded.
        shadow_golds.append(sel[0][3]["answer"])
        r = F.structure_for(vid, uid, a, b, q, so, fps, n)
        r["video_path"] = vp
        r["video_bytes"] = nbytes
        r["video_sha256"] = sha
        r["n_selected_in_video"] = len(sel)
        r["n_eligible_in_video"] = len(cand)
        r["draw_index"] = int(k)
        r["qid_uid_min"] = "%s#%d" % (vid, int(sel[0][0]))
        units.append(r)
        per_video.append({"video_id": vid, "n_candidates": len(sel), "k": int(k),
                          "uid": int(uid), "qid": "%s#%d" % (vid, int(uid)),
                          "uid_min": int(sel[0][0]), "changed_from_minuid": bool(int(k) != 0)})
    units.sort(key=lambda r: r["video_id"])
    per_video.sort(key=lambda r: r["video_id"])
    cand_rows.sort(key=lambda r: (r["video_id"], r["uid"]))

    _write_jsonl(CAND_REL, cand_rows)
    cand_sha = _sha_file(CAND_REL)

    # Recheck: read the candidate file back from disk and recompute every k from the rule
    # alone (video id + seed + ordered candidate list).  The recorded draw must be
    # reproducible from the written artefacts, not only from this process's memory.
    back = {}
    for line in open(CAND_REL):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o["rule_selected"]:
            back.setdefault(o["video_id"], []).append(o)
    recheck_bad = []
    for vid, rows in sorted(back.items()):
        rows.sort(key=lambda r: r["uid"])
        kk = draw_index(vid, len(rows))
        got = [r["uid"] for r in rows if r["selected"]]
        if len(got) != 1 or int(got[0]) != int(rows[kk]["uid"]):
            recheck_bad.append({"video_id": vid, "k_recomputed": kk, "selected": got,
                                "expect": int(rows[kk]["uid"])})

    dist = {}
    for p in per_video:
        dist[p["n_candidates"]] = dist.get(p["n_candidates"], 0) + 1
    n_index0 = int(sum(1 for p in per_video if p["k"] == 0))
    n_single = int(sum(1 for p in per_video if p["n_candidates"] == 1))
    record = {
        "rule_version": RULE_VERSION, "rule_text": RULE_TEXT, "seed": QDRAW_SEED,
        "seed_fixed_not_exp_seed": True,
        "seed_formula": "SeedSequence([20261021, int(sha256(video_id)[:8], 16)]) -> "
                        "default_rng -> integers(0, |C(v)|)",
        "candidate_order": "ascending integer uid",
        "tie_rule": "none can occur: integer uids are unique inside a video; a duplicate uid "
                    "crashes the run",
        "empty_set_rule": "|C(v)| = 0 crashes the run; no video dropped, no substitution, n=24",
        "n_videos": len(per_video),
        "k_distribution": {str(k): v for k, v in sorted(dist.items())},
        "k_distribution_expected": {str(k): v for k, v in sorted(EXPECTED_K_DISTRIBUTION.items())},
        "n_singleton_videos": n_single, "n_index0": n_index0,
        "n_differ_from_minuid": int(len(per_video) - n_index0),
        "per_video": per_video,
        "candidates_file": {"path": CAND_REL, "sha256": cand_sha,
                            "rows": len(cand_rows)},
        "selector_source_sha256": _sha_file("paper4/four.py"),
        "qdraw_source_sha256": _sha_file("twin5/qdraw.py"),
        "recheck_from_disk_ok": bool(not recheck_bad),
        "recheck_from_disk_mismatches": recheck_bad[:20],
        "forwards": 0,
        "ordering_note": "the rule and its seed were written in the Plan and committed before "
                         "this file existed; this record and the candidate file are written by "
                         "the run that first executed the committed selector",
    }
    _jdump(record, DRAW_REL)
    # The draw file's own hash is not embedded in itself (that would be stale by construction);
    # the run's Evidence records the hash of the declared output.
    log("qdraw: %d videos, k distribution %s, index0 %d, differ-from-minuid %d, recheck_ok %s"
        % (len(per_video), record["k_distribution"], n_index0,
           record["n_differ_from_minuid"], record["recheck_from_disk_ok"]))
    return units, shadow_golds, record
