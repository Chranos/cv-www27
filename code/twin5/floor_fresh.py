"""F2_fresh_pool_floor: the floor measurement on a pool of LVBench videos that no run
in this project has ever touched.

Plan: plan-floorpool-fresh (version 1).

Why this run exists
-------------------
C1 (plan-floorpool-selection) measured a pre-registered selection rule on the
carrier's 20-video copy, but its "confirmation" questions had already been measured in
F1, so it could not tell "this rule fails" from "this carrier cannot host a floor".
The independent check (review-floorpool-selection-check-r2) listed exactly what
is missing: (1) a genuinely untouched video pool of >= 17 independent video units,
(2) rule and split fixed before any run, (3) no_frames and uniform@16 at the video-level
unit with intervals, (4) the single-frame capability control, (5) at least one mechanism
reading.  This module produces exactly those.

Modes (one command each)
------------------------
build    read-only (executor=ssh, no GPU, 0 forwards): recompute the dataset manifest and
         annotation hash, rebuild the pool from the frozen inclusion rule and the frozen
         SELECT rule, apply the frozen split, and write the per-unit manifest.  It must
         reproduce the counts pre-registered in the Plan; otherwise it exits non-zero and
         no model step runs.
pilot    GPU: the 8 design-side units, five arms.  It exists only to prove the harness
         runs end to end before the confirmation side is touched.  Its readings are
         reported but never used for the verdict.
measure  GPU: the 24 confirmation-side units, five arms, plus the identity positive
         control (the 8 r6 screen questions, two arms, against the archived letters).

Arms (five, all on the same question set, same prompt, same greedy decoding, same pixel
cap, no per-arm tuning):
  no_frames            zero images
  uniform@16           the 16 grid frames at lv_common.I16U over a 32-slot grid
  single_frame_j_star  the single frame at the referenced interval's midpoint (j*)
  uniform16_swap       uniform@16 with the I16U slot nearest j* replaced by the frame at
                       (j* + floor(n/2)) mod n
  blank16              sixteen constant black frames of the same shape as the grid frames

Nothing here is a claim reading: no operator, no marked/unmarked role, no (e) cell, no
headroom_screen, no full run.  stage=smoke, decides_claim=false.
"""
import glob
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np

import gen_common as G
import mvp_common as C
import twin5.lv_common as L

RUN_BUILD = "R0_fresh_build"
RUN_PILOT = "R1_fresh_pilot"
RUN_MEASURE = "R2_fresh_confirm"

STAGE = "smoke"
PLAN_ID = os.environ.get("FRESH_PLAN_ID", "plan-floorpool-fresh")
PLAN_VERSION = int(os.environ.get("FRESH_PLAN_VERSION", "1"))
EXP_SEED = int(os.environ.get("EXP_SEED", "0"))

DATA = os.environ.get("DATA_ROOT", "/path/to/data")
FRESH_DIR = os.path.join(DATA, "lvbench-fresh-24209d43cbdd")
OLD_DIR = os.path.join(DATA, "lvbench-24209d43cbdd")
FRESH_META = os.path.join(FRESH_DIR, "video_info.meta.jsonl")
OLD_META = os.path.join(OLD_DIR, "video_info.meta.jsonl")
MODEL_DIR = L.MODEL_DIR

# Frozen dataset identity of the new pool (written into the Plan before any run).
FRESH_MANIFEST_SHA = "d8c989db399b32a842d069f75ecd281f23a5a630d351b009bb4120b6877fa448"
FRESH_N_FILES = 38
ANNOT_SHA = "a3042f766f568d457e2a91a30cdcf32a886c5421226eefa98a006ffaae038d8f"

# Frozen split constant (NOT EXP_SEED: the split must not move between reruns).
SPLIT_SEED = 20261020
DESIGN_N = 8

# Frozen inclusion rule (r6 / F1 / C1, unchanged).
MAX_INTERVAL_S = L.MAX_INTERVAL_S          # 3.0 s
MAX_VIDEO_BYTES = 300 * 1000 * 1000        # same download-feasibility cap as the old carrier
RULE_POS = {"event understanding", "temporal grounding"}
RULE_NEG = {"key information retrieval", "summarization"}

# Frozen pre-registered prediction (Plan, written before any arm run on this pool).
PRE_REG = {
    "pool_videos": 32,
    "pool_eligible_questions": 134,
    "pool_selected_questions": 55,
    "design_units": 8,
    "confirmation_units": 24,
    "bar_hits_confirmation": 8,
    "positive_control_n": 8,
    "positive_control_expected_agree_no_frames": 8,
    "positive_control_expected_agree_uniform": 8,
}

BAR_FRAC = 0.35                            # unchanged pre-registered line
CAPABILITY_MIN_RATE = 0.50                 # Plan: j* arm, control criterion (not a claim threshold)
CAPABILITY_MIN_CP_LOWER = 0.25             # Plan: j* CP95 lower bound must clear chance

ARMS = ("no_frames", "uniform@16", "single_frame_j_star", "uniform16_swap",
        "uniform16_evidence", "blank16")

ARCHIVE_TASK = "task-68cebbe16b"
ARCHIVE_REL = "results/twin5_P1_carrier_samples.jsonl"

MANIFEST_REL = "results/fresh_manifest.json"
QUESTIONS_REL = "results/fresh_questions.jsonl"
PILOT_REL = "results/fresh_pilot_readings.jsonl"
READINGS_REL = "results/fresh_readings.jsonl"

T0 = time.time()
PHASE = {"identity_s": 0.0, "model_load_s": 0.0, "decode_s": 0.0, "forward_s": 0.0}
FWD = G.Fwd()
NOTES = []
N_ROWS = [0]


def log(*a):
    print(*a, flush=True)


def app_jsonl(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    N_ROWS[0] += 1


def jdump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)


# ---------------------------------------------------------------- frozen rules

def eligible_by_video(meta_path, data_dir):
    """Every question admitted by the frozen r6 inclusion rule, grouped by video."""
    rows = [json.loads(l) for l in open(meta_path) if l.strip()]
    by_key = {r["key"]: r for r in rows}
    vids = sorted(f[:-4] for f in os.listdir(data_dir) if f.endswith(".mp4"))
    out, skipped = [], []
    for vid in vids:
        r = by_key.get(vid)
        if r is None:
            skipped.append({"video": vid, "why": "no annotation row"})
            continue
        try:
            dur = float(r["video_info"]["duration_minutes"]) * 60.0
        except Exception as e:                                        # noqa: BLE001
            skipped.append({"video": vid, "why": "no parsable duration: %s" % str(e)[:120]})
            continue
        cand = []
        for q in r["qa"]:
            p = L.parse_time_reference(q.get("time_reference", ""))
            if p is None:
                continue
            a, b = p
            if b < a or (b - a) > MAX_INTERVAL_S or b > dur + 1.0:
                continue
            o = L.parse_options(q.get("question", ""))
            if o is None:
                continue
            if q.get("answer") not in list("ABCD"):
                continue
            cand.append((int(q["uid"]), a, b, q, o))
        if not cand:
            skipped.append({"video": vid, "why": "no eligible question"})
            continue
        cand.sort(key=lambda x: x[0])
        out.append((vid, cand))
    return out, skipped


def rule_select(q):
    """Frozen SELECT rule. Reads only question_type; exact case-sensitive membership."""
    tags = set(q.get("question_type") or [])
    return bool(tags & RULE_POS) and not (tags & RULE_NEG)


def split_videos(groups):
    order = [v for v, _c in groups]
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(order)).tolist()
    shuffled = [order[int(i)] for i in perm]
    design = sorted(shuffled[:DESIGN_N])
    confirmation = sorted(shuffled[DESIGN_N:])
    return design, confirmation, shuffled


def structure_for(vid, uid, a, b, q, stem_opts, fps, n):
    stem, opts = stem_opts
    t_ev = 0.5 * (a + b)
    j_star = int(min(max(round(t_ev * fps), 0), n - 1))
    idx = [int(min(max(round((i + 0.5) * n / L.K), 0), n - 1)) for i in range(L.K)]
    n_slots_equal_j = int(sum(1 for x in idx if x == j_star))
    idx_guarded = list(idx)
    for s in range(L.K):
        if idx_guarded[s] == j_star:
            idx_guarded[s] = int(min(max(idx[s] + (1 if idx[s] + 1 < n else -1), 0), n - 1))
    i_star = L.nearest_slot_to_frame(j_star, idx_guarded)
    e_star = L.evidence_slot(i_star)
    # mechanism arm: the I16U slot whose source frame is nearest j*, and its donor frame
    s_local = int(min(range(len(L.I16U)), key=lambda s: (abs(int(idx[L.I16U[s]]) - j_star), s)))
    j_donor = int((j_star + n // 2) % n)
    return {
        "video_id": vid, "uid": int(uid), "qid": "%s#%d" % (vid, int(uid)),
        "time_reference": q.get("time_reference"), "t_ev_s": t_ev, "interval_s": b - a,
        "fps": float(fps), "n_frames": int(n),
        "j_star": int(j_star), "i_star": int(i_star), "e_star": int(e_star),
        "n_grid_slots_equal_j_star_before_guard": n_slots_equal_j,
        "e_star_in_i16u": bool(int(e_star) in L.I16U_SET),
        "swap_slot_local": int(s_local), "swap_slot_grid_index": int(L.I16U[s_local]),
        "swap_slot_frame_index": int(idx[L.I16U[s_local]]),
        "swap_slot_distance_frames": int(abs(int(idx[L.I16U[s_local]]) - j_star)),
        "swap_donor_frame_index": int(j_donor),
        "question": stem, "options": list(opts), "gold": q["answer"],
        "question_type": list(q.get("question_type") or []),
        "grid_idx": idx, "grid_idx_guarded": idx_guarded,
    }


def build_units(groups, side_ids):
    """One unit per video: smallest integer uid among the rule-selected questions."""
    side = set(side_ids)
    rows, skipped = [], []
    for vid, cand in groups:
        if vid not in side:
            continue
        sel = [(uid, a, b, q, so) for (uid, a, b, q, so) in cand if rule_select(q)]
        if not sel:
            skipped.append({"video": vid, "why": "no rule-selected question"})
            continue
        uid, a, b, q, so = sel[0]
        vp = os.path.join(FRESH_DIR, vid + ".mp4")
        fps, n = L.probe_video(vp)
        r = structure_for(vid, uid, a, b, q, so, fps, n)
        r["video_path"] = vp
        r["video_bytes"] = os.path.getsize(vp)
        r["video_sha256"] = L.sha256_file(vp)
        r["n_selected_in_video"] = len(sel)
        r["n_eligible_in_video"] = len(cand)
        rows.append(r)
    rows.sort(key=lambda r: r["video_id"])
    return rows, skipped


# ---------------------------------------------------------------- arms

def to_pil1(frame):
    return C.to_pil(frame[None, :, :, :], G.MAX_PIXELS)[0]


def ask(model, proc, pils, prompt, dev, arm, kind=None):
    t = time.time()
    try:
        raw, letter = G.answer_arm(model, proc, pils, prompt, dev, counter=FWD,
                                   kind=kind or ("answer:" + arm))
    except Exception as e:                                            # noqa: BLE001
        NOTES.append("arm %s failed: %s" % (arm, str(e)[:200]))
        raw, letter = "ERR:" + str(e)[:200], None
    PHASE["forward_s"] += time.time() - t
    return letter, (raw or "")


def measure_units(units, model, proc, dev, out_rel, tag):
    readings = []
    for v_idx, u in enumerate(units):
        vp = u["video_path"]
        t = time.time()
        grid = L.decode_indices(vp, u["grid_idx"])                 # 32 slots
        ev = L.decode_indices(vp, [u["j_star"]])[0]                # j*
        donor = L.decode_indices(vp, [u["swap_donor_frame_index"]])[0]
        grid_pils = C.to_pil(grid, G.MAX_PIXELS)
        ev_pil = to_pil1(ev)
        donor_pil = to_pil1(donor)
        u_pils = [grid_pils[i] for i in L.I16U]
        swap_pils = list(u_pils)
        swap_pils[int(u["swap_slot_local"])] = donor_pil
        evid_pils = list(u_pils)
        evid_pils[int(u["swap_slot_local"])] = ev_pil
        blank_all = C.to_pil(np.zeros_like(grid), G.MAX_PIXELS)
        blank_pils = [blank_all[i] for i in L.I16U]
        PHASE["decode_s"] += time.time() - t

        q = {"question": u["question"], "candidates": list(u["options"]),
             "correct_letter": u["gold"]}
        prompt = G.prompt_for(q)
        letters, raws = {}, {}
        for arm, pils in (("no_frames", []), ("uniform@16", u_pils),
                          ("single_frame_j_star", [ev_pil]),
                          ("uniform16_swap", swap_pils),
                          ("uniform16_evidence", evid_pils), ("blank16", blank_pils)):
            letter, raw = ask(model, proc, pils, prompt, dev, arm)
            letters[arm] = letter
            raws[arm] = raw[:60]
        correct = {a: int(letters[a] == u["gold"]) for a in ARMS}
        row = {
            "qid": u["qid"], "video_id": u["video_id"], "uid": u["uid"],
            "set": tag, "question_type": u["question_type"], "gold": u["gold"],
            "time_reference": u["time_reference"], "j_star": u["j_star"],
            "fps": u["fps"], "n_frames": u["n_frames"],
            "swap_slot_local": u["swap_slot_local"],
            "swap_slot_frame_index": u["swap_slot_frame_index"],
            "swap_slot_distance_frames": u["swap_slot_distance_frames"],
            "swap_slot_distance_seconds": (u["swap_slot_distance_frames"] / float(u["fps"])
                                           if u["fps"] else None),
            "swap_donor_frame_index": u["swap_donor_frame_index"],
            "letters": letters, "raw": raws, "correct": correct,
        }
        readings.append(row)
        app_jsonl(out_rel, row)
        log("%s %2d/%d %-12s %s" % (tag, v_idx + 1, len(units), u["video_id"], u["qid"]))
    return readings


# ---------------------------------------------------------------- statistics

def bar_hits(n):
    return int(math.floor(BAR_FRAC * n))


def arm_entry(rows, arm):
    hits = int(sum(int(r["correct"][arm]) for r in rows))
    n = len(rows)
    return {"hits": hits, "n": n, "value": (float(hits) / n if n else None),
            "cp95": L.cp_ci(hits, n)}


def cells(rows, a1, a2):
    out = {"neither": 0, "a1_only": 0, "a2_only": 0, "both": 0}
    for r in rows:
        x, y = int(r["correct"][a1]), int(r["correct"][a2])
        if x == 0 and y == 0:
            out["neither"] += 1
        elif x == 1 and y == 0:
            out["a1_only"] += 1
        elif x == 0 and y == 1:
            out["a2_only"] += 1
        else:
            out["both"] += 1
    return out


def mcnemar(rows, a1, a2):
    n01 = sum(1 for r in rows if int(r["correct"][a1]) == 0 and int(r["correct"][a2]) == 1)
    n10 = sum(1 for r in rows if int(r["correct"][a1]) == 1 and int(r["correct"][a2]) == 0)
    n = n01 + n10
    if n == 0:
        p = 1.0
    else:
        from scipy.stats import binom
        k = min(n01, n10)
        p = float(min(1.0, 2.0 * binom.cdf(k, n, 0.5)))
    return {"n01": int(n01), "n10": int(n10), "n": int(n), "p": p}


def holm(pvals):
    """Holm-corrected p for a small pre-declared family (descriptive)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


def tables(rows, with_floor):
    n = len(rows)
    out = {"n": n, "arms": {a: arm_entry(rows, a) for a in ARMS},
           "cells_vs_no_frames": {a: cells(rows, "no_frames", a) for a in ARMS if a != "no_frames"},
           "mcnemar_vs_no_frames": {a: mcnemar(rows, "no_frames", a)
                                    for a in ARMS if a != "no_frames"},
           "mcnemar_pairwise": {
               "uniform@16_vs_blank16": mcnemar(rows, "uniform@16", "blank16"),
               "uniform@16_vs_uniform16_swap": mcnemar(rows, "uniform@16", "uniform16_swap"),
               "uniform@16_vs_uniform16_evidence": mcnemar(rows, "uniform@16",
                                                           "uniform16_evidence")}}
    out["holm_descriptive"] = holm({
        "uniform@16_vs_no_frames": out["mcnemar_vs_no_frames"]["uniform@16"]["p"],
        "single_frame_j_star_vs_no_frames": out["mcnemar_vs_no_frames"]["single_frame_j_star"]["p"],
        "uniform@16_vs_blank16": out["mcnemar_pairwise"]["uniform@16_vs_blank16"]["p"],
        "uniform@16_vs_uniform16_swap": out["mcnemar_pairwise"]["uniform@16_vs_uniform16_swap"]["p"],
        "uniform@16_vs_uniform16_evidence":
            out["mcnemar_pairwise"]["uniform@16_vs_uniform16_evidence"]["p"]})
    golds = [r["gold"] for r in rows]
    cnt = Counter(golds)
    best_letter, best_hits = max(sorted(cnt.items()), key=lambda kv: kv[1]) if cnt else (None, 0)
    dsec = sorted(float(r["swap_slot_distance_seconds"]) for r in rows
                  if r.get("swap_slot_distance_seconds") is not None)
    if dsec:
        out["swap_slot_geometry"] = {
            "n": len(dsec), "min_s": dsec[0], "median_s": dsec[len(dsec) // 2],
            "max_s": dsec[-1],
            "note": "seconds between the evidence midpoint j* and the nearest uniform@16 "
                    "frame that uniform16_swap replaces; a large gap means a uniform frame "
                    "could not have been carrying the referenced moment"}
    out["constant_letter_baseline"] = {
        "gold_letter_counts": dict(sorted(cnt.items())), "best_letter": best_letter,
        "best_letter_hits": int(best_hits), "n": n,
        "value": (float(best_hits) / n if n else None),
        "note": "oracle prior baseline: always answer the modal gold letter of these units; "
                "computed from annotations only, no model call"}
    if with_floor:
        bar = bar_hits(n)
        nf = out["arms"]["no_frames"]
        u = out["arms"]["uniform@16"]
        out["bar_hits"] = bar
        out["floor_no_frames"] = bool(nf["hits"] <= bar)
        out["floor_uniform"] = bool(u["hits"] <= bar)
        out["floor_holds"] = bool(out["floor_no_frames"] and out["floor_uniform"])
        out["certified_no_frames"] = bool(nf["cp95"][1] is not None and nf["cp95"][1] < BAR_FRAC)
        out["certified_uniform"] = bool(u["cp95"][1] is not None and u["cp95"][1] < BAR_FRAC)
        out["floor_certified"] = bool(out["floor_holds"] and out["certified_no_frames"]
                                      and out["certified_uniform"])
        j = out["arms"]["single_frame_j_star"]
        out["capability_control"] = {
            "criterion": "hits/n >= %.2f AND cp95 lower bound > %.2f"
                         % (CAPABILITY_MIN_RATE, CAPABILITY_MIN_CP_LOWER),
            "value": j["value"], "cp95": j["cp95"],
            "established": bool(j["value"] is not None
                               and j["value"] >= CAPABILITY_MIN_RATE
                               and j["cp95"][0] is not None
                               and j["cp95"][0] > CAPABILITY_MIN_CP_LOWER)}
        hu = u["hits"]
        hb = out["arms"]["blank16"]["hits"]
        hs = out["arms"]["uniform16_swap"]["hits"]
        he = out["arms"]["uniform16_evidence"]["hits"]
        if hb >= hu - 1:
            cls = "M1_prior_or_option_structure"
        elif he >= hu + 3:
            cls = "M4_evidence_frame_carries_answer"
        elif hs <= hu - 3:
            cls = "M3_evidence_adjacent_frame"
        elif hb <= hu - 3 and abs(hs - hu) <= 1 and abs(he - hu) <= 1:
            cls = "M2_global_content"
        else:
            cls = "mixed_or_unresolved"
        out["mechanism_reading"] = {
            "rule": "in this order: M1 if blank16 >= uniform-1; else M4 if "
                    "uniform16_evidence >= uniform+3; else M3 if uniform16_swap <= uniform-3; "
                    "else M2 if blank16 <= uniform-3 and |swap-uniform| <= 1 and "
                    "|evidence-uniform| <= 1; else mixed",
            "uniform_hits": hu, "blank16_hits": hb, "swap16_hits": hs,
            "evidence16_hits": he,
            "j_star_hits": out["arms"]["single_frame_j_star"]["hits"],
            "classification": cls,
            "note": "descriptive reading rule fixed in the Plan before the run; it does not "
                    "change the floor verdict"}
    return out


# ---------------------------------------------------------------- build

def run_build():
    t = time.time()
    got_manifest, n_files = L.manifest_sha256(root=FRESH_DIR)
    annot = L.sha256_file(FRESH_META)
    ident = {"dataset": "lvbench-fresh-24209d43cbdd", "path": FRESH_DIR,
             "source": "AI-ModelScope/LVBench (ModelScope mirror, plain HTTP)",
             "annotation_revision": "24209d43cbddb260fcb43dbdee2de0899c87163f",
             "video_file_revision": "935fac4ccca85178eb7c6b10614c1cf21cf0aed6",
             "license": "README cc-by-nc-sa-4.0; LICENSE file is Apache-2.0 text with an "
                        "unresolved merge marker; non-commercial academic research only",
             "manifest_sha256_recomputed": got_manifest,
             "manifest_sha256_expected": FRESH_MANIFEST_SHA,
             "n_files": n_files, "n_files_expected": FRESH_N_FILES,
             "manifest_matches": bool(got_manifest == FRESH_MANIFEST_SHA and n_files == FRESH_N_FILES),
             "annotation_sha256": annot, "annotation_sha256_expected": ANNOT_SHA,
             "annotation_matches": bool(annot == ANNOT_SHA),
             "verified_by": "the run itself, not the sandbox (no data_dirs configured on this "
                            "device; the platform dataset schema has no ModelScope provider)"}
    PHASE["identity_s"] = time.time() - t
    groups, skipped = eligible_by_video(FRESH_META, FRESH_DIR)
    design, confirmation, shuffled = split_videos(groups)
    design_units, des_skip = build_units(groups, design)
    conf_units, conf_skip = build_units(groups, confirmation)

    per_video = []
    for vid, cand in groups:
        sel = [c for c in cand if rule_select(c[3])]
        per_video.append({
            "video_id": vid, "side": "design" if vid in set(design) else "confirmation",
            "n_eligible": len(cand), "n_selected": len(sel),
            "selected_uids": [c[0] for c in sel],
            "video_bytes": os.path.getsize(os.path.join(FRESH_DIR, vid + ".mp4")),
            "in_old_copy": bool(os.path.exists(os.path.join(OLD_DIR, vid + ".mp4")))})
    counts = {
        "pool_videos": len(groups),
        "pool_eligible_questions": int(sum(len(c) for _v, c in groups)),
        "pool_selected_questions": int(sum(1 for _v, c in groups for q in c if rule_select(q[3]))),
        "design_units": len(design_units),
        "confirmation_units": len(conf_units),
        "bar_hits_confirmation": bar_hits(len(conf_units)),
    }
    checks = {
        "no_video_in_old_copy": bool(all(not p["in_old_copy"] for p in per_video)),
        "all_sizes_under_cap": bool(all(p["video_bytes"] <= MAX_VIDEO_BYTES for p in per_video)),
        "design_confirmation_disjoint": bool(set(design).isdisjoint(set(confirmation))),
        "e_star_outside_i16u_all": bool(all(not u["e_star_in_i16u"]
                                            for u in design_units + conf_units)),
        "swap_slot_is_i16u_member": bool(all(u["swap_slot_grid_index"] in L.I16U_SET
                                             for u in design_units + conf_units)),
        "swap_donor_differs_from_j_star": bool(all(u["swap_donor_frame_index"] != u["j_star"]
                                                   for u in design_units + conf_units)),
        "one_unit_per_video": bool(len({u["video_id"] for u in design_units + conf_units})
                                   == len(design_units) + len(conf_units)),
        "gold_all_in_ABCD": bool(all(u["gold"] in list("ABCD")
                                     for u in design_units + conf_units)),
    }
    pred_match = {k: {"actual": counts[k], "predicted": PRE_REG[k], "match": counts[k] == PRE_REG[k]}
                  for k in PRE_REG if k in counts}

    manifest = {
        "dataset": ident, "rule": {
            "inclusion": "time_reference parses MM:SS-MM:SS; 0 <= end-start <= 3 s; "
                         "end <= duration + 1.0 s; exactly A-D non-empty options; answer in ABCD",
            "select": "SELECT(q) iff T(q) intersects {event understanding, temporal grounding} "
                      "and T(q) does not intersect {key information retrieval, summarization}; "
                      "T(q)=set(question_type), exact case-sensitive strings",
            "video_unit": "smallest integer uid among the video's rule-selected questions",
            "max_interval_s": MAX_INTERVAL_S, "max_video_bytes": MAX_VIDEO_BYTES},
        "split": {"seed": SPLIT_SEED, "design_n": DESIGN_N, "shuffled_order": shuffled,
                  "design_videos": design, "confirmation_videos": confirmation},
        "counts": counts, "per_video": per_video, "skipped_videos": skipped,
        "skipped_units": {"design": des_skip, "confirmation": conf_skip},
        "units": {"design": [u["qid"] for u in design_units],
                  "confirmation": [u["qid"] for u in conf_units]},
        "questions": design_units + conf_units,
    }
    jdump(manifest, MANIFEST_REL)
    for u in design_units + conf_units:
        app_jsonl(QUESTIONS_REL, u)

    problems = []
    if not (ident["manifest_matches"] and ident["annotation_matches"]):
        problems.append("data_identity_mismatch")
    if not all(checks.values()):
        problems.append("structure_check_failed")
    if not all(v["match"] for v in pred_match.values()):
        problems.append("prediction_mismatch")
    out = {"run": {"run": RUN_BUILD, "tag": "fresh_build", "stage": STAGE, "plan_id": PLAN_ID,
                   "plan_version": PLAN_VERSION, "exp_seed": EXP_SEED,
                   "decides_claim": False},
           "data_identity": ident, "counts": counts, "prediction_matches": pred_match,
           "structure_checks": checks, "split": manifest["split"],
           "units": manifest["units"], "per_video": per_video,
           "forward_counts": FWD.report(), "phases_seconds": PHASE,
           "elapsed_seconds": round(time.time() - T0, 1), "notes": NOTES,
           "metrics_problem": (";".join(problems) if problems else None)}
    jdump(out, "metrics.json")
    log("build: counts=%s checks=%s problems=%s" % (json.dumps(counts), all(checks.values()),
                                                    problems))
    if problems:
        raise SystemExit(3)


# ---------------------------------------------------------------- measure

def load_manifest():
    cands = sorted(glob.glob(os.path.join("sources", "*", MANIFEST_REL)))
    if len(cands) != 1:
        raise SystemExit("expected exactly one archived fresh manifest under sources/, found %r"
                         % cands)
    with open(cands[0]) as f:
        return json.load(f), cands[0]


def positive_control(model, proc, dev):
    """Re-measure the 8 r6 screen questions with the two original arms and compare letters.

    The screen set is rebuilt from the OLD carrier with the frozen r6 rule (sorted videos
    that have an eligible question, default_rng(11), first 8, smallest uid per video); the
    archive supplies the reference letters.
    """
    path = os.path.join("sources", ARCHIVE_TASK, ARCHIVE_REL)
    out = {"available": bool(os.path.exists(path)), "path": path, "archive_task": ARCHIVE_TASK}
    if not out["available"]:
        NOTES.append("archived P1 samples not found at %s; positive control unavailable" % path)
        out["pass"] = False
        return out
    arch = {}
    for line in open(path):
        o = json.loads(line)
        arch[o["video_id"]] = o
    groups, _skip = eligible_by_video(OLD_META, OLD_DIR)
    gold_by_video = {v: cand[0][3]["answer"] for v, cand in groups}
    order = [v for v, _c in groups]
    perm = np.random.default_rng(L.SPLIT_SEED).permutation(len(order)).tolist()
    screens = sorted(order[int(i)] for i in perm[:L.SCREEN_N])
    by = dict(groups)
    rows, mism = [], []
    for vid in screens:
        cand = by[vid]
        uid, a, b, q, so = cand[0]
        vp = os.path.join(OLD_DIR, vid + ".mp4")
        fps, n = L.probe_video(vp)
        st = structure_for(vid, uid, a, b, q, so, fps, n)
        idx = st["grid_idx"]
        grid = L.decode_indices(vp, idx)
        grid_pils = C.to_pil(grid, G.MAX_PIXELS)
        u_pils = [grid_pils[i] for i in L.I16U]
        qq = {"question": st["question"], "candidates": list(st["options"]),
              "correct_letter": st["gold"]}
        prompt = G.prompt_for(qq)
        lnf, rnf = ask(model, proc, [], prompt, dev, "no_frames", kind="control:no_frames")
        lu, ru = ask(model, proc, u_pils, prompt, dev, "uniform@16", kind="control:uniform@16")
        o = arch.get(vid, {})
        row = {"qid": st["qid"], "video_id": vid,
               "this": {"no_frames": lnf, "uniform@16": lu, "j_star": st["j_star"],
                        "e_star": st["e_star"]},
               "archive": {"no_frames": (o.get("letters") or {}).get("no_frames"),
                           "uniform@16": (o.get("letters") or {}).get("uniform@16"),
                           "j_star": o.get("j_star"), "e_star": o.get("e_star")}}
        row["agree_no_frames"] = bool(row["this"]["no_frames"] == row["archive"]["no_frames"])
        row["agree_uniform@16"] = bool(row["this"]["uniform@16"] == row["archive"]["uniform@16"])
        row["agree_structure"] = bool(row["this"]["j_star"] == row["archive"]["j_star"]
                                      and row["this"]["e_star"] == row["archive"]["e_star"])
        rows.append(row)
        if not (row["agree_no_frames"] and row["agree_uniform@16"] and row["agree_structure"]):
            mism.append(row)
    out.update({
        "n": len(rows), "n_compared": len(rows),
        "n_agree_no_frames": int(sum(1 for r in rows if r["agree_no_frames"])),
        "n_agree_uniform": int(sum(1 for r in rows if r["agree_uniform@16"])),
        "n_agree_structure": int(sum(1 for r in rows if r["agree_structure"])),
        "n_mismatch": len(mism), "mismatches": mism[:20], "rows": rows,
        "archive_hits": {
            "no_frames": int(sum(1 for r in rows
                                 if r["archive"]["no_frames"] == gold_by_video.get(r["video_id"]))),
            "uniform@16": int(sum(1 for r in rows
                                  if r["archive"]["uniform@16"] == gold_by_video.get(r["video_id"])))},
        "this_hits": {
            "no_frames": int(sum(1 for r in rows
                                 if r["this"]["no_frames"] == gold_by_video.get(r["video_id"]))),
            "uniform@16": int(sum(1 for r in rows
                                  if r["this"]["uniform@16"] == gold_by_video.get(r["video_id"])))},
    })
    out["pass"] = bool(out["n_agree_no_frames"] == out["n"]
                       and out["n_agree_uniform"] == out["n"])
    return out


def run_measure(mode):
    tag = "pilot" if mode == "pilot" else "confirmation"
    out_rel = PILOT_REL if mode == "pilot" else READINGS_REL
    t = time.time()
    got_manifest, n_files = L.manifest_sha256(root=FRESH_DIR)
    annot = L.sha256_file(FRESH_META)
    ident = {"dataset": "lvbench-fresh-24209d43cbdd", "path": FRESH_DIR,
             "manifest_sha256_recomputed": got_manifest,
             "manifest_sha256_expected": FRESH_MANIFEST_SHA,
             "n_files": n_files, "n_files_expected": FRESH_N_FILES,
             "manifest_matches": bool(got_manifest == FRESH_MANIFEST_SHA and n_files == FRESH_N_FILES),
             "annotation_sha256": annot, "annotation_sha256_expected": ANNOT_SHA,
             "annotation_matches": bool(annot == ANNOT_SHA),
             "verified_by": "the run itself, not the sandbox (no data_dirs configured on this "
                            "device; the platform dataset schema has no ModelScope provider)"}
    PHASE["identity_s"] = time.time() - t
    if not (ident["manifest_matches"] and ident["annotation_matches"]):
        jdump({"run": {"run": RUN_PILOT if mode == "pilot" else RUN_MEASURE, "stage": STAGE,
                       "plan_id": PLAN_ID, "plan_version": PLAN_VERSION,
                       "exp_seed": EXP_SEED, "decides_claim": False},
               "data_identity": ident, "metrics_problem": "data_identity_mismatch",
               "notes": NOTES}, "metrics.json")
        raise SystemExit(2)

    manifest, manifest_path = load_manifest()
    side_key = "design" if mode == "pilot" else "confirmation"
    side_units = [u for u in manifest["questions"]
                  if u["qid"] in set(manifest["units"][side_key])]
    side_units.sort(key=lambda u: u["video_id"])
    expected_n = PRE_REG["design_units"] if mode == "pilot" else PRE_REG["confirmation_units"]
    counts_ok = bool(all(manifest["counts"][k] == PRE_REG[k] for k in PRE_REG
                         if k in manifest["counts"]))
    structure_ok = bool(all(not u["e_star_in_i16u"] for u in side_units)
                        and all(u["swap_slot_grid_index"] in L.I16U_SET for u in side_units)
                        and all(u["swap_donor_frame_index"] != u["j_star"] for u in side_units)
                        and len(side_units) == expected_n
                        and len({u["video_id"] for u in side_units}) == expected_n
                        and counts_ok)
    if not structure_ok:
        jdump({"run": {"run": RUN_PILOT if mode == "pilot" else RUN_MEASURE, "stage": STAGE,
                       "plan_id": PLAN_ID, "plan_version": PLAN_VERSION,
                       "exp_seed": EXP_SEED, "decides_claim": False},
               "data_identity": ident, "manifest_path": manifest_path,
               "metrics_problem": "structure_mismatch", "notes": NOTES}, "metrics.json")
        raise SystemExit(5)

    t = time.time()
    model, proc, dev, minfo = G.load_vlm(MODEL_DIR, G.MAX_PIXELS)
    PHASE["model_load_s"] = time.time() - t
    mident = L.model_identity()
    prior = "2402ffda2977a2ff7450b96c18771bfb0fe416ed840c2cff51014572f8f3cb9b"
    mident["matches_prior_measured_f1_p0_p1"] = bool(
        mident["model_fingerprint"]["name_size_manifest_sha256"] == prior)
    mident["note"] = ("model_manifest_matches compares with the stale lv_common constant "
                      "47cf90f8...; recorded for traceability, not used as a gate")
    log("model loaded: %r" % (minfo.get("gpu_name"),))

    readings = measure_units(side_units, model, proc, dev, out_rel, tag)

    none_rows = [r["qid"] for r in readings if any(r["letters"][a] is None for a in ARMS)]
    out_of_alpha = [r["qid"] for r in readings for a in ARMS
                    if r["letters"][a] is not None and r["letters"][a] not in list("ABCD")]
    tab = tables(readings, with_floor=(mode != "pilot"))

    out = {"run": {"run": RUN_PILOT if mode == "pilot" else RUN_MEASURE, "tag": "fresh_" + tag,
                   "stage": STAGE, "plan_id": PLAN_ID, "plan_version": PLAN_VERSION,
                   "exp_seed": EXP_SEED, "decides_claim": False},
           "data_identity": ident, "manifest_path": manifest_path,
           "split": manifest["split"], "counts": manifest["counts"],
           "units": manifest["units"], "n_measured": len(readings),
           "tables": tab,
           "extraction": {"n_none": len(none_rows), "none_qids": none_rows,
                          "out_of_alphabet_qids": out_of_alpha,
                          "rule": "mvp_common.parse_letter: first A-Ea-e, uppercased; "
                                  "None counts as wrong and is not dropped"},
           "model_load": dict(minfo, model_identity=mident),
           "notes": NOTES,
           "metrics_problem": None}
    if mode == "measure":
        out["positive_control"] = positive_control(model, proc, dev)
        out["verdict"] = _verdict(tab, out["positive_control"])
    # accounting is taken AFTER every model call (the positive control included), so the
    # forward ledger and the phase timing cover the whole run, not just the main arms.
    out["forward_counts"] = FWD.report()
    out["phases_seconds"] = {k: round(v, 1) for k, v in PHASE.items()}
    out["elapsed_seconds"] = round(time.time() - T0, 1)
    jdump(out, "metrics.json")
    log("done: %s" % json.dumps({a: [tab["arms"][a]["hits"], tab["arms"][a]["n"]] for a in ARMS}))
    if mode == "measure":
        log("verdict: %s" % out["verdict"]["label"])


def _verdict(tab, pc):
    nf, u = tab["arms"]["no_frames"], tab["arms"]["uniform@16"]
    cap = tab["capability_control"]
    pc_pass = bool(pc.get("pass"))
    out = {"bar_hits": tab["bar_hits"], "n": tab["n"],
           "no_frames": [nf["hits"], nf["n"], nf["cp95"]],
           "uniform@16": [u["hits"], u["n"], u["cp95"]],
           "floor_holds": tab["floor_holds"],
           "floor_certified": tab["floor_certified"],
           "capability_established": cap["established"],
           "positive_control_pass": pc_pass,
           "mechanism": tab["mechanism_reading"]["classification"]}
    if not pc_pass:
        out["label"] = "positive_control_failed_identity_not_checked"
        out["statement"] = ("positive control failed: the archived screen letters were not "
                            "reproduced; the confirmation readings are reported but marked "
                            "not identity-checked")
    elif tab["floor_holds"] and tab["floor_certified"]:
        out["label"] = "floor_holds_on_untouched_videos"
        out["statement"] = ("the floor holds on genuinely untouched videos: both arms' "
                            "exact 95% intervals lie entirely below 0.35")
    elif tab["floor_holds"]:
        out["label"] = "floor_holds_mechanically_unresolved"
        out["statement"] = ("the floor holds mechanically (both arms <= bar) but at least one "
                            "exact 95% interval reaches 0.35; not certifiable at this n")
    else:
        out["label"] = "floor_fails_on_untouched_videos"
        if cap["established"]:
            out["statement"] = ("the floor does not hold on genuinely untouched videos while "
                                "the single-frame control shows the questions are answerable "
                                "when the referenced moment is given: this carrier cannot "
                                "support the construction with these tools")
        else:
            out["statement"] = ("the floor does not hold on genuinely untouched videos, and "
                                "the single-frame control does not establish that the "
                                "questions are answerable even with the referenced moment: "
                                "either way this carrier cannot support the construction "
                                "with these tools")
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "build":
        run_build()
    elif mode in ("pilot", "measure"):
        run_measure(mode)
    else:
        raise SystemExit("usage: floor_fresh.py build|pilot|measure")


if __name__ == "__main__":
    main()
