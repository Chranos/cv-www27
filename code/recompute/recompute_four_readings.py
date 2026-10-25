#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recompute_four_readings.py -- independent recomputation for the four
publication-scale readings of project (commit 76ca8512).

Written to close the items that `review-four-readings-full-check@v2`
(#aa3630db...) left marked unverified.  Labels below are that review's.

  U1  exact intervals.  Every Clopper-Pearson 95% interval in the four runs:
      the 6 selection pool arms, 6 x 2 selection subset arms, the 4 single
      arms, the 13 distortion arms and the 3 donor arms (41 intervals).

  U2  video-clustered intervals.  Every cluster_boot95 in the four runs, with
      the runs' own resampling unit (UNIT_KEY = video_id, questions of one
      video resampled together, per-video mean of the paired difference) and
      the runs' own seed stream s_boot = (0*1000003 + 37) mod (2**31-1) = 37,
      N_BOOT = 10000, percentile [2.5, 97.5].  32 comparisons.

  U3  the sign tests.  Every video-level sign test (n_pos, n_neg, exact
      two-sided binomial p) attached to those same 32 comparisons, computed on
      the per-video mean differences as the runs computed it.

  U4  the whole-pool aggregation over all 110 questions for the first two
      readings (a)+(b), which live in the selection run: 6 arms over 110
      questions / 56 videos, the 9 pool comparisons, the two pre-registered
      segments pinned40 / new70 with their arms and 4 comparisons each, the
      zero-information option prior, the constant-letter baseline, both Holm
      families, the b-branch and headroom flags, selection overlap, the
      forward ledger and the extraction counters, plus the additive identity
      pinned40 + new70 = pool per arm.

  U5  a cell-by-cell identity recount per run, from the run's own sample file
      and the archived file it was compared against:
        selection   110 rows vs task-ca0a24e61b  (40 matched, arm pred_uniform)
        single       24 rows vs task-67bfc74ca9  (24 matched, 2 arms)
        distortion   17 rows vs task-68cebbe16b  (8 matched, 15 arms), with the
                     corrected tally over the 13 arms full mode actually ran
                     (104 cells) and the 16 null cells listed one by one
        donor       design P4 has no identity control; the only cell-level
                     check that exists (donor_pair != pair) is recounted and
                     the absence of identity_vs_archive is asserted.

WHAT IT DOES NOT DO
  no model forward, no GPU, no network, no read of any data directory; no
  import of any producer module; no rerun of any measurement.  It reads only
  the runs' declared outputs and the three archived sample files, and writes
  one file.

INPUTS (relative to --root, default ".")
  sources/task-23cd232b2b/metrics.json  + results/four_selection_samples.jsonl
  sources/task-88f6a1123d/metrics.json  + results/four_single_samples.jsonl
  sources/task-259e5c2b3a/metrics.json  + results/four_distortion_samples.jsonl
  sources/task-07c9beea59/metrics.json  + results/four_donor_samples.jsonl
  sources/task-ca0a24e61b/results/mvp2_samples.jsonl
  sources/task-67bfc74ca9/results/fresh_readings.jsonl
  sources/task-68cebbe16b/results/twin5_P1_carrier_samples.jsonl
  (the primary task's own outputs are also accepted at original/)

OUTPUT
  one JSON file, default metrics.json: one record per quantity with the value,
  the value it is checked against, the tolerance and PASS/FAIL.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter

import numpy as np
from scipy.stats import beta as BetaDist
from scipy.stats import binomtest

# ============================================================ parameters, all literal
SEED = 0
S_BOOT = (SEED * 1000003 + 37) % (2 ** 31 - 1)
N_BOOT = 10000
BOOT_PCT = (2.5, 97.5)
ALPHA = 0.05
BAR_FRAC = 0.35
KS = (0, 1, 2)
UNIT_KEY = "video_id"
UNIT_RULE = "one row per question; rows sharing a video_id are resampled together"
TOL_INT = 0
TOL_FLOAT = 1e-12
TOL_INTERVAL = 1e-9
TOL_BOOT = 1e-9
TOL_P = 1e-9

SEL_ARMS = ["uniform@16", "no_frames", "contrast@16_k0", "contrast@16_k1",
            "contrast@16_k2", "constant_prior"]
SEL_ANSWER_ARMS = ["uniform@16", "no_frames", "contrast@16_k0", "contrast@16_k1",
                   "contrast@16_k2"]
SINGLE_ARMS = ["uniform@16", "single_frame_j_star", "no_frames", "constant_prior"]
DIST_ARMS = (["single_clean"] +
             ["distorted_%s_%s_k%d" % (kind, form, k)
              for form in ("patchcopy", "noise") for k in KS
              for kind in ("evidence", "nonevidence")])
DONOR_ARMS = ["donor_floor_probe@16", "no_frames", "constant_prior"]

SEL_COMPS = [
    ("contrast@16_k0_vs_uniform@16", "contrast@16_k0", "uniform@16"),
    ("contrast@16_k0_vs_no_frames", "contrast@16_k0", "no_frames"),
    ("contrast@16_k1_vs_uniform@16", "contrast@16_k1", "uniform@16"),
    ("contrast@16_k1_vs_no_frames", "contrast@16_k1", "no_frames"),
    ("contrast@16_k2_vs_uniform@16", "contrast@16_k2", "uniform@16"),
    ("contrast@16_k2_vs_no_frames", "contrast@16_k2", "no_frames"),
    ("uniform@16_vs_no_frames", "uniform@16", "no_frames"),
    ("no_frames_vs_constant_prior", "no_frames", "constant_prior"),
    ("uniform@16_vs_constant_prior", "uniform@16", "constant_prior"),
]
SINGLE_COMPS = [
    ("single_frame_j_star_vs_uniform@16", "single_frame_j_star", "uniform@16"),
    ("uniform@16_vs_no_frames", "uniform@16", "no_frames"),
    ("single_frame_j_star_vs_no_frames", "single_frame_j_star", "no_frames"),
    ("single_frame_j_star_vs_constant_prior", "single_frame_j_star", "constant_prior"),
    ("uniform@16_vs_constant_prior", "uniform@16", "constant_prior"),
    ("no_frames_vs_constant_prior", "no_frames", "constant_prior"),
]
DONOR_COMPS = [
    ("comparison", "donor_floor_probe@16", "no_frames"),
    ("comparison_vs_prior", "donor_floor_probe@16", "constant_prior"),
    ("no_frames_vs_prior", "no_frames", "constant_prior"),
]

RUNS = [
    ("selection", "task-23cd232b2b", "results/four_selection_samples.jsonl"),
    ("single", "task-88f6a1123d", "results/four_single_samples.jsonl"),
    ("distortion", "task-259e5c2b3a", "results/four_distortion_samples.jsonl"),
    ("donor", "task-07c9beea59", "results/four_donor_samples.jsonl"),
]
ARCHIVES = [
    ("selection", "task-ca0a24e61b", "results/mvp2_samples.jsonl"),
    ("single", "task-67bfc74ca9", "results/fresh_readings.jsonl"),
    ("distortion", "task-68cebbe16b", "results/twin5_P1_carrier_samples.jsonl"),
]

ID_MAP_SELECTION = {"uniform@16": "pred_uniform"}
ID_MAP_SINGLE = {"uniform@16": "uniform@16",
                 "single_frame_j_star": "single_frame_j_star"}
ID_MAP_DISTORTION = {
    "single_clean": "single_clean",
    "uniform@16": "uniform@16",
    "no_frames": "no_frames",
    "distorted_evidence_patchcopy_k0": "single_distorted_evidence_ours16_patchcopy_k0",
    "distorted_evidence_patchcopy_k1": "single_distorted_evidence_ours16_patchcopy_k1",
    "distorted_evidence_patchcopy_k2": "single_distorted_evidence_ours16_patchcopy_k2",
    "distorted_evidence_noise_k0": "single_distorted_evidence_ours16_noise_k0",
    "distorted_evidence_noise_k1": "single_distorted_evidence_ours16_noise_k1",
    "distorted_evidence_noise_k2": "single_distorted_evidence_ours16_noise_k2",
    "distorted_nonevidence_patchcopy_k0": "single_distorted_nonevidence_ours16_patchcopy_k0",
    "distorted_nonevidence_patchcopy_k1": "single_distorted_nonevidence_ours16_patchcopy_k1",
    "distorted_nonevidence_patchcopy_k2": "single_distorted_nonevidence_ours16_patchcopy_k2",
    "distorted_nonevidence_noise_k0": "single_distorted_nonevidence_ours16_noise_k0",
    "distorted_nonevidence_noise_k1": "single_distorted_nonevidence_ours16_noise_k1",
    "distorted_nonevidence_noise_k2": "single_distorted_nonevidence_ours16_noise_k2",
}

REC = []
SKIP = []


def fmt(v):
    if isinstance(v, float):
        return "%.17g" % v
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    return str(v)


def close_enough(got, want, tol):
    if want is None or got is None:
        return (want is None) and (got is None)
    if isinstance(want, (list, tuple)):
        if not isinstance(got, (list, tuple)) or len(got) != len(want):
            return False
        return all(close_enough(g, w, tol) for g, w in zip(got, want))
    if isinstance(want, bool) or isinstance(got, bool):
        return bool(got) == bool(want)
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return abs(float(got) - float(want)) <= tol
    return got == want


def chk(item, got, want, tol, note=""):
    ok = close_enough(got, want, tol)
    rec = {"item": item, "got": got, "want": want, "tol": tol,
           "status": "PASS" if ok else "FAIL"}
    if note:
        rec["note"] = note
    REC.append(rec)
    print("[%s] %-76s got=%s want=%s tol=%s%s"
          % ("PASS" if ok else "FAIL", item, fmt(got), fmt(want), fmt(tol),
             ("  # " + note) if note else ""))
    return ok


def skip(item, why):
    SKIP.append({"item": item, "why": why})
    print("[SKIP] %-76s %s" % (item, why))


def need(obj, *path):
    cur = obj
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            raise KeyError("missing field %r (have %r)"
                           % ("/".join(str(x) for x in path),
                              sorted(cur.keys())[:12] if isinstance(cur, dict) else type(cur).__name__))
    return cur


def load_rows(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cp_ci(hits, n):
    k = int(hits)
    n = int(n)
    if n <= 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(BetaDist.ppf(ALPHA / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(BetaDist.ppf(1.0 - ALPHA / 2.0, k + 1, n - k))
    return [lo, hi]


def arm_stats(rows, arm):
    vals = [r["correct"].get(arm) for r in rows]
    n = len(vals)
    hits = int(sum(int(v) for v in vals if v is not None))
    return {"hits": hits, "n": n,
            "value": (float(hits) / n if n else None),
            "cp95": cp_ci(hits, n)}


def per_video_means(rows, arm):
    by = {}
    for r in rows:
        by.setdefault(r[UNIT_KEY], []).append(int(r["correct"][arm]))
    return dict((v, float(np.mean(x))) for v, x in by.items())


def cluster_boot(rows, arm_a, arm_b):
    by = {}
    for r in rows:
        by.setdefault(r[UNIT_KEY], []).append(
            (int(r["correct"][arm_a]), int(r["correct"][arm_b])))
    vids = sorted(by)
    per = np.array([np.mean([x - y for x, y in by[v]]) for v in vids], float)
    rng = np.random.default_rng(int(S_BOOT))
    idx = rng.integers(0, len(per), size=(N_BOOT, len(per)))
    means = per[idx].mean(axis=1)
    lo, hi = np.percentile(means, list(BOOT_PCT))
    return [float(lo), float(hi)]


def mcnemar(rows, arm_a, arm_b):
    n01 = int(sum(1 for r in rows
                  if int(r["correct"][arm_a]) == 0 and int(r["correct"][arm_b]) == 1))
    n10 = int(sum(1 for r in rows
                  if int(r["correct"][arm_a]) == 1 and int(r["correct"][arm_b]) == 0))
    m = n01 + n10
    p = 1.0 if m == 0 else float(binomtest(min(n01, n10), m, 0.5).pvalue)
    return {"n01": n01, "n10": n10, "p": p}


def sign_test(rows, arm_a, arm_b):
    va = per_video_means(rows, arm_a)
    vb = per_video_means(rows, arm_b)
    vd = [va[v] - vb[v] for v in sorted(va)]
    pos = int(sum(1 for d in vd if d > 0))
    neg = int(sum(1 for d in vd if d < 0))
    p = 1.0 if pos + neg == 0 else float(binomtest(min(pos, neg), pos + neg, 0.5).pvalue)
    return {"n_pos": pos, "n_neg": neg, "p": p}, len(vd)


def compare(rows, arm_a, arm_b):
    d = [int(r["correct"][arm_a]) - int(r["correct"][arm_b]) for r in rows]
    st, nv = sign_test(rows, arm_a, arm_b)
    return {"mean_diff": float(np.mean(d)),
            "cluster_boot95": cluster_boot(rows, arm_a, arm_b),
            "mcnemar": mcnemar(rows, arm_a, arm_b),
            "video_sign_test": st, "n_videos": nv}


def holm_monotone(pairs):
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    m = len(pairs)
    out = [None] * m
    prev = 0.0
    for rank, i in enumerate(order):
        name, p = pairs[i]
        adj = min(1.0, max(prev, (m - rank) * p))
        prev = adj
        out[i] = {"name": name, "p_raw": p, "p_holm": adj,
                  "reject_005": bool(adj < 0.05)}
    return out


def holm_stepdown(pairs, alpha=ALPHA):
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


def row_keys(r):
    keys = []
    if r.get("qid") is not None:
        keys.append(("qid", r.get("qid")))
    if r.get("video_id") is not None and r.get("pair") is not None:
        keys.append(("vp", r.get("video_id"), int(r.get("pair"))))
    return keys


def index_archive(rows):
    d = {}
    for o in rows:
        for k in row_keys(o):
            d[k] = o
    return d


def identity_cells(rows, arc_index, arm_map):
    cells = []
    n_overlap = 0
    for r in rows:
        a = None
        for k in row_keys(r):
            if k in arc_index:
                a = arc_index[k]
                break
        if a is None:
            continue
        n_overlap += 1
        letters = a.get("letters") or {}
        mine_letters = r.get("letters") or {}
        for mine, theirs in arm_map.items():
            want = letters.get(theirs, a.get(theirs))
            if want is None:
                continue
            got = mine_letters.get(mine)
            cells.append({"qid": r.get("qid"), "pair": r.get("pair"), "arm": mine,
                          "this": got, "archive": want, "equal": bool(got == want)})
    return n_overlap, cells


def per_arm_tally(cells):
    per = {}
    for c in cells:
        d = per.setdefault(c["arm"], [0, 0])
        d[1] += 1
        if c["equal"]:
            d[0] += 1
    return per


def check_compare(prefix, rows, arm_a, arm_b, rep):
    got = compare(rows, arm_a, arm_b)
    chk(prefix + "/mean_diff", got["mean_diff"], need(rep, "mean_diff"), TOL_FLOAT)
    chk(prefix + "/cluster_boot95/lo", got["cluster_boot95"][0],
        need(rep, "cluster_boot95")[0], TOL_BOOT)
    chk(prefix + "/cluster_boot95/hi", got["cluster_boot95"][1],
        need(rep, "cluster_boot95")[1], TOL_BOOT)
    chk(prefix + "/mcnemar/n01", got["mcnemar"]["n01"], need(rep, "mcnemar", "n01"), TOL_INT)
    chk(prefix + "/mcnemar/n10", got["mcnemar"]["n10"], need(rep, "mcnemar", "n10"), TOL_INT)
    chk(prefix + "/mcnemar/p", got["mcnemar"]["p"], need(rep, "mcnemar", "p"), TOL_P)
    chk(prefix + "/sign_test/n_pos", got["video_sign_test"]["n_pos"],
        need(rep, "video_sign_test", "n_pos"), TOL_INT)
    chk(prefix + "/sign_test/n_neg", got["video_sign_test"]["n_neg"],
        need(rep, "video_sign_test", "n_neg"), TOL_INT)
    chk(prefix + "/sign_test/p", got["video_sign_test"]["p"],
        need(rep, "video_sign_test", "p"), TOL_P)
    chk(prefix + "/n_videos", got["n_videos"], need(rep, "n_videos"), TOL_INT)
    return got


def check_arms(prefix, rows, rep, arms):
    for a in arms:
        r_rep = need(rep, "arms", a)
        s = arm_stats(rows, a)
        chk(prefix + "/" + a + "/hits", s["hits"], need(r_rep, "hits"), TOL_INT)
        chk(prefix + "/" + a + "/n", s["n"], need(r_rep, "n"), TOL_INT)
        chk(prefix + "/" + a + "/value", s["value"], need(r_rep, "value"), TOL_FLOAT)
        chk(prefix + "/" + a + "/cp95/lo", s["cp95"][0],
            need(r_rep, "cp95")[0], TOL_INTERVAL)
        chk(prefix + "/" + a + "/cp95/hi", s["cp95"][1],
            need(r_rep, "cp95")[1], TOL_INTERVAL)


def section_inputs(rows_by, rep_by):
    print("---- inputs and run parameters")
    for name, _tid, _rel in RUNS:
        rows = rows_by[name]
        rep = rep_by[name]
        chk("inputs/" + name + "/n_rows", len(rows), need(rep, "n_measured"), TOL_INT)
        chk("inputs/" + name + "/seed", need(rep, "run", "exp_seed"), SEED, TOL_INT)
        chk("inputs/" + name + "/s_boot_from_seed", S_BOOT, 37, TOL_INT,
            "s_boot = (seed*1000003+37) mod (2**31-1)")
        chk("inputs/" + name + "/reading", need(rep, "run", "reading"), name, 0)
        chk("inputs/" + name + "/mode", need(rep, "run", "mode"), "full", 0)
        chk("inputs/" + name + "/unit_is_video",
            "video" in str(need(rep, "independent_unit")), True, 0, UNIT_RULE)
        chk("inputs/" + name + "/unit_key_present_in_rows",
            all(UNIT_KEY in r for r in rows), True, 0)


def section_selection(rows, rep, arc_rows):
    print("---- U1/U4 selection: readings (a)+(b) over all 110 questions")
    check_arms("U1/selection/pool110", rows, rep, SEL_ARMS)

    chk("U4/selection/pool/n_questions", len(rows), need(rep, "verdict", "n_questions"), TOL_INT)
    chk("U4/selection/pool/n_videos", len(set(r[UNIT_KEY] for r in rows)),
        need(rep, "verdict", "n_videos"), TOL_INT)
    chk("U4/selection/pool/resolution_question_pp", 100.0 / len(rows),
        need(rep, "verdict", "resolution_question_pp"), TOL_FLOAT)
    chk("U4/selection/pool/resolution_video_pp",
        100.0 / len(set(r[UNIT_KEY] for r in rows)),
        need(rep, "verdict", "resolution_video_pp"), TOL_FLOAT)
    chk("U4/selection/pool/n_questions_matches_identity", len(rows),
        need(rep, "identity", "n_questions"), TOL_INT)
    chk("U4/selection/pool/n_videos_matches_identity", len(set(r[UNIT_KEY] for r in rows)),
        need(rep, "identity", "n_videos"), TOL_INT)
    chk("U4/selection/pool/structure_matches_expected", True,
        need(rep, "structure", "matches_expected"), 0)

    gold = Counter(r["gold"] for r in rows)
    prior = need(rep, "pretest_prior", "pool110")
    chk("U4/selection/prior/n", len(rows), need(prior, "n"), TOL_INT)
    chk("U4/selection/prior/gold_letter_counts", dict(sorted(gold.items())),
        need(prior, "gold_letter_counts"), 0)
    best = max(sorted(gold.items()), key=lambda kv: kv[1])
    chk("U4/selection/prior/prior_letter", best[0], need(prior, "prior_letter"), 0,
        "alphabetically first among the maxima")
    chk("U4/selection/prior/prior_hits", int(best[1]), need(prior, "prior_hits"), TOL_INT)
    chk("U4/selection/prior/prior_value", float(best[1]) / len(rows),
        need(prior, "prior_value"), TOL_FLOAT)

    base = need(rep, "constant_letter_baseline")
    chk("U4/selection/constant_baseline/best_letter", best[0], need(base, "best_letter"), 0)
    chk("U4/selection/constant_baseline/hits", int(best[1]),
        need(base, "best_letter_hits"), TOL_INT)
    chk("U4/selection/constant_baseline/n", len(rows), need(base, "n"), TOL_INT)
    chk("U4/selection/constant_baseline/value", float(best[1]) / len(rows),
        need(base, "value"), TOL_FLOAT)
    chk("U4/selection/constant_baseline/gold_letter_counts", dict(sorted(gold.items())),
        need(base, "gold_letter_counts"), 0)

    pool_comps = {}
    for key, a, b in SEL_COMPS:
        pool_comps[key] = check_compare("U2U3/selection/pool/" + key, rows, a, b,
                                        need(rep, "comparisons", key))

    streams = [(key, pool_comps[key]["mcnemar"]["p"]) for key in
               ("contrast@16_k0_vs_uniform@16", "contrast@16_k1_vs_uniform@16",
                "contrast@16_k2_vs_uniform@16")]
    stream_holm = holm_monotone(streams)
    rep_streams = need(rep, "holm_streams_vs_uniform")
    for i, h in enumerate(stream_holm):
        r_h = rep_streams[i]
        chk("U4/selection/holm_streams/%d/name" % i, h["name"], need(r_h, "name"), 0)
        chk("U4/selection/holm_streams/%d/p_raw" % i, h["p_raw"], need(r_h, "p_raw"), TOL_P)
        chk("U4/selection/holm_streams/%d/p_holm" % i, h["p_holm"],
            need(r_h, "p_holm"), TOL_P)
        chk("U4/selection/holm_streams/%d/reject" % i, h["reject_005"],
            need(r_h, "reject_005"), 0)

    fam2_in = [("no_frames_vs_constant_prior",
                pool_comps["no_frames_vs_constant_prior"]["mcnemar"]["p"]),
               ("no_frames_vs_uniform@16",
                pool_comps["uniform@16_vs_no_frames"]["mcnemar"]["p"]),
               ("uniform@16_vs_constant_prior",
                pool_comps["uniform@16_vs_constant_prior"]["mcnemar"]["p"])]
    fam2 = holm_stepdown(fam2_in)
    rep_fam2 = need(rep, "holm_family2_identification")
    chk("U4/selection/holm_family2/family_size", fam2["family_size"],
        need(rep_fam2, "family_size"), TOL_INT)
    for nm, dec in fam2["decisions"].items():
        r_dec = need(rep_fam2, "decisions", nm)
        chk("U4/selection/holm_family2/" + nm + "/p", dec["p"], need(r_dec, "p"), TOL_P)
        chk("U4/selection/holm_family2/" + nm + "/threshold", dec["threshold"],
            need(r_dec, "threshold"), TOL_FLOAT)
        chk("U4/selection/holm_family2/" + nm + "/p_holm", dec["p_holm"],
            need(r_dec, "p_holm"), TOL_P)
        chk("U4/selection/holm_family2/" + nm + "/reject", dec["reject"],
            need(r_dec, "reject"), 0)

    nf = pool_comps["no_frames_vs_constant_prior"]
    uf = pool_comps["uniform@16_vs_no_frames"]
    up = pool_comps["uniform@16_vs_constant_prior"]
    rej = fam2["decisions"]
    sep_above = bool(rej["no_frames_vs_constant_prior"]["reject"] and
                     nf["cluster_boot95"][0] is not None and
                     nf["cluster_boot95"][0] > 0)
    sep_below = bool(rej["no_frames_vs_constant_prior"]["reject"] and
                     nf["cluster_boot95"][1] is not None and
                     nf["cluster_boot95"][1] < 0)
    headroom = bool(rej["no_frames_vs_uniform@16"]["reject"] and
                    uf["cluster_boot95"][0] is not None and
                    uf["cluster_boot95"][0] > 0 and
                    rej["uniform@16_vs_constant_prior"]["reject"] and
                    up["cluster_boot95"][0] is not None and
                    up["cluster_boot95"][0] > 0)
    branch = "b1" if sep_above else ("b3" if sep_below else "b2")
    br = need(rep, "branches")
    chk("U4/selection/branch/b_branch", branch, need(br, "b_branch"), 0)
    chk("U4/selection/branch/headroom", headroom, need(br, "headroom"), 0)
    chk("U4/selection/branch/b4_identification_failure", (not headroom),
        need(br, "b4_identification_failure"), 0)

    pinned_qids = set()
    for o in arc_rows:
        if o.get("qid") is not None:
            pinned_qids.add(o["qid"])
    chk("U4/selection/pinned_qids_from_archive", len(pinned_qids), 40, TOL_INT)
    rows40 = [r for r in rows if r["qid"] in pinned_qids]
    rows70 = [r for r in rows if r["qid"] not in pinned_qids]
    for sub, subrows in (("pinned40", rows40), ("new70", rows70)):
        srep = need(rep, "subsets", sub)
        chk("U4/selection/subset/" + sub + "/n_questions", len(subrows),
            need(srep, "n_questions"), TOL_INT)
        chk("U4/selection/subset/" + sub + "/n_videos",
            len(set(r[UNIT_KEY] for r in subrows)), need(srep, "n_videos"), TOL_INT)
        check_arms("U1/U4/selection/" + sub, subrows, srep, SEL_ARMS)
        for k in KS:
            key = "contrast@16_k%d" % k
            check_compare("U2U3/selection/" + sub + "/" + key + "_vs_uniform@16",
                          subrows, key, "uniform@16", need(srep, "deltas", key))
        check_compare("U2U3/selection/" + sub + "/no_frames_vs_constant_prior",
                      subrows, "no_frames", "constant_prior",
                      need(srep, "no_frames_vs_constant_prior"))
    for a in SEL_ARMS:
        chk("U4/selection/split_sum/" + a,
            arm_stats(rows40, a)["hits"] + arm_stats(rows70, a)["hits"],
            arm_stats(rows, a)["hits"], TOL_INT, "pinned40 + new70 = pool")

    for k in KS:
        key = "contrast@16_k%d" % k
        m = float(np.mean([r["overlap_vs_uniform"][key] for r in rows]))
        chk("U4/selection/selection_overlap/" + key, m,
            need(rep, "selection_overlap", key, "mean"), TOL_FLOAT)

    n = len(rows)
    fc = need(rep, "forward_counts")
    chk("U4/selection/forward/total", n * (5 + 32 + 3 * 32), need(fc, "total"), TOL_INT)
    for a in SEL_ANSWER_ARMS:
        chk("U4/selection/forward/by_arm/answer:" + a, n,
            need(fc, "by_arm", "answer:" + a), TOL_INT)
    chk("U4/selection/forward/by_kind/score:clean@32", n * 32,
        need(fc, "by_kind", "score:clean@32"), TOL_INT)
    chk("U4/selection/forward/by_kind/score:distorted@32", n * 3 * 32,
        need(fc, "by_kind", "score:distorted@32"), TOL_INT)

    ext = need(rep, "extraction")
    n_none = int(sum(1 for r in rows for a in SEL_ANSWER_ARMS
                     if (r.get("letters") or {}).get(a) is None))
    n_bad = int(sum(1 for r in rows for a in SEL_ANSWER_ARMS
                    if ((r.get("letters") or {}).get(a) is None or
                        (r.get("letters") or {}).get(a) not in list("ABCDE"))))
    chk("U4/selection/extraction/n_none_letter_cells", n_none,
        need(ext, "n_none_letter_cells"), TOL_INT)
    chk("U4/selection/extraction/n_bad", n_bad, need(ext, "n_bad"), TOL_INT)


def section_single(rows, rep):
    print("---- U1/U2/U3 single: reading (c)")
    check_arms("U1/single", rows, rep, SINGLE_ARMS)
    comps = {}
    for key, a, b in SINGLE_COMPS:
        comps[key] = check_compare("U2U3/single/" + key, rows, a, b,
                                   need(rep, "comparisons", key))

    bd = need(rep, "bounded_difference")
    n = len(rows)
    pair = comps["single_frame_j_star_vs_uniform@16"]
    n10 = pair["mcnemar"]["n10"]
    n01 = pair["mcnemar"]["n01"]
    m = n10 + n01
    chk("U2U3/single/bounded/point", pair["mean_diff"], need(bd, "point"), TOL_FLOAT)
    chk("U2U3/single/bounded/point_units", int(round(pair["mean_diff"] * n)),
        need(bd, "point_units"), TOL_INT)
    chk("U2U3/single/bounded/n10_single_only", n10, need(bd, "n10_single_only"), TOL_INT)
    chk("U2U3/single/bounded/n01_uniform_only", n01, need(bd, "n01_uniform_only"), TOL_INT)
    chk("U2U3/single/bounded/discordant_pairs", m, need(bd, "discordant_pairs"), TOL_INT)
    chk("U2U3/single/bounded/n_units", n, need(bd, "n_units"), TOL_INT)
    chk("U2U3/single/bounded/unit_pp", 100.0 / n, need(bd, "unit_pp"), TOL_FLOAT)
    chk("U2U3/single/bounded/smallest_symmetric_halfwidth_units",
        (int(math.ceil(1.96 * math.sqrt(m))) if m > 0 else 0),
        need(bd, "smallest_symmetric_halfwidth_units_at_this_m"), TOL_INT)
    chk("U2U3/single/bounded/approx_halfwidth_units", 1.96 * math.sqrt(m) / n,
        need(bd, "approx_halfwidth_units"), TOL_FLOAT)
    chk("U2U3/single/bounded/exact_upper_bound_if_m_zero", 1.0 - 0.025 ** (1.0 / n),
        need(bd, "exact_upper_bound_if_m_zero"), TOL_FLOAT)
    chk("U2U3/single/bounded/equivalence_claimed", False,
        need(bd, "equivalence_claimed"), 0)

    gold = Counter(r["gold"] for r in rows)
    best = max(sorted(gold.items()), key=lambda kv: kv[1])
    base = need(rep, "constant_letter_baseline")
    chk("U1/single/constant_baseline/gold_letter_counts", dict(sorted(gold.items())),
        need(base, "gold_letter_counts"), 0)
    chk("U1/single/constant_baseline/best_letter", best[0], need(base, "best_letter"), 0)
    chk("U1/single/constant_baseline/hits", int(best[1]),
        need(base, "best_letter_hits"), TOL_INT)

    prior = need(rep, "pretest_prior")
    chk("U1/single/prior/n", n, need(prior, "n"), TOL_INT)
    chk("U1/single/prior/gold_letter_counts", dict(sorted(gold.items())),
        need(prior, "gold_letter_counts"), 0)
    chk("U1/single/prior/prior_letter", best[0], need(prior, "prior_letter"), 0)
    chk("U1/single/prior/prior_hits", int(best[1]), need(prior, "prior_hits"), TOL_INT)

    fc = need(rep, "forward_counts")
    chk("U1/single/forward/total", 3 * n, need(fc, "total"), TOL_INT)
    chk("U1/single/forward/by_arm/single_frame_j_star", n,
        need(fc, "by_arm", "answer:single_frame_j_star"), TOL_INT)

    ext = need(rep, "extraction")
    n_none = int(sum(1 for r in rows for a in SINGLE_ARMS[:3]
                     if (r.get("letters") or {}).get(a) is None))
    chk("U1/single/extraction/n_none_letter_cells", n_none,
        need(ext, "n_none_letter_cells"), TOL_INT)


def section_distortion(rows, rep):
    print("---- U1/U2/U3 distortion: reading (d1)")
    check_arms("U1/distortion", rows, rep, DIST_ARMS)
    n = len(rows)
    bar = int(math.floor(BAR_FRAC * n))
    chk("U1/distortion/bar_hits", bar, need(rep, "bar_hits"), TOL_INT)
    for form in ("patchcopy", "noise"):
        for k in KS:
            pf = need(rep, "per_form", form, str(k))
            a1 = "distorted_evidence_%s_k%d" % (form, k)
            a2 = "distorted_nonevidence_%s_k%d" % (form, k)
            chk("U1/distortion/%s_k%d/evidence_hits" % (form, k),
                arm_stats(rows, a1)["hits"], need(pf, "evidence_hits"), TOL_INT)
            chk("U1/distortion/%s_k%d/nonevidence_hits" % (form, k),
                arm_stats(rows, a2)["hits"], need(pf, "nonevidence_hits"), TOL_INT)
            chk("U1/distortion/%s_k%d/bar" % (form, k), bar, need(pf, "bar"), TOL_INT)
            chk("U1/distortion/%s_k%d/evidence_at_or_below_bar" % (form, k),
                bool(arm_stats(rows, a1)["hits"] <= bar),
                need(pf, "evidence_at_or_below_bar"), 0)
            chk("U1/distortion/%s_k%d/cp95_upper_below_bar_frac" % (form, k),
                bool(arm_stats(rows, a1)["cp95"][1] < BAR_FRAC),
                need(pf, "evidence_cp95_upper_below_bar_frac"), 0)
            check_compare("U2U3/distortion/%s_k%d/evidence_vs_nonevidence" % (form, k),
                          rows, a1, a2, need(pf, "evidence_vs_nonevidence"))

    fam = []
    for form in ("patchcopy", "noise"):
        for k in KS:
            a1 = "distorted_evidence_%s_k%d" % (form, k)
            a2 = "distorted_nonevidence_%s_k%d" % (form, k)
            fam.append(("%s_k%d_evidence_vs_nonevidence" % (form, k),
                        mcnemar(rows, a1, a2)["p"]))
    holm = holm_monotone(fam)
    rep_holm = need(rep, "holm_evidence_vs_nonevidence")
    for i, h in enumerate(holm):
        r_h = rep_holm[i]
        chk("U4b/distortion/holm/%d/name" % i, h["name"], need(r_h, "name"), 0)
        chk("U4b/distortion/holm/%d/p_raw" % i, h["p_raw"], need(r_h, "p_raw"), TOL_P,
            "monotone Holm")
        chk("U4b/distortion/holm/%d/p_holm" % i, h["p_holm"], need(r_h, "p_holm"), TOL_P)
        chk("U4b/distortion/holm/%d/reject" % i, h["reject_005"],
            need(r_h, "reject_005"), 0)

    fc = need(rep, "forward_counts")
    chk("U1/distortion/forward/total", len(DIST_ARMS) * n, need(fc, "total"), TOL_INT)
    ext = need(rep, "extraction")
    n_none = int(sum(1 for r in rows for a in DIST_ARMS
                     if (r.get("letters") or {}).get(a) is None))
    chk("U1/distortion/extraction/n_none_letter_cells", n_none,
        need(ext, "n_none_letter_cells"), TOL_INT)


def section_donor(rows, rep):
    print("---- U1/U2/U3 donor: reading (d2)")
    check_arms("U1/donor", rows, rep, DONOR_ARMS)
    for key, a, b in DONOR_COMPS:
        check_compare("U2U3/donor/" + key, rows, a, b, need(rep, key))

    gold = Counter(r["gold"] for r in rows)
    best = max(sorted(gold.items()), key=lambda kv: kv[1])
    prior = need(rep, "pretest_prior")
    chk("U1/donor/prior/n", len(rows), need(prior, "n"), TOL_INT)
    chk("U1/donor/prior/gold_letter_counts", dict(sorted(gold.items())),
        need(prior, "gold_letter_counts"), 0)
    chk("U1/donor/prior/prior_letter", best[0], need(prior, "prior_letter"), 0)
    chk("U1/donor/prior/prior_hits", int(best[1]), need(prior, "prior_hits"), TOL_INT)
    chk("U1/donor/prior/tie",
        int(sum(1 for v in gold.values() if v == best[1])) > 1, need(prior, "tie"), 0)

    fc = need(rep, "forward_counts")
    chk("U1/donor/forward/total", 2 * len(rows), need(fc, "total"), TOL_INT)

    n_diff = int(sum(1 for r in rows if int(r["donor_pair"]) != int(r["pair"])))
    chk("U5/donor/donor_pair_differs_from_pair", n_diff, len(rows), TOL_INT,
        "D1: the donor is another video")
    p_d = need(rep, "p_d")
    chk("U1/donor/p_d/n_pass_plus_n_fail",
        need(p_d, "n_pass") + need(p_d, "n_fail"), need(p_d, "n_visible"), TOL_INT)
    chk("U1/donor/p_d/retry_histogram_sum",
        sum(int(v) for v in need(p_d, "retry_histogram").values()),
        need(p_d, "n_pass"), TOL_INT)
    chk("U1/donor/p_d/n_excluded", 0, need(p_d, "n_excluded_recounted"), TOL_INT)
    skip("U1/donor/p_d/cell_level_D1_D2",
         "per-question D1/D2 pixel records are not among the run's declared outputs")


def section_identity(rows_by, rep_by, arc_by):
    print("---- U5 identity recount, cell by cell")
    sel_rows = rows_by["selection"]
    sel_arc = index_archive(arc_by["selection"])
    n_overlap, cells = identity_cells(sel_rows, sel_arc, ID_MAP_SELECTION)
    irep = need(rep_by["selection"], "identity_vs_archive")
    chk("U5/selection/overlap_rows", n_overlap, need(irep, "n_overlap"), TOL_INT)
    chk("U5/selection/mismatches", len([c for c in cells if not c["equal"]]),
        need(irep, "n_mismatch"), TOL_INT)
    chk("U5/selection/n_expected", n_overlap, need(irep, "n_expected"), TOL_INT)
    per = per_arm_tally(cells)
    for arm, (agree, tot) in per.items():
        r_ag = need(irep, "agreement", arm)
        chk("U5/selection/" + arm + "/agree", agree, need(r_ag, "agree"), TOL_INT)
        chk("U5/selection/" + arm + "/n", tot, need(r_ag, "n"), TOL_INT)

    sgl_rows = rows_by["single"]
    sgl_arc = index_archive(arc_by["single"])
    n_overlap, cells = identity_cells(sgl_rows, sgl_arc, ID_MAP_SINGLE)
    irep = need(rep_by["single"], "identity_vs_archive")
    chk("U5/single/overlap_rows", n_overlap, need(irep, "n_overlap"), TOL_INT)
    chk("U5/single/mismatches", len([c for c in cells if not c["equal"]]),
        need(irep, "n_mismatch"), TOL_INT)
    chk("U5/single/n_expected", n_overlap, need(irep, "n_expected"), TOL_INT)
    per = per_arm_tally(cells)
    for arm, (agree, tot) in per.items():
        r_ag = need(irep, "agreement", arm)
        chk("U5/single/" + arm + "/agree", agree, need(r_ag, "agree"), TOL_INT)
        chk("U5/single/" + arm + "/n", tot, need(r_ag, "n"), TOL_INT)
    chk("U5/single/ok", len([c for c in cells if not c["equal"]]) == 0,
        need(irep, "ok"), 0)

    dst_rows = rows_by["distortion"]
    dst_arc = index_archive(arc_by["distortion"])
    n_overlap, cells = identity_cells(dst_rows, dst_arc, ID_MAP_DISTORTION)
    irep = need(rep_by["distortion"], "identity_vs_archive")
    chk("U5/distortion/overlap_rows", n_overlap, need(irep, "n_overlap"), TOL_INT)
    mism = [c for c in cells if not c["equal"]]
    chk("U5/distortion/mismatches_as_reported", len(mism),
        need(irep, "n_mismatch"), TOL_INT)
    per = per_arm_tally(cells)
    for arm, (agree, tot) in per.items():
        r_ag = need(irep, "agreement", arm)
        chk("U5/distortion/" + arm + "/agree", agree, need(r_ag, "agree"), TOL_INT)
        chk("U5/distortion/" + arm + "/n", tot, need(r_ag, "n"), TOL_INT)
    rep_keys = sorted([[m["arm"], int(m["pair"]), m["qid"]]
                       for m in need(irep, "mismatches")])
    got_keys = sorted([[c["arm"], int(c["pair"]), c["qid"]] for c in mism])
    chk("U5/distortion/mismatch_identities", got_keys, rep_keys, 0)

    arms_present = sorted(set(a for r in dst_rows
                              for a in (r.get("letters") or {}).keys()))
    run_cells = [c for c in cells if c["this"] is not None]
    null_cells = [c for c in cells if c["this"] is None]
    chk("U5/distortion/corrected/arms_run_in_full", len(arms_present), 13, TOL_INT,
        "13 distortion arms = 12 distorted + single_clean")
    chk("U5/distortion/corrected/cells_in_run", len(run_cells), 104, TOL_INT,
        "13 arms x 8 matched pairs")
    chk("U5/distortion/corrected/cells_in_run_equal",
        sum(1 for c in run_cells if c["equal"]), 104, TOL_INT)
    chk("U5/distortion/corrected/cells_null", len(null_cells), 16, TOL_INT,
        "16 null cells from the two pilot-only arms")
    chk("U5/distortion/corrected/null_arms_absent_from_run",
        all(c["arm"] not in arms_present for c in null_cells), True, 0)
    chk("U5/distortion/corrected/cell_total_identity",
        len(run_cells) + len(null_cells), len(cells), TOL_INT)
    chk("U5/distortion/corrected/cells_total", len(cells), 15 * 8, TOL_INT,
        "15 mapped arms x 8 matched pairs")

    chk("U5/donor/identity_control_absent_by_design",
        "identity_vs_archive" in rep_by["donor"], False, 0,
        "Plan P4: no identity control for (d2)")


def find_dir(tid, root, primary):
    cands = []
    if tid == primary:
        cands.append(os.path.join(root, "original"))
    cands.append(os.path.join(root, "sources", tid))
    cands.append(os.path.join(root, tid))
    for c in cands:
        if os.path.isdir(c):
            return c
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="independent recomputation of the four readings")
    ap.add_argument("--root", default=".", help="directory holding original/ and sources/")
    ap.add_argument("--primary", default="task-23cd232b2b",
                    help="task id whose outputs are at original/")
    ap.add_argument("--out", default="metrics.json", help="the single result file written")
    ap.add_argument("--dir", action="append", default=[], metavar="TASK_ID=PATH",
                    help="override the directory of one task (repeatable)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 3 if any check fails (default: always exit 0)")
    args = ap.parse_args(argv)

    overrides = {}
    for spec in args.dir:
        tid, _sep, path = spec.partition("=")
        if tid and path:
            overrides[tid.strip()] = path.strip()

    try:
        script_sha = sha256_file(os.path.abspath(__file__))
    except Exception:
        script_sha = None

    print("recompute_four_readings.py")
    print("parameters: seed=%d s_boot=%d n_boot=%d pct=%s alpha=%s bar_frac=%s unit=%s"
          % (SEED, S_BOOT, N_BOOT, list(BOOT_PCT), ALPHA, BAR_FRAC, UNIT_RULE))

    dirs = {}
    for _name, tid, _rel in RUNS + ARCHIVES:
        if tid in overrides:
            dirs[tid] = overrides[tid]
            continue
        d = find_dir(tid, args.root, args.primary)
        if d is None:
            raise SystemExit("cannot find outputs of %s under %s" % (tid, args.root))
        dirs[tid] = d

    rows_by = {}
    rep_by = {}
    inputs_meta = {}
    for name, tid, rel in RUNS:
        sp = os.path.join(dirs[tid], rel)
        mp = os.path.join(dirs[tid], "metrics.json")
        if not os.path.exists(sp):
            raise SystemExit("missing sample file: " + sp)
        if not os.path.exists(mp):
            raise SystemExit("missing metrics file: " + mp)
        rows_by[name] = load_rows(sp)
        rep_by[name] = read_json(mp)
        inputs_meta[tid] = {"dir": dirs[tid], "samples": rel,
                            "n_rows": len(rows_by[name]),
                            "samples_sha256": sha256_file(sp),
                            "metrics_sha256": sha256_file(mp)}
        print("input %s <- %s (%d rows)" % (name, sp, len(rows_by[name])))
    arc_by = {}
    for name, tid, rel in ARCHIVES:
        p = os.path.join(dirs[tid], rel)
        if not os.path.exists(p):
            raise SystemExit("missing archive file: " + p)
        arc_by[name] = load_rows(p)
        inputs_meta[tid] = {"dir": dirs[tid], "samples": rel,
                            "n_rows": len(arc_by[name]),
                            "samples_sha256": sha256_file(p)}
        print("archive %s <- %s (%d rows)" % (name, p, len(arc_by[name])))

    sections = [("inputs", section_inputs, (rows_by, rep_by)),
                ("selection", section_selection,
                 (rows_by["selection"], rep_by["selection"], arc_by["selection"])),
                ("single", section_single, (rows_by["single"], rep_by["single"])),
                ("distortion", section_distortion, (rows_by["distortion"], rep_by["distortion"])),
                ("donor", section_donor, (rows_by["donor"], rep_by["donor"])),
                ("identity", section_identity, (rows_by, rep_by, arc_by))]
    for label, fn, a in sections:
        try:
            fn(*a)
        except Exception as e:
            REC.append({"item": "section/" + label, "got": None, "want": "no exception",
                        "tol": 0, "status": "FAIL",
                        "note": type(e).__name__ + ": " + str(e)[:300]})
            print("[FAIL] section/%s raised %s: %s" % (label, type(e).__name__, str(e)[:300]))

    n_pass = sum(1 for r in REC if r["status"] == "PASS")
    n_fail = sum(1 for r in REC if r["status"] == "FAIL")
    summary = {
        "script": "recompute_four_readings.py",
        "script_sha256": script_sha,
        "closes": ["U1 exact intervals", "U2 video-clustered intervals",
                   "U3 video-level sign tests", "U4 whole-pool aggregation (a)+(b) 110",
                   "U5 cell-by-cell identity recount per run"],
        "parameters": {"seed": SEED, "s_boot": S_BOOT, "n_boot": N_BOOT,
                       "boot_percentiles": list(BOOT_PCT), "alpha": ALPHA,
                       "bar_frac": BAR_FRAC, "ks": list(KS), "unit_key": UNIT_KEY,
                       "unit_rule": UNIT_RULE},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": __import__("scipy").__version__},
        "inputs": inputs_meta,
        "counts": {"pass": n_pass, "fail": n_fail, "skip": len(SKIP),
                   "total": len(REC)},
        "failures": [r for r in REC if r["status"] == "FAIL"],
        "not_checkable": SKIP,
        "checks": REC,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print("")
    print("SUMMARY pass=%d fail=%d skip=%d total=%d -> %s"
          % (n_pass, n_fail, len(SKIP), len(REC), args.out))
    if args.strict and n_fail:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
