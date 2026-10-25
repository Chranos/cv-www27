"""Shared helpers for the r2 pilot batch (Plan plan-matrix-longvideo-r2@v1).

This module adds exactly what the r2 design adds over the fifth batch, and nothing else:

  * the per-question frame-dependence PRE-SCREEN (design setup section 2): keep a question
    if and only if the no_frames arm answers it wrongly AND the same 16 frames all replaced
    by sigma=0.2 distorted versions still answer it wrongly.  Both readings already exist in
    the arm battery, so the pre-screen costs no forward the battery does not already make.
    The distortion stream is FIXED (PS_FIXED_SEED below), not derived from EXP_SEED, so
    the retained set is identical across run seeds and reproducible.
  * the pre-registered POSITIVE CONTROLS PC1/PC2/PC3 on the carrier-3 B-layer control set
    (2_needle + 1_plotQA), each with the threshold written down in the Plan before the run.
  * the instrument reading helper (G2 responsiveness: median / mean / count over the bar /
    quantiles), reported with its independent unit.
  * the carrier-3 B-layer loader with in-run identity verification against the pin.

Everything else is imported from the fourth/fifth batch modules verbatim, not forked:
mvp_common.decode_frames / to_pil / uniform_indices / distort, gen_common.Fwd /
run_seeds / distort_seeded / prompt_for / top_b / quota_select / tv / option_probs /
option_probs_multi / answer_arm / boot_ci / boot_ci_cluster / compare / load_vlm, and
mx_common.cdir manifest / CLIP helpers / its_select / maxvol_select.
"""
import hashlib
import json
import os

import numpy as np

import mvp_common as C
import gen_common as G
import mx_common as MX

PLAN_ID = "plan-matrix-longvideo-r2"
PLAN_VERSION = 1
STAGE = "smoke"

# ---------------------------------------------------------------- pre-screen (frozen)
# Written down before any run of this batch.  A fixed constant, NOT EXP_SEED, so that
# every run seed decides the same retained set.
PS_FIXED_SEED = 20261016
S_PRE_STREAM = int(G.seed_streams(PS_FIXED_SEED)["s_noise"])
PS_RULE = ("keep q  <=>  a_no_frames != gold  AND  "
           "a_sigma0.2_all16(fixed pre-screen stream s_pre=%d) != gold" % S_PRE_STREAM)
PS_NO_EXTRA_FORWARDS = ("the three readings (no_frames letter, uniform@16 letter, sigma=0.2 "
                        "all-16-distorted letter) are arms the battery already runs; the "
                        "pre-screen adds 0 forwards")

# ---------------------------------------------------------------- PC thresholds (frozen)
PC1_BAR = 0.35            # acc(no_frames) on the control set must be <= this
PC2_GAIN = 0.15           # acc(uniform@16) - acc(all-16 sigma=0.2 distorted) must be >= this
PC2_SIGN_P = 0.05         # and the one-sided sign test must be below this
PC3_MEDIAN_BAR = 0.15     # median G2 TV on the control set must be >= this
PC_CRASH_RULE = ("PC1, PC2 or PC3 not holding => that cell is a CRASH condition (fix the "
                 "design/pipeline), never an effect")
# the design's other frozen numbers, quoted so the evidence can show them next to the reading
FROZEN = {"G2_MEDIAN_TV_BAR": G.G2_MEDIAN_TV_BAR, "G3_SD_MAX": G.G3_SD_MAX,
          "W_GAIN": G.W_GAIN, "M_LOSS": G.M_LOSS, "F_BAND": G.F_BAND,
          "COUNT_AXIS_BAR": G.COUNT_AXIS_BAR, "SIGMA_MAIN": G.SIGMA_MAIN,
          "K_CAND": G.K_CAND, "B_MAIN": G.B_MAIN, "PS_FIXED_SEED": PS_FIXED_SEED,
          "S_PRE_STREAM": S_PRE_STREAM}
QUANTILES = [10, 25, 50, 75, 90]


def pre_distort(frame, q_index, frame_index):
    """The pre-screen distortion of ONE frame: sigma=0.2, the FIXED stream, the same
    (q_index, frame_index, sigma_index) key shape the run-seed arms use, so the tensors are
    rebuilt from the same rule."""
    return G.distort_seeded(frame, G.SIGMA_MAIN, S_PRE_STREAM, q_index, frame_index,
                            G.SIGMA_INDEX)


def pre_distort_pils(arr, q_index, max_pixels=G.MAX_PIXELS):
    return [C.to_pil(pre_distort(arr[i], q_index, i)[None, :, :, :], max_pixels)[0]
            for i in range(len(arr))]


# ---------------------------------------------------------------- instrument statistics

def tv_stats(tvs, bar=None):
    """Median / mean / count over the bar / quantiles of a per-question TV list.

    The independent unit is the QUESTION (one TV per question); it is stated in the return
    value so no caller can report it without the unit.
    """
    bar = G.G2_MEDIAN_TV_BAR if bar is None else bar
    v = np.asarray([float(x) for x in tvs if x is not None], float)
    n = int(len(v))
    out = {"n_units": n, "unit": "question (one G2 TV per question)",
           "median_tv": None, "mean_tv": None, "count_ge_bar": None, "bar": float(bar)}
    if n:
        q = np.percentile(v, QUANTILES)
        out.update({"median_tv": float(np.median(v)), "mean_tv": float(v.mean()),
                    "count_ge_bar": int((v >= bar).sum()),
                    "quantiles": {("p%d" % p): float(x) for p, x in zip(QUANTILES, q)},
                    "min": float(v.min()), "max": float(v.max()), "sd": float(v.std(ddof=1)) if n > 1 else None})
    return out


def prescreen_block(records):
    """Retention counts and the selection-bias reads, per cell, from the battery records.

    `records` are the per-question dicts written by the runner: they must carry
    `letters['no_frames']`, `letters['uniform@16']`, `letters['sigma_calib@0.2']`,
    `gold`, `g2_tv`, `task` and `_qid`.
    """
    keep, drop = [], []
    why = {"no_frames_only": [], "distorted_only": [], "both": []}
    for r in records:
        gold = r["gold"]
        bad_no = r["letters"].get("no_frames") != gold
        bad_dist = r["letters"].get("sigma_calib@0.2") != gold
        if bad_no and bad_dist:
            keep.append(r)
        else:
            drop.append(r)
            key = "both" if (not bad_no and not bad_dist) else ("no_frames_only" if not bad_no else "distorted_only")
            why[key].append(r["_qid"])
    n = len(records)

    def acc(rs, arm):
        ok = [1.0 if r["letters"].get(arm) == r["gold"] else 0.0 for r in rs]
        return (float(np.mean(ok)) if ok else None), len(ok)

    a_keep_u16 = acc(keep, "uniform@16")
    a_drop_u16 = acc(drop, "uniform@16")
    out = {
        "rule": PS_RULE, "n_questions": n, "n_retained": len(keep),
        "n_dropped": len(drop),
        "retention_rate": (len(keep) / float(n)) if n else None,
        "dropped_because_no_frames_answered_correctly": len(why["no_frames_only"]),
        "dropped_because_distortion_left_it_correct": len(why["distorted_only"]),
        "dropped_because_both": len(why["both"]),
        "dropped_qids": {"no_frames_only": why["no_frames_only"],
                         "distorted_only": why["distorted_only"], "both": why["both"]},
        "retained_qids": [r["_qid"] for r in keep],
        "g2_retained": tv_stats([r["g2_tv"] for r in keep]),
        "g2_dropped": tv_stats([r["g2_tv"] for r in drop]),
        "acc_uniform16_retained": a_keep_u16[0], "n_retained_units": a_keep_u16[1],
        "acc_uniform16_dropped": a_drop_u16[0], "n_dropped_units": a_drop_u16[1],
        "acc_uniform16_all": acc(records, "uniform@16")[0],
        "selection_bias": [
            "the rule is conditioned on the cell's backbone, so the retained set is "
            "cell- and backbone-specific and must never be pooled across cells",
            "the rule systematically drops questions answerable from the language prior "
            "(no_frames correct) and questions whose answer survives destroying all frames, "
            "i.e. it drops exactly the questions this diagnosis is about; the retained set is "
            "not a representative sample of the carrier and cannot be used to infer carrier "
            "difficulty",
            "the rule merges 'the answer does not need frames' with 'the frames are still "
            "usable after distortion'; sigma=0.4 and greying are descriptive companions only "
            "and never take part in the decision",
            "the pre-screen uses greedy decoding (one model state, no sampling noise); with "
            "the frames pinned and the fixed pre-screen stream it is reproducible",
        ],
        "underpowered_fallback_pre_registered":
            "retained n < 20, or acc(uniform@16) on the retained set <= 0.15 or >= 0.90 => the "
            "cell's frame-choice axis is underpowered / not decidable (checked before any "
            "delta is read); only instrument readings are reported",
    }
    rn, un = out["n_retained"], out["acc_uniform16_retained"]
    out["fallback_triggered"] = bool(rn < 20 or un is None or un <= 0.15 or un >= 0.90)
    return out


# ---------------------------------------------------------------- positive controls

def _acc(records, arm):
    ok = [1.0 if r["letters"].get(arm) == r["gold"] else 0.0 for r in records]
    return (float(np.mean(ok)), len(ok)) if ok else (None, 0)


def pc_block(records, label="carrier 3 B layer control set"):
    """PC1 / PC2 / PC3, each with the threshold written in the Plan before the run.

    PC1  acc(no_frames) <= 0.35
    PC2  acc(uniform@16) - acc(all-16 sigma=0.2 distorted) >= 0.15 AND one-sided sign test
         (correct-on-true / wrong-on-distorted vs the reverse) p < 0.05
    PC3  median G2 TV >= 0.15 on the control set, with the count over the bar reported
    Any one of them not holding => crash condition for that cell, not an effect.
    """
    import scipy.stats as st
    a_no, n = _acc(records, "no_frames")
    a_u16, _ = _acc(records, "uniform@16")
    a_d, _ = _acc(records, "sigma_calib@0.2")
    win = loss = 0
    for r in records:
        ok_u = r["letters"].get("uniform@16") == r["gold"]
        ok_d = r["letters"].get("sigma_calib@0.2") == r["gold"]
        if ok_u and not ok_d:
            win += 1
        elif ok_d and not ok_u:
            loss += 1
    sign_p = (float(st.binomtest(win, win + loss, 0.5, alternative="greater").pvalue)
              if (win + loss) else 1.0)
    gain = (a_u16 - a_d) if (a_u16 is not None and a_d is not None) else None
    g2 = tv_stats([r["g2_tv"] for r in records])
    pc1 = {"control": "PC1 no-frames floor", "set": label, "n_units": n, "unit": "question",
           "measured_acc_no_frames": a_no, "threshold": PC1_BAR, "rule": "acc <= %.2f" % PC1_BAR,
           "holds": (a_no is not None and a_no <= PC1_BAR),
           "crash_if_not": "acc >= 0.50 => the control set itself is mis-constructed"}
    pc2 = {"control": "PC2 destroying all 16 frames costs accuracy", "set": label,
           "n_units": n, "unit": "question (paired)", "acc_uniform16": a_u16,
           "acc_all16_sigma0.2_distorted": a_d, "gain": gain,
           "sign_test": {"n_win": win, "n_loss": loss, "p_one_sided_greater": sign_p},
           "threshold": {"gain >= %.2f" % PC2_GAIN: (gain is not None and gain >= PC2_GAIN),
                         "sign p < %.2f" % PC2_SIGN_P: sign_p < PC2_SIGN_P},
           "holds": bool(gain is not None and gain >= PC2_GAIN and sign_p < PC2_SIGN_P),
           "crash_if_not": "the control set is not known to need frames"}
    pc3 = {"control": "PC3 the control set's own instrument response", "set": label,
           "threshold_median": PC3_MEDIAN_BAR, "median_tv": g2["median_tv"], "mean_tv": g2["mean_tv"],
           "count_ge_bar": g2["count_ge_bar"], "quantiles": g2.get("quantiles"),
           "n_units": g2["n_units"], "unit": g2["unit"],
           "holds": bool(g2["median_tv"] is not None and g2["median_tv"] >= PC3_MEDIAN_BAR),
           "crash_if_not": "the instrument does not move on samples known to need frames"}
    return {"PC1": pc1, "PC2": pc2, "PC3": pc3,
            "all_hold": bool(pc1["holds"] and pc2["holds"] and pc3["holds"]),
            "rule": PC_CRASH_RULE}


# ---------------------------------------------------------------- instrument subset

def stratified_take(questions, per_family, tag):
    """Family-stratified, pre-registered instrument subset: for each family, questions
    sorted by qid and permuted with numpy.random.default_rng(<fixed constant>); the union is
    returned in the pinned question order.  Fixed before any model output of this batch and
    independent of the pre-screen result."""
    fams = sorted(set(q.get("task") or "all" for q in questions))
    take = {}
    for fam in fams:
        pool = sorted([q for q in questions if (q.get("task") or "all") == fam],
                      key=lambda q: q["_qid"])
        want = int(per_family.get(fam, 0))
        k = int(hashlib.sha256(("%s|%s" % (tag, fam)).encode()).hexdigest()[:6], 16)
        rng = np.random.default_rng(k)
        perm = rng.permutation(len(pool)).tolist() if want < len(pool) else list(range(len(pool)))
        take[fam] = set(id(pool[i]) for i in perm[:want])
    return [q for q in questions if id(q) in take.get(q.get("task") or "all", set())]


def family_mix(questions):
    out = {}
    for q in questions:
        f = q.get("task") or "all"
        out[f] = out.get(f, 0) + 1
    return out


# ---------------------------------------------------------------- identity

def manifest_of(root):
    """The sandbox's own listing command, reproduced in-run:
    cd <root> && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum"""
    rows = []
    for base, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(base, f)
            rel = "./" + os.path.relpath(p, root).replace(os.sep, "/")
            rows.append((rel.encode(), rel, p))
    rows.sort(key=lambda t: t[0])
    lines = ["%s  %s" % (G.sha256_file(p), rel) for _b, rel, p in rows]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(), len(lines)


def verify_identity(pins, roots, limit_gb=None):
    """Recompute the per-file sha256 manifest of every root this run reads and compare with
    the repo-pinned value.  A mismatch is a hard failure (non-zero exit)."""
    out = {}
    for name, root in roots.items():
        want = (pins.get("dirs") or {}).get(name)
        if not os.path.isdir(root):
            raise SystemExit("identity: %s is not a directory: %s" % (name, root))
        if want is None:
            raise SystemExit("identity: no pinned manifest for %s" % name)
        got, n = manifest_of(root)
        ok = (got == want["manifest_sha256"]) and (n == int(want["n_files"]))
        out[name] = {"path": root, "manifest_sha256_recomputed": got,
                     "manifest_sha256_pinned": want["manifest_sha256"],
                     "n_files_recomputed": n, "n_files_pinned": int(want["n_files"]),
                     "match": bool(ok)}
        if not ok:
            raise SystemExit("identity mismatch for %s: %s (%d files) != pinned %s (%d files)"
                             % (name, got, n, want["manifest_sha256"], int(want["n_files"])))
    return out


# ---------------------------------------------------------------- carrier 3 B layer

B_ANNOTATION_FILES = ("1_plotQA", "2_needle")


def load_c3b_questions(c3b_dir, pin_path, n_q=None):
    """Load the pinned carrier-3 B-layer control set (needle + plotQA) and verify its
    identity in-run: published dir, annotation file sha256+bytes, every video's bytes and
    sha256, item order, question-set fingerprint and the directory manifest."""
    pin = json.load(open(pin_path))
    pub = pin["published_dir"]
    if os.path.realpath(c3b_dir) != os.path.realpath(pub):
        raise SystemExit("carrier 3 B dir %s is not the pinned published dir %s" % (c3b_dir, pub))
    ann, ann_sha = {}, {}
    for task in sorted(pin["annotation_files"]):
        p = os.path.join(c3b_dir, "json", task + ".json")
        got = G.sha256_file(p)
        ann_sha[task] = got
        meta = pin["annotation_files"][task]
        if got != meta["sha256"] or os.path.getsize(p) != int(meta["bytes"]):
            raise SystemExit("carrier 3 B annotation mismatch for %s" % p)
        ann[task] = json.load(open(p))
    by_video = {}
    for task, rows in ann.items():
        for r in rows:
            by_video.setdefault((task, r["video"]), []).append(r)
    qs, lines, skipped = [], [], []
    for i, it in enumerate(pin["items"]):
        task, vid = it["task"], it["video"]
        rows = by_video.get((task, vid)) or []
        if not rows:
            raise SystemExit("carrier 3 B: %s/%s has no annotation record" % (task, vid))
        # The documented selection rule takes the FIRST annotation row for the video; the
        # pinned question_set_sha256 is computed from exactly those rows, so picking any other
        # row changes the fingerprint and the run exits.
        r = rows[0]
        n_rows_for_video = len(rows)
        cands = list(r["candidates"])
        if r["answer"] not in cands:
            raise SystemExit("carrier 3 B: gold %r not among candidates for %s" % (r["answer"], vid))
        vp = os.path.join(c3b_dir, "videos", task, vid)
        if not os.path.exists(vp):
            raise SystemExit("carrier 3 B: missing video %s" % vp)
        if os.path.getsize(vp) != int(it["bytes"]):
            raise SystemExit("carrier 3 B: video size mismatch for %s" % vp)
        got = G.sha256_file(vp)
        if got != it["sha256"]:
            raise SystemExit("carrier 3 B: video sha256 mismatch for %s" % vp)
        lines.append("%s\t%s\t%s\t%s\t%s" % (task, vid, r["question"], "|".join(cands), r["answer"]))
        qs.append({"_qid": "%s/%s" % (task, vid), "_idx": i, "task": task, "video_id": vid,
                   "_video_path": vp, "question": r["question"], "candidates": cands,
                   "correct_letter": C.LETTERS[cands.index(r["answer"])],
                   "question_type": r.get("question_type"), "duration_s": r.get("duration"),
                   "n_annotation_rows_for_video": n_rows_for_video,
                   "video_bytes": int(it["bytes"])})
    qsha = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    if qsha != pin["question_set_sha256"]:
        raise SystemExit("carrier 3 B question-set fingerprint mismatch: %s != %s"
                         % (qsha, pin["question_set_sha256"]))
    man_sha, n_files = manifest_of(c3b_dir)
    if man_sha != pin["manifest_sha256"]:
        raise SystemExit("carrier 3 B directory manifest mismatch: %s != %s"
                         % (man_sha, pin["manifest_sha256"]))
    fp = {"published_dir": pub, "source": pin.get("source"), "provider": pin.get("provider"),
          "identity_strength": pin.get("identity_strength"),
          "mirror_revision_observed": pin.get("mirror_revision_observed"),
          "manifest_sha256_recomputed": man_sha, "manifest_sha256_pin": pin["manifest_sha256"],
          "n_files": n_files, "annotation_files": ann_sha,
          "annotation_files_pin": {k: v["sha256"] for k, v in pin["annotation_files"].items()},
          "annotation_platform_record_comparison": pin.get("annotation_platform_record_comparison"),
          "n_items_pinned": len(pin["items"]),
          "task_mix": {t: sum(1 for it in pin["items"] if it["task"] == t)
                       for t in sorted(set(it["task"] for it in pin["items"]))},
          "bytes_total": int(sum(it["bytes"] for it in pin["items"])),
          "licence_note": pin.get("licence_note"), "selection_rule": pin.get("selection_rule"),
          "skipped_items": skipped}
    if n_q:
        qs = qs[:int(n_q)]
    return qs, qsha, fp


# ---------------------------------------------------------------- carriers 1/2/3A

def load_lvb(lvb_dir, pinned_jsonl, n_q=None):
    return G.load_lvb_questions(lvb_dir, pinned_jsonl, n_q=n_q)


def load_lvbench(lvbench_dir, chosen_path, pin_path, n_q=None):
    qs, _d = G.load_lvbench_questions(chosen_path, lvbench_dir, pin_path)
    if n_q and n_q < len(qs):
        qs = qs[:int(n_q)]
    return qs


def load_c3a(mlvu_dir, pin_path, n_q=None):
    qs, qsha, fp = MX.load_mlvu_questions(mlvu_dir, pin_path, n_q=n_q)
    return qs, qsha, fp


# ---------------------------------------------------------------- selection arms

def random_draws(qi, s_sel, k=G.K_CAND, b=G.B_MAIN, draws=3):
    out = []
    for j in range(draws):
        rng = np.random.default_rng(np.random.SeedSequence([int(s_sel), int(qi), 101 + j]))
        out.append(sorted(int(i) for i in rng.permutation(k)[:b]))
    return out


def shuffled_pick(scores, qi, s_sel, b=G.B_MAIN):
    sc = np.asarray(scores, float)
    rng = np.random.default_rng(np.random.SeedSequence([int(s_sel), int(qi), 202]))
    perm = rng.permutation(len(sc))
    return G.top_b(sc[perm], b)


def mean_letters(letters_by_seed):
    """Accuracy of a multi-seed arm: per-question mean over seeds (the question is the unit)."""
    if not letters_by_seed:
        return []
    return [float(np.mean(values)) for values in zip(*letters_by_seed)]
