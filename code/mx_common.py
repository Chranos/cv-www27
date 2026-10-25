"""Shared helpers for the fifth project batch (Plan plan-matrix-longvideo-r1@v1).

Carriers
  1 = LongVideoBench val subset   (LVB, pinned 40 questions, subtitle channel)
  2 = LVBench subset              (pinned 24 questions, no subtitle channel)
  3 = MLVU subset                 (pinned 40 questions via data/carrier3_pin.json, no subtitle
                                    channel: the MLVU mirror ships no subtitle track)

Everything the fourth batch already built is imported and reused verbatim, not forked:
  * gen_common.run_seeds / seed_streams      -- the project's one seed-derivation function
  * gen_common.Fwd                           -- the one forward accumulator
  * gen_common.prompt_for / top_b / quota_select / tv / sim_matrix / selection_metrics / overlap
  * gen_common.compare / boot_ci / boot_ci_cluster / choice_axis_verdict / count_axis_read
  * gen_common.load_lvb_questions / load_lvbench_questions / question_set_sha / sha256_file
  * gen_common.load_vlm / model_fingerprint / option_probs / option_probs_multi / answer_arm
  * gen_common.distort_seeded / SIGMA_MAIN / SIGMA_LIST / SIGMA_INDEX
  * gen_common.parse_subtitles / subtitle_gaps / subtitle_text_all
  * mvp_common.decode_frames / to_pil / uniform_indices / letter_token_ids / parse_letter
  * mvp_common.mcnemar_exact / holm / data_fingerprint / distort

New in this batch (this file): the MLVU carrier loader with its pin verification, the
re-implemented BOLT ITS baseline, the re-implemented MaxInfo-style maximum-volume baseline,
the CLIP embedding helpers they share, the published-directory manifest check, and the
before/after comparison against the Plan's expected_results intervals.
"""
import hashlib
import json
import os

import numpy as np

import mvp_common as C
import gen_common as G

# ---------------------------------------------------------------- pre-registered constants
# Same quantity, same bar, same operating point as the third and fourth batches.
G2_MEDIAN_TV_BAR = G.G2_MEDIAN_TV_BAR          # 0.15
ITS_ALPHA = 2.5                                # BOLT's alpha for LongVideoBench (kg:paper:W4413146798 Table 8)
SIGMA_CALIB_LIST = [0.2, 0.4]                  # the two sigma calibration points this batch runs
MAXVOL_RIDGE = 1e-8                            # numerical ridge for the greedy log-det inverse

# The 6 cells of the matrix: (cell id -> carrier, backbone, n, subtitles, P4 archive check)
CELLS = {
    "pilot": {"pairs": [("mlvu", "qwen2vl"), ("mlvu", "llavaov")], "n": 4,
              "subtitles": False, "p4": False},
    "m1": {"pairs": [("lvb", "qwen2vl")], "n": 40, "subtitles": True, "p4": True},
    "m2": {"pairs": [("lvb", "llavaov")], "n": 40, "subtitles": True, "p4": False},
    "m3": {"pairs": [("lvbench", "qwen2vl")], "n": 24, "subtitles": False, "p4": False},
    "m4": {"pairs": [("lvbench", "llavaov")], "n": 24, "subtitles": False, "p4": False},
    "m5": {"pairs": [("mlvu", "qwen2vl")], "n": 40, "subtitles": False, "p4": False},
    "m6": {"pairs": [("mlvu", "llavaov")], "n": 40, "subtitles": False, "p4": False},
}

CARRIER_NAMES = {"lvb": "LongVideoBench val subset (carrier 1)",
                 "lvbench": "LVBench subset (carrier 2)",
                 "mlvu": "MLVU subset (carrier 3)"}
BACKBONE_NAMES = {"qwen2vl": "Qwen2-VL-7B-Instruct (backbone 1)",
                  "llavaov": "LLaVA-OneVision-7B-Qwen2-ov HF copy (backbone 2)"}

# Expected intervals written down BEFORE any run of this batch (Plan expected_results).
EXPECT_G2_MEDIAN = {"lvb": [0.02, 0.13], "lvbench": [0.02, 0.15], "mlvu": [0.02, 0.15]}
EXPECT_NOFRAMES = {"lvb": [0.30, 0.62], "lvbench": [0.10, 0.45], "mlvu": [0.15, 0.40]}
EXPECT_PRIMARY = [-0.20, +0.10]
EXPECT_ITS = [-0.10, +0.15]
EXPECT_MAXVOL = [-0.10, +0.15]
EXPECT_CLIPTOP = [-0.10, +0.10]
EXPECT_SUBON = [-0.20, +0.10]
EXPECT_SUBONLY = [-0.35, +0.05]
EXPECT_B16_B8 = [-0.15, +0.25]
EXPECT_B32_B16 = [-0.15, +0.08]
EXPECT_OVERLAP_QUERY = [0.4, 0.7]

# Carrier identity constants carried over from the fourth batch (same published directories).
EXPECT_LVB = {"lvb_val_json_sha256":
              "d32bf2e0cbbdb4fa2e99666a5a49b5a19bca2ceec69222c9a7d3aad567d1fa23",
              "n_val_records": 1337, "n_video_files": 56,
              "videos_manifest_sha256":
              "35dd905b810db2d5ee0f8bbee4dc22e5798ed7f4923f65ca8f420498be478ce5"}
PINNED_QSET_SHA = G.PINNED_QSET_SHA            # carrier 1, from the archive task-ca0a24e61b
LVBENCH_QSET_SHA = "8ac114901dadca4d045f400c690a411a9dd727e1ab8d9b7284ec0ca3e1781cdc"


# ---------------------------------------------------------------- provenance

def dir_manifest_sha(root):
    """sha256 of the same listing the device-side manifest uses:

        cd <root> && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum

    i.e. lines '<sha256>  ./<path>' in byte order of the './<path>' strings, joined by
    newline with a trailing newline.  Returns (sha256, n_files).
    """
    rows = []
    for base, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(base, f)
            rel = "./" + os.path.relpath(p, root).replace(os.sep, "/")
            rows.append((rel.encode(), rel, p))
    rows.sort(key=lambda t: t[0])
    lines = ["%s  %s" % (G.sha256_file(p), rel) for _b, rel, p in rows]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(), len(lines)


def lvb_fingerprint(lvb_dir):
    fp = C.data_fingerprint(lvb_dir)
    bad = [k for k, v in EXPECT_LVB.items() if fp.get(k) != v]
    if bad:
        raise SystemExit("carrier 1 identity mismatch: %s (got %r)" % (bad, fp))
    return fp


def lvbench_fingerprint(lvbench_dir, chosen_path):
    names = []
    for base, _dirs, files in os.walk(lvbench_dir):
        for f in files:
            if f.endswith(".mp4"):
                names.append((f, os.path.getsize(os.path.join(base, f))))
    names.sort()
    h = hashlib.sha256()
    for n_, s_ in names:
        h.update(("%s %d\n" % (n_, s_)).encode())
    ch = json.load(open(chosen_path))
    return {"n_video_files": len(names), "videos_name_size_manifest_sha256": h.hexdigest(),
            "chosen_list_sha256": G.sha256_file(chosen_path),
            "chosen_annotation_sha256": ch.get("annotation_sha256"),
            "chosen_selection_rule": ch.get("selection_rule"),
            "chosen_videos": sorted(set(q["video"] for q in ch.get("chosen", [])))}


# ---------------------------------------------------------------- carrier 3 (MLVU subset)

def load_mlvu_questions(mlvu_dir, pin_path, n_q=None):
    """Load the pinned MLVU subset and verify its identity in-run.

    The pin (data/carrier3_pin.json) fixes the published directory, the three annotation
    files (sha256 + bytes), every video (bytes + sha256), the item order and the
    question-set fingerprint; any mismatch exits.  The pin itself does not carry the
    question text, so the questions are read back from the published annotation files and
    the fingerprint is recomputed from them.
    """
    if not os.path.exists(pin_path):
        raise SystemExit("carrier 3 pin not found: %s" % pin_path)
    pin = json.load(open(pin_path))
    pub = pin.get("published_dir")
    if os.path.realpath(mlvu_dir) != os.path.realpath(pub or ""):
        raise SystemExit("carrier 3 dir %s is not the pinned published dir %s"
                         % (mlvu_dir, pub))
    ann = {}
    ann_sha = {}
    for task, meta in sorted(pin["annotation_files"].items()):
        p = os.path.join(mlvu_dir, "json", task + ".json")
        got = G.sha256_file(p)
        ann_sha[task] = got
        if got != meta["sha256"] or os.path.getsize(p) != int(meta["bytes"]):
            raise SystemExit("carrier 3 annotation mismatch for %s" % p)
        ann[task] = json.load(open(p))
    by_video = {}
    for task, rows in ann.items():
        for r in rows:
            by_video.setdefault((task, r["video"]), []).append(r)

    qs, lines, skipped = [], [], []
    for i, it in enumerate(pin["items"]):
        task, vid = it["task"], it["video"]
        rows = by_video.get((task, vid)) or []
        if len(rows) != 1:
            raise SystemExit("carrier 3: %s/%s has %d annotation records" % (task, vid, len(rows)))
        r = rows[0]
        cands = list(r["candidates"])
        if r["answer"] not in cands:
            raise SystemExit("carrier 3: gold %r not among candidates for %s" % (r["answer"], vid))
        vp = os.path.join(mlvu_dir, "videos", task, vid)
        if not os.path.exists(vp):
            raise SystemExit("carrier 3: missing video %s" % vp)
        if os.path.getsize(vp) != int(it["bytes"]):
            raise SystemExit("carrier 3: video size mismatch for %s" % vp)
        got = G.sha256_file(vp)
        if got != it["sha256"]:
            raise SystemExit("carrier 3: video sha256 mismatch for %s" % vp)
        lines.append("%s\t%s\t%s\t%s\t%s"
                     % (task, vid, r["question"], "|".join(cands), r["answer"]))
        qs.append({"_qid": "%s/%s" % (task, vid), "_idx": i, "task": task, "video_id": vid,
                   "_video_path": vp, "question": r["question"], "candidates": cands,
                   "correct_letter": C.LETTERS[cands.index(r["answer"])],
                   "question_type": r.get("question_type"), "duration_s": r.get("duration"),
                   "video_bytes": int(it["bytes"])})
    qsha = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    if qsha != pin["question_set_sha256"]:
        raise SystemExit("carrier 3 question-set fingerprint mismatch: %s != %s"
                         % (qsha, pin["question_set_sha256"]))
    man_sha, n_files = dir_manifest_sha(mlvu_dir)
    if man_sha != pin["manifest_sha256"]:
        raise SystemExit("carrier 3 directory manifest mismatch: %s != %s"
                         % (man_sha, pin["manifest_sha256"]))
    fp = {"published_dir": pub, "repo": pin.get("repo"), "provider": pin.get("provider"),
          "mirror_commit": pin.get("mirror_commit"),
          "manifest_sha256_recomputed": man_sha, "manifest_sha256_pin": pin["manifest_sha256"],
          "n_files": n_files, "annotation_files": ann_sha,
          "annotation_files_pin": {k: v["sha256"] for k, v in pin["annotation_files"].items()},
          "n_items_pinned": len(pin["items"]),
          "task_mix": {t: sum(1 for it in pin["items"] if it["task"] == t)
                       for t in sorted(set(it["task"] for it in pin["items"]))},
          "bytes_total": int(sum(it["bytes"] for it in pin["items"])),
          "licence_note": pin.get("licence_note"),
          "selection_rule": pin.get("selection_rule"),
          "selection_result": pin.get("selection_result"),
          "skipped_items": skipped}
    if n_q:
        qs = qs[:int(n_q)]
    return qs, qsha, fp


# ---------------------------------------------------------------- CLIP helpers (B1 / B2 / clip_top16)

def load_clip(clip_dir, dev):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    cproc = CLIPProcessor.from_pretrained(clip_dir)
    cmodel = CLIPModel.from_pretrained(clip_dir, torch_dtype=torch.float32)
    cmodel = cmodel.to("cuda:%d" % dev) if dev >= 0 else cmodel.to("cpu")
    cmodel.eval()
    return cmodel, cproc


def clip_frame_features(cmodel, cproc, pils, dev):
    import torch
    feats = []
    with torch.inference_mode():
        for i in range(0, len(pils), 16):
            inp = cproc(images=pils[i:i + 16], return_tensors="pt")
            inp = {k: (v.to("cuda:%d" % dev) if dev >= 0 else v) for k, v in inp.items()}
            feats.append(cmodel.get_image_features(**inp).float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def clip_text_feature(cmodel, cproc, text, dev):
    import torch
    with torch.inference_mode():
        inp = cproc(text=[text], return_tensors="pt", truncation=True, max_length=77,
                    padding=True)
        inp = {k: (v.to("cuda:%d" % dev) if dev >= 0 else v) for k, v in inp.items()}
        f = cmodel.get_text_features(**inp)
    return f.float().cpu().numpy()[0]


def cos_sims(frame_feats, text_feat):
    fn = np.asarray(frame_feats, dtype=np.float64)
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    tn = np.asarray(text_feat, dtype=np.float64)
    tn = tn / max(float(np.linalg.norm(tn)), 1e-12)
    return fn @ tn


def its_select(sims, b, alpha=ITS_ALPHA):
    """BOLT's inverse transform sampling, RE-IMPLEMENTED (not the official code).

    kg:paper:W4413146798 Eq.(5)(6)(7): the per-frame relevance score is the CLIP
    text-frame cosine similarity (Eq. 5); the score is min-max normalised and raised to
    alpha (Eq. 6); the normalised scores form a distribution over the candidate frames
    (Eq. 7); frames are then obtained by inverting that distribution's CDF.

    Deviations, recorded in the evidence: (i) CLIP ViT-B/32 instead of the paper's
    CLIP-L/14 or SigLIP; (ii) the candidate pool is this batch's same 32-frame pool rather
    than every frame at 1 FPS of the whole video; (iii) alpha is FIXED at 2.5 for every
    cell (no per-cell tuning); (iv) the CDF is inverted on the fixed quantile grid
    u_j = (j+0.5)/B so the arm is deterministic -- the paper's sampling noise is not
    reproduced; (v) duplicates are dropped and the remaining slots filled with the
    next-highest probability frames, an invented rule (only the paper's text was
    available, its duplicate handling is not reproduced); (vi) no official code was used.
    """
    s = np.asarray(sims, dtype=np.float64)
    k = len(s)
    b = min(int(b), k)
    lo, hi = float(s.min()), float(s.max())
    w = np.zeros(k) if hi <= lo else ((s - lo) / (hi - lo)) ** float(alpha)
    tot = float(w.sum())
    p = np.full(k, 1.0 / k) if tot <= 0 else w / tot
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    picks = []
    for j in range(b):
        u = (j + 0.5) / float(b)
        i = int(np.searchsorted(cdf, u, side="left"))
        i = min(max(i, 0), k - 1)
        if i not in picks:
            picks.append(i)
    if len(picks) < b:
        for i in sorted(range(k), key=lambda t: (-p[t], t)):
            if len(picks) >= b:
                break
            if i not in picks:
                picks.append(i)
    return sorted(int(i) for i in picks[:b])


def maxvol_select(feats, b, ridge=MAXVOL_RIDGE):
    """MaxInfo-style greedy maximum-volume (log-det) frame selection, RE-IMPLEMENTED.

    kg:paper:W4407230825: frames are picked so that the volume spanned by their embeddings
    is as large as possible, i.e. log det of the Gram matrix G = X_sel X_sel^T is maximised.
    Adding a frame x multiplies the determinant by the Schur complement
        x^T x - (X_sel x)^T G^-1 (X_sel x),
    i.e. by the squared norm of x after projecting it onto the span of the frames already
    selected, so the greedy rule is argmax_i of that quantity.  Query-blind.

    Deviations, recorded in the evidence: (i) only the maximum-volume greedy described in
    the paper's abstract is implemented -- its Fast/Slow/Chunk variants are not in the
    available text; (ii) embeddings are the raw (unnormalised) CLIP ViT-B/32 image
    features, not the VLLM's own visual features; (iii) a ridge of
    1e-8 * trace(G)/k is added to G before inverting it (invented here for numerical
    stability), and because the greedy starts from the empty set the first frame it picks is
    the largest-norm one -- the abstract states no initialisation; (iv) no official code was
    used.
    """
    X = np.asarray(feats, dtype=np.float64)
    k = X.shape[0]
    b = min(int(b), k)
    norms2 = np.einsum("ia,ia->i", X, X)
    sel = []
    while len(sel) < b:
        if sel:
            S = X @ X[sel].T              # (k, n): inner product of each candidate with each
            Gs = S[sel]                   # selected frame; G = X_sel X_sel^T = S[sel]
            Gs = Gs + ridge * max(float(np.trace(Gs)) / len(sel), 1.0) * np.eye(len(sel))
            Ginv = np.linalg.inv(Gs)
            gains = norms2 - np.einsum("ia,ab,ib->i", S, Ginv, S)
        else:
            gains = norms2.copy()
        for i in sel:
            gains[i] = -np.inf
        sel.append(int(np.argmax(gains)))
    return sorted(int(i) for i in sel)


# ---------------------------------------------------------------- expectations bookkeeping

def dev_item(prediction, expected, measured, note=None, extra=None):
    """One line of the 'deviations from the pre-run quantitative expectations' block."""
    d = {"prediction": prediction, "expected_interval": expected,
         "measured": (None if measured is None else float(measured))}
    d["in_interval"] = (None if (measured is None or expected is None)
                        else bool(expected[0] <= float(measured) <= expected[1]))
    if note:
        d["note"] = note
    if extra:
        d.update(extra)
    return d


def boundary_note(n):
    one_q = 100.0 / n
    return ("thresholds are also stated as whole questions: n=%d, 1 question = %.2f pp, "
            "0.025 = %.2f q, 0.05 = %.2f q, 0.02 = %.2f q"
            % (n, one_q, 0.025 * n, 0.05 * n, 0.02 * n))


def resolution_note(n):
    if n >= 40:
        return ("40-question design: paired per-question CI half-width about +-0.12-0.15, so the "
                "smallest effect this matrix can resolve is about 0.15-0.17 (80% power, two-sided "
                "alpha=0.05).  F/I readings must be written as 'not separable / undetermined at "
                "this resolution', never as 'no effect'.")
    if n >= 24:
        return ("24-question design: paired per-question CI half-width about +-0.17-0.20, so the "
                "smallest effect this matrix can resolve is about 0.21-0.24 (80% power, two-sided "
                "alpha=0.05).  F/I readings must be written as 'not separable / undetermined at "
                "this resolution', never as 'no effect'.")
    return ("n=%d: this is the pilot, a throughput/plumbing run, not a design for effect readings; "
            "no resolution claim is made for it and none of its numbers are results." % n)
