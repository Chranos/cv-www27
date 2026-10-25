"""r4 shared code (Plan plan-matrix-longvideo-r4@v2, stage=smoke).

Everything the r4 design adds on top of r3, in one module:

  * the destruction operator D = frames_replaced@16 (setup 1.1): the 16 fed slots are
    replaced by ANOTHER pinned video's frames at the same pool slots; the donor is chosen by
    the pinned deterministic rule (pin_index + 7 + 3k) mod n_videos -- no RNG at draw time;
  * the pixel verification D1/D2/D3, run BEFORE any forward, per question and per cell;
  * the D branch of setup 3.4: retry with donor offset +1 up to 3 times (order pinned), then
    exclude the question with reason replacement_not_clean; more than 4 visible questions
    excluded => crash_destruction_unverified and the cell produces NO control readings;
  * G2 on D (the diagnostic main reading, unit = question), G2_runseed on the D donor
    offsets k=1/2/3, and tv_sigma_0.2 / tv_sigma_0.4 kept as DESCRIPTIVE readings of the old
    noise operator -- they never enter a gate;
  * PC2a and PC3 on D; PC1 and PC2b unchanged (they do not use D);
  * compare_block with resolved keys, the r4 missing-value semantics (mean and CI null,
    readable = false; never a fallback to 0.0 and never a fallback to the first arm) and
    S-cmp over every comparison;
  * the S-cmp negative test: a frozen copy of the r3 resolver applied to the r3 archive
    task-f1539f3d67 must reproduce the known 21/21 missing pattern, while the current
    resolver on the same records must not flag.

Nothing in this module hard-codes a reading.
"""
import collections
import hashlib
import json
import os
import time

import numpy as np

import mvp_common as C
import gen_common as G
import mx_common as MX
import r2_common as R
import r3_common as R3
import r4_arms as A4

PLAN_ID = "plan-matrix-longvideo-r4"
PLAN_VERSION = 2
STAGE = "smoke"
RUN_PURPOSES = ("control", "pipeline_smoke", "debug")

# ---- destruction operator D (setup 1.1) -------------------------------------------------
D_DONOR_BASE = 7
D_DONOR_STEP = 3
D_MAX_ATTEMPTS = 4                  # the initial draw + the 3 pinned retries (+1 each)
D_MAX_VISIBLE_EXCLUDED = 4          # more than this => crash_destruction_unverified
D_RUNSEED_DONOR_KS = (1, 2, 3)      # G2_runseed's three streams (k=0 is the main reading)

SIGMA_04_INDEX = G.SIGMA_LIST.index(0.4)

# ---- stage-1 mechanical exclusions (r3 evidence task-907d017857) ------------------------
STAGE1_EXCLUDED_QIDS = ("1_plotQA/movie101_38.mp4#visible",
                        "1_plotQA/movie101_38.mp4#placebo",
                        "1_plotQA/movie101_79.mp4#placebo")
STAGE1_ROWS_SHA = "a521b67e77e9930d9e639f40c46beb75f020ce1bb8c658d4324650726f78c5b9"

R3_ARCHIVE_SAMPLES = "sources/task-f1539f3d67/results/r3_c3b_qwen_samples.jsonl"


def log(*a):
    print(*a, flush=True)


def sha256_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)


def progress_append(path, obj):
    """Append one JSON line and fsync: a dropped device loses at most the question in flight."""
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ================================================================ the sealed pin
def video_path_of(pair, c3b_dir):
    return os.path.join(c3b_dir, "videos", pair["task"], pair["video"])


def load_pin_and_check(c3b_dir, pin_path="data/carrier3b_pin_r3.json",
                       pin_sha_out="data/carrier3b_pin_r3.sha256",
                       c3b_pin="data/carrier3b_pin.json", gen_mod=None):
    """Regenerate the sealed 24-pair pin in-run and verify it; never rewrite it."""
    import r3_gen_pin as PG
    pin_c3b, items, by_fam = PG.load_pinned_items(c3b_pin)
    qset = PG.check_pinned_question_sets(pin_c3b, items, by_fam)
    sel = R3.select_videos(items)
    videos = [(fam, it) for fam in R3.R3_PAIR_FAMILY_ORDER for it in sel[fam]]
    if len(videos) != R3.R3_N_PAIRS:
        raise SystemExit("the generator selected %d videos, expected %d"
                         % (len(videos), R3.R3_N_PAIRS))
    meta = PG.container_metadata(videos, c3b_dir=c3b_dir)
    pin = R3.build_pin(items, n_frames={k: v["n_frames"] for k, v in meta.items()},
                       fps={k: v["fps"] for k, v in meta.items()})
    sha, nbytes = R3.pin_sha256_and_bytes(pin)
    check = {"regenerated_pin_sha256": sha, "regenerated_pin_bytes": nbytes,
             "regenerated_correct_letter_counts": pin["correct_letter_counts"],
             "qset_check": qset, "written_by_this_run": False}
    if os.path.exists(pin_path):
        b = open(pin_path, "rb").read()
        check["committed_pin_bytes"] = len(b)
        check["committed_pin_file_sha256"] = hashlib.sha256(b).hexdigest()
        check["committed_pin_byte_identical"] = bool(b == R3.canonical_json(pin))
        if not check["committed_pin_byte_identical"]:
            raise SystemExit("committed %s is present but NOT byte-identical to the regenerated "
                             "pin" % pin_path)
    if os.path.exists(pin_sha_out):
        want = open(pin_sha_out).read().strip().split()[0]
        check["committed_sha256_value"] = want
        check["sha_matches_regenerated"] = bool(want == sha)
        if want != sha:
            raise SystemExit("regenerated pin sha256 %s != committed %s" % (sha, want))
    return pin, sha, check


def select_pairs_stratified(pin_pairs, n_pairs):
    """Pair selection for a run: proportional by family, systematic (evenly spread) inside each
    family's pin order.

    n_pairs >= len(pin_pairs) returns every pair in pin order, so the full 24-pair stage keeps
    exactly the selection it had before.  Below 24 the selection is NOT the pin-order prefix:
    the pilot's units are spread over both families and over the whole index range.  Pure
    arithmetic, no RNG, a deterministic function of (pin, n_pairs).
    """
    pairs = list(pin_pairs)
    n = int(n_pairs)
    if n <= 0 or n >= len(pairs):
        return pairs, {"rule": "all pairs (identity selection)", "n_pairs": len(pairs),
                       "pair_indices": [int(p["pair"]) for p in pairs],
                       "by_family": {f: sum(1 for p in pairs if p["family"] == f)
                                     for f in dict.fromkeys(p["family"] for p in pairs)},
                       "is_prefix_of_pin_order": True}
    fams = list(dict.fromkeys(p["family"] for p in pairs))
    by = {f: [p for p in pairs if p["family"] == f] for f in fams}
    prop = {f: n * len(by[f]) / float(len(pairs)) for f in fams}
    take = {f: int(np.floor(prop[f])) for f in fams}
    if n >= len(fams):
        for f in fams:
            take[f] = max(1, take[f])
    order = sorted(fams, key=lambda f: (-(prop[f] - np.floor(prop[f])), fams.index(f)))
    while sum(take.values()) < n:
        moved = False
        for f in order:
            if sum(take.values()) >= n:
                break
            if take[f] < len(by[f]):
                take[f] += 1
                moved = True
        if not moved:
            break
    while sum(take.values()) > n:
        moved = False
        for f in reversed(order):
            if sum(take.values()) <= n:
                break
            if take[f] > 1:
                take[f] -= 1
                moved = True
        if not moved:
            break
    chosen, info = [], {}
    for f in fams:
        pool, m = by[f], take[f]
        pos = sorted({int(np.floor((j + 0.5) * len(pool) / float(m))) for j in range(m)})
        chosen.extend(pool[i] for i in pos)
        info[f] = {"n": len(pos), "positions_in_family_pin_order": pos,
                   "pair_indices": [int(pool[i]["pair"]) for i in pos]}
    chosen.sort(key=lambda p: int(p["pair"]))
    return chosen, {"rule": ("proportional by family, systematic (evenly spread) inside each "
                             "family; NOT a prefix of the pin order"),
                    "n_pairs": len(chosen),
                    "pair_indices": [int(p["pair"]) for p in chosen],
                    "by_family": info,
                    "is_prefix_of_pin_order": bool(
                        [int(p["pair"]) for p in chosen]
                        == [int(p["pair"]) for p in pairs[:len(chosen)]])}


def build_questions(pin, n_pairs, c3b_dir):
    """The control-set questions, built from the pin (question text and the four options are
    pinned arithmetic, not MLVU annotations).

    Pair selection: r4_common.select_pairs_stratified (identity at n_pairs = 24).  Callers that
    need the selection record for their Evidence recompute it with the same pure function.
    """
    pairs, _sel = select_pairs_stratified(pin["pairs"], int(n_pairs))
    qs = []
    for pair in pairs:
        vp = video_path_of(pair, c3b_dir)
        if not os.path.exists(vp):
            raise SystemExit("control-set video missing: %s" % vp)
        if os.path.getsize(vp) != int(pair["bytes"]):
            raise SystemExit("control-set video size mismatch: %s" % vp)
        for role in ("visible", "placebo"):
            qs.append({"_qid": (pair["visible_qid"] if role == "visible" else pair["placebo_qid"]),
                       "_idx": len(qs), "task": pair["task"], "video_id": pair["video"],
                       "_video_path": vp, "question": pair["question_text"],
                       "candidates": list(pair["options"]),
                       "correct_letter": pair["correct_letter"],
                       "question_type": "r4_control_set_marker", "duration_s": None,
                       "video_bytes": int(pair["bytes"]),
                       "_r4_pair": int(pair["pair"]), "_r4_role": role,
                       "_r4_family": pair["family"],
                       "_r4_i_star": int(pair["i_star"]), "_r4_j_star": pair["j_star"],
                       "_r4_spec": pair["marker"],
                       "_frames_key": (vp, role, int(pair["pair"]))})
    return qs


# ================================================================ the arrays the arms consume
class PoolCache(object):
    """LRU decode cache + the independent second decode used by D2."""

    def __init__(self, cap=2):
        self.cap = int(cap)
        self.arr = collections.OrderedDict()
        self.ref = {}

    def get(self, path):
        if path in self.arr:
            self.arr.move_to_end(path)
            return self.arr[path]
        while len(self.arr) >= self.cap:
            self.arr.popitem(last=False)
        arr, fps, nfr = C.decode_frames(path, G.K_CAND)
        e = {"arr": arr, "fps": float(fps), "n_frames": int(nfr)}
        self.arr[path] = e
        return e

    def ref_slot_sha(self, path):
        """sha256 of the donor video's 16 pool-slot frames from an INDEPENDENT second decode."""
        if path not in self.ref:
            arr2, _f, _n = C.decode_frames(path, G.K_CAND)
            u16 = C.uniform_indices(G.K_CAND, G.B_MAIN)
            self.ref[path] = [sha256_arr(arr2[i]) for i in u16]
        return self.ref[path]


def fed_array(q, cache):
    """The array the arms of THIS question consume: the marker drawn at i* for a visible
    question, the untouched clean decode for its placebo twin (the r3 construction)."""
    own = cache.get(q["_video_path"])["arr"]
    if q.get("_r4_role") == "visible":
        spec, i = q["_r4_spec"], int(q["_r4_i_star"])
        out = own.copy()
        out[i], _g = R3.apply_marker(own[i], spec,
                                     R3.marker_geometry(spec, own.shape[1], own.shape[2]))
        return out
    return own


def donor_slot_array(q, pin, c3b_dir, k, offset, cache):
    """The 16 donor frames for one question and one donor offset."""
    pair_index = int(q["_r4_pair"])
    n_videos = len(pin["pairs"])
    di = int((pair_index + D_DONOR_BASE + D_DONOR_STEP * int(k) + int(offset)) % n_videos)
    dp = pin["pairs"][di]
    path = video_path_of(dp, c3b_dir)
    u16 = C.uniform_indices(G.K_CAND, G.B_MAIN)
    arr = cache.get(path)["arr"]
    return arr[u16], di, path, u16


# ================================================================ D1/D2/D3
def d_attempt(q, pin, c3b_dir, k, offset, cache):
    """One donor draw: D1 (marker gone), D2 (identity), D3 (the record)."""
    own_fed = fed_array(q, cache)
    donor_fed, di, donor_path, u16 = donor_slot_array(q, pin, c3b_dir, k, offset, cache)
    spec = q["_r4_spec"]
    totals = [0, 0, 0, 0]
    frames_bbox_out = 0
    for f in donor_fed:
        g = R3.marker_geometry(spec, f.shape[0], f.shape[1])
        if not g["in_bounds"]:
            frames_bbox_out += 1
            continue
        cc = R3.combo_counts(f, spec, geom=g)
        totals = [t + max(0, int(c)) for t, c in zip(totals, cc)]
    slot_sha = [sha256_arr(f) for f in donor_fed]
    ref_sha = cache.ref_slot_sha(donor_path)
    n_match_donor = int(sum(1 for a, b in zip(slot_sha, ref_sha) if a == b))
    own_slots = own_fed[u16]
    n_equal_own = int(sum(1 for a, b in zip(donor_fed, own_slots) if np.array_equal(a, b)))
    d1_pass = bool(all(t == 0 for t in totals))
    d2_pass = bool(n_match_donor == len(u16) and n_equal_own <= 1)
    dp = pin["pairs"][di]
    return {"qid": q["_qid"], "pair": int(q["_r4_pair"]), "role": q["_r4_role"],
            "donor_k": int(k), "donor_offset": int(offset), "donor_pair_index": int(di),
            "donor_task": dp["task"], "donor_video": dp["video"],
            "own_video": q["video_id"],
            "D1": {"option_match_pixel_totals": totals, "all_zero": d1_pass,
                   "frames_bbox_out_of_bounds": int(frames_bbox_out),
                   "tol_linf": R3.COLOUR_TOL_LINF, "n_frames": int(len(donor_fed))},
            "D2": {"slot_sha256": slot_sha, "n_slots_matching_donor_redecode": n_match_donor,
                   "n_slots": int(len(u16)), "n_slots_equal_to_own_fed": n_equal_own,
                   "passes": d2_pass},
            "D3": {"donor_pair_index": int(di), "option_match_pixel_totals": totals,
                   "slot_sha256": slot_sha, "n_slots_equal_to_own_fed": n_equal_own,
                   "donor_spec": {"colour_key": dp["marker"]["colour_key"],
                                  "shape_key": dp["marker"]["shape_key"]}},
            "d1_pass": d1_pass, "d2_pass": d2_pass,
            "pass": bool(d1_pass and d2_pass)}


def d_precheck(questions, pin, c3b_dir, k=0, cache=None):
    """D1/D2/D3 for every question, BEFORE any forward, with the pinned retry order."""
    cache = cache or PoolCache(cap=2)
    rows = []
    for q in questions:
        attempts = []
        for off in range(D_MAX_ATTEMPTS):
            a = d_attempt(q, pin, c3b_dir, k, off, cache)
            attempts.append(a)
            if a["pass"]:
                break
        acc = attempts[-1] if attempts[-1]["pass"] else None
        rows.append({"qid": q["_qid"], "pair": int(q["_r4_pair"]), "role": q["_r4_role"],
                     "video": q["video_id"], "family": q["_r4_family"],
                     "i_star": int(q["_r4_i_star"]),
                     "attempts": attempts, "n_attempts": len(attempts),
                     "accepted_offset": (int(acc["donor_offset"]) if acc else None),
                     "donor_pair_index": (int(acc["donor_pair_index"]) if acc else None),
                     "donor_video": (acc["donor_video"] if acc else None),
                     "d1_pass": bool(acc and acc["d1_pass"]),
                     "d2_pass": bool(acc and acc["d2_pass"]),
                     "pass": bool(acc is not None),
                     "excluded_replacement_not_clean": bool(acc is None)})
        log("   D %-42s %-7s pass=%s attempts=%d donor=%s"
            % (q["_qid"], q["_r4_role"], rows[-1]["pass"], rows[-1]["n_attempts"],
               rows[-1]["donor_video"]))
    return rows


def aggregate_d(rows, n_pairs):
    vis = [r for r in rows if r["role"] == "visible"]
    plc = [r for r in rows if r["role"] == "placebo"]
    failed_vis = [r["qid"] for r in vis if not r["pass"]]
    failed_plc = [r["qid"] for r in plc if not r["pass"]]
    out = {"n_questions": len(rows), "n_visible": len(vis), "n_placebo": len(plc),
           "n_pairs_built": int(n_pairs),
           "n_pass": int(sum(1 for r in rows if r["pass"])),
           "n_fail": int(sum(1 for r in rows if not r["pass"])),
           "failed_qids": [r["qid"] for r in rows if not r["pass"]],
           "d1_pass_count": int(sum(1 for r in rows if r["d1_pass"])),
           "d2_pass_count": int(sum(1 for r in rows if r["d2_pass"])),
           "n_retried": int(sum(1 for r in rows if r["n_attempts"] > 1)),
           "retry_histogram": {str(n): int(sum(1 for r in rows if r["n_attempts"] == n))
                               for n in range(1, D_MAX_ATTEMPTS + 1)},
           "n_visible_excluded_replacement_not_clean": len(failed_vis),
           "n_placebo_excluded_replacement_not_clean": len(failed_plc),
           "visible_excluded_qids": failed_vis,
           "placebo_excluded_qids": failed_plc,
           "max_visible_excluded_allowed": D_MAX_VISIBLE_EXCLUDED,
           "holds": bool(len(failed_vis) <= D_MAX_VISIBLE_EXCLUDED),
           "rule": ("P-D: D1 (all four option combos match 0 pixels in all 16 fed frames) and "
                    "D2 (16/16 slots equal the donor's independently re-decoded frames, <=1/16 "
                    "equal to this video's own fed slots) on every in-force question; a question "
                    "that still fails after donor offsets +1,+2,+3 is excluded with reason "
                    "replacement_not_clean; more than 4 visible questions excluded => "
                    "crash_destruction_unverified and the cell produces no control readings")}
    out["verdict"] = ("ok" if out["holds"] else "crash_destruction_unverified")
    return out


# ================================================================ model-side D readings
def g2_d_block(model, proc, dev, ids, q, pin, c3b_dir, prompt, fwd, p_original, cache,
               accepted_offset):
    """WITHDRAWN.  Operator D and every reading built on it were withdrawn in r7; r8 removed
    the computation from the execution path, so this function cannot produce a value at all.
    Archived g2_d rows stay readable as data, never through a constructor or this function."""
    raise A4.WithdrawnArmError(
        "g2_d_block is withdrawn: operator D was withdrawn in r7 and removed from the "
        "execution path in r8; no G2 reading under that operator may be produced")


def g2_runseed_d_block(model, proc, dev, ids, q, pin, c3b_dir, prompt, fwd, seeds,
                       p_original, cache):
    """WITHDRAWN, same as g2_d_block: no reader may produce p_donor_runseed_k* again."""
    raise A4.WithdrawnArmError(
        "g2_runseed_d_block is withdrawn: the donor streams were withdrawn in r7 and removed "
        "from the execution path in r8; no p_donor_runseed_k* reading may be produced")


def g2_runseed_summary(per_question, n_seeds=3):
    ks = [str(k) for k in range(int(n_seeds))]
    by_seed = {}
    for k in ks:
        vals = [q["per_seed"][k]["tv_vs_original_u16"] for q in per_question
                if q.get("per_seed") and k in q["per_seed"]]
        by_seed[k] = {"median_tv": (float(np.median(vals)) if vals else None),
                      "mean_tv": (float(np.mean(vals)) if vals else None),
                      "count_ge_bar": int(sum(1 for v in vals if v >= G.G2_MEDIAN_TV_BAR)),
                      "n_units": len(vals), "per_question_tv": vals,
                      "donor_offset_k": int(D_RUNSEED_DONOR_KS[int(k)])
                      if int(k) < len(D_RUNSEED_DONOR_KS) else None}
    meds = [v["median_tv"] for v in by_seed.values() if v["median_tv"] is not None]
    return {"per_run_seed_stream": by_seed, "n_streams": len(ks),
            "cross_stream_range_of_medians": (float(max(meds) - min(meds)) if meds else None),
            "unit": "question",
            "note": ("the same D operator with the three run-internal donor streams; reported "
                     "beside the gate reading, never pooled with it, and NOT a seed-robustness "
                     "reading (the pilot has one run seed)")}


def tv_sigma_block(model, proc, dev, ids, arr, u16, prompt, q_index, fwd, sigma, sigma_index,
                   s_stream, p_original):
    pils = [C.to_pil(G.distort_seeded(arr[i], sigma, s_stream, q_index, i, sigma_index)[None,
                                                                                       :, :, :],
                     G.MAX_PIXELS)[0] for i in u16]
    p = G.option_probs_multi(model, proc, pils, prompt, dev, ids, counter=fwd,
                             kind="multi:tv_sigma_%.1f" % float(sigma))
    return {"sigma": float(sigma), "sigma_index": int(sigma_index),
            "tv_vs_original_u16": G.tv(p_original, p),
            "p_distorted_u16": [round(float(x), 6) for x in np.asarray(p).ravel()],
            "role": "descriptive old-operator reading; never enters a gate"}


# ================================================================ comparisons and S-cmp
def spec_keys(spec):
    """The exact keys an arm spec reads, in order.  Never built by string concatenation at the
    call site: every key comes from r4_arms.arm_key (explicit `keys` are themselves produced
    by that constructor)."""
    if spec.get("keys"):
        return [str(k) for k in spec["keys"]]
    return [A4.arm_key(spec["kind"], k=spec.get("k"), j=spec.get("j"))]


def spec_agg_key(spec):
    keys = spec_keys(spec)
    if len(keys) == 1:
        return keys[0]
    return "agg:%s" % spec.get("kind", "arm")


def spec_letter(rec, spec):
    return (rec.get("letters") or {}).get(spec_keys(spec)[0])


def spec_correct(rec, spec):
    """Per-question correctness of an arm spec: one key as-is, several keys averaged."""
    keys = spec_keys(spec)
    vals = []
    for key in keys:
        l = (rec.get("letters") or {}).get(key)
        if l is not None:
            vals.append(1.0 if l == rec["gold"] else 0.0)
    if not vals:
        return None
    if len(keys) == 1:
        return vals[0]
    return float(np.mean(vals))


def accuracy_per_arm(recs, keys, agg_specs=()):
    out = {}
    for key in keys:
        vals = [None if (r.get("letters") or {}).get(key) is None
                else (1.0 if r["letters"][key] == r["gold"] else 0.0) for r in recs]
        ok = [v for v in vals if v is not None]
        out[key] = {"arm_key": key, "n_units": len(ok), "unit": "question",
                    "accuracy": (float(np.mean(ok)) if ok else None), "per_question": vals}
    for spec in agg_specs:
        vals = [spec_correct(r, spec) for r in recs]
        ok = [v for v in vals if v is not None]
        out[spec_agg_key(spec)] = {
            "arm_key": spec_agg_key(spec), "n_units": len(ok), "unit": "question",
            "accuracy": (float(np.mean(ok)) if ok else None), "per_question": vals,
            "aggregation": ("per-question mean over the run-seed streams %r"
                            % (spec.get("seeds"),))}
    return out


def compare_block(label, recs, spec_a, spec_b, seeds, s_boot, descriptive=False):
    a = [spec_correct(r, spec_a) for r in recs]
    b = [spec_correct(r, spec_b) for r in recs]
    miss_a = [i for i, x in enumerate(a) if x is None]
    miss_b = [i for i, x in enumerate(b) if x is None]
    readable = bool(recs) and not miss_a and not miss_b
    blk = {"label": label, "n_units": len(recs), "unit": "question",
           "arm_first": spec_a.get("kind"), "arm_second": spec_b.get("kind"),
           "resolved_key_first": spec_agg_key(spec_a),
           "resolved_key_second": spec_agg_key(spec_b),
           "resolved_keys_first": spec_keys(spec_a), "resolved_keys_second": spec_keys(spec_b),
           "n_missing_first": len(miss_a), "n_missing_second": len(miss_b),
           "readable": readable, "descriptive_only": bool(descriptive),
           "missing_semantics": ("n_missing > 0 => mean and both CIs are null and readable is "
                                 "false; never 0.0, never a fallback to the first arm")}
    if readable:
        aa = np.asarray(a, float)
        bb = np.asarray(b, float)
        blk.update(G.compare(label, aa, bb, [r["video_id"] for r in recs], s_boot, label=label,
                             n_units_note=MX.resolution_note(len(recs))))
        blk["accuracy_first"] = float(aa.mean())
        blk["accuracy_second"] = float(bb.mean())
    else:
        blk.update({"mean": None, "ci95_question": [None, None],
                    "ci95_video_cluster": [None, None],
                    "accuracy_first": (None if not any(x is not None for x in a)
                                       else float(np.mean([x for x in a if x is not None]))),
                    "accuracy_second": (None if not any(x is not None for x in b)
                                        else float(np.mean([x for x in b if x is not None]))),
                    "why_unreadable": ("missing per-question values: first %d, second %d"
                                       % (len(miss_a), len(miss_b)))})
    pq = []
    single_a = len(spec_keys(spec_a)) == 1
    single_b = len(spec_keys(spec_b)) == 1
    for i, r in enumerate(recs):
        pq.append({"qid": r["_qid"], "video_id": r["video_id"], "gold": r["gold"],
                   "first": a[i], "second": b[i],
                   "delta": (None if (a[i] is None or b[i] is None) else a[i] - b[i]),
                   "first_letter": (spec_letter(r, spec_a) if single_a else None),
                   "second_letter": (spec_letter(r, spec_b) if single_b else None)})
    blk["per_question_values"] = pq
    blk["accuracy_on_this_set"] = {
        "first": (None if not any(x is not None for x in a)
                  else float(np.mean([x for x in a if x is not None]))),
        "second": (None if not any(x is not None for x in b)
                   else float(np.mean([x for x in b if x is not None]))),
        "note": ("the per-arm accuracies recomputed on THIS comparison's own units; for the "
                 "full in-force comparisons this set is the same set accuracy_per_arm uses"),
        "unit_qids": [r["_qid"] for r in recs]}
    return blk


def s_cmp(comparisons, recs, acc, required_labels):
    """S-cmp: every comparison must be readable and agree with the per-arm accuracies.

    (1) n_missing_second == 0; (2) accuracy_second == accuracy_per_arm[resolved_key_second]
    float-exactly; (3) accuracy_first == accuracy_per_arm[resolved_key_first] float-exactly;
    (4) every per-question second letter equals the record's letter under the resolved key;
    (5) resolved keys are written out.  An empty group (n_units == 0) is recorded as
    not_applicable_empty_group, never as a pass of the comparison itself.
    """
    by_qid = {r["_qid"]: r for r in recs}
    rows, missing_labels = {}, []
    for label in required_labels:
        blk = comparisons.get(label)
        if blk is None:
            rows[label] = {"status": "not_produced", "passes": False}
            missing_labels.append(label)
            continue
        if int(blk.get("n_units") or 0) == 0:
            rows[label] = {"status": "not_applicable_empty_group", "passes": True, "n_units": 0,
                           "note": "the group has no questions in this run; nothing to assert"}
            continue
        onset = blk.get("accuracy_on_this_set") or {}
        full_set = (len(recs) == int(blk.get("n_units") or -1))
        e2 = acc.get(blk["resolved_key_second"]) or {}
        e1 = acc.get(blk["resolved_key_first"]) or {}
        c = {"n_missing_second_zero": blk["n_missing_second"] == 0,
             "n_missing_first_zero": blk["n_missing_first"] == 0,
             "readable": bool(blk.get("readable")),
             "accuracy_second_exact": (blk.get("accuracy_second") is not None
                                       and onset.get("second") is not None
                                       and float(blk["accuracy_second"]) == float(onset["second"])),
             "accuracy_first_exact": (blk.get("accuracy_first") is not None
                                      and onset.get("first") is not None
                                      and float(blk["accuracy_first"]) == float(onset["first"])),
             "block_covers_the_whole_in_force_set": bool(full_set),
             "accuracy_second": blk.get("accuracy_second"),
             "accuracy_on_this_set_second": onset.get("second"),
             "accuracy_first": blk.get("accuracy_first"),
             "accuracy_on_this_set_first": onset.get("first")}
        if full_set:
            c["accuracy_second_equals_global_per_arm"] = (
                blk.get("accuracy_second") is not None and e2.get("accuracy") is not None
                and float(blk["accuracy_second"]) == float(e2["accuracy"]))
            c["accuracy_first_equals_global_per_arm"] = (
                blk.get("accuracy_first") is not None and e1.get("accuracy") is not None
                and float(blk["accuracy_first"]) == float(e1["accuracy"]))
        else:
            c["accuracy_second_equals_global_per_arm"] = None
            c["accuracy_first_equals_global_per_arm"] = None
            c["subgroup_note"] = ("this block is a subgroup comparison (%d of %d in-force "
                                  "questions): the design's 'equals accuracy_per_arm' is applied "
                                  "to accuracy on the block's own units, and the global per-arm "
                                  "accuracy is not expected to equal it"
                                  % (int(blk.get("n_units") or 0), len(recs)))
        c["resolved_keys_written"] = bool(blk.get("resolved_key_first")
                                          and blk.get("resolved_key_second"))
        c["resolved_key_first"] = blk.get("resolved_key_first")
        c["resolved_key_second"] = blk.get("resolved_key_second")
        bad = []
        for pq in blk["per_question_values"]:
            rec = by_qid.get(pq["qid"])
            if rec is None:
                bad.append(pq["qid"])
                continue
            if len(blk["resolved_keys_second"]) == 1:
                want = (rec.get("letters") or {}).get(blk["resolved_key_second"])
                if want != pq.get("second_letter"):
                    bad.append(pq["qid"])
        c["per_question_second_letters_equal"] = (not bad)
        c["per_question_mismatch_qids"] = bad
        c["status"] = "checked"
        c["passes"] = bool(all(c[k] for k in ("n_missing_second_zero", "n_missing_first_zero",
                                              "readable", "accuracy_second_exact",
                                              "accuracy_first_exact", "resolved_keys_written",
                                              "per_question_second_letters_equal"))
                           and c.get("accuracy_second_equals_global_per_arm") is not False
                           and c.get("accuracy_first_equals_global_per_arm") is not False)
        rows[label] = c
    ok = all(v.get("passes") for v in rows.values()) and not missing_labels
    return {"self_check": "S-cmp", "n_comparisons_required": len(required_labels),
            "n_comparisons_checked": int(sum(1 for v in rows.values()
                                             if v.get("status") == "checked")),
            "n_not_applicable_empty": int(sum(1 for v in rows.values()
                                              if v.get("status") == "not_applicable_empty_group")),
            "missing_labels": missing_labels, "per_comparison": rows, "passes": bool(ok),
            "rule": ("every required comparison must be readable, have n_missing_second == 0, "
                     "match the per-arm accuracy recomputed on ITS OWN units exactly on both "
                     "arms (identical to accuracy_per_arm for a full in-force comparison, and "
                     "recomputed on the subgroup for a pre-screen group comparison) and match "
                     "every per-question second letter; otherwise the run exits non-zero after "
                     "writing its outputs")}


# ================================================================ prescreen (r4, D operator)
def prescreen_block_r4(records):
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
            key = ("both" if (not bad_no and not bad_dist)
                   else ("no_frames_only" if not bad_no else "distorted_only"))
            why[key].append(r["_qid"])

    def acc(rs, arm):
        ok = [1.0 if r["letters"].get(arm) == r["gold"] else 0.0 for r in rs]
        return (float(np.mean(ok)) if ok else None), len(ok)

    out = {"rule": R.PS_RULE, "n_questions": len(records), "n_retained": len(keep),
           "n_dropped": len(drop),
           "retention_rate": (len(keep) / float(len(records)) if records else None),
           "dropped_because_no_frames_answered_correctly": len(why["no_frames_only"]),
           "dropped_because_distortion_left_it_correct": len(why["distorted_only"]),
           "dropped_because_both": len(why["both"]),
           "dropped_qids": why, "retained_qids": [r["_qid"] for r in keep],
           "g2_retained": R.tv_stats([r.get("g2_tv") for r in keep]),
           "g2_dropped": R.tv_stats([r.get("g2_tv") for r in drop]),
           "tv_sigma_0.2_retained": R.tv_stats([r.get("tv_sigma_0.2") for r in keep]),
           "tv_sigma_0.2_dropped": R.tv_stats([r.get("tv_sigma_0.2") for r in drop]),
           "acc_uniform16_retained": acc(keep, "uniform@16")[0],
           "n_retained_units": len(keep),
           "acc_uniform16_dropped": acc(drop, "uniform@16")[0],
           "n_dropped_units": len(drop),
           "g2_operator": "D (frames_replaced@16); the sigma=0.2 companion is beside it",
           "inference_rule_prewritten": ("inference on the stratum only when n >= 20; below that, "
                                         "description only"),
           "inference_allowed": bool(len(keep) >= 20),
           "selection_bias": [
               "the rule is conditioned on the cell's backbone; the retained set is cell- and "
               "backbone-specific and is never pooled across cells",
               "the rule drops questions answerable from the language prior and questions whose "
               "answer survives the sigma=0.2 distortion; it is not a representative sample",
               "it merges 'the answer does not need frames' with 'the frames are still usable "
               "after sigma=0.2 distortion'; sigma=0.4 and the D operator are companions only",
               "greedy decoding, one model state; with the frames pinned and the fixed "
               "pre-screen stream it is reproducible"]}
    return out


# ================================================================ S-cmp negative test
def _old_arm_key(kind, k=None, j=None):
    """FROZEN copy of the r3 resolver (commit 32106add), used only by the negative test."""
    if k is None and j is None:
        return str(kind)
    if j is None:
        return "%s_k%d" % (kind, int(k))
    if k is None:
        return "%s_j%d" % (kind, int(j))
    return "%s_k%d_j%d" % (kind, int(k), int(j))


def _old_correct(rec, kind, k=None, j=None):
    l = (rec.get("letters") or {}).get(_old_arm_key(kind, k=k, j=j))
    return None if l is None else (1.0 if l == rec["gold"] else 0.0)


def _cur_correct(rec, kind, k=None, j=None):
    l = (rec.get("letters") or {}).get(A4.arm_key(kind, k=k, j=j))
    return None if l is None else (1.0 if l == rec["gold"] else 0.0)


def scmp_negative_test(archive_path, excluded_qids, n_seeds=3):
    """Apply the frozen r3 resolver to the r3 archive: it must reproduce 21/21 missing.

    The same records under the r4 resolver must NOT be flagged.  This is what makes the
    self-check demonstrably able to go red.
    """
    if not os.path.exists(archive_path):
        return {"executed": False, "reason": "archive not mounted at %s" % archive_path,
                "passes": False}
    recs = [json.loads(l) for l in open(archive_path) if l.strip()]
    excl = set(excluded_qids or ())
    recs = [r for r in recs if r["_qid"] not in excl]
    if not recs:
        return {"executed": False, "reason": "archive has no in-force records", "passes": False}
    keep = [r for r in recs
            if r["letters"].get("no_frames") != r["gold"]
            and r["letters"].get("sigma_calib@0.2") != r["gold"]]
    drop = [r for r in recs if r not in keep]
    comps = [("main: ours16 - uniform@16", recs, "ours16", "uniform@16")]
    for kind in ("ours_quota16", "nq16", "mis16", "shuffled16", "clip_top16", "its16",
                 "maxvol16", "no_frames", "sigma_calib@0.2", "random@16"):
        comps.append(("%s - uniform@16" % kind, recs, kind, "uniform@16"))
    comps.append(("ours16 - uniform@16 (pre-screen retained group)", keep, "ours16", "uniform@16"))
    comps.append(("ours16 - uniform@16 (pre-screen dropped group)", drop, "ours16", "uniform@16"))
    rows = []
    for label, rs, kind_a, kind_b in comps:
        # the r3 read path called arm_key(kind, k=k) for EVERY arm and EVERY run seed, flat
        # arms included -- that is exactly what produced "uniform@16_k0".  So the frozen
        # resolver is applied with k=0 to BOTH arms here.
        old_b = [_old_correct(r, kind_b, k=0) for r in rs]
        old_a = ([_old_correct(r, kind_a, k=0) for r in rs]
                 if kind_a != A4.RANDOM_KIND
                 else [_old_correct(r, A4.RANDOM_KIND, k=0, j=0) for r in rs])
        cur_b = [_cur_correct(r, kind_b, k=0) for r in rs]
        cur_a = ([_cur_correct(r, kind_a, k=0) for r in rs]
                 if kind_a != A4.RANDOM_KIND
                 else [_cur_correct(r, A4.RANDOM_KIND, k=0, j=0) for r in rs])
        rows.append({
            "comparison": label, "n_units": len(rs),
            "old_resolver": {"n_missing_second": int(sum(1 for x in old_b if x is None)),
                             "n_missing_first": int(sum(1 for x in old_a if x is None)),
                             "accuracy_second": (float(np.mean([0.0 if x is None else x
                                                                for x in old_b])) if old_b
                                                 else None)},
            "current_resolver": {"n_missing_second": int(sum(1 for x in cur_b if x is None)),
                                 "n_missing_first": int(sum(1 for x in cur_a if x is None)),
                                 "accuracy_second": (float(np.mean([x for x in cur_b
                                                                    if x is not None]))
                                                     if any(x is not None for x in cur_b) else None)}})
    n = len(recs)
    old_flagged = [r for r in rows
                   if r["old_resolver"]["n_missing_second"] == r["n_units"]
                   and r["old_resolver"]["accuracy_second"] == 0.0]
    cur_flagged = [r for r in rows if r["current_resolver"]["n_missing_second"] > 0]
    return {"executed": True, "old_commit": "32106addbb903bb132a1bfed851e0d4b031f0a4a",
            "archive": archive_path, "n_records": n,
            "n_comparisons": len(rows),
            "n_old_flagged": len(old_flagged),
            "n_current_flagged": len(cur_flagged),
            "pattern": "%d/%d" % (n, n),
            "per_comparison_missing_pattern": [
                {"comparison": r["comparison"], "n_units": r["n_units"],
                 "old_n_missing_second": r["old_resolver"]["n_missing_second"],
                 "old_accuracy_second": r["old_resolver"]["accuracy_second"],
                 "current_n_missing_second": r["current_resolver"]["n_missing_second"]}
                for r in rows],
            "old_every_comparison_all_units_missing": bool(len(old_flagged) == len(rows)),
            "current_resolver_flagged": bool(cur_flagged),
            "rows": rows,
            "passes": bool(len(old_flagged) == len(rows) and not cur_flagged),
            "design_phrase_note": (
                "the Plan writes 'all 21 comparisons report 21/21 missing'; read literally the "
                "count of comparisons in the family is %d (main + 10 secondary + 2 pre-screen "
                "groups) and 21 is the number of in-force questions. The assertion implemented "
                "here is the substantive one: EVERY comparison reports n_missing_second equal "
                "to ALL of its units (21/21 on the full set, 9/9 and 12/12 on the pre-screen "
                "groups) with accuracy_second = 0.0, while the r4 resolver reports 0 missing on "
                "the same records." % len(rows))}


# ================================================================ stage-1 table
def load_stage1_exclusions(path):
    if not os.path.exists(path):
        raise SystemExit("the stage-1 per-question visibility table is not mounted at %s"
                         % path)
    got = G.sha256_file(path)
    if STAGE1_ROWS_SHA and got != STAGE1_ROWS_SHA:
        raise SystemExit("stage-1 visibility table sha256 %s != the recorded %s"
                         % (got, STAGE1_ROWS_SHA))
    rows = [json.loads(l) for l in open(path) if l.strip()]
    excl = [{"_qid": r["_qid"], "pair": r["pair"], "role": r["role"], "video": r["video"],
             "reason": r["exclusion_reason"]} for r in rows if r.get("excluded_mechanical")]
    return {"path": path, "sha256": got, "n_rows": len(rows),
            "exclusion_list": excl,
            "excluded_qids": sorted(r["_qid"] for r in rows if r.get("excluded_mechanical")),
            "rule": ("applied MECHANICALLY: an excluded question is dropped from every "
                     "positive-control, instrument and comparison reading of this run")}
