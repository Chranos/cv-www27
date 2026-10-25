"""r5 twin-pair rebuild on LVBench real evidence frames (Plan plan-paper-full-twin-lvbench-r5).

Everything here is carrier code written for this Plan version:
  * the LVBench subset on the device is identified by a recomputed manifest (no download);
  * a pair is one video plus one question whose time_reference interval is <= 3 s;
  * the evidence frame j* is decoded at the interval midpoint and placed at pool slot e*,
    which is outside the uniform-16 budget I16u;
  * the unmarked twin is that pool (evidence present, marker absent); the marked twin is the
    same pool with a fixed, answer-neutral magenta border drawn on the e* frame only;
  * each operator (patchcopy, noise) carries its own precondition, including a behavioural
    destruction check on the real evidence frame and a weak probe that must fail to destroy.

No reading is hard-coded.  The pin is a pure function of the frozen data and the rules below.
"""
import hashlib
import json
import os
import re

import numpy as np

import gen_common as G
import mvp_common as C

DATA = os.environ.get("DATA_ROOT", "/path/to/data")
LVB_DIR = os.path.join(DATA, "lvbench-24209d43cbdd")
META = os.path.join(LVB_DIR, "video_info.meta.jsonl")
MODEL_DIR = os.path.join(DATA, "qwen2vl-7b-2cbada593a2b")

MANIFEST_SHA = "224e854da5ed0a527dd03dac9c42e384f2a11da194a9c29d6ef3678de405a184"
MANIFEST_N = 27
ANNOT_SHA = "a3042f766f568d457e2a91a30cdcf32a886c5421226eefa98a006ffaae038d8f"
MODEL_MANIFEST_SHA = "47cf90f85514f119a5f77c6e91a77b46d97fa9860a8fca81f678e916b246831c"

K = G.K_CAND                      # 32 candidate slots
B = G.B_MAIN                      # 16 fed frames
I16U = C.uniform_indices(K, B)    # {0,2,...,14,17,...,31}
I16U_SET = set(I16U)
SCREEN_N = 8
SPLIT_SEED = 11
MAX_INTERVAL_S = 3.0
MARK_FRAC = 0.06
PATCHCOPY_SIDE_FRAC = 0.40
PATCHCOPY_WEAK_FRAC = 0.10
PATCHCOPY_STREAM_KEY = 991
PATCHCOPY_OFFSETS = ((1, 0), (0, 1), (-1, 0), (0, -1),
                     (1, 1), (-1, -1), (1, -1), (-1, 1))
NOISE_SIGMA = G.SIGMA_MAIN
CHANCE_BAR = 0.35                 # 1.4 x chance, unchanged from r3/r4
READ_MIN_HITS = 7                 # of 8, unchanged

ROLES = ("marked", "unmarked")

# ---------------------------------------------------------------- identity


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def manifest_sha256(root=LVB_DIR):
    """The same listing the device-exploration manifest command produces:
    cd root && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum"""
    rels = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            rels.append(os.path.relpath(os.path.join(dirpath, f), root))
    rels.sort(key=lambda s: s.encode())
    h = hashlib.sha256()
    for rel in rels:
        p = os.path.join(root, rel)
        h.update(("%s  ./%s\n" % (sha256_file(p), rel)).encode())
    return h.hexdigest(), len(rels)


def identity_check():
    got, n = manifest_sha256()
    return {"dataset": "lvbench-24209d43cbdd", "path": LVB_DIR,
            "source": "AI-ModelScope/LVBench (ModelScope mirror)",
            "revision": "24209d43cbddb260fcb43dbdee2de0899c87163f",
            "license": "README cc-by-nc-sa-4.0 / LICENSE Apache-2.0 text with a merge marker; "
                       "non-commercial academic research only",
            "manifest_sha256_recomputed": got, "manifest_sha256_expected": MANIFEST_SHA,
            "n_files": n, "n_files_expected": MANIFEST_N,
            "manifest_matches": bool(got == MANIFEST_SHA and n == MANIFEST_N),
            "annotation_sha256": sha256_file(META), "annotation_sha256_expected": ANNOT_SHA,
            "annotation_matches": bool(sha256_file(META) == ANNOT_SHA),
            "model_dir": MODEL_DIR,
            "model_manifest_sha256": G.model_fingerprint(MODEL_DIR)["name_size_manifest_sha256"],
            "model_manifest_sha256_expected": MODEL_MANIFEST_SHA,
            "verified_by": "the run itself, not the sandbox (no data_dirs configured on this device)"}
    # note: model fingerprint is filled by the caller when it loads the model


def model_identity():
    fp = G.model_fingerprint(MODEL_DIR)
    return {"model_dir": MODEL_DIR, "model_fingerprint": fp,
            "model_manifest_sha256_expected": MODEL_MANIFEST_SHA,
            "model_manifest_matches": bool(
                fp["name_size_manifest_sha256"] == MODEL_MANIFEST_SHA)}

# ---------------------------------------------------------------- pin


def parse_time_reference(tr):
    m = re.match(r"^\s*(\d+):(\d+)\s*-\s*(\d+):(\d+)\s*$", tr or "")
    if not m:
        return None
    a = int(m.group(1)) * 60 + int(m.group(2))
    b = int(m.group(3)) * 60 + int(m.group(4))
    return a, b


def parse_options(question):
    parts = re.split(r"\(([A-D])\)\s*", question or "")
    if len(parts) != 9:
        return None
    stem = parts[0].strip()
    opts = {}
    for i in range(1, 9, 2):
        opts[parts[i]] = parts[i + 1].strip()
    if sorted(opts) != ["A", "B", "C", "D"] or any(not v for v in opts.values()):
        return None
    return stem, [opts[L] for L in "ABCD"]


def probe_video(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("cv2 cannot open %s" % path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if n <= 1:
        raise RuntimeError("no usable frame count for %s" % path)
    return fps, n


def decode_indices(path, idxs):
    """Decode exactly the given source frame indices (uint8 RGB)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("cv2 cannot open %s" % path)
    out = []
    try:
        for i in idxs:
            i = int(i)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok or fr is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, i - 1))
                ok, fr = cap.read()
            if not ok or fr is None:
                raise RuntimeError("cv2 failed to read frame %d of %s" % (i, path))
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.stack(out)


def evidence_slot(i_star):
    i = int(i_star)
    for d in range(1, K):
        for cand in (i + d, i - d):
            if 0 <= cand < K and cand not in I16U_SET:
                return int(cand)
    raise SystemExit("no slot outside I16u for i_star=%d" % i)


def nearest_slot_to_frame(j_star, idx):
    return int(min(range(len(idx)), key=lambda s: (abs(int(idx[s]) - int(j_star)), s)))


def build_pin():
    rows = [json.loads(l) for l in open(META) if l.strip()]
    by_key = {r["key"]: r for r in rows}
    vids = sorted(f[:-4] for f in os.listdir(LVB_DIR) if f.endswith(".mp4"))
    pairs, skipped = [], []
    for vid in vids:
        r = by_key.get(vid)
        if r is None:
            skipped.append({"video": vid, "why": "no annotation row"})
            continue
        dur = float(r["video_info"]["duration_minutes"]) * 60.0
        cand = []
        for q in r["qa"]:
            p = parse_time_reference(q.get("time_reference", ""))
            if p is None:
                continue
            a, b = p
            if b < a or (b - a) > MAX_INTERVAL_S or b > dur + 1.0:
                continue
            o = parse_options(q.get("question", ""))
            if o is None:
                continue
            if q.get("answer") not in list("ABCD"):
                continue
            cand.append((int(q["uid"]), a, b, q, o))
        if not cand:
            skipped.append({"video": vid, "why": "no eligible question"})
            continue
        cand.sort(key=lambda x: x[0])
        uid, a, b, q, (stem, opts) = cand[0]
        vp = os.path.join(LVB_DIR, vid + ".mp4")
        fps, n = probe_video(vp)
        t_ev = 0.5 * (a + b)
        j_star = int(min(max(round(t_ev * fps), 0), n - 1))
        idx = [int(min(max(round((i + 0.5) * n / K), 0), n - 1)) for i in range(K)]
        n_slots_equal_j = int(sum(1 for x in idx if x == j_star))
        # keep j* only at e*: any grid slot whose frame is j* is nudged by one frame
        idx_guarded = list(idx)
        for s in range(K):
            if idx_guarded[s] == j_star:
                idx_guarded[s] = int(min(max(idx[s] + (1 if idx[s] + 1 < n else -1), 0), n - 1))
        i_star = nearest_slot_to_frame(j_star, idx_guarded)
        e_star = evidence_slot(i_star)
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
            "video_sha256": sha256_file(vp),
        })
    pairs.sort(key=lambda p: p["video_id"])
    for i, p in enumerate(pairs):
        p["pair"] = i
    n = len(pairs)
    perm = np.random.default_rng(SPLIT_SEED).permutation(n).tolist()
    screen_ids = sorted(int(i) for i in perm[:SCREEN_N])
    report_ids = sorted(int(i) for i in perm[SCREEN_N:])
    pin = {
        "carrier": "lvbench-24209d43cbdd",
        "manifest_sha256": MANIFEST_SHA,
        "rule": {"max_interval_s": MAX_INTERVAL_S, "one_question_per_video": "smallest uid",
                 "pool": "32 uniform midpoints, round((i+0.5)*n/32)",
                 "I16u": I16U, "e_star_rule": "nearest slot to i* not in I16u, +d first",
                 "evidence_frame": "round(midpoint_seconds * fps)",
                 "mark": "magenta border, frac=%.2f of min(H,W)" % MARK_FRAC,
                 "split": "np.random.default_rng(%d).permutation(n)" % SPLIT_SEED},
        "n_pairs": n, "screen_pairs": screen_ids, "report_pairs": report_ids,
        "skipped_videos": skipped, "pairs": pairs,
    }
    pin["pin_sha256"] = hashlib.sha256(
        json.dumps(pin, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return pin


def check_pin(pin):
    """Structural facts that must hold before any model call."""
    rows = []
    for p in pin["pairs"]:
        rows.append({
            "pair": p["pair"], "video_id": p["video_id"], "j_star": p["j_star"],
            "i_star": p["i_star"], "e_star": p["e_star"],
            "e_star_in_I16u": bool(p["e_star"] in I16U_SET),
            "e_star_outside_I16u": bool(p["e_star"] not in I16U_SET),
            "e_star_distance_to_i_star": int(abs(p["e_star"] - p["i_star"])),
            "grid_slots_equal_j_star_after_guard": int(
                sum(1 for x in p["grid_idx_guarded"] if x == p["j_star"])),
            "n_grid_slots_equal_j_star_before_guard":
                int(p["n_grid_slots_equal_j_star_before_guard"]),
            "uid": p["uid"], "time_reference": p["time_reference"],
            "interval_s": p["interval_s"], "gold": p["gold"],
            "question": p["question"], "options": p["options"],
            "video_sha256": p["video_sha256"], "video_bytes": p["video_bytes"],
        })
    ok = all(r["e_star_outside_I16u"] and r["grid_slots_equal_j_star_after_guard"] == 0
             for r in rows)
    return {"pass": bool(ok), "n_pairs": len(rows), "rows": rows}


def pair_questions(pin, pair_ids, role):
    out = []
    for p in pin["pairs"]:
        if int(p["pair"]) not in set(int(x) for x in pair_ids):
            continue
        out.append({
            "_pair": int(p["pair"]), "_role": role, "_video_path": p["video_path"],
            "video_id": p["video_id"], "question": p["question"],
            "candidates": list(p["options"]), "correct_letter": p["gold"],
            "_j_star": int(p["j_star"]), "_e_star": int(p["e_star"]),
            "_idx": int(p["pair"]) * 2 + (0 if role == "marked" else 1),
        })
    return out

# ---------------------------------------------------------------- frames


class FrameCache:
    def __init__(self, cap=2):
        self.cap = int(cap)
        self.order = []
        self.d = {}

    def get(self, p):
        if p in self.d:
            return self.d[p]
        fps, n = probe_video(p)
        grid_idx = [int(min(max(round((i + 0.5) * n / K), 0), n - 1)) for i in range(K)]
        grid = decode_indices(p, grid_idx)
        self.d[p] = {"grid": grid, "fps": fps, "n": n, "grid_idx": grid_idx}
        self.order.append(p)
        while len(self.order) > self.cap:
            self.d.pop(self.order.pop(0), None)
        return self.d[p]


def evidence_frame(path, j_star):
    return decode_indices(path, [int(j_star)])[0]


def apply_mark(frame, frac=MARK_FRAC):
    H, W = frame.shape[0], frame.shape[1]
    t = max(2, int(round(frac * min(H, W))))
    out = frame.copy()
    out[:t, :, :] = np.array([255, 0, 255], dtype=frame.dtype)
    out[H - t:, :, :] = np.array([255, 0, 255], dtype=frame.dtype)
    out[:, :t, :] = np.array([255, 0, 255], dtype=frame.dtype)
    out[:, W - t:, :] = np.array([255, 0, 255], dtype=frame.dtype)
    interior_same = bool(np.array_equal(out[t:H - t, t:W - t], frame[t:H - t, t:W - t]))
    changed = int(np.any(out != frame, axis=2).sum())
    return out, {"thickness": int(t), "changed_pixels": changed,
                 "fraction_changed": float(changed) / float(H * W),
                 "interior_bit_identical": interior_same,
                 "frac": float(frac)}


def build_pool(cache, p, role):
    vpath = p.get("video_path") or p["_video_path"]
    rec = cache.get(vpath)
    pool = rec["grid"].copy()
    ev = evidence_frame(vpath, p["_j_star"])
    mark_info = None
    if role == "marked":
        ev, mark_info = apply_mark(ev)
    pool[int(p["_e_star"])] = ev
    return pool, ev, mark_info, rec

# ---------------------------------------------------------------- operators


def geom_of(H, W, frac=PATCHCOPY_SIDE_FRAC):
    side = max(1, int(round(frac * min(H, W))))
    y0 = (H - side) // 2
    x0 = (W - side) // 2
    return {"side": int(side), "y0": int(max(0, y0)), "x0": int(max(0, x0)),
            "y1": int(min(H, max(0, y0) + side)), "x1": int(min(W, max(0, x0) + side)),
            "frac": float(frac), "H": int(H), "W": int(W)}


def patchcopy_offsets(geom, s_noise, q_idx, slot):
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(s_noise) & 0x7FFFFFFF, int(q_idx), int(slot), PATCHCOPY_STREAM_KEY]))
    j = int(rng.integers(0, len(PATCHCOPY_OFFSETS)))
    dy, dx = PATCHCOPY_OFFSETS[j]
    return int(dy) * int(geom["side"]), int(dx) * int(geom["side"]), int(j)


def patchcopy_frame(frame, geom, dy, dx):
    out = frame.copy()
    donor = np.roll(np.roll(frame, int(dy), axis=0), int(dx), axis=1)
    ys = slice(geom["y0"], geom["y1"])
    xs = slice(geom["x0"], geom["x1"])
    out[ys, xs] = donor[ys, xs]
    return out


def operator_frame(form, frame, geom, k, q_idx, slot, s_noise, frac=None):
    if form == "patchcopy":
        g = geom if frac is None else geom_of(frame.shape[0], frame.shape[1], frac)
        dy, dx, j = patchcopy_offsets(g, s_noise, q_idx, slot)
        return patchcopy_frame(frame, g, dy, dx), {"offset": [dy, dx], "offset_index": j,
                                                   "geom": g}
    if form == "noise":
        rng = np.random.default_rng(np.random.SeedSequence(
            [int(s_noise), int(q_idx), int(slot), G.SIGMA_INDEX]))
        return C.distort(frame, NOISE_SIGMA, rng), {"sigma": NOISE_SIGMA}
    raise SystemExit("unknown operator %r" % form)


def operator_frames(form, pool, geom, k, q_idx, s_noise):
    out, meta = [], []
    for i in range(len(pool)):
        f, m = operator_frame(form, pool[i], geom, k, q_idx, i, s_noise)
        out.append(f)
        meta.append(m)
    return out, meta


def pixel_precondition(form, pool, geom, k, q_idx, s_noise):
    dist, meta = operator_frames(form, pool, geom, k, q_idx, s_noise)
    changed = [int(not np.array_equal(dist[i], pool[i])) for i in range(len(pool))]
    constant = [int(uint8_min_max(pool[i])[0] == uint8_min_max(pool[i])[1]) for i in range(len(pool))]
    nonconstant = [i for i in range(len(pool)) if not constant[i]]
    applied = bool(all(changed[i] for i in nonconstant))
    identity = None
    if form == "patchcopy":
        g = geom
        same = []
        for i in range(len(pool)):
            dy, dx = meta[i]["offset"]
            ys, xs = slice(g["y0"], g["y1"]), slice(g["x0"], g["x1"])
            outside = np.ones(pool.shape[1:3], bool)
            outside[ys, xs] = False
            ok = bool(np.array_equal(dist[i][outside], pool[i][outside]))
            recon = np.roll(np.roll(pool[i], dy, axis=0), dx, axis=1)
            inside = bool(np.array_equal(dist[i][ys, xs], recon[ys, xs]))
            same.append(int(ok and inside))
        identity = {"n_ok": int(sum(same)), "n": len(pool), "all": bool(all(same))}
    n_distinct = len({hashlib.sha256(np.ascontiguousarray(d).tobytes()).hexdigest()
                      for d in dist})
    return {"form": form, "k": int(k), "applied": applied,
            "n_slots": len(pool), "n_changed": int(sum(changed)),
            "n_constant_slots": int(sum(constant)),
            "frame_identity": identity, "n_distinct_distorted_frames": int(n_distinct)}


def uint8_min_max(a):
    return int(a.min()), int(a.max())

# ---------------------------------------------------------------- screen (pure function)


def cp_ci(hits, n, alpha=0.05):
    from scipy.stats import beta
    k = int(hits)
    n = int(n)
    if n <= 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return [lo, hi]


def acc_entry(hits, n):
    return {"hits": int(hits), "n": int(n), "value": (float(hits) / n if n else None),
            "cp95": cp_ci(hits, n)}


def endpoint(hits, n):
    return bool(int(hits) == 0 or int(hits) == int(n))


def headroom_screen(marked, unmarked, licensed, n):
    """The r4 screen, unchanged, applied to a measured pair table.

    marked/unmarked: dict pair -> {"method": [0/1 per stream], "uniform": 0/1,
                                   "no_frames": 0/1, "evidence_fed": 0/1}

    Endpoint rule, applied at the ARM level exactly as the Plan writes it: an arm whose
    total hits over the n pairs is 0 or n is discarded; every per-pair d resting on that
    arm is null.  A stream whose method arm is an endpoint is dropped for every pair.
    """
    pairs = sorted(marked)
    n_streams = len(marked[pairs[0]]["method"]) if pairs else 0

    def total(role, arm):
        return int(sum(int(role[p][arm]) for p in role))

    m_no, m_uni, m_ef = total(marked, "no_frames"), total(marked, "uniform"), total(marked, "evidence_fed")
    u_no, u_uni, u_ef = total(unmarked, "no_frames"), total(unmarked, "uniform"), total(unmarked, "evidence_fed")

    m_stream_hits = [int(sum(int(marked[p]["method"][k]) for p in pairs))
                     for k in range(n_streams)]
    u_stream_hits = [int(sum(int(unmarked[p]["method"][k]) for p in pairs))
                     for k in range(n_streams)]
    m_valid = [k for k, h in enumerate(m_stream_hits) if not endpoint(h, n)]
    u_valid = [k for k, h in enumerate(u_stream_hits) if not endpoint(h, n)]
    m_any_endpoint = len(m_valid) != n_streams
    u_any_endpoint = len(u_valid) != n_streams

    def per_pair_d(role, valid, uni_end):
        # the endpoint rule applies to an ARM TOTAL (hits over the n pairs), never to a
        # per-pair 0/1 indicator: a pair whose uniform answer is wrong is not an endpoint.
        out = {}
        for p in sorted(role):
            if not valid or uni_end:
                out[p] = None
            else:
                out[p] = float(np.mean([int(role[p]["method"][k]) for k in valid])) \
                    - float(int(role[p]["uniform"]))
        return out

    d_marked = per_pair_d(marked, m_valid, endpoint(m_uni, n))
    d_unmarked = per_pair_d(unmarked, u_valid, endpoint(u_uni, n))

    out = {"n_pairs": int(n), "n_streams": int(n_streams),
           "stream_hits": {"marked": m_stream_hits, "unmarked": u_stream_hits},
           "stream_valid": {"marked": m_valid, "unmarked": u_valid},
           "d_marked": d_marked}
    A1 = bool(float(m_no) / n <= CHANCE_BAR)
    A2 = bool(float(m_uni) / n <= CHANCE_BAR)
    A3 = bool(int(m_ef) >= READ_MIN_HITS)
    A4 = bool((not m_any_endpoint) and (not endpoint(m_uni, n))
              and 0 < m_uni < n and all(0 < h < n for h in m_stream_hits))
    A5 = bool(len(licensed) >= 1)
    B1 = bool((not u_any_endpoint) and all(0 < h < n for h in u_stream_hits))
    B2 = bool((not endpoint(u_uni, n)) and 0 < u_uni < n)
    B3 = bool(any((v is not None and v != 0.0) for v in d_unmarked.values()))
    B4 = bool((not u_any_endpoint) and not endpoint(u_uni, n))

    out["conditions"] = {
        "A1_PC1_no_frames_le_0.35": A1,
        "A2_PC_no_ceiling_uniform_le_0.35": A2,
        "A3_PC_read_evidence_fed_ge_7_of_8": A3,
        "A4_marked_arms_strictly_inside_0_n": A4,
        "A5_at_least_one_operator_licensed": A5,
        "B1_unmarked_method_strictly_inside_0_n": B1,
        "B2_unmarked_uniform_strictly_inside_0_n": B2,
        "B3_at_least_one_nonzero_control_half_difference": B3,
        "B4_no_endpoint_in_control_half": B4,
    }
    out["blocks"] = {"A_readability": bool(A1 and A2 and A3 and A4 and A5),
                     "B_control_half_live": bool(B1 and B2 and B3 and B4)}
    out["d_unmarked"] = d_unmarked
    if out["blocks"]["A_readability"] and out["blocks"]["B_control_half_live"]:
        out["reading_scope"] = "single_arm_and_controlled_contrast"
    elif out["blocks"]["A_readability"]:
        out["reading_scope"] = "single_arm_only"
    else:
        out["reading_scope"] = "none"
    out["counts"] = {"marked": {"no_frames": m_no, "uniform": m_uni,
                                "evidence_fed": m_ef, "n": n},
                     "unmarked": {"no_frames": u_no, "uniform": u_uni,
                                  "evidence_fed": u_ef, "n": n}}
    return out


def u_method_inside(role, n):
    vals = [v for p in role for v in role[p]["method"]]
    return bool(all(0 < int(v) < n for v in vals)) and len(vals) > 0
