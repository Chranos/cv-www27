"""Shared helpers for the r3 batch (Plan plan-matrix-longvideo-r3@v1).

Imported verbatim, never forked: mvp_common (the decoder, uniform_indices, to_pil, distort),
gen_common (the project's ONE seed-derivation function seed_streams, the ONE forward
accumulator Fwd, prompt_for, top_b, tv, the statistics), mx_common (the per-directory manifest,
the expectations bookkeeping), r2_common (the FIXED pre-screen stream and its per-frame
distortion, tv_stats, the in-run identity recomputation).

New in r3 (this module) -- the code the r3 design says is not yet written:

  * the rebuilt carrier-3 B control set's pinned constants and its sections 3.3-3.5
    construction: I32 / I16u membership, the slot-boundary placebo index, the deterministic
    (no RNG at draw time) marker, and the 6/6/6/6 correct-letter balance;
  * the per-question frame-visibility check V0-V7 and its aggregate rule;
  * the three-state k-relative PC1, PC2a / PC2b / PC3;
  * G2_runseed, the run-seed-side instrument reading the r2 design promised and never
    produced (r3 text section 2.3c);
  * arm_key(kind, k, j), the ONE arm-key constructor shared by the writing side and the
    summarising side, plus the self-check S-keys (record defect 3a);
  * the COMPUTED P4 applicability plus the self-check S-p4 (3b) and the readings assertion
    S-readings (3c).
"""
import hashlib
import json
import os
import tempfile

import numpy as np

import mvp_common as C
import gen_common as G
import mx_common as MX
import r2_common as R

PLAN_ID = "plan-matrix-longvideo-r3"
PLAN_VERSION = 1
STAGE = "smoke"

# ================================================================ pinned r3 constants
# The new pinned constant of the rebuilt control set.  The generator's whole stream is derived
# from it through gen_common.seed_streams, the project's one seed-derivation function, so the
# control set is independent of EXP_SEED.
R3_PIN_SEED = 20261017
R3_FAMILY_OFFSET = {"2_needle": 0, "1_plotQA": 1000}
R3_PAIRS_PER_FAMILY = 12
R3_PAIR_FAMILY_ORDER = ("2_needle", "1_plotQA")     # alternating; pair 0 = needle[0]
R3_K_OPTIONS = 4
R3_N_PAIRS = 24
R3_N_QUESTIONS = 48
R3_MECHANICAL_REASONS = (
    "decode failure",
    "video shorter than 32 decodable pool points",
    "marker bbox out of bounds",
    "V3 uniqueness failure",
    "V5 failure",
)
R3_AGGREGATE_RULE = ("after mechanical exclusion n_visible >= 20 and n_placebo >= 12 must BOTH "
                     "hold over the whole control set, else the cell is crash_incomplete_control_set")

# marker constants (pinned; the marker is drawn deterministically, no RNG at draw time)
MARKER_SIDE_FRAC = 0.40          # bbox side as a fraction of min(H, W)
MARKER_JITTER_FRAC = 0.10        # pinned central region: centre +- this fraction of W / H
COLOUR_TOL_LINF = 3              # tight per-channel tolerance for the V5 pixel counting
# why these two numbers: the device's 40 control-set videos run from 540x720 to 796x1908, i.e.
# min(H,W)^2/(H*W) is as low as 0.417 on the 2.4:1 portrait ones, where the thinnest shapes
# (triangle / diamond, area = half the bounding box) would sit at 2.0-2.4% for a 0.34 side and
# only 3.3% for a 0.40 side -- the >= 2% V2 target needs the margin; and the marker is drawn into
# the DECODED array, so its own pixels are exact and a tolerance of 3 per channel still finds it
# while making a natural coincidence of a pinned saturated colour essentially impossible.
MARKER_AREA_MIN = 0.02           # area fraction of the frame handed to the model
MARKER_MIN_STROKE_PX = 3.0       # smallest stroke, in px, at the resolution handed to the model

MARKER_COLOURS = [
    {"key": "magenta", "name": "\u6d0b\u7ea2", "rgb": [255, 0, 255]},
    {"key": "cyan", "name": "\u9752\u8272", "rgb": [0, 255, 255]},
    {"key": "yellow", "name": "\u9ec4\u8272", "rgb": [255, 255, 0]},
    {"key": "green", "name": "\u7eaf\u7eff", "rgb": [0, 255, 0]},
]
MARKER_SHAPES = [
    {"key": "disc", "name": "\u5706\u76d8"},
    {"key": "square", "name": "\u65b9\u5757"},
    {"key": "triangle", "name": "\u4e09\u89d2"},
    {"key": "diamond", "name": "\u83f1\u5f62"},
]
MARKER_STROKE_SEMANTICS = (
    "smallest stroke = the diameter of the largest inscribed disc of the drawn marker mask "
    "(2 x max distance-to-background), measured on the mask resampled to the resolution the "
    "model actually receives (nearest neighbour, the same resize factor mvp_common.to_pil uses); "
    "the bbox side and the central-row/column run length are reported next to it"
)
QUESTION_TEXT = ("\u8fd9\u6bb5\u89c6\u9891\u91cc\u51fa\u73b0\u7684\u724c\u5b50\u662f"
                 "\u4ec0\u4e48\u5f62\u72b6\u548c\u989c\u8272\uff1f"
                 "\u7528\u9009\u9879\u5b57\u6bcd\u56de\u7b54\u3002")
OPTION_LETTERS = C.LETTERS[:R3_K_OPTIONS]


def r3_streams():
    """The one derivation of the pinned control set's stream."""
    st = G.seed_streams(R3_PIN_SEED)
    return {"R3_PIN_SEED": int(R3_PIN_SEED), "s_sel_base": int(st["s_sel"]),
            "s_noise_base": int(st["s_noise"]),
            "note": ("derived from the pinned constant R3_PIN_SEED through "
                     "gen_common.seed_streams; independent of EXP_SEED")}


def _rng(*keys):
    return np.random.default_rng(np.random.SeedSequence([int(k) for k in keys]))


def canonical_json(obj):
    """Canonical JSON: sort_keys, compact separators, UTF-8, no trailing newline."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def pin_sha256_and_bytes(pin):
    b = canonical_json(pin)
    return hashlib.sha256(b).hexdigest(), len(b)


# ================================================================ selection (pure arithmetic)
def select_videos(pinned_items):
    """12 of the 20 videos of each family, chosen by the pinned stream + a per-family offset.

    Pure arithmetic: the input is the pinned item table of data/carrier3b_pin.json; no video
    file is opened and no frame is decoded.
    """
    base = r3_streams()["s_sel_base"]
    out = {}
    for fam in R3_PAIR_FAMILY_ORDER:
        pool = sorted([it for it in pinned_items if it["task"] == fam],
                      key=lambda it: it["video"])
        if len(pool) < R3_PAIRS_PER_FAMILY:
            raise SystemExit("control set: family %s has only %d pinned videos" % (fam, len(pool)))
        rng = _rng(base, R3_FAMILY_OFFSET[fam])
        perm = [int(x) for x in rng.permutation(len(pool)).tolist()]
        out[fam] = [pool[i] for i in perm[:R3_PAIRS_PER_FAMILY]]
        out[fam + "_pool_size"] = len(pool)
        out[fam + "_permutation"] = perm
        out[fam + "_k_rng"] = int(R3_FAMILY_OFFSET[fam])
    return out


def _pool_index_to_source(i, n):
    """The decoder's own rule (mvp_common._decode_cv2 / decode_frames): the source frame index
    a pool position lands on."""
    k = G.K_CAND
    v = int(np.floor((float(i) + 0.5) * n / float(k)))
    return int(min(max(v, 0), n - 1))


def pair_specs(pair_index, fam, item):
    """One twin pair's pinned arithmetic: i*, the marker, the four options, the correct index."""
    base = r3_streams()["s_sel_base"]
    i16u = C.uniform_indices(G.K_CAND, G.B_MAIN)
    i_star = int(i16u[int(_rng(base, 7, pair_index).integers(0, len(i16u)))])
    rng_m = _rng(base, 13, pair_index)
    c0 = int(rng_m.integers(0, R3_K_OPTIONS))
    s0 = int(rng_m.integers(0, R3_K_OPTIONS))
    jx = float(_rng(base, 17, pair_index).uniform(-MARKER_JITTER_FRAC, MARKER_JITTER_FRAC))
    jy = float(_rng(base, 19, pair_index).uniform(-MARKER_JITTER_FRAC, MARKER_JITTER_FRAC))
    # correct-letter balance: block b (4 pairs) draws one permutation, each letter lands exactly
    # 6 times over the 24 visible questions
    block = pair_index // R3_K_OPTIONS
    perm4 = [int(x) for x in _rng(base, 29, block).permutation(R3_K_OPTIONS).tolist()]
    pos = int(perm4[pair_index % R3_K_OPTIONS])
    # presentation order: position `pos` holds the drawn combo (o = 0), the other positions hold
    # o = 1, 2, 3 in ascending order
    order = [None] * R3_K_OPTIONS
    order[pos] = 0
    rest = iter([1, 2, 3])
    for i in range(R3_K_OPTIONS):
        if order[i] is None:
            order[i] = next(rest)
    options = [option_text((c0 + o) % R3_K_OPTIONS, (s0 + o) % R3_K_OPTIONS) for o in order]
    return {"pair": int(pair_index), "family": fam, "task": item["task"],
            "video": item["video"], "sha256": item["sha256"], "bytes": int(item["bytes"]),
            "i_star": i_star, "i_star_in_I32": True, "i_star_in_I16u": True,
            "marker": {"colour_index": c0, "colour_key": MARKER_COLOURS[c0]["key"],
                       "colour_name": MARKER_COLOURS[c0]["name"],
                       "colour_rgb": list(MARKER_COLOURS[c0]["rgb"]),
                       "shape_index": s0, "shape_key": MARKER_SHAPES[s0]["key"],
                       "shape_name": MARKER_SHAPES[s0]["name"],
                       "side_frac": MARKER_SIDE_FRAC,
                       "cx_frac": round(0.5 + jx, 6), "cy_frac": round(0.5 + jy, 6),
                       "jitter_frac": {"x": round(jx, 6), "y": round(jy, 6)},
                       "jitter_stream": ("r3_streams()['s_sel_base'], keys (17, pair) and "
                                         "(19, pair)"),
                       "colour_tol_linf": COLOUR_TOL_LINF},
            "options": options, "option_order_o": order,
            "correct_index": pos, "correct_letter": OPTION_LETTERS[pos],
            "question_text": QUESTION_TEXT,
            "option_combo_rule": ("option at position p holds (colour (c0+o)%4, shape (s0+o)%4) "
                                  "for o = option_order_o[p]; any two options differ in BOTH "
                                  "colour and shape"),
            "correct_letter_stream": ("r3_streams()['s_sel_base'], keys (29, pair//4), "
                                      "permutation, position pair%4")}


def option_text(ci, si):
    return "%s %s" % (MARKER_COLOURS[ci]["name"], MARKER_SHAPES[si]["name"])


def build_pin(pinned_items, n_frames=None, fps=None):
    """The pinned control set.  Pure arithmetic, zero decoding, independent of EXP_SEED.

    `n_frames` / `fps` are container header metadata per video filename, passed IN by the
    caller (they are the only run-independent way to reach the slot-boundary placebo index
    j* = floor((i*+1)*n/32)).  Called with n_frames=None the same pin is produced with
    j_star = null and j_star_rule pinned, and the caller can see that those are the only two
    fields that differ.
    """
    sel = select_videos(pinned_items)
    n_frames = n_frames or {}
    fps = fps or {}
    pairs = []
    for p in range(R3_N_PAIRS):
        fam = R3_PAIR_FAMILY_ORDER[p % 2]
        item = sel[fam][p // 2]
        sp = pair_specs(p, fam, item)
        n = n_frames.get(item["video"])
        f = fps.get(item["video"])
        j_star = None
        gap = None
        if n and int(n) > 0:
            j_star = int((sp["i_star"] + 1) * int(n) // G.K_CAND)
            if f and float(f) > 0:
                gap = (j_star - _pool_index_to_source(sp["i_star"], int(n))) / float(f)
        sp["n_frames_source"] = (int(n) if n else None)
        sp["fps"] = (float(f) if f else None)
        sp["j_star"] = j_star
        sp["j_star_rule"] = "floor((i_star+1) * n_frames_source / 32)"
        sp["gap_s"] = (round(gap, 6) if gap is not None else None)
        sp["placebo_rule"] = ("floor((i_star+1) * n / 32), the source-frame index of the slot "
                              "boundary between pool slot i_star and slot i_star+1")
        sp["visible_qid"] = "%s/%s#visible" % (fam, item["video"])
        sp["placebo_qid"] = "%s/%s#placebo" % (fam, item["video"])
        pairs.append(sp)
    pin = {
        "plan_id": PLAN_ID, "plan_version": PLAN_VERSION,
        "what": ("rebuilt carrier-3 B control set: 24 twin pairs (12 needle + 12 plotQA) on the "
                 "24 videos the pinned generator selects out of the 40 in the device root, with "
                 "the evidence frame index, the placebo frame index, the marker spec, the four "
                 "options and the correct index per pair"),
        "generator": "r3_gen_pin.py / r3_common.build_pin (pure arithmetic, zero decoding, zero models)",
        "streams": r3_streams(),
        "selection": {k: v for k, v in sel.items()},
        "pair_family_order": list(R3_PAIR_FAMILY_ORDER),
        "pinned_constants": {
            "R3_PIN_SEED": R3_PIN_SEED, "R3_FAMILY_OFFSET": R3_FAMILY_OFFSET,
            "R3_PAIRS_PER_FAMILY": R3_PAIRS_PER_FAMILY, "K_CAND": G.K_CAND,
            "B_MAIN": G.B_MAIN, "B_LOW": G.B_LOW, "K_OPTIONS": R3_K_OPTIONS,
            "MARKER_SIDE_FRAC": MARKER_SIDE_FRAC, "MARKER_JITTER_FRAC": MARKER_JITTER_FRAC,
            "COLOUR_TOL_LINF": COLOUR_TOL_LINF, "MARKER_AREA_MIN": MARKER_AREA_MIN,
            "MARKER_MIN_STROKE_PX": MARKER_MIN_STROKE_PX,
            "SIGMA_MAIN": G.SIGMA_MAIN, "SIGMA_INDEX": G.SIGMA_INDEX,
            "S_PRE_STREAM": R.S_PRE_STREAM, "PS_FIXED_SEED": R.PS_FIXED_SEED,
            "I32": C.uniform_indices(G.K_CAND, G.K_CAND),
            "I16u": C.uniform_indices(G.K_CAND, G.B_MAIN),
        },
        "marker_colour_table": MARKER_COLOURS, "marker_shape_table": MARKER_SHAPES,
        "marker_stroke_semantics": MARKER_STROKE_SEMANTICS,
        "question_text": QUESTION_TEXT,
        "letters": list(OPTION_LETTERS),
        "correct_letter_balance_rule": ("each of the 4 letters is the correct position exactly 6 "
                                        "times over the 24 visible questions"),
        "correct_letter_counts": {L: sum(1 for p in pairs if p["correct_letter"] == L)
                                  for L in OPTION_LETTERS},
        "n_pairs": len(pairs), "n_questions": 2 * len(pairs),
        "pairs": pairs,
    }
    return pin


# ================================================================ the marker (deterministic)
def marker_geometry(spec, H, W):
    """Pin the marker's pixel geometry.  No randomness: the jitter was drawn once, at pin time,
    and is carried in the spec as a fraction."""
    side = int(round(float(spec["side_frac"]) * float(min(H, W))))
    side = max(side, 8)
    cx = int(round(float(spec["cx_frac"]) * W))
    cy = int(round(float(spec["cy_frac"]) * H))
    x0, y0 = int(cx - side // 2), int(cy - side // 2)
    x1, y1 = x0 + side, y0 + side
    return {"side": side, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "cx": cx, "cy": cy,
            "H": int(H), "W": int(W),
            "in_bounds": bool(x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H)}


def shape_mask(side, shape_index):
    """The drawn shape's mask, side x side, uint8 0/1.  Deterministic rasterisation."""
    s = int(side)
    yy, xx = np.mgrid[0:s, 0:s]
    c = (s - 1) / 2.0
    X = (xx - c) / (s / 2.0)
    Y = (yy - c) / (s / 2.0)
    key = MARKER_SHAPES[int(shape_index)]["key"]
    if key == "disc":
        m = (X ** 2 + Y ** 2) <= 1.0
    elif key == "square":
        m = np.ones((s, s), bool)
    elif key == "triangle":
        m = (Y <= 1.0) & (np.abs(X) <= (Y + 1.0) / 2.0)
    elif key == "diamond":
        m = (np.abs(X) + np.abs(Y)) <= 1.0
    else:
        raise SystemExit("unknown marker shape %r" % key)
    return m.astype(np.uint8)


def apply_marker(frame, spec, geom=None):
    """Return a copy of `frame` with the pinned marker drawn into it.  No RNG."""
    H, W = frame.shape[:2]
    g = geom if geom is not None else marker_geometry(spec, H, W)
    out = frame.copy()
    if not g["in_bounds"]:
        return out, g
    m = shape_mask(g["side"], int(spec["shape_index"])).astype(bool)
    sub = out[g["y0"]:g["y0"] + g["side"], g["x0"]:g["x0"] + g["side"]]
    sub[m] = np.asarray(spec["colour_rgb"], np.uint8)
    return out, g


def combo_match_count(frame, spec, colour_rgb, shape_index, geom=None):
    """Pixels matching one (colour, shape) combo: inside that shape's mask at the pinned
    geometry, within the pinned per-channel tolerance of that colour.  -1 if out of bounds."""
    H, W = frame.shape[:2]
    g = geom if geom is not None else marker_geometry(spec, H, W)
    if not g["in_bounds"]:
        return -1
    s = g["side"]
    sub = frame[g["y0"]:g["y0"] + s, g["x0"]:g["x0"] + s].astype(np.int16)
    m = shape_mask(s, shape_index).astype(bool)
    d = np.abs(sub - np.asarray(colour_rgb, np.int16))
    return int(np.count_nonzero((d.max(axis=2) <= COLOUR_TOL_LINF) & m))


def combo_counts(frame, spec, geom=None):
    """The four pinned (colour, shape) combos' pixel counts in one frame, in option order."""
    return [combo_match_count(frame, spec, MARKER_COLOURS[(int(spec["colour_index"]) + o) % 4]["rgb"],
                              (int(spec["shape_index"]) + o) % 4, geom) for o in range(4)]


def pil_target_size(H, W, max_pixels=None):
    """The size mvp_common.to_pil hands the model for a frame of H x W."""
    import math
    max_pixels = G.MAX_PIXELS if max_pixels is None else max_pixels
    if W * H <= max_pixels:
        return (int(W), int(H)), False
    s = math.sqrt(max_pixels / float(W * H))
    nw = max(28, int(W * s) // 28 * 28)
    nh = max(28, int(H * s) // 28 * 28)
    return (int(nw), int(nh)), True


def stroke_measures(mask, size):
    """Area fraction, bbox side and smallest stroke of a mask resampled to `size`."""
    import cv2
    from PIL import Image
    m = Image.fromarray((mask.astype(np.uint8) * 255)).resize(tuple(size), Image.NEAREST)
    a = np.asarray(m) > 0
    n = int(np.count_nonzero(a))
    if n == 0:
        return {"area_frac": 0.0, "mask_pixels": 0, "canvas_pixels": int(a.size),
                "bbox_side_px": 0, "inscribed_diameter_px": 0.0,
                "central_row_run_px": 0, "central_col_run_px": 0, "size": list(size)}
    ys, xs = np.nonzero(a)
    side = int(min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
    dt = cv2.distanceTransform((a.astype(np.uint8) * 255), cv2.DIST_L2, 5)
    cy = int(round((ys.min() + ys.max()) / 2.0))
    cx = int(round((xs.min() + xs.max()) / 2.0))
    row = a[cy, :]
    col = a[:, cx]
    return {"area_frac": float(n) / float(a.size), "mask_pixels": n,
            "canvas_pixels": int(a.size), "bbox_side_px": side,
            "inscribed_diameter_px": float(2.0 * float(dt.max())),
            "central_row_run_px": int(np.count_nonzero(row)),
            "central_col_run_px": int(np.count_nonzero(col)),
            "size": [int(size[0]), int(size[1])],
            "definition": MARKER_STROKE_SEMANTICS}


# ================================================================ V0-V7
def _max_abs_diff(a, b):
    if a.shape != b.shape:
        return None
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())


def visibility_checks(pair, clean, visible_arr, placebo_fed, placebo_marker_frame,
                      n_dec, fps_dec, decoded_extra=None):
    """V0-V7 for one twin pair: one row per question (visible, placebo).

    clean              the CLEAN decode of the video's 32 pool frames
    visible_arr        the array the visible question's arm consumes (marker drawn at i*)
    placebo_fed        the array the placebo question's arm consumes (the CLEAN copy)
    placebo_marker_frame  the separately decoded off-pool frame at j*, with the marker drawn
    """
    i32 = C.uniform_indices(G.K_CAND, G.K_CAND)
    i16u = C.uniform_indices(G.K_CAND, G.B_MAIN)
    i = int(pair["i_star"])
    spec = pair["marker"]
    H, W = clean.shape[1], clean.shape[2]
    geom = marker_geometry(spec, H, W)
    n = int(n_dec)
    fps = float(fps_dec or 0.0)
    src_idx = [_pool_index_to_source(t, n) for t in range(G.K_CAND)]
    t_pool = [(src_idx[t] / fps if fps > 0 else None) for t in range(G.K_CAND)]
    j = pair.get("j_star")
    notes = list(decoded_extra or [])

    # ---- V0 pool consistency -------------------------------------------------------------
    v0 = {"I32": i32, "I16u": i16u,
          "I32_len": len(i32), "I16u_len": len(i16u),
          "I32_sha256": hashlib.sha256(",".join(str(x) for x in i32).encode()).hexdigest(),
          "I16u_sha256": hashlib.sha256(",".join(str(x) for x in i16u).encode()).hexdigest(),
          "I16u_subset_I32": bool(set(i16u) <= set(i32)),
          "I32_equals_range_0_31": bool(i32 == list(range(G.K_CAND))),
          "recomputed_with": "mvp_common.uniform_indices(32, 32) and uniform_indices(32, 16), the "
                             "same functions the arms use; both are pure index arithmetic and "
                             "independent of duration",
          "source_frame_indices_of_pool": src_idx,
          "pass": bool(len(i32) == G.K_CAND and len(i16u) == G.B_MAIN and set(i16u) <= set(i32))}

    # ---- V1 evidence position ------------------------------------------------------------
    d_i = None
    if i in i32 and fps > 0:
        d_i = abs(t_pool[i] - t_pool[i])
    v1_visible = {"i_star": i, "i_star_in_I32": bool(i in i32), "i_star_in_I16u": bool(i in i16u),
                  "t_star_s": (t_pool[i] if (fps > 0 and 0 <= i < len(t_pool)) else None),
                  "source_frame_index_of_i_star": src_idx[i] if 0 <= i < len(src_idx) else None,
                  "distance_to_nearest_pool_timestamp_s": 0.0,
                  "remark": ("i* IS a pool timestamp, so its distance to the nearest pool "
                             "timestamp is 0 by construction"),
                  "pass": bool(i in i32 and i in i16u)}
    v1_placebo = {"j_star": j, "j_star_rule": pair.get("j_star_rule"),
                  "j_star_in_I32": (None if j is None else bool(j in i32)),
                  "j_star_in_decoded_pool_source_indices": (None if j is None
                                                            else bool(j in src_idx)),
                  "t_placebo_s": (j / fps if (j is not None and fps > 0) else None),
                  "gap_rule": "|t(j*) - t(nearest pool timestamp)|",
                  "pass": False}
    if j is not None and fps > 0:
        gaps = [abs(j / fps - t_pool[t]) for t in range(len(t_pool))]
        near = int(np.argmin(gaps))
        v1_placebo.update({"nearest_pool_position": near, "nearest_pool_timestamp_s": t_pool[near],
                           "distance_to_nearest_pool_timestamp_s": float(gaps[near]),
                           "gap_ge_0.5s": bool(float(gaps[near]) >= 0.5),
                           "literal_membership_test_vacuous": bool(j >= G.K_CAND),
                           "literal_membership_note": (
                               "I32 is the pool INDEX space {0..31} (design setup section 3.4); "
                               "j* is a SOURCE frame index, so 'j* not in I32' is vacuous for "
                               "every video whose frame count exceeds 32 -- reported as "
                               "j_star_in_I32 besides the meaningful test j* not in the decoded "
                               "pool source indices")})
        v1_placebo["pass"] = bool((j not in i32) and (j not in src_idx)
                                  and v1_placebo.get("gap_ge_0.5s"))

    # ---- V2 marker present in the array the arm consumes ---------------------------------
    expected, _g = apply_marker(clean[i], spec, geom)
    v2 = {"marker_drawn_at_frame": i, "compared_against": "re-synthesis from the clean decode",
          "max_abs_diff_visible_frame": _max_abs_diff(visible_arr[i], expected),
          "patch_bbox": {"x0": geom["x0"], "y0": geom["y0"], "x1": geom["x1"], "y1": geom["y1"],
                         "side_px": geom["side"], "frame_hw": [H, W]},
          "marker_bbox_in_bounds": geom["in_bounds"]}
    pil_size, resized = pil_target_size(H, W)
    mask = shape_mask(geom["side"], int(spec["shape_index"]))
    pil_actual = None
    try:
        # mvp_common.to_pil takes a STACK of frames; pass a stack of one.  The first cut passed a
        # bare (H,W,3) frame, so to_pil iterated over its ROWS and the cross-check reported a
        # meaningless size: the run whose Evidence is evidence.task-907d017857 carries
        # that wart in metrics.cells[].pair_info[].V2.pil_actual_size_from_to_pil and
        # pil_target_matches_to_pil (e.g. [3, 1920] for 1920x792).  The V2 verdict itself used
        # pil_target_size(), which replicates to_pil's own rule for a frame, so the area and
        # stroke measurements and the V2 pass are unaffected.
        pil_actual = list(C.to_pil(clean[i][None], G.MAX_PIXELS)[0].size)
    except Exception as e:                                       # noqa: BLE001
        notes.append("to_pil check failed: %s" % str(e)[:120])
    pre = stroke_measures(mask, (geom["side"], geom["side"]))
    sx = pil_size[0] / float(W)
    sy = pil_size[1] / float(H)
    bbox_post = (max(1, int(round(geom["side"] * sx))), max(1, int(round(geom["side"] * sy))))
    post = stroke_measures(mask, bbox_post)
    frame_pre, frame_post = float(H) * float(W), float(pil_size[0]) * float(pil_size[1])
    pre, post = dict(pre), dict(post)
    pre["area_frac_of_frame"] = pre["mask_pixels"] / frame_pre
    post["area_frac_of_frame"] = post["mask_pixels"] / frame_post
    v2.update({"pre_resize": pre, "post_resize": post,
               "pil_target_size": list(pil_size), "pil_actual_size_from_to_pil": pil_actual,
               "pil_target_matches_to_pil": bool(pil_actual == list(pil_size)),
               "frame_was_resized": bool(resized),
               "resize_scale": {"x": sx, "y": sy},
               "bbox_size_at_model_input": list(bbox_post),
               "measurement_note": ("the marker mask is resampled to the resolution the model "
                                    "receives (nearest neighbour, the same factor mvp_common.to_pil "
                                    "uses) and the area fraction is taken of that frame"),
               "area_frac_at_model_input": post["area_frac_of_frame"],
               "min_stroke_px_at_model_input": post["inscribed_diameter_px"],
               "area_target": MARKER_AREA_MIN, "stroke_target_px": MARKER_MIN_STROKE_PX,
               "area_ge_target": bool(post["area_frac_of_frame"] >= MARKER_AREA_MIN),
               "stroke_ge_target": bool(post["inscribed_diameter_px"] >= MARKER_MIN_STROKE_PX)})
    v2["pass"] = bool(v2["max_abs_diff_visible_frame"] == 0.0 and geom["in_bounds"]
                      and v2["area_ge_target"] and v2["stroke_ge_target"])

    # ---- V3 uniqueness -------------------------------------------------------------------
    other_frames = [t for t in range(len(clean)) if t != i]
    diffs_other = [_max_abs_diff(visible_arr[t], clean[t]) for t in other_frames]
    v3_visible = {"n_frames_compared": len(other_frames),
                  "max_abs_diff_over_other_pool_frames": (max(diffs_other) if diffs_other else None),
                  "max_abs_diff_at_i_star": _max_abs_diff(visible_arr[i], clean[i]),
                  "rule": "every pool frame other than i* must be bit-identical to the clean decode",
                  "pass": bool(diffs_other and max(diffs_other) == 0.0)}
    dp = [_max_abs_diff(placebo_fed[t], clean[t]) for t in range(len(clean))]
    v3_placebo = {"n_frames_compared": len(dp), "max_abs_diff_over_all_pool_frames":
                  (max(dp) if dp else None),
                  "rule": "all 32 pool frames of the placebo question must be bit-identical to "
                          "the clean decode",
                  "pass": bool(dp and max(dp) == 0.0)}
    placebo_not_from_visible = {"placebo_fed_equals_clean": bool(np.array_equal(placebo_fed, clean)),
                                "placebo_fed_equals_visible": bool(np.array_equal(placebo_fed, visible_arr)),
                                "max_abs_diff_placebo_vs_visible": _max_abs_diff(placebo_fed, visible_arr)}
    placebo_not_from_visible["assert_holds"] = bool(
        placebo_not_from_visible["placebo_fed_equals_clean"]
        and not placebo_not_from_visible["placebo_fed_equals_visible"])

    # ---- V4 destroyability (pre-registered input of PC2a) --------------------------------
    distorted = R.pre_distort(visible_arr[i], pair["pair"], i)
    diff = distorted.astype(np.float32) - visible_arr[i].astype(np.float32)
    noise_mag = np.linalg.norm(diff, axis=2)
    bbox_clean = clean[i][geom["y0"]:geom["y1"], geom["x0"]:geom["x1"]].astype(np.float32)
    bbox_mask = mask.astype(bool)
    bbox_vis = visible_arr[i][geom["y0"]:geom["y1"], geom["x0"]:geom["x1"]].astype(np.float32)
    local_contrast = np.linalg.norm(bbox_vis - bbox_clean, axis=2)[bbox_mask]
    noise_in_mask = noise_mag[geom["y0"]:geom["y1"], geom["x0"]:geom["x1"]][bbox_mask]
    lc = float(local_contrast.mean()) if local_contrast.size else 0.0
    nm = float(noise_in_mask.mean()) if noise_in_mask.size else 0.0
    frac = (float(np.count_nonzero(noise_in_mask > 0.5 * lc) / float(noise_in_mask.size))
            if noise_in_mask.size else None)
    snr = (lc / nm) if nm > 0 else None
    v4 = {"sigma": G.SIGMA_MAIN, "stream": "r2_common.pre_distort (fixed pre-screen stream s_pre=%d)"
          % R.S_PRE_STREAM,
          "patch_pixels": int(bbox_mask.sum()),
          "patch_mean_pixel_change": nm, "patch_mean_pixel_change_unit": "L2 over the 3 channels, 0-255",
          "patch_mean_pixel_change_0_1": (nm / 255.0),
          "local_contrast_mean": lc, "local_contrast_mean_unit": "L2 over the 3 channels, 0-255",
          "local_contrast_definition": ("mean over the marker mask of ||marker colour - the clean "
                                        "pixel the marker replaced||_2"),
          "frac_patch_pixels_changed_gt_half_local_contrast": frac,
          "patch_snr": snr,
          "preregistered_criterion": "marker unreadable at sigma=0.2 <=> patch SNR < 1",
          "snr_lt_1": (None if snr is None else bool(snr < 1.0)),
          "note": ("recorded BEFORE any positive-control reading, as PC2a's pre-registered input; "
                   "no model runs in this dispatch so no PC reading consumes it yet")}

    # ---- V5 option availability ----------------------------------------------------------
    clean_counts = [combo_counts(clean[t], spec, geom) for t in range(len(clean))]
    vis_counts = combo_counts(visible_arr[i], spec, geom)
    j_counts = (combo_counts(placebo_marker_frame, spec, geom) if placebo_marker_frame is not None
                else None)
    j_pos = (j if (j is not None and 0 <= j < len(clean)) else None)
    v5 = {"tol_linf": COLOUR_TOL_LINF,
          "clean_all_frames_counts": clean_counts,
          "visible_i_star_counts": vis_counts,
          "placebo_j_star_frame_counts": j_counts,
          "clean_correct_combo_nonzero_positions": [t for t in range(len(clean_counts))
                                                    if clean_counts[t][0] > 0],
          "clean_distractor_nonzero_positions": {str(o): [t for t in range(len(clean_counts))
                                                          if clean_counts[t][o] > 0]
                                                 for o in (1, 2, 3)},
          "visible_correct_combo_nonzero_positions": ([i] if vis_counts[0] > 0 else []),
          "visible_distractor_nonzero_counts": {str(o): vis_counts[o] for o in (1, 2, 3)},
          "pass": False}
    vis_ok = bool(vis_counts[0] > 0 and all(vis_counts[o] == 0 for o in (1, 2, 3))
                  and all(c[o] == 0 for c in clean_counts for o in (0, 1, 2, 3)))
    plc_ok = bool(all(c[o] == 0 for c in clean_counts for o in (0, 1, 2, 3)))
    if j_counts is not None:
        plc_ok = bool(plc_ok and j_counts[0] > 0 and all(j_counts[o] == 0 for o in (1, 2, 3)))
    v5["visible_pass"] = vis_ok
    v5["placebo_pass"] = plc_ok
    v5["pass"] = bool(vis_ok and plc_ok)

    # ---- V6 letter balance (set level, identical in every row) ---------------------------
    v6 = {"source": "the pin's correct_index / correct_letter over the 24 visible questions",
          "note": ("the no_frames max-single-letter baseline is NOT available in this dispatch: "
                   "no model runs, so no no_frames letters exist; left null and to be produced by "
                   "stage 2")}

    # ---- V7 decode integrity -------------------------------------------------------------
    v7 = {"pool_points_expected": int(G.K_CAND), "pool_points_decoded": int(len(clean)),
          "source_frame_count_decoded": n, "fps_decoded": fps,
          "n_frames_source_pinned": pair.get("n_frames_source"),
          "exclusion_reason": None}
    if len(clean) < G.K_CAND:
        v7["exclusion_reason"] = "video shorter than 32 decodable pool points"
    elif pair.get("n_frames_source") is not None and int(pair["n_frames_source"]) != n:
        v7["exclusion_reason"] = ("decode failure (container frame count %s differs from the "
                                 "decoded frame count %d)" % (pair["n_frames_source"], n))
    elif not geom["in_bounds"]:
        v7["exclusion_reason"] = "marker bbox out of bounds"

    v6 = dict(v6)
    v6["pass"] = None                     # set level; filled by the aggregate

    base_row = {"pair": int(pair["pair"]), "family": pair["family"], "video": pair["video"],
                "sha256": pair["sha256"], "bytes": int(pair["bytes"]),
                "task": pair["task"], "n_frames_source": pair.get("n_frames_source"),
                "fps": pair.get("fps"), "i_star": i, "j_star": j, "gap_s": pair.get("gap_s"),
                "marker": spec, "options": pair["options"], "correct_index": pair["correct_index"],
                "correct_letter": pair["correct_letter"], "gold": pair["correct_letter"],
                "question_text": pair["question_text"], "n_candidates": R3_K_OPTIONS,
                "label_used_in_scoring": False,
                "label_used_in_scoring_note": ("the gold letter is used only to score, never to "
                                               "score-frame or to build the marker"),
                "V0": v0}

    def prompt_of(role):
        q = {"question": pair["question_text"], "candidates": list(pair["options"])}
        p = G.prompt_for(q)
        return p, hashlib.sha256(p.encode()).hexdigest()

    rows = []
    for role, v1, v2b, v3b, v5b, excl in (
            ("visible", v1_visible, v2, v3_visible, v5, None),
            ("placebo", v1_placebo, None, v3_placebo, v5, None)):
        p, ph = prompt_of(role)
        row = dict(base_row)
        row["role"] = role
        row["_qid"] = pair["visible_qid"] if role == "visible" else pair["placebo_qid"]
        row["prompt"] = p
        row["prompt_sha256"] = ph
        row["V1"] = v1
        row["V2"] = (v2b if role == "visible" else
                     {"applies_to": "visible", "note": "the placebo's own marker is the same "
                      "marker, drawn only into the off-pool frame at j*; the fed array carries no "
                      "marker at all (see V3 placebo)"})
        row["V3"] = v3b
        row["V3_placebo_not_from_visible"] = placebo_not_from_visible
        row["V4"] = (v4 if role == "visible" else
                     {"applies_to": "visible", "note": "recorded on the visible question's marker "
                      "frame; the placebo's fed frames carry no marker"})
        row["V5"] = v5b
        row["V6"] = v6
        row["V7"] = v7
        v5_fail = (not v5["visible_pass"]) if role == "visible" else (not v5["placebo_pass"])
        v3_fail = not row["V3"]["pass"]
        reason = v7["exclusion_reason"]
        if reason is None:
            if v3_fail:
                reason = "V3 uniqueness failure"
            elif v5_fail:
                reason = "V5 failure"
        row["excluded_mechanical"] = bool(reason is not None)
        row["exclusion_reason"] = reason
        row["exclusion_reason_is_preregistered"] = bool(reason is None or
                                                        reason in R3_MECHANICAL_REASONS)
        rows.append(row)
    info = {"pair": int(pair["pair"]), "video": pair["video"], "family": pair["family"],
            "i_star": i, "j_star": j, "gap_s": pair.get("gap_s"),
            "n_frames_source_decoded": n, "fps_decoded": fps,
            "geom": geom, "V0_pass": v0["pass"], "V1_visible_pass": v1_visible["pass"],
            "V1_placebo_pass": v1_placebo["pass"], "V2_pass": v2["pass"],
            "V3_visible_pass": v3_visible["pass"], "V3_placebo_pass": v3_placebo["pass"],
            "V5_visible_pass": v5["visible_pass"], "V5_placebo_pass": v5["placebo_pass"],
            "notes": notes}
    return rows, info


def aggregate_visibility(rows, pairs_total=R3_N_PAIRS):
    """The pre-written aggregate rule of design setup section 3.4."""
    vis = [r for r in rows if r["role"] == "visible"]
    plc = [r for r in rows if r["role"] == "placebo"]
    vis_keep = [r for r in vis if not r["excluded_mechanical"]]
    plc_keep = [r for r in plc if not r["excluded_mechanical"]]
    excl = [{"_qid": r["_qid"], "role": r["role"], "pair": r["pair"], "video": r["video"],
             "reason": r["exclusion_reason"]} for r in rows if r["excluded_mechanical"]]
    v6_counts = {L: sum(1 for r in vis if r["correct_letter"] == L) for L in OPTION_LETTERS}
    v6_exact = all(v6_counts[L] == 6 for L in OPTION_LETTERS)
    out = {"n_pairs_built": int(pairs_total), "n_questions_built": int(len(rows)),
           "n_visible_built": len(vis), "n_placebo_built": len(plc),
           "n_visible": len(vis_keep), "n_placebo": len(plc_keep),
           "n_visible_excluded": len(vis) - len(vis_keep),
           "n_placebo_excluded": len(plc) - len(plc_keep),
           "exclusion_list": excl, "exclusion_reasons_allowed": list(R3_MECHANICAL_REASONS),
           "aggregate_rule": R3_AGGREGATE_RULE,
           "aggregate_rule_holds": bool(len(vis_keep) >= 20 and len(plc_keep) >= 12),
           "V6_position_balance": {"counts": v6_counts, "expected": {L: 6 for L in OPTION_LETTERS},
                                   "exactly_6_each": bool(v6_exact),
                                   "n_visible_units": len(vis),
                                   "no_frames_max_single_letter_baseline": None,
                                   "no_frames_baseline_note": (
                                       "NOT available: no model runs in this dispatch, so there are "
                                       "no no_frames letters; stage 2 produces it"),
                                   "dispatch_text_note": (
                                       "the dispatch text says '6/6/6/4-balanced (6 of each "
                                       "letter)'; 6+6+6+4 = 22 != 24, so the two halves of that "
                                       "sentence disagree.  The r3 Plan (setup 3.3 item 4) says "
                                       "each of the 4 positions exactly 6 times over 24 "
                                       "questions and that is what is pinned and checked here.  "
                                       "Reported, not silently reinterpreted.")}}
    for key, role in (("V0", None), ("V1", None), ("V2", "visible"), ("V3", None),
                      ("V5", None)):
        _ = key, role
    def pass_count(name, pred):
        ok = [r for r in rows if pred(r)]
        return {"pass_count": len(ok), "n_units": len(rows), "fail_qids": [r["_qid"] for r in rows
                                                                          if not pred(r)]}
    out["V0"] = pass_count("V0", lambda r: bool(r["V0"]["pass"]))
    out["V1"] = pass_count("V1", lambda r: bool(r["V1"]["pass"]))
    out["V2"] = {"pass_count": sum(1 for r in vis if r["V2"].get("pass")),
                 "n_units": len(vis), "note": "applies to the visible question of each pair",
                 "fail_qids": [r["_qid"] for r in vis if not r["V2"].get("pass")]}

    def role_pass(role, pred):
        """Per-role pass counter: only that role's rows are counted, so fail_qids lists real
        failures of that role.  (The first cut used a role-filtered predicate on the WHOLE row
        list, so each role's fail_qids listed the other role's rows; the counts were right and
        the real exclusion list was always the one above.  The run whose Evidence is
        evidence.task-907d017857 carries that wart in
        metrics.control_visibility.V3_visible/V3_placebo/V5_visible/V5_placebo.)"""
        rs = [r for r in rows if r["role"] == role]
        ok = [r for r in rs if pred(r)]
        return {"pass_count": len(ok), "n_units": len(rs), "role": role,
                "fail_qids": [r["_qid"] for r in rs if not pred(r)]}

    out["V3_visible"] = role_pass("visible", lambda r: bool(r["V3"]["pass"]))
    out["V3_placebo"] = role_pass("placebo", lambda r: bool(r["V3"]["pass"]))
    out["V3_placebo_not_from_visible"] = {
        "pass_count": sum(1 for r in plc
                          if r["V3_placebo_not_from_visible"]["assert_holds"]),
        "n_units": len(plc),
        "fail_qids": [r["_qid"] for r in plc
                      if not r["V3_placebo_not_from_visible"]["assert_holds"]]}
    out["V5_visible"] = role_pass("visible", lambda r: bool(r["V5"]["visible_pass"]))
    out["V5_placebo"] = role_pass("placebo", lambda r: bool(r["V5"]["placebo_pass"]))
    out["V4_readings"] = {
        "n_units": len(vis), "applies_to": "the visible question's marker frame",
        "patch_snr": [r["V4"]["patch_snr"] for r in vis],
        "patch_mean_pixel_change": [r["V4"]["patch_mean_pixel_change"] for r in vis],
        "frac_patch_pixels_changed_gt_half_local_contrast":
            [r["V4"]["frac_patch_pixels_changed_gt_half_local_contrast"] for r in vis],
        "local_contrast_mean": [r["V4"]["local_contrast_mean"] for r in vis],
        "snr_lt_1_count": sum(1 for r in vis if r["V4"]["snr_lt_1"]),
        "median_patch_snr": (float(np.median([r["V4"]["patch_snr"] for r in vis]))
                             if vis and all(r["V4"]["patch_snr"] is not None for r in vis) else None),
        "criterion": "marker unreadable at sigma=0.2 <=> patch SNR < 1"}
    out["V7_decode_integrity"] = {
        "n_units": len(rows),
        "pool_points_expected": int(G.K_CAND),
        "pool_points_decoded_min": min((r["V7"]["pool_points_decoded"] for r in rows), default=None),
        "pool_points_decoded_max": max((r["V7"]["pool_points_decoded"] for r in rows), default=None),
        "decode_failure_qids": [r["_qid"] for r in rows
                                if (r["V7"]["exclusion_reason"] or "").startswith("decode failure")],
        "exclusion_list": excl}
    out["verdict"] = ("ok" if out["aggregate_rule_holds"] else "crash_incomplete_control_set")
    return out


# ================================================================ arm keys (record defect 3a)
def arm_key(kind, k=None, j=None):
    """THE arm-key constructor.  The writing side and the summarising side both call this one
    function; the r2 defect was that they each built random@16's key by hand, in different
    orders (random@16_k{k}_j{j} on the writing side, random@16_j{j}_k{k} on the summarising
    side), so four cells reported null."""
    if k is None and j is None:
        return str(kind)
    if j is None:
        return "%s_k%d" % (kind, int(k))
    if k is None:
        return "%s_j%d" % (kind, int(j))
    return "%s_k%d_j%d" % (kind, int(k), int(j))


SEEDED_ARM_KINDS = ("ours16", "ours_quota16", "nq16", "mis16", "shuffled16")
FLAT_ARM_KINDS = ("no_frames", "uniform@8", "uniform@16", "uniform@32", "uniform@16_repeat",
                  "clip_top16", "its16", "maxvol16", "sigma_calib@0.2", "sigma_calib@0.4")
RANDOM_DRAWS = 3


def declared_arm_keys(n_seeds=3, random_draws=RANDOM_DRAWS):
    """Every arm key this run declares, produced by arm_key only."""
    keys = [arm_key(k) for k in FLAT_ARM_KINDS]
    for k in range(int(n_seeds)):
        for kind in SEEDED_ARM_KINDS:
            keys.append(arm_key(kind, k=k))
        for j in range(int(random_draws)):
            keys.append(arm_key("random@16", k=k, j=j))
    return keys


def s_keys_check(records, accuracy_per_arm, keys, n_questions):
    """S-keys: every declared arm key must appear in every per-question record AND in
    accuracy_per_arm with n_units == the number of questions of this run."""
    missing_in_records, missing_in_summary, bad_n = {}, {}, {}
    for key in keys:
        miss = [r.get("_qid") for r in records if key not in (r.get("letters") or {})]
        if miss:
            missing_in_records[key] = miss[:5] + (["..."] if len(miss) > 5 else [])
        entry = (accuracy_per_arm or {}).get(key)
        if entry is None:
            missing_in_summary[key] = True
        elif int(entry.get("n_units") or -1) != int(n_questions):
            bad_n[key] = entry.get("n_units")
    ok = not (missing_in_records or missing_in_summary or bad_n)
    return {"self_check": "S-keys", "declared_keys": list(keys), "n_declared_keys": len(keys),
            "n_questions": int(n_questions), "key_constructor": "r3_common.arm_key(kind, k, j)",
            "missing_in_records": missing_in_records, "missing_in_accuracy_per_arm": missing_in_summary,
            "wrong_n_units": bad_n, "passes": bool(ok),
            "rule": ("any missing key fails the run AFTER all declared outputs are written, and "
                     "the failing keys are named in metrics")}


# ================================================================ k-relative PC1 (three states)
def k_relative_pc1(acc, n, k, max_single_letter=None, label="control set"):
    """PC1 with the r3 k-relative pass / unmet / construction-failure bands.

    pass = acc <= 1.4/k ; unmet band = (1.4/k, 2.5/k) ; construction failure = acc >= 2.5/k.
    All three are separately decidable branches; the last two are both crash_positive_control.
    Mandatory companions: acc, n, k, 1/k, the excess ratio acc/(1/k), the exact binomial 95% CI
    and the maximum single-letter baseline.
    """
    import scipy.stats as st
    k = int(k)
    chance = 1.0 / float(k)
    pass_line = 1.4 * chance
    fail_line = 2.5 * chance
    hits = None
    ci = [None, None]
    if acc is not None and n:
        hits = int(round(float(acc) * int(n)))
        lo, hi = st.binomtest(hits, int(n), chance).proportion_ci(confidence_level=0.95,
                                                                 method="exact")
        ci = [float(lo), float(hi)]
    state = None
    if acc is not None:
        if float(acc) <= pass_line:
            state = "pass"
        elif float(acc) < fail_line:
            state = "unmet_band"
        else:
            state = "construction_failure"
    return {"control": "PC1 no-frames floor, k-relative (r3 rewrite of record defect)",
            "set": label, "n_units": (int(n) if n else None), "unit": "question",
            "acc": (None if acc is None else float(acc)), "k": k, "chance_1_over_k": chance,
            "excess_ratio_acc_over_chance": (None if acc is None else float(acc) / chance),
            "pass_line_1.4_over_k": pass_line, "fail_line_2.5_over_k": fail_line,
            "exact_binomial_95_ci": ci, "hits": hits,
            "max_single_letter_baseline": max_single_letter,
            "state": state,
            "three_states_are_separate_branches": True,
            "holds": bool(state == "pass"),
            "crash_positive_control": bool(state in ("unmet_band", "construction_failure")),
            "rule": ("pass = acc <= 1.4/k ; unmet band = (1.4/k, 2.5/k) ; construction failure = "
                     "acc >= 2.5/k (k=4 -> 0.35 / (0.35, 0.625) / 0.625); the band and the failure "
                     "state are BOTH labelled crash_positive_control and exit non-zero"),
            "precedent_note": ("BOLT's Video-MME Blind Test 40.85 on five options is 2.04/k, so it "
                               "falls in the unmet band and is no longer read as 'the control set "
                               "was built wrong'")}


def pc2_gain_sign(readings):
    """PC2a / PC2b share this: gain and the one-sided sign test over the paired units, plus an
    exact McNemar on those same units.

    `readings` is a list of (correct_on_first, correct_on_second) booleans, one per unit.
    """
    import scipy.stats as st
    units = [(bool(a), bool(b)) for a, b in readings]
    n = len(units)
    if not n:
        return {"n_units": 0, "gain": None, "sign_test": {"n_win": 0, "n_loss": 0,
                                                          "p_one_sided_greater": 1.0},
                "mcnemar": {"n01": 0, "n10": 0, "p": 1.0}, "holds": False}
    a = np.asarray([u[0] for u in units], float)
    b = np.asarray([u[1] for u in units], float)
    gain = float(a.mean() - b.mean())
    win = int(np.sum((a > 0) & (b == 0)))
    loss = int(np.sum((b > 0) & (a == 0)))
    p = float(st.binomtest(win, win + loss, 0.5, alternative="greater").pvalue) if (win + loss) else 1.0
    return {"n_units": n, "gain": gain, "acc_first": float(a.mean()), "acc_second": float(b.mean()),
            "sign_test": {"n_win": win, "n_loss": loss, "p_one_sided_greater": p},
            "mcnemar": C.mcnemar_exact(np.asarray([u[0] for u in units], bool),
                                       np.asarray([u[1] for u in units], bool)),
            "threshold_gain": 0.15, "threshold_sign_p": 0.05,
            "holds": bool(gain >= 0.15 and p < 0.05)}


def pc2a_block(records, arm_a="uniform@16", arm_b="sigma_calib@0.2",
               label="control set, visible family"):
    out = pc2_gain_sign([(r["letters"].get(arm_a) == r["gold"], r["letters"].get(arm_b) == r["gold"])
                         for r in records])
    out.update({"control": "PC2a destroying all 16 frames costs accuracy", "set": label,
                "arms": [arm_a, arm_b],
                "crash_if_not": "the control set is not known to need frames"})
    return out


def pc2b_block(pairs, arm_a="visible@16", arm_b="placebo@16",
               label="control set, the 24 twin pairs"):
    """PC2b, paired over the twin pairs (the pair is the unit)."""
    out = pc2_gain_sign([(p["correct_visible"], p["correct_placebo"]) for p in pairs])
    out.update({"control": "PC2b visibility contrast on the twin pairs", "set": label,
                "arms": [arm_a, arm_b], "unit": "pair",
                "note": "the same thresholds as PC2a; the exact McNemar is over the 24 pairs",
                "crash_if_not": "an answer that is verifiably inside the pool is not measurable"})
    return out


def pc3_block(tvs, label="control set, visible family"):
    s = R.tv_stats(tvs)
    s.update({"control": "PC3 the control set's own instrument response",
              "set": label, "threshold_median": 0.15,
              "holds": bool(s["median_tv"] is not None and s["median_tv"] >= 0.15),
              "crash_if_not": "the instrument does not move on samples known to need frames"})
    return s


# ================================================================ G2_runseed (r2 promise, never produced)
def g2_runseed_block(model, proc, dev, ids, arr, u16, prompt, q_index, fwd, run_seeds, p_original,
                     max_pixels=None):
    """One extra whole-battery forward per question per internal run seed: the question's 16 fed
    frames are all replaced by sigma=0.2 distorted versions from THAT seed's s_noise stream, and
    the TV is taken against the question's original 16-frame uniformity distribution.

    The gate reading G2 still uses the fixed pre-screen stream; this one is reported beside it,
    never pooled with it (r3 setup section 5).
    """
    max_pixels = G.MAX_PIXELS if max_pixels is None else max_pixels
    per_seed = {}
    for s in run_seeds:
        k = int(s["k"])
        dp = [C.to_pil(G.distort_seeded(arr[i], G.SIGMA_MAIN, s["s_noise"], q_index, i,
                                        G.SIGMA_INDEX)[None, :, :, :], max_pixels)[0] for i in u16]
        p = G.option_probs_multi(model, proc, dp, prompt, dev, ids, counter=fwd,
                                 kind=arm_key("p_alldist_runseed", k=k))
        per_seed[str(k)] = {"tv_vs_original_u16": G.tv(p_original, p),
                            "p_alldist_runseed": [round(float(x), 6)
                                                  for x in np.asarray(p).ravel()],
                            "s_noise": int(s["s_noise"])}
    return {"per_seed": per_seed, "n_seeds": len(run_seeds),
            "unit": "question", "note": ("run-seed-side robustness of the instrument; the gate "
                                         "reading G2 uses the fixed pre-screen stream")}


def g2_runseed_summary(per_question, n_seeds=3):
    """Per seed and per question, the median per seed, and the cross-seed range = max - min of
    the medians.  This is the reading the r2 design promised and no r2 run produced."""
    ks = [str(k) for k in range(int(n_seeds))]
    by_seed = {}
    for k in ks:
        vals = [q["per_seed"][k]["tv_vs_original_u16"] for q in per_question
                if q.get("per_seed") and k in q["per_seed"]]
        by_seed[k] = {"median_tv": (float(np.median(vals)) if vals else None),
                      "mean_tv": (float(np.mean(vals)) if vals else None),
                      "count_ge_bar": int(sum(1 for v in vals if v >= G.G2_MEDIAN_TV_BAR)),
                      "n_units": len(vals),
                      "quantiles": ({("p%d" % p): float(x) for p, x in
                                     zip(R.QUANTILES, np.percentile(vals, R.QUANTILES))}
                                    if vals else None),
                      "per_question_tv": vals}
    meds = [v["median_tv"] for v in by_seed.values() if v["median_tv"] is not None]
    return {"per_seed": by_seed,
            "cross_seed_range_max_minus_min_of_medians": (max(meds) - min(meds) if meds else None),
            "gate_reading_stream": "fixed pre-screen stream s_pre=%d (r2_common)" % R.S_PRE_STREAM,
            "run_seed_reading_stream": "each run seed's own s_noise stream",
            "n_units": len(per_question), "unit": "question"}


# ================================================================ P4 applicability + S-p4
ARCHIVE_TASK = G.PINNED_SOURCE_TASK            # task-ca0a24e61b
ARCHIVE_PATH = "sources/%s/results/mvp2_samples.jsonl" % ARCHIVE_TASK


def p4_applicability(cells, archive_path=ARCHIVE_PATH):
    """P4's applicability COMPUTED from this run's cell matrix and the mounted archive.

    applicable <=> this run has a carrier-1 x backbone-1 cell AND the archive's declared output
    is mounted.  Never a hard-coded constant (that was r2's defect).
    """
    has_cell = any((c.get("carrier") == "lvb") and (c.get("backbone") == "qwen2vl")
                   for c in (cells or []))
    mounted = bool(archive_path) and os.path.exists(archive_path)
    return {"applicable": bool(has_cell and mounted),
            "has_carrier1_backbone1_cell": bool(has_cell),
            "archive_mounted": bool(mounted), "archive_path": archive_path,
            "rule": ("applicable <=> a carrier-1 x backbone-1 cell is in this run's cell matrix "
                     "AND the archive output is mounted; computed, never hard-coded")}


def p4_run(per_question_letters, archive_path=ARCHIVE_PATH, max_records=None):
    """P4 itself: compare this run's carrier-1 x backbone-1 uniform@16 per-question letters with
    the archive's per-question prediction field.  The field name is DISCOVERED from the archive's
    own keys and reported; if there is none the call fails loudly."""
    if not os.path.exists(archive_path):
        raise SystemExit("P4: applicable but the archive is not mounted at %s" % archive_path)
    recs = []
    with open(archive_path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if max_records:
        recs = recs[:int(max_records)]
    if not recs:
        raise SystemExit("P4: the archive %s has no records" % archive_path)
    keys = sorted(set().union(*[set(r.keys()) for r in recs]))
    wanted = ("pred_uniform", "pred_uniform16", "pred_uniform@16", "uniform16", "letter_uniform",
              "pred_letter_uniform", "pred")
    cand = [k for k in keys if k in wanted]
    if not cand:
        raise SystemExit("P4: no per-question prediction field found in the archive keys: %r" % keys)
    field = cand[0]
    by_qid = {}
    for r in recs:
        qid = r.get("qid") or r.get("_qid") or r.get("question_id")
        if qid is None:
            raise SystemExit("P4: archive record without a question identifier: keys %r" % keys)
        by_qid[qid] = r.get(field)
    agree, mismatch = 0, []
    for qid, letter in per_question_letters.items():
        got = by_qid.get(qid)
        if got == letter:
            agree += 1
        else:
            mismatch.append({"qid": qid, "run_letter": letter, "archive_pred": got})
    return {"self_check": "P4 archive reproduction", "archive_path": archive_path,
            "field_discovered": field, "field_candidates_from_archive_keys": cand,
            "archive_keys": keys, "n_archive_records": len(recs), "n_units": len(per_question_letters),
            "agree": agree, "mismatches": mismatch,
            "agreement_str": "%d/%d" % (agree, len(per_question_letters)),
            "passes": bool(agree == len(per_question_letters) and len(per_question_letters) > 0)}


def s_p4(cells, produced, archive_path=ARCHIVE_PATH):
    """S-p4: assert no applicability/pass field is a constant, and fail loudly when the run has a
    carrier-1 x backbone-1 cell and the archive is not mounted."""
    fresh = p4_applicability(cells, archive_path)
    # prove the field is a function of the inputs, not a constant: probe both branches of the
    # archive-mount condition with a throwaway file inside our own run directory
    tmpd = os.environ.get("TMPDIR", ".") or "."
    os.makedirs(tmpd, exist_ok=True)
    fd, probe_path = tempfile.mkstemp(prefix="s_p4_probe_", dir=tmpd)
    os.close(fd)
    try:
        with_cell_mounted = p4_applicability([{"carrier": "lvb", "backbone": "qwen2vl"}], probe_path)
        without_cell_mounted = p4_applicability([], probe_path)
        with_cell_unmounted = p4_applicability([{"carrier": "lvb", "backbone": "qwen2vl"}],
                                              probe_path + ".absent")
    finally:
        try:
            os.remove(probe_path)
        except OSError:
            pass
    branch_values = [with_cell_mounted["applicable"], without_cell_mounted["applicable"],
                     with_cell_unmounted["applicable"]]
    not_a_constant = bool(len(set(branch_values)) > 1)
    agrees = (produced is None) or (bool(produced.get("applicable")) == fresh["applicable"])
    return {"self_check": "S-p4", "recomputed": fresh,
            "produced": (None if produced is None else {k: produced.get(k) for k in
                                                        ("applicable", "has_carrier1_backbone1_cell",
                                                         "archive_mounted")}),
            "recomputation_agrees_with_produced": bool(agrees),
            "branch_probe": {"has_cell_and_archive_mounted": with_cell_mounted["applicable"],
                             "no_cell_archive_mounted": without_cell_mounted["applicable"],
                             "has_cell_archive_unmounted": with_cell_unmounted["applicable"]},
            "not_a_constant": bool(not_a_constant),
            "hard_failure": bool(fresh["has_carrier1_backbone1_cell"] and not fresh["archive_mounted"]),
            "passes": bool(agrees and not_a_constant
                           and not (fresh["has_carrier1_backbone1_cell"]
                                    and not fresh["archive_mounted"])),
            "hard_failure_rule": ("a run with a carrier-1 x backbone-1 cell and no mounted archive "
                                  "exits non-zero")}


# ================================================================ S-readings
S_READINGS_ITEMS = (
    {"key": "per_cell_deviation_table",
     "where": "metrics.cells[].deviations_from_predictions",
     "source": "r2 data_analysis section 7 (promised, never landed in r2)"},
    {"key": "label_used_in_scoring_false",
     "where": "every per-question record carries label_used_in_scoring == false",
     "source": "r2 setup section 6 (promised, never landed in r2)"},
    {"key": "record_defect_random16",
     "where": "metrics.record_defect_readings.random16",
     "source": "r2 review section 6 main item 3a: the arm-key defect, hence per-draw values + mean"},
    {"key": "record_defect_p4",
     "where": "metrics.record_defect_readings.p4",
     "source": "r2 review section 6 main item 3b: the hard-coded P4 applicability"},
    {"key": "record_defect_g2_runseed",
     "where": "metrics.record_defect_readings.g2_runseed",
     "source": "r2 review section 6 main item 3c: the run-seed G2 robustness reading"},
)


def s_readings(metrics, records, scope="full_battery"):
    """S-readings: the r3 design's mandatory reading list must be present and non-empty in
    metrics and in the per-question records.

    A reading that this run structurally cannot produce (a model-based one in a no-model run)
    must still be PRESENT, non-empty, and carry measured=false with the reason -- never absent.
    """
    items = {}
    cells = metrics.get("cells") or []
    dev_tables = [c.get("deviations_from_predictions") for c in cells]
    items["per_cell_deviation_table"] = {
        "present": bool(dev_tables) and all(bool(t) for t in dev_tables),
        "n_cells": len(cells), "n_items": [len(t or []) for t in dev_tables]}
    lab = [r.get("label_used_in_scoring") for r in records]
    items["label_used_in_scoring_false"] = {
        "present": bool(records) and all(v is False for v in lab),
        "n_records": len(records), "n_false": int(sum(1 for v in lab if v is False)),
        "n_true": int(sum(1 for v in lab if v is True)),
        "n_missing": int(sum(1 for v in lab if v is None))}
    rd = metrics.get("record_defect_readings") or {}
    for key in ("random16", "p4", "g2_runseed"):
        d = rd.get(key)
        items["record_defect_" + key] = {
            "present": bool(d), "non_empty": bool(d),
            "measured": (None if not d else bool(d.get("measured"))),
            "not_measured_reason": (None if not d else d.get("not_measured_reason"))}
    all_present = all(v.get("present") for v in items.values())
    return {"self_check": "S-readings", "scope": scope, "required_readings": list(S_READINGS_ITEMS),
            "items": items, "passes": bool(all_present),
            "rule": ("every declared mandatory reading must be present and non-empty; a reading a "
                     "no-model run cannot produce stays present with measured=false and a reason")}


# ================================================================ predictions this dispatch checks
R3_PREDICTIONS = (
    {"name": "n_pairs_built", "prediction": "24 twin pairs (12 needle + 12 plotQA)",
     "expected": [24, 24], "unit": "pair"},
    {"name": "n_visible_after_mechanical_exclusion", "prediction": ">= 20 (design setup 3.4)",
     "expected": [20, 24], "unit": "question"},
    {"name": "n_placebo_after_mechanical_exclusion", "prediction": ">= 12 (design setup 3.4)",
     "expected": [12, 24], "unit": "question"},
    {"name": "i_star_in_I16u", "prediction": "every visible question's i* is one of the 16 fed frames",
     "expected": [24, 24], "unit": "question"},
    {"name": "placebo_gap_ge_0.5s", "prediction": "every placebo j* is >= 0.5 s from every pool "
                                                  "timestamp", "expected": [24, 24], "unit": "question"},
    {"name": "V2_marker_area_ge_0.02", "prediction": "marker area fraction >= 2% at the model's "
                                                     "own resolution", "expected": [24, 24],
     "unit": "question"},
    {"name": "V2_min_stroke_ge_3px", "prediction": "smallest stroke >= 3 px at the model's own "
                                                   "resolution", "expected": [24, 24], "unit": "question"},
    {"name": "V5_distractors_zero_everywhere", "prediction": "each distractor matches 0 pixels in "
                                                             "every pool frame", "expected": [24, 24],
     "unit": "question"},
    {"name": "V6_correct_letter_balance", "prediction": "each of the 4 letters is correct exactly "
                                                        "6 times", "expected": [24, 24],
     "unit": "question"},
)
