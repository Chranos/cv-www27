#!/usr/bin/env python3
"""r3_gen_pin.py -- the pinned control-set generator of Plan plan-matrix-longvideo-r3@v1.

Pure arithmetic, ZERO decoding, ZERO models, independent of EXP_SEED.  Everything it selects
comes from

  * the new pinned constant R3_PIN_SEED = 20261017, through gen_common.seed_streams (the
    project's one seed-derivation function) plus a per-family offset (r3_common.R3_FAMILY_OFFSET);
  * data/carrier3b_pin.json, which already pins the bytes and sha256 of the 40 videos of the
    device root mlvu-needle-plotqa-38e93f060520;
  * the container HEADER metadata (frame count, fps) of the 24 selected videos -- read from the
    container index, no frame is decoded -- which is what makes the slot-boundary placebo index
    j* = floor((i*+1) * n / 32) concrete.  Called without it, the same pin is produced with
    j_star = null and j_star_rule pinned; those are the only fields that differ.

What it pins per pair: the video filename, its sha256 and bytes, the evidence frame index i*,
the placebo frame index j*, the marker spec (colour, shape, size, position, tolerance), the four
options and the correct index; plus the 6/6/6/6 correct-letter balance over the 24 visible
questions.  Output is canonical JSON (sort_keys, compact).

Usage (all optional; the run calls the functions directly):
  r3_gen_pin.py --write data/carrier3b_pin_r3.json --probe results/pin_probe.json
  r3_gen_pin.py --check data/carrier3b_pin_r3.json --expect-sha256 <sha>
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

import r3_common as R3

PINS_FILE = os.environ.get("R3_C3B_PIN", "data/carrier3b_pin.json")
DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/data")
C3B_DIR = os.environ.get("C3B_DIR", os.path.join(DATA_ROOT, "mlvu-needle-plotqa-38e93f060520"))


def load_pinned_items(path=PINS_FILE):
    """The pinned item table of the device root (order preserved: it is the pinned order)."""
    pin = json.load(open(path))
    items = pin["items"]
    by_fam = {}
    for it in items:
        by_fam.setdefault(it["task"], []).append(it)
    return pin, items, by_fam


def check_pinned_question_sets(pin, items, by_fam):
    """Refuse to start if the count/order of the pinned question sets is not what the design
    pinned: 40 items, 20 per family, task order and per-family order exactly as pinned."""
    out = {"n_items_pinned": len(items),
           "family_counts": {k: len(v) for k, v in by_fam.items()},
           "pinned_order_first5": [(it["task"], it["video"]) for it in items[:5]],
           "question_set_sha256_pin": pin.get("question_set_sha256"),
           "n_files_pin": pin.get("n_files"), "manifest_sha256_pin": pin.get("manifest_sha256"),
           "matches_design": None}
    ok = (len(items) == 40 and by_fam.get("2_needle") and by_fam.get("1_plotQA")
          and len(by_fam["2_needle"]) == 20 and len(by_fam["1_plotQA"]) == 20
          and set(by_fam) == {"2_needle", "1_plotQA"})
    out["matches_design"] = bool(ok)
    if not ok:
        raise SystemExit("pinned question sets differ from the design's: %r" % out)
    return out


def container_metadata(videos, c3b_dir=C3B_DIR):
    """Container HEADER metadata only: frame count and fps.  No frame is decoded."""
    import cv2
    out = {}
    for fam, it in videos:
        path = os.path.join(c3b_dir, "videos", it["task"], it["video"])
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                raise SystemExit("container metadata: cv2 cannot open %s" % path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()
        out[it["video"]] = {"n_frames": n, "fps": fps, "path": path}
    return out


def probe_files(c3b_dir=C3B_DIR):
    """Per-file sha256 and bytes of the 24 selected videos, checked against the pin's table."""
    pin, items, by_fam = load_pinned_items()
    sel = R3.select_videos(items)
    out = {}
    for fam in R3.R3_PAIR_FAMILY_ORDER:
        for it in sel[fam]:
            p = os.path.join(c3b_dir, "videos", it["task"], it["video"])
            if not os.path.exists(p):
                raise SystemExit("missing selected video %s" % p)
            got = R3.G.sha256_file(p)
            if got != it["sha256"] or os.path.getsize(p) != int(it["bytes"]):
                raise SystemExit("selected video %s does not match the pinned bytes/sha256" % p)
            out[it["video"]] = {"sha256": got, "bytes": os.path.getsize(p)}
    return out


def make_pin(with_metadata=True, c3b_dir=C3B_DIR):
    pin, items, by_fam = load_pinned_items()
    check_pinned_question_sets(pin, items, by_fam)
    if not with_metadata:
        return R3.build_pin(items, n_frames=None, fps=None), {}
    sel = R3.select_videos(items)
    videos = [(fam, it) for fam in R3.R3_PAIR_FAMILY_ORDER for it in sel[fam]]
    meta = container_metadata(videos, c3b_dir=c3b_dir)
    n_frames = {k: v["n_frames"] for k, v in meta.items()}
    fps = {k: v["fps"] for k, v in meta.items()}
    return R3.build_pin(items, n_frames=n_frames, fps=fps), meta


def pin_probe(pin, meta=None, extra=None):
    """The small probe file: the pin's sha256, its byte size, the rule summary and a compact
    per-pair table."""
    sha, nbytes = R3.pin_sha256_and_bytes(pin)
    rows = [{"pair": p["pair"], "family": p["family"], "video": p["video"],
             "sha256": p["sha256"], "bytes": p["bytes"], "i_star": p["i_star"], "j_star": p["j_star"],
             "n_frames_source": p["n_frames_source"], "fps": p["fps"], "gap_s": p["gap_s"],
             "colour": p["marker"]["colour_key"], "shape": p["marker"]["shape_key"],
             "correct_index": p["correct_index"], "correct_letter": p["correct_letter"]}
            for p in pin["pairs"]]
    out = {"pin_id": "data/carrier3b_pin_r3.json",
           "pin_sha256": sha, "pin_bytes": nbytes,
           "pin_canonical_form": "json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False)",
           "generator": "r3_gen_pin.py / r3_common.build_pin",
           "R3_PIN_SEED": R3.R3_PIN_SEED,
           "R3_FAMILY_OFFSET": R3.R3_FAMILY_OFFSET,
           "stream": R3.r3_streams(),
           "n_pairs": pin["n_pairs"], "n_questions": pin["n_questions"],
           "correct_letter_counts": pin["correct_letter_counts"],
           "selection": {k: v for k, v in pin["selection"].items()},
           "per_pair": rows,
           "j_star_rule": "floor((i_star+1) * n_frames_source / 32)",
           "marker_rule": ("colour (c0+o)%4 and shape (s0+o)%4 for o=0..3; the correct option is "
                           "the drawn combo (o=0) and the presentation order puts it at position "
                           "correct_index"),
           "aggregate_targets": {"n_visible_min": 20, "n_placebo_min": 12},
           "metadata_source": ("cv2 container header metadata (frame count, fps); no frame decoded"
                               if meta else "not read (arithmetic-only form)")}
    if extra:
        out.update(extra)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default=None, help="write the canonical pin here")
    ap.add_argument("--probe", default=None, help="write the probe JSON here")
    ap.add_argument("--check", default=None, help="regenerate and compare with this pin file")
    ap.add_argument("--expect-sha256", default=None)
    ap.add_argument("--no-metadata", action="store_true",
                    help="pure arithmetic form: j_star = null, j_star_rule pinned")
    args = ap.parse_args(argv)

    pin, meta = make_pin(with_metadata=not args.no_metadata)
    sha, nbytes = R3.pin_sha256_and_bytes(pin)
    print("pin sha256=%s bytes=%d pairs=%d" % (sha, nbytes, pin["n_pairs"]))
    print("correct letter counts: %r" % (pin["correct_letter_counts"],))
    if args.write:
        d = os.path.dirname(args.write)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.write, "wb") as f:
            f.write(R3.canonical_json(pin))
        got = R3.G.sha256_file(args.write)
        print("wrote %s (%d bytes, file sha256=%s)" % (args.write, os.path.getsize(args.write), got))
        if got != sha:
            raise SystemExit("written pin file's sha256 != the pin's canonical sha256")
    if args.probe:
        d = os.path.dirname(args.probe)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.probe, "w") as f:
            json.dump(pin_probe(pin, meta), f, indent=1, sort_keys=True, ensure_ascii=False)
        print("wrote %s" % args.probe)
    if args.check or args.expect_sha256:
        want = args.expect_sha256
        if args.check:
            if not os.path.exists(args.check):
                raise SystemExit("committed pin not found: %s" % args.check)
            got_file = R3.G.sha256_file(args.check)
            if want is not None and got_file != want:
                raise SystemExit("committed pin sha256 %s != expected %s" % (got_file, want))
            want = got_file
            if os.path.getsize(args.check) != nbytes:
                raise SystemExit("committed pin byte size %d != regenerated %d"
                                 % (os.path.getsize(args.check), nbytes))
            if open(args.check, "rb").read() != R3.canonical_json(pin):
                raise SystemExit("committed pin file is present but NOT byte-identical to the "
                                 "regenerated pin")
        if want is not None and sha != want:
            raise SystemExit("regenerated pin sha256 %s != committed %s" % (sha, want))
        print("pin check OK: %s" % sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
