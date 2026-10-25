"""Backbone probe: does the second backbone load and answer through the SAME call shapes the
four readings use, and what does one unit of each call cost?

This writes NO reading.  It is a plumbing + timing run:
  1. load the named backbone through paper4/backbone.py (architecture from the checkpoint);
  2. letter token ids (assert single-token per letter);
  3. decode a 32-frame pool for one carrier-1 question (the pool the selector scores);
  4. time option_logits(..., batch=4) over 8 single-image prompts -- the call the frame
     selector makes 128 times per question;
  5. time one 16-frame answer, one 1-frame answer, one 0-frame answer -- the three answer
     shapes every cell uses;
  6. determinism control: repeat the same 16-frame call on two questions, cell for cell;
  7. turn the measured rates into the card-hour estimate for the four full cells.

Everything it writes is tagged probe_only: true and decides_claim: false.
"""
import json
import os
import time

import numpy as np

import backbone as BK
import gen_common as G
import mvp_common as C

NAME = os.environ.get("P4_BACKBONE", "llavaov").strip()
MODE = os.environ.get("P4_MODE", "pilot").strip()
PLAN_ID = os.environ.get("P4_PLAN_ID", "").strip()
PLAN_VERSION = int(os.environ.get("P4_PLAN_VERSION", "1"))
EXP_SEED = int(os.environ.get("EXP_SEED", "0"))
DEEP = os.environ.get("P4_DEEP_IDENTITY", "0") == "1"
DATA = os.environ.get("DATA_ROOT", "/path/to/data")
LVB_DIR = os.path.join(DATA, "lvb-val-e383ac257332")

FWD = G.Fwd()
T0 = time.time()


def log(*a):
    print(*a, flush=True)


def main():
    log("probe backbone=%s mode=%s plan=%s@%d seed=%d" % (NAME, MODE, PLAN_ID, PLAN_VERSION,
                                                          EXP_SEED))
    t = time.time()
    model, proc, dev, minfo = BK.load(NAME, G.MAX_PIXELS)
    load_s = time.time() - t
    ids = BK.letter_ids(proc)
    tok = proc.tokenizer
    log("loaded %s arch=%s in %.1fs; letter ids %s" % (minfo["family"], minfo["architecture"],
                                                       load_s, ids))

    pool = C.load_pool(LVB_DIR)
    qs = sorted(pool, key=lambda q: q["_qid"])[:2]
    out = {"probe_only": True, "decides_claim": False, "backbone": NAME, "mode": MODE,
           "plan_id": PLAN_ID, "plan_version": PLAN_VERSION, "exp_seed": EXP_SEED,
           "model": minfo, "model_identity": BK.identity(NAME, deep=DEEP),
           "letter_token_ids": dict(zip("ABCDE", ids)),
           "letter_tokens_decode": {L: tok.convert_ids_to_tokens(i) for L, i in
                                    zip("ABCDE", ids)},
           "carriers": {"carrier1": LVB_DIR, "pool_questions": len(pool),
                        "pool_videos": len({q["video_id"] for q in pool})},
           "timings": {}, "answers": {}, "determinism": {}, "cost_model": {}}

    # ---- 1. 32-frame pool + selector scoring shape --------------------------------
    q0 = qs[0]
    t = time.time()
    arr, fps, nfr = C.decode_frames(q0["_video_path"], G.K_CAND)
    pils = C.to_pil(arr, G.MAX_PIXELS)
    dec_s = time.time() - t
    log("decoded %d frames (%dx%d) fps=%.2f n=%d in %.2fs"
        % (len(arr), arr.shape[1], arr.shape[2], fps, nfr, dec_s))

    prompt0 = C.prompt_for(q0)
    t = time.time()
    lo = C.option_logits(model, proc, pils[:8], prompt0, dev, ids, batch=4)
    score8_s = time.time() - t
    FWD.count("probe:score_single_image", 8)
    log("option_logits on 8 single-image prompts (batch=4): %.2fs -> %.3fs/image"
        % (score8_s, score8_s / 8.0))

    # ---- 2. the three answer shapes ----------------------------------------------
    u16 = C.uniform_indices(G.K_CAND, G.B_MAIN)
    t = time.time()
    raw16, l16 = G.answer_arm(model, proc, [pils[i] for i in u16], prompt0, dev,
                              counter=FWD, kind="probe:ans16")
    ans16_s = time.time() - t
    t = time.time()
    raw1, l1 = G.answer_arm(model, proc, [pils[0]], prompt0, dev, counter=FWD,
                            kind="probe:ans1")
    ans1_s = time.time() - t
    t = time.time()
    raw0, l0 = G.answer_arm(model, proc, [], prompt0, dev, counter=FWD, kind="probe:ans0")
    ans0_s = time.time() - t
    log("answers 16f=%.2fs raw=%r letter=%r | 1f=%.2fs raw=%r -> %r | 0f=%.2fs raw=%r -> %r"
        % (ans16_s, raw16, l16, ans1_s, raw1, l1, ans0_s, raw0, l0))
    out["answers"] = {"qid": q0["_qid"], "gold": C.correct_letter(q0),
                      "uniform@16": {"raw": raw16, "letter": l16, "seconds": round(ans16_s, 3)},
                      "single_frame": {"raw": raw1, "letter": l1, "seconds": round(ans1_s, 3)},
                      "no_frames": {"raw": raw0, "letter": l0, "seconds": round(ans0_s, 3)},
                      "option_probs_first8_head": np.round(lo[0], 4).tolist()}

    # ---- 3. determinism control: the same call twice, two questions ---------------
    det = []
    for q in qs:
        parr, _fps, _n = C.decode_frames(q["_video_path"], G.K_CAND)
        ppils = C.to_pil(parr, G.MAX_PIXELS)
        pr = C.prompt_for(q)
        sel = [ppils[i] for i in u16]
        a_raw, a_l = G.answer_arm(model, proc, sel, pr, dev, counter=FWD,
                                  kind="probe:det16")
        b_raw, b_l = G.answer_arm(model, proc, sel, pr, dev, counter=FWD,
                                  kind="probe:det16")
        det.append({"qid": q["_qid"], "first": {"raw": a_raw, "letter": a_l},
                    "second": {"raw": b_raw, "letter": b_l},
                    "agree": bool(a_l == b_l and a_raw == b_raw)})
    log("determinism %s" % [d["agree"] for d in det])
    out["determinism"] = {"pairs": det, "n": len(det),
                          "n_agree": int(sum(1 for d in det if d["agree"])),
                          "note": "same inputs, same call, greedy decode, one GPU; this is the "
                                  "instrument control that replaces the archive-identity control "
                                  "(which is backbone-specific and cannot apply here)"}

    # ---- 4. card-hour estimate for the four full cells ---------------------------
    t_img = score8_s / 8.0
    cell_units = {"P1_selection": 110, "P2_single": 24, "P3_distortion": 17, "P4_donor": 23}
    per_unit = {
        "P1_selection": {"decode_32frames_s": dec_s, "score_single_image_calls": 128,
                         "score_s_per_unit": 128 * t_img,
                         "answer16_calls": 4, "answer16_s_per_unit": 4 * ans16_s,
                         "answer0_calls": 1, "answer0_s_per_unit": ans0_s},
        "P2_single": {"decode_s_per_unit": dec_s, "answer16_calls": 1,
                      "answer16_s_per_unit": ans16_s, "answer1_calls": 1,
                      "answer1_s_per_unit": ans1_s, "answer0_calls": 1,
                      "answer0_s_per_unit": ans0_s},
        "P3_distortion": {"decode_s_per_unit": dec_s, "answer1_calls": 13,
                          "answer1_s_per_unit": 13 * ans1_s},
        "P4_donor": {"decode_s_per_unit": dec_s, "answer16_calls": 1,
                     "answer16_s_per_unit": ans16_s, "answer0_calls": 1,
                     "answer0_s_per_unit": ans0_s},
    }
    cells = {}
    for cell, n in cell_units.items():
        pu = per_unit[cell]
        s_per_unit = sum(v for k, v in pu.items() if k.endswith("_s_per_unit")) + \
            pu.get("decode_32frames_s", pu.get("decode_s_per_unit", 0.0))
        cells[cell] = {"units": n, "seconds_per_unit": round(s_per_unit, 3),
                       "seconds_total": round(s_per_unit * n, 1),
                       "minutes_total": round(s_per_unit * n / 60.0, 2)}
    total_s = sum(c["seconds_total"] for c in cells.values()) + 4 * load_s
    out["timings"] = {"model_load_s": round(load_s, 2), "decode_32frames_s": round(dec_s, 3),
                      "option_logits_8_single_images_s": round(score8_s, 3),
                      "option_logits_s_per_image": round(t_img, 4),
                      "answer_16frames_s": round(ans16_s, 3),
                      "answer_1frame_s": round(ans1_s, 3),
                      "answer_0frames_s": round(ans0_s, 3)}
    out["cost_model"] = {"per_unit": per_unit, "cells": cells,
                         "seconds_total_all_cells_plus_4_loads": round(total_s, 1),
                         "card_hours_all_cells": round(total_s / 3600.0, 3),
                         "note": "measured rates x the fixed unit counts of the four cells; "
                                 "excludes per-run process start and file IO"}
    out["forward_counts"] = FWD.report()
    out["elapsed_seconds"] = round(time.time() - T0, 1)
    with open("metrics.json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
    log("probe done elapsed=%.1fs forwards=%d card_hours_estimate=%.3f"
        % (time.time() - T0, FWD.report()["total"], total_s / 3600.0))


if __name__ == "__main__":
    main()
