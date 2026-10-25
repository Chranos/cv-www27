"""Exploratory, post-hoc addition (2): step timing for the selection cell's existing
arms.

Plan: plan-exploratory-bolt-timing (stage=smoke, decides_claim=false,
exploratory, single device, single seed).

This re-executes the registered selection cell's five arms on the same 110 questions
(uniform@16, contrast@16_k0/k1/k2, no_frames; constant prior is 0 forwards), with the
same forwards structure as the archived full run evidence.task-23cd232b2b
(550 answer forwards + 14080 frame-scoring forwards), adding per-step wall timers:

  model load | video decode (32-candidate pool, cv2) | PIL prep | scoring
  (option-contrast: clean@32 + per-stream distorted@32 + distortion generation)
  | selection (top_b) | answer forward (per arm)

The letters and selected sets are compared per question against the archived run
(identity record, not a gate). The timing rows are the deliverable; no reading of the
cell is produced here and nothing enters the registered families.

Timer method: time.time() around each step, inside the run; the device is not isolated
(confined=false), so these are self-reported wall times on one A100 with nothing else
running. They follow BOLT Table 11's reporting STYLE only; absolute seconds are not
comparable to that table (different backbone, carrier, candidate pool, timer method).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import four  # noqa: E402
import gen_common as G  # noqa: E402
import mvp_common as C  # noqa: E402
import twin5.lv_common as L  # noqa: E402

log = four.log
app_jsonl = four.app_jsonl

LVB_JSON_SHA = "d32bf2e0cbbdb4fa2e99666a5a49b5a19bca2ceec69222c9a7d3aad567d1fa23"
N_VAL_RECORDS = 1337
VIDEOS_MANIFEST_SHA = "35dd905b810db2d5ee0f8bbee4dc22e5798ed7f4923f65ca8f420498be478ce5"
QSET_SHA = "57d5985387598a8286a47be78e7021e1433c1aaf101eaf335e0841b75de4de09"
EXPECTED_POOL_Q = 110
EXPECTED_POOL_V = 56
KS = four.KS
SIGMA = four.SIGMA


def main():
    t0 = time.time()
    st = G.seed_streams(four.EXP_SEED)
    streams = G.run_seeds(four.EXP_SEED, ks=KS)
    pool = C.load_pool(four.LVB_DIR)
    groups = {}
    for q in pool:
        groups.setdefault(q["video_id"], []).append(q)
    qs = sorted(pool, key=lambda q: q["_qid"])
    fp = C.data_fingerprint(four.LVB_DIR)
    checks = {
        "pool_110": len(pool) == EXPECTED_POOL_Q,
        "pool_56_videos": len(groups) == EXPECTED_POOL_V,
        "qset_sha_matches": C.question_set_sha(qs) == QSET_SHA,
        "lvb_json_sha_matches": fp.get("lvb_val_json_sha256") == LVB_JSON_SHA,
        "n_val_records_matches": fp.get("n_val_records") == N_VAL_RECORDS,
        "videos_manifest_sha_matches": fp.get("videos_manifest_sha256") == VIDEOS_MANIFEST_SHA,
    }
    bad = [k for k, v in checks.items() if v is False]
    if bad:
        raise SystemExit("timing structure check failed: %s" % bad)
    structure = {"pool_questions": len(pool), "pool_videos": len(groups),
                 "question_set_sha256": C.question_set_sha(qs),
                 "data_fingerprint": fp, "checks": checks,
                 "note": "same pool and same identity constants as the archived full "
                         "run evidence.task-23cd232b2b"}
    prior = four.prior_block([C.correct_letter(q) for q in qs], "timing_pool110",
                             expect_letter="C", expect_hits=31)
    model, proc, dev, minfo, ids = four.load_model()
    mident = L.model_identity()
    u16 = C.uniform_indices(G.K_CAND, G.B_MAIN)
    arm_keys = ["uniform@16", "no_frames"] + ["contrast@16_k%d" % k for k in KS]
    rows = []
    agg = {a: {"forward_s": [], "scoring_s": [], "select_s": []} for a in arm_keys}
    agg["decode_s"] = []
    agg["prep_s"] = []
    agg["distort_s"] = []
    for qi, q in enumerate(qs):
        four.TOKENS.clear()
        t = time.time()
        arr, fps, nfr = C.decode_frames(q["_video_path"], G.K_CAND)
        t_decode = time.time() - t
        t = time.time()
        pils = C.to_pil(arr, G.MAX_PIXELS)
        t_prep = time.time() - t
        four.PHASE["decode_s"] += t_decode + t_prep
        agg["decode_s"].append(t_decode)
        agg["prep_s"].append(t_prep)
        prompt = C.prompt_for(q)
        gold = C.correct_letter(q)
        t = time.time()
        sink = C.TokenSink()
        lo = C.option_logits(model, proc, pils, prompt, dev, ids, batch=4,
                             token_sink=sink)
        four.FWD.count("score:clean@32", len(pils))
        four.TOKENS["score:clean@32"] = sink.summary()
        t_score_clean = time.time() - t
        four.PHASE["forward_s"] += t_score_clean
        letters, raws, sel, ov = {}, {}, {}, {}
        timing = {"decode_s": t_decode, "prep_s": t_prep,
                  "score_clean_s": t_score_clean}
        t = time.time()
        letters["no_frames"], raws["no_frames"] = four.ask(
            model, proc, [], prompt, dev, "no_frames")
        timing["forward_no_frames_s"] = time.time() - t
        agg["no_frames"]["forward_s"].append(timing["forward_no_frames_s"])
        t = time.time()
        letters["uniform@16"], raws["uniform@16"] = four.ask(
            model, proc, [pils[i] for i in u16], prompt, dev, "uniform@16")
        timing["forward_uniform_s"] = time.time() - t
        agg["uniform@16"]["forward_s"].append(timing["forward_uniform_s"])
        t_all_stream = 0.0
        for s in streams:
            k = s["k"]
            t = time.time()
            darr = np.stack([G.distort_seeded(arr[i], SIGMA, s["s_noise"], qi, i,
                                              G.SIGMA_INDEX)
                             for i in range(len(arr))])
            t_distort = time.time() - t
            t = time.time()
            dpils = C.to_pil(darr, G.MAX_PIXELS)
            t_distort_prep = time.time() - t
            t = time.time()
            sink = C.TokenSink()
            ld = C.option_logits(model, proc, dpils, prompt, dev, ids, batch=4,
                                 token_sink=sink)
            four.FWD.count("score:distorted@32", len(dpils))
            four.TOKENS["score:distorted@32"] = sink.summary()
            t_score = time.time() - t
            four.PHASE["forward_s"] += t_score
            t = time.time()
            tv, _l1 = C.contrast_scores(lo, ld)
            key = "contrast@16_k%d" % k
            sel[key] = G.top_b(tv, G.B_MAIN)
            t_sel = time.time() - t
            t = time.time()
            letters[key], raws[key] = four.ask(model, proc,
                                               [pils[i] for i in sel[key]],
                                               prompt, dev, key)
            t_fwd = time.time() - t
            timing["stream_k%d" % k] = {
                "distort_s": t_distort, "distort_prep_s": t_distort_prep,
                "score_distorted_s": t_score, "select_s": t_sel, "forward_s": t_fwd}
            t_all_stream += t_distort + t_distort_prep + t_score + t_sel
            agg["contrast@16_k%d" % k]["forward_s"].append(t_fwd)
            agg["contrast@16_k%d" % k]["scoring_s"].append(
                t_score_clean + t_score + t_distort + t_distort_prep)
            agg["contrast@16_k%d" % k]["select_s"].append(t_sel)
            agg["distort_s"].append(t_distort + t_distort_prep)
            ov[key] = len(set(sel[key]) & set(u16)) / float(G.B_MAIN)
        letters["constant_prior"] = prior["prior_letter"]
        raws["constant_prior"] = "PRIOR:" + str(prior["prior_letter"])
        correct = {a: int(letters[a] == gold) for a in arm_keys + ["constant_prior"]}
        row = {"qid": q["_qid"], "video_id": q["video_id"], "gold": gold,
               "question_type": q.get("question_category"), "fps": fps, "n_frames": nfr,
               "letters": letters, "raw": raws, "correct": correct, "selected": sel,
               "overlap_vs_uniform": ov, "vision_tokens": dict(four.TOKENS),
               "duration_group": q.get("duration_group"), "timing_s": timing}
        rows.append(row)
        app_jsonl("results/exploratory_timing_samples.jsonl", row)
        log("timing %2d/%d %-28s %s" % (qi + 1, len(qs), q["_qid"],
                                        {k: letters[k] for k in arm_keys}))

    def _m(x):
        return float(np.mean(x)) if x else None

    per_arm = {}
    for a in arm_keys:
        per_arm[a] = {
            "forward_s_per_question_mean": _m(agg[a]["forward_s"]),
            "forward_s_total": float(sum(agg[a]["forward_s"])),
            "scoring_s_per_question_mean": _m(agg[a]["scoring_s"]),
            "selection_s_per_question_mean": _m(agg[a]["select_s"]),
            "n_questions": len(agg[a]["forward_s"]),
        }
    timing = {
        "model_load_s": four.PHASE["model_load_s"],
        "decode_s_per_question_mean": _m(agg["decode_s"]),
        "prep_s_per_question_mean": _m(agg["prep_s"]),
        "distort_s_per_question_mean": _m(agg["distort_s"]),
        "decode_s_total": float(sum(agg["decode_s"])),
        "per_arm": per_arm,
        "attribution": {
            "decode": "the 32-candidate pool is decoded once per question and shared by "
                      "every frame arm (uniform, contrast streams); no_frames does not "
                      "use it but is charged the same per-question row for comparability",
            "scoring": "contrast arms: clean@32 scoring + this stream's distortion "
                       "generation + distorted@32 scoring; uniform/no_frames: none",
            "selection": "top_b over contrast scores (contrast arms); ITS for the bolt "
                         "arm is reported in the bolt run's own metrics",
            "forward": "one greedy answer generation per arm and question",
        },
        "bolt_table11_style": (
            [{"step": "Qwen2-VL-7B load", "seconds": four.PHASE["model_load_s"],
              "per": "run"},
             {"step": "video decode (32-candidate pool, cv2)",
              "seconds": _m(agg["decode_s"]), "per": "question"}]
            + [{"step": "VLM answer forward, %s" % a,
                "seconds": per_arm[a]["forward_s_per_question_mean"], "per": "question"}
               for a in ("no_frames", "uniform@16")]
            + [{"step": "scoring (clean+distorted@32), contrast@16_k%d" % k,
                "seconds": per_arm["contrast@16_k%d" % k]["scoring_s_per_question_mean"],
                "per": "question"} for k in KS]
            + [{"step": "VLM answer forward, contrast@16_k%d" % k,
                "seconds": per_arm["contrast@16_k%d" % k]["forward_s_per_question_mean"],
                "per": "question"} for k in KS]),
        "note": "time.time() inside the run; device not isolated (confined=false), "
                "self-reported wall times on one A100 with nothing else running; "
                "BOLT Table 11 style only, absolute values not comparable to it",
    }
    ident = four.archive_identity(
        rows, "sources/task-23cd232b2b/results/four_selection_samples.jsonl",
        {a: a for a in arm_keys})
    ident["n_expected"] = len(rows)
    ident["note"] = ("re-execution check of all five arms against the archived full "
                     "run; greedy decode has reproduced archived letters exactly in "
                     "this project's history; any disagreement is reported, not gated, "
                     "and does not by itself invalidate the timing rows")
    sel_agree = {"n_compared": 0, "n_agree": 0}
    arc = four.load_jsonl("sources/task-23cd232b2b/results/four_selection_samples.jsonl") or {}
    for r in rows:
        a = arc.get(("qid", r["qid"]))
        if a is None:
            continue
        for key, vs in (r.get("selected") or {}).items():
            sel_agree["n_compared"] += 1
            if sorted(int(x) for x in vs) == sorted(int(x) for x in a["selected"][key]):
                sel_agree["n_agree"] += 1

    results = {
        "run": {},
        "exploratory": True, "post_hoc": True,
        "labels_required_in_paper": "exploratory, post-hoc, single device, single seed, "
                                    "step-timing rows only, stage=smoke",
        "claim_tested": None,
        "claim_note": "this run produces no reading of the cell; the letters below are a "
                      "re-execution by-product checked against the archive",
        "identity": {"dataset": "lvb-val-e383ac257332",
                     "re_execution_of": "evidence.task-23cd232b2b"},
        "model": four.model_block(minfo, mident),
        "structure": structure,
        "pretest_prior": prior,
        "arms": {a: four.arm_entry([r["correct"][a] for r in rows])
                 for a in arm_keys + ["constant_prior"]},
        "timing": timing,
        "identity_vs_archive": ident,
        "selected_sets_vs_archive": sel_agree,
        "extraction": four.extraction_block(rows, arm_keys),
        "n_measured": len(rows),
        "verdict": {"n_questions": len(rows),
                    "n_videos": len({r["video_id"] for r in rows}),
                    "deliverable": "per-step latency rows (timing block); no reading"},
    }
    four.finish(results, "results/exploratory_timing_samples.jsonl")
    log("timing exploratory run done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
