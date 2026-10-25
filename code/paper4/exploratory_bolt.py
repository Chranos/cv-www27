"""Exploratory, post-hoc addition (1): the official BOLT training-free selector as one
more arm (bolt_its@16) on the registered 110-question selection cell.

Plan: plan-exploratory-bolt-timing (stage=smoke, decides_claim=false,
outside the registered Holm families; single device, single seed).

What is official here (vendored byte-exact under paper4/bolt_official/, sha-checked at
run start):
  - the CLIP class of extract_feature.py (openai/clip-vit-large-patch14, fp16, autocast,
    cosine x exp(logit_scale) similarity), and
  - inverse_transform_sampling of select_frames.py (Eq. 5-7 of the paper),
  - the official LongVideoBench parameters: power 2.5, num_frames 16.
Registered deviations (all in the Plan, none hidden):
  - the score vector is computed over the cell's registered 32-frame uniform midpoint
    candidate pool (the official pipeline scores a 1 fps dense sequence; the cell pool is
    the frame universe every arm of this cell draws from);
  - the CLIP weights load from the local device directory (offline run) via a subclass
    that only replaces __init__'s from_pretrained arguments -- methods inherited
    unmodified;
  - frames are decoded by the cell's cv2 path (device has no decord), same indices the
    archived full run used;
  - the readout is the cell's Qwen2-VL-7B + prompt + greedy decode + parse_letter.
Measured in the same run, for pairing: uniform@16 (also the identity control against
evidence.task-23cd232b2b), plus the annotation-only constant prior (0 forwards).
"""
import hashlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BOLT_OFFICIAL = os.path.join(HERE, "bolt_official")
# Offline shims (ours, not vendored): the official files import decord and pandas at
# module top for code paths this arm never executes (fps=1 extraction, parquet
# loaders). The shims raise NotImplementedError if those paths are ever reached.
sys.path.insert(0, os.path.join(HERE, "offline_shims"))
sys.path.insert(0, ROOT)
sys.path.insert(0, BOLT_OFFICIAL)

import torch  # noqa: E402

import extract_feature as bolt_ef  # noqa: E402  official, verbatim
import select_frames as bolt_sf  # noqa: E402  official, verbatim

import backbone as BK  # noqa: E402
import four  # noqa: E402
import gen_common as G  # noqa: E402
import mvp_common as C  # noqa: E402
import twin5.lv_common as L  # noqa: E402

log = four.log
app_jsonl = four.app_jsonl

# ---------------------------------------------------------------- pinned constants
VENDOR_SHA = {
    "select_frames.py": "1e2297a0ae33dee9183cc30d71cc1dee566ff2acf5ae05d3d337ee809be2ce5c",
    "extract_feature.py": "23156ebd4433bc542106ba3525e2668b1836a3ce75c7f626fe153a7cc6ef4c12",
}
UPSTREAM = {
    "repo": "github.com/sming256/BOLT",
    "commit": "8a9ba455e8c7668ae4443b3273bbf283b700cc7f",
    "commit_date": "2026-02-05",
    "tarball_sha256": "fd86933ae83f515fad9fb9b036e64cf9ef3a662e0c6ef2defad8dadf2b06ad35",
    "license": "MIT (LICENSE vendored; see PROVENANCE.md)",
    "dataset_record": "plan-dataset-bolt-code-8a9ba45",
}
CLIP_DIR = os.environ.get(
    "CLIP_L14_DIR",
    "/path/to/data/clip-vit-large-patch14-fcf9b040423c")
CLIP_MANIFEST_SHA = "318590fc7a08d9d9d29e16e16bbc919f1b4ebe1e06033d4bea3b8d22a444b7e4"
CLIP_SOURCE = {
    "upstream_id": "openai/clip-vit-large-patch14",
    "mirror": "openai-mirror/clip-vit-large-patch14 (ModelScope, plain HTTP)",
    "mirror_revision": "fcf9b040423c8d6b168d6b592494ec64b0519b77",
    "device_record": "plan-exploratory-bolt-timing setup sec.2 (platform dataset "
                     "schema has no ModelScope provider; identity re-verified in-run)",
    "dtype": "float16 (official)",
}
LVB_JSON_SHA = "d32bf2e0cbbdb4fa2e99666a5a49b5a19bca2ceec69222c9a7d3aad567d1fa23"
N_VAL_RECORDS = 1337
VIDEOS_MANIFEST_SHA = "35dd905b810db2d5ee0f8bbee4dc22e5798ed7f4923f65ca8f420498be478ce5"
QSET_SHA = "57d5985387598a8286a47be78e7021e1433c1aaf101eaf335e0841b75de4de09"
EXPECTED_POOL_Q = 110
EXPECTED_POOL_V = 56
OFFICIAL_PARAMS = {"model": "clip (ViT-L/14)", "power": 2.5, "num_frames": 16,
                   "official_source": "scripts/frame_selection.sh of the pinned commit "
                                      "(longvideobench entries: power 2.5, frame 16)",
                   "fps_note": "official scores a 1 fps dense sequence; this arm scores "
                               "the cell's registered 32-frame candidate pool instead "
                               "(registered deviation, see module docstring)"}
PILOT_N = int(os.environ.get("P4_BOLT_PILOT_N", "0") or 0)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_vendored():
    out = {}
    for name, want in VENDOR_SHA.items():
        got = sha256_file(os.path.join(BOLT_OFFICIAL, name))
        out[name] = {"sha256": got, "expected": want, "matches": got == want}
        if got != want:
            raise SystemExit("vendored %s sha256 %s != pinned %s" % (name, got, want))
    return out


def check_clip_dir():
    """Recompute the publish manifest (find|sort|sha256sum, same command) and compare."""
    entries = []
    for dirpath, _dirnames, filenames in os.walk(CLIP_DIR):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = "./" + os.path.relpath(full, CLIP_DIR)
            entries.append((rel, sha256_file(full)))
    entries.sort()
    lines = ["%s  %s" % (sha, rel) for rel, sha in entries]
    manifest_sha = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    ok = manifest_sha == CLIP_MANIFEST_SHA
    return {"path": CLIP_DIR, "n_files": len(entries), "manifest_sha256": manifest_sha,
            "expected": CLIP_MANIFEST_SHA, "matches": ok,
            "files": [{"path": rel, "sha256": sha} for rel, sha in entries],
            "note": "same listing command as the publish step (find -type f | sort | "
                    "sha256sum); the platform did not verify the mirror against the "
                    "openai upstream, this only pins the bytes on this device"}


def its_reference(score, n, power):
    """Independent re-implementation of Eq. 5-7 for the self-test (not used for
    selection; the official function is)."""
    s = np.asarray(score, dtype=np.float64)
    sr = (s - s.min()) / (s.max() - s.min())
    if power != -1:
        sr = sr ** power
    probabilities = sr / np.sum(sr)
    cdf = np.cumsum(probabilities)
    uniform_sampling = np.linspace(1 / n, 1 - 1 / n, n)
    return np.searchsorted(cdf, uniform_sampling)


def its_selftest():
    rng = np.random.default_rng(20261022)
    vectors = {
        "ramp": np.linspace(0.0, 1.0, 32),
        "sine": np.sin(np.linspace(0.0, 6.0 * np.pi, 32)),
        "fixed_random": rng.random(32),
        "peaked": np.exp(-((np.arange(32) - 11) ** 2) / 2.0),
    }
    out = {"vectors": {}, "near_constant_note": None}
    for name, v in vectors.items():
        a = bolt_sf.inverse_transform_sampling(v.copy(), 16, OFFICIAL_PARAMS["power"])
        b = its_reference(v, 16, OFFICIAL_PARAMS["power"])
        same = bool(np.array_equal(np.asarray(a), np.asarray(b)))
        out["vectors"][name] = {"official": [int(x) for x in a],
                                "reference": [int(x) for x in b], "identical": same}
        if not same:
            raise SystemExit("ITS self-test failed on %s: official != Eq.5-7 reference"
                             % name)
    near = np.linspace(0.0, 1e-7, 32)  # near-constant scores: ITS reduces ~ uniform
    idx = np.asarray(bolt_sf.inverse_transform_sampling(near, 16, OFFICIAL_PARAMS["power"]))
    u16 = np.asarray(C.uniform_indices(G.K_CAND, G.B_MAIN))
    dev = int(np.max(np.abs(np.sort(idx) - np.sort(u16))))
    out["near_constant_uniform_deviation_slots"] = dev
    out["near_constant_note"] = ("with near-constant scores the official sampler returns "
                                 "the uniform 16-of-32 grid up to %d slot(s) (paper Fig. 4 "
                                 "property); exactly constant scores would make the "
                                 "official (s-min)/max division 0/0, which is why the "
                                 "per-question score range is recorded" % dev)
    return out


def load_local_clip(device="cuda"):
    """Build the official CLIP wrapper against the local weights directory. Everything
    except the from_pretrained arguments is the official __init__ body, line for line."""
    t = time.time()
    model = bolt_ef.AutoModel.from_pretrained(CLIP_DIR, dtype=torch.float16,
                                              device_map=device)
    processor = bolt_ef.CLIPProcessor.from_pretrained(CLIP_DIR)
    tokenizer = bolt_ef.AutoTokenizer.from_pretrained(CLIP_DIR)
    wrapper = bolt_ef.CLIP.__new__(bolt_ef.CLIP)
    wrapper.model = model
    wrapper.processor = processor
    wrapper.tokenizer = tokenizer
    wrapper.device = device
    wrapper.model.eval()
    load_s = time.time() - t
    return wrapper, load_s


def clip_sanity(clip, probe_images):
    t_feat = clip.extract_text_features("a test query")
    t_feat2 = clip.extract_text_features("a test query")
    v1 = clip.extract_visual_features(probe_images)
    v2 = clip.extract_visual_features(probe_images)
    return {
        "text_features_bitwise_identical": bool(torch.equal(t_feat, t_feat2)),
        "visual_features_bitwise_identical": bool(torch.equal(v1, v2)),
        "logit_scale": float(clip.model.logit_scale.exp().item()),
        "visual_feature_shape": list(v1.shape),
        "note": "same call twice on the same inputs; determinism control only",
    }


def main():
    t0 = time.time()
    vendored = check_vendored()
    clip_id = check_clip_dir()
    if not clip_id["matches"]:
        raise SystemExit("CLIP dir manifest mismatch: %s != %s"
                         % (clip_id["manifest_sha256"], CLIP_MANIFEST_SHA))
    selftest = its_selftest()
    log("bolt: vendored sha ok, clip manifest ok, ITS selftest ok")

    # ---- structure: the same 110-question pool the full run measured -----------------
    pool = C.load_pool(four.LVB_DIR)
    groups = {}
    for q in pool:
        groups.setdefault(q["video_id"], []).append(q)
    qs = sorted(pool, key=lambda q: q["_qid"])
    fp = C.data_fingerprint(four.LVB_DIR)
    structure = {
        "pool_questions": len(pool), "pool_videos": len(groups),
        "question_set_sha256": C.question_set_sha(qs),
        "data_fingerprint": fp,
        "checks": {
            "pool_110": len(pool) == EXPECTED_POOL_Q,
            "pool_56_videos": len(groups) == EXPECTED_POOL_V,
            "qset_sha_matches": C.question_set_sha(qs) == QSET_SHA,
            "lvb_json_sha_matches": fp.get("lvb_val_json_sha256") == LVB_JSON_SHA,
            "n_val_records": fp.get("n_val_records"),
            "n_val_records_matches": fp.get("n_val_records") == N_VAL_RECORDS,
            "videos_manifest_sha_matches":
                fp.get("videos_manifest_sha256") == VIDEOS_MANIFEST_SHA,
        },
    }
    bad = [k for k, v in structure["checks"].items() if v is False]
    if bad:
        raise SystemExit("bolt structure check failed: %s" % bad)
    if PILOT_N > 0:
        qs = qs[:PILOT_N]
        structure["pilot_n"] = PILOT_N
        structure["note"] = ("pipeline pilot: first %d questions in fixed qid order; no "
                             "reading" % PILOT_N)
    # ---- annotation-only prior, hashed before any model step (crash calibration) -----
    prior = four.prior_block([C.correct_letter(q) for q in qs],
                             "bolt_pool110" + ("_pilot%d" % PILOT_N if PILOT_N else ""),
                             expect_letter="C",
                             expect_hits=(31 if not PILOT_N else None))

    # ---- models ---------------------------------------------------------------------
    clip, clip_load_s = load_local_clip("cuda")
    probe = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    sanity = clip_sanity(clip, probe)
    model, proc, dev, minfo, ids = four.load_model()
    mident = L.model_identity()
    u16 = C.uniform_indices(G.K_CAND, G.B_MAIN)

    # ---- per-video: decode the registered 32-candidate pool + official CLIP features
    vids = sorted({q["video_id"] for q in qs})
    vcache = {}
    per_video = {}
    for vi, vid in enumerate(vids):
        q0 = next(q for q in qs if q["video_id"] == vid)
        n_q_in_video = sum(1 for q in qs if q["video_id"] == vid)
        t = time.time()
        arr, fps, nfr = C.decode_frames(q0["_video_path"], G.K_CAND)
        t_decode = time.time() - t
        t = time.time()
        pils = C.to_pil(arr, G.MAX_PIXELS)
        t_prep = time.time() - t
        t = time.time()
        feats = clip.extract_visual_features([arr[i] for i in range(len(arr))])
        feats = feats.cpu().numpy()
        t_clip = time.time() - t
        vcache[vid] = {"arr": arr, "pils": pils, "feats": feats, "fps": fps,
                       "n_frames": nfr, "n_q": n_q_in_video}
        per_video[vid] = {"decode_s": t_decode, "prep_s": t_prep,
                          "clip_visual_s": t_clip, "n_questions": n_q_in_video}
        four.PHASE["decode_s"] += t_decode + t_prep
        log("bolt video %2d/%d %-20s decode %.2fs clip %.2fs (%d q)"
            % (vi + 1, len(vids), vid, t_decode, t_clip, n_q_in_video))

    # ---- per-question: official text+similarity+ITS, then the two arms ---------------
    rows = []
    arm_keys = ["uniform@16", "bolt_its@16"]
    for qi, q in enumerate(qs):
        vid = q["video_id"]
        vc = vcache[vid]
        pv = per_video[vid]
        prompt = C.prompt_for(q)
        gold = C.correct_letter(q)
        query = q["question"]
        t = time.time()
        text_feat = clip.extract_text_features(query)
        vf = torch.from_numpy(vc["feats"]).to(clip.model.device)
        score = clip.compute_similarity(vf, text_feat).numpy()
        t_textsim = time.time() - t
        t = time.time()
        sel = bolt_sf.inverse_transform_sampling(score.copy(), G.B_MAIN,
                                                 OFFICIAL_PARAMS["power"])
        sel = [int(x) for x in sel]
        t_its = time.time() - t
        assert len(sel) == G.B_MAIN
        letters, raws, timings = {}, {}, {}
        t = time.time()
        letters["uniform@16"], raws["uniform@16"] = four.ask(
            model, proc, [vc["pils"][i] for i in u16], prompt, dev, "uniform@16")
        timings["forward_uniform_s"] = time.time() - t
        t = time.time()
        letters["bolt_its@16"], raws["bolt_its@16"] = four.ask(
            model, proc, [vc["pils"][i] for i in sel], prompt, dev, "bolt_its@16")
        timings["forward_bolt_s"] = time.time() - t
        letters["constant_prior"] = prior["prior_letter"]
        raws["constant_prior"] = "PRIOR:" + str(prior["prior_letter"])
        correct = {a: int(letters[a] == gold) for a in arm_keys + ["constant_prior"]}
        overlap = len(set(sel) & set(u16)) / float(G.B_MAIN)
        row = {
            "qid": q["_qid"], "video_id": vid, "gold": gold,
            "question_type": q.get("question_category"), "fps": vc["fps"],
            "n_frames": vc["n_frames"],
            "letters": letters, "raw": raws, "correct": correct,
            "selected_uniform": [int(i) for i in u16], "selected_bolt": sel,
            "overlap_vs_uniform": overlap,
            "clip_query": query,
            "clip_score_summary": {
                "min": float(np.min(score)), "max": float(np.max(score)),
                "argmax": int(np.argmax(score)),
                "range_over_max": float((np.max(score) - np.min(score))
                                        / max(1e-9, abs(np.max(score)))),
            },
            "clip_score": [float(x) for x in score],
            "timing_s": dict(timings, decode_attr=pv["decode_s"] / pv["n_questions"],
                             prep_attr=pv["prep_s"] / pv["n_questions"],
                             clip_visual_attr=pv["clip_visual_s"] / pv["n_questions"],
                             clip_text_sim=t_textsim, its=t_its),
        }
        rows.append(row)
        app_jsonl("results/bolt_samples.jsonl", row)
        log("bolt %2d/%d %-28s u=%s b=%s ov=%.2f"
            % (qi + 1, len(qs), q["_qid"], letters["uniform@16"],
               letters["bolt_its@16"], overlap))

    # ---- aggregates ------------------------------------------------------------------
    st = G.seed_streams(four.EXP_SEED)
    all_arms = arm_keys + ["constant_prior"]
    arms = {a: four.arm_entry([r["correct"][a] for r in rows]) for a in all_arms}
    comps = {
        "bolt_its@16_vs_uniform@16":
            four.compare(rows, "bolt_its@16", "uniform@16", st["s_boot"]),
        "bolt_its@16_vs_constant_prior":
            four.compare(rows, "bolt_its@16", "constant_prior", st["s_boot"]),
        "uniform@16_vs_constant_prior":
            four.compare(rows, "uniform@16", "constant_prior", st["s_boot"]),
    }
    overlap = [r["overlap_vs_uniform"] for r in rows]
    fw = [r["timing_s"]["forward_uniform_s"] for r in rows]
    fb = [r["timing_s"]["forward_bolt_s"] for r in rows]
    its_t = [r["timing_s"]["its"] for r in rows]
    ts = [r["timing_s"]["clip_text_sim"] for r in rows]
    dec = [r["timing_s"]["decode_attr"] for r in rows]
    prep = [r["timing_s"]["prep_attr"] for r in rows]
    clipv = [r["timing_s"]["clip_visual_attr"] for r in rows]

    def _m(x):
        return float(np.mean(x)) if x else None

    timing = {
        "clip_load_s": clip_load_s,
        "model_load_s": four.PHASE["model_load_s"],
        "per_question_mean_s": {
            "decode_32pool_attr": _m(dec), "prep_pil_attr": _m(prep),
            "clip_visual_32frames_attr": _m(clipv), "clip_text_plus_similarity": _m(ts),
            "inverse_transform_sampling": _m(its_t),
            "forward_uniform@16": _m(fw), "forward_bolt_its@16": _m(fb),
        },
        "per_video_mean_s": {
            k: _m([pv[k] for pv in per_video.values()])
            for k in ("decode_s", "prep_s", "clip_visual_s")
        },
        "bolt_table11_style": [
            {"step": "CLIP ViT-L/14 load", "seconds": clip_load_s, "per": "run"},
            {"step": "Qwen2-VL-7B load", "seconds": four.PHASE["model_load_s"],
             "per": "run"},
            {"step": "video decode (32-candidate pool, cv2)",
             "seconds": _m([pv["decode_s"] for pv in per_video.values()]), "per": "video"},
            {"step": "CLIP visual feature (32 frames, fp16)",
             "seconds": _m([pv["clip_visual_s"] for pv in per_video.values()]),
             "per": "video"},
            {"step": "CLIP text feature + similarity",
             "seconds": _m(ts), "per": "question"},
            {"step": "inverse transform sampling (official)", "seconds": _m(its_t),
             "per": "question"},
            {"step": "VLM answer forward, uniform@16 (16 frames)", "seconds": _m(fw),
             "per": "question"},
            {"step": "VLM answer forward, bolt_its@16 (16 frames)", "seconds": _m(fb),
             "per": "question"},
        ],
        "note": "time.time() inside the run; device not isolated (confined=false), so "
                "these are self-reported wall times on one A100 with nothing else "
                "running; NOT comparable in absolute terms to BOLT Table 11 (different "
                "backbone, carrier, candidate pool and timer method)",
    }
    ident = four.archive_identity(
        rows, "sources/task-23cd232b2b/results/four_selection_samples.jsonl",
        {"uniform@16": "uniform@16"})
    ident["n_expected"] = len(rows)
    ident["note"] = ("uniform@16 re-measured here vs the archived full run; greedy "
                     "decode has reproduced the archived letters exactly in this "
                     "project's history, agreement below 110/110 is reported, not gated")

    results = {
        "run": {},
        "exploratory": True, "post_hoc": True,
        "labels_required_in_paper": "exploratory, post-hoc, single device, single seed, "
                                    "outside the registered Holm families, stage=smoke",
        "claim_tested": None,
        "claim_note": "no claim is decided by this run; the paired difference is an "
                      "exploratory bounded statement with no family, no adjustment and "
                      "no threshold",
        "independent_unit": "video (questions clustered inside videos; both levels "
                            "reported)",
        "provenance": {
            "official_code": UPSTREAM, "vendored_files": vendored,
            "clip_weights": CLIP_SOURCE, "clip_dir_check": {
                k: v for k, v in clip_id.items() if k != "files"},
            "official_params": OFFICIAL_PARAMS,
            "registered_deviations": [
                "score vector over the cell's registered 32-frame uniform midpoint "
                "candidate pool (official: 1 fps dense sequence over the whole video)",
                "CLIP weights loaded from the local device directory (offline); class "
                "methods inherited unmodified",
                "candidate frames decoded by the cell's cv2 path (no decord on device), "
                "same indices as the archived run",
                "readout is the cell's Qwen2-VL-7B + prompt + greedy + parse_letter "
                "(not lmms-eval / LLaVA-OneVision)",
            ],
        },
        "identity": {"dataset": "lvb-val-e383ac257332",
                     "structure": structure,
                     "re_execution_of": "evidence.task-23cd232b2b"},
        "model": four.model_block(minfo, mident),
        "structure": structure,
        "controls": {
            "vendored_sha": vendored,
            "clip_manifest": {k: v for k, v in clip_id.items() if k != "files"},
            "its_selftest": selftest,
            "clip_sanity": sanity,
            "prior_calibration": {
                "letter": prior["prior_letter"], "hits": prior["prior_hits"],
                "n": prior["n"], "sha256": prior["sha256"],
                "computed_before_model_load": True, "forwards": 0},
        },
        "pretest_prior": prior,
        "arms": arms,
        "comparisons": comps,
        "selection_overlap": {"bolt_its@16_vs_uniform@16": {
            "mean": _m(overlap), "min": float(np.min(overlap)),
            "max": float(np.max(overlap)),
            "random_baseline_note": "two independent 16-of-32 subsets overlap 0.5 in "
                                    "expectation"}},
        "timing": timing,
        "extraction": four.extraction_block(rows, arm_keys),
        "identity_vs_archive": ident,
        "n_measured": len(rows),
        "verdict": {"n_questions": len(rows),
                    "n_videos": len({r["video_id"] for r in rows}),
                    "resolution_question_pp": (100.0 / len(rows) if rows else None),
                    "resolution_video_pp": (100.0 / len({r["video_id"] for r in rows})
                                            if rows else None),
                    "exploratory": True},
    }
    four.finish(results, "results/bolt_samples.jsonl")
    log("bolt exploratory run done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
