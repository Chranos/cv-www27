"""Shared harness for the fourth minimum-experiment batch of 

Plan: plan-generalisation-r1 (version 1).

A1  second backbone (LLaVA-OneVision-7B) x carrier 1 (LongVideoBench val subset, pinned 40 questions)
A2  original backbone (Qwen2-VL-7B)      x carrier 2 (LVBench subset, pinned 24 questions)
A3  second backbone x carrier 2 (cross cell, optional)
A4  text/subtitle channel on carrier 1 (24 questions): no-sub / interleaved-sub / subtitles-only /
    no frames
A5  the no-frames floor is an arm inside every run.

The three accounting errors the batch-3 check found are fixed structurally here, not in prose:
  * every counted forward goes through one accumulator (`Fwd`), so sub-totals and the grand total
    come from the same source (batch 3 reported R's single-image forwards as 256 instead of 128,
    L1's U16 block as 33 instead of 35, L1's per-question total as 160 instead of 161);
  * there is no hand-written formula in the write-up to disagree with the code (the batch-3
    self-correction fixed 63->62 but left the same paragraph's other numbers wrong);
  * criteria are evaluated as an ordered, mutually exclusive chain and every threshold-boundary
    comparison is reported both as a difference and as a whole number of questions (batch 3
    claimed its U clause "did not hold" because |0.05| is not below 0.05, while the code's IEEE
    comparison made it true; what actually decided the verdict was the chain order).
"""
import hashlib
import json
import os
import re
import time

import numpy as np

import mvp_common as C

LETTERS = ["A", "B", "C", "D", "E"]
SIGMA_MAIN = 0.2
SIGMA_LIST = [0.05, 0.1, 0.2, 0.4]
SIGMA_INDEX = SIGMA_LIST.index(SIGMA_MAIN)          # = 2, the same operating point as batch 3
ANCHORS = [8, 16, 24]
K_CAND = 32
B_MAIN = 16
B_LOW = 8
B_HIGH = 32
NSEG = 8
MAX_PIXELS = 448 * 448

# pre-registered thresholds (Plan plan-generalisation-r1@v1, expected_results)
G2_MEDIAN_TV_BAR = 0.15
G3_SD_MAX = 0.02
W_GAIN = 0.025
M_LOSS = -0.025
F_BAND = 0.025
COUNT_AXIS_BAR = 0.02
PINNED_QSET_SHA = "c1ee771fb2f963c9367ccb98d9b75c3770c55a2eb305abfbc212a2ef94050075"
PINNED_SOURCE_TASK = "task-ca0a24e61b"
QSET_N_LVB = 40


# ---------------------------------------------------------------- provenance helpers

def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def manifest_sha(paths):
    """Hash of the (basename, size) list -- the same shape the batch-1..3 runs recorded."""
    lines = ["%s %d" % (os.path.basename(p), os.path.getsize(p)) for p in sorted(paths)]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


class Fwd:
    """Forward accumulator. The only place any forward count is produced."""

    def __init__(self):
        self.kind = {}
        self.arm = {}

    def count(self, kind, n=1, arm=None):
        if n <= 0:
            return
        self.kind[kind] = self.kind.get(kind, 0) + int(n)
        if arm:
            self.arm[arm] = self.arm.get(arm, 0) + int(n)

    def report(self):
        kinds = dict(sorted(self.kind.items()))
        return {"by_kind": kinds, "by_arm": dict(sorted(self.arm.items())),
                "total": int(sum(kinds.values())),
                "note": "accumulated inside the run by the code path that actually ran; the same "
                        "accumulator produces every sub-total and the grand total"}


# ---------------------------------------------------------------- seeds

def seed_streams(exp_seed):
    s = int(exp_seed)
    return {"seed": s, "s_sel": (s * 1000003 + 11) % (2 ** 31 - 1),
            "s_noise": (s * 1000003 + 23) % (2 ** 31 - 1),
            "s_boot": (s * 1000003 + 37) % (2 ** 31 - 1)}


def run_seeds(exp_seed, ks=(0, 1, 2)):
    out = []
    for k in ks:
        seed = int(exp_seed) * 10 + int(k)
        st = seed_streams(seed)
        out.append({"k": int(k), "seed": seed, "s_sel": int(st["s_sel"]),
                    "s_noise": int(st["s_noise"]), "s_boot": int(st["s_boot"])})
    return out


def distort_seeded(arr, sigma, s_noise, q_index, frame_index, sigma_index):
    ss = np.random.SeedSequence([int(s_noise), int(q_index), int(frame_index), int(sigma_index)])
    rng = np.random.default_rng(ss)
    return C.distort(arr, sigma, rng)


def noise_stream_selfcheck(s_noise):
    """Position independent, and different sigma_index must give a different tensor."""
    arr = (np.arange(8 * 8 * 3).reshape(8, 8, 3) % 251).astype(np.uint8)
    a = distort_seeded(arr, SIGMA_MAIN, s_noise, 7, 3, SIGMA_INDEX)
    _ = np.random.default_rng(12345).normal(size=(64,))          # unrelated draw in between
    b = distort_seeded(arr, SIGMA_MAIN, s_noise, 7, 3, SIGMA_INDEX)
    c = distort_seeded(arr, SIGMA_MAIN, s_noise, 7, 3, 1)
    return bool(np.array_equal(a, b)) and not bool(np.array_equal(a, c))


def assert_noise_selfcheck(s_noise_by_k):
    checks = {str(int(k)): bool(noise_stream_selfcheck(int(v))) for k, v in s_noise_by_k.items()}
    if not all(checks.values()):
        raise SystemExit("noise-stream self-check FAILED for %r" % checks)
    return checks


# ---------------------------------------------------------------- model

def load_vlm(model_dir, max_pixels=MAX_PIXELS, device=None):
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor
    dev = int(device) if device is not None else 0
    if torch.cuda.is_available():
        free0, total0 = torch.cuda.mem_get_info(dev)
        torch.cuda.set_device(dev)
        gpu = torch.cuda.get_device_name(dev)
    else:
        free0 = total0 = 0
        gpu = "cpu"
        dev = -1
    cfg = AutoConfig.from_pretrained(model_dir)
    arch = (getattr(cfg, "architectures", None) or [""])[0]
    cls = getattr(transformers, arch, None)
    if cls is None:
        raise SystemExit("transformers %s cannot build %r (from %s)"
                         % (transformers.__version__, arch, model_dir))
    proc = AutoProcessor.from_pretrained(model_dir)
    ip = getattr(proc, "image_processor", None)
    if ip is not None and hasattr(ip, "max_pixels"):
        ip.min_pixels = 28 * 28 * 4
        ip.max_pixels = max_pixels
        pixel_control = "min/max_pixels"
    else:
        pixel_control = "processor default"
    model = cls.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                device_map=("cuda:%d" % dev) if dev >= 0 else None,
                                attn_implementation="sdpa")
    model.eval()
    info = {"model_dir": model_dir, "architecture": arch, "dtype_mode": "bf16",
            "device_index": dev, "gpu_name": gpu,
            "gpu_free_before_mib": round(free0 / 2 ** 20, 1),
            "gpu_total_mib": round(total0 / 2 ** 20, 1),
            "pixel_control": pixel_control, "transformers": transformers.__version__,
            "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available())}
    return model, proc, dev, info


def model_fingerprint(model_dir):
    names = []
    for root, _dirs, files in os.walk(model_dir):
        for f in files:
            p = os.path.join(root, f)
            names.append((os.path.relpath(p, model_dir), os.path.getsize(p)))
    names.sort()
    h = hashlib.sha256()
    for n, s in names:
        h.update(("%s %d\n" % (n, s)).encode())
    return {"files": len(names), "bytes": int(sum(s for _n, s in names)),
            "name_size_manifest_sha256": h.hexdigest()}


# ---------------------------------------------------------------- prompt, forwards

def prompt_for(q, letters=LETTERS, with_query=True):
    cands = list(q["candidates"])
    opts = "\n".join("%s. %s" % (L, c) for L, c in zip(letters, cands))
    if with_query:
        head = q["question"]
    else:                                   # unused here; kept as in mvp_common
        head = "What is shown in this image? Choose the best option."
    rng = letters[:len(cands)]
    return ("%s\n%s\nAnswer with the single letter (%s-%s) of the correct option.\nAnswer:"
            % (head, opts, rng[0], rng[-1]))


def _msgs(pils, prompt, gaps=None):
    content = []
    for i, _ in enumerate(pils):
        content.append({"type": "image"})
        if gaps and i < len(gaps) and gaps[i]:
            content.append({"type": "text", "text": gaps[i]})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def option_probs(model, proc, pils, prompt, device, ids, batch=4, counter=None, kind=None,
                 gaps=None):
    """Option-letter distribution for one prompt holding len(pils) frames (batching only for
    the single-image calls; a multi-frame prompt is always one call)."""
    import torch
    out = []
    for i in range(0, max(1, len(pils)), batch):
        chunk = list(pils[i:i + batch])
        msgs = [_msgs([c], prompt) for c in chunk]
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]
        inputs = proc(text=texts, images=chunk, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        idx = C._pad_index(proc, inputs["attention_mask"]).to(logits.device)
        row = logits[torch.arange(logits.shape[0], device=logits.device), idx]
        out.append(np.asarray([[float(row[j, t]) for t in ids] for j in range(len(chunk))]))
    if counter is not None:
        counter.count(kind or "option_probs", len(pils))
    return C.softmax(np.concatenate(out, axis=0))


def option_probs_multi(model, proc, pils, prompt, device, ids, counter=None, kind=None, gaps=None):
    """Option-letter distribution for ONE prompt holding all of `pils` (order preserved)."""
    import torch
    msgs = _msgs(pils, prompt, gaps=gaps)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=list(pils), return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits
    idx = C._pad_index(proc, inputs["attention_mask"]).to(logits.device)
    row = logits[torch.arange(logits.shape[0], device=logits.device), idx]
    if counter is not None:
        counter.count(kind or "option_probs_multi", 1)
    return C.softmax(np.asarray([[float(row[0, t]) for t in ids]]))


def _first_step_blocker(ids, begin_index, steps=(0,), is_ws=None):
    """A logits processor that forbids `ids` at the first generated token (and, optionally,
    at the next one).

    Used by the free-text naming control so that its answer cannot degenerate into a bare
    letter.  It never favours any option: it only removes the blocked ids from the step, so the
    model has to write something else.  Written here rather than passed as a generate kwarg so
    it does not depend on which generation-config keys this transformers version accepts.

    `steps` names the generated positions to block, counted from the first generated token.
    `is_ws(token_id)` is consulted only for position 1: if the first generated token decodes to
    whitespace alone, the block is applied again, so a reply of " A" cannot slip past a
    first-position-only block.
    """
    block = [int(i) for i in ids]
    b0 = int(begin_index)
    offs = {b0 + int(s) for s in steps}

    def _proc(input_ids, scores):
        n = int(input_ids.shape[1])
        fire = n in offs
        if not fire and is_ws is not None and n == b0 + 1:
            fire = bool(is_ws(int(input_ids[0, -1])))
        if fire:
            scores = scores.clone()
            for i in block:
                scores[:, i] = float("-inf")
        return scores

    return _proc


def _ws_checker(tok):
    """True when a token id decodes to whitespace only (used by the naming-step block)."""
    def f(i):
        try:
            return str(tok.decode([int(i)])).strip() == ""
        except Exception:                                        # noqa: BLE001
            return False
    return f


def answer_arm(model, proc, pils, prompt, device, counter=None, kind=None, gaps=None,
               max_new_tokens=8, token_sink=None, first_token_block_ids=None,
               block_steps=(0,), prompt_suffix=None):
    """One greedy answer.  `max_new_tokens=8` is what every measurement arm uses and is
    unchanged; only the free-text naming control of the instrument battery asks for more.
    `token_sink`, when given, records the vision-token count of this call's own input_ids.
    `first_token_block_ids` / `block_steps` are used only by that naming control (see
    _first_step_blocker).  `prompt_suffix` is appended AFTER the chat template's generation
    prompt, so it is the opening of the assistant turn and not part of the user message: the
    naming control puts `The option text is: "` there, which fixes the shape of the answer in
    the input itself rather than asking the model to obey an instruction."""
    import torch
    from transformers import LogitsProcessorList
    msgs = _msgs(pils, prompt, gaps=gaps)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if prompt_suffix:
        text = text + str(prompt_suffix)
    if pils:
        inputs = proc(text=[text], images=list(pils), return_tensors="pt").to(device)
    else:
        inputs = proc(text=[text], return_tensors="pt").to(device)
    if token_sink is not None:
        _nv, _src = C.count_vision_tokens(model, proc, inputs)
        token_sink.record(len(pils), _nv, int(inputs["input_ids"].shape[1]))
    extra = {}
    if first_token_block_ids:
        extra["logits_processor"] = LogitsProcessorList(
            [_first_step_blocker(first_token_block_ids, inputs["input_ids"].shape[1],
                                 steps=tuple(block_steps),
                                 is_ws=_ws_checker(proc.tokenizer))])
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False,
                             **extra)
    s = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    if counter is not None:
        counter.count(kind or "answer", 1, arm=kind)
    return s, C.parse_letter(s)


def tv(p, q):
    return float(0.5 * np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


# ---------------------------------------------------------------- selection rules

def top_b(scores, b):
    sc = np.asarray(scores, float)
    order = sorted(range(len(sc)), key=lambda i: (-sc[i], int(i)))
    return sorted(int(i) for i in order[:b])


def quota_select(scores, b, k, nseg=NSEG):
    sc = np.asarray(scores, float)
    segs = np.array_split(np.arange(k), nseg)
    per = b // nseg
    chosen = []
    for s in segs:
        order = sorted(s.tolist(), key=lambda i: (-sc[i], int(i)))
        chosen.extend(order[:per])
    cset = set(chosen)
    rest = [i for i in range(k) if i not in cset]
    rest.sort(key=lambda i: (-sc[i], int(i)))
    chosen.extend(rest[:max(0, b - len(chosen))])
    return sorted(int(i) for i in set(chosen))


def sim_matrix(arrs):
    import cv2
    gs = []
    for a in arrs:
        g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gs.append(cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0)
    k = len(gs)
    m = np.zeros((k, k), np.float32)
    for i in range(k):
        for j in range(i + 1, k):
            v = 1.0 - float(np.mean(np.abs(gs[i] - gs[j])))
            m[i, j] = m[j, i] = v
    return m


def selection_metrics(sel, sim, k, nseg=NSEG):
    sel = sorted(int(i) for i in sel)
    segs = np.array_split(np.arange(k), nseg)
    sset = set(sel)
    cov = sum(1 for s in segs if any(int(i) in sset for i in s.tolist())) / float(nseg)
    pairs = [(sel[a], sel[b]) for a in range(len(sel)) for b in range(a + 1, len(sel))]
    red = float(np.mean([sim[i][j] for i, j in pairs])) if pairs else None
    gaps = [sel[i + 1] - sel[i] for i in range(len(sel) - 1)]
    return {"segments_covered": float(cov), "span": (sel[-1] - sel[0]) / float(k - 1),
            "mean_gap_frames": float(np.mean(gaps)) if gaps else None, "redundancy": red}


def overlap(a, b):
    a, b = set(int(x) for x in a), set(int(x) for x in b)
    return len(a & b) / float(max(1, len(a)))


# ---------------------------------------------------------------- statistics

def boot_ci(values, s_boot, n=10000, alpha=0.05):
    d = np.asarray(values, float)
    if len(d) == 0:
        return [None, None]
    rng = np.random.default_rng(int(s_boot))
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def boot_ci_cluster(values, clusters, s_boot, n=10000, alpha=0.05):
    """The video is the independent unit: whole clusters are resampled."""
    v = np.asarray(values, float)
    cl = list(clusters)
    uq = sorted(set(cl))
    groups = {c: v[[i for i, cc in enumerate(cl) if cc == c]] for c in uq}
    rng = np.random.default_rng(int(s_boot) + 1)
    means = np.empty(n)
    for t in range(n):
        pick = rng.integers(0, len(uq), size=len(uq))
        means[t] = np.concatenate([groups[uq[i]] for i in pick]).mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def sign_test(n_win, n_loss):
    import scipy.stats as st
    n = n_win + n_loss
    if n == 0:
        return {"n_win": 0, "n_loss": 0, "p": 1.0}
    return {"n_win": int(n_win), "n_loss": int(n_loss),
            "p": float(st.binomtest(int(n_win), n, 0.5).pvalue)}


def compare(name, a, b, clusters, s_boot, label=None, n_units_note=None):
    """Paired comparison of two per-question correctness lists; both units reported."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    n = len(d)
    one_q = 1.0 / n if n else None
    return {"label": label or name, "n_units": int(n), "unit": "question",
            "mean": float(d.mean()),
            "ci95_question": boot_ci(d, s_boot),
            "ci95_video_cluster": boot_ci_cluster(d, clusters, s_boot),
            "sd": float(d.std(ddof=1)) if n > 1 else None,
            "min": float(d.min()), "max": float(d.max()),
            "sign_test": sign_test(int(np.sum(d > 0)), int(np.sum(d < 0))),
            "mcnemar": C.mcnemar_exact(a.astype(bool), b.astype(bool)),
            "one_question_pp": one_q,
            "boundary_note": ("thresholds are also stated as whole questions: 0.025 = %.2f q, "
                              "0.05 = %.2f q, 0.02 = %.2f q" % (0.025 / one_q if one_q else 0,
                                                                 0.05 / one_q if one_q else 0,
                                                                 0.02 / one_q if one_q else 0)),
            "resolution_note": n_units_note}


def choice_axis_verdict(point, ci_lo, ci_hi):
    """Ordered, mutually exclusive chain (Plan expected_results). One verdict only."""
    if point >= W_GAIN and ci_lo > 0:
        return "W_gain"
    if point <= M_LOSS and ci_hi < 0:
        return "M_worse"
    if abs(point) <= F_BAND or (ci_lo <= 0 <= ci_hi and abs(point) <= 0.05):
        return "F_not_separable"
    return "I_resolution"


def count_axis_read(b16_minus_b8, b32_minus_b16):
    out = {}
    out["uniform32_minus_uniform16"] = (
        "no_observed_gain_from_more_frames"
        if b32_minus_b16 is not None and b32_minus_b16 <= COUNT_AXIS_BAR else
        ("gain_from_more_frames" if b32_minus_b16 is not None else "not_measured"))
    out["uniform16_minus_uniform8"] = (
        "count_axis_flat_too"
        if b16_minus_b8 is not None and b16_minus_b8 <= COUNT_AXIS_BAR else
        ("count_axis_has_spread" if b16_minus_b8 is not None else "not_measured"))
    out["applies_to"] = "frame-COUNT axis only; never to the frame-CHOICE axis"
    return out


# ---------------------------------------------------------------- carriers

def load_lvb_questions(lvb_dir, pinned_jsonl, n_q=None):
    meta = json.load(open(os.path.join(lvb_dir, "lvb_val.json")))
    rows = [json.loads(l) for l in open(pinned_jsonl)]
    pool = {}
    for i, q in enumerate(meta):
        vp = C.video_path(lvb_dir, q["video_id"])
        if vp is None:
            continue
        pool["%s#%d" % (q["video_id"], i)] = {
            "_qid": "%s#%d" % (q["video_id"], i), "_idx": i,
            "video_id": q["video_id"], "_video_path": vp,
            "question": q["question"], "candidates": q["candidates"],
            "correct_letter": C.correct_letter(q),
            "duration_group": q.get("duration_group"), "question_category": q.get("question_category"),
            "position": q.get("position"),
            "subtitle_path": q.get("subtitle_path"),
            "subtitle_offset": q.get("starting_timestamp_for_subtitles") or 0}
    qs = []
    for r in rows:
        qid = r.get("qid")
        if qid not in pool:
            raise SystemExit("pinned qid %r not in the pool (%d)" % (qid, len(pool)))
        qs.append(pool[qid])
    sha = C.question_set_sha(qs)
    if sha != PINNED_QSET_SHA:
        raise SystemExit("question-set fingerprint mismatch: %s != %s" % (sha, PINNED_QSET_SHA))
    if n_q:
        qs = qs[:int(n_q)]
    return qs, sha


def load_lvbench_questions(chosen_path, video_dir, pin_path=None):
    d = json.load(open(chosen_path))
    if pin_path and os.path.exists(pin_path):
        pin = json.load(open(pin_path))
        got = sha256_file(chosen_path)
        if got != pin.get("frozen_list_sha256"):
            raise SystemExit("LVBench frozen-list hash mismatch: %s != %s"
                             % (got, pin.get("frozen_list_sha256")))
        want = list(pin.get("qids") or [])
        have = ["%s#%s" % (q["video"], q["uid"]) for q in d["chosen"]]
        if want and want != have:
            raise SystemExit("LVBench pinned qid list differs from the frozen list")
        if pin.get("annotation_sha256") != d.get("annotation_sha256"):
            raise SystemExit("LVBench annotation hash differs between the pin and the frozen list")
    qs = []
    for q in d["chosen"]:
        vp = os.path.join(video_dir, os.path.basename(q["video_path"]))
        if not os.path.exists(vp):
            raise SystemExit("missing LVBench video %s" % vp)
        if os.path.getsize(vp) != int(q["video_bytes"]):
            raise SystemExit("LVBench video size mismatch for %s" % vp)
        got = sha256_file(vp)
        if got != q["video_sha256"]:
            raise SystemExit("LVBench video sha256 mismatch for %s: %s != %s"
                             % (vp, got, q["video_sha256"]))
        qs.append({"_qid": "%s#%s" % (q["video"], q["uid"]), "video_id": q["video"],
                   "_video_path": vp, "question": q["stem"], "candidates": q["options"],
                   "correct_letter": q["answer"], "question_type": q.get("question_type"),
                   "time_reference": q.get("time_reference"), "video_bytes": q["video_bytes"]})
    return qs, d


def question_set_sha(qs):
    h = hashlib.sha256()
    for q in qs:
        h.update(q["_qid"].encode())
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------- subtitles (carrier 1, A4)

SRT_TIME = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})")


def _secs(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _hms(t):
    """'HH:MM:SS.mmm' or 'HH:MM:SS,mmm' -> seconds (None if unparseable)."""
    m = re.match(r"\s*(\d+):(\d{2}):(\d{2})[.,](\d{1,3})\s*$", str(t))
    if not m:
        return None
    return _secs(*m.groups())


def parse_subtitles_json(path):
    """LongVideoBench ships its subtitle channel as a JSON list of {start, end, line}."""
    out = []
    try:
        data = json.load(open(path, "r", encoding="utf-8", errors="replace"))
    except Exception:
        return out
    if not isinstance(data, list):
        return out
    for c in data:
        if not isinstance(c, dict):
            continue
        s, e = _hms(c.get("start")), _hms(c.get("end"))
        txt = " ".join(str(c.get("line") or c.get("text") or "").replace("&nbsp;", " ").split())
        if s is None or e is None or not txt:
            continue
        out.append((float(s), float(e), txt))
    out.sort(key=lambda x: x[0])
    return out


def parse_subtitles(path):
    """Return [(start_s, end_s, text), ...]; JSON cue lists and .srt are both handled."""
    if str(path).lower().endswith(".json"):
        return parse_subtitles_json(path)
    out = []
    try:
        raw = open(path, "r", encoding="utf-8", errors="replace").read()
    except Exception:
        return out
    for block in re.split(r"\n\s*\n", raw.replace("\r\n", "\n")):
        lines = [l for l in block.split("\n") if l.strip() != ""]
        if not lines:
            continue
        m = None
        for i, l in enumerate(lines):
            m = SRT_TIME.search(l)
            if m:
                lines = lines[i + 1:]
                break
        if not m:
            continue
        start = _secs(*m.groups()[:4])
        end = _secs(*m.groups()[4:])
        text = " ".join(x.strip() for x in lines if x.strip())
        if text:
            out.append((start, end, text))
    return out


def subtitle_gaps(cues, frame_times, n_frames):
    """Insert each cue between the two frames whose mid-timestamps bracket the cue's midpoint."""
    gaps = [""] * max(0, n_frames - 1)
    for start, end, text in cues:
        mid = 0.5 * (start + end)
        idx = None
        for i in range(n_frames - 1):
            if frame_times[i] <= mid < frame_times[i + 1]:
                idx = i
                break
        if idx is None:
            idx = 0 if mid < frame_times[0] else n_frames - 2
        gaps[idx] = (gaps[idx] + " " + text).strip() if gaps[idx] else text
    return gaps


def subtitle_text_all(cues):
    return " ".join(t for _s, _e, t in cues)
