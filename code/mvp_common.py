"""Shared utilities for the MVP experiments of 

Mechanism under test: per-frame "contrast evidence score" = how much the answer-option
distribution changes when a candidate frame is replaced by a distorted (Gaussian-noise)
version, under the same reference query.  Frames are selected top-B under a fixed budget.

Everything here is training-free: no module is trained, all weights are frozen.
The scorer never sees the ground-truth label (asserted in code below).
"""
import json
import math
import os
import re
import time

import numpy as np

LETTERS = ["A", "B", "C", "D", "E"]

# ---------------------------------------------------------------- seeds / config


def seed_streams(seed):
    s = int(seed)
    return {
        "seed": s,
        "s_sel": (s * 1000003 + 11) % (2 ** 31 - 1),
        "s_noise": (s * 1000003 + 23) % (2 ** 31 - 1),
        "s_boot": (s * 1000003 + 37) % (2 ** 31 - 1),
    }


def env_config():
    return {
        "LVB_DIR": os.environ.get("LVB_DIR", ""),
        "QWEN_DIR": os.environ.get("QWEN_DIR", ""),
        "CLIP_DIR": os.environ.get("CLIP_DIR", ""),
        "N_QUESTIONS": int(os.environ.get("N_QUESTIONS", "40")),
        "K_CAND": int(os.environ.get("K_CAND", "32")),
        "B_MAIN": int(os.environ.get("B_MAIN", "16")),
        "SIGMA_MAIN": float(os.environ.get("SIGMA_MAIN", "0.2")),
        "MAX_PIXELS": int(os.environ.get("MAX_PIXELS", str(448 * 448))),
    }


# ---------------------------------------------------------------- dataset


def video_path(lvb_dir, video_id):
    for rel in (os.path.join("videos", "videos", video_id + ".mp4"),
                os.path.join("videos", video_id + ".mp4")):
        p = os.path.join(lvb_dir, rel)
        if os.path.exists(p):
            return p
    return None


def load_pool(lvb_dir):
    """All val questions whose video file is present in the downloaded subset."""
    meta = json.load(open(os.path.join(lvb_dir, "lvb_val.json")))
    pool = []
    for i, q in enumerate(meta):
        vp = video_path(lvb_dir, q["video_id"])
        if vp is None:
            continue
        item = dict(q)
        item["_idx"] = i
        item["_video_path"] = vp
        item["_qid"] = "%s#%d" % (q["video_id"], i)
        pool.append(item)
    pool.sort(key=lambda x: x["_qid"])
    return pool


def sample_questions(pool, n, s_sel):
    rng = np.random.default_rng(s_sel)
    if n >= len(pool):
        sel = list(range(len(pool)))
    else:
        sel = sorted(rng.permutation(len(pool))[:n].tolist())
    out = [pool[i] for i in sel]
    return out


def question_set_sha(qs):
    import hashlib
    h = hashlib.sha256()
    for q in qs:
        h.update(q["_qid"].encode())
        h.update(b"\n")
    return h.hexdigest()


def correct_letter(q):
    cc = q["correct_choice"]
    if isinstance(cc, str):
        m = re.search(r"[A-E]", cc.upper())
        return m.group(0) if m else None
    return LETTERS[int(cc)]


def prompt_for(q, with_query=True):
    opts = "\n".join("%s. %s" % (L, c) for L, c in zip(LETTERS, q["candidates"]))
    if with_query:
        head = q["question"]
    else:
        # Query-free (variant A) template: same options, no question-specific content.
        head = "What is shown in this image? Choose the best option."
    return ("%s\n%s\nAnswer with the single letter (A-E) of the correct option.\nAnswer:" % (head, opts))


# ---------------------------------------------------------------- frames


def _decode_decord(path, k):
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0), num_threads=4)
    n = len(vr)
    idx = np.clip(((np.arange(k) + 0.5) * n / k).astype(int), 0, n - 1)
    arr = vr.get_batch([int(i) for i in idx]).asnumpy()
    fps = float(vr.get_avg_fps()) or 30.0
    del vr
    return arr, fps, n


# Which decoder actually ran, and what failed.  The four MVP runs must use the same
# decoder for their frames to be comparable, so the name is written into every metrics.json.
DECODER_USED = {"name": None}
DECODE_CALLS = {"n": 0}
DECODE_FAILURES = []


def decode_report():
    return {"decoder": DECODER_USED["name"], "n_decode_calls": DECODE_CALLS["n"],
            "failures": DECODE_FAILURES[-20:]}


def _decode_cv2(path, k):
    """OpenCV fallback: this is the decoder the device actually has (no decord, no av)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("cv2 cannot open %s" % path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n <= 1:
            # No usable frame count: count the frames once (sequential decode, nothing kept),
            # then reopen and seek, so memory stays bounded.
            cnt = 0
            while True:
                ok, _fr = cap.read()
                if not ok:
                    break
                cnt += 1
            if cnt <= 1:
                frames = []
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                if not frames:
                    raise RuntimeError("cv2 decoded no frames from %s" % path)
                n = len(frames)
                idx = np.clip(((np.arange(k) + 0.5) * n / k).astype(int), 0, n - 1)
                return np.stack([frames[int(i)] for i in idx]), fps, n
            cap.release()
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError("cv2 re-open failed for %s" % path)
            n = cnt
        idx = np.clip(((np.arange(k) + 0.5) * n / k).astype(int), 0, n - 1)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok or fr is None:
                # seeking can land on the missing last frame; back off by one
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(i) - 1))
                ok, fr = cap.read()
            if not ok or fr is None:
                raise RuntimeError("cv2 failed to read frame %d of %s" % (i, path))
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        return np.stack(frames), fps, n
    finally:
        cap.release()


def _decode_av(path, k):
    import av
    container = av.open(path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    dur = None
    if stream.duration is not None:
        dur = float(stream.duration * stream.time_base)
    elif container.duration is not None:
        dur = container.duration / 1e6
    if dur is None or dur <= 0:
        raise RuntimeError("cannot determine duration of %s" % path)
    targets = (np.arange(k) + 0.5) * dur / k
    frames = []
    for t in targets:
        container.seek(int(t * 1e6), backward=True)
        last = None
        got = 0
        for frame in container.decode(stream):
            got += 1
            last = frame
            ft = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            if ft >= t or got > 3000:
                break
        if last is None:
            raise RuntimeError("no frame decoded at t=%.2f for %s" % (t, path))
        frames.append(last.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames), fps, int(dur * fps)


def decode_frames(path, k):
    """Return (uint8 array (k,H,W,3), fps, n_frames) at k uniform midpoints."""
    import sys
    last = None
    for name, fn in (("decord", _decode_decord), ("av", _decode_av), ("cv2", _decode_cv2)):
        try:
            out = fn(path, k)
            DECODER_USED["name"] = name
            DECODE_CALLS["n"] += 1
            return out
        except Exception as e:
            last = e
            DECODE_FAILURES.append({"path": os.path.basename(path), "decoder": name,
                                    "error": str(e)[:200]})
            print("decoder %s failed on %s (%s)" % (name, path, e), file=sys.stderr)
    raise RuntimeError("no decoder could read %s: %s" % (path, last))


def frame_timestamps(fps, n, k):
    idx = np.clip(((np.arange(k) + 0.5) * n / k).astype(int), 0, n - 1)
    return idx, idx / float(fps)


def to_pil(arr, max_pixels):
    from PIL import Image
    out = []
    for a in arr:
        im = Image.fromarray(a)
        w, h = im.size
        if w * h > max_pixels:
            s = math.sqrt(max_pixels / float(w * h))
            nw = max(28, int(w * s) // 28 * 28)
            nh = max(28, int(h * s) // 28 * 28)
            im = im.resize((nw, nh))
        out.append(im)
    return out


def distort(arr, sigma, rng):
    """Additive Gaussian noise in [0,1] pixel space, clipped back to [0,1]."""
    x = arr.astype(np.float32) / 255.0
    noise = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    y = np.clip(x + noise, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)


def uniform_indices(k, b):
    if b >= k:
        return list(range(k))
    return sorted(set(np.linspace(0, k - 1, b).round().astype(int).tolist()))


# ---------------------------------------------------------------- model


def pick_device(torch):
    forced = os.environ.get("MVP_GPU")
    if forced is not None and forced != "":
        return int(forced)
    best, bestfree = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        if free > bestfree:
            best, bestfree = i, free
    return best


def load_model(model_dir, max_pixels):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    dev = pick_device(torch)
    torch.cuda.set_device(dev)
    free0, total0 = torch.cuda.mem_get_info(dev)
    proc = AutoProcessor.from_pretrained(model_dir)
    proc.image_processor.min_pixels = 28 * 28 * 4
    proc.image_processor.max_pixels = max_pixels
    mode = "bf16"
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="cuda:%d" % dev,
            attn_implementation="sdpa")
    except torch.cuda.OutOfMemoryError:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        from transformers import BitsAndBytesConfig
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_type="nf4")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir, quantization_config=qc, device_map="cuda:%d" % dev)
        mode = "4bit"
    model.eval()
    info = {"device_index": int(dev), "gpu_name": torch.cuda.get_device_name(dev),
            "gpu_free_before_mib": round(free0 / 2 ** 20, 1),
            "gpu_total_mib": round(total0 / 2 ** 20, 1), "dtype_mode": mode}
    return model, proc, dev, info


def letter_token_ids(proc):
    tok = proc.tokenizer
    ids = []
    for L in LETTERS:
        e = tok.encode(L, add_special_tokens=False)
        assert len(e) == 1, "letter %s is not a single token: %r" % (L, e)
        ids.append(e[0])
    return ids


def _pad_index(proc, attention_mask):
    import torch
    if proc.tokenizer.padding_side == "left":
        return torch.full((attention_mask.shape[0],), attention_mask.shape[1] - 1,
                          dtype=torch.long)
    return attention_mask.sum(dim=1) - 1


def count_vision_tokens(model, proc, inputs):
    """How many vision-token slots the prompt the model is fed actually holds.

    Counted on the processor's own input_ids, with no forward pass and no re-encoding: it is a
    control quantity, not a reading.  The primary source is the model's own image-token id;
    when the config does not expose one, the token names are counted instead.
    """
    ids = inputs["input_ids"]
    for attr in ("image_token_index", "image_token_id"):
        v = getattr(model.config, attr, None)
        if v is not None:
            return int((ids == int(v)).sum()), attr
    toks = proc.tokenizer.convert_ids_to_tokens([int(t) for t in ids[0].tolist()])
    # InternVL's chat template marks each image with <IMG_CONTEXT> (the processor then repeats
    # the img-context token image_seq_length times), so those names count as vision slots too.
    n = sum(1 for t in toks if ("image_pad" in str(t))
            or str(t) in ("<image>", "<IMG_CONTEXT>", "<img_context>"))
    return int(n), "token_names"


class TokenSink(list):
    """Per-prompt vision-token record for one call.

    A list and not a counter: a batched scoring call is several prompts, and each prompt is
    recorded separately, so a per-unit count can be recomputed from the run's own samples file.
    """

    def record(self, n_images, n_vision, n_total):
        self.append({"n_images": int(n_images), "vision_tokens": int(n_vision),
                     "total_tokens": int(n_total)})

    def summary(self):
        imgs = int(sum(r["n_images"] for r in self))
        vis = int(sum(r["vision_tokens"] for r in self))
        return {"n_prompts": len(self), "n_images": imgs, "vision_tokens_total": vis,
                "vision_tokens_per_image": (float(vis) / imgs if imgs else None),
                "per_prompt": [dict(r) for r in self]}


def option_logits(model, proc, images, prompt, device, ids, batch=4, token_sink=None):
    """Logits of the five option letters for each image, one image per prompt."""
    import torch
    out_rows = []
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        msgs = [[{"role": "user", "content": [{"type": "image"},
                                              {"type": "text", "text": prompt}]}]
                for _ in chunk]
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]
        inputs = proc(text=texts, images=chunk, return_tensors="pt", padding=True).to(device)
        if token_sink is not None:
            _nv, _src = count_vision_tokens(model, proc, inputs)
            token_sink.record(len(chunk), _nv, int(inputs["input_ids"].shape[1]))
        with torch.inference_mode():
            logits = model(**inputs).logits
        idx = _pad_index(proc, inputs["attention_mask"]).to(logits.device)
        rows = logits[torch.arange(logits.shape[0], device=logits.device), idx]
        out_rows.append(torch.stack([rows[:, t] for t in ids], dim=1).float().cpu().numpy())
    return np.concatenate(out_rows, axis=0)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def contrast_scores(logits_orig, logits_dist):
    """TV distance between the option distributions (0..1) plus raw logit L1."""
    p0 = softmax(logits_orig)
    p1 = softmax(logits_dist)
    tv = 0.5 * np.abs(p0 - p1).sum(axis=1)
    l1 = np.abs(logits_orig - logits_dist).sum(axis=1)
    return tv, l1


def answer_question(model, proc, images, prompt, device):
    import torch
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in images] +
             [{"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=list(images), return_tensors="pt").to(device)
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    s = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return s, parse_letter(s)


def parse_letter(s):
    m = re.search(r"[A-Ea-e]", s)
    return m.group(0).upper() if m else None


# ---------------------------------------------------------------- stats / io


def boot_ci(diffs, s_boot, n=10000, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return [None, None]
    rng = np.random.default_rng(s_boot)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def mcnemar_exact(a, b):
    """Two-sided exact McNemar on paired boolean correctness lists."""
    import scipy.stats as st
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    p = float(st.binomtest(min(n01, n10), n, 0.5).pvalue)
    return {"n01": n01, "n10": n10, "p": p}


def holm(pairs):
    """pairs: list of (name, p). Returns list of dicts with Holm-adjusted decisions."""
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    m = len(pairs)
    out = [None] * m
    prev = 0.0
    for rank, i in enumerate(order):
        name, p = pairs[i]
        adj = min(1.0, max(prev, (m - rank) * p))
        prev = adj
        out[i] = {"name": name, "p_raw": p, "p_holm": adj, "reject_005": bool(adj < 0.05)}
    return out


def _sha256_file(path, buf=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def data_fingerprint(lvb_dir):
    """Fingerprint of the frozen data this run actually read (val json + video files)."""
    import hashlib
    fp = {}
    jp = os.path.join(lvb_dir, "lvb_val.json")
    if os.path.exists(jp):
        fp["lvb_val_json_sha256"] = _sha256_file(jp)
        try:
            fp["n_val_records"] = len(json.load(open(jp)))
        except Exception:
            fp["n_val_records"] = None
    lines = []
    vd = os.path.join(lvb_dir, "videos")
    if os.path.isdir(vd):
        for name in sorted(os.listdir(vd)):
            p = os.path.join(vd, name)
            if os.path.isfile(p):
                lines.append("%s %d" % (name, os.path.getsize(p)))
    fp["n_video_files"] = len(lines)
    fp["videos_manifest_sha256"] = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    return fp


class Recorder:
    """Writes per-sample jsonl plus the run's metrics.json."""

    def __init__(self, mvp, out_path):
        self.mvp = mvp
        self.path = out_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self.f = open(out_path, "w")

    def write(self, obj):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def write_metrics(d, path="metrics.json"):
    d.setdefault("decode_report", decode_report())
    with open(path, "w") as f:
        json.dump(d, f, ensure_ascii=False, indent=2, sort_keys=True)


def now():
    return time.time()
