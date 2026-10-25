"""Backbone resolution for the four readings (first and second backbone).

Plan: plan-four-readings-second-backbone (version bound by P4_PLAN_VERSION).

Only the WEIGHTS change between backbones.  Everything else -- carriers, question sets,
splits, pins, frame pools, arm definitions, branch rules, thresholds, statistics -- is the
code path that already ran on the first backbone (commit 76ca8512) and is not touched here.

What does change, and has to be written down, is the per-frame pixel budget: each backbone
sees frames at its own native resolution/processor budget.
  * qwen2vl : dynamic resolution, min_pixels = 28*28*4, max_pixels = 448*448 (the setting
              every earlier run used).
  * llavaov : one 384x384 tile per frame (image_grid_pinpoints = [[384, 384]], i.e. anyres
              turned off), processor size 384.  The packaged anyres grid list would spend many
              tiles per frame; capping it at one tile keeps the per-frame budget in the same
              range as the first backbone's and makes the 16-frame budget the thing that is
              held fixed.  The actual per-image vision-token count is measured by each run on
              the processor's own input_ids, not derived from this config.
That difference is a backbone property, not a knob: it is identical for every arm inside a
run, so within-backbone paired comparisons are unaffected, and cross-backbone statements are
limited to whether a reading's direction repeats (see the Plan).

No constant is asserted against a stale pin.  The weight identity is computed by the run
that reports it: a name+size manifest for every file, plus head/tail hashes of every
safetensors shard, plus full hashes of the config/tokenizer files; `deep=True` hashes the
whole shards (used once, by the probe).
"""
import hashlib
import os

DATA = os.environ.get("DATA_ROOT", "/path/to/data")

BACKBONES = {
    "qwen2vl": {
        "family": "Qwen2-VL-7B-Instruct",
        "model_dir": os.path.join(DATA, "qwen2vl-7b-2cbada593a2b"),
        "pixel_control": "min_pixels=28*28*4, max_pixels=448*448 (native dynamic resolution)",
        "max_pixels": 448 * 448,
        "image_grid_pinpoints": None,
        "source": "device copy qwen2vl-7b-2cbada593a2b (first backbone, unchanged)",
        "license": "Apache-2.0 (Qwen2-VL-7B-Instruct)",
    },
    "llavaov": {
        "family": "LLaVA-OneVision-7B-Qwen2 (ov, HF format)",
        "model_dir": os.path.join(DATA, "llavaov-7b-ov-hf-ad4cb27d2a3f"),
        "pixel_control": "384x384 per frame; the anyres grid list is pinned to the single point "
                         "[[384,384]] in BOTH the processor and the model config.  The vision-"
                         "token count is NOT derived from the config: it is measured on the "
                         "processor's own input_ids (see the run's vision_tokens block).  Runs "
                         "of this design measured 1,149 vision tokens per image on the standard "
                         "path (selection / single / distortion) and 1,037 on the donor path, "
                         "the same on all three image-bearing prompt paths inside one run; the "
                         "difference between readings is recorded and was not traced.",
        "max_pixels": 384 * 384,
        "image_grid_pinpoints": [[384, 384]],
        "source": "device copy llavaov-7b-ov-hf-ad4cb27d2a3f, provenance "
                  "_provenance.json: AI-ModelScope/llava-onevision-qwen2-7b-ov-hf "
                  "(upstream lmms-lab/LLaVA-OneVision-7B-Qwen2-ov)",
        "license": "Apache-2.0 (ModelScope listing and upstream repo)",
    },
    # r6 second backbone (Plan plan-four-readings-second-backbone-r6 §1).  The model
    # directory is passed in P4_MODEL_DIR: it is the published device dataset
    # internvl3-8b-hf-<manifest12>, whose name carries the manifest hash, so it is
    # never hard-coded here.  The per-frame budget is pinned by patches, not by qwen-style
    # pixels: min_patches=max_patches=1 -> one 448x448 view per frame -> 256 vision tokens
    # per frame (image_seq_length=256 in processor_config.json), so a 16-frame prompt holds
    # 16*256 = 4,096 vision tokens -- the same budget the first backbone had.  The count is
    # measured on the processor's own input_ids by every run; this string is not the source.
    "internvl3hf": {
        "family": "InternVL3-8B-hf (InternViT-300M-448px vision, Qwen2.5-7B text stack)",
        "model_dir": os.environ.get("P4_MODEL_DIR", ""),
        "pixel_control": "min_patches=max_patches=1 (one 448x448 view per frame, 256 vision "
                         "tokens per frame, image_seq_length=256; same budget on every "
                         "image-bearing arm inside one run; measured per call, not derived)",
        "max_pixels": None,          # this backbone pins patches; the qwen-style pixel knobs
        "image_grid_pinpoints": None,  # and the llava-style grid pins do not apply
        "min_patches": 1,
        "max_patches": 1,
        "source": "ModelScope OpenGVLab/InternVL3-8B-hf, fetched with staging/ms_fetch.py "
                  "(per-file sha256 checked against the ModelScope API values, resumable "
                  "curl), published atomically as the device dataset "
                  "internvl3-8b-hf-<manifest12> with the manifest beside it",
        "license": "MIT for InternVL3-8B-hf (project README); the embedded Qwen2.5-7B text "
                   "stack is under the Qwen License as the README states for the embedded "
                   "components -- recorded as the source states it, not re-licensed here",
    },
}

CONFIG_FILES = ("config.json", "preprocessor_config.json", "processor_config.json",
                "tokenizer_config.json", "generation_config.json", "chat_template.json",
                "chat_template.jinja", "video_preprocessor_config.json")


def get(name):
    if name not in BACKBONES:
        raise SystemExit("unknown backbone %r (have %s)" % (name, sorted(BACKBONES)))
    b = dict(BACKBONES[name])
    if name == "internvl3hf" and not b["model_dir"]:
        raise SystemExit("backbone internvl3hf needs P4_MODEL_DIR: the published device "
                         "dataset directory internvl3-8b-hf-<manifest12>")
    return b


def _sha_range(path, start, length):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        left = length
        while left > 0:
            b = f.read(min(1 << 20, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def configure_processor(proc, name):
    """Apply the per-backbone pixel budget to a freshly loaded processor; return what was
    actually set so the run records the setting rather than the intention."""
    b = get(name)
    ip = getattr(proc, "image_processor", None)
    applied = {}
    if ip is None:
        return {"image_processor": None}
    if b["image_grid_pinpoints"] is not None and hasattr(ip, "image_grid_pinpoints"):
        ip.image_grid_pinpoints = [list(p) for p in b["image_grid_pinpoints"]]
        applied["image_grid_pinpoints"] = [list(p) for p in ip.image_grid_pinpoints]
    # internvl3hf pins the budget by patches (GotOcr2-style image processor).  Applied only
    # when the backbone declares one, so qwen2vl/llavaov behaviour is bit-identical to before.
    if b.get("min_patches") is not None and hasattr(ip, "min_patches"):
        ip.min_patches = int(b["min_patches"])
        ip.max_patches = int(b["max_patches"])
        applied["min_patches"] = int(ip.min_patches)
        applied["max_patches"] = int(ip.max_patches)
    if b.get("max_pixels") is not None and hasattr(ip, "max_pixels"):
        ip.min_pixels = 28 * 28 * 4
        ip.max_pixels = b["max_pixels"]
        applied["min_pixels"] = int(ip.min_pixels)
        applied["max_pixels"] = int(ip.max_pixels)
    size = getattr(ip, "size", None)
    if size is not None:
        applied["size"] = dict(size) if isinstance(size, dict) else size
    applied["class"] = type(ip).__name__
    return applied


def load(name, max_pixels=None, device=0):
    """Load the named backbone.  Mirrors gen_common.load_vlm, but the processor budget comes
    from this module and the architecture is whatever the checkpoint declares."""
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor

    b = get(name)
    dev = int(device)
    if torch.cuda.is_available():
        free0, total0 = torch.cuda.mem_get_info(dev)
        torch.cuda.set_device(dev)
        gpu = torch.cuda.get_device_name(dev)
    else:
        free0 = total0 = 0
        gpu = "cpu"
        dev = -1
    cfg = AutoConfig.from_pretrained(b["model_dir"])
    arch = (getattr(cfg, "architectures", None) or [""])[0]
    cls = getattr(transformers, arch, None)
    if cls is None:
        raise SystemExit("transformers %s cannot build %r (from %s)"
                         % (transformers.__version__, arch, b["model_dir"]))
    proc = AutoProcessor.from_pretrained(b["model_dir"])
    applied = configure_processor(proc, name)
    model = cls.from_pretrained(b["model_dir"], torch_dtype=torch.bfloat16,
                                device_map=("cuda:%d" % dev) if dev >= 0 else None,
                                attn_implementation="sdpa")
    model.eval()
    # The processor and the model must agree on how many 384x384 tiles one image becomes:
    # the processor slices with its own image_grid_pinpoints, the model recomputes the tile
    # count from config.image_grid_pinpoints.  If only one of them is pinned the forward pass
    # fails in torch.split.  Pin both.
    if b["image_grid_pinpoints"] is not None:
        model.config.image_grid_pinpoints = [list(p) for p in b["image_grid_pinpoints"]]
    applied["model_config_image_grid_pinpoints"] = list(
        getattr(model.config, "image_grid_pinpoints", []) or [])
    info = {"backbone": name, "family": b["family"], "model_dir": b["model_dir"],
            "architecture": arch, "dtype_mode": "bf16", "device_index": dev,
            "gpu_name": gpu, "gpu_free_before_mib": round(free0 / 2 ** 20, 1),
            "gpu_total_mib": round(total0 / 2 ** 20, 1),
            "pixel_control": b["pixel_control"], "processor_settings_applied": applied,
            "transformers": transformers.__version__, "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available())}
    return model, proc, dev, info


def letter_ids(proc):
    """Token ids of the five option letters, asserted single-token."""
    tok = proc.tokenizer
    ids, seen = [], {}
    for L in ("A", "B", "C", "D", "E"):
        e = tok.encode(L, add_special_tokens=False)
        if len(e) != 1:
            raise SystemExit("letter %s is not a single token on this backbone: %r" % (L, e))
        ids.append(e[0])
        seen[L] = e[0]
    return ids


def identity(name, deep=False):
    b = get(name)
    d = b["model_dir"]
    files = []
    for root, _dirs, fs in os.walk(d):
        for f in fs:
            p = os.path.join(root, f)
            files.append((os.path.relpath(p, d), os.path.getsize(p)))
    files.sort()
    ns = hashlib.sha256()
    for rel, s in files:
        ns.update(("%s %d\n" % (rel, s)).encode())
    shards = {}
    for rel, s in files:
        if not rel.endswith(".safetensors"):
            continue
        p = os.path.join(d, rel)
        rec = {"bytes": s, "head_sha256": _sha_range(p, 0, 1 << 20),
               "tail_sha256": _sha_range(p, max(0, s - (1 << 20)), 1 << 20)}
        if deep:
            rec["sha256"] = sha256_file(p)
        shards[rel] = rec
    cfgs = {}
    for rel in CONFIG_FILES:
        p = os.path.join(d, rel)
        if os.path.exists(p):
            cfgs[rel] = sha256_file(p)
    return {"name": name, "family": b["family"], "model_dir": d,
            "source": b["source"], "license": b["license"],
            "pixel_control": b["pixel_control"],
            "n_files": len(files), "bytes": int(sum(s for _rel, s in files)),
            "name_size_manifest_sha256": ns.hexdigest(),
            "shards": shards, "config_sha256": cfgs, "deep": bool(deep),
            "note": "computed by the run that reports it; no constant is asserted against a "
                    "stale pin. deep=True hashes whole shards (probe only)."}
