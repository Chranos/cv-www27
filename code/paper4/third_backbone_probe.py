#!/usr/bin/env python3
"""Read-only feasibility probe for the third-backbone attempt (Plan r5).

What it does: records which open weights exist on this device, and for the one
candidate that is plausibly a different family (Mistral-Small-3.2-24B-Instruct-2506)
records, probe by probe, whether the installed stack can load it as released.

What it does NOT do: no forward passes, no weight loading, no downloads, no
installs, no writes outside the run directory, no changes to any cell, arm,
threshold, criterion or claim. The instrument check itself is NOT executed here;
this file only records why it cannot be.

Writes: metrics.json (repo root) and probes/probes.jsonl.
"""
import json
import os
import platform
import stat as statmod
import struct
import sys
import traceback

DATA_ROOT = "/path/to/data"
MODELS_ROOT = os.environ.get("MODELS_ROOT", "/path/to/models")
STARTLUX_ROOT = os.environ.get("STARTLUX_ROOT", "/path/to/startlux")
SHARED_MODELS = os.environ.get("SHARED_MODELS", "/path/to/shared-models")
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

CANDIDATE = os.path.join(MODELS_ROOT, "Mistral-Small-3.2-24B-Instruct-2506")

PASS_LINE = ("mapping_ok = (n_pairs>0) and (n_evaluable==n_pairs) and "
             "(n_contradictions==0)")

probes = []


def rec(name, ok, detail):
    entry = {"probe": name, "ok": bool(ok), "detail": detail}
    probes.append(entry)
    line = json.dumps(entry, ensure_ascii=False)
    print(line[:2000], flush=True)
    return entry


def dir_size(path):
    """Sum of regular-file bytes and file count, metadata only."""
    total, n = 0, 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            fp = os.path.join(dirpath, name)
            try:
                st = os.lstat(fp)
            except OSError:
                continue
            if statmod.S_ISREG(st.st_mode):
                total += st.st_size
                n += 1
    return total, n


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def dir_record(path, label):
    """Facts about one directory of weights; never reads tensor bodies."""
    out = {"label": label, "path": path, "exists": os.path.isdir(path)}
    if not out["exists"]:
        return out
    try:
        total, n = dir_size(path)
        out["bytes_total"] = total
        out["n_files"] = n
    except Exception as exc:  # pragma: no cover - defensive
        out["dir_size_error"] = f"{type(exc).__name__}: {exc}"
    try:
        names = sorted(os.listdir(path))
    except OSError as exc:
        out["list_error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["has_config_json"] = "config.json" in names
    out["has_preprocessor_config"] = "preprocessor_config.json" in names
    out["has_tokenizer_config"] = "tokenizer_config.json" in names
    out["has_tokenizer_json"] = "tokenizer.json" in names
    out["hf_weight_files"] = [n for n in names
                              if n in ("model.safetensors", "pytorch_model.bin",
                                       "model.safetensors.index.json",
                                       "pytorch_model.bin.index.json")]
    out["consolidated_safetensors"] = "consolidated.safetensors" in names
    out["zero_byte_weight_like"] = []
    for n in names:
        if n.endswith((".safetensors", ".bin")) or n in ("tokenizer.json", "vocab.json",
                                                         "merges.txt", "tokenizer.model"):
            fp = os.path.join(path, n)
            try:
                if statmod.S_ISREG(os.lstat(fp).st_mode) and os.path.getsize(fp) == 0:
                    out["zero_byte_weight_like"].append(n)
            except OSError:
                pass
    if out["has_config_json"]:
        try:
            cfg = read_json(os.path.join(path, "config.json"))
            out["model_type"] = cfg.get("model_type")
            out["architectures"] = cfg.get("architectures")
            out["has_vision_config"] = "vision_config" in cfg
            vc = cfg.get("vision_config") or {}
            out["vision_config_subset"] = {k: vc.get(k) for k in
                                           ("model_type", "patch_size", "image_size",
                                            "hidden_size", "num_hidden_layers")
                                           if k in vc}
        except Exception as exc:
            out["config_error"] = f"{type(exc).__name__}: {exc}"
    return out


def safetensors_header(path):
    """Read only the JSON header: tensor names, dtypes. No tensor bodies."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    keys = [k for k in header if k != "__metadata__"]
    return {
        "n_tensors": len(keys),
        "dtypes": sorted({v.get("dtype") for k, v in header.items()
                          if k != "__metadata__"}),
        "first_keys": keys[:6],
        "vision_keys": [k for k in keys if "vision" in k][:6],
        "embedding_keys": [k for k in keys
                           if "tok_embeddings" in k or "lm_head" in k][:4],
        "hf_style_naming": any(k.startswith("model.language_model")
                               or k.startswith("model.vision_tower") for k in keys),
    }


def main():
    metrics = {
        "what": ("Read-only feasibility probe for the third-backbone attempt: what open "
                 "weights exist on this device, and why the selected candidate cannot be "
                 "loaded as released. No forward passes, no weight loading, no downloads, "
                 "no installs."),
        "state": "blocked",
        "design": {
            "plan_id": "plan-four-readings-second-backbone-r5",
            "plan_version": 1,
            "instrument_check_executed": False,
            "pass_line": PASS_LINE,
            "thresholds_changed": False,
            "stimulus_changed": False,
            "cells_arms_changed": False,
            "nothing_recomputed_from_other_backbones": True,
        },
        "seed": {"EXP_SEED": os.environ.get("EXP_SEED")},
        "environment": {
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "counts": {
            "forward_passes": 0,
            "model_calls": 0,
            "weights_loaded": 0,
            "downloads": 0,
            "installs": 0,
            "gpu_used": False,
        },
    }

    # ---------------------------------------------------------------- inventory
    inventory = []
    inventory.append(dir_record(os.path.join(DATA_ROOT, "qwen2vl-7b-2cbada593a2b"),
                                "backbone_1_qwen2vl_7b_already_tried"))
    inventory.append(dir_record(os.path.join(DATA_ROOT, "llavaov-7b-ov-hf-ad4cb27d2a3f"),
                                "backbone_2_llavaov_7b_hf_already_tried"))
    inventory.append(dir_record(os.path.join(DATA_ROOT, "llavaov-7b-ov-b15850a51ee6"),
                                "llavaov_non_hf_copy_build_failed"))
    inventory.append(dir_record(os.path.join(DATA_ROOT, "clip-vit-b32-5587187bcb3a"),
                                "clip_vit_b32_not_a_generative_vlm"))
    for size in ("SL-9B", "SL-4B", "SL-2B", "SL-0.8B"):
        inventory.append(dir_record(os.path.join(STARTLUX_ROOT, size),
                                    f"startlux_{size}"))
    for name in ("Qwen3-VL-8B-Instruct", "llava-v1.6-mistral-7b",
                 "clip-vit-large-patch14"):
        inventory.append(dir_record(os.path.join(SHARED_MODELS, name),
                                    f"shared_public_{name}"))
    for name in ("GLM-4.5-Air-AWQ-4bit", "Meta-Llama-3.1-8B-Instruct",
                 "Mistral-7B-Instruct-v0.3", "Phi-4-mini-instruct",
                 "Qwen2.5-7B-Instruct", "granite-3.1-8b-instruct",
                 "granite-3.1-8b-instruct-ms", "llama-3.3-70b-instruct-awq"):
        inventory.append(dir_record(os.path.join(MODELS_ROOT, name), f"text_only_{name}"))
    if os.path.isdir(HF_CACHE):
        cache_entries = []
        for name in sorted(os.listdir(HF_CACHE)):
            p = os.path.join(HF_CACHE, name)
            if os.path.isdir(p):
                b, n = dir_size(p)
                cache_entries.append({"name": name, "bytes_total": b, "n_files": n})
        metrics["hf_hub_cache"] = {"path": HF_CACHE, "n_entries": len(cache_entries),
                                   "entries": cache_entries}
    metrics["inventory"] = inventory
    rec("inventory", True, {"n_dirs": len(inventory),
                            "with_vision_config": [d["label"] for d in inventory
                                                   if d.get("has_vision_config")],
                            "complete_vlm_like": [d["label"] for d in inventory
                                                  if d.get("has_vision_config")
                                                  and not d.get("zero_byte_weight_like")]})

    # ------------------------------------------------------- candidate probes
    cand = dir_record(CANDIDATE, "candidate_mistral_small_3_2_24b")
    metrics["selected_candidate"] = {
        "path": CANDIDATE,
        "why_plausibly_different": (
            "Four axes differ at once from both tried backbones: Pixtral vision encoder "
            "(vision_config.model_type=pixtral, patch_size 14, hidden 1024, 24 layers, "
            "image_size 1540) vs backbone 1's Qwen dynamic-resolution ViT (M-RoPE) and vs "
            "backbone 2's SigLIP-so400m-384 + MLP connector; Mistral text decoder vs "
            "Qwen2-7B-Instruct in both; Tekken tokenizer vs Qwen BPE."),
        "local_license_file": "absent (no LICENSE in the local copy; upstream licence not verifiable offline)",
        "inventory_record": cand,
    }
    rec("candidate_dir", True, {k: cand.get(k) for k in
                                ("exists", "bytes_total", "n_files", "has_config_json",
                                 "has_preprocessor_config", "has_tokenizer_config",
                                 "hf_weight_files", "consolidated_safetensors")})

    # safetensors header: names and dtypes only
    st_path = os.path.join(CANDIDATE, "consolidated.safetensors")
    try:
        hdr = safetensors_header(st_path)
        metrics["selected_candidate"]["safetensors_header"] = hdr
        rec("safetensors_header", True, hdr)
    except Exception as exc:
        metrics["selected_candidate"]["safetensors_header"] = {
            "error": f"{type(exc).__name__}: {exc}"}
        rec("safetensors_header", False, metrics["selected_candidate"]["safetensors_header"])

    # transformers: what it can and cannot build from this directory
    loader = {}
    try:
        import transformers
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        loader["transformers_version"] = transformers.__version__
        loader["n_config_types"] = len(CONFIG_MAPPING.keys())
        loader["config_types_of_interest"] = {
            k: (k in CONFIG_MAPPING.keys())
            for k in ("qwen3_vl", "qwen2_vl", "qwen2_5_vl", "mistral3",
                      "llava_onevision", "pixtral", "qwen3_5")
        }
        rec("transformers_config_types", True, loader["config_types_of_interest"])

        try:
            cfg = AutoConfig.from_pretrained(CANDIDATE, local_files_only=True)
            loader["autoconfig"] = {"ok": True, "class": type(cfg).__name__,
                                    "model_type": cfg.model_type}
        except Exception as exc:
            loader["autoconfig"] = {"ok": False,
                                    "error": f"{type(exc).__name__}: {exc}"}
        rec("autoconfig", loader["autoconfig"]["ok"], loader["autoconfig"])

        for key, fn in (("autotokenizer", AutoTokenizer.from_pretrained),
                        ("autoprocessor", AutoProcessor.from_pretrained)):
            try:
                obj = fn(CANDIDATE, local_files_only=True)
                loader[key] = {"ok": True, "class": type(obj).__name__}
            except Exception as exc:
                loader[key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            rec(key, loader[key]["ok"], loader[key])

        # from_pretrained: fails at weight-file resolution before any tensor is read
        try:
            from transformers import Mistral3ForConditionalGeneration
            Mistral3ForConditionalGeneration.from_pretrained(
                CANDIDATE, local_files_only=True, low_cpu_mem_usage=True)
            loader["from_pretrained"] = {"ok": True}
        except Exception as exc:
            loader["from_pretrained"] = {"ok": False,
                                         "error": f"{type(exc).__name__}: {exc}"}
        rec("from_pretrained", loader["from_pretrained"]["ok"],
            {"error": loader["from_pretrained"].get("error", "")[:600]})

        # processor defaults vs the checkpoint's own vision config
        try:
            from transformers import PixtralImageProcessor
            proc = PixtralImageProcessor()
            defaults = {"patch_size": proc.patch_size, "size": proc.size}
            cfg_raw = read_json(os.path.join(CANDIDATE, "config.json"))
            vc = cfg_raw.get("vision_config", {})
            loader["processor_defaults_vs_checkpoint"] = {
                "processor_defaults": defaults,
                "checkpoint_vision_config": {"patch_size": vc.get("patch_size"),
                                             "image_size": vc.get("image_size")},
                "match": (defaults.get("patch_size") == vc.get("patch_size")),
            }
        except Exception as exc:
            loader["processor_defaults_vs_checkpoint"] = {
                "error": f"{type(exc).__name__}: {exc}"}
        rec("processor_defaults_vs_checkpoint", True,
            loader["processor_defaults_vs_checkpoint"])

        # config fields the input pipeline would need
        try:
            cfg_raw = read_json(os.path.join(CANDIDATE, "config.json"))
            loader["checkpoint_config_subset"] = {
                k: cfg_raw.get(k) for k in
                ("model_type", "architectures", "image_token_index",
                 "spatial_merge_size", "vision_feature_layer", "torch_dtype")
                if k in cfg_raw
            }
        except Exception as exc:
            loader["checkpoint_config_subset"] = {"error": f"{type(exc).__name__}: {exc}"}
        rec("checkpoint_config_subset", True, loader["checkpoint_config_subset"])

    except Exception as exc:
        loader["transformers_import_error"] = f"{type(exc).__name__}: {exc}"
        rec("transformers_import", False, loader["transformers_import_error"])
    metrics["selected_candidate"]["loader_probes"] = loader

    # alternative loaders / kernels, and the startlux requirements
    modules = {}
    for mod in ("mistral_inference", "mistral_common", "vllm",
                "flash_linear_attention", "fla", "causal_conv1d"):
        try:
            m = __import__(mod)
            modules[mod] = {"ok": True, "version": getattr(m, "__version__", None)}
        except Exception as exc:
            modules[mod] = {"ok": False, "error": type(exc).__name__}
    metrics["modules"] = modules
    rec("modules", True, modules)

    # --------------------------------------------------- outcome of the attempt
    loadable = bool(loader.get("from_pretrained", {}).get("ok"))
    metrics["outcome"] = {
        "loadable_as_released": loadable,
        "blocked": not loadable,
        "blocking_reasons": [
            "weights are in Mistral original format: consolidated.safetensors holds "
            f"{metrics['selected_candidate'].get('safetensors_header', {}).get('n_tensors')} "
            "tensors named layers.*/tok_embeddings/vision_encoder.*, not the HF "
            "model.language_model.*/model.vision_tower.* naming",
            "no model.safetensors / model.safetensors.index.json / pytorch_model.bin in the directory",
            "no preprocessor_config.json, so AutoProcessor cannot build the image processor",
            "PixtralImageProcessor defaults (patch 16, longest_edge 1024) contradict this "
            "checkpoint's vision_config (patch 14, image_size 1540)",
            "no tokenizer_config.json / chat template in the directory",
            "no Mistral->HF converter in the installed transformers, and mistral_inference is absent",
        ],
        "why_not_ported": (
            "Hand-renaming 585 tensors and hand-building the image preprocessing would "
            "produce an object that is not the released model; one silent mismatch would be "
            "read by the instrument as 'this model fails the pass line'. That is changing the "
            "measured object, not the minimal wiring needed to make the naming step answerable."),
        "instrument_check": {
            "executed": False,
            "reason": "no loadable weights; the Plan's blocked branch applies",
            "pass_line_unchanged": PASS_LINE,
        },
    }

    # ------------------------------------------------------------ other candidates
    ruled_out = []
    for entry in inventory:
        if entry["label"].startswith("startlux"):
            ruled_out.append({
                "label": entry["label"],
                "reason": "model_type qwen3_5 absent from the installed transformers; "
                          "requirements demand transformers>=5.8 + flash-linear-attention + "
                          "causal-conv1d, none installed, no network; CC-BY-NC-4.0; typed-decision API",
            })
        elif entry["label"].startswith("shared_public"):
            ruled_out.append({
                "label": entry["label"],
                "reason": "zero-byte shell: weight and tokenizer files are 0 bytes",
                "zero_byte_files": entry.get("zero_byte_weight_like", []),
            })
        elif entry["label"].startswith("text_only"):
            ruled_out.append({"label": entry["label"],
                              "reason": "no vision_config: text-only model"})
    metrics["ruled_out"] = ruled_out

    metrics["limits"] = [
        "No forward passes, no weight loading, no downloads, no installs; nothing outside "
        "the run directory is written.",
        "The inventory covers the roots listed in `inventory`; /tmp, /root and other users' "
        "private directories were not scanned.",
        "'Cannot be loaded' means in this environment without changing the measured object; "
        "no weight conversion was attempted, so it is not a claim about other environments.",
        "The licence of the candidate is unverified (no LICENSE file locally, no network).",
        "Numbers from the two tried backbones are cited elsewhere and are not recomputed here.",
    ]

    os.makedirs("probes", exist_ok=True)
    with open(os.path.join("probes", "probes.jsonl"), "w", encoding="utf-8") as fh:
        for entry in probes:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    metrics["probes"] = {
        "n_probes": len(probes),
        "n_failed": sum(1 for p in probes if not p["ok"]),
        "details_file": "probes/probes.jsonl",
    }

    with open("metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"wrote": "metrics.json", "state": metrics["state"],
                      "loadable": loadable, "n_probes": len(probes)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        # Still leave a metrics.json behind so the failure is recorded, not lost.
        with open("metrics.json", "w", encoding="utf-8") as fh:
            json.dump({"state": "crash", "error": traceback.format_exc()[-4000:]}, fh,
                      ensure_ascii=False, indent=2)
        sys.exit(1)
