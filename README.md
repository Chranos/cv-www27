# Frame Dependence in Long-Video Question Answering: Four Readings

Code and per-question records for the paper *Frame Dependence in Long-Video Question Answering: Four Readings and What They Do Not Establish*.

| Directory | What it holds |
|---|---|
| `records/` | The archived outputs of the measurement runs behind every number the paper prints, the script that recomputes those numbers from them, and the exact prompt-building code each run used. See `records/README.md` and `records/manifest.json`. |
| `code/` | The code that took the measurements: `paper4/four.py` runs one reading per invocation, with the frame sampling, distortion and question-draw helpers it imports (`twin5/`, `gen_common.py`, `mvp_common.py`, `r*_common.py`) and the pinned question sets it reads (`data/`). |

## Recompute the paper's numbers (no GPU, no model, standard library only)

```sh
cd records
python3 recompute/pack_regenerate_acc.py
```

This rewrites `numbers.tex` and the six `tab-*.tex` tables in `records/` from the records under `records/sources/`; each comes out byte-identical to the copy in `records/regenerated_pack/`.

## The readings and the runs that produced them

| Paper | Run (`records/sources/`) | Command (from `code/`) |
|---|---|---|
| R1 selection at 16 frames, LongVideoBench, 110 questions; R2 the no-frames arm on the same carrier | `task-23cd232b2b` | `P4_READING=selection P4_MODE=full P4_PLAN_VERSION=4 P4_PLAN_ID=plan-paper-full-four-readings-r3 bash paper4/run.sh` |
| R3 single frame against sixteen, LVBench, 24 units | `task-240d7a5d60` | `P4_READING=single P4_MODE=full P4_QSELECT=draw P4_QDRAW_SEED=20261021 P4_PLAN_VERSION=4 P4_PLAN_ID=plan-single-frame-random-question-r4 bash paper4/run.sh` |
| R4 pre-registered distortion, LVBench, 17 pairs | `task-b595df723d` (random question draw), `task-259e5c2b3a` (original set) | `P4_READING=distortion P4_MODE=full P4_QSELECT=draw P4_PLAN_VERSION=2 P4_PLAN_ID=plan-distortion-random-question bash paper4/run.sh` |
| Donor diagnostic (instrument check, no finding) | `task-07c9beea59` | `P4_READING=donor P4_MODE=full P4_PLAN_VERSION=4 P4_PLAN_ID=plan-paper-full-four-readings-r3 bash paper4/run.sh` |

`records/manifest.json` (`runs`) gives each run's commit and the sha256 of `four.py` it ran. The plan identifiers only label the run in its `metrics.json`.

## Re-running a measurement

A run needs one CUDA GPU, Python 3 with `torch`, `transformers`, `numpy`, `scipy` and `opencv-python`, the weights of Qwen2-VL-7B-Instruct, and the benchmark videos, which are not redistributed here (LongVideoBench validation, LVBench, MLVU; see their own licences). Put them under one directory and pass it as `DATA_ROOT`, with the directory names the code reads:

| Directory under `DATA_ROOT` | Contents |
|---|---|
| `qwen2vl-7b-2cbada593a2b` | Qwen2-VL-7B-Instruct |
| `lvb-val-e383ac257332` | LongVideoBench validation videos and annotations |
| `lvbench-24209d43cbdd` | LVBench videos and `video_info.meta.jsonl` |
| `mlvu-needle-plotqa-38e93f060520` | the MLVU needle and plotQA videos of the donor diagnostic |

The suffix of each name is the hash of the copy the runs used; the runs record the manifest of what they read in their `metrics.json`, so a re-run on another copy can be compared file by file. Set `PYTHON` to the interpreter to use (default `python3`) and `EXP_SEED` to the seed stream (the paper's runs used 0). Each run writes `metrics.json` and `results/four_<reading>_samples.jsonl` into `code/`.

`data/carrier3b_pin_r3.json`, the pinned question set of the donor diagnostic, is not included: it carries MLVU question text and answers under MLVU's licence. `r3_gen_pin.py --write data/carrier3b_pin_r3.json` rebuilds it from those MLVU videos, and `four.py` refuses a rebuilt file whose sha256 differs from the one the run froze.

## Not included

Model weights and benchmark data (third-party licences), the raw run logs and environment dumps (device-identifying; superseded by each run's `metrics.json`), and code for the project's earlier pilots that no number in the paper depends on.
