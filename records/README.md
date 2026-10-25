# Four-readings release bundle (anonymized)

Contents
- sources/<task>/...  archived outputs of the measurement runs and the pack
  chain, scrubbed of user names, account ids, device paths, host name, node id
  and agent-session UUIDs. Each file's pre-scrub sha256 was verified against
  the run's Evidence pin before scrubbing (see manifest.json).
    measurement runs:
      task-23cd232b2b  (a)+(b) selection, 110 questions
      task-240d7a5d60  (c) single-frame, random-question re-run, 24 units
                       (+ results/single_qdraw*.json[l]: the drawn question set)
      task-259e5c2b3a  (d1) distortion, original question set, 17 pairs
      task-07c9beea59  (d2) donor diagnostic, 23 units
      task-b595df723d  (d1) distortion, random-question re-run, 17 pairs
                       (+ results/distortion_ddraw*.json[l]: the drawn set)
    pack chain (verification pins for the recompute script):
      task-1f59540a60, task-219f5f1017, task-282e07a239, task-63b0f5bf56
- recompute/pack_regenerate_acc.py  derives every printed number in the pack
  from these archived records (pure arithmetic; no model calls). Run it from
  THIS directory:  python3 recompute/pack_regenerate_acc.py
  It rewrites the seven pack files (numbers.tex, tab-*.tex) in the working
  directory. regenerated_pack/ holds the copy produced inside the packaging
  run, with its sha256 comparison against the pinned pack recorded in
  manifest.json (6/7 byte-identical; tab-distortion.tex carries the known
  r2 header fix).
- recompute/pack_full_stage.py  the verifier of record for the pack chain;
  it expects the repository layout (paper4/ + sources/) and is included for
  reference.
- prompts/  the exact prompt-building code each run used at its commit
  (see manifest.prompt_provenance for the run -> builder mapping and the
  source file hashes). All arms share one prompt template per module; answers
  are decoded greedily with max_new_tokens=8.

Kept on purpose (they identify runs and data, not people): task/plan/evidence
ids, commit hashes, dataset directory names, video ids, question ids.

Not included, and why: see manifest.json "excluded".
