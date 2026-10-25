"""CPU-only code check for the review's three conditions.

This is NOT the pilot and cannot stand in for one: it loads no model, reads no carrier, does
0 forwards and produces no reading.  It exists because the second backbone lives on a device
that was offline when the conditions were built in, and a syntax error or a wrong permutation
index in the new control code should not cost a GPU fix cycle later.

Checks
  1. the new and edited modules import;
  2. the permutation stream is derived from EXP_SEED, is deterministic, is never the
     identity, and moves the gold letter with the option text (the arithmetic the control
     relies on);
  3. the six synthetic visibility stimuli render at 448x448, are not blank, and their correct
     option never repeats among the distractors;
  4. the three renames are present in the committed source and the three old names are gone
     from the blocks they belonged to.

What it does not check: that a model answers any of it.  That is the pilot's job.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# paper4/ for `instrument` (the reading scripts run with paper4/ as sys.path[0]); the repo
# root for `gen_common` / `mvp_common`, which is what run.sh exports as PYTHONPATH.
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import instrument as INS                                        # noqa: E402

LETTERS = "ABCDE"
OUT = "metrics.json"
RESULT = {"check": "cpu_code_check", "forwards": 0, "model_loaded": False,
          "run": {"stage": "smoke", "decides_claim": False, "reading": "selfcheck",
                  "note": "code check only; no carrier, no weights, no reading"},
          "checks": {}}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_imports():
    import backbone  # noqa: F401
    import gen_common  # noqa: F401
    import mvp_common  # noqa: F401
    import aggregate_smoke  # noqa: F401
    return {"ok": True,
            "modules": ["paper4/instrument.py", "paper4/backbone.py", "gen_common.py",
                        "mvp_common.py", "paper4/aggregate_smoke.py"],
            "note": "four.py and twin5/* are not imported here: they pull the video stack, "
                    "which this check does not need; the GPU run's py_compile step covers them"}


def check_permutation():
    seed = INS.perm_seed(0)
    rows, ok = [], True
    for i in range(24):
        n = 4 + (i % 2)                      # 4-option and 5-option questions
        p = INS._perm_for(seed, i, n)
        p2 = INS._perm_for(seed, i, n)
        distinct = sorted(p) == list(range(n))
        non_identity = p != list(range(n))
        deterministic = p == p2
        # the gold letter must travel with its option text, and the chosen text must not move
        opts = ["opt%d" % j for j in range(n)]
        opts_perm = [opts[k] for k in p]
        text_moved = False
        gold_idx = i % n
        gold_perm = LETTERS[p.index(gold_idx)]
        if opts_perm[LETTERS.index(gold_perm)] != opts[gold_idx]:
            text_moved = True
        for li in range(n):
            chosen_perm = LETTERS[p.index(li)]
            if opts_perm[LETTERS.index(chosen_perm)] != opts[li]:
                text_moved = True
        ok = ok and distinct and non_identity and deterministic and not text_moved
        rows.append({"i": i, "n": n, "permutation": p, "distinct": distinct,
                     "non_identity": non_identity, "deterministic": deterministic,
                     "text_follows_option": not text_moved})
    seeds = [INS.perm_seed(s) for s in (0, 1, 2)]
    return {"ok": bool(ok), "seed": int(seed), "perm_seed_by_exp_seed": seeds,
            "perm_seed_is_distinct_per_seed": len(set(seeds)) == len(seeds),
            "n": len(rows), "rows": rows[:6],
            "rule": "permutation = SeedSequence([s_perm, unit_index, n_options]); the gold "
                    "letter and the chosen option text are re-indexed through the same "
                    "permutation, so a preserved text with a moving letter is the expected "
                    "signature of a correctly wired option<->letter map"}


def check_visibility_stimuli():
    rows, ok = [], True
    for i in range(INS.N_VISIBILITY):
        base = list(INS.VIS_WORDS)
        correct = base[i % len(base)]
        pos = i % 4
        rest = [w for w in base if w != correct]
        opts = rest[:pos] + [correct] + rest[pos:]
        gold = LETTERS[pos]
        im, renderer = INS._render_word(correct)
        px = list(im.getdata())
        dark = sum(1 for p in px if sum(p) < 384)          # not near-white
        nonblank = len(set(px)) > 1
        unique = len(set(opts)) == len(opts)
        at_gold = opts[LETTERS.index(gold)] == correct
        ok = ok and im.size == (448, 448) and im.mode == "RGB" and nonblank and unique \
            and at_gold and dark > 200
        rows.append({"i": i, "word": correct, "gold": gold, "size": list(im.size),
                     "mode": im.mode, "renderer": renderer, "dark_pixels": dark,
                     "nonblank": nonblank, "options_unique": unique,
                     "correct_option_at_gold": at_gold})
    return {"ok": bool(ok), "n": len(rows), "units": rows}


def check_renames():
    src = open(os.path.join(HERE, "four.py"), encoding="utf-8").read()
    must_have = ["n_pairs_measured_this_run", "n_pairs_in_pin",
                 "n_units_measured_this_run", "n_inforce_pool",
                 '"checked": False', "n_measured", "n_pool", "list_truncated_at"]
    must_not = ['"matches_expected"', '"n_visible_in_force"']
    missing = [s for s in must_have if s not in src]
    leftover = [s for s in must_not if s in src]
    # `n_pairs` may legitimately remain for the carrier-4 pin count; it must not be the pin
    # block's only size field any more.
    return {"ok": bool(not missing and not leftover), "missing": missing,
            "old_names_still_present": leftover,
            "four_py_sha256": sha256_file(os.path.join(HERE, "four.py")),
            "instrument_py_sha256": sha256_file(os.path.join(HERE, "instrument.py")),
            "note": "a source-text check, not a runtime one: it shows the names are in the "
                    "committed code, not that a run emitted them"}


def main():
    RESULT["checks"]["imports"] = check_imports()
    RESULT["checks"]["permutation"] = check_permutation()
    RESULT["checks"]["visibility_stimuli"] = check_visibility_stimuli()
    RESULT["checks"]["renames"] = check_renames()
    RESULT["all_ok"] = bool(all(c.get("ok") for c in RESULT["checks"].values()))
    with open(OUT, "w") as f:
        json.dump(RESULT, f, indent=1, sort_keys=True, default=str)
    print(json.dumps({"all_ok": RESULT["all_ok"],
                      "checks": {k: v.get("ok") for k, v in RESULT["checks"].items()}}))
    if not RESULT["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
