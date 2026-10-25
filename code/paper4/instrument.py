"""Instrument controls for the four readings, on either backbone.

Plan: plan-four-readings-second-backbone-r3 (these controls are bound to that version)
Review: review-pilot-gate-four-readings-second-backbone-r2 (six required fixes)

  r2 §5.1 / r3 §1.1  the option control is SPLIT IN TWO, and this is an AFTER-THE-FACT change,
      written down as one: the old rule (`n_text_preserved == n` to pass) was seen, in
      task-49c7675427, to be reporting the model's sensitivity to option order as a broken
      option<->letter map.  It demanded a preserved chosen TEXT and got 5/6, 4/6, 2/3, 5/6
      (0 of 4 cells passed) while 0 of 21 units carried the mapping-fault signature, so the
      rule was measuring something other than what it claimed.  It is therefore split into
        * option_control.mapping           -- THE CRITERION.  The letter a unit's own ask
          produced must point at the option the model NAMES IN FREE TEXT, read from a naming
          ask on the same frames, the same question and the same option list.  The text is
          never derived from the letter: the old `chosen_text = opts[LETTERS.index(letter)]`
          was self-consistent by construction and could prove nothing about the map.
        * option_control.order_sensitivity -- A DESCRIPTIVE QUANTITY with NO pass line:
          n_text_preserved/n, the four pattern classes and the mapping-fault signature count.
          It goes into table 5 and gates nothing.
      The r2 result is NOT retro-fitted into a pass: it was 'criterion not well defined', and
      the corrected control is re-run once under the r3 plan.
  r2 §5.2 / r3 §1.2  instrument_failure is IMPLEMENTED here, not only worded.  `instrument_guard`
      names the triggers and the consequence (four.py writes `landing = null` and marks the
      branch blocks withheld).  The "letter distribution nearly constant" trigger is the plan's
      own literal reading -- every answered cell the same letter -- evaluated only at
      n >= NEAR_CONSTANT_MIN_N, and it EXCLUDES the annotation-only arms: `constant_prior` is
      constant by construction (single: 24/24 A) and would otherwise fire in every cell.
  r2 §5.3 / r3 §1.3  vision tokens are counted, never derived from config strings: per arm and
      prompt path in the `vision_tokens` block here, and per unit and per arm in the samples
      file by the caller (four.py).  Measured: 1,149 per image on the standard path, 1,037 on
      the donor path, equal across the three image-bearing prompt paths inside one run.
  r2 §5.5 / r3 §1.5  KEPT, unchanged: letter_decode, determinism on all three prompt paths,
      answer_visibility.

  r3 §2 / r4 §3.1  the NAMING STEP is redesigned, and this too is an after-the-fact change:
      the criterion could not be evaluated at all (v3: 26 of 42 pairs unevaluable, 6 of the 16
      evaluable ones contradicting, the donor cell 0 of 12).  A letter reply is no longer
      silently counted as unparseable.  Channel A prints the option TEXTS only, the assistant
      turn is opened by a forced prefix, and the first generated token cannot be a bare letter
      (computed from the vocabulary, not from five hard-coded ids).  A bare letter is then
      handled by a stated rule -- one escalation retry, then a named `letter_reply_*` failure --
      and the letter is never turned into option text.  Channel B is a test-retest of the same
      question with the same option texts, giving the naming step's own noise floor.  A
      deliberate offset of the letter->option map is run through the SAME comparison function
      to show the criterion fires.  The criterion, its pass line, the arms, the cells and every
      threshold are untouched.

Nothing in this module is an arm, a cell, a threshold or a criterion of a reading.  Every block
is a control a reader can re-check against the raw strings the run writes to its samples file.

Costs per reading, in forwards: 2 on the 16-frame answer path (determinism), 2 on the 1-frame
answer path (determinism), 2 single-image scorings (determinism), up to 6 asks on the reading's
own driving arm plus 2x6 free-text naming asks (the option control), 6 synthetic asks + 6
single-image scorings (visibility).  All of it goes through the same Fwd accumulator, tagged
`control:*`, so the forward account stays closed and no measurement arm is touched.
"""
from collections import Counter
import re

import numpy as np

import gen_common as G
import mvp_common as C

LETTERS = ["A", "B", "C", "D", "E"]

N_PERM_UNITS = 6          # questions whose option order is rotated, when the run has them
N_VISIBILITY = 6          # synthetic "the answer is in the picture" units
NAMED_MAX_NEW_TOKENS = 96  # the free-text naming ask only; every measurement arm uses 8

# ---- v4 naming step (review r3 §2; Plan r4 §3.1 items (3a)-(3f)) -----------------------
# The forced opening of the assistant turn.  This is not an instruction: it is already in the
# input, so the model continues inside a quote and cannot answer with a bare letter.
NAMING_FORCED_PREFIX = 'The option text is: "'
# The one escalation retry, used only after a bare-letter reply.
NAMING_ESCALATION_INSTRUCTION = ("Write only that option's words. Do not write a letter or "
                                 "a number.")
# A bare single Latin letter, optionally wrapped in ()/[] and/or closed with . : ) -
_LETTER_TOKEN = re.compile(r"^\(?[A-Za-z]\)?[.:)\-]?$")
# The same, restricted to the option letters: a reply of this shape is a `letter_reply`.
_LETTER_ONLY = re.compile(r"^\(?[A-Ea-e]\)?[.:)\-]?$")
# Leading-space markers used by byte-level vocabs (Ġ, ▁) plus a plain space.
_LEAD_MARKS = ("\u0120", "\u2581", " ")
# The bit-identical repeat of channel A, on this many units (proves the decode path is
# deterministic; it is structurally zero under greedy decoding and is NOT the noise floor).
N_IDENTICAL_REPEAT_UNITS = 2

# ---- the registered refusal rule (Plan plan-single-frame-random-question-r3 §9) -----
# WRITTEN DOWN AS AN AFTER-THE-FACT CORRECTION (old behaviour / new behaviour / reason):
#   old: any naming reply that did not match an option was `unparseable`; the pair was
#        unevaluable and one such pair made the whole cell `incomplete`.  In task-0cdb808f53
#        two pairs (both orders of one unit) were lost that way, and that was the only reason
#        the cell could not report.
#   new: a reply that normalises to one of REFUSAL_PHRASES is a REFUSAL, not a failure to
#        parse.  If the unit's OWN option list carries an option that normalises to the same
#        phrase, the refusal names that option (`refusal_matched_option`) and the pair is
#        judged by the same comparison as any other pair.  If it does not, the pair stays
#        unevaluable (`refusal_no_option`) and is counted.
#   reason: a refusal is a legitimate answer shape on a forced-choice question; when the
#        question itself offers a "none of the above" option, that is an option's name and not
#        a parse failure.  The restriction to "the option list actually contains it" is the
#        point: a refusal is never mapped onto a neighbouring option by meaning, which would
#        invent the very mapping the criterion exists to test.
# The list is closed and matched on normalised string equality (no fuzz, no letters).  A bare
# `none` is deliberately NOT in it: when an option is literally "None", equality already
# matches it and no extra rule is needed.
REFUSAL_PHRASES = ("none of the above", "none of these", "none of the options",
                   "none of the choices", "not enough information", "cannot be determined",
                   "insufficient information")

VIS_WORDS = ("SEVENTEEN", "ORANGE", "GUITAR", "PLANET")   # no word a substring of another

# Arms whose letters are written by the code from the annotations and never produced by the
# model.  `constant_prior` is constant by construction, so the "letter distribution nearly
# constant" trigger must not see it (review r2 §5.2).
ANNOTATION_ONLY_ARMS = ("constant_prior",)

# The floor under the near-constant trigger.  Under a uniform-letter null, all n cells landing
# on one letter has probability 5*(1/5)^n: at n=3 that is ~4% for one arm and certain to happen
# somewhere across a dozen arms, at n=5 it is 0.16%.  Below the floor the check is recorded
# `not_evaluable`, which is a statement about the run's size and not a pass.
NEAR_CONSTANT_MIN_N = 5


# ---------------------------------------------------------------- helpers

def perm_seed(exp_seed):
    """A seed stream of its own, derived from EXP_SEED (nothing hard-coded)."""
    return (int(exp_seed) * 1000003 + 53) % (2 ** 31 - 1)


def _perm_for(seed, i, n):
    ss = np.random.SeedSequence([int(seed), int(i), int(n)])
    p = [int(x) for x in np.random.default_rng(ss).permutation(n)]
    if p == list(range(n)):                 # an identity rotation would test nothing
        p = list(range(1, n)) + [0]
    return p


def _build_inputs(proc, pils, prompt):
    msgs = G._msgs(pils, prompt)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if pils:
        return proc(text=[text], images=list(pils), return_tensors="pt")
    return proc(text=[text], return_tensors="pt")


# ---------------------------------------------------------------- letter decode

def letter_decode(proc, ids):
    rows, ok = [], True
    for L, i in zip(LETTERS, ids):
        d = proc.tokenizer.decode([int(i)])
        hit = bool(d.strip() == L)
        ok = ok and hit
        rows.append({"letter": L, "id": int(i), "decoded": d, "ok": hit})
    return {"rows": rows, "all_ok": bool(ok),
            "rule": "each option letter is a single token; decoding that id must give the same "
                    "letter back, so an id->letter offset cannot pass unnoticed"}


# ---------------------------------------------------------------- vision tokens

def vision_tokens(model, proc, cases):
    """cases: [{'arm','path','pils','prompt'}] -> per-path measured token counts."""
    out = {}
    for case in cases:
        inputs = _build_inputs(proc, case["pils"], case["prompt"])
        n, src = C.count_vision_tokens(model, proc, inputs)
        k = len(case["pils"])
        out[case["arm"]] = {
            "prompt_path": case["path"],
            "n_images": int(k),
            "vision_tokens_total": int(n),
            "vision_tokens_per_image": (float(n) / k if k else None),
            "n_non_vision_tokens": int(inputs["input_ids"].shape[1]) - int(n),
            "counted_from": src,
            "counted_with": "the processor's own input_ids for the exact prompt this arm "
                            "sends; no forward pass, so this is not a reading",
        }
    with_images = [v["vision_tokens_per_image"] for v in out.values()
                   if v["n_images"] > 0 and v["vision_tokens_per_image"] is not None]
    out["per_image_equality"] = {
        "image_bearing_paths": len(with_images),
        "values": sorted(set(with_images)),
        "all_equal": bool(len(set(with_images)) <= 1),
        "rule": "the three image-bearing prompt paths (16-frame answer, 1-frame answer, "
                "single-image scoring) must feed the same number of vision tokens per image. "
                "Runs of this design measured 1,149 (selection / single / distortion) and "
                "1,037 (donor); the difference between readings is recorded but was not traced"}
    out["valid_within"] = ("this run only: the equality above is inside one run, on this "
                           "backbone. It is not a fixed per-frame budget and it is not carried "
                           "across readings or backbones")
    return out


# ---------------------------------------------------------------- determinism

def determinism(model, proc, dev, ids, cases, counter):
    """cases: [{'name','kind','pils','prompt'}], kind in {'answer','score'}; each run twice."""
    rows = []
    for case in cases:
        name = case["name"]
        if case["kind"] == "answer":
            kind = "control:det_" + name
            raw1, l1 = G.answer_arm(model, proc, case["pils"], case["prompt"], dev,
                                    counter=counter, kind=kind)
            raw2, l2 = G.answer_arm(model, proc, case["pils"], case["prompt"], dev,
                                    counter=counter, kind=kind)
            rows.append({"path": name, "kind": "answer", "n_images": len(case["pils"]),
                         "letters": [l1, l2], "raw": [raw1, raw2],
                         "letter_agree": bool(l1 == l2), "raw_agree": bool(raw1 == raw2),
                         "logits_identical": None,
                         "agree": bool(l1 == l2 and raw1 == raw2)})
        else:
            kind = "control:det_" + name
            a = C.option_logits(model, proc, case["pils"], case["prompt"], dev, ids, batch=1)
            counter.count(kind, len(case["pils"]))
            b = C.option_logits(model, proc, case["pils"], case["prompt"], dev, ids, batch=1)
            counter.count(kind, len(case["pils"]))
            la = [LETTERS[int(i)] for i in np.argmax(a, axis=1)]
            lb = [LETTERS[int(i)] for i in np.argmax(b, axis=1)]
            ident = bool(np.array_equal(a, b))
            rows.append({"path": name, "kind": "score", "n_images": len(case["pils"]),
                         "letters": [la, lb], "raw": [None, None],
                         "letter_agree": bool(la == lb), "raw_agree": None,
                         "logits_identical": ident,
                         "agree": bool(la == lb and ident)})
    return {"paths": rows, "n_paths": len(rows),
            "n_agree": int(sum(1 for r in rows if r["agree"])),
            "all_agree": bool(rows and all(r["agree"] for r in rows)),
            "rule": "the same call, twice, on each of the three prompt paths. The 16-frame "
                    "answer path, the 1-frame answer path and the single-image scoring path "
                    "are all repeated inside the run; a repeat that only re-runs one path "
                    "cannot say anything about the other two."}


# ---------------------------------------------------------------- the option control

def new_state():
    return {"perm_units": [], "paths": None}


def add_perm_unit(state, qid, question, options, gold, letter, raw, pils, arm,
                  limit=N_PERM_UNITS):
    if len(state["perm_units"]) >= limit:
        return
    state["perm_units"].append({"qid": qid, "question": question,
                                "options": list(options), "gold": gold, "letter": letter,
                                "raw": raw, "pils": list(pils), "arm": arm})


def set_paths(state, frames16, frame1, prompt, arm16, arm1):
    state["paths"] = {"frames16": list(frames16), "frame1": frame1, "prompt": prompt,
                      "arm16": arm16, "arm1": arm1}


def prompt_for_named_text(question, options, letters=None, strict=False):
    """The naming ask.  Channel A (letters=None) prints the option TEXTS only; channel B
    (letters=LETTERS) prints the same texts in the same order with the letter labels the
    letter ask uses.  Neither channel is an arm prompt.

    History, written down because it decides the design: v1 ended on "Answer:" and v2 on free
    text, and in both the model replied with a bare letter every time (task-bbb62d92f8 and
    task-83b6aa08bc: 0 of 30 pairs evaluable each); v3 blocked the five upper-case ids and the
    model answered lower-case `d`/`c`/`b` or out-of-range `F`/`G`/`H`/`M` instead
    (task-f0d4a231c4: 26 of 42 pairs unevaluable).  So the answer shape is now fixed in the
    input: channel A's option list carries no letters to copy, the assistant turn is opened by
    NAMING_FORCED_PREFIX (appended by answer_arm as `prompt_suffix`), and the first generated
    token cannot be a bare letter (see letter_block_ids).
    """
    opts = list(options)
    if letters:
        body = "\n".join("%s. %s" % (L, c) for L, c in zip(letters, opts))
    else:
        body = "\n".join("- %s" % c for c in opts)
    if strict:
        instr = "Which option answers the question? " + NAMING_ESCALATION_INSTRUCTION
    else:
        instr = ("Which option answers the question? Copy that option's text exactly as it is "
                 "written above, without its letter.")
    return "%s\n%s\n%s" % (question, body, instr)


def letter_block_ids(proc):
    """Every token id that decodes to a bare single Latin letter, computed from the vocabulary.

    v3 hard-coded ids 32-36 (upper-case A-E) and the model answered with lower-case d/c/b or
    out-of-range F/G/H/M instead.  This walks the whole vocabulary, so no case and no letter is
    left out, and it is recomputed per run on this backbone's tokenizer rather than assumed.
    It removes only those ids and only on the naming ask, so no option is favoured.
    """
    tok = proc.tokenizer
    out = []
    for t, i in (tok.get_vocab() or {}).items():
        s = str(t)
        for m in _LEAD_MARKS:
            if s.startswith(m):
                s = s[len(m):]
        if not _LETTER_TOKEN.match(s):
            continue
        try:
            d = str(tok.decode([int(i)])).strip()
        except Exception:                                        # noqa: BLE001
            continue
        if _LETTER_TOKEN.match(d):
            out.append(int(i))
    return sorted(set(out))


def _unit_block_ids(block_ids, proc, options):
    """The block for one unit.  If an option's whole text IS a single letter, that letter's ids
    are not blocked for this unit (blocking them would forbid a correct answer); the conflict
    is recorded instead of silently ignored."""
    bad = set()
    for o in options:
        s = _norm(o)
        if len(s) == 1 and s.isalpha():
            bad.add(s.lower())
    if not bad:
        return list(block_ids), []
    keep, dropped = [], []
    for i in block_ids:
        try:
            d = _norm(proc.tokenizer.decode([int(i)])).strip("().:-")
        except Exception:                                        # noqa: BLE001
            keep.append(int(i))
            continue
        if len(d) == 1 and d in bad:
            dropped.append(int(i))
        else:
            keep.append(int(i))
    return keep, dropped


def _name_once(model, proc, dev, pils, question, options, letters, tag, counter,
               block_ids, strict=False, block_steps=(0,)):
    """One naming ask, its raw string and how it parsed.  No letter is ever looked up."""
    blk, dropped = _unit_block_ids(block_ids, proc, options)
    prompt = prompt_for_named_text(question, options, letters=letters, strict=strict)
    raw, _ = G.answer_arm(model, proc, pils, prompt, dev, counter=counter,
                          kind="control:named_" + tag,
                          max_new_tokens=NAMED_MAX_NEW_TOKENS,
                          first_token_block_ids=blk, block_steps=tuple(block_steps),
                          prompt_suffix=NAMING_FORCED_PREFIX)
    text, how, n_cand, shown = _parse_named_text(raw, options)
    return {"raw": raw, "shown": shown, "named_text": text, "parse": how,
            "n_matching_options": n_cand, "escalated": False,
            "block_steps": [int(s) for s in block_steps], "n_blocked_ids": len(blk),
            "blocking_conflicts_with_option": dropped}


def _name_with_rule(model, proc, dev, pils, question, options, letters, tag, counter,
                    block_ids):
    """One naming channel, plus THE STATED RULE for a bare-letter reply.

    A bare letter is classified `letter_reply` and is then handled by rule, not folded into
    `unparseable`:
      1. exactly ONE escalation retry, on the same unit and order, with the block applied to
         the first two generated positions and the instruction replaced by
         NAMING_ESCALATION_INSTRUCTION;
      2. if the retry yields option text, that text judges the pair and
         `resolved_by_escalation` is recorded;
      3. if the retry is a bare letter again (or empty/unmatchable), the pair is NOT evaluable
         and is named `letter_reply_after_escalation` -- a named sub-class of `n_unparseable`
         that makes the cell `incomplete`, which is not a pass.
    The letter is never turned into option text: that is exactly the withdrawn r2
    construction (`chosen_text = opts[LETTERS.index(letter)]`), which was self-consistent by
    construction.  A letter reply is evidence that the naming step failed, not a substitute
    reading of the mapping.
    """
    out = _name_once(model, proc, dev, pils, question, options, letters, tag, counter,
                     block_ids)
    if out["parse"] == "letter_reply":
        esc = _name_once(model, proc, dev, pils, question, options, letters,
                         tag + "_escal", counter, block_ids, strict=True,
                         block_steps=(0, 1))
        out["escalated"] = True
        out["raw_escalation"] = esc["raw"]
        out["block_steps_escalation"] = esc["block_steps"]
        if esc["named_text"] is not None:
            out["resolved_by_escalation"] = True
            out["named_text"] = esc["named_text"]
            out["parse"] = "resolved_by_escalation"
            out["n_matching_options"] = esc["n_matching_options"]
        else:
            out["resolved_by_escalation"] = False
            out["parse"] = ("letter_reply_after_escalation"
                            if esc["parse"] == "letter_reply" else esc["parse"])
    return out


# ---- the REGISTERED pass line (Plan plan-single-frame-random-question-r2 §5) -----
# The criterion itself is unchanged.  Its pass line was zero contradictions, and that line was
# withdrawn as an after-the-fact threshold: the naming step disagrees with itself on 1 of 8
# pairs when only the option-list labelling changes, so on 48 pairs a zero-tolerance rule
# passes with probability 0.0016 at that rate and can only ever withhold.  The line below is
# derived from that same measured rate, on the same pairs, in the same run.  It is registered
# before the 48-pair data exist; the only quantity it reads from the data is the noise floor
# that zero tolerance ignored.  alpha is a stated convention, not a tuned value.
MAPPING_ALPHA = 0.05          # one-sided
MAPPING_UNSTABLE_P_HI = 0.35  # a naming step that flips on >=35% of pairs cannot gate a map


def _cp_upper(k, n, alpha=MAPPING_ALPHA):
    """One-sided Clopper-Pearson upper bound for k successes in n trials (exact)."""
    from scipy.stats import beta
    n = int(n)
    if n <= 0:
        return 1.0
    k = max(0, min(int(k), n))
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def _binom_tolerance(n, p, alpha=MAPPING_ALPHA):
    """Largest b with P(X <= b) >= 1 - alpha, X ~ Binomial(n, p)."""
    from scipy.stats import binom
    n = int(n)
    if n <= 0:
        return 0
    p = min(max(float(p), 0.0), 1.0)
    for b in range(n + 1):
        if binom.cdf(b, n, p) >= 1.0 - alpha:
            return int(b)
    return n


def _smallest_detectable_rate(n, b, alpha=MAPPING_ALPHA, step=0.001):
    """The smallest true contradiction rate this rule rejects with probability >= 0.95.

    Reported with the verdict so nobody can read a pass as 'the map is correct': at 48 pairs
    the rule only sees gross mis-wiring, and this number says how gross.
    """
    from scipy.stats import binom
    n = int(n)
    if n <= 0:
        return None
    b = int(b)
    p = 0.0
    while p <= 1.0 + 1e-12:
        if 1.0 - binom.cdf(b, n, p) >= 0.95:
            return float(p)
        p += step
    return 1.0


def mapping_pass_line(n_pairs, n_evaluable, n_contradictions, retest_both, retest_disagree,
                      alpha=MAPPING_ALPHA):
    """THE REGISTERED PASS LINE for `option_control.mapping` (Plan r2 §5, control C).

    pass  <=>  every registered pair evaluable AND C <= B, where
               p_hi = one-sided 95% Clopper-Pearson upper bound on the naming test-retest
                      disagreement rate D/n_both measured on the SAME pairs,
               B    = the 95th percentile of Binomial(n_eval, p_hi), i.e. the largest
                      contradiction count that this cell's own naming noise could produce.
    fail  <=>  C > B            (more contradictions than naming noise explains at 5%)
    incomplete <=> some registered pair is unevaluable (that is not a pass)
    not_discriminating <=> n_both == 0 (no noise floor to derive a line from)

    B_strict is the same quantity at p_hi(0 of n_pairs): a pass with C <= B_strict does not
    depend on this run's retest measurement at all.
    """
    n_pairs = int(n_pairs)
    n_ev = int(n_evaluable)
    n_c = int(n_contradictions)
    nb = int(retest_both)
    nd = int(retest_disagree)
    out = {"alpha_one_sided": float(alpha),
           "n_pairs": n_pairs, "n_evaluable": n_ev, "n_contradictions": n_c,
           "n_retest_both": nb, "n_retest_disagree": nd,
           "p_retest": (float(nd) / nb if nb else None),
           "contradiction_rate": (float(n_c) / n_ev if n_ev else None)}
    p_hi = _cp_upper(nd, nb, alpha) if nb else None
    out["p_retest_upper95"] = p_hi
    if n_pairs > 0:
        out["tolerance_b_strict"] = _binom_tolerance(n_pairs, _cp_upper(0, n_pairs, alpha),
                                                    alpha)
    else:
        out["tolerance_b_strict"] = None
    out["tolerance_b"] = (None if p_hi is None
                          else _binom_tolerance(n_ev or n_pairs, p_hi, alpha))
    out["smallest_detectable_rate"] = (
        None if out["tolerance_b"] is None
        else _smallest_detectable_rate(n_ev or n_pairs, out["tolerance_b"], alpha))
    # ---- Plan r3 §10: every unevaluable pair is charged as a worst-case contradiction ------
    # Coverage is not a separate convention.  A pair that could not be evaluated is not
    # evidence of agreement, and dropping it would shrink the denominator and the line with
    # it; so it is counted as a contradiction, and the criterion passes only when the observed
    # contradictions and the unevaluable pairs together stay under the line:
    #     pass  <=>  C + e <= B(n_pairs - e, p_hi)
    # The coverage floor is the same inequality at C = 0: e <= B(n_pairs - e, p_hi).  e_max is
    # its largest integer solution, so a cell that keeps fewer than n_pairs - e_max evaluable
    # pairs cannot pass whatever its contradiction count is.  Both numbers come out of the
    # registered pair count, the registered alpha and the measured noise floor; neither is a
    # fraction chosen by convention.
    n_unev = int(n_pairs) - int(n_ev) if n_pairs else 0
    out["n_unevaluable"] = n_unev
    out["contradictions_plus_unevaluable"] = int(n_c + n_unev)
    if n_pairs > 0 and p_hi is not None:
        e_max = 0
        for e in range(0, n_pairs + 1):
            if e <= _binom_tolerance(n_pairs - e, p_hi, alpha):
                e_max = int(e)
        n_min = int(n_pairs - e_max)
        # The quantity the prose used to CLAIM for `margin_pairs` (Plan -r4 §13b): how many
        # further unevaluable pairs this run's contradiction count can still absorb.  It is
        # NOT B - C, because B is taken at n_pairs - e and shrinks as e grows; the walk below
        # uses the same inequality the verdict uses, so the number is the one the words said.
        max_absorb = None
        for e2 in range(0, int(n_pairs) + 1):
            if e2 > int(e_max):
                break
            if int(n_c) + e2 <= _binom_tolerance(n_pairs - e2, p_hi, alpha):
                max_absorb = int(e2)
            else:
                break
        out["coverage"] = {
            "n_pairs": int(n_pairs), "n_evaluable": int(n_ev), "n_unevaluable": int(n_unev),
            "e_max_allowed": int(e_max),
            "required_pairs_for_a_pass": n_min,
            "required_coverage_frac": float(n_min) / float(n_pairs),
            "observed_coverage_frac": float(n_ev) / float(n_pairs),
            "slack_b_minus_c": (None if out.get("tolerance_b") is None
                                else int(out["tolerance_b"]) - int(n_c)),
            "max_absorbable_unevaluable_pairs": (None if (out.get("tolerance_b") is None
                                                          or max_absorb is None)
                                                 else int(max_absorb)),
            "sufficient": bool(int(n_unev) <= int(e_max)),
            "rule": "e_max = the largest e with e <= B(n_pairs - e, p_hi) (C = 0, the most "
                    "favourable case); a cell with more than e_max unevaluable pairs cannot "
                    "pass at any contradiction count.  required_coverage_frac = "
                    "(n_pairs - e_max)/n_pairs.  slack_b_minus_c = B - C is the distance from "
                    "the line at the observed n_eval; it is NOT how many further unevaluable "
                    "pairs this run could absorb, because B shrinks with each one.  That "
                    "quantity is max_absorbable_unevaluable_pairs = the largest e' with "
                    "C + e' <= B(n_pairs - e', p_hi); quote that one for 'how many more the "
                    "run could take'.  (This replaces the field `margin_pairs`, whose "
                    "definition was B - C while its prose claimed the absorbable count.)",
        }
        if out["coverage"]["max_absorbable_unevaluable_pairs"] is None:
            out["coverage"]["max_absorbable_note"] = (
                "None with a non-null tolerance_b means no e' satisfies "
                "C + e' <= B(n_pairs - e'): the run is already over the line at e = 0")
    else:
        out["coverage"] = {"n_pairs": int(n_pairs), "n_evaluable": int(n_ev),
                           "n_unevaluable": int(n_unev), "e_max_allowed": None,
                           "required_pairs_for_a_pass": None,
                           "required_coverage_frac": None,
                           "observed_coverage_frac": (float(n_ev) / float(n_pairs)
                                                      if n_pairs else None),
                           "slack_b_minus_c": None,
                           "max_absorbable_unevaluable_pairs": None,
                           "sufficient": None,
                           "rule": "no noise floor, so no coverage floor can be derived"}

    demoted = bool(p_hi is not None and p_hi >= MAPPING_UNSTABLE_P_HI)
    if n_pairs <= 0:
        status, reason = "not_available", "no pair was available"
    elif p_hi is None:
        status, reason = ("not_discriminating",
                          "no noise floor: no pair had both naming channels parsed, so no "
                          "pass line can be derived at this cell size")
    elif demoted:
        status, reason = ("not_discriminating",
                          "naming step unstable: p_hi = %.4f >= %.2f (the registered demotion): "
                          "reported, no pass" % (p_hi, MAPPING_UNSTABLE_P_HI))
    elif n_c > int(out["tolerance_b"]):
        status, reason = "fail", ("%d contradictions exceed the line %d"
                                  % (int(n_c), int(out["tolerance_b"])))
    elif int(out["coverage"]["e_max_allowed"]) is not None and \
            n_unev > int(out["coverage"]["e_max_allowed"]):
        status, reason = "incomplete", ("coverage %d/%d is below the derived floor %d/%d "
                                        "(e = %d > e_max = %d)"
                                        % (int(n_ev), int(n_pairs),
                                           int(out["coverage"]["required_pairs_for_a_pass"]),
                                           int(n_pairs), int(n_unev),
                                           int(out["coverage"]["e_max_allowed"])))
    elif n_c + n_unev > int(out["tolerance_b"]):
        status, reason = "incomplete", ("worst case: %d contradictions + %d unevaluable = %d "
                                        "exceed the line %d"
                                        % (int(n_c), int(n_unev), int(n_c + n_unev),
                                           int(out["tolerance_b"])))
    else:
        status, reason = "pass", None
    out["status"] = status
    out["reason"] = reason
    out["mapping_ok"] = bool(status == "pass")
    out["demoted_naming_unstable"] = demoted
    # The r2 line, computed on the same numbers with no extra input, so a reader can see that
    # the two lines differ by the unevaluable pairs and by nothing else.
    out["r2_rule_status"] = ("not_available" if n_pairs <= 0
                             else "not_discriminating" if p_hi is None
                             else "fail" if n_c > int(out["tolerance_b"])
                             else "incomplete" if n_ev < n_pairs
                             else "pass")
    out["naming_step_unstable"] = bool(p_hi is not None and p_hi >= MAPPING_UNSTABLE_P_HI)
    out["passes_only_because_of_noise"] = bool(
        status == "pass" and out["tolerance_b_strict"] is not None
        and n_c > int(out["tolerance_b_strict"]))
    out["can_prove"] = ("a gross mis-wiring only: a fault that contradicts on a large share "
                        "of the 48 pairs is rejected (the smallest detectable rate is "
                        "reported)")
    out["cannot_prove"] = ("that the letter<->option map is correct; a fault on a minority "
                           "of pairs is invisible at 48 pairs")
    out["rule"] = ("registered in Plan plan-single-frame-random-question-r3 §9-§10 "
                   "(control C; it supersedes the r2 §5 line, and the r2 verdict on the same "
                   "numbers is reported as r2_rule_status). A pair is judged by the same "
                   "comparison as before: the option the letter points at must equal the option "
                   "the model names in free text on that same order, never derived from the "
                   "letter. What changed is the treatment of a pair that could not be judged: "
                   "it is charged as a worst-case contradiction. pass iff C + e <= B and "
                   "e <= e_max; fail iff C > B (the contradictions alone exceed the line); "
                   "incomplete iff C <= B < C + e, or e > e_max (too little coverage: no "
                   "verdict can be given and the cell does not report); not_discriminating iff "
                   "no pair was parsed on both channels, or p_hi >= 0.35 (the naming step is too "
                   "unstable to gate a map). B = quantile_0.95(Binomial(n_eval, p_hi)), p_hi = "
                   "one-sided 95% Clopper-Pearson upper bound on the naming test-retest rate on "
                   "the same pairs, e_max = the largest e with e <= B(n_pairs - e, p_hi). "
                   "B_strict is the same line at retest 0/n_pairs. A pass says no gross "
                   "mis-wiring, never that the map is correct.")
    out["why_the_line_changed"] = ("two after-the-fact corrections, both written down before the "
                                   "run that uses them. (1) r2's zero tolerance was withdrawn "
                                   "because the naming step disagrees with itself on 1 of 8 "
                                   "pairs on the same question, so a 48-pair cell passed at "
                                   "0.0016 and the criterion could only ever withhold. (2) r2's "
                                   "coverage = 1 rule was withdrawn for the same reason one "
                                   "level up: one unit that refuses to name an option made the "
                                   "whole cell incomplete, and under the same 95% convention a "
                                   "coverage = 1 rule withholds about 99% of runs of this size "
                                   "whatever the map does. Unevaluable pairs are now charged as "
                                   "contradictions, which is never looser than the r2 line: at "
                                   "the same n_eval it demands C <= B - e, i.e. strictly fewer "
                                   "contradictions.")
    out["refusal_rule"] = ("a reply that normalises to one of REFUSAL_PHRASES is a refusal, not "
                           "an unparseable answer: if the unit's own option list contains an "
                           "option with that normalised text it names that option "
                           "(refusal_matched_option) and the pair is judged; otherwise the pair "
                           "stays unevaluable (refusal_no_option) and is charged as a "
                           "worst-case contradiction. A refusal is never mapped onto a "
                           "neighbouring option by meaning.")
    return out


def verdict_sensitivity(n_pairs, n_evaluable, n_contradictions, retest_both, retest_disagree,
                        alpha=MAPPING_ALPHA):
    """The SAME pairs read under two other defensible conventions.  Zero forwards.

    Recorded WITH the verdict because the pilot-gate review established that this line's pass
    is a knife-edge: either of these choices flips it.  That is a property of the instrument
    (Plan -r4 §14b), not a footnote to one run, and neither line decides anything -- the
    registered verdict is the one in `status` of `mapping_pass_line`.

    (1) point_estimate_noise: B taken at the observed retest rate instead of its one-sided
        95% upper bound.  Two stacked 95%s (the CP bound and the binomial quantile) widen the
        line; a reader who used the point estimate would get a different verdict.
    (2) worst_case_noise_denominator: the criterion charges an unevaluable pair as a
        worst-case contradiction, but the noise floor drops the pairs whose retest channel
        named nothing.  Applying the SAME worst-case principle to the noise denominator
        charges them as disagreements.
    The third fragility (a different option permutation) needs new letter asks, so it cannot
    be recomputed from these pairs; it is recorded as a labelled quote, not as a measurement.
    """
    n_pairs, n_ev = int(n_pairs), int(n_evaluable)
    n_c = int(n_contradictions)
    nb, nd = int(retest_both), int(retest_disagree)
    n_unev = n_pairs - n_ev
    out = {"rule": "reported only, no forward pass; the registered verdict is `status`. Both "
                   "lines are computed from this run's own pairs."}
    if nb > 0:
        p_hat = float(nd) / float(nb)
        b_pt = int(_binom_tolerance(n_ev or n_pairs, p_hat, alpha))
        out["point_estimate_noise"] = {
            "p_retest_point": p_hat, "tolerance_b": b_pt,
            "contradictions_plus_unevaluable": int(n_c + n_unev),
            "status": ("fail" if n_c > b_pt
                       else "incomplete" if n_c + n_unev > b_pt else "pass"),
            "what": "B at the observed retest rate (p_hat) instead of its 95% upper bound"}
    else:
        out["point_estimate_noise"] = {"available": False,
                                       "why": "no pair parsed on both naming channels"}
    nb_w = n_ev
    nd_w = nd + max(0, n_ev - nb)
    p_hi_w = _cp_upper(nd_w, nb_w, alpha) if nb_w else None
    if p_hi_w is None or p_hi_w >= MAPPING_UNSTABLE_P_HI:
        st_w = "not_discriminating"
    else:
        b_w = int(_binom_tolerance(n_ev or n_pairs, p_hi_w, alpha))
        st_w = ("fail" if n_c > b_w
                else "incomplete" if n_c + n_unev > b_w else "pass")
    out["worst_case_noise_denominator"] = {
        "n_both_charged": int(nb_w), "n_disagree_charged": int(nd_w),
        "p_retest_upper95": p_hi_w, "status": st_w,
        "what": "pairs whose retest channel named no option are charged as disagreements -- "
                "the same worst-case principle the criterion applies to unevaluable pairs"}
    out["permutation_seed"] = {
        "measured_here": False,
        "quoted_from": "evidence.task-6e86f0d18c (rerun_of task-c0dda04c23, seed 1, "
                       "same commit, same 24 units, same draw; only the option permutation "
                       "differs): n_retest_disagree/n_retest_both = 8/29, "
                       "p_retest_upper95 = 0.4429, verdict not_discriminating",
        "why": "a different permutation needs new letter asks (forwards), so it cannot be "
               "recomputed from this run's pairs; quoted, not measured here"}
    return out


def _criterion(pairs, shift_fn=None):
    """THE criterion, in one place, used for the real verdict AND for the fault injection.

    For each pair: the option that the letter points at must equal the option the model names
    in free text.  `shift_fn(k)` is a deliberate offset of the letter->option index map (0 is
    the real map).  Every pair is judged by the same expression, so an injection cannot take a
    different code path than the verdict it is meant to test.
    """
    rows = []
    for k, p in enumerate(pairs):
        li = p.get("letter_index")
        opts = p.get("options") or []
        named = p.get("named_text")
        sh = 0 if shift_fn is None else int(shift_fn(k))
        idx = None if (li is None or not opts) else (int(li) + sh) % len(opts)
        target = opts[idx] if idx is not None else None
        evaluable = bool(named is not None and li is not None)
        rows.append({"k": k, "shift": sh, "target": target, "named_text": named,
                     "evaluable": evaluable,
                     "contradiction": bool(evaluable and target != named)})
    n_ev = int(sum(1 for r in rows if r["evaluable"]))
    n_c = int(sum(1 for r in rows if r["contradiction"]))
    n_diff = int(sum(1 for r in rows
                     if r["evaluable"] and r["target"] != r["named_text"]))
    return {"rows": rows, "n_pairs": len(rows), "n_evaluable": n_ev,
            "n_contradictions": n_c, "n_corrupted_target_differs": n_diff}


def fault_injection(pairs, reported_contradictions, seed, retest_both, retest_disagree):
    """A deliberately corrupted letter->option map must make the criterion fire.

    Every injected verdict goes through the SAME registered pass line
    (`mapping_pass_line`) that decides the real one, with this run's retest counts, so an
    injection cannot be judged by a rule the verdict is not judged by.

    Four injections, all through `_criterion`, all zero forwards:
      identity      shift 0             -> must reproduce the reported contradiction count
      shift_plus1   index -> index+1    -> the verdict must be fail, and every pair whose
                                           corrupted target differs from the named text must
                                           be caught (no silent pass)
      shift_minus1  index -> index-1    -> same
      partial_half  +1 on one parity    -> contradictions exactly on the corrupted pairs whose
                                           target differs, verdict fail
    What this proves: the comparison is sensitive to the letter->option map.  A criterion that
    derived the text from the letter (the withdrawn r2 construction) would be invariant to all
    four injections.  What it does NOT prove: that the real wiring is correct -- that is what
    letter_decode, the permutation arithmetic and answer_visibility are for.
    """
    par = int(np.random.default_rng(int(seed) + 7).integers(0, 2))
    modes = [("identity", lambda k: 0),
             ("shift_plus1", lambda k: 1),
             ("shift_minus1", lambda k: -1),
             ("partial_half", lambda k: 1 if (k % 2) == par else 0)]
    out = {}
    for name, fn in modes:
        c = _criterion(pairs, fn)
        zero_ok = bool(c["n_pairs"] > 0 and c["n_evaluable"] == c["n_pairs"]
                       and c["n_contradictions"] == 0)
        out[name] = {
            "n_pairs": c["n_pairs"], "n_evaluable": c["n_evaluable"],
            "n_contradictions": c["n_contradictions"],
            "n_pairs_where_corrupted_target_differs": c["n_corrupted_target_differs"],
            "status": mapping_pass_line(c["n_pairs"], c["n_evaluable"],
                                        c["n_contradictions"], retest_both,
                                        retest_disagree)["status"],
            # the r6 registered line (zero contradictions, every pair evaluable), so the
            # injection is judged by the same line that governs `mapping.mapping_ok`
            "status_zero_line": ("pass" if zero_ok else "fail"),
            "every_differing_pair_caught": bool(
                c["n_contradictions"] == c["n_corrupted_target_differs"]),
        }
    out["partial_half_parity"] = par
    ident = out["identity"]
    sh = out["shift_plus1"]
    out["identity_reproduces_reported"] = bool(
        ident["n_contradictions"] == int(reported_contradictions))
    out["verdict_changes"] = bool(sh["status"] != ident["status"])
    # ---- r6 correction of the `fired` boolean (Plan r6 controls.negative (vi); review of
    # task-78e2c44d4a §7).  The v4 conjunction demanded, besides the checks below, that the
    # offset RAISE the contradiction count (or that identity pass).  On the llavaov donor cell
    # shift_plus1 caught all 8 misaligned pairs and returned status=fail, yet fired=false --
    # an over-strict boolean, not a criterion that failed to fire.  The conjunction is removed;
    # every other term is unchanged.  The removed quantity is still REPORTED
    # (`n_contradictions_increased`), it just no longer gates `fired`.
    out["n_contradictions_increased"] = bool(
        sh["n_contradictions"] > ident["n_contradictions"])
    out["fired"] = bool(
        ident["n_evaluable"] > 0
        and out["identity_reproduces_reported"]
        and sh["status_zero_line"] == "fail"
        and sh["every_differing_pair_caught"]
        and sh["n_pairs_where_corrupted_target_differs"] >= 1)
    out["rule"] = ("the same comparison function that produces `mapping` is re-run with the "
                   "letter->option index map deliberately offset (identity, +1, -1, and +1 on "
                   "one parity). A criterion that derived the option text from the letter "
                   "would be invariant to all four. `fired` is true when the identity run "
                   "reproduces the reported contradiction count AND shift_plus1 is judged "
                   "fail AND every pair whose corrupted target differs from the named text is "
                   "caught (with at least one such pair). The v4 extra term -- the offset must "
                   "also RAISE the contradiction count -- was removed by the r6 correction "
                   "(Plan plan-four-readings-second-backbone-r6): a cell already "
                   "contradicting on every evaluable pair cannot raise the count further, and "
                   "the donor cell of task-78e2c44d4a was exactly that case. This tests the "
                   "comparison, not the real wiring.")
    return out


def _norm(s):
    return " ".join(str(s or "").split()).strip().lower().strip(" \t\"'`*_.,;:!?()[]{}")


def _strip_lead(s):
    """Drop a leading announcement or option marker the model may have put in front of the
    text it copied: "B.", "(B)", "B:", "The correct option is", "Answer:", "It is".

    This only removes boilerplate.  It never supplies a letter and never looks one up: the text
    that remains is what the model wrote, and the criterion still compares our letter against
    it.  It matters because several options are short ("48%"), where any prefix at all would
    otherwise make the answer unmatchable.
    """
    t = str(s or "").strip()
    t = re.sub(r"^[-*\u2022]+\s+", "", t)          # list bullet copied from channel A's list
    t = re.sub(r"^the\s+option\s+text\s+is\s*[:.\-]?\s*", "", t, flags=re.I)
    t = re.sub(r"^\(?[A-Ea-e]\)?\s*[:.\-)]\s*", "", t)
    t = re.sub(r"^(?:the\s+)?(?:correct\s+)?(?:answer|option)(?:\s+is)?\s*[:.\-]?\s*", "", t,
               flags=re.I)
    t = re.sub(r"^it\s+is\s+", "", t, flags=re.I)
    return t.strip()


def _parse_named_text(raw, options):
    """Which option does the model's free text name?

    Returns (text or None, how, n_candidates, shown).  It never looks at a letter to build the
    text: the whole point of the mapping criterion is that the option text must come from what
    the model said, not from the letter it also gave.  Matching is on normalised strings; a
    candidate must either be the whole answer or overlap the answer by at least 12 characters,
    so a refusal or a paraphrase is reported rather than silently mapped onto an option.

    `how == "letter_reply"` is its own class, not `unparseable`: a bare letter is a specific,
    named failure of the naming step and `_name_with_rule` escalates it once before a pair is
    given up on.
    """
    t = _strip_lead(raw).strip()
    while len(t) >= 2 and t[0] in "\"'`\u201c\u201d" and t[-1] in "\"'`\u201c\u201d":
        t = t[1:-1].strip()
    t = t.strip("\"'`\u201c\u201d").strip()
    if _LETTER_ONLY.match(t):
        return None, "letter_reply", 0, t
    r = _norm(t)
    if not r:
        return None, "empty", 0, t
    # Plan r3 §9: a refusal is a naming when the unit's own option list carries that phrase.
    if r in {_norm(p) for p in REFUSAL_PHRASES}:
        for j, o in enumerate(options):
            if _norm(o) == r:
                return list(options)[j], "refusal_matched_option", 1, t
        return None, "refusal_no_option", 0, t
    cands = []
    for j, o in enumerate(options):
        oo = _norm(o)
        if not oo:
            continue
        if oo == r:
            cands.append((j, len(oo), "exact"))
        elif len(oo) >= 12 and oo in r:
            cands.append((j, len(oo), "option_in_raw"))
        elif len(r) >= 12 and r in oo:
            cands.append((j, len(oo), "raw_in_option"))
    if not cands:
        return None, "unparseable", 0, t
    # An EXACT match outranks a containment match (Plan -r4 §13a; pilot-gate review §1.1).
    # The old rule took the LONGEST candidate, so a short reply that was literally one option
    # lost to a longer option that merely contained it, and the pair was charged as a
    # contradiction although the model had named the option exactly (instance:
    # 2sriHX3PbXw#511, permuted order).  Longest-first survives only as the fallback when no
    # candidate is exact; a length tie is still reported as `ambiguous`, never guessed.
    exact = [c for c in cands if c[2] == "exact"]
    pool = exact if exact else cands
    best_len = max(c[1] for c in pool)
    best = [c for c in pool if c[1] == best_len]
    if len({c[0] for c in best}) != 1:
        return None, "ambiguous", len(cands), t
    return list(options)[best[0][0]], best[0][2], len(cands), t


def _parse_named_text_old_rule(raw, options):
    """The SUPERSEDED parse (before Plan -r4 §13a): the LONGEST candidate wins, even when a
    shorter candidate is an exact match of what the model wrote.

    It exists for one purpose only: to print the old-rule contradiction count beside the
    corrected one, as the pilot-gate review requires (review §1.2 condition 3, §6 item 4).
    Nothing in the criterion, the pass line, the guard or any verdict reads this function;
    it is a counterfactual re-read of the SAME recorded strings.
    """
    t = _strip_lead(raw).strip()
    while len(t) >= 2 and t[0] in "\"'`\u201c\u201d" and t[-1] in "\"'`\u201c\u201d":
        t = t[1:-1].strip()
    t = t.strip("\"'`\u201c\u201d").strip()
    if _LETTER_ONLY.match(t):
        return None, "letter_reply"
    r = _norm(t)
    if not r:
        return None, "empty"
    if r in {_norm(p) for p in REFUSAL_PHRASES}:
        for j, o in enumerate(options):
            if _norm(o) == r:
                return list(options)[j], "refusal_matched_option"
        return None, "refusal_no_option"
    cands = []
    for j, o in enumerate(options):
        oo = _norm(o)
        if not oo:
            continue
        if oo == r:
            cands.append((j, len(oo), "exact"))
        elif len(oo) >= 12 and oo in r:
            cands.append((j, len(oo), "option_in_raw"))
        elif len(r) >= 12 and r in oo:
            cands.append((j, len(oo), "raw_in_option"))
    if not cands:
        return None, "unparseable"
    best_len = max(c[1] for c in cands)
    best = [c for c in cands if c[1] == best_len]
    if len({c[0] for c in best}) != 1:
        return None, "ambiguous"
    return list(options)[best[0][0]], best[0][2]


def option_control(model, proc, dev, units, counter, seed, letter_ids=None):
    """The mapping criterion and the order-sensitivity quantity (r3 §3.1 items 3 and 5).

    Old rule (r2, withdrawn as an after-the-fact change): one ask per unit, both orders, and
    `n_text_preserved == n` to pass.  New rule: for each unit and each option order, (a) the
    letter that order's own letter-ask produced must point at the option the model NAMES IN
    FREE TEXT on that same order, and (b) the permutation arithmetic is a plain check.  The
    order-sensitivity numbers are reported and are not a criterion.
    """
    rows = []
    block_ids = letter_block_ids(proc)     # the vocabulary-level block, computed once per run
    for i, u in enumerate(units):
        opts = list(u["options"])
        n = len(opts)
        perm = _perm_for(seed, i, n)
        opts_perm = [opts[p] for p in perm]
        gold = u.get("gold")
        gold_perm = (LETTERS[perm.index(LETTERS.index(gold))] if gold in LETTERS else None)
        prompt_perm = G.prompt_for({"question": u["question"], "candidates": opts_perm})
        raw_p, letter_p = G.answer_arm(model, proc, u["pils"], prompt_perm, dev,
                                       counter=counter, kind="control:perm_" + u["arm"])
        # Two naming channels per order (Plan r4 §3.1 (3a), (3e)).  Channel A is the primary
        # read and feeds the criterion: option TEXTS only, so there is no letter to copy.
        # Channel B is the test-retest channel: the same texts in the same order, with the
        # letter labels the letter ask uses.  A bare-letter reply is handled by the stated rule
        # inside `_name_with_rule`; it is never turned into option text.
        nam = {
            "original": {
                "A": _name_with_rule(model, proc, dev, u["pils"], u["question"], opts, None,
                                     "A_" + u["arm"], counter, block_ids),
                "B": _name_with_rule(model, proc, dev, u["pils"], u["question"], opts, LETTERS,
                                     "B_" + u["arm"], counter, block_ids),
            },
            "permuted": {
                "A": _name_with_rule(model, proc, dev, u["pils"], u["question"], opts_perm,
                                     None, "Aperm_" + u["arm"], counter, block_ids),
                "B": _name_with_rule(model, proc, dev, u["pils"], u["question"], opts_perm,
                                     LETTERS, "Bperm_" + u["arm"], counter, block_ids),
            },
        }
        # A bit-identical repeat of channel A on the first units.  Under greedy decoding this
        # is structurally zero, so it is NOT the noise floor -- it only shows the naming decode
        # path is deterministic.  The noise floor is the A-vs-B disagreement below.
        repeat = None
        if i < N_IDENTICAL_REPEAT_UNITS:
            rep = _name_with_rule(model, proc, dev, u["pils"], u["question"], opts, None,
                                  "Arep_" + u["arm"], counter, block_ids)
            repeat = {"raw": rep["raw"], "named_text": rep["named_text"],
                      "parse": rep["parse"],
                      "identical": bool(rep["named_text"] == nam["original"]["A"]["named_text"]
                                        and rep["raw"] == nam["original"]["A"]["raw"])}
        named = nam["original"]["A"]["named_text"]
        how = nam["original"]["A"]["parse"]
        n_cand = nam["original"]["A"]["n_matching_options"]
        named_p = nam["permuted"]["A"]["named_text"]
        how_p = nam["permuted"]["A"]["parse"]
        n_cand_p = nam["permuted"]["A"]["n_matching_options"]
        li = LETTERS.index(u["letter"]) if u["letter"] in LETTERS else None
        lp = LETTERS.index(letter_p) if letter_p in LETTERS else None
        text_orig = opts[li] if li is not None else None
        text_perm = opts_perm[lp] if lp is not None else None
        pairs = [
            {"order": "original", "letter": u["letter"], "letter_index": li,
             "n_options": n, "options": opts,
             "letter_points_at": text_orig, "named_text": named, "named_parse": how,
             "n_matching_options": n_cand, "raw_named": nam["original"]["A"]["raw"],
             "naming": nam["original"]["A"], "retest": nam["original"]["B"],
             "identical_repeat": repeat,
             "evaluable": bool(li is not None and named is not None),
             "agrees": bool(text_orig is not None and named is not None
                            and text_orig == named)},
            {"order": "permuted", "letter": letter_p, "letter_index": lp,
             "n_options": n, "options": opts_perm,
             "letter_points_at": text_perm, "named_text": named_p, "named_parse": how_p,
             "n_matching_options": n_cand_p, "raw_named": nam["permuted"]["A"]["raw"],
             "naming": nam["permuted"]["A"], "retest": nam["permuted"]["B"],
             "identical_repeat": None,
             "evaluable": bool(lp is not None and named_p is not None),
             "agrees": bool(text_perm is not None and named_p is not None
                            and text_perm == named_p)},
        ]
        text_preserved = bool(text_orig is not None and text_perm is not None
                              and text_orig == text_perm)
        letter_moved = bool(u["letter"] is not None and letter_p is not None
                            and u["letter"] != letter_p)
        rows.append({
            "qid": u["qid"], "arm": u["arm"], "n_options": n,
            "permutation": perm,
            "perm_is_a_permutation": bool(sorted(perm) == list(range(n))),
            "options": opts, "options_permuted": opts_perm,
            "options_permuted_is_arithmetic": bool(
                opts_perm == [opts[p] for p in perm]),
            "gold": gold, "gold_permuted": gold_perm,
            "gold_permuted_is_arithmetic": bool(
                gold_perm == LETTERS[perm.index(LETTERS.index(gold))]) if gold in LETTERS
                else None,
            "letter": u["letter"], "letter_permuted": letter_p,
            "raw": u["raw"], "raw_permuted": raw_p,
            "chosen_text": text_orig, "chosen_text_permuted": text_perm,
            "text_preserved": text_preserved, "letter_moved": letter_moved,
            "perm_fixes_chosen_index": bool(li is not None and perm[li] == li),
            "correct_orig": bool(u["letter"] == gold),
            "correct_permuted": bool(letter_p == gold_perm) if gold_perm else False,
            "naming": nam,
            "identical_repeat": repeat,
            "pairs": pairs,
        })
    n = len(rows)
    if n == 0:
        return {"available": False, "n": 0, "mapping": {"status": "not_available",
                                                         "mapping_ok": False},
                "note": "no measured unit was available for the option control"}

    pairs_flat = [p for r in rows for p in r["pairs"]]
    crit = _criterion(pairs_flat)                  # THE criterion: one code path, also injected
    n_pairs = crit["n_pairs"]
    n_evaluable = crit["n_evaluable"]
    n_contradictions = crit["n_contradictions"]
    n_unparseable = int(sum(1 for p in pairs_flat if p["named_text"] is None))
    n_letter_missing = int(sum(1 for p in pairs_flat if p["letter"] not in LETTERS))
    # Named sub-classes of `n_unparseable` (Plan r4 §3.1 (3c)): a bare-letter reply that the
    # escalation rescued is still evaluable; one that was not is a named failure, not a silent
    # "unparseable".
    n_letter_reply = int(sum(1 for p in pairs_flat
                             if str(p.get("named_parse") or "").startswith("letter_reply")))
    n_resolved = int(sum(1 for p in pairs_flat
                         if p.get("named_parse") == "resolved_by_escalation"))
    # ---- the refusal rule's accounting (Plan r3 §9) ----------------------------------------
    # Reported separately from `n_unparseable` so a reader can tell a refusal from a parse
    # failure, see whether the option list carried the refusal, and see every pair that is
    # still unevaluable and every pair the rule recovered (with its verdict).
    ref_set = {_norm(x) for x in REFUSAL_PHRASES}
    n_ref_matched = int(sum(1 for p in pairs_flat
                            if p.get("named_parse") == "refusal_matched_option"))
    n_ref_no_option = int(sum(1 for p in pairs_flat
                              if p.get("named_parse") == "refusal_no_option"))
    n_units_with_refusal_option = int(sum(
        1 for r in rows
        if any(_norm(o) in ref_set for o in (r.get("options") or []))))
    recovered_pairs = [{"qid": r["qid"], "order": p["order"], "letter": p["letter"],
                        "letter_points_at": p["letter_points_at"],
                        "named_text": p["named_text"], "raw_named": p["raw_named"],
                        "agrees": bool(p.get("agrees")),
                        "contradiction": bool(not p.get("agrees"))}
                       for r in rows for p in r["pairs"]
                       if p.get("named_parse") == "refusal_matched_option"]
    unevaluable_pairs = [{"qid": r["qid"], "order": p["order"],
                          "parse": p.get("named_parse"),
                          "letter": p["letter"], "raw_naming": p["raw_named"],
                          "retest_parse": (p.get("retest") or {}).get("parse"),
                          "raw_retest": (p.get("retest") or {}).get("raw")}
                         for r in rows for p in r["pairs"]
                         if not p.get("evaluable")]
    # ---- naming test-retest, computed BEFORE the verdict -----------------------------------
    # The registered pass line (Plan plan-single-frame-random-question-r3 §10) is a
    # function of this measured rate, so the rate has to exist before the verdict is taken.
    # Same loop as below, only earlier; the numbers it produces are unchanged.
    both, dis, rep = 0, 0, []
    for p in pairs_flat:
        a = p.get("naming") or {}
        b = p.get("retest") or {}
        ta, tb = a.get("named_text"), b.get("named_text")
        if ta is not None and tb is not None:
            both += 1
            if ta != tb:
                dis += 1
        if p.get("identical_repeat") is not None:
            rep.append(p["identical_repeat"])
    pl = mapping_pass_line(n_pairs, n_evaluable, n_contradictions, both, dis)
    # Two other defensible readings of the SAME pairs, always reported with the verdict
    # (Plan -r4 §14b): the instrument's fragility is a property of the control, not a
    # footnote to one run.
    pl["sensitivity"] = verdict_sensitivity(n_pairs, n_evaluable, n_contradictions, both, dis)
    # ---- the SUPERSEDED parse counted on the SAME pairs (report only, zero forwards) --------
    # The pilot-gate review requires the corrected count never to be printed alone: the
    # exact-first fix (Plan -r4 §13a) can only lower C, so a reader has to see the count the
    # old rule gives on the same 48 pairs (review §1.2 condition 3, §6 item 4).  This block
    # reads `raw_named` and `options` that the run already wrote; it changes no verdict.
    old_rows = []
    for r in rows:
        for p in r["pairs"]:
            opts_p = list(p.get("options") or [])
            li_p = p.get("letter_index")
            named_old, how_old = _parse_named_text_old_rule(p.get("raw_named"), opts_p)
            target_p = opts_p[int(li_p)] if (li_p is not None and opts_p) else None
            ev_old = bool(named_old is not None and li_p is not None)
            old_rows.append({"qid": r["qid"], "order": p["order"], "letter": p["letter"],
                             "letter_points_at": target_p,
                             "named_text_old_rule": named_old,
                             "named_parse_old_rule": how_old,
                             "evaluable": ev_old,
                             "contradiction": bool(ev_old and target_p != named_old)})
    n_old_ev = int(sum(1 for x in old_rows if x["evaluable"]))
    n_old_c = int(sum(1 for x in old_rows if x["contradiction"]))
    pl_old = mapping_pass_line(n_pairs, n_old_ev, n_old_c, both, dis)
    contradiction_counts = {
        "corrected_exact_first": {
            "n_contradictions": int(n_contradictions), "n_evaluable": int(n_evaluable),
            "contradictions_plus_unevaluable": int(n_contradictions + (n_pairs - n_evaluable)),
            "status": pl["status"],
            "rule": "Plan -r4 §13a: an exact match outranks a containment match; longest "
                    "match only as the fallback when nothing is exact"},
        "old_rule_longest_match": {
            "n_contradictions": n_old_c, "n_evaluable": n_old_ev,
            "contradictions_plus_unevaluable": int(n_old_c + (n_pairs - n_old_ev)),
            "status": pl_old["status"], "tolerance_b": pl_old.get("tolerance_b"),
            "rule": "the superseded parse: the longest candidate wins even when a shorter "
                    "candidate is an exact match of the reply (instance 2sriHX3PbXw#511, "
                    "permuted order)"},
        "printed_both_because": "review §1.2 condition 3: the corrected count must never be "
                                "printed alone, because the fix can only lower it",
        "pairs": old_rows,
    }
    status = pl["status"]
    reason = pl["reason"]
    # ---- r6: the REGISTERED pass line of this round and the 21-unit denominator ----------------
    # Plan r6 goal ("判据与通过线"): mapping_ok = (n_pairs>0) AND (n_evaluable==n_pairs) AND
    # (n_contradictions==0); any other outcome makes the cell instrument_failure.  The r3
    # statistical line above is still computed and reported (`status`, `pass_line`); nothing
    # is loosened -- the registered line is the stricter one whenever the two differ.
    zero_ok = bool(n_pairs > 0 and n_evaluable == n_pairs and n_contradictions == 0)
    # Unit-level aggregation (Plan r6 controls.positive (v), data_analysis §3): 42 pairs are
    # 21 units x 2 option orders; the two pairs of one unit are one unit's measurement, not
    # two independent observations.  Main denominator = units; the pair denominator is kept
    # beside it.  Mechanical split of the contradicting units: STABLE = both option orders
    # contradict; ORDER-SPECIFIC = exactly one does.  Zero extra forwards; reads only the
    # pairs already recorded.
    unit_rows = []
    n_units_wc = n_stable = n_order_specific = 0
    for r in rows:
        cs = [1 if (p.get("evaluable") and not p.get("agrees")) else 0 for p in r["pairs"]]
        c_u = int(sum(cs))
        if c_u > 0:
            n_units_wc += 1
            if all(x == 1 for x in cs):
                n_stable += 1
                cls = "stable"
            else:
                n_order_specific += 1
                cls = "order_specific"
        else:
            cls = "none"
        unit_rows.append({"qid": r["qid"], "n_pairs": len(cs),
                          "n_contradiction_pairs": c_u, "class": cls})
    unit_level = {
        "n_units": n, "n_orders_per_unit": 2,
        "units_with_contradiction": int(n_units_wc),
        "unit_contradiction_rate": (float(n_units_wc) / n if n else None),
        "units_stable_contradiction": int(n_stable),
        "units_order_specific_contradiction": int(n_order_specific),
        "stable_definition": "both option orders of the unit contradict the criterion",
        "order_specific_definition": "exactly one of the unit's two option orders contradicts",
        "per_unit": unit_rows,
        "rule": "r6 primary denominator: the unit (21 units = 42 pairs x 1/2). The pair "
                "denominator stays beside it in `n_pairs`/`n_evaluable`/`n_contradictions`. "
                "No significance test is run on either denominator (Plan r6 data_analysis §2); "
                "the criterion's verdict comes from the zero line only.",
    }
    mapping = {
        "status": status, "reason": reason,
        "mapping_ok": zero_ok,
        "zero_line": {
            "mapping_ok": zero_ok,
            "status": ("pass" if zero_ok else "fail"),
            "n_pairs": n_pairs, "n_evaluable": n_evaluable,
            "n_contradictions": n_contradictions,
            "rule": "registered in Plan plan-four-readings-second-backbone-r6 (goal): "
                    "mapping_ok = (n_pairs>0) AND (n_evaluable==n_pairs) AND "
                    "(n_contradictions==0); anything else is instrument_failure for the cell. "
                    "`status`/`pass_line` beside it keep the r3 statistical line unchanged; "
                    "when the two disagree this line is the stricter one.",
        },
        "pass_line": pl,
        "n_pairs": n_pairs, "n_units": n, "n_orders_per_unit": 2,
        "unit_level": unit_level,
        "n_evaluable": n_evaluable, "n_contradictions": n_contradictions,
        "contradiction_counts": contradiction_counts,
        "n_unevaluable_charged": int(n_pairs - n_evaluable),
        "contradictions_plus_unevaluable": int(n_contradictions + (n_pairs - n_evaluable)),
        "r2_rule_status": pl.get("r2_rule_status"),
        "coverage": pl.get("coverage"),
        "refusal_phrases": list(REFUSAL_PHRASES),
        "n_refusal_like": int(n_ref_matched + n_ref_no_option),
        "n_refusal_matched_option": n_ref_matched,
        "n_refusal_no_option": n_ref_no_option,
        "n_units_with_refusal_option": n_units_with_refusal_option,
        "n_recovered_pairs": len(recovered_pairs),
        "recovered_pairs": recovered_pairs,
        "unevaluable_pairs": unevaluable_pairs,
        "n_unparseable": n_unparseable, "n_letter_missing": n_letter_missing,
        "n_letter_reply": n_letter_reply,
        "n_resolved_by_escalation": n_resolved,
        "n_letter_reply_after_escalation": int(sum(
            1 for p in pairs_flat
            if p.get("named_parse") == "letter_reply_after_escalation")),
        "contradictions": [{"qid": r["qid"], "order": p["order"], "letter": p["letter"],
                            "letter_points_at": p["letter_points_at"],
                            "named_text": p["named_text"], "raw_named": p["raw_named"]}
                           for r in rows for p in r["pairs"]
                           if p["evaluable"] and not p["agrees"]],
        "rule": "THE CRITERION. For each unit and each option order, opts[idx(letter)] must "
                "equal the option the model names in free text on that same order, read from "
                "the naming ask and matched on normalised strings. Never derived from the "
                "letter. The PASS LINE is the one registered in Plan "
                "plan-single-frame-random-question-r3 §9-§10 and computed in "
                "`mapping_pass_line`: a pair that cannot be judged is charged as a worst-case "
                "contradiction, so pass iff C + e <= B (the contradiction count plus the "
                "unevaluable pairs does not exceed what the naming step's own test-retest rate "
                "can produce at 95%) and e <= e_max (the derived coverage floor); fail iff "
                "C > B; incomplete iff C <= B < C + e or e > e_max. A pass says 'no gross "
                "mis-wiring', never 'the map is correct'.",
        "why_the_line_changed": "two after-the-fact corrections. (1) The line used to be zero "
                                "contradictions; the naming step disagrees with itself on 1 of "
                                "8 pairs when only the option-list labelling changes, so a "
                                "48-pair cell passed at 0.0016 and the criterion could only "
                                "ever withhold; it was replaced by the line derived from that "
                                "same measured rate on the same pairs (r2 §5). (2) That line "
                                "still demanded every pair evaluable, which made one refusing "
                                "unit enough to withhold the cell; unevaluable pairs are now "
                                "charged as worst-case contradictions and the coverage floor is "
                                "derived from the same inequality (r3 §10). At e = 0 the two "
                                "versions agree; at e > 0 r3 demands strictly fewer "
                                "contradictions. The criterion, the pairs and the consequence "
                                "(landing = null unless pass) are unchanged.",
        "why_split": "the r2 rule `n_text_preserved == n` measured the model's sensitivity to "
                     "option order (5/6, 4/6, 2/3, 5/6; 0 of 4 cells passed) while 0 of 21 "
                     "units carried the mapping-fault signature; it was withdrawn as an "
                     "after-the-fact change and the corrected control re-run",
        "naming_rule": "a bare-letter reply is classified `letter_reply`, escalated ONCE with "
                       "the block applied to the first two generated positions and an "
                       "instruction not to write a letter, and either resolved by that retry "
                       "(`resolved_by_escalation`) or recorded as "
                       "`letter_reply_after_escalation` and counted unevaluable. The letter is "
                       "never converted into option text.",
    }

    # ---- naming test-retest: reported here, computed above (the pass line reads it) --------
    naming_retest = {
        "n_pairs": n_pairs,
        "n_both_parsed": int(both),
        "n_disagree": int(dis),
        "disagreement_rate": (float(dis) / both if both else None),
        "n_letter_reply_A": int(sum(1 for p in pairs_flat
                                    if str((p.get("naming") or {}).get("parse") or "")
                                    .startswith("letter_reply"))),
        "n_letter_reply_B": int(sum(1 for p in pairs_flat
                                    if str((p.get("retest") or {}).get("parse") or "")
                                    .startswith("letter_reply"))),
        "identical_repeat": {
            "n": len(rep),
            "n_identical": int(sum(1 for r in rep if r.get("identical"))),
            "structurally_zero_note": "a greedy decode repeated with the same prompt is "
                                      "bit-identical by construction, so this is a check that "
                                      "the naming decode path is deterministic and NOT the "
                                      "noise floor",
        },
        "channels": {"A": "primary (option texts only, no letter labels) -- feeds `mapping`",
                     "B": "test-retest (same texts, same order, with the letter labels)"},
        "rule": "the noise floor is the A-vs-B disagreement rate: same question, same option "
                "texts, same order, only the option-list labelling differs, so the answer "
                "should not change. Its one-sided 95% upper bound SETS the pass line of "
                "`mapping` (see `mapping.pass_line`, Plan r3 §10); it is not decoration.",
    }

    fault_block = fault_injection(pairs_flat, n_contradictions, seed, both, dis)
    mapping["fault_injection"] = fault_block
    mapping["naming_retest"] = naming_retest

    s1 = int(sum(1 for r in rows if r["text_preserved"] and r["letter_moved"]))
    s2 = int(sum(1 for r in rows if r["text_preserved"] and not r["letter_moved"]))
    s3 = int(sum(1 for r in rows if not r["text_preserved"] and r["letter_moved"]))
    s4 = int(sum(1 for r in rows if not r["text_preserved"] and not r["letter_moved"]))
    n_fault_sig = int(sum(1 for r in rows if r["text_preserved"] and r["letter_moved"] is False
                          and r["perm_fixes_chosen_index"] is False))
    order_sensitivity = {
        "n": n,
        "n_text_preserved": int(sum(1 for r in rows if r["text_preserved"])),
        "hits_original": int(sum(1 for r in rows if r["correct_orig"])),
        "hits_permuted": int(sum(1 for r in rows if r["correct_permuted"])),
        "patterns": {
            "S1_text_preserved_letter_moved": s1,
            "S2_text_preserved_letter_stayed": s2,
            "S3_text_changed_letter_moved": s3,
            "S4_text_changed_letter_stayed": s4,
            "note": "text here is the option the letter points at, as in r2: this half is a "
                    "description of how the answers moved when the options were rotated, and "
                    "it decides nothing. The criterion is `mapping` above and it does not "
                    "derive text from the letter.",
        },
        "n_mapping_fault_signature": n_fault_sig,
        "mapping_fault_signature": "text preserved AND the text's index was moved by the "
                                   "permutation AND the letter did not move",
        "mapping_fault_signature_note": "structurally zero when `text` is derived from the "
                                        "letter (text preserved forces perm[li] == li), which "
                                        "is exactly why the criterion had to move to the "
                                        "free-text naming ask",
        "no_pass_line": True,
    }
    arithmetic_ok = bool(all(r["perm_is_a_permutation"] and r["options_permuted_is_arithmetic"]
                             and (r["gold_permuted_is_arithmetic"] is not False)
                             for r in rows))
    return {"units": rows, "n": n, "mapping": mapping,
            "order_sensitivity": order_sensitivity,
            "arithmetic_ok": arithmetic_ok,
            "naming_retest": naming_retest,
            "fault_injection": fault_block,
            "naming_prompt_template": prompt_for_named_text("<question>", ["<option 1>",
                                                                          "<option 2>"]),
            "naming_prompt_template_channel_B": prompt_for_named_text(
                "<question>", ["<option 1>", "<option 2>"], letters=LETTERS[:2]),
            "naming_forced_prefix": NAMING_FORCED_PREFIX,
            "naming_max_new_tokens": int(NAMED_MAX_NEW_TOKENS),
            "naming_escalation_instruction": NAMING_ESCALATION_INSTRUCTION,
            "naming_block": {
                "ids": [int(i) for i in block_ids],
                "n_ids": len(block_ids),
                "computed_from": "the tokenizer's own vocabulary: every id that decodes to a "
                                 "bare single Latin letter (any case, optionally wrapped in "
                                 "()/[] and/or closed with . : ) -), plus the same check "
                                 "applied again at the second generated position when the "
                                 "first generated token is whitespace only",
                "what": "those ids are suppressed at the first generated token of the naming "
                        "ask only (and at the second position on the escalation retry), so "
                        "the answer cannot be a bare letter. No option is favoured: only "
                        "single-letter tokens are removed, and a letter that is itself an "
                        "option's whole text is not removed for that unit "
                        "(`blocking_conflicts_with_option`).",
                "why": "v1 and v2 both got a bare letter back under two different "
                       "instructions (task-bbb62d92f8, task-83b6aa08bc: 0 of 30 pairs "
                       "evaluable each), and v3's hard-coded ids 32-36 (upper-case A-E only) "
                       "were bypassed with lower-case d/c/b and out-of-range F/G/H/M "
                       "(task-f0d4a231c4: 26 of 42 unevaluable)",
                "supersedes": "the v3 `naming_first_token_block.ids` = [32, 33, 34, 35, 36]",
            },
            "rule": "same frames, same question, option order rotated with a permutation drawn "
                    "from EXP_SEED. What this control reports is split in two: `mapping` "
                    "(criterion) and `order_sensitivity` (quantity, no pass line). The naming "
                    "step is read on two channels (A primary, B test-retest) and the criterion "
                    "is self-tested by a deliberate offset of the letter->option map."}


# ---------------------------------------------------------------- answer visibility

def _render_word(word):
    """Paint the word on a blank frame.  Synthetic by construction and labelled as such."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        f = ImageFont.load_default(size=44)
        im = Image.new("RGB", (448, 448), "white")
        d = ImageDraw.Draw(im)
        bb = d.textbbox((0, 0), word, font=f)
        d.text(((448 - (bb[2] - bb[0])) // 2 - bb[0],
                (448 - (bb[3] - bb[1])) // 2 - bb[1]), word, fill="black", font=f)
        return im, "ImageFont.load_default(size=44) on 448x448"
    except TypeError:
        f = ImageFont.load_default()
        small = Image.new("RGB", (128, 128), "white")
        d = ImageDraw.Draw(small)
        bb = d.textbbox((0, 0), word, font=f)
        d.text(((128 - (bb[2] - bb[0])) // 2 - bb[0],
                (128 - (bb[3] - bb[1])) // 2 - bb[1]), word, fill="black", font=f)
        return small.resize((448, 448), Image.NEAREST), \
            "ImageFont.load_default() on 128x128, upscaled to 448x448 (NEAREST)"


def answer_visibility(model, proc, dev, ids, counter):
    rows, renderers = [], set()
    for i in range(N_VISIBILITY):
        base = list(VIS_WORDS)
        correct = base[i % len(base)]
        pos = i % 4
        rest = [w for w in base if w != correct]
        opts = rest[:pos] + [correct] + rest[pos:]
        gold = LETTERS[pos]
        im, renderer = _render_word(correct)
        renderers.add(renderer)
        prompt = G.prompt_for({"question": "Which word is written in the image?",
                               "candidates": opts})
        raw, letter = G.answer_arm(model, proc, [im], prompt, dev, counter=counter,
                                   kind="control:visibility_answer")
        lo = C.option_logits(model, proc, [im], prompt, dev, ids, batch=1)
        counter.count("control:visibility_score", 1)
        s_letter = LETTERS[int(np.argmax(lo[0]))]
        rows.append({"i": i, "word": correct, "options": opts, "gold": gold,
                     "answer_letter": letter, "answer_raw": raw,
                     "score_letter": s_letter,
                     "answer_correct": bool(letter == gold),
                     "score_correct": bool(s_letter == gold)})
    n = len(rows)
    return {"units": rows, "n": n,
            "hits_answer": int(sum(1 for r in rows if r["answer_correct"])),
            "hits_score": int(sum(1 for r in rows if r["score_correct"])),
            "all_answer_correct": bool(n and all(r["answer_correct"] for r in rows)),
            "all_score_correct": bool(n and all(r["score_correct"] for r in rows)),
            "chance": 0.25,
            "renderer": sorted(renderers),
            "stimulus": "SYNTHETIC CONTROL, not carrier data: a blank frame with the correct "
                        "option's word painted on it, four options, the correct option's letter "
                        "cycles A,B,C,D across the units. The model is guaranteed to be able "
                        "to answer it; anything below n/n points at the prompt template, the "
                        "option<->letter map or the letter extraction, not at the model.",
            "rule": "reported through the same answer path and the same single-image scoring "
                    "path the readings use; it is a positive control and enters no table."}


# ---------------------------------------------------------------- the instrument-failure guard

def letter_distributions(rows, arms):
    """Per-arm letter counts over the cells this run actually answered."""
    out = {}
    for a in arms:
        cnt = Counter()
        for r in rows:
            v = (r.get("letters") or {}).get(a)
            if v in LETTERS:
                cnt[v] += 1
        n = int(sum(cnt.values()))
        out[a] = {"counts": dict(sorted(cnt.items())), "n_answered": n,
                  "distinct_letters": len(cnt),
                  "constant": bool(n > 0 and len(cnt) == 1)}
    return out


def instrument_guard(controls, distributions, annotation_arms=ANNOTATION_ONLY_ARMS,
                     min_n=NEAR_CONSTANT_MIN_N):
    """The plan's instrument-failure trigger, implemented (review r2 §5.2).

    Four triggers: the letter decode, the three-path determinism, an arm whose answered letters
    are all one letter, and the mapping criterion.  The consequence is written by the caller
    (`landing = null`, branch blocks marked withheld); it is not a string in a metrics field.
    """
    triggers = []
    near = {}
    ld = controls.get("letter_decode") or {}
    if not ld.get("all_ok"):
        triggers.append({"id": "letter_decode",
                         "detail": "an option letter's token id did not decode back to it",
                         "observed": ld.get("rows")})
    det = controls.get("determinism") or {}
    if not (int(det.get("n_paths") or 0) == 3 and det.get("all_agree")):
        triggers.append({"id": "determinism",
                         "detail": "the same call did not repeat identically on all three "
                                   "prompt paths",
                         "observed": {"n_paths": det.get("n_paths"),
                                      "n_agree": det.get("n_agree")}})
    for arm in sorted(distributions):
        d = distributions[arm]
        if arm in annotation_arms:
            near[arm] = {"status": "excluded", "n_answered": d["n_answered"],
                         "counts": d["counts"],
                         "why": "annotation-only arm: its letters are written by the code from "
                                "the gold annotations and never come from the model; "
                                "constant_prior is constant by construction"}
        elif d["n_answered"] < int(min_n):
            near[arm] = {"status": "not_evaluable", "n_answered": d["n_answered"],
                         "counts": d["counts"], "min_n": int(min_n),
                         "why": "below the floor: under a uniform-letter null, n cells all the "
                                "same has probability 5*(1/5)^n, so no arm of this size can "
                                "distinguish a broken instrument from a favourite letter"}
        elif d["constant"]:
            near[arm] = {"status": "triggered", "n_answered": d["n_answered"],
                         "counts": d["counts"]}
            triggers.append({"id": "near_constant_letter_distribution", "arm": arm,
                             "counts": d["counts"], "n_answered": d["n_answered"],
                             "detail": "every answered cell of a model arm is the same letter"})
        else:
            near[arm] = {"status": "ok", "n_answered": d["n_answered"],
                         "counts": d["counts"],
                         "max_letter_share": (float(max(d["counts"].values()))
                                              / d["n_answered"] if d["n_answered"] else None)}
    mp = ((controls.get("option_control") or {}).get("mapping")) or {}
    m_status = mp.get("status")
    if m_status is None or not mp.get("mapping_ok"):
        triggers.append({"id": "option_mapping",
                         "detail": "the option<->letter mapping criterion is not satisfied",
                         "status": m_status,
                         # r6: mapping_ok is the registered zero line; the r3 statistical
                         # status is kept beside it so both readings of the same pairs show
                         "zero_line_status": ((mp.get("zero_line") or {}).get("status")),
                         "n_contradictions": mp.get("n_contradictions"),
                         "n_evaluable": mp.get("n_evaluable"),
                         "n_pairs": mp.get("n_pairs"),
                         "n_unparseable": mp.get("n_unparseable"),
                         "units_with_contradiction": ((mp.get("unit_level") or {})
                                                      .get("units_with_contradiction"))})
    return {
        "triggered": bool(triggers), "triggers": triggers,
        "near_constant": near,
        "near_constant_min_n": int(min_n),
        "excluded_arms": sorted(annotation_arms),
        "mapping_status": m_status, "mapping_ok": mp.get("mapping_ok"),
        "rule": "implemented, not worded: any trigger makes this cell report instrument_failure "
                "and write no landing. Triggers are the letter decode, the three-path "
                "determinism, a model arm whose answered letters are all one letter (at "
                "n >= %d, annotation-only arms excluded), and a mapping criterion that is not "
                "satisfied." % int(min_n),
    }


# ---------------------------------------------------------------- the whole battery

def build(model, proc, dev, ids, state, counter, exp_seed):
    """Run every instrument control for one reading and return the metrics block."""
    out = {"letter_decode": letter_decode(proc, ids)}
    paths = state.get("paths")
    if paths:
        out["vision_tokens"] = vision_tokens(model, proc, [
            {"arm": "answer:16_frames", "path": "many-image answer prompt (16 frames)",
             "pils": paths["frames16"], "prompt": paths["prompt"]},
            {"arm": "answer:1_frame", "path": "single-image answer prompt (1 frame)",
             "pils": [paths["frame1"]], "prompt": paths["prompt"]},
            {"arm": "answer:0_frames", "path": "text-only answer prompt (no image)",
             "pils": [], "prompt": paths["prompt"]},
            {"arm": "score:single_image", "path": "one image as its own prompt, option-letter "
                                                  "logits", "pils": [paths["frame1"]],
             "prompt": paths["prompt"]},
        ])
        out["vision_tokens"]["arm_remap"] = {
            "16-frame answer arms": [paths["arm16"]],
            "1-frame answer arms": [paths["arm1"]],
            "single-image scoring arms": ["score:* (option_logits scores one image per prompt)"],
            "no_frames": ["answer:0_frames"],
        }
        out["determinism"] = determinism(model, proc, dev, ids, [
            {"name": "answer_16_frames", "kind": "answer", "pils": paths["frames16"],
             "prompt": paths["prompt"]},
            {"name": "answer_1_frame", "kind": "answer", "pils": [paths["frame1"]],
             "prompt": paths["prompt"]},
            {"name": "score_1_image", "kind": "score", "pils": [paths["frame1"]],
             "prompt": paths["prompt"]},
        ], counter)
    else:
        out["vision_tokens"] = {"available": False, "note": "no control case captured"}
        out["determinism"] = {"available": False, "note": "no control case captured"}
    out["option_control"] = option_control(model, proc, dev, state.get("perm_units") or [],
                                           counter, perm_seed(exp_seed), letter_ids=ids)
    out["answer_visibility"] = answer_visibility(model, proc, dev, ids, counter)
    return out