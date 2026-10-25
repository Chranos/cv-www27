"""The ONE arm-key constructor of the r4 batch (Plan plan-matrix-longvideo-r4@v2).

Why this module exists.  r3's arm_key() built "kind_k{k}" whenever k was given, while the
writing side wrote FLAT arms under their bare key.  compare_block therefore asked for
"uniform@16_k0" on records that only have "uniform@16": every comparison against a flat arm
read as "all wrong", accuracy_second collapsed to 0.0, and the mean fell back to
accuracy_first -- a number that looked like a gain and was the negative of the truth
(review-pilot-gate-matrix-r3 section 4; n_missing_second = 21/21/40).

r4 fixes the constructor, not the call sites.  A FLAT arm resolves to its bare key whatever
k/j the caller passes, so the writing side and the reading side cannot disagree.

r7 introduced the withdrawal of the donor operator D; r8 (Plan plan-matrix-longvideo-r8)
finishes it at the CODE level, not the bookkeeping level.  The r7 revision only removed the
withdrawn arm from the declared list while `r4_model_run.run_one_cell` kept answering
frames_replaced@16 and writing g2_d / g2_runseed, so any new MAIN run created fresh values
under names that must never be quoted.  From r8:

  * `arm_key` REFUSES every withdrawn name.  There is no fallback, no alias and no
    `exclude_withdrawn=False` second list: a withdrawn arm cannot be resolved at all.
  * the execution path no longer computes or writes those names (r2_run.battery_core,
    r4_model_run.run_one_cell, r4_common.g2_d_block / g2_runseed_d_block).
  * reading an ARCHIVED record is plain dict access on the recorded key; it never goes
    through this module, so archived bytes stay readable without giving the name a
    constructor.
"""

FLAT_ARM_KINDS = (
    "no_frames", "uniform@8", "uniform@16", "uniform@32", "uniform@16_repeat",
    "clip_top16", "its16", "maxvol16", "sigma_calib@0.2", "sigma_calib@0.4",
)
SEEDED_ARM_KINDS = ("ours16", "ours_quota16", "nq16", "mis16", "shuffled16")
RANDOM_KIND = "random@16"
RANDOM_DRAWS = 3

# --- withdrawn (r7 decision, enforced from r8) --------------------------------------------
WITHDRAWN_REPLACEMENT_KINDS = ("frames_replaced@16",)
WITHDRAWN_ALIAS_KEYS = ("destroyed@16",)
WITHDRAWN_STREAM_KINDS = ("p_donor_runseed_k0", "p_donor_runseed_k1", "p_donor_runseed_k2")
WITHDRAWN_RECORD_FIELDS = ("g2_d", "g2_runseed")

WITHDRAWN_NAMES = (WITHDRAWN_REPLACEMENT_KINDS + WITHDRAWN_ALIAS_KEYS
                   + WITHDRAWN_STREAM_KINDS + WITHDRAWN_RECORD_FIELDS)


class WithdrawnArmError(ValueError):
    """Raised whenever a withdrawn name is asked for.  Never caught to produce a key."""


def is_withdrawn_key(key):
    """True for every name this batch must not resolve: the donor arm, its alias, its
    streams and the two record fields that carried its readings."""
    k = str(key)
    return (k in WITHDRAWN_REPLACEMENT_KINDS or k in WITHDRAWN_ALIAS_KEYS
            or k in WITHDRAWN_STREAM_KINDS or k in WITHDRAWN_RECORD_FIELDS
            or k.startswith("p_donor_runseed_") or k.startswith("multi:p_donor"))


def _refuse_withdrawn(name):
    if is_withdrawn_key(name):
        raise WithdrawnArmError(
            "withdrawn arm/reading name %r: operator D (the donor replacement) was withdrawn "
            "in r7 and removed from the execution path in r8; no constructor resolves it"
            % (str(name),))


def arm_key(kind, k=None, j=None):
    """The only function that turns an arm spec into the key written to a record.

    Withdrawn names raise before any branch is taken.  Flat arms ignore k and j and resolve
    to their bare key (that is the r4 fix); a run-seed arm needs k and takes no j;
    random@16 needs both.  Anything else raises.
    """
    kind = str(kind)
    _refuse_withdrawn(kind)
    if kind in FLAT_ARM_KINDS:
        return kind
    if kind == RANDOM_KIND:
        if k is None or j is None:
            raise ValueError("random@16 needs both k and j (got k=%r j=%r)" % (k, j))
        return "%s_k%d_j%d" % (kind, int(k), int(j))
    if kind in SEEDED_ARM_KINDS:
        if k is None:
            raise ValueError("%s is a run-seed arm and needs k" % kind)
        if j is not None:
            raise ValueError("%s takes no j (got j=%r)" % (kind, j))
        return "%s_k%d" % (kind, int(k))
    raise ValueError("undeclared arm kind %r: no fallback key is ever produced" % kind)


def declared_arm_keys(n_seeds=3, random_draws=RANDOM_DRAWS, exclude_withdrawn=True):
    """Every arm key the battery declares: 34 keys (10 flat + 15 seeded + 9 random draws).

    `exclude_withdrawn=False` used to reproduce the historical 35-key list.  From r8 it
    raises: the historical list declared a withdrawn arm, and giving that name a constructor
    -- even one labelled historical -- is what let a new run keep writing it.  The archived
    list itself is untouched in the r7 Evidence and can be read there as data.
    """
    if not exclude_withdrawn:
        raise WithdrawnArmError(
            "the historical arm list declared the withdrawn replacement arm %r; it is no "
            "longer producible. Read the archived list from the r7 Evidence as data."
            % (WITHDRAWN_REPLACEMENT_KINDS[0],))
    keys = [arm_key(k) for k in FLAT_ARM_KINDS]
    for k in range(int(n_seeds)):
        for kind in SEEDED_ARM_KINDS:
            keys.append(arm_key(kind, k=k))
        for j in range(int(random_draws)):
            keys.append(arm_key(RANDOM_KIND, k=k, j=j))
    bad = [k for k in keys if is_withdrawn_key(k)]
    if bad:                                                       # defensive: must never fire
        raise WithdrawnArmError("declared list contains withdrawn names: %r" % (bad,))
    return keys


def is_flat(kind):
    return str(kind) in FLAT_ARM_KINDS
