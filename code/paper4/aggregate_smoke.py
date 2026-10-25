"""Copy the blocks a gate has to look at out of the four pilot runs into the root metrics.json.

It copies, it does not compute.  Every number under `readings` is read from
runs/<reading>/metrics.json and labelled with the file and field it came from, so a reader can
check both files against each other; nothing there is recomputed, rounded or restated from an
earlier batch.

The one block that is derived rather than copied is `per_unit_vision_tokens`, and it is
labelled as such: it is plain arithmetic over the `vision_tokens` field of each unit's row in
results/four_<reading>_samples.jsonl, which is the run's own per-unit record.  It is here so a
reader does not have to page a samples file to see whether every unit fed the same number of
vision tokens per image.

Plan: plan-four-readings-second-backbone-r3
Review: review-pilot-gate-four-readings-second-backbone-r2 (six required fixes)
"""
import hashlib
import json
import os

READINGS = ["selection", "single", "distortion", "donor"]

# reading -> (block name in that run's metrics.json, the fields the r2 review renamed)
RENAMED = {
    "selection": ("structure", ["checked", "n_measured", "n_pool",
                                "n_measured_videos", "n_pool_videos"]),
    "distortion": ("pin", ["n_pairs_measured_this_run", "n_pairs_in_pin"]),
    "donor": ("identity", ["n_units_measured_this_run", "n_inforce_pool"]),
}

# The names these fields carried before the rename, and why they were wrong.
OLD_NAMES = {
    "selection": {"matches_expected": "checked (and expected_* is no longer emitted in a "
                                      "pilot at all)",
                  "pool_questions / expected_questions_full": "n_pool / n_measured"},
    "distortion": {"n_pairs": "n_pairs_measured_this_run (+ n_pairs_in_pin)"},
    "donor": {"n_visible_in_force": "n_units_measured_this_run (+ n_inforce_pool)"},
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def per_unit_vision_tokens(samples_path):
    """Derived: per-arm, per-unit vision-token counts read from the run's own samples file.

    Nothing is estimated: each value is the `vision_tokens_per_image` the arm's own call
    recorded for that unit.  A reading with one distinct value across all units is the only
    state in which "every unit fed the same per-image count" is a measurement and not an
    extrapolation from one representative unit.
    """
    if not os.path.exists(samples_path):
        return {"available": False, "path": samples_path}
    per_arm = {}
    n_units = 0
    for line in open(samples_path):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        n_units += 1
        for arm, blk in (row.get("vision_tokens") or {}).items():
            d = per_arm.setdefault(arm, {"values": set(), "units": 0, "missing": 0})
            d["units"] += 1
            v = blk.get("vision_tokens_per_image")
            if v is None:
                d["missing"] += 1
            else:
                d["values"].add(float(v))
    out = {"available": True, "path": samples_path, "n_units": n_units, "per_arm": {},
           "derived_from": "the `vision_tokens` field of each unit's row in %s; plain "
                           "arithmetic over the run's own per-unit record, not a copy of a "
                           "metrics field and not an estimate" % samples_path}
    for arm in sorted(per_arm):
        d = per_arm[arm]
        vals = sorted(d["values"])
        out["per_arm"][arm] = {"n_units": d["units"], "n_missing": d["missing"],
                               "per_image_values": vals,
                               "constant_across_units": bool(len(vals) == 1)}
    return out


def main():
    codes_path = "runs/_exit_codes.json"
    codes = {}
    plan = {}
    if os.path.exists(codes_path):
        meta = json.load(open(codes_path))
        codes = meta.get("exit_codes", {})
        plan = {"plan_id": meta.get("plan_id"), "plan_version": meta.get("plan_version")}

    out = {
        "smoke": {
            "what": "r6 instrument pilot: the v4 naming-step stimulus UNCHANGED (channel A "
                    "option texts only with a forced answer prefix and a vocabulary-level "
                    "first-token block, the stated bare-letter rule, the channel-B test-retest "
                    "noise floor, the fault injection) plus the three r6 additions -- the "
                    "corrected `fired` boolean, the 21-unit denominator with the "
                    "stable/order-specific split, and the registered zero line as the verdict "
                    "(n_pairs>0 AND all evaluable AND 0 contradictions). The same block runs "
                    "on the second backbone (internvl3hf) and, as the criterion's own positive "
                    "control, on the first backbone (qwen2vl) with the same code path.",
            "review": "review-evidence-second-backbone-v4-and-third-model-probe (the "
                      "v4 smoke it corrects is evidence.task-78e2c44d4a)",
            "stage": "smoke",
            "decides_claim": False,
            "note": "Every sub-run is stage=smoke, decides_claim=false, pilot mode. Nothing "
                    "here is a reading and nothing here may be quoted as one; the full runs "
                    "wait on the gate being re-recorded against this Plan version.",
            "plan": plan,
            "exit_codes": codes,
            "readings": {},
            "metrics_files_sha256": {},
        }
    }

    summary = {}
    for r in READINGS:
        p = "runs/%s/metrics.json" % r
        if not os.path.exists(p):
            out["smoke"]["readings"][r] = {"available": False, "exit_code": codes.get(r),
                                           "note": "no metrics.json written by this reading"}
            continue
        m = json.load(open(p))
        out["smoke"]["metrics_files_sha256"][p] = sha256_file(p)
        blk = {
            "available": True,
            "exit_code": codes.get(r),
            "copied_from": p,
            "run": m.get("run"),
            "verdict": m.get("verdict"),
            "forward_counts": {"total": (m.get("forward_counts") or {}).get("total"),
                               "by_kind": (m.get("forward_counts") or {}).get("by_kind")},
            "instrument_controls": m.get("instrument_controls"),
            "vision_tokens": m.get("vision_tokens"),
            "instrument_guard": m.get("instrument_guard"),
            "instrument_failure": m.get("instrument_failure"),
            "landing": m.get("landing"),
            "renamed_fields": {},
            "per_unit_vision_tokens": per_unit_vision_tokens(
                "results/four_%s_samples.jsonl" % r),
        }
        if r in RENAMED:
            name, fields = RENAMED[r]
            src = m.get(name) or {}
            blk["renamed_fields"] = {
                "block": name,
                "old_to_new": OLD_NAMES.get(r, {}),
                "values_copied": {f: src.get(f) for f in fields},
                "copied_from": "%s#%s" % (p, name),
            }
        out["smoke"]["readings"][r] = blk

        ic = m.get("instrument_controls") or {}
        det = ic.get("determinism") or {}
        oc = ic.get("option_control") or {}
        mp = oc.get("mapping") or {}
        osens = oc.get("order_sensitivity") or {}
        av = ic.get("answer_visibility") or {}
        ld = ic.get("letter_decode") or {}
        vt = ic.get("vision_tokens") or {}
        gd = m.get("instrument_guard") or {}
        fi = oc.get("fault_injection") or mp.get("fault_injection") or {}
        rt = oc.get("naming_retest") or mp.get("naming_retest") or {}
        summary[r] = {
            "letter_decode_all_ok": ld.get("all_ok"),
            "determinism_paths_agreeing": "%s/%s" % (det.get("n_agree"), det.get("n_paths")),
            "determinism_all_agree": det.get("all_agree"),
            # the r3 criterion, and the r2 quantity it replaced
            "mapping_status": mp.get("status"),
            "mapping_ok": mp.get("mapping_ok"),
            "mapping_pairs": mp.get("n_pairs"),
            "mapping_evaluable": mp.get("n_evaluable"),
            "mapping_contradictions": mp.get("n_contradictions"),
            "mapping_unparseable": mp.get("n_unparseable"),
            # r6: the registered zero line that governs mapping_ok, and the 21-unit
            # denominator with the stable / order-specific split (Plan r6 data_analysis §3)
            "mapping_zero_line_ok": ((mp.get("zero_line") or {}).get("mapping_ok")),
            "mapping_zero_line_status": ((mp.get("zero_line") or {}).get("status")),
            "mapping_n_units": ((mp.get("unit_level") or {}).get("n_units")),
            "units_with_contradiction": ((mp.get("unit_level") or {})
                                         .get("units_with_contradiction")),
            "units_stable_contradiction": ((mp.get("unit_level") or {})
                                           .get("units_stable_contradiction")),
            "units_order_specific_contradiction": ((mp.get("unit_level") or {})
                                                   .get("units_order_specific_contradiction")),
            "unit_contradiction_rate": ((mp.get("unit_level") or {})
                                        .get("unit_contradiction_rate")),
            # v4: a bare-letter reply is a named class, not a silent "unparseable"
            "naming_letter_reply": mp.get("n_letter_reply"),
            "naming_resolved_by_escalation": mp.get("n_resolved_by_escalation"),
            "naming_letter_reply_after_escalation": mp.get("n_letter_reply_after_escalation"),
            "naming_retest_both_parsed": rt.get("n_both_parsed"),
            "naming_retest_disagree": rt.get("n_disagree"),
            "naming_retest_disagreement_rate": rt.get("disagreement_rate"),
            "naming_identical_repeat": (rt.get("identical_repeat") or {}).get("n_identical"),
            "naming_identical_repeat_n": (rt.get("identical_repeat") or {}).get("n"),
            "naming_forced_prefix": oc.get("naming_forced_prefix"),
            "naming_block_ids": (oc.get("naming_block") or {}).get("n_ids"),
            "criterion_fault_injection_fired": fi.get("fired"),
            "criterion_fault_injection": {
                "identity_contradictions": (fi.get("identity") or {}).get("n_contradictions"),
                "shift_plus1_contradictions":
                    (fi.get("shift_plus1") or {}).get("n_contradictions"),
                "shift_plus1_status": (fi.get("shift_plus1") or {}).get("status"),
                "shift_plus1_status_zero_line":
                    (fi.get("shift_plus1") or {}).get("status_zero_line"),
                "shift_minus1_contradictions":
                    (fi.get("shift_minus1") or {}).get("n_contradictions"),
                "partial_half_contradictions":
                    (fi.get("partial_half") or {}).get("n_contradictions"),
                "identity_reproduces_reported": fi.get("identity_reproduces_reported"),
                "n_contradictions_increased": fi.get("n_contradictions_increased"),
                "verdict_changes": fi.get("verdict_changes"),
            },
            "option_arithmetic_ok": oc.get("arithmetic_ok"),
            "order_sensitivity_text_preserved": "%s/%s" % (osens.get("n_text_preserved"),
                                                           osens.get("n")),
            "order_sensitivity_patterns": osens.get("patterns"),
            "order_sensitivity_fault_signature": osens.get("n_mapping_fault_signature"),
            "order_sensitivity_is_a_criterion": False,
            "visibility_hits_answer": "%s/%s" % (av.get("hits_answer"), av.get("n")),
            "visibility_hits_score": "%s/%s" % (av.get("hits_score"), av.get("n")),
            "vision_tokens_16_frames": (vt.get("answer:16_frames") or {}).get(
                "vision_tokens_total"),
            "vision_tokens_1_frame": (vt.get("answer:1_frame") or {}).get(
                "vision_tokens_total"),
            "vision_tokens_score_1_image": (vt.get("score:single_image") or {}).get(
                "vision_tokens_total"),
            "vision_tokens_0_frames": (vt.get("answer:0_frames") or {}).get(
                "vision_tokens_total"),
            "vision_tokens_per_image_equality": vt.get("per_image_equality"),
            "instrument_guard_triggered": gd.get("triggered"),
            "instrument_failure_triggered": (m.get("instrument_failure") or {}).get(
                "triggered"),
            "landing_written": m.get("landing") is not None,
        }
    out["smoke"]["controls_summary"] = summary

    # ---- pooled totals over the four cells (sums only; no test, no estimate) ----------------
    # The pass line does not move: a cell passes only with zero contradictions and every pair
    # evaluable.  These totals exist so the four cells can be read at once without adding
    # anything to them.
    tot = {"n_pairs": 0, "n_evaluable": 0, "n_contradictions": 0, "n_unparseable": 0,
           "n_letter_reply": 0, "retest_both": 0, "retest_disagree": 0,
           "n_units": 0, "units_with_contradiction": 0, "units_stable_contradiction": 0,
           "units_order_specific_contradiction": 0}
    cells_pass = {}
    cells_zero = {}
    for r in READINGS:
        s = summary.get(r)
        if not s:
            continue
        for k_src, k_dst in (("mapping_pairs", "n_pairs"),
                             ("mapping_evaluable", "n_evaluable"),
                             ("mapping_contradictions", "n_contradictions"),
                             ("mapping_unparseable", "n_unparseable"),
                             ("naming_letter_reply", "n_letter_reply"),
                             ("naming_retest_both_parsed", "retest_both"),
                             ("naming_retest_disagree", "retest_disagree"),
                             ("mapping_n_units", "n_units"),
                             ("units_with_contradiction", "units_with_contradiction"),
                             ("units_stable_contradiction", "units_stable_contradiction"),
                             ("units_order_specific_contradiction",
                              "units_order_specific_contradiction")):
            v = s.get(k_src)
            if isinstance(v, int):
                tot[k_dst] += v
        cells_pass[r] = bool(s.get("mapping_ok"))
        cells_zero[r] = bool(s.get("mapping_zero_line_ok"))
    tot["contradiction_rate_among_evaluable"] = (
        float(tot["n_contradictions"]) / tot["n_evaluable"] if tot["n_evaluable"] else None)
    tot["evaluable_rate"] = (float(tot["n_evaluable"]) / tot["n_pairs"]
                             if tot["n_pairs"] else None)
    tot["naming_noise_floor"] = (float(tot["retest_disagree"]) / tot["retest_both"]
                                 if tot["retest_both"] else None)
    tot["unit_contradiction_rate"] = (
        float(tot["units_with_contradiction"]) / tot["n_units"] if tot["n_units"] else None)
    tot["cells_mapping_ok"] = cells_pass
    tot["cells_zero_line_ok"] = cells_zero
    tot["all_cells_pass"] = bool(cells_pass and all(cells_pass.values()))
    tot["all_cells_zero_line_ok"] = bool(cells_zero and all(cells_zero.values()))
    tot["criterion_fault_injection_fired_all_cells"] = bool(
        cells_pass and all(summary[r].get("criterion_fault_injection_fired")
                           for r in cells_pass))
    tot["denominators"] = (
        "pair denominator: n_pairs (42 registered pairs = 21 units x 2 option orders). "
        "unit denominator: n_units (21). Both are reported; the r6 registered verdict is the "
        "zero line on the pair counts, and no significance test is run on either.")
    tot["pass_line"] = ("a cell passes only when every pair is evaluable and there are zero "
                        "contradictions; the pooled block adds the four cells' counts and "
                        "changes no cell's verdict")
    out["smoke"]["pooled"] = tot

    with open("metrics.json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
    print(json.dumps({"controls_summary": summary, "exit_codes": codes},
                     ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
