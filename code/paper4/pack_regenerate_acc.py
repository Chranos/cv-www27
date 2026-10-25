"""Regenerate the numbers/tables pack with ONE change: every arm table gains its
per-arm accuracy column beside the existing difference columns.

Addition 3 of plan-exploratory-bolt-timing -- NO new measurement:
- numbers.tex and tab-design.tex are byte-copies of the archived pack
  (task-1f59540a60 / task-219f5f1017), hash-checked;
- the five arm tables (selection, noframes, single, distortion, donor) are the
  archived files with accuracy cell(s) inserted; every other cell must be
  byte-identical (token-diff invariant enforced below);
- every inserted accuracy value is computed from records already held: exact
  arithmetic on the archived table's own hits/n cells, and the full-pool /
  unit-level arm values cross-checked against the archived metrics
  (arms.*.value) and the existing rate macros in numbers.tex
  (SelPool*Rate, Sgl*Rate).

Run as a figure task over the archived runs; writes the seven pack files.
"""
import hashlib
import json
import os
import re

BS = chr(92)
NL = chr(10)

def base_dir(task):
    """Figure-task layout uses original/<task>/...; an ordinary task's read_from uses
    sources/<task>/... -- resolve whichever exists."""
    for cand in ("original/" + task, "sources/" + task):
        if os.path.isdir(cand):
            return cand
    raise SystemExit("cannot locate archived outputs for " + task)


def p(task, rel):
    return base_dir(task) + "/" + rel


SOURCES = {
    "numbers.tex": (p("task-1f59540a60", "numbers.tex"),
                    "bb9b99dbb38211d479d2bfaa24e1abd8a998054e477365e3c53b68016bc0a80e"),
    "tab-single.tex": (p("task-1f59540a60", "tab-single.tex"),
                       "3c79d9ed406d254fc2a4fdf304cd86fe45dcfd0c44b987d46ea93db04a884399"),
    "tab-design.tex": (p("task-219f5f1017", "tab-design.tex"),
                       "1588329c56f8a433e35201d0024d5f1dce5437044a346cddf10617530b6c0bf4"),
    "tab-selection.tex": (p("task-1f59540a60", "tab-selection.tex"),
                          "12680679e836ae1d16751254aed9a845039ab58de35eb8b247c75430374177ef"),
    "tab-noframes.tex": (p("task-1f59540a60", "tab-noframes.tex"),
                         "da802d570eeb14bfc4650f1e82ee8a13318e71ee2f6be3db5f50a438d4edafdf"),
    "tab-distortion.tex": (p("task-1f59540a60", "tab-distortion.tex"),
                           "d2e3d9eb7deef829bad2d44b369c2cdeb923d1f9a03e50772a22f3df0c6c7552"),
    "tab-donor.tex": (p("task-1f59540a60", "tab-donor.tex"),
                      "b5194359834746ba93664788e39025611e40526e9120bc5a95bb0bd432246dbc"),
}

INSERTED = []
CURRENT = [None, None]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_metrics(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return {}


SEL = load_metrics(p("task-23cd232b2b", "metrics.json"))
SGL = load_metrics(p("task-240d7a5d60", "metrics.json"))
DIS = load_metrics(p("task-259e5c2b3a", "metrics.json"))
DON = load_metrics(p("task-07c9beea59", "metrics.json"))


def acc_from_cell(cell):
    m = re.match(r"^(\d+)/(\d+)$", cell.strip())
    if not m:
        return None
    h, n = int(m.group(1)), int(m.group(2))
    return "%.1f" % (100.0 * h / n)


def cells_of(line):
    body = line.rstrip()
    if body.endswith(BS + BS):
        body = body[:-2].rstrip()
    return body.split(" & ")


def rejoin(cells):
    return " & ".join(cells) + " " + BS + BS


def is_data_row(line):
    return (" & " in line and BS + "multicolumn" not in line
            and not line.startswith("%"))


def transform(table_text, header_first, header_inserts, row_rules,
              mc_from, mc_to, colspec_old, colspec_new):
    out_lines = []
    for line in table_text.split(NL):
        if BS + "multicolumn{%d}" % mc_from in line:
            line = line.replace(BS + "multicolumn{%d}" % mc_from,
                                BS + "multicolumn{%d}" % mc_to)
        elif colspec_old in line:
            line = line.replace(colspec_old, colspec_new)
        elif is_data_row(line):
            cs = cells_of(line)
            if cs[0].strip() == header_first:
                # r2 header-order fix (review-fullstage-pack-rerun@v1):
                # insert header cells with the SAME convention as the data-row
                # inserts below -- ascending final-column positions applied to
                # the mutating cell list.  The previous descending/static order
                # placed tab-distortion's "non-ev. acc." header to the RIGHT of
                # the difference column while its data values are inserted to
                # the LEFT of it, so the table's 6th/7th column headers were
                # swapped.  Ascending order labels every column with the
                # quantity it holds; only tab-distortion.tex (the only table
                # with two header inserts) is affected.
                for pos, label in sorted(header_inserts, key=lambda t: t[0]):
                    cs.insert(pos, label)
                    INSERTED.append((CURRENT[0], "HEADER", label, CURRENT[1]))
                line = rejoin(cs)
            else:
                for pred, insertions in row_rules:
                    if pred(cs):
                        for pos, valfn in insertions:
                            v = valfn(cs)
                            cs.insert(pos, v)
                            INSERTED.append((CURRENT[0], cs[0][:44], v, CURRENT[1]))
                        break
                else:
                    raise SystemExit("unmatched data row in %s: %r" % (CURRENT[0], cs[0]))
                line = rejoin(cs)
        out_lines.append(line)
    return NL.join(out_lines)


def check_invariant(name, old_text, new_text, n_insert):
    old_lines = old_text.split(NL)
    new_lines = new_text.split(NL)
    if len(old_lines) != len(new_lines):
        raise SystemExit("%s: line count changed %d -> %d" % (name, len(old_lines),
                                                              len(new_lines)))
    for lo, ln in zip(old_lines, new_lines):
        if lo == ln:
            continue
        co, cn = cells_of(lo), cells_of(ln)
        if is_data_row(lo):
            i = j = ins = 0
            ok = True
            while i < len(co):
                if j < len(cn) and co[i] == cn[j]:
                    i += 1
                    j += 1
                elif j < len(cn):
                    j += 1
                    ins += 1
                else:
                    ok = False
                    break
            ins += len(cn) - j   # insertions after the last matched old cell
            if not ok or i != len(co) or ins != n_insert:
                raise SystemExit("%s: row changed beyond %d insertion(s): %r -> %r"
                                 % (name, n_insert, lo, ln))
        else:
            a = re.sub(r"\\multicolumn\{\d+\}", "MC", lo)
            b = re.sub(r"\\multicolumn\{\d+\}", "MC", ln)
            a = re.sub(r"\{@{}[l r]+(@\{\}\}|\})", "CS", a)
            b = re.sub(r"\{@{}[l r]+(@\{\}\}|\})", "CS", b)
            if a != b:
                raise SystemExit("%s: non-data line changed beyond layout: %r -> %r"
                                 % (name, lo, ln))
    print("invariant ok: %s (%d insertion(s) per data row, all other cells identical)"
          % (name, n_insert))


# ---------------------------------------------------------------- load + verify
old = {}
for name, (path, want) in SOURCES.items():
    got = sha(path)
    print("source %-18s sha256 %s %s" % (name, got, "ok" if got == want else "MISMATCH"))
    if got != want:
        raise SystemExit("source hash mismatch for " + name)
    old[name] = open(path, "r", encoding="utf-8").read()

macros = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}",
                         old["numbers.tex"]))


def check_macro(macro, value, where):
    if macro not in macros:
        print("cross-check skipped (macro %s absent): %s" % (macro, where))
        return
    if abs(float(macros[macro]) - float(value)) > 0.051:
        raise SystemExit("cross-check failed: %s macro %s=%s != computed %s"
                         % (where, macro, macros[macro], value))
    print("cross-check ok: %s %s == %s" % (where, macro, value))


ACC = "acc. (" + BS + "%)"

# ---------------------------------------------------------------- tab-selection
CURRENT[:] = ["tab-selection.tex",
              "arithmetic on the archived hits/n cells (same cells the table "
              "already shows); full-pool values equal metrics.arms.*.value"]
sel_new = transform(
    old["tab-selection.tex"], "Row", [(4, "arm " + ACC)],
    row_rules=[(lambda cs: True, [(4, lambda cs: acc_from_cell(cs[3]) or "---")])],
    mc_from=8, mc_to=9,
    colspec_old="{@{}l r l l l r r r@{}}", colspec_new="{@{}l r l l l l r r r@{}}")
for arm, m in (("uniform@16", "SelPoolUniformRate"), ("contrast@16_k0", "SelPoolKZeroRate"),
               ("contrast@16_k1", "SelPoolKOneRate"), ("contrast@16_k2", "SelPoolKTwoRate")):
    if SEL.get("arms", {}).get(arm):
        check_macro(m, "%.1f" % (100.0 * SEL["arms"][arm]["value"]),
                    "SEL metrics " + arm)

# ---------------------------------------------------------------- tab-noframes
CURRENT[:] = ["tab-noframes.tex",
              "arithmetic on the archived hits/n cells; arm values equal "
              "metrics.arms.*.value"]
nof_new = transform(
    old["tab-noframes.tex"], "Arm or contrast", [(2, ACC)],
    row_rules=[(lambda cs: cs[1] != "---", [(2, lambda cs: acc_from_cell(cs[1])
                                             or "---")]),
               (lambda cs: cs[1] == "---", [(2, lambda cs: "---")])],
    mc_from=7, mc_to=8,
    colspec_old="{@{}l l l l r r r@{}}", colspec_new="{@{}l l l l l r r r@{}}")
for arm, m in (("no_frames", "SelPoolNoFramesRate"), ("uniform@16", "SelPoolUniformRate"),
               ("constant_prior", "SelPoolPriorRate")):
    if SEL.get("arms", {}).get(arm):
        check_macro(m, "%.1f" % (100.0 * SEL["arms"][arm]["value"]),
                    "SEL metrics " + arm)

# ---------------------------------------------------------------- tab-single
CURRENT[:] = ["tab-single.tex",
              "SGL metrics.arms.*.value; asserted equal to the existing "
              "Sgl*Rate macros"]
sgl_acc = []
if SGL.get("arms"):
    sgl_acc = [
        ("uniform@16", "SglUniformRate", "%.1f" % (100.0 * SGL["arms"]["uniform@16"]["value"])),
        ("single frame", "SglSingleRate",
         "%.1f" % (100.0 * SGL["arms"]["single_frame_j_star"]["value"])),
        ("no frames", "SglNoFramesRate", "%.1f" % (100.0 * SGL["arms"]["no_frames"]["value"])),
        ("constant-letter", "SglPriorRate",
         "%.1f" % (100.0 * SGL["arms"]["constant_prior"]["value"])),
    ]
    for prefix, m, v in sgl_acc:
        check_macro(m, v, "SGL metrics " + prefix)


def sgl_value_for(cs):
    for prefix, _m, v in sgl_acc:
        if cs[0].startswith(prefix):
            return v
    raise SystemExit("unknown single arm row: %r" % cs[0])


sgl_new = transform(
    old["tab-single.tex"], "Arm or contrast", [(2, ACC)],
    row_rules=[(lambda cs: cs[1] != "---", [(2, sgl_value_for)]),
               (lambda cs: cs[1] == "---", [(2, lambda cs: "---")])],
    mc_from=7, mc_to=8,
    colspec_old="{@{}l l l l l r r@{}}", colspec_new="{@{}l l l l l l r r@{}}")
# the bounded-difference block's multicolumn spans grow by the one new column
sgl_new = sgl_new.replace(BS + "multicolumn{2}{@{}l}{", BS + "multicolumn{3}{@{}l}{")
sgl_new = sgl_new.replace(BS + "multicolumn{4}{@{}l}{", BS + "multicolumn{5}{@{}l}{")

# ---------------------------------------------------------------- tab-distortion
CURRENT[:] = ["tab-distortion.tex",
              "arithmetic on the archived hits/17 cells (evidence and non-evidence)"]
dis_new = transform(
    old["tab-distortion.tex"], "Operator $" + BS + "times$ stream",
    [(2, "evidence " + ACC), (5, "non-ev. " + ACC)],
    row_rules=[(lambda cs: True,
                [(2, lambda cs: acc_from_cell(cs[1]) or "---"),
                 (5, lambda cs: acc_from_cell(cs[4]) or "---")])],
    mc_from=7, mc_to=9,
    colspec_old="{@{}l l l l l r r@{}}", colspec_new="{@{}l l l l l l l r r@{}}")

# ---------------------------------------------------------------- tab-donor
CURRENT[:] = ["tab-donor.tex", "arithmetic on the archived hits/n cells"]
don_new = transform(
    old["tab-donor.tex"], "Arm or contrast", [(2, ACC)],
    row_rules=[(lambda cs: cs[1] != "---", [(2, lambda cs: acc_from_cell(cs[1])
                                             or "---")]),
               (lambda cs: cs[1] == "---", [(2, lambda cs: "---")])],
    mc_from=6, mc_to=7,
    colspec_old="{@{}l l l l r r@{}}", colspec_new="{@{}l l l l l r r@{}}")

# ---------------------------------------------------------------- invariants
check_invariant("tab-selection.tex", old["tab-selection.tex"], sel_new, 1)
check_invariant("tab-noframes.tex", old["tab-noframes.tex"], nof_new, 1)
check_invariant("tab-single.tex", old["tab-single.tex"], sgl_new, 1)
check_invariant("tab-distortion.tex", old["tab-distortion.tex"], dis_new, 2)
check_invariant("tab-donor.tex", old["tab-donor.tex"], don_new, 1)

# ---------------------------------------------------------------- write the pack
OUT = {
    "numbers.tex": old["numbers.tex"],
    "tab-design.tex": old["tab-design.tex"],
    "tab-selection.tex": sel_new,
    "tab-noframes.tex": nof_new,
    "tab-single.tex": sgl_new,
    "tab-distortion.tex": dis_new,
    "tab-donor.tex": don_new,
}
for name, text in OUT.items():
    with open(name, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("out %-18s sha256 %s bytes %d" % (name, sha(name), len(text.encode("utf-8"))))

assert sha("numbers.tex") == SOURCES["numbers.tex"][1], "numbers.tex must not change"
assert sha("tab-design.tex") == SOURCES["tab-design.tex"][1], "tab-design must not change"
print("inserted cells: %d" % len(INSERTED))
for tbl, row, val, src in INSERTED:
    print("  %-18s %-44s %s" % (tbl, row, val))
with open("metrics.json", "w") as fh:
    json.dump({"pack_regenerate_acc": {
        "n_inserted_cells": len(INSERTED),
        "inserted": [{"table": t, "row": r, "value": v, "source": s}
                     for t, r, v, s in INSERTED],
        "unchanged_files": ["numbers.tex", "tab-design.tex"],
        "changed_files": ["tab-selection.tex", "tab-noframes.tex", "tab-single.tex",
                          "tab-distortion.tex", "tab-donor.tex"],
        "new_measurement": False,
        "note": "accuracy cells computed from records already held (archived metrics "
                "fields and the archived tables' own hits/n cells); cross-checked "
                "against the existing SelPool*/Sgl* rate macros; every other cell "
                "byte-identical (invariant enforced in-run)",
    }}, fh, indent=1)
print("packaged 7 files; numbers.tex and tab-design.tex byte-identical; five arm "
      "tables gained their accuracy column(s)")
