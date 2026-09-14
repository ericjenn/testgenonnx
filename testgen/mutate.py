"""Mutation testing harness (strategy: "Validating pruning decisions via
mutation testing").

Every ``pruned`` / ``covered-by`` record and every class-retention
decision is a documented belief: it assumes an implementation pattern is
absent. This module *checks* the belief instead of only asserting it: a
set of plausible implementation faults (mutants) is injected into the
reference implementation, and the retained test set of every data type is
run against each mutant. A mutant that survives (produces the reference
output on every retained test) is evidence that either the pruning was too
aggressive or that a ``covered-by`` disposition does not actually cover
what it claims.

The runner is generic; the mutant set is operator-specific by design (it
mutates the operator's reference implementation), mirroring
``reference.REFERENCES``. Run via ``python -m testgen.cli --mutate`` after
generation; results are merged into each data type's manifest.json and
appended to its report.md.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .materialize import FLOAT_DTYPES


# ---------------------------------------------------------------------------
# Mutated reference implementation (MaxPool)
# ---------------------------------------------------------------------------


def _mutated_maxpool(X, dX2, dX3, dW0, dW1, s0, s1, d0, d1, p0, p1, p2, p3,
                     dY2, dY3, bug):
    """The reference window scan with exactly one deliberate fault.

    Bug points (one per mutant):
      - "index-off-by-one":   Indices shifted by one element
      - "tie-break-last":     last-encountered max wins (spec: first/lowest)
      - "swap-strides-dilations": strides and dilations swapped in the
                              window-position computation
      - "ceil-output-size":   output size rounded up (ceil_mode=1) instead
                              of the spec's floor formula
      - "undilated-kernel-extent": the output-size formula uses the raw
                              kernel shape, ignoring dilation
    """
    K0 = d0 * (dW0 - 1) + 1
    K1 = d1 * (dW1 - 1) + 1
    S0 = dX2 + p0 + p2
    S1 = dX3 + p1 + p3
    if bug == "ceil-output-size":
        dY2 = -(-(S0 - K0) // s0) + 1
        dY3 = -(-(S1 - K1) // s1) + 1
    elif bug == "undilated-kernel-extent":
        dY2 = (S0 - dW0) // s0 + 1
        dY3 = (S1 - dW1) // s1 + 1
    Y, I = [], []
    for m in range(dY2):
        row_y, row_i = [], []
        for n in range(dY3):
            best = None
            best_idx = None
            for h in range(dW0):
                for w in range(dW1):
                    if bug == "swap-strides-dilations":
                        r_p = m * d0 + h * s0
                        c_p = n * d1 + w * s1
                    else:
                        r_p = m * s0 + h * d0
                        c_p = n * s1 + w * d1
                    r = r_p - p0
                    c = c_p - p1
                    if 0 <= r < dX2 and 0 <= c < dX3:
                        v = X[r][c]
                        if bug == "tie-break-last":
                            better = best is None or v >= best
                        else:
                            better = best is None or v > best
                        if better:
                            best = v
                            best_idx = r * dX3 + c
                            if bug == "index-off-by-one":
                                best_idx += 1
            if best is None:
                # A mutant can push a window fully into the padding (e.g.
                # the ceil-output-size bug): the mutant has no output for
                # this test, which counts as detected.
                raise AssertionError("window fully inside padding")
            row_y.append(best)
            row_i.append(best_idx)
        Y.append(row_y)
        I.append(row_i)
    return Y, I


# ---------------------------------------------------------------------------
# Mutant registry (operator-specific, like reference.REFERENCES)
# ---------------------------------------------------------------------------

MUTANTS = [
    {
        "id": "MUT-01",
        "name": "Indices off-by-one",
        "bug": "index-off-by-one",
        "description": ("every returned index is shifted one element past "
                        "its input position"),
        "probes": [
            "control: must be killed by any test (index observability, "
            "template T1)",
        ],
    },
    {
        "id": "MUT-02",
        "name": "tie-break: last-encountered max wins",
        "bug": "tie-break-last",
        "description": ("the spec's tie-break (lowest h, then w over the "
                        "input) is replaced by the opposite preference"),
        "probes": [
            "retention of the tie-break templates T2/T4 on the auxiliary "
            "cases (pruning discipline restricting T2-T4 to auxiliary "
            "cases)",
            "dependency claim 'Output index vs. special/tied values'",
        ],
    },
    {
        "id": "MUT-03",
        "name": "strides and dilations swapped in window positioning",
        "bug": "swap-strides-dilations",
        "description": ("window origins advance by dilations and sample "
                        "offsets step by strides, instead of the reverse"),
        "probes": [
            "coverage of configurations with strides != dilations on an "
            "axis (equivalence classes + the dilation boundary case)",
        ],
    },
    {
        "id": "MUT-04",
        "name": "output-size rounding: ceil instead of floor",
        "bug": "ceil-output-size",
        "description": ("the output-size formula rounds up (ceil_mode=1) "
                        "instead of the spec's floor (restriction R4)"),
        "probes": [
            "Y-C1 boundary classes (fits/overflows) and restriction R4",
        ],
    },
    {
        "id": "MUT-05",
        "name": "kernel extent ignores dilation in the output-size formula",
        "bug": "undilated-kernel-extent",
        "description": ("the output size is computed with the raw kernel "
                        "shape instead of the dilated kernel extent"),
        "probes": [
            "dilation boundary case (d0 = 2) and the dilation-2 "
            "equivalence classes",
        ],
    },
]

MUTANT_SETS = {"MaxPool": MUTANTS}


# ---------------------------------------------------------------------------
# Generic runner
# ---------------------------------------------------------------------------


def _parse_value(v, dtype):
    if isinstance(v, str):
        return float(v.replace("+inf", "inf"))
    if dtype in FLOAT_DTYPES:
        return float(v)
    return int(v)


def _killed_by(mut_bug: str, test: Dict[str, object],
               dtype: str, pool_fn) -> Optional[str]:
    """Run one mutant against one test; return the test id when killed.

    The mutant is killed when its output differs from the expected
    (Y, Indices, shape) computed by the un-mutated reference, or when it
    fails to produce an output at all.
    """
    attrs = test["attributes"]
    w = test["input"]
    dX2, dX3 = w["shape"][2], w["shape"][3]
    X = [[_parse_value(v, dtype) for v in row] for row in w["values"]]
    dY2, dY3 = test["expected"]["shape"][2], test["expected"]["shape"][3]
    args = (X, dX2, dX3, attrs["kernel_shape"][0], attrs["kernel_shape"][1],
            attrs["strides"][0], attrs["strides"][1],
            attrs["dilations"][0], attrs["dilations"][1],
            attrs["pads"][0], attrs["pads"][1], attrs["pads"][2],
            attrs["pads"][3], dY2, dY3)
    try:
        Y, I = pool_fn(*args, bug=mut_bug)
    except Exception:
        return test["id"]
    if (len(Y), len(Y[0]) if Y else 0) != (dY2, dY3):
        return test["id"]
    exp_y, exp_i = test["expected"]["Y"], test["expected"]["Indices"]
    for r in range(dY2):
        for c in range(dY3):
            if _parse_value(exp_y[r][c], dtype) != Y[r][c]:
                return test["id"]
            if exp_i[r][c] != I[r][c]:
                return test["id"]
    return None


def run_for_dtype(mutants: List[Dict[str, object]], out_dir: str,
                  dtype: str) -> Dict[str, object]:
    """Run every mutant against every retained test of one data type."""
    tests = []
    for fname in sorted(os.listdir(out_dir)):
        if not fname.endswith(".json") or fname == "manifest.json":
            continue
        with open(os.path.join(out_dir, fname)) as fh:
            t = json.load(fh)
        if t.get("operator") and t.get("dtype") == dtype:
            tests.append(t)

    results = []
    for mut in mutants:
        killed: List[str] = []
        for t in tests:
            k = _killed_by(mut["bug"], t, dtype, _mutated_maxpool)
            if k is not None:
                killed.append(k)
        results.append({
            "id": mut["id"],
            "name": mut["name"],
            "description": mut["description"],
            "probes": mut["probes"],
            "survived": not killed,
            "killed_by_sample": killed[:3],
            "n_tests": len(tests),
        })
    return {
        "dtype": dtype,
        "tests": len(tests),
        "mutants": results,
        "survivors": [r["id"] for r in results if r["survived"]],
    }


def _report_section(per_dtype: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    a = lines.append
    a("## Mutation testing (pruning validation)\n")
    a("Plausible implementation faults are injected into the reference "
      "implementation and the retained test set is run against each "
      "mutant (strategy: 'Validating pruning decisions via mutation "
      "testing'). A surviving mutant is evidence that a pruning or "
      "covered-by record (or a retention decision) was too "
      "aggressive.\n")
    for res in per_dtype:
        a(f"### Data type: {res['dtype']} ({res['tests']} tests)\n")
        rows = []
        for r in res["mutants"]:
            sample = ", ".join(r["killed_by_sample"]) or "-"
            status = "**SURVIVED**" if r["survived"] else "killed"
            rows.append([r["id"], r["description"], status, sample])
        a("| mutant | fault | status | killed by (sample) |\n"
          "|---|---|---|---|")
        for row in rows:
            a("| " + " | ".join(str(c) for c in row) + " |")
        a("")
        survivors = res["survivors"]
        if survivors:
            for r in res["mutants"]:
                if r["survived"]:
                    a(f"**{r['id']} survived** -- invalidates or "
                      f"questions: {'; '.join(r['probes'])}.")
                    a("")
        else:
            a("All mutants killed: no pruning or retention decision is "
              "invalidated by this mutant set (the set is illustrative, "
              "not exhaustive).\n")
    return "\n".join(lines) + "\n"


def mutation_analysis(out_dirs: List[str]) -> Dict[str, Dict[str, object]]:
    """Run the mutation analysis for each emitted dtype directory and
    merge the results into its manifest.json / report.md. Returns a
    mapping {out_dir: result}.
    """
    all_results: Dict[str, Dict[str, object]] = {}
    for out_dir in out_dirs:
        manifest_path = os.path.join(out_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        op_name, dtype = manifest["operator"], manifest["dtype"]
        mutants = MUTANT_SETS.get(op_name)
        if mutants is None:
            continue
        res = run_for_dtype(mutants, out_dir, dtype)
        all_results[out_dir] = res

        manifest["mutation_testing"] = res
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")

        report_path = os.path.join(out_dir, "report.md")
        if os.path.isfile(report_path):
            with open(report_path) as fh:
                report = fh.read()
            report = report.split("\n## Mutation testing")[0]
            with open(report_path, "w") as fh:
                fh.write(report + _report_section([res]))
    return all_results