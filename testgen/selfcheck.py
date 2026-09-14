"""Machinery self-check against the strategy document's expected results.

This is NOT test generation: the worked examples (Add/Div/Relu/Abs) have
no specification files and are never emitted into tests/. They exist only
to validate that the p'/p'' enumeration reproduces the results stated in
the strategy document:

- Add on uint8 must yield exactly 9 classes: 4 vertices, 4 open edges,
  1 interior;
- Div on int8 must keep the domain hole (y != 0) and the hole-adjacent
  boundary classes must be satisfiable;
- Relu/Abs must yield exactly one class per branch of the piecewise
  definition.

Run:  python -m testgen.selfcheck
"""

from __future__ import annotations

import os
import sys

from .classes import enumerate_classes
from .dsl import FloatVar, Operator, fgt, nan_sibling_violations
from .evaluate import verify_witness
from .operators import CATALOG
from .worked_examples import WORKED_EXAMPLES


def _check(ok: bool, msg: str, failures: list) -> None:
    print(("  OK  " if ok else "  FAIL") + f" {msg}")
    if not ok:
        failures.append(msg)


def _verify_all(op, result, failures) -> None:
    for cls in result.classes:
        bad = verify_witness(op.variables, op.domain,
                             cls.predicates, cls.witness)
        _check(not bad, f"{op.name}/{cls.id} witness satisfies its class",
               failures)
    for cls in result.branches:
        bad = verify_witness(op.variables,
                             list(op.domain) + list(op.constraints),
                             cls.predicates, cls.witness)
        _check(not bad, f"{op.name}/{cls.id} witness satisfies its class",
               failures)


def main() -> int:
    failures: list = []
    ops = {op.name: op for op in WORKED_EXAMPLES}

    # -- Add uint8: 9 classes, corners/edges/interior --------------------
    op = ops["Add_uint8"]
    r = enumerate_classes(op)
    print(f"Add_uint8: {len(r.classes)} classes")
    _check(len(r.classes) == 9, "Add_uint8 yields exactly 9 classes "
           "(strategy: 4 vertices + 4 edges + 1 interior)", failures)
    corners = {(0, 0), (0, 255), (255, 0), (255, 255)}
    got = {(c.witness["x1"], c.witness["x2"]) for c in r.classes}
    _check(corners <= got, "all 4 corners are witnessed", failures)
    _check(len(r.unsat) == 7, "7 of the 16 sign combinations are UNSAT",
           failures)
    _verify_all(op, r, failures)

    # -- Add uint8 with the output-type constraint: domain shrinks --------
    op = ops["Add_uint8_no_overflow"]
    r2 = enumerate_classes(op)
    print(f"Add_uint8_no_overflow: {len(r2.classes)} classes")
    _check(all(c.witness["x1"] + c.witness["x2"] <= 255 for c in r2.classes),
           "no witness overflows uint8", failures)
    _check((255, 255) not in {(c.witness["x1"], c.witness["x2"])
                              for c in r2.classes},
           "the (255,255) corner leaves the valid domain", failures)
    _check(len(r2.classes) != 9,
           "the output-type constraint changes the class inventory", failures)
    _verify_all(op, r2, failures)

    # -- Div int8: domain hole respected ----------------------------------
    op = ops["Div_int8"]
    r3 = enumerate_classes(op)
    print(f"Div_int8: {len(r3.classes)} classes, "
          f"{len(r3.branches)} auxiliary")
    _check(all(c.witness["y"] != 0 for c in r3.classes),
           "no witness divides by zero", failures)
    bd = {b.name: b.witness["y"] for b in r3.branches}
    _check(bd.get("divisor adjacent to the domain hole, below (y = -1)")
           == -1, "hole-adjacent class below (y = -1) exists", failures)
    _check(bd.get("divisor adjacent to the domain hole, above (y = 1)")
           == 1, "hole-adjacent class above (y = 1) exists", failures)
    _verify_all(op, r3, failures)

    # -- Relu / Abs: one class per branch ----------------------------------
    for name, n_branches in (("Relu_f32", 7), ("Abs_f32", 7)):
        op = ops[name]
        r = enumerate_classes(op)
        print(f"{name}: {len(r.branches)} branch classes")
        _check(len(r.branches) == n_branches,
               f"{name} yields one class per spec branch ({n_branches})",
               failures)
        _check(len(r.unsat) == 0, f"{name}: all branches satisfiable",
               failures)
        _verify_all(op, r, failures)

    # -- MaxPool: spec-backed operator must exist with its spec file ------
    mp = CATALOG["MaxPool"]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _check(os.path.isfile(os.path.join(root, mp.spec_file)),
           "MaxPool spec file present (specs/maxpool.md)", failures)

    # -- NaN sibling-predicate validation ---------------------------------
    # Every operator using ordered float comparisons must carry an explicit
    # NaN class for the same variable (strategy caution), or record the
    # NaN specification gap. The worked examples all comply.
    for op in list(ops.values()) + [mp]:
        v = nan_sibling_violations(op)
        _check(not v,
               f"{op.name}: no NaN sibling-predicate violation"
               + (f" (got: {v[0]})" if v else ""),
               failures)

    # A definition that *should* be flagged: an ordered float predicate
    # over a variable with no NaN class and no recorded gap.
    _bx = FloatVar("X", 32, role="scalar operand (float32)")
    bad_op = Operator(
        name="Bad_f32", spec_file="specs/bad.md", spec_ref="selfcheck",
        variables=[_bx], domain=[], constraints=[fgt(_bx, 0)],
    )
    v = nan_sibling_violations(bad_op)
    _check(len(v) == 1, "NaN sibling check flags an ordered float "
           "predicate without an explicit NaN class", failures)

    # ...and the same definition passes once the gap is recorded.
    bad_op.params["nan_spec_gaps"] = {"X": "spec leaves NaN undefined"}
    _check(not nan_sibling_violations(bad_op),
           "a recorded NaN specification gap silences the diagnostic",
           failures)

    print()
    if failures:
        print(f"SELFCHECK FAILED ({len(failures)} failure(s))")
        return 1
    print("SELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())