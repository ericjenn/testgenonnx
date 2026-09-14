"""Functional test materialization for MaxPool.

For every equivalence class / auxiliary case of the structural constraint
system, and for every data type supported by the spec, this module
materializes concrete input tensors and computes the expected outputs
(Y, Indices) with the reference implementation of the informal
specification (reference.maxpool_ref).

Value templates (the value-level dimension of the strategy):

- T1 "distinct": every element distinct and derived from its position
  (index observability, strategy: "On values and indexes...") -- applied
  to every structural class.
- T2 "all-ties": every element equal, so every window max is tied and
  Indices is fully determined by the spec's tie-break rule.
- T3 "pad-constant": the whole tensor is filled with the *padding
  constant* (-inf / minint8 / 0). The max of windows at the padding
  boundary equals the padding value yet must still be reported as coming
  from X: this is exactly the configuration where the spec documents
  ONNX Runtime non-compliances (Y wrong AND Indices pointing into the
  padding), and the combined boundary case required by the strategy's
  cross-domain dependency check.
- T4 "duplicate maxima": the same max value placed at the first and last
  position of the first window, testing the tie-break between window
  edge positions.

T2/T3/T4 are applied to the auxiliary (boundary/property) cases only:
the structural interior classes add no expected value-level fault, and
the restriction is recorded in the report (pruning discipline).
"""

from __future__ import annotations

from typing import Dict, List

from .reference import PAD_VALUES, maxpool_output_shape, maxpool_ref

FLOAT_DTYPES = ("float16", "float", "double")

TEMPLATES = [
    {"id": "T1", "name": "distinct values (index observability)",
     "applies_to": "all",
     "note": "every element value is distinct and derived from its "
             "position, so each expected output identifies its source "
             "input position (index observability)"},
    {"id": "T2", "name": "all-ties (tie-break of Indices)",
     "applies_to": "auxiliary",
     "note": "every window max is tied, so Indices is fully determined "
             "by the spec's tie-break rule"},
    {"id": "T3", "name": "pad-constant tensor (padding/max interplay)",
     "applies_to": "auxiliary",
     "note": "the whole tensor equals the padding constant: windows at "
             "the padding boundary must still report a max from X -- the "
             "configuration where the spec documents ONNX Runtime "
             "non-compliances"},
    {"id": "T4", "name": "duplicate maxima in the first window",
     "applies_to": "auxiliary",
     "note": "the same max value at the first and last covered position "
             "of the first window tests the tie-break between window "
             "edge positions"},
]


def _as_dtype(v, dtype):
    if dtype in FLOAT_DTYPES:
        return float(v)
    return int(v)


def _render(v, dtype):
    """JSON-safe rendering (floats: -inf -> "-inf"; ints as-is)."""
    if dtype in FLOAT_DTYPES:
        if v == float("-inf"):
            return "-inf"
        if v == float("inf"):
            return "+inf"
        if v == 0.0:
            return 0.0
        return v
    return int(v)


def _window_positions(w, m, n, dX2, dX3):
    """Positions of X selected by window (m, n), row-major scan order
    (h increasing, then w increasing -- the spec's tie-break order)."""
    pos = []
    for h in range(w["dW0"]):
        for ww in range(w["dW1"]):
            r = m * w["s0"] + h * w["d0"] - w["p0"]
            c = n * w["s1"] + ww * w["d1"] - w["p1"]
            if 0 <= r < dX2 and 0 <= c < dX3:
                pos.append((r, c))
    return pos


def _first_window_positions(w, dX2, dX3):
    """Positions of X covered by the first window, row-major scan order."""
    return _window_positions(w, 0, 0, dX2, dX3)


def padding_only_windows(w):
    """Windows that select no element of X at all.

    Possible under dilation >= 2 even when the spec's pads-C2 holds
    (e.g. dX3 = 1, dilations[1] = 2: the dilated kernel samples columns
    0 and 2 of the padded tensor while X lives in column 1): neither Y
    nor Indices is defined by the spec for such configurations. They
    are reported as specification gaps instead of being materialized.
    """
    bad = []
    for m in range(w["dY2"]):
        for n in range(w["dY3"]):
            if not _window_positions(w, m, n, w["dX2"], w["dX3"]):
                bad.append((m, n))
    return bad


def _tensor_T1(w, dtype, dX2, dX3):
    """Distinct values derived from position (index observability)."""
    return [[_as_dtype(r * dX3 + c + 1, dtype) for c in range(dX3)]
            for r in range(dX2)]


def _tensor_T2(w, dtype, dX2, dX3):
    v = _as_dtype(5, dtype)
    return [[v for _ in range(dX3)] for _ in range(dX2)]


def _tensor_T3(w, dtype, dX2, dX3):
    """Whole tensor equal to the padding constant."""
    v = _as_dtype(PAD_VALUES[dtype], dtype)
    return [[v for _ in range(dX3)] for _ in range(dX2)]


def _tensor_T4(w, dtype, dX2, dX3):
    """Duplicate maxima at the first and last covered position of the
    first window; everything else distinct and lower."""
    pos = _first_window_positions(w, dX2, dX3)
    if len(pos) < 2:
        return None  # single-element windows: no tie is constructible
    X = [[_as_dtype(r * dX3 + c + 1, dtype) for c in range(dX3)]
         for r in range(dX2)]
    hi = _as_dtype(100, dtype)
    for (r, c) in (pos[0], pos[-1]):
        X[r][c] = hi
    return X


_TENSORS = {"T1": _tensor_T1, "T2": _tensor_T2, "T3": _tensor_T3,
            "T4": _tensor_T4}


def functional_tests(case_id: str, kind: str, name: str, traceability,
                     predicates_rendered, witness: Dict[str, object],
                     spec_ref: str, dtype: str):
    """Materialize the functional test cases for one structural case.

    Returns (tests, gap) where tests is a list of 0..4 test case
    documents (templates not applicable to the case are skipped, e.g.
    T4 on single-element windows) and gap is a specification-gap record
    when materialization was refused (see padding_only_windows), else
    None.
    """
    w = witness
    dX2, dX3 = w["dX2"], w["dX3"]
    is_aux = kind != "equivalence-class"
    out: List[Dict[str, object]] = []

    gap_windows = padding_only_windows(w)
    if gap_windows:
        gap = {
            "case_id": case_id,
            "kind": kind,
            "class": name,
            "inputs": {k: w[k] for k in sorted(w)},
            "windows": [list(g) for g in gap_windows],
            "reason": (
                "window(s) select only padding elements although pads-C2 "
                "holds: under dilation >= 2 the spec's rationale for "
                "pads-C2 ('guarantees that the max value returned belongs "
                "to X') is violated and neither Y nor Indices is defined "
                "by the specification -- functional test refused pending "
                "spec clarification"
            ),
        }
        return out, gap

    dY2_f, dY3_f = maxpool_output_shape(
        dX2, dX3, w["dW0"], w["dW1"], w["s0"], w["s1"], w["d0"], w["d1"],
        w["p0"], w["p1"], w["p2"], w["p3"])
    if (dY2_f, dY3_f) != (w["dY2"], w["dY3"]):
        raise AssertionError(
            f"{case_id}: witness output size {(w['dY2'], w['dY3'])} "
            f"disagrees with the spec formula {(dY2_f, dY3_f)}")

    for tpl in TEMPLATES:
        if tpl["applies_to"] == "auxiliary" and not is_aux:
            continue
        X = _TENSORS[tpl["id"]](w, dtype, dX2, dX3)
        if X is None:
            continue
        Y, I = maxpool_ref(
            X, dX2, dX3, w["dW0"], w["dW1"], w["s0"], w["s1"], w["d0"],
            w["d1"], w["p0"], w["p1"], w["p2"], w["p3"],
            PAD_VALUES[dtype], w["dY2"], w["dY3"])
        out.append({
            "id": f"MaxPool_{dtype}_{case_id}_{tpl['id']}",
            "operator": "MaxPool",
            "dtype": dtype,
            "spec_ref": spec_ref,
            "kind": kind,
            "class_id": case_id,
            "class": name,
            "class_predicates": predicates_rendered,
            "traceability": traceability,
            "value_template": {"id": tpl["id"], "name": tpl["name"]},
            "attributes": {
                "kernel_shape": [w["dW0"], w["dW1"]],
                "strides": [w["s0"], w["s1"]],
                "dilations": [w["d0"], w["d1"]],
                "pads": [w["p0"], w["p1"], w["p2"], w["p3"]],
                "auto_pad": "NOTSET",
                "ceil_mode": 0,
                "storage_order": 0,
            },
            "input": {
                "shape": [1, 1, dX2, dX3],
                "values": [[_render(v, dtype) for v in row] for row in X],
            },
            "expected": {
                "shape": [1, 1, w["dY2"], w["dY3"]],
                "Y": [[_render(v, dtype) for v in row] for row in Y],
                "Indices": I,
            },
        })
    return out, None