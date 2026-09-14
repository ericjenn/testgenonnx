"""Reference implementations (test oracle) for the catalog operators.

These are direct transcriptions of the informal specification definitions,
kept deliberately naive and independent of any production implementation.
They compute the expected output carried by each generated test case
(strategy: "the SONNX reference implementation as the test oracle").
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .domains import NAN, ValueClass


def _num(v) -> float:
    """Underlying numeric value of a witness entry (NaN sentinel -> nan)."""
    if isinstance(v, ValueClass):
        v = v.value
    if v is NAN:
        return float("nan")
    return float(v)


def add_uint8(w: Dict[str, object]) -> Optional[Dict[str, object]]:
    return {"y": w["x1"] + w["x2"]}


def add_uint8_no_overflow(w: Dict[str, object]) -> Optional[Dict[str, object]]:
    # The overflow constraint guarantees the sum is representable.
    return {"y": w["x1"] + w["x2"]}


def div_int8(w: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Integer division, truncation toward zero (C semantics, as in ONNX)."""
    x, y = w["x"], w["y"]
    q = abs(x) // abs(y)
    return {"y": q if (x >= 0) == (y >= 0) else -q}


def relu_f32(w: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Y = X for X > 0, +0 otherwise; NaN propagates, -0 maps to +0."""
    x = _num(w["X"])
    if math.isnan(x):
        return {"Y": "NaN"}
    if x > 0.0:
        return {"Y": _render_float(x)}
    return {"Y": _render_float(0.0)}


def abs_f32(w: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Y = |X| with the sign cases of the piecewise definition:
    NaN -> NaN, ±inf -> +inf, ±0 -> +0, else the absolute value."""
    x = _num(w["X"])
    if math.isnan(x):
        return {"Y": "NaN"}
    if math.isinf(x):
        return {"Y": "+inf"}
    if x == 0.0:
        return {"Y": _render_float(0.0)}
    return {"Y": _render_float(abs(x))}


def maxpool_output_shape(dX2, dX3, dW0, dW1, s0, s1, d0, d1, p0, p1, p2, p3):
    """Spec Y-C1 output-size formula (ceil_mode = 0, restriction R4):
    dY_i = floor((dX_(i+2) + pads[i] + pads[i+2] - Ke_i) / strides[i]) + 1.

    Python ``//`` is floor division, matching the spec exactly. The
    formula can yield 0 (empty output when the kernel barely does not
    fit); a *negative* result is an invalid configuration -- the
    constraint system never admits it, so it is rejected here.
    """
    K0 = d0 * (dW0 - 1) + 1
    K1 = d1 * (dW1 - 1) + 1
    S0 = dX2 + p0 + p2
    S1 = dX3 + p1 + p3
    dY2 = (S0 - K0) // s0 + 1
    dY3 = (S1 - K1) // s1 + 1
    if dY2 < 0 or dY3 < 0:
        raise ValueError("invalid configuration: negative output size")
    return dY2, dY3


def maxpool_ref(X, dX2, dX3, dW0, dW1, s0, s1, d0, d1, p0, p1, p2, p3,
                pad_value, dY2=None, dY3=None):
    """Reference implementation of the SONNX MaxPool informal specification.

    X is a list of dX2 rows, each a list of dX3 values (batch and channel
    are 1 per restriction R1 / the materializer's pinning). Returns
    (Y, Indices) with:

    - Y[b,c,m,n] = max over the dilated window of the *padded* tensor
      X_p (padding constant per dtype: -inf for floats, minint8 for
      int8, 0 for uint8, per the spec's float/int sections);
    - Indices[b,c,m,n] = flatten (row-major) index into X of the max,
      tie-broken by lowest h*dX3 + w (spec informal definition).

    The implementation scans windows in the padded coordinate system and
    maps each selected element back to its position in X; elements in the
    padding never qualify for the max (the spec's pads-C2 guarantees each
    window contains at least one element of X).
    """
    if dY2 is None or dY3 is None:
        dY2, dY3 = maxpool_output_shape(dX2, dX3, dW0, dW1, s0, s1,
                                        d0, d1, p0, p1, p2, p3)
    Y = []
    I = []
    for m in range(dY2):
        row_y, row_i = [], []
        for n in range(dY3):
            best = None
            best_idx = None
            for h in range(dW0):
                for w in range(dW1):
                    r_p = m * s0 + h * d0      # row in padded coordinates
                    c_p = n * s1 + w * d1      # col in padded coordinates
                    r = r_p - p0               # row in X
                    c = c_p - p1               # col in X
                    if 0 <= r < dX2 and 0 <= c < dX3:
                        v = X[r][c]
                        # strict >: the first encountered max wins, which
                        # realizes the spec's tie-break (lowest h, then w,
                        # i.e. the row-major minimum of h*dX3 + w).
                        if best is None or v > best:
                            best = v
                            best_idx = r * dX3 + c
            if best is None:
                raise AssertionError(
                    "window fully inside padding: pads-C2 violated")
            row_y.append(best)
            row_i.append(best_idx)
        Y.append(row_y)
        I.append(row_i)
    return Y, I


def _render_float(v: float) -> object:
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "+inf" if v > 0 else "-inf"
    if v == 0.0 and math.copysign(1.0, v) < 0:
        return "-0"
    return v


REFERENCES = {
    "Add_uint8": add_uint8,
    "Add_uint8_no_overflow": add_uint8_no_overflow,
    "Div_int8": div_int8,
    "Relu_f32": relu_f32,
    "Abs_f32": abs_f32,
}

# Padding constant per data type (spec MaxPool: "-Inf" for floats,
# "minint8" for int8, "0" for uint8).
PAD_VALUES = {
    "float16": float("-inf"),
    "float": float("-inf"),
    "double": float("-inf"),
    "int8": -128,
    "uint8": 0,
}


def reference_for(op_name: str):
    return REFERENCES.get(op_name)