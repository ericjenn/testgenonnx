"""Type-specific value domains (strategy section "Domains").

Floating point: the strategy enumerates the interesting values NaN, +inf,
-inf, +0, -0, subnormals (min/max positive) and normals (min/max positive),
per width. These are the equivalence classes of the float type itself.
Integers: min/max of the considered type.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import List

NAN = object()  # sentinel distinguishing "NaN" from float("nan") equality traps


@dataclass(frozen=True)
class ValueClass:
    """One interesting value with the name of its equivalence class."""

    cls: str  # e.g. "nan", "+inf", "-0", "subnormal-min", "normal-max"
    value: object  # float or NAN sentinel


def float_classes(width: int) -> List[ValueClass]:
    """The interesting float values for a width (32 or 64)."""
    if width == 32:
        pack = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]
        min_sub = 2.0 ** -149
        max_sub = (1.0 - 2.0 ** -23) * 2.0 ** -126
        min_norm = 2.0 ** -126
        max_norm = (2.0 - 2.0 ** -23) * 2.0 ** 127
    else:
        pack = lambda v: struct.unpack("<d", struct.pack("<d", v))[0]
        min_sub = 2.0 ** -1074
        max_sub = (1.0 - 2.0 ** -52) * 2.0 ** -1022
        min_norm = 2.0 ** -1022
        max_norm = (2.0 - 2.0 ** -52) * 2.0 ** 1023

    # Round the computed extremes through actual float storage so the values
    # are exactly representable in the given width.
    # Order matters: the solver's witness minimization prefers low indices,
    # so "typical" values come first and extremes/specials only appear when
    # the equivalence class under test requires them.
    values = [
        ValueClass("normal-typical-pos", pack(1.5)),
        ValueClass("normal-typical-neg", pack(-2.25)),
        ValueClass("+0", 0.0),
        ValueClass("-0", -0.0),
        ValueClass("subnormal-min-pos", pack(min_sub)),
        ValueClass("subnormal-max-pos", pack(max_sub)),
        ValueClass("subnormal-min-neg", pack(-min_sub)),
        ValueClass("subnormal-max-neg", pack(-max_sub)),
        ValueClass("normal-min-pos", pack(min_norm)),
        ValueClass("normal-max-pos", pack(max_norm)),
        ValueClass("normal-min-neg", pack(-min_norm)),
        ValueClass("normal-max-neg", pack(-max_norm)),
        ValueClass("+inf", math.inf),
        ValueClass("-inf", -math.inf),
        ValueClass("nan", NAN),
    ]
    return values


def integer_classes(signed: bool, bits: int) -> List[ValueClass]:
    """min/max of an integer type -- the strategy's integer domain."""
    if signed:
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    else:
        lo, hi = 0, 2 ** bits - 1
    return [ValueClass("min-int", lo), ValueClass("max-int", hi)]


def render_value(v: object) -> str:
    """Stable, reviewable rendering of a value for manifests/reports."""
    if v is NAN:
        return "NaN"
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if v == math.inf:
            return "+inf"
        if v == -math.inf:
            return "-inf"
        if v == 0.0 and math.copysign(1.0, v) < 0:
            return "-0"
        if v == int(v):
            return str(int(v))
        return repr(v)
    return str(v)


def jsonable_value(v: object):
    """Value converted for JSON output (NaN/inf become tagged strings)."""
    if v is NAN:
        return "NaN"
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "+inf" if v > 0 else "-inf"
        if v == 0.0 and math.copysign(1.0, v) < 0:
            return "-0"
        return v
    if isinstance(v, Fraction):
        return v.numerator if v.denominator == 1 else float(v)
    return v