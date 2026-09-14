"""Direct evaluation of predicates on a witness.

This is the *verification* path, deliberately independent of the solving
path: after the solver produces a representative for each class, every
witness is re-checked by plain arithmetic evaluation of its class
predicates (and of the operator domain). Generation aborts rather than
emitting a test case whose witness fails verification.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Sequence

from .dsl import Expr, FloatVar, IntVar, Predicate, Var
from .domains import ValueClass


def _value_of(env: Dict[str, object], name: str) -> object:
    v = env[name]
    return v.value if isinstance(v, ValueClass) else v


def expr_value(e, env: Dict[str, object]) -> Fraction:
    """Evaluate an expression over a witness; float classes contribute
    their numeric value (NaN/inf classes must not appear in arithmetic)."""
    if isinstance(e, (IntVar, FloatVar)):
        return _as_frac(_value_of(env, e.name))
    if e.op == "const":
        (c,) = e.args
        return Fraction(c)
    if e.op == "var":
        return _as_frac(_value_of(env, e.args[0].name))
    a = expr_value(e.args[0], env)
    b = expr_value(e.args[1], env)
    if e.op == "+":
        return a + b
    if e.op == "-":
        return a - b
    if e.op == "*":
        return a * b
    if e.op == "/":
        return a / b
    raise ValueError(f"bad expression op {e.op!r}")


def _as_frac(v: object) -> Fraction:
    if isinstance(v, Fraction):
        return v
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            raise ValueError("NaN/inf class used in arithmetic expression")
        return Fraction(v)
    raise TypeError(f"cannot evaluate {v!r}")


def holds(p: Predicate, env: Dict[str, object]) -> bool:
    p = p.normalized()
    if p.op in ("fcls", "fne"):
        var = p.lhs.args[0] if isinstance(p.lhs, Expr) else p.lhs
        w = env[var.name]
        cls = w.cls if isinstance(w, ValueClass) else None
        if cls is None:
            raise TypeError(f"{var.name} witness is not a value class")
        ok = cls == p.rhs.args[0]
        return not ok if p.op == "fne" else ok
    if p.op in ("flt", "fge", "fgt", "fle"):
        var = p.lhs.args[0] if isinstance(p.lhs, Expr) else p.lhs
        w = env[var.name]
        v = w.value if isinstance(w, ValueClass) else w
        if not isinstance(v, float) or math.isnan(v) or math.isinf(v):
            return False  # range comparisons never match NaN/inf classes
        (c,) = p.rhs.args
        bound = float(Fraction(c))
        return {"flt": v < bound, "fge": v >= bound,
                "fgt": v > bound, "fle": v <= bound}[p.op]
    l = expr_value(p.lhs, env)
    r = expr_value(p.rhs, env)
    return {"<": l < r, "<=": l <= r, "=": l == r, "!=": l != r}[p.op]


def verify_witness(
    op_vars: Sequence[Var],
    domain: Sequence[Predicate],
    predicates: Sequence[Predicate],
    witness: Dict[str, object],
) -> List[str]:
    """Return the list of violated predicates (empty = witness is sound)."""
    env = {v.name: witness[v.name] for v in op_vars}
    failures = []
    for p in list(domain) + list(predicates):
        try:
            if not holds(p, env):
                failures.append(p.render())
        except ValueError as exc:
            failures.append(f"{p.render()} ({exc})")
    return failures