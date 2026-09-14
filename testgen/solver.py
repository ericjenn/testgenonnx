"""Satisfiability checking and witness extraction.

Two backends:

  - Z3 backend (preferred; the strategy explicitly mentions Z3): translates
    the constraint system to Z3 over integers (IntVars get Z3 Ints; FloatVars
    get an integer index ranging over their interesting value classes) and
    checks each p'/p'' combination.

  - Enumeration backend (fallback when Z3 is unavailable): backtracking over
    per-variable candidate sets anchored on the domain boundaries. UNSAT
    verdicts are exact only w.r.t. the candidate sets, and float variables
    are not supported; the manifest records which backend produced a test set.

Witness *smallness*: after satisfiability is established,
each IntVar is greedily pushed to the smallest feasible candidate value so
the generated vectors stay readable.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .dsl import Expr, FloatVar, IntVar, Predicate, Var, vars_of, FLOAT_OPS
from .domains import ValueClass, float_classes

try:  # pragma: no cover - environment dependent
    import z3 as _z3

    HAVE_Z3 = True
except Exception:  # pragma: no cover
    _z3 = None
    HAVE_Z3 = False


def float_value_classes(v: FloatVar) -> List[ValueClass]:
    """Interesting values of a FloatVar (standard classes + extras)."""
    base = float_classes(v.width)
    extras = [ValueClass(f"extra[{x}]", x) for x in v.extra_values]
    return base + extras


# ---------------------------------------------------------------------------
# Z3 translation
# ---------------------------------------------------------------------------


class Z3Model:
    """Translates the DSL into Z3 terms and extracts witnesses."""

    def __init__(self, variables: Sequence[Var], domain: Sequence[Predicate]) -> None:
        self.variables: List[Var] = list(variables)
        self.domain: List[Predicate] = list(domain)
        self.zvars: Dict[str, object] = {}
        self.fclasses: Dict[str, List[ValueClass]] = {}
        for v in self.variables:
            if isinstance(v, IntVar):
                self.zvars[v.name] = _z3.Int(v.name)
            else:
                self.zvars[v.name] = _z3.Int(f"#{v.name}")
                self.fclasses[v.name] = float_value_classes(v)

    # -- terms --------------------------------------------------------------

    def expr(self, e: Union[Expr, Var]) -> object:
        if isinstance(e, (IntVar, FloatVar)):
            return self.zvars[e.name]
        if not isinstance(e, Expr):
            raise TypeError(f"bad expr {e!r}")
        if e.op == "const":
            (c,) = e.args
            if isinstance(c, Fraction) and c.denominator == 1:
                return _z3.IntVal(int(c))
            return _z3.RealVal(str(c))
        if e.op == "var":
            return self.zvars[e.args[0].name]
        a, b = (self.expr(x) for x in e.args)
        return {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
                "/": lambda: a / b}[e.op]()

    def constraint(self, p: Predicate) -> object:
        p = p.normalized()
        if p.op in FLOAT_OPS:
            return self._float_constraint(p)
        lhs = self.expr(p.lhs)
        rhs = self.expr(p.rhs)
        return {"<": lhs < rhs, "<=": lhs <= rhs, "=": lhs == rhs,
                "!=": lhs != rhs}[p.op]

    def _float_constraint(self, p: Predicate) -> object:
        """Translate a float-class comparison into an index constraint.

        The FloatVar is an index over its interesting value classes, so:

        - fcls/fne select classes *by name* (unambiguous for NaN, +inf, -0,
          ... which Python equality cannot distinguish);
        - flt/fge/fgt/fle select the finite classes whose numeric value
          satisfies the comparison. Infinities and NaN never match a range
          comparison: per the strategy's Abs example they form their own
          branches, so a range branch must exclude them explicitly (Abs
          branch p'6 does this with fne(..., "-inf")).
        """
        var = p.lhs.args[0] if isinstance(p.lhs, Expr) else p.lhs
        if not isinstance(var, FloatVar):
            raise TypeError(f"{p.op} requires a FloatVar lhs, got {var!r}")
        idx = self.zvars[var.name]
        classes = self.fclasses[var.name]

        if p.op in ("fcls", "fne"):
            if p.rhs.op != "clsname":
                raise TypeError(f"{p.op} rhs must be a class name, got {p.rhs!r}")
            (name,) = p.rhs.args
            sel = [i for i, c in enumerate(classes) if c.cls == name]
            if not sel:
                raise ValueError(f"unknown value class {name!r} for {var.name}")
            inner = _z3.Or(*[idx == i for i in sel])
            return _z3.Not(inner) if p.op == "fne" else inner

        # Range comparison against a numeric bound.
        (c,) = p.rhs.args
        bound = float(Fraction(c))

        def matches(val) -> bool:
            if not isinstance(val, float):
                return False  # NaN sentinel / non-float extras never match
            if math.isnan(val) or math.isinf(val):
                return False
            return {"flt": val < bound, "fge": val >= bound,
                    "fgt": val > bound, "fle": val <= bound}[p.op]

        sel = [i for i, cl in enumerate(classes) if matches(cl.value)]
        if not sel:
            return _z3.BoolVal(False)
        return _z3.Or(*[idx == i for i in sel])

    def _scope(self) -> List[object]:
        """Domain constraints + float index ranges."""
        s = [self.constraint(p) for p in self.domain]
        for v in self.variables:
            if isinstance(v, FloatVar):
                idx = self.zvars[v.name]
                s.append(_z3.And(idx >= 0, idx < len(self.fclasses[v.name])))
        return s

    # -- solving ------------------------------------------------------------

    def witness(self, combo: Sequence[Predicate],
                prefer_large: Sequence[str] = ()) -> Optional[Dict[str, object]]:
        """Solve domain ∧ combo; return variable -> concrete value, or None.

        Two-phase: (1) plain satisfiability; (2) greedy optimization of
        each IntVar (and of each FloatVar's class index, so 'typical'
        values are preferred over the extremes unless the class requires
        an extreme) while re-checking satisfiability. Deterministic
        order: variable declaration order.

        By default each variable is pushed to its smallest feasible
        candidate value (readable witnesses). Variables
        named in ``prefer_large`` are pushed to their *largest* feasible
        value instead -- used for tensor sizes so that structural
        boundary cases are exercised on tensors large enough for the
        boundary to matter (e.g. a stride of 3 on a 1x1 input would be
        vacuous).
        """
        solver = _z3.Solver()
        solver.add(self._scope())
        for p in combo:
            solver.add(self.constraint(p))
        if solver.check() != _z3.sat:
            return None

        # Phase 2: greedily push variables to their smallest (or, for
        # prefer_large, largest) feasible candidate value. Candidates
        # include the boundary-adjacent values and any constant the
        # predicates mention, so witnesses tend to sit on the boundaries
        # the class is about.
        cands = _candidate_values(self.variables, self.domain, list(combo))
        for v in self.variables:
            z = self.zvars[v.name]
            bound_hi = (
                len(self.fclasses[v.name]) - 1 if isinstance(v, FloatVar) else v.hi
            )
            order = cands[v.name]
            if v.name in prefer_large:
                order = list(reversed(order))
            for target in order:
                if target > bound_hi:
                    continue
                probe = _z3.Solver()
                probe.add(*solver.assertions())
                probe.add(z == target)
                if probe.check() == _z3.sat:
                    solver = probe
                    break
            # No candidate feasible: keep whatever value phase 1 found.

        model = solver.model()
        return self._extract(model)

    def _extract(self, model) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for v in self.variables:
            val = model.eval(self.zvars[v.name], model_completion=True)
            iv = int(val.as_long())
            if isinstance(v, IntVar):
                out[v.name] = iv
            else:
                out[v.name] = self.fclasses[v.name][iv]
        return out


def _candidate_values(
    variables: Sequence[Var],
    domain: Sequence[Predicate],
    combo: Sequence[Predicate],
) -> Dict[str, List[int]]:
    """Ascending candidate targets per variable.

    IntVars: {lo, lo+1, mid, hi-1, hi} ∪ {integer constants mentioned in any
    predicate, clamped to the domain} ∪ {c-1, c+1 for those constants}
    (adjacent-to-boundary values witness 'strictly inside' classes).
    FloatVars: class indices 0..n-1 (classes are ordered from typical to
    extreme so index minimization prefers typical values).
    """
    consts: set = set()
    for p in list(domain) + list(combo):
        for e in (p.lhs, p.rhs):
            _collect_consts(e, consts)

    cands: Dict[str, List[int]] = {}
    for v in variables:
        if isinstance(v, FloatVar):
            cands[v.name] = list(range(len(float_value_classes(v))))
            continue
        s = {v.lo, v.lo + 1, v.mid(), v.hi - 1, v.hi}
        for c in consts:
            for t in (c, c - 1, c + 1):
                if v.lo <= t <= v.hi:
                    s.add(t)
        cands[v.name] = sorted(s)
    return cands


def _collect_consts(e, out: set) -> None:
    if isinstance(e, (IntVar, FloatVar)) or not isinstance(e, Expr):
        return
    if e.op == "const":
        (c,) = e.args
        if isinstance(c, Fraction) and c.denominator == 1:
            out.add(int(c))
        return
    for a in e.args:
        _collect_consts(a, out)


# ---------------------------------------------------------------------------
# Enumeration fallback (IntVars only)
# ---------------------------------------------------------------------------


def _eval(e: Union[Expr, Var], env: Dict[str, int]) -> Fraction:
    if isinstance(e, (IntVar, FloatVar)):
        return Fraction(env[e.name])
    if e.op == "const":
        (c,) = e.args
        return Fraction(c)
    if e.op == "var":
        return Fraction(env[e.args[0].name])
    a, b = (_eval(x, env) for x in e.args)
    if e.op == "+":
        return a + b
    if e.op == "-":
        return a - b
    if e.op == "*":
        return a * b
    if e.op == "/":
        return a / b
    raise ValueError(f"bad expression op {e.op!r}")


def _holds(p: Predicate, env: Dict[str, int]) -> bool:
    l, r = _eval(p.lhs, env), _eval(p.rhs, env)
    return {"<": l < r, "<=": l <= r, "=": l == r, "!=": l != r}[p.normalized().op]


def check_combo_enum(
    variables: Sequence[Var],
    domain: Sequence[Predicate],
    combo: Sequence[Predicate],
) -> Optional[Dict[str, object]]:
    """Backtracking search over boundary-anchored candidate values.

    Degraded mode: verdicts are exact only w.r.t. the candidate sets, and
    FloatVars are rejected (the manifest records the degraded backend).
    """
    for v in variables:
        if isinstance(v, FloatVar):
            raise RuntimeError(
                "enumeration fallback cannot handle float variables; "
                "install z3-solver for full support"
            )
    names = [v.name for v in variables]
    preds = list(domain) + list(combo)

    def bound_preds(assigned: set) -> List[Predicate]:
        out = []
        for p in preds:
            vs = {v.name for v in vars_of(p)}
            if vs and vs.issubset(assigned):
                out.append(p)
        return out

    cands: Dict[str, List[int]] = {
        v.name: sorted(
            {v.lo, v.lo + 1, v.mid(), v.hi - 1, v.hi} & set(range(v.lo, v.hi + 1))
        )
        for v in variables
    }
    env: Dict[str, int] = {}
    result: Optional[Dict[str, object]] = None

    def dfs(i: int) -> bool:
        nonlocal result
        if i == len(names):
            result = dict(env)
            return True
        n = names[i]
        assigned = set(names[: i + 1])
        for val in cands[n]:
            env[n] = val
            if all(_holds(p, env) for p in bound_preds(assigned)):
                if dfs(i + 1):
                    return True
        return False

    return result if dfs(0) else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def backend_name() -> str:
    return "z3" if HAVE_Z3 else "enumeration"


def check_combo(
    variables: Sequence[Var],
    domain: Sequence[Predicate],
    combo: Sequence[Predicate],
    prefer_large: Sequence[str] = (),
) -> Tuple[Optional[Dict[str, object]], str]:
    """Return (witness or None, backend name) for domain ∧ combo."""
    if HAVE_Z3:
        w = Z3Model(variables, domain).witness(combo, prefer_large)
        return w, "z3"
    return check_combo_enum(variables, domain, combo), "enumeration"