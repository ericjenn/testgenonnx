"""Constraint DSL: typed variables, arithmetic expressions, predicates.

The test designer declares an operator's input domain as a system of
predicates over typed variables (strategy section "Equivalence classes based
on the input domain"). Predicates are kept as a small AST so they can be:

  - rewritten (strict -> non-strict inequalities, integer semantics),
  - split into boundary (p') and interior (p'') versions,
  - translated to a solver (Z3) for satisfiability and witness extraction.

Supported variable kinds:
  - IntVar:   bounded integer (e.g. uint8 values, ranks, dimension sizes,
              attributes such as strides/pads).
  - FloatVar: IEEE-754 binary32/binary64 values (handled through a
              discrete set of *interesting* values -- special values and
              normal representatives -- see domains.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Union


# ---------------------------------------------------------------------------
# LaTeX names of variables
# ---------------------------------------------------------------------------

_TRAILING_DIGITS = re.compile(r"^([A-Za-z][A-Za-z_]*?)(\d+)$")


def _default_latex(name: str) -> str:
    """Generic LaTeX form of a variable name: trailing digits become a
    subscript (x1 -> x_1, dX2 -> dX_2)."""
    m = _TRAILING_DIGITS.match(name)
    return f"{m.group(1)}_{m.group(2)}" if m else name


def var_latex(v: "Var") -> str:
    """LaTeX symbol of a variable (its ``latex`` field or the default)."""
    return v.latex if getattr(v, "latex", "") else _default_latex(v.name)


def _tex_class(name: str) -> str:
    """LaTeX form of a float value-class name."""
    return {"nan": "NaN", "+inf": "+\\infty", "-inf": "-\\infty",
            "+0": "+0", "-0": "-0"}.get(name, name.replace("_", "\\_"))


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntVar:
    """A bounded integer variable.

    ``lo``/``hi`` are the *type or usage* bounds of the domain. For value
    variables they are the integer type bounds (e.g. 0..255 for uint8); for
    structural variables they are usage limits (e.g. rank in 0..4), which the
    strategy treats exactly like domain constraints ("The case of unbounded
    variables").
    """

    name: str
    lo: int
    hi: int
    # Provenance for traceability: which spec notion the variable represents.
    role: str = ""
    # True when lo..hi is a usage limit rather than a type constraint
    # (recorded in the manifest).
    usage_limit: bool = False
    # Fault model assumed absent for the chosen usage bound (strategy:
    # "The case of unbounded variables" -- the bound value is an
    # engineering choice that a reviewer must be able to challenge, so it
    # carries the same kind of justification as a pruning record).
    justification: str = ""
    # LaTeX symbol for the human-readable reports (defaults to the variable
    # name with trailing digits as a subscript).
    latex: str = ""

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"{self.name}: empty domain [{self.lo},{self.hi}]")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return self.name

    def mid(self) -> int:
        return (self.lo + self.hi) // 2


@dataclass(frozen=True)
class FloatVar:
    """A floating-point variable identified by its interesting values.

    Rather than reasoning over the continuum, the generator reasons over the
    discrete equivalence classes of IEEE-754 values listed in the strategy
    (NaN, +inf, -inf, +0, -0, subnormal, normal) plus, optionally, a few
    explicitly supplied constants. The solver sees an integer index into
    this value list; the manifest records the resolved float.
    """

    name: str
    width: int  # 32 or 64
    role: str = ""
    # Extra interesting values beyond the standard special-value classes.
    extra_values: tuple = field(default=())
    # LaTeX symbol for the human-readable reports.
    latex: str = ""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return self.name


Var = Union[IntVar, FloatVar]


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Expr:
    """Arithmetic expression over variables and constants.

    ops: "+", "-", "*", "/" (exact rational arithmetic is used for
    evaluation; the Z3 translation mirrors it on Int/Real terms).
    """

    op: str
    args: tuple  # elements: Expr, Var, int, float, Fraction

    # -- readable rendering --------------------------------------------------
    # ``str`` produces classic infix notation with minimal parentheses
    # (used by the machine-readable manifests); ``render_latex`` produces
    # classical mathematical notation for the human-readable reports.

    def __str__(self) -> str:
        return self._str_prec(None)

    def _atom(self, a) -> str:
        if isinstance(a, Expr):
            return str(a)
        if isinstance(a, (IntVar, FloatVar)):
            return a.name
        f = Fraction(a)
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"

    @staticmethod
    def _plain_parens(child, parent_op: str, is_left: bool) -> bool:
        """Minimal parentheses for plain infix notation."""
        if not isinstance(child, Expr) or child.op in ("var", "const"):
            return False
        if parent_op == "+":
            return False
        if parent_op == "-":
            return not is_left and child.op in ("+", "-")
        if parent_op == "*":
            return child.op in ("+", "-", "/")
        return child.op in ("+", "-", "*", "/")  # parent "/"

    def _wrap(self, child, is_left: bool) -> str:
        s = self._atom(child)
        return f"({s})" if self._plain_parens(child, self.op, is_left) else s

    def _str_prec(self, parent_op) -> str:
        if self.op in ("var", "const"):
            return self._atom(self.args[0])
        ls, rs = self._wrap(self.args[0], True), self._wrap(self.args[1], False)
        return {"+": f"{ls} + {rs}", "-": f"{ls} - {rs}", "*": f"{ls} * {rs}",
                "/": f"{ls} / {rs}"}[self.op]

    def render_latex(self) -> str:
        """Classical mathematical notation, e.g.
        ``(dY_2 - 1) \\cdot s_0 + \\mathrm{Ke}_0 \\le dX_2 + p_0 + p_2``."""
        return self._latex_prec(None)

    def _latex_atom(self, a) -> str:
        if isinstance(a, Expr):
            return a.render_latex()
        if isinstance(a, (IntVar, FloatVar)):
            return var_latex(a)
        f = Fraction(a)
        return str(f.numerator) if f.denominator == 1 else (
            rf"\frac{{{f.numerator}}}{{{f.denominator}}}")

    @staticmethod
    def _latex_parens(child, parent_op: str, is_left: bool) -> bool:
        """Minimal parentheses for LaTeX. Division is \\frac (self-
        parenthesizing), so only sums under products and after a minus
        sign need grouping."""
        if not isinstance(child, Expr) or child.op in ("var", "const"):
            return False
        if parent_op == "+":
            return False
        if parent_op == "-":
            return not is_left and child.op in ("+", "-")
        return parent_op == "*" and child.op in ("+", "-")

    def _latex_wrap(self, child, is_left: bool) -> str:
        s = self._latex_atom(child)
        if self._latex_parens(child, self.op, is_left):
            return rf"\left({s}\right)"
        return s

    def _latex_prec(self, parent_op) -> str:
        if self.op in ("var", "const"):
            return self._latex_atom(self.args[0])
        ls = self._latex_wrap(self.args[0], True)
        rs = self._latex_wrap(self.args[1], False)
        if self.op == "+":
            return f"{ls} + {rs}"
        if self.op == "-":
            return f"{ls} - {rs}"
        if self.op == "*":
            return rf"{ls} \cdot {rs}"
        return rf"\frac{{{ls}}}{{{rs}}}"


def _as_expr(a: Union[Expr, Var, int, float, Fraction]) -> Expr:
    if isinstance(a, Expr):
        return a
    if isinstance(a, (IntVar, FloatVar)):
        return Expr("var", (a,))
    if isinstance(a, (int, float, Fraction)):
        # Fraction(float) is exact (binary float -> dyadic rational).
        return Expr("const", (Fraction(a),))
    raise TypeError(f"cannot make expression from {a!r}")


def const(v) -> Expr:
    return Expr("const", (Fraction(v),))


def add(*args) -> Expr:
    e = _as_expr(args[0])
    for a in args[1:]:
        e = Expr("+", (e, _as_expr(a)))
    return e


def sub(a, b) -> Expr:
    return Expr("-", (_as_expr(a), _as_expr(b)))


def mul(*args) -> Expr:
    e = _as_expr(args[0])
    for a in args[1:]:
        e = Expr("*", (e, _as_expr(a)))
    return e


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Predicate:
    """A comparison p_i of the constraint system.

    ``op`` is one of "<", "<=", "=", "!=", ">", ">=".
    The predicate may be an *equality constraint* (op == "="), which the
    strategy keeps as-is: it has a p' form only (an equality has no interior).

    ``origin`` carries the traceability tag / spec reference.
    ``note`` documents why the predicate is part of the domain.
    ``criterion`` names the test-completeness criterion the predicate
    serves (strategy: "Test traceability" -- generated tests record which
    criterion, or unified pair of criteria, produced them).
    """

    op: str
    lhs: Expr
    rhs: Expr
    origin: str = ""
    note: str = ""
    criterion: str = ""

    def __post_init__(self) -> None:
        if self.op not in ("<", "<=", "=", "!=", ">", ">=") + FLOAT_OPS:
            raise ValueError(f"bad comparison op {self.op!r}")

    def render(self) -> str:
        """Readable plain-text form, e.g. ``(dY2 - 1) * s0 + ... <= dX2 + ...``
        (used in the machine-readable manifest and test JSON files)."""
        if self.op == "fcls":
            return f"{self.lhs} in class {self.rhs.args[0]}"
        if self.op == "fne":
            return f"{self.lhs} not in class {self.rhs.args[0]}"
        if self.op in ("flt", "fge", "fgt", "fle"):
            return f"{self.lhs} {self.op[1:]} {self.rhs}"
        return f"{self.lhs} {self.op} {self.rhs}"

    def render_latex(self) -> str:
        """Classical mathematical notation (without surrounding $...$),
        e.g. ``(dY_2 - 1) \\cdot s_0 + \\mathrm{Ke}_0 \\le dX_2 + p_0``."""
        if self.op == "fcls":
            return rf"{self.lhs.render_latex()} \in \mathrm{{{_tex_class(self.rhs.args[0])}}}"
        if self.op == "fne":
            return rf"{self.lhs.render_latex()} \notin \mathrm{{{_tex_class(self.rhs.args[0])}}}"
        if self.op in ("flt", "fge", "fgt", "fle"):
            sym = {"flt": "<", "fge": "\\ge", "fgt": ">",
                   "fle": "\\le"}[self.op]
            return f"{self.lhs.render_latex()} {sym} {self.rhs.render_latex()}"
        sym = {"<": "<", "<=": "\\le", "=": "=", "!=": "\\ne",
               ">": ">", ">=": "\\ge"}[self.op]
        return f"{self.lhs.render_latex()} {sym} {self.rhs.render_latex()}"

    # -- normalization -----------------------------------------------------

    def normalized(self) -> "Predicate":
        """Rewrite > and >= into < and <= (lhs/rhs swapped)."""
        if self.op == ">":
            return Predicate("<", self.rhs, self.lhs, self.origin, self.note,
                             self.criterion)
        if self.op == ">=":
            return Predicate("<=", self.rhs, self.lhs, self.origin, self.note,
                             self.criterion)
        return self


def pred(op: str, lhs, rhs, origin: str = "", note: str = "",
         criterion: str = "") -> Predicate:
    return Predicate(op, _as_expr(lhs), _as_expr(rhs), origin, note, criterion)


def lt(a, b, **kw) -> Predicate:
    return pred("<", a, b, **kw)


def le(a, b, **kw) -> Predicate:
    return pred("<=", a, b, **kw)


def eq(a, b, **kw) -> Predicate:
    return pred("=", a, b, **kw)


def ne(a, b, **kw) -> Predicate:
    return pred("!=", a, b, **kw)


def ge(a, b, **kw) -> Predicate:
    return pred(">=", a, b, **kw)


def gt(a, b, **kw) -> Predicate:
    return pred(">", a, b, **kw)


# Float-class comparisons: lhs is a FloatVar, rhs a constant naming a value
# class ("nan", "+inf", "-0", ...) or a numeric bound. These predicates are
# never split: they describe branches of the functional definition (cf. the
# Abs example in the strategy), which are disjoint by construction.
FLOAT_OPS = ("fcls", "fne", "flt", "fge", "fgt", "fle")


def fcls(v, class_name: str, origin: str = "", note: str = "") -> Predicate:
    """FloatVar is exactly the named value class (e.g. "nan", "-0")."""
    return Predicate("fcls", _as_expr(v), Expr("clsname", (class_name,)),
                      origin, note)


def fne(v, class_name: str, origin: str = "", note: str = "") -> Predicate:
    """FloatVar is not the named value class."""
    return Predicate("fne", _as_expr(v), Expr("clsname", (class_name,)),
                     origin, note)


def flt(v, bound, origin: str = "", note: str = "") -> Predicate:
    """FloatVar's value class lies strictly below `bound` (finite classes)."""
    return pred("flt", v, bound, origin, note)


def fge(v, bound, origin: str = "", note: str = "") -> Predicate:
    return pred("fge", v, bound, origin, note)


def fgt(v, bound, origin: str = "", note: str = "") -> Predicate:
    return pred("fgt", v, bound, origin, note)


def fle(v, bound, origin: str = "", note: str = "") -> Predicate:
    return pred("fle", v, bound, origin, note)


# ---------------------------------------------------------------------------
# Constraint system
# ---------------------------------------------------------------------------


@dataclass
class Pruning:
    """A recorded elimination of a combination (strategy: "Eliminating
    irrelevant cases"). Every exclusion is an engineering judgment and must
    be documented with the fault model assumed absent and a disposition."""

    # The combination being excluded: {predicate_index (1-based): "'" or "''"}.
    # Unkeyed entries are wildcards (see classes._matches_pruning).
    choice: dict
    fault_model_absent: str
    # Disposition vocabulary of the strategy's field table:
    #   "pruned"            excluded from the test set
    #   "covered-by <id>"   excluded; the cited test reveals the same class
    #   "pairwise-coverage" kept (or excluded) purely to satisfy a
    #                       covering-array obligation, without an
    #                       individually stated fault model
    disposition: str

    def __post_init__(self) -> None:
        ok = (self.disposition in ("pruned", "pairwise-coverage")
              or self.disposition.startswith("covered-by"))
        if not ok:
            raise ValueError(
                f"bad pruning disposition {self.disposition!r}: expected "
                f"'pruned', 'covered-by <test id>' or 'pairwise-coverage'")


@dataclass
class Dependency:
    """One row of the per-operator cross-domain dependency table.

    ``catalog_refs`` cites the entries of the shared cross-operator
    pattern catalog (catalog.py) that this row instantiates, so a pattern
    discovered on another operator flags this one for re-review.
    """

    candidate: str
    found: bool
    combined_test: str = "-"  # test id added, or "-"
    catalog_refs: tuple = ()


@dataclass
class Branch:
    """An auxiliary class solved against the domain, kept as-is (no p'/p''
    split). Three kinds, all recorded in the manifest:

    - kind="branch":   one branch of the piecewise functional definition
                       (Abs/Relu discontinuity and special-value classes).
    - kind="boundary": a usage-limit boundary of a single structural
                       variable (base-choice complement, used when the
                       full p'/p'' cross-product over usage limits would
                       explode).
    - kind="property": an explicit symmetric/asymmetric-style relation
                       between parameters (strategy: "Symmetry and
                       asymmetry").
    """

    name: str
    predicates: List[Predicate]
    origin: str = ""
    kind: str = "branch"
    # Which test-completeness criterion this auxiliary case serves
    # (strategy: "Test traceability").
    criterion: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("branch", "boundary", "property"):
            raise ValueError(f"bad branch kind {self.kind!r}")


@dataclass
class Operator:
    """An operator's constraint system + metadata for test generation."""

    name: str
    # Path of the informal specification file (relative to the repository
    # root). Test generation is *refused* when this file does not exist:
    # tests are only derived from actual specifications.
    spec_file: str
    # Spec traceability for the operator itself.
    spec_ref: str
    variables: List[Var]
    # Type/usage constraints defining each variable's own domain. These are
    # NOT split into p'/p''; they form the background theory the functional
    # predicates are solved against.
    domain: List[Predicate]
    # The constraint system P = {p_1..p_m} that gets the p'/p'' treatment.
    constraints: List[Predicate]
    # Extra generation parameters (e.g. dtype, padding constant) carried
    # into the manifest and used by the functional materializer.
    params: dict = field(default_factory=dict)
    branches: List[Branch] = field(default_factory=list)
    prunings: List[Pruning] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)
    # Free-form notes recorded in the manifest.
    notes: List[str] = field(default_factory=list)
    # Names of the expected outputs the materializer emits for this
    # operator (rendered in the report prose; e.g. ["Y", "Indices"]).
    outputs: List[str] = field(default_factory=list)


def vars_of(*exprs: Union[Expr, Var, Predicate]) -> List[Var]:
    """Collect variables appearing in expressions/predicates."""
    seen: dict = {}

    def walk(e) -> None:
        if isinstance(e, Predicate):
            walk(e.lhs)
            walk(e.rhs)
            return
        if isinstance(e, (IntVar, FloatVar)):
            seen[e.name] = e
            return
        if isinstance(e, Expr):
            if e.op == "var":
                walk(e.args[0])
            elif e.op == "const":
                return
            else:
                for a in e.args:
                    walk(a)

    for x in exprs:
        walk(x)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Validation of the operator definition
# ---------------------------------------------------------------------------

_ORDERED_FLOAT_OPS = ("flt", "fge", "fgt", "fle")


def _mentions(pred: Predicate, v: Var) -> bool:
    return any(x is v for x in vars_of(pred))


def nan_sibling_violations(op: "Operator") -> List[str]:
    """NaN sibling-predicate check (strategy: "Eliminating irrelevant
    cases" -- the caution on unifying an ordered domain predicate with a
    special-value predicate).

    Under IEEE 754 every ordered comparison with NaN is false, so a NaN
    value satisfies neither p' nor p'' of an ordered float predicate and
    is silently excluded from the enumerated classes. For every FloatVar
    over which the operator uses ordered float comparisons, an explicit
    NaN sibling predicate (fcls/fne with class "nan") must therefore
    exist, unless the operator records the NaN gap in
    ``params["nan_spec_gaps"]`` (mapping variable name -> gap prose, e.g.
    "the spec leaves NaN behaviour undefined" -- reported, not assumed).

    Returns a list of diagnostic messages (empty when the definition is
    consistent).
    """
    violations: List[str] = []
    declared_gaps = op.params.get("nan_spec_gaps", {})
    all_preds = (list(op.domain) + list(op.constraints)
                 + [p for br in op.branches for p in br.predicates])
    for v in op.variables:
        if not isinstance(v, FloatVar):
            continue
        ordered = [p for p in all_preds
                   if p.op in _ORDERED_FLOAT_OPS and _mentions(p, v)]
        if not ordered:
            continue  # no ordered predicate: NaN cannot be silently dropped
        has_nan_sibling = any(
            p.op in ("fcls", "fne") and p.rhs.args[0] == "nan"
            and _mentions(p, v)
            for p in all_preds
        )
        if has_nan_sibling:
            continue
        if v.name in declared_gaps:
            continue  # recorded specification gap, not an oversight
        violations.append(
            f"{op.name}: variable {v.name!r} has ordered float "
            f"predicates (e.g. {ordered[0].render()!r}) but no explicit "
            f"NaN sibling predicate; NaN satisfies neither p' nor p'' of "
            f"an ordered predicate and would be silently excluded. Add an "
            f"explicit fcls/fne 'nan' predicate or record the NaN "
            f"specification gap in params['nan_spec_gaps']."
        )
    return violations