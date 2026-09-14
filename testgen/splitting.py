"""Strict-inequality rewriting and the p'/p'' split.

Strategy ("Equivalence classes based on the input domain"):

  1. Every *strict* inequality over an integer-valued comparison is first
     rewritten into a non-strict one:  x < c  ->  x <= c-1.  Over the
     integers the boundary of x < c is decidable as the equality x = c-1.
  2. Each (possibly rewritten) inequality predicate p_i is then split into
        p'_i : replace <= (resp. >=) by =     (on the boundary)
        p''_i: replace <= (resp. >=) by <     (strictly inside)
     Equality predicates keep only a p' form (an equality has no interior).

Note: when the boundary is not a decidable constant (float continuum,
symbolic right-hand side), the split still applies -- p' expresses "on the
boundary" and p'' "strictly inside" -- but we do not force a nextdown()
rewrite, which is only sound for constant right-hand sides.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .dsl import FloatVar, Expr, Predicate, const, vars_of


def _is_integer_comparison(p: Predicate) -> bool:
    """True when the comparison is integer-valued, i.e. involves no
    FloatVar (integer variables and integral constants only), so the
    strict->non-strict rewrite is exact."""
    return not any(isinstance(v, FloatVar) for v in vars_of(p))


def rewrite_strict(p: Predicate) -> Predicate:
    """Integer-domain strict -> non-strict rewriting.

    x < E  =>  x <= E - 1   -- valid over the integers even when the
    right-hand side E is symbolic (all quantities are integral, so
    nextdown(E) = E - 1 is exact). This is the strategy's prescribed
    first step ("we replace all strict inequality by non-strict
    inequality").
    """
    p = p.normalized()
    if p.op != "<":
        return p
    if not _is_integer_comparison(p):
        return p  # float continuum: see split() for the limitation
    return Predicate("<=", p.lhs, Expr("-", (p.rhs, const(1))),
                     p.origin, p.note, p.criterion)


def split(p: Predicate) -> Tuple[Optional[Predicate], Optional[Predicate]]:
    """Return (p', p'') for a predicate.

    p'  : on the boundary (equality form of the rewritten predicate)
    p'' : strictly inside
    Either form may be absent:

    - '=' predicates have no interior (p'' = None);
    - '!=' predicates (domain holes) have no boundary (p' = None);
    - strict float comparisons left native by rewrite_strict have no
      exact boundary on the continuum (p' = None) -- the strategy's
      nextdown() rewrite only applies to constant right-hand sides.
    """
    p = rewrite_strict(p).normalized()
    if p.op == "=":
        return p, None
    if p.op == "<=":
        p_prime = Predicate("=", p.lhs, p.rhs, p.origin, p.note, p.criterion)
        p_dprime = Predicate("<", p.lhs, p.rhs, p.origin, p.note, p.criterion)
        return p_prime, p_dprime
    if p.op == "<":
        # Survived rewriting: float continuum -- no exact boundary form.
        return None, p
    if p.op == "!=":
        return None, p
    raise ValueError(f"unsupported op {p.op!r} after normalization")


def effective_choice(p: Predicate, c: bool) -> bool:
    """Collapse a p'/p'' choice when a predicate has only one form."""
    p_prime, p_dprime = split(p)
    if p_prime is None:
        return False  # only p'' exists
    if p_dprime is None:
        return True   # only p' exists
    return c


def combination_predicates(
    constraints: List[Predicate], choice: Tuple[bool, ...]
) -> List[Predicate]:
    """Materialize the conjunction for one p'/p'' choice."""
    out: List[Predicate] = []
    for p, use_prime in zip(constraints, choice):
        p_prime, p_dprime = split(p)
        if use_prime:
            sel = p_prime if p_prime is not None else p_dprime
        else:
            sel = p_dprime if p_dprime is not None else p_prime
        if sel is None:
            sel = p
        out.append(sel)
    return out


def label(choice: Tuple[bool, ...]) -> str:
    """Human label of a choice, e.g. ``p'_1 ∧ p''_2 ∧ p'_3``."""
    return " ∧ ".join(("p'" if c else "p''") + f"_{i + 1}" for i, c in enumerate(choice))


def label_of(label_str: str) -> str:
    """Convert a plain ``label()`` string into its LaTeX form (used where
    only the stored label is available, e.g. UNSAT/pruned records)."""
    return label_str.replace(" ∧ ", r" \wedge ")