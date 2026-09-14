"""Equivalence-class enumeration.

Implements the strategy's DNF construction: each p_i of the constraint
system splits into (p'_i, p''_i); the domain decomposes into one sub-domain
per choice of boundary/interior for each predicate; each satisfiable
sub-domain is one equivalence class, instantiated with one representative
witness.

Branch predicates (the Abs-style piecewise definition) are handled
separately: they are already disjoint by construction, so each branch is
one class, solved against the operator domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import solver as solver_mod
from .dsl import Operator, Predicate, Pruning
from .splitting import combination_predicates, effective_choice, label, split


@dataclass
class EqClass:
    """One satisfiable equivalence class with its representative witness."""

    id: str
    choice: Tuple[bool, ...]
    label: str  # e.g. "p'_1 ∧ p''_2"
    predicates: List[Predicate]
    witness: Dict[str, object]  # variable name -> value (int or ValueClass)
    # Set when another class produced the identical witness (recorded in the
    # manifest; the classes remain distinct per the strategy).
    duplicate_of: Optional[str] = None
    # Test-completeness criterion (or unified pair) that generated the
    # class (strategy: "Test traceability"): union of its predicates'
    # criteria.
    criterion: str = ""
    # Why the combination is in the test set (strategy: "Test
    # traceability").
    retention_reason: str = ""
    # Risk-based prioritization tier (strategy: "Risk-based prioritization
    # of test generation"): 1 dependency/symmetry-flagged, 2 boundary/
    # vertex + special-value, 3 pairwise-coverage, 4 plain interior.
    priority: int = 0

    def describe(self) -> str:
        return " ∧ ".join(p.render() for p in self.predicates)


@dataclass
class UnsatCombo:
    label: str
    predicates: List[Predicate]


@dataclass
class PrunedCombo:
    label: str
    predicates: List[Predicate]
    fault_model_absent: str
    disposition: str


@dataclass
class BranchClass:
    """One auxiliary class (branch/boundary/property) with its witness."""

    id: str
    kind: str
    name: str
    origin: str
    predicates: List[Predicate]
    witness: Dict[str, object]
    # Class-provenance metadata, as on EqClass.
    criterion: str = ""
    retention_reason: str = ""
    priority: int = 0


@dataclass
class ClassifyResult:
    operator: str
    classes: List[EqClass] = field(default_factory=list)
    branches: List[BranchClass] = field(default_factory=list)
    unsat: List[UnsatCombo] = field(default_factory=list)
    pruned: List[PrunedCombo] = field(default_factory=list)
    backend: str = ""
    # Duplicates: class id -> representative class id with equal witness.
    duplicates: Dict[str, str] = field(default_factory=dict)


def _effective_choices(
    constraints: Sequence[Predicate],
) -> List[Tuple[bool, ...]]:
    """All p'/p'' choices, collapsing predicates without a p'' form.

    A predicate that is a plain equality keeps only its p' form, so choosing
    p' or p'' for it yields the same conjunction; the collapse avoids
    enumerating vacuous duplicates. Ordered most-constrained first
    (all-boundary), matching the strategy's presentation.
    """
    collapsed: List[Tuple[bool, ...]] = []
    seen = set()
    m = len(constraints)
    for mask in range(1 << m):
        choice = tuple(bool(mask >> (m - 1 - i) & 1) for i in range(m))
        eff = tuple(
            effective_choice(p, c) for c, p in zip(choice, constraints)
        )
        if eff not in seen:
            seen.add(eff)
            collapsed.append(eff)
    return collapsed


def _matches_pruning(choice: Tuple[bool, ...], rec: Pruning) -> bool:
    """A pruning record matches when every keyed entry agrees with the
    choice (1-based predicate index -> "'" boundary / "''" interior).
    Unkeyed entries are wildcards."""
    for k, v in rec.choice.items():
        idx = int(k) - 1
        want_prime = v == "'"
        if choice[idx] != want_prime:
            return False
    return True


# Risk-based prioritization tiers (strategy: "Risk-based prioritization
# of test generation").
_TIER_NAMES = {
    1: "1 (dependency/symmetry-flagged)",
    2: "2 (boundary/vertex + special-value)",
    3: "3 (pairwise-coverage)",
    4: "4 (interior, no known special role)",
}

_BRANCH_PRIORITY = {"property": 1, "boundary": 2, "branch": 2}
_BRANCH_RETENTION = {
    "property": ("explicit symmetry/asymmetry property: named interaction "
                 "risk (strategy: 'Symmetry and asymmetry')"),
    "boundary": ("usage-limit / value-domain boundary: base-choice "
                 "complement of the p'/p'' split system"),
    "branch": ("branch of the piecewise functional definition: "
               "special-value / discontinuity class"),
}


def tier_name(priority: int) -> str:
    """Human name of a prioritization tier ('' when unassigned)."""
    return _TIER_NAMES.get(priority, "")


def _class_criterion(predicates: Sequence[Predicate]) -> str:
    """Union of the predicates' test-completeness criteria."""
    seen: List[str] = []
    for p in predicates:
        if p.criterion and p.criterion not in seen:
            seen.append(p.criterion)
    return " + ".join(seen)


def _witness_key(w: Dict[str, object]) -> str:
    """Canonical string form of a witness (duplicate detection)."""
    parts = []
    for k in sorted(w):
        v = w[k]
        if hasattr(v, "cls"):
            parts.append(f"{k}={v.cls}:{v.value!r}")
        else:
            parts.append(f"{k}={v!r}")
    return "|".join(parts)


def enumerate_classes(op: Operator) -> ClassifyResult:
    """Enumerate all equivalence classes of an operator's constraint system."""
    result = ClassifyResult(operator=op.name, backend=solver_mod.backend_name())
    constraints = list(op.constraints)
    # Tensor-size variables are pushed to large values so structural
    # boundary classes are exercised on tensors where the boundary
    # matters (see solver.Z3Model.witness).
    prefer_large = tuple(op.params.get("prefer_large_vars", ()))

    seen_witness: Dict[str, str] = {}
    n_sat = 0

    # Operators whose functional definition is a pure piecewise branch
    # structure (Abs, Relu) have no inequality constraint system to split:
    # per the strategy's Abs example, the branches *are* the classes, so
    # only the auxiliary branch cases are generated.
    if constraints:
        for choice in _effective_choices(constraints):
            combo = combination_predicates(constraints, choice)
            lbl = label(choice)

            rec = next(
                (r for r in op.prunings if _matches_pruning(choice, r)), None
            )
            if rec is not None:
                result.pruned.append(
                    PrunedCombo(lbl, combo, rec.fault_model_absent,
                                rec.disposition)
                )
                continue

            witness, backend = solver_mod.check_combo(
                op.variables, op.domain, combo, prefer_large
            )
            if witness is None:
                result.unsat.append(UnsatCombo(lbl, combo))
                continue

            n_sat += 1
            cid = f"EC-{n_sat:03d}"
            interior_only = not any(choice)
            cls = EqClass(id=cid, choice=choice, label=lbl, predicates=combo,
                          witness=witness,
                          criterion=_class_criterion(combo),
                          retention_reason=(
                              "interior (rest-of-domain) class, retained "
                              "by the exhaustive p'/p'' decomposition "
                              "(no pruning record matched)"
                              if interior_only else
                              "boundary/vertex class, retained by the "
                              "exhaustive p'/p'' decomposition (no pruning "
                              "record matched)"),
                          priority=4 if interior_only else 2)
            key = _witness_key(witness)
            if key in seen_witness:
                cls.duplicate_of = seen_witness[key]
                result.duplicates[cid] = seen_witness[key]
            else:
                seen_witness[key] = cid
            result.classes.append(cls)

    # Auxiliary classes (branches of the functional definition, usage-limit
    # boundary cases, explicit symmetry properties): solved as-is against
    # the domain AND the functional constraint system. No p'/p'' split:
    # they are disjoint by construction.
    counters: Dict[str, int] = {}
    background = list(op.domain) + list(op.constraints)
    for br in op.branches:
        prefix = {"branch": "BR", "boundary": "BD", "property": "PR"}[br.kind]
        witness, _ = solver_mod.check_combo(op.variables, background,
                                            br.predicates, prefer_large)
        if witness is None:
            # An auxiliary case unsatisfiable against the domain is itself a
            # finding: the spec defines a case the type domain cannot reach.
            result.unsat.append(
                UnsatCombo(f"{br.kind}:{br.name}", list(br.predicates))
            )
            continue
        counters[br.kind] = counters.get(br.kind, 0) + 1
        result.branches.append(
            BranchClass(
                id=f"{prefix}-{counters[br.kind]:03d}",
                kind=br.kind,
                name=br.name,
                origin=br.origin,
                predicates=list(br.predicates),
                witness=witness,
                criterion=br.criterion,
                retention_reason=_BRANCH_RETENTION[br.kind],
                priority=_BRANCH_PRIORITY[br.kind],
            )
        )

    return result