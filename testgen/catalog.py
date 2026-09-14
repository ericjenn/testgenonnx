"""Cross-operator pattern catalog (strategy: "Maintaining a cross-operator
pattern catalog").

The dependency-check questions, the special-value/discontinuity findings
and any interaction discovered by mutation testing or by an actual
specification-vs-implementation discrepancy are maintained as a single,
shared, living catalog across all SONNX operators rather than as
per-operator, one-off tables. When a new pattern is found on operator A,
every other operator sharing the relevant structural feature (the
``structural_features`` tags below) is flagged for re-review against the
new entry.

Operator definitions reference catalog entries by ID (see
``Dependency.catalog_refs``); the generator emits the referenced entries
into the manifest and report of every test set, so a reviewer can see which
shared patterns motivated which combined boundary cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class CatalogPattern:
    """One reusable interaction pattern.

    - ``id``: stable identifier cited by operator definitions.
    - ``structural_features``: tags naming the operator structure the
      pattern depends on ("sliding-window", "padding", "index-output",
      "output-size-formula", "reduction-accumulator", "ordered-float-domain");
      operators carrying the same tag are flagged for re-review when the
      pattern is updated.
    - ``origin``: how the pattern was discovered (dependency check,
      mutation testing, spec-vs-implementation discrepancy, or the
      strategy document itself).
    - ``action``: what the pattern requires of a test design.
    """

    id: str
    title: str
    description: str
    structural_features: tuple
    origin: str
    action: str


PATTERNS = [
    CatalogPattern(
        id="SPECIAL-VALUE-AT-PADDING-BOUNDARY",
        title="special or tied value in a window touching a padding boundary",
        description=(
            "a special value (-inf, a pad-constant element, a tied maximum) "
            "located in the window nearest the padded edge interacts with "
            "index computation in ways that testing 'special values' and "
            "'index boundaries' separately would miss: the max may equal the "
            "padding constant yet must still be reported as coming from the "
            "input tensor"
        ),
        structural_features=("sliding-window", "padding", "index-output"),
        origin=(
            "MaxPool cross-domain dependency check; confirmed by the "
            "specification's documented discrepancies in an existing "
            "implementation (wrong Y and Indices pointing into the padding)"
        ),
        action=(
            "add a combined boundary case per operator: a special/tied value "
            "in the window at the padding boundary, distinct from the "
            "value-only and boundary-only tests"
        ),
    ),
    CatalogPattern(
        id="TIE-BREAK-INDEX",
        title="ties need explicit argmax tie-break tests",
        description=(
            "when an operator outputs the index of a selected element "
            "(argmax-style Indices), every window or reduction containing "
            "the winning value more than once makes the output depend on the "
            "spec's tie-break rule, which is invisible with distinct values"
        ),
        structural_features=("index-output",),
        origin="MaxPool symmetry/property analysis (all-ties and duplicate-maxima cases)",
        action=(
            "materialize at least one all-ties case and one duplicate-maxima "
            "case exercising the first/last position of a window"
        ),
    ),
    CatalogPattern(
        id="OUTPUT-SIZE-ATTRIBUTE-COUPLING",
        title="attributes combined in one output-size formula must be boundary-tested jointly",
        description=(
            "when an operator's output size multiplies, divides or otherwise "
            "combines several attributes in a single formula (kernel shape, "
            "strides, pads, dilations), boundary values of those attributes "
            "must be tested in combination, not only individually at their "
            "own boundaries"
        ),
        structural_features=("output-size-formula",),
        origin="strategy cross-domain dependency check, item 3",
        action=(
            "keep the coupling attributes in one shared predicate set so the "
            "boundary/interior decomposition enumerates their joint classes"
        ),
    ),
    CatalogPattern(
        id="ACCUMULATOR-RANGE-VS-SIZE",
        title="accumulator range couples with tensor size",
        description=(
            "a combining operation whose accumulator range depends on both "
            "the number of elements combined and their values cannot be "
            "tested independently on the size and value axes: the combined "
            "boundary case (max size x max value, or the size threshold at "
            "which overflow becomes possible for boundary values) must be "
            "added explicitly"
        ),
        structural_features=("reduction-accumulator",),
        origin="strategy cross-domain dependency check, item 1",
        action=(
            "add the combined size x value boundary case for every "
            "accumulating operator"
        ),
    ),
    CatalogPattern(
        id="NAN-OUTSIDE-ORDERED-SPLIT",
        title="NaN falls outside every ordered p'/p'' pair",
        description=(
            "under IEEE 754, NaN satisfies neither p' nor p'' of an ordered "
            "domain predicate; unifying an ordered float predicate with a "
            "special-value predicate over the same variable silently drops "
            "NaN unless it is added as an explicit sibling equality predicate"
        ),
        structural_features=("ordered-float-domain",),
        origin="strategy caution on unifying ordered and special-value predicates",
        action=(
            "for every float variable under ordered comparison, add an "
            "explicit NaN class (the generator enforces this via "
            "dsl.nan_sibling_violations)"
        ),
    ),
]


def get(pattern_id: str) -> CatalogPattern:
    """Look up a catalog entry by ID."""
    for p in PATTERNS:
        if p.id == pattern_id:
            return p
    raise KeyError(f"unknown catalog pattern {pattern_id!r} "
                   f"(known: {', '.join(p.id for p in PATTERNS)})")


def for_features(*features: str) -> List[CatalogPattern]:
    """All patterns whose structural features intersect ``features`` --
    the re-review set of an operator carrying those features."""
    return [p for p in PATTERNS
            if set(p.structural_features) & set(features)]