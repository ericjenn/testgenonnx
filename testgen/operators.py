"""Operator catalog.

Only operators backed by an informal specification file under ``specs/``
are eligible for test generation -- the CLI *refuses* to generate tests
for anything else. The current catalog:

- **MaxPool** (specs/maxpool.md): functional tests generated separately
  for each data type of the spec (float16, float, double, int8, uint8).

The strategy's own worked examples (Add uint8, Div int8, Relu/Abs float)
have no specification files and are therefore NOT generated; they are
kept in ``worked_examples.py`` solely to validate the class-enumeration
machinery (see selfcheck.py), which is not test generation.
"""

from __future__ import annotations

from .dsl import (
    Branch,
    Dependency,
    IntVar,
    Operator,
    Pruning,
    add,
    eq,
    gt,
    le,
    mul,
    ne,
    sub,
)

# ---------------------------------------------------------------------------
# MaxPool (SONNX informal spec: specs/maxpool.md, based on ONNX MaxPool v22)
# ---------------------------------------------------------------------------
# Structural parameters only (values are handled by the functional
# materializer). The constraint system below transcribes the spec's
# constraints with their actual anchors/tags for traceability:
#   - pads C2 (anchor Keff_less_than_pads): Ke_i > max(pads[i], pads[i+2]),
#     encoded per side so the asymmetric boundary (Ke = pad+1 on one side
#     only) forms its own class;
#   - Y C1 (anchor shape_consist): output-size formula, encoded as the
#     pair of inequalities "last window fits / next window would not"
#     (ceil_mode = 0 per restriction R4);
#   - kernel_shape C1 / dilations C1 / strides C1: strictly positive
#     integers -- part of the background domain;
#   - restriction R1 (2 spatial axes, anchor C1ia): fixed rank-4 tensors
#     with batch = channel = 1 (structural classes over batch/channel are
#     out of scope of the spec's own constraint set; recorded in notes).

# Fault models assumed absent for the chosen usage bounds (strategy: "The
# case of unbounded variables" -- the bound value is an engineering choice
# a reviewer must be able to challenge, so it carries the same kind of
# justification as a pruning record). Shared per bound family.
_JUST_SIZE = (
    "no implementation pattern is expected to distinguish spatial sizes "
    "beyond 8 from sizes within 2..8: sizes enter only the output-size "
    "formula and the window bounds checks")
_JUST_KERNEL = (
    "no implementation pattern is expected to special-case kernels above "
    "3 elements: the window scan iterates over the kernel extent uniformly")
_JUST_STRIDE = (
    "no implementation pattern is expected to distinguish strides above "
    "3: the stride appears only in the linear window-origin computation "
    "and the output-size formula")
_JUST_PADS = (
    "no implementation pattern is expected to distinguish pads above 3: "
    "padding shifts the window origin uniformly and is bounded against "
    "the kernel extent by pads-C2")
_JUST_DILATION = (
    "no implementation pattern is expected to distinguish dilations above "
    "2: dilation enters only the sample offsets and the output-size "
    "formula, both linearly")
_JUST_OUTSIZE = (
    "the output size follows from the other attributes via Y-C1 "
    "(shape_consist); bounding it at 8 adds no configuration class "
    "beyond those the other limits already exercise")

# Test-completeness criteria served by each predicate / auxiliary case
# (strategy: "Test traceability"; recorded on every generated test).
_CRIT_PRECONDITIONS = "criterion 4: operator pre-conditions"
_CRIT_FUNCTIONAL = "criterion 5: functional specification"
_CRIT_PRECOND_BOUNDS = (_CRIT_PRECONDITIONS
                        + " (usage-domain / value-domain bounds)")
_CRIT_OUTSIZE = _CRIT_FUNCTIONAL + " (output-size boundary values)"
_CRIT_SYMMETRY = (_CRIT_FUNCTIONAL
                  + " (explicit symmetry property between parameters)")

_dX2 = IntVar("dX2", 1, 8, role="spatial size of X, axis 2 (first spatial axis)",
              usage_limit=True, justification=_JUST_SIZE, latex="dX_2")
_dX3 = IntVar("dX3", 1, 8, role="spatial size of X, axis 3 (second spatial axis)",
              usage_limit=True, justification=_JUST_SIZE, latex="dX_3")
_dW0 = IntVar("dW0", 1, 3, role="kernel_shape[0]",
              usage_limit=True, justification=_JUST_KERNEL, latex="k_0")
_dW1 = IntVar("dW1", 1, 3, role="kernel_shape[1]",
              usage_limit=True, justification=_JUST_KERNEL, latex="k_1")
_s0 = IntVar("s0", 1, 3, role="strides[0]",
             usage_limit=True, justification=_JUST_STRIDE, latex="s_0")
_s1 = IntVar("s1", 1, 3, role="strides[1]",
             usage_limit=True, justification=_JUST_STRIDE, latex="s_1")
_p0 = IntVar("p0", 0, 3, role="pads[0] (begin, axis 2)",
             usage_limit=True, justification=_JUST_PADS, latex="p_0")
_p1 = IntVar("p1", 0, 3, role="pads[1] (begin, axis 3)",
             usage_limit=True, justification=_JUST_PADS, latex="p_1")
_p2 = IntVar("p2", 0, 3, role="pads[2] (end, axis 2)",
             usage_limit=True, justification=_JUST_PADS, latex="p_2")
_p3 = IntVar("p3", 0, 3, role="pads[3] (end, axis 3)",
             usage_limit=True, justification=_JUST_PADS, latex="p_3")
_d0 = IntVar("d0", 1, 2, role="dilations[0]",
             usage_limit=True, justification=_JUST_DILATION, latex=r"\delta_0")
_d1 = IntVar("d1", 1, 2, role="dilations[1]",
             usage_limit=True, justification=_JUST_DILATION, latex=r"\delta_1")
_dY2 = IntVar("dY2", 0, 8, role="spatial size of Y, axis 2",
              usage_limit=True, justification=_JUST_OUTSIZE, latex="dY_2")
_dY3 = IntVar("dY3", 0, 8, role="spatial size of Y, axis 3",
              usage_limit=True, justification=_JUST_OUTSIZE, latex="dY_3")

# Dilated kernel extents: Ke_i = dilations[i]*(kernel_shape[i]-1)+1
_K0 = add(mul(_d0, sub(_dW0, 1)), 1)
_K1 = add(mul(_d1, sub(_dW1, 1)), 1)
# Padded extents per axis: S_i = dX_(i+2) + pads[begin] + pads[end]
_S0 = add(_dX2, _p0, _p2)
_S1 = add(_dX3, _p1, _p3)

_MP_VARS = [_dX2, _dX3, _dW0, _dW1, _s0, _s1, _p0, _p1, _p2, _p3,
            _d0, _d1, _dY2, _dY3]

_MP_DOMAIN = (
    # spec C1 value-domain constraints ("strictly positive lists", "pads >= 0")
    # + usage limits for the physically unreachable upper type bounds
    # (strategy: "The case of unbounded variables" -- usage limits are
    # handled like domain constraints and are named for auditability).
    [le(1, _dX2, origin="USAGE-X-size", note="usage limit: max spatial size 8"),
     le(_dX2, 8, origin="USAGE-X-size", note="usage limit: max spatial size 8"),
     le(1, _dX3, origin="USAGE-X-size", note="usage limit: max spatial size 8"),
     le(_dX3, 8, origin="USAGE-X-size", note="usage limit: max spatial size 8"),
     le(1, _dW0, origin="kernel_shape-C1", note="strictly positive (spec)"),
     le(_dW0, 3, origin="USAGE-kernel", note="usage limit on kernel size"),
     le(1, _dW1, origin="kernel_shape-C1", note="strictly positive (spec)"),
     le(_dW1, 3, origin="USAGE-kernel", note="usage limit on kernel size"),
     le(1, _s0, origin="strides-C1", note="strictly positive (spec)"),
     le(_s0, 3, origin="USAGE-strides", note="usage limit on stride"),
     le(1, _s1, origin="strides-C1", note="strictly positive (spec)"),
     le(_s1, 3, origin="USAGE-strides", note="usage limit on stride"),
     le(0, _p0, origin="USAGE-pads", note="pads are non-negative (spec)"),
     le(_p0, 3, origin="USAGE-pads", note="usage limit on pad"),
     le(0, _p1, origin="USAGE-pads", note="pads are non-negative (spec)"),
     le(_p1, 3, origin="USAGE-pads", note="usage limit on pad"),
     le(0, _p2, origin="USAGE-pads", note="pads are non-negative (spec)"),
     le(_p2, 3, origin="USAGE-pads", note="usage limit on pad"),
     le(0, _p3, origin="USAGE-pads", note="pads are non-negative (spec)"),
     le(_p3, 3, origin="USAGE-pads", note="usage limit on pad"),
     le(1, _d0, origin="dilations-C1", note="strictly positive (spec)"),
     le(_d0, 2, origin="USAGE-dilations", note="usage limit on dilation"),
     le(1, _d1, origin="dilations-C1", note="strictly positive (spec)"),
     le(_d1, 2, origin="USAGE-dilations", note="usage limit on dilation"),
     le(0, _dY2, origin="USAGE-Y-size", note="usage limit on output size"),
     le(_dY2, 8, origin="USAGE-Y-size", note="usage limit on output size"),
     le(0, _dY3, origin="USAGE-Y-size", note="usage limit on output size"),
     le(_dY3, 8, origin="USAGE-Y-size", note="usage limit on output size")]
)

# The p'/p'' split system: the functional constraints that couple several
# structural parameters. Predicate numbering (used by pruning records):
#   1: pads-C2 axis 2, begin side   (Ke_0 > pads[0])
#   2: pads-C2 axis 2, end side     (Ke_0 > pads[2])
#   3: pads-C2 axis 3, begin side   (Ke_1 > pads[1])
#   4: pads-C2 axis 3, end side     (Ke_1 > pads[3])
#   5: Y-C1 axis 2, last window fits
#   6: Y-C1 axis 2, next window overflows
#   7: Y-C1 axis 3, last window fits
#   8: Y-C1 axis 3, next window overflows
_MP_CONSTRAINTS = [
    gt(_K0, _p0, origin="pads-C2[axis2-begin]",
                 criterion=_CRIT_PRECONDITIONS,
       note="spec pads C2 (anchor Keff_less_than_pads): the effective "
            "kernel extent exceeds the begin pad of axis 2"),
    gt(_K0, _p2, origin="pads-C2[axis2-end]",
                 criterion=_CRIT_PRECONDITIONS,
       note="spec pads C2 (anchor Keff_less_than_pads): the effective "
            "kernel extent exceeds the end pad of axis 2"),
    gt(_K1, _p1, origin="pads-C2[axis3-begin]",
                 criterion=_CRIT_PRECONDITIONS,
       note="spec pads C2 (anchor Keff_less_than_pads): the effective "
            "kernel extent exceeds the begin pad of axis 3"),
    gt(_K1, _p3, origin="pads-C2[axis3-end]",
                 criterion=_CRIT_PRECONDITIONS,
       note="spec pads C2 (anchor Keff_less_than_pads): the effective "
            "kernel extent exceeds the end pad of axis 3"),
    le(add(mul(sub(_dY2, 1), _s0), _K0), _S0,
       origin="Y-C1[axis2-fits]",
       criterion=_CRIT_FUNCTIONAL,
       note="spec Y C1 (anchor shape_consist, ceil_mode=0): the last "
            "output window fits on axis 2"),
    gt(add(mul(_dY2, _s0), _K0), _S0,
       origin="Y-C1[axis2-overflows]",
       criterion=_CRIT_FUNCTIONAL,
       note="spec Y C1 (anchor shape_consist, ceil_mode=0): shifting the "
            "last window by one stride would overflow axis 2"),
    le(add(mul(sub(_dY3, 1), _s1), _K1), _S1,
       origin="Y-C1[axis3-fits]",
       criterion=_CRIT_FUNCTIONAL,
       note="spec Y C1 (anchor shape_consist, ceil_mode=0): the last "
            "output window fits on axis 3"),
    gt(add(mul(_dY3, _s1), _K1), _S1,
       origin="Y-C1[axis3-overflows]",
       criterion=_CRIT_FUNCTIONAL,
       note="spec Y C1 (anchor shape_consist, ceil_mode=0): shifting the "
            "last window by one stride would overflow axis 3"),
]

maxpool = Operator(
    name="MaxPool",
    spec_file="specs/maxpool.md",
    spec_ref="specs/maxpool.md -- SONNX MaxPool (ONNX MaxPool v22): "
             "restrictions R1-R5, constraints pads-C2 (Keff_less_than_pads), "
             "Y-C1 (shape_consist), kernel_shape-C1, dilations-C1, "
             "strides-C1, X-C1 (C1ia), Indices-C1",
    outputs=["Y", "Indices"],
    params={
        # One test set per data type supported by the spec (generated
        # separately, per user requirement). Padding constant per spec:
        # -inf for float16/float/double, minint8 for int8, 0 for uint8.
        "dtypes": ["float16", "float", "double", "int8", "uint8"],
        # Spec restrictions with fixed attribute values (R1-R5): prose
        # for the reports/manifest, owned by the operator definition.
        "restrictions": [
            "R1 (anchor C1ia): the input tensor has 2 spatial axes "
            "(rank 4; batch = channel = 1 in the materialized tests)",
            "R2/R3: `auto_pad = NOTSET` (explicit padding only)",
            "R4: `ceil_mode = 0` (floor output-size formula)",
            "R5: `storage_order = 0` (row-major)",
        ],
        # Prose notes about the notation used in the rendered predicates.
        "notation_notes": [
            "Named sub-quantities of the spec (e.g. the effective kernel "
            "extent $\\mathrm{Ke}_i = \\delta_i \\cdot (k_i - 1) + 1$) "
            "appear in expanded form in the predicates below.",
        ],
        # Push the input spatial sizes toward their usage maximum so
        # boundary classes are exercised on tensors where the boundary
        # matters (a stride of 3 on a 1x1 input would be vacuous).
        "prefer_large_vars": ["dX2", "dX3"],
        # Value templates flagged by the cross-domain dependency check
        # ("Output index vs. special/tied values"): they are the combined
        # boundary cases of prioritization tier 1 (named interaction risk).
        "tier1_templates": ["T3", "T4"],
    },
    variables=_MP_VARS,
    domain=_MP_DOMAIN,
    constraints=_MP_CONSTRAINTS,
    branches=[
        # -- usage-limit / value-domain boundary cases (base-choice) ------
        Branch(name="minimum spatial size on axis 2 ($dX_2 = 1$)",
               predicates=[eq(_dX2, 1)], origin="USAGE-X-size",
               criterion=_CRIT_PRECOND_BOUNDS,
               kind="boundary"),
        Branch(name="minimum spatial size on axis 3 ($dX_3 = 1$)",
               predicates=[eq(_dX3, 1)], origin="USAGE-X-size",
               criterion=_CRIT_PRECOND_BOUNDS,
               kind="boundary"),
        Branch(name="maximum usage-limit spatial size ($dX_2 = 8$)",
               predicates=[eq(_dX2, 8)], origin="USAGE-X-size",
               criterion=_CRIT_PRECOND_BOUNDS,
               kind="boundary"),
        Branch(name="minimum kernel ($k_0 = 1$)",
               predicates=[eq(_dW0, 1)], origin="kernel_shape-C1",
               criterion=_CRIT_PRECOND_BOUNDS,
               kind="boundary"),
        Branch(name="maximum kernel ($k_0 = 3$)",
               predicates=[eq(_dW0, 3)], origin="kernel_shape-C1",
               criterion=_CRIT_PRECOND_BOUNDS,
               kind="boundary"),
        Branch(name="maximum stride ($s_0 = 3$)",
               predicates=[eq(_s0, 3)], origin="strides-C1",
               criterion=_CRIT_PRECOND_BOUNDS,
               kind="boundary"),
        Branch(name="null pad, begin of axis 2 ($p_0 = 0$)",
               predicates=[eq(_p0, 0)], origin="USAGE-pads",
               criterion=_CRIT_PRECOND_BOUNDS,
               kind="boundary"),
        Branch(name="maximum pad, begin of axis 2 ($p_0 = 3$)",
               predicates=[eq(_p0, 3)], origin="USAGE-pads",
               criterion=_CRIT_PRECOND_BOUNDS,
               kind="boundary"),
        Branch(name=r"maximum dilation ($\delta_0 = 2$)",
               predicates=[eq(_d0, 2)], origin="dilations-C1",
               criterion=_CRIT_PRECOND_BOUNDS,
               kind="boundary"),
        Branch(name="empty output on axis 2 ($dY_2 = 0$): kernel within one "
                    "stride of fitting, the floor formula yields 0",
               predicates=[eq(_dY2, 0)], origin="Y-C1",
               criterion=_CRIT_OUTSIZE,
               kind="boundary"),
        Branch(name="single-element output on axis 2 ($dY_2 = 1$)",
               predicates=[eq(_dY2, 1)], origin="Y-C1",
               criterion=_CRIT_OUTSIZE,
               kind="boundary"),
        # -- explicit symmetry/asymmetry properties (strategy section) ------
        Branch(name="symmetric padding on axis 2 ($p_0 = p_2$)",
               predicates=[eq(_p0, _p2)],
               origin="pads-C2[axis2]",
               criterion=_CRIT_SYMMETRY,
               kind="property"),
        Branch(name=r"asymmetric padding on axis 2 ($p_0 \ne p_2$)",
               predicates=[ne(_p0, _p2)],
               origin="pads-C2[axis2]",
               criterion=_CRIT_SYMMETRY,
               kind="property"),
        Branch(name="symmetric padding on axis 3 ($p_1 = p_3$)",
               predicates=[eq(_p1, _p3)],
               origin="pads-C2[axis3]",
               criterion=_CRIT_SYMMETRY,
               kind="property"),
        Branch(name=r"asymmetric padding on axis 3 ($p_1 \ne p_3$)",
               predicates=[ne(_p1, _p3)],
               origin="pads-C2[axis3]",
               criterion=_CRIT_SYMMETRY,
               kind="property"),
    ],
    prunings=[
        Pruning(
            # p'_1 ∧ p'_2 : Ke_0 = pads[0]+1 and Ke_0 = pads[2]+1.
            choice={"1": "'", "2": "'"},
            fault_model_absent=(
                "an implementation that special-cases the minimal-overhang "
                "configuration (dilated kernel exactly one element wider "
                "than *both* pads of axis 2 simultaneously) is not "
                "expected: the overhang is governed by the uniform "
                "comparison Ke_0 > max(pads[0], pads[2]) applied "
                "independently to each side"
            ),
            disposition="pruned",
        ),
        Pruning(
            # p'_3 ∧ p''_4 : minimal overhang on the begin side of axis 3
            # only, everything else free.
            choice={"3": "'", "4": "''"},
            fault_model_absent=(
                "minimal overhang on the begin side of axis 3 combined with "
                "a non-minimal end-side overhang exercises the same "
                "per-side comparison as the asymmetric-padding property "
                "case on axis 3, which puts differing pads on the two "
                "sides of the same axis"
            ),
            disposition="covered-by PR-004",
        ),
    ],
    dependencies=[
        Dependency(
            candidate="Accumulator range vs. tensor size",
            found=False,
            combined_test="-",
        ),
        Dependency(
            candidate="Output index vs. special/tied values",
            found=True,
            catalog_refs=("SPECIAL-VALUE-AT-PADDING-BOUNDARY",
                          "TIE-BREAK-INDEX"),
            combined_test=(
                "combined boundary cases required by the strategy: the "
                "value templates T3 (tensor entirely filled with the "
                "padding constant, mirroring the spec's -inf / minint8 "
                "examples) and T4 (duplicate maxima within a window) put "
                "special and tied values in windows interacting with the "
                "padding boundary -- precisely the interaction where "
                "discrepancies were observed on ONNX Runtime (see spec "
                "'Discrepancies observed in an existing implementation')"
            ),
        ),
        Dependency(
            candidate="Output-size formula attribute coupling "
                      "(kernel_shape x strides x pads x dilations)",
            found=True,
            catalog_refs=("OUTPUT-SIZE-ATTRIBUTE-COUPLING",),
            combined_test=(
                "the Y-C1 predicates couple all four attributes in one "
                "formula per axis and are part of the p'/p'' split "
                "system, so their boundaries are tested in combination "
                "by construction"
            ),
        ),
        Dependency(
            candidate="Broadcasting vs. special values",
            found=False,
            combined_test=(
                "N/A: MaxPool has a single data input; broadcasting "
                "between inputs does not apply"
            ),
        ),
    ],
    notes=[
        "Restriction R1 (2 spatial axes, anchor C1ia) fixes the tensor "
        "rank at 4; batch and channel dimensions are pinned to 1 in the "
        "functional materialization -- the spec's own constraint set "
        "leaves them unconstrained, and no functional branch of the "
        "spec depends on them.",
        "Restrictions R2/R3/R4/R5 pin auto_pad=NOTSET, ceil_mode=0, "
        "storage_order=0: attributes with fixed values are constants in "
        "every generated test, not test dimensions.",
        "Indices-C1 (Indices has the same shape as Y) is verified "
        "implicitly: every expected Indices tensor is materialized with "
        "the same shape as Y.",
        "NaN semantics of Max_F are not defined by the spec (the "
        "floating-point section only changes Max to Max_F and the "
        "padding constant). This is a specification gap: NaN-bearing "
        "inputs are deliberately NOT generated and the gap is reported "
        "rather than papered over with assumed semantics.",
        "Tie-break rule (spec informal definition): when the max value "
        "is present several times, Indices holds the position minimizing "
        "$h \\cdot dX_3 + w$ in "
        "row-major order over the *input* tensor; the "
        "reference implementation materializes this rule exactly.",
    ],
)

# ---------------------------------------------------------------------------
# Catalog: spec-backed operators eligible for test generation
# ---------------------------------------------------------------------------

CATALOG = {"MaxPool": maxpool}