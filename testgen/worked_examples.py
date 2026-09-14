"""The strategy document's worked examples, for machinery validation ONLY.

These operator definitions transcribe the worked examples of the test
strategy (Add on uint8, Div on int8, Relu/Abs on float) so the
class-enumeration machinery can be validated against the expected results
stated in the strategy document (e.g. Add/uint8 must yield exactly 9
classes: 4 vertices, 4 open edges, 1 interior).

They have NO specification file under specs/ and are therefore NEVER
generated into tests/ -- the generator refuses to produce tests for
operators without a specification (user requirement). Use selfcheck.py to
run them as internal validation.
"""

from __future__ import annotations

from .dsl import (
    Branch,
    FloatVar,
    IntVar,
    Operator,
    add,
    eq,
    fcls,
    fgt,
    flt,
    fne,
    ge,
    le,
    ne,
)

_x1 = IntVar("x1", 0, 255, role="A[0]: first scalar operand (uint8)")
_x2 = IntVar("x2", 0, 255, role="B[0]: second scalar operand (uint8)")

_ADD_TYPE = [
    ge(_x1, 0, origin="TYPE-x1-lo", note="uint8 lower bound"),
    le(_x1, 255, origin="TYPE-x1-hi", note="uint8 upper bound"),
    ge(_x2, 0, origin="TYPE-x2-lo", note="uint8 lower bound"),
    le(_x2, 255, origin="TYPE-x2-hi", note="uint8 upper bound"),
]

add_uint8 = Operator(
    name="Add_uint8",
    spec_file="specs/add.md",  # does not exist: never generated
    spec_ref="strategy worked example (Add on uint8 scalars)",
    variables=[_x1, _x2],
    domain=list(_ADD_TYPE),
    constraints=list(_ADD_TYPE),
    notes=[
        "Expected per the strategy: 9 classes (4 corners, 4 open edges, "
        "1 interior) out of the 16 sign combinations.",
    ],
)

add_uint8_no_overflow = Operator(
    name="Add_uint8_no_overflow",
    spec_file="specs/add.md",  # does not exist: never generated
    spec_ref="strategy worked example, output-type variant",
    variables=[_x1, _x2],
    domain=list(_ADD_TYPE),
    constraints=_ADD_TYPE + [
        le(add(_x1, _x2), 255,
           origin="FUNC-overflow",
           note="output type constraint: the sum must stay representable "
                "in uint8"),
    ],
)

_dx = IntVar("x", -128, 127, role="A[0]: dividend (int8)")
_dy = IntVar("y", -128, 127, role="B[0]: divisor (int8)")

div_int8 = Operator(
    name="Div_int8",
    spec_file="specs/div.md",  # does not exist: never generated
    spec_ref="strategy worked example (Div on int8 scalars)",
    variables=[_dx, _dy],
    domain=[
        ge(_dx, -128, origin="TYPE-x-lo", note="int8 lower bound"),
        le(_dx, 127, origin="TYPE-x-hi", note="int8 upper bound"),
        ge(_dy, -128, origin="TYPE-y-lo", note="int8 lower bound"),
        le(_dy, 127, origin="TYPE-y-hi", note="int8 upper bound"),
        ne(_dy, 0, origin="FUNC-div-by-zero",
           note="division by zero is a domain hole, not a valid input"),
    ],
    constraints=[
        ge(_dx, -128, origin="TYPE-x-lo"),
        le(_dx, 127, origin="TYPE-x-hi"),
        ge(_dy, -128, origin="TYPE-y-lo"),
        le(_dy, 127, origin="TYPE-y-hi"),
    ],
    branches=[
        Branch(name="divisor adjacent to the domain hole, below (y = -1)",
               predicates=[eq(_dy, -1)], origin="FUNC-div-by-zero",
               kind="boundary"),
        Branch(name="divisor adjacent to the domain hole, above (y = 1)",
               predicates=[eq(_dy, 1)], origin="FUNC-div-by-zero",
               kind="boundary"),
    ],
)

_rx = FloatVar("X", 32, role="X[0]: scalar operand (float32)")

relu_f32 = Operator(
    name="Relu_f32",
    spec_file="specs/relu.md",  # does not exist: never generated
    spec_ref="strategy discontinuity example (Relu threshold at 0)",
    variables=[_rx],
    domain=[],
    constraints=[],
    branches=[
        Branch(name="X is NaN", predicates=[fcls(_rx, "nan")],
               origin="E_RELU_FUNC_NAN"),
        Branch(name="X is +inf", predicates=[fcls(_rx, "+inf")],
               origin="E_RELU_FUNC_INF"),
        Branch(name="X is -inf", predicates=[fcls(_rx, "-inf")],
               origin="E_RELU_FUNC_INF"),
        Branch(name="X is -0", predicates=[fcls(_rx, "-0")],
               origin="E_RELU_FUNC_ZERO"),
        Branch(name="X is +0 (threshold)", predicates=[fcls(_rx, "+0")],
               origin="E_RELU_FUNC_THRESHOLD"),
        Branch(name="X finite negative", predicates=[flt(_rx, 0)],
               origin="E_RELU_FUNC_BRANCH"),
        Branch(name="X finite positive", predicates=[fgt(_rx, 0)],
               origin="E_RELU_FUNC_BRANCH"),
    ],
)

_ax = FloatVar("X", 32, role="X[0]: scalar operand (float32)")

abs_f32 = Operator(
    name="Abs_f32",
    spec_file="specs/abs.md",  # does not exist: never generated
    spec_ref="strategy Abs example: the piecewise branches are the classes",
    variables=[_ax],
    domain=[],
    constraints=[],
    branches=[
        Branch(name="X is NaN", predicates=[fcls(_ax, "nan")],
               origin="E_ABS_FUNC_010"),
        Branch(name="X is +inf", predicates=[fcls(_ax, "+inf")],
               origin="E_ABS_FUNC_020"),
        Branch(name="X is -inf", predicates=[fcls(_ax, "-inf")],
               origin="E_ABS_FUNC_030"),
        Branch(name="X is -0", predicates=[fcls(_ax, "-0")],
               origin="E_ABS_FUNC_040"),
        Branch(name="X is +0", predicates=[fcls(_ax, "+0")],
               origin="E_ABS_FUNC_050"),
        Branch(name="X finite negative",
               predicates=[flt(_ax, 0), fne(_ax, "-inf")],
               origin="E_ABS_FUNC_060"),
        Branch(name="X finite positive",
               predicates=[fgt(_ax, 0), fne(_ax, "+inf")],
               origin="E_ABS_FUNC_070"),
    ],
)

WORKED_EXAMPLES = [add_uint8, add_uint8_no_overflow, div_int8,
                   relu_f32, abs_f32]