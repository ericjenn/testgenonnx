# SONNX test strategy — understanding and improvements

This document records my understanding of the test strategy described in
`guidelines/strategy.md`, proposes improvements, and defines the
scope of the implementation in this repository (the equivalence-class-based
strategy only).

---

## 1. My understanding of the strategy

### 1.1 Objectives

Tests serve two distinct objectives:

1. **Validating the specification** — by comparing the SONNX informal
   specification against trusted existing implementations (e.g. ONNX Runtime).
   A discrepancy reveals a defect *somewhere* (spec, implementation, or both),
   but does not by itself locate the fault. This has already proven fruitful.
2. **Verifying implementations** — the SONNX reference implementation acts as
   the test oracle for third-party implementations. Passing the tests is
   *necessary evidence*, not a guarantee of conformance; in a certification
   context the conformance demonstration remains the applicant's
   responsibility.

Tests are essentially **functional** (what the operator does, not how it is
implemented). Implementation-based tests are allowed but must be explicitly
justified. Every test must be **traceable** to a spec section or requirement
tag (e.g. `E_DIV_REAL_FUNC_010`).

### 1.2 The test space

An operator input is a vector consisting of, for every input tensor *T*: its
shape $(dT_0,\dots,dT_{n-1})$ (its "structure") and all its element values
$T[i]$. Attributes contribute further scalar/vector values. Each element of
this vector is one **dimension of the test space**, and the test selection
problem is to choose a small subset of this space that maximizes fault-revealing
power.

### 1.3 Equivalence classes — the core idea

Exhaustive testing is impossible, so inputs are partitioned into classes such
that two inputs in the same class are *expected* to reveal the same faults
(this is a heuristic: the true relation depends on an unknown fault model, so
the working definition is "same coverage of the functional behaviour"). The
document defines class membership through several contributing criteria:

- same **boundary** of the input domain,
- same **branch/condition** of the functional specification,
- same **special values** (neutral, absorbing, dominating; discontinuities),
- same classes w.r.t. any **sub-operator** used in the definition.

### 1.4 Constructing classes from the input domain (the p′/p″ algebra)

This is the mechanical part of the strategy, and the part this repository
implements:

1. **Define the domain.** Collect the set of predicates
   $P = \{p_1,\dots,p_m\}$ constraining the inputs:
   - *type constraints* (e.g. $x \in [0,255]$ for uint8),
   - *functional constraints* from the specification (e.g. the MaxPool
     output-size inequalities coupling `strides`, `dilations`, `kernel_shape`,
     `pads`),
   - *usage limits* for variables whose type bounds are physically unreachable
     (rank, dimension sizes — see §1.6).
   The domain is the conjunction $p_1 \wedge \dots \wedge p_m$.

2. **Rewrite strict inequalities into non-strict ones.**
   - Integer domain: $x < c$ becomes $x \le c-1$.
   - Floating-point domain: $x < c$ becomes $x \le \mathrm{nextdown}(c)$
     (largest float strictly below $c$ under round-to-nearest-ties-to-even).

3. **Split every inequality predicate into a boundary and an interior version:**
   - $p'_i$: replace $\le$/$\ge$ by $=$ (**on the boundary**),
   - $p''_i$: replace $\le$ by $<$, $\ge$ by $>$ (**strictly inside**),
   - equality predicates are kept as-is (they only have a $p'$ form — an
     equality constraint has no interior).

4. **Enumerate the disjunctive normal form.** Since
   $p_i \Leftrightarrow p'_i \vee p''_i$, the domain decomposes into
   $2^m$ sub-domains, one per choice of $p'$ or $p''$ for each predicate.
   Each *satisfiable* member is one **equivalence class**; the all-$p'$
   member is the most constrained (e.g. a corner of a rectangle), the
   all-$p''$ member is the interior.

5. **Instantiate one representative per class.** The document notes this
   becomes tedious quickly and suggests a constraint solver (Z3 is mentioned).

**Worked example (Add on uint8).** With the four type constraints
($x_1 \ge 0$, $x_1 \le 255$, $x_2 \ge 0$, $x_2 \le 255$), most of the 16
combinations are unsatisfiable (e.g. $x_1=0$ and $x_1=255$ simultaneously);
the 9 surviving classes are the 4 vertices, the 4 open edges, and the open
interior — the classic boundary-value analysis picture, but *derived*
rather than hand-drawn.

### 1.5 Value domains beyond type bounds

- **Floats**: NaN, ±inf, ±0, subnormals (min/max positive, per float width),
  normals (min/max positive). Subnormals matter for underflow, degraded
  relative precision, platform-dependent handling, and comparison/branching
  corner cases.
- **Integers**: min/max of the considered type.
- **Structure**: rank 0 (scalar), 1 (vector), 2 (matrix), ≥3; sizes: zero
  dimensions (empty tensors), degenerate shapes (1×1, 1×n, n×1), ordinary
  sizes.
- **Special values from operator semantics**: neutral elements (0 for Add,
  1 for Mul/Div-denominator, −inf for Max), absorbing elements (0 for Mul,
  +inf for Max), dominating values, and **discontinuities** (division by
  zero, the Relu threshold at 0, Abs's piecewise definition enumerating
  NaN/±inf/±0/negative/positive). The Abs example shows the second use of
  the machinery: the branches of the informal definition *are* the
  equivalence classes.

### 1.6 Unbounded variables

Ranks and dimension sizes are bounded only by the type ($2^{64}-1$), which is
physically untestable. The strategy fixes arbitrary **usage limits**
(e.g. rank ≤ 4, size ≤ 1000) and treats them exactly like real domain
constraints. These define the domain in which the operator is *guaranteed*
to behave correctly, and should ideally be stated explicitly in the
specification.

### 1.7 Pruning and documentation duties

Two elimination mechanisms:

1. **Unsatisfiable combinations** — eliminated for free by the solver
   (or statically, e.g. left-bound and right-bound predicates of an interval
   can never hold simultaneously, except for singleton intervals).
2. **Non-pertinent combinations** — judged not to reveal a distinct fault.
   Because this is an engineering judgment, every exclusion must be recorded
   with: the combination, the **concrete fault model assumed absent** (stated
   precisely enough that a reviewer could disagree), and a disposition
   (`pruned`, `covered-by` an existing test ID, or `pairwise-coverage` when
   the case is kept or dropped purely to satisfy a covering-array obligation,
   without an individually stated fault model).

### 1.8 Cross-domain dependency check

Before decomposition, the designer must check four candidate families of
coupling that independent per-dimension classes would miss:

1. **Accumulation**: if a combining operation's range depends on *both* the
   number of elements and their values (conv, avg-pool, matmul), size and value
   domains cannot be tested independently — combined boundary cases (max size
   × max value) must be added.
2. **Index computation**: if output positions are computed from structural
   parameters (MaxPool indices), special values (NaN, ties, ±inf) may interact
   with structural boundaries (window at the edge) — add the combined case.
3. **Output-size formula coupling**: when several attributes are combined in
   one formula (MaxPool's `kernel × dilations + pads` vs `strides`), their
   boundaries must be tested in combination, not only individually.
4. **Broadcasting vs. special values**: for operators that broadcast, a
   broadcast-from-1 axis combined with a special value on the size-1 operand
   may produce a distinct failure mode — add the combined case.

The outcome is a per-operator **dependency table** (candidate dependency →
found? → combined test added?), whose rows cite the shared cross-operator
pattern catalog so that a pattern discovered on one operator flags every
operator sharing the relevant structural feature for re-review.

### 1.9 Explicit symmetry/asymmetry

Relations between parameters (e.g. symmetric vs asymmetric `pads`) must be
tested *deliberately*, even if boundary enumeration fortuitously covers them.

### 1.10 Completeness criterion

A test set is complete relative to an operator spec iff it covers every
equivalence class built upon: (1) data types, (2) tensor shapes, (3)
broadcasting, (4) pre-conditions/constraints, (5) the functional
specification, and (6) important implementation-risk patterns.

### 1.11 Out of the implemented scope

- **Property-based "additional tests"** (commutativity, linearity, rotation/
  translation/scaling invariance) — a separate strategy, not implemented here.
- **Index-plausibility discipline** ("values and indexes…"): test data must be
  designed so that each expected output value identifies its source input
  position (e.g. avoid all-zero operands for Mul, which masks mis-indexing).
  Not a generator feature per se, but a data-design constraint I adopt below
  as an improvement (§2.6).

---

## 2. Proposed improvements

1. **Machine-readable pruning rationale.** The justification table for pruned
   combinations should be *data consumed by the generator*, not just prose:
   the operator definition carries `(combination, absent-fault-model,
   disposition)` records, the generator emits them into the test manifest, and
   the manifest cross-references the test IDs that `covered-by` cites. This
   makes the judgment reviewable and diff-able. (Implemented.)

2. **Solver-native strict inequalities for floats.** Rewriting
   $x_1+x_2 < x_3$ to $x_1+x_2 \le \mathrm{nextdown}(x_3)$ only works when the
   right-hand side is a *constant*. With a solver, keep strict predicates
   native (the split $p'/p''$ does not need the rewrite: $p'$ is $<$-boundary
   turned $=$ only when the boundary is expressible). For integer variables
   the rewrite stays useful because it makes the boundary decidable as an
   equality. (Implemented: integer rewrite automatic; float strict
   comparisons kept native.)

3. **Witness quality objective.** A raw solver model may return awkward
   values (e.g. $x_1=99$ when $x_1=1$ also witnesses the class). Add a
   soft preference for small, human-readable witnesses, since reviewers
   must be able to eyeball the test vectors. (Implemented: greedy per-variable
   optimization pass over boundary-anchored candidate values, with an opt-out
   pushing tensor-size variables toward their usage maximum so boundaries are
   exercised on non-degenerate tensors. *Note:* the strategy now prescribes
   the rounded arithmetic midpoint of the feasible interval as the default
   representative for integer interior classes; the small-witness preference
   is a recorded deviation from that rule, to be aligned as future work —
   with the `prefer_large` opt-out recorded as operator data when it is.)

4. **Usage limits as first-class, named constraints.** Rather than ad-hoc
   bounds, each usage limit should carry an identifier and rationale so the
   "guaranteed correct" envelope is auditable and adjustable in one place
   (e.g. `max_rank=4`, `max_dim=1000`). (Implemented.)

5. **Combinatorial explosion control for float classes.** The float value
   domain has ~7 classes per element (NaN, ±inf, ±0, subnormal, normal); for
   an $n$-element test tensor even a single dimension explodes. In the
   strategy's unify-then-prune frame: the value-level dimension is unified
   with the structural classes through per-case value templates (each
   template expresses, as data, which value-level interaction matters —
   index observability, ties, the padding constant), and the full
   value-level cross-product is *pruned* with recorded rationale: T2–T4 are
   restricted to the auxiliary cases because structural interior classes
   add no expected value-level fault. (Implemented; the templates and the
   restriction rationale are recorded in the manifest and report, and the
   restriction is validated by mutation testing — the tie-break mutant is
   killed only by the retained T2/T4 tests.)

6. **Index observability as a data constraint.** Adopt the document's
   "values and indexes" rule in the generator: when materializing a tensor
   for a class witness, default to distinct element values derived from
   position (e.g. small distinct primes or $i \mapsto$ distinct value), so
   that misplaced outputs are detectable. (Implemented for element-wise
   materialization.)

7. **Determinism.** Generated tests must be reproducible byte-for-byte:
   no wall-clock, no random seeds without pinning, stable ordering of
   classes. (Implemented.)

8. **Manifest as an audit artifact.** The generator emits, per operator:
   total combinations, per-predicate split, satisfiable classes with
   witnesses, UNSAT combinations, pruned combinations with rationale, and
   the dependency table. This makes the §1.10 completeness claim checkable
   mechanically rather than by trust. (Implemented.)

---

## 3. Implementation scope

Implemented here: **equivalence-class-based test generation only**, per the
task instruction — and **specification-gated**: tests are generated only for
operators backed by an informal specification file under `specs/`. The CLI
*refuses* to generate tests for an operator without a spec (exit code 2).

### 3.1 Architecture

- `testgen/dsl.py` — DSL to declare an operator: typed integer variables with
  named usage limits (each bound carrying its fault-model justification) and
  float variables over interesting IEEE-754 value classes, predicates
  (each labelled with the test-completeness criterion it serves),
  branch/boundary/property cases, pruning records
  (combination, absent-fault-model, disposition — `pruned` /
  `covered-by` / `pairwise-coverage`), a dependency table citing shared
  catalog patterns, and the NaN sibling-predicate validation.
- `testgen/catalog.py` — the shared, living cross-operator pattern catalog:
  interaction patterns (special value at a padding boundary, tie-break,
  output-size attribute coupling, accumulator-vs-size, NaN-outside-ordered-
  split) with their structural features and origin, cited by operator
  definitions and emitted into every manifest/report.
- `testgen/splitting.py` — the p′/p″ algebra of §1.4: integer strict-inequality
  rewriting ($x < E \to x \le E-1$, applied symbolically), splitting into
  boundary/interior forms (equality predicates keep only p′; `≠` predicates
  only p″), DNF enumeration.
- `testgen/solver.py` — Z3-backed satisfiability checking with a two-phase
  witness extraction (SAT, then greedy minimization toward small readable
  witnesses — with an opt-out pushing tensor-size variables toward the usage
  maximum so structural boundaries are exercised on non-degenerate tensors).
  An enumeration backend without Z3 serves as a degraded fallback.
- `testgen/classes.py` — DNF enumeration with pruning-record elimination and
  duplicate-witness bookkeeping; auxiliary cases are solved against the
  domain *and* the functional constraints (a boundary witness must be a valid
  operator configuration).
- `testgen/evaluate.py` — an **independent verification path**: every witness
  is re-checked by direct arithmetic evaluation (Python `Fraction`s), not by
  the solver.
- `testgen/operators.py` — the catalog: currently **MaxPool** only, transcribed
  from `specs/maxpool.md` with the spec's real constraint tags
  (pads-C2/Keff_less_than_pads per side, Y-C1/shape_consist fits/overflows per
  axis, restrictions R1–R5) so every test is traceable to its spec anchor.
- `testgen/reference.py` — reference implementation of the informal spec
  (padded max, row-major tie-break over input positions, literal floor output
  -size formula).
- `testgen/materialize.py` — functional test materialization via value
  templates: T1 distinct position-derived values (index observability, §2.6);
  T2 all-ties (Indices tie-break); T3 pad-constant tensor (the configuration
  where the spec documents ONNX Runtime non-compliances); T4 duplicate maxima.
  T2–T4 are restricted to auxiliary cases with the rationale recorded
  (pruning discipline §1.7). Configurations the spec leaves undefined are
  *refused* and recorded as specification gaps (§3.3).
- `testgen/emit.py` — per-dtype emission: `tests/MaxPool_<dtype>/` with one
  JSON file per functional test, a machine-readable `manifest.json`, and a
  detailed human-readable `report.md` (constraint table, class inventory with
  witnesses, UNSAT, pruned with rationale, dependency table, spec gaps).
  Every class and test carries class-provenance metadata (criterion,
  retention reason) and a risk-based prioritization tier; the emitted test
  list is ordered by tier, and usage limits are recorded with their
  justifications and the base-choice pruning record of their cross-product.
- `testgen/mutate.py` — mutation testing harness: plausible implementation
  faults (index off-by-one, inverted tie-break, swapped strides/dilations,
  wrong output-size rounding, dilation ignored in the output size) are
  injected into the reference implementation and run against every retained
  test; a surviving mutant invalidates the pruning or retention decision it
  probes. Run with `python -m testgen.cli --mutate`; results are merged into
  each manifest and report.
- `testgen/cli.py` — spec-file gating, witness verification, per-dtype
  generation (one independent test set per data type of the spec), `tests/index.json`.
- `testgen/selfcheck.py` — the strategy document's worked examples (Add uint8
  → 9 classes with 7 UNSAT, Div int8 domain hole, Relu/Abs piecewise branches)
  are encoded in `testgen/worked_examples.py` but **never generated** (they
  have no spec files); they are used exclusively to validate the class
  enumeration machinery against the document's expected results.

### 3.2 What is generated (MaxPool, per the spec)

The spec's data types are generated **separately** (one test set each):
`float16`, `float`, `double`, `int8`, `uint8`, each with its own
spec-mandated padding constant (−inf, −inf, −inf, minint8, 0). Per dtype:
144 equivalence classes from the 8-predicate split system, 15 auxiliary
boundary/property cases, 112 combinations pruned by two documented pruning
records, 194 functional tests — every expected output verified twice (by
`evaluate.py` at generation time and by an independent brute-force oracle).
The retained test set additionally kills all five mutation-testing mutants
per dtype, validating the pruning records and the T2–T4 retention
decision.

### 3.3 Specification findings (test objective 1: validating the spec)

- **Padding-only windows under dilation.** With `dilations ≥ 2`, a window can
  select *only* padding elements while pads-C2 holds (e.g. `dX3 = 1`,
  `dilations[1] = 2` samples padded columns 0 and 2 while X lives in column 1).
  The spec's rationale for pads-C2 — "guarantees that the max value returned
  by MaxPool belongs to X" — does not hold there, and neither Y nor Indices
  is defined. 28 such valid configurations were refused materialization and
  recorded as specification gaps in the manifests/reports for spec
  clarification.
- **NaN semantics of Max_F undefined.** The spec's floating-point section
  changes Max to Max_F and the padding constant but never defines NaN
  behaviour. NaN-bearing inputs are deliberately not generated; the gap is
  reported rather than papered over with assumed semantics.

Not implemented (future work): property-based/metamorphic tests (§1.11),
broadcasting classes, sub-operator recursion, and actual execution against
the SONNX reference implementation or ONNX Runtime as oracle.