# testgen — design

This document explains the design of `testgen`: what it computes, which
module owns which responsibility, and the reasoning behind the decisions a
reviewer is most likely to challenge. The implemented method is the
equivalence-class-based strategy of
[`guidelines/strategy.md`](guidelines/strategy.md); this document references
its sections by name and records where the implementation deviates (each
deviation is also tracked in
[`guidelines/strategy-impact-analysis.md`](guidelines/strategy-impact-analysis.md)).

**Scope.** Only the equivalence-class strategy is implemented. Property-based /
metamorphic "additional tests", broadcasting-specific domains and sub-operator
recursion are explicitly future work (§ Known limitations). Tests are generated
only for operators backed by an informal specification under `specs/` — the CLI
*refuses* anything else with exit code 2 — and always **per data type**: one
independent test set per dtype of the spec.

---

## 1. The method in one paragraph

An operator's valid input space is described by a system of predicates
$P = \{p_1, \dots, p_m\}$ (type/usage constraints, functional constraints from
the spec). Every inequality predicate is split into a boundary form $p'_i$
(`<=`/`>=` turned into `=`) and an interior form $p''_i$ (strict); the domain
then decomposes into the DNF over all $2^m$ choices of $p'$ or $p''$ per
predicate. Each satisfiable member of the DNF is one **equivalence class**,
instantiated with one representative witness found by a solver (Z3).
Combinations that are unsatisfiable are eliminated; combinations judged
non-pertinent are **pruned with a recorded justification** (the fault model
assumed absent, and a disposition). Auxiliary cases — usage-limit boundaries,
explicit symmetry properties — are solved as-is against the domain. Every
retained class is materialized into concrete functional tests: an input tensor
built from a value template, and the expected output computed by a reference
implementation of the informal specification.

## 2. Generation pipeline

```
 operators.py  (operator definitions, DSL of dsl.py)
      |
      v
 classes.enumerate_classes
      |-- splitting.py   strict -> non-strict rewrite, p'/p'' split, DNF choices
      |-- solver.py      Z3 satisfiability + witness extraction (enumeration fallback)
      |
      v
 ClassifyResult  (classes, auxiliary cases, unsat, pruned)
      |
      |-- evaluate.py    independent re-verification of every witness (exact
      |                  Fraction arithmetic; generation aborts on failure)
      v
 emit.py
      |-- materialize.py value templates -> concrete input tensors
      |-- reference.py   test oracle: expected Y / Indices per the spec
      v
 tests/<Op>_<dtype>/{*.json, manifest.json, report.md}
      |
      v (with --mutate)
 mutate.py               inject faults into the oracle, run the retained
                         tests against each mutant, merge results
```

Concrete MaxPool figures (each dtype): 8 predicates in the split system → 256
combinations → **144 classes** (SAT), **0 unsatisfiable**, **112 pruned** by 2
recorded pruning records; plus **15 auxiliary cases** (11 usage-limit
boundaries, 4 symmetric/asymmetric-padding properties); materialized through 4
value templates → **194 functional tests**. Priorities: 28 tier 1, 165 tier 2,
1 tier 4 (the plain interior class).

## 3. Modules

| Module | Responsibility |
|---|---|
| `dsl.py` | The declaration language: typed variables (`IntVar`, `FloatVar`), arithmetic expressions, predicates with traceability (`origin`), note and completeness `criterion`; pruning records with disposition validation; dependency rows citing catalog patterns; auxiliary `Branch` cases; `nan_sibling_violations` (NaN check, § 5.4). |
| `domains.py` | Type-specific value domains (§ "Domains" of the strategy): the interesting IEEE-754 values per width (NaN, ±inf, ±0, subnormals, normals), integer min/max; stable renderings for JSON/reports. |
| `splitting.py` | The p′/p″ algebra: integer strict-inequality rewriting ($x < E \to x \le E - 1$, sound for symbolic integral $E$), boundary/interior split (equalities keep only p′, `!=` holes only p″, native float strict comparisons keep only p″), choice labels. |
| `solver.py` | Z3 translation (IntVars → Z3 Int; FloatVars → an integer index over their value classes) and two-phase witness extraction; enumeration fallback without Z3. |
| `classes.py` | DNF enumeration with pruning-record elimination, duplicate-witness bookkeeping, auxiliary-case solving; assigns class provenance (criterion, retention reason) and risk-based priority tiers. |
| `operators.py` | The operator catalog: MaxPool, transcribed from `specs/maxpool.md` with the spec's real constraint tags for traceability. Worked examples live separately in `worked_examples.py`. |
| `evaluate.py` | The independent verification path: every witness is re-checked by direct arithmetic evaluation (Python `Fraction`s), never by the solver. |
| `reference.py` | Reference implementations (test oracle), deliberately naive transcriptions of the informal spec definitions; also the padding constants per dtype. |
| `materialize.py` | Functional materialization via value templates T1–T4; refuses configurations the spec leaves undefined (specification gaps). |
| `emit.py` | Per-dtype emission: one JSON per test, `manifest.json` (audit artifact), `report.md` (human review). |
| `mutate.py` | Mutation testing: plausible faults injected into the oracle; a surviving mutant invalidates the pruning/retention claim it probes. |
| `catalog.py` | The shared, living cross-operator pattern catalog with structural-feature tags. |
| `cli.py` | Spec-file gating, orchestration, witness verification, `index.json`. |
| `selfcheck.py` | Validates the enumeration machinery against the strategy's worked examples (Add uint8 → exactly 9 classes with 7 UNSAT, Div int8 domain hole, Relu/Abs one class per branch). Not test generation. |

## 4. Key design decisions

### 4.1 Structure and values are two separate dimensions

The constraint system reasons over **structural parameters** (shapes, kernel,
strides, pads, dilations — integer variables); the *value* dimension of the
test space is handled at materialization time through per-case **value
templates**:

- **T1** distinct position-derived values — index observability ("On values
  and indexes": each expected output value identifies its source input
  position); applied to *every* class;
- **T2** all-ties, **T3** whole tensor = padding constant, **T4** duplicate
  maxima — restricted to the auxiliary cases.

The restriction of T2–T4 is itself a pruning decision with recorded rationale
(structural interior classes add no expected value-level fault) and is
*validated* by mutation testing: the tie-break mutant is killed only by the
retained T2/T4 tests. T3/T4 are the combined boundary cases required by the
cross-domain dependency check ("Output index vs. special/tied values") — the
configuration where the spec documents ONNX Runtime non-compliances — and are
assigned priority tier 1 via `params["tier1_templates"]`.

### 4.2 Usage limits are background theory + base choice, recorded as pruning

Structural variables are bounded by **usage limits**, not type bounds (rank
and sizes are physically untestable; § "The case of unbounded variables").
Naively adding 14 usage limits to the split system would give ~$2^{22}$
combinations instead of today's $2^8$. The design keeps usage limits in the
background domain (`IntVar` bounds + domain predicates) and covers their
boundaries through **base-choice auxiliary cases**: one variable at its bound,
all others free, solved against the domain *and* the functional constraint
system. Because the strategy demands every elimination be recorded, the
manifest emits this as a pruning record of the usage-limit boundary
cross-product (`emit.USAGE_LIMIT_CROSS_PRODUCT`: base choice, fault model
"no implementation pattern couples the boundary values of two structural
attributes", disposition `pruned`), and every limit carries its own
fault-model `justification` so a reviewer can challenge the bound itself.

### 4.3 Pruning records are data, not prose

Every excluded combination is a `Pruning(combination, fault_model_absent,
disposition)` in the operator definition, validated by the DSL
(`pruned` / `covered-by <test id>` / `pairwise-coverage`) and emitted into
manifest and report. The judgment is therefore reviewable and diff-able, and
mutation testing (§ 4.8) checks the beliefs empirically.

### 4.4 Floats: discrete value classes, and the NaN caution

A `FloatVar` is an **index over its interesting value classes** (NaN, ±inf,
±0, subnormals, normals, per width) — the strategy's type-specific domains.
`fcls`/`fne` select classes by name; ordered comparisons (`flt/fge/fgt/fle`)
never match NaN or infinities. Per the strategy's caution, NaN satisfies
neither p′ nor p″ of an ordered predicate, so an operator definition that uses
ordered float comparisons must carry an explicit NaN sibling predicate —
`dsl.nan_sibling_violations` enforces this at generation time (refusal with
diagnostic) and in `selfcheck.py`; a genuine spec silence can be recorded in
`params["nan_spec_gaps"]` instead (reported, not assumed).

### 4.5 Witness selection: small and readable — a recorded deviation

After satisfiability, each variable is greedily pushed to its smallest
feasible boundary-anchored candidate so reviewers can eyeball the vectors;
tensor-size variables named in `params["prefer_large_vars"]` are pushed to
their *largest* feasible value so structural boundaries are exercised on
non-degenerate tensors. **Deviation:** the strategy prescribes the rounded
arithmetic midpoint of the feasible interval for integer interior classes;
aligning the two rules is pending (item 2.1 of the impact analysis) and the
current preference is recorded as a deviation.

### 4.6 Traceability, provenance, prioritization

Every predicate carries a spec `origin` tag and the completeness `criterion`
it serves; every class aggregates its predicates' criteria plus a
`retention_reason`; every test inherits both. Tests are ordered by the
strategy's risk-based tiers (1 dependency/symmetry-flagged, 2 boundary/vertex
+ special-value, 3 pairwise-coverage, 4 plain interior) and the manifest
records `tests_by_priority`. The completeness claim of the strategy is thus
checkable mechanically rather than by trust.

### 4.7 Two independent correctness paths

Watches the two places a silent error could hide:

- **witnesses** are re-verified by `evaluate.py` through exact `Fraction`
  arithmetic, independent of the solver; the CLI aborts (exit 1) on any
  failure rather than emitting an unsound test;
- **expected outputs** come from `reference.py`, a deliberately naive
  transcription of the informal definition (padded max scan, row-major
  tie-break), cross-checked against the spec's output-size formula at
  materialization time.

### 4.8 Refusal discipline and specification gaps

The tool reports what it cannot do instead of papering over it:

- no spec file → generation refused (exit 2);
- NaN semantics undefined by the spec (MaxPool's Max_F) → NaN-bearing inputs
  deliberately not generated; the gap is reported;
- padding-only windows under dilation ≥ 2 (where the spec's own rationale
  for pads-C2 fails and neither Y nor Indices is defined) → materialization
  refused, the configuration recorded as a **specification gap** in
  manifest/report for spec clarification (test objective 1: validating the
  specification).

### 4.9 Mutation testing validates the pruning claims

Each pruning/retention record asserts an implementation pattern is absent.
`mutate.py` injects plausible faults into the oracle (index off-by-one,
inverted tie-break, swapped strides/dilations, ceil instead of floor output
size, dilation ignored in the kernel extent) and runs every retained test
against each mutant: a **surviving** mutant is evidence that a pruning or
`covered-by` claim is too aggressive. The MaxPool set kills all 5 mutants on
all 5 dtypes — the tie-break mutant *only* via the retained T2/T4 tests, which
is what validates their auxiliary-only restriction.

### 4.10 Cross-operator pattern catalog

Dependency-check findings, tie-break requirements and NaN cautions live once
in `catalog.py` with structural-feature tags (sliding-window, padding,
index-output, output-size-formula, …); operator dependency rows cite entries
by ID, the manifest emits the cited entries, and `catalog.for_features()`
gives the re-review set when a pattern evolves.

### 4.11 Determinism

No wall-clock, no randomness, stable enumeration order: regeneration is
byte-for-byte reproducible (verified over the whole `tests/` tree).

## 5. Adding a new operator

1. **Spec** under `specs/` (gating requirement — no spec, no tests).
2. **Definition** in `operators.py`: variables with usage limits and
   justifications; background domain; the p′/p″ constraint system with each
   predicate's spec tag and criterion; auxiliary boundary/property cases;
   pruning records for every exclusion you intend; the dependency table
   (all four candidate families) citing `catalog.py` entries.
3. **Oracle** in `reference.py` — a naive transcription of the informal
   definition, independent of any production implementation.
4. **Materializer** — value templates expressing which value-level
   interactions matter for this operator; refuse undefined configurations.
5. **Mutants** in `mutate.py` (`MUTANT_SETS`), chosen to probe exactly the
   pruning/retention records you wrote.
6. **Selfcheck** additions if the spec yields mechanically checkable
   invariants.

The Div operator (`specs/div.md`) is the current candidate: its spec exists;
what is missing is the definition, oracle, templates and mutants.

## 6. Known limitations and future work

- **Representative-value rule** (impact item 2.1, pending): the small-witness
  preference deviates from the strategy's midpoint rule for integer interior
  classes; alignment changes every generated witness and gets its own
  regeneration pass.
- **Float arithmetic predicates** need Z3 `FP` sorts once an operator requires
  float equality classes in formulas (e.g. $x_1 + x_2 = c$); today floats
  appear only as discrete value classes.
- **Broadcasting-specific domains** (per-axis pairing classes, rank mismatch)
  require **error classes** — expected outcome *rejection*, which the
  materializer/emitter cannot express yet.
- **Property/metamorphic tests** (the strategy names two MaxPool properties)
  are a separate, unimplemented tier.
- The enumeration fallback backend is degraded: exact only w.r.t. its
  candidate sets and integer-only; install `z3-solver` for full support.