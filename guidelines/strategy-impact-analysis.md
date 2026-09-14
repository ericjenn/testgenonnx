# Impact analysis — `guidelines/strategy.md` vs. the current `testgen` tool

This file records, mechanism by mechanism, what the current tool already
satisfies, what must be modified, and what is future work for operators not
yet in the catalog. File references point into `testgen/`.

**Implementation status:** items 2.2–2.10 are implemented (each section
below carries its status and where it landed); 2.1 (representative-value
midpoint rule) remains pending — the current small-witness preference is a
recorded deviation until it is aligned.

---

## 1. Already compliant — no change needed

| Strategy mechanism (strategy.md section) | Current implementation |
|---|---|
| p′/p″ algebra, strict→non-strict rewriting, DNF enumeration, UNSAT elimination (§ Equivalence classes based on the input domain) | `splitting.py`, `classes.py` |
| Pruning records: combination / fault model assumed absent / disposition (§ Eliminating irrelevant cases) | `dsl.Pruning`, emitted into manifest + report — field structure matches the strategy's field table (disposition vocabulary needs one extension, see §2.3) |
| Test traceability via spec tags (§ Test traceability) | every test JSON carries `traceability` + `spec_ref` (metadata extension needed, see §2.4) |
| Record the raw solver class count before and after pruning (§ Solving the system of equations — "open action item for MaxPool") | `manifest.json` `totals` records combinations_enumerated / classes / unsat / pruned (144 / 112 measured) — the action item is already discharged |
| Bounded-model use of the solver: usage bounds applied before enumeration (§ Solving the system of equations) | solver reasons over bounded `IntVar`s |
| Symmetry/asymmetry tested explicitly (§ Symmetry and asymmetry) | PR cases for symmetric/asymmetric pads |
| Index observability (§ On values and indexes...) | template T1 (position-derived distinct values) |
| NaN caution: NaN satisfies neither p′ nor p″ of an ordered predicate and needs an explicit sibling equality predicate (§ Eliminating irrelevant cases) | the float encoding realizes this exactly: `fcls`/`fne` select classes by name; ordered range comparisons never match NaN (`solver._float_constraint`); the Abs worked example adds NaN as an explicit branch |
| Non-pertinent combinations judged with recorded justification (§ Eliminating irrelevant cases) | pruning records with per-entry fault models |
| Unify-then-prune composition: interactions expressed as shared predicates, reduced by the existing pruning mechanism (§ Eliminating irrelevant cases) | the structural system already folds functional constraints, type constraints and spec notes into one predicate set per operator (`operators.py`) |

## 2. Needs modification

### 2.1 Representative-value selection rule (§ Selecting a representative value within a class) — **PENDING**

The strategy prescribes an explicit rule, which the tool does not
implement:

- **boundary classes (p′)**: value fixed by construction — already the case;
- **integer interior classes (p″)**: **arithmetic midpoint of the interval,
  rounded** — the tool instead greedily picks the *smallest* feasible
  candidate from {lo, lo+1, mid, hi−1, hi} ∪ predicate constants ±1
  (`solver.py::_candidate_values`, `Z3Model.witness`), a readability
  heuristic the strategy's midpoint rule supersedes;
- **float interior classes**: representative drawn from the type-specific
  classes (§ Type-specific domains) — already the case de facto (FloatVar
  is an index over the type-specific value classes); should be documented
  as the instantiation of this rule;
- **jointly-fixed equalities** (e.g. $x_1+x_2=149$): the pair should be driven
  by "whichever other axis is being exercised" — the tool's `prefer_large`
  mechanism (dX2/dX3 pushed to the usage maximum so boundaries are
  non-vacuous) is a *justified deviation* in this spirit, but it is
  currently justified only in a code comment; it should be recorded as
  operator data.

**Change**: in `solver.py`, for interior classes compute each variable's
feasible interval under (domain ∧ combo) with two extra solver queries and
take `round((lo'+hi')/2)` as the primary candidate; keep boundary-adjacent
candidates for p′ classes. Move the `prefer_large` rationale into
`operators.py` params (recorded, reviewable).

### 2.2 Usage-domain predicates as ordinary predicates (§ The case of unbounded variables)

The strategy requires the arbitrary bounds (rank, size) to be *ordinary
predicates in P*, split into p′/p″ and taking part in the same enumeration
and pruning — and each bound value must carry a **fault-model
justification** like a pruning record, "so that a reviewer can challenge
the choice of limit itself".

Current state: usage limits live in `op.domain` (background theory, **not**
split), and their boundaries are covered by per-variable `Branch(kind=
"boundary")` cases — a deliberate anti-explosion design (14 usage limits in
the split system would give ~2^22 combinations vs. today's 2^8).

**Change** (two parts):
1. keep the base-choice BD mechanism, but record it as what it is under
   the strategy: a systematic pruning of the usage-limit cross-product,
   with the disposition vocabulary of the strategy (see 2.3) instead of an
   unnamed design decision — i.e. emit the usage-limit handling as
   pruning-style records in the manifest;
2. add a `justification` field per usage limit (fault model assumed
   absent), alongside the existing `note`, and emit it in manifest/report.

**Status: implemented** — each of the 14 usage-limit variables carries a fault-model justification (`IntVar.justification`, set in `operators.py`); the manifest/report emit a per-variable usage-limit record plus the base-choice cross-product pruning record (`emit.USAGE_LIMIT_CROSS_PRODUCT`, disposition `pruned`).

### 2.3 Disposition vocabulary: add `pairwise-coverage` (§ Eliminating irrelevant cases)

`dsl.Pruning.disposition` currently documents only `"pruned"` /
`"covered-by <id>"`. The strategy's field table adds
**`pairwise-coverage`**: a combination kept (or excluded) purely to
satisfy a covering-array obligation, without an individually stated fault
model. Needed by 2.2 and by the optional covering-array mode below.

**Change**: extend the disposition; surface it in manifest/report.

**Status: implemented** — `dsl.Pruning.__post_init__` validates the three-way disposition vocabulary; emitted in manifest/report as before.

### 2.4 Class-provenance / retention-reason metadata (§ Test traceability)

Each generated test must record *which criterion* (or which unified pair
of criteria) generated it, and *whether it was retained because of a
stated fault model or to fill a pairwise-coverage obligation*. Current
test JSON has traceability tags + class label, but neither the criterion
nor the retention reason.

**Change**: add `criterion` and `retention_reason` fields to the class/test
documents and the manifest.

**Status: implemented** — `Predicate.criterion` / `Branch.criterion` (operator data), aggregated onto `EqClass`/`BranchClass` in `classes.py`; every class and every test JSON carries `criterion`, `retention_reason`, `priority` in manifest and report.

### 2.5 Risk-based prioritization (§ Risk-based prioritization of test generation)

Default ordering: (1) dependency/symmetry-flagged classes, (2) boundary and
vertex classes + special-value/discontinuity classes, (3) pairwise-coverage
combinations, (4) plain interior classes. The tool currently enumerates
most-constrained-first and appends auxiliary cases, with no recorded tier.

**Change**: assign a priority tier per class (data: dependency-table hits,
PR/BD/BR kind, split label), emit `priority` in manifest + a report section,
and order the emitted test files by tier.

**Status: implemented** — tiers assigned per class (property→1, boundary→2, EC boundary/vertex→2, interior→4) and per test (templates T3/T4 raised to tier 1 via `params["tier1_templates"]`); `tests_by_priority` in manifest totals, emitted test list ordered by tier, provenance section in every report.

### 2.6 Cross-operator pattern catalog (§ Maintaining a cross-operator pattern catalog)

The dependency-check questions, special-value/discontinuity tables, and
mutation findings must live in a single **shared, living catalog** across
operators; per-operator tables reference catalog entries; operators
sharing a structural feature (e.g. "sliding window with padding") are
flagged for re-review when a new pattern is found.

**Change**: new module (e.g. `testgen/catalog.py` + catalog data file) that
holds the patterns; `operators.py` references catalog entry IDs in its
dependency rows and notes; the manifest records which catalog entries apply.

**Status: implemented** — new `testgen/catalog.py` (5 patterns with structural-feature tags); `Dependency.catalog_refs` cites entries; manifest `pattern_catalog` + report section list the cited entries; `for_features()` gives the re-review set.

### 2.7 Mutation testing harness (§ Validating pruning decisions via mutation testing)

A set of plausible mutants is injected into the reference implementation
and the retained test set is run against each; a surviving mutant is
evidence that a pruning decision was too aggressive or a `covered-by`
claim does not hold. This is a genuinely new capability — and it directly
targets the tool's own 2 MaxPool pruning records (112 pruned combinations
justified by assumed-absent fault models).

**Change**: new module (e.g. `testgen/mutate.py`): mutate
`reference.maxpool_ref` (off-by-one index, swapped `strides`/`dilations`,
wrong floor/ceil rounding, missing pad constant special case, tie-break
order), run all retained tests per dtype, report survivors mapped back to
the pruning/`covered-by` records they invalidate. CLI subcommand +
manifest/report section for results.

**Status: implemented** — new `testgen/mutate.py` (5 MaxPool mutants), run via `python -m testgen.cli --mutate`; results merged into each manifest (`mutation_testing`) and report section. First run: all 5 mutants killed on all 5 dtypes — no pruning/retention decision invalidated.

### 2.8 Dependency table: fourth row (§ Cross-domain dependency check)

The check has four candidate dependencies — the tool's MaxPool table has
three (accumulator, index, output-size coupling); **"Broadcasting vs.
special values"** is missing (answer: N/A — MaxPool does not broadcast).

**Change**: add the row in `operators.py` dependencies.

**Status: implemented** — the row is in `operators.py` (N/A for MaxPool: single data input, no broadcasting).

### 2.9 Stale references to the V2 document

`emit.py:127` (report header), `cli.py:39` (refusal message),
`testgen/__init__.py:3`, and `guidelines/test-strategy-understanding.md`
all reference `guidelines/SONNX - Test strategy V2.md`, which no longer
exists.

**Change**: point at `guidelines/strategy.md`; revise the understanding
document against the current strategy — notably the small-witness heuristic
is now *superseded* by the midpoint rule (2.1), and explosion control via
value templates should be reworded in the unify-then-prune frame of
§ Eliminating irrelevant cases.

**Status: implemented** — `emit.py`, `cli.py`, `testgen/__init__.py` and the understanding doc now reference `guidelines/strategy.md`; the understanding doc records the small-witness deviation from the midpoint rule and rewords explosion control in the unify-then-prune frame.

### 2.10 NaN sibling-predicate validation (§ Eliminating irrelevant cases)

The strategy requires the designer to *check* that special values falling
outside every p′/p″ pair get an explicit sibling equality predicate. The
tool's encoding already behaves correctly, but nothing *checks the
operator definition*.

**Change**: validation in `selfcheck.py`/CLI: for every FloatVar whose value
classes include NaN, if the operator's predicates include ordered float
comparisons over that variable, require an explicit NaN `fcls`/`fne`
predicate (or record a spec-gap note), else fail with a diagnostic.

**Status: implemented** — `dsl.nan_sibling_violations(op)`; enforced by the CLI (refusal with diagnostic) before generation and checked in `selfcheck.py` (including a detection test and the recorded-spec-gap escape hatch `params["nan_spec_gaps"]`).

## 3. Future work — needed only when the corresponding operator gets a spec

- **Floating-point theory for float arithmetic predicates** (§ Solving the
  system of equations): once an operator needs arithmetic float equality
  classes (e.g. $x_1+x_2=c$), the solver must use Z3's `FP` sorts rather
  than real/rational arithmetic. No current operator has float arithmetic
  predicates (floats appear only as discrete value classes); record the
  caveat in the solver docstring.
- **Broadcasting-specific domains** (§ Broadcasting-specific domains):
  per-axis pairing classes (equal / broadcast-from-1 per side /
  incompatible), rank-mismatch class, degenerate cases. Requires new DSL
  concepts — in particular **error classes** (incompatible shapes: the
  expected outcome is rejection, not a computed tensor), which
  `materialize.py`/`emit.py` cannot express today.
- **Output-comparison tolerance** (§ Note on scope: input equivalence
  vs. output-comparison tolerance): per-dtype ULP / relative-error rule
  for comparing the SONNX reference output against an implementation
  under test — an independent parameter to record in the manifest for the
  future oracle-comparison stage (the tool currently only generates
  expected outputs).
- **Property/metamorphic "additional tests"** (§ Additional tests): still a
  separate strategy, not implemented. The strategy names two MaxPool
  metamorphic properties (invariance of the max index under adding a
  constant / multiplying by a strictly positive constant); they are
  candidates for that future tier.

## 4. Suggested implementation order

1. ~~Cheap compliance: 2.8 (dependency row), 2.9 (stale refs + understanding
   doc), 2.3 (disposition vocabulary), 2.10 (NaN validation)~~ — done;
2. Representative-value rule: 2.1 (+ recording `prefer_large` as data) —
   **remaining**; changes every generated witness, so it deserves its own
   regeneration pass;
3. ~~Provenance & prioritization: 2.4, 2.5, and the record-keeping half of
   2.2 (usage-limit justifications)~~ — done;
4. ~~New capabilities: 2.7 mutation harness (highest value — it checks the
   112 pruning claims), then 2.6 catalog~~ — done; optional pairwise
   covering-array mode on top of 2.3 remains future work.