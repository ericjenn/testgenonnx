# testgen — equivalence-class-based test generator for (S)ONNX operators

`testgen` implements the **equivalence-class-based test strategy** described in
[`guidelines/strategy.md`](guidelines/strategy.md) (see
[`guidelines/test-strategy-understanding.md`](guidelines/test-strategy-understanding.md)
for the recorded understanding of the strategy). It generates functional test
cases — input tensors together with the expected outputs computed by a
reference implementation of the informal specification — for operators backed
by a specification file under `specs/`.

Design rationale and internals: [`DESIGN.md`](DESIGN.md).

## Requirements

- Python ≥ 3.9 (developed and tested on 3.14), standard library only, plus
- [`z3-solver`](https://pypi.org/project/z3-solver/) — recommended; without it
  the generator falls back to a degraded enumeration backend (integer
  variables only, verdicts exact only w.r.t. the candidate sets).

## Installation

```sh
python -m venv ~/Venvs/testgenonnx
~/Venvs/testgenonnx/bin/pip install -r requirements.txt
```

## Usage

Run from the repository root:

```sh
# Generate tests for every spec-backed operator (currently: MaxPool)
~/Venvs/testgenonnx/bin/python -m testgen.cli

# One operator only
~/Venvs/testgenonnx/bin/python -m testgen.cli MaxPool

# Generate and additionally validate the retained test sets by mutation testing
~/Venvs/testgenonnx/bin/python -m testgen.cli --mutate

# Machinery self-check against the strategy document's worked examples
~/Venvs/testgenonnx/bin/python -m testgen.selfcheck
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `--out DIR` | `tests` | output directory |
| `--spec-dir DIR` | `specs` | directory holding the informal specifications |
| `--mutate` | off | run mutation testing over the generated test sets and merge the results into each manifest and report |
| `operators …` | all spec-backed | operator names to generate |

### Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | a generated witness failed independent verification (generation aborts; nothing unsound is emitted) |
| 2 | **refused**: unknown operator, no specification file for the requested operator, or the operator definition violates the NaN sibling-predicate check — tests are only derived from actual specifications |

## What is generated

Generation is **per data type** (one independent test set per dtype of the
spec). For MaxPool: `float16`, `float`, `double`, `int8`, `uint8` — each with
its spec-mandated padding constant. The output layout:

```
tests/
    index.json                     per-operator summary
    MaxPool_<dtype>/
        <test case>.json           one file per functional test
        manifest.json              machine-readable audit artifact
        report.md                  detailed human-readable report
```

Per dtype the MaxPool set contains **194 functional tests**: 144 equivalence
classes (from the 256-combination p′/p″ decomposition of the 8-predicate
constraint system; 112 combinations pruned by recorded justification) and 15
auxiliary boundary/property cases, each materialized through dtype-specific
value templates — 970 test cases in total, all byte-for-byte reproducible.

### Test case files

Each JSON file carries:

- `attributes`, `input` (shape + values) and `expected` (`Y`, `Indices`, shape)
  computed by the reference implementation of the informal specification
  ([`testgen/reference.py`](testgen/reference.py));
- full traceability: `spec_ref`, per-predicate origin tags, the class label
  and its rendered predicates;
- class provenance and prioritization: the test-completeness `criterion` that
  produced it, the `retention_reason` (why the case is in the set), and the
  risk-based `priority` tier (1 dependency/symmetry-flagged … 4 plain
  interior); files are listed tier-first.

### manifest.json

The audit artifact required by the strategy: variables and their usage limits
with the fault models assumed absent, the constraint system with criteria,
every class with its witness, unsatisfiable and pruned combinations (with
rationale and disposition), the cross-domain dependency table with its catalog
references, cited cross-operator pattern-catalog entries, specification gaps,
and totals. With `--mutate`, the mutation-testing results are merged in.

### report.md

A detailed report of the generation: constraint table, class inventory with
witnesses, pruning records, provenance/prioritization, dependency table,
specification findings. Intended for review by a human, with classical
mathematical notation.

## Operator coverage

| Operator | Spec | Status |
|---|---|---|
| MaxPool | [`specs/maxpool.md`](specs/maxpool.md) | generated (5 data types) |
| Div | [`specs/div.md`](specs/div.md) | spec present, operator definition not yet modeled |

The strategy's own worked examples (Add uint8, Div int8, Relu/Abs float) are
**never generated** — they have no spec role here; they are encoded in
`testgen/worked_examples.py` solely to validate the class-enumeration
machinery (`testgen.selfcheck`).

## Adding a new operator

1. Place the informal specification under `specs/`.
2. Declare the operator in `testgen/operators.py` with the DSL of
   `testgen/dsl.py` (variables with usage limits and justifications, domain,
   the p′/p″ constraint system, auxiliary cases, pruning records, dependency
   table with catalog references).
3. Provide the reference implementation (test oracle) in
   `testgen/reference.py` and the value templates / materializer in
   `testgen/materialize.py`.
4. Add a mutant set to `testgen/mutate.py` and register both in
   `MUTANT_SETS` / the materializer dispatch.
5. Extend `testgen/selfcheck.py` if the spec yields checkable invariants.

Details and the reasoning behind each obligation: [`DESIGN.md`](DESIGN.md).

## Documentation map

- [`DESIGN.md`](DESIGN.md) — how and why the tool works internally.
- [`guidelines/strategy.md`](guidelines/strategy.md) — the test strategy implemented.
- [`guidelines/test-strategy-understanding.md`](guidelines/test-strategy-understanding.md) — recorded understanding, proposed improvements, scope.
- [`guidelines/strategy-impact-analysis.md`](guidelines/strategy-impact-analysis.md) — strategy-vs-implementation gap analysis with per-item status.