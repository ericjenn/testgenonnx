"""Command-line entry point.

Usage:
    python -m testgen.cli [--out tests] [--spec-dir specs] [--mutate]
        [operator ...]

Generates equivalence-class-based functional tests for the spec-backed
catalog operators, *per data type*, into the output directory. An
operator is only generated when its informal specification file exists
under the spec directory; requests for operators without a specification
are refused (user requirement -- tests are derived from specifications
only). With --mutate, the generated test sets are additionally validated
by mutation testing (strategy: 'Validating pruning decisions via
mutation testing').
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .classes import enumerate_classes
from .dsl import nan_sibling_violations
from .emit import emit
from .evaluate import verify_witness
from .mutate import mutation_analysis
from .operators import CATALOG
from .solver import backend_name

# Repository root (parent of the testgen package).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def spec_path(op, spec_dir: str) -> str:
    return os.path.join(REPO_ROOT, spec_dir, os.path.basename(op.spec_file))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="testgen",
        description="Equivalence-class-based functional test generator for "
                    "SONNX operators (strategy: guidelines/strategy.md).",
    )
    ap.add_argument("--out", default="tests", help="output directory")
    ap.add_argument("--spec-dir", default="specs",
                    help="directory holding the informal specifications")
    ap.add_argument("--mutate", action="store_true",
                    help="run mutation testing over the generated test "
                         "sets and merge the results into each manifest "
                         "and report")
    ap.add_argument("operators", nargs="*",
                    help=f"operator names (default: all spec-backed: "
                         f"{sorted(CATALOG)})")
    args = ap.parse_args(argv)

    spec_backed = [n for n in sorted(CATALOG)
                   if os.path.isfile(spec_path(CATALOG[n], args.spec_dir))]

    if args.operators:
        refused = [n for n in args.operators
                   if n not in CATALOG
                   or not os.path.isfile(spec_path(CATALOG[n], args.spec_dir))]
        if refused:
            for n in refused:
                if n not in CATALOG:
                    print(f"REFUSED: unknown operator {n!r} "
                          f"(available: {', '.join(sorted(CATALOG))})",
                          file=sys.stderr)
                else:
                    print(f"REFUSED: no specification file for {n!r} "
                          f"(expected: {spec_path(CATALOG[n], args.spec_dir)})"
                          f" -- tests are only generated from a "
                          f"specification", file=sys.stderr)
            return 2
        names = args.operators
    else:
        names = spec_backed

    print(f"solver backend: {backend_name()}")
    index = {}
    for name in names:
        op = CATALOG[name]

        # NaN sibling-predicate validation (strategy: the caution on
        # unifying an ordered domain predicate with a special-value
        # predicate). A definition whose float variables are under
        # ordered comparison without an explicit NaN class would silently
        # drop NaN from the enumerated classes.
        violations = nan_sibling_violations(op)
        if violations:
            for v in violations:
                print(f"REFUSED: {v}", file=sys.stderr)
            return 2

        result = enumerate_classes(op)

        # Independent verification of every witness (evaluation path,
        # separate from the solving path -- cf. evaluate.py).
        failures = []
        checked = 0
        for cls in result.classes:
            checked += 1
            bad = verify_witness(op.variables, op.domain,
                                 cls.predicates, cls.witness)
            if bad:
                failures.append(f"{cls.id}: {bad}")
        for br in result.branches:
            checked += 1
            bad = verify_witness(op.variables,
                                 list(op.domain) + list(op.constraints),
                                 br.predicates, br.witness)
            if bad:
                failures.append(f"{br.id}: {bad}")
        if failures:
            print(f"{name}: WITNESS VERIFICATION FAILED", file=sys.stderr)
            for f in failures:
                print(f"  {f}", file=sys.stderr)
            return 1

        out_dirs = emit(op, result, args.out)
        dtypes = ", ".join(op.params.get("dtypes", []))
        print(
            f"{name} [{dtypes}]: {len(result.classes)} equivalence classes, "
            f"{len(result.branches)} auxiliary cases, "
            f"{len(result.unsat)} unsatisfiable, "
            f"{len(result.pruned)} pruned, "
            f"{checked}/{checked} witnesses verified"
        )
        for d in out_dirs:
            n_tests = len([f for f in os.listdir(d) if f.endswith(".json")
                           and f != "manifest.json"])
            print(f"  -> {d} ({n_tests} functional tests, report.md)")

        mutation = None
        if args.mutate:
            results = mutation_analysis(out_dirs)
            mutation = {}
            for d, res in results.items():
                survivors = res["survivors"]
                print(f"  mutation testing [{res['dtype']}]: "
                      f"{len(res['mutants'])} mutants, "
                      f"{len(res['mutants']) - len(survivors)} killed, "
                      f"{len(survivors)} survived"
                      + (f" ({', '.join(survivors)})" if survivors else ""))
                mutation[res["dtype"]] = {
                    "tests": res["tests"],
                    "mutants": len(res["mutants"]),
                    "survivors": survivors,
                }

        index[name] = {
            "spec_file": spec_path(op, args.spec_dir),
            "dtypes": op.params.get("dtypes", []),
            "directories": out_dirs,
            "equivalence_classes": len(result.classes),
            "auxiliary_classes": len(result.branches),
            "unsatisfiable": len(result.unsat),
            "pruned": len(result.pruned),
            "backend": result.backend,
            "mutation_testing": mutation,
        }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())