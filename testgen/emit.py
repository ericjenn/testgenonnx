"""Emission of generated test cases, manifests and reports.

For a spec-backed operator the generator writes, *per data type* (user
requirement: data types are generated separately), a directory::

    tests/<Operator>_<dtype>/
        <test case>.json     one functional test per (class, template)
        manifest.json        machine-readable audit artifact
        report.md            detailed human-readable report of the
                             generation (classes, witnesses, templates,
                             pruning rationale, dependency table)

Operators without a specification are refused upstream (cli).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from . import catalog
from .classes import ClassifyResult, tier_name
from .domains import ValueClass, jsonable_value, render_value
from .dsl import Operator, FloatVar, IntVar, var_latex
from .materialize import PAD_VALUES, TEMPLATES, functional_tests
from .splitting import label_of


# Systematic record of the usage-limit handling (strategy: "The case of
# unbounded variables" + "Eliminating irrelevant cases"). The limits are
# ordinary domain predicates; keeping them out of the p'/p'' split system
# amounts to pruning their full boundary cross-product, recorded here
# with the disposition vocabulary of the strategy.
USAGE_LIMIT_CROSS_PRODUCT = {
    "combination": ("any combination with two or more usage limits "
                    "simultaneously on their boundary form"),
    "fault_model_assumed_absent": (
        "no known implementation pattern couples the boundary values of "
        "two different structural attributes; each usage boundary is "
        "exercised independently against a base (interior) configuration "
        "of all other variables"
    ),
    "disposition": "pruned",
    "method": (
        "base choice: every usage-limit boundary case (auxiliary kind "
        "'boundary') holds one variable at its bound with all other "
        "variables free, solved against the domain and the functional "
        "constraint system"
    ),
}


def _usage_limit_records(op: Operator) -> List[Dict[str, object]]:
    """Per-variable record of the usage bounds and their fault models."""
    recs = []
    for v in op.variables:
        if isinstance(v, IntVar) and v.usage_limit:
            recs.append({
                "variable": v.name,
                "role": v.role,
                "bounds": [v.lo, v.hi],
                "fault_model_assumed_absent": v.justification or "-",
            })
    return recs


def _catalog_entries(op: Operator) -> List[Dict[str, object]]:
    """The shared cross-operator catalog entries this operator cites."""
    ids: List[str] = []
    for d in op.dependencies:
        for r in d.catalog_refs:
            if r not in ids:
                ids.append(r)
    return [{
        "id": p.id,
        "title": p.title,
        "description": p.description,
        "structural_features": list(p.structural_features),
        "origin": p.origin,
        "action": p.action,
    } for p in (catalog.get(i) for i in ids)]


def _witness_json(op: Operator, witness: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for v in op.variables:
        w = witness[v.name]
        if isinstance(v, FloatVar):
            out[v.name] = {"class": w.cls, "value": render_value(w.value)}
        else:
            out[v.name] = jsonable_value(w)
    return out


def _traceability(predicates) -> List[str]:
    tags = []
    for p in predicates:
        if p.origin and p.origin not in tags:
            tags.append(p.origin)
    return tags


def _variable_docs(op: Operator) -> List[Dict[str, object]]:
    docs = []
    for v in op.variables:
        if isinstance(v, IntVar):
            docs.append({
                "name": v.name,
                "kind": "int",
                "role": v.role,
                "bounds": [v.lo, v.hi],
                "usage_limit": v.usage_limit,
            })
        else:
            docs.append({
                "name": v.name,
                "kind": f"float{v.width}",
                "role": v.role,
            })
    return docs


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def _math(s: str) -> str:
    """Wrap a LaTeX fragment in $...$ for markdown math rendering."""
    return f"${s}$"


def _pred_math(preds) -> str:
    """Conjunction of predicates, each typeset in LaTeX."""
    return " $\\wedge$ ".join(_math(p.render_latex()) for p in preds)


def _choice_math(label_str: str) -> str:
    """A p'/p'' choice label in LaTeX, e.g. $p'_1 \\wedge p''_2$."""
    if not label_str.startswith("p'"):
        return label_str  # branch labels ("boundary:...") stay prose
    return _math(label_of(label_str))


def _report(op: Operator, result: ClassifyResult, dtype: str,
            functional: List[Dict[str, object]],
            gaps: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    a = lines.append

    a(f"# Test generation report -- {op.name} ({dtype})\n")
    a(f"Generated by `testgen` (equivalence-class-based strategy, "
      f"guidelines/strategy.md). Backend: "
      f"**{result.backend}**.\n")

    a("## Specification\n")
    a(f"- Informal specification: `{op.spec_file}`")
    a(f"- Traceability: {op.spec_ref}")
    a(f"- Data type under test: **{dtype}**")
    pad_const = PAD_VALUES.get(dtype)
    if pad_const is not None:
        a(f"- Padding constant per spec: `{pad_const}`")
    a("")

    restrictions = op.params.get("restrictions", [])
    if restrictions:
        a("## Restrictions and fixed attributes\n")
        for r in restrictions:
            a(f"- {r}")
        a("")
        a("Fixed-value attributes are constants in every test, not test "
          "dimensions.\n")

    usage_limits = _usage_limit_records(op)
    if usage_limits:
        a("## Usage limits and their justification\n")
        a("Structural variables are bounded by usage limits rather than "
          "their type bounds (strategy: 'The case of unbounded "
          "variables'). The bound value is an engineering choice, so "
          "each limit records the fault model assumed absent -- a "
          "reviewer can challenge the choice of limit itself, not only "
          "the classes derived from it.\n")
        a(_md_table(
            ["variable", "bounds", "fault model assumed absent"],
            [[r["variable"],
              f"[{r['bounds'][0]}, {r['bounds'][1]}]",
              r["fault_model_assumed_absent"]] for r in usage_limits]))
        a("Keeping the usage limits out of the $p'/p''$ split system is "
          "itself a recorded elimination (strategy: 'Eliminating "
          "irrelevant cases'):\n")
        cp = USAGE_LIMIT_CROSS_PRODUCT
        a(_md_table(
            ["combination", "fault model assumed absent", "disposition"],
            [[cp["combination"], cp["fault_model_assumed_absent"],
              cp["disposition"]]]))
        a(f"Method: {cp['method']}.\n")

    a("## Notation\n")
    rows = []
    for v in op.variables:
        rows.append([_math(var_latex(v)), f"`{v.name}`", v.role])
    a(_md_table(["symbol", "variable", "meaning (spec attribute)"], rows))
    for note in op.params.get("notation_notes", []):
        a(f"{note}\n")

    a("## Constraint system and the $p'/p''$ split\n")
    a("The strategy's equivalence-class algebra is applied to the "
      "functional constraints coupling structural parameters. Each "
      "predicate $p_i$ is split into a boundary form $p'_i$ (on the "
      "boundary, as an equality) and an interior form $p''_i$ (strictly "
      "inside); every satisfiable combination of forms is one "
      "equivalence class.\n")
    rows = []
    for i, p in enumerate(op.constraints, start=1):
        rows.append([_math(f"p_{{{i}}}"), p.origin,
                     p.criterion or "-", _math(p.render_latex()),
                     p.note or "-"])
    a(_md_table(["#", "origin (spec tag)", "criterion", "predicate",
                 "note"], rows))

    a("### Value templates (value-level dimension)\n")
    a("Every structural class is materialized into functional tests via "
      "dtype-specific value templates:\n")
    a(_md_table(["template", "name", "applied to", "note"],
                [[t["id"], t["name"], t["applies_to"],
                  t.get("note", "-")] for t in TEMPLATES]))
    aux_only = [t["id"] for t in TEMPLATES if t["applies_to"] == "auxiliary"]
    if aux_only:
        a(f"{', '.join(aux_only)} are restricted to the auxiliary "
          "(boundary/property) cases: structural interior classes add "
          "no expected value-level fault (pruning discipline -- the "
          "rationale is recorded here).\n")

    a(f"## Generated equivalence classes ({len(result.classes)})\n")
    a("Each class is identified by its $p'/p''$ choice; the predicate "
      "underlying each $p_i$ is defined in the constraint table above. "
      "The `prio` column is the risk-based prioritization tier defined "
      "in the provenance section below.\n")
    rows = []
    for cls in result.classes:
        tpl_ids = ", ".join(
            t["value_template"]["id"]
            for t in functional
            if t.get("class_id") == cls.id
        )
        rows.append([
            cls.id,
            _choice_math(cls.label),
            str(cls.priority),
            _params_str(op, cls.witness),
            tpl_ids or "-",
        ])
    a(_md_table(["class", "split", "prio", "witness (params)",
                 "templates"], rows))

    a(f"## Generated auxiliary classes ({len(result.branches)})\n")
    a("Explicit boundary cases (usage limits, spec value-domain "
      "boundaries) and symmetry properties -- never left to fortuitous "
      "coverage (strategy: 'Symmetry and asymmetry').\n")
    rows = []
    for br in result.branches:
        tpl_ids = ", ".join(
            t["value_template"]["id"]
            for t in functional
            if t.get("class_id") == br.id
        )
        rows.append([
            br.id, br.kind, str(br.priority), br.name,
            _params_str(op, br.witness),
            _pred_math(br.predicates),
            tpl_ids or "-",
        ])
    a(_md_table(["case", "kind", "prio", "name", "witness (params)",
                 "predicates", "templates"], rows))

    a("## Class provenance and risk-based prioritization\n")
    a("Each generated test records the test-completeness criterion that "
      "produced it and why the combination is in the test set (per class "
      "and per test in the manifest; strategy: 'Test traceability'). "
      "Tests are ordered by the risk-based prioritization tiers of the "
      "strategy ('Risk-based prioritization of test generation'):\n")
    a(_md_table(
        ["tier", "meaning"],
        [[str(k), v] for k, v in sorted(
            {1: "dependency/symmetry-flagged classes",
             2: "boundary and vertex classes + special-value classes",
             3: "pairwise-coverage combinations",
             4: "interior classes with no known special role"}.items())]))
    tier1_tpls = [t for t in op.params.get("tier1_templates", [])]
    if tier1_tpls:
        a(f"Value templates {', '.join(tier1_tpls)} are the combined "
          "boundary cases flagged by the cross-domain dependency check: "
          "their tests are assigned tier 1.\n")
    prov: Dict[tuple, int] = {}
    for c in list(result.classes) + list(result.branches):
        key = (c.priority, c.criterion or "-", c.retention_reason)
        prov[key] = prov.get(key, 0) + 1
    a(_md_table(
        ["prio", "criterion", "retention reason", "classes"],
        [[str(k[0]), k[1], k[2], str(n)]
         for k, n in sorted(prov.items(), key=lambda kv: (kv[0][0], kv[0][1]))]))
    a("Per-test priorities (dependency-flagged templates raise a test "
      "to tier 1) are recorded in each test JSON and in the manifest.\n")

    if result.unsat:
        a(f"## Unsatisfiable combinations ({len(result.unsat)})\n")
        a("Detected by the solver and eliminated (strategy: they cannot "
          "carry a test).\n")
        rows = [[_choice_math(u.label),
                 _pred_math(u.predicates)] for u in result.unsat]
        a(_md_table(["combination", "predicates"], rows))

    if result.pruned:
        a(f"## Pruned combinations ({len(result.pruned)}) -- recorded "
          f"engineering judgments\n")
        a("Each exclusion records the fault model assumed absent, stated "
          "concretely enough that a reviewer could disagree (strategy: "
          "'Eliminating irrelevant cases').\n")
        rows = [[_choice_math(p.label), p.fault_model_absent,
                 p.disposition] for p in result.pruned]
        a(_md_table(["combination", "fault model assumed absent",
                     "disposition"], rows))

    a("## Cross-domain dependency table (strategy check)\n")
    rows = [[d.candidate, "yes" if d.found else "no", d.combined_test]
            for d in op.dependencies]
    a(_md_table(["candidate dependency", "found?",
                 "combined boundary test"], rows))

    catalog_entries = _catalog_entries(op)
    if catalog_entries:
        a("## Cross-operator pattern catalog entries cited\n")
        a("Rows of the dependency table above cite entries of the shared, "
          "living pattern catalog (strategy: 'Maintaining a cross-operator "
          "pattern catalog'); operators sharing the listed structural "
          "features are flagged for re-review when an entry evolves.\n")
        for e in catalog_entries:
            a(f"- **{e['id']}** -- {e['title']}. {e['description']} "
              f"*Action:* {e['action']} *Origin:* {e['origin']}")
        a("")

    if result.duplicates:
        a("## Duplicate witnesses\n")
        a("Classes whose witness coincides with another class's witness; "
          "the classes remain distinct per the strategy but the fact is "
          "recorded for review.\n")
        rows = sorted(result.duplicates.items())
        a(_md_table(["class", "same witness as"], rows))

    if gaps:
        a(f"## Specification gaps -- materialization refused "
          f"({len(gaps)} case(s))\n")
        reasons: List[str] = []
        for g in gaps:
            if g.get("reason") and g["reason"] not in reasons:
                reasons.append(g["reason"])
        for r in reasons:
            a(f"{r}.")
            a("")
        a("No functional test is generated for these configurations; "
          "they are reported for spec clarification (test objective 1: "
          "validating the specification).\n")
        rows = [[g["case_id"], g["class"], str(g["inputs"]),
                 str(g["windows"])] for g in gaps]
        a(_md_table(["case", "class", "configuration", "window(s)"], rows))

    if op.notes:
        a("## Notes and specification gaps\n")
        for n in op.notes:
            a(f"- {n}")
        a("")

    a("## Totals\n")
    a(_md_table(["quantity", "value"], [
        ["constraint predicates (split system)", len(op.constraints)],
        ["combinations enumerated", len(result.classes)
         + len(result.unsat) + len(result.pruned)],
        ["equivalence classes (SAT)", len(result.classes)],
        ["auxiliary cases (boundary/property)", len(result.branches)],
        ["unsatisfiable combinations", len(result.unsat)],
        ["pruned combinations", len(result.pruned)],
        ["functional tests generated", len(functional)],
        ["materializations refused (specification gaps)", len(gaps)],
    ]))
    out_names = ", ".join(op.outputs) if op.outputs else "the operator's"
    a("Test cases are JSON files next to this report; each carries its "
      "own traceability tags, class description, input tensor and "
      f"expected outputs ({out_names}) computed by the reference "
      "implementation of the informal specification.\n")
    return "\n".join(lines)


def _params_str(op: Operator, witness: Dict[str, object]) -> str:
    parts = []
    for v in op.variables:
        w = witness[v.name]
        if isinstance(w, ValueClass):
            val = rf"\text{{{w.cls}}}"
        else:
            val = str(w)
        parts.append(_math(f"{var_latex(v)} = {val}"))
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def emit(op: Operator, result: ClassifyResult, out_root: str) -> List[str]:
    """Write per-dtype test sets; returns the output directories."""
    out_dirs: List[str] = []
    for dtype in op.params.get("dtypes", [""]):
        out_dir = os.path.join(out_root, f"{op.name}_{dtype}")
        os.makedirs(out_dir, exist_ok=True)

        functional: List[Dict[str, object]] = []
        gaps: List[Dict[str, object]] = []
        for cls in result.classes:
            tests, gap = functional_tests(
                cls.id, "equivalence-class",
                f"{cls.label} ({cls.describe()})",
                [p.render() for p in cls.predicates],
                _traceability(cls.predicates), cls.witness,
                op.spec_ref, dtype)
            functional.extend(tests)
            if gap:
                gaps.append(gap)
        for br in result.branches:
            tests, gap = functional_tests(
                br.id, f"auxiliary-{br.kind}", br.name,
                [p.render() for p in br.predicates],
                _traceability(br.predicates) + ([br.origin] if br.origin
                                                else []),
                br.witness, op.spec_ref, dtype)
            functional.extend(tests)
            if gap:
                gaps.append(gap)

        # Class-provenance and prioritization metadata on every test, then
        # tier ordering of the emitted test list (strategy: "Test
        # traceability" + "Risk-based prioritization of test generation").
        meta = {c.id: (c.criterion, c.retention_reason, c.priority)
                for c in list(result.classes) + list(result.branches)}
        tier1_tpls = set(op.params.get("tier1_templates", []))
        for t in functional:
            criterion, retention, prio = meta[t["class_id"]]
            tpl_id = t["value_template"]["id"]
            if tpl_id in tier1_tpls:
                prio = 1
                retention = (
                    f"{retention}; raised to tier 1 as a dependency-flagged "
                    f"combined boundary case (template {tpl_id}, "
                    f"cross-domain dependency check)")
            t["criterion"] = criterion
            t["retention_reason"] = retention
            t["priority"] = prio
        functional.sort(key=lambda t: (t["priority"], t["id"]))

        for t in functional:
            with open(os.path.join(out_dir, f"{t['id']}.json"), "w") as fh:
                json.dump(t, fh, indent=2)
                fh.write("\n")

        manifest = {
            "operator": op.name,
            "dtype": dtype,
            "spec_file": op.spec_file,
            "spec_ref": op.spec_ref,
            "restrictions": op.params.get("restrictions", []),
            "solver_backend": result.backend,
            "variables": _variable_docs(op),
            "usage_limits": {
                "records": _usage_limit_records(op),
                "cross_product_pruning": USAGE_LIMIT_CROSS_PRODUCT,
            },
            "pattern_catalog": _catalog_entries(op),
            "domain": [p.render() for p in op.domain],
            "constraint_system": [
                {"origin": p.origin, "criterion": p.criterion,
                 "note": p.note, "predicate": p.render()}
                for p in op.constraints
            ],
            "equivalence_classes": [
                {
                    "id": c.id,
                    "label": c.label,
                    "predicates": [p.render() for p in c.predicates],
                    "traceability": _traceability(c.predicates),
                    "criterion": c.criterion,
                    "retention_reason": c.retention_reason,
                    "priority": c.priority,
                    "priority_name": tier_name(c.priority),
                    "inputs": _witness_json(op, c.witness),
                    "duplicate_of": c.duplicate_of,
                }
                for c in result.classes
            ],
            "auxiliary_classes": [
                {
                    "id": b.id,
                    "kind": b.kind,
                    "name": b.name,
                    "origin": b.origin,
                    "predicates": [p.render() for p in b.predicates],
                    "criterion": b.criterion,
                    "retention_reason": b.retention_reason,
                    "priority": b.priority,
                    "priority_name": tier_name(b.priority),
                    "inputs": _witness_json(op, b.witness),
                }
                for b in result.branches
            ],
            "unsatisfiable_combinations": [
                {"label": u.label,
                 "predicates": [p.render() for p in u.predicates]}
                for u in result.unsat
            ],
            "pruned_combinations": [
                {
                    "label": p.label,
                    "predicates": [q.render() for q in p.predicates],
                    "fault_model_assumed_absent": p.fault_model_absent,
                    "disposition": p.disposition,
                }
                for p in result.pruned
            ],
            "dependency_table": [
                {"candidate": d.candidate, "found": d.found,
                 "catalog_refs": list(d.catalog_refs),
                 "combined_test": d.combined_test}
                for d in op.dependencies
            ],
            "duplicates": result.duplicates,
            "notes": op.notes,
            "specification_gaps": gaps,
            "value_templates": TEMPLATES,
            "totals": {
                "constraint_predicates": len(op.constraints),
                "combinations_enumerated": len(result.classes)
                + len(result.unsat) + len(result.pruned),
                "equivalence_classes": len(result.classes),
                "auxiliary_classes": len(result.branches),
                "unsatisfiable": len(result.unsat),
                "pruned": len(result.pruned),
                "functional_tests": len(functional),
                "tests_by_priority": {
                    str(p): sum(1 for t in functional
                                if t["priority"] == p)
                    for p in sorted({t["priority"] for t in functional})
                },
            },
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")

        with open(os.path.join(out_dir, "report.md"), "w") as fh:
            fh.write(_report(op, result, dtype, functional, gaps))

        out_dirs.append(out_dir)
    return out_dirs