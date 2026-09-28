"""Audit how Gurobi presolve treats indicator constraints.

The report builds, but does not optimize, each supplied fleet instance.  It
then compares the presolved model under three SOS1 reformulation settings:

* automatic (the experiment/default setting),
* disabled (PreSOS1BigM=0),
* permissive (PreSOS1BigM supplied by --maximum-big-m).

This comparison is more informative than merely observing that general
constraints disappear: presolve may also prove rows redundant or eliminate
the complete model.  The generated LP files permit manual verification.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path

import gurobipy as gp
import yaml
from gurobipy import GRB

from fleet_management.config import load_config
from fleet_management.degradation_model.base import (
    build_fleet,
    resolve_run_options,
)


GENCONSTR_NAMES = {
    GRB.GENCONSTR_MAX: "max",
    GRB.GENCONSTR_MIN: "min",
    GRB.GENCONSTR_ABS: "abs",
    GRB.GENCONSTR_AND: "and",
    GRB.GENCONSTR_OR: "or",
    GRB.GENCONSTR_NORM: "norm",
    GRB.GENCONSTR_INDICATOR: "indicator",
    GRB.GENCONSTR_PWL: "pwl",
    GRB.GENCONSTR_POLY: "poly",
    GRB.GENCONSTR_EXP: "exp",
    GRB.GENCONSTR_EXPA: "expa",
    GRB.GENCONSTR_LOG: "log",
    GRB.GENCONSTR_LOGA: "loga",
    GRB.GENCONSTR_POW: "pow",
    GRB.GENCONSTR_SIN: "sin",
    GRB.GENCONSTR_COS: "cos",
    GRB.GENCONSTR_TAN: "tan",
}


def safe_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def general_constraint_type(constraint: gp.GenConstr) -> int:
    """Return a general-constraint type across supported gurobipy APIs.

    In current gurobipy releases ``GenConstrType`` belongs to the GenConstr
    object.  Keep the explicit getAttr fallback for releases/environments that
    do not expose attributes through Python property access.
    """
    try:
        return int(constraint.GenConstrType)
    except AttributeError:
        return int(constraint.getAttr(GRB.Attr.GenConstrType))


def statistics(model: gp.Model) -> dict:
    model.update()
    type_counts = Counter()
    indicator_names = []
    for constraint in model.getGenConstrs():
        kind = general_constraint_type(constraint)
        name = GENCONSTR_NAMES.get(kind, f"type_{kind}")
        type_counts[name] += 1
        if kind == GRB.GENCONSTR_INDICATOR and len(indicator_names) < 25:
            indicator_names.append(constraint.GenConstrName)

    attrs = {}
    for name in (
        "NumVars", "NumBinVars", "NumIntVars", "NumConstrs", "NumQConstrs",
        "NumSOS", "NumGenConstrs", "NumNZs", "MinCoeff", "MaxCoeff",
        "MinRHS", "MaxRHS", "MinBound", "MaxBound",
    ):
        try:
            attrs[name] = safe_number(model.getAttr(name))
        except (AttributeError, gp.GurobiError):
            attrs[name] = None
    attrs["general_constraint_types"] = dict(sorted(type_counts.items()))
    attrs["indicator_names_sample"] = indicator_names
    return attrs


def presolve_variant(
    original: gp.Model,
    *,
    output: Path,
    pre_sos1_big_m: float | None,
) -> dict:
    work = original.copy()
    work.Params.OutputFlag = 0
    if pre_sos1_big_m is not None:
        work.Params.PreSOS1BigM = float(pre_sos1_big_m)
    selected = float(work.Params.PreSOS1BigM)
    presolved = work.presolve()
    presolved.write(str(output))
    result = statistics(presolved)
    result["PreSOS1BigM"] = selected
    result["lp_file"] = str(output)
    return result


def interpretation(original: dict, variants: dict[str, dict]) -> dict:
    original_indicators = int(
        original.get("general_constraint_types", {}).get("indicator", 0)
    )
    automatic = variants["automatic"]
    disabled = variants["disabled"]
    permissive = variants["permissive"]

    if original_indicators == 0:
        return {
            "classification": "no_indicators_in_original_model",
            "explanation": "The built model contains no indicator constraints.",
        }
    if int(automatic.get("NumVars") or 0) == 0:
        return {
            "classification": "inconclusive_complete_presolve_elimination",
            "explanation": (
                "Automatic presolve eliminated the complete model, so the "
                "absence of indicators/SOS1 does not demonstrate Big-M conversion."
            ),
        }

    automatic_indicators = int(
        automatic.get("general_constraint_types", {}).get("indicator", 0)
    )
    automatic_sos = int(automatic.get("NumSOS") or 0)
    disabled_indicators = int(
        disabled.get("general_constraint_types", {}).get("indicator", 0)
    )
    disabled_sos = int(disabled.get("NumSOS") or 0)
    permissive_indicators = int(
        permissive.get("general_constraint_types", {}).get("indicator", 0)
    )
    permissive_sos = int(permissive.get("NumSOS") or 0)

    if (
        automatic_indicators == 0
        and automatic_sos == 0
        and (disabled_indicators > 0 or disabled_sos > 0)
    ):
        return {
            "classification": "strong_evidence_of_automatic_big_m_reformulation",
            "explanation": (
                "Indicators/SOS1 disappear under automatic presolve but remain "
                "when SOS1-to-binary reformulation is disabled. This is strong "
                "instance-specific evidence of linear Big-M reformulation."
            ),
        }
    if automatic_sos > 0 or automatic_indicators > 0:
        return {
            "classification": "automatic_presolve_retains_logical_structure",
            "explanation": (
                "Automatic presolve retains indicator/SOS1 structure; do not "
                "claim that every indicator was converted to Big-M."
            ),
        }
    if permissive_indicators == 0 and permissive_sos == 0 and (
        disabled_indicators > 0 or disabled_sos > 0
    ):
        return {
            "classification": "big_m_possible_but_not_proven_under_automatic_setting",
            "explanation": (
                "A permissive threshold removes the logical structure while "
                "the disabled run retains it. Automatic presolve is not "
                "sufficiently distinct for a stronger claim."
            ),
        }
    return {
        "classification": "inconclusive",
        "explanation": (
            "The count comparison cannot distinguish reformulation from other "
            "presolve eliminations. Inspect the emitted LP files and presolve log."
        ),
    }


def audit_case(
    path: Path,
    output_directory: Path,
    *,
    h1: int | None,
    h2: int | None,
    maximum_big_m: float,
) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    if (h1 is None) != (h2 is None):
        raise ValueError("--h1 and --h2 must be supplied together")
    if h1 is not None:
        raw["H"] = [int(h1), int(h2)]

    cfg = load_config(raw)
    opts = resolve_run_options(
        cfg,
        verbose=0,
        relaxation_warm_start=False,
    )
    context = build_fleet(cfg, opts, model_name=f"indicator_audit_{path.stem}")
    original_model = context.model
    original_model.update()

    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    case_directory = output_directory / label
    case_directory.mkdir(parents=True, exist_ok=True)
    original_lp = case_directory / "original.lp"
    original_model.write(str(original_lp))

    original = statistics(original_model)
    original["lp_file"] = str(original_lp)
    variants = {
        "automatic": presolve_variant(
            original_model,
            output=case_directory / "presolved_automatic.lp",
            pre_sos1_big_m=None,
        ),
        "disabled": presolve_variant(
            original_model,
            output=case_directory / "presolved_sos1_big_m_disabled.lp",
            pre_sos1_big_m=0.0,
        ),
        "permissive": presolve_variant(
            original_model,
            output=case_directory / "presolved_sos1_big_m_permissive.lp",
            pre_sos1_big_m=maximum_big_m,
        ),
    }
    return {
        "input": str(path),
        "dimensions": {
            "F": cfg.F, "M": cfg.M, "L": cfg.L,
            "H1": cfg.H1, "H2": cfg.H2, "T": cfg.T,
        },
        "models": cfg.models,
        "reliability_impl": raw.get("reliability_impl"),
        "original": original,
        "presolved": variants,
        "interpretation": interpretation(original, variants),
    }


def csv_row(case: dict) -> dict:
    original = case["original"]
    automatic = case["presolved"]["automatic"]
    disabled = case["presolved"]["disabled"]
    permissive = case["presolved"]["permissive"]

    def indicator_count(stats):
        return stats.get("general_constraint_types", {}).get("indicator", 0)

    return {
        "input": case["input"],
        **case["dimensions"],
        "original_indicators": indicator_count(original),
        "original_sos": original.get("NumSOS"),
        "automatic_indicators": indicator_count(automatic),
        "automatic_sos": automatic.get("NumSOS"),
        "automatic_linear_rows": automatic.get("NumConstrs"),
        "disabled_indicators": indicator_count(disabled),
        "disabled_sos": disabled.get("NumSOS"),
        "disabled_linear_rows": disabled.get("NumConstrs"),
        "permissive_indicators": indicator_count(permissive),
        "permissive_sos": permissive.get("NumSOS"),
        "permissive_linear_rows": permissive.get("NumConstrs"),
        "classification": case["interpretation"]["classification"],
    }


def compact_summary(case: dict) -> dict:
    """Return presentation-ready, instance-specific presolve evidence."""

    def count(stats: dict, kind: str) -> int:
        return int(
            stats.get("general_constraint_types", {}).get(kind, 0) or 0
        )

    original = case["original"]
    automatic = case["presolved"]["automatic"]
    disabled = case["presolved"]["disabled"]
    permissive = case["presolved"]["permissive"]
    automatic_signature = (
        count(automatic, "indicator"),
        int(automatic.get("NumSOS") or 0),
        int(automatic.get("NumVars") or 0),
        int(automatic.get("NumConstrs") or 0),
        int(automatic.get("NumNZs") or 0),
    )
    permissive_signature = (
        count(permissive, "indicator"),
        int(permissive.get("NumSOS") or 0),
        int(permissive.get("NumVars") or 0),
        int(permissive.get("NumConstrs") or 0),
        int(permissive.get("NumNZs") or 0),
    )
    classification = case["interpretation"]["classification"]
    return {
        "scope": "instance_specific",
        "input": case["input"],
        "dimensions": case["dimensions"],
        "original_indicators": count(original, "indicator"),
        "automatic_presolved_indicators": count(automatic, "indicator"),
        "automatic_presolved_sos1": int(automatic.get("NumSOS") or 0),
        "disabled_conversion_presolved_sos1": int(
            disabled.get("NumSOS") or 0
        ),
        "permissive_big_m_presolved_indicators": count(
            permissive, "indicator"
        ),
        "permissive_big_m_presolved_sos1": int(
            permissive.get("NumSOS") or 0
        ),
        "automatic_matches_permissive_counts": (
            automatic_signature == permissive_signature
        ),
        "no_logical_constraints_after_automatic_presolve": (
            count(automatic, "indicator") == 0
            and int(automatic.get("NumSOS") or 0) == 0
        ),
        "classification": classification,
        "strong_evidence_of_automatic_big_m_reformulation": (
            classification
            == "strong_evidence_of_automatic_big_m_reformulation"
        ),
        "caveat": (
            "Original indicators may be eliminated as redundant before the "
            "remaining logical constraints are reformulated; therefore this "
            "does not assert one Big-M reformulation per original indicator."
        ),
    }


def append_to_summary(path: Path, cases: list[dict]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Summary file does not exist: {path}")
    summary = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    evidence = [compact_summary(case) for case in cases]
    summary["indicator_presolve_audit"] = (
        evidence[0] if len(evidence) == 1 else evidence
    )
    path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h1", type=int)
    parser.add_argument("--h2", type=int)
    parser.add_argument("--maximum-big-m", type=float, default=1e6)
    parser.add_argument(
        "--append-summary",
        type=Path,
        help="Append compact presolve evidence to an existing summary YAML.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cases = [
        audit_case(
            path,
            args.output,
            h1=args.h1,
            h2=args.h2,
            maximum_big_m=args.maximum_big_m,
        )
        for path in args.inputs
    ]
    report = {
        "purpose": "indicator/SOS1/Big-M presolve audit",
        "caveat": (
            "The classification is instance-specific. Disappearance of an "
            "indicator alone is not proof of Big-M because presolve can also "
            "remove redundant constraints or variables."
        ),
        "maximum_big_m_tested": args.maximum_big_m,
        "cases": cases,
    }
    yaml_path = args.output / "indicator_presolve_report.yaml"
    yaml_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")

    rows = [csv_row(case) for case in cases]
    csv_path = args.output / "indicator_presolve_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if args.append_summary is not None:
        append_to_summary(args.append_summary, cases)

    for row in rows:
        print(yaml.safe_dump(row, sort_keys=False).strip())
        print()
    print(f"YAML: {yaml_path}")
    print(f"CSV : {csv_path}")
    if args.append_summary is not None:
        print(f"Updated summary: {args.append_summary}")


if __name__ == "__main__":
    main()
