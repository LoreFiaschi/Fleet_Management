from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from fleet_management.config import load_config
from fleet_management.greedy_warm_start import build_greedy_warm_start
from fleet_management.solver import solve


def first_incumbent_seconds(result: dict):
    for row in result.get("optimization_progress", []):
        if row.get("incumbent") is not None:
            return float(row["runtime_seconds"])
    return None


def compact(result: dict) -> dict:
    performance = result.get("performance", {})
    return {
        "status": result.get("status"),
        "objective": result.get("objective"),
        "bound": result.get("bound"),
        "mip_gap": result.get("mip_gap"),
        "first_incumbent_seconds": first_incumbent_seconds(result),
        "solutions_found": performance.get("solutions_found"),
        "runtime_seconds": performance.get("gurobi_runtime_seconds"),
        "nodes": performance.get("branch_and_bound_nodes"),
        "simplex_iterations": performance.get("simplex_iterations"),
        "warm_start": result.get("warm_start"),
        "relaxation_warm_start": result.get("relaxation_warm_start"),
    }


def mip_start_messages(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "MIP start" in line
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--repair-trigger", type=float, default=0.70)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    # This experiment isolates the explicit greedy start.
    raw["relaxation_warm_start"] = False
    cfg = load_config(raw)

    baseline_log = args.output_directory / "gurobi_baseline.log"
    baseline_raw = dict(raw)
    baseline_raw["gurobi_params"] = dict(raw.get("gurobi_params", {}))
    baseline_raw["gurobi_params"]["LogFile"] = str(baseline_log.resolve())
    baseline_input = args.output_directory / "input_baseline.yaml"
    baseline_input.write_text(
        yaml.safe_dump(baseline_raw, sort_keys=False), encoding="utf-8"
    )
    baseline = solve(
        str(baseline_input),
        str(args.output_directory / "result_baseline.yaml"),
    )

    start, greedy_report = build_greedy_warm_start(
        cfg, repair_trigger=args.repair_trigger
    )
    if not greedy_report.feasible_assignment:
        raise RuntimeError(
            "Greedy start does not cover every mission with a real vehicle."
        )
    if not greedy_report.repeatability_feasible:
        raise RuntimeError(
            "Greedy start cannot satisfy terminal repeatability with the "
            "available operating-phase depot slots."
        )
    greedy_log = args.output_directory / "gurobi_greedy.log"
    greedy_raw = dict(raw)
    greedy_raw["gurobi_params"] = dict(raw.get("gurobi_params", {}))
    greedy_raw["gurobi_params"]["LogFile"] = str(greedy_log.resolve())
    greedy_input = args.output_directory / "input_greedy.yaml"
    greedy_input.write_text(
        yaml.safe_dump(greedy_raw, sort_keys=False), encoding="utf-8"
    )
    greedy = solve(
        str(greedy_input),
        str(args.output_directory / "result_greedy.yaml"),
        warm_start=start,
    )

    summary = {
        "experiment": "greedy_warm_start_controlled_comparison",
        "input_dimensions": {
            "F": cfg.F, "M": cfg.M, "L": cfg.L,
            "H1": cfg.H1, "H2": cfg.H2, "T": cfg.T,
        },
        "greedy_construction": greedy_report.as_dict(),
        "baseline": {
            **compact(baseline),
            "mip_start_messages": mip_start_messages(baseline_log),
        },
        "greedy": {
            **compact(greedy),
            "mip_start_messages": mip_start_messages(greedy_log),
        },
    }
    (args.output_directory / "summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
