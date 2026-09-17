from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from fleet_management.config import load_config
from fleet_management.greedy_warm_start import build_greedy_warm_start
from fleet_management.solver import solve


def _one_step_tangent_screen(cfg) -> dict:
    """Check whether each rainflow mission is feasible from its initial state."""
    failures = []
    reference_fraction = float(cfg.options.get("tangent_ref", 0.5))
    for i in range(cfg.F):
        for l in range(cfg.L):
            if str(cfg.model[i, l]) != "rainflow":
                continue
            if str(cfg.bound_method[i, l]) != "cantelli":
                continue
            tau = float(cfg.tau[i, l])
            epsilon = float(cfg.epsilon[i, l])
            coefficient = epsilon / (1.0 - epsilon)
            reference = float(np.clip(reference_fraction, 0.0, 1.0)) * tau
            distance = tau - reference
            intercept = coefficient * distance * distance
            slope = -2.0 * coefficient * distance
            for mission in range(cfg.M):
                mean = float(cfg.mu_0[i, l] + cfg.mu_trans[i, l, mission, 0])
                variance = float(cfg.v_0[i, l] + cfg.v_trans[i, l, mission, 0])
                cap = intercept + slope * (mean - reference)
                if mean > tau + 1e-12 or variance > cap + 1e-12:
                    failures.append({
                        "vehicle": i,
                        "component": l,
                        "mission": mission,
                        "mean": mean,
                        "variance": variance,
                        "tangent_cap": cap,
                    })
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "first_failures": failures[:20],
    }


def _mip_start_messages(path: Path) -> list[str]:
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
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--repair-trigger", type=float, default=0.70)
    arguments = parser.parse_args()
    arguments.run_directory.mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(arguments.input.read_text(encoding="utf-8"))
    log_path = (arguments.run_directory / "gurobi.log").resolve()
    raw["gurobi_params"] = dict(raw.get("gurobi_params", {}))
    raw["gurobi_params"]["LogFile"] = str(log_path)
    effective_input = arguments.run_directory / "input_effective.yaml"
    effective_input.write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )
    cfg = load_config(raw)

    screen = _one_step_tangent_screen(cfg)
    (arguments.run_directory / "one_step_reliability.yaml").write_text(
        yaml.safe_dump(screen, sort_keys=False), encoding="utf-8"
    )
    if not screen["passed"]:
        raise RuntimeError(
            "strict one-step tangent reliability screen failed; see "
            "one_step_reliability.yaml"
        )

    construction_start = time.perf_counter()
    start, report = build_greedy_warm_start(
        cfg, repair_trigger=arguments.repair_trigger
    )
    construction_seconds = time.perf_counter() - construction_start
    greedy = {
        **report.as_dict(),
        "construction_seconds": construction_seconds,
    }
    (arguments.run_directory / "greedy_report.yaml").write_text(
        yaml.safe_dump(greedy, sort_keys=False), encoding="utf-8"
    )

    use_start = report.feasible_assignment and report.repeatability_feasible
    if use_start:
        # Leave enough room inside the 16-hour allocation for model building,
        # exact fixed-binary completion, extraction and archiving.
        start["completion_time_limit"] = 600.0
    else:
        # The relaxation is only a fallback start generator. Its output is not
        # accepted as a valid schedule unless the original model subsequently
        # produces an incumbent.
        raw["relaxation_warm_start"] = True
        effective_input.write_text(
            yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
        )

    result_path = arguments.run_directory / "result.yaml"
    result = solve(
        str(effective_input),
        str(result_path),
        warm_start=start if use_start else None,
    )
    performance = result.get("performance", {})
    summary = {
        "experiment": "garage_hardau_strict_greedy_feasibility",
        "interpretation": (
            "synthetic reduced Garage-Hardau-inspired application; not a "
            "calibrated VBZ operational case"
        ),
        "dimensions": {
            "F": cfg.F, "M": cfg.M, "L": cfg.L,
            "H1": cfg.H1, "H2": cfg.H2, "T": cfg.T,
        },
        "safety": {
            "tau": sorted(set(np.asarray(cfg.tau).ravel().tolist())),
            "epsilon": sorted(set(np.asarray(cfg.epsilon).ravel().tolist())),
            "reliability_impl": raw.get("reliability_impl"),
        },
        "one_step_reliability": screen,
        "greedy": greedy,
        "greedy_submitted": use_start,
        "status": result.get("status"),
        "usable_incumbent": result.get("objective") is not None,
        "objective": result.get("objective"),
        "bound": result.get("bound"),
        "mip_gap": result.get("mip_gap"),
        "warm_start": result.get("warm_start"),
        "relaxation_warm_start": result.get("relaxation_warm_start"),
        "mip_start_messages": _mip_start_messages(log_path),
        "runtime_seconds": performance.get("gurobi_runtime_seconds"),
        "variables": performance.get("variables"),
        "binary_variables": performance.get("binary_variables"),
        "linear_constraints": performance.get("linear_constraints"),
        "general_constraints": performance.get("general_constraints"),
        "quadratic_constraints": performance.get("quadratic_constraints"),
        "nodes": performance.get("branch_and_bound_nodes"),
        "simplex_iterations": performance.get("simplex_iterations"),
    }
    (arguments.run_directory / "summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()

