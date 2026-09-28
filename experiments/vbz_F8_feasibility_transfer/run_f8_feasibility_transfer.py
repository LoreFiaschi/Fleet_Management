"""Test whether a feasible F6 schedule transfers to the matching F8 model.

The experiment has three cases:

``fixed_transfer``
    Embed the F6 assignment/maintenance decisions in F8, set both added
    vehicles to no activity, and fix every shared binary decision.  A feasible
    result proves that the transferred schedule can be completed by continuous
    degradation states in the current formulation.

``warm_start``
    Supply the same binary decisions as a complete MIP start, but leave all
    variables free.  Stop after the first feasible solution.

``cold_start``
    Solve the identical free F8 model without a MIP start and stop after the
    first feasible solution.

This is an experiment runner, not part of the public solver interface.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import yaml

from fleet_management.config import load_config
from fleet_management.degradation_model.base import (
    build_fleet,
    extract_solution,
    get_cell_builder,
    resolve_run_options,
)
from fleet_management.degradation_model.gamma_utils.gamma_diagnostics import (
    collect_gurobi_model_statistics,
    compare_estimate_with_actual,
    estimate_gamma_formulation,
)
from fleet_management.solver import _save_results
from fleet_management import validate_gamma_schedule_files


EXPERIMENT_MODES = ("fixed_transfer", "warm_start", "cold_start")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _git(command: str) -> str:
    completed = subprocess.run(
        ["git", *command.split()], capture_output=True, text=True, check=False,
    )
    return completed.stdout.strip()


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(np.asarray(value).shape)


def _assert_matched_inputs(f6_data: dict[str, Any], f8_data: dict[str, Any]) -> None:
    """Require an exact input match except for the intended fleet size."""
    if int(f6_data.get("F", -1)) != 6 or int(f8_data.get("F", -1)) != 8:
        raise ValueError("the source and target inputs must have F=6 and F=8")
    left, right = dict(f6_data), dict(f8_data)
    left.pop("F")
    right.pop("F")
    if left != right:
        differing = sorted(
            key for key in set(left) | set(right) if left.get(key) != right.get(key)
        )
        raise ValueError(
            "F6 and F8 inputs differ in fields other than F: "
            + ", ".join(differing)
        )


def _load_binary_start(result_path: Path, cfg) -> dict[str, np.ndarray]:
    source = _read_yaml(result_path)
    if source.get("objective") is None:
        raise ValueError("the F6 source result contains no feasible schedule")
    if int(source.get("F", -1)) != 6:
        raise ValueError("the source result must describe F=6")
    for key, expected in {
        "x": (6, cfg.M + 1, cfg.T),
        "m": (6, cfg.L, cfg.T),
        "r": (6, cfg.L, cfg.T),
    }.items():
        if _shape(source.get(key)) != expected:
            raise ValueError(
                f"source {key} has shape {_shape(source.get(key))}, expected {expected}"
            )

    decisions = {
        key: np.rint(np.asarray(source[key], dtype=float)).astype(int)
        for key in ("x", "m", "r")
    }
    for key, values in decisions.items():
        original = np.asarray(source[key], dtype=float)
        if not np.allclose(original, values, atol=1e-6, rtol=0.0):
            raise ValueError(f"source {key} contains non-binary values")
        if np.any((values < 0) | (values > 1)):
            raise ValueError(f"source {key} contains values outside {{0,1}}")

    # The copied assignment must still cover every mission exactly once.
    if not np.all(np.sum(decisions["x"][:, 1:, :], axis=0) == 1):
        raise ValueError("the source assignment does not cover each mission once")
    return {"source": source, **decisions}


def _target_binary_values(start: dict[str, Any], cfg) -> dict[str, np.ndarray]:
    """Embed F6 in F8; the two added vehicles receive no activity or repair."""
    x = np.zeros((cfg.F, cfg.M + 1, cfg.T), dtype=int)
    m = np.zeros((cfg.F, cfg.L, cfg.T), dtype=int)
    r = np.zeros((cfg.F, cfg.L, cfg.T), dtype=int)
    x[:6] = start["x"]
    m[:6] = start["m"]
    r[:6] = start["r"]
    nb = 1 - m - r
    if np.any(nb < 0):
        raise ValueError("the source selects repair and replacement simultaneously")
    return {"x": x, "m": m, "r": r, "nb": nb}


def _set_value(variable, value: int, *, fixed: bool) -> None:
    value = int(value)
    if fixed:
        variable.LB = value
        variable.UB = value
    else:
        variable.Start = value


def _apply_binary_schedule(ctx, values: dict[str, np.ndarray], *, fixed: bool) -> int:
    assigned = 0
    for i in range(ctx.F):
        for k in range(ctx.T):
            for j in range(ctx.M + 1):
                _set_value(ctx.x[i, j, k], values["x"][i, j, k], fixed=fixed)
                assigned += 1
            for l in range(ctx.L):
                _set_value(ctx.m_rep[i, l, k], values["m"][i, l, k], fixed=fixed)
                _set_value(ctx.nb[i, l, k], values["nb"][i, l, k], fixed=fixed)
                assigned += 2
                if ctx.allow_replacement:
                    _set_value(ctx.r_rep[i, l, k], values["r"][i, l, k], fixed=fixed)
                    assigned += 1
    return assigned


def _first_solution_callback(holder: dict[str, float | None]):
    def callback(model, where):
        if where == GRB.Callback.MIPSOL and holder["seconds"] is None:
            holder["seconds"] = float(model.cbGet(GRB.Callback.RUNTIME))
    return callback


def _extract_full_result(ctx, cfg, *, construction_seconds: float,
                         optimizer_seconds: float, backend_start: float) -> dict:
    extraction_start = time.perf_counter()
    out = extract_solution(ctx, cfg, ctx.model)
    for name in sorted({str(value) for value in np.asarray(cfg.model).ravel()}):
        hook = getattr(get_cell_builder(name), "extract", None)
        if hook is not None:
            hook(ctx, cfg, out)
    extraction_seconds = time.perf_counter() - extraction_start

    performance = collect_gurobi_model_statistics(ctx.model)
    performance.update({
        "model_construction_seconds": construction_seconds,
        "optimizer_call_seconds": optimizer_seconds,
        "solution_extraction_seconds": extraction_seconds,
        "backend_wall_seconds": time.perf_counter() - backend_start,
        "requested_time_limit_seconds": float(ctx.model.Params.TimeLimit),
        "requested_mip_gap": float(ctx.model.Params.MIPGap),
        "threads": int(ctx.model.Params.Threads),
        "seed": int(ctx.model.Params.Seed),
    })
    if "gamma" in ctx.extras:
        performance["gamma_calibration_seconds"] = float(sum(
            ctx.extras["gamma"]["calibration_seconds"].values()
        ))
        formulation = estimate_gamma_formulation(
            cfg, allow_replacement=ctx.allow_replacement,
        )
        formulation["actual_gurobi_model"] = {
            key: performance[key]
            for key in (
                "variables", "continuous_variables", "integer_variables",
                "binary_variables", "linear_constraints", "general_constraints",
                "indicator_constraints", "quadratic_constraints", "nonzeros",
            )
        }
        formulation["comparison"] = compare_estimate_with_actual(
            formulation, formulation["actual_gurobi_model"],
        )
        out["gamma_formulation"] = formulation

    out.update({
        "backend": "modular",
        "degradation": "mixed",
        "models": cfg.models,
        "model_assignment": cfg.model.astype(str).tolist(),
        "component_names": list(cfg.component_names),
        "performance": performance,
    })
    return out


def _build_case(cfg, mode: str, values: dict[str, np.ndarray], output_dir: Path,
                fixed_time_limit: float, free_time_limit: float,
                *, build_only: bool) -> dict[str, Any]:
    backend_start = time.perf_counter()
    construction_start = time.perf_counter()
    ctx = build_fleet(cfg, resolve_run_options(cfg), model_name=f"f8_{mode}")
    ctx.model.update()
    construction_seconds = time.perf_counter() - construction_start

    use_schedule = mode in {"fixed_transfer", "warm_start"}
    assigned = 0
    if use_schedule:
        assigned = _apply_binary_schedule(
            ctx, values, fixed=(mode == "fixed_transfer"),
        )

    ctx.model.Params.TimeLimit = (
        fixed_time_limit if mode == "fixed_transfer" else free_time_limit
    )
    ctx.model.Params.SolutionLimit = 1
    ctx.model.Params.LogFile = str(output_dir / f"{mode}.log")
    ctx.model.update()
    formulation = collect_gurobi_model_statistics(ctx.model)

    case = {
        "mode": mode,
        "binary_schedule_use": (
            "fixed" if mode == "fixed_transfer"
            else "mip_start" if mode == "warm_start"
            else "none"
        ),
        "binary_values_assigned": assigned,
        "time_limit_seconds": float(ctx.model.Params.TimeLimit),
        "formulation": {
            key: formulation[key] for key in (
                "variables", "continuous_variables", "integer_variables",
                "binary_variables", "linear_constraints", "general_constraints",
                "indicator_constraints", "quadratic_constraints",
            )
        },
    }
    if build_only:
        case.update({"status": "build_only", "feasible_schedule_found": None})
        return case

    first_solution = {"seconds": None}
    optimizer_start = time.perf_counter()
    ctx.model.optimize(_first_solution_callback(first_solution))
    optimizer_seconds = time.perf_counter() - optimizer_start

    result = _extract_full_result(
        ctx, cfg,
        construction_seconds=construction_seconds,
        optimizer_seconds=optimizer_seconds,
        backend_start=backend_start,
    )
    result_path = output_dir / f"{mode}_result.yaml"
    _save_results(result, result_path)

    validation_path = None
    validation_valid = None
    if result.get("objective") is not None:
        validation_path = output_dir / f"{mode}_validation.yaml"
        validation = validate_gamma_schedule_files(
            output_dir / "f8_input_snapshot.yaml",
            result_path,
            validation_path,
            mode="deterministic",
            raise_on_failure=True,
        )
        validation_valid = bool(validation["valid"])

    case.update({
        "status": result.get("status"),
        "feasible_schedule_found": result.get("objective") is not None,
        "objective": result.get("objective"),
        "objective_bound": result.get("bound"),
        "mip_gap": result.get("mip_gap"),
        "first_feasible_solution_seconds": first_solution["seconds"],
        "gurobi_runtime_seconds": result["performance"].get("gurobi_runtime_seconds"),
        "branch_and_bound_nodes": result["performance"].get("branch_and_bound_nodes"),
        "deterministic_gamma_validation": validation_valid,
        "result": str(result_path),
        "validation": None if validation_path is None else str(validation_path),
        "log": str(output_dir / f"{mode}.log"),
    })
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f6-input", required=True, type=Path)
    parser.add_argument("--f8-input", required=True, type=Path)
    parser.add_argument("--f6-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument(
        "--modes", nargs="+", choices=EXPERIMENT_MODES,
        default=list(EXPERIMENT_MODES),
    )
    parser.add_argument("--fixed-time-limit", type=float, default=600.0)
    parser.add_argument("--free-time-limit", type=float, default=3600.0)
    parser.add_argument("--build-only", action="store_true")
    arguments = parser.parse_args()

    for path in (arguments.f6_input, arguments.f8_input, arguments.f6_result):
        if not path.is_file():
            raise FileNotFoundError(path)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    f6_data = _read_yaml(arguments.f6_input)
    f8_data = _read_yaml(arguments.f8_input)
    _assert_matched_inputs(f6_data, f8_data)
    cfg = load_config(f8_data)
    start = _load_binary_start(arguments.f6_result, cfg)
    target_values = _target_binary_values(start, cfg)

    # The exact input used for validation remains beside every result.
    _write_yaml(f8_data, arguments.output_dir / "f8_input_snapshot.yaml")
    _write_yaml(f6_data, arguments.output_dir / "f6_input_snapshot.yaml")
    source_validation_path = arguments.output_dir / "f6_source_validation.yaml"
    source_validation = validate_gamma_schedule_files(
        arguments.output_dir / "f6_input_snapshot.yaml",
        arguments.f6_result,
        source_validation_path,
        mode="deterministic",
        raise_on_failure=True,
    )

    summary: dict[str, Any] = {
        "report": "F6-to-F8 feasible-schedule transfer experiment",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "git_commit": _git("rev-parse HEAD"),
        "git_status": _git("status --short"),
        "build_only": arguments.build_only,
        "source": {
            "result": str(arguments.f6_result),
            "status": start["source"].get("status"),
            "objective": start["source"].get("objective"),
            "mip_gap": start["source"].get("mip_gap"),
            "fleet_size": 6,
            "deterministic_gamma_validation": bool(source_validation["valid"]),
            "validation": str(source_validation_path),
        },
        "target": {
            "input": str(arguments.f8_input),
            "F": cfg.F, "M": cfg.M, "L": cfg.L,
            "H1": cfg.H1, "H2": cfg.H2, "T": cfg.T,
            "added_vehicles": 2,
            "added_vehicle_policy": "all x, m, and r decisions set to zero",
        },
        "interpretation": {
            "fixed_transfer": (
                "Feasibility proves that the embedded F6 binary schedule can be "
                "completed by continuous states in the current F8 formulation."
            ),
            "warm_vs_cold": (
                "Compare first-feasible-solution time only. Both free cases use "
                "the same F8 input, limits, threads, seed, and SolutionLimit=1."
            ),
            "warning": (
                "A MIP start being supplied does not by itself prove that Gurobi "
                "accepted it; use the fixed-transfer result as the feasibility proof."
            ),
        },
        "cases": [],
    }

    for mode in arguments.modes:
        try:
            case = _build_case(
                cfg, mode, target_values, arguments.output_dir,
                arguments.fixed_time_limit, arguments.free_time_limit,
                build_only=arguments.build_only,
            )
        except Exception as error:
            case = {
                "mode": mode,
                "status": "runner_error",
                "feasible_schedule_found": False,
                "error": f"{type(error).__name__}: {error}",
            }
        summary["cases"].append(case)
        _write_yaml(summary, arguments.summary)

    print("\nF6 -> F8 FEASIBILITY TRANSFER")
    print("mode             status           feasible   first solution [s]")
    for case in summary["cases"]:
        first = case.get("first_feasible_solution_seconds")
        shown = "-" if first is None else f"{float(first):.3f}"
        print(
            f"{case['mode']:<16} {str(case.get('status')):<16} "
            f"{str(case.get('feasible_schedule_found')):<10} {shown}"
        )
    print("Summary:", arguments.summary)

    errors = [case for case in summary["cases"] if case.get("status") == "runner_error"]
    if errors:
        raise RuntimeError("one or more experiment cases failed; inspect the summary")


if __name__ == "__main__":
    main()
