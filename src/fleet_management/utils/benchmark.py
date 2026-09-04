"""Reproducible synthetic benchmarks for the common-beta Gamma backend.

Run the standard benchmark suite with::

    python -m fleet_management.benchmark --output-dir results/benchmarks

The large profile is deliberately opt-in::

    python -m fleet_management.benchmark --profiles small medium large
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


@dataclass(frozen=True)
class BenchmarkCase:
    """Dimensions of one deterministic synthetic fleet instance."""

    name: str
    vehicles: int
    missions: int
    components: int
    horizon: int


BENCHMARK_CASES = {
    "small": BenchmarkCase("small", vehicles=4, missions=2, components=2, horizon=4),
    "medium": BenchmarkCase("medium", vehicles=5, missions=3, components=3, horizon=6),
    "large": BenchmarkCase("large", vehicles=12, missions=6, components=4, horizon=12),
}


def build_synthetic_gamma_instance(
    case: BenchmarkCase,
    *,
    seed: int = 20260803,
    verbose: int = 0,
    mip_gap: float = 0.05,
) -> dict:
    """Create a reproducible, nontrivial common-beta Gamma input.

    The second horizon contains at least one maintenance opportunity per
    vehicle (``H >= F`` in the supplied profiles), which gives the optimizer
    room to satisfy the repeatability constraint using repair or replacement.
    """

    F = case.vehicles
    M = case.missions
    L = case.components
    H = case.horizon
    if F <= M:
        raise ValueError("A benchmark needs more vehicles than missions.")
    if H < F:
        raise ValueError(
            "Use H >= F so the repeated horizon can service every vehicle."
        )

    rng = np.random.default_rng(seed)

    # Low but heterogeneous damage increments keep the generated cases
    # feasible while ensuring assignment choices are not interchangeable.
    vehicle_factor = rng.uniform(0.85, 1.15, size=(F, 1, 1, 1))
    mission_factor = np.linspace(0.8, 1.35, M)[None, :, None, None]
    component_factor = np.linspace(0.9, 1.25, L)[None, None, :, None]
    time_factor = rng.uniform(0.9, 1.1, size=(1, 1, 1, H))
    mu = 0.006 * vehicle_factor * mission_factor * component_factor * time_factor

    mu_0 = rng.uniform(0.025, 0.055, size=(F, L))
    replacement_mu = rng.uniform(0.003, 0.008, size=(F, L))

    return {
        "F": F,
        "M": M,
        "L": L,
        "H": H,
        "mu": mu.tolist(),
        "mu_0": mu_0.tolist(),
        "replacement_mu": replacement_mu.tolist(),
        "tau": np.linspace(0.45, 0.60, L).tolist(),
        "gamma_beta": np.linspace(20.0, 32.0, L).tolist(),
        "repair_rho": np.linspace(0.55, 0.70, L).tolist(),
        "epsilon": 0.05,
        "C_M": 0.25,
        "C_R": 0.65,
        "C_rep": 3.0,
        "C_S": 0.15,
        "C_P": 0.10,
        "verbose": int(verbose),
        "mip_gap": float(mip_gap),
    }


def expected_gamma_model_size(case: BenchmarkCase) -> dict[str, int]:
    """Return model-size formulas before any Gurobi presolve."""

    F, M, L, H = (
        case.vehicles,
        case.missions,
        case.components,
        case.horizon,
    )
    steps = 2 * H
    binary = steps * F * ((M + 1) + 3 * L)  # x plus m, r and q
    continuous = steps * (2 * F * L + 1)  # A, z and u
    linear = (
        steps
        + 3 * F * L * steps
        + F * L
        + F * steps
        + (M + 1) * steps
    )
    general = 6 * F * L * steps
    return {
        "expected_variables": binary + continuous,
        "expected_binary_variables": binary,
        "expected_continuous_variables": continuous,
        "expected_linear_constraints": linear,
        "expected_general_constraints": general,
    }


def run_gamma_benchmarks(
    output_dir: str | Path,
    *,
    profiles: Iterable[str] = ("small", "medium"),
    repetitions: int = 1,
    seed: int = 20260803,
    verbose: int = 0,
    mip_gap: float = 0.05,
) -> list[dict]:
    """Generate, solve, validate and summarize synthetic Gamma cases."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    profile_names = tuple(profiles)

    # Lazy imports keep instance generation usable without a Gurobi install.
    from fleet_management.degradation_model.gamma_utils.gamma_validator import (
        validate_gamma_result,
    )
    from fleet_management.solver import solve

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for profile in profile_names:
        if profile not in BENCHMARK_CASES:
            raise ValueError(
                f"Unknown profile {profile!r}; choose from {sorted(BENCHMARK_CASES)}."
            )
        case = BENCHMARK_CASES[profile]
        for repetition in range(1, repetitions + 1):
            run_seed = seed + repetition - 1
            run_name = f"{profile}_run_{repetition:02d}"
            input_path = destination / f"{run_name}_input.yaml"
            result_path = destination / f"{run_name}_result.yaml"
            validation_path = destination / f"{run_name}_validation.yaml"

            generation_start = time.perf_counter()
            instance = build_synthetic_gamma_instance(
                case,
                seed=run_seed,
                verbose=verbose,
                mip_gap=mip_gap,
            )
            with input_path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(instance, stream, sort_keys=False)
            generation_seconds = time.perf_counter() - generation_start

            run_start = time.perf_counter()
            try:
                result = solve(
                    str(input_path),
                    degradation="gamma",
                    results_path=str(result_path),
                )
                validation = validate_gamma_result(
                    str(input_path),
                    str(result_path),
                    validation_path=str(validation_path),
                )
                row = _successful_row(
                    case=case,
                    repetition=repetition,
                    seed=run_seed,
                    generation_seconds=generation_seconds,
                    total_seconds=time.perf_counter() - run_start,
                    result=result,
                    validation=validation,
                )
                row.update(
                    input_path=str(input_path),
                    result_path=str(result_path),
                    validation_path=str(validation_path),
                )
            except Exception as error:  # preserve partial benchmark evidence
                row = {
                    **asdict(case),
                    **expected_gamma_model_size(case),
                    **_environment_metadata(),
                    "repetition": repetition,
                    "seed": run_seed,
                    "generation_seconds": generation_seconds,
                    "benchmark_wall_seconds": time.perf_counter() - run_start,
                    "status": "error",
                    "validation_passed": False,
                    "error": f"{type(error).__name__}: {error}",
                    "input_path": str(input_path),
                    "result_path": str(result_path),
                    "validation_path": str(validation_path),
                }
            rows.append(row)
            _write_csv(destination / "benchmark_runs.csv", rows)

    _write_aggregate_csv(destination / "benchmark_aggregate.csv", rows)
    with (destination / "benchmark_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "profiles": list(profile_names),
                "repetitions": repetitions,
                "base_seed": seed,
                "mip_gap": mip_gap,
                "verbose": verbose,
                "environment": _environment_metadata(),
                "cases": {name: asdict(case) for name, case in BENCHMARK_CASES.items()},
                "runs": rows,
            },
            stream,
            indent=2,
        )
    return rows


def _environment_metadata() -> dict[str, str]:
    """Record enough environment information to compare benchmark runs."""

    return {
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _successful_row(
    *,
    case: BenchmarkCase,
    repetition: int,
    seed: int,
    generation_seconds: float,
    total_seconds: float,
    result: dict,
    validation: dict,
) -> dict:
    performance = result.get("performance", {})
    summary = validation.get("summary", {})
    m = np.asarray(result.get("m", []), dtype=float)
    r = np.asarray(result.get("r", []), dtype=float)
    return {
        **asdict(case),
        **expected_gamma_model_size(case),
        **_environment_metadata(),
        "repetition": repetition,
        "seed": seed,
        "status": result.get("status"),
        "objective": result.get("objective"),
        "repairs": int(np.rint(m).sum()) if m.size else 0,
        "replacements": int(np.rint(r).sum()) if r.size else 0,
        "validation_passed": bool(validation.get("passed", False)),
        "validation_checks": len(validation.get("checks", [])),
        "generation_seconds": generation_seconds,
        "validation_seconds": summary.get("validation_wall_seconds"),
        "benchmark_wall_seconds": total_seconds,
        **performance,
        "error": "",
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_aggregate_csv(path: Path, rows: list[dict]) -> None:
    numeric_metrics = (
        "expected_variables",
        "expected_binary_variables",
        "expected_continuous_variables",
        "expected_linear_constraints",
        "expected_general_constraints",
        "variables",
        "binary_variables",
        "linear_constraints",
        "general_constraints",
        "nonzeros",
        "model_construction_seconds",
        "gurobi_runtime_seconds",
        "solution_extraction_seconds",
        "backend_wall_seconds",
        "solve_api_wall_seconds",
        "serialization_seconds",
        "validation_seconds",
        "benchmark_wall_seconds",
        "branch_and_bound_nodes",
        "simplex_iterations",
        "barrier_iterations",
        "work_units",
        "relative_mip_gap",
        "objective",
        "repairs",
        "replacements",
    )
    aggregates: list[dict] = []
    for profile in sorted({row["name"] for row in rows}):
        all_profile_rows = [row for row in rows if row["name"] == profile]
        group = [row for row in all_profile_rows if not row.get("error")]
        dimensions = all_profile_rows[0]
        aggregate: dict = {
            "profile": profile,
            "vehicles": dimensions["vehicles"],
            "missions": dimensions["missions"],
            "components": dimensions["components"],
            "horizon": dimensions["horizon"],
            "attempted_runs": len(all_profile_rows),
            "successful_runs": len(group),
            "failed_runs": len(all_profile_rows) - len(group),
            "all_valid": bool(group) and all(row["validation_passed"] for row in group),
        }
        for metric in numeric_metrics:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            if values:
                aggregate[f"{metric}_mean"] = statistics.fmean(values)
                aggregate[f"{metric}_min"] = min(values)
                aggregate[f"{metric}_max"] = max(values)
        aggregates.append(aggregate)
    _write_csv(path, aggregates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/benchmarks")
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(BENCHMARK_CASES),
        default=["small", "medium"],
        help="Large is opt-in because runtime is machine- and license-dependent.",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--mip-gap", type=float, default=0.05)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rows = run_gamma_benchmarks(
        args.output_dir,
        profiles=args.profiles,
        repetitions=args.repetitions,
        seed=args.seed,
        verbose=int(args.verbose),
        mip_gap=args.mip_gap,
    )
    for row in rows:
        print(
            f"{row['name']} run {row['repetition']}: "
            f"status={row['status']}, "
            f"runtime={row.get('gurobi_runtime_seconds', 'n/a')} s, "
            f"validated={row['validation_passed']}"
        )
    if any(row.get("error") for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
