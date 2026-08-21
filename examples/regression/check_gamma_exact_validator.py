"""Regression for exact post-solve modular Gamma schedule validation."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from fleet_management import solve
from fleet_management.config import load_config
from fleet_management.degradation_model.base import (
    build_fleet,
    extract_solution,
    get_cell_builder,
    resolve_run_options,
)
from fleet_management.degradation_model.gamma_utils.gamma_tail_validator import (
    validate_gamma_tail_bound_files,
    validate_gamma_tail_bound_schedule,
)


HERE = Path(__file__).resolve().parent
PUBLIC_INPUT = HERE / "gamma_tail_bound_public.yaml"


def validate_public_replacement_case() -> dict:
    with TemporaryDirectory(prefix="gamma-exact-public-") as directory:
        result_path = Path(directory) / "result.yaml"
        report_path = Path(directory) / "validation.yaml"
        solve(str(PUBLIC_INPUT), str(result_path))
        report = validate_gamma_tail_bound_files(
            PUBLIC_INPUT,
            result_path,
            report_path,
            include_steps=True,
            raise_on_failure=True,
        )
        if not report_path.is_file():
            raise AssertionError("exact validation report was not saved")

        # A deliberately unsafe bound must invalidate the result rather than
        # merely appear as an informational diagnostic.
        unsafe_path = Path(directory) / "unsafe_result.yaml"
        unsafe = yaml.safe_load(result_path.read_text(encoding="utf-8"))
        unsafe["gamma_shape_bound"][1][0][0] = 0.0
        unsafe["gamma_tail_bound"][1][0][0] = 0.0
        unsafe_path.write_text(
            yaml.safe_dump(unsafe, sort_keys=False), encoding="utf-8"
        )
        unsafe_report = validate_gamma_tail_bound_files(PUBLIC_INPUT, unsafe_path)
        if unsafe_report["valid"]:
            raise AssertionError("unsafe Gamma bound did not invalidate the result")
        if not any(
            "non-conservative tail margin" in reason
            for item in unsafe_report["violations"]
            for reason in item["reasons"]
        ):
            raise AssertionError("unsafe result did not report its signed tail failure")

    if report["transitions_checked"] != 10:
        raise AssertionError("wrong number of public Gamma transitions")
    if report["replacements"] < 1:
        raise AssertionError("public exact validation did not exercise replacement")
    if report["repairs"] != 0:
        raise AssertionError("public replacement case unexpectedly used repair")
    return report


def validate_forced_ardinf_case() -> dict:
    cfg = load_config(
        {
            "F": 2,
            "M": 1,
            "L": 1,
            "H": [2, 2],
            "model": "gamma",
            "repair_model": "ardinf",
            "tau": 0.6,
            "epsilon": 0.1,
            "rho": 0.5,
            "mu_0": 0.02,
            "replacement_mu": 0.01,
            "mu": 0.05,
            "gamma_beta": 10.0,
            "gamma_beta_trans": 10.0,
            "gamma_beta_bound": 10.0,
            "gamma_beta_0": 10.0,
            "gamma_beta_new": 10.0,
            "C_M": 1.0,
            "C_R": 0.5,
            "C_D": 2.0,
            "C_rep": 0.2,
            "allow_replacement": True,
            "depot_capacity": 1,
            "mip_gap": 0.0,
            "verbose": 0,
        }
    )
    ctx = build_fleet(
        cfg,
        resolve_run_options(cfg),
        model_name="gamma_exact_validator_repair_test",
    )
    ctx.model.addConstr(ctx.m_rep[0, 0, 0] == 1, name="force_gamma_repair")
    ctx.model.optimize()
    if ctx.model.SolCount == 0:
        raise AssertionError("forced-repair model has no incumbent")

    result = extract_solution(ctx, cfg, ctx.model)
    get_cell_builder("gamma").extract(ctx, cfg, result)
    result["backend"] = "modular"
    result["degradation"] = "gamma"

    report = validate_gamma_tail_bound_schedule(
        cfg,
        result,
        include_steps=True,
        raise_on_failure=True,
    )
    if report["repairs"] < 1:
        raise AssertionError("exact validation did not replay the forced repair")

    repaired_step = next(
        step
        for step in report["steps"]
        if step["i"] == 0 and step["l"] == 0 and step["k"] == 0
    )
    if repaired_step["event"] != "repair":
        raise AssertionError("first forced transition was not identified as repair")
    if len(repaired_step["history"]) != 1:
        raise AssertionError("repair did not preserve the initial Gamma term")
    term = repaired_step["history"][0]
    if abs(term["shape"] - 0.2) > 1e-10:
        raise AssertionError("ARD-inf changed the exact Gamma shape")
    if abs(term["rate"] - 20.0) > 1e-10:
        raise AssertionError("ARD-inf did not transform beta to beta/(1-rho)")
    if abs(term["mean"] - 0.01) > 1e-10:
        raise AssertionError("ARD-inf exact repaired mean is wrong")
    return report


def main() -> None:
    public = validate_public_replacement_case()
    repair = validate_forced_ardinf_case()

    print("PASS exact post-solve Gamma tail validation")
    print("public transitions :", public["transitions_checked"])
    print("public replacements:", public["replacements"])
    print("public worst margin:", public["minimum_conservativeness_margin"])
    print("repair transitions :", repair["transitions_checked"])
    print("Gamma repairs      :", repair["repairs"])
    print("repair worst margin:", repair["minimum_conservativeness_margin"])
    print("maximum remainder  :", max(
        public["numerics"]["maximum_remaining_mass"],
        repair["numerics"]["maximum_remaining_mass"],
    ))


if __name__ == "__main__":
    main()
