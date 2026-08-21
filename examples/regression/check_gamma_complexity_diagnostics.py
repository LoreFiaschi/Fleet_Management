"""Regression for Gamma complexity counts and timing report fields."""

from __future__ import annotations

from report_gamma_complexity import build_complexity_report


def require_nonnegative(mapping: dict, keys: tuple[str, ...], where: str) -> None:
    for key in keys:
        if key not in mapping or mapping[key] is None or mapping[key] < 0.0:
            raise AssertionError(f"{where}.{key} is missing or negative")


def main() -> None:
    report = build_complexity_report()
    uniform = report["cases"]["uniform_gamma"]
    mixed = report["cases"]["mixed_gamma_rainflow"]
    ard1 = report["cases"]["mixed_gamma_ard1"]

    uniform_formulation = uniform["gurobi_formulation"]
    if not uniform_formulation["comparison"]["known_subtotal_matches_actual"]:
        raise AssertionError("uniform Gamma count estimate does not match Gurobi")
    if uniform_formulation["known_subtotal"] != {
        "variables": 85,
        "linear_constraints": 79,
        "general_constraints": 90,
        "quadratic_constraints": 0,
    }:
        raise AssertionError("uniform Gamma formulation baseline changed")

    mixed_formulation = mixed["gurobi_formulation"]
    remainder = mixed_formulation["comparison"]["non_gamma_remainder"]
    if mixed_formulation["comparison"]["known_subtotal_matches_actual"]:
        raise AssertionError("mixed formulation unexpectedly has no rainflow remainder")
    if any(value < 0 for value in remainder.values()):
        raise AssertionError("known shared/Gamma counts exceed actual mixed totals")
    if not any(value > 0 for value in remainder.values()):
        raise AssertionError("mixed report identified no non-Gamma contribution")

    ard1_formulation = ard1["gurobi_formulation"]
    if ard1_formulation["gamma_cells"] != 2:
        raise AssertionError("mixed ARD1 report has wrong Gamma cell count")
    if ard1_formulation["gamma_ard1_cells"] != 2:
        raise AssertionError("mixed ARD1 report has wrong ARD1 cell count")
    if ard1_formulation["known_subtotal"] != {
        "variables": 145,
        "linear_constraints": 81,
        "general_constraints": 120,
        "quadratic_constraints": 0,
    }:
        raise AssertionError("mixed ARD1 known formulation baseline changed")
    if ard1_formulation["actual_gurobi_model"] != {
        "variables": 165,
        "continuous_variables": 85,
        "integer_variables": 80,
        "binary_variables": 80,
        "linear_constraints": 125,
        "general_constraints": 210,
        "indicator_constraints": 210,
        "quadratic_constraints": 0,
        "nonzeros": 270,
    }:
        raise AssertionError("mixed ARD1 actual formulation baseline changed")
    ard1_validation = ard1["exact_validation"]
    if ard1_validation["gamma_ard1_cells"] != 2:
        raise AssertionError("mixed ARD1 validator selected wrong cells")
    if ard1_validation["repairs"] < 1:
        raise AssertionError("mixed ARD1 complexity case contains no Gamma repair")
    if ard1_validation["maximum_latch_error"] > 1e-8:
        raise AssertionError("mixed ARD1 complexity case has a latch mismatch")

    for case_name, case in report["cases"].items():
        if case["outcome"]["status"] != "optimal":
            raise AssertionError(f"{case_name} did not solve optimally")
        performance = case["solve_performance"]
        require_nonnegative(
            performance,
            (
                "model_construction_seconds",
                "gamma_calibration_seconds",
                "optimizer_call_seconds",
                "solution_extraction_seconds",
                "backend_wall_seconds",
            ),
            f"{case_name}.solve_performance",
        )
        cells = case["offline_calibration"]["cells"]
        if not cells:
            raise AssertionError(f"{case_name} has no Gamma calibration diagnostics")
        for cell in cells:
            for key in (
                "increment_opportunities",
                "increment_types",
                "calibration_lp_variables",
                "tail_constraints",
                "total_convolution_series_terms",
                "maximum_convolution_series_terms",
            ):
                if cell[key] <= 0:
                    raise AssertionError(f"{case_name} calibration {key} is not positive")
            if cell["calibration_seconds"] < 0.0:
                raise AssertionError(f"{case_name} calibration time is negative")

        validation = case["exact_validation"]
        if not validation["valid"]:
            raise AssertionError(f"{case_name} exact validation failed")
        require_nonnegative(
            validation["timing"],
            (
                "validation_wall_seconds",
                "exact_tail_seconds",
                "replay_and_reporting_seconds",
                "exact_tail_evaluations",
                "nonempty_convolutions",
            ),
            f"{case_name}.exact_validation.timing",
        )

    print("PASS Gamma complexity and timing diagnostics")
    print("uniform actual model :", uniform_formulation["actual_gurobi_model"])
    print("uniform count match  :", True)
    print("mixed non-Gamma part :", remainder)
    print("ARD1 known subtotal   :", ard1_formulation["known_subtotal"])
    print("ARD1 actual model     :", ard1_formulation["actual_gurobi_model"])
    print("ARD1 Gamma repairs    :", ard1_validation["repairs"])
    print("uniform calibration s:", uniform["offline_calibration"]["total_seconds"])
    print("uniform optimizer s  :", uniform["solve_performance"]["optimizer_call_seconds"])
    print("uniform validation s :", uniform["exact_validation"]["timing"]["validation_wall_seconds"])


if __name__ == "__main__":
    main()
