"""Run a compact operating-horizon sweep from the command line."""

from __future__ import annotations

import argparse

from fleet_management import sweep_operating_horizons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Base YAML scenario")
    parser.add_argument("output", help="Compact YAML sweep report")
    horizons = parser.add_mutually_exclusive_group(required=True)
    horizons.add_argument(
        "--h2", type=int, nargs="+",
        help="Operating-horizon candidates, for example --h2 8 16 24 32",
    )
    horizons.add_argument(
        "--h2-range", type=int, nargs=2, metavar=("START", "STOP"),
        help="Inclusive operating-horizon range, for example --h2-range 2 32",
    )
    parser.add_argument(
        "--h2-step", type=int, default=1,
        help="Step used with --h2-range (default: 1)",
    )
    parser.add_argument(
        "--stop-on-gradient", action="store_true",
        help=(
            "Stop after a gap-qualified cost increase or a sufficiently flat "
            "sequence; the final value from --h2-range remains a hard limit"
        ),
    )
    parser.add_argument(
        "--gradient-tolerance", type=float, default=1e-3,
        help="Maximum absolute relative cost gradient per H2 considered flat",
    )
    parser.add_argument(
        "--flat-gradients", type=int, default=2,
        help="Consecutive flat gradients required before stopping (default: 2)",
    )
    parser.add_argument(
        "--minimum-cases", type=int, default=3,
        help="Minimum solved cases before gradient stopping is allowed",
    )
    parser.add_argument(
        "--maximum-stopping-gap", type=float, default=0.05,
        help="Largest relative MIP gap allowed to trigger gradient stopping",
    )
    arguments = parser.parse_args()
    if arguments.h2_range is not None:
        start, stop = arguments.h2_range
        if arguments.h2_step <= 0:
            parser.error("--h2-step must be positive")
        if stop < start:
            parser.error("--h2-range STOP must be at least START")
        operating_horizons = list(range(start, stop + 1, arguments.h2_step))
    else:
        operating_horizons = arguments.h2

    report = sweep_operating_horizons(
        arguments.input,
        operating_horizons,
        output_path=arguments.output,
        stop_on_gradient=arguments.stop_on_gradient,
        gradient_tolerance=arguments.gradient_tolerance,
        flat_gradients_required=arguments.flat_gradients,
        minimum_cases=arguments.minimum_cases,
        maximum_mip_gap_for_stopping=arguments.maximum_stopping_gap,
    )
    print("best proven H2   :", report["best_proven_H2"])
    print("best feasible H2 :", report["best_feasible_H2"])
    print("stopping reason  :", report["stopping_rule"]["reason"])
    print(
        "\nH2   T   J_op/H2   best bound   gap       gradient      "
        "continuous   integer   linear rows   optimizer s"
    )
    for row in report["cases"]:
        formulation = row["formulation"]
        objective = row["J_op_average"]
        bound = row["objective_bound"]
        gap = row["mip_gap"]
        gradient = row["relative_cost_gradient_per_H2"]
        print(
            f"{row['H2']:>2}  {row['T']:>2}  "
            f"{_format_float(objective, 8, '.5f')}   "
            f"{_format_float(bound, 10, '.5f')}   "
            f"{_format_float(gap, 7, '.2%')}   "
            f"{_format_float(gradient, 10, '.3%')}   "
            f"{formulation['continuous_variables']:>10}   "
            f"{formulation['integer_variables']:>7}   "
            f"{formulation['linear_constraints']:>11}   "
            f"{row['optimizer_seconds']:>11.3f}"
        )
    print("report            :", arguments.output)


def _format_float(value, width: int, specification: str) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{format(float(value), specification):>{width}}"


if __name__ == "__main__":
    main()
