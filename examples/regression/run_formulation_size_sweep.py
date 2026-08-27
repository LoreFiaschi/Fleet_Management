"""Command-line runner for the deterministic F/M/L/T formulation-size sweep."""

from __future__ import annotations

import argparse

from fleet_management.formulation_size_sweep import sweep_formulation_dimensions


def comma_separated_integers(text: str) -> list[int]:
    try:
        return [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Scalar/broadcastable base YAML scenario")
    parser.add_argument("output", help="Output YAML report")
    parser.add_argument("--F", type=comma_separated_integers, default=[2, 4, 6, 8])
    parser.add_argument("--M", type=comma_separated_integers, default=[1, 2, 3, 4])
    parser.add_argument("--L", type=comma_separated_integers, default=[1, 2, 4, 8])
    parser.add_argument("--T", type=comma_separated_integers, default=[8, 12, 16, 20])
    arguments = parser.parse_args()

    report = sweep_formulation_dimensions(
        arguments.input,
        candidates={
            "F": arguments.F,
            "M": arguments.M,
            "L": arguments.L,
            "T": arguments.T,
        },
        output_path=arguments.output,
    )

    print("PASS deterministic F/M/L/T formulation-size sweep")
    print("baseline:", report["baseline_dimensions"])
    print("parameter  value   F   M   L   T   variables   linear rows")
    for parameter, sweep in report["sweeps"].items():
        for case in sweep["cases"]:
            dimensions = case["dimensions"]
            counts = case["counts"]
            print(
                f"{parameter:^9}  {case['changed_value']:>5}  "
                f"{dimensions['F']:>2}  {dimensions['M']:>2}  "
                f"{dimensions['L']:>2}  {dimensions['T']:>2}  "
                f"{counts['variables']:>9}   "
                f"{counts['linear_constraints']:>11}"
            )
    print("report:", arguments.output)


if __name__ == "__main__":
    main()
