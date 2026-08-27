"""Run a compact operating-horizon sweep from the command line."""

from __future__ import annotations

import argparse

from fleet_management import sweep_operating_horizons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Base YAML scenario")
    parser.add_argument("output", help="Compact YAML sweep report")
    parser.add_argument(
        "--h2", type=int, nargs="+", required=True,
        help="Operating-horizon candidates, for example --h2 8 16 24 32",
    )
    parser.add_argument(
        "--transitory-budget", type=float,
        help="Override B_trans from the input YAML",
    )
    arguments = parser.parse_args()
    report = sweep_operating_horizons(
        arguments.input,
        arguments.h2,
        transitory_budget=arguments.transitory_budget,
        output_path=arguments.output,
    )
    print("best H2          :", report["best_H2"])
    print("best J_op / H2   :", report["best_J_op_average"])
    print("\nH2   T   variables   linear rows   m* min/max   optimizer s")
    for row in report["cases"]:
        formulation = row["formulation"]
        calibration = row["calibration"]
        print(
            f"{row['H2']:>2}  {row['T']:>2}  "
            f"{formulation['variables']:>9}   "
            f"{formulation['linear_constraints']:>11}   "
            f"{str(calibration['minimum_safe_count']):>3}/"
            f"{str(calibration['maximum_safe_count']):<3}   "
            f"{row['optimizer_seconds']:>11.3f}"
        )
    print("report            :", arguments.output)


if __name__ == "__main__":
    main()