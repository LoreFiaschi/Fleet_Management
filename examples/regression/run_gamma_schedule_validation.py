"""Run deterministic and/or stochastic Gamma schedule validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator import (
    VALIDATION_MODES,
    validate_gamma_schedule_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Solved model input YAML/JSON")
    parser.add_argument("result", type=Path, help="Saved solver result YAML/JSON")
    parser.add_argument("output", type=Path, help="Validation report YAML/JSON")
    parser.add_argument("--mode", choices=VALIDATION_MODES)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--maximum-schedule-failure-rate", type=float)
    parser.add_argument("--confidence-level", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--raise-on-failure", action="store_true")
    arguments = parser.parse_args()

    report = validate_gamma_schedule_files(
        arguments.input,
        arguments.result,
        arguments.output,
        mode=arguments.mode,
        repetitions=arguments.repetitions,
        random_seed=arguments.random_seed,
        maximum_schedule_failure_rate=(
            arguments.maximum_schedule_failure_rate
        ),
        confidence_level=arguments.confidence_level,
        batch_size=arguments.batch_size,
        tolerance=arguments.tolerance,
        raise_on_failure=arguments.raise_on_failure,
    )

    print(f"Wrote {arguments.output}")
    print("mode                 :", report["mode"])

    if report["mode"] in {"deterministic", "both"}:
        deterministic = (
            report
            if report["mode"] == "deterministic"
            else report["deterministic"]
        )
        print(
            "deterministic replay :",
            "PASS" if deterministic["valid"] else "FAIL",
        )

    if report["mode"] in {"stochastic", "both"}:
        stochastic = (
            report
            if report["mode"] == "stochastic"
            else report["stochastic"]
        )
        schedule = stochastic["schedule"]
        target = schedule["maximum_failure_rate"]

        print("repetitions          :", stochastic["repetitions"])
        print("failed replays       :", schedule["failed_replays"])
        print("failure rate         :", schedule["failure_rate"])
        print(
            "upper confidence     :",
            schedule["failure_rate_upper_confidence_bound"],
        )
        print("complete fleet       :", stochastic["complete_fleet_validation"])

        if target is None:
            print("stochastic assessment: REPORTED — no threshold configured")
        else:
            print("maximum failure rate :", target)
            print(
                "observed assessment  :",
                "PASS" if schedule["observed_target_met"] else "FAIL",
            )
            print(
                "confidence assessment:",
                "PASS"
                if schedule["confidence_qualified_target_met"]
                else "FAIL",
            )


if __name__ == "__main__":
    main()
