"""Regression for the mentor-style m*, beta*, alpha* Gamma calibration."""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma

from fleet_management.degradation_model.gamma_utils.gamma_repeated_calibration import (
    calibrate_gamma_cell_tail_bound,
    calibrate_repeated_increment_tail_bound,
)


def main() -> None:
    tolerance = 1e-10
    result = calibrate_repeated_increment_tail_bound(
        shape=2.0,
        rate=20.0,
        common_rate=10.0,
        threshold=0.5,
        epsilon=0.1,
        maximum_count=10,
    )

    if result.maximum_safe_count != 3:
        raise AssertionError(f"expected m*=3, got {result.maximum_safe_count}")
    if abs(result.bounded_shape - 0.7997909142967875) > tolerance:
        raise AssertionError("unexpected fitted alpha*")
    if not result.all_checks_conservative or not result.reliability_feasible:
        raise AssertionError("repeated-increment calibration is not valid")
    if any(item.tail_margin < -tolerance for item in result.checks):
        raise AssertionError("a repeated common-rate tail is not conservative")
    if result.checks[-1].bounded_tail_probability > 0.1 + tolerance:
        raise AssertionError("the m* common-rate tail exceeds epsilon")

    next_exact_tail = float(
        gamma.sf(
            0.5,
            a=(result.maximum_safe_count + 1) * result.original_shape,
            scale=1.0 / result.original_rate,
        )
    )
    if next_exact_tail <= 0.1:
        raise AssertionError("m* is not the maximum safe count")

    equal_rate = calibrate_repeated_increment_tail_bound(
        shape=1.0,
        rate=10.0,
        common_rate=10.0,
        threshold=0.5,
        epsilon=0.1,
        maximum_count=4,
    )
    if abs(equal_rate.bounded_shape - equal_rate.original_shape) > tolerance:
        raise AssertionError("equal rates must retain the original shape")

    unsafe = calibrate_repeated_increment_tail_bound(
        shape=4.0,
        rate=10.0,
        common_rate=10.0,
        threshold=0.5,
        epsilon=0.1,
        maximum_count=4,
    )
    if unsafe.maximum_safe_count != 0:
        raise AssertionError("an unsafe singleton must report m*=0")
    if unsafe.checks[0].reliability_slack >= 0.0:
        raise AssertionError("an m*=0 increment was not marked above epsilon")

    profile = calibrate_gamma_cell_tail_bound(
        expected_damage=np.asarray([[0.1] * 4, [0.2] * 4]),
        rates=np.asarray([[20.0] * 4, [15.0] * 4]),
        threshold=0.5,
        epsilon=0.1,
        max_total_count=4,
        initial_expected_damage=0.01,
        initial_rate=12.0,
        replacement_expected_damage=0.005,
        replacement_rate=11.0,
        common_rate=10.0,
    )
    if profile.common_rate != 10.0:
        raise AssertionError("the selected beta* was not retained")
    if profile.type_max_counts.tolist() != [4, 4]:
        raise AssertionError("mission-type occurrence counts are incorrect")
    if profile.maximum_safe_counts.tolist() != [3, 1]:
        raise AssertionError("mission-type m* values are incorrect")
    if profile.initial_calibration is None or profile.replacement_calibration is None:
        raise AssertionError("initial/replacement states were not calibrated")
    if not profile.all_constraints_conservative:
        raise AssertionError("profile calibration is not conservative")

    print("PASS repeated-increment Gamma calibration")
    print("maximum safe count m*:", result.maximum_safe_count)
    print("original alpha       :", result.original_shape)
    print("common beta*         :", result.common_rate)
    print("bounded alpha*       :", result.bounded_shape)
    print("worst tail margin    :", result.worst_tail_margin)
    print("profile safe counts  :", profile.maximum_safe_counts)
    print("profile type counts  :", profile.type_max_counts)


if __name__ == "__main__":
    main()
