"""Offline signed check for the conservative ARD-inf Gamma transition."""

from __future__ import annotations

from scipy.stats import gamma

from fleet_management.degradation_model.gamma_utils.gamma_tail_bound import (
    calculate_seeded_profile_tail_bound_parameters,
    moschopoulos_tail_probability,
)


def main() -> None:
    threshold = 0.6
    common_rate = 10.0
    remaining = 0.5
    calibration = calculate_seeded_profile_tail_bound_parameters(
        expected_damage=[[0.05, 0.05, 0.05]],
        rates=[[18.0, 10.0, 20.0]],
        threshold=threshold,
        max_total_count=3,
        initial_expected_damage=0.02,
        initial_rate=14.0,
        common_rate=common_rate,
    )

    # Initial state and the first two increments survive one ARD-inf repair.
    # Scaling Gamma(A,beta) by c changes its rate to beta/c. The third mission
    # occurs after repair and keeps its original rate.
    exact = moschopoulos_tail_probability(
        shapes=[0.02 * 14.0, 0.05 * 18.0, 0.05 * 10.0, 0.05 * 20.0],
        rates=[14.0 / remaining, 18.0 / remaining, 10.0 / remaining, 20.0],
        threshold=threshold,
    )

    # The solver gives no tail-bound credit for repair, retaining every bounded
    # shape. This represents the un-repaired history, which is pathwise larger.
    bounded_shape = (
        calibration.initial_bounded_shape + float(calibration.bounded_shapes.sum())
    )
    bounded_tail = float(
        gamma.sf(threshold, a=bounded_shape, scale=1.0 / common_rate)
    )
    margin = bounded_tail - exact.upper_bound
    if margin < -1e-9:
        raise AssertionError(f"ARD-inf no-credit bound failed: margin={margin:.3e}")

    print("PASS conservative Gamma ARD-inf repair bound")
    print("exact repaired tail:", exact.upper_bound)
    print("solver bound tail  :", bounded_tail)
    print("signed margin      :", margin)
    print("repair shape rule  : A_after = A_before")


if __name__ == "__main__":
    main()
