"""Fixed-seed randomized property checks for Gamma tail-bound calibration."""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import gamma

from fleet_management.degradation_model.gamma_utils.gamma_tail_bound import (
    calculate_tail_bound_parameters,
)
from gamma_quadrature_reference import quadrature_gamma_sum_tail


RANDOM_SEED = 20260821
NUMBER_OF_CASES = 30


def independent_tail(shapes: np.ndarray, rates: np.ndarray,
                     counts: np.ndarray, threshold: float) -> tuple[float, float]:
    active = counts > 0
    grouped_shapes = shapes[active] * counts[active]
    grouped_rates = rates[active]
    if grouped_shapes.size == 1:
        value = float(
            gamma.sf(
                threshold,
                a=float(grouped_shapes[0]),
                scale=1.0 / float(grouped_rates[0]),
            )
        )
        return value, 0.0
    return quadrature_gamma_sum_tail(
        grouped_shapes,
        grouped_rates,
        threshold,
    )


def main(number_of_cases: int = NUMBER_OF_CASES,
         random_seed: int = RANDOM_SEED) -> None:
    if number_of_cases <= 0:
        raise ValueError("number_of_cases must be positive")
    rng = np.random.default_rng(random_seed)
    constraints_checked = 0
    quadratures_evaluated = 0
    worst_convolution_error = 0.0
    worst_independent_margin = float("inf")
    maximum_quadrature_error = 0.0
    largest_rate_ratio = 0.0

    for case_number in range(number_of_cases):
        shapes = rng.uniform(0.15, 8.0, size=2)
        # Log-uniform rates deliberately include substantially different scales.
        rates = np.exp(rng.uniform(np.log(3.0), np.log(60.0), size=2))
        if case_number == 0:
            rates = np.array([3.0, 60.0])  # guarantee one explicit 1:20 case
        total_mean = float(np.sum(shapes / rates))
        threshold = total_mean * float(rng.uniform(0.65, 2.25))
        common_rate = float(np.min(rates) * rng.uniform(0.45, 1.0))

        calibration = calculate_tail_bound_parameters(
            shapes=shapes,
            rates=rates,
            threshold=threshold,
            common_rate=common_rate,
            max_counts=[2, 2],
            max_total_count=3,
            convolution_tolerance=1e-12,
            feasibility_tolerance=1e-9,
        )
        if not calibration.all_constraints_conservative:
            raise AssertionError(f"case {case_number}: production calibration failed")

        for constraint in calibration.constraints:
            counts = np.asarray(constraint.counts, dtype=np.int64)
            reference, quadrature_error = independent_tail(
                shapes, rates, counts, threshold
            )
            if np.count_nonzero(counts) > 1:
                quadratures_evaluated += 1
            allowance = max(3e-9, 5.0 * quadrature_error)
            convolution_error = abs(constraint.exact_tail_estimate - reference)
            independent_margin = constraint.bounded_tail_probability - reference

            if convolution_error > allowance:
                raise AssertionError(
                    f"case {case_number}, counts {counts.tolist()}: production "
                    f"exact tail differs from quadrature by {convolution_error:.3e} "
                    f"(allowance {allowance:.3e})."
                )
            if independent_margin < -allowance:
                raise AssertionError(
                    f"case {case_number}, counts {counts.tolist()}: bounded tail "
                    f"is below independent reference by {-independent_margin:.3e}."
                )
            constraints_checked += 1
            worst_convolution_error = max(
                worst_convolution_error, convolution_error
            )
            worst_independent_margin = min(
                worst_independent_margin, independent_margin
            )
            maximum_quadrature_error = max(
                maximum_quadrature_error, quadrature_error
            )

        largest_rate_ratio = max(
            largest_rate_ratio, float(np.max(rates) / np.min(rates))
        )

    print("PASS randomized Gamma tail-bound properties")
    print("random seed             :", random_seed)
    print("cases checked           :", number_of_cases)
    print("constraints checked     :", constraints_checked)
    print("quadratures evaluated   :", quadratures_evaluated)
    print("largest rate ratio      :", largest_rate_ratio)
    print("worst convolution error :", worst_convolution_error)
    print("worst independent margin:", worst_independent_margin)
    print("maximum quadrature error:", maximum_quadrature_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=NUMBER_OF_CASES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    arguments = parser.parse_args()
    main(arguments.cases, arguments.seed)
