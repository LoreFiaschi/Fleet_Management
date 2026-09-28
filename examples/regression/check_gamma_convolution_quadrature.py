"""Compare production Gamma convolution tails with independent quadrature."""

from __future__ import annotations

import numpy as np

from fleet_management.degradation_model.gamma_utils.gamma_tail_bound import (
    moschopoulos_tail_probability,
)
from gamma_quadrature_reference import quadrature_gamma_sum_tail


CASES = (
    # equal rates exercise the production closed-form shortcut
    {"shapes": [0.5, 2.3], "rates": [10.0, 10.0], "threshold": 0.30},
    # non-integer shapes and unequal rates
    {"shapes": [0.7, 3.2], "rates": [5.0, 20.0], "threshold": 0.35},
    # rate ratio 1:20
    {"shapes": [1.2, 0.4], "rates": [3.0, 60.0], "threshold": 0.25},
    # threshold below the total mean
    {"shapes": [2.5, 1.1], "rates": [8.0, 17.0], "threshold": 0.20},
    # threshold well above the total mean
    {"shapes": [0.8, 4.0], "rates": [12.0, 25.0], "threshold": 0.75},
    # three different rates, evaluated by nested independent quadrature
    {"shapes": [0.8, 1.7, 2.4], "rates": [5.0, 12.0, 30.0], "threshold": 0.45},
)


def main() -> None:
    worst_error = 0.0
    worst_interval_violation = 0.0
    maximum_remainder = 0.0
    maximum_series_terms = 0

    for case_number, case in enumerate(CASES, start=1):
        reference, quadrature_error = quadrature_gamma_sum_tail(**case)
        production = moschopoulos_tail_probability(
            **case,
            tolerance=1e-12,
            max_series_terms=100_000,
        )
        error = abs(production.estimate - reference)
        allowance = max(2e-9, 5.0 * quadrature_error)
        interval_violation = max(
            production.estimate - reference - allowance,
            reference - production.upper_bound - allowance,
            0.0,
        )
        if interval_violation > 0.0:
            raise AssertionError(
                f"case {case_number}: quadrature tail {reference:.16g} lies "
                f"outside production interval "
                f"[{production.estimate:.16g}, {production.upper_bound:.16g}] "
                f"with allowance {allowance:.3e}."
            )
        worst_error = max(worst_error, error)
        worst_interval_violation = max(worst_interval_violation, interval_violation)
        maximum_remainder = max(maximum_remainder, production.remaining_mass)
        maximum_series_terms = max(maximum_series_terms, production.series_terms)

    print("PASS Gamma convolution against independent quadrature")
    print("cases checked       :", len(CASES))
    print("worst absolute error:", worst_error)
    print("interval violation  :", worst_interval_violation)
    print("maximum remainder   :", maximum_remainder)
    print("maximum series terms:", maximum_series_terms)


if __name__ == "__main__":
    main()
