"""Independent numerical reference for small sums of Gamma variables.

This helper intentionally does not import the production Moschopoulos code.
It evaluates the survival probability recursively from

    P(X + S > tau)
      = P(X > tau) + integral_0^tau f_X(x) P(S > tau-x) dx.

It is intended for regression checks with two or three independent terms, not
large production histories.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.stats import gamma


def quadrature_gamma_sum_tail(
    shapes,
    rates,
    threshold: float,
    *,
    epsabs: float = 2e-11,
    epsrel: float = 2e-10,
    limit: int = 250,
) -> tuple[float, float]:
    """Return ``(tail, quadrature_error_estimate)`` for a small Gamma sum."""

    a = np.asarray(shapes, dtype=float).reshape(-1)
    beta = np.asarray(rates, dtype=float).reshape(-1)
    tau = float(threshold)
    if a.size == 0 or a.shape != beta.shape:
        raise ValueError("shapes and rates must be nonempty vectors of equal length.")
    if np.any(~np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError("shapes must be finite and positive.")
    if np.any(~np.isfinite(beta)) or np.any(beta <= 0.0):
        raise ValueError("rates must be finite and positive.")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("threshold must be finite and positive.")

    def recurse(term_shapes: np.ndarray, term_rates: np.ndarray,
                remaining_threshold: float) -> tuple[float, float]:
        if remaining_threshold <= 0.0:
            return 1.0, 0.0
        if term_shapes.size == 1:
            value = float(
                gamma.sf(
                    remaining_threshold,
                    a=float(term_shapes[0]),
                    scale=1.0 / float(term_rates[0]),
                )
            )
            return value, 0.0

        # Integrating the term with the largest shape generally avoids the
        # strongest x=0 density singularity. The identity is symmetric, so this
        # ordering changes numerical conditioning but not the probability.
        index = int(np.argmax(term_shapes))
        shape_x = float(term_shapes[index])
        rate_x = float(term_rates[index])
        rest_shapes = np.delete(term_shapes, index)
        rest_rates = np.delete(term_rates, index)

        direct_tail = float(
            gamma.sf(remaining_threshold, a=shape_x, scale=1.0 / rate_x)
        )
        maximum_nested_error = 0.0

        def integrand(x: float) -> float:
            nonlocal maximum_nested_error
            rest_tail, nested_error = recurse(
                rest_shapes, rest_rates, remaining_threshold - x
            )
            maximum_nested_error = max(maximum_nested_error, nested_error)
            density = gamma.pdf(x, a=shape_x, scale=1.0 / rate_x)
            return float(density * rest_tail)

        integral, outer_error = quad(
            integrand,
            0.0,
            remaining_threshold,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
        value = min(1.0, max(0.0, direct_tail + float(integral)))
        # For nested quadrature this is a conservative diagnostic scale rather
        # than a formal interval: combine the reported outer error with the
        # largest inner error observed during evaluation.
        return value, float(outer_error + maximum_nested_error)

    return recurse(a, beta, tau)
