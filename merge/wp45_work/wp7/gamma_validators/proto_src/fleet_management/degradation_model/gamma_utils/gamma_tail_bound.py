"""Reduced copy (scratch): the series only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import gamma

FloatArray = NDArray[np.float64]

@dataclass(frozen=True)
class ConvolutionTail:
    """Numerical tail value and its certified truncation interval."""

    estimate: float
    upper_bound: float
    remaining_mass: float
    series_terms: int


def moschopoulos_tail_probability(
    shapes: ArrayLike,
    rates: ArrayLike,
    threshold: float,
    *,
    tolerance: float = 1e-12,
    max_series_terms: int = 100_000,
) -> ConvolutionTail:
    """Tail probability of a sum of independent, differently rated Gammas.

    The returned interval is ``[estimate, upper_bound]``.  All omitted series
    terms have non-negative mixture weights, so ``estimate + remaining_mass``
    is a rigorous truncation upper bound up to floating-point roundoff.
    """

    shape_array, rate_array = _validate_parameters(shapes, rates, threshold)
    if tolerance <= 0.0 or tolerance >= 1.0:
        raise ValueError("tolerance must lie strictly between zero and one.")
    if max_series_terms <= 0:
        raise ValueError("max_series_terms must be positive.")

    total_shape = float(np.sum(shape_array))
    if np.allclose(rate_array, rate_array[0], rtol=1e-13, atol=0.0):
        value = float(
            gamma.sf(threshold, a=total_shape, scale=1.0 / rate_array[0])
        )
        return ConvolutionTail(value, value, 0.0, 1)

    # Moschopoulos is written with the smallest scale.  In rate notation this
    # is the largest rate, beta_ref.  Every ratio beta_i / beta_ref is in (0, 1].
    beta_ref = float(np.max(rate_array))
    ratios = rate_array / beta_ref
    log_initial_weight = float(np.dot(shape_array, np.log(ratios)))
    initial_weight = np.exp(np.longdouble(log_initial_weight))
    if initial_weight == 0.0:
        raise RuntimeError(
            "The Moschopoulos initial mixture weight underflowed. Split this "
            "extreme convolution or use higher-precision offline arithmetic."
        )

    mixture_weights: list[np.longdouble] = [initial_weight]
    gamma_coefficients: list[np.longdouble] = [np.longdouble(0.0)]
    accumulated_mass = np.longdouble(initial_weight)
    tail = np.longdouble(initial_weight) * np.longdouble(
        gamma.sf(threshold, a=total_shape, scale=1.0 / beta_ref)
    )

    for k in range(1, max_series_terms):
        coefficient = np.longdouble(
            np.dot(shape_array, np.power(1.0 - ratios, k)) / k
        )
        gamma_coefficients.append(coefficient)
        weight = sum(
            np.longdouble(index)
            * gamma_coefficients[index]
            * mixture_weights[k - index]
            for index in range(1, k + 1)
        ) / np.longdouble(k)
        mixture_weights.append(weight)
        accumulated_mass += weight
        tail += weight * np.longdouble(
            gamma.sf(threshold, a=total_shape + k, scale=1.0 / beta_ref)
        )

        remaining = max(np.longdouble(0.0), np.longdouble(1.0) - accumulated_mass)
        if remaining <= tolerance:
            estimate = float(tail)
            remainder = float(remaining)
            return ConvolutionTail(
                estimate=estimate,
                upper_bound=min(1.0, estimate + remainder),
                remaining_mass=remainder,
                series_terms=k + 1,
            )

    remaining = float(max(np.longdouble(0.0), np.longdouble(1.0) - accumulated_mass))
    raise RuntimeError(
        "Moschopoulos series did not reach the requested tolerance within "
        f"{max_series_terms} terms; remaining mass={remaining:.3e}."
    )


def _validate_parameters(
    shapes: ArrayLike,
    rates: ArrayLike,
    threshold: float,
) -> tuple[FloatArray, FloatArray]:
    shape_array = np.asarray(shapes, dtype=float)
    rate_array = np.asarray(rates, dtype=float)
    if shape_array.ndim != 1 or shape_array.size == 0:
        raise ValueError("shapes must be a non-empty one-dimensional array.")
    if rate_array.shape != shape_array.shape:
        raise ValueError("rates must have the same one-dimensional shape as shapes.")
    if np.any(~np.isfinite(shape_array)) or np.any(shape_array <= 0.0):
        raise ValueError("all shapes must be finite and positive.")
    if np.any(~np.isfinite(rate_array)) or np.any(rate_array <= 0.0):
        raise ValueError("all rates must be finite and positive.")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive.")
    return shape_array, rate_array


