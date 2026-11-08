"""Exact tail of a sum of independent Gamma variables with different rates.

Spec, section "Validators of the Gamma Model", exact validator, steps 3 and 4.

Exact convolution tails are evaluated with the positive Gamma-series of
Moschopoulos (1985).  Truncation leaves a known non-negative mixture mass; the
exact validator uses ``estimate + remaining_mass`` and is therefore conservative
with respect to numerical series truncation.

Reference
---------
P. G. Moschopoulos, "The distribution of the sum of independent gamma random
variables", Annals of the Institute of Statistical Mathematics 37, 541-544
(1985), https://www.ism.ac.jp/editsec/aism/pdf/037_3_0541.pdf
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaincc
from scipy.stats import gamma


FloatArray = NDArray[np.float64]

_BLOCK = 256        # series terms whose Gamma tails are evaluated in one call


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
    # The weights sum to 1 only if the ratios and their complements agree: the
    # ratios are replaced by 1 - (1 - ratio), exact in floating point near 1, and
    # the complements are raised to the power k directly (no running product, whose
    # rounding error grows with k and shifts the total mass by about alpha * eps).
    complement = 1.0 - ratios
    ratios = 1.0 - complement
    log_initial_weight = float(np.dot(shape_array, np.log(ratios)))

    # The mixture weights satisfy  w_k = (1/k) sum_{i=1..k} (i gamma_i) w_{k-i},
    # with  i gamma_i = sum_q shape_q (1 - ratio_q)^i.  Each step is one dot
    # product; the Gamma tails of the mixture are summed in blocks. The weights,
    # the mass and the tail are kept in np.longdouble: 80 bits where the platform
    # has them (Linux), float64 on Windows.
    #
    # The first weight  C = exp(log_initial_weight)  can underflow (float64 on
    # Windows: shapes of a few thousand). The recursion is linear in w, so it runs
    # on  u_k = w_k / C  (u_0 = 1), whose size is tracked by a running log scale:
    # the true value is  delta_k = u_k exp(log_scale), and the recursion rescales
    # u whenever it exceeds 1e100. Each weight is formed in log scale,
    # w_k = exp(log C + log_scale + log u_k), so it underflows only when it is
    # itself negligible.
    shapes_ld = shape_array.astype(np.longdouble)
    complement = complement.astype(np.longdouble)
    scaled = np.empty(max_series_terms, dtype=np.longdouble)        # u_k, at scale log_scale
    weights = np.empty(max_series_terms, dtype=np.longdouble)       # w_k, true values
    index_gamma = np.empty(max_series_terms, dtype=np.longdouble)   # entry i: i * gamma_i
    log_scale = np.longdouble(0.0)
    log_c = np.longdouble(log_initial_weight)
    scaled[0] = 1.0
    initial_weight = np.exp(log_c)
    weights[0] = initial_weight
    index_gamma[0] = 0.0
    argument = beta_ref * threshold
    accumulated_mass = initial_weight
    tail = initial_weight * np.longdouble(gammaincc(total_shape, argument))
    summed = 1                                         # weights[:summed] are in the tail
    zero, one = np.longdouble(0.0), np.longdouble(1.0)

    for k in range(1, max_series_terms):
        index_gamma[k] = np.dot(shapes_ld, np.power(complement, k))
        scaled[k] = np.dot(index_gamma[1:k + 1], scaled[k - 1::-1]) / k
        if scaled[k] > 1e100:
            factor = scaled[k]
            scaled[:k + 1] /= factor
            log_scale += np.log(factor)
        weights[k] = (np.exp(log_c + log_scale + np.log(scaled[k]))
                      if scaled[k] > 0.0 else zero)
        accumulated_mass += weights[k]

        remaining = max(zero, one - accumulated_mass)
        if k + 1 - summed == _BLOCK or remaining <= tolerance:
            orders = np.arange(summed, k + 1)
            tail += np.dot(weights[summed:k + 1],
                           gammaincc(total_shape + orders, argument).astype(np.longdouble))
            summed = k + 1
        if remaining <= tolerance:
            return ConvolutionTail(
                estimate=float(tail),
                upper_bound=min(1.0, float(tail + remaining)),
                remaining_mass=float(remaining),
                series_terms=k + 1,
            )

    remaining = float(max(zero, one - accumulated_mass))
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
