"""Offline construction of conservative common-rate Gamma tail bounds.

Suppose independent damage increments satisfy

    X_q ~ Gamma(A_q, beta_q),

using the shape-rate convention.  Their exact sum is generally not Gamma when
the rates differ.  This module replaces them by

    X'_q ~ Gamma(A'_q, beta_bar)

with one common rate ``beta_bar <= min(beta_q)``.  Common-rate approximations
are closed under addition, so a count vector ``c`` gives

    sum_q c_q X'_q ~ Gamma(c @ A', beta_bar).

For a fixed failure threshold tau, the exact convolution tail for each count
vector is converted into a required common-rate shape.  The conditions

    c @ A' >= required_shape(c)

are linear because the Gamma tail is increasing in shape at fixed rate and
threshold.  A linear program then minimizes the individual shapes subject to
all requested tail conditions and ``0 <= A'_q <= A_q``.

The guarantee is deliberately precise: it covers the supplied finite set of
count vectors at the supplied threshold.  ``max_counts`` can enumerate every
combination reachable in a finite scheduling horizon.  A single-threshold
bound does not imply first-order stochastic dominance at every threshold, and
finite enumeration does not prove a bound for unbounded repetition.

Exact convolution tails are evaluated with the positive Gamma-series of
Moschopoulos (1985).  Truncation leaves a known non-negative mixture mass; the
optimizer uses ``partial_tail + remaining_mass`` and is therefore conservative
with respect to numerical series truncation.

Reference
---------
P. G. Moschopoulos, "The distribution of the sum of independent gamma random
variables", Annals of the Institute of Statistical Mathematics 37, 541-544
(1985), https://www.ism.ac.jp/editsec/aism/pdf/037_3_0541.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq, linprog
from scipy.stats import gamma


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ConvolutionTail:
    """Numerical tail value and its certified truncation interval."""

    estimate: float
    upper_bound: float
    remaining_mass: float
    series_terms: int


@dataclass(frozen=True)
class TailConstraint:
    """One exact-sum versus common-rate tail comparison."""

    counts: tuple[int, ...]
    exact_tail_estimate: float
    exact_tail_upper_bound: float
    convolution_remaining_mass: float
    required_common_shape: float
    bounded_common_shape: float
    bounded_tail_probability: float
    tail_margin: float


@dataclass(frozen=True)
class GammaTailBoundResult:
    """Parameters and diagnostics produced by the tail-bound calculation."""

    original_shapes: FloatArray
    original_rates: FloatArray
    common_rate: float
    bounded_shapes: FloatArray
    original_means: FloatArray
    bounded_means: FloatArray
    threshold: float
    constraints: tuple[TailConstraint, ...]
    objective_value: float

    @property
    def shape_reduction(self) -> FloatArray:
        return self.original_shapes - self.bounded_shapes

    @property
    def all_constraints_conservative(self) -> bool:
        return all(item.tail_margin >= -1e-9 for item in self.constraints)

    @property
    def worst_tail_margin(self) -> float:
        return min(item.tail_margin for item in self.constraints)

    def as_dict(self) -> dict[str, object]:
        """Return a YAML/JSON-friendly representation."""

        return {
            "original_shapes": self.original_shapes.tolist(),
            "original_rates": self.original_rates.tolist(),
            "common_rate": self.common_rate,
            "bounded_shapes": self.bounded_shapes.tolist(),
            "shape_reduction": self.shape_reduction.tolist(),
            "original_means": self.original_means.tolist(),
            "bounded_means": self.bounded_means.tolist(),
            "threshold": self.threshold,
            "objective_value": self.objective_value,
            "all_constraints_conservative": self.all_constraints_conservative,
            "worst_tail_margin": self.worst_tail_margin,
            "constraints": [
                {
                    "counts": list(item.counts),
                    "exact_tail_estimate": item.exact_tail_estimate,
                    "exact_tail_upper_bound": item.exact_tail_upper_bound,
                    "convolution_remaining_mass": item.convolution_remaining_mass,
                    "required_common_shape": item.required_common_shape,
                    "bounded_common_shape": item.bounded_common_shape,
                    "bounded_tail_probability": item.bounded_tail_probability,
                    "tail_margin": item.tail_margin,
                }
                for item in self.constraints
            ],
        }


def calculate_tail_bound_parameters(
    shapes: ArrayLike,
    rates: ArrayLike,
    threshold: float,
    *,
    common_rate: float | None = None,
    max_counts: ArrayLike | None = None,
    count_vectors: ArrayLike | None = None,
    max_total_count: int | None = None,
    objective_weights: ArrayLike | None = None,
    convolution_tolerance: float = 1e-12,
    feasibility_tolerance: float = 1e-9,
    max_series_terms: int = 100_000,
    max_combinations: int = 100_000,
) -> GammaTailBoundResult:
    """Calculate the smallest common-rate shapes for specified finite sums.

    Parameters
    ----------
    shapes, rates:
        One-dimensional original Gamma shapes and rates.  Entry ``q`` defines
        one distinct independent increment type.
    threshold:
        Failure threshold at which conservativeness is required.
    common_rate:
        Common approximation rate.  It must not exceed the smallest original
        rate.  The default is ``min(rates)``: the largest rate for which keeping
        all original shapes gives an immediate stochastic upper bound.
    max_counts:
        Maximum occurrence count of every increment type.  Every nonzero count
        vector in the Cartesian product is checked.  The default is one of each
        type, i.e. every nonempty subset.
    count_vectors:
        Explicit non-negative integer count vectors.  Use this instead of
        ``max_counts`` when only particular sums are reachable.  Singleton
        vectors are always added so each individual approximation is checked.
    max_total_count:
        Optional cap on ``sum(counts)`` during Cartesian enumeration.
    objective_weights:
        Positive weights for minimizing ``weights @ A'``.  Defaults to equal
        weights.  They resolve how additional shape required by a sum constraint
        is distributed among increment types.
    convolution_tolerance:
        Maximum unaccounted Gamma-mixture mass in each exact convolution tail.
    feasibility_tolerance:
        Accepted numerical slack when checking the final LP solution.
    max_series_terms, max_combinations:
        Safety limits for offline convolution and combination enumeration.

    Notes
    -----
    If an increment already has ``beta_q == common_rate``, its singleton
    constraint normally forces ``A'_q = A_q``.  This is the mentor's "no room
    for further improvement" case.  Other shapes can be lowered only as far as
    the complete set of individual and accumulated tail constraints permits.
    """

    original_shapes, original_rates = _validate_parameters(shapes, rates, threshold)
    n = original_shapes.size

    beta_bar = float(np.min(original_rates) if common_rate is None else common_rate)
    if not np.isfinite(beta_bar) or beta_bar <= 0.0:
        raise ValueError("common_rate must be finite and positive.")
    if beta_bar > float(np.min(original_rates)):
        raise ValueError(
            "common_rate must be no greater than the smallest original rate; "
            "otherwise the original shapes are not a guaranteed feasible bound."
        )

    combinations = _resolve_count_vectors(
        n=n,
        max_counts=max_counts,
        count_vectors=count_vectors,
        max_total_count=max_total_count,
        max_combinations=max_combinations,
    )

    if objective_weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(objective_weights, dtype=float)
        if weights.shape != (n,) or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError(f"objective_weights must contain {n} finite positive values.")

    required_shapes = np.empty(len(combinations), dtype=float)
    exact_tails: list[ConvolutionTail] = []
    for row, counts in enumerate(combinations):
        active = counts > 0
        convolution = moschopoulos_tail_probability(
            shapes=original_shapes[active] * counts[active],
            rates=original_rates[active],
            threshold=threshold,
            tolerance=convolution_tolerance,
            max_series_terms=max_series_terms,
        )
        exact_tails.append(convolution)
        required_shapes[row] = required_shape_for_tail(
            tail_probability=convolution.upper_bound,
            rate=beta_bar,
            threshold=threshold,
        )

    optimization = linprog(
        c=weights,
        A_ub=-combinations.astype(float),
        b_ub=-required_shapes,
        bounds=[(0.0, float(value)) for value in original_shapes],
        method="highs",
    )
    if not optimization.success or optimization.x is None:
        raise RuntimeError(
            "No tail-bounding shapes were found. Try a lower common_rate or "
            "a tighter convolution tolerance. LP message: "
            f"{optimization.message}"
        )

    bounded_shapes = np.asarray(optimization.x, dtype=float)
    shape_slack = combinations @ bounded_shapes - required_shapes
    if float(np.min(shape_slack)) < -feasibility_tolerance:
        raise RuntimeError(
            "The tail-bound LP returned a numerically infeasible solution; "
            f"minimum shape slack is {float(np.min(shape_slack)):.3e}."
        )

    diagnostics: list[TailConstraint] = []
    for counts, required, convolution in zip(
        combinations, required_shapes, exact_tails, strict=True
    ):
        summed_shape = float(counts @ bounded_shapes)
        bounded_tail = float(
            gamma.sf(threshold, a=summed_shape, scale=1.0 / beta_bar)
        )
        margin = bounded_tail - convolution.upper_bound
        if margin < -feasibility_tolerance:
            raise RuntimeError(
                "Post-optimization tail validation failed for counts "
                f"{counts.tolist()}: margin={margin:.3e}."
            )
        diagnostics.append(
            TailConstraint(
                counts=tuple(int(value) for value in counts),
                exact_tail_estimate=convolution.estimate,
                exact_tail_upper_bound=convolution.upper_bound,
                convolution_remaining_mass=convolution.remaining_mass,
                required_common_shape=float(required),
                bounded_common_shape=summed_shape,
                bounded_tail_probability=bounded_tail,
                tail_margin=float(margin),
            )
        )

    return GammaTailBoundResult(
        original_shapes=original_shapes.copy(),
        original_rates=original_rates.copy(),
        common_rate=beta_bar,
        bounded_shapes=bounded_shapes,
        original_means=original_shapes / original_rates,
        bounded_means=bounded_shapes / beta_bar,
        threshold=float(threshold),
        constraints=tuple(diagnostics),
        objective_value=float(weights @ bounded_shapes),
    )


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


def required_shape_for_tail(
    tail_probability: float,
    rate: float,
    threshold: float,
) -> float:
    """Invert ``P(Gamma(A, rate) > threshold)`` with respect to shape A."""

    probability = float(tail_probability)
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError("tail_probability must lie in [0, 1].")
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("rate must be finite and positive.")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive.")
    if probability == 0.0:
        return 0.0

    # Avoid an infinite requested shape when roundoff turns a near-one
    # probability into exactly one.
    target = min(probability, float(np.nextafter(1.0, 0.0)))

    def residual(shape: float) -> float:
        return float(gamma.sf(threshold, a=shape, scale=1.0 / rate) - target)

    lower = float(np.nextafter(0.0, 1.0))
    upper = max(1.0, rate * threshold)
    while residual(upper) < 0.0:
        upper *= 2.0
        if upper > 1e12:
            raise RuntimeError("Could not bracket the required Gamma shape.")
    return float(brentq(residual, lower, upper, xtol=1e-12, rtol=1e-12))


def enumerate_count_vectors(
    max_counts: ArrayLike,
    *,
    max_total_count: int | None = None,
    max_combinations: int = 100_000,
) -> IntArray:
    """Enumerate every nonzero count vector within component-wise maxima."""

    maxima = _integer_vector(max_counts, "max_counts")
    if np.any(maxima < 0) or not np.any(maxima > 0):
        raise ValueError("max_counts must be non-negative with at least one positive entry.")
    if max_total_count is not None and max_total_count <= 0:
        raise ValueError("max_total_count must be positive when provided.")
    if max_combinations <= 0:
        raise ValueError("max_combinations must be positive.")

    unconstrained_size = prod(int(value) + 1 for value in maxima) - 1
    if max_total_count is None and unconstrained_size > max_combinations:
        raise ValueError(
            f"max_counts generates {unconstrained_size} combinations, exceeding "
            f"max_combinations={max_combinations}."
        )

    rows: list[tuple[int, ...]] = []
    ranges = [range(int(value) + 1) for value in maxima]
    for candidate in product(*ranges):
        total = sum(candidate)
        if total == 0 or (max_total_count is not None and total > max_total_count):
            continue
        rows.append(candidate)
        if len(rows) > max_combinations:
            raise ValueError(
                "Count-vector enumeration exceeded "
                f"max_combinations={max_combinations}."
            )
    rows.sort(key=lambda row: (sum(row), row))
    return np.asarray(rows, dtype=np.int64)


def shapes_from_expected_damage(
    expected_damage: ArrayLike,
    rates: ArrayLike,
) -> FloatArray:
    """Convert data-derived means to shapes using ``A_q = mu_q * beta_q``."""

    means = np.asarray(expected_damage, dtype=float)
    rate_array = np.asarray(rates, dtype=float)
    if means.ndim != 1 or means.size == 0:
        raise ValueError("expected_damage must be a non-empty one-dimensional array.")
    if rate_array.shape != means.shape:
        raise ValueError("rates must have the same shape as expected_damage.")
    if np.any(~np.isfinite(means)) or np.any(means <= 0.0):
        raise ValueError("expected_damage values must be finite and positive.")
    if np.any(~np.isfinite(rate_array)) or np.any(rate_array <= 0.0):
        raise ValueError("rates must be finite and positive.")
    return means * rate_array


def _resolve_count_vectors(
    *,
    n: int,
    max_counts: ArrayLike | None,
    count_vectors: ArrayLike | None,
    max_total_count: int | None,
    max_combinations: int,
) -> IntArray:
    if max_counts is not None and count_vectors is not None:
        raise ValueError("Give max_counts or count_vectors, not both.")

    if count_vectors is None:
        maxima = (
            np.ones(n, dtype=np.int64)
            if max_counts is None
            else _integer_vector(max_counts, "max_counts")
        )
        if maxima.shape != (n,):
            raise ValueError(f"max_counts must contain exactly {n} entries.")
        return enumerate_count_vectors(
            maxima,
            max_total_count=max_total_count,
            max_combinations=max_combinations,
        )

    values = np.asarray(count_vectors, dtype=float)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape[1] != n:
        raise ValueError(f"count_vectors must have shape (K, {n}).")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("count_vectors must contain finite non-negative values.")
    rounded = np.rint(values)
    if np.any(np.abs(values - rounded) > 1e-12):
        raise ValueError("count_vectors must contain integers.")

    rows = {tuple(int(value) for value in row) for row in rounded}
    rows.discard((0,) * n)
    # Individual constraints are necessary before reasoning about sums.
    rows.update(tuple(1 if q == index else 0 for q in range(n)) for index in range(n))
    if max_total_count is not None:
        rows = {row for row in rows if sum(row) <= max_total_count or sum(row) == 1}
    if len(rows) > max_combinations:
        raise ValueError(
            f"count_vectors contains {len(rows)} combinations, exceeding "
            f"max_combinations={max_combinations}."
        )
    return np.asarray(sorted(rows, key=lambda row: (sum(row), row)), dtype=np.int64)


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


def _integer_vector(value: ArrayLike, name: str) -> IntArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array.")
    rounded = np.rint(array)
    if np.any(np.abs(array - rounded) > 1e-12):
        raise ValueError(f"{name} must contain integers.")
    return rounded.astype(np.int64)
