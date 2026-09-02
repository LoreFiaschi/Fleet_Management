"""Current repeated-increment Gamma tail calibration.

This module implements the calibration used by the public Gamma workflow.
For each distinct Gamma increment it:

1. finds the maximum safe repetition count under the original distribution;
2. selects a common rate no larger than any original rate; and
3. fits the smallest per-increment common-rate shape whose repeated tails are
   at least as large as the original tails through that safe count.

Initial and replacement states are calibrated as alternative singleton seeds.
The resulting common-rate shapes are additive and can therefore be propagated
by the linear Gamma state dynamics in the optimization model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.stats import gamma


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


__all__ = [
    "GammaRepeatedSeededProfileTailBoundResult",
    "RepeatedIncrementTailBoundResult",
    "RepeatedTailCheck",
    "calibrate_gamma_cell_tail_bound",
    "calibrate_repeated_increment_tail_bound",
    "maximum_safe_repetitions",
    "required_shape_for_tail",
]


@dataclass(frozen=True)
class RepeatedTailCheck:
    """One repeated-sum comparison used by the calibration."""

    repetitions: int
    exact_tail_probability: float
    bounded_tail_probability: float
    tail_margin: float
    reliability_slack: float


@dataclass(frozen=True)
class RepeatedIncrementTailBoundResult:
    """Common-rate bound for repeated copies of one Gamma increment."""

    original_shape: float
    original_rate: float
    common_rate: float
    bounded_shape: float
    threshold: float
    epsilon: float
    maximum_count: int
    maximum_safe_count: int
    checks: tuple[RepeatedTailCheck, ...]

    @property
    def all_checks_conservative(self) -> bool:
        return all(item.tail_margin >= -1e-9 for item in self.checks)

    @property
    def worst_tail_margin(self) -> float:
        return min((item.tail_margin for item in self.checks), default=0.0)

    @property
    def reliability_feasible(self) -> bool:
        if self.maximum_safe_count <= 0:
            return False
        limiting = self.checks[self.maximum_safe_count - 1]
        return limiting.reliability_slack >= -1e-9

    def as_dict(self) -> dict[str, object]:
        return {
            "original_shape": self.original_shape,
            "original_rate": self.original_rate,
            "common_rate": self.common_rate,
            "bounded_shape": self.bounded_shape,
            "threshold": self.threshold,
            "epsilon": self.epsilon,
            "maximum_count": self.maximum_count,
            "maximum_safe_count": self.maximum_safe_count,
            "all_checks_conservative": self.all_checks_conservative,
            "worst_tail_margin": self.worst_tail_margin,
            "reliability_feasible": self.reliability_feasible,
            "checks": [
                {
                    "repetitions": item.repetitions,
                    "exact_tail_probability": item.exact_tail_probability,
                    "bounded_tail_probability": item.bounded_tail_probability,
                    "tail_margin": item.tail_margin,
                    "reliability_slack": item.reliability_slack,
                }
                for item in self.checks
            ],
        }


@dataclass(frozen=True)
class GammaRepeatedSeededProfileTailBoundResult:
    """Repeated-increment calibration for one Gamma cell."""

    expected_damage: FloatArray
    original_rates: FloatArray
    original_shapes: FloatArray
    bounded_shapes: FloatArray
    type_indices: IntArray
    type_max_counts: IntArray
    max_total_count: int
    initial_expected_damage: float
    initial_original_rate: float | None
    initial_original_shape: float
    initial_bounded_shape: float
    replacement_expected_damage: float
    replacement_original_rate: float | None
    replacement_original_shape: float
    replacement_bounded_shape: float
    increment_offset: int
    common_rate: float
    increment_calibrations: tuple[RepeatedIncrementTailBoundResult, ...]
    initial_calibration: RepeatedIncrementTailBoundResult | None
    replacement_calibration: RepeatedIncrementTailBoundResult | None

    @property
    def checks(self) -> tuple[RepeatedTailCheck, ...]:
        rows: list[RepeatedTailCheck] = []
        for calibration in self.increment_calibrations:
            rows.extend(calibration.checks)
        if self.initial_calibration is not None:
            rows.extend(self.initial_calibration.checks)
        if self.replacement_calibration is not None:
            rows.extend(self.replacement_calibration.checks)
        return tuple(rows)

    @property
    def maximum_safe_counts(self) -> IntArray:
        return np.asarray(
            [item.maximum_safe_count for item in self.increment_calibrations],
            dtype=np.int64,
        )

    @property
    def all_constraints_conservative(self) -> bool:
        return all(item.tail_margin >= -1e-9 for item in self.checks)

    @property
    def worst_tail_margin(self) -> float:
        return min((item.tail_margin for item in self.checks), default=0.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "method": "repeated_increment",
            "expected_damage": self.expected_damage.tolist(),
            "original_rates": self.original_rates.tolist(),
            "original_shapes": self.original_shapes.tolist(),
            "bounded_shapes": self.bounded_shapes.tolist(),
            "type_indices": self.type_indices.tolist(),
            "type_max_counts": self.type_max_counts.tolist(),
            "maximum_safe_counts": self.maximum_safe_counts.tolist(),
            "max_total_count": self.max_total_count,
            "initial_expected_damage": self.initial_expected_damage,
            "initial_original_rate": self.initial_original_rate,
            "initial_original_shape": self.initial_original_shape,
            "initial_bounded_shape": self.initial_bounded_shape,
            "replacement_expected_damage": self.replacement_expected_damage,
            "replacement_original_rate": self.replacement_original_rate,
            "replacement_original_shape": self.replacement_original_shape,
            "replacement_bounded_shape": self.replacement_bounded_shape,
            "common_rate": self.common_rate,
            "all_constraints_conservative": self.all_constraints_conservative,
            "worst_tail_margin": self.worst_tail_margin,
            "increment_calibrations": [
                item.as_dict() for item in self.increment_calibrations
            ],
            "initial_calibration": (
                None
                if self.initial_calibration is None
                else self.initial_calibration.as_dict()
            ),
            "replacement_calibration": (
                None
                if self.replacement_calibration is None
                else self.replacement_calibration.as_dict()
            ),
        }


def maximum_safe_repetitions(
    shape: float,
    rate: float,
    threshold: float,
    epsilon: float,
    *,
    maximum_count: int,
    tolerance: float = 1e-12,
) -> int:
    """Return the largest ``m <= maximum_count`` with tail at most epsilon."""

    alpha = float(shape)
    beta = float(rate)
    tau = float(threshold)
    eps = float(epsilon)
    count_limit = int(maximum_count)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("shape must be finite and positive.")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("rate must be finite and positive.")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("threshold must be finite and positive.")
    if not np.isfinite(eps) or not 0.0 < eps < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one.")
    if count_limit <= 0 or count_limit != maximum_count:
        raise ValueError("maximum_count must be a positive integer.")

    safe = 0
    for repetitions in range(1, count_limit + 1):
        exact_tail = float(
            gamma.sf(tau, a=repetitions * alpha, scale=1.0 / beta)
        )
        if exact_tail <= eps + float(tolerance):
            safe = repetitions
        else:
            break
    return safe


def calibrate_repeated_increment_tail_bound(
    shape: float,
    rate: float,
    common_rate: float,
    threshold: float,
    epsilon: float,
    *,
    maximum_count: int,
    feasibility_tolerance: float = 1e-9,
) -> RepeatedIncrementTailBoundResult:
    """Fit the smallest common-rate shape for all repetitions through ``m*``.

    ``m*`` is the maximum safe count under the original Gamma increment.  The
    returned shape is the smallest value whose common-rate repeated sums have
    at least the original tail probability for every ``1 <= m <= m*``.
    """

    alpha = float(shape)
    beta = float(rate)
    beta_star = float(common_rate)
    tau = float(threshold)
    eps = float(epsilon)
    if not np.isfinite(beta_star) or beta_star <= 0.0:
        raise ValueError("common_rate must be finite and positive.")
    if beta_star > beta + 1e-12:
        raise ValueError("common_rate cannot exceed the original rate.")

    safe_count = maximum_safe_repetitions(
        alpha,
        beta,
        tau,
        eps,
        maximum_count=maximum_count,
    )
    # m*=0 is meaningful: the exact increment is unsafe even once. Calibrate
    # its singleton tail anyway. The resulting alpha* lies above the common-
    # rate reliability limit, so the Gurobi reliability row automatically
    # forbids selecting that mission type instead of aborting preprocessing.
    checked_count = max(1, safe_count)

    required_per_increment: list[float] = []
    exact_tails: list[float] = []
    for repetitions in range(1, checked_count + 1):
        exact_tail = float(
            gamma.sf(tau, a=repetitions * alpha, scale=1.0 / beta)
        )
        required_total_shape = required_shape_for_tail(
            exact_tail, beta_star, tau
        )
        required_per_increment.append(required_total_shape / repetitions)
        exact_tails.append(exact_tail)

    bounded_shape = max(required_per_increment)
    if bounded_shape > alpha + feasibility_tolerance:
        raise RuntimeError(
            "repeated-increment calibration increased alpha even though the "
            "common rate was not larger than the original rate."
        )

    checks: list[RepeatedTailCheck] = []
    for repetitions, exact_tail in enumerate(exact_tails, start=1):
        bounded_tail = float(
            gamma.sf(
                tau,
                a=repetitions * bounded_shape,
                scale=1.0 / beta_star,
            )
        )
        checks.append(
            RepeatedTailCheck(
                repetitions=repetitions,
                exact_tail_probability=exact_tail,
                bounded_tail_probability=bounded_tail,
                tail_margin=bounded_tail - exact_tail,
                reliability_slack=eps - bounded_tail,
            )
        )

    limiting_slack = checks[-1].reliability_slack
    if safe_count > 0 and limiting_slack < -feasibility_tolerance:
        raise ValueError(
            "no common-rate alpha satisfies both repeated-tail dominance and "
            f"reliability through m*={safe_count}; slack={limiting_slack:.3e}."
        )

    return RepeatedIncrementTailBoundResult(
        original_shape=alpha,
        original_rate=beta,
        common_rate=beta_star,
        bounded_shape=bounded_shape,
        threshold=tau,
        epsilon=eps,
        maximum_count=int(maximum_count),
        maximum_safe_count=safe_count,
        checks=tuple(checks),
    )


def calibrate_gamma_cell_tail_bound(
    expected_damage: ArrayLike,
    rates: ArrayLike,
    threshold: float,
    epsilon: float,
    *,
    max_total_count: int,
    initial_expected_damage: float = 0.0,
    initial_rate: float | None = None,
    replacement_expected_damage: float = 0.0,
    replacement_rate: float | None = None,
    common_rate: float | None = None,
    feasibility_tolerance: float = 1e-9,
) -> GammaRepeatedSeededProfileTailBoundResult:
    """Calibrate mission types independently with the contract."""

    means = np.asarray(expected_damage, dtype=float)
    if means.size == 0 or np.any(~np.isfinite(means)) or np.any(means <= 0.0):
        raise ValueError("expected_damage must contain finite positive values.")
    rate_values = np.asarray(rates, dtype=float)
    try:
        rate_profile = np.broadcast_to(rate_values, means.shape).astype(
            float, copy=True
        )
    except ValueError as error:
        raise ValueError(
            f"rates shape {rate_values.shape} cannot broadcast to "
            f"expected_damage shape {means.shape}."
        ) from error
    if np.any(~np.isfinite(rate_profile)) or np.any(rate_profile <= 0.0):
        raise ValueError("rates must contain finite positive values.")

    count_limit = int(max_total_count)
    if count_limit <= 0 or count_limit != max_total_count:
        raise ValueError("max_total_count must be a positive integer.")

    type_lookup: dict[tuple[float, float], int] = {}
    type_means: list[float] = []
    type_rates: list[float] = []
    type_occurrences: list[int] = []
    inverse = np.empty(means.size, dtype=np.int64)
    for position, (mean, beta) in enumerate(
        zip(means.ravel(), rate_profile.ravel(), strict=True)
    ):
        key = (float(mean), float(beta))
        type_index = type_lookup.get(key)
        if type_index is None:
            type_index = len(type_means)
            type_lookup[key] = type_index
            type_means.append(key[0])
            type_rates.append(key[1])
            type_occurrences.append(0)
        inverse[position] = type_index
        type_occurrences[type_index] += 1

    def seed_values(name: str, mean_value: float, beta_value: float | None):
        mean = float(mean_value)
        if not np.isfinite(mean) or mean < 0.0:
            raise ValueError(f"{name}_expected_damage must be non-negative.")
        if mean == 0.0:
            return mean, None, 0.0
        if beta_value is None or not np.isfinite(beta_value) or beta_value <= 0.0:
            raise ValueError(f"positive {name} damage needs a positive rate.")
        beta = float(beta_value)
        return mean, beta, mean * beta

    initial_mean, initial_beta, initial_shape = seed_values(
        "initial", initial_expected_damage, initial_rate
    )
    replacement_mean, replacement_beta, replacement_shape = seed_values(
        "replacement", replacement_expected_damage, replacement_rate
    )

    positive_rates = list(type_rates)
    if initial_beta is not None:
        positive_rates.append(initial_beta)
    if replacement_beta is not None:
        positive_rates.append(replacement_beta)
    beta_star = min(positive_rates) if common_rate is None else float(common_rate)
    if not np.isfinite(beta_star) or beta_star <= 0.0:
        raise ValueError("common_rate must be finite and positive.")
    if beta_star > min(positive_rates) + 1e-12:
        raise ValueError("common_rate cannot exceed the smallest exact rate.")

    increment_calibrations = tuple(
        calibrate_repeated_increment_tail_bound(
            mean * beta,
            beta,
            beta_star,
            threshold,
            epsilon,
            # A normalized increment type can occur only as often as that
            # exact (mean, rate) pair appears in the finite input profile.
            # Time-independent mission data therefore receive up to T
            # repetitions, while a genuinely time-specific type is checked
            # only at the positions where it can occur.
            maximum_count=min(count_limit, occurrences),
            feasibility_tolerance=feasibility_tolerance,
        )
        for mean, beta, occurrences in zip(
            type_means, type_rates, type_occurrences, strict=True
        )
    )
    bounded_type_shapes = np.asarray(
        [item.bounded_shape for item in increment_calibrations], dtype=float
    )
    type_indices = inverse.reshape(means.shape)

    initial_calibration = (
        None
        if initial_shape == 0.0
        else calibrate_repeated_increment_tail_bound(
            initial_shape,
            float(initial_beta),
            beta_star,
            threshold,
            epsilon,
            maximum_count=1,
            feasibility_tolerance=feasibility_tolerance,
        )
    )
    replacement_calibration = (
        None
        if replacement_shape == 0.0
        else calibrate_repeated_increment_tail_bound(
            replacement_shape,
            float(replacement_beta),
            beta_star,
            threshold,
            epsilon,
            maximum_count=1,
            feasibility_tolerance=feasibility_tolerance,
        )
    )

    return GammaRepeatedSeededProfileTailBoundResult(
        expected_damage=means.copy(),
        original_rates=rate_profile,
        original_shapes=means * rate_profile,
        bounded_shapes=bounded_type_shapes[type_indices],
        type_indices=type_indices,
        type_max_counts=np.asarray(type_occurrences, dtype=np.int64),
        max_total_count=count_limit,
        initial_expected_damage=initial_mean,
        initial_original_rate=initial_beta,
        initial_original_shape=initial_shape,
        initial_bounded_shape=(
            0.0 if initial_calibration is None else initial_calibration.bounded_shape
        ),
        replacement_expected_damage=replacement_mean,
        replacement_original_rate=replacement_beta,
        replacement_original_shape=replacement_shape,
        replacement_bounded_shape=(
            0.0
            if replacement_calibration is None
            else replacement_calibration.bounded_shape
        ),
        increment_offset=int(initial_shape > 0.0) + int(replacement_shape > 0.0),
        common_rate=beta_star,
        increment_calibrations=increment_calibrations,
        initial_calibration=initial_calibration,
        replacement_calibration=replacement_calibration,
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




