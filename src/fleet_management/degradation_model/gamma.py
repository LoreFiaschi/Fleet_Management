"""Constant-rate Gamma degradation model.

The model uses the shape-rate convention

    D ~ Gamma(A, beta)

with

    E[D]   = A / beta
    Var[D] = A / beta**2

The accumulated state stored by this model is the Gamma shape A.

Supported transitions
---------------------
Idle:
    A_k = A_{k-1}

Mission:
    A_k = A_{k-1} + A_increment
    A_increment = beta * expected_increment

Replacement:
    A_k = A_new
    A_new = beta * expected_damage_new

Imperfect repair uses a documented common-beta approximation.  For repair
effectiveness ``rho``, the shape is updated as

    A_plus = (1-rho) * A_minus.

This preserves the mean of the exact scaled repair while retaining the shared
rate required by the optimization model.  It does not preserve the exact
post-repair variance; validators should compare it with the exact scaled
Gamma distribution offline.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import gamma
from scipy.optimize import brentq


FloatArray = NDArray[np.float64]


class GammaAction(str, Enum):
    """Possible component-state transitions."""

    IDLE = "idle"
    MISSION = "mission"
    REPLACEMENT = "replacement"
    IMPERFECT_REPAIR = "imperfect_repair"


@dataclass(frozen=True)
class GammaModel:
    """Accumulated degradation represented by a constant-rate Gamma model."""

    beta: float
    name: str = "gamma"

    def __post_init__(self) -> None:
        if not np.isfinite(self.beta):
            raise ValueError("Gamma rate beta must be finite.")

        if self.beta <= 0.0:
            raise ValueError("Gamma rate beta must be positive.")

    # ------------------------------------------------------------------
    # Parameter conversions
    # ------------------------------------------------------------------

    def increment_parameter(self, expected_damage: ArrayLike) -> FloatArray:
        """Convert expected damage increments into Gamma shapes.

        For a common rate beta,

            A_increment = beta * expected_damage.
        """

        expected_damage_array = self._nonnegative_array(
            expected_damage,
            name="expected_damage",
        )

        return self.beta * expected_damage_array

    def shape_from_expected_damage(self, expected_damage: ArrayLike) -> FloatArray:
        """Alias for increment_parameter for readability."""

        return self.increment_parameter(expected_damage)

    def expected_damage(self, shape: ArrayLike) -> FloatArray:
        """Return the expected damage represented by a Gamma shape."""

        shape_array = self._nonnegative_array(shape, name="shape")
        return shape_array / self.beta

    def variance(self, shape: ArrayLike) -> FloatArray:
        """Return the variance represented by a Gamma shape."""

        shape_array = self._nonnegative_array(shape, name="shape")
        return shape_array / self.beta**2

    def accumulate(self, current_shape: ArrayLike, increment_shape: ArrayLike) -> FloatArray:
        """Add shapes belonging to independent Gamma variables.

        This operation is exact only because both variables use this model's
        common rate ``beta``. The rate is attached to the model rather than
        passed per increment, preventing accidental mixed-rate accumulation.
        """

        current = self._nonnegative_array(
            current_shape,
            name="current_shape",
        )
        increment = self._nonnegative_array(
            increment_shape,
            name="increment_shape",
        )
        try:
            return current + increment
        except ValueError as error:
            raise ValueError(
                "current_shape and increment_shape cannot be broadcast to compatible shapes."
            ) from error

    # ------------------------------------------------------------------
    # Probability calculations
    # ------------------------------------------------------------------

    def tail_probability(self, shape: ArrayLike, threshold: float) -> FloatArray:
        """Return P(D > threshold).

        A zero shape is interpreted as the deterministic undamaged state D=0.
        SciPy's Gamma implementation does not represent this degenerate case
        directly, so it is handled explicitly.
        """

        shape_array = self._nonnegative_array(shape, name="shape")

        if not np.isfinite(threshold):
            raise ValueError("Damage threshold must be finite.")

        if threshold < 0.0:
            raise ValueError("Damage threshold cannot be negative.")

        probabilities = np.zeros_like(shape_array, dtype=float)
        positive_shape = shape_array > 0.0

        probabilities[positive_shape] = gamma.sf(threshold, a=shape_array[positive_shape], scale=1.0 / self.beta)

        # For a positive-shape Gamma variable, P(D > 0) = 1.
        if threshold == 0.0:
            probabilities[positive_shape] = 1.0

        return probabilities

    def satisfies_reliability(
        self,
        shape: ArrayLike,
        threshold: float,
        epsilon: float,
        tolerance: float = 1e-10,
    ) -> NDArray[np.bool_]:
        """Check whether P(D > threshold) <= epsilon."""

        if not np.isfinite(epsilon):
            raise ValueError("epsilon must be finite.")

        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("epsilon must lie between zero and one.")

        if tolerance < 0.0:
            raise ValueError("tolerance cannot be negative.")

        probability = self.tail_probability(shape, threshold)
        return probability <= epsilon + tolerance

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def idle(self, current_shape: ArrayLike) -> FloatArray:
        """Apply an idle step: no degradation is added."""

        current = self._nonnegative_array(current_shape, name="current_shape")

        # Return a copy so the caller cannot accidentally modify the old state.
        return current.copy()

    def mission(self, current_shape: ArrayLike, expected_increment: ArrayLike) -> FloatArray:
        """Apply a mission damage increment.

        The increment has the same rate beta as the current state. Therefore,
        exact Gamma closure under addition applies and the shapes can be added.
        """

        current = self._nonnegative_array(current_shape, name="current_shape")
        increment_shape = self.increment_parameter(expected_increment)

        try:
            return current + increment_shape
        except ValueError as error:
            raise ValueError(
                "current_shape and expected_increment cannot be broadcast to compatible shapes."
            ) from error

    def replacement(self, expected_damage_new: ArrayLike) -> FloatArray:
        """Replace the component and reset it to a new initial state.
        expected_damage_new may be zero for an ideal new component, or positive
        if newly installed components have nonzero initial damage.
        """

        return self.shape_from_expected_damage(expected_damage_new)

    def imperfect_repair(self, current_shape: ArrayLike, rho: float) -> FloatArray:
        """Apply the mean-matched common-beta imperfect-repair approximation.

        If D_plus = (1-rho) * D_minus and

            D_minus ~ Gamma(A, beta),

        then

            D_plus ~ Gamma(A, beta / (1-rho)).

        The exact repaired variable has rate ``beta / (1-rho)``.  The solver
        instead keeps ``beta`` and reduces the shape to ``(1-rho)A``.  Both
        representations have the same repaired mean, but different variance.
        """

        current = self._nonnegative_array(current_shape, name="current_shape")

        if not np.isfinite(rho):
            raise ValueError("Repair effectiveness rho must be finite.")

        if rho < 0.0 or rho > 1.0:
            raise ValueError("Repair effectiveness rho must lie in [0, 1].")

        return (1.0 - rho) * current

    def transition(
        self,
        current_shape: ArrayLike,
        action: Union[GammaAction, str],
        expected_increment: Optional[ArrayLike] = None,
        expected_damage_new: Optional[ArrayLike] = None,
        rho: Optional[float] = None,
    ) -> FloatArray:
        """Apply one degradation transition.

        Parameters
        ----------
        current_shape:
            Shape parameter before the decision.
        action:
            One of idle, mission, replacement or imperfect_repair.
        expected_increment:
            Expected mission damage. Required for a mission.
        expected_damage_new:
            Expected initial damage of the installed component. Required for
            replacement.
        rho:
            Fraction of expected damage removed by imperfect repair.
        """

        try:
            selected_action = GammaAction(action)
        except ValueError as error:
            supported = ", ".join(item.value for item in GammaAction)
            raise ValueError(
                f"Unsupported Gamma action {action!r}. "
                f"Supported actions: {supported}."
            ) from error

        if selected_action == GammaAction.IDLE:
            self._reject_unexpected_argument(
                expected_increment,
                "expected_increment",
                selected_action,
            )
            self._reject_unexpected_argument(
                expected_damage_new,
                "expected_damage_new",
                selected_action,
            )
            self._reject_unexpected_argument(rho, "rho", selected_action)
            return self.idle(current_shape)

        if selected_action == GammaAction.MISSION:
            if expected_increment is None:
                raise ValueError(
                    "expected_increment is required for a mission transition."
                )

            self._reject_unexpected_argument(
                expected_damage_new,
                "expected_damage_new",
                selected_action,
            )
            self._reject_unexpected_argument(rho, "rho", selected_action)

            return self.mission(current_shape=current_shape, expected_increment=expected_increment)

        if selected_action == GammaAction.REPLACEMENT:
            if expected_damage_new is None:
                raise ValueError(
                    "expected_damage_new is required for replacement."
                )

            self._reject_unexpected_argument(
                expected_increment,
                "expected_increment",
                selected_action,
            )
            self._reject_unexpected_argument(rho, "rho", selected_action)

            # current_shape is intentionally not used in the result, but it is
            # validated because it is part of the common transition contract.
            self._nonnegative_array(current_shape, name="current_shape")

            return self.replacement(expected_damage_new)

        if selected_action == GammaAction.IMPERFECT_REPAIR:
            if rho is None:
                raise ValueError("rho is required for an imperfect-repair transition.")

            self._reject_unexpected_argument(
                expected_increment,
                "expected_increment",
                selected_action,
            )
            self._reject_unexpected_argument(
                expected_damage_new,
                "expected_damage_new",
                selected_action,
            )

            return self.imperfect_repair(current_shape=current_shape, rho=rho)

        # The Enum conversion above should make this unreachable.
        raise RuntimeError("Unhandled Gamma transition.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nonnegative_array(value: ArrayLike, name: str) -> FloatArray:
        """Convert a value to a finite, nonnegative float array."""

        array = np.asarray(value, dtype=float)

        if np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")

        if np.any(array < 0.0):
            raise ValueError(f"{name} cannot contain negative values.")

        return array

    @staticmethod
    def _reject_unexpected_argument(
        value: object,
        argument_name: str,
        action: GammaAction,
    ) -> None:
        """Reject parameters that do not belong to the selected action."""

        if value is not None:
            raise ValueError(
                f"{argument_name} is not used for action {action.value!r}."
            )


def _validate_positive(name: str, value: float) -> None:
    """Require a finite, strictly positive scalar value."""

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}.")

    if value <= 0.0:
        raise ValueError(f"{name} must be positive; got {value!r}.")


def maximum_reliable_shape(
    beta: float,
    threshold: float,
    epsilon: float,
    *,
    initial_upper_bound: float = 1.0,
    maximum_search_shape: float = 1e9,
) -> float:
    """Return the largest Gamma shape satisfying the reliability constraint.

    The shape-rate Gamma convention is used:

        D ~ Gamma(A, beta)

        E[D]   = A / beta
        Var[D] = A / beta**2

    This function finds A_max such that

        P(D > threshold) = epsilon.

    Because the Gamma tail probability increases monotonically with A
    for fixed beta and threshold, every shape A <= A_max satisfies

        P(D > threshold) <= epsilon.

    Parameters
    ----------
    beta:
        Positive Gamma rate parameter.
    threshold:
        Positive failure threshold.
    epsilon:
        Maximum accepted failure probability. Must lie strictly between
        zero and one.
    initial_upper_bound:
        Initial upper search bound for the shape parameter.
    maximum_search_shape:
        Safety limit for the expanding search interval.

    Returns
    -------
    float
        Largest admissible Gamma shape A_max.
    """
    _validate_positive("beta", beta)
    _validate_positive("threshold", threshold)
    _validate_positive("initial_upper_bound", initial_upper_bound)
    _validate_positive("maximum_search_shape", maximum_search_shape)

    if not 0.0 < epsilon < 1.0:
        raise ValueError(
            f"epsilon must lie strictly between 0 and 1; got {epsilon}."
        )

    def reliability_residual(shape: float) -> float:
        """Return tail probability minus the permitted probability."""

        # A = 0 represents the deterministic undamaged state D = 0.
        # Since threshold > 0, its failure probability is zero.
        if shape == 0.0:
            return -epsilon

        tail_probability = gamma.sf(threshold, a=shape, scale=1.0 / beta)

        if not np.isfinite(tail_probability):
            raise RuntimeError(
                "Gamma tail calculation returned a non-finite value for "
                f"shape={shape}, beta={beta}, threshold={threshold}."
            )

        return float(tail_probability - epsilon)

    # At A = 0, damage is treated as deterministically zero. Since the
    # threshold is positive, the failure probability is zero.
    lower_shape = 0.0
    upper_shape = initial_upper_bound

    # Find an upper bound whose tail probability exceeds epsilon.
    while reliability_residual(upper_shape) < 0.0:
        upper_shape *= 2.0

        if upper_shape > maximum_search_shape:
            raise RuntimeError(
                "Could not bracket the maximum reliable Gamma shape. "
                "Check beta, threshold, epsilon and maximum_search_shape."
            )

    return float(
        brentq(
            reliability_residual,
            lower_shape,
            upper_shape,
            xtol=1e-12,
            rtol=1e-12,
        )
    )


def maximum_reliable_expected_damage(
    beta: float,
    threshold: float,
    epsilon: float,
) -> float:
    """Return the largest admissible expected accumulated damage.

    Since E[D] = A / beta, this is simply A_max / beta.
    """
    maximum_shape = maximum_reliable_shape(
        beta=beta,
        threshold=threshold,
        epsilon=epsilon,
    )
    return maximum_shape / beta