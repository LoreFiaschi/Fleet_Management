"""
Gamma-process degradation utilities.

The project document uses the shape-rate parameterisation

    X ~ Gamma(shape=A, rate=beta)

with

    E[X]   = A / beta
    Var[X] = A / beta**2

For a Gamma process, independent increments remain Gamma distributed if they
share the same rate beta. Therefore, for one component l, beta_l should stay
constant across the trajectory and mission increments accumulate by adding
their shape parameters.

This module provides small helper functions only. It does not depend on the
solver or validator data structures.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import gamma


def mean_to_shape(mean_damage: float | np.ndarray, beta: float | np.ndarray) -> float | np.ndarray:
    """
    Convert expected damage to Gamma shape under the shape-rate convention.

    For X ~ Gamma(A, beta), using rate beta:

        E[X] = A / beta

    therefore:

        A = beta * E[X]

    Parameters
    ----------
    mean_damage:
        Expected damage value or array of expected damage values.

    beta:
        Gamma rate parameter. Must be strictly positive.

    Returns
    -------
    float | np.ndarray
        Gamma shape parameter(s).
    """

    mean_damage_arr = np.asarray(mean_damage, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)

    if np.any(mean_damage_arr < 0.0):
        raise ValueError("mean_damage must be non-negative.")

    if np.any(beta_arr <= 0.0):
        raise ValueError("beta must be strictly positive.")

    shape = beta_arr * mean_damage_arr

    if np.isscalar(mean_damage) and np.isscalar(beta):
        return float(shape)

    return shape


def shape_to_mean(shape: float | np.ndarray, beta: float | np.ndarray) -> float | np.ndarray:
    """
    Convert Gamma shape to expected damage under the shape-rate convention.

    For X ~ Gamma(A, beta):

        E[X] = A / beta

    Parameters
    ----------
    shape:
        Gamma shape parameter. Must be non-negative.

    beta:
        Gamma rate parameter. Must be strictly positive.

    Returns
    -------
    float | np.ndarray
        Expected damage value(s).
    """

    shape_arr = np.asarray(shape, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)

    if np.any(shape_arr < 0.0):
        raise ValueError("shape must be non-negative.")

    if np.any(beta_arr <= 0.0):
        raise ValueError("beta must be strictly positive.")

    mean = shape_arr / beta_arr

    if np.isscalar(shape) and np.isscalar(beta):
        return float(mean)

    return mean


def shape_to_variance(shape: float | np.ndarray, beta: float | np.ndarray) -> float | np.ndarray:
    """
    Compute Gamma variance under the shape-rate convention.

    For X ~ Gamma(A, beta):

        Var[X] = A / beta**2

    Parameters
    ----------
    shape:
        Gamma shape parameter. Must be non-negative.

    beta:
        Gamma rate parameter. Must be strictly positive.

    Returns
    -------
    float | np.ndarray
        Variance value(s).
    """

    shape_arr = np.asarray(shape, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)

    if np.any(shape_arr < 0.0):
        raise ValueError("shape must be non-negative.")

    if np.any(beta_arr <= 0.0):
        raise ValueError("beta must be strictly positive.")

    variance = shape_arr / beta_arr**2

    if np.isscalar(shape) and np.isscalar(beta):
        return float(variance)

    return variance


def add_shape_increment(
    current_shape: float | np.ndarray,
    shape_increment: float | np.ndarray,
) -> float | np.ndarray:
    """
    Add a Gamma shape increment to the current accumulated shape.

    This is valid when both the current damage state and the increment use the
    same rate beta.

        Gamma(A_current, beta) + Gamma(A_increment, beta)
        = Gamma(A_current + A_increment, beta)

    Parameters
    ----------
    current_shape:
        Current accumulated Gamma shape.

    shape_increment:
        Mission-induced Gamma shape increment.

    Returns
    -------
    float | np.ndarray
        Updated Gamma shape.
    """

    current_shape_arr = np.asarray(current_shape, dtype=float)
    shape_increment_arr = np.asarray(shape_increment, dtype=float)

    if np.any(current_shape_arr < 0.0):
        raise ValueError("current_shape must be non-negative.")

    if np.any(shape_increment_arr < 0.0):
        raise ValueError("shape_increment must be non-negative.")

    updated_shape = current_shape_arr + shape_increment_arr

    if np.isscalar(current_shape) and np.isscalar(shape_increment):
        return float(updated_shape)

    return updated_shape


def failure_probability(
    shape: float | np.ndarray,
    beta: float | np.ndarray,
    threshold: float | np.ndarray,
) -> float | np.ndarray:
    """
    Compute P(D > threshold) for D ~ Gamma(shape, beta).

    scipy.stats.gamma uses shape-scale convention, so the scale passed to scipy is:

        scale = 1 / beta

    Parameters
    ----------
    shape:
        Gamma shape parameter. Must be non-negative.

    beta:
        Gamma rate parameter. Must be strictly positive.

    threshold:
        Failure threshold. Must be non-negative.

    Returns
    -------
    float | np.ndarray
        Failure probability P(D > threshold).
    """

    shape_arr = np.asarray(shape, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)
    threshold_arr = np.asarray(threshold, dtype=float)

    if np.any(shape_arr < 0.0):
        raise ValueError("shape must be non-negative.")

    if np.any(beta_arr <= 0.0):
        raise ValueError("beta must be strictly positive.")

    if np.any(threshold_arr < 0.0):
        raise ValueError("threshold must be non-negative.")

    # Degenerate no-damage case: Gamma with shape=0 is a point mass at zero.
    # scipy's gamma distribution is defined for shape > 0, so handle shape=0 manually.
    fail_prob = np.zeros(np.broadcast_shapes(shape_arr.shape, beta_arr.shape, threshold_arr.shape))

    positive_shape = shape_arr > 0.0

    if np.any(positive_shape):
        fail_prob = np.where(
            positive_shape,
            gamma.sf(
                threshold_arr,
                a=shape_arr,
                scale=1.0 / beta_arr,
            ),
            np.where(threshold_arr < 0.0, 1.0, 0.0),
        )

    if np.isscalar(shape) and np.isscalar(beta) and np.isscalar(threshold):
        return float(fail_prob)

    return fail_prob


def reliability_passed(
    shape: float | np.ndarray,
    beta: float | np.ndarray,
    threshold: float | np.ndarray,
    epsilon: float,
    tol: float = 1e-12,
) -> bool | np.ndarray:
    """
    Check the reliability condition

        P(D > threshold) <= epsilon

    for D ~ Gamma(shape, beta).

    Parameters
    ----------
    shape:
        Gamma shape parameter.

    beta:
        Gamma rate parameter.

    threshold:
        Failure threshold.

    epsilon:
        Maximum allowed failure probability.

    tol:
        Numerical tolerance.

    Returns
    -------
    bool | np.ndarray
        True where the reliability condition is satisfied.
    """

    if epsilon < 0.0 or epsilon > 1.0:
        raise ValueError("epsilon must be in [0, 1].")

    fail_prob = failure_probability(shape, beta, threshold)
    passed = fail_prob <= epsilon + tol

    if np.isscalar(fail_prob):
        return bool(passed)

    return passed


def gamma_quantile(
    shape: float | np.ndarray,
    beta: float | np.ndarray,
    probability: float,
) -> float | np.ndarray:
    """
    Compute the damage quantile q such that

        P(D <= q) = probability

    for D ~ Gamma(shape, beta).

    This is useful for chance constraints. For example, requiring

        q_{1-epsilon} <= threshold

    is equivalent to

        P(D > threshold) <= epsilon.

    Parameters
    ----------
    shape:
        Gamma shape parameter.

    beta:
        Gamma rate parameter.

    probability:
        CDF probability in [0, 1].

    Returns
    -------
    float | np.ndarray
        Gamma quantile.
    """

    shape_arr = np.asarray(shape, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)

    if probability < 0.0 or probability > 1.0:
        raise ValueError("probability must be in [0, 1].")

    if np.any(shape_arr < 0.0):
        raise ValueError("shape must be non-negative.")

    if np.any(beta_arr <= 0.0):
        raise ValueError("beta must be strictly positive.")

    quantile = np.zeros_like(np.broadcast_to(shape_arr, np.broadcast_shapes(shape_arr.shape, beta_arr.shape)), dtype=float)

    positive_shape = shape_arr > 0.0

    if np.any(positive_shape):
        quantile = np.where(
            positive_shape,
            gamma.ppf(probability, a=shape_arr, scale=1.0 / beta_arr),
            0.0,
        )
    
    if np.isscalar(shape) and np.isscalar(beta):
        return float(quantile)

    return quantile


def loop_constraint_passed(
    shape_mid_horizon: float | np.ndarray,
    shape_end_horizon: float | np.ndarray,
    tol: float = 1e-12,
) -> bool | np.ndarray:
    """
    Check the Gamma loop constraint under the shared-rate assumption.

    If both states use the same beta, then stochastic ordering reduces to
    comparing the accumulated shape parameters. The end of the horizon should
    be no worse than the midpoint:

        A_2H <= A_H

    Parameters
    ----------
    shape_mid_horizon:
        Gamma shape at k = H.

    shape_end_horizon:
        Gamma shape at k = 2H.

    tol:
        Numerical tolerance.

    Returns
    -------
    bool | np.ndarray
        True where A_2H <= A_H.
    """

    shape_mid_arr = np.asarray(shape_mid_horizon, dtype=float)
    shape_end_arr = np.asarray(shape_end_horizon, dtype=float)

    if np.any(shape_mid_arr < 0.0):
        raise ValueError("shape_mid_horizon must be non-negative.")

    if np.any(shape_end_arr < 0.0):
        raise ValueError("shape_end_horizon must be non-negative.")

    passed = shape_end_arr <= shape_mid_arr + tol

    if np.isscalar(shape_mid_horizon) and np.isscalar(shape_end_horizon):
        return bool(passed)

    return passed


def mission_mean_increment_to_gamma_row(
    mean_increment: float,
    beta: float,
    current_shape: float,
    threshold: float,
    epsilon: float,
) -> dict:
    """
    Convenience helper for diagnostics.

    Converts one expected mission damage increment to a Gamma shape increment,
    updates the current shape, and returns a small dictionary with the most
    useful quantities for logs or dataframes.

    Parameters
    ----------
    mean_increment:
        Expected mission damage increment.

    beta:
        Component-specific Gamma rate.

    current_shape:
        Accumulated Gamma shape before the mission.

    threshold:
        Failure threshold in damage units.

    epsilon:
        Maximum allowed failure probability.

    Returns
    -------
    dict
        Diagnostic quantities.
    """

    shape_increment = mean_to_shape(mean_increment, beta)
    shape_after = add_shape_increment(current_shape, shape_increment)

    mean_before = shape_to_mean(current_shape, beta)
    mean_after = shape_to_mean(shape_after, beta)
    variance_after = shape_to_variance(shape_after, beta)
    fail_prob_after = failure_probability(shape_after, beta, threshold)

    return {
        "beta": float(beta),
        "shape_before": float(current_shape),
        "shape_increment": float(shape_increment),
        "shape_after": float(shape_after),
        "mean_before": float(mean_before),
        "mean_increment": float(mean_increment),
        "mean_after": float(mean_after),
        "variance_after": float(variance_after),
        "threshold": float(threshold),
        "failure_probability_after": float(fail_prob_after),
        "epsilon": float(epsilon),
        "reliability_passed": bool(fail_prob_after <= epsilon),
    }