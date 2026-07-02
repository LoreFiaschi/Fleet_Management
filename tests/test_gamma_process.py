import numpy as np

from fleet_management.gamma_process import (
    mean_to_shape,
    shape_to_mean,
    shape_to_variance,
    failure_probability,
    reliability_passed,
    loop_constraint_passed,
)


def test_mean_shape_round_trip():
    beta = 20.0
    mean_damage = 0.15

    shape = mean_to_shape(mean_damage, beta)
    recovered_mean = shape_to_mean(shape, beta)

    assert np.isclose(recovered_mean, mean_damage)


def test_variance_shape_rate_convention():
    beta = 10.0
    shape = 5.0

    expected_variance = shape / beta**2

    assert np.isclose(shape_to_variance(shape, beta), expected_variance)


def test_failure_probability_between_zero_and_one():
    fail_prob = failure_probability(
        shape=5.0,
        beta=20.0,
        threshold=0.5,
    )

    assert 0.0 <= fail_prob <= 1.0


def test_reliability_passed_returns_bool():
    result = reliability_passed(
        shape=1.0,
        beta=50.0,
        threshold=0.5,
        epsilon=0.05,
    )

    assert isinstance(result, bool)


def test_loop_constraint_passed():
    assert loop_constraint_passed(
        shape_mid_horizon=5.0,
        shape_end_horizon=4.0,
    )

    assert not loop_constraint_passed(
        shape_mid_horizon=5.0,
        shape_end_horizon=6.0,
    )
