import pytest
from scipy.stats import gamma

from fleet_management.degradation.gamma import (
    maximum_reliable_expected_damage,
    maximum_reliable_shape,
)


def gamma_tail(shape: float, beta: float, threshold: float) -> float:
    return float(
        gamma.sf(
            threshold,
            a=shape,
            scale=1.0 / beta,
        )
    )


def test_maximum_shape_reaches_requested_tail_probability() -> None:
    beta = 20.0
    threshold = 0.55
    epsilon = 0.05

    maximum_shape = maximum_reliable_shape(
        beta=beta,
        threshold=threshold,
        epsilon=epsilon,
    )

    probability = gamma_tail(maximum_shape, beta, threshold)

    assert probability == pytest.approx(epsilon, abs=1e-10)


def test_smaller_shape_is_reliable() -> None:
    beta = 20.0
    threshold = 0.55
    epsilon = 0.05

    maximum_shape = maximum_reliable_shape(
        beta=beta,
        threshold=threshold,
        epsilon=epsilon,
    )

    smaller_shape = 0.9 * maximum_shape

    assert gamma_tail(smaller_shape, beta, threshold) < epsilon


def test_larger_shape_is_not_reliable() -> None:
    beta = 20.0
    threshold = 0.55
    epsilon = 0.05

    maximum_shape = maximum_reliable_shape(
        beta=beta,
        threshold=threshold,
        epsilon=epsilon,
    )

    larger_shape = 1.1 * maximum_shape

    assert gamma_tail(larger_shape, beta, threshold) > epsilon


def test_expected_damage_corresponds_to_shape_bound() -> None:
    beta = 20.0
    threshold = 0.55
    epsilon = 0.05

    maximum_shape = maximum_reliable_shape(
        beta=beta,
        threshold=threshold,
        epsilon=epsilon,
    )
    maximum_mean = maximum_reliable_expected_damage(
        beta=beta,
        threshold=threshold,
        epsilon=epsilon,
    )

    assert maximum_mean == pytest.approx(maximum_shape / beta)


@pytest.mark.parametrize(
    ("beta", "threshold", "epsilon"),
    [
        (5.0, 0.25, 0.10),
        (20.0, 0.55, 0.05),
        (50.0, 0.80, 0.01),
    ],
)
def test_multiple_parameter_combinations(
    beta: float,
    threshold: float,
    epsilon: float,
) -> None:
    maximum_shape = maximum_reliable_shape(
        beta=beta,
        threshold=threshold,
        epsilon=epsilon,
    )

    assert gamma_tail(maximum_shape, beta, threshold) == pytest.approx(
        epsilon,
        abs=1e-10,
    )


@pytest.mark.parametrize(
    ("beta", "threshold", "epsilon"),
    [
        (0.0, 0.55, 0.05),
        (-1.0, 0.55, 0.05),
        (20.0, 0.0, 0.05),
        (20.0, -0.1, 0.05),
        (20.0, 0.55, 0.0),
        (20.0, 0.55, 1.0),
        (20.0, 0.55, -0.1),
        (20.0, 0.55, 1.1),
    ],
)
def test_invalid_parameters_are_rejected(
    beta: float,
    threshold: float,
    epsilon: float,
) -> None:
    with pytest.raises(ValueError):
        maximum_reliable_shape(
            beta=beta,
            threshold=threshold,
            epsilon=epsilon,
        )