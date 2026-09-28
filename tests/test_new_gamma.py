import numpy as np
import pytest

from fleet_management.degradation.gamma import GammaModel


def test_shape_from_expected_damage():
    model = GammaModel(beta=20.0)

    shape = model.increment_parameter(np.array([0.1, 0.2]))

    np.testing.assert_allclose(shape, [2.0, 4.0])


def test_expected_damage_round_trip():
    model = GammaModel(beta=20.0)
    means = np.array([0.1, 0.2])

    shapes = model.increment_parameter(means)
    reconstructed = model.expected_damage(shapes)

    np.testing.assert_allclose(reconstructed, means)


def test_common_beta_shapes_add():
    model = GammaModel(beta=20.0)

    a_1 = model.increment_parameter(np.array([0.1]))
    a_2 = model.increment_parameter(np.array([0.2]))
    accumulated = model.accumulate(a_1, a_2)

    np.testing.assert_allclose(accumulated, [6.0])
    np.testing.assert_allclose(
        model.expected_damage(accumulated),
        [0.3],
    )


def test_rejects_invalid_beta():
    with pytest.raises(ValueError):
        GammaModel(beta=0.0)