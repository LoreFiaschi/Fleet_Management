import numpy as np
import pytest

from fleet_management.degradation.gamma import GammaAction, GammaModel


@pytest.fixture
def model() -> GammaModel:
    return GammaModel(beta=20.0)


def test_idle_preserves_shape(model: GammaModel):
    current = np.array([2.0, 4.0])

    updated = model.transition(
        current_shape=current,
        action=GammaAction.IDLE,
    )

    np.testing.assert_allclose(updated, current)
    assert updated is not current


def test_mission_adds_increment_shape(model: GammaModel):
    current_shape = np.array([2.0])

    updated = model.transition(
        current_shape=current_shape,
        action=GammaAction.MISSION,
        expected_increment=np.array([0.15]),
    )

    # Increment shape = beta * mu = 20 * 0.15 = 3.
    np.testing.assert_allclose(updated, [5.0])

    # Expected accumulated damage = 5 / 20 = 0.25.
    np.testing.assert_allclose(
        model.expected_damage(updated),
        [0.25],
    )


def test_multiple_missions_accumulate_exactly(model: GammaModel):
    state = model.shape_from_expected_damage([0.05])

    state = model.transition(
        current_shape=state,
        action="mission",
        expected_increment=[0.10],
    )
    state = model.transition(
        current_shape=state,
        action="mission",
        expected_increment=[0.20],
    )

    np.testing.assert_allclose(
        model.expected_damage(state),
        [0.35],
    )


def test_replacement_resets_state(model: GammaModel):
    highly_degraded_shape = np.array([12.0])

    updated = model.transition(
        current_shape=highly_degraded_shape,
        action=GammaAction.REPLACEMENT,
        expected_damage_new=np.array([0.02]),
    )

    # Replacement does not add to the old state.
    np.testing.assert_allclose(updated, [0.4])
    np.testing.assert_allclose(
        model.expected_damage(updated),
        [0.02],
    )


def test_replacement_can_reset_to_zero(model: GammaModel):
    updated = model.transition(
        current_shape=[10.0],
        action="replacement",
        expected_damage_new=[0.0],
    )

    np.testing.assert_allclose(updated, [0.0])
    np.testing.assert_allclose(
        model.tail_probability(updated, threshold=0.5),
        [0.0],
    )


def test_imperfect_repair_is_explicitly_rejected(model: GammaModel):
    with pytest.raises(
        NotImplementedError,
        match="Scaling damage changes the Gamma rate",
    ):
        model.transition(
            current_shape=[5.0],
            action=GammaAction.IMPERFECT_REPAIR,
            rho=0.45,
        )


def test_mission_requires_expected_increment(model: GammaModel):
    with pytest.raises(
        ValueError,
        match="expected_increment is required",
    ):
        model.transition(
            current_shape=[2.0],
            action="mission",
        )


def test_replacement_requires_new_damage(model: GammaModel):
    with pytest.raises(
        ValueError,
        match="expected_damage_new is required",
    ):
        model.transition(
            current_shape=[2.0],
            action="replacement",
        )


def test_idle_rejects_mission_increment(model: GammaModel):
    with pytest.raises(
        ValueError,
        match="expected_increment is not used",
    ):
        model.transition(
            current_shape=[2.0],
            action="idle",
            expected_increment=[0.1],
        )


def test_invalid_action_is_rejected(model: GammaModel):
    with pytest.raises(
        ValueError,
        match="Unsupported Gamma action",
    ):
        model.transition(
            current_shape=[2.0],
            action="magic_repair",
        )


def test_negative_damage_is_rejected(model: GammaModel):
    with pytest.raises(
        ValueError,
        match="cannot contain negative",
    ):
        model.transition(
            current_shape=[2.0],
            action="mission",
            expected_increment=[-0.1],
        )


def test_tail_probability_and_reliability(model: GammaModel):
    shape = model.shape_from_expected_damage([0.20])

    probability = model.tail_probability(
        shape=shape,
        threshold=0.55,
    )

    reliability = model.satisfies_reliability(
        shape=shape,
        threshold=0.55,
        epsilon=0.05,
    )

    assert probability.shape == (1,)
    assert 0.0 <= probability[0] <= 1.0
    assert reliability.shape == (1,)