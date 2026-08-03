import numpy as np
import pytest

from fleet_management.degradation.gamma import GammaAction, GammaModel
from fleet_management.gamma_validator import (
    _exact_repair_diagnostic,
    _reconstruct_shapes,
    _repair_cost_violation,
)


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


def test_imperfect_repair_preserves_the_repaired_mean(model: GammaModel):
    updated = model.transition(
        current_shape=[5.0],
        action=GammaAction.IMPERFECT_REPAIR,
        rho=0.45,
    )

    np.testing.assert_allclose(updated, [2.75])
    np.testing.assert_allclose(model.expected_damage(updated), [0.1375])


@pytest.mark.parametrize("rho", [-0.01, 1.01, np.inf])
def test_imperfect_repair_rejects_invalid_effectiveness(
    model: GammaModel, rho: float
):
    with pytest.raises(ValueError, match="rho"):
        model.imperfect_repair([5.0], rho=rho)


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


def test_validator_reconstructs_repair_and_replacement_separately():
    F, H, M, L = 2, 1, 1, 1
    beta = np.array([20.0])
    x = np.zeros((F, M + 1, 2 * H), dtype=int)
    m = np.zeros((F, L, 2 * H), dtype=int)
    r = np.zeros((F, L, 2 * H), dtype=int)

    # Vehicle 0 is repaired before its mission. Vehicle 1 completes its
    # mission first and is then replaced.
    x[0, 0, 0] = 1
    m[0, 0, 0] = 1
    x[0, 1, 1] = 1
    x[1, 1, 0] = 1
    x[1, 0, 1] = 1
    r[1, 0, 1] = 1

    reconstructed = _reconstruct_shapes(
        x=x,
        m=m,
        r=r,
        F=F,
        H=H,
        M=M,
        L=L,
        beta=beta,
        mu_param=np.full((F, M, L, H), 0.05),
        mu_0=np.array([[0.20], [0.30]]),
        replacement_mu=np.array([[0.01], [0.02]]),
        repair_rho=np.array([0.50]),
    )

    np.testing.assert_allclose(reconstructed[:, 0, :], [[2.0, 3.0], [7.0, 0.4]])


def test_repair_cost_tracks_removed_expected_damage_only():
    m = np.zeros((1, 1, 2), dtype=int)
    m[0, 0, 0] = 1
    violation = _repair_cost_violation(
        m=m,
        z=np.array([[[0.10, 0.00]]]),
        reconstructed_mu=np.array([[[0.10, 0.15]]]),
        mu_0=np.array([[0.20]]),
        repair_rho=np.array([0.50]),
    )

    assert violation == pytest.approx(0.0)


def test_exact_scaled_repair_is_reported_as_offline_diagnostic():
    diagnostic = _exact_repair_diagnostic(
        m=np.ones((1, 1, 1), dtype=int),
        reconstructed_A=np.array([[[2.0]]]),
        initial_shape=np.array([[4.0]]),
        beta=np.array([20.0]),
        tau=np.array([0.20]),
        repair_rho=np.array([0.50]),
    )

    assert diagnostic["repair_events"] == 1
    assert diagnostic["maximum_absolute_tail_difference"] > 0.0
    assert diagnostic["worst_event"]["vehicle"] == 0
    assert diagnostic["worst_event"]["component"] == 0