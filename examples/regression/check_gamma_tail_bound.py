"""Numerical smoke checks for the standalone Gamma tail-bound calculation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def find_repository_root(start: Path) -> Path:
    """Find the project root independently of the checker's directory depth."""

    for candidate in (start, *start.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src" / "fleet_management").is_dir()
        ):
            return candidate

    raise RuntimeError("Could not locate the fleet-management repository root.")


REPOSITORY_ROOT = find_repository_root(Path(__file__).resolve().parent)

MODULE_PATH = (
    REPOSITORY_ROOT
    / "src/fleet_management/degradation_model/gamma_utils/gamma_tail_bound.py"
)
# Load the standalone numerical module without importing fleet_management's
# solver-facing package __init__, which imports Gurobi.
SPEC = importlib.util.spec_from_file_location("gamma_tail_bound_standalone", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
TAIL_BOUND = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TAIL_BOUND
SPEC.loader.exec_module(TAIL_BOUND)
calculate_tail_bound_parameters = TAIL_BOUND.calculate_tail_bound_parameters
calculate_profile_tail_bound_parameters = (
    TAIL_BOUND.calculate_profile_tail_bound_parameters
)
calculate_fleet_tail_bound_parameters = (
    TAIL_BOUND.calculate_fleet_tail_bound_parameters
)
moschopoulos_tail_probability = TAIL_BOUND.moschopoulos_tail_probability
shapes_from_expected_damage = TAIL_BOUND.shapes_from_expected_damage


def main() -> None:
    np.testing.assert_allclose(
        shapes_from_expected_damage([0.1, 0.4, 0.2], [20.0, 10.0, 15.0]),
        [2.0, 4.0, 3.0],
    )

    # Two unit-shape Gammas are exponentials, whose convolution tail has a
    # closed form.  This checks the different-rate convolution independently.
    rates = np.array([2.0, 5.0])
    threshold = 0.5
    convolution = moschopoulos_tail_probability(
        shapes=[1.0, 1.0], rates=rates, threshold=threshold
    )
    closed_form = (
        rates[1] * np.exp(-rates[0] * threshold)
        - rates[0] * np.exp(-rates[1] * threshold)
    ) / (rates[1] - rates[0])
    np.testing.assert_allclose(convolution.estimate, closed_form, atol=2e-12, rtol=0.0)
    assert convolution.upper_bound >= closed_form - 2e-12

    result = calculate_tail_bound_parameters(
        shapes=[2.0, 4.0, 3.0],
        rates=[20.0, 10.0, 15.0],
        threshold=0.5,
        max_counts=[1, 1, 1],
    )

    assert result.common_rate == 10.0
    assert len(result.constraints) == 7  # all nonempty subsets
    assert result.all_constraints_conservative
    assert np.all(result.bounded_shapes <= result.original_shapes + 1e-9)
    assert result.bounded_shapes[0] < result.original_shapes[0]
    assert result.bounded_shapes[2] < result.original_shapes[2]
    # This increment already had the selected common rate, so its singleton
    # tail leaves no room to reduce the shape.
    np.testing.assert_allclose(result.bounded_shapes[1], 4.0, atol=1e-8, rtol=0.0)

    # A real per-cell profile has layout (M, H).  The first increment type is
    # repeated twice, while at most H=2 increments can be accumulated because
    # a vehicle can execute at most one mission per time step.
    profile = calculate_profile_tail_bound_parameters(
        expected_damage=[[0.1, 0.2], [0.1, 0.3]],
        rates=[[20.0, 15.0], [20.0, 10.0]],
        threshold=0.5,
        max_total_count=2,
    )

    assert profile.expected_damage.shape == (2, 2)
    assert profile.original_rates.shape == (2, 2)
    assert profile.bounded_shapes.shape == (2, 2)
    assert profile.common_rate == 10.0
    assert profile.all_constraints_conservative
    np.testing.assert_array_equal(profile.type_max_counts, [2, 1, 1])
    np.testing.assert_array_equal(profile.type_indices, [[0, 1], [0, 2]])
    # Three singletons, three mixed pairs, and two occurrences of the repeated
    # first type form the conservative count-vector superset.
    assert len(profile.compressed.constraints) == 7
    np.testing.assert_allclose(
        profile.bounded_shapes[0, 0], profile.bounded_shapes[1, 0]
    )
    # The third type already uses the selected common rate.
    np.testing.assert_allclose(profile.bounded_shapes[1, 1], 3.0, atol=1e-8, rtol=0.0)

    # The fleet-facing wrapper consumes normalized (F, L, M, H2) arrays and
    # applies the unequal-horizon fallback used by the Gamma backend.
    fleet = calculate_fleet_tail_bound_parameters(
        expected_damage=np.array(
            [
                [[[[0.1, 0.1], [0.2, 0.2]]]],
                [[[[0.1, 0.1], [0.2, 0.2]]]],
            ]
        ).reshape(2, 1, 2, 2),
        rates=[20.0, 10.0],
        thresholds=0.5,
        H1=1,
    )
    assert (fleet.H1, fleet.H2, fleet.T) == (1, 2, 3)
    assert fleet.bounded_shapes_trans.shape == (2, 1, 2, 1)
    assert fleet.bounded_shapes_operating.shape == (2, 1, 2, 2)
    np.testing.assert_allclose(fleet.common_rates, 10.0)
    assert fleet.all_constraints_conservative
    # Mission 1 already uses beta_bar=10, so every occurrence keeps A=mu*beta=2.
    np.testing.assert_allclose(fleet.bounded_shapes_trans[:, :, 1, :], 2.0)
    np.testing.assert_allclose(fleet.bounded_shapes_operating[:, :, 1, :], 2.0)

    print("PASS standalone Gamma tail bound")
    print("original shapes:", result.original_shapes)
    print("bounded shapes :", result.bounded_shapes)
    print("common rate    :", result.common_rate)
    print("worst margin   :", result.worst_tail_margin)
    print("profile types  :", profile.type_max_counts)
    print("profile checks :", len(profile.compressed.constraints))
    print("fleet shape    :", fleet.bounded_shapes_operating.shape)


if __name__ == "__main__":
    main()