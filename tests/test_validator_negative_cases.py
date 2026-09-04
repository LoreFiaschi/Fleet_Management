from pathlib import Path

import pytest

from fleet_management.validation.validator import validate


INPUT = "input/data_test_baseline.yaml"
TEST_DIR = Path("results/validator_tests")


EXPECTED_FAILED_CHECKS = {
    "bad_status.yaml": "solver_status_optimal",
    "bad_x_binary.yaml": "x_binary",
    "bad_assignment.yaml": "assignment_sum_j_x_le_1",
    "bad_demand.yaml": "demand_sum_i_x_eq_1",
    "bad_u_ge_mu.yaml": "u_ge_mu",
    "bad_capacity.yaml": "capacity_sum_mu_le_F_minus_M",
    "bad_mu_periodic.yaml": "mu_periodic",
    "bad_v_periodic.yaml": "v_periodic",
    "bad_objective.yaml": "objective_recomputation",
}


@pytest.mark.parametrize("filename, expected_failed_check", EXPECTED_FAILED_CHECKS.items())
def test_validator_catches_corrupted_outputs(filename, expected_failed_check):
    report = validate(
        input_path=INPUT,
        degradation="gaussian",
        results_path=str(TEST_DIR / filename),
        validation_path=str(TEST_DIR / f"validation_{filename}"),
    )

    failed_checks = {
        check["name"]
        for check in report["checks"]
        if not check["passed"]
    }

    assert expected_failed_check in failed_checks
    assert report["passed"] is False


def test_validator_catches_dimension_error():
    with pytest.raises(ValueError, match="shape"):
        validate(
            input_path=INPUT,
            degradation="gaussian",
            results_path=str(TEST_DIR / "bad_dimension.yaml"),
            validation_path=str(TEST_DIR / "validation_bad_dimension.yaml"),
        )
