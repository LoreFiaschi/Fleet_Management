from fleet_management.validator.validator import validate


def test_validator_accepts_valid_baseline():
    report = validate(
        input_path="input/data_test_baseline.yaml",
        degradation="gaussian",
        results_path="results/output_baseline.yaml",
        validation_path="results/validation_baseline_test.yaml",
    )

    assert report["passed"] is True
    assert report["max_violation"] <= report["tolerance"]
    