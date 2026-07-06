from fleet_management.validator import validate_gamma_synthetic_diagnostic, build_gamma_diagnostic_dataframe


def _assert_gamma_dataframe_compatible(df):
    expected_columns = {
        "degradation",
        "state_interpretation",
        "time_step",
        "input_day",
        "vehicle",
        "activity",
        "mission",
        "component",
        "beta",
        "tau",
        "epsilon",
        "shape_before",
        "shape_increment",
        "shape_after",
        "mean_before",
        "mean_increment",
        "mean_after",
        "variance_after",
        "failure_probability_after",
        "failure_probability_percent",
        "reliability_passed",
        "damage_before",
        "expected_increment",
        "damage_after",
        "threshold",
        "margin_to_threshold",
        "utilization_of_threshold",
        "threshold_utilization_percent",
        "feasible",
        "assignment_feasible",
        "violating_components",
        "status",
    }

    assert expected_columns.issubset(set(df.columns))
    assert len(df) == 16
    assert df["degradation"].eq("gamma").all()
    assert df["state_interpretation"].eq("gamma_shape_rate").all()
    assert df["failure_probability_after"].between(0.0, 1.0).all()
    assert df["failure_probability_percent"].between(0.0, 100.0).all()
    assert df["threshold_utilization_percent"].ge(0.0).all()
    assert df["violating_components"].apply(lambda x: isinstance(x, list)).all()


def test_gamma_no_maintenance_diagnostic_fails_loop(tmp_path):
    log_path = tmp_path / "gamma_no_maintenance_diagnostic.log"

    report = validate_gamma_synthetic_diagnostic(
        input_path="input/tiny_gamma_synthetic.yaml",
        log_path=str(log_path),
    )

    assert report["rows_checked"] == 16
    assert report["reliability_failures"] == 0
    assert report["loop_passed"] is False
    assert report["passed"] is False
    assert "max_failure_probability" in report
    assert "max_failure_probability_location" in report
    assert "max_shape_after" in report
    assert report["max_failure_probability"] >= 0.0
    assert report["max_shape_after"] >= 0.0
    assert isinstance(report["max_failure_probability_location"], dict)
    assert log_path.exists()


def test_gamma_replacement_diagnostic_passes_loop(tmp_path):
    log_path = tmp_path / "gamma_replacement_diagnostic.log"

    report = validate_gamma_synthetic_diagnostic(
        input_path="input/tiny_gamma_synthetic_replacement.yaml",
        log_path=str(log_path),
    )

    assert report["rows_checked"] == 16
    assert report["reliability_failures"] == 0
    assert report["loop_passed"] is True
    assert report["passed"] is True
    assert "max_failure_probability" in report
    assert "max_failure_probability_location" in report
    assert "max_shape_after" in report
    assert report["max_failure_probability"] >= 0.0
    assert report["max_shape_after"] >= 0.0
    assert isinstance(report["max_failure_probability_location"], dict)
    assert log_path.exists()


def test_gamma_no_maintenance_dataframe_columns():
    df = build_gamma_diagnostic_dataframe(
        input_path="input/tiny_gamma_synthetic.yaml",
    )

    _assert_gamma_dataframe_compatible(df)


def test_gamma_replacement_dataframe_columns():
    df = build_gamma_diagnostic_dataframe(
        input_path="input/tiny_gamma_synthetic_replacement.yaml",
    )

    _assert_gamma_dataframe_compatible(df)
