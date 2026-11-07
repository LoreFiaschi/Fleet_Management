"""Tests of the horizon sweeps of ``fleet_management.horizon_sweep``.

None of them needs Gurobi: ``solve_mapping`` is replaced by a stub that returns a result
from a table. The expected values come from the formulas of the spec (Section "Horizon
Sweeps"): the relative gradient (Eq. d_gradient), the classes, the eligibility, the stop
rule, the ranking with its tie rule, and the projected cost (Eq. d_projected).
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import yaml

from fleet_management import horizon_sweep


# ---------------------------------------------------------------------------
# Tables of cases: (status, J_op_average, mip_gap, requested_mip_gap), or "error"
# ---------------------------------------------------------------------------
_OPT = "optimal"

# Ranking, errors and warm-start sources: an infeasible case, a case error, a tie.
_T1 = {
    2: (_OPT, 1.0, 0.0, 0.1),
    3: (_OPT, 0.9, 0.0, 0.1),
    4: ("time_limit", 0.8, 0.05, 0.1),
    5: "error",
    6: ("infeasible", None, None, 0.1),
    7: (_OPT, 0.9, 0.0, 0.1),
}
# The cost increases at H2 = 6; the steps of H2 are uneven.
_T2 = {
    2: (_OPT, 1.0, 0.0, 0.1),
    3: (_OPT, 0.9, 0.0, 0.1),
    5: (_OPT, 0.85, 0.0, 0.1),
    6: (_OPT, 0.95, 0.08, 0.1),
    8: (_OPT, 0.7, 0.0, 0.1),
}
# The gradients are flat from H2 = 3 to H2 = 5, then the cost falls.
_T3 = {
    2: (_OPT, 1.0, 0.0, 0.1),
    3: (_OPT, 1.0005, 0.0, 0.1),
    4: (_OPT, 1.0004, 0.0, 0.1),
    5: (_OPT, 1.0, 0.0, 0.1),
    6: (_OPT, 0.5, 0.0, 0.1),
}
# A grid, keyed by (H1, H2), and the initialization cost of each H1.
_GRID = {
    (2, 2): (_OPT, 1.0, 0.0, 0.1),
    (2, 4): (_OPT, 0.9, 0.0, 0.1),
    (3, 2): ("time_limit", 0.85, 0.05, 0.1),
    (3, 4): (_OPT, 0.9, 0.0, 0.1),
}
_J_INIT = {2: 5.0, 3: 4.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_input(tmp_path, suffix=".yaml", **changes):
    """Write a small input without the optional key ``L``; return its path."""
    data = {
        "F": 3, "M": 1, "H": [2, 9], "model": "gaussian", "tau": 1.0, "epsilon": 0.1,
        "rho": 0.5, "mu_0": 0.1, "mu": 0.2, "v": 0.001, "C_M": 1.0, "C_R": 1.0,
        "C_rep": 5.0, "C_D": 0.0,
    }
    data.update(changes)
    path = tmp_path / f"input{suffix}"
    if suffix == ".json":
        path.write_text(json.dumps(data), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _result(H1, H2, entry, j_init, objective_mode, evaluation_horizon, warm_start):
    """The result dictionary of a case, with the keys that the sweep reads."""
    status, J, gap, requested = entry
    T = H1 + H2
    performance = {
        "variables": 10 * T, "continuous_variables": 4 * T, "integer_variables": 6 * T,
        "binary_variables": 6 * T, "linear_constraints": 20 * T,
        "general_constraints": 0, "indicator_constraints": 0, "quadratic_constraints": 0,
        "nonzeros": 50 * T, "requested_mip_gap": requested,
        "requested_time_limit_seconds": 60.0, "optimizer_call_seconds": 0.5,
        "solutions_found": 0 if J is None else 1,
    }
    performance.update({
        key: 0.1 for key in horizon_sweep.TIMING_KEYS if key not in performance
    })
    result = {
        "status": status, "objective": None, "mip_gap": gap, "bound": None,
        "F": 3, "M": 1, "L": 1, "H1": H1, "H2": H2, "T": T,
        "objective_mode": objective_mode, "evaluation_horizon": evaluation_horizon,
        "x": None, "m": None, "r": None,
        "warm_start": {"applied": warm_start is not None},
        "performance": performance,
    }
    result.update({key: None for key in horizon_sweep.COST_KEYS})
    if J is not None:
        J_init = j_init.get(H1, 2.0)
        projected = (
            None if evaluation_horizon is None
            else J_init + (evaluation_horizon - H1) * J
        )
        result.update({
            "J_initialization": J_init, "J_op": J * H2, "J_op_average": J,
            "peak_damage": 0.5, "damage_penalty": 0.0, "operating_objective": J,
            "J_total": J_init + J * H2, "projected_evaluation_cost": projected,
            "evaluation_objective": projected,
            "x": np.zeros((3, 2, T)), "m": np.zeros((3, 1, T)), "r": np.zeros((3, 1, T)),
        })
        objective = {
            "operating_average": J,
            "evaluation_total": projected,
            "total": J_init + J * H2,
        }[objective_mode]
        result["objective"] = objective
        result["bound"] = objective * (1.0 - gap)
    return result


class _Stub:
    """Replaces ``solve_mapping``: a result from a table, and a record of the calls."""

    def __init__(self, table, j_init=None, delay=None, report_path=None):
        self.table = table
        self.j_init = {} if j_init is None else j_init
        self.delay = {} if delay is None else delay
        self.report_path = report_path
        self.calls = []

    def __call__(self, data, results_path=None, *, warm_start=None, warm_start_mode="start"):
        assert results_path is None
        H1, H2 = data["H"]
        checkpoint = None
        if self.report_path is not None and self.report_path.exists():
            written = yaml.safe_load(self.report_path.read_text(encoding="utf-8"))
            checkpoint = (
                written["complete"],
                written["stopping_rule"]["reason"],
                written["evaluated_operating_horizons"],
            )
        self.calls.append({
            "H": list(data["H"]), "H1": H1, "H2": H2,
            "source_H2": None if warm_start is None else warm_start["H2"],
            "mode": warm_start_mode,
            "objective_mode": data.get("objective_mode"),
            "evaluation_horizon": data.get("evaluation_horizon"),
            "checkpoint": checkpoint,
        })
        time.sleep(self.delay.get(H2, 0))
        entry = self.table[(H1, H2) if (H1, H2) in self.table else H2]
        if entry == "error":
            raise RuntimeError(f"boom {H2}")
        return _result(
            H1, H2, entry, self.j_init, data.get("objective_mode"),
            data.get("evaluation_horizon"), warm_start,
        )


def _patch(monkeypatch, table, **options):
    stub = _Stub(table, **options)
    monkeypatch.setattr(horizon_sweep, "solve_mapping", stub)
    return stub


def _count_writes(monkeypatch):
    """Wrap ``_write_sweep_report``; the list receives ``complete`` of every report written."""
    writes = []
    original = horizon_sweep._write_sweep_report

    def counting(report, target):
        writes.append(report["complete"])
        original(report, target)

    monkeypatch.setattr(horizon_sweep, "_write_sweep_report", counting)
    return writes


_REPORT_KEYS = [
    "input", "complete", "last_completed_H2", "objective", "varied_parameter",
    "fixed_dimensions", "H1", "planned_operating_horizons",
    "evaluated_operating_horizons", "warm_start", "n_workers", "stopping_rule", "cases",
    "formulation_growth", "best_proven_H2", "best_feasible_H2",
    "best_proven_J_op_average", "best_feasible_J_op_average", "best_feasible_status",
]


# ---------------------------------------------------------------------------
# Operating sweep: rows, ranking, errors
# ---------------------------------------------------------------------------
def test_rows_ranking_and_errors(tmp_path, monkeypatch, caplog):
    path = _write_input(tmp_path)
    _patch(monkeypatch, _T1)
    report = horizon_sweep.sweep_operating_horizons(path, [7, 2, 3, 4, 5, 6])
    rows = report["cases"]

    assert list(report) == _REPORT_KEYS
    assert report["evaluated_operating_horizons"] == [2, 3, 4, 5, 6, 7]
    assert report["fixed_dimensions"] == {"F": 3, "M": 1, "L": 1}
    assert [row["gradient_classification"] for row in rows] == [
        "unavailable", "decrease", "decrease", "unavailable", "unavailable", "unavailable",
    ]
    assert rows[1]["relative_cost_gradient_per_H2"] == pytest.approx(-0.1)
    assert rows[2]["relative_cost_gradient_per_H2"] == pytest.approx(-0.11111111111111108)
    assert [row["gradient_stopping_eligible"] for row in rows] == [
        False, True, True, False, False, False,
    ]

    assert report["best_proven_H2"] == 3
    assert report["best_proven_J_op_average"] == 0.9
    assert report["best_feasible_H2"] == 4
    assert report["best_feasible_J_op_average"] == 0.8
    assert report["best_feasible_status"] == "time_limit"
    assert rows[2]["absolute_mip_gap"] == pytest.approx(0.04)
    assert report["stopping_rule"]["reason"] == "candidate_range_exhausted"
    assert report["stopping_rule"]["stopped_early"] is False

    error_row = rows[3]
    assert error_row["status"] == "error: boom 5"
    assert error_row["objective"] is None
    assert error_row["formulation"] is None
    assert error_row["timing"] is None
    assert error_row["warm_start"] is None
    assert error_row["dimensions"] == {"F": 3, "M": 1, "L": 1, "H1": 2, "H2": 5, "T": 7}
    assert all(list(row) == list(horizon_sweep.ROW_KEYS) for row in rows)

    errors = [
        record for record in caplog.records
        if record.levelno == logging.ERROR and record.name == horizon_sweep.logger.name
    ]
    assert len(errors) == 1
    assert "H1=2" in errors[0].getMessage() and "H2=5" in errors[0].getMessage()

    growth = report["formulation_growth"]
    assert growth["from_H2"] == 2 and growth["to_H2"] == 7
    assert growth["added_time_steps"] == 5
    assert growth["added_variables"] == 50
    assert growth["variables_per_added_time_step"] == 10.0


def test_warm_start_sources(tmp_path, monkeypatch):
    path = _write_input(tmp_path)
    stub = _patch(monkeypatch, _T1)
    report = horizon_sweep.sweep_operating_horizons(
        path, [2, 3, 4, 5, 6, 7], warm_start=True, warm_start_mode="both"
    )
    # an error and an infeasible case give no source: the preceding case with a schedule stays
    assert [call["source_H2"] for call in stub.calls] == [None, 2, 3, 4, 4, 4]
    assert all(call["mode"] == "both" for call in stub.calls)
    assert report["warm_start"] == {"enabled": True, "mode": "both"}


# ---------------------------------------------------------------------------
# Operating sweep: the early stop
# ---------------------------------------------------------------------------
def test_stop_cost_increase(tmp_path, monkeypatch):
    path = _write_input(tmp_path)
    stub = _patch(monkeypatch, _T2)
    report = horizon_sweep.sweep_operating_horizons(
        path, [2, 3, 5, 6, 8], stop_on_gradient=True
    )
    rows = report["cases"]
    assert report["evaluated_operating_horizons"] == [2, 3, 5, 6]
    assert 8 not in [call["H2"] for call in stub.calls]
    assert report["stopping_rule"]["reason"] == "cost_increase"
    assert report["stopping_rule"]["stopped_early"] is True
    assert report["complete"] is True
    assert rows[1]["relative_cost_gradient_per_H2"] == pytest.approx(-0.1)
    assert rows[2]["relative_cost_gradient_per_H2"] == pytest.approx(-0.0277777777777778)
    assert rows[3]["relative_cost_gradient_per_H2"] == pytest.approx(0.11764705882352938)
    assert rows[3]["gradient_classification"] == "increase"


def test_stop_needs_qualified_gaps(tmp_path, monkeypatch):
    path = _write_input(tmp_path)
    _patch(monkeypatch, _T2)
    report = horizon_sweep.sweep_operating_horizons(
        path, [2, 3, 5, 6, 8], stop_on_gradient=True, maximum_mip_gap_for_stopping=0.05
    )
    # the case H2 = 6 has the gap 0.08, above the threshold 0.05
    assert report["cases"][3]["gradient_stopping_eligible"] is False
    assert report["evaluated_operating_horizons"] == [2, 3, 5, 6, 8]
    assert report["stopping_rule"]["reason"] == "candidate_range_exhausted"
    assert report["stopping_rule"]["maximum_mip_gap_for_stopping"] == 0.05


@pytest.mark.parametrize(
    "minimum_cases, flat_gradients_required, evaluated",
    [(3, 2, [2, 3, 4]), (4, 2, [2, 3, 4, 5]), (3, 3, [2, 3, 4, 5])],
)
def test_stop_flat_gradient(
    tmp_path, monkeypatch, minimum_cases, flat_gradients_required, evaluated
):
    path = _write_input(tmp_path)
    _patch(monkeypatch, _T3)
    report = horizon_sweep.sweep_operating_horizons(
        path, [2, 3, 4, 5, 6], stop_on_gradient=True, minimum_cases=minimum_cases,
        flat_gradients_required=flat_gradients_required,
    )
    assert report["evaluated_operating_horizons"] == evaluated
    assert report["stopping_rule"]["reason"] == "flat_gradient"
    assert report["stopping_rule"]["stopped_early"] is True


# ---------------------------------------------------------------------------
# Operating sweep: checkpoints, arguments, formats
# ---------------------------------------------------------------------------
def test_checkpoints(tmp_path, monkeypatch):
    path = _write_input(tmp_path)
    target = tmp_path / "out" / "sweep.yaml"       # the folder does not exist
    stub = _patch(monkeypatch, _T1, report_path=target)
    candidates = [2, 3, 4, 5, 6, 7]
    report = horizon_sweep.sweep_operating_horizons(path, candidates, output_path=target)

    assert stub.calls[0]["checkpoint"] is None
    for n in range(2, len(candidates) + 1):
        # before call n the file holds the checkpoint after case n - 1
        assert stub.calls[n - 1]["checkpoint"] == (False, "in_progress", candidates[: n - 1])
    loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert loaded == report
    assert loaded["complete"] is True


def test_checkpoint_after_a_stop_is_in_progress(tmp_path, monkeypatch):
    path = _write_input(tmp_path)
    target = tmp_path / "sweep.yaml"
    _patch(monkeypatch, _T2)
    writes = _count_writes(monkeypatch)
    report = horizon_sweep.sweep_operating_horizons(
        path, [2, 3, 5, 6, 8], stop_on_gradient=True, output_path=target
    )
    # four cases, four checkpoints (the last one is written after the stop), one final report
    assert writes == [False, False, False, False, True]
    assert report["stopping_rule"]["reason"] == "cost_increase"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"operating_horizons": []}, "operating_horizons must contain at least one value."),
        ({"operating_horizons": [2, 2]}, "operating_horizons must not contain duplicates."),
        ({"operating_horizons": [0]},
         "operating_horizons must contain positive integers (got 0)."),
        ({"operating_horizons": [2.5]},
         "operating_horizons must contain positive integers (got 2.5)."),
        ({"operating_horizons": [True]},
         "operating_horizons must contain positive integers (got True)."),
        ({"gradient_tolerance": -1}, "gradient_tolerance must be a finite number >= 0."),
        ({"gradient_tolerance": float("inf")},
         "gradient_tolerance must be a finite number >= 0."),
        ({"flat_gradients_required": 0}, "flat_gradients_required must be an integer >= 1."),
        ({"minimum_cases": 1}, "minimum_cases must be an integer >= 2."),
        ({"maximum_mip_gap_for_stopping": -0.1},
         "maximum_mip_gap_for_stopping must be None or a finite number >= 0."),
        ({"warm_start_mode": "hints"},
         "warm_start_mode must be one of ('start', 'hint', 'both') (got 'hints')."),
        ({"n_workers": 0}, "n_workers must be an integer >= 1."),
        ({"n_workers": 2, "warm_start": True},
         "warm_start=True needs n_workers=1: a warm start uses the preceding case."),
        ({"n_workers": 2, "stop_on_gradient": True},
         "stop_on_gradient=True needs n_workers=1: the early stop needs the cases in order."),
        ({"objective_mode": "average"},
         "objective_mode must be one of ('total', 'operating_average', 'evaluation_total') "
         "(got 'average')."),
        ({"objective_mode": "evaluation_total"},
         "objective_mode 'evaluation_total' needs evaluation_horizon."),
        ({"evaluation_horizon": 2},
         "evaluation_horizon must be an integer > H1 = 2 (got 2)."),
        ({"output_path": "r.json"}, "output_path must end in .yaml or .yml."),
    ],
)
def test_argument_errors(tmp_path, monkeypatch, kwargs, message):
    path = _write_input(tmp_path)
    stub = _patch(monkeypatch, _T1)
    kwargs = dict(kwargs)
    horizons = kwargs.pop("operating_horizons", [2, 3])
    with pytest.raises(ValueError, match=re.escape(message)):
        horizon_sweep.sweep_operating_horizons(path, horizons, **kwargs)
    assert stub.calls == []


def test_input_errors(tmp_path, monkeypatch):
    stub = _patch(monkeypatch, _T1)
    path = _write_input(tmp_path, H=[1, 2, 3])
    with pytest.raises(ValueError, match="a pair"):
        horizon_sweep.sweep_operating_horizons(path, [2, 3])
    listing = tmp_path / "list.yaml"
    listing.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        horizon_sweep.sweep_operating_horizons(listing, [2, 3])
    assert stub.calls == []


def test_case_input_and_formats(tmp_path, monkeypatch):
    path = _write_input(tmp_path, suffix=".json")
    stub = _patch(monkeypatch, _T1)
    report = horizon_sweep.sweep_operating_horizons(
        path, [3, 4], objective_mode="evaluation_total", evaluation_horizon=20
    )
    assert [call["H"] for call in stub.calls] == [[2, 3], [2, 4]]
    assert all(call["objective_mode"] == "evaluation_total" for call in stub.calls)
    assert all(call["evaluation_horizon"] == 20 for call in stub.calls)
    assert report["fixed_dimensions"] == {"F": 3, "M": 1, "L": 1}
    assert report["cases"][0]["evaluation_horizon"] == 20
    assert report["objective"].startswith("minimize J_initialization + (evaluation_horizon")


def test_parallel_cases(tmp_path, monkeypatch):
    path = _write_input(tmp_path)
    target = tmp_path / "sweep.yaml"
    table = {key: _T1[key] for key in (2, 3, 4)}
    stub = _patch(monkeypatch, table, delay={2: 0.3})
    monkeypatch.setattr(
        horizon_sweep, "ProcessPoolExecutor",
        lambda max_workers, mp_context=None: ThreadPoolExecutor(max_workers=max_workers),
    )
    writes = _count_writes(monkeypatch)
    report = horizon_sweep.sweep_operating_horizons(
        path, [2, 3, 4], n_workers=3, output_path=target
    )
    rows = report["cases"]

    assert [row["H2"] for row in rows] == [2, 3, 4]
    assert [row["gradient_classification"] for row in rows] == [
        "unavailable", "decrease", "decrease",
    ]
    assert [row["gradient_stopping_eligible"] for row in rows] == [False, True, True]
    assert rows[1]["relative_cost_gradient_per_H2"] == pytest.approx(-0.1)
    assert rows[2]["relative_cost_gradient_per_H2"] == pytest.approx(-0.11111111111111108)
    # the case H2 = 2 is the slowest: it completes last
    assert report["last_completed_H2"] == 2
    assert report["n_workers"] == 3
    assert report["warm_start"] == {"enabled": False, "mode": None}
    assert all(call["source_H2"] is None for call in stub.calls)
    assert writes == [False, False, False, True]


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("evaluation_horizon", [None, 10])
def test_grid(tmp_path, monkeypatch, evaluation_horizon):
    path = _write_input(tmp_path)
    stub = _patch(monkeypatch, _GRID, j_init=_J_INIT)
    report = horizon_sweep.sweep_horizon_grid(
        path, [3, 2, 2], [4, 2], evaluation_horizon=evaluation_horizon
    )
    mode = "operating_average" if evaluation_horizon is None else "evaluation_total"

    assert report["planned_transitory_horizons"] == [2, 3]
    assert report["planned_operating_horizons"] == [2, 4]
    assert [(row["H1"], row["H2"]) for row in report["cases"]] == [(2, 2), (2, 4), (3, 2), (3, 4)]
    assert [call["source_H2"] for call in stub.calls] == [None, 2, None, 2]
    assert all(call["objective_mode"] == mode for call in stub.calls)
    assert all(call["evaluation_horizon"] == evaluation_horizon for call in stub.calls)
    assert report["evaluation_horizon"] == evaluation_horizon
    assert report["warm_start_enabled"] is True
    assert report["complete"] is True
    assert report["varied_parameters"] == ["H1", "H2"]
    assert report["fixed_dimensions"] == {"F": 3, "M": 1, "L": 1}
    assert report["mip_certificate_metric"] == "objective"
    assert [sub["H1"] for sub in report["H1_sweeps"]] == [2, 3]
    assert all(sub["input"] == str(path) for sub in report["H1_sweeps"])

    best_keys = [
        "H1", "H2", "T", "status", "mip_gap", "objective", "J_initialization",
        "J_op_average", "peak_damage", "damage_penalty", "operating_objective",
        "projected_evaluation_cost", "evaluation_objective",
    ]
    best_proven, best_feasible = report["best_proven"], report["best_feasible"]
    assert list(best_proven) == best_keys and list(best_feasible) == best_keys
    if evaluation_horizon is None:
        assert report["comparison_metric"] == "J_op_average"
        # a tie between (2, 4) and (3, 4) goes to the smaller H1
        assert (best_proven["H1"], best_proven["H2"]) == (2, 4)
        assert (best_feasible["H1"], best_feasible["H2"]) == (3, 2)
        assert best_feasible["status"] == "time_limit"
    else:
        assert report["comparison_metric"] == "projected_evaluation_cost"
        assert (best_proven["H1"], best_proven["H2"]) == (3, 4)
        assert best_proven["projected_evaluation_cost"] == pytest.approx(10.3)
        assert (best_feasible["H1"], best_feasible["H2"]) == (3, 2)
        assert best_feasible["projected_evaluation_cost"] == pytest.approx(9.95)


def test_grid_checkpoints_and_errors(tmp_path, monkeypatch):
    path = _write_input(tmp_path)
    target = tmp_path / "grid" / "grid.yaml"
    stub = _patch(monkeypatch, _GRID, j_init=_J_INIT)
    writes = _count_writes(monkeypatch)
    report = horizon_sweep.sweep_horizon_grid(path, [2, 3], [2, 4], output_path=target)
    # four cases, four checkpoints, one final report
    assert writes == [False, False, False, False, True]
    loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert loaded == report
    assert len(stub.calls) == 4

    stub.calls.clear()
    with pytest.raises(ValueError, match="sets objective_mode itself"):
        horizon_sweep.sweep_horizon_grid(path, [2], [2, 4], objective_mode="total")
    with pytest.raises(ValueError, match="exceeds every H1 candidate"):
        horizon_sweep.sweep_horizon_grid(path, [2, 3], [2, 4], evaluation_horizon=3)
    with pytest.raises(ValueError, match="warm_start=True needs n_workers=1"):
        horizon_sweep.sweep_horizon_grid(path, [2], [2, 4], n_workers=2)
    with pytest.raises(TypeError):
        horizon_sweep.sweep_horizon_grid(path, [2], [2, 4], stop_on_gradients=True)
    assert stub.calls == []


def test_grid_passes_the_sweep_options(tmp_path, monkeypatch):
    path = _write_input(tmp_path)
    stub = _patch(monkeypatch, _T2)
    report = horizon_sweep.sweep_horizon_grid(
        path, [2, 3], [2, 3, 5, 6, 8], stop_on_gradient=True, warm_start_mode="hint"
    )
    # the early stop and the warm-start mode act within each H1
    assert [sub["evaluated_operating_horizons"] for sub in report["H1_sweeps"]] == [
        [2, 3, 5, 6], [2, 3, 5, 6],
    ]
    assert all(sub["stopping_rule"]["reason"] == "cost_increase" for sub in report["H1_sweeps"])
    assert len(report["cases"]) == 8 and len(stub.calls) == 8
    assert all(call["mode"] == "hint" for call in stub.calls)
    assert [call["source_H2"] for call in stub.calls] == [None, 2, 3, 5, None, 2, 3, 5]


def test_grid_with_parallel_cases(tmp_path, monkeypatch):
    path = _write_input(tmp_path)
    stub = _patch(monkeypatch, _GRID, j_init=_J_INIT)
    monkeypatch.setattr(
        horizon_sweep, "ProcessPoolExecutor",
        lambda max_workers, mp_context=None: ThreadPoolExecutor(max_workers=max_workers),
    )
    report = horizon_sweep.sweep_horizon_grid(
        path, [2, 3], [2, 4], warm_start=False, n_workers=2
    )
    assert [(row["H1"], row["H2"]) for row in report["cases"]] == [(2, 2), (2, 4), (3, 2), (3, 4)]
    assert all(call["source_H2"] is None for call in stub.calls)
    assert [sub["n_workers"] for sub in report["H1_sweeps"]] == [2, 2]
    assert report["warm_start_enabled"] is False
