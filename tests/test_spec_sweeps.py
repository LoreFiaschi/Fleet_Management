"""Horizon sweeps against the spec (spec.tex, section "Horizon Sweeps", lines 1887-2106).

Every expected value is derived by hand from the spec, never from running the code.

Most tests replace ``fleet_management.horizon_sweep.solve_mapping`` by a fake (spec 1949: each
case is solved with ``solve_mapping``). The fake reads ``H`` from the mapping it receives,
records the call, and returns a hand-built result dictionary (spec 1716-1753, 1788-1812) built
from a table of ``J_op_average`` per ``H2`` written in each test. No Gurobi is needed for them.
The few real solves skip when ``gurobipy`` is absent (``importorskip`` first in the test).
"""
import copy
import logging
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

import fleet_management.horizon_sweep as hs

DATA = Path(__file__).parent / "data" / "tools" / "sweeps"
SMALL = DATA / "gaussian_small.yaml"          # F = 3, M = 1, L = 1, H = [2, 3]
SCALAR_H = DATA / "gaussian_scalar_H.yaml"    # H = 2, objective_mode: total
H_THREE = DATA / "gaussian_h_three.yaml"
NOT_MAPPING = DATA / "not_a_mapping.yaml"
REAL = DATA / "gaussian_real.yaml"            # H = [1, 2]
PROFILE2 = DATA / "gaussian_profile_len2.yaml"

EPS_MACH = np.finfo(float).eps

INFEASIBLE_STATUSES = ("infeasible", "unbounded", "inf_or_unbounded")

# Spec 2005-2023: keys of the sweep report.
REPORT_KEYS = {
    "input", "complete", "last_completed_H2", "objective", "varied_parameter",
    "fixed_dimensions", "H1", "planned_operating_horizons", "evaluated_operating_horizons",
    "warm_start", "n_workers", "stopping_rule", "cases", "formulation_growth",
    "best_proven_H2", "best_feasible_H2", "best_proven_J_op_average",
    "best_feasible_J_op_average", "best_feasible_status",
}
# Spec 2026-2040: keys of a case row (the scalar keys of the cost report are those of 1744-1747).
COST_SCALAR_KEYS = (
    "J_initialization", "J_op", "J_op_average", "peak_damage", "damage_penalty",
    "operating_objective", "J_total", "projected_evaluation_cost", "evaluation_objective",
)
ROW_KEYS = {
    "H1", "H2", "T", "dimensions", "status", "objective_mode", "evaluation_horizon",
    "objective", "warm_start", "objective_bound", "mip_gap", "absolute_mip_gap",
    "requested_mip_gap", "requested_time_limit_seconds", "optimizer_seconds",
    "solutions_found", "formulation", "timing", "cost_gradient_per_H2",
    "relative_cost_gradient_per_H2", "gradient_classification", "gradient_stopping_eligible",
} | set(COST_SCALAR_KEYS)
GRADIENT_KEYS = {
    "cost_gradient_per_H2", "relative_cost_gradient_per_H2", "gradient_classification",
    "gradient_stopping_eligible",
}
# Spec 1794-1796: the size entries of ``performance`` ("formulation", from variables to nonzeros).
FORMULATION_KEYS = (
    "variables", "continuous_variables", "integer_variables", "binary_variables",
    "linear_constraints", "general_constraints", "indicator_constraints",
    "quadratic_constraints", "nonzeros",
)
# Spec 1804-1807 and 1800-1801: wall-clock entries and the work entries of ``timing``.
TIMING_KEYS = (
    "model_construction_seconds", "optimizer_call_seconds", "solution_extraction_seconds",
    "backend_wall_seconds", "branch_and_bound_nodes", "simplex_iterations", "work_units",
)


def close(a, b, rel=1e-9, abs_=1e-12):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


# ---------------------------------------------------------------------------
# The fake solve
# ---------------------------------------------------------------------------

def _count(name, T):
    """Hand-made size entries: a linear function of the number of days T = H1 + H2."""
    slope_and_offset = {
        "variables": (10, 7), "continuous_variables": (4, 0), "integer_variables": (0, 0),
        "binary_variables": (6, 7), "linear_constraints": (20, 3), "general_constraints": (0, 0),
        "indicator_constraints": (0, 0), "quadratic_constraints": (2, 0), "nonzeros": (55, 11),
    }[name]
    return slope_and_offset[0] * T + slope_and_offset[1]


def make_result(H1, H2, jbar, *, status="optimal", gap=0.0, req_gap=0.1, jinit=2.0,
                bound_delta=0.0, E=None, mode="operating_average", solution=True):
    """A hand-built result dictionary with the keys that the sweep reads.

    J_op = jbar * H2; the damage penalty is 0.5 (C_D u); the objective is the key of the mode
    (spec 1778-1783); P_E = J_init + (E - H1) jbar (Eq. d_projected, spec 1900).
    """
    F, M, L = 3, 1, 1
    T = H1 + H2
    performance = {
        "gurobi_version": "13.0.0",
        **{key: _count(key, T) for key in FORMULATION_KEYS},
        "solutions_found": 1 if solution else 0,
        "gurobi_runtime_seconds": 0.25,
        "branch_and_bound_nodes": 3, "simplex_iterations": 17, "barrier_iterations": 0,
        "work_units": 0.5,
        "model_construction_seconds": 0.1, "optimizer_call_seconds": 0.3,
        "solution_extraction_seconds": 0.05, "backend_wall_seconds": 0.5,
        "requested_time_limit_seconds": 30.0, "requested_mip_gap": req_gap,
        "threads": 1, "seed": 0,
    }
    result = {
        "status": status, "F": F, "M": M, "L": L, "H1": H1, "H2": H2, "T": T,
        "H": [H1, H2], "objective_mode": mode, "evaluation_horizon": E,
        "performance": performance,
        "optimization_progress": [],
        "warm_start": {"applied": False, "mode": None, "tag": H2},
        "relaxation_warm_start": {"status": "disabled"},
    }
    if not solution:
        result.update({
            "objective": None, "mip_gap": None, "bound": None,
            "x": None, "m": None, "r": None, "idle": None, "mu": None, "v": None, "z": None,
            "step_costs": None, "u": None, "component_costs": None,
        })
        for key in COST_SCALAR_KEYS:
            result[key] = None
        return result
    J_op = jbar * H2
    penalty = 0.5
    operating_objective = jbar + penalty
    J_total = jinit + J_op + penalty
    projected = None if E is None else jinit + (E - H1) * jbar
    evaluation_objective = None if E is None else projected + penalty
    objective = {"operating_average": operating_objective, "total": J_total,
                 "evaluation_total": evaluation_objective}[mode]
    result.update({
        "objective": objective, "mip_gap": gap,
        "bound": None if bound_delta is None else objective - bound_delta,
        "x": np.zeros((F, M + 1, T)), "m": np.zeros((F, L, T)), "r": np.zeros((F, L, T)),
        "idle": np.zeros((F, L, T)), "mu": np.zeros((F, L, T)), "v": np.zeros((F, L, T)),
        "z": np.zeros((F, L, T)), "step_costs": np.zeros(T), "u": penalty,
        "component_costs": {"C_M": 1.0, "C_D": 1.0},
        "J_initialization": jinit, "J_op": J_op, "J_op_average": jbar,
        "peak_damage": penalty, "damage_penalty": penalty,
        "operating_objective": operating_objective, "J_total": J_total,
        "projected_evaluation_cost": projected, "evaluation_objective": evaluation_objective,
    })
    return result


class Fake:
    """Stands for ``solve_mapping``: same signature, table of results, records every call.

    ``table`` maps ``(H1, H2)`` or ``H2`` to a dict of ``make_result`` options (``jbar`` ...), to
    the string ``"infeasible"`` or ``"nosolution"``, or to an exception that the call raises.
    ``hook(fake, H1, H2)`` runs at the start of each call (before the result is built).
    """

    def __init__(self, table, hook=None):
        self.table = table
        self.hook = hook
        self.calls = []
        self.results = {}

    def __call__(self, data, results_path=None, *, warm_start=None, warm_start_mode="start"):
        H1, H2 = list(data["H"])
        mode = data.get("objective_mode", "total")
        E = data.get("evaluation_horizon")
        self.calls.append({
            "data": copy.deepcopy(data), "results_path": results_path, "H1": H1, "H2": H2,
            "warm_start": warm_start, "warm_start_mode": warm_start_mode,
        })
        if self.hook is not None:
            self.hook(self, H1, H2)
        spec = self.table[(H1, H2)] if (H1, H2) in self.table else self.table[H2]
        if isinstance(spec, Exception):
            raise spec
        if spec == "infeasible":
            result = make_result(H1, H2, 0.0, status="infeasible", solution=False, E=E, mode=mode)
        elif spec == "nosolution":
            result = make_result(H1, H2, 0.0, status="time_limit", solution=False, E=E, mode=mode)
        elif isinstance(spec, dict):
            options = dict(spec)
            jinit = options.pop("jinit", 2.0)
            if callable(jinit):
                jinit = jinit(H1, H2)
            result = make_result(H1, H2, E=E, mode=mode, jinit=jinit, **options)
        else:
            result = make_result(H1, H2, spec, E=E, mode=mode)
        self.results[(H1, H2)] = result
        return result


def install(monkeypatch, table, hook=None):
    fake = Fake(table, hook)
    monkeypatch.setattr(hs, "solve_mapping", fake)
    return fake


def sweep(monkeypatch, table, horizons, path=SMALL, hook=None, **options):
    fake = install(monkeypatch, table, hook)
    report = hs.sweep_operating_horizons(path, horizons, **options)
    return report, fake


def uniform(horizons, jbar):
    return {H2: jbar for H2 in horizons}


def classes(report):
    return [row["gradient_classification"] for row in report["cases"]]


def eligible(report):
    return [row["gradient_stopping_eligible"] for row in report["cases"]]


def evaluated(report):
    return list(report["evaluated_operating_horizons"])


def is_error(row):
    return isinstance(row["status"], str) and row["status"].startswith("error: ")


def is_feasible(row):
    """Spec 1904: a solution (here: a numeric J_op_average) and a status that is not infeasible."""
    return row["J_op_average"] is not None and row["status"] not in INFEASIBLE_STATUSES


def load_report(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def plain(report):
    return yaml.safe_load(yaml.safe_dump(report))


def check_gradient_annotation(rows, tolerance):
    """Eq. d_gradient (spec 1966) and the classes (spec 1972-1978) recomputed from the rows."""
    for n, row in enumerate(rows):
        if n == 0 or not (is_feasible(rows[n - 1]) and is_feasible(row)):
            assert row["gradient_classification"] == "unavailable"
            assert row["gradient_stopping_eligible"] is False
            continue
        prev = rows[n - 1]
        g = (row["J_op_average"] - prev["J_op_average"]) / (
            (row["H2"] - prev["H2"]) * max(abs(prev["J_op_average"]), EPS_MACH))
        assert close(row["relative_cost_gradient_per_H2"], g, rel=1e-6, abs_=1e-12)
        expected = "increase" if g > tolerance else "decrease" if g < -tolerance else "flat"
        assert row["gradient_classification"] == expected


# ---------------------------------------------------------------------------
# Case inputs (spec 1929, 1946-1949)
# ---------------------------------------------------------------------------

def test_case_inputs_from_list_H(monkeypatch):
    """Spec 1929, 1948-1949: H = [H1, H2], H1 = first entry, second entry ignored; increasing order."""
    report, fake = sweep(monkeypatch, uniform([3, 5], 4.0), [5, 3])
    stored = yaml.safe_load(SMALL.read_text(encoding="utf-8"))
    assert [list(c["data"]["H"]) for c in fake.calls] == [[2, 3], [2, 5]]   # input H = [2, 3]
    for call in fake.calls:
        assert call["results_path"] is None                                  # spec 1949
        for key in ("F", "M", "L", "model", "tau", "mu", "v", "C_M"):
            assert call["data"][key] == stored[key]
        assert call["data"]["objective_mode"] == "operating_average"
        assert call["data"].get("evaluation_horizon") is None
    assert report["H1"] == 2
    assert evaluated(report) == [3, 5]


def test_case_inputs_from_scalar_H_and_objective_mode(monkeypatch):
    """Spec 1929, 1940-1941, 1948: scalar H gives H1 = H; objective_mode and E are written in."""
    report, fake = sweep(monkeypatch, uniform([2, 3], 4.0), [2, 3], path=SCALAR_H)
    assert [list(c["data"]["H"]) for c in fake.calls] == [[2, 2], [2, 3]]
    # the stored objective_mode "total" is replaced
    assert all(c["data"]["objective_mode"] == "operating_average" for c in fake.calls)
    assert report["H1"] == 2

    report, fake = sweep(monkeypatch, uniform([2, 3], 4.0), [2, 3], path=SCALAR_H,
                         objective_mode="evaluation_total", evaluation_horizon=10)
    assert all(c["data"]["objective_mode"] == "evaluation_total" for c in fake.calls)
    assert all(c["data"]["evaluation_horizon"] == 10 for c in fake.calls)
    assert all(c["data"]["objective_mode"] == "evaluation_total" for c in fake.calls)


def test_unsorted_candidates_are_evaluated_in_increasing_order(monkeypatch):
    """Spec 1930: 'evaluated in increasing order'; spec 2018: the rows are by H2."""
    report, fake = sweep(monkeypatch, uniform([2, 3, 4], 4.0), [4, 2, 3])
    assert [c["H2"] for c in fake.calls] == [2, 3, 4]
    assert [row["H2"] for row in report["cases"]] == [2, 3, 4]
    assert list(report["planned_operating_horizons"]) == [2, 3, 4]


# ---------------------------------------------------------------------------
# Arguments (spec 1928-1942, 1957)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizons, options", [
    ([2, 2], {}),                                    # duplicates, spec 1930
    ([0, 2], {}),                                    # zero, spec 1930
    ([-1, 2], {}),
    ([2.5], {}),                                     # not an integer
    ([], {}),                                        # at least one
    ([2, 3], {"gradient_tolerance": -1.0}),          # finite and >= 0, spec 1933
    ([2, 3], {"gradient_tolerance": float("nan")}),
    ([2, 3], {"gradient_tolerance": float("inf")}),
    ([2, 3], {"flat_gradients_required": 0}),        # >= 1, spec 1934
    ([2, 3], {"minimum_cases": 1}),                  # >= 2, spec 1935
    ([2, 3], {"maximum_mip_gap_for_stopping": -1.0}),    # None or finite >= 0, spec 1936
    ([2, 3], {"maximum_mip_gap_for_stopping": float("inf")}),
    ([2, 3], {"warm_start": True, "n_workers": 2}),      # spec 1957
    ([2, 3], {"stop_on_gradient": True, "n_workers": 2}),
    ([2, 3], {"warm_start": True, "warm_start_mode": "bogus"}),
    ([2, 3], {"objective_mode": "bogus"}),
    ([2, 3], {"objective_mode": "evaluation_total"}),    # needs E, spec 1940
    ([2, 3], {"objective_mode": "evaluation_total", "evaluation_horizon": 2}),   # E <= H1 = 2
    ([2, 3], {"objective_mode": "evaluation_total", "evaluation_horizon": 1}),
    ([2, 3], {"evaluation_horizon": 2}),             # if given, E > H1, spec 1941
    ([2, 3], {"evaluation_horizon": 3.5}),           # an integer
])
def test_invalid_arguments_raise_before_the_first_case(monkeypatch, horizons, options):
    """Spec 1957: ValueError before the first case, never recorded as a case error."""
    fake = install(monkeypatch, uniform(range(1, 10), 4.0))
    with pytest.raises(ValueError):
        hs.sweep_operating_horizons(SMALL, horizons, **options)
    assert fake.calls == []


def test_list_H_of_length_three_raises(monkeypatch):
    """Spec 1957: a list H whose length is not two."""
    fake = install(monkeypatch, uniform(range(1, 10), 4.0))
    with pytest.raises(ValueError):
        hs.sweep_operating_horizons(H_THREE, [2, 3])
    assert fake.calls == []


def test_data_that_is_not_a_mapping_raises_type_error(monkeypatch):
    """Spec 1946, 1956."""
    fake = install(monkeypatch, uniform(range(1, 10), 4.0))
    with pytest.raises(TypeError):
        hs.sweep_operating_horizons(NOT_MAPPING, [2, 3])
    assert fake.calls == []


def test_output_path_with_another_extension_raises(monkeypatch, tmp_path):
    """Spec 1931: the report file is .yaml or .yml."""
    fake = install(monkeypatch, uniform(range(1, 10), 4.0))
    with pytest.raises(ValueError):
        hs.sweep_operating_horizons(SMALL, [2, 3], output_path=tmp_path / "report.txt")
    assert fake.calls == []


def test_parallel_option_is_refused_with_warm_start_and_stop(monkeypatch):
    """Spec 2054, 1957, 2076: n_workers > 1 refused with warm_start and with stop_on_gradient."""
    fake = install(monkeypatch, uniform(range(1, 10), 4.0))
    with pytest.raises(ValueError):
        hs.sweep_operating_horizons(SMALL, [2, 3], n_workers=2, warm_start=True)
    with pytest.raises(ValueError):
        hs.sweep_operating_horizons(SMALL, [2, 3], n_workers=2, stop_on_gradient=True)
    # grid: warm_start defaults to True, so n_workers > 1 needs warm_start=False (spec 2076)
    with pytest.raises(ValueError):
        hs.sweep_horizon_grid(SMALL, [1, 2], [2, 3], n_workers=2)
    with pytest.raises(ValueError):
        hs.sweep_horizon_grid(SMALL, [1, 2], [2, 3], n_workers=2, warm_start=False,
                              stop_on_gradient=True)
    assert fake.calls == []


class _Reached(Exception):
    """Raised by the fake executor on entry: the test needs only its keyword arguments."""


def test_parallel_workers_start_with_spawn(monkeypatch):
    """The parent may hold Gurobi threads, so the worker processes start with 'spawn', not 'fork'."""
    recorded = {}

    class FakeExecutor:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def __enter__(self):
            raise _Reached

        def __exit__(self, *exc_info):
            return False

    install(monkeypatch, uniform(range(1, 10), 4.0))
    monkeypatch.setattr(hs, "ProcessPoolExecutor", FakeExecutor)
    with pytest.raises(_Reached):
        hs.sweep_operating_horizons(SMALL, [2, 3], n_workers=2)
    assert recorded["max_workers"] == 2
    assert recorded["mp_context"].get_start_method() == "spawn"


# ---------------------------------------------------------------------------
# Gradient and classes (Eq. d_gradient, spec 1964-1985)
# ---------------------------------------------------------------------------

# Candidates H2 = 2, 4, 5, 9, 10 and Jbar = 10, 8, 8.004, 8.0, 9.0 (all optimal, gap 0).
# n=2: (8 - 10) / ((4 - 2) * 10) = -0.1                  -> decrease (t = 1e-3)
# n=3: (8.004 - 8) / ((5 - 4) * 8) = 5e-4                -> flat
# n=4: (8.0 - 8.004) / ((9 - 5) * 8.004) = -1.2494e-4    -> flat
# n=5: (9.0 - 8.0) / ((10 - 9) * 8.0) = 0.125            -> increase
GRADIENT_TABLE = {2: 10.0, 4: 8.0, 5: 8.004, 9: 8.0, 10: 9.0}
GRADIENT_EXPECTED = [None, -0.1, 0.004 / (1 * 8.0), -0.004 / (4 * 8.004), 1.0 / (1 * 8.0)]
GRADIENT_CLASSES = ["unavailable", "decrease", "flat", "flat", "increase"]


def test_gradient_values_classes_and_eligibility_without_stop(monkeypatch):
    """Eq. d_gradient (spec 1966), classes (spec 1972-1978), annotation (spec 1984)."""
    report, fake = sweep(monkeypatch, GRADIENT_TABLE, [2, 4, 5, 9, 10])
    assert len(fake.calls) == 5
    assert classes(report) == GRADIENT_CLASSES
    assert eligible(report) == [False, True, True, True, True]
    for row, expected in zip(report["cases"][1:], GRADIENT_EXPECTED[1:]):
        assert close(row["relative_cost_gradient_per_H2"], expected, rel=1e-9)
    check_gradient_annotation(report["cases"], 1e-3)


def test_gradient_annotation_is_the_same_with_stop_on_gradient(monkeypatch):
    """Spec 1984: the annotation is also written if stop_on_gradient is false; the same values
    appear for the cases that a sweep with stop_on_gradient=True evaluates.

    With the stop rule: n=3 flat, n=4 flat, minimum_cases = 3, flat_gradients_required = 2:
    the sweep stops after n=4 (H2 = 9), reason flat_gradient (spec 1988-1997).
    """
    report, fake = sweep(monkeypatch, GRADIENT_TABLE, [2, 4, 5, 9, 10], stop_on_gradient=True)
    assert evaluated(report) == [2, 4, 5, 9]
    assert classes(report) == GRADIENT_CLASSES[:4]
    assert eligible(report) == [False, True, True, True]
    for row, expected in zip(report["cases"][1:], GRADIENT_EXPECTED[1:4]):
        assert close(row["relative_cost_gradient_per_H2"], expected, rel=1e-9)
    assert report["stopping_rule"]["reason"] == "flat_gradient"


def test_gradient_class_boundaries_are_flat(monkeypatch):
    """Spec 1974-1976: flat is -t <= g <= t, so g = +-t is flat.

    t = 0.125. Jbar = 4.0, 4.5, 5.0625, 6.0 on H2 = 2..5:
      n=2: 0.5 / 4.0 = 0.125 (flat); n=3: 0.5625 / 4.5 = 0.125 (flat);
      n=4: 0.9375 / 5.0625 = 0.185 (increase).
    Jbar = 4.0, 3.5, 3.0625, 2.0: n=2: -0.5 / 4.0 = -0.125 (flat);
      n=3: -0.4375 / 3.5 = -0.125 (flat); n=4: -1.0625 / 3.0625 = -0.347 (decrease).
    """
    report, _ = sweep(monkeypatch, {2: 4.0, 3: 4.5, 4: 5.0625, 5: 6.0}, [2, 3, 4, 5],
                      gradient_tolerance=0.125)
    assert classes(report) == ["unavailable", "flat", "flat", "increase"]
    report, _ = sweep(monkeypatch, {2: 4.0, 3: 3.5, 4: 3.0625, 5: 2.0}, [2, 3, 4, 5],
                      gradient_tolerance=0.125)
    assert classes(report) == ["unavailable", "flat", "flat", "decrease"]


def test_gradient_denominator_uses_machine_precision_and_absolute_value(monkeypatch):
    """Eq. d_gradient (spec 1966): max{|Jbar_{n-1}|, eps_mach} in the denominator.

    Jbar = 0 then 1: g = 1 / (1 * eps_mach), about 4.5e15 (increase); 0 then 0: g = 0 (flat).
    Jbar = -4 then -3: g = 1 / (1 * 4) = 0.25 > 1e-3 (increase).
    """
    report, _ = sweep(monkeypatch, {2: 0.0, 3: 1.0}, [2, 3])
    assert classes(report) == ["unavailable", "increase"]
    assert close(report["cases"][1]["relative_cost_gradient_per_H2"], 1.0 / EPS_MACH, rel=1e-6)
    report, _ = sweep(monkeypatch, {2: 0.0, 3: 0.0}, [2, 3])
    assert classes(report) == ["unavailable", "flat"]
    report, _ = sweep(monkeypatch, {2: -4.0, 3: -3.0}, [2, 3])
    assert classes(report) == ["unavailable", "increase"]
    assert close(report["cases"][1]["relative_cost_gradient_per_H2"], 0.25)


def test_unavailable_next_to_an_infeasible_case(monkeypatch):
    """Spec 1977: 'unavailable' for n = 1, or if one of the two cases is not feasible.

    H2 = 2..5: Jbar = 5, infeasible, 4, 4.
    n=1: unavailable; n=2 (infeasible): unavailable; n=3 (neighbour infeasible): unavailable;
    n=4: (4 - 4) / (1 * 4) = 0, flat; its two cases qualify, so it is eligible (spec 1983).
    """
    table = {2: 5.0, 3: "infeasible", 4: 4.0, 5: 4.0}
    report, _ = sweep(monkeypatch, table, [2, 3, 4, 5])
    assert classes(report) == ["unavailable", "unavailable", "unavailable", "flat"]
    assert eligible(report) == [False, False, False, True]
    check_gradient_annotation(report["cases"], 1e-3)


# ---------------------------------------------------------------------------
# Stop rule (spec 1987-1998)
# ---------------------------------------------------------------------------

def test_stop_cost_increase(monkeypatch):
    """Jbar = 10, 8, 9: n=2 g = -0.2 (decrease); n=3 g = 1 / 8 = 0.125 (increase), n = 3 >= 3: stop."""
    report, fake = sweep(monkeypatch, {2: 10.0, 3: 8.0, 4: 9.0, 5: 1.0, 6: 1.0, 7: 1.0},
                         [2, 3, 4, 5, 6, 7], stop_on_gradient=True)
    assert evaluated(report) == [2, 3, 4]
    assert len(fake.calls) == 3
    assert report["stopping_rule"]["reason"] == "cost_increase"
    assert report["stopping_rule"]["stopped_early"] is True
    assert report["stopping_rule"]["enabled"] is True
    assert report["complete"] is True
    assert list(report["planned_operating_horizons"]) == [2, 3, 4, 5, 6, 7]
    assert report["last_completed_H2"] == 4


def test_increase_before_minimum_cases_does_not_stop(monkeypatch):
    """Jbar = 10, 12, 12, 12: n=2 increase but n = 2 < minimum_cases = 3 (no stop);
    n=3 flat, previous class increase, so not two flat cases; n=4 flat, flat: stop (flat_gradient)."""
    report, _ = sweep(monkeypatch, {2: 10.0, 3: 12.0, 4: 12.0, 5: 12.0, 6: 99.0, 7: 99.0},
                      [2, 3, 4, 5, 6, 7], stop_on_gradient=True)
    assert classes(report) == ["unavailable", "increase", "flat", "flat"]
    assert evaluated(report) == [2, 3, 4, 5]
    assert report["stopping_rule"]["reason"] == "flat_gradient"
    assert report["stopping_rule"]["stopped_early"] is True


def test_flat_gradients_required_and_minimum_cases(monkeypatch):
    """Constant Jbar: the gradients of n = 2, 3, ... are flat (0).

    defaults (2 flat cases, minimum 3): stop at n = 3 (cases 2 and 3 flat);
    flat_gradients_required = 3: stop at n = 4 (cases 2, 3, 4);
    minimum_cases = 5: no stop before n = 5.
    """
    table = uniform(range(2, 8), 5.0)
    candidates = [2, 3, 4, 5, 6, 7]
    report, _ = sweep(monkeypatch, table, candidates, stop_on_gradient=True)
    assert evaluated(report) == [2, 3, 4]
    report, _ = sweep(monkeypatch, table, candidates, stop_on_gradient=True,
                      flat_gradients_required=3)
    assert evaluated(report) == [2, 3, 4, 5]
    report, _ = sweep(monkeypatch, table, candidates, stop_on_gradient=True, minimum_cases=5)
    assert evaluated(report) == [2, 3, 4, 5, 6]
    assert report["stopping_rule"]["reason"] == "flat_gradient"
    assert report["stopping_rule"]["minimum_cases"] == 5
    assert report["stopping_rule"]["stopped_early"] is True


def test_candidate_range_exhausted(monkeypatch):
    """Spec 1998 and 2017: the last candidate is a hard limit; complete with all cases is not early."""
    report, fake = sweep(monkeypatch, {2: 10.0, 3: 9.0, 4: 8.0}, [2, 3, 4], stop_on_gradient=True)
    assert evaluated(report) == [2, 3, 4]
    assert report["stopping_rule"]["reason"] == "candidate_range_exhausted"
    assert report["stopping_rule"]["stopped_early"] is False


def test_no_stop_without_stop_on_gradient(monkeypatch):
    """Jbar increasing 10, 12, 14, 16: classes increase, but stop_on_gradient is false."""
    report, fake = sweep(monkeypatch, {2: 10.0, 3: 12.0, 4: 14.0, 5: 16.0}, [2, 3, 4, 5])
    assert evaluated(report) == [2, 3, 4, 5]
    assert len(fake.calls) == 4
    assert classes(report) == ["unavailable", "increase", "increase", "increase"]
    assert report["stopping_rule"]["enabled"] is False
    assert report["stopping_rule"]["reason"] == "candidate_range_exhausted"
    assert report["stopping_rule"]["stopped_early"] is False


# Jbar = 10, 12, 14, 16, 18 on H2 = 2..6; the case H2 = 4 (n = 3) differs per test.
# n=2: 2 / 10 = 0.2; n=3: 2 / 12 = 0.1667; n=4: 2 / 14 = 0.143; n=5: 2 / 16 = 0.125: all increase.
def _gap_table(case3):
    return {2: 10.0, 3: 12.0, 4: dict(jbar=14.0, **case3), 5: 16.0, 6: 18.0, 7: 99.0}


def test_case_above_the_mip_gap_does_not_qualify(monkeypatch):
    """Spec 1982-1983: case n=3 has gap 0.2 above the requested gap 0.1: it does not qualify, so
    the gradients of n = 3 and n = 4 are not eligible; n = 5 (cases 4 and 5 qualify) is eligible and
    its class is increase: stop after n = 5 (H2 = 6)."""
    table = _gap_table({"gap": 0.2})
    candidates = [2, 3, 4, 5, 6, 7]
    report, _ = sweep(monkeypatch, table, candidates, stop_on_gradient=True)
    assert eligible(report) == [False, True, False, False, True]
    assert classes(report) == ["unavailable"] + ["increase"] * 4
    assert evaluated(report) == [2, 3, 4, 5, 6]
    assert report["stopping_rule"]["reason"] == "cost_increase"
    # the annotation is written also without the stop
    report, _ = sweep(monkeypatch, table, candidates)
    assert eligible(report)[:5] == [False, True, False, False, True]
    assert evaluated(report) == candidates


def test_maximum_mip_gap_for_stopping_replaces_the_requested_gap(monkeypatch):
    """Spec 1982: the threshold is maximum_mip_gap_for_stopping if it is a number.

    Case n=3 has gap 0.2: with the threshold 0.25 it qualifies, n=3 is eligible, g = 2 / 12 > t,
    n = 3 >= 3: stop after H2 = 4. The report echoes the argument (spec 2017)."""
    table = _gap_table({"gap": 0.2})
    report, _ = sweep(monkeypatch, table, [2, 3, 4, 5, 6, 7], stop_on_gradient=True,
                      maximum_mip_gap_for_stopping=0.25)
    assert evaluated(report) == [2, 3, 4]
    assert eligible(report) == [False, True, True]
    assert report["stopping_rule"]["maximum_mip_gap_for_stopping"] == 0.25
    assert report["stopping_rule"]["reason"] == "cost_increase"


def test_threshold_below_the_requested_gap(monkeypatch):
    """Case n=3 has gap 0.08. Requested gap 0.1: qualifies (stop after H2 = 4).
    maximum_mip_gap_for_stopping = 0.05: 0.08 > 0.05, no qualification: stop after n = 5."""
    table = _gap_table({"gap": 0.08})
    candidates = [2, 3, 4, 5, 6, 7]
    report, _ = sweep(monkeypatch, table, candidates, stop_on_gradient=True)
    assert evaluated(report) == [2, 3, 4]
    report, _ = sweep(monkeypatch, table, candidates, stop_on_gradient=True,
                      maximum_mip_gap_for_stopping=0.05)
    assert evaluated(report) == [2, 3, 4, 5, 6]


def test_threshold_is_the_requested_gap_of_the_case(monkeypatch):
    """Spec 1982: without the argument, the threshold is requested_mip_gap of the row of the case.
    Case n=3: gap 0.08 with requested gap 0.05: not qualified."""
    table = _gap_table({"gap": 0.08, "req_gap": 0.05})
    report, _ = sweep(monkeypatch, table, [2, 3, 4, 5, 6, 7], stop_on_gradient=True)
    assert eligible(report)[:5] == [False, True, False, False, True]
    assert evaluated(report) == [2, 3, 4, 5, 6]


def test_case_without_a_finite_gap_does_not_qualify(monkeypatch):
    """Spec 1982: 'a finite MIP gap'. Case n=3 is time_limit with a solution and gap null."""
    table = _gap_table({"status": "time_limit", "gap": None})
    report, _ = sweep(monkeypatch, table, [2, 3, 4, 5, 6, 7], stop_on_gradient=True)
    assert eligible(report)[:5] == [False, True, False, False, True]
    assert evaluated(report) == [2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Ranking (spec 1903-1910, 2020-2022)
# ---------------------------------------------------------------------------

def test_best_proven_and_best_feasible(monkeypatch):
    """H2 = 2..8:
      2 optimal 6.0 (proven); 3 time_limit 3.0 (feasible, not proven); 4 optimal 5.0 (proven);
      5 infeasible (neither); 6 optimal 5.0 (proven, ties with H2 = 4: smaller H2 wins);
      7 unbounded 0.1 with a solution (neither, spec 1904); 8 suboptimal 3.0 (feasible,
      ties with H2 = 3: smaller H2 wins).
    Best proven: H2 = 4 (5.0). Best feasible: H2 = 3 (3.0), status time_limit."""
    table = {
        2: 6.0, 3: dict(jbar=3.0, status="time_limit", gap=0.3), 4: 5.0, 5: "infeasible",
        6: 5.0, 7: dict(jbar=0.1, status="unbounded"), 8: dict(jbar=3.0, status="suboptimal"),
    }
    report, _ = sweep(monkeypatch, table, [2, 3, 4, 5, 6, 7, 8])
    assert report["best_proven_H2"] == 4
    assert close(report["best_proven_J_op_average"], 5.0)
    assert report["best_feasible_H2"] == 3
    assert close(report["best_feasible_J_op_average"], 3.0)
    assert report["best_feasible_status"] == "time_limit"


def test_ranking_without_a_feasible_case_is_null(monkeypatch):
    """Spec 2020-2022: null without such a case. One infeasible case, one without a solution."""
    report, _ = sweep(monkeypatch, {2: "infeasible", 3: "nosolution"}, [2, 3])
    for key in ("best_proven_H2", "best_feasible_H2", "best_proven_J_op_average",
                "best_feasible_J_op_average", "best_feasible_status"):
        assert report[key] is None


def test_ranking_without_a_proven_case(monkeypatch):
    """A time_limit case with a solution is feasible, not proven."""
    report, _ = sweep(monkeypatch, {2: dict(jbar=3.0, status="time_limit", gap=0.5)}, [2])
    assert report["best_proven_H2"] is None
    assert report["best_proven_J_op_average"] is None
    assert report["best_feasible_H2"] == 2
    assert report["best_feasible_status"] == "time_limit"


def test_ranking_value_is_jbar_also_with_evaluation_total(monkeypatch):
    """Spec 1906: in sweep_operating_horizons the ranking value is Jbar_op.

    H1 = 2, E = 10, J_init = 100 - 10 H2, Jbar = 3.0, 3.5, 4.0 on H2 = 2, 3, 4:
    P_E = 80 + 8 * 3 = 104; 70 + 28 = 98; 60 + 32 = 92 (so P_E would prefer H2 = 4),
    but the ranking by Jbar prefers H2 = 2."""
    table = {H2: dict(jbar=j, jinit=lambda H1, h2: 100.0 - 10.0 * h2)
             for H2, j in ((2, 3.0), (3, 3.5), (4, 4.0))}
    report, _ = sweep(monkeypatch, table, [2, 3, 4], objective_mode="evaluation_total",
                      evaluation_horizon=10)
    assert report["best_proven_H2"] == 2
    assert report["best_feasible_H2"] == 2
    assert close(report["best_proven_J_op_average"], 3.0)


# ---------------------------------------------------------------------------
# Report keys, rows, growth (spec 2003-2047)
# ---------------------------------------------------------------------------

def test_report_keys_and_values(monkeypatch):
    """Spec 2005-2023 with the defaults of 1918-1924; H1 = 2, F = 3, M = 1, L = 1."""
    report, _ = sweep(monkeypatch, {2: 5.0, 3: 4.5, 4: 4.25}, [2, 3, 4])
    assert REPORT_KEYS <= set(report)
    assert Path(report["input"]) == SMALL
    assert report["complete"] is True
    assert report["last_completed_H2"] == 4
    assert isinstance(report["objective"], str)
    assert report["varied_parameter"] == "H2"
    assert dict(report["fixed_dimensions"]) == {"F": 3, "M": 1, "L": 1}
    assert report["H1"] == 2
    assert list(report["planned_operating_horizons"]) == [2, 3, 4]
    assert evaluated(report) == [2, 3, 4]
    assert report["warm_start"]["enabled"] is False
    assert "mode" in report["warm_start"]                  # the default mode "start" is not fixed when disabled
    assert report["n_workers"] == 1
    rule = report["stopping_rule"]
    assert rule["enabled"] is False
    assert close(rule["relative_gradient_tolerance_per_H2"], 1e-3)
    assert rule["flat_gradients_required"] == 2
    assert rule["minimum_cases"] == 3
    assert rule["maximum_mip_gap_for_stopping"] is None
    assert rule["reason"] == "candidate_range_exhausted"
    assert rule["stopped_early"] is False
    assert report["best_proven_H2"] == 4
    assert close(report["best_proven_J_op_average"], 4.25)
    assert report["best_feasible_H2"] == 4
    assert report["best_feasible_status"] == "optimal"
    assert [row["H2"] for row in report["cases"]] == [2, 3, 4]


def test_report_echoes_the_stop_arguments(monkeypatch):
    """Spec 2017: the stop arguments appear under their own names."""
    report, _ = sweep(monkeypatch, uniform(range(2, 8), 5.0), [2, 3, 4], stop_on_gradient=True,
                      gradient_tolerance=0.02, flat_gradients_required=4, minimum_cases=2,
                      maximum_mip_gap_for_stopping=0.3, warm_start=True, warm_start_mode="both")
    rule = report["stopping_rule"]
    assert rule["enabled"] is True
    assert close(rule["relative_gradient_tolerance_per_H2"], 0.02)
    assert rule["flat_gradients_required"] == 4
    assert rule["minimum_cases"] == 2
    assert close(rule["maximum_mip_gap_for_stopping"], 0.3)
    assert report["warm_start"]["enabled"] is True
    assert report["warm_start"]["mode"] == "both"


def test_row_keys_and_values(monkeypatch):
    """Spec 2025-2040. Hand values from the fake: T = H1 + H2 = 2 + H2; objective = Jbar + 0.5
    (mode operating_average, spec 1781); requested gap 0.1, time limit 30, one solution."""
    report, fake = sweep(monkeypatch, {2: 5.0, 3: 4.5}, [2, 3])
    row = report["cases"][0]
    assert ROW_KEYS <= set(row)
    assert (row["H1"], row["H2"], row["T"]) == (2, 2, 4)
    assert dict(row["dimensions"]) == {"F": 3, "M": 1, "L": 1, "H1": 2, "H2": 2, "T": 4}
    assert row["status"] == "optimal"
    assert row["objective_mode"] == "operating_average"
    assert row["evaluation_horizon"] is None
    assert close(row["objective"], 5.5)
    assert close(row["J_op_average"], 5.0)
    assert close(row["J_op"], 10.0)
    assert close(row["J_initialization"], 2.0)
    assert close(row["damage_penalty"], 0.5)
    assert close(row["operating_objective"], 5.5)
    assert close(row["J_total"], 2.0 + 10.0 + 0.5)
    assert row["projected_evaluation_cost"] is None
    assert row["warm_start"]["tag"] == 2                  # the warm-start report of the result
    assert close(row["mip_gap"], 0.0)
    assert close(row["requested_mip_gap"], 0.1)
    assert close(row["requested_time_limit_seconds"], 30.0)
    assert row["solutions_found"] == 1
    assert row["optimizer_seconds"] is not None
    for key in FORMULATION_KEYS:
        assert row["formulation"][key] == _count(key, 4)
    for key in TIMING_KEYS:
        assert key in row["timing"]
    assert row["timing"]["branch_and_bound_nodes"] == 3
    assert row["timing"]["simplex_iterations"] == 17
    assert close(row["timing"]["work_units"], 0.5)
    assert row["gradient_classification"] == "unavailable"
    assert row["gradient_stopping_eligible"] is False


def test_row_with_evaluation_total(monkeypatch):
    """Spec 2029, 1747: objective_mode, E, P_E = J_init + (E - H1) Jbar = 2 + 8 * 5 = 42 (E = 10),
    objective = evaluation_objective = 42.5."""
    report, _ = sweep(monkeypatch, {2: 5.0}, [2], objective_mode="evaluation_total",
                      evaluation_horizon=10)
    row = report["cases"][0]
    assert row["objective_mode"] == "evaluation_total"
    assert row["evaluation_horizon"] == 10
    assert close(row["projected_evaluation_cost"], 42.0)
    assert close(row["evaluation_objective"], 42.5)
    assert close(row["objective"], 42.5)


def test_absolute_mip_gap(monkeypatch):
    """Spec 2033: max{0, objective - bound}; null if the objective or the bound is null.
    Objective = 5.5; bound = 5.2 (0.3), 5.7 (bound above: 0), null (null); no solution (null)."""
    table = {2: dict(jbar=5.0, bound_delta=0.3), 3: dict(jbar=5.0, bound_delta=-0.2),
             4: dict(jbar=5.0, bound_delta=None), 5: "infeasible"}
    report, _ = sweep(monkeypatch, table, [2, 3, 4, 5])
    rows = report["cases"]
    assert close(rows[0]["absolute_mip_gap"], 0.3, rel=1e-9, abs_=1e-9)
    assert close(rows[0]["objective_bound"], 5.2, rel=1e-9)
    assert close(rows[1]["absolute_mip_gap"], 0.0, abs_=1e-12)
    assert rows[2]["absolute_mip_gap"] is None
    assert rows[2]["objective_bound"] is None
    assert rows[3]["absolute_mip_gap"] is None
    assert rows[3]["objective"] is None
    assert rows[3]["status"] == "infeasible"


def test_formulation_growth(monkeypatch):
    """Spec 2019. H2 = 2, 3 (error, no counts), 4, 5 (error): rows with counts are H2 = 2 and 4,
    T = 4 and 6: added_time_steps = 2; variables 10 T + 7: 47 and 67: added 20, 10 per step;
    linear_constraints 20 T + 3: 83 and 123: added 40, 20 per step; integer_variables constant: 0, 0;
    nonzeros 55 T + 11: added 110, 55 per step."""
    table = {2: 5.0, 3: ValueError("boom"), 4: 4.0, 5: ValueError("bang")}
    report, _ = sweep(monkeypatch, table, [2, 3, 4, 5])
    growth = report["formulation_growth"]
    assert growth["from_H2"] == 2
    assert growth["to_H2"] == 4
    assert growth["added_time_steps"] == 2
    assert growth["added_variables"] == 20
    assert close(growth["variables_per_added_time_step"], 10.0)
    assert growth["added_linear_constraints"] == 40
    assert close(growth["linear_constraints_per_added_time_step"], 20.0)
    assert growth["added_integer_variables"] == 0
    assert close(growth["integer_variables_per_added_time_step"], 0.0, abs_=1e-12)
    assert growth["added_nonzeros"] == 110
    assert close(growth["nonzeros_per_added_time_step"], 55.0)
    for key in FORMULATION_KEYS:
        assert "added_" + key in growth
        assert key + "_per_added_time_step" in growth


def test_formulation_growth_with_fewer_than_two_rows(monkeypatch):
    """Spec 2019: null with fewer than two rows with counts (read as: the growth entries are null)."""
    report, _ = sweep(monkeypatch, {2: 5.0, 3: ValueError("boom")}, [2, 3])
    growth = report["formulation_growth"]
    assert growth is None or growth.get("added_variables") is None


def test_checkpoint_is_rewritten_after_every_case(monkeypatch, tmp_path):
    """Spec 2043: with output_path the report is rewritten after every case with complete = false
    and the reason in_progress; the final file is the returned report (spec 1951, 2003).
    The fake reads the file at the start of each call: after n cases it holds n rows."""
    target = tmp_path / "nested" / "folder" / "report.yaml"      # parents are created (spec 1931)
    seen = []

    def hook(fake, H1, H2):
        if len(fake.calls) == 1:
            return
        report = load_report(target)
        seen.append(report)

    report, fake = sweep(monkeypatch, {2: 5.0, 3: 4.5, 4: 4.25}, [2, 3, 4], hook=hook,
                         output_path=target)
    assert len(seen) == 2
    for n, checkpoint in enumerate(seen, start=1):
        assert checkpoint["complete"] is False
        assert checkpoint["stopping_rule"]["reason"] == "in_progress"
        assert len(checkpoint["cases"]) == n
        assert checkpoint["last_completed_H2"] == n + 1          # H2 = 2, 3
    assert report["complete"] is True
    assert target.exists()
    assert load_report(target) == plain(report)


def test_checkpoint_with_yml_extension(monkeypatch, tmp_path):
    """Spec 1931: .yml is accepted."""
    target = tmp_path / "report.yml"
    report, _ = sweep(monkeypatch, {2: 5.0}, [2], output_path=target)
    assert load_report(target) == plain(report)


def test_no_file_without_output_path(monkeypatch, tmp_path):
    """Spec 1931: None writes no file (and spec 1949: no temporary result file)."""
    monkeypatch.chdir(tmp_path)
    sweep(monkeypatch, {2: 5.0, 3: 4.0}, [2, 3])
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Case errors (spec 2044-2047)
# ---------------------------------------------------------------------------

def test_case_error_row_log_and_continuation(monkeypatch, caplog):
    """Spec 2044-2046. H2 = 2..5, the case H2 = 3 raises ValueError('boom').

    Row: status 'error: boom' and null in every entry except the sizes (H1, H2, T, dimensions;
    F, M, L from the input: 3, 1, 1). ERROR record naming H1 = 2 and H2 = 3. The sweep goes on
    (4 calls). The error row is neither feasible nor proven: best proven is H2 = 4 (Jbar 4.0 ties
    with H2 = 5, smaller wins). The neighbours of the error row have the class unavailable."""
    table = {2: 5.0, 3: ValueError("boom"), 4: 4.0, 5: 4.0}
    report, fake = sweep(monkeypatch, table, [2, 3, 4, 5])
    assert len(fake.calls) == 4
    assert evaluated(report) == [2, 3, 4, 5]
    rows = report["cases"]
    error_row = rows[1]
    assert error_row["status"] == "error: boom"
    assert (error_row["H1"], error_row["H2"], error_row["T"]) == (2, 3, 5)
    assert dict(error_row["dimensions"]) == {"F": 3, "M": 1, "L": 1, "H1": 2, "H2": 3, "T": 5}
    sizes = {"H1", "H2", "T", "dimensions", "status"} | GRADIENT_KEYS
    for key in ROW_KEYS - sizes:
        assert key in error_row and error_row[key] is None, key
    records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "2" in message and "3" in message
    assert classes(report) == ["unavailable", "unavailable", "unavailable", "flat"]
    assert report["best_proven_H2"] == 4
    assert report["best_feasible_H2"] == 4
    assert not is_feasible(error_row)


def test_case_error_at_the_first_case_does_not_stop_the_sweep(monkeypatch, caplog):
    """Spec 2046: the sweep continues after an error in the first case."""
    report, fake = sweep(monkeypatch, {2: RuntimeError("first"), 3: 4.0, 4: 3.0}, [2, 3, 4])
    assert [row["status"] for row in report["cases"]] == ["error: first", "optimal", "optimal"]
    assert classes(report) == ["unavailable", "unavailable", "decrease"]
    assert report["complete"] is True
    assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 1


def test_case_errors_in_every_case(monkeypatch, caplog):
    """Every case raises: all rows are errors, nothing is best, the report is still complete."""
    table = {2: ValueError("a"), 3: ValueError("b")}
    report, _ = sweep(monkeypatch, table, [2, 3])
    assert [row["status"] for row in report["cases"]] == ["error: a", "error: b"]
    assert report["best_proven_H2"] is None
    assert report["best_feasible_H2"] is None
    assert report["formulation_growth"] is None or report["formulation_growth"].get(
        "added_variables") is None
    assert report["complete"] is True


# ---------------------------------------------------------------------------
# Warm starts in sweeps (spec 2197-2204; the fake records the arguments)
# ---------------------------------------------------------------------------

def test_warm_start_sources_in_a_sweep(monkeypatch):
    """Spec 2201-2203, with the mode passed on (spec 1938).

    H2 = 2 optimal; 3 error; 4 infeasible; 5 time_limit without a solution; 6 time_limit with a
    solution; 7 optimal. Sources: 2: none (first case); 3, 4, 5, 6: the nearest preceding case
    with a solution, which is H2 = 2 each time; 7: H2 = 6."""
    table = {
        2: 5.0, 3: ValueError("boom"), 4: "infeasible", 5: "nosolution",
        6: dict(jbar=4.0, status="time_limit", gap=0.4), 7: 3.0,
    }
    report, fake = sweep(monkeypatch, table, [2, 3, 4, 5, 6, 7], warm_start=True,
                         warm_start_mode="both")
    sources = [None if c["warm_start"] is None else c["warm_start"]["H2"] for c in fake.calls]
    assert sources == [None, 2, 2, 2, 2, 6]
    for call in fake.calls[1:]:
        assert call["warm_start_mode"] == "both"
        assert call["warm_start"]["H1"] == 2
        assert call["warm_start"]["x"] is not None


def test_no_warm_start_by_default(monkeypatch):
    """Spec 1937: warm_start defaults to false: no source is passed."""
    _, fake = sweep(monkeypatch, {2: 5.0, 3: 4.0, 4: 3.0}, [2, 3, 4])
    assert all(c["warm_start"] is None for c in fake.calls)


def test_warm_start_default_mode_is_start(monkeypatch):
    """Spec 1938: the default mode is 'start'."""
    _, fake = sweep(monkeypatch, {2: 5.0, 3: 4.0}, [2, 3], warm_start=True)
    assert fake.calls[1]["warm_start_mode"] == "start"
    assert fake.calls[1]["warm_start"]["H2"] == 2


# ---------------------------------------------------------------------------
# Grid (spec 2058-2106)
# ---------------------------------------------------------------------------

# Hand table for the grid, E = 10, J_init(H1 = 1) = 3, J_init(H1 = 2) = 2:
#   (1, 2): Jbar 5.0, P = 3 + 9 * 5.0 = 48
#   (1, 3): Jbar 3.0, P = 3 + 9 * 3.0 = 30
#   (2, 2): Jbar 3.0, P = 2 + 8 * 3.0 = 26
#   (2, 3): Jbar 4.0, P = 2 + 8 * 4.0 = 34
def _grid_table(jbars, override=None, jinit=None):
    jinit = jinit or {1: 3.0, 2: 2.0}
    table = {}
    for (H1, H2), jbar in jbars.items():
        table[(H1, H2)] = dict(jbar=jbar, jinit=jinit[H1])
    for key, extra in (override or {}).items():
        table[key] = dict(table[key], **extra)
    return table


GRID_JBARS = {(1, 2): 5.0, (1, 3): 3.0, (2, 2): 3.0, (2, 3): 4.0}


def grid(monkeypatch, table, h1, h2, hook=None, **options):
    fake = install(monkeypatch, table, hook)
    report = hs.sweep_horizon_grid(SMALL, h1, h2, **options)
    return report, fake


def sweeps_of(report):
    value = report["H1_sweeps"]
    return list(value.values()) if isinstance(value, dict) else list(value)


def test_grid_candidates_and_case_inputs(monkeypatch):
    """Spec 2073-2074, 2083: H1 and H2 sorted, repeated values removed; mode operating_average
    without E; calls by H1 then H2."""
    report, fake = grid(monkeypatch, _grid_table(GRID_JBARS), [2, 1, 2, 1], [3, 2, 2])
    assert [(c["H1"], c["H2"]) for c in fake.calls] == [(1, 2), (1, 3), (2, 2), (2, 3)]
    for call in fake.calls:
        assert call["data"]["objective_mode"] == "operating_average"
        assert list(call["data"]["H"]) == [call["H1"], call["H2"]]
    assert list(report["planned_transitory_horizons"]) == [1, 2]
    assert list(report["planned_operating_horizons"]) == [2, 3]
    assert [(row["H1"], row["H2"]) for row in report["cases"]] == [(1, 2), (1, 3), (2, 2), (2, 3)]


def test_grid_ranking_by_projected_cost(monkeypatch):
    """Spec 1906, 2083, 2096-2098, 2105: with E = 10 the mode is evaluation_total, the ranking
    key is P_E (Eq. d_projected): 48, 30, 26, 34; best (2, 2)."""
    report, fake = grid(monkeypatch, _grid_table(GRID_JBARS), [1, 2], [2, 3],
                        evaluation_horizon=10)
    for call in fake.calls:
        assert call["data"]["objective_mode"] == "evaluation_total"
        assert call["data"]["evaluation_horizon"] == 10
    assert report["comparison_metric"] == "projected_evaluation_cost"
    assert report["mip_certificate_metric"] == "objective"
    assert report["evaluation_horizon"] == 10
    best = report["best_proven"]
    assert (best["H1"], best["H2"], best["T"]) == (2, 2, 4)
    assert best["status"] == "optimal"
    assert close(best["projected_evaluation_cost"], 26.0)
    assert close(best["objective"], 26.5)                         # P_E + C_D u, spec 1782
    assert report["best_feasible"]["H1"] == 2 and report["best_feasible"]["H2"] == 2
    for key in ("best_proven", "best_feasible"):
        entry = report[key]
        for name in ("H1", "H2", "T", "status", "mip_gap", "objective", "J_initialization",
                     "J_op_average", "peak_damage", "damage_penalty", "operating_objective",
                     "projected_evaluation_cost", "evaluation_objective"):
            assert name in entry, name
        assert "J_op" not in entry and "J_total" not in entry


def test_grid_ranking_by_jbar_without_evaluation_horizon(monkeypatch):
    """Spec 1906, 2096, 2098: without E the ranking key is Jbar. Jbar = 5, 3, 3, 4: tie between
    (1, 3) and (2, 2): the smaller H1 wins (spec 1909)."""
    report, _ = grid(monkeypatch, _grid_table(GRID_JBARS), [1, 2], [2, 3])
    assert report["comparison_metric"] == "J_op_average"
    assert report["evaluation_horizon"] is None
    assert (report["best_proven"]["H1"], report["best_proven"]["H2"]) == (1, 3)
    assert (report["best_feasible"]["H1"], report["best_feasible"]["H2"]) == (1, 3)


def test_grid_tie_in_projected_cost_goes_to_the_smaller_H1(monkeypatch):
    """Jbar = 5.0, 3.0 (H1 = 1), 3.5, 4.0 (H1 = 2); E = 10:
    P(1, 3) = 3 + 9 * 3.0 = 30; P(2, 2) = 2 + 8 * 3.5 = 30; tie: (1, 3)."""
    jbars = {(1, 2): 5.0, (1, 3): 3.0, (2, 2): 3.5, (2, 3): 4.0}
    report, _ = grid(monkeypatch, _grid_table(jbars), [1, 2], [2, 3], evaluation_horizon=10)
    assert (report["best_proven"]["H1"], report["best_proven"]["H2"]) == (1, 3)


def test_grid_tie_in_the_same_H1_goes_to_the_smaller_H2(monkeypatch):
    """Jbar = 3.0, 3.0 (H1 = 1), 4.0, 5.0 (H1 = 2), no E: (1, 2) and (1, 3) tie: (1, 2)."""
    jbars = {(1, 2): 3.0, (1, 3): 3.0, (2, 2): 4.0, (2, 3): 5.0}
    report, _ = grid(monkeypatch, _grid_table(jbars), [1, 2], [2, 3])
    assert (report["best_proven"]["H1"], report["best_proven"]["H2"]) == (1, 2)


def test_grid_best_proven_differs_from_best_feasible(monkeypatch):
    """The best case by P_E, (2, 2) with P = 26, is time_limit: best feasible (2, 2); the best
    proven case is the next one by P_E: (1, 3) with P = 30."""
    table = _grid_table(GRID_JBARS, {(2, 2): {"status": "time_limit", "gap": 0.4}})
    report, _ = grid(monkeypatch, table, [1, 2], [2, 3], evaluation_horizon=10)
    assert (report["best_feasible"]["H1"], report["best_feasible"]["H2"]) == (2, 2)
    assert report["best_feasible"]["status"] == "time_limit"
    assert (report["best_proven"]["H1"], report["best_proven"]["H2"]) == (1, 3)


def test_grid_without_a_feasible_case(monkeypatch):
    """Spec 2105: best_proven and best_feasible are null without such a case."""
    table = {(h1, h2): "infeasible" for h1 in (1, 2) for h2 in (2, 3)}
    report, _ = grid(monkeypatch, table, [1, 2], [2, 3])
    assert report["best_proven"] is None
    assert report["best_feasible"] is None


def test_grid_report_keys(monkeypatch):
    """Spec 2093-2105."""
    report, _ = grid(monkeypatch, _grid_table(GRID_JBARS), [1, 2], [2, 3])
    for key in ("input", "complete", "objective", "comparison_metric", "mip_certificate_metric",
                "evaluation_horizon", "varied_parameters", "fixed_dimensions",
                "planned_transitory_horizons", "planned_operating_horizons",
                "warm_start_enabled", "H1_sweeps", "cases", "best_proven", "best_feasible"):
        assert key in report, key
    assert Path(report["input"]) == SMALL
    assert report["complete"] is True
    assert isinstance(report["objective"], str)
    assert list(report["varied_parameters"]) == ["H1", "H2"]
    assert dict(report["fixed_dimensions"]) == {"F": 3, "M": 1, "L": 1}
    assert report["warm_start_enabled"] is True                 # default, spec 2066
    sub = sweeps_of(report)
    assert len(sub) == 2
    for h1, one in zip((1, 2), sub):
        assert Path(one["input"]) == SMALL                       # the input file of the grid
        assert one["H1"] == h1
        assert [row["H2"] for row in one["cases"]] == [2, 3]


def test_grid_warm_starts_act_within_each_H1(monkeypatch):
    """Spec 2084, 2201-2202: the first case of every H1 has no source; later cases take the
    preceding case of the same H1. warm_start_enabled is the flag."""
    report, fake = grid(monkeypatch, _grid_table(GRID_JBARS), [1, 2], [2, 3])
    by_case = {(c["H1"], c["H2"]): c["warm_start"] for c in fake.calls}
    assert by_case[(1, 2)] is None
    assert by_case[(2, 2)] is None
    assert by_case[(1, 3)]["H1"] == 1 and by_case[(1, 3)]["H2"] == 2
    assert by_case[(2, 3)]["H1"] == 2 and by_case[(2, 3)]["H2"] == 2
    assert report["warm_start_enabled"] is True
    report, fake = grid(monkeypatch, _grid_table(GRID_JBARS), [1, 2], [2, 3], warm_start=False)
    assert all(c["warm_start"] is None for c in fake.calls)
    assert report["warm_start_enabled"] is False


def test_grid_early_stop_acts_within_each_H1_and_uses_jbar(monkeypatch):
    """Spec 2084: the early stop acts within each H1 and uses Jbar also when E is given.

    E = 3; H2 = 2..6. H1 = 1: Jbar = 10, 12, 14, 1, 1 with J_init = 100 - 20 H2: P = J_init + 2 Jbar
    = 80, 64, 48 (decreasing, so a stop on P would not come); n = 3 (H2 = 4): g = 2 / 12 = 0.167
    (increase): stop after 3 cases. H1 = 2: Jbar = 10, 9, 8, 7, 6: always decreasing (g = -0.1,
    -0.111, ...): five cases. Eight calls in all."""
    table = {}
    for h2, jbar in zip((2, 3, 4, 5, 6), (10.0, 12.0, 14.0, 1.0, 1.0)):
        table[(1, h2)] = dict(jbar=jbar, jinit=100.0 - 20.0 * h2)
    for h2, jbar in zip((2, 3, 4, 5, 6), (10.0, 9.0, 8.0, 7.0, 6.0)):
        table[(2, h2)] = dict(jbar=jbar, jinit=2.0)
    report, fake = grid(monkeypatch, table, [1, 2], [2, 3, 4, 5, 6], evaluation_horizon=3,
                        stop_on_gradient=True)
    assert len(fake.calls) == 8
    sub = sweeps_of(report)
    assert evaluated(sub[0]) == [2, 3, 4]
    assert sub[0]["stopping_rule"]["reason"] == "cost_increase"
    assert evaluated(sub[1]) == [2, 3, 4, 5, 6]
    assert sub[1]["stopping_rule"]["reason"] == "candidate_range_exhausted"
    assert len(report["cases"]) == 8


def test_grid_case_error_is_captured_within_its_H1(monkeypatch, caplog):
    """Spec 2084, 2046: an error in the case (1, 3) is a row; the other cases are solved."""
    table = _grid_table(GRID_JBARS)
    table[(1, 3)] = ValueError("boom")
    report, fake = grid(monkeypatch, table, [1, 2], [2, 3])
    assert len(fake.calls) == 4
    statuses = {(row["H1"], row["H2"]): row["status"] for row in report["cases"]}
    assert statuses[(1, 3)] == "error: boom"
    assert statuses[(1, 2)] == statuses[(2, 2)] == statuses[(2, 3)] == "optimal"
    assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 1
    # the error row is never best: best is (2, 2) with Jbar 3.0 (Jbar = 5.0, 3.0, 4.0 remain)
    assert (report["best_proven"]["H1"], report["best_proven"]["H2"]) == (2, 2)


@pytest.mark.parametrize("transitory, evaluation", [
    ([1, 2, 3], 3),       # E must exceed every H1 (spec 2077, 2090)
    ([1, 2], 2),
    ([1, 2], 1),
    ([1, 2], 2.5),
])
def test_grid_evaluation_horizon_must_exceed_every_H1(monkeypatch, transitory, evaluation):
    fake = install(monkeypatch, _grid_table(GRID_JBARS))
    with pytest.raises(ValueError):
        hs.sweep_horizon_grid(SMALL, transitory, [2, 3], evaluation_horizon=evaluation)
    assert fake.calls == []


def test_grid_evaluation_horizon_above_every_H1_is_accepted(monkeypatch):
    """Spec 2077: E = 3 exceeds H1 = 1, 2."""
    report, _ = grid(monkeypatch, _grid_table(GRID_JBARS), [1, 2], [2, 3], evaluation_horizon=3)
    assert report["evaluation_horizon"] == 3


@pytest.mark.parametrize("transitory, operating", [([0, 1], [2, 3]), ([1, 2], [0, 3]),
                                                   ([1.5], [2, 3]), ([1, 2], [-2])])
def test_grid_candidates_must_be_positive_integers(monkeypatch, transitory, operating):
    """Spec 2073-2074."""
    fake = install(monkeypatch, _grid_table(GRID_JBARS))
    with pytest.raises(ValueError):
        hs.sweep_horizon_grid(SMALL, transitory, operating)
    assert fake.calls == []


def test_grid_checkpoints_and_final_file(monkeypatch, tmp_path):
    """Spec 2085: the grid report is rewritten after every case with complete = false and at the end.
    The fake reads the file at the start of each call after the first: n cases are in it."""
    target = tmp_path / "grid" / "grid.yaml"
    seen = []

    def hook(fake, H1, H2):
        if len(fake.calls) == 1:
            return
        seen.append(load_report(target))

    report, _ = grid(monkeypatch, _grid_table(GRID_JBARS), [1, 2], [2, 3], hook=hook,
                     output_path=target)
    assert len(seen) == 3
    for n, checkpoint in enumerate(seen, start=1):
        assert checkpoint["complete"] is False
        assert len(checkpoint["cases"]) == n
    assert report["complete"] is True
    assert load_report(target) == plain(report)


# ---------------------------------------------------------------------------
# Real solves (Gurobi): small Gaussian fleet, F = 3, M = 1, L = 1
# ---------------------------------------------------------------------------

def _recording_solver(monkeypatch):
    """Wrap the real ``solve_mapping`` and keep each result by (H1, H2)."""
    real = hs.solve_mapping
    results = {}

    def wrapper(data, results_path=None, **kwargs):
        result = real(data, results_path, **kwargs)
        results[(result["H1"], result["H2"])] = result
        return result

    monkeypatch.setattr(hs, "solve_mapping", wrapper)
    return results


def _check_rows_against_results(rows, results):
    for row in rows:
        result = results[(row["H1"], row["H2"])]
        assert row["status"] == result["status"]
        assert row["T"] == row["H1"] + row["H2"]
        if result["x"] is None:
            assert row["J_op_average"] is None
            continue
        assert close(row["J_op_average"], result["J_op_average"], rel=1e-9, abs_=1e-9)
        assert close(row["J_op_average"], result["J_op"] / row["H2"], rel=1e-9, abs_=1e-9)
        assert close(row["objective"], result["objective"], rel=1e-9, abs_=1e-9)


def test_real_sweep_with_output(monkeypatch, tmp_path):
    """Real sweep over H2 in [2, 3, 4], H1 = 1, with output_path (spec 1949-1951, 2043)."""
    pytest.importorskip("gurobipy")
    results = _recording_solver(monkeypatch)
    target = tmp_path / "out" / "sweep.yaml"
    report = hs.sweep_operating_horizons(REAL, [2, 3, 4], output_path=target)
    assert report["complete"] is True
    assert report["H1"] == 1
    assert evaluated(report) == [2, 3, 4]
    assert [row["H2"] for row in report["cases"]] == [2, 3, 4]
    _check_rows_against_results(report["cases"], results)
    check_gradient_annotation(report["cases"], 1e-3)
    assert load_report(target) == plain(report)
    assert report["best_proven_H2"] is not None
    proven = [row for row in report["cases"] if row["status"] == "optimal"]
    best = min(proven, key=lambda row: (row["J_op_average"], row["H2"]))
    assert report["best_proven_H2"] == best["H2"]
    assert close(report["best_proven_J_op_average"], best["J_op_average"])


def test_real_parallel_sweep(monkeypatch, tmp_path):
    """Real sweep with n_workers = 2 (spec 2053-2055): rows ordered by H2, annotations
    recomputed over the ordered rows."""
    pytest.importorskip("gurobipy")
    target = tmp_path / "parallel.yaml"
    report = hs.sweep_operating_horizons(REAL, [4, 2, 3], n_workers=2, output_path=target)
    assert report["n_workers"] == 2
    assert report["complete"] is True
    assert [row["H2"] for row in report["cases"]] == [2, 3, 4]
    assert evaluated(report) == [2, 3, 4]
    for row in report["cases"]:
        assert row["T"] == 1 + row["H2"]
        if row["J_op"] is not None:
            assert close(row["J_op_average"], row["J_op"] / row["H2"], rel=1e-9, abs_=1e-9)
    check_gradient_annotation(report["cases"], 1e-3)
    assert load_report(target) == plain(report)


def test_real_grid(monkeypatch, tmp_path):
    """Real grid over H1 in [1, 2], H2 in [2, 3] (spec 2058-2106)."""
    pytest.importorskip("gurobipy")
    results = _recording_solver(monkeypatch)
    target = tmp_path / "grid.yaml"
    report = hs.sweep_horizon_grid(REAL, [2, 1], [3, 2], output_path=target)
    rows = report["cases"]
    assert [(row["H1"], row["H2"]) for row in rows] == [(1, 2), (1, 3), (2, 2), (2, 3)]
    _check_rows_against_results(rows, results)
    assert report["comparison_metric"] == "J_op_average"
    feasible = [row for row in rows if is_feasible(row)]
    best = min(feasible, key=lambda row: (row["J_op_average"], row["H1"], row["H2"]))
    assert (report["best_feasible"]["H1"], report["best_feasible"]["H2"]) == (best["H1"], best["H2"])
    assert report["complete"] is True
    assert load_report(target) == plain(report)


def test_real_case_error_from_a_profile_of_the_wrong_length(monkeypatch):
    """Spec 1948: a profile whose length is not H2 fails the checks, as an error of that case.
    mu has two operating entries: H2 = 2 is solved, H2 = 3 is an error row."""
    pytest.importorskip("gurobipy")
    report = hs.sweep_operating_horizons(PROFILE2, [2, 3])
    rows = report["cases"]
    assert not is_error(rows[0])
    assert is_error(rows[1])
    assert dict(rows[1]["dimensions"]) == {"F": 3, "M": 1, "L": 1, "H1": 1, "H2": 3, "T": 4}
    assert rows[1]["J_op_average"] is None
    assert report["complete"] is True
