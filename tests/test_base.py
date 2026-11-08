"""Tests of the shared Gurobi model, ``degradation_model/base.py``.

The tests need ``gurobipy`` and are skipped without it. Every model below has
at most about 120 variables and rows, far below the size limit of the free
Gurobi licence.

The cell builders of the real degradation models are not used. Three small
builders (``_MeanVarianceBuilder``, ``_WienerLikeBuilder``, ``_RowOnlyBuilder``)
combine the helpers of ``base`` the way a real builder does. They are registered
under the model name ``gaussian``.

The reference case is worked by hand: two trains, one mission, one component,
horizon ``H1 = H2 = 2``. Train 0 flies, is repaired, flies and is replaced.
Train 1 flies on days 2 and 4 and adds no degradation.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

gp = pytest.importorskip("gurobipy")

from fleet_management.degradation_model import base  # noqa: E402

# ---------------------------------------------------------------------------
# The configuration of the hand-worked case (the attributes of FleetConfig)
# ---------------------------------------------------------------------------
_NAN = float("nan")

_SELECTORS = {
    "model": "gaussian",
    "repair_model": "ard1",
    "bound_method": "",
    "reliability_impl": "tangent",
    "encoding": "bigm",
}
_CELL_VALUES = {
    "tau": 1.0,
    "epsilon": 0.1,
    "tangent_ref": 0.5,
    "rho": 0.5,
    "mu_0": 0.2,
    "v_0": 0.0,
    "anchor_mu_0": 0.2,
    "anchor_v_0": 0.0,
    "anchor_R_0": _NAN,
    "anchor_K_0": _NAN,
    "replacement_mu": 0.1,
    "replacement_v": 0.0,
    "R_0": _NAN,
    "cgf_0": _NAN,
    "R_new": _NAN,
    "cgf_new": _NAN,
    "support_0": _NAN,
    "support_new": _NAN,
    "s_chernoff": _NAN,
    "eta": _NAN,
    "v_max_user": _NAN,
    "gamma_beta_0": _NAN,
    "gamma_beta_new": _NAN,
    "support_const": _NAN,
    "sigma": np.array([[0.1], [0.0]]),
    "C_R": 2.0,
    "C_rep": 3.0,
    "range_mu": 1.0,
    "range_v": 1.0 / norm.ppf(0.9) ** 2,
    "range_R": _NAN,
    "range_K": _NAN,
}
_SCALARS = {
    "C_M": 1.0,
    "C_D": 0.0,
    "penalty_type": "inf_norm",
    "pwl_points": 8,
    "allow_replacement": True,
    "replacement_as_new": False,
}
_OPTIONS = {
    "verbose": 0,
    "mip_gap": 0.0,
    "time_limit": 60.0,
    "fast": False,
    "gurobi_params": None,
    "objective_mode": "total",
    "evaluation_horizon": None,
    "progress_interval_seconds": 5.0,
    "relaxation_warm_start": False,
    "relaxation_time_limit": 60.0,
    "validation": None,
}


def _broadcast(value, shape, dtype) -> np.ndarray:
    return np.array(np.broadcast_to(np.asarray(value, dtype=dtype), shape))


def _cfg(**changes) -> SimpleNamespace:
    """The hand-worked case as the ``FleetConfig`` of the interface.

    ``changes`` replace any attribute; a scalar for a cell value is broadcast to
    ``(F, L)`` and ``options`` is merged into the default options. Setting ``H``
    sets ``H1``, ``H2`` and ``T`` and the shape of the default profiles.
    """
    F, M, L = changes.pop("F", 2), changes.pop("M", 1), changes.pop("L", 1)
    H1, H2 = H = changes.pop("H", [2, 2])
    cfg = dict(
        F=F,
        M=M,
        L=L,
        H=list(H),
        H1=H1,
        H2=H2,
        T=H1 + H2,
        component_names=[f"Component {l + 1}" for l in range(L)],
        **_SCALARS,
    )
    for name, default in _SELECTORS.items():
        cfg[name] = _broadcast(changes.pop(name, default), (F, L), str)
    for name, default in _CELL_VALUES.items():
        cfg[name] = _broadcast(changes.pop(name, default), (F, L), float)
    ard1 = cfg["repair_model"] == "ard1"
    for name in ("anchor_mu_0", "anchor_v_0"):
        cfg[name] = np.where(ard1, cfg[name], _NAN)
    for name, value in (("mu", 0.3), ("v", 0.01)):
        cfg[name] = np.zeros((F, L, M, H2))
        cfg[name][0] = value
    for name in ("support", "cgf", "gamma_beta"):
        cfg[name] = None
    for name in ("mu", "v", "support", "cgf", "gamma_beta"):
        cfg[name + "_trans"] = None
    cfg["options"] = {**_OPTIONS, **changes.pop("options", {})}
    cfg.update(changes)
    return SimpleNamespace(**cfg)


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------
def _build_schedule() -> dict:
    x, m, r = np.zeros((2, 2, 4)), np.zeros((2, 1, 4)), np.zeros((2, 1, 4))
    for index in [(0, 1, 0), (0, 0, 1), (0, 1, 2), (0, 0, 3), (1, 1, 1), (1, 1, 3)]:
        x[index] = 1.0
    m[0, 0, 1] = 1.0
    r[0, 0, 3] = 1.0
    return {"x": x, "m": m, "r": r}


_SCHEDULE = _build_schedule()

_SOLUTION_KEYS = (
    "x",
    "m",
    "r",
    "idle",
    "mu",
    "v",
    "z",
    "R",
    "K",
    "g_mu",
    "g_v",
    "g_R",
    "g_K",
    "y_mu",
    "y_v",
    "y_R",
    "y_K",
    "y_g_mu",
    "y_g_v",
    "y_g_R",
    "y_g_K",
    "u",
)
_COST_KEYS = (
    "step_costs",
    "J_initialization",
    "J_op",
    "J_op_average",
    "peak_damage",
    "damage_penalty",
    "operating_objective",
    "J_total",
    "projected_evaluation_cost",
    "evaluation_objective",
    "component_costs",
)
_RESULT_KEYS = frozenset(
    (
        "status",
        "objective",
        "mip_gap",
        "bound",
        "F",
        "M",
        "L",
        "H1",
        "H2",
        "T",
        "H",
        "model",
        "component_names",
        "tau",
        "mu_0",
        "v_0",
        "bound_method",
        "repair_model",
        "reliability_impl",
        "encoding",
        "objective_mode",
        "evaluation_horizon",
        "performance",
        "optimization_progress",
        "warm_start",
        "relaxation_warm_start",
    )
    + _SOLUTION_KEYS
    + _COST_KEYS
)
_PERFORMANCE_KEYS = frozenset(
    (
        "gurobi_version",
        "variables",
        "continuous_variables",
        "integer_variables",
        "binary_variables",
        "linear_constraints",
        "general_constraints",
        "indicator_constraints",
        "quadratic_constraints",
        "nonzeros",
        "solutions_found",
        "gurobi_runtime_seconds",
        "branch_and_bound_nodes",
        "simplex_iterations",
        "barrier_iterations",
        "work_units",
        "model_construction_seconds",
        "optimizer_call_seconds",
        "solution_extraction_seconds",
        "backend_wall_seconds",
        "requested_time_limit_seconds",
        "requested_mip_gap",
        "threads",
        "seed",
    )
)
_INFEASIBLE = (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD)
_NO_DUAL_REDUCTIONS = {"gurobi_params": {"DualReductions": 0}}


def _register(monkeypatch, builder) -> None:
    monkeypatch.setitem(base.CELL_BUILDERS, "gaussian", builder)


def _build(cfg):
    ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
    ctx.model.update()
    return ctx


def _fix_schedule(ctx) -> None:
    """Fix the binaries of the model to ``_SCHEDULE`` by bounds."""
    for variables, name in ((ctx.x, "x"), (ctx.m_rep, "m"), (ctx.r_rep, "r")):
        if variables is None:
            continue
        for key, variable in variables.items():
            variable.LB = variable.UB = float(_SCHEDULE[name][key])


def _min_states(ctx) -> None:
    """Minimise the states; with fixed binaries the one-sided rows are then exact."""
    variables = (
        list(ctx.mu_var.values())
        + list(ctx.state_vars["v"].values())
        + list(ctx.z_var.values())
        + [y for table in ctx.start_vars.values() for y in table.values()]
        + [y for table in ctx.start_anchor_vars.values() for y in table.values()]
    )
    ctx.model.setObjective(gp.quicksum(variables), gp.GRB.MINIMIZE)


def _solve_fixed(cfg, *, min_states=True):
    """Build the model, fix the schedule, optimise and extract the result."""
    ctx = _build(cfg)
    _fix_schedule(ctx)
    if min_states:
        _min_states(ctx)
    ctx.model.optimize()
    return ctx, base.extract_solution(ctx, cfg)


def _rows(ctx, prefix: str) -> list:
    return [row for row in ctx.model.getConstrs() if row.ConstrName.startswith(prefix)]


def _row_names(ctx) -> set:
    """Names of the linear rows of the model."""
    return {row.ConstrName for row in ctx.model.getConstrs()}


# ---------------------------------------------------------------------------
# Test builders: how the helpers of ``base`` fit together
# ---------------------------------------------------------------------------
def _set_nonconvex(ctx, cells) -> None:
    if any(str(ctx.impl_of[i, l]) == "exact" for i, l in cells):
        ctx.model.Params.NonConvex = 2


class _MeanVarianceBuilder:
    """Gaussian-like cell: mean and variance, both anchored under ``ard1``."""

    name = "gaussian"

    def prepare(self, ctx, cfg, cells, opts) -> None:
        base.add_state_vars(ctx, "mu", cells)
        base.add_state_vars(ctx, "v", cells)
        base.add_selector_vars(ctx, cells)
        base.add_pwl_vars(ctx, cells)
        _set_nonconvex(ctx, cells)

    def add_cell(self, ctx, i, l) -> None:
        mu_inc = [base.day_increment(ctx, i, l, k, "mu") for k in range(ctx.T)]
        v_inc = [base.day_increment(ctx, i, l, k, "v") for k in range(ctx.T)]
        kappa2 = norm.ppf(1.0 - float(ctx.eps[i, l])) ** 2
        base.add_selector_rows(ctx, i, l)
        base.add_quantity_rows(ctx, i, l, "mu", mu_inc)
        base.add_quantity_rows(ctx, i, l, "v", v_inc)
        base.add_quadratic_reliability_rows(
            ctx, i, l, "v", c2=1.0 / kappa2, c1=0.0, mu_side=float(ctx.tau[i, l])
        )
        base.add_repair_cost_rows(ctx, i, l)


class _WienerLikeBuilder:
    """Wiener-like cell: the variance grows with the missions, a repair keeps it."""

    name = "gaussian"

    def prepare(self, ctx, cfg, cells, opts) -> None:
        base.add_state_vars(ctx, "mu", cells)
        base.add_state_vars(ctx, "v", cells, anchored=False)
        base.add_selector_vars(ctx, cells)
        base.add_pwl_vars(ctx, cells)
        _set_nonconvex(ctx, cells)
        ctx.extras["sigma"] = np.asarray(cfg.sigma, dtype=float)

    def add_cell(self, ctx, i, l) -> None:
        sigma = float(ctx.extras["sigma"][i, l])
        mu_inc = [base.day_increment(ctx, i, l, k, "mu") for k in range(ctx.T)]
        v_inc = [sigma**2 * base.mission_count(ctx, i, k) for k in range(ctx.T)]
        kappa2 = norm.ppf(1.0 - float(ctx.eps[i, l])) ** 2
        base.add_selector_rows(ctx, i, l)
        base.add_quantity_rows(ctx, i, l, "mu", mu_inc)
        base.add_quantity_rows(ctx, i, l, "v", v_inc, repair_keeps_state=True)
        base.add_quadratic_reliability_rows(
            ctx, i, l, "v", c2=1.0 / kappa2, c1=0.0, mu_side=float(ctx.tau[i, l])
        )
        base.add_repair_cost_rows(ctx, i, l)


class _RowOnlyBuilder:
    """Only the quadratic reliability row, on a variance without dynamics."""

    name = "gaussian"

    def __init__(self, c2: float, c1: float, mu_side: float):
        self.c2, self.c1, self.mu_side = c2, c1, mu_side

    def prepare(self, ctx, cfg, cells, opts) -> None:
        base.add_state_vars(ctx, "v", cells, anchored=False)
        base.add_pwl_vars(ctx, cells)
        _set_nonconvex(ctx, cells)

    def add_cell(self, ctx, i, l) -> None:
        base.add_quadratic_reliability_rows(
            ctx, i, l, "v", c2=self.c2, c1=self.c1, mu_side=self.mu_side
        )


# ---------------------------------------------------------------------------
# Registry and context
# ---------------------------------------------------------------------------
def test_unknown_model_has_no_cell_builder():
    base._load_builders(["gamma"])
    with pytest.raises(NotImplementedError, match="gamma"):
        base.get_cell_builder("no_such_model")


@pytest.mark.parametrize(
    "transitory, expected",
    [(None, [1, 2, 1, 2, 3]), ([10, 20], [10, 20, 1, 2, 3])],
)
def test_increment_follows_the_phase_lookup(transitory, expected):
    operating = np.zeros((2, 1, 1, 3))
    operating[0, 0, 0] = [1, 2, 3]
    mu_trans = None
    if transitory is not None:
        mu_trans = np.zeros((2, 1, 1, 2))
        mu_trans[0, 0, 0] = transitory
    cfg = _cfg(H=[2, 3], v=None, mu=operating, mu_trans=mu_trans)
    ctx = base.build_context(cfg, base.resolve_run_options(cfg))

    values = [ctx.increment("mu", 0, 1, 0, k) for k in range(5)]

    np.testing.assert_allclose(values, expected, atol=1e-6)
    assert ctx.increment("mu", 0, 0, 0, 3) == 0.0


def test_increment_of_a_missing_profile_raises():
    cfg = _cfg(v=None)
    ctx = base.build_context(cfg, base.resolve_run_options(cfg))

    with pytest.raises(KeyError, match="profile 'v' is not in the input"):
        ctx.increment("v", 0, 1, 0, 0)


# ---------------------------------------------------------------------------
# Variables and rows exist only where the spec builds them
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("C_D, rows", [(0.0, 0), (0.5, 8)])
def test_penalty_variable_and_rows_need_a_positive_weight(monkeypatch, C_D, rows):
    _register(monkeypatch, _MeanVarianceBuilder())

    ctx = _build(_cfg(C_D=C_D))

    assert (ctx.u_var is not None) == (C_D > 0.0)
    assert ("u_0_0" in _row_names(ctx)) == (C_D > 0.0)
    assert len(_rows(ctx, "u_")) == rows


@pytest.mark.parametrize(
    "changes, keys",
    [
        pytest.param(
            {"C_R": np.array([[2.0], [0.0]])},
            {(0, 0, k) for k in range(4)},
            id="cells-with-positive-cost",
        ),
        pytest.param(
            {"options": {"objective_mode": "operating_average"}},
            {(i, 0, k) for i in range(2) for k in (2, 3)},
            id="operating-days",
        ),
    ],
)
def test_repair_cost_variable_only_where_it_is_charged(monkeypatch, changes, keys):
    _register(monkeypatch, _MeanVarianceBuilder())

    ctx = _build(_cfg(**changes))

    assert set(ctx.z_var.keys()) == keys


@pytest.mark.parametrize("allow", [True, False])
def test_replacement_variable_and_depot_row(monkeypatch, allow):
    _register(monkeypatch, _MeanVarianceBuilder())

    ctx = _build(_cfg(allow_replacement=allow))

    assert (ctx.r_rep is not None) == allow
    depot = [row for row in ctx.model.getConstrs() if row.ConstrName == "depot_0_0_0"]
    assert len(depot) == 1
    assert ctx.model.getRow(depot[0]).size() == (3 if allow else 2)


@pytest.mark.parametrize("encoding, count", [("bigm", 0), ("indicator", 8)])
def test_selector_only_in_the_indicator_encoding(monkeypatch, encoding, count):
    _register(monkeypatch, _MeanVarianceBuilder())

    ctx = _build(_cfg(encoding=encoding))

    assert len(ctx.nb) == count
    assert len(_rows(ctx, "sel_")) == count


@pytest.mark.parametrize(
    "impl, points, count", [("pwl", 4, 32), ("pwl", 1, 0), ("tangent", 4, 0)]
)
def test_segment_binaries_only_for_pwl_with_two_points_or_more(
    monkeypatch, impl, points, count
):
    _register(monkeypatch, _MeanVarianceBuilder())

    ctx = _build(_cfg(reliability_impl=impl, pwl_points=points))

    assert len(ctx.pwl_w) == count


@pytest.mark.parametrize("repair, count", [("ardinf", 0), ("ard1", 8)])
def test_anchor_variables_only_under_ard1(monkeypatch, repair, count):
    _register(monkeypatch, _MeanVarianceBuilder())

    ctx = _build(_cfg(repair_model=repair))

    assert len(ctx.anchor_vars["mu"]) == count
    assert len(ctx.anchor_vars["v"]) == count


# ---------------------------------------------------------------------------
# Dynamics, anchors and repair cost on the hand-worked case
# ---------------------------------------------------------------------------
# Day maps of train 0 (mission, repair, mission, replacement). Train 1 flies with
# zero increments: mu = 0.2, g_mu = 0.2, v = 0, g_v = 0 and z = 0 on every day.
_TRAIN_0 = {
    ("ard1", "mu"): [0.5, 0.35, 0.65, 0.1],
    ("ardinf", "mu"): [0.5, 0.25, 0.55, 0.1],
    ("ard1", "g_mu"): [0.2, 0.35, 0.35, 0.1],
    ("ard1", "g_v"): [0.0, 0.0025, 0.0025, 0.0],
    ("ard1", "z"): [0.0, 0.15, 0.0, 0.0],
    ("ardinf", "z"): [0.0, 0.25, 0.0, 0.0],
}
_V = {
    "gaussian": [0.01, 0.0025, 0.0125, 0.0],
    "wiener": [0.01, 0.01, 0.02, 0.0],
}


def _both_trains(train_0, train_1: float) -> np.ndarray:
    """The expected ``(F, T)`` array: the day map of train 0, a constant for train 1."""
    return np.array([train_0, [train_1] * len(train_0)])


def _assert_mean_trajectory(out: dict, repair: str) -> None:
    """The mean, its anchor and the repair cost of the hand-worked case."""
    mu = _both_trains(_TRAIN_0[repair, "mu"], 0.2)
    z = _both_trains(_TRAIN_0[repair, "z"], 0.0)
    np.testing.assert_allclose(out["mu"][:, 0], mu, atol=1e-6)
    np.testing.assert_allclose(out["z"][:, 0], z, atol=1e-6)
    if repair == "ard1":
        g_mu = _both_trains(_TRAIN_0["ard1", "g_mu"], 0.2)
        np.testing.assert_allclose(out["g_mu"][:, 0], g_mu, atol=1e-6)
    else:
        assert np.isnan(out["g_mu"]).all()


def _assert_untracked_keys(out: dict) -> None:
    """The keys that no tracked quantity feeds: NaN, and no idle component."""
    for key in ("R", "K", "g_R", "g_K", "y_R", "y_K", "y_g_R", "y_g_K"):
        assert np.isnan(out[key]).all()
    np.testing.assert_allclose(out["idle"], 0.0, atol=1e-6)


@pytest.mark.parametrize("repair", ["ard1", "ardinf"])
@pytest.mark.parametrize("encoding", ["bigm", "indicator"])
def test_dynamics_reproduce_the_hand_case(monkeypatch, encoding, repair):
    _register(monkeypatch, _MeanVarianceBuilder())

    _, out = _solve_fixed(_cfg(encoding=encoding, repair_model=repair))

    assert out["status"] == "optimal"
    _assert_mean_trajectory(out, repair)
    np.testing.assert_allclose(
        out["v"][:, 0], _both_trains(_V["gaussian"], 0.0), atol=1e-6
    )
    if repair == "ard1":
        g_v = _both_trains(_TRAIN_0["ard1", "g_v"], 0.0)
        np.testing.assert_allclose(out["g_v"][:, 0], g_v, atol=1e-6)
    else:
        assert np.isnan(out["g_v"]).all()
    _assert_untracked_keys(out)


@pytest.mark.parametrize("repair", ["ard1", "ardinf"])
@pytest.mark.parametrize("encoding", ["bigm", "indicator"])
def test_wiener_variance_is_kept_by_a_repair(monkeypatch, encoding, repair):
    _register(monkeypatch, _WienerLikeBuilder())

    _, out = _solve_fixed(_cfg(encoding=encoding, repair_model=repair))

    assert out["status"] == "optimal"
    np.testing.assert_allclose(
        out["v"][:, 0], _both_trains(_V["wiener"], 0.0), atol=1e-6
    )
    assert np.isnan(out["g_v"]).all()
    _assert_mean_trajectory(out, repair)
    _assert_untracked_keys(out)


@pytest.mark.parametrize(
    "repair, present, absent",
    [
        (
            "ard1",
            ["loop_mu_0_0", "loop_v_0_0", "loop_gmu_0_0", "loop_gv_0_0"],
            [],
        ),
        ("ardinf", ["loop_mu_0_0", "loop_v_0_0"], ["loop_gmu_0_0", "loop_gv_0_0"]),
    ],
)
def test_loop_rows_cover_states_and_anchors(monkeypatch, repair, present, absent):
    _register(monkeypatch, _MeanVarianceBuilder())

    names = _row_names(_build(_cfg(repair_model=repair)))

    for name in present:
        assert name in names
    for name in absent:
        assert name not in names


def test_repair_cost_on_transitory_days_is_computed_after_the_solve(monkeypatch):
    _register(monkeypatch, _MeanVarianceBuilder())
    cfg = _cfg(
        C_R=np.array([[2.0], [0.0]]),
        options={"objective_mode": "operating_average"},
    )

    ctx, out = _solve_fixed(cfg)

    assert out["status"] == "optimal"
    assert all(k >= cfg.H1 for _, _, k in ctx.z_var)
    # The first two days are transitory. Eq. c_zpost gives the repair on day 2:
    # rho (mu - g_mu) = 0.5 (0.5 - 0.2) = 0.15, with mu and g_mu of day 1.
    np.testing.assert_allclose(out["z"][0, 0], [0.0, 0.15, 0.0, 0.0], atol=1e-6)
    assert np.isnan(out["z"][1, 0]).all()


# ---------------------------------------------------------------------------
# Reliability forms: one point (mu, v) of cell (0, 0) on day 1 is fixed by bounds
# ---------------------------------------------------------------------------
_GAUSSIAN_C2 = 1.0 / norm.ppf(0.9) ** 2
_KAPPA = math.log(10.0)
_SUPPORT = 0.3
_BERNSTEIN_C = _KAPPA * _SUPPORT / 3.0
_BERNSTEIN_ROW = {
    "c2": 1.0 / (2.0 * _KAPPA),
    "c1": -_SUPPORT / 3.0,
    "mu_side": 1.0 - _BERNSTEIN_C,
}
_BERNSTEIN_RANGE_V = (1.0 - 2.0 * _BERNSTEIN_C) / (2.0 * _KAPPA)


def _cap(c2: float, c1: float, mu: float, tau: float = 1.0) -> float:
    """The upper bound on the variance at the mean ``mu``."""
    return c2 * (tau - mu) ** 2 + c1 * (tau - mu)


def _status_at_point(monkeypatch, builder, cfg, mu: float, v: float) -> int:
    _register(monkeypatch, builder)
    ctx = _build(cfg)
    ctx.mu_var[0, 0, 0].LB = ctx.mu_var[0, 0, 0].UB = mu
    variance = ctx.state_vars["v"][0, 0, 0]
    variance.LB = variance.UB = v
    ctx.model.optimize()
    return ctx.model.Status


def _assert_feasibility(status: int, feasible: bool) -> None:
    if feasible:
        assert status == gp.GRB.OPTIMAL
    else:
        assert status in _INFEASIBLE


@pytest.mark.parametrize(
    "impl, tangent_ref, points, feasible",
    [
        ("exact", 0.5, 8, True),
        ("tangent", 0.375, 8, True),
        ("tangent", 0.5, 8, False),
        ("pwl", 0.5, 4, True),
        ("pwl", 0.5, 1, False),
    ],
)
def test_gaussian_reliability_forms(monkeypatch, impl, tangent_ref, points, feasible):
    cfg = _cfg(
        H=[1, 1],
        reliability_impl=impl,
        tangent_ref=tangent_ref,
        pwl_points=points,
        options=_NO_DUAL_REDUCTIONS,
    )
    builder = _RowOnlyBuilder(_GAUSSIAN_C2, 0.0, 1.0)
    # The variance cap at mu = 0.375 is 0.2378 on the curve, on the tangent at
    # 0.375 and on the pwl segment [0.25, 0.5] of four, but 0.2283 on the tangent
    # at 0.5 and on the single pwl tangent. The point sits 1 % under the curve.
    v = 0.99 * _cap(_GAUSSIAN_C2, 0.0, 0.375)

    status = _status_at_point(monkeypatch, builder, cfg, 0.375, v)

    _assert_feasibility(status, feasible)


@pytest.mark.parametrize(
    "mu, v_scale, feasible",
    [(0.3, 0.99, True), (0.3, 1.01, False), (0.8, 0.0, False)],
)
def test_bernstein_reliability_row(monkeypatch, mu, v_scale, feasible):
    cfg = _cfg(
        H=[1, 1],
        reliability_impl="exact",
        range_v=_BERNSTEIN_RANGE_V,
        options=_NO_DUAL_REDUCTIONS,
    )
    builder = _RowOnlyBuilder(**_BERNSTEIN_ROW)
    # The scale is relative to the cap at mu = 0.3; mu = 0.8 breaks the side row.
    v = v_scale * _cap(_BERNSTEIN_ROW["c2"], _BERNSTEIN_ROW["c1"], 0.3)

    status = _status_at_point(monkeypatch, builder, cfg, mu, v)

    _assert_feasibility(status, feasible)


# ---------------------------------------------------------------------------
# Objective and result of ``solve_mixed``
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mode, penalty, key, horizon",
    [
        ("total", "inf_norm", "J_total", None),
        ("operating_average", "inf_norm", "operating_objective", None),
        ("evaluation_total", "inf_norm", "evaluation_objective", 6),
        ("total", "quadratic", "J_total", None),
    ],
)
def test_objective_equals_the_cost_report(monkeypatch, mode, penalty, key, horizon):
    _register(monkeypatch, _MeanVarianceBuilder())
    cfg = _cfg(
        C_D=0.5,
        penalty_type=penalty,
        reliability_impl="exact" if penalty == "quadratic" else "tangent",
        options={"objective_mode": mode, "evaluation_horizon": horizon},
    )

    ctx, out = _solve_fixed(cfg, min_states=False)

    assert out["status"] == "optimal"
    assert abs(out["objective"] - out[key]) <= 1e-6 * max(1.0, abs(out[key]))
    # The variable u has no slack: it equals the value recomputed from mu (Eq. c_u).
    assert abs(ctx.u_var.X - out["u"]) <= 1e-6 * max(1.0, out["u"])


def test_solve_mixed_returns_the_documented_keys(monkeypatch):
    _register(monkeypatch, _MeanVarianceBuilder())

    out = base.solve_mixed(_cfg(C_D=0.5))

    assert out["status"] == "optimal"
    assert set(out) == _RESULT_KEYS
    assert set(out["performance"]) == _PERFORMANCE_KEYS
    assert out["performance"]["requested_mip_gap"] == 0.0
    assert out["warm_start"]["mode"] is None
    assert out["relaxation_warm_start"]["status"] == "disabled"
    assert out["model"] == [["gaussian"], ["gaussian"]]
    assert np.isnan(out["R"]).all()
    assert np.isfinite(out["v"]).all()


def test_infeasible_solve_reports_no_solution(monkeypatch):
    _register(monkeypatch, _MeanVarianceBuilder())
    cfg = _cfg(
        H=[1, 1],
        mu=np.full((2, 1, 1, 1), 0.9),
        allow_replacement=False,
        options=_NO_DUAL_REDUCTIONS,
    )

    out = base.solve_mixed(cfg)

    assert out["status"] in ("infeasible", "inf_or_unbounded")
    for key in ("objective", "mip_gap") + _SOLUTION_KEYS + _COST_KEYS:
        assert out[key] is None
    assert set(out["performance"]) == _PERFORMANCE_KEYS
    assert out["objective_mode"] == "total"


def test_unknown_warm_start_mode_is_rejected(monkeypatch):
    _register(monkeypatch, _MeanVarianceBuilder())

    with pytest.raises(ValueError):
        base.solve_mixed(_cfg(), warm_start_mode="bogus")


def test_warm_start_reports_skip_the_relaxation(monkeypatch):
    _register(monkeypatch, _MeanVarianceBuilder())
    cfg = _cfg(options={"relaxation_warm_start": True})
    source = {**_SCHEDULE, "H1": 2, "H2": 2}

    out = base.solve_mixed(cfg, warm_start=source)

    assert out["warm_start"]["applied"] is True
    assert out["warm_start"]["mode"] == "start"
    assert out["relaxation_warm_start"]["enabled"] is True
    assert out["relaxation_warm_start"]["status"] == "skipped"
