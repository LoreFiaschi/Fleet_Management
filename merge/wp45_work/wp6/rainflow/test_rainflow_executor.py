"""Tests of the rainflow cell builder, ``degradation_model/rainflow.py``.

The tests need ``gurobipy`` and are skipped without it. Every model below has
fewer than 200 variables, far below the size limit of the free Gurobi licence.

The reference case is worked by hand: two trains, one mission, one component,
horizon ``H1 = H2 = 2``. Train 0 flies, is repaired, flies and is replaced.
Train 1 flies on days 2 and 4 and adds no degradation. Each tail bound has its
own inputs (``_BOUNDS``). The expected values come from the exact day maps of
the spec, section "Rainflow Model".
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")

from fleet_management.degradation_model import base  # noqa: E402
from fleet_management.degradation_model import rainflow  # noqa: E402

# ---------------------------------------------------------------------------
# The configuration of the hand-worked case (the attributes of FleetConfig)
# ---------------------------------------------------------------------------
_NAN = float("nan")

# per bound: (epsilon that holds, epsilon whose row fails under ard1 only) and its inputs;
# a profile value is the increment of train 0 (train 1 has zero increments)
_BOUNDS = {
    "markov": {"eps": (0.7, 0.6)},
    "cantelli": {"eps": (0.2, 0.11), "v_0": 0.004, "replacement_v": 0.002, "v": 0.01},
    "hoeffding": {"eps": (0.05, 0.01), "R_0": 0.01, "R_new": 0.005, "support": 0.2},
    "bernstein": {
        "eps": (0.4, 0.3),
        "v_0": 0.004,
        "replacement_v": 0.002,
        "v": 0.01,
        "support": 0.25,
        "support_0": 0.3,
        "support_new": 0.1,
    },
    "chernoff": {"eps": (0.5, 0.4), "s_chernoff": 4.0, "cgf_0": 1.0, "cgf_new": 0.5, "cgf": 1.5},
}
_OPTIONS = {
    "verbose": 0,
    "mip_gap": 0.0,
    "time_limit": 60.0,
    "fast": False,
    "gurobi_params": {"DualReductions": 0},   # a definite status for an infeasible case
    "objective_mode": "total",
    "evaluation_horizon": None,
    "progress_interval_seconds": 5.0,
    "relaxation_warm_start": False,
    "relaxation_time_limit": 60.0,
    "validation": None,
}


def _cfg(bound, repair_model="ard1", *, reliability_impl="tangent", encoding="bigm",
         failing=False, **changes) -> SimpleNamespace:
    """The hand-worked case of one bound, as the ``FleetConfig`` of ``load_config``.

    ``failing`` takes the epsilon whose row fails under ``ard1`` only. The
    derived fields (``support_const``, ``range_*``) follow ``config._derive``.
    ``changes`` replace any attribute after the defaults are set.
    """
    F, M, L, H1, H2 = 2, 1, 1, 2, 2
    inputs = _BOUNDS[bound]
    eps = inputs["eps"][1 if failing else 0]
    kappa = math.log(1.0 / eps)
    ard1 = repair_model == "ard1"

    def cell(value) -> np.ndarray:
        return np.full((F, L), value, dtype=float)

    def selector(value) -> np.ndarray:
        return np.full((F, L), value, dtype=object)

    def profile(key):
        if key not in inputs:
            return None
        values = np.zeros((F, L, M, H2))
        values[0] = inputs[key]
        return values

    b = _NAN
    if bound == "bernstein":
        b = max(inputs["support_0"], inputs["support_new"], inputs["support"])
    c = kappa * b / 3.0
    s = inputs.get("s_chernoff", _NAN)
    v_0, R_0, cgf_0 = (inputs.get(key, _NAN) for key in ("v_0", "R_0", "cgf_0"))
    range_v = {"cantelli": eps / (1.0 - eps), "bernstein": (1.0 - 2.0 * c) / (2.0 * kappa)}
    cfg = dict(
        F=F, M=M, L=L, H=[H1, H2], H1=H1, H2=H2, T=H1 + H2, component_names=["Component 1"],
        model=selector("rainflow"), repair_model=selector(repair_model),
        bound_method=selector(bound),
        reliability_impl=selector(
            reliability_impl if bound in rainflow.QUADRATIC_BOUNDS else "exact"),
        encoding=selector(encoding),
        tau=cell(1.0), epsilon=cell(eps), tangent_ref=cell(0.5), rho=cell(0.5),
        mu_0=cell(0.2), v_0=cell(v_0), support_0=cell(inputs.get("support_0", _NAN)),
        R_0=cell(R_0), cgf_0=cell(cgf_0),
        anchor_mu_0=cell(0.2 if ard1 else _NAN), anchor_v_0=cell(v_0 if ard1 else _NAN),
        anchor_R_0=cell(R_0 if ard1 else _NAN), anchor_K_0=cell(cgf_0 if ard1 else _NAN),
        replacement_mu=cell(0.1), replacement_v=cell(inputs.get("replacement_v", _NAN)),
        support_new=cell(inputs.get("support_new", _NAN)),
        R_new=cell(inputs.get("R_new", _NAN)), cgf_new=cell(inputs.get("cgf_new", _NAN)),
        s_chernoff=cell(s), sigma=cell(_NAN), eta=cell(_NAN), v_max_user=cell(_NAN),
        gamma_beta_0=cell(_NAN), gamma_beta_new=cell(_NAN), C_R=cell(2.0), C_rep=cell(3.0),
        C_M=1.0, C_D=0.0, penalty_type="inf_norm", pwl_points=8,
        allow_replacement=True, replacement_as_new=False,
        mu=np.zeros((F, L, M, H2)), v=profile("v"), support=profile("support"),
        cgf=profile("cgf"), gamma_beta=None, mu_trans=None, v_trans=None, support_trans=None,
        cgf_trans=None, gamma_beta_trans=None,
        support_const=cell(b), range_mu=cell(1.0), range_v=cell(range_v.get(bound, _NAN)),
        range_R=cell(2.0 / kappa if bound == "hoeffding" else _NAN),
        range_K=cell(s - kappa if bound == "chernoff" else _NAN),
        options=dict(_OPTIONS),
    )
    cfg["mu"][0] = 0.3
    cfg.update(changes)
    return SimpleNamespace(**cfg)


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------
def _build_schedule(L: int = 1) -> dict:
    """Train 0 flies, is repaired, flies, is replaced; train 1 flies on days 2 and 4."""
    x, m, r = np.zeros((2, 2, 4)), np.zeros((2, L, 4)), np.zeros((2, L, 4))
    for index in [(0, 1, 0), (0, 0, 1), (0, 1, 2), (0, 0, 3), (1, 1, 1), (1, 1, 3)]:
        x[index] = 1.0
    m[0, :, 1] = 1.0
    r[0, :, 3] = 1.0
    return {"x": x, "m": m, "r": r}


_SCHEDULE = _build_schedule()
_INFEASIBLE = ("infeasible", "inf_or_unbounded")
# train 0 on days 1..4 under each law: (states, anchors); train 1 keeps its start
_EXPECTED = {
    "mu": {"ard1": ([0.5, 0.35, 0.65, 0.1], [0.2, 0.35, 0.35, 0.1]),
           "ardinf": ([0.5, 0.25, 0.55, 0.1], None)},
    "v": {"ard1": ([0.014, 0.0065, 0.0165, 0.002], [0.004, 0.0065, 0.0065, 0.002]),
          "ardinf": ([0.014, 0.0035, 0.0135, 0.002], None)},
    "R": {"ard1": ([0.05, 0.02, 0.06, 0.005], [0.01, 0.02, 0.02, 0.005]),
          "ardinf": ([0.05, 0.0125, 0.0525, 0.005], None)},
    "K": {"ard1": ([2.5, 1.75, 3.25, 0.5], [1.0, 1.75, 1.75, 0.5]),
          "ardinf": ([2.5, 1.25, 2.75, 0.5], None)},
}
_START = {"mu": 0.2, "v": 0.004, "R": 0.01, "K": 1.0}
_Z = {"ard1": [0.0, 0.15, 0.0, 0.0], "ardinf": [0.0, 0.25, 0.0, 0.0]}
_RESULT_KEYS = {"mu": ("mu", "g_mu"), "v": ("v", "g_v"), "R": ("R", "g_R"), "K": ("K", "g_K")}


def _tracked(bound) -> list:
    second = rainflow.BOUND_QUANTITY[bound]
    return ["mu"] + ([second[0]] if second else [])


def _build(cfg):
    ctx = base.build_fleet(cfg, base.resolve_run_options(cfg))
    ctx.model.update()
    return ctx


def _fix_schedule(ctx, schedule) -> None:
    """Fix the binaries ``x``, ``m``, ``r`` of the model to ``schedule`` by bounds."""
    for variables, name in ((ctx.x, "x"), (ctx.m_rep, "m"), (ctx.r_rep, "r")):
        if variables is None:
            continue
        for key, variable in variables.items():
            variable.LB = variable.UB = float(schedule[name][key])


def _min_states(ctx) -> None:
    """Minimise the states; with fixed binaries the one-sided rows are then exact."""
    variables = list(ctx.mu_var.values()) + list(ctx.z_var.values())
    for q in ("v", "R", "K"):
        variables += list(ctx.state_vars[q].values())
    ctx.model.setObjective(gp.quicksum(variables), gp.GRB.MINIMIZE)


def _solve_fixed(cfg, schedule=_SCHEDULE):
    """Build the model, fix the schedule, minimise the states, extract the result."""
    ctx = _build(cfg)
    _fix_schedule(ctx, schedule)
    _min_states(ctx)
    ctx.model.optimize()
    return ctx, base.extract_solution(ctx, cfg)


def _row_names(ctx) -> set:
    """Names of the linear rows of the model."""
    return {row.ConstrName for row in ctx.model.getConstrs()}


def _stack(first, second) -> SimpleNamespace:
    """Two one-component cases as one fleet with ``L = 2``; ``first`` is component 0."""
    joined = dict(vars(first))
    for key, a in vars(first).items():
        b = getattr(second, key)
        if a is None and b is None:
            continue
        if a is None:
            a = np.full(b.shape, _NAN)
        if b is None:
            b = np.full(a.shape, _NAN)
        if isinstance(a, np.ndarray) and a.ndim >= 2:
            joined[key] = np.concatenate([a, b], axis=1)
    joined["L"], joined["component_names"] = 2, ["Component 1", "Component 2"]
    return SimpleNamespace(**joined)


# ---------------------------------------------------------------------------
# Registry and row constants
# ---------------------------------------------------------------------------
def test_builder_is_registered():
    base._load_builders(["rainflow"])

    builder = base.get_cell_builder("rainflow")

    assert isinstance(builder, rainflow.RainflowCellBuilder)
    assert builder.name == "rainflow"


@pytest.mark.parametrize(
    "bound, args, expected",
    [
        ("markov", {"tau": 1.0, "eps": 0.6}, {"q": "mu", "cap": 0.6}),
        (
            "cantelli",
            {"tau": 1.0, "eps": 0.2},
            {"q": "v", "c2": 0.25, "c1": 0.0, "mu_side": 1.0},
        ),
        (
            "hoeffding",
            {"tau": 1.0, "eps": 0.05},
            {"q": "R", "c2": 2.0 / math.log(20.0), "c1": 0.0, "mu_side": 1.0},
        ),
        (
            "bernstein",
            {"tau": 1.0, "eps": 0.4, "b": 0.3},
            {
                "q": "v",
                "c2": 1.0 / (2.0 * math.log(2.5)),
                "c1": -0.1,
                "mu_side": 1.0 - 0.1 * math.log(2.5),
            },
        ),
        (
            "chernoff",
            {"tau": 1.0, "eps": 0.5, "s": 4.0},
            {"q": "K", "cap": 4.0 + math.log(0.5)},
        ),
    ],
)
def test_row_constants(bound, args, expected):
    got = rainflow.row_constants(bound, **args)

    assert got.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, str):
            assert got[key] == value
        else:
            assert got[key] == pytest.approx(value)


# ---------------------------------------------------------------------------
# Variables and rows of a cell
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("repair", ["ard1", "ardinf"])
@pytest.mark.parametrize("bound", list(rainflow.BOUND_QUANTITY))
def test_variables_only_where_the_bound_tracks_them(bound, repair):
    ctx = _build(_cfg(bound, repair))

    tracked = _tracked(bound)
    for q in ("v", "R", "K"):
        assert len(ctx.state_vars[q]) == (8 if q in tracked else 0)
        assert len(ctx.anchor_vars[q]) == (8 if q in tracked and repair == "ard1" else 0)
    assert len(ctx.anchor_vars["mu"]) == (8 if repair == "ard1" else 0)


@pytest.mark.parametrize(
    "bound, impl, segments, nonconvex",
    [
        ("cantelli", "exact", 0, True),
        ("hoeffding", "exact", 0, True),
        ("bernstein", "exact", 0, True),
        ("cantelli", "tangent", 0, False),
        ("cantelli", "pwl", 64, False),
        ("hoeffding", "pwl", 64, False),
        ("bernstein", "pwl", 64, False),
        ("markov", "pwl", 0, False),
        ("chernoff", "pwl", 0, False),
    ],
)
def test_segment_binaries_and_nonconvex(bound, impl, segments, nonconvex):
    # the fixture turns pwl into exact at Markov and Chernoff cells, as config.py does
    ctx = _build(_cfg(bound, reliability_impl=impl))

    assert len(ctx.pwl_w) == segments
    assert (ctx.model.Params.NonConvex == 2) == nonconvex


@pytest.mark.parametrize("repair", ["ard1", "ardinf"])
@pytest.mark.parametrize("bound", list(rainflow.BOUND_QUANTITY))
def test_rows_of_a_cell(bound, repair):
    cfg = _cfg(bound, repair, v_max_user=np.full((2, 1), 0.05))

    names = _row_names(_build(cfg))

    loops = {f"loop_{q}_{i}_0" for q in _tracked(bound) for i in (0, 1)}
    if repair == "ard1":
        loops |= {f"loop_g{q}_{i}_0" for q in _tracked(bound) for i in (0, 1)}
    assert {name for name in names if name.startswith("loop_")} == loops
    assert ("rel_side_0_0_0" in names) == (bound in rainflow.QUADRATIC_BOUNDS)
    assert "rel_0_0_3" in names
    assert ("v_max_user_1_0_3" in names) == (bound in ("cantelli", "bernstein"))
    assert "z_0_0_1" in names


# ---------------------------------------------------------------------------
# Trajectories on the hand-worked case (the schedule is fixed)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("encoding", ["bigm", "indicator"])
@pytest.mark.parametrize("repair", ["ard1", "ardinf"])
@pytest.mark.parametrize("bound", list(rainflow.BOUND_QUANTITY))
def test_trajectories(bound, repair, encoding):
    _, out = _solve_fixed(_cfg(bound, repair, encoding=encoding))

    assert out["status"] == "optimal"
    tracked = _tracked(bound)
    for q, (key, anchor_key) in _RESULT_KEYS.items():
        if q not in tracked:
            assert np.isnan(out[key]).all()
            assert np.isnan(out[anchor_key]).all()
            continue
        states, anchors = _EXPECTED[q][repair]
        np.testing.assert_allclose(out[key][0, 0], states, atol=1e-6)
        np.testing.assert_allclose(out[key][1, 0], _START[q], atol=1e-6)
        if repair == "ard1":
            np.testing.assert_allclose(out[anchor_key][0, 0], anchors, atol=1e-6)
            np.testing.assert_allclose(out[anchor_key][1, 0], _START[q], atol=1e-6)
        else:
            assert np.isnan(out[anchor_key]).all()
    np.testing.assert_allclose(out["z"][0, 0], _Z[repair], atol=1e-6)
    np.testing.assert_allclose(out["z"][1, 0], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Reliability rows
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("repair", ["ard1", "ardinf"])
@pytest.mark.parametrize(
    "bound, impl",
    [("markov", "exact"), ("chernoff", "exact")]
    + [
        (bound, impl)
        for bound in rainflow.QUADRATIC_BOUNDS
        for impl in ("exact", "tangent", "pwl")
    ],
)
def test_reliability_row_binds(bound, impl, repair):
    # the failing epsilon is too small for the ard1 trajectory only (margins in the plan)
    cfg = _cfg(bound, repair, reliability_impl=impl, failing=True, pwl_points=4)

    _, out = _solve_fixed(cfg)

    if repair == "ard1":
        assert out["status"] in _INFEASIBLE
    else:
        assert out["status"] == "optimal"


@pytest.mark.parametrize("repair", ["ard1", "ardinf"])
@pytest.mark.parametrize("impl", ["exact", "pwl"])
@pytest.mark.parametrize("bound", list(rainflow.QUADRATIC_BOUNDS))
def test_quadratic_forms_keep_the_trajectories(bound, impl, repair):
    _, out = _solve_fixed(_cfg(bound, repair, reliability_impl=impl, pwl_points=4))

    assert out["status"] == "optimal"
    for q in _tracked(bound):
        states = _EXPECTED[q][repair][0]
        np.testing.assert_allclose(out[_RESULT_KEYS[q][0]][0, 0], states, atol=1e-6)


@pytest.mark.parametrize("repair, feasible", [("ard1", False), ("ardinf", True)])
def test_v_max_user_row(repair, feasible):
    # 0.015 lies below the ard1 peak of v (0.0165) and above the ardinf peak (0.0135)
    cfg = _cfg("cantelli", repair, v_max_user=np.full((2, 1), 0.015))

    _, out = _solve_fixed(cfg)

    if feasible:
        assert out["status"] == "optimal"
    else:
        assert out["status"] in _INFEASIBLE


# ---------------------------------------------------------------------------
# The Bernstein side row, mixed fleets and bad cell data
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("b, feasible", [(0.3, False), (0.0, True)])
def test_bernstein_side_row_is_tau_minus_c(b, feasible):
    # The mean reaches tau on day 1 with v = 0. Only the side row mu <= tau - c can fail;
    # with the side row mu <= tau of the old module, b = 0.3 would be feasible.
    zeros = np.zeros((2, 1, 1, 2))
    mu_trans = zeros.copy()
    mu_trans[0, 0, 0] = [0.8, 0.0]
    cfg = _cfg(
        "bernstein", "ardinf", reliability_impl="exact",
        mu=zeros, v=zeros, support=zeros, mu_trans=mu_trans,
        v_0=np.zeros((2, 1)), replacement_v=np.zeros((2, 1)),
        support_const=np.full((2, 1), b),
    )

    ctx, out = _solve_fixed(cfg)

    if feasible:
        assert out["status"] == "optimal"
    else:
        assert out["status"] in _INFEASIBLE
    side = ctx.model.getConstrByName("rel_side_0_0_0")
    assert side.RHS == pytest.approx(1.0 - math.log(2.5) * b / 3.0)


@pytest.mark.parametrize("repair", ["ard1", "ardinf"])
def test_two_bounds_in_one_fleet(repair):
    cfg = _stack(_cfg("cantelli", repair), _cfg("chernoff", repair))

    ctx, out = _solve_fixed(cfg, _build_schedule(L=2))

    assert out["status"] == "optimal"
    assert {key[1] for key in ctx.state_vars["v"]} == {0}
    assert {key[1] for key in ctx.state_vars["K"]} == {1}
    for q, l in (("v", 0), ("K", 1), ("mu", 0), ("mu", 1)):
        np.testing.assert_allclose(
            out[_RESULT_KEYS[q][0]][0, l], _EXPECTED[q][repair][0], atol=1e-6
        )
    assert np.isnan(out["v"][:, 1]).all()
    assert np.isnan(out["K"][:, 0]).all()


@pytest.mark.parametrize(
    "bound, changes, message",
    [
        (
            "cantelli",
            {"bound_method": np.full((2, 1), "weibull", dtype=object)},
            "unknown bound_method",
        ),
        ("bernstein", {"support_const": np.full((2, 1), _NAN)}, "support_const is not finite"),
    ],
)
def test_bad_cell_data_is_an_error(bound, changes, message):
    with pytest.raises(ValueError, match=message):
        _build(_cfg(bound, **changes))
