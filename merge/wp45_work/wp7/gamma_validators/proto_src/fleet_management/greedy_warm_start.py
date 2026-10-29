"""
Greedy heuristic that builds a warm-start source from the input (spec, "Greedy Heuristic").

A construction pass assigns the missions to the trains and schedules the repairs
with a proxy state ``s_hat``. A loop pass then adds repairs, or replacements, in
the operating phase until the reference trajectory of each cell meets its loop
rows and its reliability row. The reference trajectory is the trajectory of the
rows of the model with every inequality replaced by an equality, computed from
the initial values, the starting anchors and the replacement values of the input.

The heuristic works for every degradation model. It needs numpy and scipy only:
the reliability caps of the models are imported inside the functions, and neither
``base`` nor Gurobi is loaded. The module is not exported (spec, "Public API").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from fleet_management.config import _lookup

EXCESS_TOLERANCE = 1e-10          # an excess above it is a violation (spec, loop pass)
COMPLETION_TIME_LIMIT = 60.0      # completion_time_limit of the source, in seconds


# ===========================================================================
# Report
# ===========================================================================
@dataclass(frozen=True)
class GreedyWarmStartReport:
    """What the heuristic did (spec, "Report keys")."""
    repairs: int
    replacements: int
    repeatability_feasible: bool
    repeatability_failures: int
    maximum_repeatability_excess: float
    maximum_proxy_ratio: float
    assignment_order: list[list[int]]

    def as_dict(self) -> dict:
        """The seven keys, in this order, as plain Python types."""
        return {
            "repairs": int(self.repairs),
            "replacements": int(self.replacements),
            "repeatability_feasible": bool(self.repeatability_feasible),
            "repeatability_failures": int(self.repeatability_failures),
            "maximum_repeatability_excess": float(self.maximum_repeatability_excess),
            "maximum_proxy_ratio": float(self.maximum_proxy_ratio),
            "assignment_order": [[int(j) for j in day] for day in self.assignment_order],
        }


# ===========================================================================
# Construction pass
# ===========================================================================
def _profiles(cfg) -> dict:
    """Increment profiles after the phase lookup, shape ``(F, L, M, T)``; ``None`` if the input has none."""
    profiles = {}
    for key in ("mu", "v", "support", "cgf", "gamma_beta"):
        operating = getattr(cfg, key)
        profiles[key] = None if operating is None else _lookup(
            operating, getattr(cfg, key + "_trans"), cfg.H1, cfg.H2)
    return profiles


def _construction_pass(cfg, mu_inc, repair_trigger: float):
    """Missions and repairs of every day with a proxy state (spec, "Steps of the construction pass").

    ``mu_inc`` is the mean profile ``(F, L, M, T)``. Returns ``x`` ``(F, M + 1, T)``,
    ``m`` and ``r`` ``(F, L, T)`` (``r`` stays zero), the order of the missions of
    each day (1-based) and the largest ``s_hat / tau``.
    """
    F, M, L, T = cfg.F, cfg.M, cfg.L, cfg.T
    tau = np.asarray(cfg.tau, dtype=float)
    rho = np.asarray(cfg.rho, dtype=float)
    x = np.zeros((F, M + 1, T))
    m = np.zeros((F, L, T))
    r = np.zeros((F, L, T))
    s_hat = np.asarray(cfg.mu_0, dtype=float).copy()
    assignment_order: list[list[int]] = []
    maximum_ratio = 0.0
    for k in range(T):
        # Step 1: the hardest mission first; a tie goes to the larger j.
        difficulty = [(float(np.max(mu_inc[:, :, j, k] / tau)), j) for j in range(M)]
        missions = [j for _, j in sorted(difficulty, reverse=True)]
        assignment_order.append([j + 1 for j in missions])
        # Step 2: each mission to the free train with the smallest (max, sum) of the ratios.
        free = list(range(F))
        for j in missions:
            keys = {}
            for i in free:
                ratio = (s_hat[i] + mu_inc[i, :, j, k]) / tau[i]
                keys[i] = (float(np.max(ratio)), float(np.sum(ratio)), i)
            i = min(free, key=keys.get)
            free.remove(i)
            x[i, j + 1, k] = 1.0
            s_hat[i] += mu_inc[i, :, j, k]
        # Step 3: a train without a mission is idle, unless some component is due.
        for i in free:
            due = s_hat[i] > repair_trigger * tau[i]
            if due.any():
                x[i, 0, k] = 1.0
                m[i, due, k] = 1.0
                s_hat[i, due] *= 1.0 - rho[i, due]
        maximum_ratio = max(maximum_ratio, float(np.max(s_hat / tau)))
    return x, m, r, assignment_order, maximum_ratio


# ===========================================================================
# Quantities and rows of a cell
# ===========================================================================
def _cell_quantities(cfg, i: int, l: int, profiles: dict) -> list[dict]:
    """The tracked quantities of cell ``(i, l)``, the mean first (spec, "Encoding of the Dynamics").

    One dictionary per quantity ``q`` with the keys ``q``, ``start``, ``new`` (value
    after a replacement), ``anchored``, ``anchor`` (starting anchor), ``factor``
    (of a repair), ``increment`` (array ``(M, T)``, per mission and day) and
    ``range`` (the natural range of ``q``). Gaussian and Cantelli or Bernstein
    rainflow cells track the variance ``v``. A Wiener cell tracks ``v``, which
    grows by ``sigma^2`` per mission and is not changed by a repair. A Hoeffding
    rainflow cell tracks ``R`` and a Chernoff cell tracks ``K``.
    """
    model = str(cfg.model[i, l])
    bound = str(cfg.bound_method[i, l])
    ard1 = str(cfg.repair_model[i, l]) == "ard1"
    rho = float(cfg.rho[i, l])

    def item(q, start, new, anchored, anchor, factor, increment, range_q):
        return {
            "q": q, "start": float(start[i, l]), "new": float(new[i, l]),
            "anchored": bool(anchored), "anchor": math.nan if anchor is None else float(anchor[i, l]),
            "factor": float(factor), "increment": np.asarray(increment, dtype=float),
            "range": float(range_q[i, l]),
        }

    quantities = [item("mu", cfg.mu_0, cfg.replacement_mu, ard1, cfg.anchor_mu_0, 1.0 - rho,
                       profiles["mu"][i, l], cfg.range_mu)]
    if model == "gaussian" or (model == "rainflow" and bound in ("cantelli", "bernstein")):
        quantities.append(item("v", cfg.v_0, cfg.replacement_v, ard1, cfg.anchor_v_0,
                               (1.0 - rho) ** 2, profiles["v"][i, l], cfg.range_v))
    elif model == "wiener":
        sigma_square = np.full((cfg.M, cfg.T), float(cfg.sigma[i, l]) ** 2)
        quantities.append(item("v", cfg.v_0, cfg.replacement_v, False, None, 1.0,
                               sigma_square, cfg.range_v))
    elif model == "rainflow" and bound == "hoeffding":
        quantities.append(item("R", cfg.R_0, cfg.R_new, ard1, cfg.anchor_R_0,
                               (1.0 - rho) ** 2, profiles["support"][i, l] ** 2, cfg.range_R))
    elif model == "rainflow" and bound == "chernoff":
        quantities.append(item("K", cfg.cgf_0, cfg.cgf_new, ard1, cfg.anchor_K_0,
                               1.0 - rho, profiles["cgf"][i, l], cfg.range_K))
    return quantities


def _cell_rows(cfg, i: int, l: int, profiles: dict) -> dict:
    """The reliability row of cell ``(i, l)`` (spec, the model sections; Eqs. a_quad, a_tangent, a_pwl_cap).

    ``{"kind": "cap", "q", "cap"}`` for ``state_q <= cap`` (inverse Gaussian, Gamma,
    rainflow Markov and Chernoff), or ``{"kind": "quadratic", "q", "c2", "c1", "side"}``
    for ``state_q <= c2 (tau - mu)^2 + c1 (tau - mu)`` with ``mu <= side`` (Gaussian,
    Wiener, rainflow Cantelli, Hoeffding and Bernstein). Every row also holds the form
    built at the cell (``form``), ``tangent_ref`` and the optional bound ``v_max_user``
    (NaN where ``v`` is not tracked or the input gives none). The caps are those of
    the cell builders.
    """
    model = str(cfg.model[i, l])
    bound = str(cfg.bound_method[i, l])
    tau, eps = float(cfg.tau[i, l]), float(cfg.epsilon[i, l])
    if model in ("gaussian", "wiener"):
        row = {"kind": "quadratic", "q": "v", "c2": 1.0 / float(ndtri(1.0 - eps)) ** 2,
               "c1": 0.0, "side": tau}
    elif model == "inverse_gaussian":
        from fleet_management.degradation_model.inverse_gaussian import ig_mean_cap
        row = {"kind": "cap", "q": "mu", "cap": ig_mean_cap(float(cfg.eta[i, l]), tau, eps)}
    elif model == "gamma":
        from fleet_management.degradation_model.gamma import mean_cap, smallest_rate
        beta_star = smallest_rate(
            profiles["mu"][i, l], profiles["gamma_beta"][i, l],
            float(cfg.mu_0[i, l]), float(cfg.gamma_beta_0[i, l]),
            float(cfg.replacement_mu[i, l]), float(cfg.gamma_beta_new[i, l]),
            bool(cfg.allow_replacement),
        )
        row = {"kind": "cap", "q": "mu", "cap": mean_cap(beta_star, tau, eps)}
    else:
        from fleet_management.degradation_model.rainflow import row_constants
        c = row_constants(bound, tau, eps, float(cfg.support_const[i, l]), float(cfg.s_chernoff[i, l]))
        if "cap" in c:
            row = {"kind": "cap", "q": c["q"], "cap": c["cap"]}
        else:
            row = {"kind": "quadratic", "q": c["q"], "c2": c["c2"], "c1": c["c1"], "side": c["mu_side"]}
    variance = model in ("gaussian", "wiener") or (
        model == "rainflow" and bound in ("cantelli", "bernstein"))
    row["form"] = str(cfg.reliability_impl[i, l])
    row["tangent_ref"] = float(cfg.tangent_ref[i, l])
    row["v_max_user"] = float(cfg.v_max_user[i, l]) if variance else math.nan
    return row


def _row_cap(mu, tau: float, c2: float, c1: float, form: str, tangent_ref: float, pwl_points: int) -> np.ndarray:
    """Right-hand side of the quadratic reliability row on every day, in the form built at the cell.

    ``f(m) = c2 (tau - m)^2 + c1 (tau - m)`` (Eq. a_quad). ``"exact"`` gives ``f(mu)``;
    ``"tangent"`` the tangent at ``tangent_ref * tau`` (Eq. a_tangent); ``"pwl"`` the
    cap of the segment that contains the mean of the day (Eq. a_pwl_cap), with one
    tangent at ``tau / 2`` for a single segment. The mean is clipped to ``[0, tau]``,
    which keeps the cap finite above ``tau``; the side row counts that excess.
    """
    mu = np.asarray(mu, dtype=float)

    def f(point):
        return c2 * (tau - point) ** 2 + c1 * (tau - point)

    def df(point):
        return -2.0 * c2 * (tau - point) - c1

    if form == "exact":
        return f(mu)
    if form == "tangent":
        point = tangent_ref * tau
        return f(point) + df(point) * (mu - point)
    clipped = np.clip(mu, 0.0, tau)
    cap = np.full(mu.shape, -np.inf)
    for p in range(1, pwl_points + 1):
        mid = (p - 0.5) * tau / pwl_points
        upper = tau if p == pwl_points else p * tau / pwl_points
        inside = ((p - 1) * tau / pwl_points <= clipped) & (clipped <= upper)
        cap = np.where(inside, np.maximum(cap, f(mid) + df(mid) * (mu - mid)), cap)
    return cap


# ===========================================================================
# Loop pass
# ===========================================================================
def _reference_trajectory(cfg, x, m, r, i: int, l: int, quantities: list[dict]):
    """Reference trajectory of cell ``(i, l)`` under the schedule ``(x, m, r)``.

    The rows of the model with every inequality replaced by an equality, from the
    initial values, the starting anchors and the replacement values of the input.
    Returns two dictionaries ``q -> array (T,)``: the states, and the anchors of the
    anchored quantities. On a day with a replacement the state is set to its value
    ``new`` and the anchor to the state. On a day with a repair an anchored state
    becomes ``factor * s + (1 - factor) * g`` and the anchor the state; the other
    states are multiplied by ``factor``. Otherwise the state grows by the increment of
    the mission flown that day, if any.
    """
    T = cfg.T
    flights = x[i, 1:, :]
    flown = np.where(flights.sum(axis=0) > 0.5, flights.argmax(axis=0), -1)
    states, anchors = {}, {}
    for item in quantities:
        factor, increment = item["factor"], item["increment"]
        s, g = item["start"], item["anchor"]
        state, anchor = np.zeros(T), np.zeros(T)
        for k in range(T):
            if r[i, l, k] > 0.5:
                s = item["new"]
                g = s
            elif m[i, l, k] > 0.5:
                if item["anchored"]:
                    s = factor * s + (1.0 - factor) * g
                    g = s
                else:
                    s = factor * s
            elif flown[k] >= 0:
                s += float(increment[flown[k], k])
            state[k], anchor[k] = s, g
        states[item["q"]] = state
        if item["anchored"]:
            anchors[item["q"]] = anchor
    return states, anchors


def _cell_excess(cfg, x, m, r, i: int, l: int, quantities: list[dict], rows: dict):
    """Excess of cell ``(i, l)`` under the schedule ``(x, m, r)`` (spec, step 2 of the loop pass).

    Returns ``(e, raw, mean_ratio)``: ``e`` is the largest violation divided by the
    natural range of its quantity, ``raw`` the largest violation, and ``mean_ratio``
    the largest reference mean over ``tau``. A violation is one of: the loop excess
    ``max{0, s_T - s_H1}`` of a state or an anchor; the reliability row on a day, in
    the form built at the cell, and its side row ``mu <= side``; and, if given,
    ``v <= v_max_user`` (counted so that a start that breaks it is not called feasible).
    """
    H1, T = cfg.H1, cfg.T
    states, anchors = _reference_trajectory(cfg, x, m, r, i, l, quantities)
    ranges = {item["q"]: item["range"] for item in quantities}
    mu = states["mu"]
    terms = []                                          # (value, natural range)
    for q, trace in list(states.items()) + list(anchors.items()):
        terms.append((max(0.0, float(trace[T - 1] - trace[H1 - 1])), ranges[q]))
    q = rows["q"]
    if rows["kind"] == "cap":
        terms.append((max(0.0, float(np.max(states[q] - rows["cap"]))), ranges[q]))
    else:
        caps = _row_cap(mu, float(cfg.tau[i, l]), rows["c2"], rows["c1"], rows["form"],
                        rows["tangent_ref"], cfg.pwl_points)
        terms.append((max(0.0, float(np.max(states[q] - caps))), ranges[q]))
        terms.append((max(0.0, float(np.max(mu - rows["side"]))), ranges["mu"]))
    if math.isfinite(rows["v_max_user"]):
        terms.append((max(0.0, float(np.max(states["v"] - rows["v_max_user"]))), ranges["v"]))
    e = max([value / range_q for value, range_q in terms], default=0.0)
    raw = max([value for value, _ in terms], default=0.0)
    return e, raw, float(np.max(mu)) / float(cfg.tau[i, l])


def _best_trial(cfg, x, m, r, i: int, l: int, quantities: list[dict], rows: dict, targets: tuple):
    """The intervention that lowers the excess of cell ``(i, l)`` most.

    Over every free day ``k`` of the operating phase (the train has no mission, the
    cell no intervention) and every array of ``targets`` (``m`` for a repair, ``r``
    for a replacement): put the intervention on day ``k`` (the train goes to the
    depot), compute the excess and undo it. The smallest excess wins; a tie goes to
    the array that comes first in ``targets``, then to the later day. Returns
    ``(e, k, target)``, or ``None`` if no day is free.
    """
    best = None
    for k in range(cfg.H1, cfg.T):
        if x[i, 1:, k].sum() > 0.5 or m[i, l, k] > 0.5 or r[i, l, k] > 0.5:
            continue
        for position, target in enumerate(targets):
            depot = x[i, 0, k]
            target[i, l, k] = 1.0
            x[i, 0, k] = 1.0
            e = _cell_excess(cfg, x, m, r, i, l, quantities, rows)[0]
            target[i, l, k] = 0.0
            x[i, 0, k] = depot
            key = (e, position, -k)
            if best is None or key < best[0]:
                best = (key, k, target)
    return None if best is None else (best[0][0], best[1], best[2])


def _gains(trial, e: float) -> bool:
    """Whether a trial lowers the excess ``e`` by more than ``EXCESS_TOLERANCE``."""
    return trial is not None and trial[0] < e - EXCESS_TOLERANCE


def _loop_pass(cfg, x, m, r, profiles: dict):
    """Repairs and replacements in the operating phase, cell by cell (spec, "Steps of the loop pass").

    While the excess ``e`` of a cell exceeds ``EXCESS_TOLERANCE``, one intervention is
    added. A cell with no gaining trial counts one failure. At a cell of the model
    ``wiener``, the repairs and the replacements are tried together (its variance
    falls only at a replacement). At every other cell the repairs are tried first,
    and the replacements only if no repair gains and replacement is allowed.
    Returns ``(failures, maximum raw excess, maximum reference mean over tau)``.
    """
    failures, maximum_excess, maximum_ratio = 0, 0.0, 0.0
    for i in range(cfg.F):
        for l in range(cfg.L):
            quantities = _cell_quantities(cfg, i, l, profiles)
            rows = _cell_rows(cfg, i, l, profiles)
            e = _cell_excess(cfg, x, m, r, i, l, quantities, rows)[0]
            while e > EXCESS_TOLERANCE:
                if str(cfg.model[i, l]) == "wiener":
                    targets = (m, r) if cfg.allow_replacement else (m,)
                    trial = _best_trial(cfg, x, m, r, i, l, quantities, rows, targets)
                else:
                    trial = _best_trial(cfg, x, m, r, i, l, quantities, rows, (m,))
                    if not _gains(trial, e) and cfg.allow_replacement:
                        trial = _best_trial(cfg, x, m, r, i, l, quantities, rows, (r,))
                if not _gains(trial, e):
                    failures += 1
                    break
                e, k, target = trial
                target[i, l, k] = 1.0
                x[i, 0, k] = 1.0
            _, raw, mean_ratio = _cell_excess(cfg, x, m, r, i, l, quantities, rows)
            maximum_excess = max(maximum_excess, raw)
            maximum_ratio = max(maximum_ratio, mean_ratio)
    return failures, maximum_excess, maximum_ratio


# ===========================================================================
# Builder
# ===========================================================================
def build_greedy_warm_start(cfg, *, repair_trigger: float = 0.70) -> tuple[dict, GreedyWarmStartReport]:
    """Build a warm-start source and its report from the input (spec, "Greedy Heuristic").

    ``cfg`` is the normalised configuration (``load_config``). The source holds ``H1``,
    ``H2``, ``x``, ``m``, ``r`` and the request ``complete_binary_start``, so that the
    solver completes the start with the continuous variables. A start with failures is
    returned; the report counts them. Raises ``ValueError`` if ``repair_trigger`` does
    not lie in ``[0, 1]``.
    """
    if not 0.0 <= repair_trigger <= 1.0:
        raise ValueError("repair_trigger must lie in [0, 1]")
    profiles = _profiles(cfg)
    x, m, r, assignment_order, construction_ratio = _construction_pass(
        cfg, profiles["mu"], repair_trigger)
    failures, maximum_excess, loop_ratio = _loop_pass(cfg, x, m, r, profiles)
    source = {
        "H1": cfg.H1,
        "H2": cfg.H2,
        "x": x,
        "m": m,
        "r": r,
        "complete_binary_start": True,
        "completion_time_limit": COMPLETION_TIME_LIMIT,
    }
    report = GreedyWarmStartReport(
        repairs=int(m.sum()),
        replacements=int(r.sum()),
        repeatability_feasible=failures == 0,
        repeatability_failures=failures,
        maximum_repeatability_excess=float(maximum_excess),
        maximum_proxy_ratio=float(max(construction_ratio, loop_ratio)),
        assignment_order=assignment_order,
    )
    return source, report
