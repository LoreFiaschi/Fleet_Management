"""Reference model ("oracle") of the fleet-management specification.

An independent re-implementation, in numpy and scipy, of what a schedule means under
``spec/spec.tex`` (the ``lore`` copy): the state of every component on every day, the
constants of the reliability rows, the repair cost, the damage penalty and the objective.
It is written from the spec only. It reads the input through ``fleet_management.config``
(``load_config`` and ``read_input``) and uses no other function of the package, and none of
the derived fields ``support_const``, ``range_mu``, ``range_v``, ``range_R``, ``range_K``.

Conventions (spec, "Index Sets and Notation"):
* a cell is ``(i, l)``; days are ``k = 1..T``; an array of states has length ``T + 1`` and
  index 0 holds the initial value, so day ``k`` is index ``k``;
* a schedule is ``x`` with shape ``(F, M+1, T)`` (``x[i, 0, k-1]`` is a depot day of train
  ``i``), ``m`` and ``r`` with shape ``(F, L, T)`` (repair, replacement); the day axis of
  ``x``, ``m``, ``r``, ``z`` and of the profiles has length ``T`` and starts at day 1;
* a slack is "allowed value minus value": positive or zero means that the row holds.

Layout for the second task (O2: Gamma and rainflow): the per-model code is dispatched on
the cell kind (``cell_kind(...)["kind"]``) through the dictionaries ``_QUANTITIES`` (the
tracked quantities of a kind, for ``reference_trajectory``) and ``_SLACK`` (the reliability
rows of a kind, for ``reliability_slack``). To add a kind, write one function and register
it in the dictionary; ``constants``, ``initial_and_new`` and ``quadratic_slack`` already
cover every kind.
"""

import collections
import itertools
import math

import numpy as np
from scipy import integrate, optimize, special, stats

# ---------------------------------------------------------------------------
# Cell kinds (spec 1673-1700)
# ---------------------------------------------------------------------------
_KIND_RAINFLOW = {"markov": "Mk", "cantelli": "Ca", "hoeffding": "Ho",
                  "bernstein": "Be", "chernoff": "Ch"}
_TRACKED = {"G": ("mu", "v"), "IG": ("mu",), "W": ("mu", "v"), "Ga": ("mu",),
            "Mk": ("mu",), "Ca": ("mu", "v"), "Ho": ("mu", "R"), "Be": ("mu", "v"),
            "Ch": ("mu", "K")}
# kinds with a quadratic reliability row (spec 865-885); the others have a linear row
_QUADRATIC = ("G", "W", "Ca", "Ho", "Be")


def _nan_to(value, default):
    """``value`` as float, ``default`` where it is NaN (a key that does not apply or is omitted)."""
    value = float(value)
    return default if math.isnan(value) else value


def cell_kind(cfg, i, l):
    """The kind of the cell ``(i, l)`` (spec 1673-1700).

    Returns a dict: ``model``, ``bound`` ('' if not rainflow), ``kind`` (G, IG, W, Ga, Mk, Ca,
    Ho, Be, Ch), ``tracked`` (the tracked quantities in the order of the spec table),
    ``law`` ('ardinf' or 'ard1'), ``impl`` (the form built at the cell: 'exact' at a linear
    row), ``encoding``, ``quadratic`` (True if the row is quadratic).
    """
    model = str(cfg.model[i, l])
    bound = str(cfg.bound_method[i, l]) if model == "rainflow" else ""
    if model == "gaussian":
        kind = "G"
    elif model == "inverse_gaussian":
        kind = "IG"
    elif model == "wiener":
        kind = "W"
    elif model == "gamma":
        kind = "Ga"
    elif model == "rainflow":
        kind = _KIND_RAINFLOW[bound]
    else:
        raise ValueError(f"unknown model {model!r}")
    quadratic = kind in _QUADRATIC
    impl = str(cfg.reliability_impl[i, l]) if quadratic else "exact"
    return {"model": model, "bound": bound, "kind": kind, "tracked": _TRACKED[kind],
            "law": str(cfg.repair_model[i, l]), "impl": impl,
            "encoding": str(cfg.encoding[i, l]), "quadratic": quadratic}


# ---------------------------------------------------------------------------
# Profiles and the phase lookup (spec Eq. c_phase, 438-449)
# ---------------------------------------------------------------------------
def _to_4d(arr, F, L, M, Hs):
    """An (F, L, M, Hs) copy of a profile; a shape without day axis is repeated (spec 457)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:        # (F, L)
        arr = arr[:, :, None, None]
    elif arr.ndim == 3:      # (F, L, M)
        arr = arr[:, :, :, None]
    return np.array(np.broadcast_to(arr, (F, L, M, Hs)))


def day_profile(cfg, key):
    """The profile ``key`` after the phase lookup, array ``(F, L, M, T)``; ``None`` if the key is absent.

    ``key`` is ``mu``, ``v``, ``support``, ``cgf`` or ``gamma_beta``. Day ``k`` (1-based) reads, by
    spec Eq. c_phase: the transitory profile at ``k`` if ``k <= H1`` and it is given; the operating
    profile at ``((k-1) mod H2) + 1`` if ``k <= H1`` and it is not; the operating profile at
    ``((k-H1-1) mod H2) + 1`` if ``k > H1``.
    """
    op = getattr(cfg, key)
    if op is None:
        return None
    F, L, M, H1, H2, T = cfg.F, cfg.L, cfg.M, cfg.H1, cfg.H2, cfg.T
    op = _to_4d(op, F, L, M, H2)
    tr = getattr(cfg, key + "_trans")
    if tr is not None:
        tr = _to_4d(tr, F, L, M, H1)
    out = np.empty((F, L, M, T))
    for k in range(1, T + 1):
        if k <= H1 and tr is not None:
            out[..., k - 1] = tr[..., k - 1]
        elif k <= H1:
            out[..., k - 1] = op[..., (k - 1) % H2]
        else:
            out[..., k - 1] = op[..., (k - H1 - 1) % H2]
    return out


# ---------------------------------------------------------------------------
# Initial, anchor and replacement values (spec 408-428, 774-778, 1389-1396)
# ---------------------------------------------------------------------------
def initial_and_new(cfg):
    """Per cell ``(i, l)``: ``{"init": {q: value}, "anchor0": {q: value}, "new": {q: value}}``.

    ``q`` runs over the tracked quantities of the cell (``mu``, ``v``, ``R``, ``K``).
    Defaults: ``v_0``, ``R_0``, replacement values 0; ``cgf_0 = s mu_0`` and ``cgf_new = s mu_new``;
    a starting anchor equals the initial value. ``replacement_as_new`` sets every replacement
    value to 0 (spec 857-860, 1393).
    """
    out = {}
    as_new = bool(cfg.replacement_as_new)
    for i in range(cfg.F):
        for l in range(cfg.L):
            kind = cell_kind(cfg, i, l)["kind"]
            tracked = _TRACKED[kind]
            s = _nan_to(cfg.s_chernoff[i, l], 0.0)
            init, anchor0, new = {}, {}, {}
            mu0 = float(cfg.mu_0[i, l])
            mu_new = 0.0 if as_new else _nan_to(cfg.replacement_mu[i, l], 0.0)
            for q in tracked:
                if q == "mu":
                    init[q], new[q] = mu0, mu_new
                    anchor0[q] = _nan_to(cfg.anchor_mu_0[i, l], mu0)
                elif q == "v":
                    init[q] = _nan_to(cfg.v_0[i, l], 0.0)
                    new[q] = 0.0 if as_new else _nan_to(cfg.replacement_v[i, l], 0.0)
                    anchor0[q] = _nan_to(cfg.anchor_v_0[i, l], init[q])
                elif q == "R":
                    init[q] = _nan_to(cfg.R_0[i, l], 0.0)
                    new[q] = 0.0 if as_new else _nan_to(cfg.R_new[i, l], 0.0)
                    anchor0[q] = _nan_to(cfg.anchor_R_0[i, l], init[q])
                elif q == "K":
                    init[q] = _nan_to(cfg.cgf_0[i, l], s * mu0)
                    new[q] = 0.0 if as_new else _nan_to(cfg.cgf_new[i, l], s * mu_new)
                    anchor0[q] = _nan_to(cfg.anchor_K_0[i, l], init[q])
            out[(i, l)] = {"init": init, "anchor0": anchor0, "new": new}
    return out


# ---------------------------------------------------------------------------
# Constants (spec 324-343, 862-918, 1022, 1094-1119, 1274-1310, 1399-1405, 1508-1529)
# ---------------------------------------------------------------------------
def ig_cap(eta, tau, eps):
    """Inverse Gaussian mean cap ``mu_bar = tau / E[Z | Z > q]`` (spec 1094-1099), independent method.

    ``Z ~ IG(1, 1/eta)`` is ``scipy.stats.invgauss(mu=eta, scale=1/eta)`` (mean 1, shape 1/eta);
    ``q`` is its ``(1-eps)``-quantile; ``E[Z; Z > q]`` is integrated numerically, and
    ``E[Z | Z > q] = E[Z; Z > q] / eps``.
    """
    law = stats.invgauss(mu=eta, scale=1.0 / eta)
    q = law.isf(eps)
    tail_mean, _ = integrate.quad(lambda z: z * law.pdf(z), q, np.inf, limit=500,
                                  epsabs=1e-14, epsrel=1e-12)
    return tau * eps / tail_mean


def _worst_tail(A, t):
    """``w(A) = max over c in (0, 1] of Q(A/c, t/c)`` (spec 1290-1292): grid on ``log c``, then refined."""
    logc = np.linspace(math.log(1e-8), 0.0, 2000)
    vals = special.gammaincc(A / np.exp(logc), t / np.exp(logc))
    j = int(np.argmax(vals))
    best = float(vals[j])
    lo, hi = logc[max(j - 1, 0)], logc[min(j + 1, len(logc) - 1)]
    if hi > lo:
        res = optimize.minimize_scalar(
            lambda lc: -special.gammaincc(A / math.exp(lc), t / math.exp(lc)),
            bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        best = max(best, -float(res.fun))
    return best


def gamma_cap(t, eps):
    """The sharp Gamma shape cap ``A_1(t, eps)``: the largest ``A < t`` with ``w(A) <= eps`` (spec 1293-1308).

    Independent method: for ``eps >= 1/2`` every ``A < t`` qualifies (``w(A) < 1/2``) and the
    supremum ``t`` is returned. Otherwise ``A^`` is the root of ``Q(A, t) = eps`` (brentq);
    ``w >= Q(., t)`` (take ``c = 1``), so ``A_1 <= min(A^, t)``. If ``A^ < t`` and
    ``w(A^) <= eps`` the answer is ``A^``; if ``A^ >= t`` and ``w(t) <= eps`` it is the supremum
    ``t``; else it is the root of ``w(A) - eps`` on ``[eps t, min(A^, t)]`` (brentq; ``w`` is
    increasing in ``A`` and ``w(eps t) <= eps``, spec 1308).
    """
    if eps >= 0.5:
        return float(t)
    def tail(A):
        return float(special.gammaincc(A, t)) - eps
    hi = max(1.0, 2.0 * t)
    while tail(hi) <= 0.0:
        hi *= 2.0
    a_hat = optimize.brentq(tail, 1e-12, hi, xtol=1e-14, rtol=1e-14)
    top = min(a_hat, float(t))
    if _worst_tail(top, t) <= eps:
        return top
    return optimize.brentq(lambda A: _worst_tail(A, t) - eps, eps * t, top,
                           xtol=1e-14, rtol=1e-13)


def _bernstein_b(cfg, i, l, prof_support):
    """Bernstein support constant ``b`` (spec Eq. rf_bsupport, 1508-1513)."""
    b = _nan_to(cfg.support_0[i, l], 0.0)
    if cfg.allow_replacement and not cfg.replacement_as_new:
        b = max(b, _nan_to(cfg.support_new[i, l], 0.0))
    if prof_support is not None:
        b = max(b, float(np.max(prof_support[i, l])))
    return b


def _beta_star(cfg, i, l, prof_mu, prof_beta, mu_new):
    """The smallest Gamma rate ``beta*`` (spec Eq. gm_betastar) or ``None`` if no term has a rate."""
    rates = []
    if prof_beta is not None:
        mask = prof_mu[i, l] > 0
        rates.extend(prof_beta[i, l][mask].tolist())
    b0 = float(cfg.gamma_beta_0[i, l])
    if not math.isnan(b0) and float(cfg.mu_0[i, l]) > 0:
        rates.append(b0)
    bn = float(cfg.gamma_beta_new[i, l])
    if cfg.allow_replacement and not math.isnan(bn) and mu_new > 0:
        rates.append(bn)
    return min(rates) if rates else None


def constants(cfg):
    """Per cell ``(i, l)`` a dict of the constants of the spec, computed again from the spec.

    Keys (present where they apply): ``tau``, ``eps``, ``rho``; ``kappa_G``, ``V_max`` (G, W);
    ``kappa_C``, ``V_C`` (Ca); ``kappa`` (Ho, Be, Ch); ``b``, ``c``, ``V_B`` (Be); ``s`` (Ch);
    ``M`` (dict of the natural range ``M_s`` of every tracked quantity, spec 832-836);
    ``mu_bar`` (IG cap); ``beta_star``, ``gamma_t``, ``gamma_A1`` and ``mu_cap`` (Ga: ``A1 / beta*``,
    or ``tau`` if no term has a rate); ``mu_cap`` is also the Markov cap ``eps tau``;
    ``K_cap`` (Ch: ``s tau + ln eps``); for a quadratic row ``chi`` (name of the quantity on the left),
    ``c2``, ``c1``, ``side`` (table 875-884).
    """
    prof_mu = day_profile(cfg, "mu")
    prof_support = day_profile(cfg, "support")
    prof_beta = day_profile(cfg, "gamma_beta")
    ini = initial_and_new(cfg)
    out = {}
    for i in range(cfg.F):
        for l in range(cfg.L):
            info = cell_kind(cfg, i, l)
            kind = info["kind"]
            tau, eps, rho = (float(cfg.tau[i, l]), float(cfg.epsilon[i, l]), float(cfg.rho[i, l]))
            d = {"tau": tau, "eps": eps, "rho": rho, "M": {"mu": tau}}
            if kind in ("G", "W"):
                kappa_G = float(stats.norm.isf(eps))          # Phi^{-1}(1 - eps)
                d["kappa_G"] = kappa_G
                d["V_max"] = tau ** 2 / kappa_G ** 2
                d["M"]["v"] = d["V_max"]
                d.update(chi="v", c2=1.0 / kappa_G ** 2, c1=0.0, side=tau)
            if kind in ("Ca", "Ho", "Be", "Ch"):
                d["kappa"] = math.log(1.0 / eps)
            if kind == "Ca":
                d["kappa_C"] = math.sqrt((1.0 - eps) / eps)
                d["V_C"] = eps * tau ** 2 / (1.0 - eps)
                d["M"]["v"] = d["V_C"]
                d.update(chi="v", c2=eps / (1.0 - eps), c1=0.0, side=tau)
            if kind == "Ho":
                d["M"]["R"] = 2.0 * tau ** 2 / d["kappa"]
                d.update(chi="R", c2=2.0 / d["kappa"], c1=0.0, side=tau)
            if kind == "Be":
                kappa = d["kappa"]
                b = _bernstein_b(cfg, i, l, prof_support)
                c = kappa * b / 3.0
                d["b"], d["c"] = b, c
                d["V_B"] = tau * (tau - 2.0 * c) / (2.0 * kappa)
                d["M"]["v"] = d["V_B"]
                d.update(chi="v", c2=1.0 / (2.0 * kappa), c1=-b / 3.0, side=tau - c)
            if kind == "Ch":
                s = float(cfg.s_chernoff[i, l])
                d["s"] = s
                d["M"]["K"] = s * tau - d["kappa"]
                d["K_cap"] = s * tau + math.log(eps)
            if kind == "IG":
                d["mu_bar"] = ig_cap(float(cfg.eta[i, l]), tau, eps)
            if kind == "Mk":
                d["mu_cap"] = eps * tau
            if kind == "Ga":
                bs = _beta_star(cfg, i, l, prof_mu, prof_beta, ini[(i, l)]["new"]["mu"])
                d["beta_star"] = bs
                if bs is None:
                    d["mu_cap"] = tau
                else:
                    d["gamma_t"] = bs * tau
                    d["gamma_A1"] = gamma_cap(bs * tau, eps)
                    d["mu_cap"] = d["gamma_A1"] / bs
            out[(i, l)] = d
    return out


# ---------------------------------------------------------------------------
# Reference trajectory (spec 829, day maps, anchors)
# ---------------------------------------------------------------------------
def _day_increment(x_i, prof_il):
    """``sum_j x_ijk * profile_jk`` for one train and one cell: ``x_i`` is ``(M+1, T)``, ``prof_il`` ``(M, T)``."""
    return np.einsum("jk,jk->k", x_i[1:, :], prof_il)


def _quantities_gaussian(cfg, i, l, ctx):
    """G: ``mu`` (factor ``1-rho``) and ``v`` (factor ``(1-rho)^2``) with anchors (spec Eqs. g_map_mu, g_map_v)."""
    rho = float(cfg.rho[i, l])
    return [
        {"name": "mu", "delta": _day_increment(ctx["x"][i], ctx["prof"]["mu"][i, l]),
         "a": 1.0 - rho, "anchored": True},
        {"name": "v", "delta": _day_increment(ctx["x"][i], ctx["prof"]["v"][i, l]),
         "a": (1.0 - rho) ** 2, "anchored": True},
    ]


def _quantities_inverse_gaussian(cfg, i, l, ctx):
    """IG: ``mu`` only (spec Eq. ig_map)."""
    rho = float(cfg.rho[i, l])
    return [{"name": "mu", "delta": _day_increment(ctx["x"][i], ctx["prof"]["mu"][i, l]),
             "a": 1.0 - rho, "anchored": True}]


def _quantities_wiener(cfg, i, l, ctx):
    """W: ``mu`` as G; ``v`` grows by ``sigma^2`` per mission, a repair keeps it (factor 1, no anchor;
    spec Eq. w_map_v)."""
    rho = float(cfg.rho[i, l])
    sigma2 = float(cfg.sigma[i, l]) ** 2
    missions = ctx["x"][i][1:, :].sum(axis=0)
    return [
        {"name": "mu", "delta": _day_increment(ctx["x"][i], ctx["prof"]["mu"][i, l]),
         "a": 1.0 - rho, "anchored": True},
        {"name": "v", "delta": sigma2 * missions, "a": 1.0, "anchored": False},
    ]


def _quantities_mean_only(cfg, i, l, ctx):
    """Ga and RF Markov: ``mu`` only (spec Eqs. gm_map, rf_mu_*; Markov table row 1359)."""
    rho = float(cfg.rho[i, l])
    return [{"name": "mu", "delta": _day_increment(ctx["x"][i], ctx["prof"]["mu"][i, l]),
             "a": 1.0 - rho, "anchored": True}]


def _quantities_rainflow_variance(cfg, i, l, ctx):
    """RF Cantelli and Bernstein: ``mu`` and ``v``, factor ``(1-rho)^2`` for ``v`` (spec 1381-1387)."""
    rho = float(cfg.rho[i, l])
    return _quantities_mean_only(cfg, i, l, ctx) + [
        {"name": "v", "delta": _day_increment(ctx["x"][i], ctx["prof"]["v"][i, l]),
         "a": (1.0 - rho) ** 2, "anchored": True}]


def _quantities_rainflow_hoeffding(cfg, i, l, ctx):
    """RF Hoeffding: ``mu`` and ``R``; ``delta R = sum_j x_ijk (b_jk)^2`` (spec 1377), factor ``(1-rho)^2``."""
    rho = float(cfg.rho[i, l])
    return _quantities_mean_only(cfg, i, l, ctx) + [
        {"name": "R", "delta": _day_increment(ctx["x"][i], ctx["prof"]["support"][i, l] ** 2),
         "a": (1.0 - rho) ** 2, "anchored": True}]


def _quantities_rainflow_chernoff(cfg, i, l, ctx):
    """RF Chernoff: ``mu`` and ``K``; ``delta K = sum_j x_ijk K_jk`` (spec 1378), factor ``1-rho``."""
    rho = float(cfg.rho[i, l])
    return _quantities_mean_only(cfg, i, l, ctx) + [
        {"name": "K", "delta": _day_increment(ctx["x"][i], ctx["prof"]["cgf"][i, l]),
         "a": 1.0 - rho, "anchored": True}]


# kind -> function(cfg, i, l, ctx) returning the list of tracked quantities of the cell.
_QUANTITIES = {"G": _quantities_gaussian, "IG": _quantities_inverse_gaussian,
               "W": _quantities_wiener, "Ga": _quantities_mean_only,
               "Mk": _quantities_mean_only, "Ca": _quantities_rainflow_variance,
               "Be": _quantities_rainflow_variance, "Ho": _quantities_rainflow_hoeffding,
               "Ch": _quantities_rainflow_chernoff}

_PROFILE_KEYS = ("mu", "v", "support", "cgf", "gamma_beta")


def _profiles(cfg):
    """The day profiles of the config (``day_profile`` for every key), kept on the config object."""
    cache = cfg.__dict__.get("_oracle_profiles")
    if cache is None:
        cache = {key: day_profile(cfg, key) for key in _PROFILE_KEYS}
        cfg.__dict__["_oracle_profiles"] = cache
    return cache


def _advance(s_prev, g_prev, delta, a, event, anchored, law, s_new):
    """One day of one tracked quantity (spec Eqs. a_psi, a_anchor, with every inequality at equality).

    Returns ``(s_k, g_k)``; ``event`` is 'carry', 'repair' or 'replace'.
    """
    if event == "replace":
        s = s_new
    elif event == "repair":
        s = a * s_prev
        if law == "ard1" and anchored:
            s += (1.0 - a) * g_prev
    else:
        s = s_prev + delta
    g = g_prev if event == "carry" else s
    return s, g


def reference_trajectory(cfg, x, m, r=None, profiles=None):
    """The reference trajectory of a schedule: every dynamics row of the spec at equality.

    ``x`` is ``(F, M+1, T)``, ``m`` and ``r`` are ``(F, L, T)`` (``r`` may be ``None``: no
    replacement). Returns a dict of arrays ``(F, L, T+1)``: ``mu``, ``v``, ``R``, ``K``, ``g_mu``,
    ``g_v``, ``g_R``, ``g_K`` (NaN where the cell does not track the quantity, and for the
    anchors at ``ardinf`` cells and at the Wiener variance). Index 0 holds the initial values
    and the starting anchors. The horizon is the day axis of ``x``; ``profiles`` (a dict of
    day profiles of that length, as ``day_profile`` gives) replaces the profiles of the config,
    for a schedule that is longer than ``cfg.T`` (used by ``repeated_safety``).
    """
    F, L = cfg.F, cfg.L
    x = np.asarray(x)
    T = x.shape[-1]
    m = np.asarray(m)
    r = np.zeros_like(m) if r is None else np.asarray(r)
    ini = initial_and_new(cfg)
    ctx = {"x": x, "prof": _profiles(cfg) if profiles is None else profiles}
    out = {name: np.full((F, L, T + 1), np.nan) for name in
           ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K")}
    for i in range(F):
        for l in range(L):
            info = cell_kind(cfg, i, l)
            quantities = _QUANTITIES[info["kind"]](cfg, i, l, ctx)
            events = []
            for k in range(T):
                if m[i, l, k] and r[i, l, k]:
                    raise ValueError(f"repair and replacement of cell {(i, l)} on day {k + 1}")
                events.append("replace" if r[i, l, k] else "repair" if m[i, l, k] else "carry")
            for q in quantities:
                name = q["name"]
                s = ini[(i, l)]["init"][name]
                g = ini[(i, l)]["anchor0"][name]
                track_anchor = q["anchored"] and info["law"] == "ard1"
                out[name][i, l, 0] = s
                if track_anchor:
                    out["g_" + name][i, l, 0] = g
                for k in range(T):
                    s, g = _advance(s, g, q["delta"][k], q["a"], events[k], q["anchored"],
                                    info["law"], ini[(i, l)]["new"][name])
                    out[name][i, l, k + 1] = s
                    if track_anchor:
                        out["g_" + name][i, l, k + 1] = g
    return out


# ---------------------------------------------------------------------------
# Reliability rows (spec 862-922 and the model sections)
# ---------------------------------------------------------------------------
def quadratic_slack(c2, c1, side, tau, chi, mu, impl="exact", tangent_ref=0.5, P=8):
    """Smallest slack of a quadratic reliability row ``chi <= f(mu)`` and its side row ``mu <= side``.

    ``f(mu) = c2 (tau - mu)^2 + c1 (tau - mu)``, ``f'(mu) = -2 c2 (tau - mu) - c1`` (spec Eq. a_quad).
    Forms: ``exact``: ``f(mu) - chi``; ``tangent``: the tangent at ``mu_r = tangent_ref tau``
    (Eq. a_tangent); ``pwl``: ``P`` equal segments of ``[0, tau]``, midpoints
    ``(p - 1/2) tau / P``; the cap is the largest among the segments that contain ``mu``
    (Eqs. a_pwl_sum to a_pwl_cap; ``-inf`` if no segment contains ``mu``); ``P = 1`` is the
    tangent at ``tau / 2``. Returns the smaller of the row slack and ``side - mu``.
    """
    def f(u):
        return c2 * (tau - u) ** 2 + c1 * (tau - u)

    def fp(u):
        return -2.0 * c2 * (tau - u) - c1

    if impl == "exact":
        row = f(mu) - chi
    elif impl == "tangent":
        mu_r = tangent_ref * tau
        row = f(mu_r) + fp(mu_r) * (mu - mu_r) - chi
    elif impl == "pwl":
        row = -math.inf
        tol = 1e-12 * max(1.0, abs(tau))
        for p in range(1, P + 1):
            if (p - 1) * tau / P - tol <= mu <= p * tau / P + tol:
                mid = (p - 0.5) * tau / P
                row = max(row, f(mid) + fp(mid) * (mu - mid) - chi)
    else:
        raise ValueError(f"unknown reliability form {impl!r}")
    return min(row, side - mu)


def _v_user_slack(cfg, i, l, state):
    """Slack of the optional row ``v <= v_max_user`` (``inf`` if not given)."""
    vmax = float(cfg.v_max_user[i, l]) if cfg.v_max_user is not None else math.nan
    if math.isnan(vmax) or "v" not in state:
        return math.inf
    return vmax - state["v"]


def _tangent_ref(cfg, i, l):
    return _nan_to(cfg.tangent_ref[i, l], 0.5)


def _constants_cached(cfg):
    """``constants(cfg)``, kept on the config object (build a new config after changing a field)."""
    cache = cfg.__dict__.get("_oracle_constants")
    if cache is None:
        cache = constants(cfg)
        cfg.__dict__["_oracle_constants"] = cache
    return cache


def _slack_quadratic(cfg, i, l, state):
    """G, W (and, for O2, Ca, Ho, Be): the quadratic row in the form built at the cell, the side
    row, and ``v <= v_max_user``."""
    info = cell_kind(cfg, i, l)
    const = _constants_cached(cfg)[(i, l)]
    row = quadratic_slack(const["c2"], const["c1"], const["side"], const["tau"],
                          state[const["chi"]], state["mu"], impl=info["impl"],
                          tangent_ref=_tangent_ref(cfg, i, l), P=int(cfg.pwl_points))
    return min(row, _v_user_slack(cfg, i, l, state))


def _slack_inverse_gaussian(cfg, i, l, state):
    """IG: ``mu <= mu_bar`` (spec Eq. ig_row); a linear row is always exact."""
    return _constants_cached(cfg)[(i, l)]["mu_bar"] - state["mu"]


def _slack_mean_cap(cfg, i, l, state):
    """Ga: ``mu <= A_1(beta* tau, eps) / beta*`` (spec Eq. gm_row; ``mu <= tau`` if no term has a
    rate, 1282-1283); RF Markov: ``mu <= eps tau`` (Eq. rf_markov). Linear rows, always exact."""
    return _constants_cached(cfg)[(i, l)]["mu_cap"] - state["mu"]


def _slack_chernoff(cfg, i, l, state):
    """RF Chernoff: ``K - s tau <= ln eps`` (spec Eq. rf_chernoff); a linear row, always exact."""
    const = _constants_cached(cfg)[(i, l)]
    return const["K_cap"] - state["K"]


# kind -> function(cfg, i, l, state) returning the smallest slack of the day.
# Ca, Ho and Be use the generic quadratic row (table 875-884; ``constants`` gives chi, c2, c1, side).
_SLACK = {"G": _slack_quadratic, "W": _slack_quadratic, "IG": _slack_inverse_gaussian,
          "Ga": _slack_mean_cap, "Mk": _slack_mean_cap, "Ca": _slack_quadratic,
          "Ho": _slack_quadratic, "Be": _slack_quadratic, "Ch": _slack_chernoff}


def reliability_slack(cfg, i, l, state):
    """Smallest slack of the reliability rows of the cell ``(i, l)`` on one day.

    ``state`` maps the tracked quantities of the day (``mu``, ``v``, ``R``, ``K``) to floats.
    The reliability row is taken in the form built at the cell, together with its side row and
    ``v <= v_max_user`` where given. Zero or positive: every row holds.
    """
    kind = cell_kind(cfg, i, l)["kind"]
    if kind not in _SLACK:
        raise NotImplementedError(f"reliability_slack: kind {kind!r} (task O2)")
    return _SLACK[kind](cfg, i, l, state)


# ---------------------------------------------------------------------------
# Repair cost, damage penalty, costs (spec 653-695, 1562-1599, 1756-1783, 1897-1902)
# ---------------------------------------------------------------------------
def minimal_z(cfg, traj, m):
    """The smallest ``z`` that satisfies Eq. a_z, array ``(F, L, T)``.

    ``max(0, rho (mu_{k-1} - g_{k-1}))`` on a repair day, 0 elsewhere (replacement days included);
    ``g = 0`` under ``ardinf``; NaN at a cell with ``C_R = 0``.
    """
    m = np.asarray(m)
    z = np.zeros((cfg.F, cfg.L, cfg.T))
    for i in range(cfg.F):
        for l in range(cfg.L):
            if float(cfg.C_R[i, l]) == 0.0:
                z[i, l, :] = np.nan
                continue
            law = str(cfg.repair_model[i, l])
            for k in range(1, cfg.T + 1):
                if m[i, l, k - 1]:
                    g = traj["g_mu"][i, l, k - 1] if law == "ard1" else 0.0
                    z[i, l, k - 1] = max(0.0, float(cfg.rho[i, l]) *
                                         (traj["mu"][i, l, k - 1] - g))
    return z


def u_value(cfg, mu):
    """The damage penalty value ``u`` (spec Eq. c_u): ``max_{i,k} sum_l mu`` (``inf_norm``) or
    ``max_{i,k} sum_l mu^2`` (``quadratic``), over the days 1..T.

    ``mu`` has shape ``(F, L, T)``; an array ``(F, L, T+1)`` has its index-0 (initial) values dropped.
    """
    mu = np.asarray(mu, dtype=float)
    if mu.shape[-1] == cfg.T + 1:
        mu = mu[..., 1:]
    if str(cfg.penalty_type) == "quadratic":
        per_train_day = (mu ** 2).sum(axis=1)
    else:
        per_train_day = mu.sum(axis=1)
    return float(per_train_day.max())


def costs(cfg, x, r, z, u):
    """The cost report of the spec on a schedule: returns a dict.

    ``step_costs`` (``c_k``, Eq. a_daycost), ``J_initialization``, ``J_op`` (Eq. a_phases),
    ``J_op_average = J_op / H2``, ``peak_damage = u``, ``damage_penalty = C_D u``,
    ``operating_objective = J_op_average + C_D u``, ``J_total = J_init + J_op + C_D u``,
    ``projected_evaluation_cost`` (``P_E`` of Eq. d_projected) and ``evaluation_objective``
    (``P_E + C_D u``; ``None`` if no evaluation horizon is given), ``objective`` (the value
    of the mode of ``cfg``, Eq. a_objective). ``z`` is ``(F, L, T)`` with ``z = 0`` where it is NaN;
    ``r`` may be ``None``. The term ``C_D u`` is present only if ``C_D > 0``.
    """
    x = np.asarray(x, dtype=float)
    z = np.nan_to_num(np.asarray(z, dtype=float), nan=0.0)
    r = np.zeros((cfg.F, cfg.L, cfg.T)) if (r is None or not cfg.allow_replacement) \
        else np.asarray(r, dtype=float)
    H1, H2, T = cfg.H1, cfg.H2, cfg.T
    depot = x[:, 0, :].sum(axis=0)                                  # per day
    repair = (np.asarray(cfg.C_R, dtype=float)[:, :, None] * z).sum(axis=(0, 1))
    replace = (np.asarray(cfg.C_rep, dtype=float)[:, :, None] * r).sum(axis=(0, 1))
    step = float(cfg.C_M) * depot + repair + replace
    J_init = float(step[:H1].sum())
    J_op = float(step[H1:].sum())
    C_D = float(cfg.C_D)
    penalty = C_D * u if C_D > 0 else 0.0
    E = cfg.options.get("evaluation_horizon")
    out = {"step_costs": step, "J_initialization": J_init, "J_op": J_op,
           "J_op_average": J_op / H2, "peak_damage": u, "damage_penalty": penalty,
           "operating_objective": J_op / H2 + penalty, "J_total": J_init + J_op + penalty,
           "projected_evaluation_cost": None, "evaluation_objective": None}
    if E is not None:
        P_E = J_init + (E - H1) * (J_op / H2)
        out["projected_evaluation_cost"] = P_E
        out["evaluation_objective"] = P_E + penalty
    mode = cfg.options.get("objective_mode", "total")
    if mode == "total":
        out["objective"] = out["J_total"]
    elif mode == "operating_average":
        out["objective"] = out["operating_objective"]
    elif mode == "evaluation_total":
        out["objective"] = J_init + (E - H1) / H2 * J_op + penalty
    else:
        raise ValueError(f"unknown objective_mode {mode!r}")
    return out


# ---------------------------------------------------------------------------
# Enumeration of schedules (spec 704-732)
# ---------------------------------------------------------------------------
def _day_options(cfg):
    """Every assignment of one day: list of ``(x_day (F, M+1), m_day (F, L), r_day (F, L))``.

    Each mission is flown by exactly one train, a train flies at most one mission (Eqs.
    a_coverage, a_one_activity). A train that does not fly is idle, or in the depot; on a depot
    day each component gets no intervention, a repair or (if ``allow_replacement``) a replacement
    (Eq. a_depot). An idle day and a depot day without intervention are both listed.
    """
    F, M, L = cfg.F, cfg.M, cfg.L
    per_comp = ("none", "repair", "replace") if cfg.allow_replacement else ("none", "repair")
    options = []
    for flyers in itertools.permutations(range(F), M):       # flyers[j-1] flies mission j
        others = [i for i in range(F) if i not in flyers]
        # one state per non-flying train: 'idle' or a tuple of per-component choices
        states = ["idle"] + list(itertools.product(per_comp, repeat=L))
        for combo in itertools.product(states, repeat=len(others)):
            xd = np.zeros((F, M + 1), dtype=np.int8)
            md = np.zeros((F, L), dtype=np.int8)
            rd = np.zeros((F, L), dtype=np.int8)
            for j, i in enumerate(flyers, start=1):
                xd[i, j] = 1
            for i, state in zip(others, combo):
                if state == "idle":
                    continue
                xd[i, 0] = 1
                for l, choice in enumerate(state):
                    if choice == "repair":
                        md[i, l] = 1
                    elif choice == "replace":
                        rd[i, l] = 1
            options.append((xd, md, rd))
    return options


def enumerate_schedules(cfg):
    """Generate every binary schedule ``(x, m, r)`` of the horizon that satisfies the assignment rows
    (one activity, coverage, depot-day compatibility); arrays ``(F, M+1, T)``, ``(F, L, T)``,
    ``(F, L, T)``. ``r`` is all zero if ``allow_replacement`` is false."""
    options = _day_options(cfg)
    for combo in itertools.product(range(len(options)), repeat=cfg.T):
        x = np.stack([options[c][0] for c in combo], axis=-1)
        m = np.stack([options[c][1] for c in combo], axis=-1)
        r = np.stack([options[c][2] for c in combo], axis=-1)
        yield x, m, r


# ---------------------------------------------------------------------------
# Rows of the spec at a given solution (spec 698-1625)
# ---------------------------------------------------------------------------
Violation = collections.namedtuple("Violation", "row i l k violation")


def _events(m_cell, r_cell):
    """The event of each day of one cell: 'carry', 'repair' or 'replace' (binaries read at 1/2)."""
    return ["replace" if r_cell[k] > 0.5 else "repair" if m_cell[k] > 0.5 else "carry"
            for k in range(len(m_cell))]


def row_violations(cfg, sol, tol=1e-6, bigm_tol=2e-5, quad_rel=1e-5):
    """Every row of the spec evaluated at the solution ``sol``; the list of the rows that fail.

    ``sol`` is a mapping with the keys of the result dictionary: ``x`` ``(F, M+1, T)``, ``m``, ``r``
    (``r`` may be absent or ``None``), the tracked arrays ``mu``, ``v``, ``R``, ``K``, ``g_mu``,
    ``g_v``, ``g_R``, ``g_K`` and ``z``, all ``(F, L, T)`` for the days ``1..T`` (day 0 is taken
    from the inputs), and optionally ``u``. Binaries are read at 1/2 to choose the event of a day.
    Each record is ``Violation(row, i, l, k, violation)`` with ``k`` the 1-based day and
    ``violation`` the amount by which the row fails (a NaN counts as a failure).

    Rows, with their spec location: ``one_activity`` (Eq. a_one_activity), ``coverage_j<j>``
    (a_coverage), ``depot`` (a_depot); per tracked quantity ``<q>``: ``<q>_carry``, ``<q>_repair``
    (the one-sided rows of the day maps, Eq. a_psi and the model sections), ``<q>_replace``
    (equality, a_bigm_repl); under ard1 ``g_<q>_set`` and ``g_<q>_keep`` (a_anchor); ``reliability``
    (the row of the cell in the form built at the cell, its side row and ``v <= v_max_user``, via
    ``reliability_slack``); ``<q>_loop`` and ``g_<q>_loop`` (Eq. a_loop, record day ``T``);
    ``z_row`` (Eq. a_z, where ``C_R > 0``, and in mode ``operating_average`` only on operating
    days) and ``z_nonneg``; ``u_row`` (Eqs. a_u_inf, a_u_quad, if ``u`` is given and ``C_D > 0``).

    Tolerance of a row: ``tol`` plus ``bigm_tol * M_s`` for a row with a big-M constant ``M_s``
    (the natural range of the quantity; ``rho tau`` for ``z_row``), because the integrality
    tolerance of the solver lets a binary leak; a linear reliability row has ``tol``; an exact
    quadratic reliability row has ``tol + quad_rel * M_chi``; a tangent or pwl row
    ``tol + bigm_tol * M_chi`` (``M_chi`` the range of the quantity ``chi`` on the left).
    """
    F, L, M, T, H1 = cfg.F, cfg.L, cfg.M, cfg.T, cfg.H1
    x = np.asarray(sol["x"], dtype=float)
    m = np.asarray(sol["m"], dtype=float)
    r = sol.get("r")
    r = np.zeros_like(m) if r is None else np.asarray(r, dtype=float)
    out = []

    def add(row, i, l, k, violation, allowed):
        if not (violation <= allowed):          # also true for NaN
            out.append(Violation(row, i, l, k, float(violation)))

    # --- fleet rows
    for k in range(1, T + 1):
        for i in range(F):
            add("one_activity", i, None, k, x[i, :, k - 1].sum() - 1.0, tol)
        for j in range(1, M + 1):
            add(f"coverage_j{j}", None, None, k, abs(x[:, j, k - 1].sum() - 1.0), tol)
        for i in range(F):
            for l in range(L):
                add("depot", i, l, k, m[i, l, k - 1] + r[i, l, k - 1] - x[i, 0, k - 1], tol)

    # --- rows of the cells
    const = _constants_cached(cfg)
    ini = initial_and_new(cfg)
    ctx = {"x": x, "prof": _profiles(cfg)}
    mode = cfg.options.get("objective_mode", "total")
    for i in range(F):
        for l in range(L):
            info = cell_kind(cfg, i, l)
            c = const[(i, l)]
            law = info["law"]
            events = _events(m[i, l], r[i, l])
            quantities = _QUANTITIES[info["kind"]](cfg, i, l, ctx)
            for q in quantities:
                name = q["name"]
                Ms = c["M"][name]
                allowed = tol + bigm_tol * Ms
                s_arr = np.asarray(sol[name], dtype=float)[i, l]
                anchored = q["anchored"] and law == "ard1"
                g_arr = np.asarray(sol["g_" + name], dtype=float)[i, l] if anchored else None
                prev = ini[(i, l)]["init"][name]
                gprev = ini[(i, l)]["anchor0"][name]
                new = ini[(i, l)]["new"][name]
                for k in range(1, T + 1):
                    s, ev = s_arr[k - 1], events[k - 1]
                    if ev == "replace":
                        add(f"{name}_replace", i, l, k, abs(s - new), allowed)
                    elif ev == "repair":
                        psi = q["a"] * prev + ((1.0 - q["a"]) * gprev if anchored else 0.0)
                        add(f"{name}_repair", i, l, k, psi - s, allowed)
                    else:
                        add(f"{name}_carry", i, l, k, prev + q["delta"][k - 1] - s, allowed)
                    if anchored:
                        g = g_arr[k - 1]
                        if ev == "carry":
                            add(f"g_{name}_keep", i, l, k, abs(g - gprev), allowed)
                        else:
                            add(f"g_{name}_set", i, l, k, abs(g - s), allowed)
                        gprev = g
                    prev = s
                # loop rows (Eq. a_loop)
                add(f"{name}_loop", i, l, T, s_arr[T - 1] - s_arr[H1 - 1], allowed)
                if anchored:
                    add(f"g_{name}_loop", i, l, T, g_arr[T - 1] - g_arr[H1 - 1], allowed)
            # reliability row, its side row and the optional row
            if info["quadratic"]:
                M_chi = c["M"][c["chi"]]
                scale = quad_rel if info["impl"] == "exact" else bigm_tol
                allowed = tol + scale * M_chi
            else:
                allowed = tol
            for k in range(1, T + 1):
                state = {name: float(np.asarray(sol[name], dtype=float)[i, l, k - 1])
                         for name in info["tracked"]}
                add("reliability", i, l, k, -reliability_slack(cfg, i, l, state), allowed)
            # repair cost rows (Eq. a_z)
            if float(cfg.C_R[i, l]) > 0.0 and sol.get("z") is not None:
                z_arr = np.asarray(sol["z"], dtype=float)[i, l]
                rho, tau = c["rho"], c["tau"]
                mu_arr = np.asarray(sol["mu"], dtype=float)[i, l]
                g_arr = np.asarray(sol["g_mu"], dtype=float)[i, l] if law == "ard1" else None
                for k in range(1, T + 1):
                    if mode == "operating_average" and k <= H1:
                        continue
                    mu_prev = ini[(i, l)]["init"]["mu"] if k == 1 else mu_arr[k - 2]
                    if law == "ard1":
                        g_prev = ini[(i, l)]["anchor0"]["mu"] if k == 1 else g_arr[k - 2]
                    else:
                        g_prev = 0.0
                    rhs = rho * (mu_prev - g_prev) - rho * tau * (1.0 - float(m[i, l, k - 1]))
                    add("z_row", i, l, k, rhs - z_arr[k - 1], tol + bigm_tol * rho * tau)
                    add("z_nonneg", i, l, k, -z_arr[k - 1], tol)

    # --- damage penalty rows (Eqs. a_u_inf, a_u_quad)
    if sol.get("u") is not None and float(cfg.C_D) > 0.0:
        mu_all = np.asarray(sol["mu"], dtype=float)
        per = (mu_all ** 2).sum(axis=1) if str(cfg.penalty_type) == "quadratic" else mu_all.sum(axis=1)
        for i in range(F):
            for k in range(1, T + 1):
                add("u_row", i, None, k, per[i, k - 1] - float(sol["u"]), tol)
    return out


def repeated_safety(cfg, x, m, r, cycles):
    """The smallest reliability slack over all days of the schedule repeated ``cycles`` times.

    Spec 1612-1617: with the loop rows, "every day of every repetition of the operating phase
    satisfies the reliability requirement". The schedule is extended by repeating its days
    ``H1+1..T`` ``cycles`` times (the profiles of the operating phase continue with period
    ``H2``, so the extended profiles repeat their days ``H1+1..T`` the same way); the reference
    trajectory is computed over the extended horizon and the smallest ``reliability_slack`` over
    all cells and all days is returned. The loop rows are not applied to the extension.
    """
    H1 = cfg.H1

    def extend(a):
        a = np.asarray(a)
        return np.concatenate([a] + [a[..., H1:]] * cycles, axis=-1)

    profiles = {key: (None if val is None else extend(val)) for key, val in _profiles(cfg).items()}
    traj = reference_trajectory(cfg, extend(x), extend(m),
                                None if r is None else extend(r), profiles=profiles)
    best = math.inf
    T_ext = traj["mu"].shape[-1] - 1
    for i in range(cfg.F):
        for l in range(cfg.L):
            tracked = cell_kind(cfg, i, l)["tracked"]
            for k in range(1, T_ext + 1):
                state = {name: float(traj[name][i, l, k]) for name in tracked}
                best = min(best, reliability_slack(cfg, i, l, state))
    return best


def best_reference(cfg, tol=1e-9):
    """The smallest cost over the schedules whose reference trajectory breaks no row.

    Every schedule of ``enumerate_schedules(cfg)`` is taken with its reference trajectory, the
    smallest ``z`` (``minimal_z``) and ``u = u_value`` of the reference mean; it is feasible if
    ``row_violations(cfg, ..., tol, bigm_tol=0, quad_rel=0)`` is empty (exact rows, no leak). The
    cost is ``costs(...)["objective"]``. Returns a dict: ``cost`` (``None`` if no schedule is
    feasible), ``schedule`` (the first schedule of that cost, ``(x, m, r)``), ``n_feasible``,
    ``n_total`` and ``costs`` (the list of the costs of the feasible schedules).
    """
    best_cost, best_schedule, n_total, feasible_costs = None, None, 0, []
    for x, m, r in enumerate_schedules(cfg):
        n_total += 1
        traj = reference_trajectory(cfg, x, m, r)
        z = minimal_z(cfg, traj, m)
        u = u_value(cfg, traj["mu"])
        sol = {"x": x, "m": m, "r": r, "z": z, "u": u}
        for name in ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K"):
            sol[name] = traj[name][..., 1:]
        if row_violations(cfg, sol, tol=tol, bigm_tol=0.0, quad_rel=0.0):
            continue
        cost = costs(cfg, x, r, z, u)["objective"]
        feasible_costs.append(cost)
        if best_cost is None or cost < best_cost - 1e-12:
            best_cost, best_schedule = cost, (x, m, r)
    return {"cost": best_cost, "schedule": best_schedule, "n_feasible": len(feasible_costs),
            "n_total": n_total, "costs": feasible_costs}


# --- O5: least one-sided solution ---
# Spec 780-836 (the dynamics rows are one-sided, ``s_k >= map(s_{k-1})``), 1602-1617 (loop rows),
# 862-922 (reliability rows), 1562-1594 (z and u).
#
# Why the least solution decides feasibility. Fix a schedule. Every row that bounds a continuous
# tracked quantity from below is monotone (non-decreasing) in the other quantities:
#   * carry:  s_k >= s_{k-1} + delta s_k;  repair:  s_k >= a s_{k-1} + (1 - a) g_{k-1}  (a in [0, 1],
#     Eq. a_psi; for the Wiener variance a = 1, Eq. w_vrow);  replacement:  s_k >= s_new  (the half of
#     the equality of Eq. a_bigm_repl that is a lower bound);
#   * anchors (Eq. a_anchor): ``g_k = s_k`` on an intervention day and ``g_k = g_{k-1}`` otherwise, read
#     as the two lower bounds ``g_k >= s_k`` and ``s_k >= g_k``, respectively ``g_k >= g_{k-1}`` and
#     ``g_{k-1} >= g_k``;
#   * loop (Eq. a_loop):  s_{H1} >= s_T  for every tracked quantity and every anchor.
# A set of rows of the form ``y_j >= f_j(y)`` with every ``f_j`` non-decreasing and the initial values
# fixed has a least solution (the Knaster-Tarski fixed point, reached by Kleene iteration from any start
# below it, here the exact trajectory); every solution of the set lies above the least one. A lower bound
# on a fixed initial value (index 0) that exceeds it is a violation of an equality: "no solution".
# The remaining rows are upper bounds on the tracked quantities: the other half of the replacement
# equality (``s_k <= s_new``), the reliability rows, ``v <= v_max_user`` and the domains ``0 <= s <= M_s``.
# Each reliability row is a cap on the mean (``mu <= cap``, ``mu <= side``) or a bound ``chi <= f(mu)``
# of a second quantity ``chi`` by a function ``f`` that is decreasing in ``mu`` (spec 862-922): the slack
# is decreasing in ``mu`` and in ``chi``, so the row is downward closed (lowering ``mu`` and ``chi``
# keeps it true; for the tangent and pwl forms ``f`` is also decreasing in ``mu``, on ``[0, tau]``).
# If a one-sided solution ``y`` of the whole system exists, the least solution ``y*`` of the lower-bound
# rows satisfies ``y* <= y`` componentwise, so ``y*`` satisfies every downward-closed row that ``y`` does.
# Hence the least solution is feasible exactly when the schedule is feasible under the one-sided rows.
# The costs ``z`` and ``u`` are free upwards and do not decide feasibility.
_LEAST_DOMAIN_TOL = 1e-9


def _least_quantity(s, g, q, events, law, new, Ms, H1, T, max_passes, tol):
    """Kleene iteration for one tracked quantity of one cell. ``s`` and ``g`` (or ``None``) are arrays of
    length ``T + 1`` that start at the exact trajectory and are raised in place. Returns ``(ok, reason)``."""
    a, delta = q["a"], q["delta"]
    anchored = g is not None
    lim = Ms + _LEAST_DOMAIN_TOL
    state = {"max": 0.0}

    def rise(arr, k, val):
        if k == 0:                      # fixed initial value: a larger lower bound is a violation
            return val <= arr[0] + _LEAST_DOMAIN_TOL
        if val > arr[k]:
            state["max"] = max(state["max"], val - arr[k])
            arr[k] = val
            return val <= lim
        return True

    if np.max(s) > lim or (anchored and np.max(g) > lim):
        return False, "initial value above the natural range"
    for _ in range(max_passes):
        state["max"] = 0.0
        for k in range(1, T + 1):
            ev = events[k - 1]
            if ev == "replace":
                ok = rise(s, k, new)
            elif ev == "repair":
                ok = rise(s, k, a * s[k - 1] + ((1.0 - a) * g[k - 1] if anchored else 0.0))
            else:
                ok = rise(s, k, s[k - 1] + delta[k - 1])
            if not ok:
                return False, f"day {k}: lower bound above the range"
            if anchored:
                if ev == "carry":
                    ok = rise(g, k, g[k - 1]) and rise(g, k - 1, g[k])
                else:
                    ok = rise(g, k, s[k]) and rise(s, k, g[k])
                if not ok:
                    return False, f"day {k}: anchor row"
        if not rise(s, H1, s[T]) or (anchored and not rise(g, H1, g[T])):
            return False, "loop row"
        for k in range(T, 0, -1):       # anchors backwards along the carry days
            if anchored and events[k - 1] == "carry" and not rise(g, k - 1, g[k]):
                return False, f"day {k}: anchor row"
        if state["max"] <= tol:
            return True, ""
    return False, "max_passes reached"


def _least_cell(cfg, i, l, x, m, r, max_passes, tol):
    """The least one-sided solution of the cell ``(i, l)`` for the schedule (cached on the config by the
    train's activity row and the cell's binaries). Returns a dict: ``ok``, ``reason``, ``s`` (``name ->
    array (T+1,)``), ``g`` (same, ``None`` where no anchor), ``z`` (``(T,)``, NaN if ``C_R = 0``)."""
    cache = cfg.__dict__.setdefault("_oracle_least_cache", {})
    key = (i, l, x[i].tobytes(), m[i, l].tobytes(), r[i, l].tobytes(), max_passes, tol)
    hit = cache.get(key)
    if hit is not None:
        return hit
    T = x.shape[-1]
    H1 = cfg.H1
    info = cell_kind(cfg, i, l)
    law = info["law"]
    c = _constants_cached(cfg)[(i, l)]
    ini = initial_and_new(cfg)[(i, l)]
    ctx = {"x": x, "prof": _profiles(cfg)}
    events = _events(m[i, l], r[i, l])
    res = {"ok": True, "reason": "", "s": {}, "g": {}, "z": None}
    for q in _QUANTITIES[info["kind"]](cfg, i, l, ctx):
        name = q["name"]
        anchored = q["anchored"] and law == "ard1"
        s = np.empty(T + 1)
        g = np.empty(T + 1) if anchored else None
        s[0] = ini["init"][name]
        if anchored:
            g[0] = ini["anchor0"][name]
        sk, gk = s[0], (g[0] if anchored else 0.0)
        for k in range(T):              # exact trajectory as the start
            sk, gk = _advance(sk, gk, q["delta"][k], q["a"], events[k], q["anchored"], law, ini["new"][name])
            s[k + 1] = sk
            if anchored:
                g[k + 1] = gk
        ok, reason = _least_quantity(s, g, q, events, law, ini["new"][name], c["M"][name], H1, T,
                                     max_passes, tol)
        res["s"][name], res["g"][name] = s, g
        if not ok:
            res["ok"], res["reason"] = False, f"{name}: {reason}"
            break
    if res["ok"]:                       # upper-bound rows and domains
        eps = _LEAST_DOMAIN_TOL
        for name, s in res["s"].items():
            g = res["g"][name]
            Ms = c["M"][name]
            if np.min(s) < -eps or np.max(s) > Ms + eps or \
                    (g is not None and (np.min(g) < -eps or np.max(g) > Ms + eps)):
                res["ok"], res["reason"] = False, f"{name}: domain"
            for k in range(1, T + 1):
                if events[k - 1] == "replace" and abs(s[k] - ini["new"][name]) > eps:
                    res["ok"], res["reason"] = False, f"{name}: replacement value, day {k}"
        for k in range(1, T + 1):
            state = {name: float(s[k]) for name, s in res["s"].items()}
            if reliability_slack(cfg, i, l, state) < -eps:
                res["ok"], res["reason"] = False, f"reliability, day {k}"
                break
    if res["ok"]:
        z = np.zeros(T)
        if float(cfg.C_R[i, l]) == 0.0:
            z[:] = np.nan
        else:
            gm = res["g"]["mu"] if law == "ard1" else None
            for k in range(1, T + 1):
                if m[i, l, k - 1]:
                    z[k - 1] = max(0.0, float(cfg.rho[i, l]) *
                                   (res["s"]["mu"][k - 1] - (gm[k - 1] if gm is not None else 0.0)))
        res["z"] = z
    cache[key] = res
    return res


def least_solution(cfg, x, m, r=None, max_passes=10000, tol=1e-12, verify=True):
    """The least one-sided solution of the dynamics rows for a fixed schedule (see the comment above).

    Returns a dict: ``feasible`` (bool), ``reason`` (first failure, ``""`` if feasible), and, if feasible,
    the arrays ``(F, L, T+1)`` ``mu``, ``v``, ``R``, ``K``, ``g_mu``, ``g_v``, ``g_R``, ``g_K`` (NaN where
    not tracked; index 0 holds the initial values), ``z`` (``(F, L, T)``, ``minimal_z`` of the least
    trajectory) and ``u`` (``u_value`` of its mean), and ``row_violations`` (the list returned by
    ``row_violations`` on the least solution with exact tolerances ``tol = 1e-9``, ``bigm_tol = quad_rel
    = 0``; empty if feasible). With ``verify = False`` the full ``row_violations`` pass is skipped
    (``best_onesided`` does that for speed; the cell checks are the same rows).
    """
    F, L = cfg.F, cfg.L
    x = np.asarray(x)
    T = x.shape[-1]
    m = np.asarray(m)
    r = np.zeros_like(m) if r is None else np.asarray(r)
    out = {name: np.full((F, L, T + 1), np.nan) for name in
           ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K")}
    z = np.zeros((F, L, T))
    for i in range(F):
        for l in range(L):
            cell = _least_cell(cfg, i, l, x, m, r, max_passes, tol)
            if not cell["ok"]:
                return {"feasible": False, "reason": f"cell {(i, l)}: {cell['reason']}",
                        "row_violations": None}
            for name, s in cell["s"].items():
                out[name][i, l] = s
                if cell["g"][name] is not None:
                    out["g_" + name][i, l] = cell["g"][name]
            z[i, l] = cell["z"]
    out["z"] = z
    out["u"] = u_value(cfg, out["mu"])
    out["feasible"], out["reason"], out["row_violations"] = True, "", []
    if verify:
        sol = {"x": x, "m": m, "r": r, "z": z, "u": out["u"]}
        for name in ("mu", "v", "R", "K", "g_mu", "g_v", "g_R", "g_K"):
            sol[name] = out[name][..., 1:]
        out["row_violations"] = row_violations(cfg, sol, tol=1e-9, bigm_tol=0.0, quad_rel=0.0)
        if out["row_violations"]:
            out["feasible"], out["reason"] = False, f"row {out['row_violations'][0].row}"
    return out


def best_onesided(cfg, max_passes=10000, tol=1e-12):
    """The smallest cost over the schedules that have a one-sided solution (spec 780-836, 1602-1617).

    A schedule is feasible if ``least_solution`` is; its cost is ``costs(...)["objective"]`` with ``z``
    from the least trajectory and ``u = u_value`` of its mean. With only ``ardinf`` cells every cost term
    is non-decreasing in the tracked states (``z = rho mu_{k-1}``, Eq. a_z; ``u``, Eqs. a_u_inf and
    a_u_quad), so the least solution is the cheapest one: the result is the exact optimum of the spec's
    problem. With an ``ard1`` cell the term ``rho (mu_{k-1} - g_{k-1})`` is not monotone: the result is an
    upper bound. Returns a dict: ``cost`` (``None`` if no schedule is feasible), ``schedule`` (the first of
    that cost), ``n_feasible``, ``n_total``, ``costs`` and ``exact`` (every cell ``ardinf``).
    """
    F, L = cfg.F, cfg.L
    exact = all(cell_kind(cfg, i, l)["law"] == "ardinf" for i in range(F) for l in range(L))
    best_cost, best_schedule, n_total, feasible_costs = None, None, 0, []
    for x, m, r in enumerate_schedules(cfg):
        n_total += 1
        cells = []
        for i in range(F):
            for l in range(L):
                cell = _least_cell(cfg, i, l, x, m, r, max_passes, tol)
                if not cell["ok"]:
                    break
                cells.append(cell)
            else:
                continue
            break
        else:
            mu = np.array([c["s"]["mu"] for c in cells]).reshape(F, L, -1)
            z = np.array([c["z"] for c in cells]).reshape(F, L, -1)
            cost = costs(cfg, x, r, z, u_value(cfg, mu))["objective"]
            feasible_costs.append(cost)
            if best_cost is None or cost < best_cost - 1e-12:
                best_cost, best_schedule = cost, (x, m, r)
    return {"cost": best_cost, "schedule": best_schedule, "n_feasible": len(feasible_costs),
            "n_total": n_total, "costs": feasible_costs, "exact": exact}
