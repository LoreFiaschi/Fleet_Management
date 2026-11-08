"""Reference implementation of the greedy heuristic of the spec (section "Greedy Heuristic").

Written by the WP7 sweeps planner, independently of greedy_warm_start.py, to produce the
expected values of tests/test_warm_start.py. Run: python greedy_ref.py
"""
import math
import sys

import numpy as np
from scipy.special import ndtri

sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
from fleet_management.config import load_config, _lookup  # noqa: E402

TOL = 1e-10


def quantities(cfg, i, l):
    """List of (q, start, new, anchored, anchor0, a, increments (M, T), range)."""
    model = str(cfg.model[i, l]); bound = str(cfg.bound_method[i, l])
    ard1 = str(cfg.repair_model[i, l]) == "ard1"
    rho = float(cfg.rho[i, l])
    H1, H2 = cfg.H1, cfg.H2
    mu_inc = _lookup(cfg.mu, cfg.mu_trans, H1, H2)[i, l]
    out = [("mu", cfg.mu_0[i, l], cfg.replacement_mu[i, l], ard1, cfg.anchor_mu_0[i, l],
            1 - rho, mu_inc, cfg.range_mu[i, l])]
    if model == "gaussian" or (model == "rainflow" and bound in ("cantelli", "bernstein")):
        out.append(("v", cfg.v_0[i, l], cfg.replacement_v[i, l], ard1, cfg.anchor_v_0[i, l],
                    (1 - rho) ** 2, _lookup(cfg.v, cfg.v_trans, H1, H2)[i, l], cfg.range_v[i, l]))
    elif model == "wiener":
        inc = np.full(mu_inc.shape, float(cfg.sigma[i, l]) ** 2)
        out.append(("v", cfg.v_0[i, l], cfg.replacement_v[i, l], False, math.nan, 1.0, inc,
                    cfg.range_v[i, l]))
    elif model == "rainflow" and bound == "hoeffding":
        out.append(("R", cfg.R_0[i, l], cfg.R_new[i, l], ard1, cfg.anchor_R_0[i, l],
                    (1 - rho) ** 2, _lookup(cfg.support, cfg.support_trans, H1, H2)[i, l] ** 2,
                    cfg.range_R[i, l]))
    elif model == "rainflow" and bound == "chernoff":
        out.append(("K", cfg.cgf_0[i, l], cfg.cgf_new[i, l], ard1, cfg.anchor_K_0[i, l],
                    1 - rho, _lookup(cfg.cgf, cfg.cgf_trans, H1, H2)[i, l], cfg.range_K[i, l]))
    return out


def rows(cfg, i, l):
    """('quad', chi, c2, c1, side) or ('cap', q, cap); plus v_max_user (nan if none)."""
    model = str(cfg.model[i, l]); bound = str(cfg.bound_method[i, l])
    tau, eps = float(cfg.tau[i, l]), float(cfg.epsilon[i, l])
    if model in ("gaussian", "wiener"):
        row = ("quad", "v", 1.0 / ndtri(1 - eps) ** 2, 0.0, tau)
    elif model == "inverse_gaussian":
        from fleet_management.degradation_model.inverse_gaussian import ig_mean_cap
        row = ("cap", "mu", ig_mean_cap(float(cfg.eta[i, l]), tau, eps))
    elif model == "gamma":
        from fleet_management.degradation_model.gamma import mean_cap, smallest_rate
        mu_inc = _lookup(cfg.mu, cfg.mu_trans, cfg.H1, cfg.H2)[i, l]
        beta_inc = _lookup(cfg.gamma_beta, cfg.gamma_beta_trans, cfg.H1, cfg.H2)[i, l]
        star = smallest_rate(mu_inc, beta_inc, float(cfg.mu_0[i, l]), float(cfg.gamma_beta_0[i, l]),
                             float(cfg.replacement_mu[i, l]), float(cfg.gamma_beta_new[i, l]),
                             bool(cfg.allow_replacement))
        row = ("cap", "mu", mean_cap(star, tau, eps))
    else:
        from fleet_management.degradation_model.rainflow import row_constants
        c = row_constants(bound, tau, eps, float(cfg.support_const[i, l]), float(cfg.s_chernoff[i, l]))
        row = ("cap", c["q"], c["cap"]) if "cap" in c else ("quad", c["q"], c["c2"], c["c1"], c["mu_side"])
    vmax = float(cfg.v_max_user[i, l]) if (model in ("gaussian", "wiener") or bound in ("cantelli", "bernstein")) and model in ("gaussian", "wiener", "rainflow") else math.nan
    return row, vmax


def cap_of(mu, tau, c2, c1, impl, tref, P):
    f = lambda m: c2 * (tau - m) ** 2 + c1 * (tau - m)
    fp = lambda m: -2 * c2 * (tau - m) - c1
    if impl == "exact":
        return f(mu)
    if impl == "tangent":
        r = tref * tau
        return f(r) + fp(r) * (mu - r)
    mc = min(max(mu, 0.0), tau)
    best = -math.inf
    for p in range(1, P + 1):
        if (p - 1) * tau / P <= mc <= p * tau / P:
            h = (p - 0.5) * tau / P
            best = max(best, f(h) + fp(h) * (mu - h))
    return best


def trajectory(cfg, x, m, r, i, l, Q):
    T = cfg.T
    s = {q[0]: float(q[1]) for q in Q}
    g = {q[0]: float(q[4]) for q in Q if q[3]}
    traj = {q[0]: np.zeros(T) for q in Q}
    gtraj = {q: np.zeros(T) for q in g}
    for k in range(T):
        flown = [j for j in range(cfg.M) if x[i, j + 1, k] > 0.5]
        for (q, start, new, anchored, g0, a, inc, rng) in Q:
            if r[i, l, k] > 0.5:
                s[q] = float(new)
                if anchored:
                    g[q] = s[q]
            elif m[i, l, k] > 0.5:
                s[q] = a * s[q] + (1 - a) * g[q] if anchored else a * s[q]
                if anchored:
                    g[q] = s[q]
            else:
                s[q] = s[q] + (float(inc[flown[0], k]) if flown else 0.0)
            traj[q][k] = s[q]
            if anchored:
                gtraj[q][k] = g[q]
    return traj, gtraj


def excess(cfg, x, m, r, i, l, Q, row, vmax):
    H1, T = cfg.H1, cfg.T
    traj, gtraj = trajectory(cfg, x, m, r, i, l, Q)
    rng = {q[0]: float(q[7]) for q in Q}
    norm, raw = [0.0], [0.0]
    for q, t in list(traj.items()) + list(gtraj.items()):
        d = max(0.0, t[T - 1] - t[H1 - 1])
        norm.append(d / rng[q]); raw.append(d)
    tau = float(cfg.tau[i, l])
    mu = traj["mu"]
    if row[0] == "cap":
        q, cap = row[1], row[2]
        d = max(0.0, float(np.max(traj[q] - cap)))
        norm.append(d / rng[q]); raw.append(d)
    else:
        chi, c2, c1, side = row[1:]
        impl = str(cfg.reliability_impl[i, l])
        caps = np.array([cap_of(v, tau, c2, c1, impl, float(cfg.tangent_ref[i, l]) if impl == "tangent" else 0.5, cfg.pwl_points) for v in mu])
        d = max(0.0, float(np.max(traj[chi] - caps)))
        norm.append(d / rng[chi]); raw.append(d)
        d = max(0.0, float(np.max(mu - side)))
        norm.append(d / rng["mu"]); raw.append(d)
    if math.isfinite(vmax):
        d = max(0.0, float(np.max(traj["v"] - vmax)))
        norm.append(d / rng["v"]); raw.append(d)
    return max(norm), max(raw), float(np.max(mu)) / tau


def greedy(cfg, r_trig=0.70, joint=False, joint_models=()):
    """joint=False: the spec (repairs first, a replacement only if no repair helps).
    joint_models: models whose cells use the joint trial; ("wiener",) is the user's rule of
    29 Oct 2026 (merge/wp7_interface.md, decision 2).
    joint=True: option (b) of question Q1 (repairs and replacements in one trial, the smallest
    e wins; a tie goes to the repair, then to the later day)."""
    F, M, L, T, H1 = cfg.F, cfg.M, cfg.L, cfg.T, cfg.H1
    x = np.zeros((F, M + 1, T)); m = np.zeros((F, L, T)); r = np.zeros((F, L, T))
    mu_inc = _lookup(cfg.mu, cfg.mu_trans, cfg.H1, cfg.H2)       # (F, L, M, T)
    tau = np.asarray(cfg.tau, float); rho = np.asarray(cfg.rho, float)
    s = np.asarray(cfg.mu_0, float).copy()
    order_all, max_ratio = [], 0.0
    for k in range(T):
        diff = [(float(np.max(mu_inc[:, :, j, k] / tau)), j) for j in range(M)]
        order = [j for _, j in sorted(diff, reverse=True)]
        order_all.append([j + 1 for j in order])
        free = list(range(F))
        for j in order:
            def key(i):
                ratios = (s[i] + mu_inc[i, :, j, k]) / tau[i]
                return (float(np.max(ratios)), float(np.sum(ratios)), i)
            i = min(free, key=key)
            free.remove(i)
            x[i, j + 1, k] = 1
            s[i] += mu_inc[i, :, j, k]
        for i in free:
            due = s[i] > r_trig * tau[i]
            if due.any():
                x[i, 0, k] = 1
                m[i, due, k] = 1
                s[i, due] *= 1 - rho[i, due]
        max_ratio = max(max_ratio, float(np.max(s / tau)))
    failures, max_raw = 0, 0.0
    for i in range(F):
        for l in range(L):
            Q = quantities(cfg, i, l); row, vmax = rows(cfg, i, l)
            e, raw, ratio = excess(cfg, x, m, r, i, l, Q, row, vmax)
            while e > TOL:
                done = False
                kinds = [m] + ([r] if cfg.allow_replacement else [])
                if joint or str(cfg.model[i, l]) in joint_models:
                    best = None
                    for k in range(H1, T):
                        if x[i, 1:, k].sum() > 0.5 or m[i, l, k] > 0.5 or r[i, l, k] > 0.5:
                            continue
                        for rank, arr in enumerate(kinds):
                            old = x[i, 0, k]
                            arr[i, l, k] = 1; x[i, 0, k] = 1
                            ek = excess(cfg, x, m, r, i, l, Q, row, vmax)[0]
                            arr[i, l, k] = 0; x[i, 0, k] = old
                            if best is None or (ek, rank, -k) < best[0]:
                                best = ((ek, rank, -k), arr, k)
                    if best is not None and best[0][0] < e - TOL:
                        best[1][i, l, best[2]] = 1; x[i, 0, best[2]] = 1
                        e = best[0][0]
                        continue
                    failures += 1
                    break
                for arr in kinds:
                    best = None
                    for k in range(H1, T):
                        if x[i, 1:, k].sum() > 0.5 or m[i, l, k] > 0.5 or r[i, l, k] > 0.5:
                            continue
                        old = x[i, 0, k]
                        arr[i, l, k] = 1; x[i, 0, k] = 1
                        ek = excess(cfg, x, m, r, i, l, Q, row, vmax)[0]
                        arr[i, l, k] = 0; x[i, 0, k] = old
                        if best is None or ek <= best[0]:
                            best = (ek, k)
                    if best is not None and best[0] < e - TOL:
                        arr[i, l, best[1]] = 1; x[i, 0, best[1]] = 1
                        e = best[0]
                        done = True
                        break
                if not done:
                    failures += 1
                    break
            e, raw, ratio = excess(cfg, x, m, r, i, l, Q, row, vmax)
            max_raw = max(max_raw, raw); max_ratio = max(max_ratio, ratio)
    report = {"repairs": int(m.sum()), "replacements": int(r.sum()),
              "repeatability_feasible": failures == 0, "repeatability_failures": failures,
              "maximum_repeatability_excess": max_raw, "maximum_proxy_ratio": max_ratio,
              "assignment_order": order_all}
    return x, m, r, report


COSTS = {"C_M": 1.0, "C_R": 1.0, "C_rep": 5.0, "C_D": 0.0}

CASES = {
    # A: hand case. Gaussian, ardinf, tangent form; two trains, one mission.
    "A_gaussian": dict(F=2, M=1, H=[2, 2], model="gaussian", tau=1.0, epsilon=0.1, rho=0.5,
                       mu_0=0.125, v_0=0.0, mu=0.25, v=0.00390625, reliability_impl="tangent",
                       **COSTS),
    # A2: as A with three trains and H2 = 4: the loop pass succeeds.
    "A2_gaussian": dict(F=3, M=1, H=[2, 4], model="gaussian", tau=1.0, epsilon=0.1, rho=0.5,
                        mu_0=0.125, v_0=0.0, mu=0.25, v=0.00390625, reliability_impl="tangent",
                        **COSTS),
    # B2: Wiener with three trains and H2 = 4.
    "B2_wiener": dict(F=3, M=1, H=[2, 4], model="wiener", tau=1.0, epsilon=0.1, rho=0.5,
                      mu_0=0.125, v_0=0.0, sigma=0.0625, mu=0.25, reliability_impl="tangent",
                      **COSTS),
    # F: as A2 with mu_0 = 0.25 and rho = 1 (a repair empties the state): feasible.
    "F_gaussian": dict(F=3, M=1, H=[2, 2], model="gaussian", tau=1.0, epsilon=0.1, rho=1.0,
                       mu_0=0.25, v_0=0.0, mu=0.25, v=0.00390625, reliability_impl="tangent",
                       **COSTS),
    # G: linear rows only (IG, Gamma, rainflow Markov and Chernoff), ardinf, rho = 1.
    "G_linear": dict(F=3, M=1, H=[2, 2],
                     model=["inverse_gaussian", "gamma", "rainflow", "rainflow"],
                     bound_method=["markov", "markov", "markov", "chernoff"], tau=1.0,
                     epsilon=0.5, rho=1.0, mu_0=0.0625, eta=0.25, gamma_beta=8.0,
                     s_chernoff=4.0, mu=[[0.125], [0.0625], [0.0625], [0.125]],
                     cgf=[[0.0], [0.0], [0.0], [0.625]], **COSTS),
    # B: Wiener, ardinf: the variance falls only by replacement.
    "B_wiener": dict(F=2, M=1, H=[2, 2], model="wiener", tau=1.0, epsilon=0.1, rho=0.5,
                     mu_0=0.125, v_0=0.0, sigma=0.0625, mu=0.25, reliability_impl="tangent",
                     **COSTS),
    # H: one Wiener and one Gaussian component per train (as A and B): the rule of decision 2
    # applies the joint trial at the Wiener cells only.
    "H_wiener_gaussian": dict(F=2, M=1, H=[2, 2], model=["wiener", "gaussian"], tau=1.0,
                              epsilon=0.1, rho=0.5, mu_0=0.125, v_0=0.0, sigma=[0.0625, 0.0],
                              mu=0.25, v=[[0.0], [0.00390625]], reliability_impl="tangent",
                              **COSTS),
    # C: Wiener without replacement: one failure per flying cell.
    "C_wiener_norepl": dict(F=2, M=1, H=[2, 2], model="wiener", tau=1.0, epsilon=0.1, rho=0.5,
                            mu_0=0.125, v_0=0.0, sigma=0.0625, mu=0.25,
                            reliability_impl="tangent", allow_replacement=False, **COSTS),
    # D: mixed fleet, ard1, three trains, two missions, L = 3.
    "D_mixed_ard1": dict(F=3, M=2, H=[2, 3], model=["inverse_gaussian", "gamma", "rainflow"],
                         bound_method="chernoff", repair_model="ard1", tau=1.0, epsilon=0.1,
                         rho=0.5, mu_0=0.0625, eta=0.25, gamma_beta=8.0, s_chernoff=4.0,
                         mu=[[0.125, 0.25], [0.0625, 0.125], [0.125, 0.0625]],
                         cgf=[[0.625, 1.25], [0.3125, 0.625], [0.625, 0.3125]], **COSTS),
    # E: rainflow Cantelli (exact form) and Hoeffding (pwl), ardinf, transitory profile.
    "E_rainflow": dict(F=2, M=1, H=[3, 2], model=["rainflow", "rainflow"],
                       bound_method=["cantelli", "hoeffding"], tau=1.0, epsilon=0.2, rho=0.75,
                       mu_0=0.0, reliability_impl=["exact", "pwl"], pwl_points=4,
                       mu=[[0.125], [0.125]], mu_trans=[[0.25], [0.0625]],
                       v=[[0.001953125], [0.0]], support=[[0.0], [0.125]], **COSTS),
}


def main():
    np.set_printoptions(linewidth=150)
    for name, data in CASES.items():
        cfg = load_config(dict(data, verbose=0))
        x, m, r, rep = greedy(cfg)
        print("=" * 20, name)
        for i in range(cfg.F):
            print(f" train {i}: activity per day (0 depot, -1 idle, j mission):",
                  [int(np.argmax(x[i, :, k])) if x[i, :, k].sum() > 0.5 else -1 for k in range(cfg.T)])
            for l in range(cfg.L):
                print(f"   cell ({i},{l}) m:", m[i, l].astype(int).tolist(), " r:", r[i, l].astype(int).tolist())
        print(" report:", rep)
        xw, mw, rw, repw = greedy(cfg, joint_models=("wiener",))
        print(" rule (Wiener joint):", {k: repw[k] for k in ("repairs", "replacements", "repeatability_failures", "maximum_repeatability_excess", "maximum_proxy_ratio")})
        for i in range(cfg.F):
            for l in range(cfg.L):
                if (mw[i, l] != m[i, l]).any() or (rw[i, l] != r[i, l]).any() or (xw[i] != x[i]).any():
                    print(f"   (rule) cell ({i},{l}) m:", mw[i, l].astype(int).tolist(), " r:", rw[i, l].astype(int).tolist())
        if name in ("A_gaussian", "B_wiener", "D_mixed_ard1", "F_gaussian", "G_linear"):
            xj, mj, rj, repj = greedy(cfg, joint=True)
            print(" option (b):", {k: repj[k] for k in ("repairs", "replacements", "repeatability_failures", "maximum_repeatability_excess", "maximum_proxy_ratio")})
            for i in range(cfg.F):
                for l in range(cfg.L):
                    if (mj[i, l] != m[i, l]).any() or (rj[i, l] != r[i, l]).any():
                        print(f"   (b) cell ({i},{l}) m:", mj[i, l].astype(int).tolist(), " r:", rj[i, l].astype(int).tolist())
        if name == "A_gaussian":
            print(" repair_trigger 0.5:", greedy(cfg, 0.5)[3])


if __name__ == "__main__":
    main()
