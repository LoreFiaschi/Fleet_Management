"""
File for solving fleet management with remaining-life (rainflow branch).

This is the rainflow / accumulated-damage counterpart of the Gaussian-degradation
solver.  The fleet mechanics (assignment, maintenance removal, safety/repair
costs, capacity, periodic horizon) are kept identical to the Gaussian version so
the two models can be compared directly.  Only two things change, and they are
exactly the two blocks the rainflow formulation touches:

  * RELIABILITY   P(D_{ilk} > tau) <= eps                              (slides 34-35)
        In the Gaussian model this was the linearised quantile bound
        mu + Phi^{-1}(1-eps) * sqrt(v) <= alpha.  Rainflow makes NO normality
        assumption: the accumulated Palmgren-Miner damage D is only known through
        its mean, variance, support, or CGF (all AFFINE in the per-mission counts,
        slide 33), so the tail P(D > tau) is bounded by a concentration
        inequality.  Five are provided, in increasing tightness (slide 35):
            method='markov'     mean only            -> LINEAR   (MILP)
            method='cantelli'   mean, variance       -> quadratic (MIQCP)   [default]
            method='hoeffding'  mean, support        -> quadratic (MIQCP)
            method='bernstein'  mean, variance, supp -> quadratic (MIQCP)
            method='chernoff'   CGF at fixed s       -> LINEAR   (MILP)
        Cantelli is the natural drop-in for the old Gaussian bound: it reuses the
        very same (mu, v) state and only swaps the coefficient
        Phi^{-1}(1-eps)  ->  sqrt((1-eps)/eps).

  * REPEATABILITY  loop the MOMENTS, not the bound                     (slide 36)
        mu(2H) <= mu(H)  AND  var(2H) <= var(H)  (per vehicle / component).
        Looping the reliability bound U(2H) <= U(H) is NOT sufficient because
        different (mean, variance) pairs give the same bound value.  The Gaussian
        reference already loops the moments; we keep that and, for the two methods
        that carry an extra descriptor (Hoeffding's support-sum, Chernoff's CGF),
        we loop that descriptor too so the repeated horizon is dominated.

Author: Johann Tschan
"""

import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB


# Reliability methods and which extra data each one meeds:
_METHODS = ("markov", "cantelli", "hoeffding", "bernstein", "chernoff")
_QUADRATIC = ("cantelli", "hoeffding", "bernstein")  # need NonConvex=2
_NEEDS_SUPPORT = ("hoeffding", "bernstein")           # need support_param
_NEEDS_CGF = ("chernoff",)                            # need cgf_param + s_chernoff


def validate_inputs(
    F: int,
    H: int,
    M: int,
    L: int,
    mu_param: np.ndarray,
    v_param: np.ndarray,
    tau: float,
    epsilon: float,
    xi: np.ndarray,
    C_M: float,
    C_R: float,
    C_S: float,
    C_P: float,
    mu_0: np.ndarray,
    v_0: np.ndarray,
    method: str,
    support_param: np.ndarray = None,
    cgf_param: np.ndarray = None,
    s_chernoff: float = None,
) -> None:
    """Consistency checks for the rainflow model.
    """
    if F <= 0 or H <= 0 or M <= 0 or L <= 0:
        raise ValueError("F, H, M, L must be positive integers.")
    if tau <= 0:
        raise ValueError("tau (damage threshold) must be positive.")
    if not (0.0 < epsilon < 1.0):
        raise ValueError(f"epsilon must be in (0, 1) (got {epsilon}).")
    if C_M <= 0 or C_R <= 0 or C_S <= 0 or C_P <= 0:
        raise ValueError("All cost coefficients must be positive.")

    if F <= M:
        raise ValueError(f"F must be greater than M (got F={F}, M={M}).")

    if mu_param.shape != (F, M, L, H):
        raise ValueError(f"mu_param shape must be {(F, M, L, H)}, got {mu_param.shape}.")
    if v_param.shape != (F, M, L, H):
        raise ValueError(f"v_param shape must be {(F, M, L, H)}, got {v_param.shape}.")

    if xi.shape != (F, L):
        raise ValueError(f"xi must have shape {(F, L)}.")
    if not np.all(xi > 0) or not np.all(xi <= 1):
        raise ValueError("xi must be in (0, 1] element-wise.")

    if mu_0.shape != (F, L) or v_0.shape != (F, L):
        raise ValueError(f"mu_0 and v_0 must have shape {(F, L)}.")

    # Per-mission damage-increment moments must be positive; totals non-negative.
    if not np.all(mu_param > 0):
        raise ValueError("All entries of mu_param must be positive.")
    if not np.all(v_param > 0):
        raise ValueError("All entries of v_param must be positive.")
    if not np.all(mu_0 >= 0) or not np.all(v_0 >= 0):
        raise ValueError("mu_0 and v_0 must be >= 0 element-wise.")

    # A component must start below the failure threshold, else it is already dead.
    # if not np.all(mu_0 < tau):
    #     raise ValueError(f"mu_0 must be < tau={tau} element-wise.")

    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS} (got '{method}').")

    if method in _NEEDS_SUPPORT:
        if support_param is None:
            raise ValueError(f"method='{method}' requires support_param "
                             f"(per-mission increment support width).")
        if support_param.shape != (F, M, L, H):
            raise ValueError(f"support_param shape must be {(F, M, L, H)}.")
        if not np.all(support_param > 0):
            raise ValueError("support_param must be positive element-wise.")

    if method in _NEEDS_CGF:
        if cgf_param is None or s_chernoff is None:
            raise ValueError("method='chernoff' requires cgf_param (per-mission "
                             "CGF evaluated at s) and s_chernoff > 0.")
        if cgf_param.shape != (F, M, L, H):
            raise ValueError(f"cgf_param shape must be {(F, M, L, H)}.")
        if not np.all(cgf_param > 0):
            raise ValueError("cgf_param must be positive element-wise.")
        if s_chernoff <= 0:
            raise ValueError("s_chernoff must be positive.")


def solve_fleet_management(
    F: int,
    H: int,
    M: int,
    L: int,
    mu_param: np.ndarray,
    v_param: np.ndarray,
    tau: float,
    epsilon: float,
    xi: np.ndarray,
    C_M: float,
    C_R: float,
    C_S: float,
    C_P: float,
    mu_0: np.ndarray,
    v_0: np.ndarray,
    method: str = "cantelli",
    support_param: np.ndarray = None,
    cgf_param: np.ndarray = None,
    s_chernoff: float = None,
    verbose: int = 1,
    mip_gap: float = 0.12,
) -> dict:
    """
    Solve the rainflow (accumulated-damage) fleet-management problem with Gurobi.

    Parameters
    ----------
    F, H, M, L : int
        Fleet size, horizon (model spans 2H steps), number of missions, number of
        components per vehicle.  ``F > M`` is required.
    mu_param, v_param : np.ndarray, shape (F, M, L, H)
        Mean and variance of the *per-mission* Palmgren-Miner damage increment
        (from rainflow counting).  These accumulate linearly in the schedule
        counts (slide 33).  For k >= H the parameters wrap (index ``k % H``).
    tau : float
        Damage failure threshold.  Reliability is P(D > tau) <= epsilon.
    epsilon : float
        Reliability level, in (0, 1).
    xi : np.ndarray, shape (F, L)
        Fraction of accumulated damage removed by one maintenance step.
    C_M, C_R, C_S, C_P : float
        Maintenance / repair / safety / periodicity cost coefficients.
    mu_0, v_0 : np.ndarray, shape (F, L)
        Initial accumulated-damage mean and variance.
    method : str
        Reliability bound: 'markov' | 'cantelli' (default) | 'hoeffding'
        | 'bernstein' | 'chernoff'.
    support_param : np.ndarray, shape (F, M, L, H), optional
        Per-mission increment support width (b_j - a_j).  Required for
        'hoeffding' and 'bernstein'.
    cgf_param : np.ndarray, shape (F, M, L, H), optional
    s_chernoff : float, optional
        Per-mission cumulant generating function evaluated at ``s_chernoff`` and
        the (offline-tuned) tilt ``s_chernoff``.  Required for 'chernoff'.
    verbose : int
        Gurobi OutputFlag.
    mip_gap : float, optional
        Relative MIP gap tolerance.

    Returns
    -------
    dict
        Keys: "status", "objective", "method", "x", "mu", "v", "u", "z",
              "F", "H", "M", "L", "tau", "model".
    """
    validate_inputs(F, H, M, L, mu_param, v_param, tau, epsilon, xi,
                    C_M, C_R, C_S, C_P, mu_0, v_0,
                    method, support_param, cgf_param, s_chernoff)

    # ---- precomputed constants -------------------------------------------
    Le = math.log(1.0 / epsilon)      # ln(1/eps), used by Hoeffding/Bernstein
    ln_eps = math.log(epsilon)        # ln(eps), used by Chernoff

    # Big-M for deactivating each accumulation lower bound during maintenance.
    # It must exceed that state's largest reachable value (initial + every
    # increment over 2H with no maintenance); otherwise maintenance cannot pull
    # the state down and the model is spuriously infeasible.  Each descriptor is
    # sized independently -- the CGF, in particular, is much larger than tau.
    bigM_mu = float(mu_0.max()) + 2 * H * float(mu_param.max())
    bigM_v = float(v_0.max()) + 2 * H * float(v_param.max())
    bigM_R = 2 * H * float(support_param.max() ** 2) if method == "hoeffding" else 0.0
    bigM_K = 2 * H * float(cgf_param.max()) if method == "chernoff" else 0.0

    # wrapped parameter accessors (0-indexed everywhere; j is the 0-based mission)
    def mu_inc(i, j, l, k):
        return float(mu_param[i, j, l, k % H])

    def v_inc(i, j, l, k):
        return float(v_param[i, j, l, k % H])

    def w2_inc(i, j, l, k):                      # squared support width
        return float(support_param[i, j, l, k % H] ** 2)

    def cgf_inc(i, j, l, k):
        return float(cgf_param[i, j, l, k % H])

    # ---- model -----------------------------------------------------------
    model = gp.Model("fleet_management_rainflow")
    model.Params.OutputFlag = int(verbose)
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap
    if method in _QUADRATIC:
        # Cantelli/Hoeffding/Bernstein feasible sets are quadratic and non-convex
        # in (mu, v); this flag lets Gurobi accept the algebraic form.
        model.Params.NonConvex = 2

    # decision variables (same layout as the Gaussian model)
    # x[i,j,k]: j=0 -> maintenance, j=1..M -> missions
    x = model.addVars(F, M + 1, 2 * H, vtype=GRB.BINARY, name="x")
    mu_var = model.addVars(F, L, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="mu")
    v_var = model.addVars(F, L, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="v")
    u_var = model.addVars(2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="u")
    z_var = model.addVars(F, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="z")

    # optional extra descriptor states, only created when the method needs them
    R_var = None   # Hoeffding: accumulated squared-support sum
    K_var = None   # Chernoff:  accumulated CGF at s
    if method == "hoeffding":
        R_var = model.addVars(F, L, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="R")
    if method == "chernoff":
        K_var = model.addVars(F, L, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="K")

    # ---- objective ---------------------
    obj = gp.LinExpr()
    for k in range(2 * H):
        obj += C_S * u_var[k]
        for i in range(F):
            obj += C_M * x[i, 0, k]
            obj += C_R * z_var[i, k]
    for i in range(F):
        for l in range(L):
            obj += C_P * (
                mu_var[i, l, H - 1] - mu_var[i, l, 2 * H - 1]
                + v_var[i, l, H - 1] - v_var[i, l, 2 * H - 1]
            )
    model.setObjective(obj, GRB.MINIMIZE)

    # ---- capacity: aggregate mean-damage cap per step --------------------
    for k in range(2 * H):
        model.addConstr(
            gp.quicksum(mu_var[i, l, k] for i in range(F) for l in range(L)) <= F - M,
            name=f"capacity_{k}",
        )

    # ---- state recursion + reliability -----------------------------------
    for i in range(F):
        for l in range(L):
            for k in range(2 * H):
                mu_prev = mu_0[i, l] if k == 0 else mu_var[i, l, k - 1]
                v_prev = v_0[i, l] if k == 0 else v_var[i, l, k - 1]

                # mean of accumulated damage: grow by the mission increment,
                # or drop to (1 - xi) fraction under maintenance (j=0).
                model.addConstr(
                    mu_var[i, l, k]
                    >= mu_prev
                    + gp.quicksum(x[i, j, k] * mu_inc(i, j - 1, l, k)
                                  for j in range(1, M + 1))
                    - bigM_mu * x[i, 0, k],
                    name=f"mu_update_{i}_{l}_{k}",
                )
                model.addConstr(
                    mu_var[i, l, k] >= mu_prev * (1 - xi[i, l]),
                    name=f"mu_lb_{i}_{l}_{k}",
                )

                # variance of accumulated damage (variances add; slide 33).
                model.addConstr(
                    v_var[i, l, k]
                    >= v_prev
                    + gp.quicksum(x[i, j, k] * v_inc(i, j - 1, l, k)
                                  for j in range(1, M + 1))
                    - bigM_v * x[i, 0, k],
                    name=f"v_update_{i}_{l}_{k}",
                )
                model.addConstr(
                    v_var[i, l, k] >= v_prev * (1 - xi[i, l]),
                    name=f"v_lb_{i}_{l}_{k}",
                )

                # extra descriptor recursions (same shape as variance)
                if method == "hoeffding":
                    R_prev = 0.0 if k == 0 else R_var[i, l, k - 1]
                    model.addConstr(
                        R_var[i, l, k]
                        >= R_prev
                        + gp.quicksum(x[i, j, k] * w2_inc(i, j - 1, l, k)
                                      for j in range(1, M + 1))
                        - bigM_R * x[i, 0, k],
                        name=f"R_update_{i}_{l}_{k}",
                    )
                    model.addConstr(R_var[i, l, k] >= R_prev * (1 - xi[i, l]),
                                    name=f"R_lb_{i}_{l}_{k}")
                if method == "chernoff":
                    K_prev = 0.0 if k == 0 else K_var[i, l, k - 1]
                    model.addConstr(
                        K_var[i, l, k]
                        >= K_prev
                        + gp.quicksum(x[i, j, k] * cgf_inc(i, j - 1, l, k)
                                      for j in range(1, M + 1))
                        - bigM_K * x[i, 0, k],
                        name=f"K_update_{i}_{l}_{k}",
                    )
                    model.addConstr(K_var[i, l, k] >= K_prev * (1 - xi[i, l]),
                                    name=f"K_lb_{i}_{l}_{k}")

                # ---- RELIABILITY  P(D > tau) <= eps  (slides 34-35) -------
                mu_ik = mu_var[i, l, k]
                v_ik = v_var[i, l, k]
                rname = f"rel_{i}_{l}_{k}"

                if method == "markov":
                    #  mean <= eps * tau                                (linear)
                    model.addConstr(mu_ik <= epsilon * tau, name=rname)

                elif method == "cantelli":
                    #  (1-eps) var <= eps (tau - mean)^2 ,  mean <= tau  (quad)
                    #  == Gaussian bound with Phi^{-1}(1-eps) replaced by
                    #     sqrt((1-eps)/eps); reuses the same (mu, v) state.
                    model.addConstr(mu_ik <= tau, name=f"{rname}_gap")
                    model.addQConstr(
                        (1.0 - epsilon) * v_ik
                        <= epsilon * (tau - mu_ik) * (tau - mu_ik),
                        name=rname,
                    )

                elif method == "hoeffding":
                    #  (tau - mean)^2 >= 0.5 ln(1/eps) * support_sum     (quad)
                    model.addConstr(mu_ik <= tau, name=f"{rname}_gap")
                    model.addQConstr(
                        (tau - mu_ik) * (tau - mu_ik) >= 0.5 * Le * R_var[i, l, k],
                        name=rname,
                    )

                elif method == "bernstein":
                    #  0.5 t^2 - (Le b/3) t - Le var >= 0 ,  t = tau-mean (quad)
                    b = float(support_param.max())   # global per-increment bound
                    t = tau - mu_ik
                    model.addConstr(mu_ik <= tau, name=f"{rname}_gap")
                    model.addQConstr(
                        0.5 * t * t - (Le * b / 3.0) * t - Le * v_ik >= 0,
                        name=rname,
                    )

                elif method == "chernoff":
                    #  K(s) - s tau <= ln(eps)                          (linear)
                    #  valid for any fixed s>0; tune s offline for tightness.
                    model.addConstr(
                        K_var[i, l, k] - s_chernoff * tau <= ln_eps, name=rname
                    )

    # ---- REPEATABILITY: loop the moments, not the bound (slide 36) -------
    for i in range(F):
        for l in range(L):
            model.addConstr(mu_var[i, l, 2 * H - 1] <= mu_var[i, l, H - 1],
                            name=f"repeat_mu_{i}_{l}")
            model.addConstr(v_var[i, l, 2 * H - 1] <= v_var[i, l, H - 1],
                            name=f"repeat_v_{i}_{l}")
            # dominate the extra descriptor too, for the methods that use one
            if method == "hoeffding":
                model.addConstr(R_var[i, l, 2 * H - 1] <= R_var[i, l, H - 1],
                                name=f"repeat_R_{i}_{l}")
            if method == "chernoff":
                model.addConstr(K_var[i, l, 2 * H - 1] <= K_var[i, l, H - 1],
                                name=f"repeat_K_{i}_{l}")

    # ---- safety (worst mean damage) and repair amount --------------------
    for k in range(2 * H):
        for i in range(F):
            for l in range(L):
                model.addConstr(u_var[k] >= mu_var[i, l, k], name=f"u_{i}_{l}_{k}")

    for i in range(F):
        for k in range(2 * H):
            model.addConstr(
                z_var[i, k]
                >= gp.quicksum(mu_var[i, l, k] * xi[i, l] for l in range(L))
                - tau + tau * x[i, 0, k],
                name=f"z_{i}_{k}",
            )

    # ---- assignment logic (slide 43) -------------------------------------
    # each vehicle does at most one activity per step ...
    for i in range(F):
        for k in range(2 * H):
            model.addConstr(gp.quicksum(x[i, j, k] for j in range(M + 1)) <= 1,
                            name=f"assign_{i}_{k}")
    # ... and every activity (maintenance slot + each mission) is covered once.
    for j in range(M + 1):
        for k in range(2 * H):
            model.addConstr(gp.quicksum(x[i, j, k] for i in range(F)) == 1,
                            name=f"demand_{j}_{k}")

    # ---- solve -----------------------------------------------------------
    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_sol = np.zeros((F, M + 1, 2 * H))
        mu_sol = np.zeros((F, L, 2 * H))
        v_sol = np.zeros((F, L, 2 * H))
        u_sol = np.zeros(2 * H)
        z_sol = np.zeros((F, 2 * H))
        for k in range(2 * H):
            u_sol[k] = u_var[k].X
            for i in range(F):
                z_sol[i, k] = z_var[i, k].X
                for l in range(L):
                    mu_sol[i, l, k] = mu_var[i, l, k].X
                    v_sol[i, l, k] = v_var[i, l, k].X
                for j in range(M + 1):
                    x_sol[i, j, k] = x[i, j, k].X
        return {
            "status": "optimal", "objective": model.ObjVal, "method": method,
            "F": F, "H": H, "M": M, "L": L, "tau": tau,
            "x": x_sol, "mu": mu_sol, "v": v_sol, "u": u_sol, "z": z_sol,
            "model": model,
        }
    return {
        "status": model.status, "objective": None, "method": method,
        "F": F, "H": H, "M": M, "L": L, "tau": tau,
        "x": None, "mu": None, "v": None, "u": None, "z": None, "model": model,
    }


# ----------------------------------------------------------------------------
# Runnable demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Fleet management (rainflow / accumulated-damage) demo")
    rng = np.random.default_rng(0)

    F, H, M, L = 3, 30, 1, 1          # 3 vehicles, horizon 3 (2H=6), 1 mission, 1 comp
    tau = 0.30                       # Palmgren-Miner failure threshold (tight, so
                                     # the reliability bound actually binds)
    epsilon = 0.10

    # per-mission damage-increment moments (rainflow + Miner), shape (F,M,L,H)
    mu_param = np.full((F, M, L, H), 0.06)
    v_param = np.full((F, M, L, H), 0.0015)
    support_param = np.full((F, M, L, H), 0.08)   # increment support width (b-a)

    # Chernoff needs the per-mission CGF at a fixed, offline-tuned tilt s.
    # Here the increment is modelled as Gamma(shape a, rate b) matching the
    # mean/variance above, so K(s) = -a ln(1 - s/b); s is tuned once, offline.
    a_shape = 0.06 ** 2 / 0.0015
    b_rate = 0.06 / 0.0015
    s_tilt = 20.0
    cgf_param = np.full((F, M, L, H), -a_shape * math.log(1 - s_tilt / b_rate))

    xi = np.full((F, L), 0.5)                     # maintenance removes half
    mu_0 = np.full((F, L), 0.02)
    v_0 = np.full((F, L), 4e-4)
    C_M, C_R, C_S, C_P = 1.0, 0.5, 2.0, 1.0

    for method in _METHODS:
        kwargs = dict(method=method, support_param=support_param, verbose=1)
        if method == "chernoff":
            kwargs.update(cgf_param=cgf_param, s_chernoff=s_tilt)
        res = solve_fleet_management(
            F, H, M, L, mu_param, v_param, tau, epsilon, xi,
            C_M, C_R, C_S, C_P, mu_0, v_0, **kwargs,
        )
        print("=" * 68)
        if res["status"] != "optimal":
            note = ("mean <= eps*tau = %.3f is far below one 0.06 increment"
                    % (epsilon * tau)) if method == "markov" else \
                   "this bound is too loose at tau=%.2f to admit any schedule" % tau
            print(f"method = {method:9s} -> INFEASIBLE  ({note})")
            continue
        print(f"method = {method:9s} -> objective = {res['objective']:.4f}")
        mu, v = res["mu"], res["v"]
        for i in range(F):
            muH, mu2H = mu[i, 0, H - 1], mu[i, 0, 2 * H - 1]
            vH, v2H = v[i, 0, H - 1], v[i, 0, 2 * H - 1]
            ok = "OK" if (mu2H <= muH + 1e-6 and v2H <= vH + 1e-6) else "VIOLATED"
            print(f"    veh{i}: repeatability  mu(H)={muH:.3f} >= mu(2H)={mu2H:.3f} , "
                  f"v(H)={vH:.5f} >= v(2H)={v2H:.5f}   [{ok}]")