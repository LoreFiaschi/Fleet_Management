import numpy as np
from scipy.stats import norm
import gurobipy as gp
from gurobipy import GRB


def validate_inputs(
    F: int,
    H: int,
    M: int,
    mu_param: np.ndarray,
    c: np.ndarray,
    alpha: float,
    epsilon: float,
    C_M: float,
    C_R: float,
    C_S: float,
    C_P: float,
    mu_0: np.ndarray,
) -> None:
    """Run all consistency checks for the inverse Gaussian degradation model."""
    if F <= 0 or H <= 0 or M <= 0:
        raise ValueError("F, H, M must be positive integers.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if C_M <= 0 or C_R <= 0 or C_S <= 0 or C_P <= 0:
        raise ValueError("All cost coefficients must be positive.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    if F <= M:
        raise ValueError(f"F must be greater than M (got F={F}, M={M}).")

    if mu_param.shape != (F, M, H):
        raise ValueError(
            f"mu shape must be ({F}, {M}, {H}), got {mu_param.shape}."
        )

    if epsilon >= 0.5:
        raise ValueError(f"epsilon must be < 0.5 (got {epsilon}).")

    if c.shape != (F,):
        raise ValueError(f"c must be a 1D vector of length {F}.")

    if mu_0.shape != (F,):
        raise ValueError(f"mu_0 must be a 1D vector of length {F}.")

    if not np.all(mu_param > 0):
        raise ValueError("All entries of mu must be positive.")
    if not np.all(c > 0):
        raise ValueError("All entries of c must be positive.")
    if not np.all(mu_0 > 0):
        raise ValueError("All entries of mu_0 must be positive.")

    if not np.all(mu_param < alpha):
        raise ValueError(f"mu must be < alpha={alpha} element-wise.")

    if not np.all(mu_0 < alpha):
        raise ValueError(f"mu_0 must be < alpha={alpha} element-wise.")


def solve_fleet_management(
    F: int,
    H: int,
    M: int,
    mu_param: np.ndarray,
    c: np.ndarray,
    alpha: float,
    epsilon: float,
    C_M: float,
    C_R: float,
    C_S: float,
    C_P: float,
    mu_0: np.ndarray,
    verbose: int = 1,
    mip_gap: float = None,
) -> dict:
    """
    Solve the Fleet Management MILP with inverse Gaussian degradation using Gurobi.

    Parameters
    ----------
    F : int
        Number of flights.
    H : int
        Time horizon (the model spans 2H time steps).
    M : int
        Number of maintenance levels.
    mu_param : np.ndarray, shape (F, M, H)
        Mean degradation parameters.
    c : np.ndarray, shape (F,)
        Shape parameter per flight for the inverse Gaussian model.
    alpha : float
        Upper bound for degradation mean (must be positive).
    epsilon : float
        Reliability threshold (must be in (0, 0.5)).
    C_M : float
        Maintenance cost coefficient.
    C_R : float
        Repair cost coefficient.
    C_S : float
        Safety cost coefficient.
    C_P : float
        Penalty cost coefficient.
    mu_0 : np.ndarray, shape (F,)
        Initial mean values per flight.
    verbose : int, optional
        Gurobi output verbosity flag (0 = silent, 1 = normal). Default is 1.
    mip_gap : float, optional
        Relative MIP optimality gap tolerance. If None, Gurobi's default is used.

    Returns
    -------
    dict
        Keys: "status", "objective", "x", "mu", "u", "z", "F", "H", "M", "alpha", "model".
    """
    # --- Consistency checks ---
    validate_inputs(F, H, M, mu_param, c, alpha, epsilon, C_M, C_R, C_S, C_P, mu_0)

    # --- Precompute constants ---
    phi_inv = norm.ppf(1 - epsilon)

    # Reliability upper bound per flight: alpha - sqrt(alpha / c_i) * Phi^{-1}(1 - epsilon)
    mu_upper = np.zeros(F)
    for i in range(F):
        mu_upper[i] = alpha - np.sqrt(alpha / c[i]) * phi_inv

    # Helper: get wrapped input parameter values (0-indexed everywhere)
    def mu_input(i, j, k):
        return float(mu_param[i, j, k % H])

    # --- Build Gurobi model ---
    # Index mapping (PDF 1-based -> Python 0-based):
    #   i: PDF {1,...,F}   -> Python 0..F-1
    #   j: PDF {0,...,M}   -> Python 0..M
    #   k: PDF {1,...,2H}  -> Python 0..2H-1
    model = gp.Model("fleet_management_inverse_gaussian_degradation")
    model.Params.OutputFlag = int(verbose)
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    # Decision variables
    x = model.addVars(F, M + 1, 2 * H, vtype=GRB.BINARY, name="x")
    mu_var = model.addVars(F, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="mu")
    u_var = model.addVars(2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="u")
    z_var = model.addVars(F, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="z")

    # --- Objective ---
    # min C_M * sum_{k,i} x_{i,0,k}
    #   + C_R * sum_{k,i} z_{ik}
    #   + C_S * sum_{k} u_{k}
    #   + C_P * sum_i (mu_{i,H-1} - mu_{i,2H-1})
    obj = gp.LinExpr()
    for k in range(2 * H):
        obj += C_S * u_var[k]
        for i in range(F):
            obj += C_M * x[i, 0, k]
            obj += C_R * z_var[i, k]
    for i in range(F):
        obj += C_P * (mu_var[i, H - 1] - mu_var[i, 2 * H - 1])
    model.setObjective(obj, GRB.MINIMIZE)

    # --- Constraints ---

    # (1) sum_i mu_{ik} <= F - M,  for all k
    for k in range(2 * H):
        model.addConstr(
            gp.quicksum(mu_var[i, k] for i in range(F)) <= F - M,
            name=f"capacity_{k}",
        )

    # (2) Reliability constraint:
    #   mu_{ik} <= alpha - sqrt(alpha / c_i) * Phi^{-1}(1 - epsilon),  for all i, k
    for i in range(F):
        for k in range(2 * H):
            model.addConstr(
                mu_var[i, k] <= mu_upper[i],
                name=f"reliability_{i}_{k}",
            )

    for i in range(F):
        for k in range(2 * H):
            # Previous-step values: when k=0 (PDF k=1), use initial conditions
            if k == 0:
                mu_prev = mu_0[i]
            else:
                mu_prev = mu_var[i, k - 1]

            # (3) mu update:
            #   mu_{ik} >= mu_{ik-1} + sum_{j=1}^{M} x_{ijk} * mu_input_{ijk}
            #              - 2H * x_{i,0,k}
            model.addConstr(
                mu_var[i, k]
                >= mu_prev
                + gp.quicksum(
                    x[i, j, k] * mu_input(i, j - 1, k)
                    for j in range(1, M + 1)
                )
                - 2 * H * x[i, 0, k],
                name=f"mu_update_{i}_{k}",
            )

            # z bound:
            #   z_{ik} >= mu_{ik-1} - 2H + 2H * x_{i,0,k}
            model.addConstr(
                z_var[i, k] >= mu_prev - 2 * H + 2 * H * x[i, 0, k],
                name=f"z_bound_{i}_{k}",
            )

    # mu_{i,2H} <= mu_{iH}  ->  mu_var[i, 2H-1] <= mu_var[i, H-1]
    for i in range(F):
        model.addConstr(
            mu_var[i, 2 * H - 1] <= mu_var[i, H - 1],
            name=f"mu_periodic_{i}",
        )

    # u_k >= mu_{ik},  for all i, k
    for k in range(2 * H):
        for i in range(F):
            model.addConstr(
                u_var[k] >= mu_var[i, k],
                name=f"u_bound_{i}_{k}",
            )

    # sum_{j=0}^{M} x_{ijk} <= 1,  for all i, k
    for i in range(F):
        for k in range(2 * H):
            model.addConstr(
                gp.quicksum(x[i, j, k] for j in range(M + 1)) <= 1,
                name=f"assignment_{i}_{k}",
            )

    # sum_{i=1}^{F} x_{ijk} = 1,  for all j, k
    for j in range(M + 1):
        for k in range(2 * H):
            model.addConstr(
                gp.quicksum(x[i, j, k] for i in range(F)) == 1,
                name=f"demand_{j}_{k}",
            )

    # --- Solve ---
    model.optimize()

    # --- Extract results ---
    if model.status == GRB.OPTIMAL:
        x_sol = np.zeros((F, M + 1, 2 * H))
        mu_sol = np.zeros((F, 2 * H))
        u_sol = np.zeros(2 * H)
        z_sol = np.zeros((F, 2 * H))

        for k in range(2 * H):
            u_sol[k] = u_var[k].X
            for i in range(F):
                mu_sol[i, k] = mu_var[i, k].X
                z_sol[i, k] = z_var[i, k].X
                for j in range(M + 1):
                    x_sol[i, j, k] = x[i, j, k].X

        return {
            "status": "optimal",
            "objective": model.ObjVal,
            "F": F,
            "H": H,
            "M": M,
            "alpha": alpha,
            "x": x_sol,
            "mu": mu_sol,
            "u": u_sol,
            "z": z_sol,
            "model": model,
        }
    else:
        return {
            "status": model.status,
            "objective": None,
            "F": F,
            "H": H,
            "M": M,
            "alpha": alpha,
            "x": None,
            "mu": None,
            "u": None,
            "z": None,
            "model": model,
        }
