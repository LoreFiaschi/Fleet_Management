import numpy as np
from scipy.stats import norm
import gurobipy as gp
from gurobipy import GRB


def validate_inputs(
    F: int,
    H: int,
    M: int,
    L: int,
    mu_param: np.ndarray,
    c: np.ndarray,
    alpha: float,
    epsilon: float,
    xi: np.ndarray,
    C_M: float,
    C_R: float,
    C_S: float,
    C_P: float,
    mu_0: np.ndarray,
) -> None:
    """Run all consistency checks for the inverse Gaussian degradation model."""
    if F <= 0 or H <= 0 or M <= 0 or L <= 0:
        raise ValueError("F, H, M, L must be positive integers.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if C_M <= 0 or C_R <= 0 or C_S <= 0 or C_P <= 0:
        raise ValueError("All cost coefficients must be positive.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    if F <= M:
        raise ValueError(f"F must be greater than M (got F={F}, M={M}).")

    if mu_param.shape != (F, M, L, H):
        raise ValueError(
            f"mu shape must be ({F}, {M}, {L}, {H}), got {mu_param.shape}."
        )

    if epsilon >= 0.5:
        raise ValueError(f"epsilon must be < 0.5 (got {epsilon}).")

    if xi.shape != (F, L):
        raise ValueError(f"xi must have shape ({F}, {L}).")
    if not np.all(xi > 0):
        raise ValueError("All entries of xi must be positive.")
    if not np.all(xi <= 1):
        raise ValueError("xi must be <= 1 element-wise.")

    if c.shape != (F, L):
        raise ValueError(f"c must have shape ({F}, {L}).")

    if mu_0.shape != (F, L):
        raise ValueError(f"mu_0 must have shape ({F}, {L}).")

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
    L: int,
    mu_param: np.ndarray,
    c: np.ndarray,
    alpha: float,
    epsilon: float,
    xi: np.ndarray,
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
    L : int
        Number of components per train.
    mu_param : np.ndarray, shape (F, M, L, H)
        Mean degradation parameters.
    c : np.ndarray, shape (F, L)
        Shape parameter per flight and component for the inverse Gaussian model.
    alpha : float
        Upper bound for degradation mean (must be positive).
    epsilon : float
        Reliability threshold (must be in (0, 0.5)).
    xi : np.ndarray, shape (F, L)
        Fraction of damage repairable in one maintenance day per train and component.
    C_M : float
        Maintenance cost coefficient.
    C_R : float
        Repair cost coefficient.
    C_S : float
        Safety cost coefficient.
    C_P : float
        Penalty cost coefficient.
    mu_0 : np.ndarray, shape (F, L)
        Initial mean values per flight and component.
    verbose : int, optional
        Gurobi output verbosity flag (0 = silent, 1 = normal). Default is 1.
    mip_gap : float, optional
        Relative MIP optimality gap tolerance. If None, Gurobi's default is used.

    Returns
    -------
    dict
        Keys: "status", "objective", "x", "mu", "u", "z",
              "F", "H", "M", "L", "alpha", "model".
    """
    # --- Consistency checks ---
    validate_inputs(F, H, M, L, mu_param, c, alpha, epsilon, xi,
                    C_M, C_R, C_S, C_P, mu_0)

    # --- Precompute constants ---
    phi_inv = norm.ppf(1 - epsilon)

    # Reliability upper bound per (i, l): alpha - sqrt(alpha / c_{il}) * Phi^{-1}(1 - epsilon)
    mu_upper = np.zeros((F, L))
    for i in range(F):
        for l in range(L):
            mu_upper[i, l] = alpha - np.sqrt(alpha / c[i, l]) * phi_inv

    # Helper: get wrapped input parameter values (0-indexed everywhere)
    def mu_input(i, j, l, k):
        return float(mu_param[i, j, l, k % H])

    # --- Build Gurobi model ---
    # Index mapping (PDF 1-based -> Python 0-based):
    #   i: PDF {1,...,F}   -> Python 0..F-1
    #   j: PDF {0,...,M}   -> Python 0..M
    #   k: PDF {1,...,2H}  -> Python 0..2H-1
    #   l: PDF {1,...,L}   -> Python 0..L-1
    model = gp.Model("fleet_management_inverse_gaussian_degradation")
    model.Params.OutputFlag = int(verbose)
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    # Decision variables
    x = model.addVars(F, M + 1, 2 * H, vtype=GRB.BINARY, name="x")
    mu_var = model.addVars(F, L, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="mu")
    u_var = model.addVars(2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="u")
    z_var = model.addVars(F, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="z")

    # --- Objective ---
    # min C_M * sum_{k,i} x_{i,0,k}
    #   + C_R * sum_{k,i} z_{ik}
    #   + C_S * sum_{k} u_{k}
    #   + C_P * sum_{i,l} (mu_{il,H} - mu_{il,2H})
    obj = gp.LinExpr()
    for k in range(2 * H):
        obj += C_S * u_var[k]
        for i in range(F):
            obj += C_M * x[i, 0, k]
            obj += C_R * z_var[i, k]
    for i in range(F):
        for l in range(L):
            obj += C_P * (mu_var[i, l, H - 1] - mu_var[i, l, 2 * H - 1])
    model.setObjective(obj, GRB.MINIMIZE)

    # --- Constraints ---

    # (1) sum_{i,l} mu_{ilk} <= F - M,  for all k
    for k in range(2 * H):
        model.addConstr(
            gp.quicksum(mu_var[i, l, k] for i in range(F) for l in range(L))
            <= F - M,
            name=f"capacity_{k}",
        )

    # (2) Reliability constraint:
    #   mu_{ilk} <= alpha - sqrt(alpha / c_{il}) * Phi^{-1}(1 - epsilon),  for all i, l, k
    for i in range(F):
        for l in range(L):
            for k in range(2 * H):
                model.addConstr(
                    mu_var[i, l, k] <= mu_upper[i, l],
                    name=f"reliability_{i}_{l}_{k}",
                )

    for i in range(F):
        for l in range(L):
            for k in range(2 * H):
                # Previous-step values: when k=0 (PDF k=1), use initial conditions
                if k == 0:
                    mu_prev = mu_0[i, l]
                else:
                    mu_prev = mu_var[i, l, k - 1]

                # (3) mu update:
                #   mu_{ilk} >= mu_{ilk-1} + sum_{j=1}^{M} x_{ijk} * mu_input_{ijlk}
                #               - alpha * x_{i,0,k}
                model.addConstr(
                    mu_var[i, l, k]
                    >= mu_prev
                    + gp.quicksum(
                        x[i, j, k] * mu_input(i, j - 1, l, k)
                        for j in range(1, M + 1)
                    )
                    - alpha * x[i, 0, k],
                    name=f"mu_update_{i}_{l}_{k}",
                )

                # mu lower bound during maintenance:
                #   mu_{ilk} >= mu_{ilk-1} * (1 - xi_{il})
                model.addConstr(
                    mu_var[i, l, k] >= mu_prev * (1 - xi[i, l]),
                    name=f"mu_lb_{i}_{l}_{k}",
                )

                # z bound:
                #   z_{ik} >= mu_{ilk-1} * xi_{il} - alpha + alpha * x_{i,0,k}
                model.addConstr(
                    z_var[i, k] >= mu_prev * xi[i, l] - alpha + alpha * x[i, 0, k],
                    name=f"z_bound_{i}_{l}_{k}",
                )

    # mu periodic: mu_{il,2H} <= mu_{il,H}
    for i in range(F):
        for l in range(L):
            model.addConstr(
                mu_var[i, l, 2 * H - 1] <= mu_var[i, l, H - 1],
                name=f"mu_periodic_{i}_{l}",
            )

    # u_k >= mu_{ilk},  for all i, l, k
    for k in range(2 * H):
        for i in range(F):
            for l in range(L):
                model.addConstr(
                    u_var[k] >= mu_var[i, l, k],
                    name=f"u_bound_{i}_{l}_{k}",
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
        mu_sol = np.zeros((F, L, 2 * H))
        u_sol = np.zeros(2 * H)
        z_sol = np.zeros((F, 2 * H))

        for k in range(2 * H):
            u_sol[k] = u_var[k].X
            for i in range(F):
                z_sol[i, k] = z_var[i, k].X
                for l in range(L):
                    mu_sol[i, l, k] = mu_var[i, l, k].X
                for j in range(M + 1):
                    x_sol[i, j, k] = x[i, j, k].X

        return {
            "status": "optimal",
            "objective": model.ObjVal,
            "F": F,
            "H": H,
            "M": M,
            "L": L,
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
            "L": L,
            "alpha": alpha,
            "x": None,
            "mu": None,
            "u": None,
            "z": None,
            "model": model,
        }
