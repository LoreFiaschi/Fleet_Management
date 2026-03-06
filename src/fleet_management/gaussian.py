import numpy as np
from scipy.stats import norm
import gurobipy as gp
from gurobipy import GRB


def validate_inputs(
    F: int,
    H: int,
    M: int,
    mu_param: np.ndarray,
    v_param: np.ndarray,
    alpha: float,
    epsilon: float,
    C_M: float,
    C_R: float,
    C_S: float,
    C_P: float,
    mu_0: np.ndarray,
    v_0: np.ndarray,
) -> None:
    """Run all consistency checks from the problem specification."""
    # All variables must be positive
    if F <= 0 or H <= 0 or M <= 0:
        raise ValueError("F, H, M must be positive integers.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if C_M <= 0 or C_R <= 0 or C_S <= 0 or C_P <= 0:
        raise ValueError("All cost coefficients must be positive.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    # F > M
    if F <= M:
        raise ValueError(f"F must be greater than M (got F={F}, M={M}).")

    # mu and v shape should be F x M x H
    if mu_param.shape != (F, M, H):
        raise ValueError(
            f"mu shape must be ({F}, {M}, {H}), got {mu_param.shape}."
        )
    if v_param.shape != (F, M, H):
        raise ValueError(
            f"v shape must be ({F}, {M}, {H}), got {v_param.shape}."
        )

    # epsilon < 0.5
    if epsilon >= 0.5:
        raise ValueError(f"epsilon must be < 0.5 (got {epsilon}).")

    # mu_0 and v_0 are 1D vectors of length F
    if mu_0.shape != (F,):
        raise ValueError(f"mu_0 must be a 1D vector of length {F}.")
    if v_0.shape != (F,):
        raise ValueError(f"v_0 must be a 1D vector of length {F}.")

    # All values must be positive
    if not np.all(mu_param > 0):
        raise ValueError("All entries of mu must be positive.")
    if not np.all(v_param > 0):
        raise ValueError("All entries of v must be positive.")
    if not np.all(mu_0 > 0):
        raise ValueError("All entries of mu_0 must be positive.")
    if not np.all(v_0 > 0):
        raise ValueError("All entries of v_0 must be positive.")

    # mu >= 3*sqrt(v) element-wise
    if not np.all(mu_param >= 3 * np.sqrt(v_param)):
        raise ValueError("mu must be >= 3*sqrt(v) element-wise.")

    # mu_0 >= 3*sqrt(v_0) element-wise
    if not np.all(mu_0 >= 3 * np.sqrt(v_0)):
        raise ValueError("mu_0 must be >= 3*sqrt(v_0) element-wise.")

    # mu < alpha element-wise
    if not np.all(mu_param < alpha):
        raise ValueError(f"mu must be < alpha={alpha} element-wise.")

    # mu_0 < alpha element-wise
    if not np.all(mu_0 < alpha):
        raise ValueError(f"mu_0 must be < alpha={alpha} element-wise.")


def compute_W(
    F: int,
    H: int,
    M: int,
    mu_param: np.ndarray,
    v_param: np.ndarray,
    phi_inv_sq: float,
) -> np.ndarray:
    """
    Precompute the W tensor of dimension F x M x 2H.

    W[i,j,k] = mu_ij(k)^2 + 2*mu_ij(k) - v_ij(k) * (Phi^{-1}(1-eps))^2

    For k >= H the input parameters wrap: index (k % H) is used.
    All indices are 0-based.
    """
    W = np.zeros((F, M, 2 * H))
    for k in range(2 * H):
        k_wrapped = k % H
        mu_slice = mu_param[:, :, k_wrapped]  # (F, M)
        v_slice = v_param[:, :, k_wrapped]  # (F, M)
        W[:, :, k] = mu_slice ** 2 + 2 * mu_slice - v_slice * phi_inv_sq
    return W


def solve_fleet_management(
    F: int,
    H: int,
    M: int,
    mu_param: np.ndarray,
    v_param: np.ndarray,
    alpha: float,
    epsilon: float,
    C_M: float,
    C_R: float,
    C_S: float,
    C_P: float,
    mu_0: np.ndarray,
    v_0: np.ndarray,
    verbose: int = 1,
) -> dict:
    """
    Solve the Fleet Management MILP with Gaussian degradation using Gurobi.

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
    v_param : np.ndarray, shape (F, M, H)
        Variance degradation parameters.
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
    v_0 : np.ndarray, shape (F,)
        Initial variance values per flight.
    verbose : int, optional
        Gurobi output verbosity flag (0 = silent, 1 = normal). Default is 1.

    Returns
    -------
    dict
        Keys: "status", "objective", "x", "mu", "v", "z", "F", "H", "M", "alpha", "model".
    """
    # --- Consistency checks ---
    validate_inputs(F, H, M, mu_param, v_param, alpha, epsilon, C_M, C_R, C_S, C_P, mu_0, v_0)

    # --- Precompute constants ---
    phi_inv = norm.ppf(1 - epsilon)
    phi_inv_sq = phi_inv ** 2

    W = compute_W(F, H, M, mu_param, v_param, phi_inv_sq)

    # Helper: get wrapped input parameter values (0-indexed everywhere)
    def mu_input(i, j, k):
        return float(mu_param[i, j, k % H])

    def v_input(i, j, k):
        return float(v_param[i, j, k % H])

    # --- Build Gurobi model ---
    # Index mapping (PDF 1-based -> Python 0-based):
    #   i: PDF {1,...,F}   -> Python 0..F-1
    #   j: PDF {0,...,M}   -> Python 0..M
    #   k: PDF {1,...,2H}  -> Python 0..2H-1
    model = gp.Model("fleet_management_gaussian_degradation")
    model.Params.OutputFlag = int(verbose)

    # Decision variables
    x = model.addVars(F, M + 1, 2 * H, vtype=GRB.BINARY, name="x")
    mu_var = model.addVars(F, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="mu")
    v_var = model.addVars(F, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="v")
    z_var = model.addVars(F, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="z")

    # --- Objective ---
    # min C_M * sum_{k,i} x_{i,0,k}
    #   + C_R * sum_{k,i} z_{ik}
    #   + C_S * sum_{k,i} mu_{ik}
    #   + C_P * sum_i (mu_{i,H-1} - mu_{i,2H-1} + v_{i,H-1} - v_{i,2H-1})
    obj = gp.LinExpr()
    for k in range(2 * H):
        for i in range(F):
            obj += C_M * x[i, 0, k]
            obj += C_R * z_var[i, k]
            obj += C_S * mu_var[i, k]
    for i in range(F):
        obj += C_P * (
            mu_var[i, H - 1] - mu_var[i, 2 * H - 1]
            + v_var[i, H - 1] - v_var[i, 2 * H - 1]
        )
    model.setObjective(obj, GRB.MINIMIZE)

    # --- Constraints ---

    # (1) sum_i mu_{ik} <= F - M,  for all k
    for k in range(2 * H):
        model.addConstr(
            gp.quicksum(mu_var[i, k] for i in range(F)) <= F - M,
            name=f"capacity_{k}",
        )

    for i in range(F):
        for k in range(2 * H):
            # Previous-step values: when k=0 (PDF k=1), use initial conditions
            if k == 0:
                mu_prev = mu_0[i]
                v_prev = v_0[i]
            else:
                mu_prev = mu_var[i, k - 1]
                v_prev = v_var[i, k - 1]

            # (2) Reliability constraint:
            #   v_{ik-1} * (phi_inv^2 - 9) + 2*mu_{ik-1} - 2H*x_{i,0,k}
            #     <= alpha + sum_{j=1}^{M} x_{i,j,k} * W_{i,j-1,k}
            model.addConstr(
                v_prev * (phi_inv_sq - 9)
                + 2 * mu_prev
                - 2 * H * x[i, 0, k]
                <= alpha + gp.quicksum(
                    x[i, j, k] * W[i, j - 1, k] for j in range(1, M + 1)
                ),
                name=f"reliability_{i}_{k}",
            )

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

            # (4) v update:
            #   v_{ik} >= v_{ik-1} + sum_{j=1}^{M} x_{ijk} * v_input_{ijk}
            #             - 2H * x_{i,0,k}
            model.addConstr(
                v_var[i, k]
                >= v_prev
                + gp.quicksum(
                    x[i, j, k] * v_input(i, j - 1, k)
                    for j in range(1, M + 1)
                )
                - 2 * H * x[i, 0, k],
                name=f"v_update_{i}_{k}",
            )

            # (7) z bound:
            #   z_{ik} >= mu_{ik-1} - 2H + 2H * x_{i,0,k}
            model.addConstr(
                z_var[i, k] >= mu_prev - 2 * H + 2 * H * x[i, 0, k],
                name=f"z_bound_{i}_{k}",
            )

    # (5) mu_{i,2H} <= mu_{iH}  ->  mu_var[i, 2H-1] <= mu_var[i, H-1]
    for i in range(F):
        model.addConstr(
            mu_var[i, 2 * H - 1] <= mu_var[i, H - 1],
            name=f"mu_periodic_{i}",
        )

    # (6) v_{i,2H} <= v_{iH}  ->  v_var[i, 2H-1] <= v_var[i, H-1]
    for i in range(F):
        model.addConstr(
            v_var[i, 2 * H - 1] <= v_var[i, H - 1],
            name=f"v_periodic_{i}",
        )

    # (8) sum_{j=0}^{M} x_{ijk} <= 1,  for all i, k
    for i in range(F):
        for k in range(2 * H):
            model.addConstr(
                gp.quicksum(x[i, j, k] for j in range(M + 1)) <= 1,
                name=f"assignment_{i}_{k}",
            )

    # (9) sum_{i=1}^{F} x_{ijk} = 1,  for all j, k
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
        v_sol = np.zeros((F, 2 * H))
        z_sol = np.zeros((F, 2 * H))

        for i in range(F):
            for k in range(2 * H):
                mu_sol[i, k] = mu_var[i, k].X
                v_sol[i, k] = v_var[i, k].X
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
            "v": v_sol,
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
            "v": None,
            "z": None,
            "model": model,
        }
