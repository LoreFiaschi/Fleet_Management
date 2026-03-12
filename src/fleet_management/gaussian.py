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
    v_param: np.ndarray,
    alpha: float,
    epsilon: float,
    xi: np.ndarray,
    C_M: float,
    C_R: float,
    C_S: float,
    C_P: float,
    mu_0: np.ndarray,
    v_0: np.ndarray,
) -> None:
    """Run all consistency checks from the problem specification."""
    # All variables must be positive
    if F <= 0 or H <= 0 or M <= 0 or L <= 0:
        raise ValueError("F, H, M, L must be positive integers.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if C_M <= 0 or C_R <= 0 or C_S <= 0 or C_P <= 0:
        raise ValueError("All cost coefficients must be positive.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    # F > M
    if F <= M:
        raise ValueError(f"F must be greater than M (got F={F}, M={M}).")

    # mu and v shape should be F x M x L x H
    if mu_param.shape != (F, M, L, H):
        raise ValueError(
            f"mu shape must be ({F}, {M}, {L}, {H}), got {mu_param.shape}."
        )
    if v_param.shape != (F, M, L, H):
        raise ValueError(
            f"v shape must be ({F}, {M}, {L}, {H}), got {v_param.shape}."
        )

    # epsilon < 0.5
    if epsilon >= 0.5:
        raise ValueError(f"epsilon must be < 0.5 (got {epsilon}).")

    # xi shape is F x L, element-wise <= 1
    if xi.shape != (F, L):
        raise ValueError(f"xi must have shape ({F}, {L}).")
    if not np.all(xi > 0):
        raise ValueError("All entries of xi must be positive.")
    if not np.all(xi <= 1):
        raise ValueError("xi must be <= 1 element-wise.")

    # mu_0 and v_0 shape is F x L
    if mu_0.shape != (F, L):
        raise ValueError(f"mu_0 must have shape ({F}, {L}).")
    if v_0.shape != (F, L):
        raise ValueError(f"v_0 must have shape ({F}, {L}).")

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
    L: int,
    mu_param: np.ndarray,
    v_param: np.ndarray,
    phi_inv_sq: float,
) -> np.ndarray:
    """
    Precompute the W tensor of dimension F x M x L x 2H.

    W[i,j,l,k] = mu_ijlk^2 + 2*mu_ijlk - v_ijlk * (Phi^{-1}(1-eps))^2

    For k >= H the input parameters wrap: index (k % H) is used.
    All indices are 0-based.
    """
    W = np.zeros((F, M, L, 2 * H))
    for k in range(2 * H):
        k_wrapped = k % H
        mu_slice = mu_param[:, :, :, k_wrapped]  # (F, M, L)
        v_slice = v_param[:, :, :, k_wrapped]  # (F, M, L)
        W[:, :, :, k] = mu_slice ** 2 + 2 * mu_slice - v_slice * phi_inv_sq
    return W


def solve_fleet_management(
    F: int,
    H: int,
    M: int,
    L: int,
    mu_param: np.ndarray,
    v_param: np.ndarray,
    alpha: float,
    epsilon: float,
    xi: np.ndarray,
    C_M: float,
    C_R: float,
    C_S: float,
    C_P: float,
    mu_0: np.ndarray,
    v_0: np.ndarray,
    verbose: int = 1,
    mip_gap: float = None,
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
    L : int
        Number of components per train.
    mu_param : np.ndarray, shape (F, M, L, H)
        Mean degradation parameters.
    v_param : np.ndarray, shape (F, M, L, H)
        Variance degradation parameters.
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
        Initial mean degradation values per flight and component.
    v_0 : np.ndarray, shape (F, L)
        Initial variance values per flight and component.
    verbose : int, optional
        Gurobi output verbosity flag (0 = silent, 1 = normal). Default is 1.
    mip_gap : float, optional
        Relative MIP optimality gap tolerance. If None, Gurobi's default is used.

    Returns
    -------
    dict
        Keys: "status", "objective", "x", "mu", "v", "z", "u",
              "F", "H", "M", "L", "alpha", "model".
    """
    # --- Consistency checks ---
    validate_inputs(F, H, M, L, mu_param, v_param, alpha, epsilon, xi,
                    C_M, C_R, C_S, C_P, mu_0, v_0)

    # --- Precompute constants ---
    phi_inv = norm.ppf(1 - epsilon)
    phi_inv_sq = phi_inv ** 2

    W = compute_W(F, H, M, L, mu_param, v_param, phi_inv_sq)

    # Helper: get wrapped input parameter values (0-indexed everywhere)
    def mu_input(i, j, l, k):
        return float(mu_param[i, j, l, k % H])

    def v_input(i, j, l, k):
        return float(v_param[i, j, l, k % H])

    # --- Build Gurobi model ---
    # Index mapping (PDF 1-based -> Python 0-based):
    #   i: PDF {1,...,F}   -> Python 0..F-1
    #   j: PDF {0,...,M}   -> Python 0..M
    #   k: PDF {1,...,2H}  -> Python 0..2H-1
    #   l: PDF {1,...,L}   -> Python 0..L-1
    model = gp.Model("fleet_management_gaussian_degradation")
    model.Params.OutputFlag = int(verbose)
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    # Decision variables
    x = model.addVars(F, M + 1, 2 * H, vtype=GRB.BINARY, name="x")
    mu_var = model.addVars(F, L, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="mu")
    v_var = model.addVars(F, L, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="v")
    u_var = model.addVars(2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="u")
    z_var = model.addVars(F, 2 * H, vtype=GRB.CONTINUOUS, lb=0.0, name="z")

    # --- Objective ---
    # min C_M * sum_{k,i} x_{i,0,k}
    #   + C_R * sum_{k,i} z_{ik}
    #   + C_S * sum_{k} u_{k}
    #   + C_P * sum_{i,l} (mu_{i,l,H} - mu_{i,l,2H} + v_{i,l,H} - v_{i,l,2H})
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

    # --- Constraints ---

    # (1) sum_{i,l} mu_{ilk} <= F - M,  for all k
    for k in range(2 * H):
        model.addConstr(
            gp.quicksum(mu_var[i, l, k] for i in range(F) for l in range(L))
            <= F - M,
            name=f"capacity_{k}",
        )

    for i in range(F):
        for l in range(L):
            for k in range(2 * H):
                # Previous-step values: when k=0 (PDF k=1), use initial conditions
                if k == 0:
                    mu_prev = mu_0[i, l]
                    v_prev = v_0[i, l]
                else:
                    mu_prev = mu_var[i, l, k - 1]
                    v_prev = v_var[i, l, k - 1]

                # (2) Reliability constraint:
                #   v_{ilk-1} * (phi_inv^2 - 9) + 2*mu_{ilk-1} - 3*alpha*x_{i,0,k}
                #     <= alpha + sum_{j=1}^{M} x_{i,j,k} * W_{i,j-1,l,k}
                model.addConstr(
                    v_prev * (phi_inv_sq - 9)
                    + 2 * mu_prev
                    - 3 * alpha * x[i, 0, k]
                    <= alpha + gp.quicksum(
                        x[i, j, k] * W[i, j - 1, l, k] for j in range(1, M + 1)
                    ),
                    name=f"reliability_{i}_{l}_{k}",
                )

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

                # (3b) mu lower bound during maintenance:
                #   mu_{ilk} >= mu_{ilk-1} * (1 - xi_{il})
                model.addConstr(
                    mu_var[i, l, k] >= mu_prev * (1 - xi[i, l]),
                    name=f"mu_lb_{i}_{l}_{k}",
                )

                # (4) v update:
                #   v_{ilk} >= v_{ilk-1} + sum_{j=1}^{M} x_{ijk} * v_input_{ijlk}
                #              - alpha * x_{i,0,k}
                model.addConstr(
                    v_var[i, l, k]
                    >= v_prev
                    + gp.quicksum(
                        x[i, j, k] * v_input(i, j - 1, l, k)
                        for j in range(1, M + 1)
                    )
                    - alpha * x[i, 0, k],
                    name=f"v_update_{i}_{l}_{k}",
                )

                # (4b) v lower bound during maintenance:
                #   v_{ilk} >= v_{ilk-1} * (1 - xi_{il})
                model.addConstr(
                    v_var[i, l, k] >= v_prev * (1 - xi[i, l]),
                    name=f"v_lb_{i}_{l}_{k}",
                )

    # (5) mu_{il,2H} <= mu_{il,H}  ->  mu_var[i, l, 2H-1] <= mu_var[i, l, H-1]
    for i in range(F):
        for l in range(L):
            model.addConstr(
                mu_var[i, l, 2 * H - 1] <= mu_var[i, l, H - 1],
                name=f"mu_periodic_{i}_{l}",
            )

    # (6) v_{il,2H} <= v_{il,H}  ->  v_var[i, l, 2H-1] <= v_var[i, l, H-1]
    for i in range(F):
        for l in range(L):
            model.addConstr(
                v_var[i, l, 2 * H - 1] <= v_var[i, l, H - 1],
                name=f"v_periodic_{i}_{l}",
            )

    # u_k >= mu_{ilk},  for all i, l, k
    for k in range(2 * H):
        for i in range(F):
            for l in range(L):
                model.addConstr(
                    u_var[k] >= mu_var[i, l, k],
                    name=f"u_bound_{i}_{l}_{k}",
                )

    # z bound: z_{ik} >= sum_l mu_{ilk} * xi_{il} - alpha + alpha * x_{i,0,k}
    for i in range(F):
        for k in range(2 * H):
            model.addConstr(
                z_var[i, k]
                >= gp.quicksum(mu_var[i, l, k] * xi[i, l] for l in range(L))
                - alpha + alpha * x[i, 0, k],
                name=f"z_bound_{i}_{k}",
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
            "status": "optimal",
            "objective": model.ObjVal,
            "F": F,
            "H": H,
            "M": M,
            "L": L,
            "alpha": alpha,
            "x": x_sol,
            "mu": mu_sol,
            "v": v_sol,
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
            "v": None,
            "u": None,
            "z": None,
            "model": model,
        }
