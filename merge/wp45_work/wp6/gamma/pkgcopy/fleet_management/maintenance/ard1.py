"""Shared ARD1 (Arithmetic Reduction of Degradation, order 1) big-M dynamics.

Used by every degradation model that supports ARD1 maintenance. On a repair day
the tracked state is scaled by a fixed ``repair_factor`` (``1 - rho`` for a
mean/shape state, ``(1 - rho) ** 2`` for a variance state); on any other day it
accumulates the mission-induced increment. Both rows are the standard big-M
linearisation of the two-branch exact recursion and are exact at binary points
of ``(x_m, x_r)`` (spec/spec.tex Sections 5.3-5.7).
"""


def accumulate_and_repair(model, var, prev_expr, delta_expr, repair_factor,
                           x_m, x_r, big_m, name):
    """Add the ARD1 big-M accumulate/repair rows for one (component, step).

    Parameters
    ----------
    model : gurobipy.Model
    var : gurobipy.Var
        The tracked-state variable at the current step (mu, v, or alpha).
    prev_expr : gurobipy.Var or float
        The tracked state at the previous step (or the initial condition).
    delta_expr : gurobipy.LinExpr
        The mission-induced increment at the current step.
    repair_factor : float
        Multiplicative factor applied to ``prev_expr`` on a repair day
        (``1 - rho`` for mean/shape states, ``(1 - rho) ** 2`` for variance).
    x_m, x_r : gurobipy.Var
        Repair and replacement binaries at the current step.
    big_m : float
        Big-M constant (``tau``, ``V_max``, or ``alpha_bar`` depending on state).
    name : str
        Constraint name prefix.
    """
    model.addConstr(
        var >= prev_expr + delta_expr - big_m * x_m - big_m * x_r,
        name=f"{name}_accum",
    )
    model.addConstr(
        var >= repair_factor * prev_expr - big_m * (1 - x_m),
        name=f"{name}_repair",
    )
