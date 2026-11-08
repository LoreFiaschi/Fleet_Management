"""Shared ARA1 (Arithmetic Reduction of Age, order 1) big-M dynamics.

Used by every degradation model that supports ARA1 maintenance (Wiener on its
mean, Gamma on its shape). Unlike ARD1, repair scales the tracked state toward
an auxiliary "anchor" -- the state at the last maintenance epoch -- rather
than toward zero: on a repair day the state is reduced by a fraction rho of
the degradation accumulated *since that anchor*, not of its total value. Both
rows are the standard big-M linearisation of the two-branch exact recursion
and are exact at binary points of (x_m, x_r) (spec/spec.tex Sections 5.4.2,
5.5.2).

Deviation from the spec's literal (ARA1-only) equations, flagged explicitly:
``update_anchor`` also resets the anchor on a replacement day. The spec's own
normative rows don't do this -- and its "Open Questions" section names the
gap directly, calling the reset "the natural fix" and confirming it "being a
constant map, also preserves the monotonicity argument" used by the loop-
constraint chaining proof. Implementing it isn't optional here: without it,
the invariant ``tracked_state >= anchor`` (which every ARD1 component
trivially satisfies, anchor=0) can break the instant a component is ever
replaced -- the state resets low while the stale anchor stays high, so the
next `gap = state - anchor` used by the repair-cost McCormick envelope
(``models/base.add_repair_cost_mccormick``) goes negative, which combined
with ``z``'s ``lb=0`` makes the model spuriously infeasible. Confirmed via
Gurobi's IIS on a forced-replacement test before this fix.
"""


def update_anchor(model, anchor_var, prev_expr, anchor_prev_expr, new_val, x_m, x_r,
                   big_m, name):
    """Three-way big-M update of the maintenance-epoch anchor variable.

    On a repair day (x_m=1) the anchor snaps to the pre-repair state; on a
    replacement day (x_r=1) it snaps to the reset value (see module
    docstring); otherwise it holds its previous value. x_m and x_r are
    mutually exclusive (spec's maintenance/replacement compatibility
    constraint), so exactly one case is ever active; all three rows are
    two-sided, hence exact at binary points.
    """
    model.addConstr(
        anchor_var >= prev_expr - big_m * (1 - x_m) - big_m * x_r, name=f"{name}_i_lb"
    )
    model.addConstr(
        anchor_var <= prev_expr + big_m * (1 - x_m) + big_m * x_r, name=f"{name}_i_ub"
    )
    model.addConstr(
        anchor_var >= new_val - big_m * (1 - x_r) - big_m * x_m, name=f"{name}_ii_lb"
    )
    model.addConstr(
        anchor_var <= new_val + big_m * (1 - x_r) + big_m * x_m, name=f"{name}_ii_ub"
    )
    model.addConstr(
        anchor_var >= anchor_prev_expr - big_m * x_m - big_m * x_r, name=f"{name}_iii_lb"
    )
    model.addConstr(
        anchor_var <= anchor_prev_expr + big_m * x_m + big_m * x_r, name=f"{name}_iii_ub"
    )


def accumulate_and_repair(model, var, prev_expr, delta_expr, anchor_prev_expr, rho,
                           x_m, x_r, big_m, name):
    """Add the ARA1 big-M accumulate/repair rows for one (component, step).

    The accumulate row is identical in form to ARD1's; only the repair row
    differs, reducing the state toward the anchor (rather than toward zero)
    by a fraction rho of the gap accumulated since it.

    Parameters
    ----------
    model : gurobipy.Model
    var : gurobipy.Var
        The tracked-state variable at the current step (mu or alpha).
    prev_expr : gurobipy.Var or float
        The tracked state at the previous step (or the initial condition).
    delta_expr : gurobipy.LinExpr
        The mission-induced increment at the current step.
    anchor_prev_expr : gurobipy.Var or float
        The maintenance-epoch anchor at the previous step (or its initial
        condition, equal to the state's own initial condition).
    rho : float
        Repair efficiency, applied to the gap (prev_expr - anchor_prev_expr).
    x_m, x_r : gurobipy.Var
        Repair and replacement binaries at the current step.
    big_m : float
        Big-M constant (tau or alpha_bar depending on state).
    name : str
        Constraint name prefix.
    """
    model.addConstr(
        var >= prev_expr + delta_expr - big_m * x_m - big_m * x_r,
        name=f"{name}_accum",
    )
    model.addConstr(
        var >= prev_expr - rho * (prev_expr - anchor_prev_expr) - big_m * (1 - x_m),
        name=f"{name}_repair",
    )
