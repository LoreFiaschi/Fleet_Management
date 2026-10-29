import hand_rows as h
import numpy as np
from scipy.stats import norm
kappa2 = norm.ppf(0.9) ** 2
cfg = h.make_cfg(H=[1, 1], H1=1, H2=1, T=2, reliability_impl="tangent", tangent_ref=np.full((2, 1), 0.375), pwl_points=8)
ctx = h.build(cfg, h.RowOnlyBuilder(1 / kappa2, 0.0, 0.7697415))
for key, var in ctx.x.items():
    var.LB = var.UB = 1.0 if key in ((0, 1, 0), (0, 1, 1)) else 0.0
for var in list(ctx.m_rep.values()) + list(ctx.r_rep.values()):
    var.LB = var.UB = 0.0
ctx.mu_var[0,0,0].LB = ctx.mu_var[0,0,0].UB = 0.7
v = ctx.state_vars["v"][0,0,0]; v.LB = v.UB = 0.0
ctx.model.setObjective(h.base.gp.LinExpr(), h.base.GRB.MINIMIZE)
ctx.model.optimize()
print(ctx.model.Status, ctx.model.SolCount)
# rows involving mu_var[0,0,0]
for tc, name in ctx.model.rows:
    if name.startswith("rel") :
        print(name, {vv.VarName: c for vv, c in tc.expr.terms.items()}, tc.expr.const, tc.sense)
