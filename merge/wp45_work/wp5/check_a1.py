"""Step A1 check: FleetModel.increment on the T2 case of the plan (section 4.4)."""
import numpy as np

from load_base import load

base = load()
mu = np.zeros((2, 1, 1, 3))
mu[0, 0, 0, :] = [1.0, 2.0, 3.0]
mu_tr = np.zeros((2, 1, 1, 2))
mu_tr[0, 0, 0, :] = [10.0, 20.0]
names = ["model", "x", "m_rep", "r_rep", "mu_var", "z_var", "u_var", "model_of", "repair_of", "bound_of",
         "impl_of", "encoding_of", "tau", "eps", "rho", "tangent_ref", "initial", "replacement",
         "anchor_start", "state_range", "opts"]
kw = {n: None for n in names}
for transitory, expected in ((None, [1, 2, 1, 2, 3]), (mu_tr, [10, 20, 1, 2, 3])):
    ctx = base.FleetModel(F=2, H1=2, H2=3, M=1, L=1, T=5, allow_replacement=True, pwl_points=8,
                          profiles={"mu": (mu, transitory)}, **kw)
    got = [ctx.increment("mu", 0, 1, 0, k) for k in range(5)]
    print("transitory" if transitory is not None else "no transitory", got, "OK" if got == expected else "FAIL")
    assert ctx.increment("mu", 0, 0, 0, 3) == 0.0
try:
    ctx.increment("v", 0, 1, 0, 0)
except KeyError as error:
    print("KeyError:", error)
print("fields:", [f for f in base.FleetModel.__dataclass_fields__])
print("state_vars default separate:", ctx.state_vars is not base.FleetModel(F=2, H1=2, H2=3, M=1, L=1, T=5, allow_replacement=True, pwl_points=8, profiles={}, **kw).state_vars)
