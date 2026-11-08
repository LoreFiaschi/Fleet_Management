"""Exact day maps of the hand-worked test case of the WP5 plan (section 4)."""
import math

from scipy.stats import norm

rho, tau = 0.5, 1.0
# train 0: mission, repair, mission, replacement; increments mu 0.3, v 0.01; mu_0 0.2, v_0 0
plan = ["fly", "repair", "fly", "replace"]


def run(law, wiener=False):
    mu, v = 0.2, 0.0
    g_mu, g_v = 0.2, 0.0
    out = []
    for day in plan:
        mu_prev, g_prev = mu, g_mu
        if day == "fly":
            mu += 0.3
            v += 0.01
            z = 0.0
        elif day == "repair":
            a_mu, a_v = 1 - rho, (1 - rho) ** 2
            if law == "ard1":
                mu = a_mu * mu + (1 - a_mu) * g_mu
                if not wiener:
                    v = a_v * v + (1 - a_v) * g_v
                z = rho * (mu_prev - g_prev)
                g_mu, g_v = mu, v
            else:
                mu = a_mu * mu
                if not wiener:
                    v = a_v * v
                z = rho * mu_prev
        else:
            mu, v = 0.1, 0.0
            z = 0.0
            if law == "ard1":
                g_mu, g_v = mu, v
        out.append((round(mu, 6), round(v, 6), round(g_mu, 6), round(g_v, 6), round(z, 6)))
    return out


for law in ("ard1", "ardinf"):
    print(law, "gaussian-like", run(law))
    print(law, "wiener-like", run(law, wiener=True))

kappa2 = norm.ppf(0.9) ** 2
f = lambda m: (tau - m) ** 2 / kappa2
fp = lambda m: -2 * (tau - m) / kappa2
v_a = 0.99 * f(0.375)
print("kappa_G^2", kappa2, "V_max", tau ** 2 / kappa2)
print("point A v", v_a, "tangent 0.5 cap", f(0.5) + fp(0.5) * (0.375 - 0.5), "pwl P=4 cap", f(0.375))
kappa = math.log(10)
b = 0.3
c = kappa * b / 3
c2, c1 = 1 / (2 * kappa), -b / 3
fb = lambda m: c2 * (tau - m) ** 2 + c1 * (tau - m)
print("bernstein c", c, "side", tau - c, "f(0.3)", fb(0.3),
      "spec form", ((tau - c - 0.3) ** 2 - c ** 2) / (2 * kappa), "range", tau * (tau - 2 * c) / (2 * kappa))
# reliability on the hand trajectories, tangent at 0.5
for law in ("ard1", "ardinf"):
    for mu, v, *_ in run(law):
        assert v <= f(0.5) + fp(0.5) * (mu - 0.5) + 1e-12, (law, mu, v)
    for mu, v, *_ in run(law, wiener=True):
        assert v <= f(0.5) + fp(0.5) * (mu - 0.5) + 1e-12, (law, mu, v)
print("tangent rows hold on all hand trajectories")
