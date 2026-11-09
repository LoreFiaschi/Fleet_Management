"""Largest mean damage mu with Pr(D > tau) <= eps, as a fraction of tau, for the tyre calibrations.

IG (library row): mu <= tau / qbar, qbar = E[Z | Z > q], Z ~ IG(1, 1/eta), Pr(Z > q) = eps.
Gamma (exact tail, independent increments): D ~ Gamma(shape beta*mu, rate beta), beta = 1/(c^2 tau).
Gaussian: D ~ N(mu, c^2 tau mu).
"""
import sys
import numpy as np
from scipy import stats, integrate, optimize

sys.stdout.reconfigure(encoding="utf-8")
tau = 18.4
for c in (0.19, 0.30):
    eta = c * c
    lam = 1.0 / eta
    Z = stats.invgauss(mu=1.0 / lam, scale=lam)  # scipy: IG(mean=mu*scale, shape=scale)
    beta = 1.0 / (c * c * tau)
    for eps in (0.01, 0.05):
        q = Z.isf(eps)
        tail, _ = integrate.quad(lambda z: z * Z.pdf(z), q, np.inf)
        qbar = tail / eps
        f_ig = 1.0 / qbar
        g = lambda m: stats.gamma(a=beta * m, scale=1.0 / beta).sf(tau) - eps
        m_ga = optimize.brentq(g, 1e-6, tau)
        h = lambda m: stats.norm(loc=m, scale=np.sqrt(c * c * tau * m)).sf(tau) - eps
        m_n = optimize.brentq(h, 1e-6, tau)
        print(f"c={c:.2f} eps={eps:.2f}: mean of Z={Z.mean():.3f} sd={Z.std():.3f} | IG cap {f_ig:.3f} tau | "
              f"Gamma exact {m_ga / tau:.3f} tau | Gaussian {m_n / tau:.3f} tau")
