"""Above t1: is max over c in (0, 1) of Q(A_hat/c, t/c) at most Q(A_hat, t) = eps? Grid of t in [t1, 1e5] (R2 review)."""
import numpy as np
from scipy.special import gammaincc as Q
from scipy.optimize import brentq
t1 = {0.001: 0.436143, 0.01: 0.44839, 0.1: 0.614223}
for eps in (0.001, 0.01, 0.1):
    worst = 0.0; wt = None
    for t in np.logspace(np.log10(t1[eps]), 5, 200):
        Ah = brentq(lambda A: Q(A, t) - eps, 1e-300, 10*t + 100, xtol=1e-300, rtol=1e-14)
        c = np.logspace(-8, 0, 20001)[:-1]
        r = np.max(Q(Ah/c, t/c))/eps
        if r > worst: worst, wt = r, t
    print(f'eps={eps}: largest max_(c<1) Q(A_hat/c,t/c)/eps over t in [t1, 1e5] = {worst:.9f} at t={wt:.4g}')
