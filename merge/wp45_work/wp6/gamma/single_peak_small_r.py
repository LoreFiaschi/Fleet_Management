"""Single peak of s -> Q(r s, s) for small r (below the range 1e-5 .. 0.999 checked in the scoping).

Same method as merge/analysis/scripts/verify_gamma_sharp/single_peak_check.py: sign changes of the
slope on a log grid of s in [1e-12, 1e8]; also the cap A_1 at small epsilon against brute force.
"""
import numpy as np
from scipy.special import gammaincc

import cap_proto as cp

bad = 0
for r in np.logspace(-10, -5, 200):
    s = np.logspace(-12, 8, 4000)
    v = gammaincc(r * s, s)
    d = np.diff(v)
    sg = np.sign(d[np.abs(d) > 1e-15 * np.max(v)])
    if int(np.sum(sg[1:] != sg[:-1])) > 1:
        bad += 1
print(f"r in [1e-10, 1e-5], 200 values: {bad} with more than one sign change of the slope")
for eps in (1e-6, 1e-8):
    for t in (0.01, 1.0, 30.0):
        a, b = cp.sharp_cap(t, eps), cp.ref_a1(t, eps)
        print(f"eps={eps:g} t={t}: A_1={a:.10g} brute force {b:.10g} rel diff {abs(a - b) / b:.1e}")
