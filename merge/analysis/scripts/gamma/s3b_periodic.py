"""S3, second part: periodic schedule [repair, mission of shape rho*A] after a first mission of shape A = A_hat.
Tracked peaks all equal A_hat (rows hold with equality). True peak after n cycles:
(1-rho)^n X + sum_{j=0}^{n-1} (1-rho)^j Z_j, X ~ Gamma(A, b), Z_j ~ Gamma(rho A, b).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_lib import sum_tail, shape_for_tail  # noqa: E402

tau = 1.0
print("ratio of the true peak tail to eps after n = 1..4 cycles")
worst_regular = 0.0
for eps in [0.01, 0.1]:
    for btau in [0.01, 0.1, 1.0, 3.0, 10.0, 30.0]:
        b = btau / tau
        A = shape_for_tail(eps, b, tau)
        for rho in [0.3, 0.5, 0.8]:
            row = []
            for n in range(1, 5 if rho <= 0.5 else 3):
                comps = [(A, b / (1 - rho) ** n)] + [(rho * A, b / (1 - rho) ** j) for j in range(n)]
                lo, up = sum_tail(comps, tau, cut=6000)
                row.append(lo / eps)
                if btau >= 1.0:
                    worst_regular = max(worst_regular, up / eps)
            print(f"  eps={eps:<5} b*tau={btau:<5} A_hat={A:<9.4g} rho={rho}: " + " ".join(f"{r:.3f}" for r in row))
print(f"largest ratio (upper end of the series) for b*tau >= 1: {worst_regular:.3f}")
