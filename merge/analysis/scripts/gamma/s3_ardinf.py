"""S3: ARD-infinity repair in parameter space versus the pathwise law.

One rate b (lore: b = 1/beta; main: one increment type with b = b*, where the calibration is exact).
Schedule on a new part (state 0): day 1 mission of shape A (A = A_hat, the cap); day 2 repair;
day 3 mission of shape rho*A. Tracked shape: A, (1-rho)A, A: every row A <= A_hat holds.
True state on day 3: (1-rho)X + Z, X ~ Gamma(A, b), Z ~ Gamma(rho*A, b) independent,
i.e. Gamma(A, b/(1-rho)) + Gamma(rho*A, b).
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_lib import sum_tail_fast, mc_tail, tail, shape_for_tail  # noqa: E402

tau = 1.0


def day3_true(A, b, rho):
    return sum_tail_fast([(A, b / (1.0 - rho)), (rho * A, b)], tau)


# 1) the worked counterexample
eps, rho, b = 0.01, 0.5, 0.01
A = shape_for_tail(eps, b, tau)
lo, up = day3_true(A, b, rho)
print(f"example: eps={eps} rho={rho} rate b={b} (lore beta={1/b:g}), tau={tau}")
print(f"  A_hat={A:.6g}; mean at cap A_hat/b={A/b:.4f} (<= tau, so lore's alpha<=tau/beta holds)")
print(f"  tracked shapes: day1 {A:.6g}, day2 {(1-rho)*A:.6g}, day3 {(1-rho)*A+rho*A:.6g}; rows hold")
print(f"  true P(D_1>tau)={tail(A, b, tau):.6f}, P(D_2>tau)={tail(A, b/(1-rho), tau):.6f}, "
      f"P(D_3>tau) in [{lo:.6f}, {up:.6f}] = {lo/eps:.3f} eps")


def sampler(rng):
    return (1 - rho) * rng.gammavariate(A, 1 / b) + rng.gammavariate(rho * A, 1 / b)


p, se = mc_tail(sampler, tau, 1000000, seed=3)
print(f"  Monte Carlo (1e6 draws): {p:.5f} +- {1.96*se:.5f}")

# 2) where does it happen? ratio true/eps on day 3 over a grid of b*tau (shape scale) and rho
print("ratio P(D_3>tau)/eps on the grid (b*tau; A_hat; mean at cap/tau):")
for eps in [0.01, 0.1]:
    for btau in [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]:
        b = btau / tau
        A = shape_for_tail(eps, b, tau)
        row = []
        for rho in [0.1, 0.3, 0.5, 0.8, 0.95]:
            lo, up = day3_true(A, b, rho)
            row.append(f"{lo/eps:.3f}")
        print(f"  eps={eps:<5} b*tau={btau:<6} A_hat={A:<10.4g} mean/tau={A/btau:<7.3f} "
              f"rho=0.1,0.3,0.5,0.8,0.95: {' '.join(row)}")

# 3) more increments after the repair, same total shape rho*A (law unchanged), or a longer
#    tail-end: add k missions of shape rho*A/k: identical law, so the grid above covers it.
# 4) immediately after the repair: the true tail never exceeds the checked pre-repair tail
worst = 0.0
for btau in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    for rho in [0.1, 0.5, 0.9]:
        A = shape_for_tail(0.01, btau, 1.0)
        worst = max(worst, tail(A, btau / (1 - rho), 1.0) - tail(A, btau, 1.0))
print(f"day 2 (just repaired): max of true tail - pre-repair tail = {worst:.2e} (<= 0 expected)")
