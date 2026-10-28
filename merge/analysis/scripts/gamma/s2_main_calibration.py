"""S2: can main's repeated_increment surrogate accept an unsafe schedule?

Schedule = n1 missions of type 1 and n2 of type 2 on one component, no repair, new part
(mu_0 = 0), n1 + n2 <= T. Surrogate accepts iff n1*a1' + n2*a2' <= A_max.
True tail = P(Gamma(n1*a1, b1) + Gamma(n2*a2, b2) > tau), exact series.
"""
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_lib import sum_tail, sum_tail_fast, mc_tail, tail  # noqa: E402
from main_calibration import calibrate_cell  # noqa: E402

tau, T = 1.0, 20

# consistency of the fast series with the plain one
for comps in [[(3.0, 10.0), (40.0, 100.0)], [(0.7, 2.0), (5.0, 9.0)]]:
    a, b = sum_tail(comps, tau), sum_tail_fast(comps, tau)
    assert abs(a[0] - b[0]) < 1e-12 and abs(a[1] - b[1]) < 1e-12, (a, b)


def worst_pair(m1, b1, m2, b2, eps):
    (a1p, a2p), (s1, s2), bstar, Amax = calibrate_cell([m1, m2], [b1, b2], tau, eps, T)
    a1, a2 = m1 * b1, m2 * b2
    worst = None
    for n1 in range(T + 1):
        # largest accepted n2 for this n1 (the true tail increases in n2)
        n2 = -1
        for k in range(T - n1, -1, -1):
            if n1 * a1p + k * a2p <= Amax:
                n2 = k
                break
        if n2 < 0 or (n1 == 0 and n2 == 0):
            continue
        lo, up = sum_tail_fast([(n1 * a1, b1), (n2 * a2, b2)], tau)
        if worst is None or lo > worst[0]:
            worst = (lo, up, n1, n2)
    return worst, (a1p, a2p, s1, s2, bstar, Amax)


grid_bstar = [2.0, 10.0, 50.0]
grid_ratio = [2.0, 5.0, 20.0]
grid_mean = [0.02, 0.05, 0.1, 0.2, 0.4]
results = []
for eps in [0.01, 0.1]:
    for bstar, r, m1, m2 in itertools.product(grid_bstar, grid_ratio, grid_mean, grid_mean):
        w, cal = worst_pair(m1, bstar, m2, r * bstar, eps)
        if w is None:
            continue
        results.append((w[0] / eps, eps, bstar, r, m1, m2, w, cal))
n_bad = sum(1 for x in results if x[6][0] > x[1])
print(f"searched {len(results)} parameter sets (eps in 0.01, 0.1; b* in {grid_bstar}; "
      f"b2/b1 in {grid_ratio}; means in {grid_mean}; T={T})")
print(f"parameter sets with an accepted schedule whose true tail exceeds eps: {n_bad}")
results.sort(key=lambda x: -x[0])
for ratio, eps, bstar, r, m1, m2, w, cal in results[:6]:
    a1p, a2p, s1, s2, bs, Amax = cal
    print(f"  eps={eps} b1={bstar} b2={r*bstar} m1={m1} m2={m2}: a1'={a1p:.4f} (a1={m1*bstar:.4f}) "
          f"a2'={a2p:.4f} (a2={m2*r*bstar:.4f}) m*=({s1},{s2}) A_max={Amax:.4f} "
          f"worst (n1,n2)=({w[2]},{w[3]}) A'={w[2]*a1p+w[3]*a2p:.4f} true tail in [{w[0]:.5f},{w[1]:.5f}] "
          f"= {ratio:.2f} eps")

# single-type overshoot: n = m* + 1 accepted?
print("single type, user-chosen common rate b* < b (modular route needs gamma_beta_bound):")
over = 0
for eps in [0.01, 0.1]:
    for b, frac, m in itertools.product([10.0, 50.0, 200.0], [0.2, 0.5, 0.9], grid_mean):
        bstar = frac * b
        (ap,), (s,), _, Amax = calibrate_cell([m], [b], tau, eps, T, common_rate=bstar)
        if s < T and (s + 1) * ap <= Amax:
            over += 1
            print(f"  eps={eps} b={b} b*={bstar} mean={m}: m*={s}, a'={ap:.4f}, (m*+1)a'={(s+1)*ap:.4f} "
                  f"<= A_max={Amax:.4f}; true tail at m*+1 = {tail((s+1)*m*b, b, tau):.4f}")
print(f"single-type cases with m*+1 repetitions accepted: {over}")
