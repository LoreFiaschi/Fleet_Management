"""S2, second part: mixing two types while each count stays within its certified m*.

Also prints the two worked examples used in the notes, with Monte Carlo cross-checks.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_lib import sum_tail_fast, mc_tail, tail  # noqa: E402
from main_calibration import calibrate_cell  # noqa: E402

tau, T = 1.0, 20
grid_bstar = [2.0, 10.0, 50.0]
grid_ratio = [2.0, 5.0, 20.0]
grid_mean = [0.02, 0.05, 0.1, 0.2, 0.4]
count, bad, best = 0, 0, None
for eps in [0.01, 0.1]:
    for bstar, r, m1, m2 in itertools.product(grid_bstar, grid_ratio, grid_mean, grid_mean):
        b1, b2 = bstar, r * bstar
        (a1p, a2p), (s1, s2), _, Amax = calibrate_cell([m1, m2], [b1, b2], tau, eps, T)
        count += 1
        found = None
        for n1 in range(1, s1 + 1):
            for n2 in range(s2, 0, -1):
                if n1 + n2 <= T and n1 * a1p + n2 * a2p <= Amax:
                    lo, up = sum_tail_fast([(n1 * m1 * b1, b1), (n2 * m2 * b2, b2)], tau)
                    if found is None or lo > found[0]:
                        found = (lo, up, n1, n2)
                    break
        if found and found[0] > eps:
            bad += 1
            key = found[0] / eps
            if best is None or key > best[0]:
                best = (key, eps, b1, b2, m1, m2, a1p, a2p, s1, s2, Amax, found)
print(f"searched {count} parameter sets, counts limited to n1 <= m1*, n2 <= m2*, n1 + n2 <= {T}")
print(f"sets with an accepted unsafe mix: {bad}")
if best:
    key, eps, b1, b2, m1, m2, a1p, a2p, s1, s2, Amax, f = best
    print(f"  worst: eps={eps} type1 (mean {m1}, rate {b1}) type2 (mean {m2}, rate {b2}): "
          f"a'=({a1p:.4f},{a2p:.4f}) m*=({s1},{s2}) A_max={Amax:.4f} (n1,n2)=({f[2]},{f[3]}) "
          f"A'={f[2]*a1p+f[3]*a2p:.4f} true tail {f[0]:.5f} = {key:.2f} eps")

# worked example A (mix within m*): eps 0.01, type 1 (mean 0.1, rate 10), type 2 (mean 0.1, rate 200)
eps = 0.01
(a1p, a2p), (s1, s2), bstar, Amax = calibrate_cell([0.1, 0.1], [10.0, 200.0], tau, eps, T)
print(f"example A: b*={bstar} a1'={a1p:.6f} a2'={a2p:.6f} m*=({s1},{s2}) A_max={Amax:.6f}")
for n1, n2 in [(3, 0), (0, 8), (3, 1), (3, 2), (2, 4), (1, 7)]:
    Ap = n1 * a1p + n2 * a2p
    lo, up = sum_tail_fast([(n1 * 1.0, 10.0), (n2 * 20.0, 200.0)], tau)
    print(f"  (n1,n2)=({n1},{n2}): A'={Ap:.4f} accepted={Ap <= Amax} true tail [{lo:.6f},{up:.6f}]")
n1, n2 = 3, 2


def samp(rng):
    return sum(rng.gammavariate(1.0, 0.1) for _ in range(n1)) + sum(rng.gammavariate(20.0, 1 / 200) for _ in range(n2))


p, se = mc_tail(samp, tau, 500000, seed=5)
print(f"  MC for (3,2): {p:.5f} +- {1.96*se:.5f}")

# worked example B (one type repeated beyond m*): type 2 (mean 0.4, rate 50) next to type 1 (mean 0.1, rate 10)
(a1p, a2p), (s1, s2), bstar, Amax = calibrate_cell([0.1, 0.4], [10.0, 50.0], tau, eps, T)
print(f"example B: b*={bstar} a1'={a1p:.6f} a2'={a2p:.6f} (true shape 20) m*=({s1},{s2}) A_max={Amax:.6f}")
for n2 in [1, 2, 3, 5, 10, 20]:
    print(f"  n2={n2}: A'={n2*a2p:.4f} accepted={n2*a2p <= Amax} true mean {0.4*n2:.1f} "
          f"true tail {tail(n2*20.0, 50.0, tau):.6f}")
