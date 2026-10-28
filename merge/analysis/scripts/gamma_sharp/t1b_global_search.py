"""T1b: global numerical search for the worst member of C(m) at the cap m = A_hat(t).

Members: S = delta + sum_{i<=K} c_i Gamma(m_i / c_i, 1), sum m_i + delta = m, c_i in [cmin, 1].
Nelder-Mead with random restarts on (c_i, weights incl. drift). Output: best tail / eps.
A ratio > 1 is a witness that the single Gamma is not the worst case (exact tail, lower end).
Usage: python -I t1b_global_search.py [eps ...]
"""
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sharp_lib import A_hat, member_tail, nelder_mead, gammaQ  # noqa: E402


def decode(x, K, cmin):
    cs = []
    for i in range(K):
        xi = x[i]
        cs.append(1.0 if xi > 30 else cmin + (1.0 - cmin) / (1.0 + math.exp(-xi)))
    ys = x[K:]
    mx = max(ys)
    es = [math.exp(y - mx) for y in ys]
    s = sum(es)
    ws = [e / s for e in es]          # ws[0] = drift share, ws[1:] = term shares
    return cs, ws


def tail_of(x, K, cmin, m, t):
    cs, ws = decode(x, K, cmin)
    terms = [(c, w * m) for c, w in zip(cs, ws[1:]) if w * m > 1e-300]
    lo, _ = member_tail(terms, ws[0] * m, t)
    return lo


def search(t, m, K, cmin, restarts, rng, iters=250):
    best = (-1.0, None)
    for r in range(restarts):
        x0 = [rng.uniform(-3, 4) for _ in range(K)] + [rng.uniform(-6, 0)] + \
             [rng.uniform(-1, 1) for _ in range(K)]
        if r == 0:   # start near one term at c* = t / 0.4348 (small-shape optimum), no drift
            cstar = min(0.999, max(cmin + 1e-6, t / 0.4348))
            z = (cstar - cmin) / (1 - cmin)
            x0 = [math.log(z / (1 - z))] + [rng.uniform(-3, 4) for _ in range(K - 1)] + [-12.0] + \
                 [2.0] + [-2.0] * (K - 1)
        f = lambda x: -tail_of(x, K, cmin, m, t)
        x, v = nelder_mead(f, x0, step=0.8, iters=iters)
        if -v > best[0]:
            best = (-v, x)
    p, x = best
    cs, ws = decode(x, K, cmin)
    return p, cs, ws


def main():
    eps_list = [float(a) for a in sys.argv[1:]] or [0.001, 0.01, 0.1]
    rng = random.Random(11)
    ts = [0.01, 0.1, 0.3, 0.4, 0.43, 0.45, 0.47, 0.5, 0.6, 0.62, 0.65, 0.7, 1.0, 3.0, 10.0, 100.0]
    for eps in eps_list:
        for t in ts:
            m = A_hat(t, eps)
            base = gammaQ(m, t)
            line = f"eps={eps:<6} t={t:<6} m={m:<10.4g} single={base / eps:.6f}"
            t0 = time.time()
            for K, cmin, rs in [(1, 0.002, 6), (2, 0.002 if t < 10 else 0.05, 8),
                                (3, 0.05 if t < 10 else 0.2, 4)]:
                p, cs, ws = search(t, m, K, cmin, rs, rng)
                desc = " ".join(f"(c={c:.3f},w={w:.3f})" for c, w in zip(cs, ws[1:]))
                line += f" | K={K}: {p / eps:.6f} drift={ws[0]:.3f} {desc}"
            print(line + f"  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
