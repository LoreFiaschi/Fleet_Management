# Reviewer's spot check of Theorem "One term is the worst case", with an exact series for the
# tail of a sum of independent Gamma variables of different scales (Moschopoulos 1985 mixture series),
# standard library only. Members: S = delta + sum_i c_i Y_i, Y_i ~ Gamma(a_i, 1), mean m.
import math, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gq import Q

def tail_sum(delta, terms, t, kmax=4000, tol=1e-15):
    # terms: list of (shape a_i, scale c_i); P(delta + sum c_i Y_i > t)
    tp = t - delta
    if tp <= 0:
        return 1.0
    terms = [(a, c) for a, c in terms if a > 0]
    b1 = min(c for a, c in terms)
    rho = sum(a for a, c in terms)
    logC = sum(a * math.log(b1 / c) for a, c in terms)
    # gamma_k = sum_j a_j (1 - b1/c_j)^k / k
    qs = [(a, 1 - b1 / c) for a, c in terms]
    gam = [0.0]
    deltas = [1.0]
    total = deltas[0] * Q(rho, tp / b1)
    acc = deltas[0]
    for k in range(1, kmax):
        gam.append(sum(a * (q ** k) for a, q in qs) / k)
        dk = sum(i * gam[i] * deltas[k - i] for i in range(1, k + 1)) / k
        deltas.append(dk)
        acc += dk
        term = dk * Q(rho + k, tp / b1)
        total += term
        if dk * math.exp(logC) < tol and k > 5:
            break
    return math.exp(logC) * total, math.exp(logC) * acc  # second value should tend to 1

def W(m, t, n=3000):
    best, bc = 0.0, None
    for i in range(n):
        c = math.exp(math.log(1e-7) * (1 - i / (n - 1)))
        v = Q(m / c, t / c)
        if v > best:
            best, bc = v, c
    # golden refine
    lo, hi = bc / 1.01, min(1.0, bc * 1.01)
    for _ in range(100):
        a = lo + (hi - lo) / 3; b = hi - (hi - lo) / 3
        if Q(m / a, t / a) < Q(m / b, t / b):
            lo = a
        else:
            hi = b
    c = 0.5 * (lo + hi)
    return max(best, Q(m / c, t / c)), c

# sanity: equal scales reproduce one term
p, s = tail_sum(0.0, [(0.01, 0.2), (0.02, 0.2)], 0.1)
print("sanity equal scales: series %.10f  one-term %.10f  mass %.12f" % (p, Q(0.03, 0.5), s))
p, s = tail_sum(0.0, [(0.5, 0.3), (0.7, 0.6)], 1.0)
print("sanity mass of mixture weights (should be 1): %.12f" % s)

random.seed(11)
worst = 0
for t in [0.05, 0.1, 0.2, 0.448, 1.0, 3.0]:
    for _ in range(60):
        m = t * random.uniform(0.01, 0.6)
        Wm, cstar = W(m, t)
        nt = random.choice([2, 2, 3])
        fd = random.random() * 0.3 if random.random() < 0.5 else 0.0
        delta = fd * m
        fr = [random.random() for _ in range(nt)]
        sfr = sum(fr)
        # scales near c* half of the time, else anywhere
        if random.random() < 0.5:
            cs = [min(1.0, cstar * math.exp(random.uniform(-0.7, 0.7))) for _ in range(nt)]
        else:
            cs = [math.exp(random.uniform(math.log(1e-3), 0)) for _ in range(nt)]
        terms = [((m - delta) * f / sfr / c, c) for f, c in zip(fr, cs)]
        p, mass = tail_sum(delta, terms, t)
        r = p / Wm
        if r > worst:
            worst = r
            arg = (t, m, delta, terms, p, Wm, mass)
print("random members (2 or 3 terms, optional constant): largest P(S>t)/W(m,t) = %.8f" % worst)
print("  at t=%g m=%.5g delta=%.4g terms=%s P=%.8g W=%.8g mass=%.10f" % arg)

print("targeted: optimal one term split into two scales around c*")
for t, m in [(0.1, 0.0035), (0.3, 0.0106), (1.0, 0.0437), (0.448, 0.0158), (3.0, 0.8)]:
    Wm, cstar = W(m, t)
    best = 0
    for f1 in [0.7, 0.9, 1.0]:
        for f2 in [0.5, 0.8, 1.0, 1.1, 1.3]:
            for fr in [0.1, 0.5, 0.9]:
                c1 = min(1.0, cstar * f1); c2 = min(1.0, cstar * f2)
                p, mass = tail_sum(0.0, [(m * fr / c1, c1), (m * (1 - fr) / c2, c2)], t)
                best = max(best, p)
    print("  t=%g m=%g c*=%.4f W=%.8g best two-term=%.8g ratio=%.8f" % (t, m, cstar, Wm, best, best / Wm))
