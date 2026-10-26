# Reviewer's independent spot check of Theorem "One term is the worst case" (standard library only):
# two-term members S = delta + c1*Y1 + c2*Y2 of the class C(m) never exceed W(m,t) = max_c Q(m/c, t/c).
import math, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gq import Q

def W(m, t, n=1500):
    best = 0.0
    for i in range(n):
        c = math.exp(math.log(1e-6) * (1 - i / (n - 1)))
        best = max(best, Q(m / c, t / c))
    return best

def tail2(delta, m1, c1, m2, c2, t, n=4000):
    # P(delta + c1*Y1 + c2*Y2 > t), Y_i ~ Gamma(m_i/c_i, 1)
    tp = t - delta
    if tp <= 0:
        return 1.0
    a1, a2 = m1 / c1, m2 / c2
    Y = tp / c1
    if a1 >= 1:
        lg1 = math.lgamma(a1)
        h = Y / n
        s = 0.0
        for k in range(n + 1):
            y = k * h
            fy = math.exp((a1 - 1) * math.log(y) - y - lg1) if y > 0 else (math.exp(-lg1) if a1 == 1 else 0.0)
            g = fy * Q(a2, max(tp - c1 * y, 0.0) / c2)
            wgt = 1 if k in (0, n) else (4 if k % 2 else 2)
            s += wgt * g
        return Q(a1, Y) + s * h / 3
    U = Y ** a1
    # Simpson in u, y = u^(1/a1), weight e^{-y}/Gamma(a1+1)
    lg = math.lgamma(a1 + 1)
    h = U / n
    s = 0.0
    for k in range(n + 1):
        u = k * h
        y = u ** (1 / a1) if u > 0 else 0.0
        g = math.exp(-y - lg) * Q(a2, max(tp - c1 * y, 0.0) / c2)
        wgt = 1 if k in (0, n) else (4 if k % 2 else 2)
        s += wgt * g
    integral = s * h / 3
    return Q(a1, Y) + integral

random.seed(7)
worst = -1
cases = 0
for t in [0.05, 0.2, 0.448, 1.0, 3.0]:
    for eps in [0.01, 0.1]:
        # mean near the sharp cap scale: use m = r*t with r between 0.02 and 0.6
        for _ in range(25):
            m = t * random.uniform(0.02, 0.6)
            Wm = W(m, t)
            fr = random.random()
            fd = random.random() * 0.3
            delta = fd * m
            m1 = (m - delta) * fr
            m2 = (m - delta) - m1
            c1 = math.exp(random.uniform(math.log(1e-3), 0))
            c2 = math.exp(random.uniform(math.log(1e-3), 0))
            p = tail2(delta, m1, c1, m2, c2, t)
            cases += 1
            ratio = p / Wm
            if ratio > worst:
                worst = ratio
                arg = (t, m, delta, m1, c1, m2, c2, p, Wm)
print("random two-term members:", cases, " largest P(S>t)/W(m,t) = %.6f" % worst)
print("  at t=%.3g m=%.4g delta=%.4g (m1,c1)=(%.4g,%.4g) (m2,c2)=(%.4g,%.4g) P=%.6g W=%.6g" % arg)

# targeted search near the one-term optimum: split the optimal term into two scales close to c*
for t, m in [(0.1, 0.0035), (0.3, 0.0106), (1.0, 0.0437), (0.448, 0.0158)]:
    n = 1500
    cs = [math.exp(math.log(1e-6) * (1 - i / (n - 1))) for i in range(n)]
    cstar = max(cs, key=lambda c: Q(m / c, t / c))
    Wm = Q(m / cstar, t / cstar)
    best = 0
    for f1 in [0.8, 1.0, 1.25]:
        for f2 in [0.5, 0.8, 1.25, 2.0]:
            for fr in [0.2, 0.5, 0.8]:
                c1 = min(1.0, cstar * f1); c2 = min(1.0, cstar * f2)
                p = tail2(0.0, m * fr, c1, m * (1 - fr), c2, t)
                best = max(best, p)
    print("  t=%.3g m=%.4g c*=%.4g W=%.6g best two-term near c*: %.6g (ratio %.6f)" % (t, m, cstar, Wm, best, best / Wm))
