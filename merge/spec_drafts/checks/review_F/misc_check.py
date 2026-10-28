# Reviewer's checks of small numerical claims (standard library only).
import math
from statistics import NormalDist
N = NormalDist()
print("V_max/tau^2 at eps 0.01 0.1 0.3 0.49:", [round(1 / N.inv_cdf(1 - e) ** 2, 4) for e in (0.01, 0.1, 0.3, 0.49)])
# Cantelli pwl example: eps 0.1, kappa_C^2 = 9, tau 1, v 0.004
f = lambda mu: (1 - mu) ** 2 / 9
fp = lambda mu: -2 * (1 - mu) / 9
v = 0.004
def ok_pwl(mu, P):
    p = min(int(mu * P), P - 1)
    mh = (p + 0.5) / P
    return v <= f(mh) + fp(mh) * (mu - mh)
lo, hi = 0.0, 1.0
best = max(i / 1e6 for i in range(0, 1000001) if ok_pwl(i / 1e6, 8))
print("exact %.4f  tangent0 %.4f  tangent0.5 %.4f  pwl8 %.4f" % (1 - math.sqrt(9 * v), (1 - 9 * v) / 2, 0.75 - 9 * v, best))
# Chernoff example
K = math.log(0.5 + 0.5 * math.exp(2.5))
print("K_inc %.4f; row with K0=3: %.4f vs ln0.1 %.4f; with K0=0: %.4f" % (K, 3 + K - 5, math.log(0.1), K - 5))
# loop remark example (ARD1, rho 0.5)
mu, g = 0.4, 0.1
peaks = []
for rep in range(4):
    mu += 0.1; peaks.append(round(mu, 4)); mu = 0.5 * mu + 0.5 * g; g = mu; mu += 0.05
    print("rep", rep + 1, "end (mu,g)=(%.4f,%.4f)" % (mu, g), "peak", peaks[-1])
# scalar bound example
U = lambda m, v: v / (v + (10 - m) ** 2)
m, vv = 4.0, 9.0
print("U start %.4f" % U(m, vv))
for rep in range(3):
    m = 0.5 * m + 3; vv = 0.25 * vv + 3.99
    print("U end rep %d: %.4f" % (rep + 1, U(m, vv)))
print("U limit %.4f" % U(6.0, 3.99 / 0.75))
# Bernstein tangent emptiness: tau=1, c=0.1 -> sqrt(tau(tau-2c)) vs tau - c
tau, c = 1.0, 0.1
print("Bernstein tangent point bound: sqrt(tau(tau-2c)) = %.4f, check tau - c = %.4f" % (math.sqrt(tau * (tau - 2 * c)), tau - c))
kap = 2.0  # any kappa
for mur in (0.5, 0.85, 0.89, 0.895, 0.899):
    dr = tau - mur
    fB = lambda mu: ((tau - mu) ** 2 - 2 * c * (tau - mu)) / (2 * kap)
    fBp = lambda mu: (-(2 * (tau - mu)) + 2 * c) / (2 * kap)
    T0 = fB(mur) + fBp(mur) * (0 - mur)
    print("  mu_r=%.3f: tangent value at mu=0: %.5f (>=0 needed for any feasible point)" % (mur, T0))
