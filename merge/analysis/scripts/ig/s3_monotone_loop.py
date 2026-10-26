"""S3: monotonicity in mu, the spec's density and MLR derivative, and FSD of the true laws in the loop."""
import math, sys
sys.path.insert(0, sys.argv[1])
from iglib import *

TAU = 1.0
print("== A. tail increasing in mu on a grid ==")
for eta in (0.05, 1.0, 10.0):
    vals = [ig_sf(TAU, m, m / eta) for m in [0.01 * i for i in range(1, 100)]]
    print(f"fixed eta={eta}: increasing = {all(b > a for a, b in zip(vals, vals[1:]) if a > 1e-300)}")
for xi in (0.2, 20.0, 2000.0):
    vals = [ig_sf(TAU, m, xi * m * m) for m in [0.01 * i for i in range(1, 100)]]
    print(f"fixed xi={xi}: increasing = {all(b >= a for a, b in zip(vals, vals[1:]))}")

print("\n== B. spec density (lore:spec/spec.tex:906-907) vs the IG density with lam = mu/eta ==")
eta = 0.5
spec = lambda x, m: math.sqrt(m / (2 * math.pi * eta * x ** 3)) * math.exp(-(x - m) ** 2 / (2 * eta * x))
for m in (0.25, 1.0, 4.0):
    n, X = 400000, 60.0 * max(1, m)
    h = X / n
    s_spec = simpson([0.0] + [spec(i * h, m) for i in range(1, n + 1)], h)
    s_true = simpson([ig_pdf(i * h, m, m / eta) for i in range(n + 1)], h)
    print(f"mu={m}: integral of spec density {s_spec:.6f} (mu^-1/2 = {m ** -0.5:.6f}); of IG(mu, mu/eta) {s_true:.6f}")
m1, m2 = 0.5, 1.5
for x in (0.3, 1.0, 3.0):
    dx = 1e-5
    lr = lambda t: math.log(ig_pdf(t, m2, m2 / eta)) - math.log(ig_pdf(t, m1, m1 / eta))
    num = (lr(x + dx) - lr(x - dx)) / (2 * dx)
    formula = (m2 - m1) / (2 * eta) * (1 / (m1 * m2) + 1 / x ** 2)
    print(f"x={x}: d/dx log f(x;mu2)/f(x;mu1): numeric {num:.6f}, spec formula {formula:.6f}")

print("\n== C. FSD between the true laws at day H and day 2H (eta = 0.25) ==")
eta = 0.25
P = lambda a: (a, a / eta)
group = lambda n, a: (n * a, n * n * a / eta)  # n equal terms P(a) sum to IG(n a, n^2 a/eta)
DH = [P(0.1), group(10, 0.07)]  # mean 0.8: reset draw + 10 missions
D2H = [(0.5 * 0.1, 0.5 * 0.1 / eta), (0.5 * 0.7, 0.5 * 100 * 0.07 / eta), P(0.35)]  # 0.5*D_H + one mission
print(f"means: D_H {sum(m for m, _ in DH):.3f}, D_2H {sum(m for m, _ in D2H):.3f}")
for t in (0.6, 0.8, 1.0, 1.2, 1.5):
    print(f"t={t}: P(D_H > t) = {sum_tail(DH, t, 2000):.5f}   P(D_2H > t) = {sum_tail(D2H, t, 2000):.5f}")
p_mc, se = mc_tail(D2H, 1.2, 1000000, seed=4)
print(f"MC check P(D_2H > 1.2) = {p_mc:.5f} +- {se:.5f}")
