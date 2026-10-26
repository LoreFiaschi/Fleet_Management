"""S4: provably safe caps (CVaR cap, sub-gamma Bernstein, Cantelli) vs lore's cap and the exact cap."""
import math, sys
sys.path.insert(0, sys.argv[1])
from iglib import *

TAU = 1.0


def partial_exp(u, mu, lam, n=20000):
    """E(X - u)_+ = int_u^inf P(X > x) dx for X ~ IG(mu, lam)."""
    top = u + 40.0 * math.sqrt(mu ** 3 / lam) + 80.0 * mu * mu / lam + 10.0 * mu
    h = (top - u) / n
    return simpson([ig_sf(u + i * h, mu, lam) for i in range(n + 1)], h)


def cvar(eps, mu, lam):
    """CVaR = q + E(X - q)_+/eps, q the (1-eps)-quantile (Rockafellar-Uryasev minimiser)."""
    q = bisect(lambda t: eps - ig_sf(t, mu, lam), 1e-12, 1e4 * mu)
    return q + partial_exp(q, mu, lam) / eps, q


print("== A. CVaR cap mu_hat = tau / CVaR(Z) vs lore's cap mu_bar = tau / VaR(Z), Z ~ IG(1, 1/eta) ==")
for eta in (0.05, 0.25, 1.0, 4.0, 10.0, 100.0):
    row = f"eta={eta:<6}"
    for eps in (0.001, 0.01, 0.1):
        cv, q = cvar(eps, 1.0, 1.0 / eta)
        mb = mu_bar_eta(eta, TAU, eps)
        row += f" | eps={eps}: mu_bar={mb:.4f} mu_hat={TAU / cv:.4f} ratio={TAU / cv / mb:.3f}"
    print(row)

print("\n== A'. check: Markov stop-loss bound at d = VaR equals eps; mu_hat is safe on the S1 counterexamples ==")
for eta, eps in ((10.0, 0.1), (100.0, 0.01), (4.0, 0.2)):
    cv, q = cvar(eps, 1.0, 1.0 / eta)
    mh = TAU / cv
    worst = max((ig_sf(TAU, mh, n * mh / eta), n) for n in (1, 2, 3, 5, 10, 30, 100, 1000, 10 ** 6))
    # bound at the optimal d = mu_hat * q
    d = mh * q
    bound = mh * partial_exp(q, 1.0, 1.0 / eta) / (TAU - d)
    print(f"eta={eta} eps={eps}: mu_hat={mh:.4f}; stop-loss bound {bound:.5f}; worst true tail over n-equal = {worst[0]:.5f} (n={worst[1]})")

print("\n== B. sub-gamma check: psi(s) = (1/eta)(1 - sqrt(1 - 2 eta a s)) - a s <= eta a^2 s^2 / (2 (1 - 2 eta a s)) ==")
ck = [0.5]
for k in range(1, 40):
    ck.append(ck[-1] * (2 * k - 1) / (2 * k + 2))  # coefficient of u^(k+1) in 1 - sqrt(1-u)
print("coefficients c_k of 1 - sqrt(1-u), k=1..6:", [round(c, 6) for c in ck[:6]], "; max_{k>=2} c_k =", max(ck[1:]))
for eta, a in ((0.25, 1.0), (10.0, 0.3)):
    smax = 1 / (2 * eta * a)
    worst = -1e9
    for i in range(1, 2000):
        s = smax * i / 2000
        psi = (1 - math.sqrt(1 - 2 * eta * a * s)) / eta - a * s
        bnd = eta * a * a * s * s / (2 * (1 - 2 * eta * a * s))
        worst = max(worst, psi - bnd)
    print(f"eta={eta} a={a}: max over s in (0, 1/(2 eta a)) of psi - bound = {worst:.3e} (must be <= 0)")


def caps(eta, eps, mu0, m):
    """Caps for the structure: reset draw P(mu0) + equal missions P(m) (sum of n: IG(nm, n^2 m/eta))."""
    kap, kC = math.log(1 / eps), math.sqrt((1 - eps) / eps)
    amax = max(mu0, m)
    csg = 2 * eta * amax
    var = lambda mu: eta * (mu0 ** 2 + (mu - mu0) * m)  # eta * sum a_t^2 for this structure
    true_tail = lambda mu: sum_tail([(mu0, mu0 / eta), (mu - mu0, (mu - mu0) ** 2 / (eta * m))], TAU, 1500)
    def cap(f, lo, hi, it=200):
        """Largest mu in [lo, hi] with f(mu) <= 0, f increasing; 0.0 if none."""
        if f(lo) > 0:
            return 0.0
        if f(hi) <= 0:
            return hi
        return bisect(f, lo, hi, it)

    out = {}
    out["lore mu_bar (not proven)"] = mu_bar_eta(eta, TAU, eps)
    out["CVaR cap (S4.1)"] = TAU / cvar(eps, 1.0, 1.0 / eta)[0]
    bern_mean = lambda mu: mu + kap * csg + math.sqrt((kap * csg) ** 2 + 2 * kap * eta * amax * mu) - TAU
    out["Bernstein, mean only (S4.3)"] = cap(bern_mean, 1e-9, TAU)
    bern_v = lambda mu: mu + kap * csg + math.sqrt((kap * csg) ** 2 + 2 * kap * var(mu)) - TAU
    out["Bernstein, tracked v (S4.2)"] = cap(bern_v, mu0 + 1e-9, TAU)
    cant_v = lambda mu: mu + kC * math.sqrt(var(mu)) - TAU
    out["Cantelli, tracked v (S4.4)"] = cap(cant_v, mu0 + 1e-9, TAU)
    out["exact cap for this structure"] = cap(lambda mu: true_tail(mu) - eps, mu0 + 1e-6, TAU * (1 - 1e-9), it=50)
    return out


print("\n== C. caps on example structures (reset draw mu0 + equal missions m), tau = 1 ==")
for eta, eps, mu0, m in ((0.25, 0.01, 0.05, 0.02), (0.25, 0.01, 0.05, 0.2), (1.0, 0.01, 0.05, 0.02), (10.0, 0.1, 0.05, 0.05)):
    print(f"eta={eta} eps={eps} mu0={mu0} m={m}:")
    for k, v in caps(eta, eps, mu0, m).items():
        print(f"   {k:<32} {v:.4f}")
