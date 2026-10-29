"""High-precision survival of Z ~ IG(1, 1/eta) at the two candidate quantiles. Usage: python ig_mp.py"""
import mpmath as mp
mp.mp.dps = 60
def sf(x, eta):
    x, eta = mp.mpf(x), mp.mpf(eta)
    r = 1 / mp.sqrt(eta * x)
    F = mp.ncdf(r * (x - 1)) + mp.exp(2 / eta) * mp.ncdf(-r * (x + 1))
    return 1 - F
for eta, eps, mine, sp in ((1e-6, 0.5, 0.9999995000002919, 0.999999501948993),
                           (1e-3, 1e-12, 1.2479105239702215, 1.247910600495865),
                           (1e6, 1e-12, 18194156.479684554, 18194156.51479229)):
    print(eta, eps, "mine:", mp.nstr(sf(mine, eta) / eps - 1, 5), "scipy:", mp.nstr(sf(sp, eta) / eps - 1, 5))
