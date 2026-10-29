"""Extreme parameters of the IG cap prototype. Usage: python ig_extreme.py"""
from ig_cap import ig_unit_quantile, ig_mean_cap, invgauss
worst = 0.0
for eta in (1e-6, 1e-5, 1e-4, 1e-3, 0.1, 1.0, 10.0, 1e4, 1e6):
    for eps in (1e-12, 1e-9, 1e-6, 1e-3, 0.5, 0.9, 0.999):
        q = ig_unit_quantile(eta, eps)
        c = ig_mean_cap(eta, 1.0, eps)
        ref = float(invgauss.isf(eps, mu=eta, scale=1 / eta))
        rel = abs(q - ref) / ref
        worst = max(worst, rel)
        assert 0.0 < c < 1.0 and c <= 1.0 / q, (eta, eps, q, c)
        if rel > 1e-9:
            print("large diff", eta, eps, q, ref)
print("extreme grid passed; max rel. diff of q vs invgauss.isf:", worst)
