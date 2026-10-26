"""S4b: lore's cap on the structure reset 0.05 + 8 equal missions, eta = 10, eps = 0.1."""
import sys
sys.path.insert(0, sys.argv[1])
from iglib import *
eta, eps, mu0 = 10.0, 0.1, 0.05
mb = mu_bar_eta(eta, 1.0, eps)
m = (mb - mu0) / 8
terms = [(mu0, mu0 / eta), (8 * m, 64 * m / eta)]
print(f"mu_bar={mb:.4f}, missions 8 x {m:.5f}: true tail {sum_tail(terms, 1.0, 2000):.5f}")
single = [(mu0, mu0 / eta)] + [(m, m / eta)] * 8
p, se = mc_tail(single, 1.0, 1000000, seed=9)
print(f"MC (each mission sampled): {p:.5f} +- {se:.5f}")
