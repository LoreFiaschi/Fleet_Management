"""Apply the config.py change of the GIW plan (section 4) to a copy, and run test_config.py.

Usage: python config_patch.py
Writes patched/fleet_management/config.py and patched/test_config_plus.py (the worktree's
tests/test_config.py plus the two new tests of the plan), then runs the tests on the copy.
The worktree is not changed.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge")
CONFIG = WORKTREE / "src" / "fleet_management" / "config.py"
TESTS = WORKTREE / "tests" / "test_config.py"

IMPORT_OLD = "from scipy.special import ndtri\n"
IMPORT_NEW = "from scipy.optimize import brentq\nfrom scipy.special import log_ndtr, ndtri\n"

ANCHOR_SECTION = ("# ---------------------------------------------------------------------------\n"
                  "# Consistency checks\n")
SECTION = '''# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
def _ig_unit_log_cdf(x: float, eta: float) -> float:
    """ln F(x) of Z ~ IG(1, 1/eta), x > 0 (spec, "Inverse Gaussian Model")."""
    r = 1.0 / math.sqrt(eta * x)
    return float(np.logaddexp(log_ndtr(r * (x - 1.0)), 2.0 / eta + log_ndtr(-r * (x + 1.0))))


def _ig_unit_log_sf(x: float, eta: float) -> float:
    """ln (1 - F(x)) of Z ~ IG(1, 1/eta), x > 0, without the cancellation of 1 - F."""
    r = 1.0 / math.sqrt(eta * x)
    first = float(log_ndtr(-r * (x - 1.0)))
    second = 2.0 / eta + float(log_ndtr(-r * (x + 1.0)))
    return first + math.log1p(-math.exp(second - first))


def _ig_unit_quantile(eta: float, epsilon: float) -> float:
    """The (1 - epsilon)-quantile q of Z ~ IG(1, 1/eta): Pr(Z > q) = epsilon.

    Root of ln Pr(Z > e^u) = ln epsilon in u = ln q, by brentq on [-50, ln b], with
    b = 1 + sqrt(eta (2/epsilon - 1)): Var Z = eta, so Pr(Z > b) <= epsilon/2 (Cantelli).
    """
    target = math.log(epsilon)

    def gap(u):
        return _ig_unit_log_sf(math.exp(u), eta) - target

    lo, hi = -50.0, math.log(1.0 + math.sqrt(eta * (2.0 / epsilon - 1.0)))
    if not (gap(lo) > 0.0 > gap(hi)):
        raise ValueError(f"no quantile bracket for eta={eta}, epsilon={epsilon}")
    return math.exp(brentq(gap, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200))


def ig_mean_cap(eta: float, tau: float, epsilon: float) -> float:
    """Mean cap mu_bar = tau / E[Z | Z > q] of an inverse Gaussian cell (spec, Eq. ig_row).

    E[Z; Z > q] = F(1/q), so mu_bar = tau * epsilon / F(1/q), computed in log space.
    The builder ``degradation_model/inverse_gaussian.py`` imports it; it lives here
    because the check of the replacement state needs it without Gurobi.
    """
    q = _ig_unit_quantile(eta, epsilon)
    return tau * math.exp(math.log(epsilon) - _ig_unit_log_cdf(1.0 / q, eta))


'''

MARKOV_OLD = '''        check(m["markov"], _le(mu, cfg.epsilon * cfg.tau), "replacement_mu",
              "must be <= epsilon*tau at a markov cell")
'''
MARKOV_NEW = MARKOV_OLD + '''        ig = m["inverse_gaussian"]
        mu_bar = np.full(mu.shape, np.nan)
        for i, l in zip(*np.nonzero(ig)):
            mu_bar[i, l] = ig_mean_cap(float(cfg.eta[i, l]), float(cfg.tau[i, l]),
                                       float(cfg.epsilon[i, l]))
        check(ig, _le(mu, mu_bar), "replacement_mu",
              "must be <= the mean cap tau/E[Z | Z > q] at an inverse_gaussian cell")
'''

NEW_TESTS = '''

# ---------------------------------------------------------------------------
# Inverse Gaussian mean cap (merge/wp6_plan_giw.md, section 2)
# ---------------------------------------------------------------------------
def test_inverse_gaussian_cap():
    from scipy.stats import invgauss

    # (eta, tau, epsilon, q, mu_bar), computed by brentq and checked by quadrature
    for eta, tau, eps, q, mu_bar in ((0.5, 1.0, 0.1, 1.8859953361, 0.3823093163),
                                     (0.1, 1.0, 0.01, 1.9488253601, 0.4627586840),
                                     (2.0, 2.0, 0.05, 3.5525960458, 0.3469170694)):
        assert abs(config._ig_unit_quantile(eta, eps) - q) <= 1e-9 * q
        assert abs(config.ig_mean_cap(eta, tau, eps) - mu_bar) <= 1e-9 * mu_bar
        assert abs(invgauss.sf(q, eta, scale=1.0 / eta) - eps) <= 1e-8 * eps
    for eta in (1e-3, 1.0, 1e3):
        for eps in (1e-9, 0.5, 0.99):
            cap = config.ig_mean_cap(eta, 1.0, eps)
            assert 0.0 < cap <= 1.0 / config._ig_unit_quantile(eta, eps)


def test_inverse_gaussian_replacement_cap():
    # cap 0.5493234 for eta = 0.15, epsilon = 0.1, tau = 1
    rule = "must be <= the mean cap"
    _fails("inverse_gaussian", None, "replacement_mu", rule, replacement_mu=0.56)
    load_config(_small("inverse_gaussian", replacement_mu=0.54))
    load_config(_small("inverse_gaussian", replacement_mu=0.56, allow_replacement=False))
'''


def patched_config() -> str:
    text = CONFIG.read_bytes().decode("utf-8").replace("\r\n", "\n")
    for old, new in ((IMPORT_OLD, IMPORT_NEW), (ANCHOR_SECTION, SECTION + ANCHOR_SECTION),
                     (MARKOV_OLD, MARKOV_NEW)):
        assert text.count(old) == 1, old[:60]
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    out = HERE / "patched" / "fleet_management"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.py").write_bytes(patched_config().replace("\n", "\r\n").encode("utf-8"))
    tests = TESTS.read_bytes().decode("utf-8").replace("\r\n", "\n") + NEW_TESTS
    (HERE / "patched" / "test_config_plus.py").write_text(tests, encoding="utf-8", newline="\n")
    result = subprocess.run([sys.executable, str(HERE / "run_config_tests.py"),
                             str(HERE / "patched" / "test_config_plus.py")],
                            capture_output=True, text=True)
    print(result.stdout[-3000:], result.stderr[-3000:])
