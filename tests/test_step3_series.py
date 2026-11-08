"""Step 3 tests: the Moschopoulos series without underflow (M13) and the margins (M16).

No Gurobi. The values "before" were computed with the code before the change of M13
(the first mixture weight was formed in ``numpy.longdouble`` and raised when it underflowed).
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtri
from scipy.stats import gamma

from fleet_management import greedy_warm_start
from fleet_management.config import load_config
from fleet_management.degradation_model.gamma_utils.gamma_tail_bound import (
    moschopoulos_tail_probability,
)
from fleet_management.validation.dashboard_inspector import load_pair, reliability_margins

FIXTURES = Path(__file__).resolve().parents[1] / "results" / "validator_tests"

# (shapes, rates, threshold, estimate, upper bound) with the code before the change.
_BEFORE = [
    ([2.0, 3.0], [1.0, 0.5], 2.0, 0.9892590978985508, 0.9892590978991399),
    ([2.0, 3.0], [1.0, 0.5], 10.0, 0.2552701871586962, 0.2552701871592853),
    ([2.0, 3.0], [1.0, 0.5], 20.0, 0.007627132548360343, 0.007627132548949428),
    ([5.0, 1.5, 4.0], [2.0, 0.7, 1.1], 3.0, 0.995093271636349, 0.9950932716371499),
    ([5.0, 1.5, 4.0], [2.0, 0.7, 1.1], 8.0, 0.4913762305592069, 0.4913762305600078),
    ([40.0, 60.0], [1.0, 0.3], 150.0, 0.9999610414058284, 0.9999610414067853),
    ([40.0, 60.0], [1.0, 0.3], 250.0, 0.34062878958742593, 0.34062878958838283),
    ([40.0, 60.0], [1.0, 0.3], 400.0, 2.7316567184197523e-07, 2.7316662874320016e-07),
    ([300.0, 200.0], [1.0, 0.3], 900.0, 0.9106288248488253, 0.9106288248497524),
    ([300.0, 200.0], [1.0, 0.3], 1100.0, 0.005484814397511543, 0.005484814398438579),
]


@pytest.mark.parametrize("threshold", [11500.0, 12000.0, 12500.0, 13500.0])
def test_series_large_shapes(threshold):
    """Shapes 2000 and 3000: the first weight is exp(-1000) or smaller, below float64."""
    shapes, rates = [2000.0, 3000.0], [1.0, 0.3]
    result = moschopoulos_tail_probability(shapes, rates, threshold)
    assert 0.0 <= result.estimate <= result.upper_bound <= 1.0
    assert result.remaining_mass <= 1e-12
    # P(X + Y > t) = P(X > t) + int_0^t f_X(x) P(Y > t - x) dx, X ~ Gamma(2000, 1), Y ~ Gamma(3000, rate 0.3)
    integral, _ = quad(
        lambda x: gamma.pdf(x, 2000.0) * gamma.sf(threshold - x, 3000.0, scale=1.0 / 0.3),
        0.0, threshold, points=[1900.0, 2000.0, 2100.0], limit=500, epsabs=1e-13, epsrel=1e-13)
    reference = gamma.sf(threshold, 2000.0) + integral
    assert abs(result.upper_bound - reference) < 1e-6
    assert abs(result.estimate - reference) < 1e-6


def test_series_large_shapes_normal_sanity():
    """Sanity check only: the normal approximation at the mean gives about one half."""
    result = moschopoulos_tail_probability([2000.0, 3000.0], [1.0, 0.3], 12000.0)
    assert abs(result.upper_bound - 0.5) < 0.01


@pytest.mark.parametrize("shapes, rates, threshold, estimate, upper", _BEFORE)
def test_series_unchanged_cases(shapes, rates, threshold, estimate, upper):
    result = moschopoulos_tail_probability(shapes, rates, threshold)
    # the stopping rule keeps the remaining mass below 1e-12, so the values agree to that
    assert result.estimate == pytest.approx(estimate, rel=1e-12, abs=1e-12)
    assert result.upper_bound == pytest.approx(upper, rel=1e-12, abs=1e-12)


def test_series_equal_rates_shortcut():
    result = moschopoulos_tail_probability([2.0, 3.0], [0.5, 0.5], 7.0)
    value = float(gamma.sf(7.0, a=5.0, scale=2.0))
    assert (result.estimate, result.upper_bound, result.remaining_mass, result.series_terms) == (
        value, value, 0.0, 1)


def test_series_too_few_terms_raises():
    with pytest.raises(RuntimeError, match="did not reach the requested tolerance"):
        moschopoulos_tail_probability([40.0, 60.0], [1.0, 0.3], 250.0, max_series_terms=50)


def test_series_no_underflow_error():
    """The message about an underflowed initial weight no longer exists."""
    result = moschopoulos_tail_probability([2000.0, 3000.0], [1.0, 0.3], 12000.0)
    assert result.series_terms > 1000


# ---------------------------------------------------------------------------
# M16: the margins with eps = 0.05 are unchanged by -ndtri(eps)
# ---------------------------------------------------------------------------
def test_inspector_margins_unchanged():
    cfg, result = load_pair(FIXTURES / "input.yaml", FIXTURES / "result.yaml")
    margins = reliability_margins(cfg, result)
    # train 2 Wheelset (Wiener) and train 1 Wheelset (Gaussian), eps = 0.05, values before
    np.testing.assert_allclose(margins[1, 0], [0.6203122391642665, 0.5266772500555805,
                                               0.4402422609468946, 0.9602917782618361,
                                               0.8438567891531503], rtol=1e-12)
    np.testing.assert_allclose(margins[0, 0], [0.6547472282729524, 0.5759766739275427,
                                               0.9602917782618361, 0.8460212239164266,
                                               0.7562506695710172], rtol=1e-12)


def test_greedy_margin_constant_unchanged():
    cfg = load_config(dict(F=2, M=1, H=[2, 2], model="gaussian", tau=1.0, epsilon=0.05, rho=0.5,
                           mu_0=0.125, v_0=0.0, mu=0.25, v=0.00390625,
                           reliability_impl="tangent", C_M=1.0, C_R=1.0, C_rep=5.0, C_D=0.0,
                           verbose=0))
    row = greedy_warm_start._cell_rows(cfg, 0, 0, greedy_warm_start._profiles(cfg))
    assert row["c2"] == pytest.approx(0.3696115094681951, rel=1e-12)
    assert row["c2"] == pytest.approx(1.0 / float(ndtri(0.95)) ** 2, rel=1e-12)
    report = greedy_warm_start.build_greedy_warm_start(cfg)[1].as_dict()
    assert (report["repairs"], report["replacements"], report["repeatability_failures"]) == (2, 0, 1)
    assert report["maximum_repeatability_excess"] == pytest.approx(0.0625)
    assert report["maximum_proxy_ratio"] == pytest.approx(0.625)
