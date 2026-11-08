"""Step 3 input fixes (no Gurobi needed): M2 verbose, M3 component_names, M16 kappa_G, M17 IG cap.

Expected behaviour comes from spec/spec.tex: null and integer rules (359-364), component_names
(386-391), verbose (489), Gaussian row and kappa_G (1013-1031), inverse Gaussian cap (1092-1119).
Inputs are the files of tests/data/io/valid/.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.special import ndtri

from fleet_management.config import load_config
from fleet_management.degradation_model.inverse_gaussian import ig_mean_cap

DATA = Path(__file__).parent / "data" / "io" / "valid"


def _base(name="gaussian"):
    with open(DATA / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# M2: verbose is an integer key, a flag is refused (spec 364)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [True, False])
def test_verbose_flag_refused(value):
    data = _base()
    data["verbose"] = value
    with pytest.raises(ValueError, match="verbose"):
        load_config(data)


@pytest.mark.parametrize("value", [0, 1, 1.0])
def test_verbose_integer_accepted(value):
    data = _base()
    data["verbose"] = value
    assert load_config(data).options["verbose"] == int(value)


# ---------------------------------------------------------------------------
# M3: component_names null is refused, an absent key gives the default names (spec 359)
# ---------------------------------------------------------------------------
def test_component_names_null_refused():
    data = _base()
    data["component_names"] = None
    with pytest.raises(ValueError, match="component_names"):
        load_config(data)


def test_component_names_absent_gives_defaults():
    data = _base()
    del data["component_names"]
    cfg = load_config(data)
    assert cfg.component_names == [f"Component {l + 1}" for l in range(data["L"])]


# ---------------------------------------------------------------------------
# M16: kappa_G = -ndtri(eps) (spec 1013-1031)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eps", [1e-3, 1e-2, 0.05, 0.1, 0.25, 0.49])
def test_kappa_g_same_value(eps):
    data = _base()
    data["epsilon"] = eps
    cfg = load_config(data)
    kappa_g = ndtri(1.0 - eps)
    expected = cfg.tau ** 2 / kappa_g ** 2          # V_max = tau^2 / kappa_G^2
    got = cfg.range_v[np.isfinite(cfg.range_v)]
    assert got.size > 0
    np.testing.assert_allclose(got, expected[np.isfinite(cfg.range_v)], rtol=1e-14)
    assert abs(-ndtri(eps) - kappa_g) <= 1e-14 * kappa_g


def test_v_max_finite_for_tiny_epsilon():
    data = _base()
    data["epsilon"] = 1e-17
    data["v_0"] = 1e-4
    data["replacement_v"] = 1e-6
    data["v"] = [[1e-5], [1e-5]]
    cfg = load_config(data)
    v_max = cfg.range_v[np.isfinite(cfg.range_v)]
    assert v_max.size > 0 and np.all(np.isfinite(v_max)) and np.all(v_max > 0)


# ---------------------------------------------------------------------------
# M17: the inverse Gaussian cap down to epsilon = 1e-18 (spec 1092-1119)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eta,eps", [(1.0, 1e-16), (0.01, 1e-16), (1.0, 1e-18)])
def test_ig_cap_tiny_epsilon(eta, eps):
    cap = ig_mean_cap(eta, 1.0, eps)
    assert np.isfinite(cap) and 0.0 < cap < 1.0


def test_ig_cap_decreases_with_epsilon():
    caps = [ig_mean_cap(1.0, 1.0, eps) for eps in (1e-2, 1e-6, 1e-12, 1e-16, 1e-18)]
    assert all(a > b for a, b in zip(caps, caps[1:]))


def test_ig_cap_unchanged_for_normal_epsilon():
    # value computed with the unguarded tail function before the change
    assert ig_mean_cap(1.0, 1.0, 0.05) == pytest.approx(0.23714775097188218, rel=1e-12)
