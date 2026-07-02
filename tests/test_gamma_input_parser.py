from pathlib import Path

from fleet_management.solver import _read_input
from fleet_management.model_registry import extract_degradation_parameters


def test_tiny_gamma_synthetic_input_parses():
    input_path = Path("input/tiny_gamma_synthetic.yaml")

    data = _read_input(input_path)
    params = extract_degradation_parameters(data, "gamma")

    assert params["degradation"] == "gamma"
    assert params["F"] == 2
    assert params["M"] == 1
    assert params["L"] == 2
    assert params["H"] == 2

    assert params["mu_0"].shape == (2, 2)
    assert params["mu_param"].shape == (2, 1, 2, 2)
    assert params["gamma_beta"].shape == (2,)

    assert params["tau"] == 1.0
    assert params["epsilon"] == 0.05