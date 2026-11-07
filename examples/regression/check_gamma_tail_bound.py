"""Numerical smoke check for the Moschopoulos tail of a sum of Gamma variables."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def find_repository_root(start: Path) -> Path:
    """Find the project root independently of the checker's directory depth."""

    for candidate in (start, *start.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src" / "fleet_management").is_dir()
        ):
            return candidate

    raise RuntimeError("Could not locate the fleet-management repository root.")


REPOSITORY_ROOT = find_repository_root(Path(__file__).resolve().parent)

MODULE_PATH = (
    REPOSITORY_ROOT
    / "src/fleet_management/degradation_model/gamma_utils/gamma_tail_bound.py"
)
# Load the standalone numerical module without importing fleet_management's
# solver-facing package __init__, which imports Gurobi.
SPEC = importlib.util.spec_from_file_location("gamma_tail_bound_standalone", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
TAIL_BOUND = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TAIL_BOUND
SPEC.loader.exec_module(TAIL_BOUND)
moschopoulos_tail_probability = TAIL_BOUND.moschopoulos_tail_probability


def main() -> None:
    # Two unit-shape Gammas are exponentials, whose convolution tail has a
    # closed form.  This checks the different-rate convolution independently.
    rates = np.array([2.0, 5.0])
    threshold = 0.5
    convolution = moschopoulos_tail_probability(
        shapes=[1.0, 1.0], rates=rates, threshold=threshold
    )
    closed_form = (
        rates[1] * np.exp(-rates[0] * threshold)
        - rates[0] * np.exp(-rates[1] * threshold)
    ) / (rates[1] - rates[0])
    np.testing.assert_allclose(convolution.estimate, closed_form, atol=2e-12, rtol=0.0)
    assert convolution.upper_bound >= closed_form - 2e-12

    print("PASS Moschopoulos tail of a sum of Gamma variables")
    print("estimate   :", convolution.estimate)
    print("closed form:", closed_form)
    print("upper bound:", convolution.upper_bound)


if __name__ == "__main__":
    main()
