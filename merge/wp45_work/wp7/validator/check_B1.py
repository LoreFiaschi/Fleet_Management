"""Check of step B1: load_pair, cell_label, reliability_caps, reliability_margins."""
import sys
import types
from pathlib import Path

import numpy as np

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package

from fleet_management.validation.dashboard_inspector import (  # noqa: E402
    cell_label, load_pair, reliability_caps, reliability_margins)

FIX = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\results\validator_tests")
cfg, result = load_pair(FIX / "input.yaml", FIX / "result.yaml")
print("T", cfg.T)
print([cell_label(cfg, i, l) for i in range(cfg.F) for l in range(cfg.L)])
caps = reliability_caps(cfg)
print(caps)
assert abs(caps[0, 1] - 0.49234697216756246) < 1e-9
assert abs(caps[1, 1] - 0.6726905060677382) < 1e-9
assert np.isnan(caps).sum() == 4

expected = np.array([
    [0.654747228, 0.575976674, 0.960291778, 0.846021224, 0.756250670],
    [0.352346972, 0.282346972, 0.472346972, 0.432346972, 0.362346972],
    [0.620312239, 0.526677250, 0.440242261, 0.960291778, 0.843856789],
    [0.522690506, 0.482690506, 0.442690506, 0.652690506, 0.612690506],
    [0.756412851, 0.756412851, 0.671289976, 0.591167100, 0.884998013],
    [1.997414907, 2.137414907, 1.757414907, 1.377414907, 2.597414907]])
margins = reliability_margins(cfg, result)
print(margins.shape)
flat = margins.reshape(6, 5)
print(np.abs(flat - expected).max())
assert np.abs(flat - expected).max() < 1e-9
print("B1 ok")
