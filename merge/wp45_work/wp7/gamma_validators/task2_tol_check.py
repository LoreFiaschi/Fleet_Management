"""Smallest convolution_tolerance that the vectorised series (float64) reaches; new code against git HEAD.

Usage: python task2_tol_check.py
"""
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge")
package = types.ModuleType("fleet_management")
package.__path__ = [str(ROOT / "src" / "fleet_management")]
sys.modules["fleet_management"] = package
sys.modules["gurobipy"] = None
sys.path.insert(0, str(HERE / "old_tb"))

import numpy as np  # noqa: E402
import old_tail_bound as old  # noqa: E402
from fleet_management.degradation_model.gamma_utils import gamma_tail_bound as new  # noqa: E402

print("longdouble precision (eps):", np.finfo(np.longdouble).eps)
shapes, rates, threshold = [6.0, 6.0], [40.0, 20.0], 0.8
for tol in (1e-12, 1e-13, 1e-14, 1e-15, 1e-16):
    row = [f"tol={tol:.0e}"]
    for name, module in (("new", new), ("old", old)):
        try:
            tail = module.moschopoulos_tail_probability(shapes, rates, threshold, tolerance=tol,
                                                        max_series_terms=2000)
            row.append(f"{name}: terms={tail.series_terms} mass={tail.remaining_mass:.1e}")
        except RuntimeError as error:
            row.append(f"{name}: RuntimeError ({str(error)[-30:]})")
    print("  ".join(row))
