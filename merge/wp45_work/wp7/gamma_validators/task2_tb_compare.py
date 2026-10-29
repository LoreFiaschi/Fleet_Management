"""Compare the new gamma_tail_bound.py (worktree) with the old one (git HEAD copy in old_tb/).

Usage: python task2_tb_compare.py
"""
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge")
package = types.ModuleType("fleet_management")
package.__path__ = [str(ROOT / "src" / "fleet_management")]
sys.modules["fleet_management"] = package
sys.modules["gurobipy"] = None
sys.path.insert(0, str(HERE / "old_tb"))

import old_tail_bound as old  # noqa: E402
from fleet_management.degradation_model.gamma_utils import gamma_tail_bound as new  # noqa: E402

cases = [
    ([6.0, 6.0], [40.0, 20.0], 0.8),
    ([1.0, 3.0, 6.0, 6.0], [20.0, 40.0, 40.0, 20.0], 1.0),
    ([0.7, 3.2], [5.0, 20.0], 0.35),
    ([0.8, 1.7, 2.4], [5.0, 12.0, 30.0], 0.45),
    ([2.0, 3.0], [5.0, 5.0], 1.0),
]
for repairs in (2, 4):
    cases.append(([6.0] * (repairs + 1), [20.0 * 2.0 ** n for n in range(repairs + 1)], 0.8))
worst = 0.0
for shapes, rates, threshold in cases:
    t0 = time.perf_counter()
    a = new.moschopoulos_tail_probability(shapes, rates, threshold)
    t1 = time.perf_counter()
    b = old.moschopoulos_tail_probability(shapes, rates, threshold)
    t2 = time.perf_counter()
    diff = abs(a.upper_bound - b.upper_bound)
    worst = max(worst, diff)
    print(f"n={len(shapes)} terms new={a.series_terms} old={b.series_terms} "
          f"upper new={a.upper_bound:.15e} diff={diff:.1e} mass new={a.remaining_mass:.2e} "
          f"old={b.remaining_mass:.2e} t_new={t1 - t0:.3f}s t_old={t2 - t1:.3f}s")
print("worst difference of the upper bounds:", worst)
for repairs in (6, 8):
    shapes, rates = [6.0] * (repairs + 1), [20.0 * 2.0 ** n for n in range(repairs + 1)]
    t0 = time.perf_counter()
    a = new.moschopoulos_tail_probability(shapes, rates, 0.8)
    print(f"{repairs} repairs: terms={a.series_terms} upper={a.upper_bound:.6e} "
          f"t_new={time.perf_counter() - t0:.2f}s")
try:
    new.moschopoulos_tail_probability([6.0] * 9, [20.0 * 2.0 ** n for n in range(9)], 0.8,
                                      max_series_terms=1000)
except RuntimeError as error:
    print("RuntimeError:", error)
