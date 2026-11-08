"""Task B, step B3: import fleet_management with gurobipy and streamlit refused.

Usage: PYTHONPATH=<worktree>/src python taskB_check_api.py
Checks: the import succeeds; __all__ is the spec's list in the spec's order; each name
comes from the module of the plan; gurobipy, streamlit and degradation_model.base are not loaded.
"""
import importlib.abc
import sys

REFUSED = ("gurobipy", "streamlit")


class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in REFUSED:
            raise ImportError(f"import of {name} refused by the check")
        return None


sys.meta_path.insert(0, Refuse())

import fleet_management  # noqa: E402

SPEC_ORDER = [
    ("solve", "fleet_management.solver"),
    ("sweep_operating_horizons", "fleet_management.horizon_sweep"),
    ("sweep_horizon_grid", "fleet_management.horizon_sweep"),
    ("plot_management", "fleet_management.utils.plotter"),
    ("plot_horizon_sweep", "fleet_management.utils.plotter"),
    ("validate", "fleet_management.validation.validator"),
    ("validate_gamma_replay_schedule",
     "fleet_management.degradation_model.gamma_utils.gamma_replay_validator"),
    ("validate_gamma_replay_files",
     "fleet_management.degradation_model.gamma_utils.gamma_replay_validator"),
    ("validate_gamma_stochastic_schedule",
     "fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator"),
    ("validate_gamma_schedule",
     "fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator"),
    ("validate_gamma_schedule_files",
     "fleet_management.degradation_model.gamma_utils.gamma_stochastic_validator"),
]

ok = True
names = [name for name, _ in SPEC_ORDER]
if list(fleet_management.__all__) != names:
    ok = False
    print("FAIL __all__ differs:", fleet_management.__all__)
else:
    print("ok   __all__ is the spec's list in the spec's order:", len(names), "names")

for name, module in SPEC_ORDER:
    obj = getattr(fleet_management, name)
    if obj.__module__ != module:
        ok = False
        print("FAIL", name, "comes from", obj.__module__, "expected", module)
    else:
        print("ok  ", name, "<-", module)

public = {k for k in vars(fleet_management) if not k.startswith("_")}
public -= {"solver", "horizon_sweep", "utils", "validation", "degradation_model", "config"}
extra = public - set(names)
print("non-listed public attributes of the package:", sorted(extra) or "none")

for banned in ("gurobipy", "streamlit", "fleet_management.degradation_model.base",
               "fleet_management.utils.mixed_plotter"):
    if banned in sys.modules:
        ok = False
        print("FAIL loaded by the import:", banned)
    else:
        print("ok   not loaded:", banned)

print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
