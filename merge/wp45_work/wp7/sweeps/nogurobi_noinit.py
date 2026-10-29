import sys, types
SRC = r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src"
sys.path.insert(0, SRC)
sys.modules['gurobipy'] = None
pkg = types.ModuleType("fleet_management"); pkg.__path__ = [SRC + r"\fleet_management"]
sys.modules["fleet_management"] = pkg
import greedy_ref  # noqa  (only for the case data)
from fleet_management.config import load_config
from fleet_management.greedy_warm_start import build_greedy_warm_start
cfg = load_config(dict(greedy_ref.CASES["G_linear"], verbose=0))
source, report = build_greedy_warm_start(cfg)
assert report.repairs == 8 and report.repeatability_feasible
print("greedy without gurobipy: OK;", "base loaded:", "fleet_management.degradation_model.base" in sys.modules)
