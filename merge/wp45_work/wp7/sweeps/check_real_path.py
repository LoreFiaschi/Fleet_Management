"""Run the sweeps through the real load_config and, with the fake gurobipy emulator, through solve_mixed."""
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAKE = "--fake" in sys.argv
if FAKE:
    sys.path.insert(0, str(HERE / "fakegp"))
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
import yaml
from fleet_management import horizon_sweep as h
from fleet_management.config import load_config

data = {"F": 3, "M": 1, "H": [2, 9], "model": "gaussian", "tau": 1.0, "epsilon": 0.1, "rho": 0.5, "mu_0": 0.1,
        "mu": 0.2, "v": 0.001, "C_M": 1.0, "C_R": 1.0, "C_rep": 5.0, "C_D": 0.0}
sizes = h._input_sizes(data)
print("sizes", sizes)
base = dict(output_path=None, stop_on_gradient=False, gradient_tolerance=1e-3, flat_gradients_required=2,
            minimum_cases=3, maximum_mip_gap_for_stopping=None, warm_start=False, warm_start_mode="start", n_workers=1,
            objective_mode="operating_average", evaluation_horizon=None)
for mode, E in (("operating_average", None), ("evaluation_total", 20), ("total", None)):
    settings = h._check_sweep_arguments(2, [3], **dict(base, objective_mode=mode, evaluation_horizon=E))
    # the case input of the sweep is accepted by the real loader
    import copy
    case = copy.deepcopy(data)
    case["H"] = [2, 3]
    case["objective_mode"] = mode
    if E is not None:
        case["evaluation_horizon"] = E
    cfg = load_config(case)
    print(mode, "load_config ok: H1, H2 =", cfg.H1, cfg.H2)

tmp = HERE / "tmp"
tmp.mkdir(exist_ok=True)
path = tmp / "real_path_input.yaml"
path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
logging.basicConfig(level=logging.ERROR, format="LOG %(levelname)s %(message)s")
report = h.sweep_operating_horizons(path, [2, 3], output_path=tmp / "real_path_report.yaml")
for row in report["cases"]:
    print(row["H2"], row["status"], row["objective"])
print("reason", report["stopping_rule"]["reason"], "keys", len(report))
