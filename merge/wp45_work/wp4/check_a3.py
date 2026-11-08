import sys
import types

import numpy as np

package = types.ModuleType("fleet_management")
package.__path__ = [r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src\fleet_management"]
sys.modules["fleet_management"] = package
from fleet_management import config as c


def attempt(label, func, *args):
    try:
        print(f"{label}: ok -> {func(*args)!r}")
    except Exception as exc:
        print(f"{label}: {type(exc).__name__}: {exc}")


attempt("integer 3", c._integer, 3, "F")
attempt("integer 3.0", c._integer, 3.0, "F")
attempt("integer np.int64", c._integer, np.int64(4), "F")
attempt("integer 2.5", c._integer, 2.5, "F")
attempt("integer True", c._integer, True, "F")
attempt("integer nan", c._integer, float("nan"), "F")
attempt("number 2", c._number, 2, "x")
attempt("number np.float32", c._number, np.float32(0.5), "x")
attempt("number True", c._number, True, "x")
attempt("number 'a'", c._number, "a", "x")
attempt("flag True", c._flag, True, "x")
attempt("flag np.bool_", c._flag, np.bool_(False), "x")
attempt("flag 1", c._flag, 1, "x")
attempt("horizon 3", c._parse_horizon, 3)
attempt("horizon [2,3]", c._parse_horizon, [2, 3])
attempt("horizon (2,3)", c._parse_horizon, (2, 3))
attempt("horizon [1]", c._parse_horizon, [1])
attempt("horizon [1,0]", c._parse_horizon, [1, 0])
attempt("horizon 0", c._parse_horizon, 0)
attempt("horizon 1.5", c._parse_horizon, 1.5)
attempt("L str", c._infer_L, {"model": "gaussian"}, 2)
attempt("L str given", c._infer_L, {"model": "gaussian", "L": 3}, 2)
attempt("L list", c._infer_L, {"model": ["gaussian", "gamma"]}, 2)
attempt("L FxL", c._infer_L, {"model": [["gaussian", "gamma"]] * 2}, 2)
attempt("L differs", c._infer_L, {"model": [["gaussian", "gamma"]] * 2, "L": 3}, 2)
attempt("L 3-d", c._infer_L, {"model": [[["a"]]]}, 2)
attempt("L 0", c._infer_L, {"model": "gaussian", "L": 0}, 2)
attempt("names None", c._component_names, None, 2)
attempt("names ok", c._component_names, [" a ", "b"], 2)
attempt("names str", c._component_names, "a", 1)
attempt("names len", c._component_names, ["a"], 2)
attempt("names empty", c._component_names, ["a", " "], 2)
attempt("names int", c._component_names, ["a", 5], 2)
attempt("run defaults", c._run_options, {}, 2)
attempt("run ok", c._run_options, {"verbose": True, "time_limit": 600, "objective_mode": "evaluation_total",
                                  "evaluation_horizon": 30.0, "gurobi_params": {"Threads": 4}}, 2)
for bad in ({"objective_mode": "avg"}, {"objective_mode": "evaluation_total"}, {"evaluation_horizon": 2},
            {"evaluation_horizon": 2.5}, {"mip_gap": -0.1}, {"time_limit": 0}, {"progress_interval_seconds": 0},
            {"relaxation_time_limit": 0}, {"gurobi_params": [1]}, {"validation": "x"}, {"fast": "yes"},
            {"relaxation_warm_start": 1}, {"verbose": 2}, {"verbose": 1.5}):
    attempt(f"run bad {bad}", c._run_options, bad, 2)
