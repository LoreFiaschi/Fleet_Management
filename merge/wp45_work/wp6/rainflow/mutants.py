"""Mutation check: each defect planted in rainflow.py must make some local test fail.

Usage: python mutants.py
Runs tests/test_rainflow.py through the emulator (review_run.py) once per mutant.
"""
import importlib.util
import sys
import types
from pathlib import Path

import review_run as rr  # installs the emulator, the package stubs and the pytest stub

RF = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src\fleet_management\degradation_model\rainflow.py")
TR = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\tests\test_rainflow.py")
SOURCE = RF.read_text(encoding="utf-8")

MUTANTS = {
    "Bernstein side row tau (main)": ('"mu_side": tau - c}', '"mu_side": tau}'),
    "Bernstein c1 sign": ('"c1": -b / 3.0', '"c1": b / 3.0'),
    "Cantelli c2 = eps": ('"c2": eps / (1.0 - eps)', '"c2": eps'),
    "Hoeffding c2 = kappa / 2": ('"c2": 2.0 / kappa', '"c2": kappa / 2.0'),
    "Markov cap tau": ('"cap": eps * tau}', '"cap": tau}'),
    "Chernoff cap s tau": ('"cap": s * tau + math.log(eps)}', '"cap": s * tau}'),
    "Hoeffding increment not squared": ('("R", "support", True)', '("R", "support", False)'),
    "Chernoff tracks v": ('"chernoff": ("K", "cgf", False)', '"chernoff": ("v", "v", False)'),
    "v_max_user at every bound": ('if constants["q"] == "v"', 'if True'),
    "no v_max_user row": ('if math.isfinite(rows["v_max_user"]):', 'if False:'),
    "NonConvex at linear rows": ('for i, l in quadratic):\n            ctx.model.Params.NonConvex',
                                 'for i, l in cells):\n            ctx.model.Params.NonConvex'),
    "no anchors": ('base.add_state_vars(ctx, BOUND_QUANTITY[bound][0], own)',
                   'base.add_state_vars(ctx, BOUND_QUANTITY[bound][0], own, anchored=False)'),
    "no repair cost": ('        base.add_repair_cost_rows(ctx, i, l)\n', '        pass\n'),
    "no support check": ('if bound == "bernstein" and not math.isfinite(b):', 'if False:'),
}


def run(source):
    for name in ("fleet_management.degradation_model.rainflow", "test_rainflow"):
        sys.modules.pop(name, None)
    rr_base = sys.modules["fleet_management.degradation_model.base"]
    rr_base.CELL_BUILDERS.pop("rainflow", None)
    module = types.ModuleType("fleet_management.degradation_model.rainflow")
    module.__file__ = str(RF)
    sys.modules[module.__name__] = module
    exec(compile(source, str(RF), "exec"), module.__dict__)
    spec = importlib.util.spec_from_file_location("test_rainflow", TR)
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    original = tests._fix_schedule

    def fix_with_selector(ctx, schedule):
        original(ctx, schedule)
        for (i, l, k), variable in ctx.nb.items():
            variable.LB = variable.UB = 1.0 - schedule["m"][i, l, k] - schedule["r"][i, l, k]
    tests._fix_schedule = fix_with_selector
    failed = []
    for name, func in sorted(vars(tests).items()):
        if not name.startswith("test_") or not callable(func) or getattr(func, "_skip", None):
            continue
        for label, kwargs in rr.cases(func):
            try:
                func(**kwargs)
            except (NotImplementedError, RuntimeError) as error:
                if "fake gurobipy" not in str(error):
                    failed.append(f"{name}[{label}]")
            except Exception:
                failed.append(f"{name}[{label}]")
    return failed


import fleet_management.degradation_model.base  # noqa: E402,F401
print("original:", len(run(SOURCE)), "failures")
for label, (old, new) in MUTANTS.items():
    assert SOURCE.replace("\r\n", "\n").count(old) == 1, label
    failed = run(SOURCE.replace("\r\n", "\n").replace(old, new))
    tests = sorted({f.split("[")[0] for f in failed})
    print(f"{'caught' if failed else 'MISSED'}  {label}: {len(failed)} cases, {tests}")
