"""Run tests/test_rainflow.py through the linear emulator, without pytest and without Gurobi.

Usage: python run_local.py <path to test_rainflow.py> [name substring]

Expands ``pytest.mark.parametrize`` (stacked decorators = product), emulates
``pytest.raises(match=...)`` and ``pytest.approx``. A case that needs a real
Gurobi (quadratic rows, ``getConstrByName``, free binaries) is reported as
SKIP, not as a failure. Shims on the emulator: ``getConstrs`` returns objects
with ``ConstrName``; ``Params.NonConvex`` defaults to -1; ``getConstrByName``
returns an object with ``RHS`` for a one-sided linear row.
"""
import contextlib
import importlib.util
import inspect
import itertools
import re
import sys
import traceback
import types
from pathlib import Path

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
SCRATCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRATCH / "wp5" / "fakegp"))
for name, path in (("fleet_management", SRC / "fleet_management"),
                   ("fleet_management.degradation_model",
                    SRC / "fleet_management" / "degradation_model")):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package

import gurobipy as gp  # noqa: E402  (the emulator)

_orig_init = gp.Model.__init__


def _init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self.Params.NonConvex = -1


def _get_constrs(self):
    return [types.SimpleNamespace(ConstrName=name) for _, name in self.rows]


def _get_constr_by_name(self, name):
    for tc, row_name in self.rows:
        if row_name == name:
            return types.SimpleNamespace(RHS=-tc.expr.const, ConstrName=name)
    return None


gp.Model.__init__ = _init
gp.Model.getConstrs = _get_constrs
gp.Model.getConstrByName = _get_constr_by_name


class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected, self.rel, self.abs = expected, rel, abs

    def __eq__(self, other):
        return abs(other - self.expected) <= max(self.rel * abs(self.expected), self.abs)

    def __repr__(self):
        return f"approx({self.expected})"


@contextlib.contextmanager
def _raises(expected, match=None):
    try:
        yield
    except expected as error:
        if match is not None and not re.search(match, str(error)):
            raise AssertionError(f"message {str(error)!r} does not match {match!r}")
        return
    raise AssertionError(f"DID NOT RAISE {expected.__name__}")


class _Mark:
    @staticmethod
    def parametrize(argnames, argvalues):
        names = [n.strip() for n in argnames.split(",")]

        def decorate(func):
            func.__dict__.setdefault("_params", []).append((names, list(argvalues)))
            return func
        return decorate


pytest_stub = types.ModuleType("pytest")
pytest_stub.importorskip = lambda name: __import__(name)
pytest_stub.raises = _raises
pytest_stub.approx = _Approx
pytest_stub.mark = _Mark
sys.modules["pytest"] = pytest_stub

import numpy as np  # noqa: E402


def cases(func):
    groups = getattr(func, "_params", [])
    if not groups:
        yield "", {}
        return
    expanded = []
    for names, values in groups:
        rows = []
        for value in values:
            value = value if len(names) > 1 else (value,)
            rows.append(dict(zip(names, value)))
        expanded.append(rows)
    for combo in itertools.product(*expanded):
        kwargs = {}
        for part in combo:
            kwargs.update(part)
        yield "-".join(str(v) for v in kwargs.values() if isinstance(v, (str, int, float, bool))), kwargs


def main():
    path = Path(sys.argv[1]).resolve()
    pattern = sys.argv[2] if len(sys.argv) > 2 else ""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "_fix_schedule"):
        # the emulator cannot search over binaries: also fix the selector n = 1 - x^m - x^r
        original = module._fix_schedule

        def fix_with_selector(ctx, schedule):
            original(ctx, schedule)
            for (i, l, k), variable in ctx.nb.items():
                variable.LB = variable.UB = 1.0 - schedule["m"][i, l, k] - schedule["r"][i, l, k]
        module._fix_schedule = fix_with_selector
    passed, skipped, failed = 0, 0, []
    for name, func in sorted(vars(module).items()):
        if not name.startswith("test_") or not callable(func) or pattern not in name:
            continue
        for label, kwargs in cases(func):
            try:
                func(**kwargs)
                passed += 1
            except (NotImplementedError, RuntimeError) as error:
                if "fake gurobipy" not in str(error):
                    failed.append(f"{name}[{label}]")
                    print(f"FAIL {name}[{label}]", traceback.format_exc())
                    continue
                skipped += 1
                print(f"SKIP {name}[{label}]: {error}")
            except Exception:
                failed.append(f"{name}[{label}]")
                print(f"FAIL {name}[{label}]\n{traceback.format_exc()}")
    print(f"{passed} passed, {skipped} skipped (need Gurobi), {len(failed)} failed: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
