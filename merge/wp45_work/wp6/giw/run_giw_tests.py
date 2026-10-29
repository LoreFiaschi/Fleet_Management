"""Run a GIW test module (merge/wp6_plan_giw.md) without pytest and without Gurobi.

Usage:  python run_giw_tests.py <test file> [--proto] [name substring]

* ``gurobipy`` is the linear emulator ``fakegp/`` of this folder: it solves the LP that
  remains when every binary is fixed; a model with quadratic rows is built but not
  solved. A test that needs a real solve is reported as NEEDS GUROBI, not as a failure.
* The package ``fleet_management`` is loaded from the merge worktree without running its
  ``__init__.py``. With ``--proto`` the builder modules of ``proto/degradation_model/``
  (the planner's prototypes) shadow those of the worktree.
* No injection: since Claude's decision 1 (wp6_interface.md) the cap lives in
  ``inverse_gaussian.py`` and ``config.py`` imports it from there.
* ``pytest`` is a stub with ``mark.parametrize``, ``importorskip`` and ``raises``.
"""
import contextlib
import importlib
import importlib.util
import itertools
import os
import re
import sys
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
sys.path.insert(0, str(HERE / "fakegp"))


class _Mark:
    def parametrize(self, names, values, **kwargs):
        def decorate(function):
            function.__dict__.setdefault("_params", []).append((names, list(values)))
            return function
        return decorate


class _Info:
    value = None


@contextlib.contextmanager
def _raises(expected, match=None):
    info = _Info()
    try:
        yield info
    except expected as exc:
        info.value = exc
        if match is not None and not re.search(match, str(exc)):
            raise AssertionError(f"message {str(exc)!r} does not match {match!r}")
        return
    raise AssertionError(f"DID NOT RAISE {expected.__name__}")


def _install_pytest():
    stub = types.ModuleType("pytest")
    stub.mark = _Mark()
    stub.importorskip = importlib.import_module
    stub.raises = _raises
    stub.param = lambda *args, **kwargs: args
    sys.modules["pytest"] = stub


def _install_package(proto: bool):
    package = types.ModuleType("fleet_management")
    package.__path__ = [str(SRC / "fleet_management")]
    sys.modules["fleet_management"] = package
    importlib.import_module("fleet_management.config")
    sub = importlib.import_module("fleet_management.degradation_model")
    if proto:
        folder = os.environ.get("GIW_PROTO_DIR", str(HERE / "proto" / "degradation_model"))
        sub.__path__.insert(0, folder)
        print(f"note: builder modules from {folder}")


def _cases(function):
    params = function.__dict__.get("_params", [])
    if not params:
        return [("", {})]
    axes = []
    for names, values in params:
        names = [n.strip() for n in names.split(",")]
        axes.append([dict(zip(names, v if len(names) > 1 else (v,))) for v in values])
    out = []
    for combo in itertools.product(*axes):
        kwargs = {}
        for part in combo:
            kwargs.update(part)
        out.append((str(list(kwargs.values())), kwargs))
    return out


def _needs_gurobi(exc) -> bool:
    return "fake gurobipy" in str(exc) and isinstance(exc, (NotImplementedError, RuntimeError))


def main():
    args = [a for a in sys.argv[1:] if a != "--proto"]
    proto = "--proto" in sys.argv
    path = Path(args[0]).resolve()
    pattern = args[1] if len(args) > 1 else ""
    _install_pytest()
    _install_package(proto)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    passed, failed, gurobi = 0, [], []
    for name, function in sorted(vars(module).items()):
        if not name.startswith("test_") or not callable(function) or pattern not in name:
            continue
        for label, kwargs in _cases(function):
            case = f"{name}{label}"
            try:
                function(**kwargs)
                passed += 1
                print(f"PASS {case}")
            except Exception as exc:
                if _needs_gurobi(exc):
                    gurobi.append(case)
                    print(f"NEEDS GUROBI {case}")
                else:
                    failed.append(case)
                    print(f"FAIL {case}\n{traceback.format_exc()}")
    print(f"{passed} passed, {len(failed)} failed, {len(gurobi)} need Gurobi")
    if failed:
        print("failed:", failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
