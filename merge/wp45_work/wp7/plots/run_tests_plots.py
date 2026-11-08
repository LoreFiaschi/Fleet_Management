"""Run a pytest-style test module without pytest (local check of tests/test_gamma.py).

Usage:  python run_tests_gamma.py <path to test module> [name substring]

As ..\\..\\wp4\\run_tests.py, plus ``pytest.importorskip`` (a missing module skips the
test) and ``pytest.mark.parametrize`` (one run per parameter set). The package
``fleet_management`` is loaded from the worktree without its ``__init__.py``; no
gurobipy is on the path, so the model tests skip.
"""
import contextlib
import importlib
import importlib.util
import inspect
import os
import re
import sys
import tempfile
import traceback
import types
import warnings
from pathlib import Path

SRC = Path(os.environ.get("FM_SRC") or r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")

TMP = Path(__file__).resolve().parent / "tmp"      # tmp_path folders, in the scratchpad

package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package


class _Skipped(Exception):
    pass


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


@contextlib.contextmanager
def _warns(expected, match=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught
    hits = [w for w in caught if issubclass(w.category, expected)
            and (match is None or re.search(match, str(w.message)))]
    if not hits:
        raise AssertionError(f"DID NOT WARN {expected.__name__} matching {match!r}")


def _importorskip(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        raise _Skipped(f"no module {name!r}") from None


def _parametrize(names, values):
    names = [n.strip() for n in names.split(",")]

    def mark(func):
        cases = [v if isinstance(v, tuple) else (v,) for v in values]
        func._cases = getattr(func, "_cases", [{}])
        func._cases = [dict(old, **dict(zip(names, case))) for old in func._cases for case in cases]
        return func
    return mark


pytest_stub = types.ModuleType("pytest")
pytest_stub.raises = _raises
pytest_stub.warns = _warns
pytest_stub.importorskip = _importorskip
pytest_stub.mark = types.SimpleNamespace(parametrize=_parametrize)
sys.modules["pytest"] = pytest_stub


def _load_gamma_from(path):
    """Planner's pre-check only: use the module at ``path`` as degradation_model.gamma."""
    sub = types.ModuleType("fleet_management.degradation_model")
    sub.__path__ = [str(SRC / "fleet_management" / "degradation_model")]
    sys.modules["fleet_management.degradation_model"] = sub
    spec = importlib.util.spec_from_file_location("fleet_management.degradation_model.gamma", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sub.gamma = module


def _load_plotter_from(path):
    """Planner's pre-check only: use the module at ``path`` as fleet_management.utils.plotter."""
    utils = types.ModuleType("fleet_management.utils")
    utils.__path__ = [str(SRC / "fleet_management" / "utils")]
    sys.modules["fleet_management.utils"] = utils
    spec = importlib.util.spec_from_file_location("fleet_management.utils.plotter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    utils.plotter = module


def main():
    args = sys.argv[1:]
    if args and args[0] == "--plotter":
        _load_plotter_from(args[1])
        args = args[2:]
    if args and args[0] == "--gamma":
        _load_gamma_from(args[1])
        args = args[2:]
    path = Path(args[0]).resolve()
    pattern = args[1] if len(args) > 1 else ""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    passed, skipped, failed = 0, 0, []
    for name, func in sorted(vars(module).items()):
        if not name.startswith("test_") or not callable(func) or pattern not in name:
            continue
        for kwargs in getattr(func, "_cases", [{}]):
            label = f"{name}{kwargs if kwargs else ''}"
            if "tmp_path" in inspect.signature(func).parameters:
                TMP.mkdir(exist_ok=True)
                kwargs = dict(kwargs, tmp_path=Path(tempfile.mkdtemp(dir=TMP)))
            try:
                func(**kwargs)
                passed += 1
            except _Skipped:
                skipped += 1
            except Exception:
                failed.append(label)
                print(f"FAIL {label}\n{traceback.format_exc()}")
    print(f"{passed} passed, {skipped} skipped, {len(failed)} failed: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
