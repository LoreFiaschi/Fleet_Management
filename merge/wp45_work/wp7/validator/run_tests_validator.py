"""Run a pytest-style test module without pytest (local check of tests/test_validator.py).

Usage:  python run_tests_validator.py <path to test module> [name substring]

As ..\\..\\wp6\\gamma\\run_tests_gamma.py, plus ``pytest.approx`` (scalars and arrays; the
default tolerances of pytest). The package ``fleet_management`` is loaded from the worktree
without its ``__init__.py``; no gurobipy is on the path.
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

import numpy as np

SRC = Path(os.environ.get("FM_SRC") or r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")

TMP = Path(r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\validator\tmp")

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


class _Approx:
    """pytest.approx for scalars and arrays: tolerance max(rel * |expected|, abs)."""

    def __init__(self, expected, rel=None, abs=None):
        self.expected, self.rel, self.abs = expected, rel, abs

    def _tolerance(self, expected):
        absolute = 1e-12 if self.abs is None else self.abs
        if self.rel is None and self.abs is not None:
            return absolute
        relative = (1e-6 if self.rel is None else self.rel) * np.abs(expected)
        return np.maximum(relative, absolute)

    def __eq__(self, other):
        expected = np.asarray(self.expected, dtype=float)
        other = np.asarray(other, dtype=float)
        if expected.shape != other.shape:
            return False
        return bool(np.all(np.abs(other - expected) <= self._tolerance(expected)))

    def __repr__(self):
        return f"approx({self.expected})"


def _importorskip(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        raise _Skipped(f"no module {name!r}") from None


def _parametrize(names, values):
    names = [n.strip() for n in names.split(",")]

    def mark(func):
        cases = [v if isinstance(v, tuple) and len(names) > 1 else (v,) for v in values]
        func._cases = getattr(func, "_cases", [{}])
        func._cases = [dict(old, **dict(zip(names, case))) for old in func._cases for case in cases]
        return func
    return mark


pytest_stub = types.ModuleType("pytest")
pytest_stub.raises = _raises
pytest_stub.warns = _warns
pytest_stub.approx = _Approx
pytest_stub.importorskip = _importorskip
pytest_stub.mark = types.SimpleNamespace(parametrize=_parametrize)
sys.modules["pytest"] = pytest_stub


def main():
    args = sys.argv[1:]
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
