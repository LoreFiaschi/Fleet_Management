"""Run a test module as pytest would on a machine without gurobipy.

Usage: python run_without_gurobi.py <test file>
``gurobipy`` is blocked (``sys.modules['gurobipy'] = None``); ``pytest.importorskip``
raises a skip; the real package is imported from the worktree (its ``__init__.py`` runs).
Prints one line per case: PASS, SKIP or FAIL.
"""
import contextlib
import importlib
import importlib.util
import itertools
import re
import sys
import traceback
import types
from pathlib import Path

sys.modules["gurobipy"] = None
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")


class Skipped(Exception):
    pass


def importorskip(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        raise Skipped(name) from None


class _Mark:
    def parametrize(self, names, values, **kwargs):
        def decorate(function):
            function.__dict__.setdefault("_params", []).append((names, list(values)))
            return function
        return decorate


@contextlib.contextmanager
def _raises(expected, match=None):
    try:
        yield
    except expected as exc:
        if match is not None and not re.search(match, str(exc)):
            raise AssertionError(f"message {str(exc)!r} does not match {match!r}")
        return
    raise AssertionError(f"DID NOT RAISE {expected.__name__}")


stub = types.ModuleType("pytest")
stub.mark, stub.importorskip, stub.raises = _Mark(), importorskip, _raises
sys.modules["pytest"] = stub


def cases(function):
    axes = []
    for names, values in function.__dict__.get("_params", []):
        names = [n.strip() for n in names.split(",")]
        axes.append([dict(zip(names, v if len(names) > 1 else (v,))) for v in values])
    out = []
    for combo in itertools.product(*axes):
        kwargs = {}
        for part in combo:
            kwargs.update(part)
        out.append(kwargs)
    return out or [{}]


path = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location(path.stem, path)
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
except Skipped as exc:
    print(f"whole module skipped (importorskip {exc})")
    sys.exit(0)
counts = {"PASS": 0, "SKIP": 0, "FAIL": 0}
for name, function in sorted(vars(module).items()):
    if not name.startswith("test_") or not callable(function):
        continue
    for kwargs in cases(function):
        try:
            function(**kwargs)
            status = "PASS"
        except Skipped:
            status = "SKIP"
        except Exception:
            status = "FAIL"
            traceback.print_exc()
        counts[status] += 1
        print(status, name, list(kwargs.values()))
print(counts, "gurobipy loaded:", sys.modules.get("gurobipy") is not None)
