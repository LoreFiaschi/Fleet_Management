"""(GIW copy: config.py from patched/ first.) Run a pytest-style test module without pytest and without gurobipy.

Usage:  python run_tests.py <path to test module> [name substring]
"""
import contextlib
import importlib.util
import inspect
import re
import sys
import tempfile
import traceback
import types
import warnings
from pathlib import Path

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
SCRATCH = Path(__file__).resolve().parent / "tmp"

# Package stub: import fleet_management.<module> without running __init__.py,
# which imports the solver stack and gurobipy.
package = types.ModuleType("fleet_management")
package.__path__ = [str(Path(__file__).resolve().parent / "patched" / "fleet_management"), str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package


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


pytest_stub = types.ModuleType("pytest")
pytest_stub.raises = _raises
pytest_stub.warns = _warns
sys.modules["pytest"] = pytest_stub


def main():
    path = Path(sys.argv[1]).resolve()
    pattern = sys.argv[2] if len(sys.argv) > 2 else ""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    SCRATCH.mkdir(exist_ok=True)
    passed, failed = 0, []
    for name, func in sorted(vars(module).items()):
        if not name.startswith("test_") or not callable(func) or pattern not in name:
            continue
        kwargs = {}
        if "tmp_path" in inspect.signature(func).parameters:
            kwargs["tmp_path"] = Path(tempfile.mkdtemp(dir=SCRATCH))
        try:
            func(**kwargs)
            passed += 1
        except Exception:
            failed.append(name)
            print(f"FAIL {name}\n{traceback.format_exc()}")
    print(f"{passed} passed, {len(failed)} failed: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
