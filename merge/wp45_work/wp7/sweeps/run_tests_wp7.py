"""Run a pytest-style test module of WP7 without pytest.

Usage: python run_tests_wp7.py <test file> [name substring] [--fake]

Without --fake: the package of the worktree, no gurobipy (``pytest.importorskip("gurobipy")``
skips the test). With --fake: the linear emulator of ``fakegp`` (this folder) stands in for
gurobipy; a test that needs a real Gurobi raises NotImplementedError or RuntimeError with
"fake gurobipy" and is reported as SKIP.

Supported: ``pytest.raises(match=)``, ``pytest.approx``, ``pytest.skip``,
``pytest.importorskip``, ``pytest.mark.parametrize`` (stacked = product),
``pytest.mark.skipif``; the fixtures ``tmp_path``, ``monkeypatch`` (setattr, setitem,
delattr; undone after the test) and ``caplog`` (``records``, ``text``, ``set_level``).
"""
import contextlib
import importlib.util
import inspect
import itertools
import logging
import re
import sys
import tempfile
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
FAKE = "--fake" in sys.argv
if FAKE:
    sys.argv.remove("--fake")
    sys.path.insert(0, str(HERE / "fakegp"))
sys.path.insert(0, str(SRC))
if "--no-init" in sys.argv:
    # load the subpackages without running fleet_management/__init__.py (useful while another
    # package that __init__ imports is being rewritten)
    sys.argv.remove("--no-init")
    _package = types.ModuleType("fleet_management")
    _package.__path__ = [str(SRC / "fleet_management")]
    sys.modules["fleet_management"] = _package
TMP = HERE / "tmp"
TMP.mkdir(exist_ok=True)


class Skip(Exception):
    pass


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
    raise AssertionError(f"DID NOT RAISE {expected}")


def _importorskip(name):
    try:
        return __import__(name)
    except ModuleNotFoundError as error:
        raise Skip(f"no module {name}") from error


def _skip(reason=""):
    raise Skip(reason)


class _Mark:
    @staticmethod
    def skipif(condition, reason=""):
        def decorate(func):
            if condition:
                func._skip = reason
            return func
        return decorate

    @staticmethod
    def parametrize(argnames, argvalues):
        names = [n.strip() for n in argnames.split(",")] if isinstance(argnames, str) else list(argnames)

        def decorate(func):
            func.__dict__.setdefault("_params", []).append((names, list(argvalues)))
            return func
        return decorate


pytest_stub = types.ModuleType("pytest")
pytest_stub.importorskip = _importorskip
pytest_stub.raises = _raises
pytest_stub.approx = _Approx
pytest_stub.skip = _skip
pytest_stub.param = lambda *values, **options: values      # id and marks are ignored
pytest_stub.mark = _Mark
sys.modules["pytest"] = pytest_stub


class _MonkeyPatch:
    _missing = object()

    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value, raising=True):
        old = getattr(target, name, self._missing)
        if old is self._missing and raising:
            raise AttributeError(f"{target!r} has no attribute {name!r}")
        self._undo.append(("attr", target, name, old))
        setattr(target, name, value)

    def delattr(self, target, name):
        self._undo.append(("attr", target, name, getattr(target, name)))
        delattr(target, name)

    def setitem(self, mapping, key, value):
        self._undo.append(("item", mapping, key, mapping.get(key, self._missing)))
        mapping[key] = value

    def undo(self):
        for kind, target, name, old in reversed(self._undo):
            if kind == "attr":
                if old is self._missing:
                    delattr(target, name)
                else:
                    setattr(target, name, old)
            elif old is self._missing:
                target.pop(name, None)
            else:
                target[name] = old
        self._undo.clear()


class _CapLog(logging.Handler):
    def __init__(self):
        super().__init__(level=0)
        self.records = []
        self._levels = []

    def emit(self, record):
        self.records.append(record)

    @property
    def text(self):
        return "\n".join(r.getMessage() for r in self.records)

    def set_level(self, level, logger=None):
        target = logging.getLogger(logger)
        self._levels.append((target, target.level))
        target.setLevel(level)

    def close(self):
        for target, level in reversed(self._levels):
            target.setLevel(level)
        super().close()


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
        yield "-".join(str(v) for v in kwargs.values()
                       if isinstance(v, (str, int, float, bool))), kwargs


def run_one(func, kwargs):
    params = inspect.signature(func).parameters
    patch, caplog = _MonkeyPatch(), None
    if "tmp_path" in params:
        kwargs = dict(kwargs, tmp_path=Path(tempfile.mkdtemp(dir=TMP)))
    if "monkeypatch" in params:
        kwargs = dict(kwargs, monkeypatch=patch)
    if "caplog" in params:
        caplog = _CapLog()
        logging.getLogger().addHandler(caplog)
        kwargs = dict(kwargs, caplog=caplog)
    try:
        func(**kwargs)
    finally:
        patch.undo()
        if caplog is not None:
            logging.getLogger().removeHandler(caplog)
            caplog.close()


def main():
    path = Path(sys.argv[1]).resolve()
    pattern = sys.argv[2] if len(sys.argv) > 2 else ""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Skip as reason:
        print(f"module skipped: {reason}")
        return
    passed, skipped, failed = 0, 0, []
    for name, func in sorted(vars(module).items()):
        if not name.startswith("test_") or not callable(func) or pattern not in name:
            continue
        for label, kwargs in cases(func):
            if getattr(func, "_skip", None):
                skipped += 1
                print(f"SKIP {name}[{label}]: {func._skip}")
                continue
            try:
                run_one(func, kwargs)
                passed += 1
            except Skip as reason:
                skipped += 1
                print(f"SKIP {name}[{label}]: {reason}")
            except (NotImplementedError, RuntimeError) as error:
                if "fake gurobipy" in str(error):
                    skipped += 1
                    print(f"SKIP {name}[{label}]: {error}")
                    continue
                failed.append(f"{name}[{label}]")
                print(f"FAIL {name}[{label}]\n{traceback.format_exc()}")
            except Exception:
                failed.append(f"{name}[{label}]")
                print(f"FAIL {name}[{label}]\n{traceback.format_exc()}")
    print(f"{passed} passed, {skipped} skipped, {len(failed)} failed: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
