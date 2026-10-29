"""Write review_run.py: run_local.py plus pytest.mark.skipif and a --real mode (no emulator)."""
src = open("run_local.py", encoding="utf-8").read()


def swap(old, new):
    global src
    assert src.count(old) == 1, old[:60]
    src = src.replace(old, new)


swap('SCRATCH = Path(__file__).resolve().parents[2]\n'
     'sys.path.insert(0, str(SCRATCH / "wp5" / "fakegp"))\n'
     'for name, path in (',
     'SCRATCH = Path(__file__).resolve().parents[2]\n'
     'REAL = "--real" in sys.argv          # real package, no gurobipy: the tests without a model\n'
     'if REAL:\n'
     '    sys.argv.remove("--real")\n'
     '    sys.path.insert(0, str(SRC))\n'
     'else:\n'
     '    sys.path.insert(0, str(SCRATCH / "wp5" / "fakegp"))\n'
     'for name, path in [] if REAL else (')
swap('import gurobipy as gp  # noqa: E402  (the emulator)\n',
     'try:\n'
     '    import gurobipy as gp  # noqa: E402  (the emulator)\n'
     'except ModuleNotFoundError:\n'
     '    gp = None\n')
swap('gp.Model.__init__ = _init\n'
     'gp.Model.getConstrs = _get_constrs\n'
     'gp.Model.getConstrByName = _get_constr_by_name\n',
     'if gp is not None:\n'
     '    _orig_init = gp.Model.__init__\n'
     '    gp.Model.__init__ = _init\n'
     '    gp.Model.getConstrs = _get_constrs\n'
     '    gp.Model.getConstrByName = _get_constr_by_name\n')
swap('_orig_init = gp.Model.__init__\n\n\n', '_orig_init = None\n\n\n')
swap('class _Mark:\n    @staticmethod\n',
     'class _Mark:\n'
     '    @staticmethod\n'
     '    def skipif(condition, reason=""):\n'
     '        def decorate(func):\n'
     '            if condition:\n'
     '                func._skip = reason\n'
     '            return func\n'
     '        return decorate\n\n'
     '    @staticmethod\n')
swap('        for label, kwargs in cases(func):\n            try:\n',
     '        for label, kwargs in cases(func):\n'
     '            if getattr(func, "_skip", None):\n'
     '                skipped += 1\n'
     '                print(f"SKIP {name}[{label}]: {func._skip}")\n'
     '                continue\n'
     '            try:\n')
swap('    if hasattr(module, "_fix_schedule"):', '    if hasattr(module, "_fix_schedule") and not REAL:')
open("review_run.py", "w", encoding="utf-8").write(src)
print("written")
