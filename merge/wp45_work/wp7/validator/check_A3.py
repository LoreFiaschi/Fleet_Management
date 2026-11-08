"""Check of step A3: check_result gives the violations of reference_checks.py on the fixture and
on each corruption of the plan (section 3)."""
import contextlib
import io
import sys

HERE = r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\validator"
sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):
    import reference_checks as ref  # builds the package stub and the cases; prints are dropped

from fleet_management.config import load_config, read_input  # noqa: E402
from fleet_management.validation import validator  # noqa: E402

assert "gurobipy" not in sys.modules


def compare(label, cfg, res):
    reference, failed, rec = ref.checks(cfg, res)
    report = validator.check_result(cfg, res)
    got = {c["name"]: c["violation"] for c in report["checks"]}
    assert list(got) == list(reference) or set(got) == set(reference), (label, list(got), list(reference))
    for name in reference:
        assert got[name] == reference[name], (label, name, got[name], reference[name])
    assert report["recomputed_objective"] == rec, (label, report["recomputed_objective"], rec)
    failed_new = {c["name"]: c["violation"] for c in report["checks"] if not c["passed"]}
    assert failed_new == failed, (label, failed_new, failed)
    print(f"{label:18s} failed {failed_new}  recomputed {rec!r}")


compare("valid", ref.cfg, ref.base)
for name, fn in ref.cases.items():
    compare(name, ref.cfg, ref.corrupt(fn))

data = read_input(ref.HERE / "input.yaml")
data["allow_replacement"] = False
compare("allow_replacement", load_config(data), ref.base)
for mode, extra in (("total", {}), ("operating_average", {}), ("evaluation_total", {}), ("total", {"C_D": 0.0})):
    data = read_input(ref.HERE / "input.yaml")
    data["objective_mode"] = mode
    data.update(extra)
    compare(f"mode {mode} {extra}", load_config(data), ref.base)

report = validator.check_result(ref.cfg, ref.base)
print([c["name"] for c in report["checks"]])
print({k: v for k, v in report.items() if k != "checks"})
print("A3 check passed")
