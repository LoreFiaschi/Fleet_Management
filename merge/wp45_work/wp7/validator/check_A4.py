"""Check of step A4: validate on the fixture; the file equals the returned dictionary; no print."""
import contextlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path

import yaml

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
FIX = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\results\validator_tests")
TMP = Path(r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\validator\tmp")
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package

from fleet_management.validation.validator import validate  # noqa: E402

TMP.mkdir(exist_ok=True)
tmp = Path(tempfile.mkdtemp(dir=TMP))
out = io.StringIO()
with contextlib.redirect_stdout(out):
    report = validate(FIX / "input.yaml", FIX / "result.yaml", tmp / "v.yaml")
assert out.getvalue() == "", "validate printed"
assert report["passed"] is True
print([c["name"] for c in report["checks"]])
assert len(report["checks"]) == 14
with open(tmp / "v.yaml", encoding="utf-8") as f:
    assert yaml.safe_load(f) == report
report_json = validate(FIX / "input.yaml", FIX / "result.yaml", tmp / "sub" / "r")  # folder created, .yaml added
assert (tmp / "sub" / "r.yaml").exists()
report_json = validate(FIX / "input.yaml", FIX / "result.yaml", tmp / "r.json")
with open(tmp / "r.json", encoding="utf-8") as f:
    assert json.load(f) == report_json
print({k: v for k, v in report.items() if k != "checks"})
print("A4 check passed")
