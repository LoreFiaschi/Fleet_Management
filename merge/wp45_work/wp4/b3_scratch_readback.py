"""Scratch check: an HDF5 result file read back through config.read_input."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\tests")
sys.path.insert(0, r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp4")
import run_tests  # noqa: F401  (registers the stub package)

from fleet_management import config, solver  # noqa: E402
import test_io  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "r.h5"
    solver._save_results(test_io._result(), p)
    data = config.read_input(p)
    print(sorted(data))
    print(data["H"], type(data["F"]).__name__, data["bound_method"], data["repair_model"])
    print(data["v"][0][1], data["status"], data["objective"])
