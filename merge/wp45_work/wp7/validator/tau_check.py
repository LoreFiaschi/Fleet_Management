import sys, types
from pathlib import Path
SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
p = types.ModuleType("fleet_management"); p.__path__ = [str(SRC / "fleet_management")]; sys.modules["fleet_management"] = p
from fleet_management.config import load_config, read_input
d = read_input(Path(__file__).parent / "input.yaml"); d["tau"] = -1.0
try:
    load_config(d)
except Exception as e:
    print(type(e).__name__, e)
