"""Emulate the licensed machine: a stub gurobipy, then import the package as pytest would."""
import sys, traceback, types
from unittest import mock
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
gp = mock.MagicMock(name="gurobipy")
sys.modules["gurobipy"] = gp
sys.modules["gurobipy.GRB"] = gp.GRB
try:
    import fleet_management
    print("import fleet_management: OK")
except Exception:
    traceback.print_exc()
