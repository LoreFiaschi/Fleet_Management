"""Test T1 of tests/test_base.py, run through the gurobipy stub (no pytest, no Gurobi).

Usage: python t1_check.py
Repeats the body of test_unknown_model_has_no_cell_builder, with the new first line.
"""
import re
import sys

sys.path.insert(0, r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management"
                   r"\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp5\stubs")
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")

from fleet_management.degradation_model import base

print("registered before the load:", tuple(base.CELL_BUILDERS))
base._load_builders(["gamma"])
print("registered after the load:", tuple(base.CELL_BUILDERS))
try:
    base.get_cell_builder("no_such_model")
except NotImplementedError as error:
    print("message:", error)
    print("T1:", "pass" if re.search("gamma", str(error)) else "FAIL")
else:
    print("T1: FAIL (no error)")
