"""Extend the WP7 copy of the linear emulator for the warm-start tests (run once).

Adds: Params.NonConvex = -1 (Gurobi's default); Start and VarHintVal default to GRB.UNDEFINED, as in Gurobi; Model.copy (a deep
copy, variables in the same order); computeIIS (marks every row: the emulator cannot find a
minimal subset); IISConstr, GenConstrName and IISGenConstr on the row objects;
feasRelaxS raises NotImplementedError ("fake gurobipy"), so the runner reports SKIP.
"""
from pathlib import Path

path = Path(__file__).resolve().parent / "fakegp" / "gurobipy" / "__init__.py"
s = path.read_text(encoding="utf-8")

edits = [
    ("import itertools\n", "import copy as _copy\nimport itertools\n"),
    ("        self.X = None\n        self.Start = None\n",
     "        self.X = None\n        self.Start = 1e101          # GRB.UNDEFINED\n"
     "        self.VarHintVal = 1e101     # GRB.UNDEFINED\n"),
    ("        self.Params.Seed = 0\n",
     "        self.Params.Seed = 0\n        self.Params.NonConvex = -1   # Gurobi's default\n"),
    ("                if v.Start is None:\n",
     "                if v.Start is None or v.Start >= 1e100:\n"),
    ("            out.append(type(\"Constr\", (), {\"ConstrName\": name, \"RHS\": -tc.expr.const,\n"
     "                                           \"Sense\": sense})())\n",
     "            out.append(type(\"Constr\", (), {\"ConstrName\": name, \"RHS\": -tc.expr.const,\n"
     "                                           \"Sense\": sense,\n"
     "                                           \"IISConstr\": getattr(self, \"_iis\", False)})())\n"),
    ("        return [type(\"GenConstr\", (), {\"GenConstrType\": GRB.GENCONSTR_INDICATOR})() for _ in self.indicators]\n",
     "        return [type(\"GenConstr\", (), {\"GenConstrType\": GRB.GENCONSTR_INDICATOR,\n"
     "                                       \"GenConstrName\": name,\n"
     "                                       \"IISGenConstr\": getattr(self, \"_iis\", False)})()\n"
     "                for _, _, _, name in self.indicators]\n\n"
     "    def getConstrByName(self, name):\n"
     "        \"\"\"The linear row of that name, or None (as Gurobi).\"\"\"\n"
     "        for row in self.getConstrs():\n"
     "            if row.ConstrName == name:\n"
     "                return row\n"
     "        return None\n\n"
     "    def copy(self):\n"
     "        \"\"\"A deep copy; getVars() of the copy lists the variables in the same order.\"\"\"\n"
     "        return _copy.deepcopy(self)\n\n"
     "    def computeIIS(self):\n"
     "        \"\"\"Marks every row and indicator row (not a minimal subset).\"\"\"\n"
     "        self._iis = True\n\n"
     "    def feasRelaxS(self, *args):\n"
     "        raise NotImplementedError(\"fake gurobipy: feasRelaxS needs Gurobi\")\n"),
]
for old, new in edits:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
s = s.replace('"""A tiny linear emulator', '"""WP7 copy (patch_fakegp.py). A tiny linear emulator', 1)
path.write_text(s, encoding="utf-8", newline="\n")
print("patched", path)
