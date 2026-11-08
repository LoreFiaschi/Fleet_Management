"""Check of step B4: the page runs against a recording stand-in of streamlit."""
import ast
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package

# 1. the module imports with streamlit absent
assert "streamlit" not in sys.modules
from fleet_management.validation import dashboard_inspector as di  # noqa: E402

tree = ast.parse((SRC / "fleet_management/validation/dashboard_inspector.py").read_text("utf-8"))
top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
assert not any("streamlit" in ast.dump(n) for n in top)

# 2. the page runs with the stand-in
import fake_streamlit  # noqa: E402

fake_streamlit.install()
FIX = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\results\validator_tests")
state = di.load_inspector(FIX / "input.yaml", FIX / "result.yaml", 1e-6)
di.render_result_inspector(state)
names = [c[0] for c in fake_streamlit.calls]
print(names)
labels = fake_streamlit.calls[0][1][0]
print(labels)
assert labels == ["Overview", "Reliability", "Repeatability", "Train timeline", "Heatmap",
                  "Component comparison", "Validation", "Raw data"]
assert names.count("pyplot") == 3 and names.count("download_button") == 1
assert ("success", ("Every loop row holds.",), {}) in fake_streamlit.calls
assert ("success", ("All checks passed.",), {}) in fake_streamlit.calls
print("B4 ok")
