"""Check of steps B5 and B6: the Gamma diagnostic on the fixture; figures to tmp/; the pages run."""
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

assert "streamlit" not in sys.modules
from fleet_management.validation import dashboard_gamma as dg  # noqa: E402

FIX = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\results\validator_tests")
state = dg.load_gamma_diagnostic(FIX / "input.yaml", FIX / "result.yaml", 1e-6)
print({k: v for k, v in state["overview"].items()})
print(state["tails"].to_string())
print(state["means"].to_string())
print(sorted(state["replay"]), sorted(state["exact"]))
print(state["exact"]["steps"][0])

cfg = state["cfg"]
TMP = HERE / "tmp"
TMP.mkdir(exist_ok=True)
for name, fig in {"mean": dg.gamma_mean_figure(cfg, state["means"], 1, 1),
                  "tail": dg.gamma_tail_figure(cfg, state["tails"], 1, 1),
                  "tail_heatmap": dg.gamma_tail_heatmap_figure(cfg, state["tails"])}.items():
    fig.savefig(TMP / f"B5_{name}.png", dpi=110)

# the page, with the recording stand-in
import fake_streamlit  # noqa: E402

fake_streamlit.install()
dg.render_gamma_diagnostic(state)
names = [c[0] for c in fake_streamlit.calls]
print(names)
assert fake_streamlit.calls[0][1][0] == ["Overview", "Mean and cap", "Exact tail", "Raw data"]
assert names.count("pyplot") == 3 and names.count("download_button") == 2
assert [c for c in fake_streamlit.calls if c[0] == "selectbox"][1][2].get("key") == "tail_cell"
print("B5/B6 ok")
