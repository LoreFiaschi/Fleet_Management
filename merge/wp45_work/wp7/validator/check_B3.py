"""Check of step B3: the three figures; saves PNG files in tmp/."""
import sys
import types
from pathlib import Path

SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
package = types.ModuleType("fleet_management")
package.__path__ = [str(SRC / "fleet_management")]
sys.modules["fleet_management"] = package

from matplotlib.figure import Figure  # noqa: E402

from fleet_management.validation.dashboard_inspector import (  # noqa: E402
    component_figure, load_inspector, margin_heatmap_figure, timeline_figure)

TMP = Path(__file__).parent / "tmp"
TMP.mkdir(exist_ok=True)
FIX = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\results\validator_tests")
state = load_inspector(FIX / "input.yaml", FIX / "result.yaml", 1e-6)
cfg, result = state["cfg"], state["result"]
figs = {"timeline_t3": timeline_figure(cfg, result, 2),
        "timeline_t1": timeline_figure(cfg, result, 0),
        "heatmap": margin_heatmap_figure(cfg, state["margins"]),
        "component": component_figure(cfg, result, 0)}
for name, fig in figs.items():
    assert isinstance(fig, Figure)
    fig.savefig(TMP / f"B3_{name}.png", dpi=110)
print(len(figs["timeline_t3"].axes[0].lines), len(figs["heatmap"].axes[0].images))
print("B3 ok")
