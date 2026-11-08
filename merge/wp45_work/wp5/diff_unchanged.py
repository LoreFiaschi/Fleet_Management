"""Check that the parts of base.py that the plan leaves alone are unchanged against git HEAD.

Usage: python diff_unchanged.py
Compares: the Gamma block (from its banner to the registration line) and the three
warm-start functions (the last one with the expected removal of the five idle lines).
"""
import pathlib
import subprocess

WT = r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge"
REL = "src/fleet_management/degradation_model/base.py"

old = subprocess.run(["git", "-C", WT, "show", f"HEAD:{REL}"], capture_output=True, text=True, encoding="utf-8").stdout
new = pathlib.Path(WT, REL).read_text(encoding="utf-8")


def block(text, start_marker, end_marker):
    a = text.index(start_marker)
    b = text.index(end_marker, a) + len(end_marker)
    return text[a:b]


gamma_start = "# ###################  GAMMA FINITE-HORIZON TAIL BLOCK  #####################"
gamma_end = 'register_cell_builder("gamma", GammaCellBuilder())'
same_gamma = block(old, gamma_start, gamma_end).replace("\r\n", "\n") == block(new, gamma_start, gamma_end).replace("\r\n", "\n")
print("Gamma block identical:", same_gamma)

ws_start = "def apply_binary_warm_start("
ws_end = "    diagnostics[\"values_copied\"] = copied\n    return diagnostics"
old_ws = block(old, ws_start, ws_end).replace("\r\n", "\n")
new_ws = block(new, ws_start, ws_end).replace("\r\n", "\n")
removed = (
    "                depot_value = float(round(x_old[i, 0, source_k]))\n"
    "                ctx.idle[i, l, k].Start = max(\n"
    "                    0.0, depot_value - m_value - r_value\n"
    "                )\n"
    "                values += 1\n"
)
print("warm-start functions: old == new + the five idle lines:", old_ws.replace(removed, "") == new_ws, "; removed text present in old:", removed in old_ws)
print("apply_performance_params and the progress helpers: ",
      block(old, "def apply_performance_params(", "            model.setParam(key, val)").replace("\r\n", "\n")
      == block(new, "def apply_performance_params(", "            model.setParam(key, val)").replace("\r\n", "\n"))
