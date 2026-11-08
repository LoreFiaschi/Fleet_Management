"""Build a scratch copy of the worktree src, with the Task 1 prototype for the modules not yet written.

Usage: python task2_overlay.py <dest folder name> <module> [<module> ...]
The modules named (gamma_tail_validator, gamma_stochastic_validator, gamma_tail_bound) are taken
from proto/ (or proto_src/ for gamma_tail_bound), the others from the worktree. The worktree is
only read. Run the tests of an intermediate state with FM_SRC=<dest>/ python run_tests_gamma.py ...
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
GAMMA = "fleet_management/degradation_model/gamma_utils"

dest = HERE / "overlays" / sys.argv[1]
shutil.copytree(SRC, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
for name in sys.argv[2:]:
    if name == "gamma_tail_bound":
        source = HERE / "proto_src" / GAMMA / "gamma_tail_bound.py"
    else:
        source = HERE / "proto" / f"{name}.py"
    shutil.copyfile(source, dest / GAMMA / f"{name}.py")
print("overlay written to", dest)
