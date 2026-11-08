"""Build proto_src/: a copy of the worktree src with the prototype modules of Task 2 on top.

The worktree is only read. The reduced gamma_tail_bound.py keeps ConvolutionTail,
moschopoulos_tail_probability and _validate_parameters of the current file (step C4 without the
vectorisation), so that the test of the removed names can be run.
"""
import ast
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
DEST = HERE / "proto_src"
GAMMA = "fleet_management/degradation_model/gamma_utils"

shutil.copytree(SRC, DEST, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
for name in ("gamma_replay_validator", "gamma_tail_validator", "gamma_stochastic_validator"):
    shutil.copyfile(HERE / "proto" / f"{name}.py", DEST / GAMMA / f"{name}.py")

source = (SRC / GAMMA / "gamma_tail_bound.py").read_text(encoding="utf-8")
tree = ast.parse(source)
keep = {"ConvolutionTail", "moschopoulos_tail_probability", "_validate_parameters"}
parts = ['"""Reduced copy (scratch): the series only."""\n\nfrom __future__ import annotations\n\n'
         "from dataclasses import dataclass\n\nimport numpy as np\nfrom numpy.typing import ArrayLike, NDArray\n"
         "from scipy.stats import gamma\n\nFloatArray = NDArray[np.float64]\n\n"]
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in keep:
        start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        parts.append("\n".join(source.splitlines()[start - 1:node.end_lineno]) + "\n\n\n")
(DEST / GAMMA / "gamma_tail_bound.py").write_text("".join(parts), encoding="utf-8")
print("overlay written to", DEST)
