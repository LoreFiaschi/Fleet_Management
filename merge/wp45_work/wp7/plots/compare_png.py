"""Compare the PNG files of render_task_a with those of render_out (planner's prototype)."""
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

base = Path(r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\plots")
for name in ("schedule.png", "sweep.png", "sweep_requested.png"):
    a = mpimg.imread(base / "render_out" / name)
    b = mpimg.imread(base / "render_task_a" / name)
    if a.shape != b.shape:
        print(name, "shape differs", a.shape, b.shape)
        continue
    diff = np.abs(a - b)
    print(name, "shape", a.shape, "max abs diff", float(diff.max()), "pixels differing", int(np.count_nonzero(diff.max(axis=2) > 0)))
