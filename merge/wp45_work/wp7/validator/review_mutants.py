"""Review: mutants of validator.py and the dashboards that the tests must kill.

Copies the package into mutR\<n>\src, applies one textual change, runs the two test files
with FM_SRC pointing at the copy, and prints killed or survived.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge")
PKG = WORKTREE / "src" / "fleet_management"
RUNNER = HERE / "run_tests_validator.py"

MUTANTS = [
    ("validation/validator.py", 'z = np.nan_to_num(np.asarray(result["z"], dtype=float), nan=0.0)',
     'z = np.asarray(result["z"], dtype=float)'),
    ("validation/validator.py", "if cfg.C_D > 0 else 0.0", "if cfg.C_D > 0 else 1.0"),
    ("validation/validator.py", "excess = q[:, :, T - 1] - q[:, :, H1 - 1]",
     "excess = q[:, :, T - 1] - q[:, :, H1]"),
    ("validation/validator.py", "np.minimum(np.abs(a), np.abs(a - 1.0))", "np.abs(a - np.round(a))"),
    ("validation/dashboard_inspector.py", "(tau - mu) ** 2 - kappa / 2.0 * R",
     "(tau - mu) ** 2 - kappa * R"),
    ("validation/dashboard_inspector.py", "\"rainflow (markov)\": eps * tau - mu",
     "\"rainflow (markov)\": tau - mu"),
]


def main():
    run = time.strftime("%H%M%S")     # a new folder per run: nothing is deleted
    for n, (rel, old, new) in enumerate(MUTANTS):
        root = HERE / "mutR" / f"{run}_{n}"
        shutil.copytree(PKG, root / "src" / "fleet_management")
        target = root / "src" / "fleet_management" / rel
        text = target.read_text(encoding="utf-8")
        assert text.count(old) == 1, (rel, old)
        target.write_text(text.replace(old, new), encoding="utf-8")
        env = dict(os.environ, FM_SRC=str(root / "src"), PYTHONIOENCODING="utf-8")
        failed = 0
        for test in ("tests/test_validator.py", "tests/test_validator_dashboards.py"):
            out = subprocess.run([sys.executable, str(RUNNER), test], cwd=WORKTREE, env=env,
                                 capture_output=True, text=True).stdout.strip().splitlines()[-1]
            failed += int(out.split(",")[2].split()[0])
        print(n, rel, "killed" if failed else "SURVIVED", f"({failed} failing tests)")


if __name__ == "__main__":
    main()
