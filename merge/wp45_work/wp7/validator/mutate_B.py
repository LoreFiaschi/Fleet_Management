"""Mutation check of Task B: each mutation of a copy of the source must fail a test.

The copy lives in mutB/src (scratchpad); the real files are not touched. The runner
run_tests_validator.py reads the source from the environment variable FM_SRC.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
REAL = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge")
COPY = HERE / "mutB" / "src"
TESTS = REAL / "tests" / "test_validator_dashboards.py"

MUTATIONS = {
    "inspector": [
        ("side row dropped (Gaussian)",
         "gaussian = np.minimum((tau - mu) ** 2 - kg2 * v, tau - mu)",
         "gaussian = (tau - mu) ** 2 - kg2 * v"),
        ("cap sign", '"inverse_gaussian": cap - mu,', '"inverse_gaussian": mu - cap,'),
        ("chernoff sign", 'np.log(eps) - (K - s * tau)', '(K - s * tau) - np.log(eps)'),
        ("bernstein c", "(tau - c - mu) ** 2 - c ** 2 - 2.0 * kappa * v", "(tau - c - mu) ** 2 - 2.0 * kappa * v"),
        ("holds threshold", 'table["holds"] = table["margin"] >= -tol', 'table["holds"] = table["margin"] >= tol'),
        ("phase", 'np.where(kk < cfg.H1, "transitory", "operating")', 'np.where(kk <= cfg.H1, "transitory", "operating")'),
        ("loop margin sign", "first - last, first - last >= -tol", "last - first, last - first >= -tol"),
        ("model check", "!= cfg.model.astype(str).tolist()", "== cfg.model.astype(str).tolist()"),
        ("mark order", 'if r[i, l, k] > 0.5:\n            marks.append("P")\n        elif m[i, l, k] > 0.5:\n            marks.append("R")',
         'if m[i, l, k] > 0.5:\n            marks.append("R")\n        elif r[i, l, k] > 0.5:\n            marks.append("P")'),
        ("mark depot", 'elif x[i, 0, k] > 0.5:\n            marks.append("D")', 'elif x[i, 0, k] > 0.5:\n            marks.append("I")'),
        ("gamma cap rate", "float(cfg.gamma_beta_0[i, l]),", "float(cfg.gamma_beta_new[i, l]),"),
        ("violation count", "int((~table[\"holds\"]).sum())", "int((table[\"holds\"]).sum())"),
        ("smallest margins", 'table.nsmallest(n, "margin")', 'table.nlargest(n, "margin")'),
        ("timeline start", "np.concatenate([[cfg.mu_0[i, l]], mu[i, l, :]])", "np.concatenate([[0.0], mu[i, l, :]])"),
        ("heatmap boundary", "ax.axvline(cfg.H1 - 0.5, color=\"black\")\n    ax.set_xlabel(\"day\")\n    ax.set_title(\"Reliability margins", "ax.axvline(cfg.H1 + 0.5, color=\"black\")\n    ax.set_xlabel(\"day\")\n    ax.set_title(\"Reliability margins"),
    ],
    "gamma": [
        ("slack sign", "epsilon - bound, epsilon - bound >= -tol", "bound - epsilon, bound - epsilon >= -tol"),
        ("mean margin", "margin = float(caps[i, l] - mu[i, l, k])", "margin = float(mu[i, l, k] - caps[i, l])"),
        ("violations number", 'len(replay["violations"])', 'len(replay["global_violations"])'),
        ("day offset", "rows.append([i + 1, cfg.component_names[l], k + 1,", "rows.append([i + 1, cfg.component_names[l], k,"),
        ("no gamma error", 'raise ValueError("the input has no Gamma cell.")', "pass"),
        ("bang mark", 'ax.text(day - 1, row, "!"', 'ax.text(day - 1, row, "?"'),
    ],
}


def main():
    COPY.parent.mkdir(exist_ok=True)     # each run overwrites the copy in place; nothing is deleted
    shutil.copytree(REAL / "src", COPY, dirs_exist_ok=True)
    survivors = []
    for which, items in MUTATIONS.items():
        name = {"inspector": "dashboard_inspector.py", "gamma": "dashboard_gamma.py"}[which]
        target = COPY / "fleet_management" / "validation" / name
        original = (REAL / "src" / "fleet_management" / "validation" / name).read_text(encoding="utf-8")
        for label, old, new in items:
            if original.count(old) != 1:
                print(f"SKIP {label}: pattern occurs {original.count(old)} times")
                survivors.append(label + " (pattern)")
                continue
            target.write_text(original.replace(old, new), encoding="utf-8")
            env = dict(os.environ, FM_SRC=str(COPY), PYTHONIOENCODING="utf-8")
            run = subprocess.run([sys.executable, str(HERE / "run_tests_validator.py"), str(TESTS)],
                                 capture_output=True, text=True, env=env)
            last = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else run.stderr[-200:]
            killed = run.returncode != 0
            print(("killed  " if killed else "SURVIVED"), label, "|", last[:110])
            if not killed:
                survivors.append(label)
        target.write_text(original, encoding="utf-8")
    print("survivors:", survivors)


if __name__ == "__main__":
    main()
