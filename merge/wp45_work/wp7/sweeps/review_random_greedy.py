"""Review: compare greedy_warm_start.build_greedy_warm_start with greedy_ref.greedy (Wiener rule)
on random small fleets of all five models, both repair laws and the three forms.

Run: python review_random_greedy.py [cases]
"""
import sys
import types

SRC = r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src"
sys.path.insert(0, SRC)
sys.modules["gurobipy"] = None
pkg = types.ModuleType("fleet_management")
pkg.__path__ = [SRC + r"\fleet_management"]
sys.modules["fleet_management"] = pkg

import numpy as np  # noqa: E402
import greedy_ref  # noqa: E402
from fleet_management.config import load_config  # noqa: E402
from fleet_management.greedy_warm_start import build_greedy_warm_start  # noqa: E402

MODELS = ["gaussian", "inverse_gaussian", "wiener", "gamma", "rainflow"]
BOUNDS = ["markov", "cantelli", "hoeffding", "bernstein", "chernoff"]


def random_input(rng):
    M = int(rng.integers(1, 3))
    F = M + int(rng.integers(1, 3))
    L = int(rng.integers(1, 4))
    H = [int(rng.integers(1, 4)), int(rng.integers(1, 5))]
    model = [str(rng.choice(MODELS)) for _ in range(L)]
    bound = [str(rng.choice(BOUNDS)) for _ in range(L)]
    q = 0.0625
    mu = (rng.integers(0, 4, size=(L, M)) * q).tolist()
    data = dict(F=F, M=M, L=L, H=H, model=model, bound_method=bound,
                repair_model=str(rng.choice(["ardinf", "ard1"])),
                reliability_impl=[str(rng.choice(["exact", "tangent", "pwl"])) for _ in range(L)],
                pwl_points=int(rng.integers(1, 5)), tau=1.0, epsilon=float(rng.choice([0.1, 0.2])),
                rho=float(rng.choice([0.5, 0.75, 1.0])), mu_0=float(rng.choice([0.0, q, 2 * q])),
                mu=mu, v=(np.asarray(mu) ** 2 / 16).tolist(), sigma=0.05, eta=0.25,
                gamma_beta=8.0, s_chernoff=6.0, cgf=(np.asarray(mu) * 6.0).tolist(),
                support=(np.asarray(mu) * 0.5).tolist(), support_0=0.0, v_0=0.0,
                allow_replacement=bool(rng.random() < 0.7), C_M=1.0, C_R=1.0, C_rep=5.0, C_D=0.0,
                verbose=0)
    if rng.random() < 0.5:
        data["v_max_user"] = 0.01
    return data


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(7)
    compared = rejected = 0
    failures_seen = replacements_seen = 0
    for _ in range(n):
        data = random_input(rng)
        try:
            cfg = load_config(data)
        except ValueError:
            rejected += 1
            continue
        x1, m1, r1, rep1 = greedy_ref.greedy(cfg, joint_models=("wiener",))
        source, report = build_greedy_warm_start(cfg)
        rep2 = report.as_dict()
        same = (np.array_equal(x1, source["x"]) and np.array_equal(m1, source["m"])
                and np.array_equal(r1, source["r"])
                and all(np.isclose(float(rep1[k]), float(rep2[k])) for k in
                        ("repairs", "replacements", "repeatability_failures",
                         "maximum_repeatability_excess", "maximum_proxy_ratio"))
                and rep1["assignment_order"] == rep2["assignment_order"])
        if not same:
            print("DIFFERENT:", data)
            print(" ref:", {k: rep1[k] for k in rep1 if k != "assignment_order"})
            print(" pkg:", {k: rep2[k] for k in rep2 if k != "assignment_order"})
            return
        compared += 1
        failures_seen += rep2["repeatability_failures"] > 0
        replacements_seen += rep2["replacements"] > 0
    print(f"compared {compared}, rejected by load_config {rejected}: all equal; "
          f"{failures_seen} with failures, {replacements_seen} with replacements")


if __name__ == "__main__":
    main()
