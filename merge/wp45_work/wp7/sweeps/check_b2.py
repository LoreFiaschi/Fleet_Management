"""Check of step B2: quantities and rows of every cell against the reference greedy_ref.py."""
import math
import sys

SP = r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\sweeps"
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
sys.path.insert(0, SP)

import numpy as np  # noqa: E402
from fleet_management import greedy_warm_start as g  # noqa: E402
from fleet_management.config import load_config  # noqa: E402
from fleet_management.degradation_model.inverse_gaussian import ig_mean_cap  # noqa: E402
from fleet_management.degradation_model.gamma import mean_cap  # noqa: E402
import greedy_ref  # noqa: E402

assert "gurobipy" not in sys.modules

cfg = load_config(dict(greedy_ref.CASES["G_linear"], verbose=0))
profiles = g._profiles(cfg)
caps = [g._cell_rows(cfg, 0, l, profiles) for l in range(4)]
print(caps)
assert caps[0]["kind"] == "cap" and math.isclose(caps[0]["cap"], ig_mean_cap(0.25, 1.0, 0.5))
assert math.isclose(caps[1]["cap"], mean_cap(8.0, 1.0, 0.5))
assert caps[2]["cap"] == 0.5 and caps[2]["q"] == "mu"
assert caps[3]["q"] == "K" and math.isclose(caps[3]["cap"], 4.0 + math.log(0.5))

n = 0
for name, data in greedy_ref.CASES.items():
    cfg = load_config(dict(data, verbose=0))
    profiles = g._profiles(cfg)
    for i in range(cfg.F):
        for l in range(cfg.L):
            Q = greedy_ref.quantities(cfg, i, l)
            mine = g._cell_quantities(cfg, i, l, profiles)
            assert [a["q"] for a in mine] == [b[0] for b in Q], (name, i, l)
            for a, b in zip(mine, Q):
                q, start, new, anchored, anchor, factor, inc, rng = b
                assert a["start"] == float(start) and a["new"] == float(new)
                assert a["anchored"] == bool(anchored)
                assert (math.isnan(a["anchor"]) and math.isnan(float(anchor))) or a["anchor"] == float(anchor)
                assert math.isclose(a["factor"], factor) and a["range"] == float(rng)
                assert a["increment"].shape == (cfg.M, cfg.T)
                np.testing.assert_array_equal(a["increment"], np.asarray(inc, float))
            row, vmax = greedy_ref.rows(cfg, i, l)
            mine_row = g._cell_rows(cfg, i, l, profiles)
            if row[0] == "cap":
                assert mine_row["kind"] == "cap" and mine_row["q"] == row[1]
                assert mine_row["cap"] == row[2]
            else:
                assert mine_row["kind"] == "quadratic" and mine_row["q"] == row[1]
                assert (mine_row["c2"], mine_row["c1"], mine_row["side"]) == tuple(row[2:])
            assert (math.isnan(vmax) and math.isnan(mine_row["v_max_user"])) or vmax == mine_row["v_max_user"]
            # the cap function against the reference over a grid of means
            if row[0] == "quad":
                impl = str(cfg.reliability_impl[i, l])
                tref = float(cfg.tangent_ref[i, l]) if impl == "tangent" else 0.5
                mus = np.linspace(-0.1, 1.3, 29) * float(cfg.tau[i, l])
                mine_cap = g._row_cap(mus, float(cfg.tau[i, l]), row[2], row[3], impl,
                                      float(cfg.tangent_ref[i, l]), cfg.pwl_points)
                ref_cap = [greedy_ref.cap_of(v, float(cfg.tau[i, l]), row[2], row[3], impl, tref, cfg.pwl_points)
                           for v in mus]
                np.testing.assert_allclose(mine_cap, ref_cap, rtol=1e-12, atol=1e-14)
            n += 1
print("cells compared:", n)
print("B2 ok")
