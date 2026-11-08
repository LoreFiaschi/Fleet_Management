import contextlib
import io
import sys
sys.path.insert(0, r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src")
sys.path.insert(0, r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\sweeps")
from fleet_management import horizon_sweep as h

with contextlib.redirect_stdout(io.StringIO()):
    import sweep_expected as se


def make_row(H2, c, H1=2):
    row = {k: None for k in h.ROW_KEYS}
    if c == "error":
        row.update({"H1": H1, "H2": H2, "status": "error: x"})
        return row
    status, J, gap, requested = c
    row.update({"H1": H1, "H2": H2, "T": H1 + H2, "status": status, "J_op_average": J,
                "objective": J, "mip_gap": gap, "requested_mip_gap": requested})
    row.update({k: None for k in h.GRADIENT_KEYS})
    return row


def run(name, table, **kw):
    stop = kw.get("stop", False)
    rows = []
    reason = None
    for H2 in sorted(table):
        rows.append(make_row(H2, table[H2]))
        h._annotate_gradients(rows, gradient_tolerance=1e-3,
                              maximum_mip_gap_for_stopping=kw.get("threshold"))
        if stop:
            reason = h._gradient_stop_reason(rows, minimum_cases=kw.get("minimum", 3),
                                             flat_gradients_required=kw.get("flat", 2))
            if reason:
                break
    exp_rows, exp_reason, exp_prov, exp_feas = se.sweep(
        table, tol=1e-3, flat=kw.get("flat", 2), minimum=kw.get("minimum", 3),
        threshold=kw.get("threshold"), stop=stop)
    assert [r["H2"] for r in rows] == [e[0] for e in exp_rows], name
    for r, e in zip(rows, exp_rows):
        assert r["cost_gradient_per_H2"] == e[1], (name, r["H2"])
        assert r["relative_cost_gradient_per_H2"] == e[2], (name, r["H2"], r["relative_cost_gradient_per_H2"], e[2])
        assert r["gradient_classification"] == e[3], (name, r["H2"])
        assert r["gradient_stopping_eligible"] == e[4], (name, r["H2"])
    assert (reason or "candidate_range_exhausted") == exp_reason, (name, reason, exp_reason)
    bp, bf = h._select_horizon_cases(rows)
    assert (None if bp is None else (bp["J_op_average"], bp["H2"])) == exp_prov, (name, bp, exp_prov)
    assert (None if bf is None else (bf["J_op_average"], bf["H2"])) == exp_feas, (name, bf, exp_feas)
    print(name, "ok", [r["H2"] for r in rows], reason)


run("T1", se.TABLES["T1"])
run("T2", se.TABLES["T2"], stop=True)
run("T2b", se.TABLES["T2"], stop=True, threshold=0.05)
run("T3", se.TABLES["T3"], stop=True)
run("T3b", se.TABLES["T3"], stop=True, minimum=4)
run("T3c", se.TABLES["T3"], stop=True, flat=3)

# grid ranking
grid = {(2, 2): ("optimal", 1.0, 0.0), (2, 4): ("optimal", 0.9, 0.0),
        (3, 2): ("time_limit", 0.85, 0.05), (3, 4): ("optimal", 0.9, 0.0)}
rows = []
for (H1, H2), (status, J, gap) in grid.items():
    r = make_row(H2, (status, J, gap, 0.1), H1=H1)
    r["projected_evaluation_cost"] = {2: 5.0, 3: 4.0}[H1] + (10 - H1) * J
    rows.append(r)
bp, bf = h._select_grid_cases(rows, "J_op_average")
assert (bp["H1"], bp["H2"]) == (2, 4) and (bf["H1"], bf["H2"]) == (3, 2)
bp, bf = h._select_grid_cases(rows, "projected_evaluation_cost")
assert (bp["H1"], bp["H2"]) == (3, 4) and (bf["H1"], bf["H2"]) == (3, 2)
print("A3 check OK")
