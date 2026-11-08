"""Expected values of tests/test_horizon_sweep.py, from the spec's formulas (Eq. d_gradient,
classes, eligibility, stop rule, ranking, tie rule, P_E). Independent of horizon_sweep.py.

A case is (status, J_op_average, mip_gap, requested_mip_gap) or "error".
"""
import sys

EPS = sys.float_info.epsilon
NO_SOLUTION = ("infeasible", "unbounded", "inf_or_unbounded")

TABLES = {
    # ranking, errors, warm-start sources; no stop
    "T1": {2: ("optimal", 1.0, 0.0, 0.1), 3: ("optimal", 0.9, 0.0, 0.1),
           4: ("time_limit", 0.8, 0.05, 0.1), 5: "error", 6: ("infeasible", None, None, 0.1),
           7: ("optimal", 0.9, 0.0, 0.1)},
    # cost_increase, with uneven steps of H2
    "T2": {2: ("optimal", 1.0, 0.0, 0.1), 3: ("optimal", 0.9, 0.0, 0.1),
           5: ("optimal", 0.85, 0.0, 0.1), 6: ("optimal", 0.95, 0.08, 0.1),
           8: ("optimal", 0.7, 0.0, 0.1)},
    # flat_gradient
    "T3": {2: ("optimal", 1.0, 0.0, 0.1), 3: ("optimal", 1.0005, 0.0, 0.1),
           4: ("optimal", 1.0004, 0.0, 0.1), 5: ("optimal", 1.0, 0.0, 0.1),
           6: ("optimal", 0.5, 0.0, 0.1)},
}


def feasible(c):
    return c != "error" and c[1] is not None and c[0] not in NO_SOLUTION


def qualifies(c, threshold):
    if c == "error" or c[2] is None:
        return False
    limit = c[3] if threshold is None else threshold
    return c[2] <= limit


def sweep(table, *, tol=1e-3, flat=2, minimum=3, threshold=None, stop=False):
    rows, reason = [], None
    hs = sorted(table)
    for n, h in enumerate(hs, start=1):
        c = table[h]
        g = rel = None
        cls, elig = "unavailable", False
        if n > 1 and feasible(table[hs[n - 2]]) and feasible(c):
            p = table[hs[n - 2]]
            g = (c[1] - p[1]) / (h - hs[n - 2])
            rel = g / max(abs(p[1]), EPS)
            cls = "increase" if rel > tol else "decrease" if rel < -tol else "flat"
            elig = qualifies(p, threshold) and qualifies(c, threshold)
        rows.append((h, g, rel, cls, elig))
        if stop and n >= minimum and elig:
            if cls == "increase":
                reason = "cost_increase"
            elif len(rows) >= flat and all(r[3] == "flat" and r[4] for r in rows[-flat:]):
                reason = "flat_gradient"
            if reason:
                break
    feas = [(table[h][1], h) for h, *_ in rows if feasible(table[h])]
    prov = [(table[h][1], h) for h, *_ in rows if feasible(table[h]) and table[h][0] == "optimal"]
    return rows, reason or "candidate_range_exhausted", min(prov, default=None), min(feas, default=None)


def show(name, **kw):
    rows, reason, prov, feas = sweep(TABLES[name[:2]], **kw)
    print(f"--- {name} {kw}")
    for h, g, rel, cls, elig in rows:
        print(f"  H2={h}: grad={g!r} rel={rel!r} class={cls} eligible={elig}")
    print(f"  reason={reason} evaluated={[r[0] for r in rows]} best_proven={prov} best_feasible={feas}")


show("T1")
show("T2", stop=True)
show("T2b", stop=True, threshold=0.05)
show("T3", stop=True)
show("T3b", stop=True, minimum=4)
show("T3c", stop=True, flat=3)

# Grid: J_init by H1, (status, J_op_average, gap) by (H1, H2); E = 10.
J_INIT = {2: 5.0, 3: 4.0}
GRID = {(2, 2): ("optimal", 1.0, 0.0), (2, 4): ("optimal", 0.9, 0.0),
        (3, 2): ("time_limit", 0.85, 0.05), (3, 4): ("optimal", 0.9, 0.0)}
for E in (None, 10):
    value = {k: (v[1] if E is None else J_INIT[k[0]] + (E - k[0]) * v[1]) for k, v in GRID.items()}
    feas = min((value[k], k[0], k[1]) for k in GRID)
    prov = min((value[k], k[0], k[1]) for k, v in GRID.items() if v[0] == "optimal")
    print(f"--- grid E={E}: values={value} best_proven={prov} best_feasible={feas}")

# formulation_growth of T1: rows with counts are H2 = 2, 3, 4, 6, 7 (5 is an error row);
# the stub gives variables = 10*T, ..., so added_variables = 10*(T_last - T_first).
print("--- growth T1: from_H2=2 to_H2=7 added_time_steps=5 added_variables=50 per_step=10.0")
