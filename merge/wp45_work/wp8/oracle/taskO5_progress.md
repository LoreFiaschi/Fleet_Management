Status: done. Local: `test_oracle_unit.py` 144 passed (6 new tests), `test_oracle_bruteforce.py` 85 passed, 42 skipped (Gurobi), 0 failed (runner copy in `scratchpad\wp8\O5\`). G3 is unexecuted until Euler runs the Gurobi test. Next: run the brute-force test on Euler.

# Task O5 progress

- [x] read reference_model.py, spec rows
- [x] least_solution (cell-level Kleene iteration, cached per train row and cell binaries; `verify=True` runs `row_violations` too)
- [x] best_onesided
- [x] unit tests (a) loop rescued, (b) divergence and max_passes, (c) ard1 with anchors (feasible + no solution), (d) best_onesided <= best_reference (2 inputs), tiny input by hand (best 3.0)
- [x] G3 in `test_bruteforce_case`; new `test_bruteforce_case_onesided` (no Gurobi, 42 cases, about 1.5 min for the file)
- [x] local run

## Per-case table (local, from `scratchpad\wp8\O5\table.py`; time of best_reference / best_onesided in s)
Result: all 19 cases that Euler printed with G2 agree with best_onesided (difference below 1e-11). 

```
Be_ard1_tangent        ref=                  12.0 os=                  12.0 exact=False nf=80/88 t_ref=1.4 t_os=0.2
Be_ardinf              ref=                   7.5 os=    2.8999999999992725 exact=True  nf=144/320 t_ref=1.3 t_os=0.2
Be_pwl1                ref=                   7.5 os=    2.8999999999992725 exact=True  nf=144/320 t_ref=1.4 t_os=0.2
Ca_ard1_tangent        ref=                  12.0 os=                  12.0 exact=False nf=80/88 t_ref=2.6 t_os=0.4
Ca_ardinf              ref=                   7.5 os=    2.8999999999992725 exact=True  nf=144/416 t_ref=2.2 t_os=0.3
Ca_penalty_quad        ref=                 6.045 os=                 6.045 exact=False nf=80/112 t_ref=1.5 t_os=0.2
Ch_ard1                ref=                  12.0 os=                  12.0 exact=False nf=80/112 t_ref=1.4 t_os=0.5
Ch_ardinf              ref=                  7.25 os=    2.4499999999992723 exact=True  nf=144/416 t_ref=2.6 t_os=0.5
Ch_indicator           ref=                  7.25 os=    2.4499999999992723 exact=True  nf=144/416 t_ref=1.3 t_os=0.2
G_ard1_anchor          ref=                  12.0 os=                  12.0 exact=False nf=80/88 t_ref=1.3 t_os=0.2
G_ard1_indicator       ref=                  12.0 os=                  12.0 exact=False nf=80/88 t_ref=1.4 t_os=0.2
G_ard1_pwl3            ref=                  12.0 os=                  12.0 exact=False nf=80/88 t_ref=1.5 t_os=0.2
G_ard1_tangent         ref=                  13.3 os=                  13.3 exact=False nf=20/22 t_ref=1.5 t_os=0.2
G_ardinf_exact         ref=                   7.7 os=    3.2999999999992724 exact=True  nf=144/320 t_ref=2.7 t_os=0.4
G_ardinf_pwl1          ref=                   7.7 os=    3.2999999999992724 exact=True  nf=144/320 t_ref=1.9 t_os=0.2
G_ardinf_pwl3          ref=                   7.7 os=    3.2999999999992724 exact=True  nf=144/320 t_ref=1.4 t_os=0.2
G_ardinf_tangent       ref=                  13.4 os=                  13.4 exact=True  nf=16/16 t_ref=1.3 t_os=0.4
G_indicator            ref=                   7.7 os=    3.2999999999992724 exact=True  nf=144/320 t_ref=2.8 t_os=0.4
G_mode_eval            ref=                  24.0 os=                  24.0 exact=False nf=80/88 t_ref=1.5 t_os=0.2
G_mode_opavg           ref=                   3.8 os=    1.5999999999992724 exact=True  nf=144/320 t_ref=1.3 t_os=0.2
G_no_replacement       ref=                   2.5 os=                   2.5 exact=True  nf=32/72 t_ref=0.4 t_os=0.1
G_penalty_inf          ref=                   8.4 os=    3.9999999999992726 exact=True  nf=144/320 t_ref=1.3 t_os=0.2
G_penalty_quad         ref=                  8.19 os=     3.789999999999272 exact=True  nf=144/320 t_ref=1.3 t_os=0.2
G_transitory           ref=                   7.6 os=    3.1999999999994544 exact=True  nf=112/320 t_ref=1.3 t_os=0.3
Ga_ard1                ref=                  12.0 os=                  12.0 exact=False nf=88/112 t_ref=2.2 t_os=0.4
Ga_ardinf              ref=                   7.4 os=    2.6999999999992723 exact=True  nf=144/416 t_ref=2.2 t_os=0.4
Ga_beta0               ref=                   7.4 os=    2.6999999999992723 exact=True  nf=140/416 t_ref=1.3 t_os=0.2
Ho_ard1_pwl3           ref=                  12.0 os=                  12.0 exact=False nf=80/88 t_ref=1.6 t_os=0.2
Ho_ardinf              ref=                   7.5 os=    2.8999999999992725 exact=True  nf=144/320 t_ref=1.5 t_os=0.5
Ho_indicator           ref=                   7.5 os=    2.8999999999992725 exact=True  nf=144/320 t_ref=3.0 t_os=0.4
IG_ard1                ref=                  18.0 os=                  18.0 exact=False nf=8/8 t_ref=1.0 t_os=0.1
IG_ardinf              ref=                  18.0 os=                  18.0 exact=True  nf=8/8 t_ref=1.0 t_os=0.1
IG_indicator           ref=                  18.0 os=                  18.0 exact=True  nf=8/8 t_ref=1.0 t_os=0.1
Mk_ard1                ref=                  12.0 os=                  12.0 exact=False nf=88/88 t_ref=1.9 t_os=0.4
Mk_ardinf              ref=                  7.25 os=    2.4499999999992723 exact=True  nf=144/320 t_ref=2.1 t_os=0.2
W_ard1                 ref=                  12.0 os=                  12.0 exact=False nf=80/88 t_ref=1.4 t_os=0.2
W_ardinf               ref=                  12.0 os=                  12.0 exact=True  nf=80/88 t_ref=1.4 t_os=0.2
W_vmax_user            ref=                  12.0 os=                  12.0 exact=False nf=80/88 t_ref=2.4 t_os=0.4
infeasible_G           ref=                  None os=                  None exact=True  nf=0/0 t_ref=0.0 t_os=0.0
mix_G_IG               ref=                  22.0 os=                  22.0 exact=False nf=20/20 t_ref=5.5 t_os=0.2
mix_Ga_Ho              ref=                  22.0 os=    12.799999999998544 exact=False nf=20/108 t_ref=5.0 t_os=0.4
mix_W_Ch               ref=                  22.0 os=                  22.0 exact=False nf=20/30 t_ref=4.2 t_os=0.2
```

## Summary
- 24 of 42 cases are exact (every cell ardinf; includes bf_infeasible_G); 18 are ard1 or mixed (upper bound only).
- best_onesided < best_reference in 19 cases: exactly the 19 cases of Euler's G2 lines; the values agree with the Euler objectives (2.9, 2.45, 3.3, 1.6, 4.0, 3.79, 3.2, 2.7, 12.8) to 1e-11 (the least solution stops about 7e-13 below the fixed point, tol 1e-12).
- In every other case best_onesided = best_reference in cost (the number of feasible schedules is larger in many of them, see the `nf` column).
- No case where best_onesided is above the Euler objective. For `mix_Ga_Ho` (not exact) best_onesided = 12.8 = Gurobi.
- Cost of the new unit tests: 3 hand cases on Markov cells (loop rescued; divergence; ard1 anchors) derived in comments.
- Edits: `tests/reference_model.py` (appended section O5: `_least_quantity`, `_least_cell`, `least_solution`, `best_onesided`), `tests/test_oracle_unit.py` (imports os, warnings, read_input; 6 tests appended), `tests/test_oracle_bruteforce.py` (import, `_onesided` cache, G3, new test, docstring).
- Note: `least_solution(..., verify=False)` skips the full `row_violations` pass; `best_onesided` uses the cell-level checks (replacement equality, reliability row, domains) and the cell cache.
