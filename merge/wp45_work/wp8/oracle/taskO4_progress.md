Status: done. Local: `test_oracle_bruteforce.py` 43 passed (inputs documented + 42 non-triviality), 42 skipped (Gurobi, one per case), 0 failed. Next: Euler run (Claude submits).

# Task O4 progress

- [x] read interface, plan, O1/O2 progress, oracle, spec (inputs 349-602, schema 2712-2830, result 1714-1784)
- [x] 42 inputs `tests/data/oracle/bf_*.yaml` (generated from the scratchpad tool `wp8\O4\gen_cases.py`; the YAML files are the deliverable); all pass `load_config`
- [x] `tests/test_oracle_bruteforce.py`: `test_bruteforce_inputs_are_documented`, `test_bruteforce_case_nontrivial[case]` (no Gurobi), `test_bruteforce_case[case]` (Gurobi, `importorskip` per test)
- [x] local run (runner copy `wp8\O4\run_tests_gamma.py`, with an ignored `ids=` argument added to its `parametrize` stub)
- [x] dry run of checks B to F on a fake result built from each best reference schedule (`wp8\O4\fake_solve.py`): 0 failures (checks catch no false positive on exact data; the Gurobi path itself is untested here)

## Cases (schedules, feasible, best cost, local time of `best_reference`)
All: F=2, M=1, `H=[2,2]` (T=4) unless stated; L=1 unless stated; replacement allowed unless stated. Every best schedule has an intervention (2 to 4).

| Case | schedules | feasible | best cost | time (s) |
|---|---|---|---|---|
| G_ardinf_exact | 4096 | 144 | 7.7 | 3.5 |
| G_ard1_anchor (anchors 0.05, 0.0002 below initial) | 4096 | 80 | 12.0 | 3.4 |
| G_ardinf_tangent (ref 0.2) | 4096 | 16 | 13.4 | 3.1 |
| G_ard1_tangent (ref 0.3) | 4096 | 20 | 13.3 | 2.8 |
| G_ardinf_pwl1 | 4096 | 144 | 7.7 | 1.8 |
| G_ardinf_pwl3 | 4096 | 144 | 7.7 | 1.6 |
| G_ard1_pwl3 | 4096 | 80 | 12.0 | 1.9 |
| G_indicator | 4096 | 144 | 7.7 | 4.0 |
| G_ard1_indicator | 4096 | 80 | 12.0 | 2.2 |
| IG_ardinf | 4096 | 8 | 18.0 | 1.2 |
| IG_ard1 (anchor 0.04) | 4096 | 8 | 18.0 | 1.3 |
| IG_indicator | 4096 | 8 | 18.0 | 2.5 |
| W_ardinf (loop forces replacement) | 4096 | 80 | 12.0 | 4.0 |
| W_ard1 | 4096 | 80 | 12.0 | 1.6 |
| W_vmax_user (ard1, tangent) | 4096 | 80 | 12.0 | 2.1 |
| Ga_ardinf | 4096 | 144 | 7.4 | 2.1 |
| Ga_ard1 (anchor 0.06) | 4096 | 88 | 12.0 | 1.4 |
| Ga_beta0 (gamma_beta_0, gamma_beta_new) | 4096 | 140 | 7.4 | 2.5 |
| Mk_ardinf | 4096 | 144 | 7.25 | 1.2 |
| Mk_ard1 | 4096 | 88 | 12.0 | 2.0 |
| Ca_ardinf | 4096 | 144 | 7.5 | 4.3 |
| Ca_ard1_tangent (anchors below) | 4096 | 80 | 12.0 | 2.2 |
| Ho_ardinf | 4096 | 144 | 7.5 | 1.6 |
| Ho_ard1_pwl3 | 4096 | 80 | 12.0 | 2.0 |
| Ho_indicator | 4096 | 144 | 7.5 | 1.9 |
| Be_ardinf | 4096 | 144 | 7.5 | 2.8 |
| Be_ard1_tangent (ref 0.4) | 4096 | 80 | 12.0 | 1.9 |
| Be_pwl1 | 4096 | 144 | 7.5 | 3.1 |
| Ch_ardinf | 4096 | 144 | 7.25 | 1.3 |
| Ch_ard1 (anchors below) | 4096 | 80 | 12.0 | 1.5 |
| Ch_indicator | 4096 | 144 | 7.25 | 1.5 |
| G_mode_opavg | 4096 | 144 | 3.8 | 1.3 |
| G_mode_eval (ard1, E = 6) | 4096 | 80 | 24.0 | 1.4 |
| G_penalty_inf (C_D = 1) | 4096 | 144 | 8.4 | 2.1 |
| G_penalty_quad (C_D = 1, exact) | 4096 | 144 | 8.19 | 2.9 |
| Ca_penalty_quad (ard1, C_D = 0.5, opavg) | 4096 | 80 | 6.045 | 2.0 |
| G_no_replacement (increments 0.1) | 1296 | 32 | 2.5 | 0.5 |
| G_transitory (mu_trans 0.35, 0.2) | 4096 | 112 | 7.6 | 1.8 |
| mix_G_IG (L=2, H=[1,2], T=3) | 8000 | 20 | 22.0 | 7.4 |
| mix_W_Ch (L=2, H=[1,2]) | 8000 | 20 | 22.0 | 6.0 |
| mix_Ga_Ho (L=2, H=[1,2]) | 8000 | 20 | 22.0 | 4.8 |
| infeasible_G (H=[1,1]; day-2 flier breaks the loop row) | 64 | 0 | none | 0.02 |

Timing is local Python, noisy; the three L=2 cases with replacement exceed the 5 s target (8000 schedules). Not reduced further: with H1 = 1 and no replacement they have no feasible schedule, and `H2 = 1` is infeasible (loop row).

## Deviations and notes
- Plan check C: `repeated_safety >= -5e-5`, not `-1e-9`. A loop row broken by the solver's feasibility tolerance (1e-6) grows by that amount in each repetition (20 cycles: 2e-5). A true schedule defect is orders of magnitude larger.
- Plan G2: printed with `print` (`G2 <case>: solver objective ... < reference best ...`), never failed; B and C are applied to every case before it.
- The infeasible case is exempt from the non-triviality test (asserts `n_feasible == 0`); the Gurobi test accepts a status other than `optimal` without a solution, or B and C holding.
- Several forms share one best cost (for example G exact, pwl P=1 and P=3: 7.7); the forms differ in the feasible set only through `G_ardinf_tangent` and `G_ard1_tangent` (tangent_ref 0.2 / 0.3, which raises the cost). The oracle's own unit tests cover the forms at the row level.
- `res["R"]`, `res["K"]` and the `g_*` arrays are read as arrays (NaN where untracked), as the Result Dictionary table says.
- No change to `reference_model.py`.

## Possible defects
- None found in the oracle or the spec. Code defects can show only in the Euler run.
