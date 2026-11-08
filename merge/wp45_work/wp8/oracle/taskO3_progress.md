Status: done (local part). 22 inputs and `tests/test_oracle_replay.py` written; the Gurobi test runs on Euler (Claude submits). Local: 23 passed, 22 skipped, 0 failed.

# Task O3 progress

- [x] read interface, plan, O1/O2 progress, oracle, spec 349-650, 1709-1785, 2712-2830
- [x] scratch tools in `scratchpad\wp8\O3\`: `design.py` (loads each input, non-triviality, hill-climb search of a feasible witness schedule checked by `row_violations`), `dry_run.py` (the Gurobi test with a fake `solve_mapping` that returns the witness with reference values)
- [x] inputs `tests/data/oracle/replay_*.yaml` (22 cases, all pass `load_config` without warnings; a feasible witness schedule exists for each, so each is feasible at exact rows)
- [x] `tests/test_oracle_replay.py`: `test_every_replay_file_is_listed`, `test_case_is_not_trivial` (22, no Gurobi), `test_solution_means_what_the_spec_says` (22, Gurobi; checks A to F, the domains of spec 612-644, the NaN pattern, `idle`)
- [x] local run: 23 passed, 22 skipped (runner copy `scratchpad\wp8\O3\run_tests_gamma.py`); dry run with the fake solver: 22 passed

## Cases
G ardinf exact; G ard1 tangent (0.4); G pwl P=4; G indicator (ard1); IG ardinf; IG ard1; W ardinf; W ard1 tangent with v_max_user; Ga ardinf with gamma_beta_0; Ga ard1 with M=2, rates (F,L,M) and gamma_beta_trans; RF Markov; RF Cantelli tangent ard1; RF Hoeffding exact; RF Bernstein pwl (b=0.1, P=4); RF Chernoff ard1 (s=3); mixed (spec example, time_limit 120); operating_average (G ard1); evaluation_total (IG ard1, E=20); penalty quadratic (G exact + IG); no replacement (G ardinf tangent); replacement_as_new (G ardinf); transitory profile (G + IG, distinct days in mu, mu_trans, v, v_trans).
All: F=3, L=2, T=5 (H=[2,3]); M=1 except Ga rates and mixed (M=2).

## Choices and deviations
- Non-triviality: with `verbose` and the rest as the plan. The argument is general: every mission is flown every day and every operating increment is positive, so some mean is larger on day T than on day H1 and the loop row `mu_loop` breaks for every schedule without intervention. The test checks the minimum operating increment > 0, then the schedule in turn with the oracle (row `mu_loop`). For all-Wiener inputs it also checks that repairs on every free depot slot do not repair `v_loop`.
- Mixed example: the spec's numbers are used unchanged (feasible; a witness exists), with `mip_gap 0`, `time_limit 120`, `Threads 1`, `verbose 0`.
- `no_replacement` uses `ardinf`, not `ard1`. Reason (from spec Eq. a_anchor and a_loop, derived by hand): under `ard1` an anchor never falls without a replacement (`g_k = s_k >= g_{k-1}` on a repair), so `g_T <= g_{H1}` forbids every repair in the operating phase that raises the anchor, and a train that flies there cannot return `mu_T <= mu_{H1}` unless `rho = 1`. Under `ard1` with `rho < 1` and no replacement the problem is infeasible; this is a spec property, not a defect (the oracle search found no witness in 27 s).
- Tolerances: B as the plan; D uses `1e-5 + 1e-4 M_s` (leaks of a one-sided chain over T days); the domain checks `1e-6 + 2e-5 M_s`; the final comparison of `objective` with the oracle cost: `1e-6` relative plus `1e-4` absolute (a leak of `1e-6` in a `z` row, times `C_R`, times up to `(E - H1)/H2` in the mode `evaluation_total`); the keys of the cost report: `1e-8` relative.
- Extra checks beyond the plan: `idle = x0 - m - r` (spec 1734); NaN pattern of the tracked arrays and anchors (spec 1736-1739); `z` is NaN exactly where `C_R = 0`; Eq. c_zpost in the mode `operating_average`.
- Not asserted: `z` equal to the smallest value on built days (plan asks only for the rows); `mip_gap` reported.
- The oracle was not changed (no helpers needed).

## Possible code defects, spec findings
- None found locally (the Gurobi checks have not run).
- Spec remark (no defect): with `ard1` and no replacement the loop makes the problem infeasible whenever a train flies in the operating phase and `rho < 1`. A user input of that kind gives `infeasible`; an input check does not exist and none is demanded by the spec.

## Open points
- The first Euler run decides whether each solve is under 5 s (mixed and `ga_ard1_rates` have M=2) and whether the tolerance of D and F is wide enough.
