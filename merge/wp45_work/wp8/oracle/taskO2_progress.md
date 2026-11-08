Status: done. Local result: `test_oracle_unit.py` 138 passed, 0 skipped, 0 failed (runner copy in the scratchpad `wp8\O2\`). Next: task O3 and O4 (they may start).

# Task O2 progress

- [x] read interface, plan, O1 progress, O1 files, spec 603-1707 and 1714-1784
- [x] scratch checks of hand numbers (`wp8\O2\chk_traj.py`, `chk_slack.py`, `chk_scan.py`, `chk_rows.py`, `chk_safe.py`)
- [x] `_QUANTITIES` / `_SLACK` for Ga, Mk, Ca, Ho, Be, Ch (`tests/reference_model.py`)
- [x] `row_violations`, `repeated_safety`, `best_reference`
- [x] unit tests 2 (Ga, RF), 3 (RF, Ga), 6 (Gamma), 8, `row_violations`, `best_reference`
- [x] local run: 138 passed

## Interface notes for O3 and O4
- `reference_trajectory(cfg, x, m, r=None, profiles=None)`: horizon is the day axis of `x`; `profiles` only for `repeated_safety`.
- `row_violations(cfg, sol, tol=1e-6, bigm_tol=2e-5, quad_rel=1e-5)` returns a list of `Violation(row, i, l, k, violation)`. `sol` keys: `x`, `m`, `r` (optional), tracked arrays `(F, L, T)`, `z` (optional), `u` (optional). Row names: `one_activity`, `coverage_j<j>`, `depot`, `<q>_carry`, `<q>_repair`, `<q>_replace`, `g_<q>_set`, `g_<q>_keep`, `reliability`, `<q>_loop`, `g_<q>_loop`, `z_row`, `z_nonneg`, `u_row`. Tolerance rules in its docstring (big-M allowance `tol + bigm_tol * M_s`; exact quadratic reliability row `tol + quad_rel * M_chi`; linear reliability row `tol`).
- `best_reference(cfg)` returns a dict: `cost` (None if infeasible), `schedule` `(x, m, r)`, `n_feasible`, `n_total`, `costs`. It calls `row_violations` with `bigm_tol = 0`, `tol = 1e-9`: exact rows. 512 schedules (F=2, M=1, L=1, T=3, replacement) take 0.12 s.
- No domain rows (variable bounds) are checked in `row_violations`: they cannot fail alone on a reference solution; O3 may check the bounds itself.
- `repeated_safety(cfg, x, m, r, cycles)` accepts `r=None`.
- Caches on the config object: `_oracle_constants`, `_oracle_profiles`. Build a new config after changing a field.

## Deviation from the task text
- The tiny `best_reference` case uses `H = [1, 2]` (T = 3, 512 schedules), not `[1, 1]`. With `H2 = 1` the loop row `s_2 <= s_1` fails for the train that flies on day 2 (it has no day to intervene), so no schedule is feasible with a positive increment. That case is a second test (`n_feasible = 0`, 64 schedules). The `H = [1, 2]` case has a hand-derived best cost 8 (8 feasible schedules, costs 8, 8, 9, 9, 9.2, 9.2, 12, 12).

## Defects and findings
- **Oracle defect found and fixed (O1's `gamma_cap`)**: for `A^ >= t` it returned `t` without testing `w(t) <= eps`; for `eps < 1/2` the maximum over `c` can exceed `Q(A, t)`, so the cap is below `t` (cases `(1, 0.4)`: 0.9533342, `(0.05, 0.2)`: 0.0303336). Now the oracle follows spec step 3 and agrees with the package's `sharp_cap` within 1e-9 relative on 36 pairs of `t` in 0.05..20 and `eps` in 0.001..0.45, and on four pairs in the test.
- No code defect found: package `sharp_cap` and `mean_cap` agree with the independent oracle within 1e-6 (checked 36 pairs plus the test pairs; `eps >= 1/2` gives `t` up to 1e-9 relative).
- No spec defect. Remark: spec line 1295 says that for `eps >= 1/2` "A_1 is the value that step 3 returns"; the oracle takes the supremum `t` (difference at most 1e-9 relative).
- Not asserted (spec silent): none.
