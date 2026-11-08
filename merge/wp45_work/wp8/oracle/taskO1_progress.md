Status: done. Next: task O2 (Gamma and rainflow in the registries `_QUANTITIES` and `_SLACK` of `tests/reference_model.py`; `row_violations`, `repeated_safety`, `best_reference`; unit tests 2 (Ga, RF), 3 (RF, Ga), 6 (Gamma), 8).

# Task O1 progress

## Written (worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge`, not committed)
- `tests/reference_model.py`: `cell_kind`, `day_profile`, `initial_and_new`, `ig_cap` (independent, scipy quad), `gamma_cap` (independent A1, brentq on w(A)), `constants` (all five models, rainflow bounds, quadratic-row constants c2, c1, side), `reference_trajectory` (G, IG, W; other kinds raise NotImplementedError), `quadratic_slack` (generic exact / tangent / pwl form, reusable by O2), `reliability_slack` (G, IG, W), `minimal_z`, `u_value`, `costs`, `enumerate_schedules`.
- Extension points for O2: dictionaries `_QUANTITIES` (kind -> tracked quantities with delta, factor, anchored flag) and `_SLACK` (kind -> slack function). `constants` and `initial_and_new` already cover Ga and all rainflow bounds.
- `tests/test_oracle_unit.py`: unit tests 1, 2 (G, IG, W), 3 (G, W, plus one Bernstein slack through `quadratic_slack`), 4, 5, 6 (IG cap), 7, plus constants, cell kinds, initial and replacement values.

## Local result
`run_tests_gamma.py` copy in the scratchpad (tests/ on sys.path): 66 passed, 0 skipped, 0 failed.

## Notes and possible defects
- No code defect found. The package's `ig_mean_cap` agrees with the oracle's independent cap within 1e-12 relative on (eta, eps) = (0.5, 0.05), (2, 0.1), (0.1, 0.01), (1, 0.3), tau 1 and 2.5. The package's Gamma `sharp_cap` agrees with `gamma_cap` within 1e-12 on four (t, eps) pairs (checked in the scratchpad only; the Gamma test is O2's).
- Plan correction: `V_max = 1/kappa_G^2` for eps = 0.05 is 0.3696115 (plan says 0.3696123); the tangent cap at mu = 0.3 is 0.1663252 (plan 0.1663255); exact 0.1811096 (plan 0.1811100). The tests use the corrected values.
- At a pwl segment boundary the two caps are equal (the tangent of c2 (tau-mu)^2 at m is f(mu) - c2 (mu-m)^2, and equal segments put the boundary halfway between midpoints), so "takes the larger cap" cannot be told apart from "takes either"; the test asserts the common value.
- A Bernstein tangent_ref must stay below sqrt(tau (tau - 2c)) (spec 590); the property test uses 0.7 for the Bernstein cell (bound 0.735).
- The pytest stub of the runner has no `approx`; the tests use their own `close` helper.
- Day-axis conventions: `x` (F, M+1, T), `m`, `r`, `z` (F, L, T); trajectory arrays (F, L, T+1) with index 0 initial.
