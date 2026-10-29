Status: done. R1-R7 complete; waiting for the planner's review.

## Steps
- R1 done: `src/fleet_management/degradation_model/rainflow.py` written (identical to `proto_rainflow.py` after
  CRLF normalisation); ast.parse ok.
- R2 done: CRLF 143/143 lines, longest line 97; `names_check.py` 0 and 0; `rf_rows.py` prints `failures: 0`.
- R3 done: `tests/test_rainflow.py` head = verbatim copy of `fixture_draft.py` (cmp equal); ast ok;
  `names_check.py` 0 and 0; `fixture_check.py` `failures: 0`, `failures after helpers: 0`.
- R4 done: T1-T5 appended; ast ok; names_check 0 and 0.
- R5 done: T6-T9 appended; ast ok; names_check 0 and 0.
- R6 done: T10-T12 appended; file converted to CRLF: 460/460 lines, longest line 97; ast ok; names_check 0 and 0.
- R7 done: final reply sent.

## Local test run (emulator, `run_local.py`, my own pytest-free runner in this folder)
`python run_local.py <tests/test_rainflow.py>`: 68 passed, 29 skipped (need Gurobi), 0 failed.
- Passed locally: T1 (1), T2 (5), T3 (10), T4 (6: tangent and pwl cases), T5 (10), T6 (20), T7 (10: Markov and
  Chernoff exact, quadratic bounds in the tangent form), T9 (2), T11 (2), T12 (2).
- Skipped, need Gurobi: T4 `exact` (3), T7 `exact` and `pwl` of the quadratic bounds (12), T8 (12), T10 (2).
- `t12_check.py`: both errors raised with the expected messages; `pwl_w` counts 64 and 0 as in T4.
- `t10_check.py` (new): T10's logic through the emulator with the quadratic row replaced by its side row: b = 0.3
  infeasible, b = 0 optimal, RHS of `rel_side_0_0_0` = 1 - ln(2.5) b / 3 (0.90837 and 1.0).

## Deviations from the plan
- None in `rainflow.py` or in the test head. The tests T1-T12 follow section 3.3; the plan does not give their code,
  so the wording, parametrisation and helper use are mine.

## Problems
- None blocking. Notes for the reviewer:
  - T4 asserts `ctx.model.Params.NonConvex == 2` against the Gurobi default (-1) at the non-exact cases; the
    emulator has no default, so `run_local.py` adds `NonConvex = -1` as a shim.
  - T10 passes `mu_trans` of train 0 `[0.8, 0.0]` with `H1 = 2`, so the transitory increment 0.8 acts on day 1
    (train 0 flies) and 0.0 on day 2 (repair day); it was checked only in the tangent/side-row-only emulation above.
  - T7 and T8 on `pwl` and `exact` and T10 were never run; they need Gurobi (WP8).
