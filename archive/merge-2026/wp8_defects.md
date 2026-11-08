# WP8: defects and spec findings

Status (7 Nov 2026): closed. Code defects D1, D2 and test defects X1, X2 fixed (commit 8cf16c1); spec findings S1 to S9 applied to the spec; F1 reported to the user. Euler run 1 (job 20153161): 1597 passed, 8 failed (X1, X2). Claude classifies each item.

- **Code defect**: the code contradicts the spec. Fixed in step 5 of `wp8_interface.md`.
- **Test defect**: the test contradicts the spec. Fixed by the task that owns the test.
- **Spec finding**: the spec is silent, ambiguous, or states something that cannot hold. Goes to the user.

## Code defects

| Id | Spec | Test | Observed | Expected | Owner of the fix | State |
|---|---|---|---|---|---|---|
| D1 | 2538-2545 ("one string per occurrence"; one bullet: "an intervention on a day with a mission or without a depot day") | `test_spec_validation.py` (T1; accepts 1 or 2 strings) | `gamma_replay_validator._schedule_violations` writes two strings for an intervention on a mission day ("with a mission" and "without a depot day") | one string per cell and day where either holds | fix: `gamma_replay_validator.py` (one loop, one string); test: expect exactly 1 | fixed locally (fix1, 7 Nov 2026); confirmed on Euler (job 20155563) |
| D2 | 2331-2336 (summary panel shows the models and repair laws present) | rendered PNG of `test_spec_plots.py` (T4), seen by Claude | In the summary panel of `plot_management`, a value longer than one line ("gaussian, gamma, wiener, inverse_gaussian") wraps onto the next row and overlaps its value ("ard1, ardinf") | each value readable, no overlap | fix: `utils/plotter.py` (`_draw_statistics`: wrap long values and move the next rows down, or shrink the font of that value); test: assert no overlap of the text bounding boxes | fixed locally (fix1, 7 Nov 2026; PNG checked by Claude); confirmed on Euler (job 20155563) |

## Test defects

| Id | Test | Spec | Fault | Fix | State |
|---|---|---|---|---|---|
| X1 | `test_spec_outputs.py::test_bad_result_extension_with_missing_input_raises_value_error` (I3) | 221 (step 1: the input exists, then its extension, then `results_path`), 1880 ("before the input file is **read**", step 2) | The plan (Claude's error) expected `ValueError`; with a missing input, step 1 raises `FileNotFoundError` first. The code is right. | Expect `FileNotFoundError`; keep the two tests with an existing, unreadable input. | fixed locally (fix1, 7 Nov 2026); confirmed on Euler (job 20155563) |
| X2 | `test_spec_outputs.py`, 7 tests of the case `rainflow_bounds` (Euler job 20153161) | 1490-1494 (Markov row `mu <= eps tau`) | The library input `tests/data/io/valid/rainflow_bounds.yaml` is infeasible by the spec: its Markov component has the cap `eps tau = 0.1`, but every mission adds 0.25 to the mean and some train flies every day. Gurobi reports `infeasible`, correctly. | Make the input feasible (a Markov increment and initial mean below the cap), checked with the reference model on an explicit schedule; keep the I1 and I2 tests passing. | fixed locally (fix1, 7 Nov 2026); confirmed on Euler (job 20155563) |

## Semantic findings (allowed by the spec; reported to the user)

| Id | Spec | Finding | Evidence |
|---|---|---|---|
| F1 | 783 (one-sided rows), 1606-1617 (loop rows), 647-650 (reported values are upper bounds) | The solver's optimum is often far below the best schedule whose **exact** trajectory (every row at equality, 829) satisfies every row: in 19 of 42 brute-force cases, for example 3.3 against 7.7 (G `ardinf`), 2.45 against 7.25 (RF Markov), 12.8 against 22.0 (mixed Ga and Ho). The one-sided rows let the solver raise the tracked state at day `H1`, so the loop row `s_T <= s_{H1}` holds for a higher state although the exact trajectory breaks it. The certified regime is then the steady state of the repeated operating phase. | Euler job 20153605 (7 Nov 2026): for each such case the returned point satisfies every spec row (check B) and the exact trajectory of the returned schedule stays reliable over 20 repetitions of the operating phase (check C). Task O5 (7 Nov 2026): the least solution of the one-sided rows (Kleene iteration of every lower-bound row, the loop rows included) gives the spec's own optimum `best_onesided`; it equals the Gurobi objective of all 19 cases within 1e-11 (exact optimum for `ardinf` fleets, an upper bound with `ard1` cells). So the code solves exactly the problem the spec defines; the difference is between the spec's one-sided meaning and the exact-trajectory meaning. |

## Spec findings

Applied to the spec on 7 Nov 2026 (user: "Apply all as proposed"), each written to match the code; S3 states that
`pwl_points` is checked in every input, as the code does (old and new text: `wp45_work/wp8/spec_fix_progress.md`).

| Id | Spec | Finding | Proposal |
|---|---|---|---|
| S1 | 435 | The rule "an array whose shape fits both `(L, M, H*)` and `(F, L, M)` is read as `(L, M, H*)`" never applies: both shapes fit only when `F = L = M` (and `H* = M`), and the check `F > M` (369, 543) excludes it (task I1, 7 Nov 2026). | Delete the sentence, or keep it with "(this cannot happen, since `F > M`)". |
| S2 | 908-913, 2255 | At a boundary between two `pwl` segments the two caps are equal: for a quadratic `f`, the tangents at the midpoints `(p - 1/2) tau / P` and `(p + 1/2) tau / P` meet at the boundary `p tau / P`. So "the larger cap where two segments contain it" (2255) always picks the same value (task O1, 7 Nov 2026; checked by Claude). | No change needed; optionally say "the caps agree there". |
| S3 | 395, 559 | `pwl_points` applies only to cells with the form `pwl` (395), but the check "`pwl_points` is an integer `>= 1`" (559) has no such condition. The spec does not say whether `pwl_points: 0` is an error in an input without a `pwl` cell (task I2, 7 Nov 2026). | Say "at cells with the form `pwl`" in 559, as for `tangent_ref`. |
| S4 | 2428, 2435 | The check `idle` runs "only if the result has the key `idle`"; the report "lists the checks in the order of the table" and "the eight checks `loop_q` are always listed". The spec does not say whether `idle` is listed (with violation 0) when the key is absent (task T1, 7 Nov 2026). | Say "`idle` is listed only when the result has the key", or "always listed, violation 0 without the key", as the code does. |
| S5 | 1727, 384 | The result key `reliability_impl` is "the selector of each cell after broadcasting". At a cell with a linear row (IG, Ga, RF Mk, Ch) the input value is ignored and the form `exact` is built (384). The spec does not say which of the two the result reports (task I3, 7 Nov 2026). | Say "the form built at the cell (`exact` at a linear row)". |
| S6 | 2039 | The row key `cost_gradient_per_H2` (the absolute gradient) has no definition; only the relative gradient `d_gradient` is defined (task T2, 7 Nov 2026). | Define it: `(J_n - J_{n-1}) / (H_{2,n} - H_{2,n-1})`, `null` where the relative one is unavailable. |
| S7 | 1984, 2045 | "Every row carries its gradient, class and eligibility" (1984) conflicts with "an error row has `null` in every entry except the sizes" (2045) (task T2). | Say that an error row carries the gradient keys too (class `unavailable`). |
| S8 | 2019, 2036, 2015, 1939 | Unstated details of the sweep report: `formulation_growth` with fewer than two counted rows (a `null` mapping, or a mapping of `null` entries); the `performance` key behind `optimizer_seconds`; `warm_start.mode` when the warm start is off; the allowed values of `n_workers` (task T2). | Fix each in one clause. |
| S9 | 2271, 2272, 1838 | Unstated details of the warm-start reports: whether `maximum_proxy_ratio` counts the initial state `mu_0/tau`; whether `assignment_order` numbers missions from 1; the type of `source_horizon` (list or mapping) (task T3, 7 Nov 2026). | Fix each in one clause. |
