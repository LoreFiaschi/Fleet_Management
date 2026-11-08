# Step 3: merged review of the code against the spec

Status (8 Nov 2026): approved by the user (all of groups 1 to 3; M30 documented; M31 a to-do item); M18 added after the Euler run. Code fixes done; spec text next.

- **Sources**: eight independent Sonnet reviews, two per area (`review_A_1.json` to `review_D_2.json`): 11 + 16 (A),
  3 + 4 (B), 8 + 12 (C), 13 + 13 (D) issues. Merged by Claude; "agreed" means both reviewers of the area found it.
- **Overall**: no deviation in the formulation (both B reviewers checked every row, coefficient by coefficient). The
  deviations are in edge cases of inputs, reports, warm starts, the sweep plot and the Gamma validators. Most items
  are behaviours of the code that the spec does not state.
- **Rule for the fixes**: the spec holds what the code does. Where the code contradicts a sound spec sentence, the
  code changes; where the code is sound and the spec is silent or wrong, the spec changes.

## 1 Code fixes (the code contradicts the spec, or fails on valid input)

| Id | Found by | What is wrong | Fix (Claude's recommendation) |
|---|---|---|---|
| M1 | A1-1, B2-4 (agreed) | A `NonConvex` entry of `gurobi_params` is overwritten by `prepare`, which sets `NonConvex = 2` afterwards; the spec says `gurobi_params` is set after every other option (493) | apply `gurobi_params` again after the `prepare` calls |
| M2 | A1-2, A2-1 (agreed) | `verbose: true` is accepted as 1; an integer key must refuse a flag (364) | refuse a flag (no input in the repository uses one) |
| M3 | A1-3, A2-3 (agreed) | `component_names: null` gives the default names; `null` is an error where a key applies (359) | raise `ValueError` |
| M4 | C1-1, C2-1 (agreed) | A Gurobi error inside the warm-start completion (or `computeIIS`) stops the solve; the spec says a failed completion never stops it (2178) | catch it, record the status `"error"`, dispose of the copy; the same for the relaxation start; add the status to the spec |
| M5 | A2-11 | An infinite best bound or MIP gap is returned as `±inf` (kept in HDF5, `null` in YAML and JSON); the spec says "if available" (1719) | report a non-finite bound or gap as `null` |
| M6 | D1-1, D2-1 (agreed; reproduced) | The sweep plot crashes (`TypeError`) with two or more reports that give no `H1`; the spec says the title leaves the size out (2370) | read `H1` only where given; drop the unused key `T` |
| M7 | D1-2 | The sweep plot draws line segments and gap labels for cases without a feasible solution; the spec shows only feasible cases there (2375-2376) | mask those cases |
| M8 | D1-3, D2-13 (agreed) | The selection for equal `H2` is skipped when only one report is given (2368) | always run the selection |
| M9 | D2-6 | The Gamma replay validator accepts `z = +inf` at a Gamma cell silently; the spec requires `ValueError` for a non-finite entry where read (2521) | raise `ValueError` |
| M10 | D2-2 | `random_seed = -1` fails inside numpy with an unclear message; the spec allows any integer (2646) | spec: "a non-negative integer"; code: a `ValueError` that names the argument |
| M11 | D1-11 | The inspector's "largest `mu/tau`" leaves out cells whose margin is `NaN` (2465) | compute it on the result's `mu` and `tau` |
| M12 | D1-12 | In the input block `validation`, a key with value `null` raises an error instead of taking the default (2702) | treat `null` as absent |
| M13 | D1-8, D2-3 (agreed; reproduced) | The exact Gamma validator stops with an unlisted `RuntimeError` when the first Moschopoulos weight underflows; on Windows this happens already at realistic sizes (shapes 2000 and 3000) | compute the weights in log scale; keep the error only for true non-convergence |
| M14 | C1-7, C2-6 (agreed) | A sweep accepts `H1 < 1`; every case then fails as a case error | `ValueError` before the first case, with a spec row |
| M15 | C1-4, C2-8 (agreed) | `completion_time_limit` of a source is not checked (`null` or a negative value fails inside Gurobi) | `ValueError` for a value that is not a number `>= 0` ("Invalid source", 2132) |
| M16 | B2-1 | `kappa_G = ndtri(1 - eps)` becomes infinite for `eps < 1.1e-16`, so `V_max = 0` without a warning | compute `kappa_G = -ndtri(eps)` (the same number, exact for small `eps`); spec line 1022 follows |
| M17 | B2-2 | `ig_mean_cap` fails with "math domain error" for `eps <= 1e-16` | guard the cancellation in the tail function |
| M18 | Euler job 20170912 (8 Nov 2026) | Gurobi returns binaries within its integrality tolerance (an `x` of 0.99999901); the Gamma validators require binaries within 1e-8 by default and refuse the result | the result extraction rounds `x`, `m`, `r` (and `idle`, derived from them) to exactly 0 or 1 (user's choice, 8 Nov 2026) |

## 2 Spec corrections (the code is right, the spec text is wrong or self-contradicting)

| Id | Found by | Spec | Correction |
|---|---|---|---|
| M20 | A2-7 | 506-525 | The messages of the rejected keys end with section pointers that no message can carry; move the pointers out of the message column |
| M21 | A2-15 | 1881 | "HDF5 files keep NaN and ±inf": true for datasets and number attributes; JSON text in an HDF5 attribute writes them as `null` |
| M22 | D2-9 | 2427 | The violation of `depot_gating` is `max{0, ...}` (as the other rows), plus `max r` when replacement is not allowed |
| M23 | D1-9 | 2511 | "Each validator has a file version": the Monte Carlo validator has none; the dispatcher's file version with mode `stochastic` serves it |
| M24 | D2-8 | 2389 | "Each tool takes the model from the result key `model`": the structural validator does not read it |
| M25 | C2-12 | 1983 | "Eligible to stop" also needs the gradient of case `n` to be available |
| M26 | A1-11, A2-16 (agreed) | 171-211 | The layout omits the tracked folder `input/`, the scoping files and the extra `dev`; `config.py` also derives constants |

## 3 Behaviours the code has and the spec does not state (document them)

- **Inputs and outputs (A)**: component names are stripped of blanks; `v_0` is `NaN` at cells without a variance; a
  file whose data are not a mapping raises `TypeError`; the meaning of the size counts (`integer_variables` includes
  the binaries, `general_constraints` includes the indicators, node and iteration counts are floats, `threads` and
  `seed` are parameter values); the errors of the HDF5 reader; where `warm_start_mode` is checked; a transitory
  profile needs its operating profile even where the key applies nowhere; `bound_method` is a `null` scalar without
  rainflow cells; the two formulas of `relative_gap`; entries added by an `extract` hook are returned but not written.
- **Formulation internals (B)**: `prepare` also creates the segment binaries `w`; the members of `ctx` and of
  `ctx.extras`; the helper functions of `base.py` and the order of their calls (the loop rows are built with the
  dynamics); the guarded import and `row_constants` of `rainflow.py`; the `ValueError`s of the builders.
- **Sweeps and warm starts (C)**: the grid's argument errors; the stop reason when the rule fires on the last
  candidate; rounding of source values (ties to even, no clipping); `model_construction_seconds` includes the warm
  starts; the parameters and logs of the two model copies; the report values of a step that did not run; how `H1`
  is read and the fixed objective texts; `objective_mode = "total"` in a sweep; the sizes are all `null` if one
  cannot be read; the relaxed copy minimises the violation only and keeps the quadratic rows; the completion's
  runtime and the integer variables it fixes; the worker pool of parallel sweeps; the two tests of "has a solution";
  the greedy's segment for a negative mean.
- **Plots and validation (D)**: the plot elements that the spec does not list (titles, footnote, grey lines, tick
  thinning, the dashed line only when `0 < H1 < T`); the tie rule of the best feasible mark; the Gamma cap `tau` for a
  cell without a rate; the equal-rate test of the series and its term count; the dashboard elements (default file
  names, lines after `H1`, extra tables, CSV export; "Raw data" is the margin table); report keys and values
  (`solver_status = "unknown"`, the records of `terms`, the `worst_step_*` keys, `dimensions`, `timing`,
  `transitions_checked`; repairs counted over the Gamma cells); `max_series_terms` is an integer; a result without
  `T` is a global violation.

## 4 Choices

| Id | Found by | Item | Options (recommended first) |
|---|---|---|---|
| M30 | D1-6 | The dashboards use the tolerance set at the click on "Load"; later changes in the sidebar have no effect until the next load (2457: "the sidebar tolerance") | document it (no code change; Streamlit is not testable here) / recompute on every change |
| M31 | B2-3 | Near the limits of `eps` (0.5 at G and W cells, 1 at Ca and Ho cells) the natural ranges, used as big-M constants, grow without bound (1.6e7·tau² at `eps = 0.4999`), so a binary within Gurobi's integrality tolerance can leak | a to-do item (no change now) / an input check on `eps` / tighter big-M constants |
