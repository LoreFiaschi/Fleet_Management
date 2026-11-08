Status: done. All five items edited in `merge/spec_drafts/fragments_D.tex`, second part only (from "Plotting API and Layout"). Braces and `\tabstart`/`\tabend` balanced; 0 CR before and after. Not built.

Paths: plotter `utils/plotter.py` (P), `validation/validator.py` (V), `validation/dashboard_inspector.py` (DI), `validation/dashboard_gamma.py` (DG), `validator_dashboard.py` (VD), `gamma_utils/gamma_replay_validator.py` (RV), `gamma_tail_validator.py` (TV), `gamma_tail_bound.py` (TB), `gamma_stochastic_validator.py` (SV).

## Item 1, group 2
| Id | Location | Old | New | Code |
|---|---|---|---|---|
| M22 | Structural, table, `depot_gating` violation | "the largest excess of the left side over the right side" | `max{0, max(m + r - x0)}`; without replacement the larger of it and `max r` | V `check_result` |
| M23 | Gamma, Common rules, File version | "It takes an input path..." (for each validator) | replay, exact and dispatcher have a file version; Monte Carlo has none, the dispatcher's with mode `"stochastic"` serves it | SV (no `*_files`) |
| M24 | Validation Tools, intro list | bullet "takes the model of each cell from the result key `model`" | bullet removed; sentence: dashboards and Gamma validators read it, structural validator does not | V; DI `load_pair`; RV `gamma_cells` |

## Item 2, group 3 (D)
| Topic | Location | New text | Code |
|---|---|---|---|
| Plot elements | Schedule Plot, Layout | dashed line and labels only if `0 < H1 < T`; thin grey cell lines; new bullet "Titles and labels" (figure title, grid title, axis labels); wrap sentence after the summary list (22 characters, at blanks, long word not split) | P `_draw_solution`, `_draw_statistics` |
| Plot elements | Sweep Plot, Content | Panels bullet: grid, panel titles, legends, tick thinning above 10 cases (every second `H2` plus the last); new bullets "Lines" (grey line in (a) if more than one feasible case, and in (b); infeasible cases break it) and "Footnote" (exact text) | P `_draw_horizon_sweep`, `_readable_horizon_ticks` |
| Tie rule of best mark | Sweep, panel (a) | smallest `J_op/H2` among feasible cases, exact equality, then smaller `H2`; noted as different from the selection | P `_normalise_horizon_sweep` |
| Gamma cap `tau` | Dashboards, new bullet "Gamma cap"; replay field `cap` | cap is `tau` at a cell with no rate; margin table, `cap` and Gamma tab use it | `mean_cap`; RV `gamma_cell_caps`; DI `reliability_caps` |
| Series | Exact, step 3 | equal-rate test `allclose`, rtol 1e-13, atol 0; one Gamma tail, remaining mass 0, 1 term | TB |
| Dashboard elements | Dashboards | defaults `input.yaml`, `output.yaml`; Reliability sorted by margin; margin table columns; Component comparison extra table; line after `H1` in margin and tail heatmaps; Raw data = margin table with CSV `result_inspector.csv`; Gamma: mean-tab table, tail heatmap (colour to largest epsilon, black line, "!"), tail table columns, raw data with two CSV files | VD, DI, DG |
| Report keys | Replay report keys (split into rows); Exact keys; Monte Carlo keys | `solver_status` `"unknown"`; `dimensions` record; `timing` record; `transitions_checked` = Gamma cells x T; `repairs`/`replacements` over Gamma cells (also noted in the Gamma overview); new row `terms` (record keys, `part`, `source` values, `created_step`); `frozen_mean`, `saved_anchor` null outside ARD1; worst step carries `terms`; exact-tail evaluations count; arguments-used keys listed; cell record keys and `worst_time_step` tie rule (earliest day, 0-based) | RV, TV, SV |
| `max_series_terms` | Exact, further arguments and Errors | integer rule as for `repetitions`; `ValueError` row for `convolution_tolerance` and `max_series_terms`; `RuntimeError` row adds "needs at least 2 terms if the rates differ" | TV `_integer`; TB loop |
| No `T` | Replay, step 1, first bullet | "or a result without the key `T`" | RV `_schedule_violations` |

## Item 3, M30
Dashboards, new bullet "When the sidebar is read": mode and tolerance are read at "Load"; a later tolerance change has no effect until the next load; a page shows only for the mode of the last load. The Validation tab and the Gamma text now say "the tolerance of the last load" (old: "the sidebar tolerance"). Code: VD `main`, DI `load_inspector`, DG `load_gamma_diagnostic`.

## Item 4, consequences
| Id | Where | Change |
|---|---|---|
| M6, M7, M8 | Sweep, Content | missing `H1` left out of the title (already stated; the code now agrees); lines and labels only for feasible cases; selection also within one report (tie: order of the reports, then of `cases`; "The selection also runs for one report") |
| M9 | Gamma Errors table, `ValueError` row | the common rule already covered it; added the clause: `+-inf` of `z` at a Gamma cell raises, `NaN` means not built and is skipped |
| M10 | Monte Carlo, `random_seed` row | non-negative integer; negative value raises `ValueError` naming the argument; "same samples" instead of "same report" (timing differs) |
| M11 | Dashboards, after the margin tables | largest `mu/tau` on the result's `mu` and `tau` over all finite cells and days, also where the margin is missing; `NaN` if none |
| M12 | Dispatcher, None rule | a block key with value `null` counts as absent; a direct call does not read the block |
| M13 | Exact, step 5 | weights in log scale, no error for a small first weight; `RuntimeError` row only for non-convergence |

## Item 5, start of the repetition
- Structural table, `loop_q`: rows `q_T <= y^q` and `q_H1 <= y^q` with violation `max{0, max(q_T - y), max(q_H1 - y)}`; without `y_q` the old row; paragraph: `y_q` counts as saved if it holds one number; a non-number difference is skipped (V `start_values`, `_loop_violation`).
- Replay: reads `y_mu`, `y_g_mu` (shape F x L); `repeatability` row and record text (`expected` = `y`, else the value on day H1) (RV).
- Dashboards: Repeatability paragraph lists `q_H1`, `y`, `q_T`, both margins, the smaller margin and "holds"; rows without a number `y` left out; without `y_q`, `y` and the H1 margin are NaN (DI `loop_table`).

## Disagreements and open points
- No disagreement between the items and the code.
- Common rule of the Gamma validators says "the Gamma validators" read the key `model`; the Monte Carlo validator also needs a global-violation check only through the others (it raises none). Unchanged.
- `check_sizes` does not check the shape of `y_*` (Y3 open point); the structural validator then fails inside numpy on a wrong shape. Not stated in the spec.
- A cell with `y_q` NaN while `q` is a number is skipped (Y3); documented as "a difference that is not a number is skipped".
- The Gamma dashboard's overview counts repairs over Gamma cells, the Result inspector and the plot over all cells; both now stated.
