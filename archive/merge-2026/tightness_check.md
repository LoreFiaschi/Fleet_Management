Status: done. All 42 inputs `tests/data/oracle/bf_*.yaml`, every schedule enumerated (176912 in total, no sampling), run 8 November 2026 on the worktree `Fleet_Management-merge` (scripts: `scratchpad/wp8/tight/run.py`, `diag.py`; raw results `all.json`).

# Tightness of the one-sided rows

## Method
- A schedule is enumerated with `enumerate_schedules(cfg)`; A = `least_solution(cfg, x, m, r, verify=False)["feasible"]` (the solver's acceptance).
- B = `repeated_safety(cfg, x, m, r, 300) >= -1e-9` (operating phase repeated 300 times, exact trajectory, smallest reliability slack). The same test at 30 repetitions is computed too; a schedule where the two answers differ counts in the column `slow`.
- `A and not B` would be a soundness defect: the solver accepts an unsafe schedule. `B and not A` is conservatism of the one-sided rows.
- Column `min cost B-not-A`: the smallest objective (`costs` of the exact reference trajectory, minimal `z`, `u_value`) over the schedules with B and not A. Column `best_onesided`: `best_onesided(cfg)["cost"]`, the optimum the solver can reach. Column `optimum changes`: YES when `min cost B-not-A` is below `best_onesided`.
- The schedule count is 4096 per case (1296 for `G_no_replacement`, 64 for `infeasible_G`, 8000 for the three `mix_*` cases).

## Table
| case | repair laws | schedules | A and B | A and not B | B and not A | min cost B-not-A | best_onesided | optimum changes | slow |
|---|---|---|---|---|---|---|---|---|---|
| Be_ard1_tangent | ard1 | 4096 | 88 | 0 | 8 | 18 | 12 | no | 0 |
| Be_ardinf | ardinf | 4096 | 320 | 0 | 64 | 8.5 | 2.9 | no | 0 |
| Be_pwl1 | ardinf | 4096 | 320 | 0 | 64 | 8.5 | 2.9 | no | 0 |
| Ca_ard1_tangent | ard1 | 4096 | 88 | 0 | 8 | 18 | 12 | no | 0 |
| Ca_ardinf | ardinf | 4096 | 416 | 0 | 96 | 8.5 | 2.9 | no | 0 |
| Ca_penalty_quad | ard1 | 4096 | 112 | 0 | 16 | 6.08 | 6.045 | no | 0 |
| Ch_ard1 | ard1 | 4096 | 112 | 0 | 16 | 18 | 12 | no | 0 |
| Ch_ardinf | ardinf | 4096 | 416 | 0 | 96 | 8.25 | 2.45 | no | 0 |
| Ch_indicator | ardinf | 4096 | 416 | 0 | 96 | 8.25 | 2.45 | no | 0 |
| G_ard1_anchor | ard1 | 4096 | 88 | 0 | 8 | 18 | 12 | no | 0 |
| G_ard1_indicator | ard1 | 4096 | 88 | 0 | 8 | 18 | 12 | no | 0 |
| G_ard1_pwl3 | ard1 | 4096 | 88 | 0 | 8 | 18 | 12 | no | 0 |
| G_ard1_tangent | ard1 | 4096 | 22 | 0 | 2 | 24 | 13.3 | no | 0 |
| G_ardinf_exact | ardinf | 4096 | 320 | 0 | 64 | 8.7 | 3.3 | no | 0 |
| G_ardinf_pwl1 | ardinf | 4096 | 320 | 0 | 64 | 8.7 | 3.3 | no | 0 |
| G_ardinf_pwl3 | ardinf | 4096 | 320 | 0 | 64 | 8.7 | 3.3 | no | 0 |
| G_ardinf_tangent | ardinf | 4096 | 16 | 0 | 0 | - | 13.4 | - | 0 |
| G_indicator | ardinf | 4096 | 320 | 0 | 64 | 8.7 | 3.3 | no | 0 |
| G_mode_eval | ard1 | 4096 | 88 | 0 | 8 | 30 | 24 | no | 0 |
| G_mode_opavg | ardinf | 4096 | 320 | 0 | 64 | 1.3 | 1.6 | YES | 0 |
| G_no_replacement | ardinf | 1296 | 72 | 0 | 0 | - | 2.5 | - | 0 |
| G_penalty_inf | ardinf | 4096 | 320 | 0 | 64 | 9.2 | 4 | no | 0 |
| G_penalty_quad | ardinf | 4096 | 320 | 0 | 64 | 8.95 | 3.79 | no | 0 |
| G_transitory | ardinf | 4096 | 320 | 0 | 64 | 8.6 | 3.2 | no | 0 |
| Ga_ard1 | ard1 | 4096 | 112 | 0 | 16 | 18 | 12 | no | 0 |
| Ga_ardinf | ardinf | 4096 | 416 | 0 | 96 | 8.4 | 2.7 | no | 0 |
| Ga_beta0 | ardinf | 4096 | 416 | 0 | 96 | 8.42 | 2.7 | no | 0 |
| Ho_ard1_pwl3 | ard1 | 4096 | 88 | 0 | 8 | 18 | 12 | no | 0 |
| Ho_ardinf | ardinf | 4096 | 320 | 0 | 64 | 8.5 | 2.9 | no | 0 |
| Ho_indicator | ardinf | 4096 | 320 | 0 | 64 | 8.5 | 2.9 | no | 0 |
| IG_ard1 | ard1 | 4096 | 8 | 0 | 0 | - | 18 | - | 0 |
| IG_ardinf | ardinf | 4096 | 8 | 0 | 0 | - | 18 | - | 0 |
| IG_indicator | ardinf | 4096 | 8 | 0 | 0 | - | 18 | - | 0 |
| Mk_ard1 | ard1 | 4096 | 88 | 0 | 8 | 18 | 12 | no | 0 |
| Mk_ardinf | ardinf | 4096 | 320 | 0 | 64 | 8.25 | 2.45 | no | 0 |
| W_ard1 | ard1 | 4096 | 88 | 0 | 8 | 18 | 12 | no | 0 |
| W_ardinf | ardinf | 4096 | 88 | 0 | 8 | 18 | 12 | no | 0 |
| W_vmax_user | ard1 | 4096 | 88 | 0 | 8 | 18 | 12 | no | 0 |
| infeasible_G | ardinf | 64 | 0 | 0 | 0 | - | - | - | 0 |
| mix_G_IG | ard1+ardinf | 8000 | 20 | 0 | 0 | - | 22 | - | 0 |
| mix_Ga_Ho | ard1+ardinf | 8000 | 108 | 0 | 52 | 18.5 | 12.8 | no | 0 |
| mix_W_Ch | ard1+ardinf | 8000 | 30 | 0 | 10 | 28 | 22 | no | 0 |

## Totals
| | count |
|---|---|
| schedules | 176912 |
| A and B | 7836 |
| A and not B (soundness defects) | **0** |
| B and not A (conservatism) | **1512** |
| neither A nor B | 167564 |
| schedules with 30 and 300 repetitions disagreeing | 0 |

## Findings
- **No soundness defect.** In no case does the solver accept a schedule that is unsafe on 300 repetitions.
- **The claim "equality for `ardinf`" is false.** 35 of the 42 cases have `B and not A` schedules. The other seven have none: `G_ardinf_tangent`, `G_no_replacement`, `IG_ard1`, `IG_ardinf`, `IG_indicator`, `mix_G_IG` and `infeasible_G`. `ardinf` cases show them as often as `ard1` cases.
- **Cause**, read from `least_solution(...)["reason"]` on four cases (`G_mode_opavg`, `G_ardinf_exact`, `W_ard1`, `mix_Ga_Ho`): every `B and not A` schedule is refused with "replacement value" on the last day of the transitory phase (day `H1`, or day 1 in `mix_Ga_Ho`). A replacement on day `H1` pins `s_{H1}` to `s_new`; the loop row `s_T <= s_{H1}` then needs the operating phase to end at or below `s_new`. The operating phase can rise from `s_new` to a level that is still safe, and it then repeats safely (it converges), but the loop row refuses it. So the loop row is sufficient for periodic safety and not necessary; the one-sided dynamics rows themselves are not the source.
- **Convergence is not slow**: 30 and 300 repetitions agree on every schedule.
- **Optimum.** The optimum changes in one case only: `G_mode_opavg` (`min cost B-not-A` 1.3 against `best_onesided` 1.6). In every other case `min cost B-not-A` is above `best_onesided` (for example 18 against 12 in the `ard1` cases), so the conservatism costs nothing there. Caveat: the cost of a B-not-A schedule is taken from the reference trajectory of the transitory phase and one operating phase, as `best_reference` does; for the case `G_mode_opavg` (a mode that averages the operating phase) the cost of such a schedule was not examined further.


# After the start-of-repetition change (8 Nov 2026)

## Method
- The same method as above, with the new `least_solution` of `tests/reference_model.py` (the rows with the start of the repetition `y`: `y >= s_{H1}`, the rows of day `H1 + 1` read `y`, the loop rows `s_T <= y`). A = `least_solution(...)["feasible"]`, B = `repeated_safety(cfg, x, m, r, 300) >= -1e-9` (exact trajectory; unchanged), 8 parallel processes, all 42 inputs, every schedule (176912), run 8 November 2026. Scripts and raw results: `scratchpad/step3/Y2/tight/` (`run.py`, `agg.py`, `all.json`).
- Column `old ...`: the values of the table above. `best_onesided` is the new `best_onesided(cfg)["cost"]`.

## Totals
| | before | after |
|---|---|---|
| schedules | 176912 | 176912 |
| A and B | 7836 | **9348** |
| A and not B (soundness defects) | 0 | **0** |
| B and not A (conservatism) | 1512 | **0** |
| schedules with 30 and 300 repetitions disagreeing | 0 | 0 |

## Table
| case | schedules | A and B | A and not B | B and not A | old A and B | old B and not A | min cost B-not-A | best_onesided | old best_onesided |
|---|---|---|---|---|---|---|---|---|---|
| Be_ard1_tangent | 4096 | 96 | 0 | 0 | 88 | 8 | - | 12 | 12 |
| Be_ardinf | 4096 | 384 | 0 | 0 | 320 | 64 | - | 2.9 | 2.9 |
| Be_pwl1 | 4096 | 384 | 0 | 0 | 320 | 64 | - | 2.9 | 2.9 |
| Ca_ard1_tangent | 4096 | 96 | 0 | 0 | 88 | 8 | - | 12 | 12 |
| Ca_ardinf | 4096 | 512 | 0 | 0 | 416 | 96 | - | 2.9 | 2.9 |
| Ca_penalty_quad | 4096 | 128 | 0 | 0 | 112 | 16 | - | 6.045 | 6.045 |
| Ch_ard1 | 4096 | 128 | 0 | 0 | 112 | 16 | - | 12 | 12 |
| Ch_ardinf | 4096 | 512 | 0 | 0 | 416 | 96 | - | 2.45 | 2.45 |
| Ch_indicator | 4096 | 512 | 0 | 0 | 416 | 96 | - | 2.45 | 2.45 |
| G_ard1_anchor | 4096 | 96 | 0 | 0 | 88 | 8 | - | 12 | 12 |
| G_ard1_indicator | 4096 | 96 | 0 | 0 | 88 | 8 | - | 12 | 12 |
| G_ard1_pwl3 | 4096 | 96 | 0 | 0 | 88 | 8 | - | 12 | 12 |
| G_ard1_tangent | 4096 | 24 | 0 | 0 | 22 | 2 | - | 13.3 | 13.3 |
| G_ardinf_exact | 4096 | 384 | 0 | 0 | 320 | 64 | - | 3.3 | 3.3 |
| G_ardinf_pwl1 | 4096 | 384 | 0 | 0 | 320 | 64 | - | 3.3 | 3.3 |
| G_ardinf_pwl3 | 4096 | 384 | 0 | 0 | 320 | 64 | - | 3.3 | 3.3 |
| G_ardinf_tangent | 4096 | 16 | 0 | 0 | 16 | 0 | - | 13.4 | 13.4 |
| G_indicator | 4096 | 384 | 0 | 0 | 320 | 64 | - | 3.3 | 3.3 |
| G_mode_eval | 4096 | 96 | 0 | 0 | 88 | 8 | - | 24 | 24 |
| G_mode_opavg | 4096 | 384 | 0 | 0 | 320 | 64 | - | 1.6 | 1.6 |
| G_no_replacement | 1296 | 72 | 0 | 0 | 72 | 0 | - | 2.5 | 2.5 |
| G_penalty_inf | 4096 | 384 | 0 | 0 | 320 | 64 | - | 4 | 4 |
| G_penalty_quad | 4096 | 384 | 0 | 0 | 320 | 64 | - | 3.79 | 3.79 |
| G_transitory | 4096 | 384 | 0 | 0 | 320 | 64 | - | 3.2 | 3.2 |
| Ga_ard1 | 4096 | 128 | 0 | 0 | 112 | 16 | - | 12 | 12 |
| Ga_ardinf | 4096 | 512 | 0 | 0 | 416 | 96 | - | 2.7 | 2.7 |
| Ga_beta0 | 4096 | 512 | 0 | 0 | 416 | 96 | - | 2.7 | 2.7 |
| Ho_ard1_pwl3 | 4096 | 96 | 0 | 0 | 88 | 8 | - | 12 | 12 |
| Ho_ardinf | 4096 | 384 | 0 | 0 | 320 | 64 | - | 2.9 | 2.9 |
| Ho_indicator | 4096 | 384 | 0 | 0 | 320 | 64 | - | 2.9 | 2.9 |
| IG_ard1 | 4096 | 8 | 0 | 0 | 8 | 0 | - | 18 | 18 |
| IG_ardinf | 4096 | 8 | 0 | 0 | 8 | 0 | - | 18 | 18 |
| IG_indicator | 4096 | 8 | 0 | 0 | 8 | 0 | - | 18 | 18 |
| Mk_ard1 | 4096 | 96 | 0 | 0 | 88 | 8 | - | 12 | 12 |
| Mk_ardinf | 4096 | 384 | 0 | 0 | 320 | 64 | - | 2.45 | 2.45 |
| W_ard1 | 4096 | 96 | 0 | 0 | 88 | 8 | - | 12 | 12 |
| W_ardinf | 4096 | 96 | 0 | 0 | 88 | 8 | - | 12 | 12 |
| W_vmax_user | 4096 | 96 | 0 | 0 | 88 | 8 | - | 12 | 12 |
| infeasible_G | 64 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mix_G_IG | 8000 | 20 | 0 | 0 | 20 | 0 | - | 22 | 22 |
| mix_Ga_Ho | 8000 | 160 | 0 | 0 | 108 | 52 | - | 12.8 | 12.8 |
| mix_W_Ch | 8000 | 40 | 0 | 0 | 30 | 10 | - | 22 | 22 |

## Findings
- **No soundness defect, and no conservatism left on these inputs.** The 1512 schedules that were `B and not A` (refused because of a replacement on day `H1`) are all accepted now, and none is unsafe on 300 repetitions. The 35 cases that had `B and not A` schedules have none now; the other seven had none before. `A and B` rises by exactly 1512.
- **Cases with `B and not A`: none.** There is no cause to diagnose. (Where the same check showed "replacement value" on day `H1` before, the least solution now has `y` above `s_{H1}` and the operating phase starts from `y`.)
- **`best_onesided` is unchanged in every one of the 42 cases** (compare the last two columns), as is the infeasibility of `infeasible_G`. The rescued schedules are not cheaper than the optimum.
- **`G_mode_opavg`, the case where the optimum had seemed to change** (min cost `B and not A` 1.3 against `best_onesided` 1.6): the 96 schedules with reference cost below 1.6 are accepted now, and the cost from the least solution (tracked `z` and `u` with the larger `y`) is 1.6, 1.625 or 1.65 (diagnosed with `scratchpad/step3/Y2/tight/diag2.py`). The reference-trajectory cost of those schedules was an underestimate of what the solver pays: the optimum stays 1.6.
