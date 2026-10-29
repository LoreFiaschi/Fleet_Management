Status: done (29 Oct 2026). All notes placed or left out with a reason; build clean (92 pages, 0 errors, 0 undefined references or citations, 0 bibtex warnings, no Overfull, no Float too large); changed pages rendered and read; `scoping.pdf` copied into the repository. Next: the owner reads the new text.

# Reasons of WP4 to WP7 placed in the scoping document (29 Oct 2026)

Sources: `merge/spec_wp7_insertions.md` ("Notes for the scoping document", notes 1-25); `merge/wp6_plan_giw.md` sections 2 and 7 (point 3); `merge/wp6_plan_gamma.md` sections 2, 7 (Q1, Q4) and 8 (point 3); `merge/wp6_plan_rainflow.md` section 5; `merge/wp4_wp5_interface.md` last section, `merge/decisions.md` "Follow-up decisions from writing the spec" (last bullet), `merge/wp4_plan.md` sections 5 and 7; `merge/wp7_interface.md` "Decisions on the plans' questions". For detail the notes were checked against the plans' question sections (`wp7_plan_sweeps.md` 7 and 8, `wp7_plan_plots.md` 7, `wp7_plan_gamma_validators.md` 7 and 8, `wp7_plan_validator.md` 7 and 8) and against the spec on `lore`.

Section numbers are those of the built PDF: 5.2 Reasons for the input checks, 5.3 Outputs, 5.5 Warm starts, 5.6 Plots (new), 5.7 Validation tools, 5.8 Package and inputs. The cap paragraphs are in 2.2.1 (Stochastic ordering within each degradation family).

## Destinations

### 1. `spec_wp7_insertions.md`, notes 1-25

| Note | Destination |
|---|---|
| 1 Dashboard tolerance at least 1e-12 | 5.7, item "Tolerance of the dashboards" |
| 2 `main()` under `__name__ == "__main__"` | 5.7, item "Start of the dashboards". Evidence for "`streamlit run` executes the script as `__main__`": Streamlit's `script_runner.py`, read in part by the validator reviewer (`wp7_plan_validator.md` 8, C); not read here |
| 3 Wiener rule of the greedy heuristic | 5.5, Greedy heuristic, sub-item "Wiener components" (with the example of case B) |
| 4 Excess with side row and `v_max_user` | 5.5, Greedy heuristic, sub-item "Rows in the excess" (reason stated through Proposition "Domination": the tracked values are at least the reference values) |
| 5 Reference trajectory | 5.5, Greedy heuristic, sub-item "Reference trajectory" (case A given as "a Gaussian test case"). Left out: "for the redesign item of `Things to change.txt`" (planned work belongs to the to-do file) |
| 6 Completion report in mode `hint` | 5.5, item "Completion report" |
| 7 Invalid source | 5.5, item "Invalid source", stated positively; the behaviour of the old code is left out (archaeology) |
| 8 `maximum_proxy_ratio` | 5.5, Greedy heuristic, sub-item "Report" |
| 9 Last "Presolved:" line | 5.6 (new subsection), item "Counts from the Gurobi log" (the note names the sweep plot; the rule belongs to the summary panel of the schedule plot, as in the spec) |
| 10 Result without `model` | 5.6, item "Result without models" |
| 11 `TypeError` for files that hold no mapping | 5.6, item "Readers" |
| 12 Empty list `cases` | 5.6, item "Sweep without cases" |
| 13 Title of the sweep plot | left out: the record gives only the choice (a fixed name with the sizes), no reason |
| 14 `numpy.longdouble` | 5.7, Exact Gamma validator, sub-item "Precision". The rounding figures are those of `review_mass_rounding_output.txt` (2, 4, 6 repairs; 1.6e-15, 9.0e-15, 7.1e-15) |
| 15 `removed_mean` on the saved states | 5.7, item "Repair cost in the replay Gamma validator" (the claim "saved difference at least the reference difference" checked by hand: both differences are 0 after an intervention, and the carry rows only add slack) |
| 16 Initial damage split at `anchor_mu_0` | 5.7, Exact Gamma validator, sub-item "Initial damage under ARD1" |
| 17 Monte Carlo validator raises no error for a schedule violation | 5.7, item "Monte Carlo validator" (two sentences added) |
| 18 Cost of the exact tail | 5.7, Exact Gamma validator, sub-item "Cost". The term counts 3,056 and 12,426 of the note are written "about 3,000" and "about 12,400": the measured counts differ by implementation (3,055 and 12,416 in `wp7\gamma_validators\moschopoulos_fast_output.txt`, 12,425 in the review). Times 2 s and 31 s from Q6 and that output. Left out: "idea not checked: a certified split of a term of tiny mean" (planned work; it is in `Things to change.txt`, decision 3 of `wp7_interface.md`) |
| 19 Unknown keys of the block `validation` | 5.7, item "Settings of the Gamma validators" |
| 20 Margin of a row with a side row | 5.7, item "Margins in the dashboards" (three sentences added, with the example mu = 1.2) |
| 21 Dashboards raise on a different `model`, `validate` does not | 5.7, item "Model of the result"; it also covers the Gamma validators (decision 4 of `wp7_interface.md`; reason from `wp7_plan_gamma_validators.md` Q1) |
| 22 Errors of `validate` for an incomplete result | 5.7, item "Incomplete results" |
| 23 Eight `loop_q` checks always listed | 5.7, item "Fixed list of checks" |
| 24 One reader in `solver.py` | 5.3, item "One reader" |
| 25 Scope of `validate` | 5.7, item "Scope of the structural validator". Left out: "checking the dynamics rows would need a spec change" (planned work) |

### 2. Inverse Gaussian cap (`wp6_plan_giw.md` 7, point 3, and 2)

| Reason | Destination |
|---|---|
| Cantelli's inequality for the right end of the bracket (`Var Z = eta`, `Pr(Z > b) <= eps/2`) | new paragraph "Computation of the inverse Gaussian cap" (after "Inverse Gaussian: a cap on the mean"), item "Quantile"; derivation in the new appendix entry `drv:ig_bracket` (Appendix A.3, page 84). No citation: the document states Cantelli's inequality as Eq. `eq:cantelli_surrogate` without one for the inequality itself |
| The identity `E[Z; Z > q] = F(1/q)` | already in "Inverse Gaussian: a cap on the mean" with `drv:ig_tailmean`; item "Tail mean" gives its role in the computation |
| Log-space evaluation (`logaddexp`, no overflow of `e^{2/eta}`; `log1p` against the cancellation of `1 - F`) | items "Distribution function" and "Survival function" |
| Accuracy figures of section 2 | item "Accuracy of the quantile" (table: `invgauss.isf` against the root search) and item "Checks" (grid, extremes, Monte Carlo) |

### 3. Gamma cap `A_1` (`wp6_plan_gamma.md` 2, 7 Q1 and Q4, 8 point 3)

| Reason | Destination |
|---|---|
| Bracket of `A_hat`, lower end returned | paragraph "Computation of the cap", item "One-rate cap" |
| Grid of 401, extensions by 100 at most 10 times, then error | item "Grid and its extensions" (reason for the error: a maximum below the last grid would be underestimated; follows from the method) |
| 60 golden-section steps, largest value returned | item "Golden-section search" (0.618^60 = 2.9e-13 computed here) |
| Tolerances 1e-12 and 1e-9; cost 30 ms and 16 evaluations of `w` | item "Tolerances and cost" |
| `eps >= 1/2` shortcut | item "Levels of at least one half" |
| Cache per `(t, eps)` | item "Cache" |
| Reference values and checks of section 2 | item "Accuracy" (table) |
| Single peak checked for r in [1e-10, 1e-5] (section 2, "Beyond the records") | one sentence in the appendix entry `drv:gamma_computation`, fact 2 |
| Placement in `gamma.py` with the guarded import (Q1, same rule for the IG cap) | 5.8, item "Modules without Gurobi" |

### 4. Rainflow (`wp6_plan_rainflow.md` 5)

| Reason | Destination |
|---|---|
| `rainflow.py` loads without gurobipy: `row_constants` needs no Gurobi object | 5.8, item "Modules without Gurobi" |
| `ctx.extras[self.name]`, `# Eq.` comments | left out: style only |
| Row names `rel`, `rel_side`, `v_max_user` (sentence 3, optional) | left out: the spec does not name the rows, so there is no choice of the specification to justify |
| "`load_config` computes b once per Bernstein cell" (sentence 1) | left out: no reason recorded |
| Builder errors not needed in the spec (`load_config` rejects first) | left out: concerns what the spec lists, not a design choice |

Section 4 of the same plan records two further reasons (no tightening of the variable bounds, Q1; the redundant rows are built, Q3). They are outside the brief (section 5 only) and are not placed.

### 5. WP4 choices (`wp4_wp5_interface.md`, `decisions.md`, `wp4_plan.md` 5 and 7)

| Choice | Destination |
|---|---|
| `null` at the three keys without a default | 5.2, "Keys, shapes and entries", new item |
| Relative tolerance `1e-12·max(1, |a|, |b|)`, example 5·0.14 | 5.2, new paragraph "Tolerance of the checks", item 1. The reason for "relative" (the rounding error grows with the size of the numbers) and for the term 1 is Claude's (a standard floating-point fact); the records give only the user's choice |
| Strict inequalities without tolerance | same paragraph, item 2 (reason by Claude, from the existing check `mu_0 < tau`) |
| Selector values case sensitive (Q6) | 5.2, "Keys, shapes and entries", new item (old inputs write `ARD1` for ARD_inf: spec, "Rejected Keys", row `maintenance_type`) |
| Integers: 3.0 accepted, 2.5 and a flag refused (Q9) | 5.2, "Keys, shapes and entries", new item |
| `NaN` and `±inf` as `null` in YAML and JSON (Q4) | 5.3, item "Numbers that are not finite" |
| `objective = None` not stored in HDF5 (Q8) | 5.3, item "Objective without a value" |
| `mip_gap` clamp (`wp4_plan.md` 7) | 5.2, "Tolerance of the checks", item 3 |

### 6. `wp7_interface.md`, decisions

| Decision | Destination |
|---|---|
| 1. One reader `solver.read_result` | 5.3, item "One reader" (with note 24) |
| 2. Wiener rule of the greedy heuristic | 5.5, sub-item "Wiener components" (with note 3) |

## Changelog bullets (paragraph "Version 1.0 (merged package)")

1. The input checks give every non-strict inequality the relative tolerance 1e-12·max(1, |a|, |b|), and they compare selector values with their case. At a key without a default, `null` means that the key is not given at that cell. YAML and JSON result files write every number that is not finite as `null`.
2. The specification fixes the numerical methods of the two caps: a root search in ln q for the inverse Gaussian quantile, and the grid, the golden-section search and the two bisections of the Gamma cap A_1.
3. One function, `read_result`, reads the result files for every tool. The greedy heuristic tries a repair and a replacement on every free day of a Wiener cell.

## Build

- `merge/scoping_build.sh` into `scratchpad\scobuild_notes\`, behind the guard (no LaTeX process was running at any build).
- Baseline before the edits (`scratchpad\scobuild_notes_base\`): 87 pages, clean.
- Final build: 92 pages; 0 errors; 0 undefined references or citations; bibtex 0 warnings; 0 Overfull; 0 Float too large; 89 Underfull (89 in the baseline). Three builds in all; the second and third followed rewording of long sentences. No build had an Overfull box.
- `scoping.pdf` of the final build copied over `C:\Users\utente\Documents\GitHub\Fleet_Management\scoping.pdf`.
- `scoping.tex` keeps LF line endings (0 CR characters) and stays ASCII.

## Pages looked at (70 dpi, `scratchpad\scopage_notes\`)

- First build: 42, 43, 44, 45, 63, 64, 65, 66, 67, 68, 69, 70, 74, 84, 87.
- After the rewording: 42, 43, 45, 67, 68, 69, 70, 73, 74; after the last split of sentences: 43, 68, 69.
- Page 64 holds no changed text (looked at for the neighbouring list). The table of item "Accuracy of the quantile" starts at the top of page 43, below its explanation at the end of page 42.

## House form

- Every new item has a bold lead-in, except the items of 5.2 (they follow that subsection's form "rule --- reason") and the changelog bullets (that list has no lead-ins). Bold lead-ins were added to the three existing sub-items of "Exact Gamma validator" (Mixture, Certified bound, Saved means), with their text unchanged, so that the sub-list has one form.
- New prose paragraphs: two short introductions (about 20 and 10 words). No new paragraph over 80 words.
- New displays (appendix `drv:ig_bracket`) are numbered `equation`s. Two tables added in lists (`center`, `\small`, booktabs, as the existing table of A_1/A_hat).
- Tell-words checked on the added lines: none of "now", "earlier", "no longer", "corrected", "was described as", "is wrong".
- Terms defined at first use: SciPy, golden-section search, HDF5, YAML and JSON, NaN, NumPy, Streamlit, free day, convolution tolerance.

## Word counts (source words, comments removed; script `scratchpad\scobuild_notes_wc.py`)

| Part | Before | After |
|---|---|---|
| IG cap paragraphs (with the new "Computation of the inverse Gaussian cap") | 682 | 1085 |
| Gamma: Computation of the cap | 45 | 583 |
| Reasons for the input checks | 1288 | 1629 |
| Outputs | 189 | 357 |
| Horizon sweeps (unchanged) | 368 | 368 |
| Warm starts | 339 | 742 |
| Plots (new subsection) | 0 | 193 |
| Validation tools | 284 | 1068 |
| Package and inputs | 427 | 518 |
| Changelog Version 1.0 | 222 | 339 |
| Appendix `drv:ig_bracket` (new) | 0 | 91 |
| Appendix `drv:gamma_computation` | 193 | 209 |
| Whole file | 41173 words, 5044 lines, 87 pages | 44318 words, 5244 lines, 92 pages |
