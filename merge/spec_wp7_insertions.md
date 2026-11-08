Status: done. All sentences of the four reviewers are in the fragments: 54 units (11 sweeps, 15 plots, 21 Gamma validators, 7 validator), of which 53 inserted or replaced, 1 merged, 0 not inserted, 0 conflicts; `spec/spec.tex` and `merge/spec_drafts/build_final/spec.tex` installed; build clean (54 pages, 0 errors, no undefined reference, no Overfull, no Float too large); 22 pages rendered and read. Next: the owner reads this file; the notes below go to the scoping document.

# Spec insertions of the WP7 reviewers (29 Oct 2026)

Sources: `merge/wp7_plan_sweeps.md` section 8 (and 7.1), `wp7_plan_plots.md` section 8, `wp7_plan_gamma_validators.md` section 8, `wp7_plan_validator.md` section 8 items 1 to 7.
Fragments edited: `merge/spec_drafts/fragments_A.tex` (Encoding of the Dynamics), `fragments_C.tex` (Output Specification), `fragments_D.tex` (Horizon Sweeps, Warm Starts, Plotting, Validation Tools). `fragments_B.tex` and `claude_full_problem.tex` needed no change.
The sentences were located by their text, not by the line numbers of the worktree spec. Each is inserted verbatim unless a row says otherwise.

## 1. Sweeps (`wp7_plan_sweeps.md`, 11 sentences, and section 7.1)

| # | Sentence | Fragment, section | Result |
|---|---|---|---|
| 1 | Wiener rule of the greedy heuristic | D, Greedy Heuristic, "Steps of the loop pass", between the end of the list and "A start with failures is returned." | inserted; the same sentence of section 7.1 is merged with it (inserted once) |
| 2 | Reference trajectory $\zeta$ | A, Encoding of the Dynamics, "Rules for both forms", new last item | inserted; the citations in the greedy step 1 and in the replay validator step 2 already point to this section |
| 3 | Excess: side-row item and the parenthesis on the `pwl` cap | D, Greedy Heuristic, loop pass step 2 | inserted; the previous item ends with `;` instead of `.` so that the list reads on |
| 4 | Completion "Report" item | D, Warm Starts, Start, Hints and Completion, "Completion" list, after "Time limit" | inserted |
| 5 | "Invalid source" item | D, Warm Starts, Interface and Reports, after "Source without x" | inserted |
| 6 | `values` of `warm_start` | C, Reports, block `warm_start` | replaced the cell "Number of values set" |
| 7 | `maximum_repeatability_excess`, `maximum_proxy_ratio` | D, Greedy Heuristic, Report keys | replaced both cells |
| 8 | `last_completed_H2` | D, Report, Checkpoint and Errors, Report keys | replaced the cell |
| 9 | `absolute_mip_gap` | D, same section, Keys of a case row | appended |
| 10 | `evaluation_horizon` rule of `sweep_operating_horizons` | D, Horizon Sweeps, Arguments | replaced the cell |
| 11 | Sizes kept in an error row | D, same section, bullet "Case error, row" | appended after "except the sizes" |

## 2. Plots (`wp7_plan_plots.md` section 8, one block)

| # | Sentence | Fragment, section | Result |
|---|---|---|---|
| 1 | Row `TypeError` of the schedule plot | D, Schedule Plot, Errors | inserted after `PermissionError` |
| 2 | `ValueError` row of the schedule plot ends with `component_names ... not positive` | D, same table | replaced the last clause "`component_names` has a length other than $L$" |
| 3 | Row `TypeError` of the sweep plot | D, Sweep Plot, Errors | inserted |
| 4 | `ValueError` row of the sweep plot: "the reports hold no case; a case has no `H2`" | D, same table | appended to the last clause |
| 5 | Reading rule: the file is read with `solver.read_result` | D, Schedule Plot, Reading rules | merged with validator item 1 (the "Reader" bullet defines the reader). The rule keeps only "The result file is read with `solver.read_result` (Section 9.4)"; the clause "a missing key is null, and a null entry of an array is NaN" is in the Reader bullet |
| 6 | Reading rule: $T$ is the length of the last axis of `mu` | D, same list | inserted |
| 7 | Reading rule: the model of each row, "unknown" without the key | D, same list | replaced the model item |
| 8 | Order of the Reading rules | D, same list | the reviewer says "two new items first; the item on the model becomes the third". The list is now: read_result, $T$, model, then the three old items (any result with a solution, missing `m` and `r`, default component name) |
| 9 | Summary panel, first item (status colours) | D, Summary panel | replaced "status, coloured by outcome;" |
| 10 | Summary panel, last item | D, Summary panel | replaced; ", counted over all cells and days" added |
| 11 | Source of the counts: log with several solves | D, Summary panel | appended to the paragraph "Source of the counts" |
| 12 | Sweep plot, Selection | D, Sweep Plot, Content | replaced |
| 13 | Sweep plot, Sizes | D, same list | inserted |
| 14 | Sweep plot, Title | D, same list | replaced "The title names the sweep and $F$, $M$, $L$, $H_1$" |
| 15 | Panels (a) and (b) | D, Sweep Plot, panel table | (a): last sentence replaced by the two sentences; (b): row replaced |

The plot's definition of a feasible case (`J_op_average` not `null` and a status outside three values) is not the definition in "Cost Measures and Ranking" (a result with a solution and a status outside the same three values). The plot reads report rows, not results, so both are kept; no contradiction.

## 3. Gamma validators (`wp7_plan_gamma_validators.md` section 8, one block)

All in `fragments_D.tex`, "Validators of the Gamma Model".

| # | Sentence | Place | Result |
|---|---|---|---|
| 1 | Gamma cells; event of a day; indices; order of a file version (4 items) | Common rules, after "Tolerance" | inserted |
| 2 | `ValueError` row of the common rules | Errors | replaced |
| 3 | "over every cell of the fleet, one string per occurrence" | Replay validator, step 1 | inserted after "(strings in `global_violations`)" |
| 4 | "starting at $\mu_0$ and, under $\ARDone$, at the starting anchor ..." | Replay validator, step 2 | replaced "with the initial value, starting anchor and replacement value of the cell" |
| 5 | `removed_mean`: "on the saved `mu` and `g_mu` (on day 1: ...)" | Error fields | inserted after the formula; its comma replaces the old one |
| 6 | Meaning of `expected` and `saved` in a record | after the table of error fields | inserted |
| 7 | Frozen and repairable parts of the initial damage | Exact validator, step 1 | inserted after the sub-list |
| 8 | Tail with a constant $c$ | Exact validator, step 2 | replaced "and $1$ if $c > \tau$"; the comma before it became a semicolon |
| 9 | `numpy.longdouble` sums | Exact validator | inserted as step 5 |
| 10 | `valid` and the records of the exact validator | after the pass rule | inserted |
| 11 | Rows "further keys" and `steps` | Exact validator, Report keys | replaced by the two rows |
| 12 | Integer rule of `repetitions`, `random_seed`, `batch_size` | Monte Carlo validator, Further arguments | appended to three rows; for `repetitions` and `batch_size` I wrote "must be positive **and** an integer; ..." (one joining word added so that the sentence reads on) |
| 13 | No error for a global violation; arguments used | after the steps of the Monte Carlo validator | inserted |
| 14 | "Reports" item of the dispatcher | Dispatcher | inserted after "Default" |
| 15 | `ValueError` row of the dispatcher | Errors | replaced |

Overlaps kept, because the statements name different tools: the clause "an input without a Gamma cell raises a `ValueError`" appears for the validators (item 2, `cfg` has no Gamma cell) and for the mode "Gamma diagnostic" (validator item 7); the clause "a result whose `model` differs from the input raises a `ValueError`" appears for the Gamma validators (item 2) and for both dashboard modes (validator item 4).

## 4. Validator and dashboards (`wp7_plan_validator.md` section 8, items 1 to 7)

| # | Item | Fragment, section | Result |
|---|---|---|---|
| 1 | "Reader" bullet | C, Result Files, new last bullet | inserted verbatim |
| 2 | Clause on the `ValueError` row of `validate` | D, Structural Validator, Errors | appended |
| 3 | Order of the checks, the eight `loop_q` checks, `check_result` | D, Structural Validator | inserted as a paragraph after "The recomputed objective is ..." |
| 4 | "Loading" becomes three bullets | D, Dashboards | inserted. Reasons moved out (notes 1 and 2). In "Code" the two lists of functions are two sentences (one per module), because the one-line form made an Overfull box |
| 5 | Margin bullet and margin table | D, Dashboards, Mode "Result inspector" | inserted; all nine labels exist; the table has no label because no sentence refers to it with `\ref` |
| 6 | Paragraph after the margin table | D, same place | inserted |
| 7 | Paragraph of the mode "Gamma diagnostic" | D, after the tab list | inserted |

## Notes for the scoping document

Reasons removed from the spec sentences (each rule stays in the spec):

1. Dashboard tolerance, "at least $10^{-12}$": the Gamma validators need a positive tolerance (validator item 4).
2. `main()` of `validator_dashboard.py` runs under `if __name__ == "__main__"`: `streamlit run` executes the script in a module named `__main__`, so the page draws (validator item 4; evidence quoted by the reviewer: Streamlit's `script_runner.py`, `self._new_module("__main__")` and `exec(code, module.__dict__)`, read in part). In the spec the clause "so every module loads without it" became the fact "every module loads without it".
3. Wiener rule of the greedy heuristic (sweeps sentence 1, section 7.1): the Wiener variance falls only at a replacement (spec, Wiener Model), so a repair placed first can take the last free day of a cell that needs a replacement. Example of the reviewer: case B (Wiener, `H = [2, 2]`): a repair on day 4 removes the mean excess and uses the only free day; the variance excess stays, one failure per train; a replacement on that day removes both.

Reasons and choices that the reviewers record next to their sentences (the spec states only the rule):

4. Excess with the side row and `v_max_user` (sweeps Q3): a start that breaks $v \le V^{\mathrm{user}}_{\max}$ can never be completed.
5. Reference trajectory (sweeps Q6): scoping line 387 defines it; the test is stricter than the one-sided rows. Case A fails it, yet its schedule is feasible for the real rows (`verify_greedy_lp.py`). For the redesign item of `Things to change.txt`.
6. Completion report (sweeps Q2): in the mode `"hint"` the completion never runs, but the greedy source requests it; no listed status fits, so `requested` is the flag of the source and the status is `"not_requested"`.
7. Invalid source (sweeps Q4): before the sentence the code took the values of the model; now `ValueError`.
8. `maximum_proxy_ratio` (sweeps Q7): the final trajectory of each cell, which is the returned schedule.
9. Sweep plot, last "Presolved:" line (plots Q1): the warm-start completion and the relaxed copy are solved before the main model in the same log, so the last "Presolved:" line is the main solve.
10. A result without `model` plots with the label "unknown" (plots Q2): old result files lack the key; `solve` always writes it.
11. `TypeError` for non-mapping files (plots Q3): one reader for every input path.
12. Empty list `cases` is an error (plots Q4): there is nothing to plot; a sweep writes its first report after its first case.
13. Title of the sweep plot (plots Q5): a fixed name with the sizes.
14. `numpy.longdouble` of the exact validator (Gamma review): 80 bits on Linux and 64 bits on Windows; float64 understates the remaining mass by $2 \cdot 10^{-15}$ to $9 \cdot 10^{-15}$ at 151 to 3,055 terms, 1 % of the default `convolution_tolerance`, far below the pass tolerance $10^{-8}$; kept for parity on Linux; no floor on `convolution_tolerance`: an unreachable value ends in the `RuntimeError`.
15. `removed_mean` on the saved states (Gamma Q2): it is the row of the model, and at binary points the saved difference is at least the reference difference under both laws, so it implies the other check.
16. Initial damage under $\ARDone$ splits at `anchor_mu_0` in the exact validator (Gamma Q3): otherwise a repair could go below the level of the last intervention.
17. Monte Carlo validator raises no error for a schedule violation (Gamma Q4): the mode `"both"` reports the schedule through the replay validator.
18. Cost of the exact tail (Gamma Q6): the series length grows with the ratio of the largest to the smallest rate, which frequent $\ARDinf$ repairs make large: 2, 4, 6, 8, 10 repairs of one term at $\rho = 0.5$ give 151, 725, 3,056, 12,426, 49,884 terms; from 11 repairs more than 100,000 terms, so the `RuntimeError`. Idea not checked: a certified split of a term of tiny mean.
19. Unknown keys of the input block `validation` are an error (Gamma Q9): `config.py` checks only that it is a mapping, so a typo such as `repetition` would be ignored.
20. Margin of a row with a side row (validator Q1): the row holds if and only if both relations hold, so the margin is the smaller of the two slacks. Example: G with $\mu = 1.2$, $\tau = 1$, $v = 0$ gives $(\tau - \mu)^2 = 0.04 > 0$ although the side row fails. The two slacks have different units, so only their sign is compared across cells.
21. The dashboards raise `ValueError` when `model` differs from the input, `validate` does not check it (validator Q2): `validate` needs no model; the margins need the input's parameters of that model.
22. Errors of `validate` for a result with `x` but without `m`, `r`, `mu`, `z`, `u`, `objective` (validator Q3): such a result cannot come from `solve`; without an error the report would hold `NaN`, and the JSON writer refuses `NaN`.
23. The eight `loop_q` checks are always listed (validator Q4): the report has a fixed list of 14 checks, 13 without `idle`.
24. One reader of result files in `solver.py` (validator Q6): the plotter, the Gamma file versions and the dashboards need the same reader; it sits next to the writers.
25. Scope of `validate` (validator Q8): the loop rows of every tracked state and anchor, not the dynamics rows nor the reliability rows; the reliability rows appear as the margins of the dashboard. Checking the dynamics rows would need a spec change.

Total: 25 notes (3 reasons removed from sentences, 22 choices recorded by the reviewers).

## Build and pages

- Assembled with `python assemble.py`, installed with CRLF into `spec/spec.tex` and `merge/spec_drafts/build_final/spec.tex`, built three times with `pdflatex` (guard: no LaTeX process was running).
- Log: 54 pages, 0 errors, no "undefined", no "Overfull", no "Float too large". One Overfull box appeared in the first build (the "Code" bullet of the dashboards); the sentence was reworded and the build repeated.
- 22 pages rendered at 70 dpi and read (physical PDF pages; the printed number is one less): 18 (reference trajectory), 25, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51. Pages 25, 37 and 40 hold no changed text (looked at to check the neighbouring text). Tables, the margin table and the formulas are not broken; the panel table of the sweep plot continues on page 44 with its header repeated.
- Images: `C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\specwp7\specpage-NN.png`.
