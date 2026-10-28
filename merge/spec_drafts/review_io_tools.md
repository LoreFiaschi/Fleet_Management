Status: done (26 Oct 2026). All sections in scope read in full; `main`'s code compared; YAML example recomputed independently (it passes every check, under the reading of item 3 below). Must fix: 19. Suggestions: 8.

# Review of spec version 1.0: inputs, outputs and tools

Reviewer: Claude (subagent), 26 Oct 2026. File: `merge/spec_drafts/spec_assembled.tex` (4440 lines). Line numbers refer
to that file. Scope: Input Specification (lines 258-685), Output Specification to Open Questions (lines 2883-4307).
Independent check script and output (scratchpad, session-scoped):
`C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\review_IO\review_checks.py`
(`review_checks_output.txt` beside it).

## Must fix

Ranked by severity, most severe first.

### 1. `validate` drops the damage penalty in the mode `evaluation_total`

- **Location.** Validation Tools, Structural Validator, lines 3870-3873.
- **Problem.** The recomputed objective for `evaluation_total` is `P_E` alone. The solve minimises `P_E + C_D u`
  (Eq. a_objective, lines 763-764). The result's `objective` equals `evaluation_objective = P_E + C_D u` (lines 2934,
  2996). So the check `objective` fails every such result with `C_D u > 0`.
- **Evidence.** Lines 763-764, 2934, 2996; the other two modes at lines 3871-3872 do add `C_D u`.
- **Fix.** Replace "for \texttt{evaluation\_total} it is the projected cost $P_E$ of Eq.~\eqref{eq:d_projected}
  (Section~\ref{sec:objective})." by "for \texttt{evaluation\_total} it is $P_E + C_D\,u$, with $P_E$ the projected cost
  of Eq.~\eqref{eq:d_projected} (Section~\ref{sec:objective})."

### 2. The Bernstein tangent check is necessary but not sufficient

- **Location.** Consistency Checks, Rainflow cells, line 667. The same claim stands at lines 1378-1380 (Linear Forms)
  and 2525-2530 (Rainflow); those are outside this scope but must change with it.
- **Problem.** For `mu_r < tau - c` the tangent row decreases in `mu`, so its largest value on `[0, tau - c]` is at
  `mu = 0`. By Proposition prop:a_tangent, `T(0) = f(0) - c_2 mu_r^2 = (tau(tau - 2c) - mu_r^2)/(2 kappa)`. So the row
  admits a point with `v >= 0` if and only if `mu_r <= sqrt(tau(tau - 2c))`, and `sqrt(tau(tau - 2c)) < tau - c`
  always. Every `mu_r` in between passes the check and leaves an empty row: the model is infeasible without a message.
  The `pwl` form has the same defect at its first segment (`P = 1` is the tangent at `tau/2`).
- **Evidence.** `tau = 1`, `eps = 0.1`, `b = 0.3`, so `c = 0.2303`, `tau - c = 0.770`, `sqrt(tau(tau - 2c)) = 0.735`.
  `tangent_ref = 0.75` passes line 667; the largest right side on `[0, tau - c]` is `-0.005`. Over 400 random
  `(tau, eps, c)` (1200 tangent points), the proposed rule matched a grid search at every point; the rule of line 667
  accepted 757 points whose row is empty (script, part 1). `pwl` with `P = 1`, `c = 0.4`: empty, and line 667 does
  not apply to `pwl`. In the YAML example the gap is `(0.9608, 0.9616)`, so writer C's boundary case
  `tangent_ref = 0.95` did not reach it.
- **Fix (line 667).** "Bernstein with the form \texttt{tangent}: $\texttt{tangent\_ref}\cdot\tau_{i\ell} <
  \sqrt{\tau_{i\ell}\,(\tau_{i\ell} - 2c_{i\ell})}$, with $c_{i\ell} = \kappa\, b_{i\ell}/3$; with the form
  \texttt{pwl}: $\tau_{i\ell}/(2P) < \sqrt{\tau_{i\ell}\,(\tau_{i\ell} - 2c_{i\ell})}$. --- The row decreases in $\mu$,
  and at $\mu = 0$ it reads $v \le f(0) - c_2\,\mu_r^2 = \bigl(\tau(\tau - 2c) - \mu_r^2\bigr)/(2\kappa)$
  (Proposition~\ref{prop:a_tangent}). Otherwise no point with $v \ge 0$ satisfies the row, or only $\mu = v = 0$."
  Lines 1378-1380 and 2525-2530: the same condition. The sentence at 2527-2530 stays true for `mu_r >= tau - c`.

### 3. The selector check contradicts the rule for absent entries

- **Location.** Input Specification, "Absent entries", lines 282-285; Consistency Checks, line 608; YAML example,
  lines 4150-4153.
- **Problem.** Line 608 requires `reliability_impl` to be `exact`, `tangent` or `pwl` "at every cell", and an unknown
  value to be an error "at every cell". Line 282 allows `null` where the key does not apply. Line 285 says a non-null
  entry there is "ignored with a `UserWarning`". The example writes `null` at the IG, Gamma and Chernoff cells, so it
  fails line 608 as written. An unknown value at an IG cell is both ignored (285) and an error (608).
- **Evidence.** Writer C's checker treats an unknown value at an IG cell as an error and accepts `null` there
  (`checks/C_checks_output.txt`), which is the decided rule ("an unknown value is an error everywhere").
- **Fix.** Line 608: "Every non-\texttt{null} entry of \texttt{reliability\_impl} is \texttt{exact}, \texttt{tangent}
  or \texttt{pwl}, also at a cell whose row is linear (IG, Ga, Mk, Ch); there the value is ignored and the form
  \texttt{exact} is built." Line 285, new last sentence: "A selector entry at such a cell is still checked against the
  values of its key (Section~\ref{sec:consistency_checks})." Two related sentences of the schema: line 4056 "the
  entries of the other cells are \texttt{null}" becomes "may be \texttt{null}"; line 4099 "read at every cell" is false
  for \texttt{tangent\_ref} and \texttt{pwl\_points}, which only cells with that form read.

### 4. The cost checks are missing from the check list

- **Location.** Consistency Checks, lines 587-676. Related: Costs table, lines 511-520; Damage Regularisation,
  lines 2638-2640.
- **Problem.** The decided checks "finite, non-negative costs" and `C_D >= 0` exist only in the input table. The check
  list has no cost block. Line 2638 says "The input check requires $C_D \geq 0$ (Section consistency_checks)", which
  points to a check that is not there. A negative `C_D` rewards damage.
- **Evidence.** decisions.md, item "Input checks" (kept from `main`: finite, non-negative costs; `C_D >= 0`); item
  "Cost shapes"; the section must hold "the consistency-check list, with these values".
- **Fix.** New block after "Run options": "\textbf{Costs.} \begin{itemize}[leftmargin=1.5em] \item $C_M$, $C_D$,
  $C_{R,i\ell}$ and $C_{\mathrm{rep},i\ell}$ are finite and $\ge 0$. --- A negative cost would reward a depot day, a
  repair, a replacement or damage. \end{itemize}"

### 5. No Jensen check on the starting anchors of a Chernoff cell

- **Location.** Consistency Checks, line 669 (Chernoff, Jensen) and line 620 (starting anchors).
- **Problem.** Under `ard1`, `anchor_mu_0` and `anchor_K_0` describe the same damage `G` after the last intervention.
  Jensen gives `K(G) >= s E[G]`, so `anchor_K_0 < s * anchor_mu_0` is a contradiction. The `K` map of
  Proposition prop:rf_maps (item 3, line 2318) needs `g^K >= K(G)`; with such an input it is not an upper bound, and
  the Chernoff certificate turns optimistic, as in the `K_0 = 0` example at lines 2371-2377.
- **Evidence.** Lines 2317-2318, 2339-2342; decisions.md, item "Names and set of repair laws" ("`gK >= K_G(s)`").
  The YAML example sits on the boundary: `anchor_K_0 = 0.50 = 5 * 0.10`.
- **Fix.** Line 669: "... $K_0 \ge s\,\mu_0$, $K^{\new} \ge s\,\mu^{\new}$ and, at \texttt{ard1} cells, $g^{K}_{0} \ge
  s\,g^{\mu}_{0}$. --- Jensen's inequality: ... The two starting anchors describe the same damage after the last
  intervention." The defaults satisfy it, since $K_0 \ge s\mu_0 \ge s\,g^{\mu}_0$.

### 6. The replacement-state check names neither the form nor the other rows

- **Location.** Consistency Checks, line 627; also line 660 (`v_max_user`).
- **Problem.** "Satisfies the reliability row of the cell" does not say in which form. At a `tangent` or `pwl` cell the
  built row is stricter (Proposition prop:a_tangent). A replacement state can pass the exact row and still be
  infeasible, which is what the check exists to catch. The Bernstein side row and the row `v <= V^user` also hold on a
  replacement day; line 660 checks `v_max_user` against `v_0` only.
- **Evidence.** Lines 1403-1416, 2470, 493. Writer C's checker tests the exact form only (`checks/C_checks.py`,
  lines 553-580).
- **Fix.** Line 627: "If \texttt{allow\_replacement} is \texttt{true}, the replacement state of every cell satisfies
  every row that holds on a replacement day, in the form built at the cell: the reliability row in the form of
  \texttt{reliability\_impl} (for \texttt{pwl}, the cap of the segment that contains $\mu^{\new}$), its side row, and
  $v^{\new} \le V^{\mathrm{user}}_{\max,i\ell}$ where \texttt{v\_max\_user} is given." Line 660: "... $\ge v_{0,i\ell}$
  and, if \texttt{allow\_replacement} is \texttt{true}, $\ge v^{\new}_{i\ell}$."

### 7. The input block `validation` of `main`'s Gamma validators is neither kept nor rejected

- **Location.** Input Specification, "Keys", lines 287-290; Rejected Keys, lines 551-585; Validators of the Gamma
  Model, line 4041.
- **Problem.** On `main`, an argument of `validate_gamma_schedule` that is `None` takes its value from the top-level
  input block `validation` (mode, repetitions, random_seed, maximum_schedule_failure_rate, confidence_level,
  batch_size), then the default. The spec lists `validation` in no table. Two tracked inputs of `main` carry it, so they
  stop at a bare "unknown key". Line 4041 drops the route without a decision; the policy keeps `main`'s features.
- **Evidence.** `main:src/fleet_management/degradation_model/gamma_utils/gamma_stochastic_validator.py:71-130`
  (`cfg.raw.get("validation", {})`); `main:examples/regression/matched_mixed_ard1.yaml:77`,
  `matched_mixed_ardinf.yaml:77`.
- **Fix (keeps the feature).** Run Options table, new row: "\optrow{\texttt{validation}}{mapping}{none}{Settings of the
  Gamma validators (Section~\ref{sec:validation_gamma}): \texttt{mode}, \texttt{repetitions}, \texttt{random\_seed},
  \texttt{maximum\_schedule\_failure\_rate}, \texttt{confidence\_level}, \texttt{batch\_size}. \texttt{solve} does not
  read it}". Line 4041: "An argument that is \texttt{None} takes the value of the input block \texttt{validation}, and
  otherwise the default of the Monte Carlo validator." Alternative: a rejected-key row "Pass the settings as arguments
  of \texttt{validate\_gamma\_schedule}". Either choice needs the user's confirmation.

### 8. The replay validator rejects valid incumbents and has no rule for `z = NaN`

- **Location.** Validators of the Gamma Model, replay validator, field `removed_mean`, line 3953.
- **Problem.** The field is two-sided and requires `z = 0` on other days. Section repair_cost allows a non-optimal
  solution to carry a larger `z` (lines 2577-2579), and every other field of the table is one-sided. So a valid
  incumbent fails. At a cell with `C_R = 0`, `z` is `NaN` (line 2924), and the field is undefined.
- **Fix.** "$\max\{0,\ x^m_{i\ell k}\,\rho_{i\ell}(\mu_{i\ell,k-1} - g_{i\ell,k-1}) - z_{i\ell k}\}$ computed from the
  saved states, with $g = 0$ under $\ARDinf$ (Section~\ref{sec:repair_cost}): a $z$ below the removed amount
  understates the repair cost. A larger $z$ is slack and is not an error. The field is skipped where $z$ is
  \texttt{NaN} ($C_{R,i\ell} = 0$)."

### 9. The reported `u` makes the `objective` check fail on incumbents

- **Location.** Output, lines 2973-2979 and 2995-2996; Structural Validator, checks `damage_rows` and `objective`,
  lines 3860-3865, 3870-3871.
- **Problem.** The result reports `u` recomputed as a maximum (Eq. c_u), not the value of the variable. The objective
  uses the variable, which only satisfies `u >= sum mu` and can exceed the maximum in a non-optimal incumbent. Then
  line 2996 ("it equals ...") is false, and `validate` fails a valid result. With the recomputed `u`, the check
  `damage_rows` can never fail.
- **Evidence.** Rows a_u_inf, a_u_quad (lines 2647, 2653); `main` recomputes `u` "independent of possible numerical
  slack in the epigraph variable" (`main:src/fleet_management/degradation_model/base.py:1906-1912`).
- **Fix.** Line 2996: "When the variable $u$ has no slack, it equals, up to the tolerance of the solver, ...;
  otherwise it is larger by $C_D$ times that slack." Check `objective`: "violation $\max\{0,\
  \texttt{recomputed\_objective} - \texttt{objective}\}/\max\{1, |\texttt{objective}|\}$: the reported objective is not
  below the cost of the reported solution". Remove the check `damage_rows`. Alternative: store the value of the
  variable `u` in the result under its own key, and check both rows and objective against it.

### 10. The quadratic-penalty check reads as an override, not an error

- **Location.** Consistency Checks, line 609; Costs table, line 515; Damage Regularisation, lines 2657-2660.
- **Problem.** Line 609 says "every cell is built in the form exact", an automatic override. Lines 515 and 2658 say the
  penalty "requires" the exact form, and writer C's checker raises an error. An implementer cannot tell which.
- **Fix.** Line 609: "If $C_D > 0$ and \texttt{penalty\_type} is \texttt{quadratic}, \texttt{reliability\_impl} is
  \texttt{exact} at every cell with a quadratic reliability row (G, W, and RF with Ca, Ho or Be); another value is an
  error. --- The quadratic penalty is a quadratic row, and the linear forms exist to give a linear model."

### 11. The day cost of the output multiplies by `NaN`

- **Location.** Output, Eq. c_step, lines 2989-2991; Structural Validator, line 3870.
- **Problem.** `z` is `NaN` in memory at cells with `C_R = 0` (line 2924). Eq. c_step multiplies `C_R` by `z` with no
  rule; in NumPy `0 * NaN` is `NaN`, so `step_costs` and every phase cost become `NaN`. Eq. a_daycost (line 750) and
  Eq. d_daycost (line 3066) state that `z` reads 0 where it is not built; Eq. c_step does not.
- **Fix.** After Eq. c_step: "where $z_{i\ell k}$ reads $0$ at the cells where it is \texttt{NaN}
  ($C_{R,i\ell} = 0$)." Line 3870: "with the saved $x$, $z$ ($0$ where \texttt{NaN}), $x^r$ ...".

### 12. The completion does not always leave a continuous copy

- **Location.** Warm Starts, Start, Hints and Completion, steps 1-2, lines 3417-3418.
- **Problem.** "All binary values are fixed, so the copy has only continuous variables" is false in three cases. The
  segment binaries `w` of the `pwl` form get no value (Section warm_map maps `x`, `x^m`, `x^r`, `n` only). A source
  without `r` leaves `x^r` free. A source with fewer trains or components leaves part of the binaries free. The copy is
  then a mixed-integer program. `main` also runs the completion only when the mapping set a value.
- **Evidence.** `main:src/fleet_management/degradation_model/base.py:1494` (`not applied` returns) and 1502-1513 (only
  integers with a `Start` are fixed).
- **Fix.** "2. Optimise the copy within \texttt{completion\_time\_limit} seconds. Binary variables without a
  \texttt{Start} value (for example the segment binaries $w$ of the \texttt{pwl} form) stay free, so the copy can be a
  mixed-integer program." Before the steps: "The completion runs only when the mapping set at least one value."

### 13. Result files omit the keys that the readers test for `null`

- **Location.** Result Files, line 3040; readers at lines 3369 (warm-start source) and 3837 (`validate`).
- **Problem.** A file omits every key whose value is `None`, except `objective`. A saved result without a solution has
  no key `x`, `mu` or `mip_gap`. These two readers speak of "`x` is `null`" (the plot and the Gamma validators already
  say "missing"). A reader that indexes `result["x"]` raises `KeyError` instead of the specified error.
- **Fix.** Line 3040, new sentence: "Every reader of a result treats a missing key as \texttt{null}." Lines 3369 and
  3837: "\texttt{x} is missing or \texttt{null}".

### 14. `gamma_beta` reads lists and matrices differently from the other profiles

- **Location.** Increment Profiles, lines 440-441, 473, 475.
- **Problem.** `gamma_beta` accepts `(L)` and `(F, L)`, not `(M)` and `(L, M)`. The other keys of the same table read
  a list per mission and a matrix as `(L, M)`. With `M = L` (as in the YAML example), `gamma_beta: [10, 20]` is per
  component while `mu: [0.1, 0.2]` is per mission. With `F = L = M`, an `(L, M)` matrix of rates is read as `(F, L)`
  without an error. Lines 440-441 list the shape ambiguities but miss this one.
- **Evidence.** `main:src/fleet_management/config.py`, `_gamma_rate_prof` and `_flmh_prof` (the two shape lists).
- **Fix.** After line 441: "The keys \texttt{gamma\_beta} and \texttt{gamma\_beta\_trans} read a list as one rate per
  component and a two-dimensional array as one rate per cell, unlike the other profiles. Rates per mission are
  written with the shape $(F, L, M)$."

### 15. The sweep section defines `u` for `inf_norm` only

- **Location.** Horizon Sweeps, line 3074.
- **Problem.** "The peak damage is $u = \max_{i,k}\sum_\ell \mu_{i\ell k}$" holds only for `inf_norm`. Under
  `quadratic` the result's `u` and `peak_damage` are `max sum mu^2` (Eq. c_u).
- **Fix.** "The peak damage $u$ is the value of Eq.~\eqref{eq:c_u}, and the damage penalty is $C_D\,u$."

### 16. The plot reads a key that no section defines

- **Location.** Plot Layout, summary panel, lines 3785-3788.
- **Problem.** The plot prefers `presolved_performance`, but the output section defines no such key, and `solve()`
  never writes it. On `main` only the plotter reads it. An implementer must guess its content.
- **Evidence.** `git grep presolved_performance main` finds only `utils/mixed_plotter.py`.
- **Fix.** Replace lines 3785-3788 by: "The model counts are those of the original model (\texttt{performance}). When a
  file \texttt{runner.out} (a Gurobi log) lies in the folder of the result file, the plot shows instead the presolved
  counts read from its lines ``Presolved: $r$ rows, $c$ columns'' and ``Variable types: $a$ continuous, $b$
  integer''. The label says which."

### 17. The condition "skipped" of the relaxation start differs between two sections

- **Location.** Warm-Start Reports, line 3473; Output, line 3030; Relaxation Start, line 3442.
- **Problem.** Line 3473 says "skipped when it is true and a start was applied". Lines 3030 and 3442 say "a start or a
  hint". The value of `enabled` in the skipped case is not stated.
- **Fix.** Line 3473: "\texttt{"skipped"} when it is true and a start or a hint was applied". In both places:
  "\texttt{enabled} is the value of the option".

### 18. The word "idle" names two concepts

- **Location.** Output, lines 2918, 2921, 2981-2984; plot marks, lines 3771-3772.
- **Problem.** The key `idle = 1` means a component in the depot without an intervention; the plot draws it as `D`.
  An "idle day" of a train means no activity; the plot draws it as `I`, and there the key `idle` is 0. The rules for the
  text ask for one term per concept. A reader who sees the legend "`I` idle" looks for `idle = 1` and finds `D`.
- **Fix.** Keep the decided key name. Line 2921: "The key \texttt{idle} is $1$ when component $\ell$ rests in the depot
  without an intervention; the plot shows it with the mark \texttt{D}. On an idle day of the train
  (Section~\ref{sec:assignment}) the key is $0$, and the plot shows the mark \texttt{I}."

### 19. Four cross-references point to the wrong section

- **Location and fix.**
  - Line 576 (`depot_capacity`): Section depot does not discuss availability. Use `\ref{sec:assignment}`, where line
    838 says "No other fleet-availability row is needed".
  - Lines 3497 and 3921 (`cfg`, "Section inputs"): the normalised configuration is defined by `load_config` in
    Section api (line 3651). Use "(\texttt{load\_config}, Section~\ref{sec:api})".
  - Line 541 (progress report): use `\ref{sec:output_reports}`.

## Suggestions

- **Ties in a parallel sweep.** Lines 3110 and 3290: "evaluated first" is undefined when cases finish out of order.
  Write "A tie goes to the case with the smaller $H_2$ (in the grid: the smaller $H_1$, then the smaller $H_2$)."
  This is what `main`'s sequential code does.
- **The gap that the proof rule reads.** Line 3096 and the early stop use `requested_mip_gap`. If `gurobi_params` sets
  `MIPGap`, Gurobi uses another gap than the run option. Report in `requested_mip_gap` the value of Gurobi's parameter
  after all options are set; the same for the time limit.
- **`evaluation_horizon` in another mode.** Lines 540 and 674 check `E > H_1` only in the mode `evaluation_total`.
  In another mode a smaller `E` gives a projected cost with a negative weight. Check `E > H_1` whenever `E` is given.
- **How a sweep solves one case.** Line 3169 says "as `solve()` solves an input file". An HDF5 input yields arrays that
  `yaml.safe_dump` cannot write back. Name an internal entry point that takes the checked mapping, or say that the case
  is written to a temporary file in the input's own format.
- **The schedule kept by a sweep.** Line 3170 says the sweep keeps the summary, not the schedule. A warm start needs the
  result of the latest case with a solution (line 3482). Add: "With \texttt{warm\_start=True} it keeps the result of the
  latest case that has a solution."
- **The greedy loop pass and the linear forms.** Line 3529 tests the reliability row in its exact form. At a `tangent`
  or `pwl` cell the built row is stricter, so a schedule without failures can still fail the completion. Test the form
  built at the cell.
- **One definition of the day cost.** The day cost and the phase costs appear three times: Eqs. a_daycost/a_phases,
  c_step/c_phases and d_daycost/d_phasecosts. Cite Eq. a_daycost and a_phases in the output and sweep sections instead.
- **Folders that the layout omits.** Lines 3565-3598 list no `experiments/`. The decision "Cluster jobs and
  experiments" keeps the experiment definitions there (only their results go), and the fixtures in
  `results/validator_tests/`. Add both, or say where the fixtures move. Also confirm that the dashboard mode "Gamma
  diagnostic" (line 3913) is a rewrite on the kept validators: `main`'s `dashboard_gamma.py` serves the synthetic
  diagnostic that the follow-up decision drops.
