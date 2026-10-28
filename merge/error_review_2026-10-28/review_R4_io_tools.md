Status: done (14 findings: 6 error, 4 inconsistency, 4 meaning; 5 doubts)

# Review R4: inputs, checks, outputs and tools

## Findings

Line numbers: `spec/spec.tex` as assembled at 19:12; `scoping.tex` as on disk at 21:33 (the block starts at
line 3525, ten lines later than in the brief); fragments as on disk at 21:27.

### F1 [error] The reason for eps < 0.5 at Gaussian and Wiener cells is false
- Where: scoping.tex:3607 (Reasons for the input checks, "Reliability, repair and initial state").
- Problem: The text says the squared normal quantile is positive only below 0.5. The square
  kappa_G^2 = Phi^{-1}(1-eps)^2 is positive for every eps other than 0.5. The real reason, given in
  Remark rem:vmax (scoping.tex:2791-2792), is the sign of kappa_G: at eps >= 0.5, kappa_G <= 0, the squared
  row no longer describes the requirement, and at eps = 0.5 the range V_max = tau^2/kappa_G^2 is infinite.
- Evidence: `r4_tools_checks.py` (output `r4_tools_checks_out.txt`): eps = 0.7 gives kappa_G = -0.5244,
  kappa_G^2 = 0.2750; eps = 0.9 gives kappa_G = -1.2816, kappa_G^2 = 1.6424.
- Fix: file `scoping.tex`
  - old: `the squared normal quantile of the Gaussian and Wiener rows is positive only below~$0.5$ (Remark~\ref{rem:vmax}).`
  - new: `the normal quantile $\kappa_G = \Phi^{-1}(1-\varepsilon)$ of the Gaussian and Wiener rows is positive only below~$0.5$; at $\varepsilon \ge 0.5$ the squared row does not describe the requirement (Remark~\ref{rem:vmax}).`
- Confidence: high

### F2 [error] "optimal" at MIP gap 0.1 does not mean "at most 10 % above the best bound"
- Where: scoping.tex:3669 (Outputs, "Solution and status").
- Problem: Gurobi's relative gap is |ObjBound - ObjVal| / |ObjVal|, relative to the incumbent, not to the
  bound. At gap 0.1 a minimisation can stop with ObjVal = ObjBound/0.9, which is 11.1 % above the bound.
- Evidence: `r4_tools_checks.py`: bound 90, gap 0.1, incumbent 100 is accepted; 100 is 11.11 % above 90.
- Fix: file `scoping.tex`
  - old: `the status \texttt{optimal} means that the best solution is at most $10\,\%$ above the best bound.`
  - new: `the status \texttt{optimal} means that the best solution exceeds the best bound by at most $10\,\%$ of its own value, that is, by at most about $11\,\%$ of the bound.`
- Confidence: high (definition of `MIPGap` in the Gurobi reference manual; the spec does not define the
  gap otherwise).

### F3 [error] The leak bound of a big-M row is 2·10^-5 M_s per row, and it adds up over the days
- Where: scoping.tex:3577-3578 (Big-M and indicator forms, "Choice of the default form").
- Problem: The carry row (scoping.tex:3547) and the anchor rows contain the sum x^m_k + x^r_k of two
  binaries. On a depot day without intervention (x_{i0k} = 1), both may sit at IntFeasTol = 10^-5, and the
  depot row x^m + x^r <= x_{i0k} allows it. The carry row then leaks 2·10^-5 M_s, not 10^-5 M_s. The leak
  of each day's carry row lowers the tracked state further, so after T days the tracked state can lie
  2·10^-5·T·M_s below the reference trajectory (0.73 % of M_s for T = 365). "A relative error of 10^-5"
  holds for neither the row nor the horizon.
- Evidence: hand derivation above; numbers in `r4_tools_checks.py` (section "IntFeasTol leak").
- Fix: file `scoping.tex`
  - old: `A big-M row can therefore leak by at most $10^{-5} M_s$.`
  - new: `A big-M row can therefore leak by at most $2 \cdot 10^{-5} M_s$, because the carry row and the anchor rows contain the sum $x^m_k + x^r_k$ of two binaries.`
  - old: `Because $M_s$ is the range of the statistic itself, this is a relative error of $10^{-5}$.`
  - new: `Because $M_s$ is the range of the statistic itself, this is a relative error of at most $2 \cdot 10^{-5}$ per row. The leaks of the carry rows of successive days add up, to at most $2 \cdot 10^{-5}\,T$ over the horizon.`
- Confidence: high for the factor 2 (it follows from the rows); the bound ignores the feasibility
  tolerance (default 10^-6, absolute), as the original text does.

### F4 [error] The natural range of a mean is not "the largest value that the exact row admits"
- Where: scoping.tex:3556-3557 (Big-M and indicator forms), first item of the list "tau for a mean".
- Problem: Two claims fail for the mean.
  - "the largest value that the exact row of the bound admits at mu >= 0": the rows cap the mean below
    tau at Markov (eps·tau, spec Eq. rf_markov), Bernstein (side row tau - c, spec Eq. rf_bernstein),
    inverse Gaussian (mu_bar < tau) and Gamma (A_1/beta* < tau, spec Section dyn_gamma). tau is an upper
    bound there, not the largest value.
  - "an upper bound of s at every point that satisfies the reliability row": the Chernoff row
    K - s·tau <= ln eps (spec Eq. rf_chernoff) has no mu, so a point with mu = 2 tau and K = 0 satisfies
    it. tau bounds the Chernoff mean only on the reference trajectory: the input checks K^inc >= s mu^inc,
    K_0 >= s mu_0, g^K_0 >= s g^mu_0, K^new >= s mu^new and the common factor 1 - rho of mu and K keep
    K >= s mu by induction, so mu <= K/s <= tau - kappa/s < tau.
  - The same definition is repeated in scoping.tex:502 (Section acceptance), outside this block.
- Evidence: spec rows cited above; example: Markov cell with eps = 0.05, tau = 1: the row admits mu up to
  0.05, not 1.
- Fix: file `scoping.tex`
  - old: `Each value below is the largest value that the exact row of the bound admits at $\mu \ge 0$.`
  - new: `For a variance, the Hoeffding budget and the Chernoff statistic, the value below is the largest value that the exact row admits at $\mu \ge 0$. For a mean, $\tau$ is an upper bound: the side row $\mu \le \tau$ or a smaller cap bounds it. The Chernoff row does not contain the mean; there $\mu \le K/s < \tau$ holds on the reference trajectory, because the input checks keep $K \ge s\mu$ (Jensen's inequality).`
- Confidence: high for the counterexamples; medium for the wording of the fix.

### F5 [error] A variance cap below the initial variance does not make the problem infeasible
- Where: scoping.tex:3643 (Model-specific checks, "Optional variance cap").
- Problem: The row v_k <= V^user holds on days k = 1, ..., T (spec, input table `v_max_user`), not on
  day 0. Counterexample: a Gaussian cell with v_0 = 2 V^user, rho = 1, law ARD_inf, and a repair on day 1:
  the repair row gives v_1 >= (1-rho)^2 v_0 = 0, so v_1 <= V^user is feasible. At a Wiener cell a
  replacement on day 1 gives v_1 = v^new <= V^user. The claim holds for the replacement variance only.
- Evidence: spec Eq. g_v_inf (repair row of the Gaussian variance) and Eq. w_v_repl.
- Fix: file `scoping.tex`
  - old: `--- A cap below either of them makes the problem infeasible.`
  - new: `--- A cap below the replacement variance makes every replacement infeasible, and a cap below the initial variance forces an intervention on day~1.`
- Confidence: high

### F6 [error] At a Bernstein cell the tangent row at tau does not reduce to chi <= 0
- Where: scoping.tex:3616 (Reasons for the input checks, tangent point).
- Problem: With f(mu) = ((tau-mu)^2 - 2c(tau-mu))/(2 kappa) (spec, Bernstein row of the table "Forms"),
  f(tau) = 0 and f'(tau) = c/kappa = b/3. The tangent row at mu_r = tau is chi <= (b/3)(mu - tau). With
  the side row mu <= tau - c it gives v <= -bc/3 < 0 when b > 0: no point is feasible. "reduces to
  chi <= 0" holds only for the rows with c_1 = 0 (Gaussian, Wiener, Cantelli, Hoeffding). The conclusion
  (exclude the value 1) stands.
- Evidence: derivation above, from spec Eq. a_quad and the table of c_2, c_1 (spec:868-871).
- Fix: file `scoping.tex`
  - old: `At the point~$\tau$ the tangent row reduces to $\chi \le 0$, and`
  - new: `At the point~$\tau$ the tangent row reduces to $\chi \le 0$ (Gaussian, Wiener, Cantelli, Hoeffding) or admits no point (Bernstein with $b > 0$), and`
- Confidence: high

### F7 [inconsistency] Which `bound_method` entries are checked
- Where: spec/spec.tex:537 (fragments_C.tex:205) against spec/spec.tex:354 (fragments_C.tex:22) and
  scoping.tex:3594.
- Problem: The check says that `bound_method` entries "at rainflow cells" are valid names. The input rules
  say that a selector entry at a cell where the key does not apply "is still checked", and the scoping
  reason says that an unknown selector value "is an error at every component". A non-`null` invalid entry
  of `bound_method` at a Gaussian cell is an error by two places and allowed by the third. The two general
  statements agree with each other and with the explicit rule for `reliability_impl` (spec:538, "also at a
  cell whose row is linear"), so the `bound_method` check is the place to align.
- Fix: file `merge/spec_drafts/fragments_C.tex`
  - old: `\item \texttt{bound\_method} entries at rainflow cells are \texttt{markov}, \texttt{cantelli}, \texttt{hoeffding}, \texttt{bernstein} or \texttt{chernoff}.`
  - new: `\item Every non-\texttt{null} entry of \texttt{bound\_method} is \texttt{markov}, \texttt{cantelli}, \texttt{hoeffding}, \texttt{bernstein} or \texttt{chernoff}, also at a cell that is not rainflow.`
- Confidence: medium (the code was not available to say which behaviour is implemented)

### F8 [inconsistency] The operating sweep does not rank by P_E when E is given
- Where: scoping.tex:3690-3695 (Horizon sweeps) against spec/spec.tex:1887 (fragments_D.tex, "Ranking
  value") and spec/spec.tex:2065.
- Problem: The scoping says "With an evaluation horizon E, it ranks by the projected cost". The spec ranks
  the cases of `sweep_operating_horizons` by J̄_op always, also when its argument `evaluation_horizon` is
  given; only `sweep_horizon_grid` ranks by P_E. The two rankings differ, because J_init changes from case
  to case. The spec is right: the scoping's next sentence says that P_E serves to compare values of H_1,
  which only the grid varies.
- Fix: file `scoping.tex`
  - old: `With an evaluation horizon~$E$, it ranks by the projected cost`
  - new: `A sweep over both horizons with an evaluation horizon~$E$ ranks by the projected cost`
- Confidence: high

### F9 [inconsistency] The changelog says that the costs are set per cell
- Where: scoping.tex:3953 (changelog, version 1.0) against spec/spec.tex:466 and 469 (fragments_C.tex,
  rows `C_M` and `C_D`).
- Problem: `C_M` and `C_D` are scalars; only `C_R` and `C_rep` are cell values.
- Fix: file `scoping.tex`
  - old: `\item The reliability level, the tangent point and the costs are set per cell.`
  - new: `\item The reliability level, the tangent point, the repair cost and the replacement cost are set per cell.`
- Confidence: high

### F10 [inconsistency] `penalty_type` is "ignored" at C_D = 0, but it selects the reported u
- Where: spec/spec.tex:469 (fragments_C.tex, row `C_D`) and spec/spec.tex:1567-1568 (fragments_A.tex:499-500)
  against spec/spec.tex:1753-1758 (fragments_C.tex:324, "Value of u").
- Problem: Two places say that with C_D = 0 the input `penalty_type` is ignored. The output rule reports u
  also when C_D = 0, with the formula of Eq. c_u that `penalty_type` selects (sum of means or sum of
  squared means). So `penalty_type` changes the keys `u` and `peak_damage` at C_D = 0.
- Fix: file `merge/spec_drafts/fragments_C.tex` (the same change in fragments_A.tex:499-500)
  - old: `if $C_D = 0$, the variable $u$ and its rows are not built and \texttt{penalty\_type} is ignored}`
  - new: `if $C_D = 0$, the variable $u$ and its rows are not built, and \texttt{penalty\_type} only selects the formula of the reported value $u$ (Section~\ref{sec:output})}`
- Confidence: medium (the alternative fix, an output rule that uses `inf_norm` at C_D = 0, is equally
  consistent; the code on `main` has only `inf_norm`)

### F11 [meaning] The output key `idle` is not an idle day
- Where: spec/spec.tex:1716 (fragments_C.tex:287) against spec/spec.tex:708 ("An idle day is an all-zero
  assignment row") and spec/spec.tex:2301-2302 (plot marks D and I).
- Problem: `idle` = x_{i0k} - x^m - x^r is 1 on a depot day without intervention, which the plot marks
  `D`. An idle day of the train (plot mark `I`) has x_{i0k} = 0, so `idle` = 0 on it. The word "idle"
  names two different states, and a reader takes `idle` = 1 for an idle day. The key name is fixed, so
  the row must say it.
- Fix: file `merge/spec_drafts/fragments_C.tex`
  - old: `component $\ell$ is in the depot without an intervention}`
  - new: `component $\ell$ is in the depot without an intervention (mark \texttt{D} of Section~\ref{sec:plotting}); it is $0$ on an idle day of the train (Section~\ref{sec:assignment})}`
- Confidence: high

### F12 [meaning] "jobs" in the row `C_M`
- Where: spec/spec.tex:466 (fragments_C.tex:134).
- Problem: "whatever the number of jobs on that day" uses an undefined word for the interventions; the
  objective section says "whatever the number of interventions on that day" (spec:679). The spec uses
  "job" for a Slurm job of the cluster (spec:180, 204), so one word names two things.
- Fix: file `merge/spec_drafts/fragments_C.tex`
  - old: `whatever the number of jobs on that day}`
  - new: `whatever the number of interventions on that day}`
- Confidence: high

### F13 [meaning] The scoping item "Start" describes the greedy start without saying so
- Where: scoping.tex:3724-3727 (Warm starts, item "Start").
- Problem: "A start with failures is still returned" and "Its completion can then end with the status
  infeasible" refer to the greedy heuristic: only its source requests the completion, and "failures" are
  the repeatability failures of its loop pass (spec:2148, 2236-2238). Under the item "Start", the reader
  reads them as a property of every start, and "failures" is undefined there. "Gurobi tries to complete it
  by itself" is Gurobi's own completion of a partial start, a different step from the completion of the
  spec (Section warm_start_modes).
- Fix: file `scoping.tex`
  - old: `    A start with failures is still returned.`
  - new: `    The greedy heuristic returns its start also when its loop pass counts failures, and it requests the completion of the specification.`
  - old: `    Its completion can then end with the status \texttt{infeasible}, with an irreducible infeasible subset of rows.`
  - new: `    That completion can then end with the status \texttt{infeasible}, with an irreducible infeasible subset of rows.`
- Confidence: high

### F14 [meaning] The relaxed solution can be a feasible point
- Where: scoping.tex:3730 and spec/spec.tex:2174 (fragments_D.tex:408).
- Problem: The relaxed copy minimises the sum of the violations of the linear rows (feasRelaxS). If the
  original model is feasible and the copy reaches its optimum 0, the relaxed solution satisfies every row
  of the original model. "is not a (feasible) point" states as a fact what is only possible.
- Fix: file `scoping.tex`
  - old: `The relaxed solution is not a feasible point of the original model.`
  - new: `The relaxed solution need not be a feasible point of the original model.`
- Fix: file `merge/spec_drafts/fragments_D.tex`
  - old: `The relaxed solution is not a point of the original model.`
  - new: `The relaxed solution need not be feasible for the original model.`
- Confidence: medium

## Doubts

- **The tie rule for profile shapes never applies.** spec:426 (fragments_C) reads an array that fits both
  (L, M, H_*) and (F, L, M) as (L, M, H_*). Equal shapes need F = L = M, which the check F > M excludes.
  The rule is harmless; it is not an error.
- **Segment boundaries of the `pwl` form.** spec:557 (replacement state) and spec:2233 (greedy) take "the
  cap of the segment that contains" the mean. At a boundary p·tau/P two segments contain it, and Eq.
  a_pwl_seg lets the solver choose either. If the code takes one fixed segment, it can reject a state that
  the model accepts. The code was not available to check.
- **Who creates the variable mu.** spec:251 and 255 list mu among the shared variables of step 1; spec:247
  says that `prepare` "creates the variables of the tracked quantities", and mu is a tracked quantity.
  Probably `prepare` creates v, R, K and tightens the bounds of mu; the text does not say so.
- **Reason for the natural-range check.** scoping.tex:3623 says that a value outside the natural range
  "violates a row at once". Counterexample: a Gaussian v_0 above V_max with a repair (rho = 1) on day 1
  violates no row; the relaxed carry row v_1 >= v_0 - V_max > 0 only cuts the point v_1 = 0. The precise
  effect is the one of the appendix derivation (a relaxed row cuts points). Left as a doubt because the
  sentence may mean the variable bound [0, M_s] of the state, which an initial value outside it does
  violate in spirit.
- **Undefined V in the changelog.** scoping.tex (version 0.3) "from F × V × 2H": V is not defined in the
  scoping document; the text is copied from the old specification.

## Checked without error

- The YAML example passes every check of the section Consistency Checks, including the replacement state
  in the form of each cell (tangent at the G cell, pwl with P = 6 at the W cell, exact Bernstein row,
  inverse Gaussian cap mu_bar = 0.4923, Gamma cap >= eps·tau = 0.05 > mu_new = 0.02, Chernoff
  K_new = 0.10 <= s·tau - kappa = 2.697) and g^K_0 = 0.50 = s·g^mu_0 (equality). Script
  `r4_yaml_example_checks.py`, output `r4_yaml_example_checks_out.txt`. The scoping's description of the
  example (anchors, K_0 = 0.70, b = 0.05, 2 kappa b/3 = 0.0768) reproduces.
- Clopper-Pearson: the Beta(f+1, n-f) quantile at the confidence level equals the root of
  P(Bin(n, p) <= f) = 1 - level (four cases, `r4_tools_checks.py`).
- Day mapping Eq. d_daymap: every property that the scoping states holds for all H_1, H_2, H_1', H_2' in
  1..5 (`r4_tools_checks.py`). Projected cost: spec and scoping forms agree. Gradient Eq. d_gradient:
  consistent with its classes and report keys.
- Keys, defaults and "Applies to" columns agree between the input tables, the check list, the table "Keys
  by Cell Type", the output tables and the HDF5 placement table (every result key has one placement).
- Rejected key `formulation`: `lp` is the tangent at mu = 0 in the `lore` code (Gaussian and rainflow
  rows), so "tangent with tangent_ref equal to 0" is right.
- Appendix derivation drv:bigM_validity: every step holds, given g = 0 under ARD_inf (scoping.tex:3062).
