Status: done (10 findings: 2 [error], 5 [inconsistency], 3 [meaning]; 2 doubts)

# Review R1: the core formulation

Block: scoping.tex lines 56-831 and 3952-4161; spec/spec.tex lines 97-164, 263-341, 593-850, 1543-1690
(fragments_A.tex, claude_full_problem.tex).

Check scripts (scratchpad): `r1_checks/check_examples.py` (every worked example of the block),
`r1_checks/gauss_loop_prob.py`, `r1_checks/gauss_loop_prob_vec.py`, `r1_checks/gauss_loop_prob_general.py`
(searches for a Gaussian failure of the probability loop), `r1_checks/initial_state.py` (F3),
`r1_checks/slack_repair_cost.py` (F5). All under the scratchpad folder.

Result of the recomputation: every number of the worked examples of the block reproduces (Example 1 and Example 2
of Section "What the one-sided rows accept", the example of Section "Replacements required by the loop constraint",
the example $E = 365$ of the objective modes, and the anchor example of the appendix). The proofs of Lemma
lem:monotone, Propositions prop:domination, prop:acceptance, prop:ard1_replacement, prop:repair_cost, Theorem
thm:perpetual and Remark rem:one_depot_row were checked step by step; the only gap found is F6.

## Findings

### F1 [error] The abstract problem has no row that defines the repair-cost variable z
- Where: scoping.tex:771 and 773-816 (Problem eq:abstract).
- Problem: in Problem (eq:abstract), $z$ appears only in the objective, through the non-decreasing $C_R$, and in
  $z \ge 0$. No row links $z$ to the damage. So every optimum can set $z = 0$, and the problem never charges a
  repair. The definition (eq:z_def) of Section "Damage dynamics" is missing. The variable $u$, by contrast, has its
  row (eq:abs_u).
- Evidence: the rows of (eq:abstract) are abs_obj, abs_assign, abs_cover, abs_maint, abs_dyn, abs_init, abs_rel,
  abs_loop, abs_u, abs_domain. Take a feasible point with some $z_{i\ell k} > 0$. Setting $z = 0$ keeps every row
  satisfied, because no row other than abs_domain contains $z$, and it does not raise $C_R$. The sentence at 771
  lists the rows and also omits (eq:z_def).
- Fix: file `scoping.tex`
  - old: `the damage dynamics~\eqref{eq:dyn_full}, the reliability constraint and the loop constraint~\eqref{eq:loop}:`
  - new: `the damage dynamics~\eqref{eq:dyn_full}, the definition~\eqref{eq:z_def} of the repair-cost variable, the reliability constraint and the loop constraint~\eqref{eq:loop}:`
  - old: `  & u = \max_{i,k}\;\phi\!\bigl(\mathbb{E}[D_{i1k}],\ldots,\mathbb{E}[D_{iLk}]\bigr),`
  - new (a row for $z$ before the row for $u$; the product form gives (eq:z_def) on a repair day and $z = 0$ on every
    other day, as line 231 states):
    ```
      & z_{i\ell k} = x^m_{i\ell k}\,\bigl(\mathbb{E}[D_{i\ell,k-1}] - \mathbb{E}[D_{i\ell k}]\bigr)
        && \forall\, i,\,\ell,\,k,
        \label{eq:abs_z} \\
      & u = \max_{i,k}\;\phi\!\bigl(\mathbb{E}[D_{i1k}],\ldots,\mathbb{E}[D_{iLk}]\bigr),
    ```
- Confidence: high

### F2 [inconsistency] The threshold and the reliability level: one value for all components, or one per component
- Where: scoping.tex:108-109 (Table tab:notation, under "Scalar and fleet-level parameters"); against
  spec/spec.tex:287-289 and 383-384 (fragments_A.tex:94-96; the input table in fragments_C.tex), scoping.tex:3818
  and 3942.
- Problem: the notation table says that $\tau$ is "the same for every component" and gives one level $\varepsilon$
  for all $(i,\ell,k)$. The spec reads `tau` and `epsilon` as cell values $\tau_{i\ell}$, $\varepsilon_{i\ell}$. The
  scoping document itself writes $\Pr(D_{i\ell k} > \tau_{i\ell}) \le \varepsilon_{i\ell}$ at line 3818, and line
  3942 says that the reliability level is set per cell. The spec is right: it describes the input format of the code.
- Evidence: spec:383 `\keyrow{\texttt{tau}}{cell value}...{Degradation threshold $\tau_{i\ell}$}`; spec:384 the
  same for `epsilon`; spec:544-545 check each cell value.
- Fix: file `scoping.tex` (move both rows to the component-level block)
  - old: `$\tau$          & Failure threshold of the damage, the same for every component ($\tau = 1$ in the Palmgren--Miner normalisation) \\`
    followed by the next line `$\varepsilon$   & Reliability threshold: $\Pr(D_{i\ell k} > \tau) \le \varepsilon$ for all $i,\ell,k$ \\`
  - new: (both lines deleted)
  - old: `$\rho_{i\ell}$      & Repair efficiency $\in (0,1]$ \\`
  - new:
    ```
    $\tau_{i\ell}$      & Failure threshold of the damage; the subscript $i\ell$ is often dropped ($\tau = 1$ in the Palmgren--Miner normalisation) \\
    $\varepsilon_{i\ell}$ & Reliability level: $\Pr(D_{i\ell k} > \tau_{i\ell}) \le \varepsilon_{i\ell}$ for all $k$; the subscript $i\ell$ is often dropped \\
    $\rho_{i\ell}$      & Repair efficiency $\in (0,1]$ \\
    ```
- Confidence: high

### F3 [inconsistency] The initial damage "must satisfy" the reliability level, but no check requires it
- Where: scoping.tex:121 (Table tab:notation, $D^{\mathrm{start}}$); against spec/spec.tex:547 and 556 (checks
  $0 \le \mu_0 < \tau$ and initial values in $[0, M_s]$, fragments_C.tex) and scoping.tex:3599-3603 (the same check,
  with its reason).
- Problem: the notation table states $\Pr(D^{\mathrm{start}}_{i\ell} > \tau) \le \varepsilon$ as a requirement. The
  spec does not check it, and the scoping list of input checks does not contain it. The spec accepts initial states
  that violate the level. No result of the block needs the condition: the reliability rows hold on days
  $1, \ldots, T$, and Theorem thm:perpetual covers the days $k \ge 1$.
- Evidence: Gaussian cell, $\tau = 1$, $\varepsilon = 0.01$, so $V_{\max} = 1/\Phi^{-1}(0.99)^2 = 0.1848$. The input
  $\mu_0 = 0.95$, $v_0 = 0.1$ passes every check: $0 \le 0.95 < 1$, $0 \le 0.1 \le 0.1848$, and the Gaussian check
  $\mu_0 \ge 3\sqrt{v_0} = 0.949$. Then $\Pr(D_0 > 1) = 1 - \Phi(0.05/\sqrt{0.1}) = 0.437 > 0.01$
  (script `r1_checks/initial_state.py`). The spec is right if the owner does not want the check; otherwise the spec
  needs a new check instead of this fix.
- Fix: file `scoping.tex`
  - old: `(known distribution; must satisfy $\Pr(D^{\mathrm{start}}_{i\ell} > \tau) \le \varepsilon$)`
  - new: `(known distribution with mean in $[0, \tau)$; the reliability level is required from day~$1$ on)`
- Confidence: medium (which side holds is the owner's choice; the two documents disagree in any case)

### F4 [meaning] In the operating-average mode the transitory phase does affect the objective
- Where: scoping.tex:701 and Table tab:objective_modes (line 692); against scoping.tex:706 and the spec rows of $u$
  (spec/spec.tex:1569-1576, fragments_A.tex:499-508, "one row per train and day", every $k = 1, \ldots, T$).
- Problem: the text says that "the solver chooses the schedule of the transitory phase for feasibility only", and the
  table says "no cost". With $C_D > 0$, the rows of $u$ hold on the transitory days too, and line 706 says that $u$
  is the maximum over the whole horizon. So the transitory schedule changes $C_D\,u$, which the objective contains.
- Evidence: Eq. a_u_inf, $u \ge \sum_\ell \mu_{i\ell k}$ for all $i$ and all $k = 1, \ldots, T$; the objective of the
  mode is $J_{\mathrm{op}}/H_2 + C_D\,u$.
- Fix: file `scoping.tex`
  - old: `The solver chooses the schedule of the transitory phase for feasibility only.`
  - new: `The solver chooses the schedule of the transitory phase for feasibility and, when $C_D > 0$, for the damage penalty, because $u$ also covers the transitory days.`
  - old: `no cost, but still constrained`
  - new: `no day cost, but still constrained and part of~$u$`
- Confidence: high

### F5 [meaning] Slack in the mean does not raise the cost of every later repair under ARD1
- Where: scoping.tex:457 (paragraph "Meaning of the reported values", bullet "Mean"); against scoping.tex:765-766.
- Problem: the bullet says that a larger mean "raises the repair cost $C_R\,z$ of every later repair". Under
  $\text{ARD}_1$ the next intervention copies the slack into the anchor ($g_k = \mu_k$), so the gap $\mu - g^{\mu}$ of
  the later repairs does not contain it. Lines 765-766 describe this mechanism. Under both laws, a replacement also
  removes the slack.
- Evidence: script `r1_checks/slack_repair_cost.py`. $\rho = 0.5$, $\mu_0 = g_0 = 0.2$; day 1 mission $+0.1$, day 2
  repair, day 3 mission $+0.1$, day 4 repair. Smallest feasible $z$ on days 2 and 4: $(0.05, 0.05)$ without slack,
  $(0.10, 0.05)$ with slack $0.1$ on day 1 ($\text{ARD}_1$). Under $\text{ARD}_\infty$: $(0.15, 0.125)$ and
  $(0.20, 0.15)$, so both repairs rise.
- Fix: file `scoping.tex`
  - old: `and raises the repair cost $C_R\,z$ of every later repair.`
  - new: `and raises the repair cost $C_R\,z$ of the next repair before a replacement. Under $\text{ARD}_\infty$ it also raises the cost of every later repair before a replacement; under $\text{ARD}_1$ the anchor takes the slack at the next intervention, and the later repairs do not change.`
- Confidence: high

### F6 [error] The proofs use that the reference values are non-negative, but never show it
- Where: scoping.tex:4039 (proof of Theorem thm:perpetual, last step), with the down-set definition at 400-404;
  scoping.tex:661-663.
- Problem: the down-set property is stated only for $s'$ with $0 \le s' \le s$. The last step of the proof concludes
  $\zeta_k \in \mathcal{C}_{i\ell}$ from $\zeta_k \le s$ alone; it also needs $\zeta_k \ge 0$. Line 663 attributes
  $\zeta^{\mu}_{i\ell k} \ge 0$ to Proposition prop:domination, which gives only $\theta_k \le \zeta_k \le \xi_k$.
  The fact is true: $\zeta_0 = \xi_0 \ge 0$ and every replacement value is $\ge 0$ by the input checks
  (spec:547-550), and by (M1) each form of the day map ($s + \delta s_k$, $g$, $a\,s + (1-a)\,g$, $s^{\mathrm{new}}$)
  sends non-negative values to non-negative values.
- Fix: file `scoping.tex`
  - old: `Since $\mathcal{C}_{i\ell}$ is a down-set, $\zeta_k \in \mathcal{C}_{i\ell}$, and (M3) gives $\Pr(D_{i\ell k} > \tau) \le \varepsilon$.`
  - new: `The reference values are non-negative: $\zeta_0 = \xi_0 \ge 0$, every replacement value is non-negative, and by (M1) each day map keeps non-negative values non-negative. Since $\mathcal{C}_{i\ell}$ is a down-set, $\zeta_k \in \mathcal{C}_{i\ell}$, and (M3) gives $\Pr(D_{i\ell k} > \tau) \le \varepsilon$.`
  - old: `By Proposition~\ref{prop:domination}, the tracked means satisfy`
  - new: `By Proposition~\ref{prop:domination} and the non-negativity of the reference values (proof of Theorem~\ref{thm:perpetual}), the tracked means satisfy`
- Confidence: high (low severity: a missing step, the claim is true)

### F7 [inconsistency] "Each statistic also has an anchor" contradicts the Wiener variance, which has none
- Where: scoping.tex:348; against scoping.tex:2664 (caption of tab:loop_per_model), 3057, and spec/spec.tex:754-755
  (fragments_A.tex:313-314, "none for the Wiener variance").
- Problem: the sentence gives an anchor to every statistic under $\text{ARD}_1$. The Wiener variance has no anchor in
  both documents. The tracked state defined next (line 349) and the loop rows (line 350) then contain an anchor
  that does not exist.
- Fix: file `scoping.tex`
  - old: `Under the repair law $\text{ARD}_1$, each statistic also has an \emph{anchor}: its value just after the last intervention (Section~\ref{sec:repair_connection}).`
  - new: `Under the repair law $\text{ARD}_1$, each statistic except the Wiener variance also has an \emph{anchor}: its value just after the last intervention (Section~\ref{sec:repair_connection}).`
- Confidence: high

### F8 [inconsistency] The full problem omits the optional variance row
- Where: spec/spec.tex:1627-1642 (claude_full_problem.tex:17-32, "Rows of a cell"); against the optional row
  $v_k \le V^{\mathrm{user}}$ of the Gaussian, Wiener and rainflow sections (fragments_B.tex:104, 271, 627) and the
  input `v_max_user` (spec:458).
- Problem: the section says that it "collects the whole problem in one place". Its list of the rows of a cell has no
  item for the optional row $v_{i\ell k} \le V^{\mathrm{user}}_{\max,i\ell}$, which the code builds when `v_max_user`
  is given.
- Fix: file `merge/spec_drafts/claude_full_problem.tex`
  - old: `  \item the loop rows Eq.~\eqref{eq:a_loop}, one per tracked quantity and per anchor;`
  - new:
    ```
      \item where \texttt{v\_max\_user} is given (G, W, RF with Ca or Be): the optional row
            $v_{i\ell k} \leq V^{\mathrm{user}}_{\max,i\ell}$ of its model section, on every day;
      \item the loop rows Eq.~\eqref{eq:a_loop}, one per tracked quantity and per anchor;
    ```
- Confidence: high (low severity)

### F9 [inconsistency] The notation defines the variance increment from $v^{\inc}$, also at Wiener cells
- Where: spec/spec.tex:297-299 (fragments_A.tex:104-106); against fragments_B.tex:219 ($\delta v_k = \sigma^2
  \sum_j x_{ijk}$ at a Wiener cell) and spec:444 (the profile `v` applies only to G and RF with Ca or Be).
- Problem: the notation row says that $\delta v$ is formed "likewise", that is from $v^{\inc}$. A Wiener cell has no
  $v^{\inc}$; its variance increment is $\sigma^2$ per mission day. So one symbol has two definitions, and the
  general one is undefined at Wiener cells.
- Fix: file `merge/spec_drafts/fragments_A.tex`
  - old: `  of the mean; $\delta v$, $\delta R$ (from $(b^{\inc})^2$) and $\delta K$ likewise; $0$`
  - new: `  of the mean; $\delta v$ (from $\sigma^2$ at a Wiener cell), $\delta R$ (from $(b^{\inc})^2$) and $\delta K$ likewise; $0$`
- Confidence: high (low severity)

### F10 [meaning] "Since $F > M$, exactly $M$ members serve missions" gives a wrong reason
- Where: scoping.tex:154; against spec/spec.tex:707 (fragments_A.tex:266, same fact without the reason) and
  scoping.tex:3589 (the actual role of $F > M$).
- Problem: exactly $M$ members serve missions because of the coverage rows (eq:mission_coverage) and the
  one-activity rows (eq:one_activity), for any $F \ge M$. The condition $F > M$ gives only $F - M \ge 1$: at least
  one member is free for the depot every day.
- Fix: file `scoping.tex`
  - old: `\item \textbf{Members on missions.} Since $F > M$, exactly $M$ members serve missions on every day, and the other $F - M$ members are in the depot or idle.`
  - new: `\item \textbf{Members on missions.} By~\eqref{eq:one_activity} and~\eqref{eq:mission_coverage}, exactly $M$ members serve missions on every day. Since $F > M$, the other $F - M \ge 1$ members are in the depot or idle.`
- Confidence: high (low severity)

## Doubts

- **The Gaussian example of scoping.tex:324-325.** The text says that "a higher variance balanced by a lower mean"
  can satisfy the probability loop (eq:loop_prob) over one repetition and fail later. Random searches (scripts
  `r1_checks/gauss_loop_prob_vec.py` and `r1_checks/gauss_loop_prob_general.py`, Gaussian component, exact moments,
  operating phases of 2 to 5 days) find many such failures. Under $\text{ARD}_\infty$, every failure found has the
  opposite direction: the mean rises and the variance falls over the first repetition. Example
  (`r1_checks/gauss_loop_prob.py`): $\varepsilon = 0.1$, $\rho = 0.191$, phase = mission $(+0.193, +0.00397)$ then
  repair; start $(0.082, 0.0433)$, end $(0.222, 0.0310)$; failure probability $5.10\cdot10^{-6}$ then
  $4.92\cdot10^{-6}$; the peak of the ninth repetition has failure probability $0.135 > 0.1$. Under $\text{ARD}_1$ the
  stated direction also occurs, but there the anchors move too. A partial argument for the two-day phase under
  $\text{ARD}_\infty$: with $t = (1-\rho)^n$, the squared z-score of each day is $(A - Bt)^2/(V - Ct^2)$, which has a
  single critical point (a minimum); (eq:loop_prob) forces $1-\rho$ below the critical point of the end day, so the
  only day that could fail is the peak of the second repetition. The searches never found it failing. I did not prove
  impossibility. If the owner wants the example to hold under $\text{ARD}_\infty$, "a lower variance balanced by a
  higher mean" is the direction that the searches confirm.
- **The repair operator of $\text{ARD}_1$.** scoping.tex:215 and 252 write $\mathcal{R}(D_{i\ell,k-1}, \rho_{i\ell})$,
  a function of the damage of the day before only. Under $\text{ARD}_1$ the operator also needs
  $D^{\mathrm{last}}$ (scoping.tex:3020-3022 says so). Problem (eq:abstract) has no row for the dynamics of
  $D^{\mathrm{last}}$. This looks like a known abuse of notation, so I did not count it as an error.
