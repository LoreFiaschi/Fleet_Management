# Review of the formulation (spec version 1.0, `spec_assembled.tex`)

Status: done (26 Oct 2026). 7 must-fix items, 8 suggestions. Read in full: lines 1-257 and 687-2882 (Project
Overview to Full Optimization Problem); lines 258-686 (inputs) read to check references. Line numbers refer to
`merge/spec_drafts/spec_assembled.tex` as built at 12:06 on 26 Oct 2026 (build log: no undefined reference).

Evidence. Reviewer's own standard-library scripts and outputs are in the session scratchpad,
`C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\review_F\`
(`gamma_check.py`, `gamma_oneterm2.py`, `ig_check.py`, `misc_check.py`, each with an `_output.txt` where kept).
Recomputed and correct: every value of the IG table, the closed form `E[Z; Z>q] = F(1/q)` (against quadrature), the
ratios 0.93 to 0.59, the three-mission tail 0.118; the Gamma table `A_1/A_hat`, `A_hat = 0.0055012`, the witness
1.565 eps, `r*` and `t_1` for the three levels; `V_max/tau^2`; the Cantelli example (0.810, 0.482, 0.714, pwl 0.810);
the Chernoff example; the loop-remark and scalar-bound examples. The one-term theorem was spot-checked with an exact
mixture series for sums of Gamma variables: 360 random members with 2 or 3 terms (and a constant) and 225 targeted
splits of the optimal term; no member exceeds `max_c Q(m/c, t/c)`. (A first attempt, `gamma_oneterm.py`, used
Simpson quadrature; its ratios up to 1.005 are quadrature error at the singular end point, and the exact series
removes them.) Every proof in scope was checked step by step;
the ones not listed below are correct. Completeness: every "`spec/spec.tex`" bullet of `decisions.md` that concerns
the formulation, and the item "Follow-up decisions from writing the spec", is implemented in scope, except the
correction recorded under "Corrections to this record" (must-fix 2). House rules: no tell-word in scope; the only
citations are the four allowed sources, with locations.

## Must fix

Ranked by severity.

### 1. The linear forms can exclude states that the input checks accept (silent infeasibility)

- **Location.** Linear Forms, lines 1377-1380; Rainflow, lines 2524-2530; Replacement, lines 1310-1313. (The
  checks themselves are in the input section, lines 627 and 667.)
- **Problem (a), Bernstein tangent point.** The text requires `mu_r < tau - c` and says that otherwise the row is
  empty. The condition is necessary, not sufficient. The tangent row at `mu_r` evaluated at `mu = 0` is
  `T(0) = (tau(tau - 2c) - mu_r^2)/(2 kappa)`. So the row admits a point with `v >= 0` only if
  `mu_r <= sqrt(tau(tau - 2c))`, which is strictly below `tau - c`.
- **Evidence (a).** `tau = 1`, `eps = 0.01` (`kappa = 4.605`), `c = 0.1`: `sqrt(0.8) = 0.894 < 0.9`. With `tangent_ref = 0.897` the check passes,
  and `T(0) = -0.0005 < 0` (`misc_check.py`): no mean in `[0, tau - c]` has `v >= 0`. The model is infeasible from
  day 1, with no message.
- **Problem (b), form of the replacement check.** Line 1310 says the input checks require the replacement state "to
  satisfy the reliability row of the cell". It does not say in which form. A state that satisfies the exact row can
  violate the tangent or pwl row. For `c_1 = 0` the tangent row admits `v >= 0` only for `mu <= (tau + mu_r)/2`.
- **Evidence (b).** Gaussian cell, `tau = 1`, default `tangent_ref = 0.5`, `replacement_mu = 0.8`,
  `replacement_v = 0`. Exact row: `0 <= 0.04` holds. Tangent row: `v <= c_2 (1 - 0.5)(1.5 - 1.6) < 0` fails. So no
  replacement is ever feasible. At a Wiener cell whose train flies in the operating phase, the loop row needs a
  replacement, so the whole problem is infeasible.
- **Fix.** (a) Lines 1377-1380 and 2525-2530: "At a Bernstein cell the tangent point must satisfy
  `mu_r < sqrt(tau(tau - 2c))`. The tangent row at `mu_r` is nonnegative at `mu = 0` exactly when
  `mu_r^2 <= tau(tau - 2c)`; above this value no point with `v >= 0` and `mu >= 0` satisfies it." Keep the argument
  for `mu_r >= tau - c` as the special case, and change the check of line 667 to match. (b) Line 1311: "... and the
  replacement state to satisfy the reliability row of the cell in the form built at the cell (exact, tangent or
  pwl)"; the same words in the check of line 627.

### 2. The acceptance statement claims more than Proposition `prop:a_accept` proves

- **Location.** Encoding, "What the one-sided rows accept", lines 1196-1198; the sentence at lines 1236-1237.
- **Problem.** The bullet says the one-sided rows reject a safe schedule "only when tracked quantities move in
  opposite directions over the first repetition". Part (b) of the proposition needs three more hypotheses: the exact
  form, the law ARD∞ (no anchor) and no replacement on day `H_1`. Without them a safe schedule with one tracked
  quantity can be rejected. The sentence at line 1236 also drops the hypothesis "exact form". `decisions.md`, item
  "Follow-up decisions", "Corrections to this record", already records the missing hypothesis on day `H_1`.
- **Evidence (counterexample).** One tracked mean, ARD∞, `rho = 0.5`, replacement on day `H_1` with
  `replacement_mu = 0`; the operating phase is a mission (+0.2) and then a repair, so `Phi(y) = 0.5 y + 0.1`. The
  replacement row forces `xi_{H_1} = 0`; the loop row needs `xi_T <= 0`, but `xi_T >= Phi(0) = 0.1`. Rejected. The
  reference trajectory has peaks 0.2, 0.3, 0.35, ... rising to 0.4, as in the first example of the section, so the
  schedule is safe whenever `0.4` is in the certificate set. Only one quantity is tracked. A second case: a tangent
  or pwl row is stricter than the exact row, so it rejects safe schedules whose peaks lie between the two.
- **Fix.** Lines 1196-1198: "*They may reject a safe schedule*, as equality rows may. Proposition (b) excludes this
  for a cell in the exact form with one tracked quantity, the law ARD∞ and no replacement on day `H_1`. Otherwise a
  rejection can occur: when tracked quantities move in opposite directions (the second example below), when the cell
  is replaced on day `H_1` (the first repetition then starts from the fixed replacement state), or when a linear form
  is stricter than the exact row." Line 1236: "So a cell in the exact form with one tracked quantity under ARD∞, not
  replaced on day `H_1`, accepts every safe schedule."

### 3. The rule for the quadratic damage penalty is ambiguous: an error, or an override?

- **Location.** Damage Regularisation, lines 2656-2660 (also the input key, line 515, and the check, line 609).
- **Problem.** Line 2658 says the quadratic penalty "requires `reliability_impl = "exact"` at every cell". The check
  at line 609 says "every cell is built in the form exact", which reads as an override of the user's form, not as
  an error. "Every cell" also covers the linear-row cells, whose value is ignored everywhere else. Line 2658 omits the
  condition `C_D > 0` that line 609 states. An implementer cannot tell whether a Gaussian cell with `tangent`, under
  `C_D > 0` and `quadratic`, raises an error or is silently made nonconvex. A silent switch to a nonconvex model is
  the behaviour that the `reliability_impl` check (line 608) exists to prevent.
- **Fix.** Line 2658: "If `C_D > 0` and `penalty_type = "quadratic"`, every cell with a quadratic reliability row
  must have `reliability_impl = "exact"`; otherwise a `ValueError` is raised (Section consistency checks). At a cell
  with a linear row the value is ignored, as elsewhere. With `C_D = 0` the penalty is not built and the rule does not
  apply." The same rule, in the same words, at lines 515 and 609.

### 4. The full problem omits the optional variance-cap row and the Gamma row without a random term

- **Location.** Full Optimization Problem, "Rows of a cell", lines 2817-2834.
- **Problem.** The section says it "collects the whole problem in one place". It omits the row
  `v_k <= V^user` that the Gaussian (line 1560), Wiener (line 1831) and rainflow (line 2532) sections add when
  `v_max_user` is given. It also cites only Eq. `eq:gm_row` for Gamma, while line 1921 replaces that row by
  `mu_k <= tau` when no term has a rate (`beta*` is then undefined).
- **Fix.** Add item 6: "where `v_max_user` is given (Gaussian, Wiener, rainflow Cantelli or Bernstein): the row
  `v_k <= V^user` on every day `k = 1, ..., T`." In item 3, after the table reference: "at a Gamma cell without a
  random term, the row `mu_k <= tau` (Section `sec:dyn_gamma`)."

### 5. "The costs computed from them are upper bounds" contradicts the repair-cost section

- **Location.** Encoding, "Meaning of the reported values", lines 1272-1278; against Repair Cost, lines 2615-2624.
- **Problem.** The paragraph says each reported value bounds its reference value "and so its true statistic", and
  then "the costs computed from them are upper bounds too". Read in sequence, this claims an upper bound on the true
  expected repair cost. Lines 2615-2624 state the exception: with a starting anchor above the true one, the first
  repair is charged less than the true removal. A difference of two upper bounds is not an upper bound.
- **Evidence.** `mu_0 = 0.5`, true anchor 0, default starting anchor 0.5, `rho = 0.5`, a repair on day 1 before any
  mission: charged `z = 0.5·(0.5 - 0.5) = 0`; true expected removal `0.5·0.5 = 0.25`.
- **Fix.** Line 1275: "The costs computed from them are upper bounds on the costs computed from the reference
  values: the repair cost (Proposition `prop:a_repair_cost`) and the damage penalty (Section `sec:damage_reg`).
  Against the true costs, Section `sec:repair_cost` states one exception, for a starting anchor above the true one."

### 6. The Gamma cap `A_1` is undefined for `eps >= 1/2`

- **Location.** Gamma, "The cap", lines 2099-2104; computation step 3, lines 2142-2148; Corollary `cor:gm_safe`,
  line 2121.
- **Problem.** The input checks accept any `eps` in (0, 1) at a Gamma cell (line 615). For `A < t`,
  `w(A) < 1/2`: with `r = A/t < 1`, `Q(rs, s) <= Q(rs, rs) < 1/2`, because the median of a Gamma law lies below its
  mean. So for `eps >= 1/2` every `A` in `[0, t)` satisfies `w(A) <= eps`, and "the largest `A` in `[0, t)`" does
  not exist. Step 3 then bisects on `[eps t, t)` and returns a value just below `t`; this is safe, but the text does
  not define it, and the corollary needs `A < t`.
- **Fix.** Line 2102: "The sharp cap `A_1(t, eps)` is the largest `A` in `[0, t)` with `w(A) <= eps`; for
  `eps >= 1/2` no largest value exists, because `w(A) < 1/2` for every `A < t` (the median of a Gamma law lies below
  its mean), and `A_1` is the value that step 3 returns, below `t`."

### 7. Symbols with two meanings across sections

- **Location and problem.**
  - `Phi` is the standard normal distribution function (lines 234, 1132, 1521, 1695) and the map of the operating
    phase (lines 227, 1204, 2691-2692; `Phi_h`). The notation table defines only the second meaning, and uses the
    first in the same table (`kappa_G = Phi^{-1}(1 - eps)`).
  - Calligraphic `C` is the certificate set `C_{il}` (line 1103) and the Gamma class `C(m)` (line 1930); both enter
    the safety argument of the Gamma cell.
  - `K` is the Chernoff quantity and the number of atoms in the proof of Theorem `thm:gm_oneterm` (line 2026);
    `kappa*`, `kappa'` (lines 2041, 2057) are levels of `phi*`, while `kappa = ln(1/eps)`; `eta` is the IG dispersion
    and an added mean in Lemma `lem:gm_rates` (line 1994); `s` is the generic tracked quantity of the encoding rows
    and the Chernoff parameter, both in the full-problem table (lines 2846-2870).
- **Fix.** Name the map of the operating phase `Pi` (and `Pi_h`) and define `Phi` in the notation table as the
  standard normal distribution function; name the Gamma class `G(m)` (calligraphic G); in the Gamma proof write `N`
  for the number of atoms, `nu*`, `nu'` for the levels and `Delta m` for the added mean; outside the rainflow section,
  write the Chernoff parameter always with its subscript, `s_{il}`, as line 1061 does.

## Suggestions

- **(M1)-(M3) labels.** Section `sec:encoding` defines (M1) as "`a` in [0, 1] and `delta s_k >= 0`" and (M2) as
  including `theta_0 <= xi_0` (lines 1110-1117). The model sections state other facts under the same labels: (M1) as
  monotone day maps, and (M2) without the day-0 inequality (lines 1564-1566, 1746-1747, 1835-1836, 2197-2198,
  2536-2539). State the same two facts in every model section, or redefine the labels once.
- **`rho = 1`.** The weights `w_t` are said to lie in (0, 1] (lines 1628, 1638, 1952, 2296). With `rho = 1` a weight
  is 0. Write `[0, 1]` and "a term of weight 0 is dropped".
- **IG remark "the largest cap for this order"** (lines 1726-1734). The two-point law is below `mu Z` in the
  stop-loss order, but it is not of the form of Proposition `prop:ig_order`; it is only a limit of such sums
  (conditionally independent copies given one common Bernoulli variable). Either add that sentence, or claim the
  statement for "every law below `mu Z` in the stop-loss order (the hypothesis of Theorem `thm:ig_cap`)".
- **Problem class in the overview** (lines 152-153): "the damage penalty is `inf_norm` or absent (`C_D = 0`)", as in
  line 2878; otherwise a model with `C_D = 0` and `penalty_type = quadratic` is classed as quadratic.
- **Acronyms.** ARD∞ and ARD1 appear in the overview (line 139) before their definition; give the expansion there.
  ARA (arithmetic reduction of age) is never expanded (line 907).
- **ARD1 with `allow_replacement = false`.** The Wiener remark states the consequence (line 1846); Proposition
  `prop:a_ard1_replacement` does not. Add: "With `allow_replacement = false`, the train of such a cell cannot fly a
  mission with a positive increment for the cell in the operating phase." A warning at input time would help the
  user, since the result is an infeasible model.
- **Replacement inputs when replacements are off.** `beta_new` enters `beta*` (line 1914) and `b_new` enters the
  Bernstein constant (line 2458) even when `allow_replacement = false`, while the range checks skip replacement values
  in that case (line 626). Excluding them then gives a larger, still safe cap.
- **Smaller points.** The Gaussian text covers "each of these terms" with the three-sigma check (line 1453), but the
  check list has no `mu_new >= 3 sqrt(v_new)` (line 639). The input key `eta` speaks of "the inverse Gaussian law of
  the state" (line 492), while Section `sec:dyn_ig` gives that law to each term only. The variance ranges
  `V^C`, `V^B` and `2 tau^2/kappa` grow without bound as `eps -> 1`, as `V_max` does as `eps -> 0.5`; one sentence
  next to the remark on `V_max` would cover them.
