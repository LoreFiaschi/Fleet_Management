Status: done (11 findings: 3 [error], 5 [inconsistency], 3 [meaning]; 3 doubts). Block read in full: scoping.tex 834-1614 and 4162-4356, spec/spec.tex 1304-1542 (= fragments_B.tex 392-629).

# Review R3a: degradation processes and the five Route-I bounds

Scripts (scratchpad): `r3a_checks.py` (output `r3a_checks_out.txt`), `r3a_tail_experiment.py`
(output `r3a_tail_experiment_out.txt`), `r3a_bernstein_closed_form.py` (output
`r3a_bernstein_closed_form_out.txt`), `r3a_refs.py`, `r3a_unique.py` (every `old` string below occurs
exactly once in its file).

Spec line numbers: fragments_B line = spec/spec.tex line - 912 in this block.

## Findings

### F1 [inconsistency] Two definitions of the support and the budget of the initial and replacement damage; the Popoviciu check rejects valid inputs
- Where: scoping.tex:1459 (A2) and spec/spec.tex:1322-1326 (fragments_B.tex:410-414, A2) define
  `support_0` as the *largest width of the independent terms* of the initial damage, and `R_0` as the *sum
  of their squared widths*. spec/spec.tex:404-405 (fragments_C.tex:72-73) define `support_0` as the
  *width of the range of the initial damage* and `R_0` as *the square of that width*. The check
  `v_0 <= (b^0)^2/4` (spec/spec.tex:578, fragments_C.tex:246; reason in scoping.tex:3634) and its
  replacement twin are valid only under the second definition of `support_0`.
- Problem: under the A2 definition, the check rejects valid data. Popoviciu's inequality bounds the
  variance of one term by its own width; the variance of a sum of terms can exceed (largest width)^2/4.
- Evidence (`r3a_checks.py`, part 5): initial damage = sum of 4 independent terms, each 0 or 0.1 with
  probability 1/2. Then v_0 = 0.01, largest term width b^0 = 0.1, (b^0)^2/4 = 0.0025 < v_0: the check
  fails. Bernstein's hypothesis holds (each centred term <= 0.05 <= b^0), so the data are valid under A2.
- Fix (recommended: take `support_0` as the width of the range of the random part, which keeps the check
  valid; the Bernstein proof stays valid because the width of a sum of independent terms is the sum of
  their widths, so every term has width at most b^0; keep `R_0` as the sum of squared widths, which is the
  tighter Hoeffding budget):
  - file `scoping.tex`
    - old: `its Hoeffding budget is the sum of their squared widths, and its support is the largest width.`
    - new: `its Hoeffding budget is the sum of their squared widths, and its support is the width of the range of their sum.`
  - file `merge/spec_drafts/fragments_B.tex`
    - old: `$R_0$ (\texttt{R\_0}) and largest width $b^0$ (\texttt{support\_0}); a`
    - new: `$R_0$ (\texttt{R\_0}); the range of their sum has width $b^0$ (\texttt{support\_0}); a`
  - file `merge/spec_drafts/fragments_C.tex`
    - old: `Initial Hoeffding budget $R_{0,i\ell}$: the square of the width of the range of the initial damage`
    - new: `Initial Hoeffding budget $R_{0,i\ell}$: the sum of the squared widths of the independent terms of the initial damage`
  - Alternative: keep A2 as it is and delete `v_0 <= (b^0)^2/4` and `v^new <= (b^new)^2/4` from the
    check (fragments_C.tex:246 and scoping.tex:3634). This keeps the tighter Bernstein constant but loses
    the check.
- Confidence: high

### F2 [error] The dominating shared-rate Gamma does not always exist
- Where: scoping.tex:1419 (Route II, "Dominating shared-rate Gamma").
- Problem: the text says that a shape alpha_j with Ga-survival >= empirical survival "for all x" always
  exists, since the Gamma survival tends to 1 as alpha grows. The empirical survival equals 1 on
  [0, smallest record). A Gamma survival is < 1 at every x > 0, for every alpha. So no Gamma law
  dominates when every recorded increment is positive, which is the usual case for a damage increment.
  Dominance exists exactly when some record equals 0.
- Evidence: `r3a_checks.py`, part 6: records {0.5, 0.7, 0.9}; at x = 0.25 the empirical survival is 1,
  the Gamma survival is 1 - P(alpha, 0.25) < 1 for alpha = 1, 10, 100, ... (log of the CDF printed:
  -1.5, -29.2, -502.6, ...). Proof of the "exactly when": with a share p_0 > 0 of zero records,
  empirical survival <= 1 - p_0 on x > 0 and = 0 beyond the largest record x_max; the Gamma survival is
  decreasing in x and its value at x_max tends to 1, so a large alpha dominates.
- Fix: file `scoping.tex`
  - old: `This is always possible, since $\overline{\mathrm{Ga}}(x;\alpha,\beta^\star) \to 1$ as $\alpha \to \infty$.`
  - new: `This is possible exactly when some recorded increment is~$0$. Then $\widehat{\bar F}_j(x) \le 1 - p_0$ for $x > 0$, with $p_0 > 0$ the share of zero records, and $\overline{\mathrm{Ga}}(x;\alpha,\beta^\star) \ge \overline{\mathrm{Ga}}(x_{\max};\alpha,\beta^\star) \to 1$ as $\alpha \to \infty$ on $[0, x_{\max}]$, with $x_{\max}$ the largest record. If every record is positive, $\widehat{\bar F}_j(x) = 1$ below the smallest record, while $\overline{\mathrm{Ga}}(x;\alpha,\beta^\star) < 1$ for every $x > 0$, so no Gamma law dominates.`
- Confidence: high

### F3 [inconsistency] The mean-and-support bound of the CGF needs the range [0, b]; the block later defines b as a width only
- Where: scoping.tex:1378-1382 (mean-and-support bound), against scoping.tex:1486-1487 ("The support of
  an increment X is the width b of an interval [l, l+b] ... The width has this one meaning in every
  bound") and 1492 ("the range may start at any point l >= 0"). Line 1164 instead says "support [0, b]".
- Problem: the bound psi(s) <= ln(1 + (mu/b)(e^{sb} - 1)) follows from convexity on [0, b]. With only the
  width known, it is false.
- Evidence (`r3a_checks.py`, part 4): X = 0.5 almost surely, range [0.5, 0.51], width b = 0.01, mean 0.5,
  s = 5. Then psi(5) = 2.5, but the right side is ln(1 + 50(e^{0.05} - 1)) = 1.271. The correct bound on
  [l, l+b] is s l + ln(1 + ((mu - l)/b)(e^{sb} - 1)) = 2.5 here.
- Fix: file `scoping.tex`
  - old: `\textbf{Mean-and-support bound.} The first is`
  - new: `\textbf{Mean-and-support bound.} For an increment whose range is $[0, b_{\ell j}]$, so that it starts at~$0$, the first is`
- Confidence: high

### F4 [error] The derivation of "closed form and per-mission support cannot be obtained together" does not hold
- Where: scoping.tex:4204-4210 (drv:bernstein_curvature) and the claim it supports, scoping.tex:1371.
- Problem: (1) "This is exactly the Chernoff--Bennett bound (eq:chernoff_permission)" is false.
  eq:chernoff_permission uses the exact CGF psi_lj(s) of the uncentred increment. The displayed sum is
  Bernstein's upper bound of the *centred* CGF: it lacks the term s mu_lj, it is valid only for
  0 < s < 3/b_lj, and it gives a looser bound. The function (s^2 v/2)/(1 - s b/3) is Bernstein's bound,
  not Bennett's (Bennett uses v(e^{sb} - 1 - sb)/b^2). (2) "can enter ... only through" is asserted, not
  derived, so the impossibility claim "cannot be obtained together" has no proof. What can be shown: with
  one constant b the choice s = a/(Var[D] + b a/3) turns the bound into the closed form; with one b per
  mission the minimisation over s stays.
- Evidence: `r3a_bernstein_closed_form.py`: the exponent -s a + s^2 V/(2(1 - s b/3)) at
  s = a/(V + b a/3) equals -a^2/(2(V + b a/3)) (sympy, difference 0); 1 - s b/3 = 3V/(3V + a b) > 0 for
  V > 0; random tests of centred CGF <= (s^2 v/2)/(1 - s b/3): maximum difference -2.9e-15.
- Fix: file `scoping.tex`
  - old (lines 4204-4210, verbatim):
    ```
    A support $b_{\ell j}$ per mission can enter the correction only through the cumulant sum per term,
    \begin{equation}
      \sum_j n_{ij}\,(s^2 v_{\ell j}/2)/(1 - s\,b_{\ell j}/3),
    \end{equation}
    minimised over~$s$.
    This is exactly the Chernoff--Bennett bound~\eqref{eq:chernoff_permission}.
    Therefore the closed form~\eqref{eq:bernstein_constraint}, which has no~$s$, and a support per mission cannot be obtained together.
    ```
  - new:
    ```
    Bernstein's bound of the cumulant generating function of one increment reads, for $0 < s < 3/b_{\ell j}$,
    \begin{equation}
      \psi_{\ell j}(s) \le s\,\mu_{\ell j} + (s^2 v_{\ell j}/2)/(1 - s\,b_{\ell j}/3).
    \end{equation}
    In~\eqref{eq:chernoff_permission}, in place of $\psi_{\ell j}$, it keeps a support per mission, and the bound is still minimised over~$s$.
    With one constant $b_\ell \ge \max_j b_{\ell j}$ in every term, the choice $s = a/(\operatorname{Var}[D] + b_\ell\, a/3)$, with $a = \tau - \mathbb{E}[D]$, gives the closed form~\eqref{eq:bernstein_constraint}, which has no~$s$.
    ```
  - old: `The closed form~\eqref{eq:bernstein_constraint}, which has no~$s$, and a support per mission cannot be obtained together.`
  - new: `The closed form~\eqref{eq:bernstein_constraint}, which has no~$s$, uses one support constant; with a support per mission, the bound keeps the minimisation over~$s$.`
- Confidence: medium (item 1 is certain; item 2 is a missing proof, not a proven false claim)

### F5 [inconsistency] "The support never enters through a shared worst-case value" contradicts the Bernstein constant
- Where: scoping.tex:1178 (bullet "Support"), against scoping.tex:1210 (table caption), 1332-1340 and
  eq:bernstein_constraint, where Bernstein's closed form uses the shared constant b_l = max of the supports.
- Problem: the bullet states an absolute rule that the same section breaks two paragraphs later.
- Fix: file `scoping.tex`
  - old: `\textbf{Support.} The support enters only through the support $b_{\ell j}$ of each mission, never through a shared worst-case value.`
  - new: `\textbf{Support.} The support enters through the support $b_{\ell j}$ of each mission; only the closed form of Bernstein's inequality merges the supports into one shared value (see below).`
- Confidence: high

### F6 [meaning] The cumulative shape of the inverse Gaussian uses the letter lambda for two quantities
- Where: scoping.tex:994-997.
- Problem: the equation reads lambda = (sum_t mu_t)^2/(mu^2/lambda). The left lambda is the cumulative
  shape; the lambda inside the bracket is a per-day shape lambda_t of the shared dispersion. As written
  it says (sum mu_t)^2 = mu^2, which is false. The content is right: with d = mu_t^2/lambda_t for every t,
  the cumulative shape is (sum mu_t)^2/d, which equals the table entry (sum mu_t)^3/sum(mu_t^3/lambda_t)
  (`r3a_checks.py`, part 7: 45.633 both ways; simulation: KS p = 0.12, mean 3.696 vs 3.7, variance
  1.102 vs 1.11).
- Fix: file `scoping.tex`
  - old: `The means add up, and the shared dispersion determines the cumulative shape,`
  - new: `The means add up, and the shared dispersion~$d$ determines the cumulative shape~$\lambda_{1:k}$,`
  - old: `\lambda = (\sum_t \mu_t)^2\big/(\mu^2/\lambda).`
  - new: `\lambda_{1:k} = (\textstyle\sum_t \mu_t)^2\big/ d, \qquad d = \mu_t^2/\lambda_t \ \text{for every } t.`
- Confidence: high

### F7 [meaning] The minimum of several tangents does not give linear constraints
- Where: scoping.tex:1369 and scoping.tex:4202 (drv:bernstein_curvature).
- Problem: "Both replacements give linear constraints in the counts". A fixed tangent gives one linear
  constraint. The minimum of several tangents gives "linear part + min_i T_i <= tau", that is
  "there is an i with linear part + T_i <= tau": a union of half-spaces, not convex, so not linear. It
  needs the SOS2 set (or binaries), as line 1368 says. The word "linear" changes the meaning.
- Fix: file `scoping.tex`
  - old: `Both replacements give linear constraints in the counts that imply~\eqref{eq:bernstein_constraint}, so they stay conservative.`
  - new: `Both replacements imply~\eqref{eq:bernstein_constraint}, so they stay conservative: a fixed tangent gives one linear constraint in the counts, and the minimum of tangents gives linear constraints together with the SOS2 set.`
  - old: `gives linear constraints in the counts that imply~\eqref{eq:bernstein_constraint}.`
  - new: `gives constraints in the counts that imply~\eqref{eq:bernstein_constraint}: one linear constraint for a fixed tangent, and linear constraints with an SOS2 set for the minimum.`
- Confidence: high

### F8 [inconsistency] Closure of the compound Poisson model: "Always" in one table, "Same F_Y" in the other
- Where: scoping.tex:878 (Table tab:deg_models: "Always"), scoping.tex:937 ("The sum of independent
  compound Poisson processes is always a compound Poisson process"), against scoping.tex:978 (Table
  tab:discrete_models: closure condition "Same F_Y").
- Problem: the sum of independent CP(nu_t, F_t) is CP(sum nu_t, F) with the mixture
  F = sum_t nu_t F_t / sum_t nu_t, whatever the F_t. So closure in the CP family needs no condition; a
  shared F_Y only keeps the jump law, which the marginal column of the second table assumes. The two
  tables state the condition for the same model differently.
- Fix: file `scoping.tex`
  - old: `& Same $F_Y$ \\`
  - new: `& Always; same $F_Y$ keeps the jump law \\`
- Confidence: medium (if "closure" is meant inside a parametric family of jump laws, the first table
  and line 937 need the change instead)

### F9 [meaning] The Gumbel row of Table tab:sn_scatter is the Weibull row in other parameters
- Where: scoping.tex:1055-1058 and 1100-1107 (Gumbel), against 1050-1054 and 1090-1098 (Weibull);
  scoping.tex:1033 says the table gives "the three most common choices".
- Problem: if ln N_f is Gumbel (minimum) with location mu(sigma_a) and scale beta, then N_f is Weibull
  with shape 1/beta and scale e^{mu(sigma_a)}. The "log-Gumbel" increment n/N_f is then the scaled
  Frechet law of the Weibull row. The text presents three distinct families; two coincide. (This is a
  mathematical identity; I do not judge the attribution to the cited paper.)
- Evidence: `r3a_checks.py`, part 9: exp(Gumbel_min(10, 0.5)) against Weibull(shape 2, scale e^10),
  KS p = 0.68 on 200000 draws.
- Fix: file `scoping.tex`
  - old: `As with the other two families, the log-Gumbel is not closed under summation.`
  - new: `Then $N_f$ is Weibull with shape $1/\beta$ and scale $e^{\mu(\sigma_a)}$, so this row is the Weibull row in other parameters, and the log-Gumbel law of the increment is a Fr\'echet law. It is not closed under summation either.`
- Confidence: high (mathematics); the wording of "three most common choices" is left to the owner

### F10 [error] Proposition prop:st_closure is cited for the law of the Gamma surrogate
- Where: scoping.tex:1420.
- Problem: "By Proposition prop:st_closure, the accumulated surrogate is then Ga(A, beta*)". The
  proposition gives the order D <=_st D-tilde. The law Ga(A, beta*) comes from the closure of the Gamma
  family under a shared rate (Table tab:discrete_models), not from the proposition.
- Fix: file `scoping.tex`
  - old: `By Proposition~\ref{prop:st_closure}, the accumulated surrogate is then $\widetilde{D} = \mathrm{Ga}(A,\beta^\star)$, with a cumulative shape`
  - new: `The accumulated surrogate is then $\widetilde{D} = \mathrm{Ga}(A,\beta^\star)$, by the closure under a shared rate, and $D \le_{\mathrm{st}} \widetilde{D}$ by Proposition~\ref{prop:st_closure}. Its law has a cumulative shape`
- Confidence: high

### F11 [inconsistency] "Diffusion coefficient" names sigma^2 in the table and sigma in the text
- Where: scoping.tex:851 (caption of tab:deg_models: "sigma^2 (diffusion coefficient)") and
  scoping.tex:903 ("sigma > 0 the diffusion coefficient").
- Problem: one term for two quantities. The increment law N(mu(t-s), sigma^2(t-s)) fits the caption.
- Fix: file `scoping.tex`
  - old: `$\mu>0$ the drift rate, and $\sigma>0$ the diffusion coefficient.`
  - new: `$\mu>0$ the drift rate, and $\sigma^2$ the diffusion coefficient, with $\sigma>0$.`
- Confidence: high

## Checked and correct (no finding)

- Table tab:deg_models and tab:discrete_models: Gamma, Wiener, IG moments and closure; IG marginal
  (sum mu)^3/sum(mu^3/lambda); Wiener first passage IG(tau/mu, tau^2/sigma^2) (mean 4.0, variance 1.44
  in `r3a_checks.py`, part 11); log-normal increment moments (part 8: 0.43996/0.03358 simulated against
  0.43982/0.03356); moment-matched Gamma (part 10).
- Markov, Cantelli, Chernoff, Bernstein and Hoeffding forms and their solution for epsilon
  (eq:bernstein_constraint, its squared form and side row; Table tab:five_bounds; all five derivations
  in drv:five_rows, including the edge cases d_k = 0, R = 0, b_l = 0); Proposition prop:rf_maps and its
  proof (Jensen step for K under ARD_inf and ARD_1); drv:width (centred term, Popoviciu); the example
  "width 0.2, variance 0.01"; the Chernoff example (1.886, -3.11, -0.11, -2.30, failure probability 0.5).
- Spec rainflow section against the scoping: big-M constants V^C, V^B, 2 tau^2/kappa, s tau - kappa
  (each is the largest value the exact row admits at mu >= 0), f(mu) and f(0) of the forms table, the
  Bernstein c_2, c_1 of the linear-forms table, the tangent condition mu_r < sqrt(tau(tau - 2c)) (it is
  exactly the condition that the tangent row is positive at mu = 0; sympy, part 1), the day maps and
  factors, the defaults (known damage: zero variance, support, budget; K = s mu), the requirements
  tau > 2 kappa b/3 and s > kappa/tau. All 34 references and 19 citations of the block resolve
  (`r3a_refs.py`). spec/spec.tex 1304-1542 equals fragments_B.tex 392-629.
- Hoeffding 1963, Theorem 2 (literature/): the theorem with a_i <= X_i <= b_i, as cited at line 1237.

## Doubts

- **The numerical experiment, scoping.tex:1395-1399.** The sentence "Bernstein lay between them, and it
  did better than Cantelli only as the tail deepened and the number of summands grew" contradicts itself:
  lying between Chernoff and Cantelli means doing better than Cantelli. The original test code is not in
  the repository. My own run (`r3a_tail_experiment.py`; increments in [0, 1], k in {30, 100}, true tail
  about 1e-4) gives Chernoff 9 to 57 times the true tail, Cantelli 404 to 2346 times, and Bernstein
  between them in 11 of 12 cases; the exception is a skewed law (Beta(0.5, 5)) at k = 30, where
  Bernstein is worse than Cantelli. So "Bernstein usually lay between them; with few summands and a
  skewed law it could be worse than Cantelli" fits my data. I cannot check the original numbers.
- **The Bernstein side row, scoping.tex:1573.** "That point is safe, but it is separated from the rest of
  the feasible set" holds only for c > 0. With b_l = 0 (c = 0) the point (tau, 0) lies on the main
  branch tau - mu >= sqrt(2 kappa v). Minor; the input b_l = 0 is allowed.
- **The natural range tau of the mean at Chernoff cells (outside the block, scoping.tex:3546-3549).** The
  text says each range is "the largest value that the exact row of the bound admits at mu >= 0". The
  Chernoff row does not bound mu. The bound mu <= tau still holds for the reference trajectory, because
  the input checks give K >= s mu along it, so mu <= (s tau - kappa)/s < tau. The justification, not the
  constant, is off.
