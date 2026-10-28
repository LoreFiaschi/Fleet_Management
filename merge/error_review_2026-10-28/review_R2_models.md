Status: done (12 findings: 8 error, 1 inconsistency, 3 meaning; 2 doubts)

# Review R2: Gaussian, inverse Gaussian, Wiener and Gamma models

Scripts and outputs: folder
`C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\r2\`
(called `r2\` below). Line numbers of `scoping.tex` are those of 28 Oct 2026, 21:33. The file changed
outside this block during the review, so the `old:` strings are the reliable locators. Every `old:` string
was checked to occur exactly once in its file. `spec/spec.tex` lines 914-1303 are identical to
`fragments_B.tex` lines 1-390.

## Findings

### F1 [error] Proof of Lemma lem:gamma_rates: the Laplace-transform identity is wrong by the factor -1/theta
- Where: scoping.tex:4684-4693 (proof of Lemma lem:gamma_rates, "This is the transform of the rate ... of Pr(S > t)").
- Problem: the displayed quantity is the rate of change of E e^{-theta S}, the Laplace transform of the law of S.
  It is not the Laplace transform of the rate of Pr(S > t). The two differ by the factor -1/theta. The lemma
  is still true, but the step as written is false.
- Evidence: integral_0^inf e^{-theta t} Pr(S > t) dt = (1 - E e^{-theta S})/theta. By Fubini,
  int e^{-theta t} Pr(t-y < S <= t) dt = E[e^{-theta S}(1 - e^{-theta y})]/theta and int e^{-theta t} f_S(t) dt = E e^{-theta S}.
  So the transform of the claimed rate is -(1/theta) times the displayed rate.
  Counterexample to the identity as written: S ~ Ga(a, 1), drift rate delta_dot = 1, Lambda_dot = 0. The displayed
  rate is -theta (1+theta)^{-a} < 0. The rate of Pr(S > t) is f_S(t) > 0, with transform (1+theta)^{-a}.
- Fix: file `scoping.tex`
  - old: `This is the transform of the rate`
  - new: `Since $\int_0^\infty e^{-\vartheta t}\,\Pr(S > t)\,\mathrm{d}t = (1 - \mathbb{E}e^{-\vartheta S})/\vartheta$, this rate times $-1/\vartheta$ is the transform of the rate`
- Confidence: high

### F2 [error] Gaussian loop paragraph: sigma^2_T denotes two things in one equation
- Where: scoping.tex:2747 (paragraph "Gaussian" of subsubsection sec:loop_per_model, "Without interventions the variance accumulates").
- Problem: on the left, sigma^2_T is the variance of D_T. On the right, sigma^2_k is the variance increment of day k,
  and the sum runs up to k = T. So sigma^2_T appears on both sides with two meanings. Read literally, the equation
  says sigma^2_{H_1} + sum_{k=H_1+1}^{T-1} sigma^2_k = 0.
- Evidence: by inspection. The paragraph defines sigma_T and sigma_{H_1} as the standard deviations of D_T and
  D_{H_1} (equation before eq:loop_gauss). The same paragraph later writes a Gaussian increment as N(mu^inc, v^inc).
- Fix: file `scoping.tex`
  - old: `  \sigma^2_T = \sigma^2_{H_1} + \sum_{k = H_1+1}^{T} \sigma^2_k > \sigma^2_{H_1}.`
  - new: `  \sigma^2_T = \sigma^2_{H_1} + \sum_{k = H_1+1}^{T} v^{\mathrm{inc}}_k > \sigma^2_{H_1},`
  - and add, on a new line right after the `\end{equation}` that closes this equation: `where $v^{\mathrm{inc}}_k$ is the variance of the increment of day~$k$ ($0$ on a day without a mission).`
- Confidence: high

### F3 [error] Computation of the Gamma cap: the numerical check of the single peak does not cover the ratios A/t that the computation uses
- Where: scoping.tex:4820 (derivation drv:gamma_computation, fact 2: "checked numerically for t from 10^-3 to 100 and
  A/t from 0.05 to 0.999"); it justifies scoping.tex:2611-2613 and the spec step 2 of "Computation of A_1"
  (spec/spec.tex:1294-1297, fragments_B.tex:381-384).
- Problem: the bisection evaluates w(A) for A/t from epsilon upwards (lower end epsilon t). The caps themselves lie
  below 0.05 for epsilon <= 0.01 and t <= t1: A_1/t = r* = 0.00355 (epsilon = 0.001) and 0.0353 (epsilon = 0.01).
  So fact 2 is not checked where the computation needs it. Also t = beta* tau can lie outside [10^-3, 100].
- Evidence: `r2\gamma_check2.py` (output `r2\gamma_check2_output.txt`). With r = A/t and s = t/c,
  Q(A/c, t/c) = Q(rs, s), so a single peak of s -> Q(rs, s) on s > 0 gives the claim for every t. On 400 values of r
  from 10^-5 to 0.999 and a log grid of 4000 points of s in [10^-12, 10^8], the slope changes sign at most once for
  every r: 0 failures. The property holds; the fix states the range that the use needs.
- Fix: file `scoping.tex`
  - old: `This was checked numerically for $t$ from $10^{-3}$ to $100$ and $A/t$ from $0.05$ to $0.999$.`
  - new: `With $r = A/t$ and $s = t/c$, the function is $s \mapsto Q(rs, s)$ on $s \ge t$. A single peak of this function on $s \in [10^{-12}, 10^{8}]$ was checked numerically for $r$ from $10^{-5}$ to $0.999$, which covers the bisection for $\varepsilon \ge 10^{-5}$.`
- Confidence: high (the range defect); the new check is mine.

### F4 [error] Derivation of the properties of w: "continuous" does not follow from the reason given
- Where: scoping.tex:4789 (derivation drv:gamma_w, first bullet); the claim stands at scoping.tex:2591.
- Problem: "w is continuous and non-decreasing in A, because Q(a, x) increases in a". The monotonicity of Q in a
  gives only that w is non-decreasing. w is a maximum over the non-compact set c in (0, 1], so continuity needs an
  argument. The claim is true.
- Evidence: c Ga(A/c, 1) has mean A and variance cA, so by Chebyshev's inequality Q(A/c, t/c) <= cA/(t - A)^2.
  For A in [0, A'] with A' < t this bound is uniformly small for small c. So the maximum can be taken over a compact
  interval [c_0, 1], where Q(A/c, t/c) is jointly continuous in (A, c). Hence w is continuous on [0, A'].
- Fix: file `scoping.tex`
  - old: `  \item $w$ is continuous and non-decreasing in~$A$, because $Q(a, x)$ increases in~$a$.`
  - new: `  \item $w$ is non-decreasing in~$A$, because $Q(a, x)$ increases in~$a$. It is continuous on $[0, A']$ for every $A' < t$: by Chebyshev's inequality $Q(A/c, t/c) \le cA/(t - A)^2$, so the maximum can be taken over a compact interval $[c_0, 1]$, where $Q(A/c, t/c)$ is continuous in $(A, c)$.`
- Confidence: high

### F5 [error] Table tab:ordering_st, compound Poisson row: the equality of the jump laws is listed as a necessary condition
- Where: scoping.tex:2336 (Table tab:ordering_st, row Comp. Poisson); scoping.tex:2247 says that the tables give
  "the necessary and sufficient conditions".
- Problem: the row reads: X <=st Y if and only if nu_1 <= nu_2, F_{Y_1} = F_{Y_2} and Y >= 0. The equality
  F_{Y_1} = F_{Y_2} is not necessary. The paragraph "Compound Poisson" states the correct result: for one jump law
  with Y >= 0 and Pr(Y > 0) > 0, the order holds if and only if nu_1 <= nu_2.
- Evidence: counterexample. X ~ CP(nu, delta_1) (unit jumps) and Y ~ CP(nu, delta_2) (jumps of size 2). With one
  Poisson count N for both, X = N <= 2N = Y, so X <=st Y although F_{Y_1} != F_{Y_2}.
- Fix: file `scoping.tex`
  - old: `  & $\nu_1 \le \nu_2$, $F_{Y_1} = F_{Y_2}$, $Y \ge 0$ \\`
  - new: `  & for $F_{Y_1} = F_{Y_2}$, $Y \ge 0$, $\Pr(Y > 0) > 0$: $\nu_1 \le \nu_2$ \\`
- Confidence: high
- Note: the example of the paragraph reproduces: likelihood ratio 463.3 at x = 9 and 1.810 at x = 10; the ratio of
  the survival functions decreases at x = 1, ..., 8 (`r2\cp_check.py`, output `r2\cp_check_output.txt`).

### F6 [error] Sharpness of the CVaR cap: the two-point law exceeds tau with probability at least epsilon', not exactly epsilon'
- Where: scoping.tex:4650 (derivation drv:ig_sharp).
- Problem: the two-point law takes the value mu E[Z | Z <= q'] with probability 1 - epsilon'. When this value is
  above tau, the law exceeds tau with probability 1. The conclusion (no larger cap is safe) still holds.
- Evidence: E[Z | Z <= q] = (1 - F_Z(1/q))/(1 - epsilon). At epsilon = 0.1, eta = 1: F_Z(1/q) = 0.1/0.2986 = 0.3349,
  so E[Z | Z <= q] = 0.739 (values from `r2\ig_check.py`). For mu = 2 tau > mu_bar, both values exceed tau.
- Fix: file `scoping.tex`
  - old: `and exceeds~$\tau$ with probability~$\varepsilon'$.`
  - new: `and exceeds~$\tau$ with probability at least~$\varepsilon'$.`
- Confidence: high

### F7 [error] Table tab:ig_cap: the entry for epsilon = 0.01, eta = 1 is 0.155, not 0.156
- Where: scoping.tex:2500 (Table tab:ig_cap, row epsilon = 0.01).
- Problem: the value is 0.15545..., which rounds to 0.155.
- Evidence: `r2\ig_check.py` (output `r2\ig_check_output.txt`) and `r2\ig_mp_check.py` (40 digits; the closed form
  tau eps/F_Z(1/q) and the quadrature of the tail mean agree): mu_bar/tau = 0.1554505574. The other 14 entries
  reproduce (0.4913, 0.2419, 0.1006, 0.03697, 0.01897; 0.5696, 0.32153, 0.06794, 0.04011; 0.6926, 0.4773, 0.2986,
  0.1826, 0.1411). The quantile q agrees with `scipy.stats.invgauss(mu=eta, scale=1/eta).isf(eps)`, as the spec says.
- Fix: file `scoping.tex`
  - old: `$\varepsilon = 0.01$ & 0.570 & 0.322 & 0.156 & 0.0679 & 0.0401 \\`
  - new: `$\varepsilon = 0.01$ & 0.570 & 0.322 & 0.155 & 0.0679 & 0.0401 \\`
- Confidence: high

### F8 [error] Gamma: A_1/A_hat at t = 0.1, epsilon = 0.01 is 0.641, not 0.642 (two places)
- Where: scoping.tex:2629 (table after "For epsilon = 0.01") and scoping.tex:2639 ("The sharp cap there is A_1 = 0.642 A_hat").
- Problem: A_1/A_hat = 0.641471, which rounds to 0.641. The value 0.642 comes from rounding 0.6415 twice.
- Evidence: `r2\gamma_check2.py`: A_hat = 0.0055012397, A_1 = 0.0035288876 = r* t with r* = 0.0352889, ratio 0.641471.
  The mpmath run `r2\gamma_check.py` (output `r2\gamma_check_output.txt`) gives 0.64147. The other entries
  reproduce: 0.14185, 0.96063, 1, 1.
- Fix: file `scoping.tex`
  - old: `$A_1/\hat A$ & 0.142 & 0.642 & 0.961 & 1 & 1 \\`
  - new: `$A_1/\hat A$ & 0.142 & 0.641 & 0.961 & 1 & 1 \\`
  - old: `The sharp cap there is $A_1 = 0.642\,\hat A$.`
  - new: `The sharp cap there is $A_1 = 0.641\,\hat A$.`
- Confidence: high

### F9 [meaning] "Dispersion" names two different quantities of the inverse Gaussian model
- Where: scoping.tex:2384 ("for IG (shared dispersion mu^2/lambda)"), 2397 ("has the dispersion mu^2/lambda = eta a"),
  2407 ("multiplies the dispersion mu^2/lambda by c"), against scoping.tex:2434 ("one dispersion eta > 0") and the
  caption of Table tab:ig_cap (2492, "five dispersions"). Also scoping.tex:3640 (input checks), fragments_B.tex:118,
  fragments_A.tex:128 (notation table) and fragments_C.tex:125 (key `eta`), which call eta the dispersion.
- Problem: in the formulation eta = mu/lambda of each term, while the closure condition calls mu^2/lambda = eta a
  the dispersion. The sentence "has the dispersion mu^2/lambda = eta a, which grows with a" meets the sentence
  "The formulation gives each inverse Gaussian component one dispersion eta". A reader cannot tell whether the
  dispersion of a component is fixed or grows with the mean. The background (scoping.tex:929, 994, 999) also
  defines "the shared dispersion d = mu_t^2/lambda_t", so the word belongs to mu^2/lambda.
- Evidence: IG(a, a/eta) has mu^2/lambda = a^2/(a/eta) = eta a and mu/lambda = eta; the text already calls eta
  "its squared coefficient of variation" (scoping.tex:2440).
- Fix: rename eta.
  - file `scoping.tex`, old: `The formulation gives each inverse Gaussian component one dispersion $\eta > 0$.`
    new: `The formulation gives each inverse Gaussian component one parameter $\eta > 0$.`
  - file `scoping.tex`, old: `for three reliability levels and five dispersions.}`
    new: `for three reliability levels and five values of~$\eta$.}`
  - file `scoping.tex`, old: `The dispersion of the inverse Gaussian law must be positive.`
    new: `The squared coefficient of variation of the inverse Gaussian terms must be positive.`
  - file `merge/spec_drafts/fragments_B.tex`, old: `and $\eta > 0$ the dispersion of the cell (input \texttt{eta}).`
    new: `and $\eta > 0$ the squared coefficient of variation of the terms of the cell (input \texttt{eta}).`
  - file `merge/spec_drafts/fragments_A.tex`, old: `& inverse Gaussian dispersion; unit law; its`
    new: `& squared coefficient of variation of an inverse Gaussian term; unit law; its`
  - file `merge/spec_drafts/fragments_C.tex`, old: `{Dispersion $\eta_{i\ell}$ of the inverse Gaussian law of each term`
    new: `{Squared coefficient of variation $\eta_{i\ell}$ of the inverse Gaussian law of each term`
  - Alternative with fewer edits: keep "dispersion" for eta and drop the word before mu^2/lambda at scoping.tex:2384,
    2397, 2407; the background (929, 994, 999) would then still use it for mu^2/lambda.
- Confidence: high (the double use); the choice of name is the owner's.

### F10 [meaning] The symbol r*(epsilon) is never defined, and the derivation does not say why the peak lies at s >= t
- Where: scoping.tex:2641-2647 (paragraph "Below the threshold") and scoping.tex:4830 (derivation drv:gamma_below).
- Problem: r*(epsilon) appears only inside the claim A_1 = r* t. The derivation says that "the peak of
  s -> Q(rs, s) at the cap lies at s >= t" but not where the peak is. The missing fact: at r = r*(epsilon),
  the peak lies at s = t1(epsilon).
- Evidence: `r2\gamma_rstar_peak.py` (output `r2\gamma_rstar_peak_output.txt`): the r at which max_s Q(rs, s) = epsilon
  is 0.00355013, 0.0352889, 0.330617, with the peak at s = 0.43614, 0.44839, 0.61422 = t1(epsilon). So
  r* = A_hat(t1)/t1, and the values 0.00355, 0.0353, 0.331 of the text reproduce.
- Fix: file `scoping.tex`
  - old: `For $t \le t_1(\varepsilon)$ the peak of $s \mapsto Q(rs, s)$ at the cap lies at $s \ge t$, so the maximum does not depend on~$t$, and`
  - new: `Let $r^*(\varepsilon)$ be the value of~$r$ at which the peak value of $s \mapsto Q(rs, s)$ equals~$\varepsilon$. This peak lies at $s = t_1(\varepsilon)$, so $r^*(\varepsilon) = \hat A(t_1, \varepsilon)/t_1$. For $t \le t_1(\varepsilon)$ the peak lies at $s \ge t$, so the maximum does not depend on~$t$, and`
- Confidence: high

### F11 [inconsistency] The CVaR of Z is C in the spec and q-bar in the scoping document
- Where: spec/spec.tex:1080-1083 (fragments_B.tex:168-171) against scoping.tex:2467 (definition of q-bar),
  Table tab:loop_per_model and Table tab:chance_rows (mu <= tau/q-bar).
- Problem: one quantity, E[Z | Z > q], has two names. In the spec, C also collides with the cost constants
  C_D, C_M, C_R, C_rep. The scoping name is the better one.
- Evidence: by inspection; the spec notation table (fragments_A.tex:127-129) writes E[Z | Z > q] in full.
- Fix: file `merge/spec_drafts/fragments_B.tex`
  - old: `$C = \mathbb{E}[Z \mid Z > q]$. On every day $k = 1, \ldots, T$:`
  - new: `$\bar q = \mathbb{E}[Z \mid Z > q]$. On every day $k = 1, \ldots, T$:`
  - old: `  \mu_k \leq \bar\mu, \qquad \bar\mu = \frac{\tau}{C}.`
  - new: `  \mu_k \leq \bar\mu, \qquad \bar\mu = \frac{\tau}{\bar q}.`
- Confidence: high

### F12 [meaning] "Member" means a fleet member and also a law of the class G(m)
- Where: scoping.tex:2554, 2556, 2560, 4707, 4710, 4717 (class G(m)) against scoping.tex:47-160 and the notation
  table (member = fleet member). In the same subsection, scoping.tex:2820 and 2827 say "the member serves a mission".
- Problem: one term for two concepts, inside one subsection.
- Evidence: by inspection (`grep -n member scoping.tex`).
- Fix: file `scoping.tex`, use "element" for the class:
  - old: `The one-term member $c\,\mathrm{Ga}(m/c, 1)$` new: `The one-term element $c\,\mathrm{Ga}(m/c, 1)$`
  - old: `A member of $\mathcal{G}(m)$ is described` new: `An element of $\mathcal{G}(m)$ is described`
  - old: `Let $S$ be a member with at least one term` new: `Let the law of~$S$ be an element of $\mathcal{G}(m)$ with at least one term`
  - old: `be the members with at most $N$ atoms` new: `be the elements with at most $N$ atoms`
  - old: `only the constant member $S = m$` new: `only the constant element $S = m$`
  - old: `Every member has finitely many atoms` new: `Every element has finitely many atoms`
- Confidence: medium (the meaning is clear from context; the double use breaks the rule of one term per concept)

## Checked and correct (no finding)

- Orderings: the Gamma, Gaussian and inverse Gaussian rows of the three tables and their derivations; the closure
  reduction mu_1 <= mu_2 of the inverse Gaussian; the icx example (X = 1, Y in {0, 2}).
- Inverse Gaussian: Proposition prop:ig_order and its proof; Theorem thm:ig_cap and its proof; the tail-mean identity
  z f_Z(z) = z^-2 f_Z(1/z); the density and the squared coefficient of variation eta; the quantile-cap example
  (tau/q = 0.4730 tau, three missions give 0.1180, Monte Carlo 0.1179, CVaR cap 0.1411 tau); the ratio 0.928 to 0.588
  (`r2\ig_check.py`); the spec formula of F and the `invgauss` parameters.
- Gamma: Lemma lem:gamma_class, Theorem thm:gamma_oneterm (steps 1-5), Lemma lem:gamma_monotone, Proposition
  prop:gamma_safe, drv:gamma_beta; A_hat for 15 pairs (t, epsilon); t1 = 0.43614, 0.44839, 0.61422 (`r2\gamma_check.py`);
  the witness rate 4.564 beta* with tail 1.56487 epsilon (`r2\gamma_check2.py`); the parameter-space repair example,
  tail 0.01329 = 1.329 epsilon by quadrature, 0.013305 +- 0.000026 by Monte Carlo (`r2\gamma_repair_example.py`).
- Gaussian and Wiener: Proposition prop:gauss_exact and its variance identity; the validity derivation;
  V_max/tau^2 = 0.1848, 0.6089, 3.636, 1591; Phi(-3) = 0.135 % (`r2\gauss_check.py`); drv:wiener_path; the loop
  argument for the Wiener variance.
- Spec: every big-M row of the four models against fragments_A (general form, natural ranges, anchor rows, Wiener
  variance row); the big-M constants are valid (on an intervention day delta_mu = delta_v = 0, and mu <= tau,
  v <= V_max follow from the rows); the day maps against the scoping; beta* and the Gamma computation steps against
  the scoping. All cited labels exist.

## Doubts

- **"For t >= t1, A_1 = A_hat" rests on a grid.** scoping.tex:2619 states it for every t >= t1; the evidence is a
  grid of t in [t1, 1000]. I extended the check to [t1, 10^5] on 200 values of t and c in [10^-8, 1): the largest
  ratio max_{c<1} Q(A_hat/c, t/c)/epsilon is 0.99999976 (`r2\gamma_above_t1.py`). So no counterexample; the claim is
  numerical, not proven. It does not affect safety: spec step 3 tests w(A_hat) <= epsilon before it uses A_hat.
- **The returned cap satisfies w(A_1) <= epsilon only for the computed w.** scoping.tex:2612 and 4823 say that
  w(A_1) <= epsilon holds at the returned value. The bisection keeps the computed w at the lower end below
  epsilon; a grid search that misses the peak would underestimate w. With the single peak (finding F3) and the
  golden-section refinement the error is negligible, so I do not report it as an error.
