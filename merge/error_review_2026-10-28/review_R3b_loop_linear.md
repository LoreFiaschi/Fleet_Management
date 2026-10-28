Status: done (6 findings, 1 doubt)

# Review R3b: loop constraint under Route-I bounds, refinements, linear forms, random fatigue life

Scope read in full: scoping.tex 1615-2183 and 4357-4524; spec/spec.tex 851-911 (identical to
merge/spec_drafts/fragments_A.tex 410-470); spec rainflow section 1304-1541 and consistency checks
525-583 for consistency; cited places in scoping (eq:loop, sec:encoding, thm:perpetual, sec:cons_surrogates,
tab:five_bounds, prop:rf_maps, sec:repair_connection, tab:sn_scatter, eq:abstract).

Check scripts (all in the scratchpad):
- `r3b_check_loop.py`: loop instance (Cantelli, Bernstein), ARD_1 anchor counterexample, fixed-margin search.
- `r3b_check_refine.py`: Bernstein support example, big-M constant, pwl table, 94 % area, tangent geometry.
- `r3b_check_linear.py`: Cantelli example of the linear forms, pwl = max of tangents, Bernstein tangent-point conditions.
- `r3b_check_miner.py`, `r3b_check_miner_conv.py`: inverse moments, quantiles, experiment of Remark miner_dichotomy.

Reproduced without error: U^(0..2) and the limit of the Cantelli instance (0.2000, 0.1997, 0.2151, 0.2495);
the Bernstein instance (U^(2) = 0.3247 > U^(0) = 0.3012); E[D_T] = 5; kappa = 4.605, c = 0.4605 and 0.0307,
budgets 0.079 and 0.939, n = 7 and 90, ratio 12.9, 1.075; derivative of the certified value in c in [1, 2];
the pwl table (all 15 entries) and the 94 % (exact 93.75 %); the tangent intersection sqrt(v_r v_{r+1}) and the
half-plane equivalence (0 mismatches in 20000 random points); Proposition tangent_safe; the Cantelli example
(0.810, 0.482, 0.714, 0.810); the pwl loss bound c_2 tau^2/(4P^2) (attained); the Bernstein tangent-point
and pwl conditions (0 mismatches in 4000 random cases); the inverse moments and quantiles of the three families;
P_fail(II) = 0.153; the reversal at high damage. Spec linear forms agree with scoping and with the spec's
rainflow section (constants c_2, c_1, side rows, f(0) = natural ranges, defaults 0.5 and 8, checks of line 580).

## Findings

### F1 [error] The derivation of the moment loop omits the anchor rows that it needs under ARD_1
- Where: scoping.tex:4393 (derivation drv:loopI_moments); compare scoping.tex:1674 and eq:loop (line 351).
- Problem: the derivation says that eq:loop_moments (the two moment rows) makes Π non-increasing, and concludes
  that the moments chain. Under ARD_1 the repair map uses the anchors, so the moment rows alone do not chain.
  The main text (line 1674) and eq:loop include the anchor rows; the derivation does not use them.
- Evidence (`r3b_check_loop.py`): ARD_1, rho = 0.5, operating phase = one repair day, then one mission day
  adding 1. Day H_1: mean 2, anchor 0. Repetitions: (E, g) = (2, 0), (2, 1), (2.5, 1.5), (3, 2), ...
  The mean row E_T = 2 <= E_H1 = 2 holds, the anchor row g_T = 1 <= 0 fails, and the mean then rises without
  bound. So "Π does not increase either moment" does not imply the induction step.
- Fix: file `scoping.tex`
  - old: `Then~\eqref{eq:loop_moments} states that $\Pi$ does not increase either moment at the state of day~$H_1$.`
  - new: `Then~\eqref{eq:loop_moments}, with the rows on the anchors under $\text{ARD}_1$, states that $\Pi$ does not increase either moment, nor either anchor, at the state of day~$H_1$.`
- Confidence: high

### F2 [error] The Miner rewriting drops the cycle count n_j and needs "no repair"
- Where: scoping.tex:2120-2123 (interpretation (II), item "Reliability constraint").
- Problem: a mission day adds sum_j x_ijk n_j / N̄_j to M (line 2101), with n_j the cycles of one execution
  and n_ij the number of days (line 1160). So, without repair, M_k = sum_j n_ij n_j / N̄_j, and M_k <= q_eps
  reads sum_j n_ij n_j/(q_eps N̄_j) <= 1. The printed form has no n_j. Also, after a repair M_k is no longer
  this count sum (M ↦ (1-rho)M), so the rewriting holds only without repair since the last replacement.
- Evidence: direct substitution of the dynamics of line 2099-2103.
- Fix: file `scoping.tex`
  - old: `\sum_j n_{ij}/(q_\varepsilon \bar N_j) \;\le\; 1`
  - new: `\sum_j n_{ij}\,n_j/(q_\varepsilon \bar N_j) \;\le\; 1`
  - old: `Rewriting $M_k \le q_\varepsilon$ as`
  - new: `Without repair since the last replacement, and with $n_{ij}$ counted from it, rewriting $M_k \le q_\varepsilon$ as`
- Confidence: high

### F3 [error] The big-M constant 2τ(τ − c_0) is valid only when 2c_j ≤ τ
- Where: scoping.tex:1786 (refinement "Schedule-dependent Bernstein support", item "Cost"); side effect 2 at 1792-1794.
- Problem: write the conditional tangent row as 2κv <= T_j(μ) (the only scaling with the units of τ²).
  When ζ_ij = 0 its largest violation over the region of the unconditional c_0 row is
  2(c_j − c_0)(τ − μ) + (μ − μ_r)², which is at most 2τ(τ − c_0) if and only if, for every μ_r in [0, τ),
  c_j <= τ/2. Side effect 2 keeps missions with 2c_j >= τ (they "force ζ_ij = 0"). For such a mission the
  big-M row with this constant cuts feasible points at ζ_ij = 0, so the model becomes infeasible instead of
  forcing ζ_ij = 0.
- Evidence (`r3b_check_refine.py`): τ = 1, c_0 = 0, c_j = 2, μ_r = 0: at the empty schedule μ = v = 0 the
  row reads 0 <= τ(τ − 2c_j) + M = −3 + 2 = −1, violated. The grid search gives "OK" for every c_j <= 0.5 and
  "FAILS" for c_j = 0.6 with μ_r = 0.99, c_j = 1 with μ_r = 0.5, c_j = 1.4 with μ_r = 0.
- Fix: file `scoping.tex`
  - old: `(a uniform valid constant is $2\tau(\tau - c_0)$)`
  - new: `(with the row scaled as $2\kappa v \le \dots$, a uniform valid constant is $2\tau(\tau - c_0)$ for every mission with $2c_j < \tau$; a mission with $2c_j \ge \tau$ gets $\zeta_{ij} = 0$ and no row)`
- Confidence: medium (the text does not state the scaling of the row; the constant has the units of τ², which
  fixes the scaling 2κv up to the choice made here)

### F4 [error] The derivation that U^(1) <= U^(0) does not chain ends with a step that does not follow
- Where: scoping.tex:4366-4367 (derivation drv:loopI_no_chain).
- Problem: "maps these two different parameter points to different bound values. So (loopI_U_rise) is
  possible." Different values do not give a larger value, and "different" is itself only asserted. The
  possibility of U^(2) > U^(1) is shown by the numerical instance of the main text (lines 1635-1646), which
  the derivation does not cite.
- Evidence (`r3b_check_loop.py`): the instance gives U^(1) = 0.1997 < U^(0) = 0.2000 < U^(2) = 0.2151.
  With sum n v = 4.00 instead of 3.99 it gives U^(1) = U^(0) = 0.2000 and U^(2) = 0.2155, which is exactly
  the relation (loopI_U_rise).
- Fix: file `scoping.tex`
  - old: `One more repetition of the dynamics maps these two different parameter points to \emph{different} bound values.`
  - new: `One more repetition of the dynamics can map these two parameter points to different bound values, the larger one at day $T + H_2$: the Cantelli instance of Section~\ref{sec:loop_routeI} with $\sum_j n_{ij} v_{\ell j} = 4$ gives $U^{(1)} = U^{(0)} = 0.2000$ and $U^{(2)} = 0.2155$.`
- Confidence: high

### F5 [inconsistency] The caption of Table sn_scatter lets the shape depend on the stress; the block fixes it
- Where: scoping.tex:1038 (caption of tab:sn_scatter, cited at 1985, 2011, 2039) against scoping.tex:1993-1994
  ("a σ that does not depend on the stress", "a common shape k") and 2090 (interpretation (II) needs a
  scatter parameter that does not depend on the stress); also line 1094-1096 (Weibull: shape k, stress-dependent scale λ).
- Problem: the caption says "μ(σ_a) and the scale/shape parameters depend on the stress level through
  Basquin's law". Basquin's law ties only the location μ (log-normal, Gumbel) or the scale λ (Weibull) to the
  stress. If σ, k or β depended on the stress, the single unit factor A of eq:miner_unit would not exist.
  The block is right; the caption is wrong.
- Fix: file `scoping.tex`
  - old: `In all cases, $\mu(\sigma_a)$ and the scale/shape parameters depend on the stress level through Basquin's law~\eqref{eq:basquin}.}`
  - new: `In all cases, the location $\mu(\sigma_a)$ or the Weibull scale $\lambda(\sigma_a)$ depends on the stress level through Basquin's law~\eqref{eq:basquin}; the scatter parameters $\sigma$, $k$ and $\beta$ do not.}`
- Confidence: medium

### F6 [meaning] Two symbols of the block carry a second meaning: n_j and μ_0
- Where: scoping.tex:1637-1639 and 1816, 1886.
- Problem:
  - n_j at 1637-1639 is the number of days of mission j in the operating phase. In the same block n_j is the
    number of cycles of one execution of mission j (lines 2049, 2080, 2101), and the day count is n_ij
    (line 1160, derivation line 4376).
  - μ_0 at 1816 and 1886 is a tangent point. Elsewhere μ_0 is the initial mean (line 1596, 1605; spec
    "$\mu_{0,i\ell}$ initial values"); Section linear_forms calls the tangent point μ_r.
- Fix: file `scoping.tex`
  - old: `  \sum_j n_j\mu_{\ell j} = 3,` new: `  \sum_j n_{ij}\mu_{\ell j} = 3,`
  - old: `  \sum_j n_j v_{\ell j} = 3.99,` new: `  \sum_j n_{ij} v_{\ell j} = 3.99,`
  - old: `and the tangent at $\mu_0 = 0$ gives` new: `and the tangent at $\mu_r = 0$ gives`
  - old: `single tangent ($\mu_0 = 0$)` new: `single tangent ($\mu_r = 0$)`
- Confidence: high

## Doubts

- **Failure probability of interpretation (I), scoping.tex:2151.** With equal daily increments, σ = 0.5,
  fifty days and median Miner sum 0.6, P_fail(I) = 2.1e-7 (importance sampling, 2e6 samples, ±0.06e-7) and
  2.4e-7 to 2.5e-7 (FFT convolution). The text says "∼10^{-6}". P_fail(II) = 0.153 reproduces. Unequal daily
  increments raise P_fail(I) (Var_I grows for a fixed sum), so the stated value may come from another
  increment profile, which the text does not give. If the profile was uniform, "∼10^{-6}" should read
  "about 2·10^{-7}". Scripts: `r3b_check_miner.py`, `r3b_check_miner_conv.py`.
