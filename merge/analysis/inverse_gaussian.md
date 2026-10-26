# Inverse Gaussian degradation: is the formulation safe?

Status: done: S1, S2, S3, S4. In progress: none. Next: none (open item: Conjecture S1, unproven).

Scripts and outputs: `C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\ig\`
(`iglib.py` is the shared library; each `sN_*.py` writes `sN_out.txt`). Standard library only.
`s0_selftest.py` checks the closed-form IG tail against quadrature (agreement to 1e-12) and the
sampler against exact moments.

## Notation

- IG(μ, λ): inverse Gaussian law. Density √(λ/(2πx³))·exp(−λ(x−μ)²/(2μ²x)), x > 0.
  Mean μ, variance μ³/λ. Laplace transform E e^{−sX} = exp((λ/μ)(1 − √(1 + 2μ²s/λ))).
- Scaling: c·IG(μ, λ) = IG(cμ, cλ) for c > 0.
- τ: failure threshold. ε: target. Tail of a law: P(D > τ). All numbers below use τ = 1.
- η = μ/λ (lore input; η = CV², CV = coefficient of variation). P_η(μ) := IG(μ, μ/η) = μ·Z,
  with Z ~ IG(1, 1/η). This is the law that lore assumes for the state.
- ξ = λ/μ². Q_ξ(μ) := IG(μ, ξμ²). This family is closed under sums (S2).
- μ̄: lore's cap, P(P_η(μ̄) > τ) = ε (`lore:src/fleet_management/models/inverse_gaussian.py:53-75`).
- Term structure. Between replacements, ARD∞ (whole-state repair) unrolls the state as
  D_k = Σ_t w_t·X_t. X_t is the reset draw or a mission increment. w_t ∈ (0, 1] is a product
  of repair factors (1−ρ). If X_t ~ P_η(m_t), then w_t·X_t ~ P_η(a_t) with a_t = w_t·m_t.
  The exact recursion gives the tracked mean μ_k = Σ_t a_t; the one-sided rows give μ_k ≥ Σ_t a_t.
- A1 (independence): all X_t are independent. This is the rainflow assumption A1
  (`lore:spec/spec.tex:1358-1363`); lore's IG section states no such assumption.
- Orders. X ≤_st Y (first-order stochastic dominance, FSD): P(X > t) ≤ P(Y > t) for all t.
  X ≤_cx Y (convex order): E g(X) ≤ E g(Y) for every convex g; it forces E X = E Y.
  ≤_icx: the same for increasing convex g. Stop-loss of X at d: E(X − d)_+.

## S1. Accumulation of increments

Script `s1_accumulation.py`, plus `s1b_threshold.py` and `s1c_eta_threshold.py`.

**S1.1 (lead confirmed).** Let X_i ~ P_η(a_i) be independent, i = 1..n, n ≥ 2. Then Σ X_i is
an IG law if and only if all a_i are equal. Even then Σ X_i ~ IG(na, n²a/η) ≠ P_η(na).
- Proof. The cumulants of IG(μ, λ) are κ1 = μ, κ2 = μ³/λ, κ3 = 3μ⁵/λ². So κ1κ3/κ2² = 3 for
  every IG law. For the sum, κ1 = Σa_i, κ2 = ηΣa_i², κ3 = 3η²Σa_i³. So
  κ1κ3/κ2² = 3(Σa_i)(Σa_i³)/(Σa_i²)² ≥ 3 (Cauchy–Schwarz), with equality iff all a_i are
  equal. For equal a_i, each term has ξ = 1/(ηa), and the Laplace transform gives
  IG(na, ξ(na)²) = IG(na, n²a/η). Its variance ηna² is n times smaller than ηn²a² of P_η(na).
- Check. Convolution of two IG(1, 4) densities (η = 0.25) vs IG(2, 16): max error 7.8e-16;
  vs the model IG(2, 8): 0.218. Means (1, 2): κ1κ3/κ2² = 3.24.

**S1.2 (the model is the comonotone sum).** P_η(Σa_i) is the law of Σ a_i·Z with one common
Z ~ IG(1, 1/η). For any joint law of terms X_i ~ P_η(a_i) (independent or not):
Σ X_i ≤_cx P_η(Σ a_i). With tracked μ_k ≥ Σ a_i: D_k ≤_icx P_η(μ_k).
- Proof. Fix d and put d_i = a_i·d/Σa_j. Pointwise, (Σ X_i − d)_+ ≤ Σ (X_i − d_i)_+. Also
  (Σa_i·Z − d)_+ = Σ (a_i·Z − d_i)_+. Take expectations; a_i·Z and X_i have the same law. So
  the stop-loss of Σ X_i is at most that of P_η(Σa_i) at every d. With equal means this is ≤_cx.
  P_η(μ) = μ·Z increases in μ for ≤_st, which gives ≤_icx for the tracked μ_k.

**S1.3 (the tail claim is refuted).** Convex order does not order tails. Same mean forces the
survival functions to cross. Above the last crossing the model tail is larger (safe); just below it
the model tail is smaller (unsafe).
- Realistic cases (A1; total mean pushed to μ̄, so the model tail equals ε). True tail / ε:

  | η | ε | μ̄ | D0 0.2 + 10×0.05 + 5×0.06 | 0.55 + 3×0.15 | 2 equal halves | 0.9 + 10×0.01 |
  |---|---|---|---|---|---|---|
  | 0.05 | 0.01 | 0.614 | 1.8e-9 | 0.023 | 0.068 | 0.59 |
  | 0.25 | 0.01 | 0.375 | 9.2e-7 | 0.054 | 0.099 | 0.66 |
  | 1 | 0.01 | 0.201 | 6.9e-5 | 0.11 | 0.17 | 0.73 |
  | 0.25 | 0.001 | 0.271 | 1.9e-10 | 0.009 | 0.015 | 0.51 |
  | 1 | 0.1 | 0.467 | 0.043 | 0.50 | 0.69 | 0.89 |

  Term means are fractions of μ̄. Quadrature error < 1e-10·ε. Monte Carlo (MC), 10⁶ draws,
  each mission sampled: 0.00429 ± 0.00007 vs 0.00429; 0.0892 ± 0.0003 vs 0.0889.
  The model is safe here, and very conservative when many small missions accumulate.
- Counterexamples (A1, n equal independent increments, total mean μ̄):
  - η = 10, ε = 0.1: μ̄ = 0.4730; three increments IG(0.1577, 0.01577). True tail 0.1180
    (MC 0.1179 ± 0.0003) > 0.1.
  - η = 100, ε = 0.01: μ̄ = 0.0519; two increments IG(0.0260, 0.00026). True tail 0.01072
    (MC 0.01064 ± 0.0001) > 0.01.
- Threshold. Over n ∈ {2, …, 10⁶} equal terms, lore's cap fails for ε above ε*(η):

  | η (CV) | 0.01 (0.1) | 0.25 (0.5) | 1 (1) | 2 (1.4) | 4 (2) | 10 (3.2) | 30 (5.5) | 100 (10) |
  |---|---|---|---|---|---|---|---|---|
  | ε* | 0.47 | 0.34 | 0.23 | 0.17 | 0.11 | 0.059 | 0.023 | 0.0074 |

  Equivalently: failures need η > 73 (CV > 8.5) at ε = 0.01, and η > 4.9 (CV > 2.2) at
  ε = 0.1. Two-group structures (one big term plus n small) gave the same picture (η = 10,
  ε = 0.1: worst 1.184). Crossing points at mean 1: η = 0.25, n = 2: t ≈ 1.08, model tail
  0.35 there; η = 10, n = 2: t ≈ 3.76, model tail 0.059 there.
- **Conjecture S1** (numerical only): for ε ≤ ε*(η) of the table, lore's cap is safe for every
  term structure. Only equal-term and two-group structures were scanned.
- Verdict: lore's formulation is not provably safe. It is unsafe for very dispersed increments
  with large ε. lore accepts every η > 0 (`lore:src/fleet_management/models/inverse_gaussian.py:30-31`)
  and ε ∈ (0, 0.01] (`lore:src/fleet_management/solver.py:135`). So in lore a failure needs
  η > 73; with ε = 0.1, as in tests, it needs only η > 4.9.
- Branch main has only a normal-quantile cap α − √(α/c)·Φ⁻¹(1−ε)
  (`main:src/fleet_management/degradation_model/inverse_gaussian.py:138-144`). It carries no
  tail guarantee for an IG law either.

## S2. The family closed under sums (fixed xi) and ARD repair

Scripts `s2_fixed_xi.py`, `s2b_fixed_xi_scan.py`, `s2c_counterexample.py`.
Here each mission increment is X_t ~ Q_ξ(m_t), with one ξ per component. The increment's
CV² is 1/(ξ·m_t): it depends on the mission size. Example: m = 0.02 with CV 0.5 gives ξ = 200.

**S2.1 (no repair: exact).** Q_ξ(a) + Q_ξ(b) = Q_ξ(a + b) for independent terms. So under A1,
without repair, D_k ~ Q_ξ(μ_k) exactly. P(Q_ξ(μ) > τ) increases in μ (S3).
- Proof. The Laplace exponent of Q_ξ(μ) is ξμ(1 − √(1 + 2s/ξ)), linear in μ.
- Check. ξ = 200: tail of Q(0.3) + Q(0.5) by convolution equals that of Q(0.8) to 1e-10.

**S2.2 (repair leaves the family).** Pathwise ARD∞ gives c·Q_ξ(μ) = IG(cμ, cξμ²) = Q_{ξ/c}(cμ),
with c = 1 − ρ. A parameter-space repair would give Q_ξ(cμ) = IG(cμ, c²ξμ²) instead. Same mean.
Adding a fresh increment Q_ξ(m) to the pathwise law gives a sum with two values of ξ, not IG.

**S2.3 (convex order holds and survives increments).** Lemma (mine): at a fixed mean μ,
IG(μ, λ1) ≤_cx IG(μ, λ2) whenever λ1 ≥ λ2. Hence c·Q_ξ(μ) ≤_cx Q_ξ(cμ), since cξμ² ≥ c²ξμ².
- Proof of the lemma. Scale to μ = 1. Take λ1/λ2 = q/p rational, q > p. Let Y_1..Y_q be iid
  IG(1, λ2/p). By S2.1 and scaling, the mean of the first r of them is IG(1, rλ2/p). So the
  mean of p is IG(1, λ2) and the mean of q is IG(1, λ1). By exchangeability,
  E[mean of p | mean of q] = mean of q. Jensen's inequality gives the convex order. For a real
  ratio, the stop-loss E(IG(1, λ) − d)_+ is continuous in λ.
- Survival. ≤_cx is kept under scaling and under adding an independent term. By induction,
  with parameter-space repair in the tracked mean: D_k ≤_icx Q_ξ(μ_k) at every day.
- Check. ξ = 200, μ = 0.8, c = 0.5. Stop-loss at d = 0.4: pathwise 0.0126 ≤ parameter-space 0.0178.

**S2.4 (tails: no order in general).** FSD between the two laws is impossible: X ≤_st Y with
E X = E Y forces equal laws. Their survival functions cross near the mean (ξ = 200, μ = 0.8,
c = 0.5: crossing at t ≈ 0.40, parameter-space tail 0.48 there).
- Scan: repair at the cap μ̄_ξ (one or two repairs, c ∈ {0.05, 0.2, 0.5, 0.8, 0.9}), then fresh
  increments back to μ̄_ξ. True tail / ε ≤ 0.98 for ξ ∈ {2, 20, 200, 2000} and ε ≤ 0.2.
- Counterexample (very dispersed increments): ξ = 0.2, ε = 0.1, μ̄_ξ = 0.4788. Repair with
  c = 0.5, then increments back to the cap. True tail 0.1068 (MC, each mission sampled,
  0.1063 ± 0.0005) > 0.1. At ε = 0.2 the worst ratio is 1.13; at ε = 0.01 it is 0.995.
- After further increments (ξ = 200, c = 0.2, ε = 0.2), the true tail at the parameter-space
  (1−ε)-quantile stayed below ε at every step (0.04 to 0.18). This is numerical only.

**Verdict S2.** With fixed ξ, accumulation is exact and repair is the only error. Parameter-space
repair (tracked mean, linear dynamics) gives a convex-order upper bound that survives later
increments (proved). A tail bound does not follow; it fails for very dispersed increments.
Fixed ξ is a different input semantics from lore's η: it needs ξ (or one reference CV at one
reference mission size) per component.

## S3. Monotonicity and the loop Proposition

Script `s3_monotone_loop.py`. Spec text: `lore:spec/spec.tex:874-925`.

**S3.1 (monotonicity holds in both families).** P(P_η(μ) > τ) and P(Q_ξ(μ) > τ) are strictly
increasing in μ. So both caps are well defined and the reliability row is increasing in μ.
- Proof. P_η(μ) = μ·Z: a scale family, and Z has a positive density on (0, ∞). Q_ξ(μ2) is
  Q_ξ(μ1) plus an independent positive term Q_ξ(μ2 − μ1) (S2.1). No MLR argument is needed.
- Check. Grid μ ∈ {0.01, …, 0.99}, η ∈ {0.05, 1, 10}, ξ ∈ {0.2, 20, 2000}: increasing.

**S3.2 (the spec's MLR proof: one typo).** The density at `lore:spec/spec.tex:906-907` has
exponent −(x−μ)²/(2ηx). The IG(μ, μ/η) density has −(x−μ)²/(2ημx). The printed function
integrates to μ^{−1/2}, not 1 (check: 2.000 at μ = 0.25, 0.500 at μ = 4). The derivative
formula at lines 910-914 is right for the correct density (numerical match to 1e-6).
The Proposition itself (P_η(μ1) ≤_st P_η(μ2) for μ1 ≤ μ2) is true.

**S3.3 (the loop argument, for the true law).**
- The Proposition compares two laws of the model family. The true laws at days H and 2H are
  sums with different term structures (S1.1). So μ_2H ≤ μ_H does not give D_2H ≤_st D_H.
- Counterexample (A1, η = 0.25). D_H: reset draw P_η(0.1) plus 10 missions P_η(0.07), mean 0.8.
  Second half: one repair with ρ = 0.5, then one mission P_η(0.35); D_2H has mean 0.75.
  P(D > 1.2): D_H 0.0029, D_2H 0.0259 (MC 0.0257 ± 0.0002). The tail of the later state is
  nine times larger although its mean is smaller.
- The repair sentence at `lore:spec/spec.tex:918-921` holds for a one-term state only.
- What the loop needs (my argument). Each day k of each repetition r of the second half has
  its own term structure. S1.2 gives D_k^(r) ≤_icx P_η(μ_k^(r)) at every such day. The loop
  row and the increasing day maps (merge decision (i)) give μ_k^(r) ≤ μ_k^(1) ≤ μ̄. The same holds
  with Q_ξ under S2.3.
- Verdict. The mean-only loop row is the right condition. It adds no new failure mode. Safety
  of every repetition reduces to one question per day: does D ≤_icx P_η(μ̄) imply
  P(D > τ) ≤ ε? S1.3 shows that it does not for lore's μ̄. S4.1 gives a cap for which it does.

## S4. A safe formulation

Scripts `s4_safe_caps.py`, `s4b_extra.py`. All proposals in this section are mine.

**S4.1 (proposal): the CVaR cap. Mean only, drop-in.** Replace μ̄ by μ̂ = τ / CVaR(Z), with
Z ~ IG(1, 1/η), q its (1−ε)-quantile, and CVaR(Z) = E[Z | Z > q] = q + E(Z − q)_+/ε
(CVaR: conditional value at risk). lore's cap is μ̄ = τ/q.
- Theorem. If D ≤_icx P_η(μ) and μ ≤ μ̂, then P(D > τ) ≤ ε. No independence is needed.
- Proof. For d < τ: 1{x > τ} ≤ (x − d)_+/(τ − d), and (x − d)_+ is increasing and convex. So
  P(D > τ) ≤ E(μZ − d)_+/(τ − d). At d = μq this equals εμ(CVaR − q)/(τ − μq), which is ≤ ε
  if and only if μ·CVaR ≤ τ.
- Sharp for the class. For μ > μ̂ there is ε' > ε with μ·E[Z | Z > q_{1−ε'}] > τ. The two-point
  law D = E[μZ | 1{Z > q_{1−ε'}}] satisfies D ≤_cx P_η(μ) (Jensen) and P(D > τ) = ε' > ε. So
  S1.2 alone allows no larger cap. The extra cost comes from the class being larger than the
  sums that a schedule can produce.
- With S3.3 this certifies every day of every repetition. The same works for Q_ξ (S2.3).
- Code change: `solve_mu_bar` only (`lore:src/fleet_management/models/inverse_gaussian.py:53-75`)
  and Eq. (ig_mubar) of the spec.
- Cost μ̂/μ̄ (τ = 1):

  | η | 0.05 | 0.25 | 1 | 4 | 10 | 100 |
  |---|---|---|---|---|---|---|
  | ε = 0.001 | 0.94 | 0.89 | 0.84 | 0.79 | 0.74 | 0.58 |
  | ε = 0.01 | 0.93 | 0.86 | 0.78 | 0.67 | 0.59 | 0.29 |
  | ε = 0.1 | 0.90 | 0.79 | 0.64 | 0.44 | 0.30 | 0.11 |

- Checks. At d = μ̂q the stop-loss bound equals ε (0.10000, 0.01000, 0.20000 in three cases).
  On the S1 counterexamples the worst true tail at μ̂ is 0.029 (η = 10, ε = 0.1) and 0.0029
  (η = 100, ε = 0.01).

**S4.2 (proposal): IG as a rainflow component (sub-gamma Bernstein, tracked variance).**
- Lemma. For X ~ P_η(a): log E e^{s(X−a)} ≤ ηa²s²/(2(1 − 2ηas)) for 0 ≤ s < 1/(2ηa).
- Proof. Put u = 2ηas. Then log E e^{sX} = (1/η)(1 − √(1 − u)) = (1/η)Σ_{k≥1} c_k u^k, with
  c_1 = 1/2 and c_{k+1} = c_k(2k − 1)/(2k + 2). The right side equals (1/η)Σ_{k≥2} u^k/8.
  Now c_2 = 1/8 and c_k decreases, so the centred series is dominated term by term.
- Under A1 the state is then sub-gamma with v = ηΣa_t² and scale 2η·a_max. Here a_max is the
  largest of μ_0, μ_new and all μ_inc: repair weights are ≤ 1. Chernoff at s = t/(v + 2η·a_max·t)
  gives P(D − μ > t) ≤ exp(−t²/(2(v + 2η·a_max·t))).
- This is exactly the rainflow Bernstein row (`lore:spec/spec.tex:1563-1575`) with derived
  inputs b = 6η·a_max, v_inc = η·μ_inc², v_0 = η·μ_0², v_new = η·μ_new². Rainflow's
  bounded-support assumption A2 is not needed: the lemma gives the same moment bound. The row
  increases in μ and v (decision (ii)); v follows ARD with factor (1−ρ)². The exact row is
  nonconvex quadratic; the rainflow LP tangent keeps a MILP.
- Check. ψ(s) − bound ≤ −3e-11 on a grid of s (η = 0.25, a = 1; η = 10, a = 0.3).

**S4.3 (variant): mean-only Bernstein.** v = ηΣa_t² ≤ η·a_max·μ gives one constant cap. Needs A1.
**S4.4 (variant): Cantelli with tracked v** (rainflow Cantelli row). Needs uncorrelated terms.
Since each certificate is valid, max(CVaR cap, mean-only Bernstein cap) is also a valid cap.

Caps on examples (τ = 1; reset draw μ_0 = 0.05 plus equal missions m; "exact" = largest mean
whose true tail is ≤ ε for this structure; "none" = no feasible mean):

| case | lore μ̄ | CVaR S4.1 | Bernstein v S4.2 | Bernstein mean S4.3 | Cantelli v S4.4 | exact |
|---|---|---|---|---|---|---|
| η = 0.25, ε = 0.01, m = 0.02 | 0.375 | 0.322 | 0.667 | 0.598 | 0.477 | 0.830 |
| η = 0.25, ε = 0.01, m = 0.2 | 0.375 | 0.322 | 0.065 | 0.053 | 0.175 | 0.530 |
| η = 1, ε = 0.01, m = 0.02 | 0.201 | 0.156 | 0.054 | 0.053 | 0.227 | 0.663 |
| η = 10, ε = 0.1, m = 0.05 | 0.473 | 0.141 | none | none | 0.158 | 0.460 |

- Last row: lore's cap exceeds the exact cap. With 8 missions of 0.0529 the true tail is 0.1059
  (MC 0.1058 ± 0.0003) > 0.1.
- Bernstein pays a margin 4κη·a_max at v = 0 (κ = ln(1/ε)); it wins only when η·a_max is small.

**S4.5 (alternatives not proposed).**
- Keep μ̄ and restrict inputs to ε ≤ ε*(η) of S1.3: rests on Conjecture S1; no proof.
- Fixed ξ (S2): the same tail gap after repairs; it needs the CVaR cap anyway.
- Chernoff with the IG moment generating function of P_η(μ): never better than S4.1, because
  e^{s(x−τ)} ≥ (1 + s(x − τ))_+, a stop-loss function with d = τ − 1/s.

**Recommendation (mine).** Port IG with the CVaR cap S4.1. It changes one function, needs no
independence, and makes the loop argument rigorous. At ε = 0.01 and η ≤ 1 it costs 7–23 % of
lore's cap. Offer S4.2 as an option for components with many small missions.
