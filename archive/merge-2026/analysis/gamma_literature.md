# Literature on extremal tails of weighted sums of independent gamma variables

Status: done (26 Oct 2026). Read in full: sections 2-5 and Janson (section 6). Not read:
Diaconis-Perlman 1990, Bock et al. 1987, Kochar-Xu 2011. Next: the user retrieves those PDFs.

## Our question

- Class C(b, m): laws of S = Σ_i c_i·Y_i, Y_i ~ Gamma(shape a_i, rate b) independent,
  c_i ∈ (0, 1], E S = m. The single Gamma(b·m, b) is one member.
- For which t is P(S > t) maximised over C(b, m) by the single Gamma(b·m, b)?

## 1. Diaconis and Perlman (1990)

- P. Diaconis, M. D. Perlman, "Bounds for tail probabilities of weighted sums of independent
  gamma random variables", IMS Lecture Notes - Monograph Series 16 (1990), 147-166,
  doi:10.1214/lnms/1215457557.
- **Not read.** Project Euclid answered the download with a bot block (Incapsula), and the
  browser page with a human-verification check ("I am human"), 26 Oct 2026. I did not pass it.
  No other full-text copy was found.
- It looks worthwhile: Yu (2017, pp. 1-2) reports that it states the "unique crossing
  conjecture" (UCC), proves it in four cases, and locates the crossing point when one weight
  vector has equal components. Can you retrieve the PDF?
- What it contains is reported below only as carried by the papers that cite it, with their
  location.

## 2. Székely and Bakirov (2003)

- G. J. Székely, N. K. Bakirov, "Extremal probabilities for Gaussian quadratic forms",
  Probab. Theory Relat. Fields 126 (2003), 184-202, doi:10.1007/s00440-003-0262-6.
  Full text: https://link.springer.com/content/pdf/10.1007/s00440-003-0262-6.pdf (ETH licence).
- 19 pages. **Read in full**; pp. 184, 186, 189, 190 also read as page images, because the
  text layer loses the symbols.
- **Setting.** Q_n = Σ λ_i Z_i², Z_i iid N(0,1), λ_i ≥ 0, E Q_n = Σ λ_i = 1. So Z_i² is
  Gamma(shape 1/2), iid, equal shapes only. They study the cdf P(Q_n ≤ x); the tail is
  1 minus it. x(d) is the crossing point of the cdfs of χ²_d/d and χ²_{d+1}/(d+1)
  (χ²_d: chi-square with d degrees of freedom); x(1) = 1.53640, x(2) = 1.29898, x(d) ↓ 1
  (Table 1 and Proposition 1, pp. 185, 188).
- **Theorem 1 (p. 189).** I_n(x) = inf over E Q_n = 1 of P(Q_n ≤ x) equals P(χ²_d/d ≤ x) for
  x ∈ [x(d), x(d-1)), d = 1, ..., n-1 (x(0) = ∞), and P(χ²_n/n ≤ x) for x < x(n-1).
  So the largest tail uses d equal weights 1/d. For x ≥ x(1) = 1.5364 it is one weight 1.
- **Corollary 3 (p. 189).** If x ≥ 2 and λ weakly majorizes μ (sorted partial sums of λ ≥
  those of μ; Σ μ_i = 1, p. 186), then P(Σ λ_i Z_i² ≤ x) ≤ P(Σ μ_i Z_i² ≤ x). So for
  x ≥ 2·mean the more concentrated weights give the larger tail P(Q > x). The bound 2 is sharp:
  the inequality for all n and all such pairs forces x ≥ 2 (or x ≤ 0).
- **Theorem 2 (p. 190)** gives the smallest tail (equal weights for x ≥ y(n); an interior
  two-weight vector for 1 < x < 2, Proposition 2). Theorems 3-4 (interval probabilities
  without a mean constraint; unimodality) do not identify a tail maximiser.
- **Conjectures.** None stated. Remark 3 (p. 190) leaves P(a ≤ Q_n ≤ b) under E Q_n = 1 open.
- **Our reading (interpretation).** Take all shapes a_i and b·m in (1/2)ℕ. Write each
  Gamma(a_i, b) as 2a_i iid Gamma(1/2, b) = Z²/(2b). Then S/m = Σ μ_j Z_j², Σ μ_j = 1, with
  μ_j = c_i/(2bm) ≤ 1/(2bm) = 1/n_0. The single Gamma(bm, b) is n_0 = 2bm equal weights 1/n_0.
  This vector majorizes every vector on the simplex with entries ≤ 1/n_0. Corollary 3 then
  gives: **for t ≥ 2m the single Gamma maximises P(S > t)** in this sub-class. For n_0 = 1
  (bm = 1/2), Theorem 1 gives the sharper range t ≥ 1.5364·m. Shapes that are not multiples
  of 1/2 are outside the theorem, and our counterexample (shape about 0.0025) is one of them.

## 3. Roosta-Khorasani, Székely and Ascher (2015)

- F. Roosta-Khorasani, G. J. Székely, U. M. Ascher, "Assessing stochastic algorithms for large
  scale nonlinear least squares problems using extremal probabilities of linear combinations
  of gamma random variables", SIAM/ASA J. Uncertainty Quantification 3 (2015), 61-90,
  doi:10.1137/14096311x. Full text read: arXiv:1404.0122v2, https://arxiv.org/pdf/1404.0122.
- 38 pages (arXiv). **Read in full** (pp. 6, 7, 29, 34, 35 also as page images). Sections 3-4
  and Appendix A (algorithms, inverse problems) do not touch our question.
- **Setting.** X_i iid Gamma(shape α, rate β), any α, β > 0; equal shapes only. Weights λ on
  the simplex Θ = {λ_i ≥ 0, Σ λ_i = 1}; mean of Σ λ_i X_i is α/β. Cdf P(Σ λ_i X_i < x).
- **Theorem 1 (p. 6; proof pp. 33-34).** For 0 < α_1 < α_2 and X_i ~ Gamma(α_i, α_i) (mean 1),
  the cdfs cross exactly once, at x(α_1, α_2), and
  1 ≤ x(α_1, α_2) ≤ (2√(α_1(α_2-α_1)) + 1) / (2√(α_1(α_2-α_1))).
- **Theorem 2 (p. 6; proof pp. 34-35).** min over Θ of P(Σ λ_i X_i < x) (largest tail) equals
  P(X_1 < x) for x > (2α+1)/(2β), and P((1/n)Σ X_i < x) for x < α/β. The max over Θ
  (smallest tail) has the mirror form. So for x > mean + 1/(2β) **one single term carries
  the largest tail over the simplex.**
- The proof (p. 35) shows more: for every x > 0 the extremes over Θ are among the equal-weight
  vectors (1/d, ..., 1/d, 0, ...), d ≤ n.
- No majorization ordering is proved. Lemmas 6-8 (pp. 28-29) give analyticity, unimodality and
  the mode of mixtures; they are tools.
- **Conjectures.** None stated.
- **Our reading (interpretation).** Theorem 2 needs the full simplex. In C(b, m) the cap
  c_i ≤ 1 is inactive only if the iid unit is the whole single Gamma: α = b·m, β = b. This
  covers the members whose shapes are integer multiples of b·m. For them, **the single
  Gamma(bm, b) maximises P(S > t) whenever b·t > b·m + 1/2.** Our counterexample is outside:
  its second member has shape A/2, not a multiple of A. For α < b·m the cap binds, and
  Theorem 2 says nothing about the capped simplex.

## 4. Yu (2017), on the Diaconis-Perlman conjecture

- Y. Yu, "On the unique crossing conjecture of Diaconis and Perlman on convolutions of gamma
  random variables", Ann. Appl. Probab. 27 (2017), 3893-3910, doi:10.1214/17-AAP1304.
  Full text read: arXiv:1607.02689v1, https://arxiv.org/pdf/1607.02689.
- 20 pages (arXiv). **Read in full** (pp. 1-3, 9 also as page images).
- **Setting.** X_i iid Gamma(α, 1); F_θ is the cdf of Σ θ_i X_i; θ a nonnegative weight vector
  (Theorems 1 and 4 assume positive vectors). No normalisation is needed; the conjecture
  compares vectors with η ≺ θ (θ majorizes η, so equal sums, so equal means).
- **Conjecture 1 = UCC (p. 1), carried from Diaconis-Perlman 1990.** If η ≺ θ and θ is not a
  permutation of η, then F_η - F_θ changes sign exactly once, from - to +. So beyond the
  crossing x_0 the less uniform θ has the larger tail.
- **Cases verified by Diaconis-Perlman, as carried by Yu (pp. 1-2):** n = 2; n = 3 with
  α = 1; n ≥ 3, α ≥ 1, vectors differ in two components; n ≥ 3 and η has equal components.
  Yu adds that Diaconis-Perlman analysed where the crossing lies when η has equal components.
- **Theorem 3 (p. 8).** The UCC holds for α ≥ 1. Proof via Theorem 4 (p. 9): α ≥ 1, positive
  η ≺_V θ, θ_(n) > η_(n), Π η_i/θ_i > 1 ⇒ exactly one crossing, F_η < F_θ before it.
- **Theorem 2 (p. 6).** For every α ∈ (0, 1) there are positive θ, η ∈ ℝ³ with η ≺ θ whose
  cdfs cross at least three times. So the UCC fails for α < 1.
- **Lemma 1 (p. 3, from Yu 2011).** log η ≺ log θ ⇒ F_η ≥ F_θ (usual stochastic order).
  **Proposition 1 (p. 3).** For n = 2: exactly one crossing from below iff θ_1θ_2 < η_1η_2 and
  max θ > max η.
- **Location of the crossing.** Yu proves no bound on x_0. He cites bounds from Bock et al.
  (1987) and Roosta-Khorasani-Székely (2015) only inside a proof (p. 13).
- **Our reading (interpretation).** For shapes and b·m that are multiples of one α ≥ 1, the
  single Gamma(bm, b) majorizes every member. So each member crosses it once, and the single
  Gamma has the larger tail beyond x_0(member). This answers our question only with a bound
  on x_0 that holds for all members; Yu gives none. For α < 1 the sign pattern itself can
  fail, consistent with our counterexample (iid unit A/2 ≈ 0.00125).

## 5. Roosta-Khorasani and Székely (2015), Schur properties (found by search; key paper)

- F. Roosta-Khorasani, G. J. Székely, "Schur properties of convolutions of gamma random
  variables", Metrika 78 (2015), 997-1014, doi:10.1007/s00184-015-0537-9.
  Full text read: arXiv:1601.04731v1, https://arxiv.org/pdf/1601.04731.
- 18 pages (arXiv). **Read in full** (pp. 2, 3 also as page images).
- **Setting.** X_i iid Gamma(α, β), any α, β > 0; equal shapes. P(λ) = P(Σ λ_i X_i < x).
  μ ≺ λ means: sorted decreasingly, partial sums of μ ≤ those of λ, equal totals s.
  **μ must be strictly positive; λ may have zeros** (p. 2, (2a)-(2d)). Any n.
- **Theorem 1 (p. 3; proof pp. 13-18).** If μ ≺ λ then
  P(μ) ≥ P(λ) for all x > (2α+1)s/(2β), that is, **the more concentrated λ has the larger
  tail** for x > mean·(1 + 1/(2α)), since the mean is αs/β.
  P(μ) ≤ P(λ) for x < αs/β (n = 2), or x < (α-1)s/β (n ≥ 3, α > 1).
- **Corollary 2 (p. 8).** The same for weak majorization (partial sums of μ ≤ those of λ for
  all k ≤ n), with s = Σ λ_i in the tail bound. **Corollary 3 (pp. 8-9).** The same for
  infinite ℓ¹ sequences (n = ∞).
- **Earlier results, as carried by this paper (p. 3).** Bock, Diaconis, Huffer, Perlman (1987,
  Theorem 3): n ≥ 3, μ ≺ λ, all λ_i > 0 ⇒ P(μ) ≥ P(λ) for x > (nα+1)·max λ_i/β. Bakirov
  (1995): α = β = 1/2 ⇒ the same for x > 2s. For α = β = 1/2, Theorem 1 gives Bakirov's
  bound (p. 7).
- **Conjectures.** The paper calls the UCC open (p. 9); Yu (2017) later settled it. It says a
  general lower-tail result for α ≤ 1, n ≥ 3 "seems likely" impossible (p. 8); this is the
  other tail, not ours.
- **Our reading (interpretation).** Let α > 0 divide every shape a_i and b·m (a_i/α and
  n_0 = bm/α are integers). Write S = Σ θ_j X_j, X_j iid Gamma(α, b), θ_j = c_i ∈ (0, 1],
  Σ θ_j = n_0 = s. The single Gamma(bm, b) is λ = (1, ..., 1, 0, ...) with n_0 ones. λ
  majorizes every such θ, and θ > 0 as the theorem needs. Theorem 1 then gives:
  **P(S > t) ≤ P(Gamma(bm, b) > t) for all t > m(1 + 1/(2α)), that is b·t > b·m + n_0/2.**
  The bound grows when α shrinks. Check on our counterexample: α = A/2, n_0 = 2, so the
  theorem needs b·t > A + 1 ≈ 1.0025; the counterexample sits at b·t = 0.01. No conflict.

## 6. Other work

- **S. Janson (2018), "Tail bounds for sums of geometric and exponential variables", Statist.
  Probab. Lett. 135, 1-6, doi:10.1016/j.spl.2017.11.017; read: arXiv:1709.08157v1,
  https://arxiv.org/pdf/1709.08157, 8 pages, read in full.** Sums of independent exponentials with rates a_i ≥ a_* = min a_i, mean μ.
  Theorem 5.1(i) (p. 7): P(X ≥ λμ) ≤ λ^{-1} exp(-a_*μ(λ - 1 - ln λ)), λ ≥ 1. The proof of
  Theorem 2.1 (pp. 2-3) bounds the moment generating function (MGF) of X by that of one
  Gamma(a_*μ, a_*), with inequality (1.5): -ln(1-x) ≤ (x/y)(-ln(1-y)), 0 < x ≤ y < 1.
  A bound, not an extremal result.
- **Bock, Diaconis, Huffer, Perlman (1987), Canad. J. Statist. 15, 387-395,
  doi:10.2307/3315257. Not read**: Wiley gave no access (download 403; the reader redirects
  to the abstract), 26 Oct 2026. Its Theorem 3 is known here only as carried by
  Roosta-Khorasani-Székely (2015, p. 3), see section 5.
- **Kochar and Xu (2011), J. Statist. Plann. Inference 141, 418-428. Not read** (Elsevier). Its
  listing (metadata only) names star and right-spread orders, not P(S > t) at a fixed t.
- Searches on 26 Oct 2026 found no paper after Yu (2017) that improves the threshold of
  section 5 or treats unequal shapes with a cap on the weights.

## Answer to our question

**Facts from the papers.**
- Every extremal result found is for weighted sums of iid gamma units of one shape α, with
  weights on the simplex or ordered by majorization. None treats unequal shapes with a cap.
- The strongest is Roosta-Khorasani-Székely (2015), Theorem 1: the more concentrated weights
  have the larger tail for x > mean·(1 + 1/(2α)). At α = 1/2 this is 2·mean, and Székely-Bakirov
  (2003, Corollary 3) show 2 is sharp over all majorization pairs.
- For α < 1 even the single-crossing pattern can fail (Yu 2017, Theorem 2).

**Interpretation (mine).**
- C(b, m) reduces to that setting only when all a_i and b·m are integer multiples of one α.
  Then the single Gamma(bm, b) majorizes every member (section 5), and:
  - **b·t > b·m + n_0/2, with n_0 = bm/α, suffices** (Roosta-Khorasani-Székely 2015, Thm 1);
  - t ≥ 2m suffices when α = 1/2 (Székely-Bakirov 2003, Cor. 3); t > m + 1/(2b) when n_0 = 1
    (Roosta-Khorasani-Székely-Ascher 2015, Thm 2).
- **No theorem settles the whole class.** The threshold grows without bound as α → 0. Members
  with tiny or incommensurable shapes, like our counterexample, get no guarantee. So the
  literature does not give a t-range for C(b, m), and it does not confirm "b·t ≥ 1".
- The thresholds above hold for all majorization pairs; ours is a special pair, so they are
  sufficient, not sharp. Example (computed with SciPy; the script lives only in the session
  scratchpad, gamma_lit\scripts\check_mapping.py):
  - α = 1, m = 3: Gamma(3,1) against Gamma(2,1) + 0.5·Gamma(2,1) cross at t = 3.10; the
    theorem needs t > 4.5.
  - Our pair (A = 0.0025): the tails cross at b·t = 0.3045; the theorem needs b·t > 1.0025.
    At b·t = 0.01 the tails are 0.01005 and 0.01336, as in our counterexample.
- **Heuristic, own first-order derivation, not a proof.** When the total shape b·m is small,
  P(S > t) ≈ Σ_i a_i·E_1(b·t/c_i), with E_1 the exponential integral. With Σ a_i c_i = b·m
  fixed, this is maximal at c = 1 iff u·E_1(u) is decreasing on [b·t, ∞). That holds iff
  **b·t ≥ u* = 0.4348**, the root of E_1(u) = e^{-u}. Below u*, the member
  c·Gamma(bm/c, b) with c = b·t/u* beats the single Gamma (by a factor near 7 at b·t = 0.01).
  This agrees with "no counterexample for b·t ≥ 1", but it is unproven for finite b·m.
- **A guarantee for the whole class (Janson's MGF argument, extended by me to real shapes).**
  By (1.5) with x = θc/b, y = θ/b: E e^{θS} ≤ (1 - θ/b)^{-bm} for 0 ≤ θ < b. Hence, for every
  member and every t ≥ m, P(S ≥ t) ≤ exp(-bm(λ - 1 - ln λ)), λ = t/m. The exponent is at
  most b·t, so this cap allows a positive mean only if b·t > ln(1/ε): 4.61 for ε = 0.01,
  2.30 for ε = 0.1. It is safe but useless in our small-b·t regime.
