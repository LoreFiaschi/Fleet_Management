# Unimodality of sums of independent Gamma variables: literature check

Status: done (26 Oct 2026). Read in full: Hasebe-Thorbjørnsen (2016), James-Roynette-Yor
(2008), Roosta-Khorasani-Székely (2015), Székely-Bakirov (2003). Read in part: Yamazato (1978),
p. 523 only (JSTOR preview; the rest blocked). Not read: Bondesson (1992), Sato (1999).
Next: the user retrieves Yamazato (1978) pp. 524-531 on the ETH network, if the proof must be
read too.

**Statement U.** Every finite sum of independent Gamma random variables (any shapes > 0, any
rates > 0), possibly plus a nonnegative constant, has a unimodal distribution on the real line.
When the law is absolutely continuous, its density is nondecreasing up to a mode and
nonincreasing after it.

Acronyms and terms: GGC = generalized gamma convolution; SD = self-decomposable (the same
class as Khintchine's class L); PDF = probability density function; JRY = James, Roynette and
Yor (2008). Gamma(a, r) has shape a and rate r.

## 1. Yamazato (1978): class L laws are unimodal

- **Reference.** M. Yamazato, "Unimodality of infinitely divisible distribution functions of
  class L", Annals of Probability 6(4) (1978), 523-531, doi:10.1214/aop/1176995474.
- **URLs tried (26 Oct 2026).**
  - Project Euclid PDF and article page: bot block (Incapsula) to PowerShell, and a
    human-verification check ("I am human") in the browser pane. Not passed. WebFetch got an
    empty page.
  - JSTOR, https://www.jstor.org/stable/2243119: the public preview shows page 523 only, as a
    page scan. The other 8 pages need a library login. The computer's address (not on the ETH
    network) has no JSTOR licence.
- **Evidence basis: read in part, p. 523 only (1 of 9 pages).** Not read: pp. 524-531, so the
  whole proof. I can report what the paper states, not how it proves it.
- **Theorem 1 (p. 523).** "All distribution functions of the class L are unimodal."
  No other hypothesis. Unimodal is defined on the same page (see link 6).
- **Class L (p. 523).** F is in class L if it is the limit law of normalized sums
  B_n^(-1) Σ_{k≤n} X_k - A_n of independent X_k, with B_n > 0 and the summands X_k/B_n
  uniformly asymptotically negligible. Reference given: Gnedenko-Kolmogorov (1954).
- **History (p. 523).** Gnedenko and Kolmogorov asserted this theorem in the Russian edition
  (1949) of their book; their proof relied on a theorem of Lapin about convolutions of
  unimodal laws. The rest of that paragraph is on p. 524, not read.
- **How it serves.** It is the final step: a law in class L is unimodal. The statement is read;
  its proof is not.

## 2. Characterization of self-decomposable laws by the Lévy measure

- **Source.** Yamazato (1978), p. 523, equation (1). Same evidence basis as link 1: read in
  part, p. 523. Yamazato attributes the result to Lévy (1937); Lévy (1937) not read.
- **Statement (p. 523, (1)), paraphrased.** F belongs to class L if and only if
  log F̂(t) = iγt - σ²t²/2
  + ∫_(-∞,0) (e^(itu) - 1 - itu/(1+u²)) |u|^(-1) l(u) du
  + ∫_(0,∞) (e^(itu) - 1 - itu/(1+u²)) u^(-1) k(u) du,
  where F̂ is the characteristic function, γ real, σ ≥ 0, and
  - k(u) ≥ 0, l(u) ≥ 0;
  - ∫_(-1,0) |u| l(u) du + ∫_(0,1) u k(u) du < ∞;
  - ∫_(1,∞) u^(-1) k(u) du + ∫_(-∞,-1) |u|^(-1) l(u) du < ∞;
  - k(u) and -l(u) are nonincreasing.
- So the Lévy measure has density k(u)/u on (0, ∞), with k nonincreasing, and l(u)/|u| on
  (-∞, 0), with l nondecreasing. This is the characterization of link 2.
- **Terms.** "Class L" (Yamazato) and "self-decomposable" (SD) name the same class; Yamazato's
  page 523 uses only "class L". The equivalence of the two definitions is not needed below:
  equation (1) is "necessary and sufficient" for class L, and the chain uses (1) directly.
- **How it serves.** To show a law is unimodal, it is enough to write its log characteristic
  function in form (1) with k nonincreasing and l = 0.

### Second source for links 1, 2 and 6: Hasebe and Thorbjørnsen (2016)

- **Reference.** T. Hasebe, S. Thorbjørnsen, "Unimodality of the freely selfdecomposable
  probability laws", Journal of Theoretical Probability 29 (2016), 922-940,
  doi:10.1007/s10959-015-0595-y (journal data from Crossref). Read: arXiv:1309.6776v3
  (7 Jan 2015), https://arxiv.org/pdf/1309.6776.
- **Evidence basis: read in full** (17 pages, text layer). The text layer loses many formula
  symbols; the statements used below are in plain words and were readable.
- **What it carries (classical case only; the paper's own results are on free convolution).**
  - p. 1, (1.1): Lévy's characterization of class L(∗): μ is in it if and only if, for every
    c ∈ (0, 1), μ = D_c μ ∗ μ_c for some probability measure μ_c (D_c μ: the law of c·X).
    "As a result ... the measures in L(∗) are called selfdecomposable."
  - p. 1: Yamazato (1978) gave the first correct proof that all self-decomposable laws are
    unimodal; the 1949 claim of Gnedenko-Kolmogorov had an error in its proof. Cited as a
    whole paper, no theorem number.
  - p. 1, (1.2): unimodal with mode a means μ(dt) = μ({a})δ_a(dt) + f(t)dt, f nondecreasing
    on (-∞, a) and nonincreasing on (a, ∞) (their "increasing" means nondecreasing, p. 1).
  - p. 5, top: (1.2) is equivalent to: the distribution function is convex on (-∞, a) and
    concave on (a, ∞). This is Yamazato's definition, so the two definitions agree.
  - p. 4, (2.11): μ ∈ L(∗) if and only if μ is infinitely divisible and its Lévy measure is
    k(t)/|t| dt, with k ≥ 0 nondecreasing on (-∞, 0) and nonincreasing on (0, ∞). Cited to
    Sato (1999) without a location.
  - p. 15, Remark 4.3: a non-degenerate self-decomposable law is absolutely continuous (Sato
    1999, Theorem 27.13, as carried); its density may have one point of discontinuity (Sato
    1999, Theorem 28.4, as carried). Sato (1999) not read.
- **How it serves.**
  - Link 1: an independent, peer-reviewed statement of Yamazato's theorem.
  - Link 2: the characterization in the modern form (2.11), with the name "selfdecomposable".
  - Link 6: the density form (1.2) of unimodality is equivalent to Yamazato's convexity form.
    This supports the density statement of Statement U with a source, beyond my own argument.

## 3. Lévy measure of Gamma(a, r)

- **Source.** L. F. James, B. Roynette, M. Yor, "Generalized Gamma Convolutions, Dirichlet
  means, Thorin measures, with explicit examples", Probability Surveys 5 (2008), 346-415,
  doi:10.1214/07-PS118. Read: arXiv:0708.3932v2 (22 Jan 2009), https://arxiv.org/pdf/0708.3932;
  it carries the journal pagination 346-415, used below. Abbreviated JRY.
- **Evidence basis: read in full** (70 pages, text layer; pp. 348, 349, 351 also as page
  images, because the text layer breaks the formulas).
- **p. 349, §1.1, (1)-(2), (4).** The standard Gamma process (γ_t) is a subordinator without
  drift, with Lévy measure e^(-x)/x dx on x > 0. So E e^(-λγ_t) = exp(-t ∫_0^∞ (1-e^(-λx))
  e^(-x)/x dx) = (1+λ)^(-t), and γ_t has the density e^(-a) a^(t-1)/Γ(t): γ_t ~ Gamma(t, 1).
- **p. 349-350, (3) (Frullani's formula).** ∫_0^∞ (1-e^(-λx)) (dx/x) ∫_0^∞ e^(-xz) ν(dz) =
  ∫_0^∞ log(1 + λ/z) ν(dz), for a σ-finite positive measure ν on ℝ₊. JRY prove it in one
  line (p. 350): both sides vanish at λ = 0 and have the same derivative in λ.
- **Consequence for Gamma(a, r)** (JRY (3) with ν = a·δ_r, my substitution): the Laplace
  transform (1 + λ/r)^(-a) equals exp(-∫_0^∞ (1-e^(-λx)) a e^(-rx)/x dx). So Gamma(a, r) is
  infinitely divisible, with Lévy measure a e^(-rx)/x dx on (0, ∞) and no drift.
- **Characteristic function, for Yamazato's form (1) (my own check; JRY use Laplace
  transforms).**
  - Claim: log E e^(itX) = -a log(1 - it/r) = a ∫_0^∞ (e^(itx) - 1) e^(-rx)/x dx, X ~ Gamma(a, r).
  - Proof: both sides vanish at t = 0. Both have the derivative a·i/(r - it) in t: the left
    side directly (principal logarithm; Re(1 - it/r) = 1 > 0); the right side by
    differentiation under the integral, dominated by a·e^(-rx). JRY's argument for (3), with
    λ = -it.
- **How it serves.** It gives k(x) = a e^(-rx) in Yamazato's form (1): nonincreasing on
  (0, ∞), with l = 0 and σ = 0.

## 4. Closure under independent sums and shifts

- **Not needed as a separate result.** The chain below writes the whole sum S = c + Σ_i X_i,
  X_i ~ Gamma(a_i, r_i) independent, c ≥ 0, directly in Yamazato's form (1):
  - log E e^(itS) = itc + Σ_i a_i ∫_0^∞ (e^(itx) - 1) e^(-r_i x)/x dx (link 3, and
    independence: the characteristic function of a sum is the product).
  - Put k(x) = Σ_i a_i e^(-r_i x). Then
    log E e^(itS) = itγ + ∫_0^∞ (e^(itx) - 1 - itx/(1+x²)) k(x)/x dx,
    with γ = c + ∫_0^∞ k(x)/(1+x²) dx. This integral is finite, since k ≤ Σ_i a_i.
  - k ≥ 0 and k is nonincreasing (a sum of nonincreasing functions). The integrability
    conditions of (1) hold: ∫_0^1 x k(x) dx ≤ (1/2) Σ_i a_i, and
    ∫_1^∞ k(x)/x dx ≤ ∫_1^∞ k(x) dx = Σ_i a_i e^(-r_i)/r_i. l = 0, σ = 0.
- **The general closure, for the record (my own one-line argument from Lévy's definition,
  Hasebe-Thorbjørnsen p. 1, (1.1)).** If μ = D_cμ ∗ μ_c and ν = D_cν ∗ ν_c, then
  μ∗ν = D_c(μ∗ν) ∗ (μ_c∗ν_c), because D_c(μ∗ν) = D_cμ ∗ D_cν. A shift: δ_b∗μ =
  D_c(δ_b∗μ) ∗ (δ_((1-c)b) ∗ μ_c). So sums of independent self-decomposable variables, and
  shifts of them, are self-decomposable.
- **JRY gives the sum directly** (p. 350-351, §1.2.a): Σ_i h_i(γ_(s_(i+1)) - γ_(s_i)), a
  linear combination of independent Gamma variables (shapes s_(i+1) - s_i, scales h_i), has
  Laplace transform exp(-∫_0^∞ (1-e^(-λy)) (dy/y) ∫ e^(-yx) μ_h(dx)). It is a GGC with
  Thorin measure μ_h = Σ_i (s_(i+1) - s_i) δ_(1/h_i). Every finite sum of independent Gamma
  variables has this form.

## 5. Direct statements for sums of Gamma variables or GGC

### James, Roynette and Yor (2008): GGC are self-decomposable

- Same source as link 3; read in full.
- **p. 351, Definition 1.0** (after Bondesson 1992, p. 29). A positive r.v. Γ is a GGC
  (generalized gamma convolution) without translation term if E e^(-λΓ) =
  exp(-∫_0^∞ (1-e^(-λx)) (dx/x) ∫ e^(-xz) μ(dz)) = exp(-∫ log(1+λ/z) μ(dz)), for a
  positive Radon measure μ on (0, ∞) (the Thorin measure) with ∫_(0,1] |log x| μ(dx) < ∞ and
  ∫_[1,∞) μ(dx)/x < ∞ ((7)-(9)).
- **p. 351, right after Definition 1.0.** The Lévy density l(x) = (1/x)∫ e^(-xz) μ(dz) has
  x·l(x) decreasing, so a GGC is self-decomposable (JRY cite Lukacs 1970; not read). A
  non-degenerate one has a density f > 0 on (0, ∞) (JRY cite Sato 1999, p. 404; not read).
- **p. 348.** G ⊂ S ⊂ J (positive GGC ⊂ positive self-decomposable ⊂ positive infinitely
  divisible). If X ∈ S then ν(dx) = h(x) dx/x with h decreasing (necessity only, as stated).
  "Another definition of a GGC" is the limit in law of sums of independent Gamma variables
  with different parameters.
- **For our sum.** A finite sum of independent Gamma(a_i, r_i) is the GGC with Thorin measure
  μ = Σ_i a_i δ_(r_i) (conditions (9) hold, since μ has finitely many atoms in (0, ∞)).
- **Not in JRY.** The word "unimodal" does not occur (text search of the whole paper).
  Unimodality needs Yamazato's theorem on top.

### Roosta-Khorasani and Székely (2015): Theorem 4, a direct statement of unimodality

- **Reference.** F. Roosta-Khorasani, G. J. Székely, "Schur properties of convolutions of
  gamma random variables", Metrika 78 (2015), 997-1014, doi:10.1007/s00184-015-0537-9
  (journal data as recorded in `gamma_literature.md`; not checked again). Read:
  arXiv:1601.04731v1, https://arxiv.org/pdf/1601.04731.
- **Evidence basis: read in full** (18 pages, text layer).
- **Theorem 4 (arXiv p. 11, Appendix A).** X_i ~ Gamma(α_i, β_i) independent, any α_i, β_i > 0,
  λ_i ≥ 0. Then the PDF (probability density function) of Σ λ_i X_i is unimodal.
  - Their definition (p. 11): f is unimodal if there is a unique a with f nondecreasing for
    x < a and nonincreasing for x > a.
  - **No proof in the paper.** They write that the proof "is identical to the proof of
    [29, Theorem 4]", that is Székely-Bakirov (2003), Theorem 4. They add that [29] states
    that theorem for chi-squared variables but proves it for arbitrary Gamma variables
    (p. 11).
  - The paper uses Theorem 4 inside its proofs (p. 11-17), for example to say that the mode
    of a convolution of Gamma variables is unique.
- **Also in Roosta-Khorasani, Székely, Ascher (2015)**, arXiv:1404.0122, p. 34 (searched, not
  read in full here; read in full in `gamma_literature.md`): "the unimodality ... was already
  proven for all gamma random variables in [32, Theorem 4]", with [32] = Székely-Bakirov
  (2003). Not used as evidence below.
- **How it serves.** A direct statement of Statement U for the absolutely continuous case
  (c = 0 and at least one term), in a peer-reviewed paper. Its proof is delegated to
  Székely-Bakirov (2003), Theorem 4; see the next entry.

### Székely and Bakirov (2003): Theorem 4 and its proof

- **Reference.** G. J. Székely, N. K. Bakirov, "Extremal probabilities for Gaussian quadratic
  forms", Probability Theory and Related Fields 126 (2003), 184-202,
  doi:10.1007/s00440-003-0262-6. Read: the journal PDF,
  https://link.springer.com/content/pdf/10.1007/s00440-003-0262-6.pdf (served 26 Oct 2026).
- **Evidence basis: read in full** (19 pages, text layer; pp. 190, 194, 195 also as page
  images).
- **Definition (p. 190).** A PDF p is unimodal if there is at least one a with p nondecreasing
  for x < a and nonincreasing for x > a. (Weaker than Roosta-Khorasani-Székely: no
  uniqueness.)
- **Theorem 4 (p. 190).** The PDF of Q_n = Σ λ_i Z_i² is unimodal for arbitrary real λ_i
  (Z_i independent standard normal; so each Z_i² is Gamma(1/2, 1/2)). The statement covers
  chi-square terms only.
- **Proof of Theorem 4 (pp. 194-195).** The argument is general:
  - Class L (self-decomposable): f(t) = f(ct) f_c(t) for every c ∈ (0, 1), with f_c a
    characteristic function (cited to Lukacs 1970, section 5.11; not read).
  - "The definition implies" that linear combinations of independent class-L variables
    are in class L (p. 194-195).
  - All class-L variables are unimodal (cited to Yamazato 1978).
  - Gamma(λ, θ), any λ > 0, θ > 0: (1 - it/θ)^(-λ) = exp{ita + ∫_0^∞ (e^(itu) - 1 -
    itu/(1+u²)) dN(u)}, with a = λ∫_0^∞ e^(-θy)/(1+y²) dy and N(u) = -λ∫_u^∞ e^(-θx)/x dx
    (cited to Lukacs 1970, Table 5, section 5.5; not read). So dN(u) = λ e^(-θu)/u du.
  - u N'(u) = λ e^(-θu) decreases on [0, ∞), so all Gamma variables are in class L (Lukacs
    1970, Theorem 5.11.2; not read).
- **How it serves.**
  - Link 3 a second time: the Lévy measure λ e^(-θu)/u du of Gamma(λ, θ), in
    characteristic-function form, with the same centring a as in my link-4 computation.
  - Link 4: closure of class L under linear combinations of independent variables.
  - Link 5: the whole chain of Statement U, written out by the authors, for any shapes and
    any rates. This is the proof that Roosta-Khorasani-Székely (2015, Theorem 4) refer to.

## 6. Meaning of "unimodal" in Yamazato (1978)

- **Definition (p. 523).** F is unimodal with mode m if F(x) is convex for x < m and concave
  for x > m. F is unimodal if it is unimodal with mode m for some m.
- **Source for the density form.** Hasebe-Thorbjørnsen (2016), p. 1, (1.2) and p. 5: this
  definition is equivalent to μ(dt) = μ({m})δ_m(dt) + f(t)dt, with f nondecreasing on
  (-∞, m) and nonincreasing on (m, ∞). See link 2, second source.
- **Consequence for a density (my own argument, elementary convex analysis).**
  - A convex function on an open interval has a right derivative there, and it is
    nondecreasing. The same holds for a concave function, with "nonincreasing".
  - So F has a right derivative f₊ on (-∞, m) ∪ (m, ∞), nondecreasing on (-∞, m) and
    nonincreasing on (m, ∞). A convex or concave function is absolutely continuous on each
    compact subinterval, so F(y) - F(x) = ∫_x^y f₊ for x < y on the same side of m.
  - F may jump at m only (an atom at the mode). If the law is absolutely continuous, F has no
    jump, so f₊ is a version of the density on ℝ \ {m}.
  - Hence: an absolutely continuous unimodal law has a density that is nondecreasing on
    (-∞, m) and nonincreasing on (m, ∞). This is the density statement of Statement U.
  - A density is defined only almost everywhere. The statement holds for the version f₊.
    f₊ is right-continuous, so a density that is continuous on one side of m equals f₊
    everywhere on that side, and has the same shape.
- **Degenerate case.** A point mass at c (zero Gamma terms) is unimodal with mode c: F is 0 on
  (-∞, c) and 1 on (c, ∞), both convex and concave.

## Sources not read

- **Yamazato (1978), pp. 524-531**: see link 1. Project Euclid gave a human-verification
  check; JSTOR needs a library login. The proof of Theorem 1 is therefore not read.
  Worth retrieving on the ETH network: it is the last link of every chain below.
- **Bondesson (1992)**, Lecture Notes in Statistics 76, doi:10.1007/978-1-4612-2948-3.
  Tried https://link.springer.com/content/pdf/10.1007/978-1-4612-2948-3.pdf and the chapter
  PDFs _1, _3, _4 (26 Oct 2026): Springer returned the landing page ("Log in via an
  institution"), so no licence from this computer. Optional; not needed for the verdict.
- **Sato (1999)**, Lévy Processes and Infinitely Divisible Distributions: not attempted. Not
  needed: the characterization is read on Yamazato p. 523, and is carried by
  Hasebe-Thorbjørnsen p. 4 and Székely-Bakirov p. 195 (via Lukacs).
- **Steutel and van Harn (2004), Lukacs (1970), Lévy (1937)**: not attempted. Known here only
  as cited by the sources above.

## Verdict

**Statement U is a known theorem.** Every link is stated in a source read here. The one link
whose proof was not read is Yamazato's Theorem 1 itself (only its statement, p. 523).

**Chain 1 (Yamazato's form (1)), S = c + Σ_i X_i, X_i ~ Gamma(a_i, r_i) independent, c ≥ 0.**

1. log E e^(itX_i) = a_i ∫_0^∞ (e^(itx) - 1) e^(-r_i x)/x dx. Source: JRY (2008), p. 349-350,
   (1)-(3) (Laplace form, Frullani); Székely-Bakirov (2003), p. 195 (characteristic-function
   form). The complex form is also checked here (link 3).
2. Then log E e^(itS) has Yamazato's form (1) with σ = 0, l = 0, k(x) = Σ_i a_i e^(-r_i x)
   (nonincreasing, ≥ 0, integrability conditions hold), γ = c + ∫_0^∞ k(x)/(1+x²) dx.
   Computation in link 4.
3. Form (1) with k nonincreasing and -l nonincreasing is necessary and sufficient for class L.
   Source: Yamazato (1978), p. 523, (1) (after Lévy 1937). Same characterization:
   Hasebe-Thorbjørnsen (2016), p. 4, (2.11).
4. Every class-L law is unimodal (F convex on (-∞, m), concave on (m, ∞)). Source: Yamazato
   (1978), Theorem 1, p. 523.
5. For an absolutely continuous law this means a density nondecreasing on (-∞, m) and
   nonincreasing on (m, ∞). Source: Hasebe-Thorbjørnsen (2016), p. 1, (1.2), and p. 5 (the
   equivalence of the two definitions); my own argument in link 6.

**Chain 2 (the authors' own proof).** Roosta-Khorasani-Székely (2015), Theorem 4 (arXiv p. 11)
states the result for Σ λ_i X_i, X_i ~ Gamma(α_i, β_i), λ_i ≥ 0. Its proof is Székely-Bakirov
(2003), proof of Theorem 4 (pp. 194-195): Gamma ∈ class L (Lukacs 1970, Theorem 5.11.2), class
L closed under independent linear combinations, class L ⊂ unimodal (Yamazato 1978).

**Chain 3 (via GGC).** A finite sum of independent Gamma variables is a GGC with Thorin measure
Σ a_i δ_(r_i) (JRY 2008, p. 350-351, §1.2.a and Definition 1.0); a GGC is self-decomposable
(JRY 2008, p. 351, citing Lukacs 1970); then step 4.

**The constant c ≥ 0.** It changes only γ in form (1), or by the one-line argument in link 4.
It shifts the mode by c. A point mass (no Gamma term) is unimodal too.

**Gaps.**
- Yamazato's proof (pp. 524-531) not read. The theorem is used as known by Székely-Bakirov
  (2003, p. 195) and Hasebe-Thorbjørnsen (2016, p. 1), both peer-reviewed. Hasebe-Thorbjørnsen
  (p. 1) report that the earlier proof of Gnedenko-Kolmogorov (1949) had an error and that
  Yamazato's 1978 proof is the correct one.
- The criterion "k nonincreasing ⇒ class L" was read only as a statement (Yamazato p. 523,
  Hasebe-Thorbjørnsen p. 4, JRY p. 351, Székely-Bakirov p. 195); its proof (Lévy 1937, Lukacs
  1970, Sato 1999) was not read. It is the classical Lévy characterization, stated alike by
  four sources.
- Uniqueness of the mode is not part of Statement U. Roosta-Khorasani-Székely define
  unimodality with a unique mode; the Székely-Bakirov argument gives only "at least one mode".
  Not needed here.
- The density statement holds for a version of the density (the right derivative of F). At
  the mode the density may be infinite or jump (for example Σ a_i ≤ 1 gives a mode at c).
