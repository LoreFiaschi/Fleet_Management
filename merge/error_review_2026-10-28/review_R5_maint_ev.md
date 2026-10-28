Status: done (17 findings: 7 error, 2 inconsistency, 8 meaning; 5 doubts)

Block: scoping.tex "Maintenance strategies", "Focus on electric vehicles", "Degradation models for the shortlisted
components", "Optimization problem for a fleet of electric vehicles", appendix drv:repair_maps; spec Repair Laws
and repair-cost rows (consistency only).
Note: scoping.tex changed on disk during the review (lines moved by about 10). Line numbers below are those of the
last read; every `old` string was copied from the file and checked to be unique.

Scripts (scratchpad):
- `r5_checks.py`: worked numbers of the Anchors paragraph, ARA = Kijima (sympy), ARD1 variance map (sympy),
  Miner sum under unit-specific scatter.
- `r5_module_replacement.py`: battery module replacement against the ARD map (Gaussian and inverse Gaussian).

Verified correct (no finding): the two worked examples of the Anchors paragraph (0.4; 0.25 and 0.5); ARA_1 =
Kijima I and ARA_inf = Kijima II; the ARD_1 variance map; the repair maps and anchor rules against spec
Eqs. (a_ardinf), (a_ard1), (a_psi), (a_anchor) and the factors a = 1-rho (mean, K), (1-rho)^2 (variance, R), 1
(Wiener variance, no anchor); the starting-anchor default; the repair-cost row (eq:z_row) against spec Eq. (a_z)
and its "first repair" exception; the EV objective (eq:objective_evs) against spec Eqs. (a_daycost),
(a_objective) in mode `total`; Proposition prop:ard1_replacement against the "Laws per model" paragraph; the CVaR
caps 0.156, 0.322, 0.141 of tab:ig_cap (reproduced in `r5_module_replacement.py`).

## Findings

### F1 [error] Battery module replacement is not the ARD operator on the damage path
- Where: scoping.tex:3306-3308 (worked example "traction-battery capacity fade", bullet "Maintenance").
- Problem: the text says that renewing the share rho of the modules "is a physical realisation of the ARD repair
  operator R". The ARD_inf operator scales the damage path, D+ = (1-rho) D-, so the variance falls by (1-rho)^2.
  Renewing modules removes summands. With independent modules the variance falls only by (1-rho). Only the mean
  map agrees. A model that tracks the variance then certifies an unsafe state.
- Evidence: `r5_module_replacement.py`, part A. Gaussian model, eps = 0.01, tau = 1, rho = 0.5. Initial damage
  N(0.45, 0.15^2) (passes the check mu >= 3 sqrt(v); row value 0.799). Module replacement, then one known mission
  increment 0.6005 that brings the tracked row to tau. True tail = 1 - Phi(kappa_G sqrt(1-rho)) = 0.0500 = 5 eps.
  Monte Carlo with 10 modules: variance ratio 0.500 (not 0.25), tail 0.0502.
  For the families the text assigns to the battery: the Gamma model stays safe, because the kept damage is still a
  sum of independent Gamma terms with rates >= beta* (Lemma lem:gamma_class, Proposition prop:gamma_safe). For the
  inverse Gaussian model the kept module terms have the dispersion eta/(1-rho), not eta, so Proposition
  prop:ig_order does not apply as stated; part B of the script found no violation (largest tail 0.34 eps), but
  this is not a proof.
- Fix: file `scoping.tex`
  - old: `    This is a physical realisation of the ARD repair operator $\mathcal{R}$, with $\rho$ the fraction of modules renewed.`
  - new: `    This matches the mean map of the $\text{ARD}_\infty$ operator~$\mathcal{R}$, with $\rho$ the fraction of modules renewed, but not its action on the damage path.\n    With independent modules the variance falls by the factor $1-\rho$, not $(1-\rho)^2$, so a model that tracks the variance cannot use the variance map of Section~\ref{sec:repair_connection} for this action.`
- Confidence: high for the mathematics; medium for the wording of the fix.

### F2 [error] The Miner damage is a convolution of its increments only under independence
- Where: scoping.tex:3127-3128 (paragraph "Maintenance under the Palmgren--Miner damage model").
- Problem: "the marginal on any day is the convolution of the individual increments, whatever their distributional
  family". The law of a sum is the convolution of the laws only for independent terms. Under the unit-specific
  scatter (II) of Section sec:miner_randomN, every increment n_j/(A N_j) shares the factor A, so the increments are
  dependent and D_k = M_k/A is not their convolution.
- Evidence: `r5_checks.py`, part 5. ln A ~ N(0, 0.5^2), two increments of median 0.3: Pr(D >= 1) = 0.15347 with
  the common factor (II), 0.10640 for the convolution of the two laws.
- Fix: file `scoping.tex`
  - old: `So the marginal on any day is the convolution of the individual increments, whatever their distributional family.`
  - new: `When the increments are independent, as under the block-independent scatter~(I) of Section~\ref{sec:miner_randomN}, the marginal on any day is the convolution of the individual increments, whatever their distributional family.\nUnder the unit-specific scatter~(II) the increments share the factor~$A$, and the marginal is the law of $M_k/A$.`
- Confidence: high.

### F3 [error] The Miner sum is not bounded by 1
- Where: scoping.tex:3098 (`D_PM = sum_i n_i/N_{f,i} \in [0,1]`) and scoping.tex:3119 ("bounded above by
  construction (failure at 1)").
- Problem: D_PM = sum n_i/N_{f,i} has no upper bound: n = 2 N_f cycles at one level give D_PM = 2. The same
  document says the critical sum is scattered between 0.5 and 2.0 (scoping.tex:1030). If D_PM were in [0,1], the
  constraint Pr(D_PM >= 1) <= eps of the next lines would read Pr(D_PM = 1) <= eps. What is fixed by construction is
  the failure level 1, the normalisation, not a bound on D_PM.
- Evidence: definition (eq:miner); scoping.tex:1018 "Failure is predicted when D_PM >= 1".
- Fix: file `scoping.tex`
  - old: `  D_{\mathrm{PM}} = \sum_i n_i/N_{f,i} \in [0,1],`
  - new: `  D_{\mathrm{PM}} = \sum_i n_i/N_{f,i} \ge 0,`
  - old: `Unlike the stochastic process models, $D_{\mathrm{PM}}$ is bounded above by construction (failure at~1).`
  - new: `Unlike the stochastic process models, $D_{\mathrm{PM}}$ is normalised so that failure occurs at~$1$.`
- Confidence: high.

### F4 [error] Three references to the remaining-life model point to the discrete-time section
- Where: scoping.tex:3096 ("The remaining-life framework of Section~\ref{sec:discrete_deg}"), 3427 ("the
  remaining-life model of Section~\ref{sec:discrete_deg}"), 3462 ("the remaining-life family of
  Section~\ref{sec:discrete_deg}").
- Problem: sec:discrete_deg labels the subsubsection "Discrete-time formulation" (scoping.tex:946-947), which treats
  the damage-accumulation processes sampled once a day. D_PM and the Palmgren--Miner rule are in the next
  subsubsection, "Remaining-life models: the S--N curve and Palmgren--Miner rule" (scoping.tex:1007), which has no
  label.
- Fix: file `scoping.tex`
  - old: `\subsubsection{Remaining-life models: the S--N curve and Palmgren--Miner rule}`
  - new: `\subsubsection{Remaining-life models: the S--N curve and Palmgren--Miner rule}\n\label{sec:remaining_life}`
  - old: `The remaining-life framework of Section~\ref{sec:discrete_deg}`
  - new: `The remaining-life framework of Section~\ref{sec:remaining_life}`
  - old: `remaining-life model of Section~\ref{sec:discrete_deg}`
  - new: `remaining-life model of Section~\ref{sec:remaining_life}`
  - old: `remaining-life family of Section~\ref{sec:discrete_deg}`
  - new: `remaining-life family of Section~\ref{sec:remaining_life}`
- Confidence: high.

### F5 [error] "The two model families of Section sec:stoch_orders" points to the section on stochastic orders
- Where: scoping.tex:3317 ("the same two model families of Section~\ref{sec:stoch_orders}"), 3322 (caption of
  tab:ev_deg_models, "the abstract model families of Section~\ref{sec:stoch_orders}"), 3466 ("the two abstract
  descriptions of Section~\ref{sec:stoch_orders}").
- Problem: the two families (damage-accumulation models and remaining-life models) are introduced at the start of
  the subsection "Stochastic degradation and fatigue models" (scoping.tex:839-844), which has no label.
  Section sec:stoch_orders ("Stochastic orderings") treats the orders within the accumulation families; it does
  not treat the remaining-life family. (Other references of the block to sec:stoch_orders name the Gamma, inverse
  Gaussian or compound Poisson families, which that section does treat; they are not reported.)
- Fix: file `scoping.tex`
  - old: `Stochastic models of component degradation form two families.`
  - new: `\label{sec:deg_families}\nStochastic models of component degradation form two families.`
    (the heading itself occurs twice, at scoping.tex:839 and in the appendix at 4173, so the label goes on the
    first sentence of the subsection, which is unique; it sits directly after the heading)
  - old: `So these components map onto the same two model families of Section~\ref{sec:stoch_orders}.`
  - new: `So these components map onto the same two model families of Section~\ref{sec:deg_families}.`
  - old: `their mapping onto the abstract model families of Section~\ref{sec:stoch_orders}.`
  - new: `their mapping onto the abstract model families of Section~\ref{sec:deg_families}.`
  - old: `Every shortlisted component reduces to one of the two abstract descriptions of Section~\ref{sec:stoch_orders}:`
  - new: `Every shortlisted component reduces to one of the two abstract descriptions of Section~\ref{sec:deg_families}:`
- Confidence: high.

### F6 [error] Table tab:laws_per_model cites, for the cap, the result that does not give it
- Where: scoping.tex:3087 (row "Inverse Gaussian") and 3089 (row "Gamma") of tab:laws_per_model.
- Problem: Gamma row: "the sharp cap holds for every weighted sum of Gamma terms (Lemma lem:gamma_class)". Lemma
  lem:gamma_class only says that the true state belongs to the class G(m); the cap is safe for the class by
  Proposition prop:gamma_safe (which scoping.tex:2708 cites for the same claim). Inverse Gaussian row: "the cap
  holds ... (Proposition prop:ig_order)". Proposition prop:ig_order gives only D <=_icx mu Z; the cap is Theorem
  thm:ig_cap (scoping.tex:2475 cites both).
- Fix: file `scoping.tex`
  - old: `the sharp cap holds for every weighted sum of Gamma terms (Lemma~\ref{lem:gamma_class})`
  - new: `both laws keep the state a weighted sum of Gamma terms (Lemma~\ref{lem:gamma_class}), and the sharp cap holds for every such sum (Proposition~\ref{prop:gamma_safe})`
  - old: `the cap holds for any repair weights and any dependence between the terms (Proposition~\ref{prop:ig_order})`
  - new: `the cap holds for any repair weights and any dependence between the terms (Proposition~\ref{prop:ig_order} and Theorem~\ref{thm:ig_cap})`
- Confidence: high.

### F7 [error] drv:repair_maps does not derive the Hoeffding budget, the ARD_inf case or the map form
- Where: scoping.tex:4935-4940 (appendix, derivation drv:repair_maps, bullet "Variance and Hoeffding budget").
- Problem: the bullet claims a = (1-rho)^2 for the variance and the Hoeffding budget. It shows only
  Var[D+] = Var[G] + (1-rho)^2 Var[Delta] under ARD_1. Three steps are missing: the map form
  a V^- + (1-a) V_G (needs Var[D^-] = Var[G] + Var[Delta]); the ARD_inf case; and the Hoeffding budget
  R = sum_t w_t^2 r_t, which is not a variance and does not follow from the variance identity.
- Evidence: `r5_checks.py`, part 3 (sympy): Var[G] + (1-rho)^2 Var[Delta] - [(1-rho)^2 (Var[G]+Var[Delta]) +
  (1-(1-rho)^2) Var[G]] = 0.
- Fix: file `scoping.tex`
  - old: `      \operatorname{Var}[D^+] = \operatorname{Var}[G] + (1-\rho)^2\operatorname{Var}[\Delta].`
  - new: `      \operatorname{Var}[D^+] = \operatorname{Var}[G] + (1-\rho)^2\operatorname{Var}[\Delta] = (1-\rho)^2\operatorname{Var}[D^-] + \bigl(1-(1-\rho)^2\bigr)\operatorname{Var}[G],`
  - and, after the `\end{equation}` that follows it (inside the same bullet), add:
    `    because $\operatorname{Var}[D^-] = \operatorname{Var}[G] + \operatorname{Var}[\Delta]$.\n    Under $\text{ARD}_\infty$, $\operatorname{Var}[(1-\rho)D^-] = (1-\rho)^2\operatorname{Var}[D^-]$.\n    The Hoeffding budget $R = \sum_t w_t^2 r_t$ follows the same map, because a repair multiplies by $1-\rho$ the weight $w_t$ of each term that it repairs (Proposition~\ref{prop:rf_maps}).`
- Confidence: high (incomplete derivation; the claimed factors are correct).

### F8 [inconsistency] Section sec:evs is cited for content it does not hold
- Where: scoping.tex:835 ("A concrete fleet fixes all of them (Section~\ref{sec:evs})") and scoping.tex:3475
  ("which degradation family describes each component when Section~\ref{sec:evs} specialises the model").
- Problem: line 834 lists four open items: the degradation model, the repair law, the aggregation function phi and
  the cost structure. Section sec:evs holds only the cost notation (tab:notation_evs) and the linear objective
  (eq:objective_evs). It fixes no degradation family, no repair law and no phi (its u still refers to eq:u_def).
  The degradation families of the EV components are in Section sec:ev_deg_models (tab:ev_deg_models). So both
  sentences point to the wrong place, and line 835 claims more than the document gives.
- Fix: file `scoping.tex`
  - old: `A concrete fleet fixes all of them (Section~\ref{sec:evs}).`
  - new: `A concrete fleet fixes all of them. For a fleet of electric vehicles, Section~\ref{sec:ev_deg_models} gives the degradation family of each component, and Section~\ref{sec:evs} gives linear cost functions.`
  - old: `This analysis also fixes which degradation family describes each component when Section~\ref{sec:evs} specialises the model.`
  - new: `This analysis also fixes which degradation family describes each component of an electric-vehicle fleet.`
- Confidence: high.

### F9 [inconsistency] Day-dependent increments do not absorb a non-repeating time trend
- Where: scoping.tex:3299 (battery worked example, bullet "Cycling term").
- Problem: the text says that the day-dependent increment parameters "absorb the non-stationary (power-law)
  dependence on time". Within the T days this holds. But the guarantee of the formulation is for the endless
  schedule, in which day T + n H_2 + h carries the increment laws of day H_1 + h (scoping.tex:267-272,
  Theorem thm:perpetual). A dependence on time that does not repeat with period H_2 is not represented there.
- Fix: file `scoping.tex`
  - old: `absorb the non-stationary (power-law) dependence on time.`
  - new: `absorb the non-stationary (power-law) dependence on time within the horizon.\n        Beyond the horizon the endless schedule repeats the increment laws of the operating phase with period~$H_2$ (Section~\ref{sec:damage_dynamics}), so a dependence on time that does not repeat is not represented there.`
- Confidence: medium.

### F10 [meaning] Tires: the legal tread limit is not the threshold tau
- Where: scoping.tex:3372 (paragraph "Tires", bullet "Fit to the framework").
- Problem: "Tread depth decreases ... to the legal limit, which plays the role of the threshold tau". In the
  formulation the damage grows and fails when it exceeds tau (scoping.tex:76). The tread depth falls, and the legal
  limit is a lower bound on it. The damage must be the tread lost, and tau the new depth minus the legal limit (the
  battery example states its damage and threshold in this way, scoping.tex:3275).
- Fix: file `scoping.tex`
  - old: `    Tread depth decreases monotonically from its new value to the legal limit, which plays the role of the threshold~$\tau$.`
  - new: `    Tread depth decreases monotonically from its new value to the legal limit.\n    The damage is the tread depth lost, and the threshold~$\tau$ is the new depth minus the legal limit.`
- Confidence: high.

### F11 [meaning] Power modules: one symbol for two concepts in consecutive equations
- Where: scoping.tex:3440-3462 (paragraph "Inverter power modules").
- Problem:
  - alpha is the coefficient in N_f = alpha (Delta T_j)^{-n} and the exponent in N_f = A (Delta T_j)^alpha exp(...)
    two lines later. Read with Eq. 3, the second law grows with Delta T_j unless alpha < 0, which is not said.
  - D is the bond-wire diameter in the Bayerer model; everywhere else D is the damage.
  - The Miner sum is called Q here and D_PM in eq:miner and in the Palmgren--Miner paragraph of the maintenance
    section; Q is also the regularised incomplete gamma function Q(a, x) (scoping.tex:2532).
- Fix: file `scoping.tex`
  - old: `      N_f = A\,(\Delta T_j)^{\alpha}\exp\!\bigl(E_a/(k_b T_{jm})\bigr).`
  - new: `      N_f = A\,(\Delta T_j)^{-n}\exp\!\bigl(E_a/(k_b T_{jm})\bigr).`
    (if Alavi 2024, Eq. 4, fits its own exponent, write it as $-n'$ with $n' > 0$)
  - old: `I^{\beta_4}\, V^{\beta_5}\, D^{\beta_6}.`
  - new: `I^{\beta_4}\, V^{\beta_5}\, d_{\mathrm{w}}^{\beta_6}.`
  - old: `the blocking voltage~$V$ and the bond-wire diameter~$D$~\cite[Eq.~1]{Bayerer2008}.`
  - new: `the blocking voltage~$V$ and the bond-wire diameter~$d_{\mathrm{w}}$~\cite[Eq.~1]{Bayerer2008}.`
  - old: `      Q = \sum_i n_i/N_{f,i},`
  - new: `      D_{\mathrm{PM}} = \sum_i n_i/N_{f,i},`
  - old: `    with failure at $Q=1$.`
  - new: `    with failure at $D_{\mathrm{PM}} = 1$.`
- Confidence: high for the clash; medium for the form of Eq. 4 (source not available).

### F12 [meaning] Brakes: the symbol S_0 is not defined
- Where: scoping.tex:3386-3388 (paragraph "Brake friction elements", N = H_0/(S_0 h_wear)).
- Problem: H_0 and h_wear are defined; S_0 is not. With H_0 a depth and h_wear "the wear per braking" (a depth),
  N = H_0/h_wear is already a number of cycles, so S_0 must be dimensionless or h_wear is not a depth. The reader
  cannot tell which.
- Fix: file `scoping.tex` (needs Sha 2022, Eqs. 14 and 22, which this review could not read)
  - old: `    with $H_0$ the admissible wear depth and $h_{\mathrm{wear}}$ the wear per braking~\cite[Eqs.~14,~22]{Sha2022}.`
  - new: `    with $H_0$ the admissible wear depth, $h_{\mathrm{wear}}$ the wear per braking and $S_0$ <definition from Sha 2022, Eq. 22>~\cite[Eqs.~14,~22]{Sha2022}.`
- Confidence: high that S_0 is undefined; the definition must come from the source.

### F13 [meaning] The Chernoff note of tab:laws_per_model names one law only
- Where: scoping.tex:3090 (row "Rainflow, every bound").
- Problem: "Chernoff under ARD_1 uses a conservative recursion for the cumulant generating function". By
  Proposition prop:rf_maps, items 2 and 3, the map of K is an upper bound under both laws, and (M2) says so
  (scoping.tex:397). The note suggests that the map is exact under ARD_inf.
- Fix: file `scoping.tex`
  - old: `Chernoff under $\text{ARD}_1$ uses a conservative recursion for the cumulant generating function (Proposition~\ref{prop:rf_maps})`
  - new: `under both laws, Chernoff uses a conservative recursion for the cumulant generating function (Proposition~\ref{prop:rf_maps})`
- Confidence: high.

### F14 [meaning] Kijima I: "damage" used for virtual age
- Where: scoping.tex:2936.
- Problem: "The repair acts only on the damage accumulated since the previous repair". Kijima models act on the
  virtual age V (the next equation and the Kijima II bullet say so). In this document "damage" is D, the
  degradation level, which the ARD laws reduce. The sentence mixes the two concepts that Table tab:reduction_classes
  separates.
- Fix: file `scoping.tex`
  - old: `The repair acts only on the damage accumulated since the previous repair:`
  - new: `The repair acts only on the virtual age accumulated since the previous repair:`
- Confidence: high.

### F15 [meaning] ARI_1: the symbol lambda^last is not defined
- Where: scoping.tex:3014 (eq:ari1).
- Problem: V^last and D^last are defined; lambda^last is not. eq:ari1 matches the ARI_1 class only when lambda^last
  is the intensity just after the previous repair (then lambda^- - lambda^last = lambda(T_n) - lambda(T_{n-1})).
- Fix: file `scoping.tex`
  - old: `ARI models are natural when maintenance targets the failure rate`
  - new: `where $\lambda^{\mathrm{last}}$ is the failure intensity just after the previous repair.\nARI models are natural when maintenance targets the failure rate`
- Confidence: high.

### F16 [meaning] EV cost function: the damage penalty is not a sum over the horizon
- Where: scoping.tex:3522 (after eq:objective_evs).
- Problem: "Each term is a weighted sum over the planning horizon." The last term C_D u is a maximum over members
  and days (eq:u_def; the next sentence says so). The sentence is false for one of the four terms.
- Fix: file `scoping.tex`
  - old: `Each term is a weighted sum over the planning horizon.`
  - new: `The first three terms are weighted sums over the planning horizon.`
- Confidence: high.

### F17 [meaning] Anchors: "damage only grows" is false for the Gaussian and Wiener models
- Where: scoping.tex:3054 (paragraph "Anchors"); the same premise at scoping.tex:3617 (reasons for the input
  checks, outside this block).
- Problem: "Damage only grows between interventions, so the true value lies between ...". Gaussian increments can
  be negative (scoping.tex:2760), and Wiener paths are not monotone (tab:deg_models, column "Mon."). The conclusion
  needs only that each tracked statistic is non-decreasing between interventions, which fact (M1) gives
  (non-negative increments of each statistic).
- Fix: file `scoping.tex`
  - old: `Damage only grows between interventions, so the true value lies between the value of a new part and the initial value.`
  - new: `Each statistic is non-decreasing between interventions (fact~(M1) of Section~\ref{sec:encoding}), so the true value lies between the value of a new part and the initial value.`
  - At scoping.tex:3617 the same change applies ("Damage only grows between interventions, so the level after the
    last intervention cannot exceed the initial level"); that line belongs to another reviewer's block.
- Confidence: high.

## Doubts

- **Count of electrical sub-families** (scoping.tex:3239): "the traction chain contains four electrical
  sub-families", but the list has three bullets. It is four only if "power semiconductors" and "converters" in the
  first bullet count as two. Not reported because the four emphasised terms may be the intended count.
- **EV objective and the objective modes** (eq:objective_evs): it is the mode `total` of spec Eq. (a_objective).
  The spec also offers `operating_average` and `evaluation_total`. The abstract objective (eq:objective) is written
  without modes too, and the modes follow it in the same subsection, so this is not reported as an inconsistency.
- **"This is exactly a mission-dependent accumulation process of the Gamma or inverse Gaussian type"**
  (scoping.tex:3297): monotone and additive increments do not imply a Gamma or inverse Gaussian law; a compound
  Poisson process with positive jumps (listed in tab:ev_deg_models as an accumulation family) is a counterexample.
  "Exactly" may mean "of the kind of", so this is not reported.
- **Wheel bearings, one family or two** (tab:ev_deg_models row "Wheel bearing": "Accumulation / remaining-life";
  "Common structure" lists bearings only under the accumulation process). The bearing paragraph describes both
  stages, so the list may be a simplification.
- **Pr(D_PM >= 1) against Pr(D > tau)** (scoping.tex:3122 and eq:abs_rel): the two differ only by the atom at
  D_PM = 1, which matters in the deterministic variant. eq:miner_quantile uses ">=" too, so the choice looks
  deliberate.
