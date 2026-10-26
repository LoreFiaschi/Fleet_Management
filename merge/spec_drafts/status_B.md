# Writer B: status

Status: done (26 Oct 2026), including the coordinator's corrections of 26 Oct 2026 (section "Corrections applied"
below). All five model sections written in `fragments_B.tex` (1109 lines), in the final order
Gaussian, inverse Gaussian, Wiener, Gamma, rainflow (one contiguous block). Built twice: 0 errors, 0 overfull
boxes; the only undefined references are other writers' labels (`sec:consistency_checks`, `sec:encoding`,
`sec:linear_forms`, `sec:loop`, `sec:repair_laws`, `sec:replacement`). PDF rendered and read (13 pages at 90 dpi).
Next: nothing; the questions below wait for Claude. Build: `bash build_B/build.sh` from `merge/spec_drafts/`.

## Checklist of the decision bullets owned by writer B

- [x] Set of degradation models: the five model sections, as one model list.
- [x] Names and set of repair laws: the laws per model, with the derivations (Gaussian ARD1 exact by independence,
      Proposition `prop:g_exact`; inverse Gaussian both laws (follow-up decision); Wiener both laws on the mean only; Chernoff ARD1
      recursion with its proof, Proposition `prop:rf_maps`; the maps of R under both laws, same proposition); the
      remark that refuses ARD1 for Gaussian is absent.
- [x] Rainflow tail bounds: the five bounds, each with its row, tracked quantities and validity argument; the
      Bernstein side row mu <= tau - c; the support constant b = max(b0, bnew, max over increments).
- [x] Replacement (rainflow part): starting and replacement values of R and K with the Jensen argument; inputs R_0,
      R_new, cgf_0, cgf_new.
- [x] Linear forms: applied to every quadratic row (Gaussian, Wiener, Cantelli, Bernstein, Hoeffding), with the
      convex function f of each row (Gaussian section "Forms"; rainflow table "Linear forms").
- [x] Reliability level epsilon: the remark on V_max as the natural range; eps < 0.5 at Gaussian and Wiener cells.
- [x] Inverse Gaussian reliability cap: the convex-order argument, the CVaR cap, its computation, the correct density,
      the loop argument.
- [x] Gamma formulation: tracked state, beta* per cell, class of true states, the theorem with its proof, the
      computation of A_1, the remark on t1(eps).

## Labels defined (for cross-references by Claude and the other writers)

`sec:dyn_gaussian` `eq:g_map_mu` `eq:g_map_v` `prop:g_exact` `eq:g_row` `sec:dyn_ig` `eq:ig_map` `prop:ig_order`
`eq:ig_row` `thm:ig_cap` `sec:dyn_wiener` `eq:w_map_v` `eq:w_vrow` `sec:dyn_gamma` `eq:gm_map` `eq:gm_betastar`
`def:gm_class` `lem:gm_class` `lem:gm_rates` `thm:gm_oneterm` `eq:gm_oneterm` `lem:gm_monotone` `eq:gm_row`
`cor:gm_safe` `sec:dyn_rainflow` `prop:rf_maps` `eq:rf_markov` `eq:rf_cantelli` `eq:rf_hoeffding` `eq:rf_bsupport`
`eq:rf_bernstein` `eq:rf_chernoff`.

- Version 0.5 labels that no longer exist in the model sections: `eq:rf_vmax` (the variance ranges are now inline,
  `V^C`, `V^B`, in the rainflow rows table), `eq:ig_mubar` (use `eq:ig_row`), `eq:g_lp_reliability`,
  `eq:w_lp_reliability`, `eq:rf_cantelli_exact`, `eq:rf_cantelli_lp`, `eq:rf_bernstein_exact`, `eq:rf_bernstein_lp`
  (forms are in `sec:linear_forms`), `eq:rf_loop`, `eq:w_loop_anchor`, `eq:gamma_loop_anchor` (loop rows are writer
  A's), `eq:g_ard_mu1`, `eq:g_ard_mu2`, `eq:ig_mu1`, `eq:ig_mu2`, `eq:rf_ard_mu1`, `eq:rf_ard_mu2`, `eq:g_rep`,
  `eq:rf_rep` (generic rows are writer A's). The v0.5 notation table cites `eq:rf_bsupport` (kept),
  `eq:rf_vmax` and `eq:ig_mubar` (gone): writer A's notation table must use `eq:rf_bsupport`, `eq:ig_row`.

## Facts from the code (file and line)

- Gamma inputs on `main`: `gamma_beta` (rates, (F,L,M,H)), `gamma_beta_trans`, `gamma_beta_0` (required when
  `mu_0 > 0`), `gamma_beta_new` (required when `replacement_mu > 0`): main:src/fleet_management/config.py:71-75,
  536-547.
- Rainflow on `main` (main:src/fleet_management/degradation_model/rainflow.py):
  - Hoeffding budget: increment `w2_inc = support**2` (lines 148-149); carry `R + sum x*w2` (291-294); ARD1
    `R+ = (1-rho)^2 R + (1-(1-rho)^2) gR` (303-307); ARDinf `(1-rho)^2 R` (309); row
    `(tau-mu)^2 >= 0.5*ln(1/eps)*R`, `mu <= tau` (437-444). So `delta R` = sum of squared support widths:
    confirmed. The spec's Hoeffding row equals main's.
  - Bernstein row `0.5 d^2 - (Le b/3) d - Le v >= 0` (447-456) = spec's `(d - c)^2 >= c^2 + 2 kappa v`;
    side row on main `mu <= tau`, spec `mu <= tau - c` (decision).
  - Chernoff row `K - s tau <= ln eps` (459-464); ARDinf `K+ = (1-rho) K` (334); K starts at 0 (327) and resets
    to 0 (337): the spec uses `K_0 = s mu_0`, `K_new = s mu_new` by default (decision, Defect B).
  - Markov `mu <= eps tau` (420-424). Cantelli `(1-eps) v <= eps d^2`, `mu <= tau` (427-434).
  - Tangent: `Q <= g(mu_p) + g'(mu_p)(mu - mu_p)` at `mu_p = tangent_ref*tau` (399-410, 470-479); pwl: segments
    of [0, tau], midpoint tangents, one binary per segment (485-524). Coefficients `_cap_coeffs` (385-396):
    Cantelli `eps/(1-eps)`, Hoeffding `2/Le`, Bernstein `1/(2 Le)` and `-b/3`: these are the spec's `f`.
  - Variance anchor under ARD1: `v+ = (1-rho)^2 v + (1-(1-rho)^2) gv` (243-247): the Gaussian and rainflow variance
    maps of the spec.
- `s_chernoff > ln(1/eps)/tau` check: main:src/fleet_management/config.py:506-518.
- Main's Chernoff ARD1 refusal: main:src/fleet_management/config.py:484-494 and rainflow.py:122-127; the spec admits
  ARD1 with `K+ = (1-rho) K + rho gK` (decision).

## Check scripts

- `checks/B_gamma.py` (output `checks/B_gamma_output.txt`): A_hat and A_1/A_hat against the T2 table (all match);
  r*(eps) = 0.003550, 0.035289, 0.330617 (constant over t = 0.001 to 0.4, below t1); A_1(t)/t nondecreasing in t;
  boundary case w(eps t) <= eps (Markov), ratio 0.28 at most; single peak of c -> Q(A/c, t/c) in 54 cases; the
  witness eps = 0.01, t = 0.1: c* = 0.2191, rate ratio 4.56, tail 1.5649 eps.
- The Gamma theorem (Statement 2) and its proof steps: checked in `merge/analysis/scripts/verify_gamma_sharp/`
  (`check.py`, `check_output.txt`) and `merge/analysis/scripts/gamma_sharp/` (t1b global search, t1d unimodality);
  t1(eps) values from `verify_gamma_sharp/check_output.txt`.
- `checks/B_rainflow.py` (output `checks/B_rainflow_output.txt`): Chernoff ARDinf and ARD1 inequalities on 2000
  random discrete laws each (largest violation 7e-15, rounding), with the boundary cases rho = 0, rho = 1 and a
  constant Delta (equality); Jensen K >= s E D; the ARD1 variance identity and the R recursion against term
  bookkeeping (both laws, random schedules with replacements); Hoeffding and Bernstein row equivalences; the forced
  region d >= 2c; V^B = f_B(0); the Bernstein tangent at mu_r >= tau - c is empty; V_max/tau^2 = 0.1848, 0.6089,
  3.636, 1591 at eps = 0.01, 0.1, 0.3, 0.49; the Chernoff example with K_0 = s mu_0.
- `checks/B_ig.py` (output `checks/B_ig_output.txt`): the closed form E[Z; Z > q] = F(1/q) (writer B's own
  derivation, from z f(z) = z^-2 f(1/z)) against quadrature, 12 cases, relative difference at most 7e-8; the cap
  table (ratios to tau/q equal those of `merge/analysis/inverse_gaussian.md` S4.1 to 2 digits; tau/q equals the
  analysis's lore cap); the stop-loss bound equals eps at the cap; boundary eta -> 0 (C -> 1); the stop-loss order
  by Monte Carlo (independent sums below the comonotone closed form); the quantile-cap example eta = 10, eps = 0.1:
  tail 0.1181 +- 0.0002 (analysis: 0.1180); the IG(mu, mu/eta) density in the spec integrates to 1.
- The proof of the stop-loss order (S1.2) and the CVaR theorem (S4.1): `merge/analysis/inverse_gaussian.md`, checks
  in `merge/analysis/scripts/ig/` and `merge/analysis/scripts/verify_ig/`. The spec's proof of the order is the
  convexity argument of the brief, a different route from S1.2 (stop-loss split); both give the same statement.
- Gaussian ARD1 exactness: algebra in Proposition `prop:g_exact`; the variance identity is checked in
  `B_rainflow.py` (4).

## Notes for Claude (points recorded as asked)

- Support width (conventions.md): the spec uses `main`'s width semantics and Popoviciu's inequality `v <= w^2/4` as
  the consistency condition (rainflow paragraph "Width of a range"); the check itself is writer C's.
- Assumption A3 (repetition) is stated in the rainflow section, as the brief asks; it is the setting of the loop
  argument for every model. If writer A states it in `sec:loop`, A3 can become a one-line reference.
- Each model section ends with three facts (M1)-(M3) for writer A's arguments: monotone day maps; the true
  statistics follow the maps (with `<=` for the Chernoff `K`); the row certifies any true state below the tracked one.

## Corrections applied (coordinator's message, 26 Oct 2026; `merge/decisions.md`, item "Follow-up decisions from
## writing the spec")

1. Inverse Gaussian accepts both laws: anchor `g^mu` under ARD1, map `mu+ = (1-rho) mu + rho g^mu` (Eq.
   `eq:ig_map`), rows table with the anchor; Proposition `prop:ig_order` stated for any repair weights in (0, 1]
   (`D = sum_t w_t a_t Z_t`); the lead-in says how ARDinf and ARD1 (`D+ = G + (1-rho) Delta`) set the weights, and
   that a starting anchor `g0 < mu_0` reads `mu_0 Z_0` as `g0 Z_0 + (mu_0 - g0) Z_0` (no independence needed). The
   sentence "IG cells accept ARDinf only" is gone. (M1) and the loop sentence include `g^mu`. New spot check
   `B_ig.py` (8): ARD1 weights with a split initial damage, Monte Carlo, stop-loss order holds and the tail at the cap
   is below eps.
2. `beta*` over the terms with a positive mean: kept, text states it.
3. Gamma: an omitted `gamma_beta_0` / `gamma_beta_new` declares a known (constant) initial or replacement damage,
   which does not enter `beta*` (model bullets and the paragraph after Eq. `eq:gm_betastar`). Added one sentence for
   the corner case: if no term has a rate, the damage is a known constant and the row is `mu_k <= tau` (safe:
   `D_k = E D_k <= mu_k <= tau`).
4. Linear rows (Markov, Chernoff, IG, Gamma): "built in its exact form, and a value of `reliability_impl` at the cell
   is ignored", in the IG, Gamma and rainflow texts.
5. Citations of `sec:consistency_checks` added where the text relies on the new checks: Bernstein tangent point;
   initial values, starting anchors and replacement values in the natural range (rainflow rows; Gaussian `V_max`
   remark); replacement states satisfy the row (Chernoff); Jensen on `cgf`, `cgf_0`, `cgf_new`.
6. Gamma for `eps >= 1/2`: definition kept.
7. **Writer D's question 28: confirmed.** Lemma `lem:gm_class` reads, under ARD1 with `g0 < mu_0`, the initial
   damage as two independent terms of rate `gamma_beta_0` with means `g0` and `mu_0 - g0` (their sum has the law
   `Gamma(beta_0 mu_0, beta_0)`), and, when `gamma_beta_0` is omitted, as the two constants `g0` and `mu_0 - g0`. The
   first part keeps its weight at a repair, the second is multiplied by `1 - rho`. This is exactly D's treatment.
   (For IG the split is `g0 Z_0` and `(mu_0 - g0) Z_0` with one common `Z_0`, since IG needs no independence; a
   validator that draws the two IG parts independently is also covered, by the same proposition.)

Build after the corrections: 0 errors, 0 overfull boxes; only other writers' labels undefined.

Symbol fixes (coordinator's second message, 26 Oct 2026, found by writer A):
- The quantity capped by a quadratic row is `\chi` (as in writer A's `eq:a_quad`, `tab:a_quad`), not `Q`; `Q` is
  only the regularised incomplete gamma function (Gamma section). Changed in the rainflow "Linear forms" sentence and
  table.
- `\kappa_G = \Phi^{-1}(1-\varepsilon)` replaces `z` in the Gaussian and Wiener sections (row `eq:g_row`, `f`,
  `V_max = tau^2/kappa_G^2`, the eps < 0.5 remark); `z` is left only as the integration variable inside the IG
  derivation of `E[Z; Z > q] = F(1/q)`.
- IG symbol `C` (the conditional tail mean) kept: no other plain `C` in any fragment (costs are subscripted, `C_M`,
  `C_R`, `C_rep`, `C_D`; the Gamma class is the calligraphic `\mathcal{C}`). Writer A's `\bar\mu_{side}` (side-row
  bound) and the IG cap `\bar\mu` differ by the subscript; no change.
Rebuilt: 0 errors, 0 overfull boxes.

## Questions for Claude (all answered on 26 Oct 2026; kept for the record)

1. **IG and ARD1 (possible conflict).** The decision keeps IG cells at ARDinf only, for the reason "under ARD1 the
   state leaves the IG family". With the CVaR cap the proof never uses the family: Proposition `prop:ig_order` holds
   for any weights in (0, 1] and any dependence, and ARD1 gives `D+ = G + (1-rho) Delta`, again of that form. So
   ARD1 would be safe at IG cells (with an anchor `g^mu`). The spec follows the decision: ARDinf only, justified only
   by "the scaling keeps the law of each term". Should ARD1 be allowed?
2. **Gamma: known initial or replacement damage.** The brief says "a known initial damage being a constant". The
   spec's class covers constants, and the sentence "A known initial damage is a constant, and the same row covers
   it" is in the model paragraph. But `main` requires `gamma_beta_0` whenever `mu_0 > 0` (config.py:540-543), so no
   input can declare a known damage. Writer C must decide whether `gamma_beta_0` / `gamma_beta_new` may be omitted
   for a known damage (then it does not enter `beta*`).
3. **Gamma `beta*`: refinement.** The spec takes the minimum over the rates of terms that carry damage: missions with
   `mu_inc > 0`, `beta_0` only if `mu_0 > 0`, `beta_new` only if `mu_new > 0` (Eq. `eq:gm_betastar`). The decision
   says "the smallest Gamma rate among its missions (both phases), its initial state and its replacement state".
   A rate of a zero term is irrelevant to safety, and including it can only lower the cap. Please confirm.
4. **`reliability_impl` at IG and Gamma cells.** The decision says Markov and Chernoff accept only `exact`. The IG and
   Gamma rows are linear too; the spec says "the forms tangent and pwl do not apply to it", without saying whether
   such a value at an IG or Gamma cell is an error or ignored. Writer C's checks need one rule.
5. **Bernstein tangent point.** With `b > 0`, a tangent point `tangent_ref*tau >= tau - c` gives an empty Bernstein
   row (proved in the spec, checked in `B_rainflow.py` (7)). The decided check is only `0 <= tangent_ref < 1`. An
   extra check `tangent_ref*tau < tau - c` at Bernstein cells (writer C) would turn a silent infeasibility into an
   error.
6. **Big-M validity needs in-range initial and replacement values.** With `M_s` the natural range, the replacement
   row `s_k >= s_new - M_s (1 - x^r_k)` forces `s_k >= s_new - M_s > 0` on every day if `s_new > M_s`; the same holds
   for the starting value on day 1. Examples: `K_new = s mu_new > s tau - kappa`, `R_new > 2 tau^2/kappa`,
   `v_new > V_max`. Such a replacement can never satisfy the reliability row anyway. Suggest an input check (writer
   C) or a sentence in `sec:encoding` (writer A): every replacement value satisfies the cell's reliability row. For
   Chernoff with the default `K_new = s mu_new`, this is `s (tau - mu_new) >= ln(1/eps)`; the decided check
   `s > ln(1/eps)/tau` covers only `mu_new = 0`.
7. **Jensen checks for the Chernoff inputs (writer C).** `cgf >= s_chernoff * mu` at every increment,
   `cgf_0 >= s mu_0`, `cgf_new >= s mu_new` are necessary conditions (Jensen); a smaller value is an inconsistent
   input that makes the row optimistic.
8. **Writer A's domination argument.** For `K` the true statistic satisfies the day map with `<=`, not `=`
   (Proposition `prop:rf_maps`). The induction is the same, but `sec:encoding` should phrase the premise as "the true
   statistic is at most the day map applied to the true statistics of the previous day".
9. **Wiener variance in the generic rows.** conventions.md says "the variance has no repair map". The spec gives it
   the repair map `psi = v_{k-1}` (factor `a = 1`, no anchor) under both laws, equivalently the single row
   `v_k >= v_{k-1} + delta v_k - V_max x^r_k` (Eq. `eq:w_vrow`). Writer A's generic rows must not drop the repair row
   for the Wiener variance without this replacement row; otherwise `v` is unconstrained below on repair days.
10. **Gamma cap for eps >= 1/2.** For `eps < 1/2` the set `{A < t : w(A) <= eps}` is a closed interval with
    `A_1 < t`. For larger `eps` its supremum can equal `t`; the spec avoids the issue by defining the row value as the
    lower end returned by the bisection (which satisfies `w(A_1) <= eps`). No action needed unless Claude prefers an
    input check `eps < 0.5` at Gamma cells too.

## Not done

- Nothing outstanding in writer B's scope.
