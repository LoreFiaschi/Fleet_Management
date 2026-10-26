# `scoping.tex`: update for the merge (step 5)

Status: done (task 2, 26 Oct 2026). Task 1 (step 5 of the plan) is done, see below. Task 2: the 145 blocks moved out
of the spec (`spec_drafts/moved_A.md`, `moved_B.md`, `moved_C.md`, `moved_D.md`, `moved_Claude.md`) are integrated
into `scoping.tex`; the record is `merge/spec_scoping_map.md` (one row per block, the spec pointers, the facts that
the condensed spec lacks). Final build: 0 errors, 0 undefined, 0 bibtex warnings, no `Float too large`, 60 pages, the
3 overfull boxes of the baseline; `scoping.pdf` replaced. Every changed page rendered at 80 dpi and read. Next: nothing;
the questions for Claude are below.

- **Task 2 brief (coordinator, 26 Oct 2026)**: the spec holds only the codebase and its mathematical constructions;
  every theorem, proof, derivation, example, rationale, comparison, literature reference and open question goes to
  `scoping.tex`; each piece of information lives in exactly one document. Merge where the scoping already holds the
  content (full proofs replace my proof sketches: perpetual safety, Gamma one term). New subsections allowed (encoding
  of the dynamics; objective modes; reasons of the input checks; reasons behind the tools near "Implementation
  details"); open questions (M-Cl-1) near "Implementation details". Spec pointers to check: "Stochastic loop
  constraint", "Conservative surrogates for the empirical damage-increment distribution", "Stochastic orderings", "Loop
  constraint for each degradation model", "Cost function". Bib keys: mercier2019 -> MercierCastro2019, doyen2004 ->
  DoyenGaudoin2004, roosta2015 -> RoostaKhorasaniSzekely2015, szekely2003 -> SzekelyBakirov2003.
- **Answers to the six questions (26 Oct 2026)**: (1) the scoping states the objective modes (block from moved_A);
  (2) the one-sided encoding goes to the scoping (blocks from moved_A); (3) keep the bullet "Provide the details to run
  an actual simulation of the system"; (4) ARI paragraph untouched; (5) line 1714 as it is; (6) keep the notation-only
  edits.

- **Task**: align `scoping.tex` (with `scoping_bib.bib`, `scoping.pdf`) with `decisions.md`, using the settled text of
  `spec_drafts/fragments_A.tex` and `spec_drafts/fragments_B.tex`. Plan: `documents_plan.md`, "Step 5. `scoping.tex`".
- Coordinator's corrections received and applied (26 Oct 2026): (1) Bernstein tangent point needs
  `mu_r < sqrt(tau(tau - 2c))` (also the pwl first-segment point `tau/(2P)`); (2) one-sided rows never accept an unsafe
  schedule and may reject a safe one; the scoping makes no acceptance claim. The map of the operating phase is `\Pi`;
  the Gamma class is `\mathcal{G}(m)`. Reviewer changes found in the fragments and applied: repair weights in `[0, 1]`
  (a term of weight 0, when `rho = 1`, is dropped); the sharp cap is defined for `eps < 1/2`.

## Resume here

- Both tasks are done. To rebuild: `sh merge/scoping_build.sh <outdir>` (Git Bash; full cycle pdflatex, bibtex,
  pdflatex, pdflatex; `<outdir>` outside the repository), then one more `pdflatex -output-directory=<outdir>
  scoping.tex` from the repository root with `TEXINPUTS` set as in the script (a 3-pass build may end with "Label(s)
  may have changed"; the 4th pass is clean). Render with `pdftoppm -png -r 80`.
- If Claude sends corrections: edit `scoping.tex`, rebuild, render the changed pages, update the line numbers of
  `merge/spec_scoping_map.md` for the moved passages.

## Task 2 record (26 Oct 2026)

- New structure: 1.3 has four subsubsections (`sec:encoding`, `sec:perpetual`, `sec:acceptance`,
  `sec:loop_replacements`); 1.4 has paragraphs "Objective modes" and "Repair cost"; 2.1.4 has "Width of a range", "Day
  maps of the statistics", "Derivation and validity of the rows"; new 2.1.7 "Linear forms of the reliability rows"
  (`sec:linear_forms`); 2.2.1 has the full Gamma proofs; 2.2.2 has per-model facts (M1)-(M3) and a rainflow paragraph;
  section 5 has subsections 5.1-5.8 (encoding forms, reasons for the input checks, outputs, sweeps, warm starts,
  validation, package and inputs, open questions).
- The proof sketches of task 1 are replaced by the full proofs of the blocks (`thm:perpetual`, `thm:gamma_oneterm`).
- Spec pointer to correct: line 808 of `spec_assembled.tex` should name "Linear forms of the reliability rows".
- Spec facts not found in the condensed spec (to restore there): `L` must equal the component axis of `model`;
  `component_names` not empty after blanks are removed.

## Checklist of passages

- [x] Scheduling and maintenance variables: `Σ_j x_ijk <= 1`; `y_ik` removed; `C_M` on `x_i0k`; every other use of `y_{i`
- [x] Damage dynamics: `H = [H_1, H_2]`, phases, `T = H_1 + H_2`; names ARD∞ and ARD1
- [x] Stochastic loop constraint: `xi_T <= xi_{H_1}` on the full tracked state, anchors included
- [x] Cost function and abstract problem: cost terms, both damage penalties, no variable with coefficient 0
- [x] Conservative surrogates: five bounds; Bernstein side row and support constant; `K_0 = s·mu_0`; forms `exact`,
      `tangent`, `pwl`; loop under the bounds; the two refinements
- [x] Stochastic ordering per family: inverse Gaussian (convex order, CVaR cap, both laws); Gamma (sharp cap); one loop rule
- [x] Reduction-based models: ARD1 to Mercier and Castro (2019) §3.1 eq. (4); ARD∞ by analogy; Doyen and Gaudoin for ARA
      and ARI only; the anchor; the ARD1 replacement proposition
- [x] Implementation details: the merged package; `spec/spec.tex` as the reference
- [x] Bibliography entries: Mercier and Castro (2019); Roosta-Khorasani and Székely (2015); Székely and Bakirov (2003)
- [x] Full build 0 errors, 0 undefined, 0 bibtex warnings, no `Float too large` (last build, 26 Oct 2026)
- [ ] Changed pages rendered and read: all read on the previous build; pp. 20, 25-27 of the last build not yet read
- [x] `scoping.pdf` replaced (last build, 42 pages)

## Baseline build (before any edit)

- Full cycle on the unchanged file (26 Oct 2026): 0 errors, 0 undefined citations or references, 35 pages,
  0 `Float too large`, 3 overfull boxes (lines 344-348, 525-528, 549), 88 underfull boxes.
- **1 bibtex warning in the baseline**: `empty publisher in WhitmoreSchenkelberg1997`. Fixed: the entry is retyped
  `@article` with `journal = {Lifetime Data Analysis}` (vol. 3, pp. 27-45, DOI 10.1023/A:1009664101413, all already in
  the entry); no other field changed.
- The original `scoping.tex` equals `HEAD:scoping.tex` (commit da7685a), line endings aside.

## Last build (26 Oct 2026)

- 0 errors; 0 undefined citations or references; 0 bibtex warnings; 0 `Float too large`; 42 pages; 3 overfull boxes,
  the same three as the baseline (untouched text: Gamma process paragraph, Proposition 2.1 statement, Table 5 header);
  88 underfull boxes, as the baseline.

## Changes of task 1 (line numbers as after task 1; task 2 moved them, see `merge/spec_scoping_map.md`)

| Passage | New lines | Change |
|---|---|---|
| Overview | 59-71 | horizon `T = H_1 + H_2`, transitory and operating phases; at most one activity, idle member; depot day; loop described on the certifying statistics |
| Notation table | 87-121 | `k` up to `T`; rows `H_1, H_2` and `T`; `C_M` on `x_{i0k}`; row `xi_{ilk}` (tracked state); row `y_ik` removed |
| Scheduling and maintenance variables | 131-153 | `Σ_j x_ijk <= 1`; idle day = row of zeros; `C_M` prices depot versus idle; one row `x^m + x^r <= x_{i0k}` (the spec's form) |
| Damage dynamics | 156-224 | label; "depot day" wording; ARD∞ and ARD1 named and expanded; paragraph "Horizon and phases" (endless schedule) |
| Stochastic loop constraint | 226-288 | probability comparison kept as eq. `loop_prob` and shown insufficient; tracked state; loop row eq. `loop` = `xi_T <= xi_{H_1}`; Proposition "Perpetual safety" (proof sketch); paragraph "Why the anchors enter the loop" (example from fragments A) |
| Cost function | 290-326 | label; `C_M` on `x_{i0k}`, once per member and depot day; two penalty types; rule "no variable for a zero coefficient" |
| Abstract problem | 332-376 | no `y`; `<= 1`; `x^m + x^r <= x_{i0k}`; loop row on `xi`; `u` not built when `C_D = 0` |
| Conservative surrogates | 627; 703-713; 731; 734-783 | caption without "loop compares survival functions"; Bernstein support constant `max(b^start, b^new, max_j b_lj)` and explicit side row; Route II loop "in the absence of repairs"; new summary; paragraphs "The five bounds of the formulation" (Table `tab:five_bounds`), "The Bernstein side row", "Starting and replacement values of the Chernoff statistic" (Jensen, example) |
| Loop under the Route-I bounds | 787-850 | notation `U^(n)`; recursions with anchors; cumulant loop valid under repair by the convexity bound `K+ <= (1-rho)K (+ rho g^K)`; new summary |
| Two refinements | 856-861; 900-918 | pwl is part of the formulation, schedule-dependent support is not; constant `b_l` with reset supports; paragraph "Linear forms in the formulation" (exact, tangent, pwl; Bernstein tangent condition) |
| Random fatigue life (Miner) | 978-982 | notation only: `M_T <= M_{H_1}`, map `\Pi`, `k = 1..T`, reference to `loop_prob` |
| Remark on stochastic orderings; ordering intro | 1031-1039 | refers to `loop_prob`; "end of the transitory phase" |
| Implications; IG and Gamma caps | 1164-1316 | closure not used; Definition icx; IG: Proposition stop-loss order, Theorem CVaR cap, closed form, largest-cap remark, counterexample to `tau/q`; Gamma: Definition class `G(m)`, Lemma, Theorem one term (proof sketch, citations), sharp cap, Proposition safe, threshold `t_1`, example, `r*`, choice of `beta*` |
| Loop per model | 1318-1411 | label; one rule; Table `tab:loop_per_model`; Gamma (parameter-space repair unsafe, S3 example); Gaussian (notation, both laws exact, row); new Wiener paragraph; IG (no probability comparison used); upper-bounds paragraph end |
| Reduction-based models | 1476-1503 | DG for ARA and ARI only (p. 8; eq. 20 p. 11; eq. 22 p. 12); ARA paragraph; ARD paragraph: ARD1 to Mercier and Castro §3.1 eq. (4), reference level, ARD∞ named by analogy |
| Connection to the formulation | 1515-1584 | label; ARD∞ default; both laws for every model; paragraphs "Anchors", "Repair maps"; Proposition "Replacement under ARD1" with proof and example; Miner bullet "since the last intervention" |
| EV specialisation | 1824; 1837-1843 | `C_M` per depot day; sums to `T` over `x_{i0k}`; "depot days" |
| Implementation details | 1853-1862 | the package, Gurobi, scope; `spec/spec.tex` as the reference; first to-do bullet removed (see question 3) |
| `scoping_bib.bib` | — | new `MercierCastro2019`, `RoostaKhorasaniSzekely2015`, `SzekelyBakirov2003`; `note` on `DoyenGaudoin2004` (preprint pagination) and on the two preprint-read entries; `WhitmoreSchenkelberg1997` retyped |

## Decisions made while writing (held here so a successor knows them)

- Label `eq:loop` now names the tracked-state row; the probability comparison is `eq:loop_prob`. Removed labels:
  `eq:loop_gamma`, `eq:loop_ig` (no remaining reference). New labels: `sec:damage_dynamics`, `sec:loop_constraint`,
  `sec:cost_function`, `sec:loop_per_model`, `sec:repair_connection`, `prop:perpetual`, `tab:five_bounds`,
  `def:icx`, `prop:ig_order`, `thm:ig_cap`, `def:gamma_class`, `lem:gamma_class`, `thm:gamma_oneterm`, `eq:gamma_cap`,
  `prop:gamma_safe`, `tab:loop_per_model`, `prop:ard1_replacement`.
- Symbols chosen to avoid clashes with the document: day map `\pi_k` (the document uses `\phi` for the aggregation
  function); map of the operating phase `\Pi` (the document used `F`, which is the fleet size); CVaR tail mean `\bar q`
  (not `C`, which names costs); in the Gamma proof sketch the rate function is `\omega` and `\gamma = 1/c`.
- Notation-only edits outside the plan's table, made because the decision on `H` changes the meaning of the symbol: the
  Miner section (lines 978-982), the remark before "Stochastic ordering within each family" and its first sentence, the
  EV objective. Listed above for review.
- The theorem "one term is the worst case" carries no citation of its own: per `decisions.md`, no source read proves it;
  only the unimodality step is cited (Roosta-Khorasani and Székely 2015, Theorem 4, p. 11 of arXiv 1601.04731v1; proof in
  Székely and Bakirov 2003, pp. 194-195).
- No other new citation was added; existing citations in untouched text are unchanged.

## Questions for Claude

Task 1 questions 1-6 are answered (see the top of this file). Open after task 2:

1. **Moschopoulos series** (5.6, exact Gamma validator): named without a citation, because no source of it was read and
   the brief allows no new citation. Cite it once a source is read?
2. **Spec pointer** at line 808 of `spec_assembled.tex`: should name "Linear forms of the reliability rows" (details in
   `merge/spec_scoping_map.md`).
3. **Two spec facts** not found in the condensed spec (`L` equals the component axis of `model`; `component_names` not
   empty after blanks are removed): restore them in the spec?
