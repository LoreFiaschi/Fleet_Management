# `scoping.tex`: update for the merge (step 5)

Status: in progress (26 Oct 2026). Done: status file. In progress: reading `scoping.tex`, the decisions and the
fragments. Next: baseline build, then the edits passage by passage.

- **Task**: align `scoping.tex` (with `scoping_bib.bib`, `scoping.pdf`) with `decisions.md`, using the settled text of
  `spec_drafts/fragments_A.tex` and `spec_drafts/fragments_B.tex`. Plan: `documents_plan.md`, "Step 5. `scoping.tex`".
- **Build folder** (scratchpad):
  `C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\scoping_build\`

## Checklist of passages

- [ ] Scheduling and maintenance variables: `Σ_j x_ijk <= 1`; `y_ik` removed; `C_M` on `x_i0k`; every other use of `y_{i`
- [ ] Damage dynamics: `H = [H_1, H_2]`, phases, `T = H_1 + H_2`; names ARD∞ and ARD1
- [ ] Stochastic loop constraint: `state[T] <= state[H_1]` on the full tracked state, anchors included
- [ ] Cost function and abstract problem: cost terms, both damage penalties, no variable with coefficient 0
- [ ] Conservative surrogates: five bounds; Bernstein side row and support constant; `K_0 = s·mu_0`; forms `exact`,
      `tangent`, `pwl`; loop under the bounds; the two refinements
- [ ] Stochastic ordering per family: inverse Gaussian (convex order, CVaR cap, both laws); Gamma (sharp cap); one loop rule
- [ ] Reduction-based models: ARD1 to Mercier and Castro (2019) §3.1 eq. (4); ARD∞ by analogy; Doyen and Gaudoin for ARA
      and ARI only; the anchor; the ARD1 replacement proposition
- [ ] Implementation details: the merged package; `spec/spec.tex` as the reference
- [ ] Bibliography entries: Mercier and Castro (2019); Roosta-Khorasani and Székely (2015); Székely and Bakirov (2003)
- [ ] Full build 0 errors, 0 undefined, 0 bibtex warnings, no `Float too large`; changed pages rendered and read
- [ ] `scoping.pdf` replaced

## Baseline build (before any edit)

(to be filled)

## Changes (new line numbers, for the user's review)

(to be filled)

## Questions for Claude

(none yet)
