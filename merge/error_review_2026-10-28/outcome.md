# Error review of spec and scoping, 28 Oct 2026: outcome

Six reviewers (one per block) wrote the findings files of this folder. Claude verified each finding against
the text, recomputed the load-bearing numbers, and applied the fixes to `scoping.tex` and to the spec
fragments (`merge/spec_drafts/`). Both documents build at 0 errors, 0 undefined references, 0 bibtex
warnings, no overfull line (spec 51 pages, scoping 85 pages).

## Count

| Block | Findings | Applied | Not applied |
|---|---|---|---|
| R1 core formulation | 10 + 2 doubts | 10; doubt "Gaussian example" made direction-neutral | doubt "ARD_1 operator notation" |
| R2 models G, IG, W, Ga | 12 + 2 doubts | 12 | 2 doubts (numerical claims, no error) |
| R3a rainflow bounds | 11 + 3 doubts | 10; doubt "Bernstein side row" (c > 0) | F9 (no error: Remark rem:miner_tails states it); doubt "experiment sentence" |
| R3b Route-I loop, linear forms | 6 + 1 doubt | 6 | doubt "~10^-6" |
| R4 inputs, outputs, tools | 14 + 5 doubts | 14 (F4 extended to the definition sentence) | 5 doubts |
| R5 maintenance, EV | 17 + 5 doubts | 16 (F11 changed: the Coffin-Manson coefficient renamed N_0, the cited Eq. 4 left as is) | F12 (S_0 needs Sha 2022) |

## Choices made while fixing (the owner may reverse them)

- R1 F3: the initial damage need not meet the reliability level; the scoping document now matches the
  spec. Alternative: a new input check in the spec.
- R3a F1: `support_0` is the width of the range of the initial damage, `R_0` the sum of the squared widths
  of its independent terms (spec input table and A2 changed). This keeps the Popoviciu check. Alternative:
  the largest term width and no check.
- R4 F7: every non-null `bound_method` entry is checked, also at non-rainflow cells. Code not checked.
- R4 F10: at C_D = 0, `penalty_type` selects the formula of the reported u. Alternative: always inf_norm.
- R2 F9: `eta` is named "squared coefficient of variation"; "dispersion" stays for mu^2/lambda.

## Follow-up the same day: open points settled

Sources read in full (PDFs in `literature/`, git-ignored): Sha 2022 (HTML full text), Alavi 2024,
Wang and Blaabjerg 2014 and 2021, Wileman 2021, Wang, Liserre and Blaabjerg 2013. New checks, with fixed
seeds, in `merge/analysis/scripts/scoping_checks/` (README there) and
`merge/analysis/scripts/verify_gamma_sharp/` (`single_peak_check.py`, `above_t1_check.py`).

- Route-I tail experiment: rerun with five laws and exact tails; the bullets state its results; the
  self-contradicting Bernstein sentence and "the exponential bounds are much tighter" (false for
  Hoeffding) are replaced.
- Remark rem:miner_dichotomy: P(I) = 2.46e-7 (not ~1e-6) with fifty equal days; the reversal is at M = 0.87.
- Loop instance: holds under ARD_inf (now stated); margin sentence made precise (delta < 0.089 at U0 = 0.2).
- Brake paragraph: Sha 2022 models the disc, not the pad; S_0 is a dimensionless wear safety factor
  (Eq. 22); the unsupported energy sentence is replaced.
- Power modules: Alavi's alpha in Eq. 4 is negative (written -n); the B10 drop belongs to SKiM63 for a
  photovoltaic inverter; "changed substantially" and "must not be extrapolated" replaced by what Alavi
  supports; CIPS08 explained; Bayerer 2008 not readable, cited through Alavi Eq. 6 (bib note).
- Capacitors and PCBs: the lifetime model is a power law in the voltage, not the ripple current;
  "power-on hours", "a substantial part" and "effectively random" removed; the footnote pointing to a
  non-existent `inaccessible_papers.md` removed; every remaining claim cited.
- Gurobi indicator constraints: cited to the Gurobi reference manual (section on general constraints).
- Text-only fixes: three electrical sub-families (not four); the "exactly Gamma or IG" claim; bearings in
  both families; the natural-range check reason; the A_1 = A_hat claim stated as a grid check up to
  t = 1e5; the ARD_1 operator argument; the changelog's undefined V removed; the spec's `prepare` does
  not create the shared mean.

- Bayerer 2008 (PDF supplied by the user, read in full): the CIPS08 model is its Eq. (1), with T_j in deg C
  (Fig. 2); two exponent sets, Eq. (5) with the maximum and Eq. (6) with the minimum junction temperature
  (Sec. 3.4). Alavi's "values based on Bayerer's research" are set (5); its "EasyPACK fit" values equal
  set (6). The no-extrapolation statement is Bayerer Secs. 3.1 and 3.4, now cited; the voltage factor does
  not apply to traction modules (Sec. 3.1, remark), now stated. "V divided by 100" comes from Alavi only.

## Open

- The scoping footnotes point to `merge/analysis/scripts/scoping_checks`. When `merge/` is archived
  (step 6), move this folder and update the paths in the footnotes.

## Spec changes that the code must follow (step 3, alignment)

`R_0` definition; `bound_method` check on every cell; `penalty_type` at C_D = 0; the optional variance row
listed in the full problem. Naming only: `eta`, q-bar.
