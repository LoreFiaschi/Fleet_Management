# `scoping.tex`: readability pass for a reader whose first language is not English

Status: done (26 Oct 2026). Every section of `scoping.tex` is revised, the changelog of the software specification is
the new last section (`sec:spec_changelog`), the final build is clean, and `scoping.pdf` is replaced. Next: nothing;
the points for the user are at the end of this file.

- **Task** (user, 26 Oct 2026): "go through the scoping file and verify (or modify it) to make it readable from a
  person whose second language is English". The whole file, the user's own text included. Readability only: no
  formula, number, citation, label or meaning changes; no technical content removed.
- **Addition**: the changelog of the software specification (`merge/spec_drafts/changelog_for_scoping.tex`) is the
  last section before the bibliography, label `sec:spec_changelog`, rewritten to the same rules (one list per version).
- **Backup of the file before this pass**: scratchpad `readability/scoping_before.tex` (session-scoped).
- **Build**: `sh merge/scoping_build.sh <outdir>` (Git Bash) plus a 4th `pdflatex` pass; outdir in the scratchpad.
- **Line endings**: a `sed -i` call in Git Bash turned the file to LF once; CRLF was restored (3292 CR, 3292 LF), as
  in the working copy before the pass (`core.autocrlf=true`).

## Builds

- Baseline (unchanged file): 0 errors, 0 undefined, 0 bibtex warnings, 0 `Float too large`, 60 pages, 3 overfull
  boxes (Gamma process paragraph; Proposition "Closure of the usual stochastic order"; header of Table
  `tab:cons_surrogates`), 88 underfull.
- Final (4 passes): 0 errors, 0 undefined citations or references, no "Label(s) may have changed", 0 bibtex
  warnings, 0 `Float too large`, 63 pages, **2 overfull boxes** (the last two of the baseline; the Gamma process one
  disappeared), 89 underfull. `scoping.pdf` replaced; no build file left in the repository.
- All 63 pages rendered at 60 dpi (scratchpad `readability/render`, sheets in `readability/sheets`) and looked at:
  no overflow, no misplaced float. Table 17 (`tab:ev_deg_models`) sits alone on a float page (p. 50, 211 words; its
  baseline page, p. 49, held 450 words). It is a normal `htbp` float page, left as it is.
- Check script (scratchpad `readability/tools/compare.py`): labels (126) and citations with locations (82) identical
  to the backup; references identical plus one new `\ref{def:st}`; every changed math segment is intended (see below).

## Sections done

- Abstract; section 1 (Overview, notation table, 1.1-1.5): about 60 sentences rewritten (split, active voice, terms).
  New definitions at first use: "intervention" (a repair or a replacement), "slack", "big-M row" and "indicator
  constraint", "chains" (carries over from one repetition to the next), "binding", "McCormick envelope". The three
  rejection cases of 1.3.3 became a sub-list.
- Section 2.1 (2.1.1 to 2.1.8): about 170 sentences rewritten, mostly long sentences with dashes and semicolons split
  (Route I and II, Bernstein curvature, the two refinements, the loop under Route I, random fatigue life). New
  definitions at first use: CDF, MGF, CGF (table caption), MILP, SOCP, PWL, SOS2, "reverse-convex", "exponentially
  tilted mean", rainflow counting, "functional", run-outs, IG and CP (table caption). Lists made of: the two steps of
  the random-N_f pipeline, the two encodings of the piecewise-linear tightening. Metaphors removed ("flatten",
  "creep", "slides", "punishing", "machinery", "knob", "resists").
- Sections 2.2 and 2.3: about 75 sentences rewritten. The facts (M1)-(M3) of the Gamma and inverse Gaussian models
  became lists, as the conditions (i)-(iii) of step 2 of the proof of Theorem "One term is the worst case".
  "Rejuvenated" -> "becomes younger" (Kijima I and II).
- Section 3 (electric vehicles): about 90 sentences rewritten. Acronyms defined in the captions of Tables 16 and 17
  (SEI, DOD, C-rate, IGBT, SiC, DC, PCB, BMS, HVAC, SOC, LAM) and in the text (SEI, B10, kerb mass, retreading,
  spall). Citations kept on the same claims; where a sentence was split, the citation stays with the claim it
  supports (Ibrahim2025 moved to the start of the description of that study).
- Sections 4 and 5: about 40 sentences rewritten; the big-M rows, the natural ranges and the relaxed rows became
  lists; YAML, "broadcasting" and the union bound glossed. The open questions (5.8) read well; one sentence split.
- Total: about 450 source lines (sentences) of the original rewritten into about 900 shorter ones.
- Changelog section: every fact of `changelog_for_scoping.tex` kept; one list item per change. Added at its start, as
  a fact from `merge/decisions.md` (lines 271-279, 374): versions 0.3-0.5 call $\text{ARD}_1$ the law that this
  document calls $\text{ARD}_\infty$, and $\text{ARA}_1$ the law $\text{ARD}_1$; their horizon has $2H$ days. The map
  $\Phi$ of version 0.5 is written $\Pi$ (Section `sec:perpetual`), with a note, because $\Phi$ is the normal CDF here.
  The broken `\texttt` (a tab character) of the draft is fixed; the macros `\ARDinf`, `\ARDone` became
  `\text{ARD}_\infty`, `\text{ARD}_1`.

## Intended changes inside math (from the check script)

- "iff" -> "if and only if" inside `\text{}` of Eq. `eq:miner_invmom`; final comma -> period in `eq:miner_routeI` and
  `eq:miner_loop_det` (the sentence now ends there).
- "big-$M$" -> "big-M" (6 times); "constant-$b$" -> "constant~$b_\ell$" (2); "$12.9\times$ gap" -> "a factor of $12.9$";
  "repeated block $=$" removed (footnote of `sec:loop_routeI` reworded).
- New inline math: $\Pi(y) \le y$ or $\Pi(y) \ge y$ (comparability of a scalar state); $u$ (gloss of "binding");
  $y$; $M_{D^-}(s)$ (sentence split); $Q$; the union bound $\Pr(A \cup B) \le \Pr(A) + \Pr(B)$.

## Terms unified (old wording -> new wording)

- "degradation state", "degradation" (for the quantity $D$) -> "damage"; the overview says once that damage is also
  called the degradation level. "Degradation model" stays the name of a model family.
- "damage regularisation (cost, variable, penalty, coefficient)" -> "damage penalty" ("a regularisation term" said
  once); also in the notation tables.
- "time step", "step", "at time k" -> "day" (the overview says once that days are the time steps).
- "maintenance policy", "repair model" -> "repair law"; "maintenance days" -> "depot days"; "reduced only by
  maintenance" -> "only an intervention reduces it"; "full replacement" -> "replacement".
- "repeated block" -> "operating phase"; "period", "cycle", "concatenation" (one repetition of the operating phase)
  -> "repetition".
- "indefinitely" -> "without end" (as "endless schedule").
- "starting value" (of $K$, of the statistics on day 0) -> "initial value" ("starting anchor" stays, it is an input
  name).
- "per-step" -> "per-day"; "big-$M$" -> "big-M"; "price" (of a construction) -> "cost"; "verbatim" -> "without
  change"; "iff" -> "if and only if"; "e.g.", "i.e.", "cf.", "via", "suf." -> "for example", "that is", "see", "by",
  "sufficient".

## Sentences left unchanged because the meaning was unclear (line numbers of the final file)

- Line 283: "... ensures that the operating phase does not use more reliability margin than the transitory phase
  created." Only "consume" -> "use"; what the transitory phase "creates" is unclear.
- Line 2488: "It removes the fraction $\rho$ of the degradation accumulated since the last maintenance." It describes
  the law of Mercier and Castro (2019), so "maintenance" was not changed to "intervention" (which also covers a
  replacement).
- Line 2854: "Bayerer (CIPS08) model": the name is not glossed, because the source was not read for it.
- Line 1048: the header "Cons.?" of Table `tab:cons_surrogates` is unchanged (baseline overfull box; changing it
  would change the column layout).
- Lines 2595, 2881 and 2945: the to-do items inside the document ("Remaining items for this section", "Provide the
  details to run an actual simulation of the system") are content of the user and stay; only the grammar of one item
  of section 4 was fixed ("Find out what can be modelled exactly and what cannot").
