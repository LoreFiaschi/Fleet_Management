# Spec version 1.0: conventions shared by every writer

## Spec versus scoping (user, 26 Oct 2026; overrides anything below that conflicts)

The user: *"spec.tex must contain only exactly what is implemented (or going to be implemented). It is not verbose,
there are no long explanations, no theorems, etc. All these stuff must belong to scoping file."* Style model:
`merge/spec_drafts/style_reference_spec_v03.tex` (read its style, ignore its content).

- **Stays in the spec**: the codebase and its mathematical constructions. Inputs (tables: key, shape, default, cells,
  one-line meaning); consistency checks (the rule only, no reason); variables with the condition under which each is
  built; the objective (equations, one line per mode); every row in its forms (exact, big-M, indicator, tangent, pwl)
  with the condition under which it is built; the definitions that the rows use (repair laws as formulas, natural
  ranges, constants); every precomputed constant with the formula or algorithm that computes it (the inverse Gaussian
  cap, the Gamma cap `A_1` and `beta*`, the Bernstein constant, the variance ranges); outputs; tool behaviour; file
  layout; input schema; one input example. A consequence that an implementer or user must know may stay as one
  sentence, with a pointer: "(scoping document, section "...")".
- **Moves to the scoping document**: theorems, propositions, lemmas, corollaries and their proofs; derivations;
  worked examples and numerical illustrations; the reasons for a choice, a default or a check; comparisons between
  models, forms or encodings; discussions of conservatism, slack, acceptance, accuracy or speed; literature and
  citations; open research questions.
- **The spec cites nothing** from the literature. The scoping document holds the sources.
- **No information is lost.** Every block you remove goes into `merge/spec_drafts/moved_<X>.md`, so that the scoping
  writer can place it. One entry per block:
  `### M-<X>-<n>: <title>`; `- From:` section and label; `- Kind:` proof, statement, derivation, example, rationale,
  comparison, literature, remark, open question; `- Suggested destination:` a section of `scoping.tex` (list below);
  `- Spec keeps:` the one line left in the spec, or "nothing"; then the removed LaTeX, verbatim, in a fenced block.
- **Sections of `scoping.tex`** (destinations): Overview of the problem (Scheduling and maintenance variables, Damage
  dynamics, Stochastic loop constraint, Cost function, Abstract problem formulation); Literature review (Stochastic
  degradation and fatigue models: damage-accumulation processes, discrete-time formulation, S-N curve and
  Palmgren-Miner, conservative surrogates for the empirical damage-increment distribution, the loop constraint under
  the Route-I bounds, two refinements, random fatigue life); Stochastic orderings (within each degradation family;
  loop constraint for each degradation model); Maintenance strategies (taxonomy, virtual age models, reduction-based
  models, connection to the fleet management formulation); Focus on electric vehicles; Optimization problem for a
  fleet of electric vehicles; Implementation details.

Written by Claude on 26 Oct 2026, for the four writers of `spec/spec.tex` version 1.0. Every writer follows this
sheet. If a convention below is wrong or missing, do not invent another one: write the problem in your status file
and use the convention as written.

## Sources of the content

- **Decisions**: `merge/decisions.md`. Every "`spec/spec.tex`: ..." bullet of its items must be covered by exactly one
  writer (the brief says which items are yours). The decisions are final; do not argue them again.
- **Plan**: `merge/documents_plan.md`, section "Step 1", table "Where each decision goes".
- **The current spec**: `spec/spec.tex` (version 0.5, on the branch `lore`, in the working tree). Keep its text where it
  stays true; rewrite only what the decisions change. Keep its prose style and its house form: prose, numbered
  equations (`align`, `equation`), `longtable` with `booktabs`, `\emph{Remark (...)}` paragraphs, `\texttt{}` for keys.
- **`main`'s code**: read it with `git show main:<path>` from the repository root
  `C:\Users\utente\Documents\GitHub\Fleet_Management`. Never check out another branch; never modify the working tree
  outside `merge/spec_drafts/`.
- **Reports and analyses**: `merge/reports/A_core.md`, `B_models.md`, `C_tooling.md` cite `main`'s and `lore`'s code by
  file and line; `merge/analysis/*.md` hold the proofs for the inverse Gaussian and Gamma caps.

## Files

- Your LaTeX goes into `merge/spec_drafts/fragments_<X>.tex` (X = A, B, C or D). No preamble, no `\begin{document}`:
  the file starts at a `\section` or `\subsection` command. If your sections are not contiguous in the final order,
  separate the blocks with a line `% ==== BLOCK <n> ====` so that Claude can place them.
- Build check: from `merge/spec_drafts/`, run
  `pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build_<X> wrap_<X>.tex` twice (create
  `build_<X>/` first). Your fragment must build with 0 errors. References to labels owned by another writer
  (table below) are expected to be undefined in your build; your own labels must all resolve.
- Status and notes: `merge/spec_drafts/status_<X>.md`, with a `Status:` line at the top (done, in progress, next), the
  checklist of decision bullets you covered, and every question or inconsistency you found. Nothing of this goes into
  the `.tex` file.
- Check scripts: `merge/spec_drafts/checks/<X>_<name>.py`, outputs `checks/<X>_<name>_output.txt`. Standard library
  only (no numpy, scipy or mpmath on this computer); run with `python -I`.
- The preamble (`preamble.tex`) is fixed. It loads `amsmath`, `amssymb`, `mathtools`, `amsthm`, `booktabs`,
  `longtable`, `enumitem`, `listings`, `tikz`, `hyperref`. Do not add packages; if you need one, say so in your status
  file. It defines the environments `theorem`, `proposition`, `lemma`, `corollary`, `definition`, `proof`, and the
  macros `\ARDinf` (ARD∞), `\ARDone` (ARD1), `\inc`, `\new`, `\tr` (for superscripts `^{\inc}`, `^{\new}`, `^{\tr}`).

## Order of the final document and owners of the labels

| Section (level) | Label | Owner |
|---|---|---|
| Project Overview & Goals | `sec:overview` | A |
| Index Sets and Notation | `sec:notation` | A |
| Input Specification (with its subsections, the consistency checks and the run options) | `sec:inputs`; `sec:input_scalars`, `sec:input_increments`, `sec:run_options`, `sec:consistency_checks` | C |
| Decision Variables | `sec:variables` | A |
| Objective Function | `sec:objective` | A |
| Constraints | `sec:constraints` | A |
| -- Assignment Constraints | `sec:assignment` | A |
| -- Depot-Day Compatibility | `sec:depot` | A |
| -- Repair Laws | `sec:repair_laws` | A |
| -- Encoding of the Dynamics | `sec:encoding` | A |
| -- Replacement | `sec:replacement` | A |
| -- Linear Forms of the Reliability Rows | `sec:linear_forms` | A |
| -- Gaussian Model | `sec:dyn_gaussian` | B |
| -- Inverse Gaussian Model | `sec:dyn_ig` | B |
| -- Wiener Model | `sec:dyn_wiener` | B |
| -- Gamma Model | `sec:dyn_gamma` | B |
| -- Rainflow Model | `sec:dyn_rainflow` | B |
| -- Repair Cost Variable | `sec:repair_cost` | A |
| -- Damage Regularisation | `sec:damage_reg` | A |
| -- Loop Constraint | `sec:loop` | A |
| Full Optimization Problem | `sec:full_problem` | Claude |
| Output Specification | `sec:output` | C |
| Horizon Sweeps | `sec:sweeps` | D |
| Warm Starts | `sec:warm_starts` | D |
| File & Folder Structure (with the plotting API and layout) | `sec:files`; `sec:plotting` | D |
| Validation Tools | `sec:validation` | D |
| Input File Schema | `sec:schema` | C |
| Open Questions | `sec:open` | Claude |
| Changelog; References | -- | Claude |

- Cross-reference another writer only through the section labels above. Equation labels carry your prefix:
  `eq:a_...` (A), `eq:g_...`, `eq:ig_...`, `eq:w_...`, `eq:gm_...`, `eq:rf_...` (B), `eq:c_...` (C), `eq:d_...` (D).

## Notation (normative)

- **Indices.** Train `i ∈ {1,...,F}`; component `\ell ∈ {1,...,L}`; activity `j ∈ {0,1,...,M}`, with `j = 0` the
  depot day; day `k ∈ {1,...,T}`; index `k = 0` holds the initial state. A **cell** is a pair `(i, \ell)`.
- **Horizon.** Input `H = [H_1, H_2]`; `T = H_1 + H_2`. Days `1,...,H_1` form the **transitory phase**, days
  `H_1+1,...,T` the **operating phase**. A scalar `H` means `H_1 = H_2 = H`. Write `H_1`, `H_2`, `T` in math.
- **Per-cell parameters** carry the subscript `i\ell`: `\tau_{i\ell}`, `\varepsilon_{i\ell}`, `\rho_{i\ell}`,
  `C_{R,i\ell}`, `C_{\mathrm{rep},i\ell}`. Inside a model section the subscript may be dropped after one sentence
  that says so.
- **Increments.** `\mu^{\inc}_{i\ell jk}` is the mean increment of cell `(i,\ell)` when train `i` flies mission `j`
  on day `k`, after the phase lookup of `sec:input_increments` (index order `i, \ell, j, k` follows the input axis
  order `(F, L, M, H)`). Likewise `v^{\inc}` (variance), `b^{\inc}` (support width), `K^{\inc}` (cumulant generating
  function, CGF, at the cell's Chernoff parameter `s_{i\ell}`), and for Gamma the rate `\beta_{i\ell jk}`.
  Day aggregates: `\delta\mu_{i\ell k} = \sum_{j=1}^{M} x_{ijk}\,\mu^{\inc}_{i\ell jk}`, and `\delta v`,
  `\delta R` (sum of squared support widths `(b^{\inc})^2`; B confirms against `main`'s Hoeffding code), `\delta K`
  in the same way. A depot day and an idle day add nothing.
- **Binary variables.** `x_{ijk}` (assignment; `x_{i0k} = 1` is a depot day); `x^m_{i\ell k}` (repair);
  `x^r_{i\ell k}` (replacement).
- **Tracked states.** `\mu_{i\ell k}` (mean, every model); `v_{i\ell k}` (variance); `R_{i\ell k}` (Hoeffding budget);
  `K_{i\ell k}` (Chernoff CGF). Which model and bound tracks which state: B states it per model.
- **Anchors** (law ARD1 only): `g^{\mu}_{i\ell k}`, `g^{v}_{i\ell k}`, `g^{R}_{i\ell k}`, `g^{K}_{i\ell k}`: the value of
  the state just after the last intervention (repair or replacement). Starting anchors `g^{\mu}_{i\ell 0}` etc. come
  from the optional inputs `anchor_mu_0`, `anchor_v_0`, `anchor_R_0`, `anchor_K_0`; default: the initial value of the
  same state.
- **Initial and replacement values.** `\mu_{0,i\ell}`, `v_{0,i\ell}`, `R_{0,i\ell}` (default 0),
  `K_{0,i\ell}` (default `s_{i\ell}\,\mu_{0,i\ell}`); `\mu^{\new}_{i\ell}`, `v^{\new}_{i\ell}`, `R^{\new}_{i\ell}`
  (default 0), `K^{\new}_{i\ell}` (default `s_{i\ell}\,\mu^{\new}_{i\ell}`); support widths `b^{0}_{i\ell}`,
  `b^{\new}_{i\ell}` (default 0). Input names: the key table of `decisions.md`, item "Input key names".
- **Repair laws.** `\ARDinf`: `D^+ = (1-\rho)\,D^-`. `\ARDone`: `D^+ = D^- - \rho\,(D^- - D^{\mathrm{last}})`,
  `D^{\mathrm{last}}` the level just after the last intervention, and the initial level before the first one. Input
  key `repair_model`, values `ardinf` and `ard1`. Citations: `\ARDone` is
  `\cite[\S3.1, eq.~(4)]{mercier2019}`; `\ARDinf` is named by analogy with ARA∞ of `\cite[eq.~(20), p.~11]{doyen2004}`,
  and the text says that the name is an analogy (ARA∞ and ARA1 reduce a virtual age, not a degradation level;
  ARA1 is `\cite[eq.~(22), p.~12]{doyen2004}`; "memory" is defined on p.~8).
- **Laws per model** (decided): Gaussian `\ARDinf` and `\ARDone`; inverse Gaussian `\ARDinf` only; Wiener both, on
  the mean only; Gamma both; rainflow both, for every bound.
- **Encoding.** Input `encoding`: `bigm` (default) or `indicator`. Every dynamics row is one-sided (`\ge`) in both
  forms; replacement rows and anchor updates are equalities. The big-M constant of a state is its natural range
  (`\tau` for a mean or a mean anchor; the variance cap for a variance; B gives it for `R` and `K`).
- **Reliability forms.** Input `reliability_impl`: `exact`, `tangent`, `pwl`. The tangent point is
  `\mu_r = \texttt{tangent\_ref}\cdot\tau`, `tangent_ref` per cell in `[0, 1)`, default 0.5; `pwl` uses `pwl_points`
  segments (default 8). Markov and Chernoff rows are linear and accept `exact` only.
- **Derived constants.** `\Phi^{-1}(1-\varepsilon)`; `V_{\max} = \tau^2/(\Phi^{-1}(1-\varepsilon))^2`;
  `\kappa_C = \sqrt{(1-\varepsilon)/\varepsilon}`; `\kappa = \ln(1/\varepsilon)`; Bernstein support constant `b_{i\ell}`
  and `c_{i\ell} = \kappa\,b_{i\ell}/3`; Chernoff parameter `s_{i\ell}` (input `s_chernoff`).
  Gamma: `\beta^*_{i\ell}` (smallest rate of the cell), `t = \beta^*\tau`, the cap `A_1(t,\varepsilon)`, and
  `Q(a, x)` the regularised upper incomplete gamma function. Inverse Gaussian: `\eta_{i\ell}`,
  `Z \sim \mathrm{IG}(1, 1/\eta)`, its `(1-\varepsilon)`-quantile `q`, the cap `\bar\mu = \tau/\mathbb{E}[Z \mid Z > q]`.
- **Costs and objective.** `C_M` (scalar, per train per depot day); `C_{R,i\ell}`, `C_{\mathrm{rep},i\ell}`; `C_D \ge 0`;
  `z_{i\ell k}` (repair cost variable, cells with `C_R > 0`); `u` (damage regularisation, only when `C_D > 0`).
  Objective modes `total`, `operating_average`, `evaluation_total` (evaluation horizon `E`).

## Generic one-sided rows (normative; A states them in `sec:encoding`, B applies them per model)

For one tracked state `s` of a cell (a mean, a variance, `R` or `K`), with day increment `\delta s_k`, natural range
`M_s`, replacement value `s^{\new}`, and repair map `\psi`:

- Repair map. `\ARDinf`: `\psi = a\,s_{k-1}`. `\ARDone`: `\psi = a\,s_{k-1} + (1-a)\,g_{k-1}`. The factor `a` is
  `1-\rho` for a mean and for `K`, and `(1-\rho)^2` for a variance and for `R` (B derives each one in its model
  section). Wiener: the variance has no repair map (repairs act on the mean only).
- Big-M form (`encoding = "bigm"`, default):
  - no intervention: `s_k \ge s_{k-1} + \delta s_k - M_s\,(x^m_k + x^r_k)`;
  - repair: `s_k \ge \psi - M_s\,(1 - x^m_k)`;
  - replacement: `s^{\new} - M_s\,(1 - x^r_k) \le s_k \le s^{\new} + M_s\,(1 - x^r_k)`;
  - anchor (`\ARDone` only), with `e_k = x^m_k + x^r_k`: `|g_k - s_k| \le M_s\,(1 - e_k)` and
    `|g_k - g_{k-1}| \le M_s\,e_k`, each written as two linear rows.
- Indicator form (`encoding = "indicator"`): one extra binary `n_k` per cell and day with `n_k + x^m_k + x^r_k = 1`,
  built only in this form; `n_k = 1 \Rightarrow s_k \ge s_{k-1} + \delta s_k`; `x^m_k = 1 \Rightarrow s_k \ge \psi`;
  `x^r_k = 1 \Rightarrow s_k = s^{\new}`; anchors: `n_k = 1 \Rightarrow g_k = g_{k-1}`,
  `x^m_k = 1 \Rightarrow g_k = s_k`, `x^r_k = 1 \Rightarrow g_k = s_k`.
- Loop row (A, `sec:loop`): `s_T \le s_{H_1}` for every tracked state and every anchor of the cell.
- Natural ranges: `\tau` for a mean and its anchor; `V_{\max}` (Gaussian, Wiener) or the bound's variance cap
  (rainflow) for a variance; B gives the ranges of `R` and `K`.

## Points settled for the writers

- **Depot-day compatibility (A).** One row `x^m_{i\ell k} + x^r_{i\ell k} \le x_{i0k}` replaces the three rows of the
  current spec. It has the same binary solutions (at most one intervention, and only on a depot day) and a tighter
  linear relaxation. Claude reports this choice to the user.
- **Output key `idle` (C).** As on `main`: `idle_{i\ell k} = x_{i0k} - x^m_{i\ell k} - x^r_{i\ell k}`, a component that
  is in the depot without an intervention. An all-zero assignment row is an idle day of the train (no mission, no
  depot). C confirms both meanings against `main`'s code and states them.
- **Replacement (A and B).** A writes the common subsection `sec:replacement`: the option `allow_replacement`
  (default true), the replacement state from the input for every model (default 0), and the shortcut
  `replacement_as_new` (default false). B writes, in the rainflow section, the starting and replacement values of
  `R` and `K` with the Jensen argument and the optional inputs `R_0`, `R_new`, `cgf_0`, `cgf_new`.
- **Loop and repeatability (A).** The chaining argument, the slack-robustness argument and the warning against a loop
  row on a scalar bound are stated once, in `sec:encoding` and `sec:loop`, for every model and both laws. B cites
  them and adds only model notes (for example, Wiener needs a replacement in the operating phase).
- **Support width (B and C).** The rainflow `support` inputs keep `main`'s meaning: the width of the range of a term.
  For a width, `lore`'s check `v \le \mu\,(b - \mu)` is not a necessary condition (it assumes the range starts at
  0). The necessary condition for a range of width `w` is Popoviciu's inequality `v \le w^2/4`
  (`\mathrm{Var} X \le \mathbb{E}(X - c)^2 \le (w/2)^2`, `c` the midpoint of the range). Use it, and record this point
  in your status file for Claude.

## Rules for the text

- **The reader's first language is not English.** One idea per sentence; short sentences; common words; active
  voice, subject first; no idioms or metaphors; no phrasal verb where a plain verb exists; one term for one concept,
  never varied for style. Define every acronym and every technical term at first use. Give one example, or two at
  most, in a row; do not pad lists, examples or adjectives to three.
- **The document is standalone.** It describes the merged package as it is. Never write what an earlier version
  said, what changed, what was corrected, what is new, or what "now" holds. Check your text for the words
  *corrected*, *earlier*, *previously*, *was described as*, *is wrong*, *no longer*, *now*, *new* (as "recent"),
  *replaces*, *formerly*, *legacy*, *deprecated*. State a correction as a positive fact.
- **No change markers and no comments** in the `.tex` file, except the section separator lines
  `% ====...` that the current spec uses.
- **Normative words** as in the current spec: "must", "shall", "is required".
- **No useless variables or rows** (decided general rule): a variable and its rows are not built when they only feed
  an objective term whose coefficient is 0, or a bound or law that the cell does not use. Each section states the
  condition under which its rows are built.
- **Proofs.** Hypotheses ("Let ..."), the chain of equalities and inequalities, and the conclusion ("Therefore ...").
  Facts that follow from the definitions, the setup or an assumption enter the chain without proof. Reuse a result
  already proved in the spec by citing it. A statement never says "under Assumption X"; the proof names the
  assumption at the step that uses it. When a published result proves the statement, cite it before the proof, in
  the heading form "\begin{proof}[Proof of Proposition~\ref{...} {\cite[Theorem~4]{roosta2015}}]".
- **Citations.** Only `\cite{doyen2004}`, `\cite{mercier2019}`, `\cite{roosta2015}`, `\cite{szekely2003}`, always with a
  location in the optional argument. Locations read and checked: Mercier and Castro §3.1, eq.~(4) (ARD1);
  Doyen and Gaudoin p.~8 (memory), eq.~(20) p.~11 (ARA∞), eq.~(22) p.~12 (ARA1); Roosta-Khorasani and Székely
  Theorem~4, p.~11 (unimodality of every sum `\sum \lambda_i X_i`, `X_i ~ Gamma(\alpha_i, \beta_i)` independent,
  `\lambda_i \ge 0`); Székely and Bakirov, proof of Theorem~4, pp.~194--195 (the proof of that unimodality, through
  self-decomposable laws). The standard inequalities (Markov, Cantelli, Chernoff, Hoeffding, Bernstein, Jensen) are
  named and stated without a citation, as the current spec does. Do not cite any other source; if you need one,
  write the need in your status file.
- **Numbers and claims.** Every new proposition gets one numerical spot check and one boundary case in a check
  script, unless `merge/analysis/` already holds one (then name it in your status file).
