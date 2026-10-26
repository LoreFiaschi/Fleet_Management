# Documents of the merged package: plan

Status: proposal (26 Oct 2026). The user chose the change markers and the archive; the plan as a whole waits for the
user's agreement. No document is edited yet.

- **Purpose.** Where each decision of [`decisions.md`](decisions.md) is written in the documents, in which order, and
  who writes it.
- **Documents in scope**: `spec/spec.tex` with `spec/references.bib`; `scoping.tex` with `scoping_bib.bib`; `README.md`;
  `CLAUDE.md`; `main`'s other Markdown files; the to-do list `Things to change.txt`.
- **Out of scope here**: the code. Its plan comes after the user agrees with this one.

## Order

| Step | Work | When | Reason |
|---|---|---|---|
| 1 | `spec/spec.tex`, new version | before any code | item "Source of truth": the spec is written first, and the code follows it |
| 2 | Code | after step 1 | it implements the spec |
| 3 | Alignment check of code and README against the spec | after step 2 | the `spec-alignment` skill: two reviewer agents, one merged list of differences, approval by the user, fixes |
| 4 | `README.md`, `CLAUDE.md`, `main`'s other Markdown files | after step 2 | they describe the code as built |
| 5 | `scoping.tex` | after step 1; may run during step 2 | it copies the settled formulation, not the code |
| 6 | Clean-up | last | `ASSESSMENT.md` removed; the merge line leaves `Things to change.txt`; `merge/` and `lore-vs-main.md` move to an archive folder |

- The to-do line on the merge in `Things to change.txt` states this order and points to this plan (26 Oct 2026).

## Step 1. `spec/spec.tex`

- **Base**: `lore`'s version 0.5. `main` never changed the spec after the common ancestor (item "Documents").
- **Procedure**: the user's `spec-creator` skill, update mode (its phase 3): read the whole spec; list the cascading
  effects; keep the untouched sections and their style; raise the version; add a changelog entry; run its mini panel
  review, because the changes are substantial.
- **Version 1.0.** The skill's rule: an architectural change raises the major version. The merged package uses `main`'s
  code base (input parsing in `config.py`, a registry of cell builders, sweeps, validators, dashboards).
- **No change markers** (user, 26 Oct 2026, answer menu): version 1.0 adds no `%% --- CHANGE ...` comments, and the 133
  existing ones are removed. The changelog and git history keep the record.
- **Structure.** Each rule that holds for every model is stated once, in a common subsection, before the five model
  sections. Today the spec repeats the repair laws and the loop constraint in each model section. The model sections
  keep only what is specific: tracked state, day map, reliability row, model notes.

### Where each decision goes

| Section of the spec (on `lore`) | Change | Items of `decisions.md` |
|---|---|---|
| Project Overview & Goals | scope of the merged package; the spec is authoritative for the formulation and the tools; five models | Source of truth; Code base of the merge; Set of degradation models |
| Index Sets and Notation | `H1`, `H2`, `T = H1 + H2` replace `2H`; the two phases (transitory, operating); names ARD∞ and ARD1 | Meaning of `H: [a, b]`; Names and set of repair laws |
| Input Specification (six subsections) | the key table; cost shapes scalar, `(L,)`, `(F, L)`; `epsilon` and `tangent_ref` per cell; starting values of the anchors; replacement inputs; Bernstein support inputs; the flag big-M or indicator; run options and defaults; `penalty_type`; objective modes; warm-start option; Gamma inputs | Input key names, shapes and axis order; Cost shapes; Reliability level; Linear forms; Starting value of the anchor; Replacement; Rainflow tail bounds; Encoding; Solver defaults; Warm starts; Gamma formulation |
| Consistency Checks | one list per model, with the values decided; errors for old and unknown keys; `null` cells | Input checks; Input key names; Reliability level; Cost shapes; Replacement |
| Decision Variables | `x` of shape `(F, M+1, T)`; `u` only when `C_D > 0`; `z` only at cells with `C_R > 0`, on repair days | General rules (no useless variables or rows); Repair cost on replacement days |
| Objective Function | `C_M·Σ x_i0k` (per depot day); cost shapes; both penalty types; three objective modes | Depot days and `C_M`; Cost shapes |
| Constraints: new common subsection "Repair laws" | ARD∞ and ARD1 with their citations; the anchor is the state after the last intervention; its starting value | Anchor; Starting value of the anchor; Names and set of repair laws |
| Constraints: new common subsection "Encoding of the dynamics" | one-sided rows in big-M form (M = natural range) and indicator form; the safety argument for every model and both laws; reported states and costs are upper bounds | Encoding of the dynamics |
| Constraints: new common subsection "Replacement" | availability; state after replacement for every model and bound; starting values of `R` and `K`; `replacement_as_new` | Replacement |
| Repair Cost Variable | defined on repair days only, for every model | Repair cost on replacement days |
| Loop Constraints | one rule: the full tracked state, anchors included, `state[T-1] <= state[H1-1]`; the repeatability argument for `H1 ≠ H2`; the per-model loop subsections go, model notes stay (Wiener needs a replacement) | States covered by the loop constraint; Meaning of `H` |
| Constraints: new common subsection "Linear forms of the reliability rows" | `exact`, `tangent` (point `tangent_ref`, default 0.5), `pwl`; why each is safe (a tangent or chord below a convex function) | Linear forms |
| Gaussian | ARD1 with anchors for mean and variance; the remark that refuses this law goes (lore:spec/spec.tex:737-742) | Names and set of repair laws |
| Inverse Gaussian | the CVaR cap with the convex-order argument; the corrected density (lore:spec/spec.tex:906-907) | Inverse Gaussian reliability cap |
| Wiener | law names; anchors | Names and set of repair laws |
| Gamma | the sharp cap: tracked mean, `b*` per cell, class of true states, the theorem with its proof, the computation of `A_1` | Gamma formulation |
| Rainflow | five bounds (Markov, Cantelli, Hoeffding, Bernstein with its side row and support constant, Chernoff with `K_0 = s·mu_0` and both laws), each with its row, tracked quantities and validity argument | Rainflow tail bounds; Replacement |
| Damage Regularisation | both penalty types; not built when `C_D = 0` | Cost shapes; Input checks |
| Full Optimization Problem | rewritten to match the sections above | all formulation items |
| Output Specification | `null` in files, `NaN` in memory, no file for a failed solve, `model` holds the names, flat HDF5; "Solver Loop Behaviour" goes (replaced by the sweeps) | Output schema; Horizon sweep |
| new section "Horizon Sweeps" | the two sweep functions, ranking criteria (`J_op/H2`; projected cost over an evaluation horizon), proof rule, early stop, checkpoint, error capture, parallel option | Horizon sweep; Solver defaults |
| new section "Warm Starts" | `Start`, `VarHintVal` or both; the greedy heuristic for all five models | Warm starts |
| File & Folder Structure; API `plot_management`; Plot Layout | `main`'s layout after the merge; the consolidated plotter, the `D` mark, the field `model` | Code base; Plotting |
| new section "Validation Tools" | the rewritten validator and dashboards; the Gamma validators | Validation tools |
| Input File Schema | the key table; the HDF5 reader reads every key; a YAML example with mixed models and the new keys | Input key names; Output schema |
| Open Questions & TODOs | the anchor item is closed; only open design questions stay; work items stay in `Things to change.txt` (one to-do list per project) | Anchor |
| Changelog | one entry for version 1.0 | all |

- `spec/references.bib`: the DOI of Mercier and Castro (2019) becomes 10.1016/j.ejor.2018.06.020; new entries only for
  sources read in full that a new argument cites (for example Roosta-Khorasani and Székely 2015, for the Gamma section).

### Who writes it

- Four agents at most, in parallel. Each owns separate sections and writes them into its own fragment file, so no two
  agents edit `spec.tex` at the same time.

  | Agent | Model | Sections |
  |---|---|---|
  | Common formulation | Opus | notation, decision variables, objective, the four new common subsections, repair cost, loop, regularisation |
  | Model sections | Opus | Gaussian, inverse Gaussian, Wiener, Gamma, rainflow |
  | Inputs and outputs | Sonnet | input specification, consistency checks, input file schema, output specification |
  | Tools | Sonnet | sweeps, warm starts, file structure, plotting, validation tools |

- Each brief carries the relevant items of `decisions.md` and the analyses in `analysis/`, and the user's rules for
  briefs (status line, small verified edits, no sub-agents, no deletions).
- Claude then:
  - assembles the fragments into `spec.tex` and rewrites "Full Optimization Problem" to match;
  - ticks every "`spec/spec.tex`: ..." bullet of `decisions.md` against the new text;
  - builds the PDF at 0 errors, 0 undefined references, 0 bibtex warnings, and reads the rendered pages that changed;
  - runs the mini panel review of the skill.
- The user reads the spec before any code is written.

## Step 4. `README.md` and the other Markdown files

- **`README.md`**: rewritten for the merged package. Base: `main`'s outline (it documents the tools), plus `lore`'s
  sections "Status", "Running the tests" and "Further documentation".

  | Section | Content |
  |---|---|
  | Purpose and status | what the package does; `spec/spec.tex` is the source of truth |
  | Prerequisites, Installation | Python >= 3.10, a Gurobi licence, the extra `dashboard` |
  | Quick start | one solve and one plot |
  | API reference | `solve`, the plotter, the two sweeps, the validator: signatures as in the code |
  | Input file | a short key table; the full rules by section name in the spec |
  | Output | the result fields; the file formats |
  | Reliability rows | one line per model and bound, with the spec section that defines it |
  | Dashboards, Validation | how to start them and what they check |
  | Project structure; Running the tests | the layout; the pytest suite and the smoke test |
  | Further documentation | the spec and the scoping document |

  - No reference to a file outside the repository: `docs/PROJECT_LAYOUT.md`, `euler/README.md` and "equations under
    `spec/`" go (items "Documents", "Cluster jobs and experiments").
  - Written by one Sonnet agent from the code and the spec; step 3 checks it.
- **`CLAUDE.md`**: rewritten by Claude for the merged package: architecture, commands, the names ARD∞ and ARD1, the
  merge policy line. Tracked by git (item "Repository hygiene").
- **`examples/regression/RegressionREADME.md`, `GAMMA_DIAGNOSTICS.md`**: updated with the code they describe. The second
  describes the calibration, which leaves the solve path (user, 26 Oct 2026). It is read in full; the parts on the
  calibration go with that code, and the file goes too if nothing else remains.
- **`experiments/*/README.md`** (9 files on `main`): read one by one; references to the removed results go (item
  "Cluster jobs and experiments").

## Step 5. `scoping.tex`

- A research document, not a code manual: it states the formulation and places it in the literature. Only the passages
  that the decisions touch change. Line numbers on `lore`:

  | Section (lines) | Change | Items |
  |---|---|---|
  | Scheduling and maintenance variables (122-147) | the one-activity row `Σ_j x_ijk <= 1`; `y_ik` goes; `C_M` charges `x_i0k` | Depot days and `C_M` |
  | Damage dynamics (148-202) | `H1`, `H2`; the names ARD∞ and ARD1 | Meaning of `H`; Names of repair laws |
  | Stochastic loop constraint (203-217) | `state[T-1] <= state[H1-1]` on the full tracked state | States covered by the loop constraint |
  | Cost function (218-242); Abstract problem formulation (243-295) | cost terms, penalty types; no useless rows | Cost shapes; General rules |
  | Conservative surrogates (500-649); loop under the Route-I bounds (650-701); two refinements (702-747) | the five rainflow bounds; the Bernstein side row and support constant; the starting value of `K` | Rainflow tail bounds; Replacement |
  | Stochastic ordering within each family (865-993); loop constraint per model (994-1054) | inverse Gaussian: convex order and CVaR cap; Gamma: the sharp cap; one loop rule | Inverse Gaussian cap; Gamma formulation; Loop |
  | Reduction-based models; connection to the formulation (1116-1181) | ARD1 cited to Mercier and Castro (2019), eq. 4; ARD∞ named by analogy; Doyen and Gaudoin (2004) for ARA and ARI only; the anchor | Names of repair laws; Anchor |
  | Implementation details (1430-) | the merged package; the spec is the reference | Code base |

  - The row ranges come from the section headings; the agent confirms each passage when it reads the file.
- Written by one Opus agent, from the finished spec. Built at 0 / 0 / 0; the changed pages are rendered and read.

## Step 6. Clean-up

- `ASSESSMENT.md`: local copy to the Recycle Bin, removed from git (item "Documents").
- `Things to change.txt`: the merge line goes once the merge is done (the list shows the current state only).
- `merge/` and `lore-vs-main.md`: kept until the merge is done (user, 25 Oct 2026); then moved to an archive folder in
  the repository (user, 26 Oct 2026, answer menu).
