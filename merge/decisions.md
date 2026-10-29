# Merge of `lore` and `main`: decisions

Status: all 30 divergences decided (26 Oct 2026).

**Resume here** (26 Oct 2026; no background task runs):
- **Spec (an SDD, Software Design Document) and scoping: done, waiting for the user's reading** (rule in `CLAUDE.md`).
  Error review of both documents (28 Oct 2026): 70 findings, 68 fixed, then the open points settled from the
  sources and new checks; choices, open items and the spec changes that the code must follow are in
  `error_review_2026-10-28/outcome.md`. Step 6 (clean-up): the scoping footnotes cite
  `merge/analysis/scripts/scoping_checks`; move that folder with `merge/` and update the paths.
  `spec/spec.tex` (51 pages; 0 errors, 0 warnings; PDF `spec_drafts/build_final/spec.pdf`) holds only the codebase
  and its mathematical constructions, in this order: overview, architecture, notation, inputs, variables, objective,
  constraints (explicit big-M rows per model), full problem, outputs, sweeps, warm starts, plotting, validation, input
  schema. From the output rules on, every tool follows one template (user's choice, 28 Oct 2026): signature,
  arguments table, numbered steps, errors table, report-key table; definitions as a bold-term list. `scoping.tex` (85 pages; 0 errors, 0 undefined references, 0 bibtex warnings, no overfull line; `scoping.pdf`)
  holds every proof, example,
  reason and source, the open questions and the spec's changelog (`sec:spec_changelog`); its whole text passed a
  readability pass for a reader whose second language is English (record: `scoping_readability_status.md`; the copy
  before that pass: `scoping_before_readability_pass.tex`), then a structure pass (user's choice, 28 Oct 2026):
  summary tables for parallel items (`tab:objective_modes`, `tab:loop_routeI_bounds`, `tab:miner_interpretations`,
  `tab:reduction_classes`), long paragraphs split into lists with bold lead-ins; statements and proofs verbatim. Its
  "Remaining items" lists moved to `Things to change.txt`. `tab:cons_surrogates` has a Hoeffding row citing
  Hoeffding (1963), Theorem 2 (read in full, 28 Oct 2026; PDF in `literature/`, supplied by the user). Second
  structure pass (user's choices, 28 Oct 2026): every proof and every prose derivation is in the appendix "Proofs and
  derivations" (17 proofs, 35 derivations; anchors `prf:<statement label>` and `drv:<name>`; each main-text pointer
  gives the appendix number and page); every display is numbered; long inline math is displayed; no prose paragraph
  over about 80 words outside footnotes. Two corrections made in that pass: for the Gumbel-minimum scatter, e^mu is
  the characteristic life, not the median; the variance ratio of the two Miner interpretations is a direct
  computation, and Cauchy--Schwarz only bounds it by the number of days. Moved blocks and their places: `spec_scoping_map.md`. Spec
  sources: the fragments and `spec_drafts/claude_*.tex`, joined by `spec_drafts/assemble.py`; copy
  `spec_assembled.tex` to `spec/spec.tex` with CRLF. The spec and scoping work is committed on `lore` (68ec120).
- **Step 2, code:** plan in [`code_plan.md`](code_plan.md). WP0 to WP3 done on the branch `merge/lore-into-main`
  (worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge`, commits ed6694c and 5f29660; list in
  `wp1_removals.md`). WP3 merged `lore` at 68ec120. Conflicts were resolved toward `main`'s code: `README.md` and
  `solver.py` from `main`; `lore`'s plotter removed (item "Plotting"). `lore`'s builders are in
  `src/fleet_management/models/` and `src/fleet_management/maintenance/`, as sources for WP6; no module of `main`
  imports them. **WP4 and WP5 done (29 Oct 2026), commit 5c8de19**: Opus plans, Sonnet executors, Opus reviews
  (`wp4_wp5_interface.md`; `wp4_plan.md` and `wp5_plan.md`, each with its review in section 7). 73 tests of WP4 pass
  here; `tests/test_base.py` needs Gurobi. The local tools of both packages (test runner, gurobipy stub, linear
  emulator of gurobipy) are copied in `merge/wp45_work/` (final versions, after the reviews).
  **WP6 done (29 Oct 2026), commit 0421e80**: five builders, three Opus plans and reviews (`wp6_interface.md`,
  `wp6_plan_giw.md`, `wp6_plan_gamma.md`, `wp6_plan_rainflow.md`); local tools copied to `merge/wp45_work/wp6/`. The spec
  on `lore` holds the reviewers' sentences (cap methods, package layout). **WP7 done (29 Oct 2026), commit 7e35db4**: sweeps, warm starts, plots,
  validators, public API; four Opus plans and reviews (`wp7_interface.md`, `wp7_plan_*.md`); local tools in
  `merge/wp45_work/wp7/`. The reviewers' spec sentences are in the spec on `lore` (54 pages, clean build;
  record `spec_wp7_insertions.md`). The reasons the WP4 to WP7 reviewers moved out of the spec are in `scoping.tex`
  (92 pages, clean build; record `scoping_notes_wp4_wp7.md`, 29 Oct 2026).
  Next: WP8 (tests), which needs `lore`'s `test/` and `input/` folders and the licensed machine.
- **Pending actions of the user:** read the spec and the scoping document; copy `lore`'s `test/` and `input/` folders
  into the repository (item "Tests and test inputs").
- Documents plan: [`documents_plan.md`](documents_plan.md), agreed by the user on 26 Oct 2026.

- **Purpose.** This file records the user's choice for each divergence between the branches `lore` and `main`.
  The merge writes these choices into `spec/spec.tex` first, then into the code, then into `scoping.tex` and the README
  ([`documents_plan.md`](documents_plan.md)).
- **Branches compared**: `lore` at 651aa71, `main` at 978ea9a. Common ancestor: 462ac1d (25 Jun 2026).
- **Evidence.** The facts behind each divergence are in [`../lore-vs-main.md`](../lore-vs-main.md) and in four
  reports written on 30 Sep 2026:
  - [`reports/A_core.md`](reports/A_core.md): core pipeline and formulation;
  - [`reports/B_models.md`](reports/B_models.md): degradation models;
  - [`reports/C_tooling.md`](reports/C_tooling.md): plotting, validation, tests, cluster jobs, experiments, data files;
  - [`reports/D_documents.md`](reports/D_documents.md): specification, scoping document, READMEs, project files.
- **Erratum to the reports.** 42 tracked files on `main` match `main`'s own `.gitignore` (checked 30 Sep 2026).
  `C_tooling.md` says 43.
- **Analyses** (26 Oct 2026): [`analysis/`](analysis/) holds the notes on the inverse Gaussian and Gamma models and
  the Gamma literature; [`analysis/scripts/`](analysis/scripts/) holds their standard-library scripts and outputs, and
  Claude's independent checks (`verify_*`). The scripts were copied from the session scratchpad; the notes cite their
  original scratchpad paths.
- **Status of an item**: decided or open. Options and a recommendation are added to an open item when it is discussed.

## General rules for the merge

- **No version is trusted by default** (user, 25 Oct 2026; item "Code base of the merge"). For each difference, both
  versions are checked and one is chosen. Everything that exists only on `lore` is ported into `main`. Every feature
  that exists only on `main` is kept, and its defects are fixed, unless an item decides otherwise.
- **The specification is authoritative for everything** (item "Source of truth"): each decision is written into
  `spec/spec.tex` before the code changes.
- **No useless variables or rows** (user, 26 Oct 2026; items "Input checks" and "Input key names"): a variable, and the
  rows that define it, are not built when they only feed an objective term whose coefficient is 0. Examples: `u` and
  its rows when `C_D = 0`; `z` and its rows at a component with `C_R = 0`.

## Agenda

The order follows the dependencies: the foundations first, then the formulation, then the tools and the documents.

| Divergence | Status |
|---|---|
| Anchor of the repair law "since the last intervention" | decided |
| Starting value of the anchor | decided |
| Source of truth: specification or code | decided |
| Code base of the merge | decided |
| Meaning of `H: [a, b]` | decided |
| Horizon sweep | decided |
| Set of degradation models | decided |
| Names and set of repair laws | decided |
| Encoding of the dynamics | decided |
| States covered by the loop constraint | decided |
| Replacement: availability and state after replacement | decided |
| Repair cost on replacement days | decided |
| Depot days, idle days and the meaning of `C_M` | decided |
| Cost shapes and objective terms | decided |
| Rainflow tail bounds | decided |
| Linear forms of the reliability rows | decided |
| Gamma formulation | decided |
| Inverse Gaussian reliability cap | decided |
| Reliability level `epsilon` | decided |
| Input checks | decided |
| Input key names, shapes and axis order | decided |
| Solver defaults | decided |
| Warm starts | decided |
| Output schema and file formats | decided |
| Plotting | decided |
| Validation tools | decided |
| Tests and test inputs | decided |
| Cluster jobs and experiments | decided |
| Documents | decided |
| Repository hygiene | decided |
| Follow-up decisions from writing the spec | decided |

## Decided

### Anchor of the repair law "since the last intervention"

- **`lore` (law `ARA1`)**: the anchor is the state *before* the last repair. The repair-cost row
  `z <= rho·(s[k-1] - anchor[k-1])` applies on every day, and `z >= 0`. Together they force the state after a
  repair back up to the state before it, so a repair lowers nothing. Derived from the code, not run
  (`lore-vs-main.md`, section 12).
- **`main` (law `ard1`)**: the anchor ("latch") is the state *after* the last intervention.
- **Decision (25 Oct 2026)**: use the `main` version. User: *"Concerning the issue you found on the "lore" branch,
  I agree. Let's use the "main" version of it."*
- **The `main` version, as coded** (main:src/fleet_management/degradation_model/rainflow.py:217-282;
  main:src/fleet_management/degradation_model/base.py:789-795, 1017-1036). State `s`, anchor `g`:

| Day | State `s` | Anchor `g` | Repair cost `z` |
|---|---|---|---|
| no intervention | `s_prev` plus the mission increments | `g_prev` | 0 |
| repair | `s_prev - rho·(s_prev - g_prev)` | `s`, the state after the repair | `rho·(s_prev - g_prev)` |
| replacement | `s_new` | `s_new` | see the item "Repair cost on replacement days" |

  - For a variance state, the repair gives `v = (1-rho)²·v_prev + (1-(1-rho)²)·gv_prev`
    (main:src/fleet_management/degradation_model/rainflow.py:244-247).
  - The starting value of the anchor is fixed by the item "Starting value of the anchor".
  - Example: `s_prev = 0.6`, `g_prev = 0.2`, `rho = 0.5`. The repair gives `s = 0.4` and `g = 0.4`. The next day
    the gap `s - g` is 0, so the repair-cost row does not force the state back up.
- **To implement**:
  - Code: the anchor update, the repair row and the repair-cost row of this law, in the code base that the merge uses.
    The loop constraint keeps the anchor among its states.
  - `spec/spec.tex`: the ARA1 equations in the Wiener and Gamma sections, the section on the repair-cost variable,
    and the open item on the anchor convention, which this decision closes.
  - `scoping.tex`: the definition of the law, where it states the anchor.

### Starting value of the anchor

- **`lore`**: `mu_0`, the initial state.
- **`main`**: 0 (main:src/fleet_management/degradation_model/rainflow.py:222, 259;
  main:src/fleet_management/degradation_model/base.py:789-795).
- **Effect**: the value matters only until the first intervention on each component. With 0, the first repair also
  removes part of the initial damage. With `mu_0`, the first repair removes only damage accumulated within the horizon.
  Example: `mu_0 = 0.5`, `rho = 0.5`, a repair on day 1 before any mission. With 0 the state drops to 0.25; with
  `mu_0` it stays at 0.5.
- **Decision (25 Oct 2026)**: an optional per-component input gives the starting anchor; its default is `mu_0`.
  User: *"mu_0 as default, with the per-component input"*.
- **Reason** (Claude's own reasoning, recommended to the user): the true anchor lies between the state of a new part
  and `mu_0`, because damage only grows between interventions. A larger anchor gives a weaker repair and a higher
  state. So `mu_0` never overstates what a repair achieves. The input is exact when the maintenance history is known.
- **Details (25 Oct 2026, both chosen in the answer menu)**:
  - Every anchored state follows the same rule: it starts at the initial value of its own state (the variance anchor
    at `v_0`, the Gamma shape anchor at the initial shape), with the same optional per-component input.
  - The input check requires `0 <= starting anchor <= initial value of its state`, for every anchor. Damage only grows
    between interventions, so a larger anchor is impossible.
- **To implement**:
  - Code: the new optional input, default the initial value of each anchored state, used as the anchor before the
    first step; the range check. The input name is fixed by the item "Input key names, shapes and axis order".
  - `spec/spec.tex`: the input tables, the consistency checks and the starting value of each anchor recursion.

### Source of truth: specification or code

- **`lore`**: `spec/spec.tex` version 0.5 is authoritative; the code follows it.
- **`main`**: the README declares the code authoritative; `spec/spec.tex` is the older version "0.2 (revised)".
  The formulation of some features exists only in the code and in notes: the Gamma common-rate calibration
  (`GAMMA_DIAGNOSTICS.md`) and the Markov, Hoeffding and Chernoff bounds. The README says the bound equations are
  "under `spec/`", but no file there contains them.
- **Options offered**: the spec for the formulation and the code for the tools (Claude's recommendation); the spec for
  everything; the code for everything.
- **Decision (25 Oct 2026, chosen in the answer menu)**: the spec for everything. `spec/spec.tex` is authoritative for
  the formulation and for the tools: every change is written into the spec first, and the code follows.
- **To implement**:
  - `spec/spec.tex`: every feature kept from `main` enters the spec, the tools included: for example the horizon sweeps,
    the warm starts, the validators and the plots, if the merge keeps them.
  - README: it names `spec/spec.tex` as the source of truth.
  - Every later item: its decision is written into the spec before the code changes.

### Code base of the merge

- **`lore`**: one shared Gurobi model per horizon; a dictionary `_DISPATCH` of five component builders; input parsing
  in `solver.py`. About 2,100 lines of Python in `src/`.
- **`main`**: input parsing in `config.py` (dataclass `FleetConfig`); a registry of cell builders (`prepare`,
  `add_cell`) with Gamma and rainflow builders; more tooling (sweeps, warm starts, validators, cluster jobs). About
  22,600 lines of Python in `src/`; the files on the public solve path alone have about 5,600 lines.
- **Options offered**: `main`'s code as the base (Claude's recommendation); `lore`'s code as the base.
- **Decision (25 Oct 2026)**, the user's instruction in neutral wording (the user chose to reword it, 26 Oct 2026): merge
  the two branches into `main`; trust neither version by default; judge every difference from `lore` carefully and
  choose one of the two versions; port everything that exists only on `lore` smoothly.
  - The merged package lives on the branch `main`.
  - No version is accepted by default. For each difference, both versions are checked, and one of them is chosen.
  - Everything that exists only on `lore` is ported into `main`.
- **Features that exist only on `main`** (26 Oct 2026, chosen in the answer menu): keep them all, and fix their
  defects. Options offered: judge each and keep or drop it (Claude's recommendation); keep all and fix defects; drop
  unless a later item needs it.
- **How later items are presented**: the recommendation states what was checked in each version (code read, equations
  derived, cases worked by hand) and lists every defect found.

### Meaning of `H: [a, b]`

- **`lore`**: a range of horizons, one solve per `H` from `a` to `b`, each with `2H` steps.
- **`main`**: one solve with a transitory phase `H1 = a` and an operating phase `H2 = b`, `T = a + b` steps.
- `Things to change.txt` asks for the split into a transitory and an operating horizon.
- **Check of `main`'s version** (25 Oct 2026, code read):
  - `_parse_horizon` (main:src/fleet_management/config.py:149-160): a scalar `H` gives `H1 = H2 = H`, `T = 2H`,
    which is `lore`'s structure; a pair gives `T = H1 + H2`; both values must be positive.
  - `make_accessor` (main:src/fleet_management/degradation_model/base.py:1291-1304): a step `k < H1` reads the
    transitory profile, or the operating profile cyclically when no transitory profile is given; a step `k >= H1` reads
    operating entry `(k - H1) mod H2`.
  - Loop row `state[T-1] <= state[H1-1]`. The repeatability argument (Claude's derivation) does not need `H1 = H2`:
    if the period map of the operating phase is monotone and maps `s[H1-1]` below itself, every repetition of the
    operating phase stays below the first one.
  - Defect: the legacy Gamma route uses `H1` only, with a `2·H1` layout, and ignores `H2`
    (main:src/fleet_management/solver.py:188-197). The item "Gamma formulation" covers this route.
- **Syntax in use on `main`**: 25 tracked YAML inputs and 20 Python call sites write `H: [H1, H2]`.
- **Old `lore` inputs are not protected** (checked 26 Oct 2026). `main`'s `load_config` accepts unknown keys without
  an error, so it ignores `maintenance_type` and `tail_bound`. Its shape check accepts a `lore` increment tensor
  (F, M, L, H) as (F, L, M, H) whenever `M = L`, so the axes are swapped without an error
  (main:src/fleet_management/config.py:241-261). An old `lore` file with `H: [4, 6]` therefore runs a different
  problem without any warning. The item "Input key names, shapes and axis order" must close this gap.
- **Options offered**: two-phase meaning with the pair syntax as on `main` (Claude's recommendation); two-phase
  meaning with explicit keys `H1` and `H2`; range meaning as on `lore`.
- **Decision (26 Oct 2026, chosen in the answer menu)**: two-phase meaning, pair syntax `H: [H1, H2]`, as on `main`.
  A scalar `H` keeps `H1 = H2 = H`. The range search moves to the item "Horizon sweep". The user confirmed the
  decision after learning that old `lore` inputs are not protected (answer menu, 26 Oct 2026).
- **To implement**:
  - Code: `main`'s `_parse_horizon` and `make_accessor` as they are; `lore`'s range loop inside `solve()` goes.
  - `spec/spec.tex`: the horizon, the increment profiles (transitory and operating), the loop constraint
    `state[T-1] <= state[H1-1]` and the repeatability argument for `H1 ≠ H2`.

### Horizon sweep

- **`lore`**: inside `solve()`; optional parallel runs (`n_workers`); a failed horizon is recorded and the loop
  continues; no best horizon is selected; reads YAML, JSON and HDF5.
- **`main`**: separate functions `sweep_operating_horizons` (one `H1`, a list of `H2`) and `sweep_horizon_grid` (an
  `H1 × H2` grid); sequential; optional early stop; a checkpoint after each case.
- **Check of `main`'s version** (26 Oct 2026, code read: main:src/fleet_management/horizon_sweep.py:99-129, 335-450;
  main:src/fleet_management/degradation_model/base.py:1735-1811):
  - Ranking by `J_op_average = J_op/H2`, the cost of the operating phase per day. Unlike a raw objective, it can be
    compared across different `H2`. It is the average cost per day of a cycle that repeats without end, where the
    transitory cost spreads to zero (Claude's reading).
  - The sweep solves with `objective_mode = operating_average`: the objective is `J_op/H2 + C_D·u`, so the
    transitory phase costs nothing in these solves.
  - The grid sweep can rank by the projected cost `J_init + (E - H1)·J_op/H2 + C_D·u` over an evaluation horizon `E`,
    which also prices the transitory phase. This is the criterion that compares different `H1`.
  - A case counts as proven only if its status is optimal and its gap is at most 1e-8.
  - Defects: one solver exception stops the whole sweep (no `try` around `solve()`, line 418); the cases run only
    one after another; the input must be YAML (`yaml.safe_load`, lines 364 and 578).
- **`lore`'s features that fix these defects**: per-case error capture, parallel runs, and every input format.
- **Options offered**: `main`'s functions plus `lore`'s fixes (Claude's recommendation); the same plus a sweep inside
  `solve()` started by an input key.
- **Decision (26 Oct 2026, chosen in the answer menu)**: keep `main`'s `sweep_operating_horizons` and
  `sweep_horizon_grid`, and port `lore`'s per-case error capture, parallel runs and input formats. There is no sweep
  inside `solve()`.
- **To implement**:
  - Code: a `try` around each case, which records the error and continues; an option for parallel runs, refused
    together with warm starts between cases; the sweep reads its input through the same reader as `solve()`.
  - `spec/spec.tex`: the two sweep functions, their ranking criteria (`J_op/H2`; the projected cost over an
    evaluation horizon), the proof rule (optimal with gap at most 1e-8), the early stop, the checkpoint, the error
    capture and the parallel option.

### Set of degradation models

- **`lore`**: Gaussian, inverse Gaussian (IG), Wiener, Gamma and rainflow, all built.
- **`main`**: Gamma and rainflow built; Gaussian and IG accepted by the input check but not built; no Wiener.
- **Check of both versions** (code read in full on both branches, 30 Sep 2026; `merge/reports/B_models.md`):
  - `main`'s `gaussian.py` and `inverse_gaussian.py` are the files of the common ancestor, unchanged. They are
    stand-alone solvers, not cell builders, so `solve()` cannot reach them. They read the old keys `alpha`, `xi`, `c`,
    `C_S`, `C_P`, have no replacement, and use the old reliability rows: for Gaussian a linear row built on
    three-sigma terms, for IG a normal approximation of the distribution function.
  - `lore`'s Gaussian, IG and Wiener are component builders on the shared model, with replacement, the exact
    `Φ⁻¹` row for Gaussian and Wiener, and the IG cap from the exact IG distribution function.
  - Gamma and rainflow exist on both branches; their versions are separate items.
  - Porting `lore`'s three builders into `main`'s registry depends on the item "Encoding of the dynamics":
    `lore` writes one-sided big-M rows, `main`'s builders write equalities.
- **Options offered**: all five models with `main`'s two old files removed (Claude's recommendation); the same with
  the old files moved to `legacy/`.
- **Decision (26 Oct 2026, chosen in the answer menu)**: all five models. `lore`'s Gaussian, IG and Wiener are ported
  as cell builders in `main`'s registry. `main`'s old `gaussian.py` and `inverse_gaussian.py` leave the package; git
  history keeps them. Gamma and rainflow follow their own items.
- **To implement**:
  - Code: three new cell builders, registered under `gaussian`, `inverse_gaussian` and `wiener`, with their input
    checks; the encoding follows the item "Encoding of the dynamics".
  - `spec/spec.tex`: the five model sections, as one model list.

### Names and set of repair laws

- `lore`: `ARD1` (the whole state times `1-rho`) and `ARA1` (since the anchor); `ARA1` only for Wiener and Gamma.
- `main`: `ardinf` (the whole state) and `ard1` (since the anchor); both for Gamma and rainflow, except rainflow with
  Chernoff and `ard1`.
- **The names in the user's own documents** (checked 26 Oct 2026):

| Law | `lore` spec (lore:spec/spec.tex:126-137) | `lore` scoping (lore:scoping.tex:1118-1130) | `main` |
|---|---|---|---|
| `D+ = (1-rho)·D-` (the whole state) | `ARD1` | `ARD∞` | `ardinf` |
| `D+ = D- - rho·(D- - D_last)` (since the last intervention) | `ARA1` | `ARD1`; `ARA1` is the same law on the virtual age `V` | `ard1` |

  - The scoping document attributes its taxonomy to Doyen and Gaudoin (2004), with a memory-one variant (subscript 1)
    and an infinite-memory variant (subscript ∞). `main`'s names agree with the scoping document; `lore`'s spec does not.
  - The papers themselves (Doyen and Gaudoin 2004; Mercier and Castro 2019) are not yet read. They must be read before
    the spec states the definitions with these citations.
- **Which model accepts which law: derivations** (Claude's own, 26 Oct 2026). Write the state after a repair as
  `G + (1-rho)·Δ` (law `ARD1`), with `G` the state at the last intervention and `Δ` the damage since then, independent
  of `G`; or as `(1-rho)·D` (law `ARD∞`).
  - Mean, any model: `mu+ = (1-rho)·mu + rho·g_mu` under `ARD1`, by linearity.
  - Variance, independent increments: `Var(D) = Var(G) + Var(Δ)`, so `v+ = (1-rho)²·v + (1-(1-rho)²)·g_v` under
    `ARD1`. This is `main`'s rainflow row (main:src/fleet_management/degradation_model/rainflow.py:244-247).
  - Gaussian: `G + (1-rho)·Δ` is a sum of independent Gaussians, so the state stays Gaussian with the mean and
    variance above. So `ARD1` is exact for Gaussian. `lore`'s spec refused it because "the ARA1 variance extension
    `v+ = v- - rho·(v- - v_last)` has no established basis" (lore:spec/spec.tex:737-742). That formula, linear in `rho`,
    is indeed wrong; the correct one is quadratic in `1-rho`, as above.
  - IG: `c·IG(mu, lambda) = IG(c·mu, c·lambda)`, so `ARD∞` keeps the family `IG(mu, mu/eta)` exactly. Under `ARD1`
    the state `G + (1-rho)·Δ` is a sum of IG terms with different `lambda/mu²`, so it leaves the family. `lore` allows
    `ARD∞` only; this agrees.
  - Wiener: `lore`'s repair acts on the mean only, and the variance falls only by replacement. This is a stated
    modelling choice of `lore`'s spec. A pathwise repair would also shrink the variance.
  - Gamma: both branches repair in parameter space, an approximation; the item "Gamma formulation" covers it.
  - Rainflow, Chernoff under `ARD∞`: `main` uses `K+ = (1-rho)·K`. It is valid: a cumulant generating function `ψ`
    is convex with `ψ(0) = 0`, so `ψ((1-rho)·s) <= (1-rho)·ψ(s)`.
  - Rainflow, Chernoff under `ARD1`: `main` refuses it, stating that no recursion closes
    (main:src/fleet_management/config.py:484-494). A conservative one closes: the true value after the repair is
    `K_G(s) + K_Δ((1-rho)·s) <= K_G(s) + (1-rho)·K_Δ(s)`. The tracked `K - gK` equals `K_Δ(s)` exactly, because the
    carry rows add exact increment values since the anchor, and `gK >= K_G(s)`. So `K+ = (1-rho)·K + rho·gK` is an
    upper bound.
- Options for the names:
  - **Spec symbols `ARD_1` and `ARD_∞`, input values `ard1` and `ardinf`**, as in the scoping document and on `main`
    (recommended).
  - **`lore`'s spec names `ARD1` and `ARA1`.**
- Options for the set:
  - **Each branch's set**: Gaussian and IG `ARD∞` only; Wiener both laws on the mean; rainflow both laws, except
    Chernoff with `ARD1`; Gamma by its own item.
  - **Each branch's set, plus `ARD1` for Gaussian** (exact, by the derivation above).
  - **The previous option, plus `ARD1` for Chernoff** with `K+ = (1-rho)·K + rho·gK` (recommended).
- Reason for the recommendations (Claude's own reasoning): the names follow the user's scoping document and the
  taxonomy it cites, and they match the code that the merge keeps. For the set, every model whose moments close under
  a law accepts that law, which makes the rule uniform. IG is the only exception, because `ARD1` leaves its family.
- **Decision on the set (26 Oct 2026, chosen in the answer menu)**: each branch's set, plus the since-last law for
  Gaussian and for Chernoff.

| Model | Whole-state law | Since-last law |
|---|---|---|
| Gaussian | yes | yes (exact; new) |
| IG | yes | yes (item "Follow-up decisions from writing the spec": the CVaR cap does not need the IG family) |
| Wiener | yes, on the mean | yes, on the mean |
| Gamma | by the item "Gamma formulation" | by the item "Gamma formulation" |
| Rainflow, Markov, Cantelli, Hoeffding, Bernstein | yes | yes |
| Rainflow, Chernoff | yes, `K+ = (1-rho)·K` | yes, `K+ = (1-rho)·K + rho·gK` (conservative; new) |

- **Decision on the names (26 Oct 2026)**. User: *"read the papers and use the correct names"*. The names are those
  that Doyen and Gaudoin (2004) and Mercier and Castro (2019) define.
- **What the papers define** (read 26 Oct 2026):
  - Doyen and Gaudoin (2004). Read in full by an agent, as page images, from the authors' preprint "submitted to RESS
    25 October 2002" (https://web.archive.org/web/20051016052209/http://www-lmc.imag.fr/SMS/ftp/DoyGau02.pdf, 22 pp.;
    the published version could not be obtained). Pages 8, 11 and 12 checked by Claude.
    - Classes on the failure intensity (ARI) and on the virtual age `A` (ARA), each with memory 1, `m` or ∞.
      "Memory" is the maximal number of previous failure times that enter the failure intensity (p. 8).
    - ARA∞, eq. (20), p. 11: `A(T_i+) = A(T_i-) - rho·A(T_i-)`, a reduction of the whole virtual age.
    - ARA1, eq. (22), p. 12: `A(T_i+) = A(T_i-) - rho·[A(T_i-) - A(T_{i-1}+)]`, measured from the value just after
      the previous repair.
    - No class acts on a degradation level; the acronym ARD does not occur (the agent's full reading).
  - Mercier and Castro (2019). Read in full by an agent from the authors' preprint of 11 June 2018
    (https://smercie1.perso.univ-pau.fr/Publications/P29_EJOR_2018_MERCIER_CASTRO_preprint.pdf); pages 3 and 8-9
    checked by Claude. The correct DOI is 10.1016/j.ejor.2018.06.020 (checked with Crossref).
    - Two models only. ARD1, "Arithmetic Reduction of Deterioration of order 1" (§3.1), eq. (4):
      `Y((n+1)T) = Y(nT) + (1-rho)·(X((n+1)T) - X(nT))`. It removes `rho` times the degradation accumulated since
      the last maintenance, measured from the level just after it, and from the origin before the first one. It acts
      on the degradation path; the age is unchanged.
    - ARA1 (§3.2): a reduction of the virtual age, after Doyen and Gaudoin (2004).
    - The paper never writes ARD∞ or ARA∞.
- **Consequences**:
  - The since-last law on the degradation level is **ARD1** (Mercier and Castro 2019, eq. 4). `lore`'s spec name
    "ARA1" for it is wrong; "ARA1" is a law on the virtual age.
  - The whole-state law on the degradation level is not named in either paper. By the memory convention of Doyen and
    Gaudoin, whose ARA∞ reduces the whole virtual age, its name is **ARD∞**. This name is an analogy, not a citation.
  - `lore`'s spec name "ARD1" for the whole-state law is wrong. `lore`'s scoping document uses ARD1 and ARD∞ correctly,
    but attributes ARD to Doyen and Gaudoin, who do not define it.
  - Paper's convention for the reference level: just after the previous maintenance, and the initial level before the
    first one. This agrees with the decided anchor and its starting value.
  - `spec/references.bib` carries a wrong DOI for Mercier and Castro (2019): 10.1016/j.ejor.2018.08.003 belongs to a
    paper on medical training.
- **Options offered for the whole-state name**: ARD∞, stated as an analogy (Claude's recommendation); a further
  literature search first.
- **Final names (26 Oct 2026, chosen in the answer menu)**:
  - Since-last law: **ARD1**, cited to Mercier and Castro (2019), eq. (4). Input value `ard1`.
  - Whole-state law: **ARD∞**, named by analogy with ARA∞ of Doyen and Gaudoin (2004), eq. (20); the spec says that
    the name is an analogy. Input value `ardinf`.
  - ARA1 and ARA∞ keep their meaning: laws on the virtual age, not used by the models of the merge.
- **To implement**:
  - Code: the set above. The Gaussian builder gets the since-last law, with anchors for mean and variance; the
    Chernoff check in `config.py` admits the since-last law, with the anchor `gK` and the row
    `K+ = (1-rho)·K + rho·gK`. `lore`'s `ARD1` becomes `ardinf`, and `lore`'s `ARA1` becomes `ard1`.
  - `spec/spec.tex`: the two definitions with these citations; the laws per model, with the derivations above; the
    removal of the remark that refuses the since-last law for Gaussian (lore:spec/spec.tex:737-742).
  - `spec/references.bib`: the DOI of Mercier and Castro (2019) becomes 10.1016/j.ejor.2018.06.020.
  - `scoping.tex`: ARD1 cited to Mercier and Castro (2019); ARD∞ stated as an analogy; Doyen and Gaudoin (2004)
    cited for ARA and ARI only (lore:scoping.tex:1118-1130).
  - `CLAUDE.md`: the names ARD1 and ARD∞ replace `lore`'s names once the code changes.

### Encoding of the dynamics

- `lore`: one-sided big-M inequalities; the tracked states are upper bounds on the moments.
- `main`: equalities, through exact binary-product hulls (Gamma) and indicator constraints (rainflow); tight variable
  bounds computed per cell and step.
- **Definitions.** Each day, a component's state follows one rule: a mission adds an increment, a repair reduces the
  state, a replacement resets it.
  - Exact: each rule is an equality, for example `s_k = s_{k-1} + increment`. The tracked state is the model's
    expected degradation.
  - One-sided: each rule is an inequality, for example `s_k >= s_{k-1} + increment`. The solver may choose a larger
    value, never a smaller one, so the tracked state is an upper bound on the expected degradation. Example: with
    `s_{k-1} = 0.3` and an increment of 0.2, exact gives 0.5; one-sided allows 0.5 or 0.6.
  - Every reliability row bounds the tracked state from above, so a larger value only makes it harder to satisfy.
    Larger values also cost more (repair cost, damage penalty). The one place where a larger value helps is the loop
    row: a larger state at the end of the first phase makes `state[T-1] <= state[H1-1]` easier.
- **Equations** for one component's mean `s`, whole-state law. Day `k`; `δ_k = Σ_j x_ijk·mu_inc_j` (0 on a depot
  day); repair and replacement binaries `m_k`, `r_k`; `n_k = 1 - m_k - r_k`.
  - Exact (main:src/fleet_management/degradation_model/rainflow.py:217-235), indicator constraints:
    `n_k = 1 ⇒ s_k = s_{k-1} + δ_k`; `m_k = 1 ⇒ s_k = (1-rho)·s_{k-1}`; `r_k = 1 ⇒ s_k = s_new`.
  - One-sided (lore:spec/spec.tex:836-851), with `M = tau`:
    `s_k >= s_{k-1} + δ_k - M·m_k - M·r_k`; `s_k >= (1-rho)·s_{k-1} - M·(1 - m_k)`;
    `s_new - M·(1 - r_k) <= s_k <= s_new + M·(1 - r_k)`. At binary points these are the exact rows with `=`
    replaced by `>=`, except replacement.
  - Common: reliability `g(s_k) <= 0` with `g` increasing; loop row `s_{T-1} <= s_{H1-1}`.
- **What each encoding accepts for a fixed schedule.** Let `φ_k` be the exact, increasing map of day `k`, `Φ` the
  exact map of the operating phase, and `s̄` the exact state at the end of the first phase.
  - Exact: accepted iff `Φ(s̄) <= s̄` and `g <= 0` along the exact trajectory.
  - One-sided: accepted iff some `y >= s̄` gives `Φ(y) <= y` and `g <= 0` along the tracked trajectory from `y`.
    `y = s̄` is the exact case.
  - Safety of one-sided: `s_k <= ŝ_k` by induction; `Φ(y) <= y` gives `Φ^n(y) <= y`; by monotonicity every later
    cycle stays below the checked first cycle.
  - Example: cycle = mission (`δ = 0.2`) then repair (`rho = 0.5`), so `Φ(y) = 0.5·y + 0.1` and `Φ(y) <= y` iff
    `y >= 0.2`. With `s̄ = 0.1`, `Φ(0.1) = 0.15 > 0.1`: exact rejects. One-sided takes `y = 0.2` (tracked cycle
    0.2, 0.4, 0.2; `g` checked at 0.4). The true trajectory 0.1, 0.15, 0.175, ... rises to 0.2 with peaks rising to
    0.4: safe. `main` can often reach an equivalent plan by skipping repairs in the first phase, which raises `s̄`;
    the difference matters when the first phase cannot raise it enough (new parts, short `H1`). Cost: each repair is
    charged `rho·0.4 = 0.2`, while the true removal rises from 0.15 toward 0.2.
- **Check of both versions** (26 Oct 2026):
  - `main`'s rows are exact. Rainflow: one indicator equality per case (no intervention, repair, replacement), with
    `nb + m + r = 1` (main:src/fleet_management/degradation_model/rainflow.py:217-282). Gamma: equality balances with
    exact product hulls (main:src/fleet_management/degradation_model/base.py:1017-1036).
  - `lore`'s spec proves that one-sided rows keep the loop guarantee: the tracked states dominate the true moments,
    and the loop row then gives `Φ(s_H) <= s_H` for the exact period map `Φ`, started from the tracked envelope
    (lore:spec/spec.tex:656-694, remark "the loop guarantee is robust to slack").
  - Consequence (Claude's derivation): every exact solution also satisfies the one-sided rows, so `lore`'s encoding
    admits every schedule that `main`'s admits, and more (example above).
  - Price of the one-sided rows: the tracked states, and the costs computed from them (`C_R·z`, `C_D·u`), are upper
    bounds, not exact values. For a fixed schedule, the solver can choose the exact states, so, with the same cost
    terms, `lore`'s optimum costs at most `main`'s.
  - Speed: no solve has been run, so no comparison of solve times exists. Big-M rows usually give weaker linear
    relaxations than hulls; indicator constraints are handled by Gurobi's own branching.
- Options:
  - **One-sided rows for every model, as on `lore`, plus an exact replay after the solve** (recommended). The replay
    runs the exact dynamics on the chosen schedule and reports the true states and costs next to the tracked ones.
  - **One-sided rows for every model, as on `lore`**; the output reports the tracked states (upper bounds).
  - **Exact equalities for every model, as on `main`.**
- Reason for the recommendation (Claude's own reasoning): the one-sided rows admit more safe schedules and never a
  costlier optimum. The replay removes their one drawback, the inexact reported values. `main` already has replay
  tools for Gamma (`gamma_replay_validator.py`).
- **User's answer (26 Oct 2026)**: *"I would opt for option 2, i.e., one-sided, as on lore if you say that it does not
  cut out feasible solutions to the true problem, nor allows unfeasible ones. However, what is on lore uses big-M by
  default. I want a flag as input that specifies if going for the big-M or for the indicator constraints. If possible,
  I would use one-sided also with the indicator constraints"*.
- **Answer to the condition** (Claude's derivation, 26 Oct 2026). The true problem: the true moments satisfy the
  reliability rows on every day of the horizon and of every later repetition of the operating phase.
  - **It never admits an unsafe schedule.** The proof above needs two facts: each day's exact map is increasing in
    the full tracked state (anchors included), and each reliability row is increasing in each tracked component.
    Both hold for every row in the merge: Cantelli `((1-eps)/eps)·v - (tau-mu)²` with `mu <= tau`, Bernstein,
    Hoeffding, Chernoff, Markov, the Gaussian row, and the IG and Gamma caps.
  - **It rejects no schedule that the exact encoding admits**, and admits more.
  - **It can reject a truly safe schedule**, as the exact encoding can, but in fewer cases. With one tracked quantity
    and the whole-state law, it rejects none: the start-of-cycle state then moves monotonically, and its supremum is a
    valid `y`. With two or more quantities it can reject one, when the quantities move in opposite directions over the
    repetitions. Example (Cantelli, `eps = 0.1`, `tau = 1`, `rho = 0.5`; cycle: a mission adds 0.2 to the mean and
    0.009 to the variance, then a repair):
    - `Φ(mu, v) = (0.5·mu + 0.1, 0.25·v + 0.00225)`, fixed point `(0.2, 0.003)`; start `s̄ = (0.5, 0)`.
    - True cycles: the mean falls to 0.2 and the variance rises to 0.003. The peak of the first cycle is `(0.7, 0.009)`:
      `9·0.009 = 0.081 <= 0.3² = 0.09`. Every later peak has mean at most 0.55 and variance below 0.012:
      `9·0.012 = 0.108 <= 0.45² = 0.2025`. So the schedule is safe forever.
    - One-sided needs `y >= (0.5, 0)` with `Φ(y) <= y`, so `y_v >= 0.003`; its tracked peak is at least
      `(0.7, 0.012)`, where `9·0.012 = 0.108 > 0.09`. Rejected.
    - Exact: `Φ(s̄) = (0.35, 0.00225)`, whose variance exceeds 0; rejected as well.
  - The one-component claim holds for the whole-state law. With the since-last law the anchor is a second tracked
    quantity, and the claim is not proven.
- **The flag** (Claude's check of feasibility): one-sided rows work with indicator constraints too. Example:
  `n_k = 1 ⇒ s_k >= s_{k-1} + δ_k`; `m_k = 1 ⇒ s_k >= (1-rho)·s_{k-1} + rho·g_{k-1}`; `r_k = 1 ⇒ s_k = s_new`.
  The anchor rows stay equalities (`g_k = s_k` on an intervention day), as on `lore`. Gurobi accepts inequalities in
  indicator constraints.
- **Decision (26 Oct 2026, chosen in the answer menu)**: one-sided rows for every model, as on `lore`, with no exact
  replay; the output reports the tracked states (upper bounds) and the costs computed on them. A new input flag
  selects big-M rows or indicator constraints, both one-sided. The default of the flag is open.
- **Guideline for the default** (Gurobi Optimizer Reference Manual, section "Dealing with Big-M Constraints",
  https://docs.gurobi.com/projects/optimizer/en/current/concepts/numericguide/tolerances_scaling.html, read on
  26 Oct 2026 through a fetch tool that paraphrases the page):
  - Big-M rows are a frequent source of numerical instability, because M exceeds any value the variable can reach.
    With the default integrality tolerance `IntFeasTol = 1e-5`, the row `x <= 1e6·y` lets `x` reach 9.999 while `y`
    counts as 0. If `x` is known to stay below `1e3`, the row `x <= 1e3·y` limits this to 0.01.
  - When the bound cannot be tightened, an indicator constraint "may produce more accurate solutions", usually at the
    cost of more processing time.
  - Application to this model (Claude's reading): `lore`'s M is the natural bound of each state (`tau` for a mean, the
    variance cap for a variance, `tau/beta` for a Gamma shape), which the reliability rows enforce anyway. M is the
    state's own range, so the leak above is at most `IntFeasTol·M`, a relative error of `1e-5` (for example `1e-5`
    when `tau = 1`, the Palmgren-Miner normalisation). So the manual's reason for indicators does not apply here.
  - Which encoding solves faster on this model is an empirical question; no solve has been run.
- **Options offered for the default**: big-M with a later benchmark of both encodings (Claude's recommendation);
  big-M without a benchmark; indicator constraints.
- **Decision on the default (26 Oct 2026, chosen in the answer menu)**: big-M by default; indicator constraints on
  request; no benchmark planned.
- **To implement**:
  - Code: one-sided rows in every builder, in two forms: big-M rows with M equal to the state's natural range, and
    indicator constraints with the same inequalities. Replacement rows and anchor updates stay equalities. `main`'s
    exact Gamma hulls and rainflow indicator equalities become one-sided. The flag's name is fixed by the item "Input
    key names, shapes and axis order".
  - `spec/spec.tex`: the one-sided rows in both forms; the slack-robustness remark extended to every model and to
    the since-last law; the answer to the user's condition above (never unsafe; may reject a truly safe schedule only
    when tracked quantities move in opposite directions); the meaning of the reported states and costs (upper bounds).

### States covered by the loop constraint

- **`lore`**: every tracked state, the anchors included.
- **`main`**: rainflow covers every tracked quantity and its anchor; Gamma covers only `A` and `mu`, not its anchors.
- **Check of both versions** (26 Oct 2026):
  - `lore`'s spec proves that the loop row must cover the full state, the anchors included: the repeatability
    argument needs `Φ(x) <= x` for the full state `x` at the end of the first phase (lore:spec/spec.tex:688-694).
  - `main`'s rainflow covers `mu`, `v`, `R`, `K` and the anchors `gmu`, `gv`, `gR`, and its comment says the anchor
    must be carried under ARD1 (main:src/fleet_management/degradation_model/rainflow.py:624-639).
  - `main`'s Gamma loop rows cover `A` and `mu` only (main:src/fleet_management/degradation_model/base.py:1053-1063).
    Its anchors `gmu` and `gA` are free at the end of the horizon.
  - Why that is a defect (Claude's derivation): under ARD1 a repair gives `s+ = (1-rho)·s + rho·g`, so a larger
    anchor gives a weaker repair. Example: the first phase ends at `(s, g) = (0.4, 0.1)`; the operating cycle ends at
    `(0.35, 0.3)`, its last repair having left the state at 0.3. The row `0.35 <= 0.4` holds, but the next cycle
    starts with anchor 0.3 instead of 0.1, so its first repair leaves the state `0.2·rho` higher than in the checked
    cycle, and its peaks are not covered. With the anchor in the loop, `0.3 <= 0.1` fails and the schedule is rejected.
- **Options offered**: the full tracked state for every model (Claude's recommendation); each branch as it is.
- **Decision (26 Oct 2026, chosen in the answer menu)**: the loop row `state[T-1] <= state[H1-1]` covers every tracked
  quantity and every anchor, for every model.
- **To implement**:
  - Code: loop rows for `gmu` and `gA` in the Gamma builder; for `gK` with the new Chernoff since-last law; for the
    anchors of the ported Gaussian and Wiener builders.
  - `spec/spec.tex`: one rule, stated once for all models: the loop row covers the full tracked state.

### Replacement: availability and state after replacement

- **`lore`**: always allowed; the state is reset to `mu_new`, `v_new` from the input (default 0).
- **`main`**: option `allow_replacement` (default true); for rainflow, option `replacement_as_new` (default true)
  resets the state to 0 and overrides the input.
- **Check of `main`'s version** (26 Oct 2026, code read):
  - `allow_replacement` (default true) removes the replacement binaries when false
    (main:src/fleet_management/degradation_model/base.py:1206-1207, 1622). No defect found.
  - With `replacement_as_new` true, `config.py` refuses a non-zero `replacement_mu` or `replacement_v` at a rainflow
    cell, and `rainflow.py` sets them to 0 (main:src/fleet_management/config.py:441-458;
    main:src/fleet_management/degradation_model/rainflow.py:104-109). The error message gives the reason: it matches
    "the R = 0 / K = 0 resets". On every replacement, the Hoeffding budget `R` and the Chernoff quantity `K` reset to
    0 (main:src/fleet_management/degradation_model/rainflow.py:337 and the Hoeffding block above it).
  - **Defect A**: with `replacement_as_new` false and a non-zero replacement state, `R` and `K` still reset to 0, which
    understates the new part's damage.
  - **Defect B**, found here: the Chernoff quantity starts at `K = 0` at the first step, whatever the initial damage
    `mu_0` (main:src/fleet_management/degradation_model/rainflow.py:327). The Chernoff row `K - s·tau <= ln(eps)`
    (lines 459-464) uses `K` only, so it ignores `mu_0`. The bound `P(D >= tau) <= exp(K_D(s) - s·tau)` needs `K_D`,
    the cumulant generating function (CGF) of the whole damage, and `K_D(s) >= s·E[D]` by Jensen's inequality.
    Example (Claude's construction): `tau = 1`, `eps = 0.1`, `s = 5` (allowed: `s > ln(1/eps)/tau = 2.30`); known
    initial damage `mu_0 = 0.6`; one mission adds `X` in {0, 0.5} with probability 1/2 each, whose CGF at `s = 5` is
    `ln(0.5 + 0.5·e^2.5) = 1.886`. True failure probability: `P(0.6 + X > 1) = 0.5`. `main`'s row:
    `1.886 - 5 = -3.11 <= -2.30`, accepted; the mean `0.85 <= tau` passes too. With the initial damage,
    `K = 3 + 1.886 = 4.886` and `4.886 - 5 = -0.11 > -2.30`, rejected.
  - Hoeffding starts at `R = 0` too. That is correct for a known initial damage, which enters through the mean; it
    understates a random initial damage.
- Options for the state after replacement:
  - **From the input, default 0, for every model, as on `lore`; `R` and `K` reset to the values implied by the
    replacement state; `replacement_as_new` kept as an explicit shortcut for "new part, all zeros", default false**
    (recommended).
  - **The same, with `replacement_as_new` removed**: with defaults of 0 it duplicates omitting the keys.
  - **`main` as it is**: `replacement_as_new` true by default; Defect A stays reachable when it is false.
- Options for the starting and replacement values of `R` and `K` (fix of Defects A and B):
  - **Known damage by default: `K_0 = s·mu_0`, `R_0 = 0`, `K_new = s·mu_new`, `R_new = 0`; optional inputs for a
    random initial or replacement damage (its CGF at `s`, its squared range)** (recommended). This mirrors how
    Cantelli and Bernstein already accept `v_0` and `v_new`.
  - **Known damage only**: the same defaults, no new inputs.
- Reason for the recommendations (Claude's own reasoning): the replacement state then follows one rule for every model
  and every bound, and both defects close. `allow_replacement` stays, as the policy keeps `main`'s features.
- **Decisions (26 Oct 2026, chosen in the answer menu)**:
  - The state after replacement comes from the input, default 0, for every model. `R` and `K` reset to the values
    implied by the replacement state. `replacement_as_new` stays as an explicit shortcut for "new part, all zeros",
    default false. `allow_replacement` stays, default true.
  - Starting and replacement values of `R` and `K`: known damage by default (`K_0 = s·mu_0`, `R_0 = 0`,
    `K_new = s·mu_new`, `R_new = 0`), with optional inputs for a random initial or replacement damage: its CGF at `s`
    and its squared range.
- **To implement**:
  - Code: the replacement rows of every builder read the input state; the Hoeffding and Chernoff starting and reset
    values follow the rule above (fixes Defects A and B); the check that refuses non-zero replacement values runs only
    when `replacement_as_new` is true; two new optional inputs, named in the item "Input key names, shapes and axis
    order".
  - `spec/spec.tex`: the replacement state for every model and bound; the starting values of `R` and `K` with the
    Jensen argument; the two options.

### Repair cost on replacement days

- **`lore`**: `z` only on repair days.
- **`main`**: rainflow also on replacement days (`z = mu_prev - mu_new`); Gamma only on repair days.
- **Check of `main`'s version** (26 Oct 2026, code read):
  - Rainflow: `r_k = 1 ⇒ z_k = mu_{k-1} - mu_k` (main:src/fleet_management/degradation_model/rainflow.py:283-286).
  - Gamma: its comment states the rule "z is the damage removed by imperfect repair. Replacement has its own binary
    cost and therefore must not also contribute to the repair cost" (main:src/fleet_management/degradation_model/base.py:313-314).
    So `main`'s two builders disagree.
  - `main`'s default replacement cost is `C_rep = C_R · max_i tau` (main:src/fleet_management/degradation_model/base.py:1272-1273),
    the repair cost of removing a full part's damage. Under the rainflow rule, a default replacement pays
    `C_rep + C_R·(mu_{k-1} - mu_new)`, so the same removal is charged twice.
  - Under the rainflow rule, a replacement also costs more the more damaged the old part is (Claude's reading). A
    replaced part is discarded, so its damage does not change the price of the new one.
- Options:
  - **`z` only on repair days, for every model** (recommended), as on `lore` and in `main`'s Gamma. A replacement
    costs `C_rep`.
  - **`z` on repair and replacement days, for every model**, as in `main`'s rainflow.
- **Decision (26 Oct 2026, chosen in the answer menu)**: `z` on repair days only, for every model. A replacement costs
  `C_rep`.
- **To implement**:
  - Code: the rainflow builder drops the row `r_k = 1 ⇒ z_k = mu_{k-1} - mu_k` and sets `z_k = 0` on replacement days.
  - `spec/spec.tex`: the repair-cost variable is defined on repair days only, for every model.

### Depot days, idle days and the meaning of `C_M`

- **`lore`** code and spec: at most one activity per train and day; an idle day is a row of zeros; `C_M` is paid once
  per train per depot day.
- **`lore`** scoping document: one activity per train and day (equality), plus a maintenance-day variable `y_ik`.
- **`main`**: `idle + m + r = x[i,0,k]`, with an explicit idle binary; `C_M[l]` is paid per repaired component.
- **Check of both versions** (26 Oct 2026):
  - Both codes use `Σ_j x_ijk <= 1` (lore:src/fleet_management/models/base.py:40-45;
    main:src/fleet_management/degradation_model/base.py:1676-1679).
  - `lore`'s spec: a row of zeros is an idle day, and `x_i0k = 1` is a maintenance day, which repairs and
    replacements require; `C_M·Σ x_i0k` is then a real decision cost (lore:spec/spec.tex:560-572).
  - `lore`'s scoping document writes the equality `Σ_j x_ijk = 1`, so `x_i0k` is a depot day, maintenance or idle, and
    a new variable `y_ik <= x_i0k` flags a maintenance day (lore:scoping.tex:125-146). Its text says "at most one
    activity per day" but its equation is an equality. The two conventions are equivalent: the scoping's `y_ik` is
    the spec's `x_i0k`. The spec's needs fewer variables.
  - `main`'s objective charges `C_M[l]·m_ilk` (main:src/fleet_management/degradation_model/base.py:1749-1766). A
    depot day with no repair costs nothing, like a row of zeros.
  - The two cost models price different things (Claude's reading): `lore`'s `C_M` prices a workshop visit (vehicle
    out of service, set-up), whatever the number of jobs; `main`'s `C_M[l]` prices each job on a component. Example:
    three components repaired on one depot day cost `C_M` on `lore`, `C_M[0] + C_M[1] + C_M[2]` on `main`. Real
    workshops have both kinds of cost.
- Options for the cost:
  - **Both terms: a cost per train per depot day (as on `lore`) and a cost per repaired component (as on `main`); each
    may be 0** (recommended). This ports `lore`'s term and keeps `main`'s.
  - **`lore`'s per-visit cost only.**
  - **`main`'s per-component cost only.**
- Options for the documents:
  - **The spec's convention** (`Σ_j x_ijk <= 1`, a row of zeros is an idle day, `x_i0k` is a depot day), used by both
    codes; the scoping document is aligned to it (recommended).
  - **The scoping document's convention** (equality plus `y_ik`); the spec and the code are aligned to it.
- **Decisions (26 Oct 2026, chosen in the answer menu)**:
  - Cost: `lore`'s per-visit cost only: `C_M` once per train per depot day (`x_i0k = 1`), whatever the number of jobs.
    `main`'s per-component `C_M[l]` is dropped. This is an explicit exception, by the user's choice, to the policy of
    keeping `main`'s features.
  - Documents: the spec's convention (`Σ_j x_ijk <= 1`, a row of zeros is an idle day, `x_i0k` is a depot day).
- **To implement**:
  - Code: the objective term `C_M·Σ_{i,k} x_i0k`; `main`'s per-component term `C_M[l]·m_ilk` removed; repairs and
    replacements allowed only on depot days, as on both branches.
  - `spec/spec.tex`: unchanged on this point (it already states the convention and the term).
  - `scoping.tex`: the one-activity row becomes `Σ_j x_ijk <= 1`; `y_ik` is removed and `C_M` charges `x_i0k`
    (lore:scoping.tex:125-146).

### Cost shapes and objective terms

- **`lore`**: `C_R`, `C_rep` scalar, (F,) or (F, L); `C_M` scalar; damage penalty `inf_norm` or `quadratic`; one
  objective.
- **`main`**: `C_M`, `C_R`, `C_rep` scalar or (L,); a default `C_rep`; damage penalty `inf_norm` only; objective
  modes `total`, `operating_average` and `evaluation_total`; `C_P` only in the legacy Gamma backend.
- **Check of both versions** (26 Oct 2026, code read):
  - `main`'s costs are per component, "intentionally independent of the vehicle index", finite and non-negative
    (main:src/fleet_management/config.py:218-238). `lore` accepts per-train costs (F,) and per-train-and-component
    costs (F, L) (lore:src/fleet_management/solver.py:366-375).
  - The union of the shapes is ambiguous when `F = L`. Example: 4 trains, 4 components, `C_R: [1, 2, 3, 4]`; `lore`
    reads it per train, `main` per component.
  - `main`'s default `C_rep = C_R · max_i tau` when the key is missing (main:src/fleet_management/degradation_model/base.py:1272-1273).
    `lore` requires `C_rep`. A silent default for a cost changes the optimal schedule without the user seeing it.
  - The quadratic damage penalty `u >= Σ_l mu²` exists only on `lore` (lore:src/fleet_management/models/base.py:103-118),
    so the policy ports it. On `lore` it requires the exact reliability form (lore:src/fleet_management/solver.py:155-156).
  - `main`'s objective modes are sound (main:src/fleet_management/degradation_model/base.py:1735-1811): `total`;
    `operating_average = J_op/H2 + C_D·u`; `evaluation_total = J_init + ((E - H1)/H2)·J_op + C_D·u`. The policy keeps
    them; no question is needed.
- Options for the shapes of `C_R` and `C_rep`:
  - **Scalar, (L,) or (F, L); (F,) is dropped because it is ambiguous when `F = L`** (recommended). A per-train cost
    is written as (F, L).
  - **Scalar, (F,), (L,) or (F, L)**, with a rule for `F = L`.
  - **`main`'s shapes only**: scalar or (L,); `lore`'s per-train costs are lost.
- Options for a missing `C_rep`:
  - **Required, as on `lore`** (recommended).
  - **`main`'s default** `C_R · max_i tau`.
- Options for the quadratic damage penalty:
  - **Ported from `lore`, with its requirement of the exact reliability form** (recommended).
  - **Not ported.**
- **Decisions (26 Oct 2026, chosen in the answer menu)**:
  - `C_R` and `C_rep`: scalar, (L,) or (F, L). The shape (F,) is not accepted.
  - `C_rep` is required.
  - The quadratic damage penalty is ported, with its requirement of the exact reliability form; `inf_norm` stays the
    default.
  - `main`'s three objective modes stay (policy).
  - `C_M`: a scalar, charged once per train per depot day (item "Depot days, idle days and the meaning of `C_M`").
- **To implement**:
  - Code: cost parsing for the three shapes, with the non-negative check of `main`; an error for a missing `C_rep`;
    the `penalty_type` option and its check.
  - `spec/spec.tex`: the cost shapes, the objective with both penalty types, and the three objective modes.

### Rainflow tail bounds

- **`lore`**: Cantelli and Bernstein (`tail_bound`); Bernstein side row `mu <= tau - c`; support constant
  `max(b_0, b_new, max b_inc)`; Bhatia-Davis consistency checks.
- **`main`**: Markov, Cantelli, Hoeffding, Bernstein and Chernoff (`bound_method`); side row `mu <= tau`; support
  constant: the largest `support` value.
- **Check of both versions** (26 Oct 2026; `d = tau - mu`, `kappa = Le = ln(1/eps)`, `c = kappa·b/3`):
  - The five bounds are kept by the policy. Markov `mu <= eps·tau` and Hoeffding `d² >= 0.5·Le·R` are the standard
    forms (Claude's check); Cantelli is identical on both branches; Chernoff is valid once Defect B is fixed (item
    "Replacement").
  - Bernstein, side row. Both quadratic rows are the same: `d² - 2c·d - 2·kappa·v >= 0`. Its roots are
    `d± = c ± sqrt(c² + 2·kappa·v)`, with `d- <= 0`. With `main`'s side row `d >= 0`, the feasible set is
    `d >= d+`, plus the single point `d = 0, v = 0` (`mu = tau`, a known damage equal to `tau`). That point is safe,
    since a known damage equal to `tau` never exceeds it, but it makes the feasible set disconnected. `lore`'s side
    row `d >= c` (`mu <= tau - c`) removes it. Both select the branch `d >= d+`.
  - Bernstein, support constant. Bernstein's inequality needs a bound `b` for every term of the sum. `main` takes the
    largest increment support only (main:src/fleet_management/degradation_model/rainflow.py:130, 450), but its tracked
    variance starts at `v_0` (line 237). So a random initial damage (`v_0 > 0`) is a term without a support bound,
    and the certificate is not proven. The same holds for a random replacement state (`v_new > 0`), which the merge
    allows (item "Replacement"). `lore` uses `b = max(b_0, b_new, max b_inc)` (lore:src/fleet_management/models/rainflow.py:26-28).
  - Support semantics: `lore`'s `b_inc` is an upper bound of the increment; `main`'s `support` is the width of its
    range. For Bernstein both bound the centred term; Hoeffding needs the width. For a non-negative increment whose
    range starts at 0, the two numbers coincide.
  - `lore` checks the Bhatia-Davis consistency of the moments (`v <= mu·(b - mu)` for increments, initial and
    replacement states) and the precondition `tau > 2·kappa·b/3`, without which no positive variance is feasible
    (lore:src/fleet_management/models/rainflow.py:66-102). `main` has neither. These are ported by the policy; their
    details go to the item "Input checks".
- Options for the Bernstein side row:
  - **`lore`'s `mu <= tau - c`** (recommended): same safe set, without the isolated point.
  - **`main`'s `mu <= tau`.**
- Options for the Bernstein support constant:
  - **`b = max(b_0, b_new, max over increments)`, with optional supports of the initial and replacement damage
    (default 0, for a known damage)** (recommended). This fixes the gap above and ports `lore`'s inputs.
  - **`main`'s increment supports only**: the certificate is not proven when `v_0 > 0` or `v_new > 0`.
- Reason for the recommendations (Claude's own reasoning): `lore`'s versions are the mathematically clean ones, and
  `main`'s extra bounds stay.
- **Decisions (26 Oct 2026, chosen in the answer menu)**:
  - The five bounds stay (policy).
  - Bernstein side row: `lore`'s `mu <= tau - c`.
  - Bernstein support constant: `b = max(b_0, b_new, max over increments)`, with optional supports of the initial
    and replacement damage, default 0 (known damage).
- **To implement**:
  - Code: the Bernstein side row in every form (exact, linear); the support constant with the two new optional
    inputs, named in the item "Input key names, shapes and axis order".
  - `spec/spec.tex`: the five bounds, each with its row, its tracked quantities and its validity argument; the
    Bernstein constant with the initial and replacement terms.

### Linear forms of the reliability rows

- **`lore`**: `formulation` is `exact` or `lp` (the tangent at `mu = 0`); an unknown value raises an error.
- **`main`**: `reliability_impl` is `exact`, `tangent` (at `tangent_ref·tau`, default 0.5) or `pwl` (`pwl_points`
  segments, default 8); an unknown value becomes `exact` without a warning.
- **Check of both versions** (26 Oct 2026; code read: main:src/fleet_management/degradation_model/rainflow.py:466-558):
  - Every quadratic row has the form `Q <= f(mu)` with `f` convex in `mu`: Cantelli and the Gaussian/Wiener row
    (`v <= (tau - mu)²/kappa`), Bernstein (`v <= (d² - 2c·d)/(2·kappa)`), Hoeffding (`R <= 2·d²/Le`). A tangent of a
    convex function lies below it, so a cap by any tangent is safe (an inner approximation) (Claude's derivation).
  - `lore`'s `lp` is the tangent at `mu = 0`; `main`'s `tangent` is the tangent at `tangent_ref·tau`. So `lore`'s form
    is `main`'s with `tangent_ref = 0`.
  - `main`'s `pwl` splits `[0, tau]` into `pwl_points` segments; a binary per step selects the segment that contains
    `mu`, and the tangent at the segment midpoint caps `Q`. Every tangent lies below `f`, so the form is safe.
  - Defect: an unknown value of `reliability_impl` becomes `exact` without a warning
    (main:src/fleet_management/degradation_model/rainflow.py:561-570), so a typo turns the model into a nonconvex one.
    The fallback for Markov and Chernoff, which are linear already, is legitimate. The policy fixes the defect with an
    error, as on `lore`.
  - Scope: `main` applies the forms to rainflow only; `lore` applies its `lp` to Gaussian, Wiener and rainflow.
- **Worked example** (Cantelli, `eps = 0.1`, so `kappa = 9`; `tau = 1`; `v = 0.004`). Largest admissible mean:
  exact `1 - sqrt(9·0.004)`, about 0.81; tangent at 0, `9v + 2·mu <= 1`, about 0.48; tangent at `0.5·tau`,
  `9v <= 0.75 - mu`, about 0.71; pwl with 8 segments, close to 0.81. The tangent at 0 never admits a mean above
  `tau/2`, even with zero variance.
- Options for the set of forms:
  - **`exact`, `tangent` and `pwl` for every quadratic row** (Gaussian, Wiener, Cantelli, Bernstein, Hoeffding)
    (recommended). `lore`'s `lp` is `tangent` with `tangent_ref = 0`.
  - **`main`'s forms for rainflow only; `lore`'s `lp` for Gaussian and Wiener.**
- Options for the default tangent point:
  - **`0.5·tau`, as on `main`** (recommended): much less conservative in the mean (example above).
  - **`0`, as on `lore`.**
- **User's answer (26 Oct 2026)**: option 1, and *"the linearity is not necessarily at tangent_ref = 0. It can be
  computed at different points, right? I would opt for this info given as input and the linear constraint is computed
  accordingly"*; default `0.5·tau`.
- **Claude's answer**: yes. `main`'s `tangent_ref` is already that input: the tangent point is `mu_r = tangent_ref·tau`
  and the row is `Q <= f(mu_r) + f'(mu_r)·(mu - mu_r)`, safe at any point because `f` is convex. Two refinements were
  proposed: a per-cell input, since the best point is near each component's operating mean; and a range check
  `0 <= tangent_ref < 1` instead of `main`'s silent clipping to [0, 1] (at `tangent_ref = 1` the tangent is `Q <= 0`).
- **Decisions (26 Oct 2026)**:
  - `exact`, `tangent` and `pwl` for every quadratic row: Gaussian, Wiener, Cantelli, Bernstein, Hoeffding. An unknown
    value raises an error; Markov and Chernoff, linear already, accept only `exact`.
  - The tangent point is the input `tangent_ref`, default 0.5; the row is computed at `tangent_ref·tau`.
  - `tangent_ref` is a per-cell input: scalar, (L,) or (F, L) (answer menu, 26 Oct 2026).
  - A value outside `0 <= tangent_ref < 1` raises an error (answer menu, 26 Oct 2026).
- **To implement**:
  - Code: the three forms in every builder with a quadratic row; `lore`'s `lp` becomes `tangent`; the error for an
    unknown value.
  - `spec/spec.tex`: the three forms, the tangent formula, the safety argument (tangent of a convex function), and the
    pwl encoding.

### Reliability level `epsilon`

- `lore`: one scalar in (0, 0.01].
- `main`: one value per cell, in (0, 1).
- **Check of both versions** (26 Oct 2026):
  - `lore`'s spec gives two reasons for `eps <= 0.01` (lore:spec/spec.tex:215-218, 754). Correctness: the squared
    Gaussian and Wiener row needs `Φ^{-1}(1 - eps) > 0`, which holds for `eps < 0.5`. Numerics: the variance big-M
    `V_max = tau²/Φ^{-1}(1 - eps)²` grows without bound as `eps -> 0.5`.
  - `V_max/tau²` is 0.18 at `eps = 0.01`, 0.61 at 0.1, 3.6 at 0.3 and about 1600 at 0.49. `V_max` is the largest
    variance that the reliability row allows, so it is the variable's natural range: the "tight M" case of Gurobi's
    guideline (item "Encoding of the dynamics"). The numerical concern is real only near 0.5 (Claude's reading).
  - The other models have no such limit: Cantelli's variance cap `tau²·eps/(1 - eps)` is finite for every `eps < 1`;
    Markov, Hoeffding, Bernstein, Chernoff, Gamma and IG are defined for every `eps` in (0, 1).
  - `main`'s README example uses `eps = 0.1` (main:README.md:121), which `lore`'s rule rejects.
  - A per-cell value lets critical components carry a smaller `eps`; `main` has it, so the policy keeps it.
- Options:
  - **Per cell (scalar, (L,) or (F, L)), in (0, 1); Gaussian and Wiener cells need `eps < 0.5`** (recommended).
  - **Per cell, in (0, 1); Gaussian and Wiener cells keep `lore`'s `eps <= 0.01`.**
  - **One scalar in (0, 0.01] for every model, as on `lore`.**
- **Decision (26 Oct 2026, chosen in the answer menu)**: `eps` per cell (scalar, (L,) or (F, L)), in (0, 1) for every
  model; Gaussian and Wiener cells need `eps < 0.5`.
- **To implement**:
  - Code: `main`'s per-cell parsing and range check; an extra check `eps < 0.5` at Gaussian and Wiener cells.
  - `spec/spec.tex`: the input table and the consistency checks; the remark on `V_max` as the natural range.

### Input checks

- `lore`: `F > M`, `mu_0 < tau`, `C_D > 0`, `epsilon <= 0.01`, and per-model checks such as Bhatia-Davis and
  `tau > 2·kappa·b/3` for Bernstein.
- `main`: none of the first four; checks only rainflow and Gamma cells; caps `mu` at `tau` through the variable bound.
- **Check of both versions** (26 Oct 2026):
  - Ported from `lore` by the policy, with no conflict: `F > M` (with `F = M` no train can ever be in the depot, so no
    repair is possible); `mu_0 < tau`; `rho` in (0, 1]; the Gaussian three-sigma checks `mu_0 >= 3·sqrt(v_0)` and
    `mu_inc >= 3·sqrt(v_inc)` (a Gaussian increment is then negative with probability about 0.13 %); `mu_inc < tau`;
    `eta > 0`; `sigma > 0`; the Bhatia-Davis checks `v <= mu·(b - mu)` for increments, initial and replacement states;
    the Bernstein precondition `tau > 2·kappa·b/3` (lore:spec/spec.tex:372-470).
  - Kept from `main` by the policy, with no conflict: `tau > 0`; finite, non-negative costs; `s_chernoff >
    ln(1/eps)/tau`; the Gamma checks (item "Gamma formulation").
  - Decided in other items: the range of `eps`; the starting anchor; `tangent_ref`; unknown keys and the axis order
    (item "Input key names, shapes and axis order").
  - Conflict 1, `C_D`: `lore` requires `C_D > 0`, "for tightness of the one-sided big-M relaxation"
    (lore:spec/spec.tex:438-440); `main` has no sign check. Safety does not need it (slack remark), but it keeps the
    reported states near the true ones, which matters because the output reports the tracked states (no replay).
    A negative `C_D` would reward damage.
  - Conflict 2, increments: `main` requires `mu > 0` at every cell and mission (main:src/fleet_management/config.py:474-475),
    and `v`, `support`, `cgf` positive. So a mission that does not wear a component cannot be written as 0. Zero is
    harmless: Bhatia-Davis then forces `v = 0`, and the CGF of 0 is 0. `lore` accepts any `mu_inc < tau`.
  - Conflict 3, Wiener `v_0`: `lore` requires `v_0 > 0` (lore:spec/spec.tex:399), with no stated reason; a known
    initial damage (`v_0 = 0`) is a valid input.
- Options:
  - `C_D`: **`> 0`, as on `lore`** (recommended); `>= 0`; no check, as on `main`.
  - Increments: **`>= 0`, with zero variance, support and CGF allowed where the mean is 0** (recommended); `> 0`, as
    on `main`.
  - Wiener `v_0`: **`>= 0`** (recommended); `> 0`, as on `lore`.
- **Decisions (26 Oct 2026, chosen in the answer menu)**:
  - `C_D >= 0`. User: *"if C_D = 0 implies a simplified problem (like some constraints are useless) I want them not to
    be added to the problem in that case"*. So with `C_D = 0` the variable `u` and its rows (one per train and day)
    are not built, and `penalty_type` is ignored. Reported states may then carry more slack.
  - Mission increments `>= 0`; where the mean is 0, the variance, support and CGF must be 0 too.
  - Wiener `v_0 >= 0`.
  - All other checks listed above: ported from `lore` or kept from `main`, by the policy.
- **To implement**:
  - Code: one validation function per model, with the checks above; the conditional construction of `u`.
  - `spec/spec.tex`: the consistency-check list, with these values.

### Input key names, shapes and axis order

- Axis order of the increment tensors: `lore` (F, M, L, H); `main` (F, L, M, H).
- Key names: `maintenance_type` or `repair_model`; `tail_bound` or `bound_method`; `formulation` or
  `reliability_impl`; `mu_new` or `replacement_mu`; `beta` or `gamma_beta`.
- Shapes: `lore` accepts (F, L) and `null` entries; `main` accepts scalar, (L,) and (F, L), and aliases.
- Only on `main`: transitory profiles (`*_trans`) and `component_names`.
- HDF5 input: `lore` reads every key; `main` reads a fixed list without `model`, so no HDF5 input loads.
- **Facts checked** (26 Oct 2026):
  - `main`'s `load_config` accepts unknown keys without an error, and accepts a `lore` tensor (F, M, L, H) as
    (F, L, M, H) when `M = L` (item "Meaning of `H: [a, b]`").
  - Aliases on `main`: `method` for `bound_method`, `xi` and `repair_rho` for `rho`, `mu_new` and `v_new` for the
    replacement state, `C_S` for `C_D` (main:src/fleet_management/config.py:349-406). Tracked inputs on `main` use
    `method` (3 files), `xi` (3), `C_S` (4); `C_P` (9) and `alpha` (15) appear only in old-format inputs and fixtures.
  - Some `lore` names cannot be aliases, because their values mean something else on `main`: `maintenance_type: ARD1`
    is `main`'s `ardinf`; `beta` is a scale, `main`'s `gamma_beta` a rate; `formulation: lp` is `reliability_impl:
    tangent` with `tangent_ref = 0`. An alias would change the model without a warning.
  - Every `lore` input contains `maintenance_type`, so an error on unknown keys stops every old `lore` file before
    its axis order is misread.
- **Proposed key table** (Claude's proposal; one name per concept; `main`'s names where they exist, since the merge
  lives on `main` and its inputs use them; Gamma keys follow the item "Gamma formulation"):

| Group | Keys |
|---|---|
| Sizes and horizon | `F`, `M`, `L`, `H` (int or `[H1, H2]`) |
| Selectors, per cell | `model`, `repair_model` (`ard1`, `ardinf`), `bound_method` (rainflow), `reliability_impl` (`exact`, `tangent`, `pwl`), `encoding` (`bigm` default, `indicator`), `component_names` |
| Reliability | `tau`, `epsilon`, `tangent_ref` (per cell), `pwl_points` |
| Repair | `rho`; starting anchors `anchor_mu_0`, `anchor_v_0`, `anchor_R_0`, `anchor_K_0` (default: the initial value of each state) |
| Initial state | `mu_0`, `v_0`, `support_0` (Bernstein), `R_0` (Hoeffding), `cgf_0` (Chernoff, default `s_chernoff·mu_0`) |
| Replacement | `replacement_mu`, `replacement_v`, `support_new`, `R_new`, `cgf_new` (default `s_chernoff·replacement_mu`), `replacement_as_new` (default false), `allow_replacement` (default true) |
| Increments, (F, L, M, H) with broadcasting | `mu`, `v`, `support`, `cgf`, and `*_trans` variants; `s_chernoff` per cell |
| Model constants, per cell | `sigma` (Wiener), `eta` (IG), `v_max_user` (Gaussian, Wiener, rainflow) |
| Costs | `C_M` (scalar, per depot day), `C_R`, `C_rep` (scalar, (L,) or (F, L)), `C_D` (>= 0), `penalty_type` (`inf_norm`, `quadratic`) |
| Run options | `verbose`, `mip_gap`, `time_limit`, `fast`, `gurobi_params`, `objective_mode`, `evaluation_horizon`, `progress_interval_seconds`, `relaxation_warm_start`, `relaxation_time_limit` |
| Removed (error with a message naming the replacement) | `maintenance_type`, `tail_bound`, `formulation`, `beta`, `alpha_inc`, `b_inc`, `b_0`, `b_new`, `n_workers`, `warm_start` (sweep options), `depot_capacity`, `C_P`, `C_S`, `alpha`, `xi`, `repair_rho`, `method`, `mu_new`, `v_new` |

  - The rainflow `support` keeps `main`'s meaning, the width of the increment's range; for Bernstein it bounds the
    centred term (item "Rainflow tail bounds").
- Options:
  - Unknown and old keys: **an error naming the canonical key and, where the meaning changed, the new meaning**
    (recommended); a warning; `lore`'s names as aliases where the meaning is unchanged.
  - `null` entries at cells where a model-specific key does not apply: **accepted, as on `lore`** (recommended);
    rejected, as on `main`.
  - The key table: **as proposed** (recommended), or edited by the user.
- **Decisions (26 Oct 2026, chosen in the answer menu)**:
  - Unknown and old keys raise an error that names the canonical key and, where the meaning changed, the new meaning.
  - `null` entries are accepted at cells where a model-specific key does not apply.
  - The key table above is adopted. The Gamma keys follow the item "Gamma formulation".
  - The canonical axis order of increment tensors is `main`'s (F, L, M, H); old `lore` files stop at the unknown-key
    error before their axis order is read.
  - The rule "no useless variables or rows" extends to every objective term with coefficient 0 (section "General
    rules for the merge").
- **To implement**:
  - Code: `load_config` with the table, the error for unknown and old keys, `null` handling, and broadcasting;
    `main`'s tracked inputs that use `method`, `xi` or `C_S` are rewritten; the HDF5 reader reads every key, as on
    `lore`, so that HDF5 inputs load.
  - `spec/spec.tex`: the input tables and the schema section, with the table above.
  - `README.md`: the input section.

### Solver defaults

- `lore`: Gurobi's default MIP gap; time limit 3600 s.
- `main`: MIP gap 0.12; no time limit; a `fast` preset, `gurobi_params` and a progress callback.
- **Check of both versions** (26 Oct 2026):
  - `main`'s default `mip_gap = 0.12` (main:src/fleet_management/degradation_model/base.py:1209) lets Gurobi stop and
    report status "optimal" with a solution up to 12 % above the best bound. Gurobi's own default gap is `1e-4`.
  - `main`'s default `time_limit = None` lets a solve run without end; `lore`'s default is 3600 s.
  - `main`'s horizon sweep counts a case as proven only if the gap is at most `1e-8`
    (main:src/fleet_management/horizon_sweep.py:99-129). With a MIP gap of `1e-4`, a case reported "optimal" can still
    fail that test, so the two thresholds should agree.
  - `fast`, `gurobi_params` and the progress callback exist only on `main`; the policy keeps them. No defect found.
- Options:
  - MIP gap default: **Gurobi's default `1e-4`** (recommended); `0.12`, as on `main`.
  - Time limit default: **3600 s, as on `lore`** (recommended); none, as on `main`.
  - Proof threshold of the sweep: **equal to the MIP gap used for the case** (recommended); `1e-8`, as on `main`.
- **Decisions (26 Oct 2026, chosen in the answer menu)**:
  - MIP gap: default 0.1; the user sets another value with `mip_gap`. (User's own option, between the two offered.)
  - Time limit: default 3600 s.
  - The sweep counts a case as proven when Gurobi reports "optimal" at the MIP gap used for that case.
  - `fast`, `gurobi_params` and the progress callback stay (policy).
- **To implement**:
  - Code: the two defaults in `resolve_run_options`; the proof test in `horizon_sweep._select_horizon_cases`.
  - `spec/spec.tex`: the run options and their defaults; the proof rule of the sweep.

### Gamma formulation

- `lore`: shape-scale convention; one scale per component; exact cap `alpha <= alpha_hat`; inputs `beta`,
  `alpha_inc`.
- `main`: shape-rate convention; a modular route with a common-rate calibration, which certifies repetitions of one
  increment type only, and a legacy route; inputs `gamma_beta`, `gamma_beta_0`, `gamma_beta_new`,
  `gamma_beta_bound`.
- **Analysis** (an agent, 26 Oct 2026; notes in [`analysis/gamma.md`](analysis/gamma.md), derivations and
  standard-library scripts; Claude checked the proof of S5 step by step and reproduced the counterexamples of S2 and
  S3 by an independent Monte Carlo script, with matching results):
  - S1: with rates `b_i >= b*`, `Σ X_i <=st Gamma(Σ a_i, b*)` (pathwise coupling). Safe, but very conservative when
    the rates differ.
  - S2, `main`'s `repeated_increment` calibration is **not safe**. Example: `tau = 1`, `eps = 0.01`; type 1 mean 0.1,
    rate 10; type 2 mean 0.1, rate 200; cap 3.9717. The mix (1 of type 1, 7 of type 2) has surrogate shape 3.30 and
    is accepted; its true tail is 0.059 (Claude's Monte Carlo: 0.0594 ± 0.0002). The mix (3, 2): 0.0142 (Claude:
    0.0143). A grid search finds accepted unsafe schedules in 360 of 441 parameter sets. The model has no row that
    limits a type to its certified count.
  - S3, repair of the shape in parameter space (both branches), whole-state law: the true state is below the
    surrogate in convex order, the repair day is safe, but later days are not. Example: `eps = 0.01`, `rho = 0.5`,
    rate 0.01; a mission at the cap, a repair, a mission of half the cap: day-3 true tail 0.0133 (Claude: 0.01325 ±
    0.00008). Unsafe cases need a cap shape at most 0.11 (`b·tau <= 0.3`); with periodic repetition the excess grows
    to 1.84·`eps` after 4 cycles. Conjecture: safe for `b·tau >= 1`.
  - S4, the since-last law in parameter space: the same verdict (day-3 tail 1.25·`eps` with an anchor at 0.25 of the
    cap).
  - Inputs on `main` (checked 26 Oct 2026): mostly `gamma_beta_bound: 10`, `gamma_beta: 20`, `tau = 0.6`, so a common
    rate below the missions' own rate. The analysis finds accepted unsafe schedules in 27 of 90 such single-type
    parameter sets (tails 0.0103 to 0.49). So **`main`'s Gamma experiment results may contain schedules less safe than
    reported**.
- **Proposed safe formulation** (the agent's proposal, S5):
  - Track `A' = b*·mu`, with `mu` the tracked mean (linear dynamics, both repair laws, one-sided rows) and `b*` the
    smallest rate of all increments, the initial state and the replacement state.
  - Row `A' <= A_icx`, a constant computed offline from `B(A_icx) = eps`, where `B(A)` is the smaller of the
    stop-loss bound `inf_{x < tau} E(Gamma(A, b*) - x)_+/(tau - x)` and the Chernoff bound.
  - Theorem: on every day of every repetition, `D_k <=icx Gamma(A'_k, b*)`, so `P(D_k > tau) <= eps`. Steps: (L1)
    `c·X <=cx Gamma(c·a, b)`, by `E[U | U + V] = c·(U + V)` for independent Gammas and Jensen; (L2) an increment of
    rate `b_q >= b*` is `<=cx Gamma(b*·m_q, b*)`, its mean-matched law; (L3) the increasing convex order survives
    sums, scaling, both repair laws and a larger shape. Claude checked each step.
  - Price: allowed shape relative to the exact one-rate cap, `eps = 0.01`: 0.04 at `b*·tau = 0.01`, 0.35 at 1, 0.81
    at 10, 0.96 at 100. With `main`'s inputs (`b*·tau` about 6): about 0.7.
  - For `lore` (one rate) only the cap changes; for `main` the calibration and the bounding shape go.
- **Other options** (from the analysis):
  - Two tracked shapes with a disjunction: the un-repaired sum with the exact cap, or `A'` with `A_icx`. Tighter, but
    it adds one binary per day and needs a replacement in every period on the days that use the first shape.
  - The exact cap with parameter-space repair, limited to `b·tau >= 1`: rests on the S3 conjecture only.
  - Chernoff at one fixed `s` with the exact cumulant, as `main`'s rainflow Chernoff: safe, no mean matching, looser.
  - No imperfect repair for Gamma (replacement only): exact with one rate, safe by S1 with several rates.
- Options for the formulation:
  - **S5, the convex-order cap** (recommended): proven safe for every rate mix and both repair laws, linear.
  - **S5, plus the two-shape disjunction as an option.**
  - **`lore`'s exact cap with an input check `b·tau >= 1`**: rests on a conjecture.
  - **`main`'s calibration as it is**: unsafe.
- Options for `main`'s other Gamma code: **the `repeated_increment` and `finite_count` calibrations and the legacy
  constant-rate route leave the solve path** (recommended; S5 supersedes them, and the legacy route ignores `H2` and the
  repair law); kept as options.
- Input names follow `main` (`mu`, `gamma_beta`, `gamma_beta_bound`, `gamma_beta_0`, `gamma_beta_new`): the
  formulation needs means and rates only.
- **User's questions (26 Oct 2026)**: *"among the first three options, is there a tighter one? Is there one that adds
  fewer constraints? Check the safety of lore's option. Than we decide."* and, on `main`'s other code: *"Are we talking
  about how to reshape all the Gammas to the same beta? Or is something else? If it is the reshaping, are you saying
  that the calibration is wrong?"*
- **Claude's answers (26 Oct 2026)**:
  - Yes: `main`'s calibration is the reshaping of each mission type to the common rate `b*`, with a shape that matches
    the tail at `tau` of repeated copies of that one type. A schedule mixes types, and the match does not survive the
    sum: the type with mean 0.1 and rate 200 gets shape 0.3287 at rate 10, a mean of 0.033 instead of 0.1. So the
    calibration is wrong for mixes (counterexample S2), as its own documentation concedes, and nothing limits a type
    to its certified count. The legacy route is a different thing: `main`'s old one-rate backend. S5 also reshapes,
    but it keeps each increment's mean, which is provably an upper bound in convex order.
  - Constraints: S5 and `lore`'s option have one tracked state and one cap row per cell and day, as `lore` today; the
    two-shape option adds a state, a binary and rows per day.
  - Tightness: `lore`'s option, if safe, is the tightest; then the two-shape option; then S5.
  - A fourth option (Claude's idea): S5's tracking with the sharpest safe cap `A_sharp`, the largest shape whose worst
    tail over the whole class of true states is at most `eps`. Same constraints as S5; if `lore`'s option is safe for
    `b*·tau >= 1`, `A_sharp` equals the exact cap there, for any rate mix.
  - The safety of `lore`'s option is open: unsafe in general (S3), conjectured safe for `b·tau >= 1`. Two agents were
    started on 26 Oct 2026: one reads the literature on extremal tails of weighted sums of independent Gamma
    variables (notes: `analysis/gamma_literature.md`), one attempts a proof or a counterexample and computes `A_sharp`
    (notes: `analysis/gamma_sharp.md`).
- **Literature result** (an agent, 26 Oct 2026; notes in [`analysis/gamma_literature.md`](analysis/gamma_literature.md)):
  - Read in full by the agent: Székely and Bakirov (2003); Roosta-Khorasani, Székely and Ascher (2015, arXiv
    1404.0122); Yu (2017, arXiv 1607.02689); Roosta-Khorasani and Székely (2015, *Metrika*, arXiv 1601.04731); Janson
    (2018, arXiv 1709.08157). Not read: Diaconis and Perlman (1990), behind a human-verification check (the user may
    retrieve it: https://doi.org/10.1214/lnms/1215457557); Bock et al. (1987) and Kochar and Xu (2011), no access.
  - Key theorem, checked by Claude on pages 2-4 of arXiv 1601.04731 (Roosta-Khorasani and Székely 2015, Theorem 1):
    for `X_i` iid `Gamma(alpha, beta)` and weights with equal sums `s`, if `lambda` majorizes `mu`, then
    `P(Σ lambda_i X_i > x) >= P(Σ mu_i X_i > x)` for all `x > (2·alpha + 1)·s/(2·beta)`, the mean times
    `1 + 1/(2·alpha)`. (The paper's `P` is the distribution function `Pr(Σ lambda_i X_i < x)`.)
  - Mapping to our class (Claude's check of the agent's step): if every shape and `b·m` are multiples of one `alpha`,
    the single Gamma at the cap majorizes every member, so `lore`'s cap is safe for `b·tau > A + A/(2·alpha)`, with
    `A` the cap shape. Example: integer shapes (`alpha = 1`), `eps = 0.01`, `b·tau = 10`: `A = 3.97`, threshold
    5.96 < 10, safe. For arbitrary shapes `alpha` can be tiny and the threshold grows without bound.
  - So the literature does not settle the whole class. The counterexample's two tails cross at `b·tau = 0.3045`
    (agent's computation).
- **Sharp-cap analysis** (an agent, 26 Oct 2026; notes in [`analysis/gamma_sharp.md`](analysis/gamma_sharp.md)). Claude
  reproduced every number with independent code ([`analysis/scripts/verify_gamma_sharp/`](analysis/scripts/verify_gamma_sharp/))
  and checked the proof step by step. Unimodality from the literature: [`analysis/gamma_unimodal.md`](analysis/gamma_unimodal.md)
  (another agent; Claude checked the statement and the proof it cites).
  - Class of true states: in units of `1/b*`, every true state of a schedule is a constant plus `Σ c_i·Y_i`, with
    `Y_i ~ Gamma(a_i, 1)` independent and `c_i ∈ (0, 1]`. Rates above `b*` and repairs give `c_i < 1`.
  - Theorem (Claude's own, by the agent): at a given mean, the largest tail over the class is reached by one term
    `c·Gamma(A/c, 1)`. Proof: first-order conditions on the terms, a count of sign changes of a Laplace transform, and
    the unimodality of sums of Gammas, a published theorem (Roosta-Khorasani and Székely 2015, *Metrika* 78, Theorem 4;
    proof by Székely and Bakirov 2003 through Yamazato 1978). No source read proves the theorem itself.
  - So the sharpest safe cap is a one-dimensional computation: `A_1(t, eps)`, the largest `A` with
    `max over c ∈ (0, 1] of Q(A/c, t/c) <= eps`, at `t = b*·tau`.
  - **`lore`'s exact cap is unsafe below a threshold `t1(eps)` of `b*·tau`**: 0.436, 0.448 and 0.614 for `eps` = 0.001,
    0.01 and 0.1. Witness: `eps = 0.01`, `b*·tau = 0.1`, one mission with rate `4.6·b*` filled to the cap: tail
    1.56·`eps`. With one rate (`lore`'s own setting), the repair counterexample S3: 1.33·`eps` at `b·tau <= 0.3`.
  - **Above the threshold, `A_1` equals `lore`'s exact cap**, for any mix of rates (checked on 121 values of `b*·tau` up
    to 1000; a per-cell computation of `A_1` needs no such check).
  - Below the threshold, the mean cap `A_1/b*` is a fixed fraction of `tau`: 0.00355, 0.0353 and 0.331 for `eps` =
    0.001, 0.01 and 0.1. Reason: `Q(A/c, t/c) = Q((A/t)·s, s)` with `s = t/c >= t`. For the same reason the mean cap
    never decreases when `b*` grows, so the best safe `b*` is the smallest rate of the cell.
  - Allowed shape relative to `lore`'s exact cap, `eps = 0.01`:

    | `b*·tau` | 0.01 | 0.1 | 0.3 | 1 | 3 | 10 | 100 |
    |---|---|---|---|---|---|---|---|
    | sharp cap `A_1` | 0.14 | 0.64 | 0.96 | 1 | 1 | 1 | 1 |
    | S5 cap `A_icx` | 0.04 | 0.18 | 0.28 | 0.35 | 0.49 | 0.81 | 0.96 |
    | `lore`'s exact cap | 1, unsafe | 1, unsafe | 1, unsafe | 1 | 1 | 1 | 1 |

- **Claude's answers after the analysis (26 Oct 2026)**:
  - Tighter option: yes, the sharp cap. It is the largest cap that is safe for every true state, with one tracked
    state; it equals `lore`'s exact cap wherever that cap is safe.
  - Fewer constraints: no option has fewer. The sharp cap, S5 and `lore`'s option all have one tracked state and one
    cap row per cell and day, as `lore` today. The two-shape option adds a binary per day; the sharp cap makes it
    useless above the threshold, so it is dropped.
  - Safety of `lore`'s option: unsafe below `t1(eps)`, safe above it.
- **Options for the formulation (revised, 26 Oct 2026)**:
  - **Sharp cap** (recommended): S5's tracking `A' = b*·mu`; `b*` computed per cell as the smallest rate of its
    increments, initial state and replacement state, so the input `gamma_beta_bound` goes; the cap `A_1(b*·tau, eps)`
    computed offline per cell (bisection on `A`, inner maximum over `c`).
  - **S5 convex-order cap `A_icx`**: proven without the theorem above; looser.
  - **`lore`'s exact cap with an input check `b*·tau >= t1(eps)`**: equal to the sharp cap where it is accepted; it
    refuses the other inputs.
- **Decision on `main`'s other Gamma code (26 Oct 2026, chosen in the answer menu)**: the `repeated_increment` and
  `finite_count` calibrations and the legacy one-rate route leave the solve path. Git history keeps them.
- **User's question on the cap (26 Oct 2026, in the answer menu)**: *"How does it work? I don't want to simply use the
  minimum (or maximum, I don't remember which of them is the most conservative) beta for all the missions because it is
  too conservative. The idea of the main's calibration was to find a beta that was tighter and at the same time safe.
  Does your proposal do the same but in a safe way this time?"*
- **Claude's answer (26 Oct 2026)**:
  - The sharp cap keeps each mission's own mean; `b*` (the smallest rate, the most conservative) sets only how
    dispersed the worst case may be. The conservative route the user means is S1: keep each mission's shape at rate
    `b*`, which multiplies its mean by `b_q/b*`.
  - With one rate per cell the sharp cap equals the exact cap. On `main`'s inputs (checked 26 Oct 2026: every Gamma
    rate is a scalar 20 in every tracked experiment input and in `examples/regression/vbz_progressive/`;
    `gamma_beta_bound: 10` is an input choice) the computed
    `b*` is 20, and the cap is exact.
  - With several rates in a cell, the sharp cap counts fast missions as if they were as dispersed as the slowest. With
    one tracked total no safe cap can do better: a schedule of slow missions only must stay certified, and for it the
    cap is already exact.
  - So no: it does not do what `main`'s calibration intended. That needs per-rate weights: one tracked weighted total
    `Σ_q w_q·mu_q` and one row, as `main`'s calibration, with weights checked against every mix (the step `main` missed).
  - Illustration (`analysis/scripts/verify_gamma_sharp/two_rates.py`; example S2: `tau = 1`, `eps = 0.01`, slow
    missions rate 10, fast missions rate 200, mean 0.1 each). Fast missions allowed: sharp cap 3.97; best safe weights
    that keep slow-only schedules exact 6.29; exact for fast-only schedules 8.42. `main`'s line through the two
    single-type limits reaches 2.96·`eps` on a mix.
  - Status of the per-rate weights: Claude's own idea. Its safety check needs an extension of the theorem to several rate
    groups (each group's worst case is one term). Claude sketched it; it is not written or checked. The illustration
    searches a small grid of scales only.
- **Decision (26 Oct 2026, chosen in the answer menu)**: the sharp cap. The per-rate weights are outstanding work,
  outside the merge (added to `Things to change.txt` on 26 Oct 2026).
  - Tracked state: the mean `mu`, with the dynamics, repair laws and one-sided rows of every model; `A' = b*·mu`.
  - `b*` per cell: the smallest Gamma rate among its missions (both phases), its initial state and its replacement state.
    The input `gamma_beta_bound` goes (an old key: an error with an explanation, item "Input key names").
  - Row on every day: `mu <= A_1(b*·tau, eps)/b*`. `A_1` is computed offline per cell: bisection on `A`, with an inner
    maximum over `c ∈ (0, 1]` of `Q(A/c, b*·tau/c)`.
  - No input check on `b*·tau`: the sharp cap is safe at every value.
- **To implement**:
  - Code: the Gamma builder with the tracked mean and the cap row; the offline computation of `A_1`; `b*` per cell;
    `main`'s calibrations, its legacy route and their inputs leave the solve path. The loop rows cover the mean and its
    anchor `gmu` (item "States covered by the loop constraint"); the shape anchor `gA` goes, because the shape is no
    longer tracked.
  - `spec/spec.tex`: the Gamma section: the tracked state; the class of true states; the theorem with its proof
    (Statement 2 of `analysis/gamma_sharp.md`, with unimodality cited to Roosta-Khorasani and Székely 2015, Theorem 4);
    the computation of `A_1`; the remark that `A_1` equals the exact one-rate cap for `b*·tau >= t1(eps)`.
  - `spec/references.bib`: Roosta-Khorasani and Székely (2015).
  - `scoping.tex`: the Gamma part of "Stochastic ordering within each degradation family".

### Inverse Gaussian reliability cap

- `lore`: from the exact IG distribution function, solved in log space.
- `main`: old code with a normal approximation, not reachable from `solve()`.
- **Question to settle in this item** (Claude's derivation, 26 Oct 2026, not yet checked against a source):
  - `lore`'s spec assumes the state law `IG(mu, mu/eta)`, with a fixed ratio `eta = mu/lambda`
    (lore:spec/spec.tex:817-818, 857-858).
  - The Laplace transform of `IG(mu, lambda)` is `exp((lambda/mu)·(1 - sqrt(1 + 2·mu²·s/lambda)))`. Its exponent is
    linear in `mu` exactly when `lambda/mu²` is fixed. So a sum of independent IG terms is IG only when every term has
    the same `lambda/mu²`.
  - With `lambda = mu/eta`, the ratio `lambda/mu² = 1/(eta·mu)` changes with `mu`. Example: two increments
    `IG(1, 1/eta)` sum to `IG(2, 4/eta)`, so `lambda/mu = 2/eta`, not `1/eta`.
  - So `IG(mu, mu/eta)` is an assumption on the family of the state, not the law of accumulated independent
    increments. The same question appears in `main:lore_check.txt` ("check that a constant lambda/mu is reasonable")
    and in review item A-05 of `main:spec/review_report.tex`.
- **Analysis** (an agent, 26 Oct 2026; notes in [`analysis/inverse_gaussian.md`](analysis/inverse_gaussian.md),
  derivations and standard-library scripts; Claude re-derived the proofs of S1.2 and S4.1 and recomputed two
  counterexamples and three costs independently, with matching results):
  - The lead is confirmed: a sum of independent `IG(a_i, a_i/eta)` is IG only if all `a_i` are equal, and then it is
    `IG(n·a, n²·a/eta)`, not `IG(n·a, n·a/eta)`.
  - The model law `IG(mu, mu/eta) = mu·Z`, `Z ~ IG(1, 1/eta)`, is the comonotone sum (all terms driven by one common
    variable). For any dependence between missions, the true state is below it in the convex order (same mean, less
    spread). With the one-sided rows, `D_k <=_icx mu_k·Z` (increasing convex order).
  - Convex order does not order tails. `lore`'s cap can be unsafe for very dispersed increments: `eta = 10`,
    `eps = 0.1`, three equal missions up to the cap give a true tail 0.118; `eta = 100`, `eps = 0.01`, two missions
    give 0.0107. For `eta <= 1` and `eps <= 0.01` it is safe and often very conservative. A failure needs `eta > 73`
    at `eps = 0.01`, `eta > 4.9` at `eps = 0.1`.
  - The tail increases in `mu` (proved without the MLR argument). The spec's density at lore:spec/spec.tex:906-907
    lacks a factor `mu` in its exponent; the Proposition built on it is true.
  - The mean-only loop row is the right condition; it adds no new failure mode. Safety of every repetition reduces
    to the per-day question that the cap answers.
- **Proposed safe cap** (the agent's proposal, S4.1): `mu <= tau / E[Z | Z > q]`, `q` the `(1 - eps)`-quantile of
  `Z`. `lore`'s cap is `mu <= tau / q`.
  - Proof: for `d < tau`, `P(D > tau) <= E(D - d)_+/(tau - d)`; with `D <=_icx mu·Z` and `d = mu·q` this gives
    `P(D > tau) <= eps` exactly when `mu·E[Z | Z > q] <= tau`. No independence is needed.
  - It is the largest cap that is safe for every law below the model in convex order.
  - Cost against `lore`'s cap at `eps = 0.01`: 7 % (`eta = 0.05`) to 22 % (`eta = 1`); Claude's recomputation:
    9.8 % at `eta = 0.1`, 18.2 % at 0.5, 22.5 % at 1.
  - Code change: `solve_mu_bar` only; the rows stay linear.
- **Other options** (from the analysis): a sub-gamma Bernstein row, as rainflow's, with derived inputs
  (`b = 6·eta·a_max`, `v_inc = eta·mu_inc²`); it needs independence and wins only with many small missions. The
  maximum of two valid caps is also valid.
- Options:
  - **The CVaR cap** (recommended).
  - **The CVaR cap, plus the sub-gamma Bernstein row as an option per component.**
  - **`lore`'s cap, with an input check `eps <= eps*(eta)`**: rests on an unproven numerical conjecture.
  - **`lore`'s cap as it is.**
- **Decision (26 Oct 2026, chosen in the answer menu)**: the CVaR cap `mu <= tau / E[Z | Z > q]`.
- **To implement**:
  - Code: `solve_mu_bar` computes the CVaR cap (the quantile `q` by root finding, the tail mean by the closed form
    or by quadrature); the rows stay linear.
  - `spec/spec.tex`: the IG reliability section with the convex-order argument and the CVaR cap; the corrected density
    at lore:spec/spec.tex:906-907; the loop argument of the analysis.

### Warm starts

- `lore`: `VarHintVal` hints, only inside the range loop, only from an optimal previous result.
- `main`: `Start` values from a previous result, with optional completion; a relaxation start; a greedy heuristic,
  which never schedules a replacement.
- **Check of both versions** (26 Oct 2026; code read in `merge/reports/C_tooling.md` section 3 and
  main:src/fleet_management/greedy_warm_start.py):
  - `main`'s `Start` gives Gurobi a candidate solution; with completion, the binaries are fixed and the continuous
    states solved, so the start is feasible or the failure is reported with an IIS (irreducible infeasible subset).
    It also uses incumbents found before a time limit. No defect found.
  - `lore`'s `VarHintVal` gives Gurobi preferred values for branching only. It works even when a start cannot be
    completed, but it is weaker than a completed start (Claude's reading). It exists only on `lore`, so the policy
    ports it.
  - The greedy heuristic handles rainflow and Gamma cells only (lines 62-73, 177-215) and never schedules a
    replacement (line 448). With the merge's Wiener model, whose variance falls only by replacement, every greedy
    start of a fleet with a Wiener cell would violate the loop row. With Gaussian and IG cells it has no replay.
- Options:
  - `lore`'s hints: **ported as an option next to `main`'s start** (recommended, by the policy; its added value is
    small); dropped as superseded by `main`'s completed start.
  - Greedy heuristic: **extended to every model, with replacements** (recommended); restricted to rainflow and Gamma
    fleets, with an error for other models.
- `main`'s start with completion and the relaxation start stay (policy).
- **User's answers (26 Oct 2026)**: hints: *"If VarHintVal is always outperformed by main option, drop it. Otherwise,
  port it as an option"*. Greedy: *"Option 1 and record that it should be smartly re-designed to get a good starting
  point"*.
- **Check of the condition on hints** (Gurobi Optimizer Reference Manual, "Warm Start Options",
  https://docs.gurobi.com/projects/optimizer/en/current/features/warmstart.html; Gurobi support article "What are the
  differences between MIP Starts and Variable Hints?",
  https://support.gurobi.com/hc/en-us/articles/20410834783377-What-are-the-differences-between-MIP-Starts-and-Variable-Hints;
  both read on 26 Oct 2026 through a fetch tool that paraphrases the pages):
  - A MIP start is for a known solution: "If you know a solution, you should use a MIP start". An infeasible start
    is repaired if possible, else discarded.
  - Hints are for "a general sense of the likely values"; they guide heuristics and branching over the whole solve,
    and poor hints "should not severely degrade performance".
  - So the start does not always outperform hints: they serve different situations. A schedule carried over from
    another horizon may not fit the new problem; its start can then be discarded, while hints still guide the search.
- **Decisions (26 Oct 2026)**:
  - `lore`'s `VarHintVal` hints are ported as an option, next to `main`'s start (the user's condition is not met).
  - The greedy heuristic is extended to every model, with replacements where the loop row needs them. It must be
    redesigned to give a good starting point, not only a feasible one (user's request).
- **To implement**:
  - Code: a warm-start option that sets `Start`, `VarHintVal`, or both; the greedy heuristic for all five models.
  - Redesign of the greedy heuristic: to be specified before it is coded; recorded as outstanding work.
  - `spec/spec.tex`: the warm-start mechanisms and the heuristic.

### Output schema and file formats

- `lore`: `x_m`, `x_r`; `model` holds the model names; `v` is NaN where it is not tracked; arrays only for an optimal
  status; `results_path=None` writes no file; HDF5 with groups.
- `main`: `m`, `r`, `idle`; `model` holds the Gurobi object and the names are in `model_assignment`; `v` is 0 where it
  is not tracked; arrays whenever a solution exists; cost breakdown, gap, bound and performance data;
  `results_path=None` writes `output.yaml`; flat HDF5.
- **Check of both versions** (26 Oct 2026; `merge/reports/A_core.md` section 9 and code read):
  - Kept from `main` by the policy, with no conflict: arrays whenever a solution exists, so an incumbent found before
    a time limit is reported with its status; the cost breakdown, gap, bound, performance and warm-start reports;
    `yaml.safe_dump` of a fixed key list; `u` recomputed from the solution as `max_{i,k} Σ_l mu`, which also works when
    `u` is not built (`C_D = 0`). The output keys `m`, `r`, `idle`, read by `main`'s tools, stay.
  - Conflict 1, `v` where no variance is tracked: `lore` writes NaN, `main` writes 0. Zero suggests a known damage
    with no spread, which is false. NaN is honest, but JSON has no NaN (Python's `json.dump` writes a non-standard
    `NaN`).
  - Conflict 2, `results_path=None`: `lore` writes no file, `main` writes `output.yaml`. `main`'s sweep passes an
    explicit path (main:src/fleet_management/horizon_sweep.py:418-422), so either rule works for it.
  - Conflict 3, the key `model`: `lore` stores the model names, as in the input; `main` stores the disposed Gurobi
    object and puts the names in `model_assignment`. One key name with two meanings across input and output.
  - Conflict 4, HDF5 layout: `lore` writes groups (`metadata`, `solution`, `parameters`) with a subset of the keys;
    `main` writes a flat file with every key (attributes for scalars and JSON strings, datasets for arrays).
- Options:
  - `v` where not tracked: **`null` in files, NaN in memory** (recommended); 0, as on `main`; NaN everywhere, as on
    `lore`.
  - `results_path=None`: **no file, as on `lore`** (recommended); `output.yaml`, as on `main`.
  - `model`: **the model names, as in the input; the Gurobi object is not stored in the result** (recommended);
    `main`'s two keys.
  - HDF5: **`main`'s flat layout with every key** (recommended); `lore`'s groups.
- **Decisions (26 Oct 2026, chosen in the answer menu)**:
  - `v` where no variance is tracked: `null` in files, NaN in memory.
  - `results_path=None` writes no file; the default argument stays `"output.yaml"`.
  - `model` holds the model names, as in the input; the Gurobi object is not stored in the result; `model_assignment`
    goes.
  - HDF5: `main`'s flat layout with every key.
  - Kept from `main` (policy): arrays for any solution found, with its status; the cost breakdown, gap, bound,
    performance and warm-start reports; `yaml.safe_dump` of a fixed key list; `u` recomputed from the solution; the
    keys `m`, `r`, `idle`.
- **To implement**:
  - Code: the serialisers for YAML, JSON and HDF5; `main`'s plotters and validators read `model` instead of
    `model_assignment`.
  - `spec/spec.tex`: the output tables (single result) and the HDF5 schema.

### Plotting

- `lore`: one row per train, `L` strips per row, border style per model.
- `main`: `plot_mixed_management`, one row per (vehicle, component) with a row label; an older `plot_management`;
  `plot_horizon_sweep`.
- `Things to change.txt` asks for a label at the start of each row.
- **Check of both versions** (26 Oct 2026; `merge/reports/C_tooling.md` section 1):
  - `plot_mixed_management` (main:src/fleet_management/utils/mixed_plotter.py): one row per (vehicle, component), a
    row label with the component name and the model, per-component marks `M_j`, `I`, `R` (repair), `P`
    (replacement), a run-summary panel (status, objective, horizons, gap, sizes, runtime, counts). It plots any result
    with a solution, which matches the decision to report incumbents. It covers the request in `Things to change.txt`.
    Defect: the README promises a `D` (depot) mark that no code draws. For archived cluster runs it reads the Gurobi
    log `runner.out` from the result's folder when that file exists (lines 783-795); this is a fallback, not a defect.
  - `main`'s older `plot_management` (main:src/fleet_management/utils/plotter.py): one row per train, as on `lore`,
    without the model; its y axis is labelled "Flight i" and it has no replacement mark. It duplicates the mixed
    plotter.
  - `lore`'s `plot_management`: one row per train, `L` strips, the model shown by border styles, which
    `Things to change.txt` asks to replace; marks for repair (gear), replacement ("R") and idle ("zzz").
  - `plot_horizon_sweep`: `J_op/H2` against `H2`, the gap, the model sizes. No defect found.
- Options:
  - **One schedule plot, `main`'s mixed plotter, published as `plot_management`; its defects fixed (draw `D` for a
    depot day with no intervention, or drop it from the documents); `plot_horizon_sweep` kept; `main`'s older plotter and `lore`'s plotter removed** (recommended). This implements the
    request in `Things to change.txt`.
  - **The same, plus `lore`'s train-level plot kept as an option**, with row labels instead of border styles.
- `Things to change.txt` says "Visualization of the results must improve"; a further redesign is open work, listed in
  the to-do file, not part of the merge.
- **Decisions (26 Oct 2026, chosen in the answer menu)**:
  - One schedule plot: `main`'s mixed plotter, published as `plot_management`; the `D` mark drawn for a depot day with
    no intervention, or removed from the documents; `plot_horizon_sweep` kept; `main`'s older plotter and `lore`'s
    plotter removed.
  - The general redesign stays open work. It is already the third item of `Things to change.txt`, whose box-contour
    part the merge settles; no new line was added.
- **To implement**:
  - Code: rename and consolidate the plotters; the `D` mark; the plotters read `model`.
  - `spec/spec.tex`: the plot layout section.

### Validation tools

- `lore`: none.
- `main`: a legacy `validate()` and dashboards that cannot read the current output; current Gamma validators.
- **Check of `main`'s version** (26 Oct 2026; `merge/reports/C_tooling.md` section 4, read in part):
  - `validate(input, degradation, results)` and the Streamlit dashboards (Streamlit: a Python library for web
    dashboards) are legacy: they require the key `alpha`, a scalar `H`, `u` of shape (2H,) and `z` of shape (F, 2H),
    and parse inputs through a registry marked deprecated. So they cannot read `main`'s current output, as `main`'s
    README warns. The demand check also counts depot days as a mission, which rejects valid schedules.
  - The Gamma validators are current: an exact replay, a bookkeeping replay, and a Monte Carlo validator that
    simulates the true damage of a fixed schedule and estimates `P(D > tau)` per day. They cover Gamma cells only.
  - A Monte Carlo replay of a solved schedule is an independent check of every safety argument in this file (Claude's
    reading): it estimates the true tail per day from the true laws, not from the tracked surrogate.
- Options:
  - Legacy `validate()` and dashboards: **rewritten for the merged schema** (recommended; the policy keeps `main`'s
    features and fixes their defects); removed.
  - Monte Carlo replay: **extended to every model** (recommended); Gamma only, as on `main`.
- **Decisions (26 Oct 2026, chosen in the answer menu)**:
  - The legacy `validate()` and the dashboards are rewritten for the merged schema: structural checks of a result
    against its input (binaries, one activity, mission coverage without depot days, depot gating, loop rows), and the
    dashboards on the new keys. The Gamma validators stay.
  - Extending the Monte Carlo replay to every model is outstanding work, outside the merge. User: *"leave the
    extension as something to do"*. Added to `Things to change.txt` on 26 Oct 2026.
- **To implement**:
  - Code: the rewritten validator and dashboards.
  - `spec/spec.tex`: the validation tools.

### Tests and test inputs

- `lore`: `remote_smoke_test.py`, plus a `test/` suite and `input/` files outside git.
- `main`: `tests/` (4 files), 30 `check_*.py` regression scripts, tracked scenario inputs.
- **Check of both versions** (26 Oct 2026; `merge/reports/C_tooling.md` sections 5 and 8):
  - `lore`'s `test/` and `input/` are ignored by git and absent from this checkout, so their content is unknown.
    `remote_smoke_test.py` needs them (`input/data_example.yaml`, `input/data_example_loop.yaml`).
  - `main`'s `tests/`: Gamma tail cap, Gamma transitions, and two tests of the legacy validator, which the merge
    rewrites. The 30 `check_*.py` scripts print `PASS` or exit with a nonzero code; four are missing from
    `RegressionREADME.md`.
  - A bare `pytest` in the root also collects `test_rainflow.py` and `test_sparse_version.py`, experiment harnesses
    whose functions take positional arguments, so pytest reports errors. No pytest configuration exists.
  - `main`'s `.gitignore` ignores `tests/*` and `test/*`, although `tests/` is tracked; a new test file is skipped by
    `git add .`.
  - The merge changes the formulation of every model, so most existing tests and checks need rewriting.
- Options:
  - Test suite: **one pytest suite in `tests/`, tracked in git (`tests/*` and `test/*` leave `.gitignore`), with
    `testpaths = ["tests"]` in `pyproject.toml`; it covers every model and every decision in this file: parsing and
    checks without Gurobi, small solves with Gurobi; `main`'s check scripts kept where they still apply** (recommended);
    `main`'s layout as it is.
  - `remote_smoke_test.py`: **ported, with its inputs tracked under `examples/`** (recommended, by the policy); dropped
    as superseded by the suite.
  - `lore`'s `test/` and `input/` folders: the user is asked whether they still exist somewhere.
- **Decisions (26 Oct 2026, chosen in the answer menu)**:
  - One pytest suite in `tests/`, tracked in git; `tests/*` and `test/*` leave `.gitignore`; `testpaths = ["tests"]`
    in `pyproject.toml`; it covers every model and every decision in this file; `main`'s check scripts are kept where
    they still apply.
  - `remote_smoke_test.py` is ported, with its inputs tracked under `examples/`. User: *"port it and record that it
    must be extended for the changes we made"*: its scenarios must cover the merge's changes (five models, the two
    repair laws and anchors, the one-sided encoding and its flag, the two-phase horizon, the five bounds and linear
    forms, the IG and Gamma caps, replacement states, the cost terms).
  - `lore`'s `test/` and `input/` folders still exist; the user will copy them into the repository. Their porting
    waits for them.
- **To implement**:
  - Waiting for the user: the copy of `lore`'s `test/` and `input/` folders.
  - Code: the suite; the extended smoke test; the pytest configuration; the `.gitignore` lines.

### Cluster jobs and experiments

- `lore`: none.
- `main`: `euler/` (the older family of job scripts calls `test.py`, renamed to `test_rainflow.py`), `experiments/`
  (9 folders), and the results of two Gamma horizon sweeps.
- **Check of `main`'s version** (26 Oct 2026; `merge/reports/C_tooling.md` section 6, read in part):
  - Family A (`setup_euler.sh`, `submit*.sh` and their arrays) calls `test.py` and loads it by path from
    `run_studies.py` and `run_year.py`; commit 1f31386 (25 Sep 2026) renamed it to `test_rainflow.py`, so family A
    fails on a fresh clone. These jobs run historical studies on archived code (`legacy/`, `rainflow_v2`).
  - Family B (study jobs) calls `examples/regression/run_*.py` and the experiment runners; runnable if the inputs
    exist; one job needs two inputs that were never committed.
  - `euler/README.md` is cited by `main`'s README but was never committed.
  - The two tracked Gamma sweeps (`results/gamma_horizon_convergence_*`) were computed with `main`'s
    `repeated_increment` calibration and `gamma_beta_bound = 10` below the mission rate 20: the case where the
    analysis found accepted unsafe schedules (item "Gamma formulation"). Their schedules are not certified.
  - Every experiment ran with `main`'s formulation; after the merge, the formulation changes for every model.
- Options:
  - Family A: **the scripts point to `test_rainflow.py`, and the jobs are kept** (recommended, by the policy); the
    family is archived as historical.
  - Results computed before the merge: **kept, with a dated note in each experiment's README and next to the tracked
    sweeps, saying which formulation produced them and that the Gamma schedules are not certified** (recommended);
    removed.
  - `euler/README.md`: **written**, describing both families (recommended); the citation removed from the README.
- **Decisions (26 Oct 2026, chosen in the answer menu)**:
  - Family A: dropped, with `test_rainflow.py` and the archived rainflow variants (item "Follow-up decisions from
    writing the spec", 26 Oct 2026).
  - Results computed before the merge are removed. Scope, to confirm file by file at implementation: the solve
    results in `results/` (the two Gamma convergence sweeps with their inputs and plot, the old Gaussian runs
    `output*.yaml`, `schedule*.png`, `validation*.yaml`). Not in scope: the validator fixtures in
    `results/validator_tests/` (test data, item "Tests and test inputs"), experiment definitions, and the start file
    of `experiments/vbz_F8_feasibility_transfer/starts/` (an input of that experiment; it is regenerated when the
    experiment is re-run). Local copies go to the Recycle Bin; git history keeps them.
  - The citation of `euler/README.md` is removed from the README; no cluster README is written.
- **To implement**:
  - Code: the removal of the family-A scripts, `test_rainflow.py` and the archived rainflow variants; the removal of
    the results above; the README line.

### Documents

- `spec/spec.tex`: version 0.5 on `lore`; version "0.2 (revised)" on `main`.
- `spec/main.tex` and `spec/review_report.tex` (a literature review from May 2026 and its automated review): deleted
  on `lore`, present on `main`.
- `scoping.tex`: revised on `lore` only.
- `ASSESSMENT.md` and `Things to change.txt`: only on `lore`. `lore_check.txt`: only on `main`.
- README: `lore` treats the specification as authoritative; `main` treats the code as authoritative and cites files
  that are not in the repository.
- **Settled by earlier decisions or facts** (26 Oct 2026):
  - `spec/spec.tex` and `scoping.tex`: `main` never changed them after the common ancestor, so `lore`'s versions are
    the base; both are updated with every decision in this file (item "Source of truth").
  - README: rewritten for the merged package, with `spec/spec.tex` as the source of truth and no reference to files
    outside the repository (`docs/PROJECT_LAYOUT.md`, `euler/README.md`, "equations under `spec/`").
  - `Things to change.txt` stays: it is the project's to-do list.
  - `CLAUDE.md` describes `lore`; it is rewritten for the merged package. `lore-vs-main.md` and `merge/` stay until
    the merge is done (user, 25 Oct 2026).
  - `main`'s `RegressionREADME.md` and `GAMMA_DIAGNOSTICS.md` are updated with the code they describe; the second
    describes the calibration that leaves the solve path (item "Gamma formulation").
- **Open points** (facts):
  - `spec/main.tex` (a literature review, May 2026) and `spec/review_report.tex` (its automated panel review): the user
    deleted both on `lore` on 7 Jul 2026; `main` keeps them unchanged. Git history keeps them either way.
  - `lore_check.txt` (only on `main`; the user deleted it on `lore` on 30 Jul 2026, when writing
    `Things to change.txt`): five notes. (1) Check that a constant `lambda/mu` is reasonable for IG; the merge makes
    IG safe for any dispersion, but the fit to data stays a modelling question. (2) Check that a constant `sigma` is
    reasonable for Wiener. (3) "Verificarlo anche per gamma" (check it for Gamma too). (4) "Capire come il lavoro di
    castillo si pone rispetto al nostro" (understand how Castillo's work relates to ours). (5) "Capire come inserire i
    cicli di vita" (understand how to include life cycles). None is in `Things to change.txt`.
  - `ASSESSMENT.md` (only on `lore`, 7 Jul 2026; the README calls it superseded): its status statements are outdated;
    its last section lists five research directions (rolling-horizon deployment; scenario chance constraints;
    distributionally robust framing; decomposition; a policy baseline).
- Options:
  - `spec/main.tex`, `spec/review_report.tex`: **stay deleted, as on `lore`** (recommended: the user's own decision);
    restored.
  - `lore_check.txt`: **stays deleted; its open notes (2) to (5), and (1) as a data question, move to
    `Things to change.txt`** (recommended); stays deleted, notes dropped.
  - `ASSESSMENT.md`: **its five research directions move to `Things to change.txt`, and the file is removed** (recommended:
    the rest is outdated, and git history keeps it); kept as a historical record; removed with its directions.
- **Decisions (26 Oct 2026, chosen in the answer menu)**:
  - `spec/main.tex` and `spec/review_report.tex` stay deleted.
  - `lore_check.txt` stays deleted, and its notes are dropped (the user's choice; Claude had recommended moving them).
  - `ASSESSMENT.md`: its five research directions were added to `Things to change.txt` on 26 Oct 2026; the file is
    removed at implementation (local copy to the Recycle Bin; git history keeps it).
- **Decisions on the plan of the documents (26 Oct 2026, chosen in the answer menu; plan in
  [`documents_plan.md`](documents_plan.md))**:
  - Change markers: version 1.0 of the spec adds no `%% --- CHANGE ...` markers, and the 133 existing markers are
    removed. The changelog and git history keep the record. Claude's recommendation, after the user's rule that a source
    file holds only the deliverable.
  - After the merge, `merge/` and `lore-vs-main.md` move to an archive folder in the repository. The user's remark about
    `main`'s authors is reworded (item "Code base of the merge"); commit 9889743 on `origin/lore` keeps the old wording.
- **To implement**:
  - `spec/spec.tex` and `scoping.tex`: every decision of this file.
  - README, `CLAUDE.md`, `RegressionREADME.md`, `GAMMA_DIAGNOSTICS.md`: rewritten or updated for the merged package.
  - The removal of `ASSESSMENT.md`.

### Repository hygiene

- `.gitignore`: 10 lines on `lore` (ignores `input/`, `test/`, `CLAUDE*`); 45 lines on `main`, which match 42 tracked
  files.
- Tracked on both: `src/fleet_management.egg-info/` and `results/`.
- `pyproject.toml`: Python >= 3.9 on `lore`; Python >= 3.10 with `pandas` and `streamlit` on `main`; on both the
  description still says "Gaussian degradation via MILP".
- **Check of both versions** (26 Oct 2026; `merge/reports/C_tooling.md` section 8):
  - `main`'s `.gitignore` mixes general rules (`__pycache__/`, `*.py[cod]`, `.venv/`, run folders of experiments) with
    personal scratch files (`ds.py`, `run_audit.py`, five archive names) and rules that match tracked project files:
    `tests/*`, `input/*`, `results/*`, and `mixed_plotter.py`, which matches the current plotter at any depth. A new
    test or plotter file is then skipped by `git add .`.
  - `src/fleet_management.egg-info/` is a build artefact, tracked on both branches and stale on both.
  - `CLAUDE*` is ignored on both branches. `CLAUDE.md` holds the user's instructions for Claude, including the merge
    policy that no version is trusted by default.
  - `main`'s dashboards need `streamlit` and its tools `pandas`; `main`'s code uses Python 3.10 syntax (`X | None`).
  - `.vscode/settings.json` (only on `main`) holds editor defaults; the policy keeps it.
- Options:
  - `.gitignore`: **one coherent file: build artefacts (`*.egg-info/`, `__pycache__/`, `*.py[cod]`, `build/`, virtual
    environments), generated outputs (`results/`, `logs/`, experiment run folders, archives), local folders
    (`literature/`, `agentic/`, `.claude/`, `bkp/`), `CLAUDE*`; no rule that matches a tracked project file** (recommended);
    `main`'s file as it is.
  - `egg-info`: **untracked and ignored** (recommended); kept tracked.
  - `pyproject.toml`: **`main`'s requirements (Python >= 3.10, `pandas`, extra `dashboard`), and a description of the
    merged package** (recommended); `lore`'s.
  - `CLAUDE.md`: **stays ignored, not shared with the other contributors** (recommended); tracked.
- **Decisions (26 Oct 2026, chosen in the answer menu)**:
  - `.gitignore`: one coherent file as described above, except that `CLAUDE.md` is tracked. User: *"One coherent file
    but sync CLAUDE.md"*.
  - `src/fleet_management.egg-info/` is untracked and ignored.
  - `pyproject.toml`: `main`'s requirements (Python >= 3.10, `pandas`, extra `dashboard`) and a description of the
    merged package.
  - `CLAUDE.md` is tracked (the user's choice; Claude had recommended keeping it ignored). Its line on the trust in
    `main`'s code becomes visible to the other contributors; the user chose a neutral wording (answer menu,
    26 Oct 2026), applied to `CLAUDE.md` the same day: "No version is trusted by default".
  - `.vscode/settings.json` stays (policy).
- **To implement**:
  - The new `.gitignore`; `git rm --cached` of the egg-info folder; `pyproject.toml`; `CLAUDE.md` added to git.

### Follow-up decisions from writing the spec

Questions raised by the four writers of spec version 1.0 (26 Oct 2026; their notes are in `spec_drafts/status_*.md`).

- **A consequence of the decided loop row, proved by writer A** (`spec_drafts/fragments_A.tex`, proposition on ARD1 and
  replacement; checked on 3000 random cycles in `spec_drafts/checks/A_encoding_loop.py`): under ARD1 with `rho < 1`, a
  cell that gains damage in the operating phase must be replaced in every operating phase. Each ARD1 repair leaves
  the state at the anchor plus `(1-rho)` times the damage since the anchor, so the anchor rises with every repair and
  never falls without a replacement; without one, the mean grows without bound over the repetitions. Example:
  `rho = 0.5`, a mission adds 0.2 between repairs; each repair leaves 0.1 more than the previous one.
- **Decisions (26 Oct 2026, chosen in the answer menu)**:
  - Default of `repair_model`: `ardinf` (`main`'s code default is `ard1`). ARD1 stays available per cell.
  - The inverse Gaussian model accepts ARD1 too. Writer B showed that the CVaR cap is safe for any repair weights in
    (0, 1] and any dependence: under ARD1 the state `G + (1-rho)·Δ` is again a weighted sum of the increments. Every
    model now accepts both laws (item "Names and set of repair laws" is updated by this decision).
  - The early stop of the horizon sweeps uses the case's MIP gap as its gap threshold (`main`'s default 0.05 would
    never fire with the decided default MIP gap 0.1).
  - `main`'s tools that serve only code the merge removes are dropped (user: *"Drop"*): `utils/model_registry.py`,
    `validate_gamma_result` (`gamma_validator.py`), the two validators of the old Gaussian format
    (`validate_baseline_assignment_feasibility`, the "Gamma synthetic diagnostic"), the formulation-count tools
    (`formulation_size_sweep.py`, `report_gamma_formulation_counts.py`, `utils/benchmark.py`, and the regression
    scripts that call them), `csv_to_plotter.py` (it targets the removed older plotter) and `fix_root_gap.py` (it
    repairs results of the archived studies).
  - `test_rainflow.py`, the archived rainflow variants (`rainflow_v2.py`, `rainflow_sparse.py`, `legacy/`) and the
    family-A cluster jobs that run them are dropped; git history keeps them. User: *"I want the rainflow model too.
    Drop the test as their implementations (if you confirmed correct) have already been tested"*. The rainflow model
    stays with its five bounds; writer B checked each bound row of `main`'s builder against its formula, with the
    decided fixes. This replaces the decision "Family A: the jobs are kept" of the item "Cluster jobs and
    experiments". `test_rainflow.py` was a bound-comparison experiment (the same case solved under each of the five
    bounds, cost and failure point compared), not a schedule validator.
- **Settled by Claude** (consequences of decisions or of the mathematics; reported to the user):
  - Depot compatibility: one row `x^m + x^r <= x_i0k` in place of three; the same binary solutions, a tighter
    relaxation.
  - Support widths: `main`'s width semantics make the Bhatia-Davis check `v <= mu·(b - mu)` invalid (it assumes a
    range starting at 0); the checks use Popoviciu's inequality `v <= w²/4`.
  - `reliability_impl` at a cell whose reliability row is linear (Markov, Chernoff, inverse Gaussian, Gamma): the row
    is built in its exact form and the value is ignored; an unknown value is an error everywhere.
  - Input checks added: every initial value, starting anchor and replacement value lies in the natural range of its
    state (the big-M rows need it); every replacement state satisfies the cell's reliability row (otherwise a
    replacement is never feasible); `cgf >= s·mu` for every increment, initial and replacement state (Jensen); at a
    Bernstein cell with the tangent form, `tangent_ref·tau < tau - c` (otherwise the row is empty).
  - Gamma: `beta*` is the smallest rate among the terms with a positive mean; `gamma_beta_0` and `gamma_beta_new` may
    be omitted, which declares a known (constant) initial or replacement damage, as rainflow's default
    (item "Replacement": known damage by default).
  - The result carries every tracked state and anchor (`R`, `K`, `g_mu`, `g_v`, `g_R`, `g_K`; NaN in memory and `null`
    in files where not tracked), so that the validator can check every loop row.
  - In the objective mode `operating_average`, `z` is not built on transitory days (coefficient 0); the transitory
    repair cost is computed after the solve.
  - `main`'s input block `validation` (settings of the Gamma validators) is kept, by the policy on `main`'s features:
    an input key that `solve` does not read (spec reviews, 26 Oct 2026).
  - Spec reviews (26 Oct 2026; `spec_drafts/review_formulation.md`, `spec_drafts/review_io_tools.md`): 26 must-fix
    items, sent to the writers for correction. Two were errors: the Bernstein tangent check (`tangent_ref·tau < tau - c` is not sufficient;
    the condition is `tangent_ref·tau < sqrt(tau·(tau - 2c))`), and the validator's objective in the mode
    `evaluation_total` (it dropped `C_D·u`).
- **Corrections to this record**:
  - Item "Horizon sweep": `main`'s grid sweep ranks by the projected cost `J_init + (E - H1)·J_op/H2` without `C_D·u`
    (writer D, from `main:src/fleet_management/horizon_sweep.py`); the item wrote the term in. The spec follows
    `main`'s code, as decided.
  - Item "Encoding of the dynamics": the claim "with one tracked quantity and the whole-state law, it rejects none"
    needs the hypothesis that the cell is not replaced on day `H1` (writer A's counterexample).
- **From the WP4 plan (29 Oct 2026, chosen in the answer menu; written into `spec_drafts/fragments_C.tex` and
  `spec/spec.tex` on `lore`, not yet merged into the merge branch)**:
  - `null` at a cell means "not given at this cell" for the three keys without a default: `v_max_user`,
    `gamma_beta_0`, `gamma_beta_new`.
  - Checks: a non-strict inequality passes up to `1e-12·max(1, |a|, |b|)`; selector values are case sensitive; an
    integer key accepts `3.0` and refuses `2.5` and a flag.
  - Result files: `NaN` and `±inf` are `null` in YAML and JSON; HDF5 keeps them; HDF5 does not store
    `objective = None`.
