# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

- A Python library, `fleet_management`, that schedules a fleet of trains over a time horizon.
  Each train has `L` components. Each component degrades stochastically and may be repaired
  (imperfect maintenance) or replaced.
- The schedule is a mixed-integer program built and solved with Gurobi (`gurobipy`). It is
  linear (MILP) with `formulation: lp`. With the default `formulation: exact` it has nonconvex
  quadratic constraints (MIQCP, mixed-integer quadratically constrained program) whenever some
  component is Gaussian, Wiener or rainflow.
- Public API: `solve(input_path, results_path)` and `plot_management(output_path, plot_path)`
  (`src/fleet_management/__init__.py`). There is no command-line entry point.
- `spec/spec.tex` (version 0.5) is the authoritative mathematical specification. Code comments
  cite its sections by name ("Consistency Checks", "Loop Constraint", "Input File Schema").
- `scoping.tex` (with `scoping.pdf`, `scoping_bib.bib`) is the research scoping document. The
  spec calls it "the project scoping document".
- **Spec and scoping hold different things; each piece of information lives in exactly one of
  them** (user, 26 Oct 2026: *"spec.tex must contain only exactly what is implemented (or going
  to be implemented). It is not verbose, there are no long explanations, no theorems, etc. All
  these stuff must belong to scoping file. There the aim of the project is given, and all the
  background info and choice explanations are provided."*).
  - `spec/spec.tex` is a Software Design Document (SDD; user's choice of title, 26 Oct
    2026): a reader consults it to learn how the code works and to plan improvements. It
    holds only the codebase and its mathematical constructions: inputs, checks,
    variables, rows, precomputed constants and how to compute them, outputs, tools. Short
    statements, equations and tables. No theorems, proofs, worked examples, rationale or
    literature; at most a pointer to the scoping section that justifies a construction.
  - `scoping.tex`: the aim of the project, the background, the literature, the derivations and
    proofs, and the reasons for each choice.
  - Style model for the spec: the version 0.3 that the user supplied, kept as
    `merge/spec_drafts/style_reference_spec_v03.tex` (its content is obsolete; its style is not).
  - Edit the two together: when a construction changes, its specification goes to the spec and
    its justification to the scoping document.

## State of this checkout (branch `lore`, checked 30 Sep 2026)

- **The merged package is on the branch `main`** (commit aa78cea, 8 Nov 2026, pushed), with its own `CLAUDE.md`.
  That file describes the merged code; the architecture below describes `lore`'s old code. For any work on the
  merged code, check out `main` and read that file.

- **`input/` and `test/` are git-ignored and absent.** The README examples, the pytest suite
  and `remote_smoke_test.py` all need them. Ask the user for these files before running anything.
- **No Gurobi, no pytest on the user's Windows workstation** (`DESKTOP-1L5C3I2`), and the package is
  not installed there; on another machine, check before running anything. Real solves run on a
  remote machine with a Gurobi licence (see *Commands*).
- **Stale tracked files:** `src/fleet_management.egg-info/` lists the old flat layout
  (`gaussian.py` at package level). `results/output.yaml` and `results/schedule.png` use the old
  single-model output format (`degradation`, `alpha` keys). Neither describes the current code.
- `.gitignore` also ignores `.claude/*`, `agentic/*` and `literature/*`. This file is versioned.
- `lore` has diverged from `main`: `main` is 116 commits ahead and `lore` is 7 commits ahead.
  The two branches are being merged into `main`. The user's choice for each divergence is
  recorded in `merge/decisions.md`; read it before changing the formulation.
- **No version is trusted by default** (user, 25 Oct 2026). Judge every difference between
  `lore` and `main` carefully before choosing a version: read both versions, derive the
  equations, work a case by hand, and report every defect found. Port everything that exists
  only on `lore` into `main`. Keep every feature that exists only on `main`, and fix its defects.
- **Known defect, decided for the merge:** an ARA1 repair cannot lower the state. The anchor is the
  state before the repair, and the repair-cost row `z <= rho·(s[k-1] - anchor[k-1])` applies on
  every day with `z >= 0`. The merge replaces it with `main`'s anchor, the state after the last
  intervention (`merge/decisions.md`).

## Commands

```bash
pip install -e ".[dev]"                          # editable install with pytest
cd test && pytest -v                             # full suite (most tests need a Gurobi licence)
pytest test/<file>.py::<test_name> -v            # one test
python remote_smoke_test.py                      # real solves of all scenarios, needs input/
```

- `remote_smoke_test.py` runs the pytest suite plus eight solve scenarios. It writes everything
  to `smoke_test_output/` and always exits 0. Read `summary.json` for pass or fail. The user runs
  it on a licensed machine and sends the zipped folder back.
- No linter, formatter or type checker is configured.

## Architecture

### Pipeline (`solver.py`)

`solve()` → `_read_input` (YAML, JSON or HDF5) → `_parse_and_validate` → one solve per horizon
→ `_extract_result` → `_save_results` (format chosen by the file extension).

- `_parse_and_validate` broadcasts every input to full shape: scalars to `(F, L)`, costs from
  `(F,)` to `(F, L)`, increments from `(F, M, L)` to `(F, M, L, H)`. It builds one boolean mask
  per model and reads model-specific fields only where some component needs them. Model-specific
  fields may hold `null` at positions of other models; these become `NaN`.
- Global checks live in `solver.py`. Per-model checks live in each module's
  `validate_inputs(mask, ...)`, which only looks at positions where `mask` is true.

### One shared Gurobi model per horizon

- Shared decision variables: `x[i, j, k]` of shape `(F, M+1, 2H)`, where `j = 0` is a
  maintenance day and `j = 1..M` are missions; `x_m`, `x_r` (repair and replacement binaries,
  `(F, L, 2H)`); `z` (degradation removed by repair); `u` (damage regularisation, a scalar).
- `base.add_assignment_constraints` adds the model-independent rows once: at most one activity
  per train and day, each mission covered every day, repair or replacement only on a maintenance
  day, never both.
- Each component `(i, l)` is then built by `_DISPATCH[model](ctx, i, l)`. `ctx` is a
  `SimpleNamespace` that carries the model, the shared variables and all parameter arrays.
- The objective is `C_D·u + C_M·(maintenance days) + Σ C_R·z + Σ C_rep·x_r`.

### Degradation-model modules (`models/*.py`)

Each module defines `MODEL_NAME`, `validate_inputs(mask, ...)` and `build_component(ctx, i, l)`.
`build_component` must store `ctx.mu[i, l, k]` for every `k`, and `ctx.v[i, l, k]` if the model
tracks variance. `base.add_damage_regularization` runs after all components and reads `ctx.mu`;
`_extract_result` reads both. Gamma stores the expression `alpha·beta` in `ctx.mu`, not a
variable, so `_value()` accepts both kinds.

| Model | Tracked state | Maintenance | Reliability constraint |
|---|---|---|---|
| `gaussian` | `mu`, `v` | ARD1 | `exact`: `Φ⁻¹(1−ε)²·v ≤ (τ−mu)²` (nonconvex quadratic); `lp`: tangent line |
| `inverse_gaussian` | `mu` | ARD1 | `mu ≤ mu_bar`, cap precomputed by `brentq` in log space |
| `wiener` | `mu`, `v` (grows by `sigma²` per mission; only replacement resets it) | ARD1, ARA1 | as Gaussian |
| `gamma` | shape `alpha` (`mu = alpha·beta`) | ARD1, ARA1 | `alpha ≤ alpha_hat`, cap precomputed by `brentq` on `gammaincc` |
| `rainflow` | `mu`, `v` | ARD1 | Cantelli or Bernstein tail bound (`tail_bound`), `exact` or `lp` form |

- `NonConvex = 2` is set only when `formulation == "exact"` and some component is Gaussian,
  Wiener or rainflow.
- To add a model, touch: a new module, `_DISPATCH` and `SUPPORTED_MODELS` in `solver.py`, the
  field parsing and the `validate_inputs` call in `_parse_and_validate`, the `NonConvex` model
  list, `_BORDER_STYLES` in `plotter.py`, and the spec.

### Maintenance and replacement (`maintenance/`, `models/base.py`)

- ARD1 (Arithmetic Reduction of Degradation): repair multiplies the state by `1−rho`, or by
  `(1−rho)²` for a variance state. ARA1 (Arithmetic Reduction of Age): repair removes a fraction
  `rho` of the degradation accumulated since an anchor, the state at the last maintenance.
- All dynamics are big-M rows that are exact at binary points. The rows are one-sided (`>=`) on
  purpose: the tracked states are upper bounds on the true moments (spec, slack-robustness remark
  in the Gaussian section).
- Big-M constants: `tau` for means, `V_max = τ²/Φ⁻¹(1−ε)²` for Gaussian and Wiener variance,
  `V_max^RF` for rainflow variance, `tau/beta` for the Gamma shape.
- Replacement: `base.add_replacement_linear` (two-sided big-M reset to `mu_new`, `v_new`).
  Repair cost `z`: `base.add_repair_cost_mccormick`, one McCormick envelope for both ARD1
  (anchor 0) and ARA1.

### Horizon structure and the loop constraint

- The horizon has `2H` days in two halves. Increment inputs cover `H` days and
  `_wrap_increment` repeats them for the second half.
- `base.add_loop_constraint` imposes `state[2H−1] ≤ state[H−1]` (0-based) on every tracked
  state, ARA1 anchors included. This makes every repetition of the second half certified.
  It is a hard constraint; the spec has no soft loop penalty.
- Wiener variance can only fall through replacement, so the loop constraint forces at least one
  Wiener replacement in the second half. No input check enforces this.
- `H: [H_min, H_max]` solves once per `H`. Increment arrays are sized `H_max`; each solve uses
  the first `H` entries. `_solve_horizon_loop` runs sequentially, optionally warm-started through
  `VarHintVal` on the overlapping days, or in a `ProcessPoolExecutor` (`n_workers > 1`, which
  forbids `warm_start`). Each parallel solve uses one Gurobi licence token. A failed horizon is
  stored with status `"error: ..."` and the loop continues.

### Output and plotting

- The result dict holds numpy arrays. A non-optimal solve sets every array to `None`.
- A multi-horizon result is keyed by `H`: integer keys in YAML, string keys in JSON, groups
  `H<h>` in HDF5. `plotter.py` detects a multi-horizon file by the absence of a `status` key.
- `plot_management` raises on a non-optimal result. It colours each component strip by `mu/τ`
  and encodes the model by the strip's border style.

## Known differences between the code and the spec

- `ara1.update_anchor` also resets the anchor on a replacement day. The spec does not, and lists
  this reset as an open item. The module docstring gives the reason: without it the model becomes
  infeasible after a replacement (confirmed with a Gurobi IIS, the irreducible infeasible subset).
- The spec's "Model interface (base.py)" describes `validate_inputs(F, H, M, L, ...)` and a
  `solve_fleet_management(...)` per model. The code instead uses per-component builders on one
  shared model, as described above.
- The output echoes `tau`, and the HDF5 output stores `model`. Neither is in the spec's tables;
  the plotter needs both.
- `mu` is required only if some component is not Gamma. The spec's table lists it as always
  present.

## How agents are used (user, 7 Nov 2026)

- **Sonnet agents for all work on code** (the global rule in `~/.claude/rules/subagents.md`, *Models*): code,
  tests, inputs, checks against the spec, copies to Euler. Claude plans, orchestrates, submits the Euler jobs and
  checks the results. This replaces the workflow of WP4 to WP7 (Opus plans, Sonnet executes, Opus revises).

## Other files

- `Things to change.txt`: the user's list of planned changes, the project's to-do list.
- `EULER.md`: how to reach the ETH Euler cluster, its folders (main and scratch),
  Slurm commands, and the user's workflow for testing the package there with Gurobi (7 Nov 2026). Read it before
  any work on Euler. Sonnet agents do the work there; the main session plans, submits jobs and checks results.
- `lore-vs-main.md`: the comparison of the branches `lore` and `main` (30 Sep 2026).
- `merge/`: `decisions.md` (the user's choices for the merge) and `reports/` (four detailed
  comparison reports, cited by file and line). Keep them until the merge is done.
- `ASSESSMENT.md`: a gap analysis from July 2026, written before all five models existed. The
  README marks it as superseded.
- Existing edits in `spec/spec.tex` carry markers `%% --- CHANGE <ID>: ... ---` (series RF, LPR,
  REV and others). Each version has a change log at the end of the file.
