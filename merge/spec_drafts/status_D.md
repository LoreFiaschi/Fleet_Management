# Status of writer D (tools: sweeps, warm starts, files and plotting, validation tools)

Status: done (26 Oct 2026). `fragments_D.tex` is rewritten as a pure specification (user's rule "Spec versus scoping"): 677 lines (1009 before; 67 %), one block in the order Horizon Sweeps, Warm Starts,
File & Folder Structure (Public API, Cell-Builder Interface, Plotting API and Layout), Validation Tools. It builds at 0 errors, 0 overfull and 0 underfull boxes; the only undefined references are labels of writers A, B and C.
All labels are kept except `eq:d_daycost` (the day cost is now cited from `eq:a_daycost`; no other fragment referred to it). Check script `checks/D_sweep_rules.py` passes; the rules it tests are unchanged.
Removed text: `moved_D.md`, 19 blocks (M-D-1 to M-D-19), each verbatim from the previous version (checked line by line against the copy in the scratchpad `spec_D/fragments_D_before_scoping_split.tex`).

## What the rewrite changed

- **Stays** (unchanged in content): every signature with defaults; the steps, errors, reports and key tables; the formulas (`eq:d_phasecosts`, `eq:d_projected`, `eq:d_gradient`, `eq:d_daymap`, the stopping rule, the proof rule, the Clopper--Pearson bound, the check violations); the layout; the builder interface; the plot layout and marks; the validator checks.
- **Moved** (kinds): rationale (why a cost per day compares horizons, why parallel cases are refused, why hints help, why the objective check is one-sided, why the loop checks use saved states), remarks (what a proof certifies, the status of a relaxed solution, a stress test is not a proof), derivation (the Moschopoulos series: mixture, lower bound), comparison (one route for every fleet), introductory sentences.
- **Shortened without loss**: the API table became one sentence; the greedy report table and the Monte Carlo key table became lists of keys; the dashboard tabs became one paragraph.
- Long `\texttt` names carry `\allowbreak` after each escaped underscore; four paragraphs sit in a local `\sloppy` group. No preamble change.

## Checklist of decision bullets (all covered)

Horizon sweep (item "Horizon sweep"; item "Solver defaults")
- [x] the two sweep functions `sweep_operating_horizons` and `sweep_horizon_grid` (`sec:sweep_operating`, `sec:sweep_grid`)
- [x] ranking by `J_op/H2` with `objective_mode = operating_average` (`sec:sweep_criteria`)
- [x] projected cost over an evaluation horizon `E` (Eq. `eq:d_projected`)
- [x] proof rule: optimal at the MIP gap used for the case (`sec:sweep_criteria`)
- [x] early stop with the case's MIP gap as default threshold (`sec:sweep_stop`), checkpoint after each case in both sweeps, per-case error capture (`sec:sweep_report`, `sec:sweep_grid`)
- [x] parallel option, refused with warm starts between cases (`sec:sweep_parallel`)
- [x] input read through the same reader as `solve()` (`sec:sweep_operating`, `sec:api`)

Warm starts (item "Warm starts")
- [x] `Start` values with optional completion and IIS report (`sec:warm_start_modes`)
- [x] `VarHintVal` hints as an option; both together (`sec:warm_start_modes`)
- [x] relaxation start (`sec:warm_relaxation`)
- [x] greedy heuristic for all five models, with replacements where the loop row needs them (`sec:greedy`); the redesign is not specified

Plotting (item "Plotting"): [x] one schedule plot `plot_management`, rows per (train, component), row labels, marks `M_j`, `R`, `P`, `D`, `I`,
run-summary panel; [x] `plot_horizon_sweep`; [x] no older plotter and no lore plotter in the layout (`sec:plotting`)

Validation tools (item "Validation tools"): [x] structural validator; [x] dashboards on the new keys; [x] Gamma validators for the sharp-cap
formulation; [x] Monte Carlo replay for every model NOT specified (the Monte Carlo validator samples Gamma cells only)

Code base of the merge: [x] package layout; [x] public API; [x] cell-builder interface (`sec:files`, `sec:api`, `sec:builders`)

Cluster jobs and experiments: [x] one paragraph in `sec:files`: `euler/` keeps the two family-B study jobs; no cluster README is mentioned

## Answers received and what I changed (26 Oct 2026, coordinator message; decisions.md, item "Follow-up decisions from writing the spec")

- **Q12, early stop.** `maximum_mip_gap_for_stopping` defaults to `None`: a case qualifies when its gap is at most the MIP gap used for that case (`requested_mip_gap`); a number replaces that threshold for every case. The report entry is `null` for the default. The plot of the gap panel shows a line for a number, otherwise a mark at each case's requested gap. The sentence on the 0.1 versus 0.05 mismatch is gone.
- **Q24, dropped tools.** None of them is in the layout (it never was). Cluster paragraph rewritten: `euler/` holds two jobs, `gamma_horizon_convergence.sbatch` (runs `run_horizon_sweep.py`) and `vbz_progressive_feasibility.sbatch` (runs `run_progressive_vbz_feasibility.py`). Read from `merge/reports/C_tooling.md` section 6 and the job files on main. `zurich_mixed_formulation_counts.sbatch` calls `report_mixed_formulation_size.py` (a formulation-count tool) and is not kept; family A is not kept. The two kept scripts import only `solve`, the sweeps and `validate_gamma_schedule_files`. The experiment folders have their own `.sbatch` files; I did not list them.
- **Q25, tail routine.** The layout lists `gamma_utils/gamma_tail_bound.py` for `moschopoulos_tail_probability`, with the three validator modules. The calibration functions of that file are not part of the package. `gamma_process.py` and `gamma_diagnostics.py` are not listed; `collect_gurobi_model_statistics`, which `base.py` needs for `performance`, is now named in the comment on `base.py` ("performance counters").
- **Q5, grid ranking** reverted to main's code: the grid ranks by `projected_evaluation_cost` = `J_init + (E-H1)*J_op/H2` (Eq. `eq:d_projected`, without `C_D*u`); the solve in the mode `evaluation_total` minimises that plus `C_D*u` (`evaluation_objective`). `comparison_metric` is `"projected_evaluation_cost"`.
- **Q7, checkpoint.** The grid report is rewritten after every case of every H1 (the operating sweep already did).
- **Q22, tracked states.** The `validate` table has one loop check per quantity `q` in `mu, v, R, K, g_mu, g_v, g_R, g_K` (NaN/null entries skipped). The dashboards no longer replay a missing quantity. The replay validator reads `g_mu`. **These key names come from decisions.md; fragments_C.tex does not list them yet (checked 26 Oct 2026, 11:29).**
- **Q28, Gamma rates.** The exact and the Monte Carlo validators treat an initial or replacement damage without a rate (`gamma_beta_0`, `gamma_beta_new` omitted) as a known constant; the exact validator shifts the threshold by the constant (tail 1 if the constant exceeds `tau`).
- **Accepted as written:** Q1, 2, 3, 4, 6, 8, 9, 10, 11, 13 to 20, 23, 26, 27, 29, 30 (numbering of my first list).

## Review fixes applied (26 Oct 2026, `review_io_tools.md`; builds at 0 errors, 0 overfull or underfull boxes)

- **Items 1, 9, 11.** The check `objective` of `validate` is one-sided (`max{0, recomputed - objective} / max{1, |objective|}`); for `evaluation_total` the recomputed value is `P_E + C_D u`; `z` reads 0 where NaN; the check `damage_rows` is removed. Day cost: Eq. `eq:a_daycost`.
- **Item 8.** `removed_mean` of the replay validator is one-sided and skipped where `z` is NaN.
- **Item 7.** The Gamma validators take an argument that is `None` from the input block `validation` (C adds the key), then the default.
- **Item 12.** The completion runs only when the mapping set a value; binaries without a `Start` stay free, so the copy can be a mixed-integer program.
- **Item 13.** Readers treat a missing key as `null` (`x` is "missing or null" in `sec:warm_interface` and in `validate`).
- **Items 15, 16, 17, 18, 19.** `u` is Eq. `eq:c_u`; the plot shows the original model counts, or the presolved counts read from `runner.out`; `skipped` applies after a start or a hint and `enabled` is the option value; the marks `D` and `I` say what the key `idle` is; `cfg` points to `load_config`, `sec:api`.
- **Suggestions.** Ties go to the smaller `H2` (grid: smaller `H1`, then `H2`). `requested_mip_gap` and `requested_time_limit_seconds` are Gurobi's parameters after all options (also `gurobi_params`); the case row entry is renamed `requested_time_limit_seconds`. Sweeps call the internal entry point `solve_mapping(data, results_path=None, *, warm_start=None, warm_start_mode="start")` (the name is my proposal), write no temporary file, and keep the latest result with a solution when `warm_start=True`. The greedy loop pass tests the reliability row in the form built at the cell. Day and phase costs are cited, not restated, in the sweep section. The layout lists `experiments/` and `results/validator_tests/`. The mode "Gamma diagnostic" runs on the replay and exact validators.
- **Not applied (not in the coordinator's list):** the check `E > H_1` whenever `E` is given (writers A and C); items 2 to 6, 10 and 14 (their sections).

## Points that other writers must align with my text

- **Result file extension.** `fragments_C.tex` says "Any other extension gives YAML". My `sec:api` says `solve()` raises `ValueError` for an extension other than `.yaml`, `.yml`, `.json`, `.h5`, `.hdf5` (accepted by the user's answers, question 2). C's sentence must change.
- **`warm_start` report.** My `sec:warm_reports` has the extra entries `warm_start.mode` and the status value `"skipped"` of `relaxation_warm_start`; C's table lists neither.
- **Keys read by my sections that C's table has**: `status`, `objective`, `bound`, `mip_gap`, `F`, `M`, `L`, `H1`, `H2`, `T`, `model`, `component_names`, `tau`, `mu_0`, `repair_model`, `objective_mode`, `evaluation_horizon`, `x`, `m`, `r`, `idle`, `mu`, `v`, `z`, `u`, the cost report keys, `performance` (all entries I use, among them `optimizer_call_seconds`, `branch_and_bound_nodes`, `requested_mip_gap`), `warm_start`, `relaxation_warm_start`, `optimization_progress`. **Keys C lacks**: `R`, `K`, `g_mu`, `g_v`, `g_R`, `g_K`; `presolved_performance` (the plot reads it when present; it is optional).

## Remaining questions for Claude

None. One check is left for Claude: after writer C's revision, compare my key list (above) with the output table again.

## Evidence basis (all read from main's code in full unless stated)

- `main:src/fleet_management/horizon_sweep.py` (716 lines): sweeps, selection (99-129), gradient and stop (146-231), reports (234-328, 638-716), loop and checkpoints (335-555, 558-635).
- `main:.../degradation_model/base.py`: registry (127-161); run options (1202-1247); warm starts (1403-1603); `build_context` (1609-1660); `build_objective` (1735-1811); `extract_solution` (1855-1972); `solve_mixed` (2007-2158). Gamma block (344-1190) not read in full (it leaves the package).
- `main:.../greedy_warm_start.py` (550 lines, in full); `main:.../utils/mixed_plotter.py` (882 lines, in full); `main:.../validation/validator.py` (read lines 1-300, 584-640, 875-900, 1084-1322 of 1322); `dashboard_rainflow.py` (lines 1-560 of 810); `dashboard_gamma.py` (lines 1-160 of 704); `validator_dashboard.py` (lines 1-60 of 1795).
- Gamma validators: `gamma_replay_validator.py` (243 lines, in full), `gamma_tail_validator.py` (561, in full), `gamma_stochastic_validator.py` (473, in full), `gamma_validator.py` (lines 1-120 of 675), `moschopoulos_tail_probability` in `gamma_tail_bound.py` (928-1007).
- `lore:src/fleet_management/solver.py` lines 55-175 and 555-790 (hint, error capture, `n_workers`, result writing). `main:README.md`, `main:pyproject.toml`, `main:examples/regression/RegressionREADME.md` (lines 1-140), `main:examples/regression/check_horizon_sweep.py` (lines 1-200), `merge/reports/C_tooling.md` (in full).
- Not read: the dashboard tabs after line 560 of `dashboard_rainflow.py`, the Gamma dashboard tabs, the legacy Gamma backend. The dashboard text is therefore a specification of tabs and margins, not a transcription of main's pages.
- Copies of main's files used for reading: scratchpad `spec_D/main_*.py` (session-scoped; nothing in the repository was changed).

## Check script

`checks/D_sweep_rules.py` (standard library only; run with `python -I`). It extracts main's functions with `ast` from `git show main:...` and runs them in an isolated namespace.
1. Day mapping (Eq. `eq:d_daymap`) against main's `old_step`: 9072 cases agree; boundary cases for `H1 > H1'` and for operating days that repeat.
2. Gradient class, eligibility and stopping rule (Section "Early Stop") against `_annotate_latest_gradient` and `_gradient_stop_reason`: 17933 random prefixes agree; both stop reasons occur (438 and 569 times); boundary case of a zero gradient with tolerance 0.
3. Proof rule: a case with status `optimal` and gap 0.08 is proven under the decided rule (main's `1e-8` rule would have chosen another case); tie rule; error rows are not feasible.
2b. Default early-stop threshold (the case's own requested gap): 3000 random pairs agree with main's `_gap_qualifies_for_stopping` called per case; worked examples of the text (gap 0.08 qualifies against 0.1, gap 0.12 does not).
4. Clopper--Pearson upper bound: the Beta(f+1, n-f) quantile equals the binomial-CDF definition (5 cases); zero failures in 100000 runs gives 2.9957e-5.
No proposition is stated in my sections, so no further proof checks apply.
