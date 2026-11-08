# WP8 plan, part `tools`: sweeps, warm starts, plots, validators

Status (7 Nov 2026): done; see section Review.

- **Goal.** Show that each tool does what its spec section says, on cases whose expected outcome is derived by hand.
- **Spec** (`lore` copy, `C:\Users\utente\Documents\GitHub\Fleet_Management\spec\spec.tex`): Package Layout 169-211;
  the reports `warm_start` and `relaxation_warm_start` 1831-1864; Horizon Sweeps 1887-2106; Warm Starts 2109-2273;
  Plotting 2276-2378; Validation Tools 2381-2710. Also the Result Dictionary 1714-1784 and Result Files 1866-1884,
  because the tools read results.
- **Independence rule** (`wp8_interface.md`). `src/` may be read only for names, signatures and exception types.
  Two seams are allowed because the spec fixes the call: `fleet_management.horizon_sweep.solve_mapping` may be
  replaced with `monkeypatch` by a fake that returns a hand-built result (spec 1949: each case is solved with
  `solve_mapping`); `matplotlib.figure.Figure.savefig` may be wrapped with `monkeypatch` to keep the figure that a
  plot function saves, so the test can inspect its axes. Do not read the existing `tests/` files before your tests
  are written.

## 1 Files

| File | Content | Task |
|---|---|---|
| `tests/test_spec_validation.py` | structural validator, dashboard loaders and margins, the three Gamma validators, the dispatcher | T1 |
| `tests/test_spec_sweeps.py` | ranking, early stop, report and checkpoint, case errors, parallel cases, the grid | T2 |
| `tests/test_spec_warm_starts.py` | interface, day mapping, modes, completion, relaxation start, warm starts in sweeps, greedy heuristic | T3 |
| `tests/test_spec_plots.py` | schedule plot and sweep plot | T4 |
| `tests/data/tools/` | hand-written inputs, results and sweep reports; each file with a first comment line saying what it covers (YAML) | all |

## 2 Tests

### 2.1 Validation (task T1; no Gurobi unless stated)

Hand-built pairs: an input (`F = 2`, `M = 1`, `L = 1` or `2`, `H = [1, 2]`) and a result file written by hand in YAML
with a schedule whose numbers the test derives from the spec (the tracked means by the day maps).
- **`validate`**: the valid pair passes every check, with the report keys of 2437-2445 and the checks in the order of
  the table 2423-2431 (eight `loop_q` always listed). Then one broken copy per check, each breaking exactly one row:
  `binary` (an entry 0.4: violation 0.4), `one_activity`, `mission_coverage`, `depot_gating` (also a replacement with
  `allow_replacement: false`), `idle`, each `loop_q` (violation `q_T - q_H1`), `objective` (saved objective below the
  recomputed one: violation `(r - o)/max(1, |o|)`); assert the violation number. A broken `binary` must not change
  the other violations more than the spec implies.
- **Errors** of 2410-2414: missing file; unsupported extension; result without `x`; input failing a check; sizes or
  shapes that differ (the message names them); a missing `m`, `r`, `mu`, `z`, `u` or `objective`; a non-finite entry
  in `x`. Report path: `.json` accepted, no extension gets `.yaml`, the folder is created.
- **Status rule**: a result with status `time_limit` and a valid schedule passes, and the status is echoed.
- **Recomputed objective** (2433): in each objective mode, by hand.
- **Dashboards** (2447-2502; streamlit absent): `import fleet_management.validator_dashboard`,
  `validation.dashboard_inspector`, `validation.dashboard_gamma` work without streamlit; `load_inspector` and
  `load_gamma_diagnostic` on the hand pairs; a result whose `model` differs from the input raises `ValueError`;
  `load_gamma_diagnostic` on an input without Gamma cells raises `ValueError`. **Margins** (table 2479-2488): one
  hand-computed margin per row of the table (G, IG, Ga, Mk, Ca, Ho, Be, Ch), through the inspector's loaded data
  (find the margin values in the returned mapping by reading `load_inspector`'s docstring and return keys).
- **Gamma replay validator** (2525-2571): a valid Gamma pair (`ardinf` and `ard1`); one broken copy per error field
  (`tracked_mean` below the reference; `mean_anchor`; `removed_mean`, `z` below `rho (mu - g)`; `cap`, `mu` above
  `A1/beta*` computed in the test with scipy as in `wp8_plan_oracle.md` section 2; `repeatability`), each giving one
  record with the right `field`, `expected`, `saved`, `error`, 0-based `i, l, k`; one copy per global violation
  (2538-2545); the event of each day (2514); `raise_on_failure=True` raises `AssertionError`; the file version writes
  the report (`null` for a non-finite value) before raising; tolerance `<= 0` raises `ValueError`; the errors of
  2519-2522.
- **Exact validator** (2573-2630): tails against an independent computation: one Gamma term `scipy.stats.gamma.sf`;
  two terms with different rates by `scipy.integrate.quad` of the convolution; a known constant `c` (rate omitted)
  shifts the threshold to `tau - c`, and the rules for `c >= tau`; a repair multiplies the rates of repairable terms
  by `1/(1 - rho)`; `ard1` frozen part. Pass rule and records with field `reliability`; `include_steps` keys;
  `RuntimeError` with `max_series_terms = 1` on a case that needs more terms.
- **Monte Carlo validator** (2632-2680): the failure rate of a one-cell schedule within 4 standard errors of the exact
  tail (seeded, `repetitions = 20000`); the same arguments give the same report; the Clopper-Pearson bound equals
  `scipy.stats.beta.ppf(confidence_level, f + 1, n - f)` (1 if `f = n`); pass rule with a target; argument errors
  (a flag for `repetitions`; `confidence_level = 1`; `maximum_schedule_failure_rate = 1.5`); `excluded_cells` in a
  mixed fleet with `complete_fleet_validation` false.
- **Dispatcher** (2682-2709): the three modes; default `deterministic`; the None rule reads the input block
  `validation`; an unknown mode and an unknown key of `validation` raise `ValueError`; `raise_on_failure` acts on the
  dispatcher's report.

### 2.2 Sweeps (task T2)

Fake solve (no Gurobi): replace `fleet_management.horizon_sweep.solve_mapping` by a function that reads `H` from the
mapping it receives and returns a hand-built result dictionary with the keys of 1716-1753 the sweep reads (status,
objective, bound, mip_gap, the cost report, performance with `requested_mip_gap`, `requested_time_limit_seconds`,
sizes, timing), from a table of `J_op_average` per `H2` written in the test. Record the mappings it receives.
- **Case inputs** (1946-1949): each case mapping has `H = [H1, H2]`, the given `objective_mode` and
  `evaluation_horizon`; `H1` from a scalar `H` or the first entry; the second entry ignored.
- **Arguments** (1928-1942, 1957): every invalid value raises `ValueError` before the first case (the fake is never
  called): duplicates, zero, empty list, `gradient_tolerance < 0`, `flat_gradients_required = 0`, `minimum_cases =
  1`, `maximum_mip_gap_for_stopping = -1`, `warm_start` with `n_workers = 2`, `stop_on_gradient` with `n_workers =
  2`, `H` a list of three, `evaluation_total` without `E`, `E <= H1`; non-mapping data raises `TypeError`.
- **Gradient and classes** (Eq. `d_gradient`, 1971-1985): a table of `J_op_average` gives, by hand, each `g_n`, class
  and eligibility, annotated on every row also with `stop_on_gradient=False`; `unavailable` for the first case and
  next to an infeasible case.
- **Stop rule** (1987-1998): a `cost_increase` stop; a `flat_gradient` stop after `flat_gradients_required` flat
  cases; no stop before `minimum_cases`; no stop when a case does not qualify (MIP gap above the threshold, with and
  without `maximum_mip_gap_for_stopping`); `candidate_range_exhausted`; `stopped_early`.
- **Ranking** (1903-1910): best proven and best feasible, ties to the smaller `H2`; a `time_limit` case is feasible
  not proven; an infeasible case is neither.
- **Report and checkpoint** (2003-2047): every report key and row key; with `output_path`, a fake that inspects the
  file after each case sees `complete: false` and reason `in_progress`; the final file is the returned report;
  `formulation_growth` by hand; a case that raises gives the row `status: "error: <message>"` with `null` entries
  except the sizes, an ERROR log record with `H1` and `H2` (`caplog`), and the sweep continues.
- **Grid** (2058-2106): `sweep_horizon_grid` with the fake: `H1` sorted and unique; `objective_mode`
  `evaluation_total` with `E`, else `operating_average`; ranking by `P_E` (Eq. `d_projected`, by hand) or by
  `J_op_average`; tie to the smaller `H1`, then `H2`; `E` not above every `H1` raises `ValueError`; report keys.
- **Real solves** (Gurobi, `pytest.importorskip`): one `sweep_operating_horizons` over `H2 in [2, 3, 4]` on a small
  G input from `tests/data/tools/`, with `output_path`; one with `n_workers = 2` (rows ordered by `H2`); one grid
  over `H1 in [1, 2]`, `H2 in [2, 3]`. Assert the structure and that each row's `J_op_average` equals its result's.

### 2.3 Warm starts and the greedy heuristic (task T3)

- **Interface** (2126-2135; Gurobi for `solve`): invalid `warm_start_mode` raises `ValueError`; a source with
  `H1 = 0`, `H1 = 1.5`, `H1 = True`, or an `x` with two axes, or a wrong last axis, raises `ValueError`; a source
  without `x` has no effect (`applied` false, `mode` null).
- **Day mapping** (Eq. `d_daymap`; Gurobi): an input with zero mean increments (`mu: 0`, `v: 0`), so every schedule is
  feasible, `C_M = 0`, solved with a source of other horizons in mode `start`, with
  `gurobi_params: {SolutionLimit: 1}` so that the accepted start is the returned solution. Expected `x`, `m`, `r` by
  hand from `sigma(k)` for (`H1 = 2, H2 = 3`) from a source with (`H1' = 1, H2' = 2`) and from one with
  (`H1' = 3, H2' = 2`); other dimensions `F' > F` (values only for shared indices). If Gurobi does not return the
  start exactly, the test checks `warm_start["values"]` instead (count derived by hand: `x`, `m`, `r`, and the
  indicator selectors at `indicator` cells).
- **Modes** (2160-2178): `hint` reports `completion.status = "not_requested"`; a greedy source in mode `start`
  reports `requested: true` and status `completed`; `both`; `values` counts each variable once.
- **Relaxation start** (2180-2195): option false gives status `disabled`; option true with a start gives `skipped`;
  option true without a start gives a Gurobi status string and `values_copied >= 0`.
- **In sweeps** (2197-2204, fake solve of 2.2 recording the `warm_start` argument): the first case of each `H1` has
  no source; a later case gets the result of the nearest preceding case with a solution; an error case and a case
  without a solution are skipped as sources.
- **Greedy heuristic** (2206-2273; no Gurobi): `load_config` of hand inputs, then `build_greedy_warm_start`. By hand:
  the mission order of each day (ties to the larger `j`), the train chosen (lexicographic key, ties to the smaller
  `i`), the repairs above `r tau` with `(1 - rho)` for every law; the loop pass on a case that needs one repair in the
  operating phase (expected day by the tie rule "the later day"); a replacement when no repair lowers the excess; the
  Wiener rule (repair and replacement tried on each free day, tie to the repair, then to the later day); a failure
  counted when nothing helps; `repair_trigger = 1.2` raises `ValueError`; the returned keys and the report keys.

### 2.4 Plots (task T4; no Gurobi: results are hand-written files)

- **Schedule plot** (2282-2343): capture the figure through `Figure.savefig`. Rows: one per cell with the two-line
  label; columns `initial` and `1..T`; the dashed line after `H1`; the labels `transitory` and `operating`; colour bar
  label `tracked mean / failure threshold`; the mark of each field by the first rule of the table (a result with one
  of each: `P`, `R`, `M1`, `D`, `I`; values above 0.5 count as 1); legend with five marks; summary panel texts
  (status colour: green `optimal`, amber otherwise, grey `unknown`); the counts from `performance`, and from a
  `runner.out` with two solves (the last `Presolved:` line and the first `Variable types:` line after it). Errors of
  2307-2311, each with a hand-made bad file. Default image `output.png` (`monkeypatch.chdir`); no extension gets
  `.png`; `.pdf` accepted.
- **Sweep plot** (2345-2378): hand-written reports; the selection rule for equal `H2` (case with a solution; smaller
  `J_op/H2` with `numpy.isclose`; larger bound; earlier report); the title with the sizes (a missing size left out);
  three panels with the marks of the table; the horizontal line at `maximum_mip_gap_for_stopping`; errors of
  2360-2363.
- Each test also writes its PNG to `tmp_path`; T4 copies one PNG of each plot to its scratchpad folder for Claude to
  look at.

## 3 Executor tasks (Sonnet agents)

Common: read `merge/wp8_interface.md`, this plan and the spec lines named. Local runs with a copy of
`merge/wp45_work/wp6/gamma/run_tests_gamma.py` in the scratchpad (Gurobi tests skip). Progress file
`merge/wp45_work/wp8/tools/task<id>_progress.md` with a `Status:` line. A test that fails locally is a test error (fix
it) or a code defect (keep the test; list it in the progress file with the spec line and the observed behaviour).

- **T1**: 2.1. **T2**: 2.2. **T3**: 2.3. **T4**: 2.4. Independent; each owns its test file and its own subfolder of
  `tests/data/tools/` (`validation/`, `sweeps/`, `warm_starts/`, `plots/`).

## 4 Questions

None open.

## 5 Removals

None.

## 6 Review

Claude, 7 Nov 2026. Tasks T1 to T4 done. Euler: all pass. Code defects D1 (replay validator, two strings for one violation) and D2 (overlapping texts in the summary panel) fixed; parallel sweeps start their workers with `spawn` (a `fork` warning on Euler). Spec findings S2, S4, S6 to S9.
