# WP7 plan, part `validator`: the structural validator and the dashboards

Status (29 Oct 2026): done. Plan (sections 1-7), execution (Tasks A, B, C) and review (section 8) finished.
Next: Claude adds the spec sentences of section 8 and commits. The scratchpad scripts named `S\...` below hold the
fixture and the expected values; copy them into `merge/` before the session ends.

Code: the worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge` at 0421e80. Spec lines are those of
the worktree's `spec/spec.tex`; its tool sections (165-261, 2256-2658) equal the `lore` copy. Abbreviations: G
Gaussian, IG inverse Gaussian, W Wiener, Ga Gamma, RF rainflow, Be Bernstein, Ch Chernoff; "cell" = component `l`
of train `i`. Every file and test path below is relative to the worktree.

## 1. Gaps

**`validation/validator.py`** (1321 lines; all of it is `main`'s legacy code)

| # | Code | Gap | Spec |
|---|---|---|---|
| V1 | 27-33 | signature `validate(input_path, degradation, results_path, validation_path, tol)`; the spec has no `degradation` | 2373 |
| V2 | 13, 109 | imports the alias `solver._read_input`; calls `extract_degradation_parameters`, which no module defines (removed with `utils/model_registry.py`) | 2362 |
| V3 | 14-21 | imports the old Gamma backend `gamma_process` (shape-rate states) | 2362-2364 |
| V4 | 24, 88, 1083-1095 | reads YAML and JSON results only, with a private reader | 2363, 2664, 2684 |
| V5 | 1098-1151 | requires the old keys `degradation`, `alpha`, scalar `H`, `u` of shape (2H,), `z` of shape (F, 2H) | 1699-1725 |
| V6 | 1176-1211 | compares F, M, H, L and shapes with `2H`; not H1, H2, T | 2389 |
| V7 | 132-138 | check `solver_status_optimal`; the status is not a check, it is echoed | 2394 |
| V8 | 149-157 | `x_binary`: only `x`, distance to the nearest integer; the spec checks `x`, `m`, `r`, distance to {0, 1} | 2399 |
| V9 | 159-171 | name `assignment_sum_j_x_le_1`; spec name `one_activity` | 2400 |
| V10 | 173-182 | `demand_sum_i_x_eq_1` sums the depot `j = 0` too, so a valid schedule fails; spec `mission_coverage`, `j >= 1` | 2401 |
| V11 | none | no `depot_gating`, no `idle` check | 2402-2403 |
| V12 | 184-195 | `u_ge_mu` (a vector `u` per day); not a check of the spec | 2398-2406 |
| V13 | 197-226 | `mu_periodic`, `v_periodic` at days `2H`, `H`, `v` only for G; spec: `loop_q` for 8 keys, days T and H1, NaN skipped | 2404 |
| V14 | 1214-1263 | objective of the old G model (`C_S`, `C_P`), test `abs(r - o) <= tol`; spec: Eq. `a_objective` in 3 modes, violation `max{0, r - o}/max{1, abs(o)}` | 2405, 2408, 662-674 |
| V15 | 123-129, 236-237 | report has `degradation`, lacks `recomputed_objective` for other models; spec key list differs | 2410-2418 |
| V16 | 23, 230-234 | only G is validated; IG raises `NotImplementedError`; W, Ga, RF refused | 2360-2364 |
| V17 | 248-580 | `validate_baseline_assignment_feasibility`, `build_assignment_feasibility_dataframe`: dropped by decision | decisions 1524-1526 |
| V18 | 583-1080 | `build_gamma_diagnostic_dataframe`, `validate_gamma_synthetic_diagnostic`: dropped by decision | decisions 1526; spec 2451 |
| V19 | 241-243 | prints to stdout; the spec writes and returns the report only | 2370 |
| V20 | none | no function that checks a result without writing a file; the inspector's tab "Validation" needs one | 2442 |

**`validation/dashboard_rainflow.py`** (810 lines) becomes `validation/dashboard_inspector.py`

| # | Code | Gap | Spec |
|---|---|---|---|
| D1 | file | name `dashboard_rainflow.py`; spec name `dashboard_inspector.py`, mode "Result inspector" for every model | 202, 2432-2433 |
| D2 | 34 | `import streamlit` at module level: the pure functions cannot be imported or tested without Streamlit (an extra) | 173, 2423 |
| D3 | 44-86 | private reader, YAML and JSON only; old keys `alpha`, `degradation`, `method`, `u` array | 2363, 2429 |
| D4 | 459-489 | reads the input through `solver._read_input` and the removed `model_registry`, falls back to raw YAML and `xi` | 2362 |
| D5 | 111-152 | `reconstruct_state` rebuilds R and K; the result now holds them | 1723, 2447 |
| D6 | 155-181 | margins for RF only, one bound for the fleet, a default bound; not the exact row of each cell's model | 2433, 2447 |
| D7 | 216-239 | `structural_checks` repeats the old checks (`u_ge_mu`, demand with the depot) | 2442 |
| D8 | 353-391 | own sidebar with epsilon and bound overrides | 2428 |
| D9 | 500-503 | tab "Vehicle timeline"; spec "Train timeline" | 2439 |
| D10 | 506-538 | Overview lacks models, H1, H2, T; shows a bound and one epsilon | 2436 |
| D11 | 582-654 | Repeatability: `mu` and `v` only | 2438 |
| D12 | 657-751 | timeline without the marks P, R, M_j, D, I; a table with `xi` and `C_P` | 2439, 2300-2306 |
| D13 | 265-343 | Heatmap of `mu` with icons; spec: margins over cells and days | 2440 |
| D14 | 356, 808-810 | debug line "BUILD CHECK"; a `__main__` runner | 2425 |

**`validation/dashboard_gamma.py`** (704 lines)

| # | Code | Gap | Spec |
|---|---|---|---|
| E1 | 8 | `import streamlit` at module level (as D2) | 2423 |
| E2 | 10-13, 55-70 | runs the dropped synthetic diagnostic; spec: the replay and exact validators | 2451 |
| E3 | 100-135 | own sidebar; an input that holds a schedule `x`; a log file | 2428-2429, 2451 |
| E4 | 72-79 | tabs (overview, shape timeline, fleet heatmap, raw); spec: overview, mean with cap, exact tail with heatmap, raw | 2453-2458 |
| E5 | 159-603 | Gamma shape `A` and loop `A_2H <= A_H` of the removed formulation; the cell tracks `mu` only | 1214, decisions "Gamma formulation" |
| E6 | 606-682 | `_render_gamma_failure_probability` is dead code (call commented out at 90-91) | none |

**`validator_dashboard.py`** (1795 lines)

| # | Code | Gap | Spec |
|---|---|---|---|
| S1 | 40-57 | modes "Baseline Gaussian validator", "Gamma synthetic diagnostic", "Rainflow Inspector"; spec: "Result inspector", "Gamma diagnostic" | 2432, 2450 |
| S2 | 59-1795 | the baseline Gaussian page (`alpha`, `2H`, `degradation_scale`, log file): dropped with V17 | decisions 1524-1526 |
| S3 | 153-199 | sidebar: log path, alpha override, degradation scale; spec: mode, input, result, tolerance 1e-6, one load button | 2428 |
| S4 | 6 | start command `streamlit run scripts/validator_dashboard.py` | 2425 |

**Tests and fixtures**

| # | Code | Gap | Spec |
|---|---|---|---|
| T1 | `tests/test_validator_negative_cases.py`, `tests/test_validator_positive_case.py` | old signature and check names; need `input/data_test_baseline.yaml` and `results/output_baseline.yaml`, absent | 2373-2418 |
| T2 | `results/validator_tests/*.yaml` (10 files) | old output format (`alpha`, `degradation`, scalar `H`) | 179, 1699-1725 |
| T3 | none | no dashboard test | interface "Rules" |

**Items of other packages (not planned here)**
- `README.md:212-232`: old validator and dashboard text (WP8, README).
- `solver.py:20`, alias `_read_input` "used by the validation tools": no caller is left after this plan (WP8 or Claude).
- `fleet_management/__init__.py:8` imports `validate` from `validation.validator`: the name stays (`plots`).
- `gamma_utils/*` validators read YAML and JSON only, with private readers, and old keys (`gamma_validators`).

## 2. Executor tasks

Three tasks with disjoint files. Order: Task A first (Tasks B and C import its names); B and C may run in parallel
with A, because every name they import is fixed here, but B runs its tests only after A is done. Scratchpad of
this plan: `S = C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\validator\`.
Local runner (no pytest, no Gurobi here): copy `<scratchpad>\wp6\gamma\run_tests_gamma.py` (it has `tmp_path`,
`parametrize`, `raises(match=)`) to `S\run_tests_validator.py`, set its `TMP` to `S\tmp`, and add `pytest.approx`
(the class `_Approx` of `<scratchpad>\wp6\rainflow\review_run.py`). Run `python S\run_tests_validator.py
tests\test_validator.py` from the worktree root.

Shared rules for every step:
- No module of this plan imports `gurobipy`, `degradation_model.base`, `gaussian.py` or `wiener.py` directly
  (`gamma.py` and `inverse_gaussian.py` load without Gurobi, spec 193 of the `lore` copy).
- `streamlit` is imported only inside the functions whose names start with `render_`.
- Day numbers shown to the user are 1-based (`day = k + 1`), trains too (`train = i + 1`); components are shown by
  their name, `cfg.component_names[l]`. Arrays stay 0-based.
- Every function gets a one-line docstring that names its spec section or equation.

### Task A: `validation/validator.py`, `tests/test_validator.py`, fixtures (6 steps)

**A1. Fixtures** (spec 179). Create `results/validator_tests/input.yaml` as a copy of `S\input.yaml`: the spec's
YAML example (spec 2727-2781 of the `lore` copy) with `anchor_mu_0[2][0] = 0.05` instead of `0.00` and without
`fast`, `gurobi_params`, `progress_interval_seconds`, `relaxation_warm_start`. Run `python S\fixture_values.py`;
it builds the valid result by the day maps of the spec (rows at equality) and writes it with
`solver._save_results` to `S\result.yaml` (also `.json`, `.h5`). Copy `S\result.yaml` to
`results/validator_tests/result.yaml`. Check: `config.read_input` of the copy has 51 keys, `objective` =
533.8820000000001, `u` = 0.56; the output of the script matches `S\fixture_values_out.txt`. The schedule
(day: train 1, train 2, train 3; `D` depot day, `-` idle):

| Day | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Train 1 | M1 | M2 | D, both replaced | M1 | M2 |
| Train 2 | M2 | M1 | M1 | D, both replaced | M1 |
| Train 3 | - | D, Brake pad repaired | M2 | M2 | D, both replaced |

**A2. `validator.py`, module head and reader** (V1-V5, V17-V19; spec 2360-2364, 1851-1868, 2684). Replace the
whole file (the deletion of the legacy code is mechanical). New content, in this order:
- Module docstring: "Structural validator of a saved result (spec, section "Structural Validator")."; then
  three sentences: `validate` reads the two files, checks, writes and returns the report; `check_result` runs the
  checks on a `FleetConfig` and a result dictionary without files (the dashboards call it); `read_result` reads a
  result file of `solve()`.
- Imports: `json`, `os`, `Path`, `numpy as np`, `yaml`; `from fleet_management.config import INPUT_EXTENSIONS,
  FleetConfig, load_config, read_input`; `from fleet_management.solver import RESULT_EXTENSIONS, RESULT_KEYS,
  _HDF5_DATASET_KEYS`. No `pandas`, no `gamma_process`.
- Constants: `REPORT_EXTENSIONS = (".yaml", ".yml", ".json")`; `LOOP_KEYS = ("mu", "v", "R", "K", "g_mu",
  "g_v", "g_R", "g_K")`; `SIZE_KEYS = ("F", "M", "L", "H1", "H2")`; `ARRAY_KEYS = _HDF5_DATASET_KEYS` (the
  float64 arrays of the result, spec 2688).
- `read_result(path) -> dict`: `path = Path(path)`; if `path.suffix.lower()` is not in `RESULT_EXTENSIONS`:
  `ValueError(f"Unsupported result file type '{path.suffix}'. Supported types: {list(RESULT_EXTENSIONS)}")`;
  `data = read_input(path)`; `result = {key: data.get(key) for key in RESULT_KEYS}` (a missing key is `None`,
  spec 1867); every key of `ARRAY_KEYS` that is not `None` becomes `np.asarray(value, dtype=float)` (a `null`
  entry becomes NaN).
- `check_sizes(cfg, result) -> None` (spec 2389), in this order, each a `ValueError`:
  1. `result.get("x") is None`: "the result has no solution: 'x' is missing or null."
  2. for `key` in `SIZE_KEYS`: `saved = result.get(key)`; if `saved is None or int(saved) != getattr(cfg, key)`:
     `f"'{key}' differs between the result ({saved}) and the input ({getattr(cfg, key)})."`
  3. expected shapes: `x` `(F, M+1, T)`; `m`, `r`, `idle`, `mu`, `v`, `z`, `R`, `K`, `g_mu`, `g_v`, `g_R`, `g_K`
     `(F, L, T)`; `tau`, `mu_0`, `v_0` `(F, L)`; `step_costs` `(T,)`. For each key whose value is not `None` and
     `np.shape(value) != shape`: `f"result array '{key}' has shape {np.shape(value)}; the input gives {shape}."`
  4. for `key` in `("m", "r", "mu", "z", "u", "objective")`, if `result.get(key) is None`:
     `f"the result has 'x' but no '{key}'."` (Question Q3)
  5. for `key` in `("x", "m", "r", "idle")`, value not `None` and not `np.all(np.isfinite(value))`:
     `f"result array '{key}' has a non-finite entry."` (Question Q3)
- Keep `_resolve_validation_path` (1295-1306) unchanged. Keep `_save_validation_report` (1309-1321), with
  `open(path, "w", encoding="utf-8")` and `json.dump(report, f, indent=2, allow_nan=False)`.
Check: the module imports without gurobipy (`S\run_tests_validator.py` loads it); `read_result` of the three
files `S\result.{yaml,json,h5}` gives equal arrays, `R` all NaN, `H == [2, 3]`.

**A3. `validator.py`, the checks** (V7-V15, V20; spec 2392-2418). Add after `check_sizes`:
- `recomputed_objective(cfg, result) -> float` (spec 2408, Eqs. `a_daycost`, `a_phases`, `a_objective`):
  `x, r = (np.asarray(result[k], dtype=float) for k in ("x", "r"))`;
  `z = np.nan_to_num(np.asarray(result["z"], dtype=float), nan=0.0)`;
  `day = cfg.C_M * x[:, 0, :].sum(axis=0) + np.einsum("il,ilk->k", cfg.C_R, z)`; if `cfg.allow_replacement`:
  `day = day + np.einsum("il,ilk->k", cfg.C_rep, r)` (the last sum of Eq. `a_daycost` is absent otherwise);
  `J_init, J_op = float(day[:cfg.H1].sum()), float(day[cfg.H1:].sum())`;
  `penalty = cfg.C_D * float(result["u"]) if cfg.C_D > 0 else 0.0`; `mode = cfg.options["objective_mode"]`:
  `total` gives `J_init + J_op + penalty`; `operating_average` gives `J_op / cfg.H2 + penalty`;
  `evaluation_total` gives `J_init + (E - cfg.H1) / cfg.H2 * J_op + penalty`, `E =
  cfg.options["evaluation_horizon"]`.
- `_loop_violation(value, H1, T) -> float` (check `loop_q`): `0.0` if `value is None`; else
  `excess = q[:, :, T - 1] - q[:, :, H1 - 1]` on `q = np.asarray(value, dtype=float)`, keep the finite entries,
  return `max(0.0, float(excess.max()))`, or `0.0` if none is finite.
- `check_result(cfg, result, tol=1e-6) -> dict`: call `check_sizes`; `x, m, r` as float arrays;
  `x0 = x[:, 0, :][:, None, :]`; checks appended in this order by a local `add(name, violation)` that appends
  `{"name": name, "passed": bool(violation <= tol), "violation": float(violation), "tolerance": float(tol)}`:
  1. `binary`: `max(float(np.max(np.minimum(np.abs(a), np.abs(a - 1.0)))) for a in (x, m, r))`;
  2. `one_activity`: `max(0.0, float(np.max(x.sum(axis=1) - 1.0)))`;
  3. `mission_coverage`: `float(np.max(np.abs(x[:, 1:, :].sum(axis=0) - 1.0)))`;
  4. `depot_gating`: `max(0.0, float(np.max(m + r - x0)))`, and if not `cfg.allow_replacement` the maximum of
     that and `float(np.max(r))`;
  5. `idle`, only if `result.get("idle") is not None`: `float(np.max(np.abs(idle - (x0 - m - r))))`;
  6. `loop_mu`, `loop_v`, `loop_R`, `loop_K`, `loop_g_mu`, `loop_g_v`, `loop_g_R`, `loop_g_K`: always all eight,
     `_loop_violation(result.get(q), cfg.H1, cfg.T)` (Question Q4);
  7. `objective`: `o = float(result["objective"])`, `r_obj = recomputed_objective(cfg, result)`, violation
     `max(0.0, r_obj - o) / max(1.0, abs(o))`.
  Return `{"status": result.get("status"), "solver_objective": o, "recomputed_objective": r_obj, "tolerance":
  float(tol), "checks": checks, "passed": all(c["passed"] for c in checks), "max_violation": max(c["violation"]
  for c in checks)}`, keys in this order (spec 2410-2418).
Check: `python S\reference_checks.py` and the new `check_result` give the same violations on the fixture and on
each corruption of section 3 (compare by hand with `S\reference_checks_out.txt`).

**A4. `validator.py`, `validate`** (V1, V19; spec 2372-2390). Add:
`validate(input_path, results_path, validation_path="validation.yaml", tol=1e-6) -> dict`, docstring with the
numpy-style "Parameters" and "Returns" of the old one, without `degradation`. Steps, in this order:
1. `FileNotFoundError(f"Input file not found: {input_path}")` and `FileNotFoundError(f"Result file not found:
   {results_path}")` if a path does not exist.
2. `ValueError(f"Unsupported input file type '{suffix}'. Supported types: {list(INPUT_EXTENSIONS)}")` and the
   same for the result with `RESULT_EXTENSIONS` and "result file type".
3. `validation_file = _resolve_validation_path(validation_path)`; suffix not in `REPORT_EXTENSIONS`:
   `ValueError(f"Unsupported validation file type '{suffix}'. Supported types: {list(REPORT_EXTENSIONS)}")`;
   `validation_file.parent.mkdir(parents=True, exist_ok=True)`; not writable:
   `PermissionError(f"Validation directory is not writable: {validation_file.parent}")`.
4. `cfg = load_config(read_input(input_file))`; `report = check_result(cfg, read_result(results_file), tol)`;
   `_save_validation_report(report, validation_file)`; return `report`. No `print`.
Check: `validate(input.yaml, result.yaml, <tmp>\v.yaml)` returns `passed: True` with the 14 check names of
section 3, and `yaml.safe_load` of the file equals the returned dictionary.

**A5. `tests/test_validator.py`, part 1** (T1; spec 2372-2418). New file, the tests 1-9 of section 3, with the
helpers of section 3. At most 150 lines.

**A6. `tests/test_validator.py`, part 2.** The tests 10-17 of section 3. Check for A5 and A6: all tests pass with
`run_tests_validator.py`; each failing-check test also asserts that no other check fails.

### Task B: `validation/dashboard_inspector.py` (new), `validation/dashboard_gamma.py`, `tests/test_validator_dashboards.py` (6 steps)

`dashboard_inspector.py` is a new file that replaces `dashboard_rainflow.py` (decision "Validation tools"; spec
202). Do not delete `dashboard_rainflow.py`: Claude removes it (section 6). Reuse from it only the tab layout.
Figures are `matplotlib.figure.Figure` objects built without `pyplot` (`fig = Figure(figsize=...)`,
`ax = fig.subplots()`), so no backend is needed; `st.pyplot(fig)` shows them.

**B1. Loading and margins** (D1-D6; spec 2362-2364, 2429, 2447). Module docstring: "Result inspector of the
dashboards (spec, section "Dashboards", mode "Result inspector")." and one sentence: the functions without the
prefix `render_` need neither Streamlit nor Gurobi. Imports: `Path`, `numpy as np`, `pandas as pd`,
`from matplotlib.figure import Figure`, `from scipy.special import ndtri`, `from fleet_management.config import
_lookup, load_config, read_input`, `from fleet_management.degradation_model.gamma import mean_cap,
smallest_rate`, `from fleet_management.degradation_model.inverse_gaussian import ig_mean_cap`,
`from fleet_management.validation.validator import LOOP_KEYS, check_result, check_sizes, read_result`.
- `load_pair(input_path, results_path) -> tuple`: `FileNotFoundError(f"Input file not found: {input_path}")`,
  `FileNotFoundError(f"Result file not found: {results_path}")`; `cfg = load_config(read_input(input_path))`;
  `result = read_result(results_path)`; `check_sizes(cfg, result)`; if `result.get("model") is None` or
  `np.asarray(result["model"], dtype=object).astype(str).tolist() != cfg.model.astype(str).tolist()`:
  `ValueError("'model' differs between the result and the input.")` (Question Q2); return `(cfg, result)`.
- `cell_label(cfg, i, l) -> str`: the model name; at a rainflow cell `f"rainflow ({cfg.bound_method[i, l]})"`.
- `reliability_caps(cfg) -> np.ndarray` (F, L), NaN elsewhere: IG cells `ig_mean_cap(eta, tau, epsilon)` (Eq.
  `ig_row`); Gamma cells `mean_cap(smallest_rate(mu_inc[i, l], beta_inc[i, l], mu_0, gamma_beta_0,
  replacement_mu, gamma_beta_new, cfg.allow_replacement), tau, epsilon)` (Eqs. `gm_betastar`, `gm_row`), with
  `mu_inc = _lookup(cfg.mu, cfg.mu_trans, cfg.H1, cfg.H2)` and `beta_inc = _lookup(cfg.gamma_beta,
  cfg.gamma_beta_trans, cfg.H1, cfg.H2)` (the same call as `config.py:828-836`, which passes `True` there).
  Scalars as `float(...)` of the `cfg` arrays at `[i, l]`.
- `reliability_margins(cfg, result) -> np.ndarray` (F, L, T): the right side minus the left side of the exact
  row of each cell (spec 2447); where the row has a side row, the smaller of the two differences (Question Q1).
  With `tau, eps` of shape (F, L, 1), `kappa = np.log(1/eps)`, `kg2 = ndtri(1 - eps)**2`, `c = kappa *
  cfg.support_const[:, :, None] / 3`, `s = cfg.s_chernoff[:, :, None]`, `cap = reliability_caps(cfg)[:, :,
  None]`, and `mu, v, R, K` the result arrays (a missing one is all NaN), select by `cell_label`:

  | Cell | Margin | Spec |
  |---|---|---|
  | `gaussian`, `wiener` | `min((tau - mu)**2 - kg2 * v, tau - mu)` | Eq. `g_row` |
  | `inverse_gaussian`, `gamma` | `cap - mu` | Eqs. `ig_row`, `gm_row` |
  | `rainflow (markov)` | `eps * tau - mu` | Eq. `rf_markov` |
  | `rainflow (cantelli)` | `min((tau - mu)**2 - (1 - eps) / eps * v, tau - mu)` | Eq. `rf_cantelli` |
  | `rainflow (hoeffding)` | `min((tau - mu)**2 - kappa / 2 * R, tau - mu)` | Eq. `rf_hoeffding` |
  | `rainflow (bernstein)` | `min((tau - c - mu)**2 - c**2 - 2 * kappa * v, tau - c - mu)` | Eq. `rf_bernstein` |
  | `rainflow (chernoff)` | `np.log(eps) - (K - s * tau)` | Eq. `rf_chernoff` |

  `min` is `np.minimum`; compute inside `np.errstate(divide="ignore", invalid="ignore")`.
Check: on the fixture, `reliability_caps` gives 0.49234697216756246 at (0, 1) and 0.6726905060677382 at (1, 1);
`reliability_margins` equals the matrix of section 3, test 22.

**B2. Tables and marks** (D9-D12; spec 2436-2439, 2443, 2297-2306). Add:
- `margin_table(cfg, result, margins, tol) -> pd.DataFrame`: one row per cell and day, `i` outer, `l`, then `k`;
  columns `train`, `component`, `model` (`cell_label`), `day`, `phase` (`"transitory"` if `k < H1`, else
  `"operating"`), `mu`, `tau`, `mu_over_tau`, `margin`, `holds` (`margin >= -tol`); rows with a NaN margin are
  dropped (spec 2447).
- `overview(cfg, result, table, n=10) -> dict`: keys `status`, `objective` (from the result), `models` (sorted
  set of `cell_label` over the cells), `H1`, `H2`, `T`, `reliability_violations` (`int((~table["holds"]).sum())`),
  `largest_mu_over_tau` (`float(table["mu_over_tau"].max())`), `smallest_margins`
  (`table.nsmallest(n, "margin")`).
- `loop_table(cfg, result, tol) -> pd.DataFrame`: one row per cell (`i` outer) and key of `LOOP_KEYS` whose
  values at days H1 and T are both finite: `train`, `component`, `quantity` (the key), `value_at_H1`
  (`a[i, l, H1-1]`), `value_at_T` (`a[i, l, T-1]`), `margin` (`value_at_H1 - value_at_T`), `holds`
  (`margin >= -tol`) (Eq. `a_loop`).
- `day_marks(result, i, l) -> list[str]`: for each day the first rule that applies, a value counting as 1 if
  above 0.5: `"P"` if `r[i, l, k]`; `"R"` if `m[i, l, k]`; `f"M{j}"` for the first `j >= 1` with `x[i, j, k]`;
  `"D"` if `x[i, 0, k]`; else `"I"`.
- `load_inspector(input_path, results_path, tol) -> dict`: `cfg, result = load_pair(...)`; keys `cfg`, `result`,
  `margins`, `table`, `overview`, `loops`, `report` (`check_result(cfg, result, tol)`), `tol`.

**B3. Figures** (D12, D13; spec 2439-2441). Add, each returning a `Figure`:
- `timeline_figure(cfg, result, i)`: x values `0..T` (0 is the initial state); per component one line of
  `[mu_0[i, l]] + mu[i, l, :]` with marker `"o"`, label the component name; a dashed horizontal line at
  `tau[i, l]` in the same colour, label `f"tau ({name})"`; above each point of day `k >= 1` the text of
  `day_marks(result, i, l)[k-1]` (fontsize 8); a dotted vertical line at `x = H1 + 0.5`; x tick labels
  `["initial", "1", ..., str(T)]`; x label "day", y label "tracked mean mu", title `f"Train {i+1}"`; legend.
- `margin_heatmap_figure(cfg, margins)`: an image of `F*L` rows (cells, train outer) and `T` columns, cmap
  `"RdYlGn"`, `vmin=-vmax`, `vmax=vmax`, `vmax = np.nanmax(np.abs(margins))` or `1.0` if that is 0 or NaN; y tick
  labels `f"Train {i+1} - {name}"`; x tick labels `1..T`; a black vertical line at `x = H1 - 0.5`; colour bar
  label "margin of the reliability row"; title "Reliability margins (negative: the row is violated)".
- `component_figure(cfg, result, l)`: one line per train of `mu[i, l, :]` over days `1..T`, label `f"Train {i+1}"`;
  dashed lines at the distinct values of `tau[:, l]`; x label "day", y label "tracked mean mu", title
  `f"{name} across the trains"`.
Check: each function returns a `Figure` on the fixture, and `fig.savefig(S\tmp\<name>.png)` gives an image that
shows what is listed (look at it).

**B4. `render_result_inspector(state)`** (D8-D14; spec 2432-2448). First line `import streamlit as st`. Tabs, in
this order: "Overview", "Reliability", "Repeatability", "Train timeline", "Heatmap", "Component comparison",
"Validation", "Raw data".
- Overview: `st.metric` for status, objective, H1, H2, T, reliability violations, largest mu/tau; the models as
  one line of text; subheader "Cells with the smallest margin" and `st.dataframe(overview["smallest_margins"])`.
- Reliability: `st.multiselect("Train", ...)`, `st.multiselect("Component", ...)` (all selected by default),
  `st.checkbox("Violations only")`; the filtered `table`, sorted by `margin`.
- Repeatability: `st.dataframe(loops)`; `st.success("Every loop row holds.")` or
  `st.error(f"{n} loop rows are violated.")`.
- Train timeline: `st.selectbox("Train", range(1, F+1))`; `st.pyplot(timeline_figure(cfg, result, train - 1))`;
  caption "Marks: P replacement, R repair, Mj mission j, D depot day without intervention, I idle day."
- Heatmap: `st.pyplot(margin_heatmap_figure(cfg, margins))`.
- Component comparison: `st.selectbox("Component", names)`; the figure; a table by train of the largest `mu`
  and the smallest `margin` of that component (`table.groupby("train")`).
- Validation: metrics "solver objective" and "recomputed objective"; `st.dataframe(report["checks"])`;
  `st.success("All checks passed.")` or `st.error(f"{n} checks failed.")`.
- Raw data: `st.dataframe(table)`; `st.download_button("Download as CSV", table.to_csv(index=False)
  .encode("utf-8"), file_name="result_inspector.csv", mime="text/csv")`.
Check: the module imports with Streamlit absent; the undefined-name check
(`<scratchpad>\wp6\rainflow\names_check_try.py`) reports nothing.

**B5. `dashboard_gamma.py`, pure part** (E1-E6; spec 2450-2458). Replace the whole file. Docstring: "Gamma
diagnostic of the dashboards (spec, section "Dashboards", mode "Gamma diagnostic")." Imports: `numpy`,
`pandas`, `Figure`, and `from fleet_management.validation.dashboard_inspector import load_pair,
reliability_caps`. Functions:
- `gamma_cells(cfg) -> list`: the cells `(i, l)` with `cfg.model[i, l] == "gamma"`, `i` outer;
  `ValueError("the input has no Gamma cell.")` if none.
- `gamma_mean_table(cfg, result, caps, tol) -> pd.DataFrame`: per Gamma cell and day: `train`, `component`,
  `day`, `mu`, `cap`, `margin` (`cap - mu`), `holds` (`margin >= -tol`).
- `gamma_tail_table(cfg, exact_report, tol) -> pd.DataFrame`: one row per record of `exact_report["steps"]`,
  which holds `i`, `l`, `k` (0-based) and `exact_tail_upper_bound` (Question Q5): `train`, `component`, `day`,
  `event`, `tail_upper_bound`, `epsilon` (`cfg.epsilon[i, l]`), `slack` (`epsilon - tail_upper_bound`), `holds`
  (`slack >= -tol`).
- `gamma_overview(replay) -> dict`: `valid`, `repairs`, `replacements`, `global_violations` (the list),
  `violations` (their number), `maximum_errors` (the mapping), from the replay report (spec 2514-2523).
- `load_gamma_diagnostic(input_path, results_path, tol) -> dict`: `cfg, result = load_pair(...)`;
  `gamma_cells(cfg)`; inside the function, import `validate_gamma_replay_schedule` from
  `fleet_management.degradation_model.gamma_utils.gamma_replay_validator` and
  `validate_gamma_tail_bound_schedule` from `...gamma_utils.gamma_tail_validator` (a module of the other plan that
  is broken during its rework then breaks only this function); `replay = validate_gamma_replay_schedule(cfg,
  result, tolerance=tol)`; `exact = validate_gamma_tail_bound_schedule(cfg, result, tolerance=tol,
  include_steps=True)`; return keys `cfg`, `result`, `replay`, `exact`, `overview`,
  `means` (with `reliability_caps(cfg)`), `tails`, `tol`.
- Figures: `gamma_mean_figure(cfg, means, i, l)` (mu over days 1..T, dashed line at the cap, title
  `f"Train {i+1} - {name}: tracked mean and cap A_1/beta*"`); `gamma_tail_figure(cfg, tails, i, l)` (the tail
  upper bound over days, dashed line at epsilon, title `f"Train {i+1} - {name}: exact tail P(D > tau)"`);
  `gamma_tail_heatmap_figure(cfg, tails)` (rows the Gamma cells, columns days, value `tail_upper_bound`, `vmin=0`,
  `vmax = max epsilon`, cmap `"Reds"`, a text `"!"` where `holds` is false).
Check: test 29 of section 3 passes; at most 150 lines for this step.

**B6. `render_gamma_diagnostic(state)` and the dashboard tests.** First line `import streamlit as st`. Tabs, in
this order: "Overview" (metrics repairs, replacements, "replay validator" `"valid"` or `"not valid"`, number of
violations; the global violations as a list; `st.json(maximum_errors)`), "Mean and cap" (`st.selectbox` of the
Gamma cells labelled `f"Train {i+1} - {name}"`; the mean figure; the rows of `means`), "Exact tail" (the same
selectbox with `key="tail_cell"`; the tail figure; the heatmap; `st.success` or `st.error` by `holds`), "Raw data"
(both tables, each with a CSV download). Then write `tests/test_validator_dashboards.py`: tests 18-30 of section 3.

### Task C: `validator_dashboard.py` (1 step)

**C1. Entry point** (S1-S4; spec 2423-2429). Replace the whole file by:
- Docstring: "Streamlit entry point of the dashboards (spec, section "Dashboards")."; the start command
  `python -m streamlit run src/fleet_management/validator_dashboard.py`; the sidebar; "A failed load shows the
  error and no page."
- `import streamlit as st`; `from fleet_management.validation.dashboard_gamma import load_gamma_diagnostic,
  render_gamma_diagnostic`; `from fleet_management.validation.dashboard_inspector import load_inspector,
  render_result_inspector`.
- `MODES = {"Result inspector": (load_inspector, render_result_inspector), "Gamma diagnostic":
  (load_gamma_diagnostic, render_gamma_diagnostic)}`.
- `st.set_page_config(page_title="Fleet Management dashboards", layout="wide")`; in `with st.sidebar:`
  `mode = st.selectbox("Mode", list(MODES))`, `input_path = st.text_input("Input file", value="input.yaml")`,
  `results_path = st.text_input("Result file", value="output.yaml")`, `tol = st.number_input("Tolerance",
  min_value=1e-12, value=1e-6, format="%.1e")` (positive: the Gamma validators need it), `load =
  st.button("Load", type="primary")`.
- If `load`: `try: st.session_state["loaded"] = (mode, MODES[mode][0](input_path, results_path, tol))`;
  `except Exception as exc:` remove `"loaded"` from the session state, `st.error(f"Could not load the files:
  {exc}")`, `st.stop()`.
- `loaded = st.session_state.get("loaded")`; if it is `None` or `loaded[0] != mode`: `st.info("Choose the files in
  the sidebar and press Load.")` and `st.stop()`; else `MODES[mode][1](loaded[1])`.
Check: `python -c "import ast; ast.parse(open(...).read())"` passes; the names check reports nothing; the file
has at most 60 lines.

## 3. Tests

No test needs Gurobi or Streamlit. Expected values come from `S\fixture_values.py` and `S\reference_checks.py`
(outputs `S\fixture_values_out.txt`, `S\reference_checks_out.txt`); compare floats with `pytest.approx(...,
rel=1e-12)` unless stated. Each file starts with a docstring that names the spec section and the fixture.

**`tests/test_validator.py`** (Task A). Helpers:
- `FIXTURES = Path(__file__).resolve().parents[1] / "results" / "validator_tests"`; `INPUT`, `RESULT` the two files.
- `CHECK_NAMES = ["binary", "one_activity", "mission_coverage", "depot_gating", "idle", "loop_mu", "loop_v",
  "loop_R", "loop_K", "loop_g_mu", "loop_g_v", "loop_g_R", "loop_g_K", "objective"]`.
- `_input(tmp_path, **changes)`: `read_input(INPUT)`, update, `yaml.safe_dump` to `tmp_path / "input.yaml"`.
- `_run(tmp_path, result, input_path=INPUT, **kwargs)`: `solver._save_results(result, tmp_path /
  "result.yaml")`, then `validate(input_path, that path, tmp_path / "validation.yaml", **kwargs)`.
- `_failed(report)`: `{c["name"]: c["violation"] for c in report["checks"] if not c["passed"]}`.
- A corruption starts from `read_result(RESULT)` and sets entries (indices 0-based: `[i, j or l, k]`).

| # | Test | Change | Expected |
|---|---|---|---|
| 1 | `test_valid_result_passes` | none | `passed`; names `== CHECK_NAMES`; `status == "optimal"`; both objectives `approx(533.882)`; `tolerance == 1e-6`; `max_violation <= 1e-12`; `yaml.safe_load` of the file `== report` |
| 2 | `test_every_result_format_gives_the_same_report` | parametrize suffix `.yaml .yml .json .h5 .hdf5` | the same names, `passed` and violations as test 1 |
| 3 | `test_read_result` | the three formats; a result written without `idle` | equal arrays (`np.testing.assert_array_equal`); `R` all NaN; `H == [2, 3]`; `idle` is `None` |
| 4 | `test_report_path_rules` | `validation_path` `tmp_path/"new"/"report"`; then `tmp_path/"report.json"` | `new/report.yaml` exists (folder created); `json.load` of the JSON file `== report` |
| 5 | `test_recomputed_objective_by_mode` | parametrize `objective_mode` `total`, `operating_average`, `evaluation_total`, and `total` with `C_D=0.0` | `recomputed_objective` 58.882, 19.84, 533.882, 58.042; `passed` (the saved objective is not below) |
| 6 | `test_objective_below_the_recomputed_value_fails` | `objective -= 2.0` | `_failed == {"objective": approx(0.003760232532779827)}` |
| 7 | `test_objective_above_the_recomputed_value_passes` | `objective += 5.0` | `passed` (slack of `u`, spec 1763) |
| 8 | `test_status_is_echoed_not_checked` | `status = "time_limit"` | `passed`; `report["status"] == "time_limit"` |
| 9 | `test_file_errors` | missing input; missing result; existing `result.txt`; `validation_path` `report.txt`; existing `input.csv` | `FileNotFoundError` "Input file not found", "Result file not found"; `ValueError` "Unsupported result file type", "Unsupported validation file type", "Unsupported input file type" |
| 10 | `test_violation_is_detected` | parametrize, below | `_failed == {name: approx(value)}` |
| 11 | `test_loop_row_violation_is_detected` | parametrize `(q, i, l)`: `mu (0,0)`, `v (1,0)`, `R (0,0)`, `K (2,1)`, `g_mu (1,1)`, `g_v (2,0)`, `g_R (0,0)`, `g_K (2,1)`; if `a[i,l,1]` is NaN set it to 0.1; set `a[i,l,4] = a[i,l,1] + 0.5` | `_failed == {f"loop_{q}": approx(0.5)}` |
| 12 | `test_nan_entries_are_skipped` | none | checks `loop_R`, `loop_g_R` have violation `0.0` (all NaN) |
| 13 | `test_replacement_not_allowed` | input `allow_replacement=False` | `_failed == {"depot_gating": 1.0}`; `recomputed_objective == approx(29.882)` |
| 14 | `test_result_without_solution` | `x = None` | `ValueError` "no solution: 'x' is missing or null" |
| 15 | `test_sizes_must_match` | parametrize `("F", 4)`, `("H1", 3)` | `ValueError` "'F' differs between the result", "'H1' differs between the result" |
| 16 | `test_shapes_must_match` | `mu = mu[:, :, :4]` | `ValueError` match `r"result array 'mu' has shape \(3, 2, 4\); the input gives \(3, 2, 5\)"` |
| 17 | `test_malformed_result_or_input` | `x[0,0,0] = nan`; `u = None`; input `tau=-1.0` | `ValueError` "result array 'x' has a non-finite entry"; "the result has 'x' but no 'u'"; a `ValueError` whose message names `tau` |

Cases of test 10 (each fails exactly one check, checked with `S\reference_checks.py`):

| Name | Change | Violation |
|---|---|---|
| `binary` | `x[2,0,0] = 0.3`; `idle[2,:,0] = 0.3`; `objective += 0.3` | 0.3 |
| `one_activity` | `x[0,1,2] = 1`; `x[1,1,2] = 0` | 1.0 |
| `mission_coverage` | `x[1,1,2] = 0` | 1.0 |
| `depot_gating` | `m[0,0,0] = 1`; `idle[0,0,0] = -1` | 1.0 |
| `idle` | `idle[2,0,1] = 0` | 1.0 |

**`tests/test_validator_dashboards.py`** (Task B). Helpers: the same `FIXTURES`; `cfg, result =
load_pair(INPUT, RESULT)` inside each test.

| # | Test | Expected |
|---|---|---|
| 18 | `test_modules_import_without_streamlit` | in `ast.parse` of `dashboard_inspector.py` and `dashboard_gamma.py`, no module-level `import streamlit`; both modules import |
| 19 | `test_load_pair` | `cfg.T == 5`; `FileNotFoundError` "Result file not found"; a result with `model[0][0] = "wiener"` gives `ValueError` "'model' differs" |
| 20 | `test_cell_label` | `(2, 0)` "rainflow (bernstein)"; `(2, 1)` "rainflow (chernoff)"; `(0, 1)` "inverse_gaussian" |
| 21 | `test_reliability_caps` | `(0, 1)` 0.49234697216756246; `(1, 1)` 0.6726905060677382 (`rel=1e-9`); NaN at the other 4 cells |
| 22 | `test_reliability_margins` | equal to the matrix below, `abs=1e-9` |
| 23 | `test_margin_violation` | with `mu[0,0,2] = 0.99`: margin `approx(-8.221738163816342e-06, rel=1e-6)`; `margin_table(..., tol=1e-6)` has one row with `holds` false: train 1, "Wheelset", day 3 |
| 24 | `test_margin_table_and_overview` | 30 rows; columns as B2; 0 violations; `largest_mu_over_tau == approx(0.33)`; first row of `smallest_margins`: train 1, "Brake pad", day 2, margin `approx(0.282346972, abs=1e-9)`; `models == ["gamma", "gaussian", "inverse_gaussian", "rainflow (bernstein)", "rainflow (chernoff)", "wiener"]`; `status == "optimal"` |
| 25 | `test_loop_table` | 18 rows (mu 6, v 3, K 1, g_mu 5, g_v 2, g_K 1), all hold; the row train 2, "Wheelset", `mu`: 0.27, 0.08, margin `approx(0.19)` |
| 26 | `test_day_marks` | `(2,0)` I D M2 M2 P; `(2,1)` I R M2 M2 P; `(0,0)` M1 M2 P M1 M2; `(1,1)` M2 M1 M1 P M1 |
| 27 | `test_figures` | the three inspector figures are `Figure`; the timeline axes has at least 4 lines; the heatmap axes has 1 image |
| 28 | `test_load_inspector` | the 8 keys of B2; `report["passed"]`; `overview["reliability_violations"] == 0` |
| 29 | `test_gamma_tables` | `gamma_cells(cfg) == [(1, 1)]`; `ValueError` "no Gamma cell" for `SimpleNamespace(model=np.array([["gaussian"]], dtype=object))`; `gamma_mean_table`: 5 rows, day 3 margin `approx(0.4426905060677382)`; `gamma_tail_table` of a hand-made report with steps `(i=1, l=1, k=0, bound 0.001)`, `(1, 1, 1, 0.06)` and `tol=1e-6`: days 1, 2, slack 0.049, -0.01, holds true, false, component "Brake pad"; `gamma_overview` of a hand-made replay report |
| 30 | `test_load_gamma_diagnostic` | `overview["valid"]`; `repairs == 0`; `replacements == 1`; `tails` 5 rows, all hold; `means` 5 rows |

Test 30 runs the Gamma validators of plan `gamma_validators`; it can pass only after that plan is carried out.
The executor of Task B runs it, reports its outcome, and does not skip it.

Margins of test 22 (rows: train 1 Wheelset, train 1 Brake pad, train 2 Wheelset, ...; columns days 1-5):

```
[[0.654747228, 0.575976674, 0.960291778, 0.846021224, 0.756250670],
 [0.352346972, 0.282346972, 0.472346972, 0.432346972, 0.362346972],
 [0.620312239, 0.526677250, 0.440242261, 0.960291778, 0.843856789],
 [0.522690506, 0.482690506, 0.442690506, 0.652690506, 0.612690506],
 [0.756412851, 0.756412851, 0.671289976, 0.591167100, 0.884998013],
 [1.997414907, 2.137414907, 1.757414907, 1.377414907, 2.597414907]]
```

## 4. Exports

- For `fleet_management/__init__.py` (owned by `plots`): `validate`, from `fleet_management.validation.validator`
  (spec 230). The import line of `__init__.py:8` stays as it is; the signature becomes `validate(input_path,
  results_path, validation_path="validation.yaml", tol=1e-6)`.
- Internal, not exported: `read_result`, `check_sizes`, `check_result`, `recomputed_objective` (validator); every
  function of the two dashboard modules. `read_result` is meant for the other tools too (Question Q6).
- `import fleet_management` stays possible without Gurobi and without Streamlit: `validator.py` imports `config`
  and `solver` only, and `__init__.py` imports no dashboard module.

## 5. Callers outside src

- `examples/`, `experiments/`, `euler/`, `remote_smoke_test.py`: no script imports `validate`, a dashboard module,
  `validator_dashboard.py` or a function of `validator.py` (grep of the worktree, 29 Oct 2026). Nothing breaks.
- `README.md:35` (extra `dashboard`) and `README.md:212-232` (the old `validate(input, degradation, results)`, the
  baseline dashboard and `validate_baseline_assignment_feasibility`) describe removed code: WP8, README.

## 6. Removals

Claude removes these files at the commit (executors do not delete files):
- `src/fleet_management/validation/dashboard_rainflow.py`: replaced by `dashboard_inspector.py` (spec 202).
- `tests/test_validator_negative_cases.py`, `tests/test_validator_positive_case.py`: replaced by
  `tests/test_validator.py`.
- `results/validator_tests/` old fixtures (10 files): `bad_assignment.yaml`, `bad_capacity.yaml`, `bad_demand.yaml`,
  `bad_dimension.yaml`, `bad_mu_periodic.yaml`, `bad_objective.yaml`, `bad_status.yaml`, `bad_u_ge_mu.yaml`,
  `bad_v_periodic.yaml`, `bad_x_binary.yaml`. They use the old output format, and the tests now build each
  corrupted result from `result.yaml`.

Code that leaves inside the files of this plan (the executors replace it): `validator.py:248-1080`
(`validate_baseline_assignment_feasibility`, `build_assignment_feasibility_dataframe`,
`build_gamma_diagnostic_dataframe`, `validate_gamma_synthetic_diagnostic`) and `_extract_results_parameters`,
`_as_2d_array`, `_check_dimensions`, `_validate_gaussian_objective`, `_is_binary`; `validator_dashboard.py:59-1795`
(the baseline page); all of `dashboard_gamma.py` (the synthetic diagnostic).

After this plan, `validator.py` no longer imports `gamma_utils/gamma_process.py`, and no module of `validation/`
calls `solver._read_input`; whether these two leave is decided by `gamma_validators` and WP8.

## 7. Questions

Each step follows the recommendation; a different answer changes only the steps named.

- **Q1. Margin of a row with a side row** (spec 2447: "the right side minus the left side of the exact form of
  its reliability row"). The G, W, Ca, Ho and Be rows are two relations, the quadratic row and the side row
  (Eqs. `g_row`, `rf_cantelli`, `rf_hoeffding`, `rf_bernstein`). The quadratic difference alone can be positive
  when the side row fails: G with `mu = 1.2`, `tau = 1`, `v = 0` gives `(tau - mu)^2 = 0.04 > 0`. Options: (a) the
  smaller of the two differences (recommended: the row holds if and only if both relations hold); (b) the
  quadratic difference only. The two differences have different units, so only their sign is compared across
  cells. Step B1.
- **Q2. `model` of the result against the input.** Spec 2364: each tool "takes the model of each cell from the
  result key `model`", but the margins need the input's parameters of that model (`eta`, `gamma_beta`, ...).
  Options: (a) the dashboards raise `ValueError` when the two differ, `validate` does not check it (recommended:
  `validate` needs no model, and its error list, spec 2389, has no such case); (b) both raise; (c) the result's
  model is used without a check. Step B1.
- **Q3. Errors that the spec does not list** (spec 2389). A result with `x` but without `m`, `r`, `mu`, `z`, `u`
  or `objective`, or with a non-finite entry in `x`, `m`, `r`, `idle`, cannot come from `solve`. Without an error
  the report would hold NaN, and the JSON writer refuses NaN. Recommendation: `ValueError` (A2, items 4-5), and
  one clause in the spec's Errors row. Step A2.
- **Q4. Loop checks of keys without numbers.** Spec 2404 skips NaN and null entries. Recommendation: the eight
  `loop_q` checks are always in the report, with violation 0 when no entry is a number, so the report has a fixed
  list (14 checks, 13 without `idle`). Option: only the keys that hold a number. Step A3.
- **Q5. Contract with plan `gamma_validators`.** The Gamma diagnostic needs: (a)
  `validate_gamma_replay_schedule(cfg, result, *, tolerance)` and `validate_gamma_tail_bound_schedule(cfg,
  result, *, tolerance, include_steps=True)` accept the dictionary of `read_result` (float arrays, NaN for null)
  and need none of the removed keys (`gamma_shape_bound`, `gamma_calibration`, `backend`, `gamma_beta_bound`);
  (b) each record of `steps` keeps `i`, `l`, `k` (0-based) and `exact_tail_upper_bound` (today's names,
  `gamma_tail_validator.py:394-423`); (c) the replay report has the keys of spec 2516-2523. Claude aligns the
  two plans. Steps B5, test 30.
- **Q6. One reader of result files.** The interface asks to read a result "with the writer's counterpart";
  `solver.py` has writers only, and `config.read_input` reads the three formats but returns lists. This plan adds
  `validator.read_result` (on `read_input`; arrays to float64, null to NaN). The plotter (`plots`) and the file
  versions of the Gamma validators (`gamma_validators`) need the same. Options: (a) `read_result` stays in
  `validator.py` and the other tools import it; (b) it moves to `solver.py`, next to the writers (recommended,
  the natural home; nobody edits `solver.py` in WP7, so Claude moves it at the commit and changes one import line
  in each caller). Step A2 places it in `validator.py` in both cases.
- **Q7. Fixture files.** The interface's file table does not list `results/validator_tests/`, but spec 179 puts
  the fixtures of the validator tests there, and section 6 removes the old ones. Options: (a) Task A creates
  `results/validator_tests/input.yaml` and `result.yaml` (recommended: the data are visible and reproducible from
  `S\fixture_values.py`); (b) the tests build the result in `tmp_path` with about 80 lines of day maps. Step A1;
  with (b), A1 becomes a helper in `tests/test_validator.py` and test 1 compares it with the script's values.
- **Q8. Scope of `validate`.** The spec (2398-2406) and the decision "Validation tools" check the loop rows of
  every tracked state and anchor, not the dynamics rows (Eqs. `a_bigm_carry` ...) nor the reliability rows. This
  plan follows the spec; the reliability rows appear as the inspector's margins. Checking the dynamics rows too
  would need a spec change. No step depends on it.

## 8. Review (29 Oct 2026)

Read in full: the diffs of `validator.py`, `validator_dashboard.py`, `solver.read_result`, the new
`dashboard_inspector.py`, `dashboard_gamma.py`, both test files; the fixtures (byte-identical to `S\input.yaml`,
`S\result.yaml`); the three progress files. `git status` of the worktree: no file deleted (the refused
`rm -rf /dev/null` of Task A removed nothing).

**Defects corrected**
- `validator.py:1-3`: the old author and date header above a module that holds none of the old code; removed (no
  other new module has one; git keeps the history).
- `validator.py:29, 97`: docstrings cited the section "Validator"; now "Structural Validator".
- `dashboard_inspector.py` (6 calls), `dashboard_gamma.py` (3 calls): `st.dataframe(..., width="stretch")`.
  Streamlit's current page for `st.dataframe` gives `"stretch"` as the default of `width`; older releases refuse a
  string. The argument is dropped, so the tables work on both.
- `validator_dashboard.py:36`: the tolerance input had Streamlit's default step 0.01; now `step=1e-6`.
- Tests that could not fail for a wrong formula (mutants that survived): the binary check as distance to the
  nearest integer; the Markov, Cantelli and Hoeffding margins (no such cell in the fixture); `z = NaN` not counted
  as 0. New tests: `test_binary_is_the_distance_to_0_and_1`, `test_null_z_counts_as_zero_and_idle_is_optional`
  (also checks the 13 names without `idle`), `test_margins_of_the_bounds_outside_the_fixture`. Six review
  mutants (`S\review_mutants.py`) are now all killed.

**Decisions on the executors' points** (accepted unless named above)
- A: no `ARRAY_KEYS` (decision 1 moved the reader); test 2 compares whole check records; extra case in test 12.
- B1 `gamma_overview["violations"]` counts the records of the replay report; the global violations stay a list.
- B2 extra assertions; B3 marks coloured and shifted per component (legibility); B4 `layout="constrained"`
  (matplotlib 3.5 or later); B7 `smallest_rate(..., cfg.allow_replacement)` as the builder and Eq. `gm_betastar`
  (config passes `True` only because its check runs with replacement allowed); B8 LF endings (`autocrlf`, no file
  mixed); B9 the two `gamma_cells` live in different modules.
- C: Streamlit inside `main()` under `if __name__ == "__main__"` is kept. `streamlit run` executes the script in a
  module named `__main__` (Streamlit's `script_runner.py`, read in part: `self._new_module("__main__")`,
  `exec(code, module.__dict__)`). So the page draws, and the module loads without Streamlit.
- Open for Claude: the two `render_*` functions and `main()` ran only against the recording stand-in; one run with
  a real Streamlit is still needed.

**Spec sentences** (LaTeX for `spec/spec.tex`)
1. "Result Files", new last bullet:
   `\item \textbf{Reader}: \texttt{read\_result(path) -> dict} in \texttt{solver.py} reads a result file of any format of the table above with the input reader (Section~\ref{sec:schema}); another extension raises a \texttt{ValueError}. The dictionary has exactly the keys of the result table (Section~\ref{sec:output}): a key missing from the file is \texttt{None}, a key outside the table is dropped. Each key stored as a \texttt{float64} dataset in HDF5 (Section~\ref{sec:schema_hdf5}) becomes a \texttt{numpy} array of floats, with \texttt{NaN} at a \texttt{null} entry. Every tool that reads a result file uses it.`
2. "Structural Validator", Errors, end of the `ValueError` row (validator Q3):
   `; the result has \texttt{x} but lacks one of \texttt{m}, \texttt{r}, \texttt{mu}, \texttt{z}, \texttt{u}, \texttt{objective}, or has a non-finite entry in \texttt{x}, \texttt{m}, \texttt{r} or \texttt{idle}`
3. "Structural Validator", after "The recomputed objective is ...":
   `The report lists the checks in the order of the table. The eight checks \texttt{loop\_}$q$ are always listed; a check whose key is absent, or has no pair of numbers at days $H_1$ and $T$, has violation $0$. The internal function \texttt{check\_result(cfg, result, tol=1e-6) -> dict} runs the checks, and the errors on sizes, shapes and arrays, on a \texttt{FleetConfig} and a result dictionary, and returns the report without writing it; \texttt{validate} and the dashboards call it.`
4. "Dashboards", the bullet "Loading" becomes three bullets (decision 4):
   `\item \textbf{Loading.} The dashboards read only these two files: the input with the input reader and \texttt{load\_config}, the result with \texttt{read\_result} (Section~\ref{sec:output_files}). The errors of \texttt{validate} on sizes, shapes and arrays apply. A result whose \texttt{model} differs from the input at some cell raises a \texttt{ValueError}. A failed load shows the error and no page.`
   `\item \textbf{Tolerance.} At least $10^{-12}$, step $10^{-6}$: the Gamma validators need a positive tolerance.`
   `\item \textbf{Code.} \texttt{main()} of \texttt{validator\_dashboard.py} builds the page; it runs under \texttt{if \_\_name\_\_ == "\_\_main\_\_"}, as \texttt{streamlit run} executes the file as \texttt{\_\_main\_\_}. Each mode has a loading function and a page function: \texttt{load\_inspector}, \texttt{render\_result\_inspector} (\texttt{validation/dashboard\_inspector.py}); \texttt{load\_gamma\_diagnostic}, \texttt{render\_gamma\_diagnostic} (\texttt{validation/dashboard\_gamma.py}). Streamlit is imported only inside \texttt{main} and the page functions, so every module loads without it.`
5. "Dashboards", the bullet "Margin" (validator Q1) becomes:
   `\item \textbf{Margin} of a cell on a day: the slack of the exact form of its reliability row on the saved tracked states, by the table below; at a row with a side row, the smaller of the two slacks. The row holds if the margin is at least $-\texttt{tolerance}$; \texttt{NaN} entries are left out. $\beta^*$ follows Eq.~\eqref{eq:gm_betastar}.`
   followed by `\tabstart{p{2.4cm}p{8.4cm}p{3.4cm}}{Cell & Margin & Row}` and the rows
   `\typerow{G, W}{$\min\{(\tau-\mu)^2-\kappa_G^2 v,\ \tau-\mu\}$}{Eq.~\eqref{eq:g_row}}`,
   `\typerow{IG}{$\bar\mu-\mu$}{Eq.~\eqref{eq:ig_row}}`,
   `\typerow{Ga}{$A_1(\beta^*\tau,\varepsilon)/\beta^*-\mu$}{Eq.~\eqref{eq:gm_row}}`,
   `\typerow{RF, Mk}{$\varepsilon\tau-\mu$}{Eq.~\eqref{eq:rf_markov}}`,
   `\typerow{RF, Ca}{$\min\{(\tau-\mu)^2-\kappa_C^2 v,\ \tau-\mu\}$}{Eq.~\eqref{eq:rf_cantelli}}`,
   `\typerow{RF, Ho}{$\min\{(\tau-\mu)^2-\tfrac{\kappa}{2}R,\ \tau-\mu\}$}{Eq.~\eqref{eq:rf_hoeffding}}`,
   `\typerow{RF, Be}{$\min\{(\tau-c-\mu)^2-c^2-2\kappa v,\ \tau-c-\mu\}$}{Eq.~\eqref{eq:rf_bernstein}}`,
   `\typerow{RF, Ch}{$\ln\varepsilon-(K-s\tau)$}{Eq.~\eqref{eq:rf_chernoff}}`, then `\tabend`.
6. "Dashboards", after the margin table:
   `Trains and days are numbered from $1$; components are shown by name. The Overview lists the ten cells and days with the smallest margins. Repeatability shows, for each cell and each saved quantity of \texttt{loop\_}$q$ with numbers at days $H_1$ and $T$, both values and the margin $q_{i\ell H_1}-q_{i\ell T}$, which holds if it is at least $-\texttt{tolerance}$. The Train timeline has the column ``initial'' ($\mu_{0,i\ell}$), a dashed line at $\tau_{i\ell}$ for each component, a dotted line after day $H_1$ and the mark of each day above its point. The Heatmap has one row per cell and a red-to-green scale symmetric about $0$. The tab Validation shows the report of \texttt{check\_result} with the sidebar tolerance.`
7. "Mode Gamma diagnostic", after the tab list:
   `The validators run with the sidebar tolerance; an input without a Gamma cell raises a \texttt{ValueError}. The overview shows \texttt{repairs}, \texttt{replacements}, \texttt{valid}, the number of records in \texttt{violations}, the list \texttt{global\_violations} and \texttt{maximum\_errors} of the replay report. The cap is that of the margin table. The tail tab takes \texttt{exact\_tail\_upper\_bound} from each record of \texttt{steps} (\texttt{include\_steps=True}); a day holds if $\varepsilon_{i\ell}$ minus that bound is at least $-\texttt{tolerance}$, and the heatmap marks a day that does not hold with ``!''.`

**Tests** (`S\run_tests_validator.py`, 29 Oct 2026, after the corrections): `tests/test_validator.py` 38 runs
passed (19 test functions, 5 of them parametrized); `tests/test_validator_dashboards.py` 14 passed (test 30 against the
finished Gamma validators). The stand-in page runs (`S\taskC_run.py`, `S\check_B4.py`, `S\check_B5.py`) pass;
undefined-name check 0; no line over 99 characters; `import fleet_management.validator_dashboard` loads neither
Streamlit nor gurobipy.
