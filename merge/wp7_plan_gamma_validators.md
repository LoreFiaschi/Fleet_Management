# WP7 plan, part `gamma_validators`: the validators of the Gamma model

Status (29 Oct 2026): done. Plan, Task 1, Task 2 and the review (section 8) are complete. Next: Claude adds the spec
sentences of section 8, reviews the diff and commits.

Scratch folder `<sp>` = `C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad`.
Scripts of this plan, in `<sp>\wp7\gamma_validators\`: `expected_values.py` (exact tails by scipy and by the
independent quadrature `examples/regression/gamma_quadrature_reference.py`; output `expected_values_output.txt`),
`moschopoulos_fast.py` (prototype of step C4; output `moschopoulos_fast_output.txt`), `import_check.py` (imports the
four modules with `gurobipy` blocked; prints their public names and `ok`), `fixture_check.py` (every test input of
section 3 loads in `load_config` without a warning; hand results round-trip through the YAML, JSON and HDF5 writers),
`plan_prototype.py` (the loops of C2 and C3 as written below, on cases A, B, C: every number of 3.2 and 3.3
reproduced; output `plan_prototype_output.txt`).

## 1. Gaps

Code: the worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge` at 0421e80. Prefixes: `rep`, `tail`,
`mc`, `tb` = `gamma_utils/gamma_replay_validator.py`, `gamma_tail_validator.py`, `gamma_stochastic_validator.py`,
`gamma_tail_bound.py`. "s" = line of the worktree's `spec/spec.tex` (its validator sections equal the `lore` copy,
checked by `diff`). Reports use 0-based array indices `i`, `l`, `k` (day `k+1` of the spec), as `config.py` does.

**Common to the three validators**
- C1 rep:65-66, tail:166-171, mc:166-173 / s 2364: the Gamma cells come from `cfg.model`; the spec takes the model of
  each cell from the result key `model`.
- C2 rep:214-223, tail:520-530, mc:469-470 / s 2362-2363, 2664: private YAML/JSON readers; HDF5 results fail. The
  spec: the input reader `config.read_input`, and every result format.
- C3 rep:233-242, tail:551-560, mc:450-459 / s 2468, 2474: the suffix of `report_path` is checked after the
  validation (a long Monte Carlo run is lost); `tail` does not create the folder; three copies of one writer.
- C4 rep:25-26, tail:78-79 / s 1722, 1724, 2509: every array must be finite, but a merged result has `NaN` in `z`
  (cells with `C_R = 0`) and in `g_mu` (cells that are not `ard1`). Finiteness applies at the Gamma cells only.
- C5 rep:43-49, tail:95-101 / s 431-438: two private copies of the phase lookup; `config._lookup` gives the profile
  on the days `1..T` (`config.py` uses it for the Gamma cap).

**Replay validator (`rep`)**
- R1 rep:75-90 / s 2480, 2491-2498: reads `gamma_shape_bound`, `gamma_maximum_shape`, `gamma_shape_increment(_trans)`,
  `gamma_mean_latch`, `gamma_shape_latch`, `gamma_calibration`, `backend`; none exists after WP6
  (`base.py:1135-1147`). The spec reads `x`, `m`, `r` (optional), `mu`, `z`, and `g_mu` at `ard1` cells.
- R2 rep:134-139 / s 2496-2497: "repair and replacement" and "intervention with a mission or without a depot day"
  are checked at Gamma cells only, inside the replay; the spec lists them as global violations.
- R3 rep:127 / s 2499, 764-767: the reference anchor starts at 0; the spec starts it at `anchor_mu_0`.
- R4 rep:98-110 / s 2505-2512: fields `physical_mean`, `bounding_shape`, `mean_latch`, `shape_latch`, `reliability`,
  two-sided (`abs`). The spec: `tracked_mean`, `mean_anchor`, `removed_mean`, `cap`, `repeatability`, one-sided
  (`max{0, ...}`), because the saved states are upper bounds.
- R5 rep:156 / s 2509: `removed_mean` compares `z` with the reference drop; the spec: `x^m rho (mu_{k-1} - g_{k-1})
  - z`, `g = 0` under `ardinf`, skipped where `z` is `NaN`.
- R6 rep:176-191 / s 2510-2511, 1281-1284: the cap compares saved shapes; the spec: `mu <= A_1(beta* tau,
  eps)/beta*` from `cfg` (`gamma.smallest_rate`, `gamma.mean_cap`). Repeatability uses the shape, not `g_mu`.
- R7 rep:179-180, 189-191 / s 2500, 2502: two record kinds lack `expected` and `saved`; the event `"repeatability"`
  is not one of the four events.

**Exact validator (`tail`)**
- E1 tail:187-223 / s 2525-2579: reads `gamma_shape_bound`, `gamma_tail_bound`, `gamma_beta_bound`, `backend` and
  `cfg.gamma_beta_bound`; `FleetConfig` has no such field (config.py:142-216), so line 214 raises `AttributeError`.
- E2 tail:183-185: `r` is required; the replay validator treats it as optional.
- E3 tail:104-110, 250-253 / s 2553: a positive mean without a rate raises `ValueError`; the spec: a known constant
  `c`, tail `Pr(terms > tau - c)`, `1` if `c > tau`. The rates are arrays with `NaN` where not given
  (config.py:187-188).
- E4 tail:254-260 / s 736-738, 764-767, 2605: under `ard1` the whole initial damage is repairable (anchor 0); with
  the spec's starting anchor, it is a frozen part of mean `anchor_mu_0` and an active part of mean
  `mu_0 - anchor_mu_0` (the split that s 2605 states for the Monte Carlo validator).
- E5 tail:290-312 / s 2550, 1225: terms are removed when `1 - rho <= tolerance` (spec: `rho = 1`; `config` accepts
  `rho` up to `1 + 1e-12`, config.py:575); a day with several missions keeps only the first.
- E6 tail:352-391 / s 2558-2564: mean and anchor compared two-sided; the reliability check uses the tail of the saved
  bounding shape. The spec: upper bound `<= eps + tolerance`; saved mean `>= E D_k - tolerance`; saved anchor `>=`
  frozen mean `- tolerance`.
- E7 tail:209-223, 269-272, 427-428, 458-507 / s 2571-2579: records are step dictionaries with `reasons`; keys
  `backend`, `minimum_conservativeness_margin`, `worst_conservativeness`, `cell_histories`, `numerics`; no
  `validator`, `maximum_errors`, top-level `maximum_remaining_mass`; the schedule checks are missing. The spec: every
  key of the replay report, plus the slack, the worst step, the remaining mass, series counts, tail timing, `steps`.

**Monte Carlo validator and dispatcher (`mc`)**
- S1 mc:233-239 / s 2605: under `ard1` the initial damage is one draw and the anchor starts at 0; the spec draws two
  independent terms of means `g_0` and `mu_0 - g_0`.
- S2 mc:204-213 / s 2605: a missing rate falls back to rate 1 only if the whole array is `None`; now `NaN` reaches
  `rng.gamma`. The spec: the constant of its mean.
- S3 mc:190-193, 250-255 / s 2472-2476: `ValueError` for schedule violations, not in the spec's error table.
- S4 mc:336-377 / s 2619-2628: extra keys `interpretation`, `failure_event`, `solver_status`, `dimensions`,
  `gamma_cells`; `maximum_schedule_failure_rate` and `tolerance` missing among the arguments used.
- S5 mc:81-95 / s 486, 2650, 2656, `lore` s 360: `_settings` reads `cfg.raw` (gone: `AttributeError`; the block is
  `cfg.options["validation"]`); `mode` is normalised by `strip().lower()`, but selector values are case sensitive.
- S6 mc:96-119: integer settings pass through `int()`, so `2.5` becomes `2` silently (`config._integer` refuses it).
- S7 mc:428-440 / s 2475: in mode `"both"` the first failing validator raises before the combined report exists.

**Exact series (`tb`)**
- T1 tb:1-47, 76-927, 1010-1150, 1171-1224 / s 198: the removed `finite_count` calibrators, their result classes and
  helpers are still there (`merge/wp1_removals.md` assigned them to WP6, which did not remove them). Kept:
  `ConvolutionTail` (66-74), `moschopoulos_tail_probability` (928-1007), `_validate_parameters` (1151-1168).

**Items of other packages (not planned here)**: `validation/validator.py:14` imports `gamma_process` (section 6);
`greedy_warm_start.py:146-150, 164` uses names that step C4 removes (`sweeps` plan; Q5); `dashboard_gamma.py` must
call the replay and exact validators (s 2451-2458; `validator` plan); `GAMMA_DIAGNOSTICS.md` (Q8).

## 2. Executor tasks

Two tasks with disjoint files. **Task 1 (tests) runs first**; Task 2 (code) uses its tests as checks. Both: do not
spawn sub-agents; edit only the files of the task; delete no file; write intermediate files only to
`<sp>\wp7\gamma_validators\`; never put the user's name or email address in any request. Style: as `gamma.py` and
`base.py` after WP6 (docstrings that cite the spec section, short comments with the equation label). Test runner
(no pytest here): `python <sp>\wp6\gamma\run_tests_gamma.py <worktree>\tests\test_gamma_validators.py [name part]`;
it supports `pytest.raises`, `pytest.mark.parametrize`, `pytest.importorskip` and `tmp_path`, not fixtures and not
`pytest.approx` (use `math.isclose`).

### Task 1: `tests/test_gamma_validators.py` (new)

1. Write the module of section 3, in its order. Check: the runner collects every test; before Task 2, the series
   tests (3.6) pass except `test_tail_bound_keeps_only_the_series`.
2. Report the test names and the runner's last line.

### Task 2: the four modules of `degradation_model/gamma_utils/`

After every step, `python <sp>\wp7\gamma_validators\import_check.py` prints `ok`, and
`python <sp>\wp6\rainflow\names_check_try.py <file>` reports 0 undefined names.

**C1. `gamma_replay_validator.py`: shared helpers** (replace lines 1-49 and 214-223; s 2362-2364, 2464-2476).
- Docstring: "Replay validator of the Gamma model and the helpers that the Gamma validators share (spec, section
  "Validators of the Gamma Model"). Indices `i`, `l`, `j`, `k` of a report are 0-based array indices; day `k` is day
  `k + 1` of the spec." Imports: `json`, `math`, `time`, `Path`, `Any`, `numpy as np`, `yaml`;
  `from fleet_management.config import FleetConfig, _lookup, load_config, read_input`;
  `from fleet_management.solver import _to_plain`. No `base`, `gamma` or `gurobipy` at module level.
- `REPORT_SUFFIXES = (".yaml", ".yml", ".json")`. `_check_tolerance(tolerance)`: `not tolerance > 0` →
  `ValueError("tolerance must be positive.")`.
- `gamma_cells(cfg, result) -> list[tuple[int, int]]`: `result.get("model")` is `None` → `ValueError("result is
  missing 'model'.")`; `names = np.asarray(...).astype(str)`; shape `!= (F, L)` → `ValueError(f"result 'model' has
  shape {names.shape}; expected {(F, L)}.")`; first cell that differs from `np.asarray(cfg.model).astype(str)` →
  `ValueError(f"result 'model' is {names[i, l]!r} at cell (i={i}, l={l}); the input says {expected[i, l]!r}.")`
  (Q1); the `"gamma"` cells, `i` outer, `l` inner; none → `ValueError("the configuration has no Gamma cell.")`.
- `_array(result, name, shape, cells=None)`: missing or `None` → `ValueError(f"result is missing {name!r}.")`;
  `np.asarray(result[name], dtype=float)` (a `None` entry becomes `NaN`); shape check as rep:23-24; finite check on
  every entry if `cells is None`, else on `np.array([value[i, l] for i, l in cells])` only → `ValueError(f"result
  {name!r} has a non-finite entry.")`.
- `_binary(result, name, shape, tolerance, optional=False)`: body of rep:30-40; message `f"result {name!r} is not
  binary within tolerance; maximum error={error:.3e}."`.
- `_increments(cfg)`: `(_lookup(cfg.mu, cfg.mu_trans, H1, H2), _lookup(cfg.gamma_beta, cfg.gamma_beta_trans, H1,
  H2))`, both `(F, L, M, T)` (Eq. c_phase); mission `j` of `x` is index `j - 1` on axis 2.
- `gamma_cell_caps(cfg, cells) -> {(i, l): cap}`: `mu_inc, beta_inc = _increments(cfg)`; import `mean_cap,
  smallest_rate` from `fleet_management.degradation_model.gamma` inside the function; `beta_star = smallest_rate(mu_inc[i, l],
  beta_inc[i, l], float(cfg.mu_0[i, l]), float(cfg.gamma_beta_0[i, l]), float(cfg.replacement_mu[i, l]),
  float(cfg.gamma_beta_new[i, l]), bool(cfg.allow_replacement))`; `cap = mean_cap(beta_star, float(cfg.tau[i, l]),
  float(cfg.epsilon[i, l]))` (as `gamma.py:182-188`; Eqs. gm_betastar, gm_row).
- `_event(x, m, r, i, l, k)`: `"replacement"` if `r[i, l, k]`, elif `m[i, l, k]` `"repair"`, elif
  `x[i, 1:, k].any()` `"mission"`, else `"idle"` (s 2500).
- `_schedule_violations(cfg, result, x, m, r) -> list[str]` (s 2491-2498), on the rounded arrays, every cell of the
  fleet, one string per occurrence in `np.argwhere` order, numpy integers cast by `int()`:
  1. `T = result.get("T")`; `T is None or int(T) != cfg.T` → `f"result horizon T={T} differs from the input T={cfg.T}"`;
  2. `x.sum(axis=1) > 1` at `(i, k)` → `f"train i={i} has {n} activities on day k={k}"`;
  3. `x[:, 1:, :].sum(axis=0) != 1` at `(j0, k)` → `f"mission j={j0 + 1} is covered {n} times on day k={k}"`;
  4. `(m == 1) & (r == 1)` → `f"cell (i={i}, l={l}) has a repair and a replacement on day k={k}"`;
  5. `m + r > 0` while train `i` flies → `f"cell (i={i}, l={l}) has an intervention on day k={k} with a mission"`;
     then `m + r > 0` while `x[i, 0, k] == 0` → `"... on day k={k} without a depot day"`.
- `_record(i, l, k, event, field, expected, saved, error)`: the eight keys of s 2502, as `int` and `float`.
- `_check_report_path(report_path)`: `None` → `None`; lower-case suffix not in `REPORT_SUFFIXES` →
  `ValueError(f"report_path must end in .yaml, .yml or .json (got {str(report_path)!r}).")`; else `Path`.
- `_read_pair(input_path, result_path)`: `(load_config(read_input(input_path)), read_input(result_path))`.
- `_finish(report, destination, raise_on_failure)`: if `destination`: `mkdir(parents=True, exist_ok=True)` of its
  folder; `plain = _to_plain(report)`; `.json` → `json.dump(plain, f, indent=2, allow_nan=False)`, else
  `yaml.safe_dump(plain, f, sort_keys=False, allow_unicode=True)`; `encoding="utf-8"`. Then `raise_on_failure and
  not report["valid"]` → `AssertionError(f"{report['validator']}: the report has valid = false.")`. Return `report`.

**C2. `gamma_replay_validator.py`: the replay validator and its file version** (replace lines 52-211, 226-243;
s 2478-2523).
- Order: `_check_tolerance`; `started = time.perf_counter()`; `cells = gamma_cells(...)`; `ard1` = cells with
  `repair_model == "ard1"`; `x`, `m`, `r` (`optional=True`) by `_binary`; `mu = _array(..., "mu", (F, L, T), cells)`;
  `z = _array(..., "z", (F, L, T), [])`; `g_mu = _array(..., "g_mu", (F, L, T), ard1) if ard1 else None`;
  `mu_inc, _ = _increments(cfg)`; `caps = gamma_cell_caps(cfg, cells)`; `global_violations =
  _schedule_violations(...)`; `maxima = dict.fromkeys(("tracked_mean", "mean_anchor", "removed_mean", "cap",
  "repeatability"), 0.0)`; a local `check(i, l, k, event, field, expected, saved, error)` updates `maxima[field]` and
  appends `_record(...)` if `error > tolerance`.
- Per cell (Eqs. gm_map, a_anchor, a_z; s 2499-2511; Q2):

```python
anchored = str(cfg.repair_model[i, l]) == "ard1"
rho = float(cfg.rho[i, l])
ref_mu = float(cfg.mu_0[i, l])
ref_g = float(cfg.anchor_mu_0[i, l]) if anchored else 0.0
prev_mu, prev_g = ref_mu, ref_g                  # saved values of day k-1; day 0: the inputs
events = []
for k in range(T):
    event = _event(x, m, r, i, l, k)
    events.append(event)
    if event == "replacement":
        replacements += 1
        ref_mu = float(cfg.replacement_mu[i, l])
    elif event == "repair":
        repairs += 1
        ref_mu = (1.0 - rho) * ref_mu + rho * ref_g          # ref_g = 0 under ardinf
    else:
        ref_mu += float(np.dot(x[i, 1:, k], mu_inc[i, l, :, k]))
    if anchored and event in ("replacement", "repair"):
        ref_g = ref_mu
    saved = float(mu[i, l, k])
    check(i, l, k, event, "tracked_mean", ref_mu, saved, max(0.0, ref_mu - saved))
    if anchored:
        saved_g = float(g_mu[i, l, k])
        check(i, l, k, event, "mean_anchor", ref_g, saved_g, max(0.0, ref_g - saved_g))
    if not math.isnan(z[i, l, k]):
        removed = float(m[i, l, k]) * rho * (prev_mu - prev_g)
        check(i, l, k, event, "removed_mean", removed, float(z[i, l, k]),
              max(0.0, removed - float(z[i, l, k])))
    check(i, l, k, event, "cap", caps[i, l], saved, max(0.0, saved - caps[i, l]))
    prev_mu, prev_g = saved, (float(g_mu[i, l, k]) if anchored else 0.0)
for array in [mu] + ([g_mu] if anchored else []):     # s 2511, on the saved states
    first, last = float(array[i, l, cfg.H1 - 1]), float(array[i, l, T - 1])
    check(i, l, T - 1, events[-1], "repeatability", first, last, max(0.0, last - first))
```

- Report, in this order: `"validator": "Gamma replay validator"`, `"valid": not global_violations and not
  violations`, `"solver_status": str(result.get("status", "unknown"))`, `"dimensions"` (`F, M, L, H1, H2, T`),
  `"gamma_cells"`, `"gamma_ard1_cells"`, `"transitions_checked": len(cells) * T`, `"repairs"`, `"replacements"`,
  `"maximum_errors"`, `"global_violations"`, `"violations"`, `"timing": {"validation_wall_seconds": ...}`. Return
  `_finish(report, None, raise_on_failure)`.
- File version; C3 and C6 copy it with their own schedule function:

```python
def validate_gamma_replay_files(input_path, result_path, report_path=None, **kwargs):
    """File version (spec, "Common rules"): input and result paths; the report as YAML or JSON."""
    destination = _check_report_path(report_path)       # before any reading
    raise_on_failure = kwargs.pop("raise_on_failure", False)
    cfg, result = _read_pair(input_path, result_path)
    report = validate_gamma_replay_schedule(cfg, result, **kwargs)
    return _finish(report, destination, raise_on_failure)  # write, then raise
```

- Check: `run_tests_gamma.py ... replay` and the replay parts of 3.7 pass.

**C3. `gamma_tail_validator.py`: the exact validator** (rewrite lines 1-37, 72-561; keep `ExactGammaTerm`, 38-69;
s 2525-2579; Q3).
- Docstring: the procedure of s 2544-2556 in five lines; "No Gurobi object is read." Imports: `math`, `time`, `Path`,
  `Any`, `numpy as np`, `dataclass`; from `fleet_management.config`: `FleetConfig, _integer`; from
  `gamma_replay_validator`: `_array, _binary, _check_report_path, _check_tolerance, _event, _finish, _increments,
  _read_pair, _record, _schedule_violations, gamma_cells`; `moschopoulos_tail_probability`. Delete
  `scipy.stats.gamma`, `json`, `yaml`, `_required_array`, `_binary_decisions`, `_phase_value`, `_seed_term`,
  `_read_mapping`.
- `_seed(mean, rate, source, step) -> (terms, constant)`: `mean <= 0` → `([], 0.0)`; `math.isnan(rate)` →
  `([], float(mean))` (s 2553); else `([ExactGammaTerm(mean * rate, rate, source, step)], 0.0)`.
- `_exact_tail(terms, constant, tau, convolution_tolerance, max_series_terms)` → dict `estimate`, `upper_bound`,
  `remaining_mass`, `series_terms`: no terms → value `1.0 if constant > tau else 0.0`; terms and `constant >= tau` →
  value 1.0 (both with mass 0 and 0 terms); else the series on `tau - constant` (s 2553-2555).
- Arguments: `_check_tolerance`; `not 0.0 < convolution_tolerance < 1.0` → `ValueError("convolution_tolerance must
  lie in (0, 1).")`; `max_series_terms = _integer(max_series_terms, "max_series_terms")`, `<= 0` →
  `ValueError("max_series_terms must be positive.")`.
- Data as C2 without `z`, with `mu_inc, beta_inc = _increments(cfg)`. Per cell (s 2547-2553, 736-738, 2605):

```python
anchored = str(cfg.repair_model[i, l]) == "ard1"
remaining = max(0.0, 1.0 - float(cfg.rho[i, l]))   # config accepts rho up to 1 + 1e-12
beta_0, beta_new = float(cfg.gamma_beta_0[i, l]), float(cfg.gamma_beta_new[i, l])
mu_0, tau, eps = float(cfg.mu_0[i, l]), float(cfg.tau[i, l]), float(cfg.epsilon[i, l])
if anchored:                                     # the starting anchor splits the initial damage
    g_0 = float(cfg.anchor_mu_0[i, l])
    frozen, c_frozen = _seed(g_0, beta_0, "initial_anchor", -1)
    active, c_active = _seed(mu_0 - g_0, beta_0, "initial", -1)
else:
    frozen, c_frozen = [], 0.0
    active, c_active = _seed(mu_0, beta_0, "initial", -1)
for k in range(T):
    event = _event(x, m, r, i, l, k)
    if event == "replacement":
        seed, c_seed = _seed(float(cfg.replacement_mu[i, l]), beta_new, "replacement", k)
        if anchored:
            frozen, c_frozen, active, c_active = seed, c_seed, [], 0.0
        else:
            frozen, c_frozen, active, c_active = [], 0.0, seed, c_seed
    elif event == "repair":                      # rate / (1 - rho); removed for rho = 1
        repaired = [] if remaining == 0.0 else [t.after_ardinf(remaining) for t in active]
        if anchored:
            frozen, c_frozen = frozen + repaired, c_frozen + remaining * c_active
            active, c_active = [], 0.0
        else:
            active, c_active = repaired, remaining * c_active
    elif event == "mission":                     # one term per mission flown
        for j0 in np.flatnonzero(x[i, 1:, k]):
            mean, rate = float(mu_inc[i, l, j0, k]), float(beta_inc[i, l, j0, k])
            if mean > 0.0:
                active.append(ExactGammaTerm(mean * rate, rate, f"mission_{j0 + 1}", k))
```

  Then per day: `terms = frozen + active`, `constant = c_frozen + c_active`; `tail = _exact_tail(...)`, timed into
  `exact_tail_seconds`, counted in `exact_tail_evaluations` if `series_terms > 0`; `exact_mean = constant + sum of
  term means`; `frozen_mean = c_frozen + sum of frozen means`; `upper = tail["upper_bound"]`; `slack = eps - upper`.
  Checks (s 2558-2564) with the `check` of C2 and `maxima` over `("reliability", "tracked_mean", "mean_anchor")`:
  `reliability` (expected `eps`, saved `upper`, error `max(0, upper - eps)`); `tracked_mean` (expected `exact_mean`,
  saved `mu`); at ard1 cells `mean_anchor` (expected `frozen_mean`, saved `g_mu`).
- Step dictionary: `i, l, k, event, epsilon, constant, exact_mean, saved_mean, frozen_mean, saved_anchor` (the last
  two `None` at ardinf), `exact_tail_estimate, exact_tail_upper_bound, remaining_mass, series_terms,
  reliability_slack`; if `include_steps` or `slack < minimum_slack`, also `"terms"`: `[{**t.as_dict(), "part":
  "frozen"} for t in frozen] + [{**t.as_dict(), "part": "active"} for t in active]`. `steps` keeps every step if
  `include_steps`; `worst_reliability` is the first step with the smallest slack.
- Report: the replay keys (`"validator": "Gamma exact validator"`, `"timing": {"validation_wall_seconds",
  "exact_tail_seconds"}`), then `"minimum_reliability_slack"`, `"worst_reliability"`, `"maximum_remaining_mass"`,
  `"maximum_series_terms_used"`, `"total_series_terms"`, `"exact_tail_evaluations"`, and `"steps"` only with
  `include_steps`. `valid = not global_violations and not violations`. A series `RuntimeError` propagates (s 2568).
  File version `validate_gamma_tail_bound_files` as in C2.
- Check: tests 3.3 pass.

**C4. `gamma_tail_bound.py`: the series only, vectorised** (s 198, 2554-2555; T1; Q5, Q6).
- Keep `ConvolutionTail` (66-74), `moschopoulos_tail_probability` (signature, docstring, checks, equal-rate
  shortcut, both `RuntimeError` messages) and `_validate_parameters` (1151-1168). Delete every other class and
  function and the imports `product`, `prod`, `brentq`, `linprog`, `IntArray`. Docstring: lines 37-46 only.
- Replace the loop 968-1007 by the function `fast` of `<sp>\wp7\gamma_validators\moschopoulos_fast.py`: float64
  arrays `weights`, `k_gamma` of length `max_series_terms`; `k_gamma[k] = np.dot(shapes, (1 - ratios)**k)` (one
  multiplication of the power per step); `weights[k] = np.dot(k_gamma[1:k + 1], weights[k - 1::-1]) / k`; tails
  `gammaincc(total_shape + idx, beta_ref * threshold)` summed in blocks of 256 and at convergence. Same stopping rule
  and return values. Reason: today's loop runs in Python; at 12,426 series terms it takes 209 s, the prototype 2 s,
  with upper bounds equal to 2e-15.
- Check: tests 3.6 pass; `examples/regression/check_gamma_convolution_quadrature.py` passes with the worktree `src`
  on `PYTHONPATH`.

**C5. `gamma_stochastic_validator.py`: the Monte Carlo validator** (replace lines 1-30, 53-68, 137-383;
s 2581-2628; Q4).
- Imports: from `fleet_management.config`: `FleetConfig, _integer, _number`; from `gamma_replay_validator`:
  `_binary, _check_report_path, _check_tolerance, _event, _finish, _increments, _read_pair, gamma_cells,
  validate_gamma_replay_schedule`. Keep `_one_sided_binomial_upper` (36-50).
- `_draw(rng, mean, rate, size)`: `mean <= 0` → `np.zeros(size)`; `math.isnan(rate)` → `np.full(size, float(mean))`;
  else `rng.gamma(shape=mean * rate, scale=1.0 / rate, size=size)` (s 2604-2605).
- Arguments: `_integer` for `repetitions`, `random_seed`, `batch_size`; `_number` for `confidence_level` and a
  non-`None` `maximum_schedule_failure_rate`; then the range checks and messages of mc:150-161.
- Data: `cells = gamma_cells(...)`; `excluded_cells` for every other cell, keys of mc:174-185 with `"model"` from
  `result["model"]`; `x`, `m`, `r` by `_binary`; no other array and no schedule `ValueError` (delete mc:190-193,
  250-255); `mu_inc, beta_inc = _increments(cfg)`. Per batch and cell, the initial draws (s 2604-2605):

```python
use_latch = str(cfg.repair_model[i, l]) == "ard1"      # as mc:243
beta_0, beta_new = float(cfg.gamma_beta_0[i, l]), float(cfg.gamma_beta_new[i, l])
if use_latch:                                          # two terms: the anchor and the rest
    g_0 = float(cfg.anchor_mu_0[i, l])
    latch = _draw(rng, g_0, beta_0, size)
    state = latch + _draw(rng, float(cfg.mu_0[i, l]) - g_0, beta_0, size)
else:
    latch = np.zeros(size)
    state = _draw(rng, float(cfg.mu_0[i, l]), beta_0, size)
```

  The day loop stays mc:246-295 with these changes: `remaining = max(0.0, 1.0 - rho)`; the branch is chosen by
  `_event(x, m, r, i, l, k)` (no conflict checks); a replacement is `_draw(rng, float(cfg.replacement_mu[i, l]),
  beta_new, size)`; each mission `j0` in `np.flatnonzero(x[i, 1:, k])` adds `_draw(rng, float(mu_inc[i, l, j0, k]),
  float(beta_inc[i, l, j0, k]), size)`. The repair lines 266-273 and the counting stay (s 2606-2608).

- Report, in this order: `"validator": "Gamma Monte Carlo validator"`, `"mode": "stochastic"`, `"valid"`,
  `"repetitions"`, `"random_seed"`, `"maximum_schedule_failure_rate"`, `"confidence_level"`, `"batch_size"`,
  `"tolerance"`, `"scope": "Gamma cells only"`, `"complete_fleet_validation"`, `"excluded_cells"`, `"schedule"`
  (keys of mc:365-372), `"cells"` (keys of mc:312-334), `"timing"`. Return `_finish(report, None,
  raise_on_failure)`.
- Check: `run_tests_gamma.py ... monte_carlo` passes.

**C6. `gamma_stochastic_validator.py`: the dispatcher and its file version** (replace lines 33, 71-134, 386-473;
s 2630-2657).
- `VALIDATION_MODES = ("deterministic", "stochastic", "both")`; `_SETTING_DEFAULTS = {"mode": "deterministic",
  "repetitions": 100_000, "random_seed": 20260908, "maximum_schedule_failure_rate": None, "confidence_level": 0.95,
  "batch_size": 20_000}`.
- `_settings(cfg, **arguments)`: `block = cfg.options.get("validation") or {}`; a key of `block` outside
  `_SETTING_DEFAULTS` → `ValueError(f"unknown key {key!r} in 'validation'; the keys are
  {list(_SETTING_DEFAULTS)}.")` (Q9); each value is the argument if not `None`, else `block.get(key, default)`
  (s 2650); `mode not in VALIDATION_MODES` (no `strip`, no `lower`) → `ValueError(f"mode must be one of
  {VALIDATION_MODES} (got {mode!r}).")`.
- `validate_gamma_schedule`: `mode = settings.pop("mode")`; sub-validators always with `raise_on_failure=False`.
  `"deterministic"`: the replay report plus `report["mode"] = "deterministic"`; `"stochastic"`: the Monte Carlo
  report (`tolerance`, `**settings`); `"both"`: `{"validator": "Gamma replay and Monte Carlo validators", "mode":
  "both", "valid": deterministic["valid"] and stochastic["valid"], "deterministic": ..., "stochastic": ...}`. Return
  `_finish(report, None, raise_on_failure)` (S7).
- `validate_gamma_schedule_files` as in C2. Delete `_save_report` (450-459).
- Check: the whole test module: `<n> passed, 0 skipped, 0 failed`.

## 3. Tests

`tests/test_gamma_validators.py`, pytest style, no Gurobi (no skip). Imports: `json`, `math`, `numpy as np`,
`pytest`, `yaml`, `gammaincc`, `load_config`, `solver._save_results`, the validators from their `gamma_utils`
modules (not from `fleet_management`, whose `__init__.py` belongs to another plan). Plain helper functions, no
fixtures. Tolerances: `1e-12` for means and exact tails (the series stops below a remaining mass of `1e-12`); five
standard errors `5*sqrt(p(1-p)/n)` for Monte Carlo rates. Values from `<sp>\wp7\gamma_validators\
expected_values_output.txt`; all inputs checked by `fixture_check.py`.

**3.1 Helpers** (the hand case of `tests/test_gamma.py:147-168`).
- `_data(**changes)`: `F=2, M=1, H=[2, 2], model="gamma", repair_model="ard1", tau=1.0, epsilon=0.1, rho=0.5,
  mu_0=0.2, replacement_mu=0.1, mu=[[[0.3]], [[0.0]]], gamma_beta=20.0, C_M=1.0, C_R=2.0, C_rep=3.0, C_D=0.0`.
- `_SCHEDULE_X = ((0, 1, 0), (0, 0, 1), (0, 1, 2), (0, 0, 3), (1, 1, 1), (1, 1, 3))`, `m[0, 0, 1] = 1`,
  `r[0, 0, 3] = 1`: train 0 flies, is repaired, flies, is replaced; train 1 flies with mean 0.
- `_result(case, **changes)`: `status="optimal"`, `F, M, L=1, H1=2, H2=2, T=4, H=[2, 2]`, `model=[["gamma"],
  ["gamma"]]`, `x`, `m`, `r`, `mu`, `g_mu`, `z` as float arrays from the table; `changes` replace keys.

| Case | `_data` changes | train 0 `mu` | train 0 `g_mu` | train 0 `z` | train 1 |
|---|---|---|---|---|---|
| A | none | 0.5, 0.35, 0.65, 0.1 | 0.2, 0.35, 0.35, 0.1 | 0, 0.15, 0, 0 | `mu` 0.2, `g_mu` 0.2, `z` 0 |
| B | `repair_model="ardinf"` | 0.5, 0.25, 0.55, 0.1 | `NaN` | 0, 0.25, 0, 0 | `mu` 0.2, `g_mu` `NaN`, `z` 0 |
| C | `gamma_beta_0=20.0, anchor_mu_0=0.05, gamma_beta_new=30.0` | 0.5, 0.275, 0.575, 0.1 | 0.05, 0.275, 0.275, 0.1 | 0, 0.225, 0, 0 | `mu` 0.2, `g_mu` 0.05, `z` 0 |

  Caps: 0.744517324566 (`beta* = 20`), except train 1 of A and B: no rate, cap `tau = 1.0`.

**3.2 Replay validator.**
- `test_replay_accepts_the_hand_cases` (A, B, C): `valid`; `maximum_errors` all `<= 1e-12`; `repairs == 1`,
  `replacements == 1`, `transitions_checked == 8`, `gamma_cells == 2`, `gamma_ard1_cells` 2, 0, 2.
- `test_replay_one_sided`: A with `mu[0, 0, 2] = 0.70`: valid.
- `test_replay_records` (parametrize, case A, one change, exactly one violation): `mu[0, 0, 2] = 0.64` →
  `(k=2, "mission", "tracked_mean", expected 0.65, saved 0.64, error 0.01)`; `z[0, 0, 1] = 0.1` → `(1, "repair",
  "removed_mean", 0.15, 0.1, 0.05)`; `mu[0, 0, 2] = 0.8` → `(2, "mission", "cap", 0.744517324566, 0.8, 0.8 - cap)`;
  `mu[0, 0, 3] = 0.4` → `(3, "replacement", "repeatability", 0.35, 0.4, 0.05)`; all with `i = l = 0`.
- `test_replay_skips_nan_repair_cost`: A with `C_R=[[2.0], [0.0]]` and `z[1, 0, :] = NaN`: valid.
- `test_replay_global_violations`: `T=5` → exactly `["result horizon T=5 differs from the input T=4"]`, not valid,
  no violation; `x[1, 1, 0] = 1` → exactly `["mission j=1 is covered 2 times on day k=0"]`; `m[0, 0, 0] = 1` → the
  list holds "cell (i=0, l=0) has an intervention on day k=0 with a mission" and "... without a depot day".
- `test_replay_anchor_array`: B without the key `g_mu`: valid; A without it: `ValueError`, "missing 'g_mu'".

**3.3 Exact validator** (train 0; upper bounds).

| Case | day 0 | day 1 | day 2 | day 3 | train 1, every day |
|---|---|---|---|---|---|
| A | `gammaincc(6, 16)` | `gammaincc(6, 32)` | 0.015106503928459624 | 0 | 0 (constant 0.2) |
| B | `gammaincc(6, 16)` | `gammaincc(6, 36)` | 0.0042209224829627395 | 0 | 0 |
| C | `gammaincc(10, 20)` | 1.0539648750813092e-06 | 0.009116347927228055 | `gammaincc(3, 30)` | `gammaincc(4, 20)` |

- `test_exact_tails_of_the_hand_cases` (A, B, C; `include_steps=True`): `valid`; each step: `|upper - expected| <=
  1e-12`, `estimate <= expected + 1e-14`; `exact_mean` = the `mu` column; at A and C `frozen_mean` = the `g_mu`
  column; `worst_reliability` at `(0, 0, 2)`; `minimum_reliability_slack = 0.1 - day-2 value`.
- `test_exact_terms_of_case_c`: the day-1 step of C has the terms (shape, rate, part) (1, 20, frozen),
  (3, 40, frozen), (6, 40, frozen).
- `test_exact_records`: A with `epsilon=0.01` → one violation `(k=2, "reliability", expected 0.01, saved = day-2
  value)`, not valid, `raise_on_failure=True` → `AssertionError`; A with `mu[0, 0, 2] = 0.64` → `tracked_mean`;
  A with `g_mu[0, 0, 2] = 0.30` → `mean_anchor`, expected 0.35.
- `test_exact_full_repair`: B with `rho=1.0`, train 0 `mu` 0.5, 0.0, 0.3, 0.1: day 1 upper bound 0, day 2
  `gammaincc(6, 20)`.
- `test_exact_tail_edge_cases` (`_exact_tail`): `([], 0.2, 1.0)` → 0; `([], 1.2, 1.0)` → 1;
  `([ExactGammaTerm(1.0, 1.0, "s", 0)], 1.0, 1.0)` → 1.
- `test_exact_report_keys`: every replay key, plus the six keys of C3; no `steps` without `include_steps`.
- `test_exact_series_limit`: A with `max_series_terms=5` → `RuntimeError`; `convolution_tolerance=1.0` → `ValueError`.

**3.4 Monte Carlo validator** (`repetitions=20000`, `batch_size=7000`, default seed).
- Helpers: `_mc_data(**changes)` = `F=2, M=1, H=[1, 1], model="gamma", repair_model="ardinf", tau=1.0,
  epsilon=0.1, rho=0.5, mu_0=0.0, mu=0.5, gamma_beta=4.0`, costs of 3.1; `_mc_result(flights, repairs, model=None)`:
  `T=2`, keys of `_result` without `mu`, `z`, `g_mu`. Default schedule: train 0 flies both days, train 1 idle.
- `test_monte_carlo_matches_the_exact_tail`: schedule rate near `gammaincc(4, 4) = 0.43347012036670896`;
  `cells[0]["worst_time_step"] == 1`; `cells[0]["failed_replays"] == schedule["failed_replays"]`;
  `cells[1]["failed_replays"] == 0`.
- `test_monte_carlo_known_constant`: `mu_0=0.6`, no `gamma_beta_0`: rate near `gammaincc(4, 1.6) =
  0.9211865127702811`; `maximum_sampled_damage >= 0.6`.
- `test_monte_carlo_splits_the_initial_damage`: `repair_model="ard1", rho=1.0, mu_0=[[0.6], [0.0]],
  gamma_beta_0=4.0, mu=[[[0.5]], [[0.0]]]`; train 0 in the depot both days, repaired on day 0; train 1 flies both
  days. Default anchor: cell rate near `gammaincc(2.4, 4) = 0.1418232030653136`; `anchor_mu_0=0.0`: no failure.
- `test_monte_carlo_is_reproducible`: two runs give equal reports without `timing`.
- `test_monte_carlo_target`: `maximum_schedule_failure_rate=0.5` → valid, both `*_target_met` true; `0.4` → not
  valid, and `AssertionError` with `raise_on_failure=True`.
- `test_clopper_pearson`: `(0, 100, 0.95)` → 0.02951304960703993; `(3, 100, 0.95)` → 0.07571079374983006;
  `(100, 100, 0.95)` → 1.0.
- `test_monte_carlo_excludes_other_cells`: `model=[["gamma"], ["inverse_gaussian"]], eta=0.5`:
  `complete_fleet_validation` false; one excluded record, `i=1, l=0, model="inverse_gaussian"`.
- `test_monte_carlo_argument_errors`: `repetitions` 0 or 2.5, `batch_size=0`, `confidence_level=1.0`,
  `maximum_schedule_failure_rate=1.5` → `ValueError`.

**3.5 Dispatcher** (case A; `repetitions=500`).
- `test_dispatcher_default_mode`: `mode == "deterministic"`, `validator == "Gamma replay validator"`.
- `test_dispatcher_reads_the_validation_block`: `_data(validation={"mode": "stochastic", "repetitions": 500,
  "batch_size": 200})` → `mode`, `repetitions`, `batch_size` from the block, `random_seed == 20260908`; with the
  argument `repetitions=300` → 300.
- `test_dispatcher_both`: keys `deterministic`, `stochastic`, `valid` = both valid; A with `mu[0, 0, 2] = 0.64` and
  `raise_on_failure=True` → `AssertionError`.
- `test_dispatcher_rejects_modes` (`"Both"`, `"monte_carlo"`, `" stochastic"`) → "mode must be one of";
  `validation={"repetition": 5}` → "unknown key".

**3.6 Moschopoulos series.**
- `([6, 6], [40, 20], 0.8)` → 0.015106503928459624 and `([7, 9], [20, 40], 1.0)` → 0.009116347927228055 (upper
  bounds, `1e-12`; `estimate <= upper_bound`; `remaining_mass <= 1e-12`); equal rates `([2, 3], [5, 5], 1.0)` →
  `gammaincc(5, 5)`, one series term, mass 0; `max_series_terms=5` → `RuntimeError`, "did not reach"; threshold 0 →
  `ValueError`.
- `test_tail_bound_keeps_only_the_series`: no name `calculate_*`, `required_shape_for_tail`,
  `enumerate_count_vectors`, `shapes_from_expected_damage` in the module.

**3.7 Files and errors.**
- `test_file_versions` (result `.yaml`, `.json`, `.h5` × report `.yaml`, `.json`): input by `yaml.safe_dump(_data())`,
  result A by `_save_results`; the replay, exact and dispatcher file versions, with `report_path = tmp_path / "out" /
  f"report{suffix}"`, return a valid report, and the file loads with `valid` true.
- `test_file_versions_keep_nan` (`.h5`, `.yaml`): B with `C_R=[[2.0], [0.0]]` and `z[1, 0, :] = NaN`: valid.
- `test_report_path_is_checked_first`: `report_path="report.txt"` and a missing result file → `ValueError`.
- `test_file_version_writes_before_raising`: A with `mu[0, 0, 2] = 0.64`, `raise_on_failure=True` →
  `AssertionError`, and the report file holds `valid: false`.
- `test_errors` (replay, exact, Monte Carlo): `model="inverse_gaussian", eta=0.5` → "no Gamma cell"; no `model` key
  → "missing 'model'"; result `model` `[["gamma"], ["inverse_gaussian"]]` on A → "result 'model'";
  `x[0, 1, 0] = 0.5` → "not binary"; `tolerance=0` → "tolerance must be positive"; replay and exact only:
  `mu[0, 0, 0] = NaN` → "non-finite", `mu` of shape `(2, 1, 3)` → "shape".

## 4. Exports

Names and signatures do not change; the import lines of today's `__init__.py` (9-17) stay.

| Module | Public API (s 231) | Internal (s 233) |
|---|---|---|
| `gamma_replay_validator` | `validate_gamma_replay_schedule`, `validate_gamma_replay_files` | `gamma_cells`, `gamma_cell_caps` (the Gamma dashboard of the `validator` plan may use them, s 2456) |
| `gamma_stochastic_validator` | `validate_gamma_stochastic_schedule`, `validate_gamma_schedule`, `validate_gamma_schedule_files` | `VALIDATION_MODES` |
| `gamma_tail_validator` | none | `validate_gamma_tail_bound_schedule`, `validate_gamma_tail_bound_files`, `ExactGammaTerm` |
| `gamma_tail_bound` | none | `moschopoulos_tail_probability`, `ConvolutionTail` |

## 5. Callers outside src

Broken by this plan (WP8 fixes them; most also use inputs or keys that WP4 to WP6 removed):

| Script | Cause |
|---|---|
| `examples/regression/check_gamma_ard1_truth_table.py:14`, `check_gamma_ard1_replacement_truth_table.py:12`, `check_gamma_ardinf_replacement_truth_table.py:5`, `check_gamma_big_m_formulation.py:16`, `gamma_replacement_truth_table_common.py:14`; `experiments/vbz_F8_feasibility_transfer/run_f8_feasibility_transfer.py:44` | import `gamma_diagnostics` (section 6) |
| `experiments/gamma_horizon_stabilization/gamma_horizon_stabilization.sbatch:67-68` | copies `gamma_diagnostics.py` |
| `examples/regression/check_gamma_tail_bound.py:39-47` | the removed `calculate_*`, `shapes_from_expected_damage` (C4) |
| `examples/regression/check_gamma_exact_validator.py`, `check_gamma_validator_edge_cases.py` | exact report keys `backend`, `minimum_conservativeness_margin`, `numerics`, `reasons`, `history` (C3) |
| `examples/regression/check_gamma_replay_validator.py`, `check_mixed_gamma_ard1_public.py`, `check_mixed_public_interface.py`, `report_gamma_complexity.py` (and `check_gamma_complexity_diagnostics.py`), `report_gamma_validator_slide8.py` | replay field `physical_mean` and others, key `backend` (C2) |

Also broken: `examples/regression/check_gamma_stochastic_validator.py`, whose hand result has no key `model` (C1).
Not broken by this plan: `check_gamma_convolution_quadrature.py`, `run_gamma_schedule_validation.py`,
`check_matched_ard_inputs.py`, `run_progressive_vbz_feasibility.py` (they read keys that stay). `remote_smoke_test.py` and `euler/` do not call the Gamma tools (grep, 29 Oct 2026).

## 6. Removals

Claude removes these files at the commit; the executors delete no file.
- **`gamma_utils/gamma_diagnostics.py`** (292 lines): not in the spec layout (s 194-198); imports `gurobipy` at module
  level; its estimators count rows of the removed Gamma formulation; `base.py` no longer uses
  `collect_gurobi_model_statistics` (grep, 29 Oct 2026). No importer in `src` or `tests`; outside, section 5.
- **`gamma_utils/gamma_process.py`** (480 lines): not in the layout; only importer `validation/validator.py:14`, the
  legacy diagnostic that the `validator` plan rewrites (decision "Validation tools"; `merge/wp1_removals.md`, U2).
  Remove it in the commit of that rewrite: `__init__.py` imports `validator.py`.
- Inside a file, by step C4: the calibrators and helpers of `gamma_tail_bound.py`.
- Not removed by this plan: `gamma_utils/GAMMA_DIAGNOSTICS.md` (Q8).

## 7. Questions

No question blocks a step: each step follows the recommendation, and another answer changes the step named.

- **Q1. A result whose `model` differs from the input** (C1). s 2364 gives no rule. Options: **`ValueError` at the
  first cell that differs** (recommended: the Gamma inputs of a cell exist only where the input says `gamma`); the
  result's model alone (fails later on `NaN` rates).
- **Q2. `removed_mean` on the saved states** (C2). s 2509 does not say whether `mu_{k-1}`, `g_{k-1}` are saved or
  reference values. Options: **saved** (recommended: it is the model's row, Eq. a_z, `base.py:767-794`, and Eq.
  c_zpost; at binary points the saved `mu_{k-1} - g_{k-1}` is at least the reference difference under both laws, so
  it implies the other check); reference.
- **Q3. Choices that s 2525-2579 leaves open for the exact validator** (C3): (a) under `ard1` the initial damage
  splits at `anchor_mu_0`, as s 2605 says for the Monte Carlo validator (else a repair could go below the level of
  the last intervention, against s 736-738); (b) `valid` also needs no global violation; (c) records have the replay
  keys, and for `reliability` `expected` is `epsilon` and `saved` the upper bound. Recommendation: all three.
- **Q4. Monte Carlo errors and keys** (C5). s 2472-2476 has no error for a schedule violation; s 2619-2628 has no
  `interpretation`, `failure_event`, `solver_status`, `dimensions`, `gamma_cells`. Recommendation: **neither**; the
  mode `"both"` reports the schedule through the replay validator.
- **Q5. Callers of the removed calibrators** (C4): `greedy_warm_start.py:146-150, 164` (`sweeps` plan) imports them
  inside a function, so `import fleet_management` still works. Recommendation: the `sweeps` plan uses
  `gamma.smallest_rate` and `gamma.mean_cap`; Claude checks that the two plans agree.
- **Q6. Cost of the exact tail** (C4). The series length grows with the ratio of the largest to the smallest rate,
  which frequent `ardinf` repairs make large. One term repaired 2, 4, 6, 8, 10 times at `rho = 0.5` beside a fresh
  one: 151, 725, 3,056, 12,426, 49,884 terms; per tail, today 0.07, 1, 13, 209 s (10 repairs not run), vectorised
  0.002, 0.009, 0.12, 2, 31 s. From 11 repairs, more than 100,000 terms: `RuntimeError` (s 2568). Options: **the
  spec's procedure, vectorised** (recommended for the merge); later, a certified split of a term `X` of tiny mean,
  `Pr(S + X > tau) <= Pr(S > tau - x0) + Pr(X > x0)` (Claude's idea, not checked), as a line of
  `Things to change.txt` if the user wants it.
- **Q7. Private names across modules** (C1, C3, C5): `config._lookup`, `_integer`, `_number`, `solver._to_plain`.
  Options: **import them** (recommended: WP7 may not edit `config.py` or `solver.py`; `solver._read_input` is a
  precedent); copy them.
- **Q8. `GAMMA_DIAGNOSTICS.md`** describes the calibration, `gamma_diagnostics.py` and `report_gamma_complexity.py`,
  which all leave; the decision "Documents" says it is updated. Options: **remove it with `gamma_diagnostics.py`**
  (recommended: nothing it describes remains, and the spec describes the validators); rewrite it. Owner: Claude.
- **Q9. Unknown keys in the input block `validation`** (C6): `config.py` checks only that it is a mapping, so a typo
  such as `repetition` is ignored; s 486 lists six keys. Options: **`ValueError`** (recommended; one line); ignore.

## 8. Review (29 Oct 2026)

Read in full: the four modules and the test module, against sections 2 and 3, `wp7_interface.md` and spec 1192-1303,
2461-2658. The modules follow C1 to C6; tails, replays, one-sided errors and report keys are right. Departures: decision
1 (`solver.read_result`: right), the docstring of `gamma_tail_bound.py` (right), float64 in the series (corrected).
Dashboard plan: no mismatch (steps carry `i`, `l`, `k`, `exact_tail_upper_bound`; replay keys as s 2516-2523).

**Defects corrected**
- `gamma_tail_bound.py:75-122`: the series summed in float64 where the replaced loop used `np.longdouble`. Restored
  `np.longdouble` for the weights, the mass and the tail (80 bits on Linux, float64 on Windows), so the series again
  reaches a `convolution_tolerance` near `1e-19` on Linux, as before. Against git HEAD: equal term counts, bounds
  within 1.8e-16; 8 repairs (12,425 terms) in 0.18 s; the quadrature check passes (error 1.6e-12).
- `gamma_replay_validator.py:290`, `gamma_tail_validator.py:246`: `str(result.get("status", "unknown"))` gave `"None"`
  for a result read by `read_result` without a status; now `"unknown"`. `gamma_replay_validator.py:192`:
  `if destination:` became `if destination is not None:`.
- Tests that could not see three behaviours, now added: the global strings of an extra activity and of a repair with
  a replacement, and the status `"unknown"` (`test_replay_global_violations`, line 188); a global violation alone
  makes the exact report not valid (`test_exact_records`, line 313); a replacement draw uses `gamma_beta_new`
  (`test_monte_carlo_replacement_draw`, line 451; `_mc_result` takes `replacements`). Section 5 now lists
  `check_gamma_stochastic_validator.py` as broken (its result has no `model`).

**Decisions**
- Float64 understates the remaining mass by 2e-15 to 9e-15 at 151 to 3,055 terms (`review_mass_rounding.py`, 40
  digits): 1 % of the default `convolution_tolerance`, far below the pass tolerance `1e-8`. `np.longdouble` is kept for
  parity on Linux; no floor on `convolution_tolerance`: an unreachable one ends in the spec's `RuntimeError`.
- The names the tests read are the intended API: `ExactGammaTerm(shape, rate, source, created_step)`;
  `_exact_tail(terms, constant, tau, convolution_tolerance, max_series_terms)` with keys `estimate`, `upper_bound`,
  `remaining_mass`, `series_terms`; the step keys of C3; `maximum_errors` keys `reliability`, `tracked_mean`,
  `mean_anchor`; `exact_tail_evaluations` counts the tails computed by the series (3 in case A).
- The `NaN` guard of `removed_mean` cannot be seen from outside (`max(0.0, nan)` is 0.0); it stays, as s 2509 says.

**Spec sentences** (worktree line numbers):

```latex
% 2469, after the item "Tolerance" of the common rules:
\item \textbf{Gamma cells.} The Gamma cells are the cells whose entry of the result key \texttt{model} is \texttt{gamma}; this key must have the shape $F \times L$ and equal the input key \texttt{model}.
\item \textbf{Event of a day.} The event of a cell on day $k$ is \texttt{"replacement"} if $x^r_{i\ell k} = 1$, else \texttt{"repair"} if $x^m_{i\ell k} = 1$, else \texttt{"mission"} if train $i$ flies a mission, else \texttt{"idle"}. Every validator applies it, also on a day with a global violation.
\item \textbf{Indices.} In a report, \texttt{i}, \texttt{l} and \texttt{k} are 0-based array indices; \texttt{k} is day $k+1$.
\item \textbf{Order of a file version.} It checks the suffix of \texttt{report\_path} before it reads a file, reads the result with \texttt{read\_result}, writes the report with \texttt{null} for a value that is not finite, and only then raises the \texttt{AssertionError}.
% 2474, the row ValueError becomes:
\rejrow{\texttt{ValueError}}{\texttt{report\_path} has another suffix than YAML or JSON; \texttt{cfg} has no Gamma cell; the result has no key \texttt{model}, or its \texttt{model} has another shape than $F \times L$ or differs from the input at a cell; a required array is missing, has a wrong shape, or has a non-finite entry at a cell where it is read; an array that must be binary is not within \texttt{tolerance} of $\{0,1\}$}
% 2491, step 1 of the replay validator, after "(strings in global_violations)":
over every cell of the fleet, one string per occurrence
% 2499, step 2: replace "with the initial value, starting anchor and replacement value of the cell" by
starting at $\mu_0$ and, under $\ARDone$, at the starting anchor $g^{\mu}_0$, with the replacement value of the cell
% 2509, row removed_mean: after "- z_{i\ell k}\}$", add
on the saved \texttt{mu} and \texttt{g\_mu} (on day~1: $\mu_0$ and $g^{\mu}_0$),
% 2512, after the table of error fields:
In a record, \texttt{expected} is the reference value, the cap (\texttt{cap}) or the saved value on day $H_1$ (\texttt{repeatability}, recorded on day $T$ with its event), and \texttt{saved} is the saved value.
% 2551, after the list of step 1 of the exact validator:
Under $\ARDone$ the initial damage is a frozen part of mean $g^{\mu}_0$ and a repairable part of mean $\mu_0 - g^{\mu}_0$, both with rate $\beta_0$ or both constants; a replacement term is frozen. A repair multiplies a repairable constant by $1 - \rho$.
% 2553, step 2: replace "and $1$ if $c > \tau$" by
it is $1$ if $c \ge \tau$ and some term exists; without terms it is $1$ if $c > \tau$, else $0$
% 2555, after step 4:
\item The weights, the mass and the tail of the series are summed in \texttt{numpy.longdouble}.
% 2564, after the pass rule:
\texttt{valid} is true if there is no global violation (step 1 of the replay validator) and every day passes. A failed condition adds a record with the keys of the replay record, with field \texttt{reliability} (\texttt{expected} $\varepsilon_{i\ell}$, \texttt{saved} the upper bound), \texttt{tracked\_mean} (\texttt{expected} $\mathbb{E}D_k$) or \texttt{mean\_anchor} (\texttt{expected} the exact mean of the frozen part); \texttt{maximum\_errors} has these three fields.
% 2577-2578, the rows "further keys" and steps become:
\rejrow{\texttt{maximum\_series\_terms\_used}, \texttt{total\_series\_terms}, \texttt{exact\_tail\_evaluations}}{largest and total number of series terms; number of tails computed by the series; \texttt{timing} adds \texttt{exact\_tail\_seconds}}
\rejrow{\texttt{steps}}{with \texttt{include\_steps=True}, per cell and day: \texttt{i}, \texttt{l}, \texttt{k}, \texttt{event}, \texttt{epsilon}, \texttt{constant}, \texttt{exact\_mean}, \texttt{saved\_mean}, \texttt{frozen\_mean}, \texttt{saved\_anchor}, \texttt{exact\_tail\_estimate}, \texttt{exact\_tail\_upper\_bound}, \texttt{remaining\_mass}, \texttt{series\_terms}, \texttt{reliability\_slack}, \texttt{terms}; \texttt{worst\_reliability} is the first step with the smallest slack}
% 2594, 2598: rows repetitions and batch_size, after "must be positive"; 2595: row random_seed, at its end:
an integer; a number with zero fractional part is accepted, a flag is not
% 2609, after the steps of the Monte Carlo validator:
The validator raises no error for a global violation of the schedule; it applies the event of each day. The arguments used are \texttt{repetitions}, \texttt{random\_seed}, \texttt{maximum\_schedule\_failure\_rate}, \texttt{confidence\_level}, \texttt{batch\_size} and \texttt{tolerance}.
% 2652, after the item "Default":
\item \textbf{Reports.} In mode \texttt{"deterministic"} the report is the replay report with the key \texttt{mode}. The validators run without raising; \texttt{raise\_on\_failure} applies to the report of the dispatcher.
% 2656, the row ValueError becomes:
\rejrow{\texttt{ValueError}}{\texttt{mode} has another value (values are case sensitive); the input block \texttt{validation} has a key other than \texttt{mode}, \texttt{repetitions}, \texttt{random\_seed}, \texttt{maximum\_schedule\_failure\_rate}, \texttt{confidence\_level}, \texttt{batch\_size}}
```

**Final counts.** `tests/test_gamma_validators.py`: 37 test functions, 60 passed, 0 skipped, 0 failed
(`run_tests_gamma.py`). `tests/test_gamma.py`: 21 passed, 15 skipped (Gurobi). `import_check.py`: `ok`; names check:
0 undefined names in each module.
