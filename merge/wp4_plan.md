# WP4 plan: the input and output layer

Status (29 Oct 2026): plan done (sections 1 to 5); executors done; review done (section 7). Next: Claude reviews
the diff and commits on `merge/lore-into-main`.

- **Scope.** WP4 (work package 4) rewrites `src/fleet_management/config.py` and `src/fleet_management/solver.py` of the
  worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge` (branch `merge/lore-into-main`), and adds
  `tests/test_config.py` and `tests/test_io.py`. Interface and file ownership: `merge/wp4_wp5_interface.md` (binding).
- **Authority.** `spec/spec.tex` of the worktree. "spec 352" means line 352 of that file; `c:` is `config.py` and `s:`
  is `solver.py` before WP4, as read on 28 Oct 2026.

## 1. Gaps

Items marked WP5, WP6 or WP7 belong to another package and are not planned here.

**Keys and errors**
- G1 c:303-318 / spec 491-516, 526: only `depot_capacity` is rejected; no other rejected key, no "unknown key" error;
  missing keys raise `KeyError`; the spec requires `ValueError`.
- G2 c:343-363, 393 / spec 499, 511-515: the aliases `method`, `xi`, `repair_rho`, `mu_new`, `v_new`, `C_S` are accepted;
  the spec rejects them.
- G3 c:37, 73, 79, 369-376, 410-415, 548-565 / spec 503-504: `gamma_beta_bound` and `gamma_calibration_method` are
  inputs; the spec rejects both.
- G4 c:355-358, 385-396 / spec 526: missing `tau`, `epsilon`, `rho`, `mu_0` give `KeyError`; `C_M`, `C_R`, `C_rep`,
  `C_D` are optional in the code.
- G5 c: none / spec 2697-2707: no check of the keys that each cell type requires.

**`FleetConfig`**
- G6 c:48-143 / interface table: `H_prof`, `costs`, `raw`, `gamma_beta_bound`, `gamma_calibration_method`, `cell()`,
  `describe()`, `is_single_model` must go. Missing: `reliability_impl`, `encoding`, `tangent_ref`, four anchors,
  `support_0`, `R_0`, `cgf_0`, `support_new`, `R_new`, `cgf_new`, `sigma`, `eta`, `v_max_user`, `C_R`, `C_rep`,
  `pwl_points`, `allow_replacement`, `replacement_as_new`, `C_M`, `C_D`, `penalty_type`, `support_const`, `range_mu`,
  `range_v`, `range_R`, `range_K`.
- G7 c:153 / spec 279: a pair `H` is stored as a tuple; it must be the list `[H1, H2]` (`yaml.safe_dump` refuses
  tuples).

**Shapes, `null` and broadcasting**
- G8 c:332-341, 365-384 / spec 424: operating profiles use `H_prof`, which is `H1` for a Gamma fleet; spec: always `H2`.
- G9 c:166-195, 241-287 / spec 352-354, 528: no `null` rule. `null` where a key applies is not an error; values where
  it does not apply give no `UserWarning` and are not set to NaN (interface).
- G10 c:220-238, 385-396 / spec 466-469, 589: `C_M` is a per-component vector; `C_R`, `C_rep` accept only a scalar or
  `(L,)`, not `(F, L)`; `C_D` has no sign check.
- G11 c:198-217 / spec 377, 534: `component_names` accepts numbers (`str(item)`) and a bare string.
- G12 c:149-160, 319-327 / spec 360-363, 533, 535: `int()` truncates `2.5` and accepts `True`; `F > M` is not checked.

**Selectors and options**
- G13 c:34 / spec 372, 536: `wiener` is not a supported model.
- G14 c:354 / spec 373: the default `repair_model` is `ard1`; spec: `ardinf`.
- G15 c:42, 484-494 / spec 136, 1335: Chernoff with `ard1` is refused; spec: every model and bound accepts both laws.
- G16 c:352-354, 397-406, 477-483 / spec 371-377, 536-538: `bound_method` and `repair_model` are checked only at
  rainflow cells; `reliability_impl` and `encoding` are one option value, not per-cell selectors.
- G17 c: none / spec 470, 539: `penalty_type` is not read; no rule for `quadratic`.
- G18 c:397-406 / spec 473-488, 587-588: run options have no defaults and no checks; `validation` is not accepted;
  keys that are not run options sit in `options`.
- G19 c:441-458 / spec 417, 551: `replacement_as_new` defaults to true and checks only `mu`, `v` at rainflow cells;
  spec: default false, five keys, every cell that reads them.

**Value checks** (absent in the code unless a location is given)
- G20 c:470 / spec 545: `epsilon < 0.5` at G (Gaussian) and W (Wiener) cells.
- G21 spec 547-549: `0 <= mu_0 < tau`; `v_0`, `support_0`, `R_0` and replacement values `>= 0`; `tangent_ref` in
  [0, 1); `pwl_points >= 1`.
- G22 spec 393-396, 550: starting anchors: inputs, defaults, `0 <= anchor <= initial value`.
- G23 spec 556-557: natural ranges of initial and replacement values; the replacement state satisfies the rows of a
  replacement day.
- G24 c:474-475, 495-503 / spec 562-564: the code requires `mu`, `v`, `support`, `cgf` `> 0`; spec: `0 <= mu < tau`;
  `v`, `support`, `cgf` `>= 0` and zero where `mu` is zero; a transitory profile needs its operating profile.
- G25 c:495-498 / spec 403, 444: every rainflow cell requires `v` and `v_0`; spec: only Ca (Cantelli) and Be
  (Bernstein) cells.
- G26 spec 569-571: G three-sigma checks; IG (inverse Gaussian) `eta > 0`; W `sigma > 0`.
- G27 c:540-547 / spec 459-460, 572: Gamma requires `gamma_beta_0` when `mu_0 > 0` and `gamma_beta_new` when
  `replacement_mu > 0`; spec: both optional (known damage), finite and `> 0` where given.
- G28 spec 573: `v_max_user` checks.
- G29 spec 578-580: Bernstein: Popoviciu `v <= b^2/4`; `tau > 2*kappa*b/3`; tangent and pwl conditions.
- G30 spec 581-582: Chernoff: `K >= s*mu` for increments, `K_0`, anchors and `K_new`. Kept from the code:
  `s > ln(1/eps)/tau` (c:506-518).
- G31 interface: the derived fields `support_const` (Eq. `rf_bsupport`) and `range_*` are not computed.

**Input reader**
- G32 s:219-223 / spec 2663-2669: YAML and JSON are opened without `encoding="utf-8"` (Windows then reads cp1252);
  no check that the file holds one mapping.
- G33 s:230-300 / spec 2671-2680: the HDF5 reader reads a fixed key list without `model` (so no HDF5 input loads),
  turns integers into floats, does not decode strings or parse JSON text.
- G34 spec 183: the spec puts the reader in `config.py`; it is in `solver.py` (Question Q1).

**`solve` and the result files**
- G35 s:16-21 / spec 210-211: the default `results_path` is `None`; `warm_start_mode` is missing.
- G36 s:303-310, 324-325 / spec 1859-1865: `None` writes `output.yaml` (spec: no file); another extension is written as
  YAML (spec: `ValueError`, raised before the input is read).
- G37 s: none / spec 220: `solve_mapping` is missing.
- G38 s:9 / interface: `base` is imported at module level, so readers and writers need Gurobi.
- G39 s:67-212 / decisions "Gamma formulation": legacy dispatch and the old Gamma backend; WP1 removed its module
  `legacy/`, so this path fails at run time.
- G40 s:58-61, 93-98 / spec 1701-1738, decisions "Output schema": `solve` adds `model_assignment`, `backend`,
  `degradation`, `models` to the result; none is a result key.
- G41 s:328-449 / spec 1701-1738, 1866-1867: YAML and JSON write `main`'s old key list and miss `model`, the selectors,
  `R`, `K`, `g_*`; NaN is written as `.nan` (YAML) or `NaN` (JSON); spec: `null`.
- G42 s:486-579 / spec 2683-2690: HDF5 writer: wrong key set, `u` as a dataset (spec: attribute), no JSON text for
  `model`, the reports and the selectors.
- G43 s:473, 482: result files are written without `encoding="utf-8"`.

**Other packages**: G44 to G48 (WP5, WP6, WP7) are in section 4, with their locations.

## 2. Executor tasks

**Rules for both executors**
- Edit only the two files of your task. Do not spawn sub-agents. Do not delete any file; write scratch files only to
  `C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp4\`.
  No installs, no git command that changes state. Never put the user's name or email address in any request.
- **Code style**: write like the existing `config.py` and `solver.py`: a module docstring; section banners `# ----`
  with a title; small private helpers with a one-line docstring; type hints; f-string messages that quote key names
  (`'tau'`); numpy idioms (`np.full`, `np.broadcast_to(...).copy()`, `np.where`); a comment only where the reason is
  not obvious. Keep `from __future__ import annotations` in `config.py`.
- **Errors**: `TypeError` for an input that is not a mapping; `ValueError` for every key, shape and value check;
  `UserWarning` for ignored values. Cell indices in messages are 0-based array indices. Selector values match exactly
  (case sensitive). Use the message templates of this plan; tests match only the quoted fragments.
- **Tests and runner**: section 3. After every step run the tests; a step is done when its tests and all earlier tests
  pass. Runner (written and checked on 28 Oct 2026): `run_tests.py` in the scratchpad folder above. Command, from that
  folder: `PYTHONIOENCODING=utf-8 python run_tests.py <test file> [name filter]`. It registers a stub package
  `fleet_management` whose `__path__` is the worktree's `src/fleet_management` (so `__init__.py`, which imports
  gurobipy, does not run), a stub `pytest` with `raises(exc, match=)` and `warns(cls, match=)`, and passes a fresh
  `tmp_path` to a test that asks for it. If the file is missing, recreate it from this description (about 80 lines).
- A step changes at most about 150 lines of code; deleting a whole old block counts as one change.

### Task A: `config.py` and `tests/test_config.py` (10 steps)

Task B starts when step A1 is done. Shorthand: G, IG, W, Ga, RF = the five models; Mk, Ca, Ho, Be, Ch = the five
rainflow bounds; `m[...]` = the masks of step A4.

**A1. Input reader** (new section "Input reader" at the end of `config.py`; spec 2659-2690).
- Top imports: add `import json`, `import warnings`, `from pathlib import Path`, `from scipy.special import ndtri`.
  Import `yaml` and `h5py` inside the functions only (interface: `config.py` imports only the standard library, numpy
  and scipy at module level).
- `INPUT_EXTENSIONS = (".yaml", ".yml", ".json", ".h5", ".hdf5")`.
- `read_input(path) -> dict`: `ext = Path(path).suffix.lower()`. YAML: `yaml.safe_load` on
  `open(path, "r", encoding="utf-8")`. JSON: `json.load`, same `open`. HDF5: `_read_hdf5(path)`. Other extension:
  `ValueError(f"Unsupported input file type '{suffix}'. Supported types: {list(INPUT_EXTENSIONS)}")`. If the data is
  not a `dict`: `TypeError(f"the input file {path} must hold one mapping from key names to values.")`.
- `_read_hdf5(path) -> dict`: one entry per root attribute (`_hdf5_value(value, text_json=True)`) and per root
  dataset (`_hdf5_value(ds[()], text_json=False)`). A root group: `ValueError(f"HDF5 input must be flat: '{name}' is a
  group.")`. A name that is both: `ValueError(f"HDF5 key '{name}' is both an attribute and a dataset.")`.
- `_hdf5_value(value, *, text_json)`: `bytes` (also `np.bytes_`) → decoded UTF-8 `str`; a `str` that begins with `{`
  or `[` → `json.loads` when `text_json` (on failure `ValueError(f"HDF5 attribute text {value[:40]!r} is not valid
  JSON.")`); a numpy array of kind `S`, `O` or `U` → nested list of decoded `str`; another array → `.tolist()`; a
  numpy scalar → `.item()`.
- Check: tests A1.

**A2. Key tables** (replace c:34-42; spec 491-516, 526).
- Constants: `SUPPORTED_MODELS = ("gaussian", "inverse_gaussian", "wiener", "gamma", "rainflow")`; `RAINFLOW_BOUNDS`
  (as now); `REPAIR_MODELS = ("ardinf", "ard1")`; `RELIABILITY_IMPLS = ("exact", "tangent", "pwl")`;
  `ENCODINGS = ("bigm", "indicator")`; `PENALTY_TYPES = ("inf_norm", "quadratic")`;
  `OBJECTIVE_MODES = ("total", "operating_average", "evaluation_total")`; `REQUIRED_KEYS = ("F", "M", "H", "model",
  "tau", "epsilon", "rho", "mu_0", "mu", "C_M", "C_R", "C_rep", "C_D")`; `RUN_OPTION_DEFAULTS = {"verbose": 1,
  "mip_gap": 0.1, "time_limit": 3600.0, "fast": False, "gurobi_params": None, "objective_mode": "total",
  "evaluation_horizon": None, "progress_interval_seconds": 5.0, "relaxation_warm_start": False,
  "relaxation_time_limit": 60.0, "validation": None}`.
- `CANONICAL_KEYS`, the 63 keys in the order of the spec tables:
  ```
  F M L H | model repair_model bound_method reliability_impl encoding component_names |
  tau epsilon tangent_ref pwl_points | rho anchor_mu_0 anchor_v_0 anchor_R_0 anchor_K_0 |
  mu_0 v_0 support_0 R_0 cgf_0 | replacement_mu replacement_v support_new R_new cgf_new replacement_as_new
  allow_replacement | mu v support cgf s_chernoff gamma_beta mu_trans v_trans support_trans cgf_trans
  gamma_beta_trans | sigma eta v_max_user gamma_beta_0 gamma_beta_new | C_M C_R C_rep C_D penalty_type |
  then the 11 keys of RUN_OPTION_DEFAULTS in that order
  ```
- `REJECTED_KEYS`: dict of the 21 keys of spec 498-515 (`tail_bound` and `method`, `n_workers` and `warm_start`, `xi`
  and `repair_rho` share a row). Value: the "Message" column, with `\texttt{x}` written `'x'`, math as plain text
  (`$0$` → `0`, `$1/\texttt{beta}$` → `1/'beta'`), "(Section ...)" removed. Examples: `"xi": "Use 'rho', with the same
  meaning."`; `"C_P": "Remove it. The loop constraint is a hard constraint and has no penalty."`.
- Delete `GAMMA_CALIBRATION_METHODS`, `_NEEDS_SUPPORT`, `_NEEDS_CGF`, `_ARD1_UNSUPPORTED`.
- `_check_keys(data) -> None`, in this order: not a `dict` → `TypeError("input data must be a mapping (dict).")`; for
  each key in input order, a rejected key → `ValueError(f"key '{key}' is not accepted. {REJECTED_KEYS[key]}")`; keys
  outside `CANONICAL_KEYS` (as `str`, sorted) → `ValueError(f"unknown key(s) {unknown}; the canonical keys are:
  {', '.join(CANONICAL_KEYS)}.")`; missing `REQUIRED_KEYS` → `ValueError(f"missing required key(s): {', '.join(repr(k)
  for k in missing)}.")`.
- Call `_check_keys(data)` as the first line of the current `load_config`, replacing c:301-318.
- Check: tests A2.

**A3. Scalar helpers** (replace `_parse_horizon`, `_infer_L`, `_component_names` in place; their signatures stay).
- `_integer(value, name) -> int`: `bool`/`np.bool_` refused; `int`/`np.integer` accepted; `float`/`np.floating` with
  an integer value accepted (Question Q9); else `ValueError(f"'{name}' must be an integer (got {value!r}).")`.
- `_number(value, name) -> float`: `int`, `float` or numpy number, not `bool`; else `ValueError(f"'{name}' must be a
  number (got {value!r}).")`.
- `_flag(value, name) -> bool`: `bool` or `np.bool_` only; else `ValueError(f"'{name}' must be true or false (got
  {value!r}).")`.
- `_parse_horizon(H_raw)`: list or tuple → length 2 else `ValueError("'H' must be an integer or a pair [H1, H2].")`;
  `H1 = _integer(H_raw[0], "H1")`, `H2` likewise; both `>= 1` else `ValueError(f"'H' entries must be >= 1 (got {H1},
  {H2}).")`; return `[H1, H2], H1, H2, H1 + H2`. Scalar → `H = _integer(H_raw, "H")`, `>= 1` else `ValueError(f"'H'
  must be >= 1 (got {H}).")`; return `H, H, H, 2 * H`.
- `_infer_L(data, F)`: `model` a `str` → `L = _integer(data["L"], "L")` if given, else 1. Else
  `np.array(model, dtype=object)`: ndim 1 → `len`; ndim 2 → `shape[1]`; else `ValueError("'model' must be a string, a
  list of L strings or an F x L array.")`. A given `L` that differs: `ValueError(f"'L'={L_given} differs from the
  component axis of 'model' ({L}).")`. `L < 1`: `ValueError("'L' must be >= 1.")`.
- `_component_names(value, L)`: `None` → `[f"Component {n + 1}" ...]` (as now); not a list or tuple →
  `ValueError("'component_names' must be a list of L strings.")`; a non-`str` entry → `ValueError(f"'component_names'
  entries must be strings (got {item!r}).")`; length and empty-name errors as now (c:210-216).
- `_run_options(data, H1) -> dict`: start from `dict(RUN_OPTION_DEFAULTS)`, take given values, then check (spec
  587-588): `verbose` `_integer` in {0, 1} (`True`/`False` accepted as 1/0); `mip_gap` `_number` `>= 0`; `time_limit`,
  `progress_interval_seconds`, `relaxation_time_limit` `_number` `> 0`; `fast`, `relaxation_warm_start` `_flag`;
  `gurobi_params`, `validation` `None` or `dict`; `objective_mode` in `OBJECTIVE_MODES`; `evaluation_horizon` `None`
  or `_integer` `> H1`; `evaluation_total` with `evaluation_horizon` `None` is an error. Message:
  `f"'{key}' must be {rule} (got {value!r})."`, rule text as in this list (for example "a number >= 0", "one of
  ('total', ...)", "an integer > H1 = 2", "given when objective_mode is 'evaluation_total'").
- Check: a scratch script (stub package as in the runner) calls each helper on one good and one bad value.

**A4. Array helpers** (new section "Cells, selectors and profiles", next to the old helpers; old ones go in A5).
- `_float_array(raw, name)`: `np.asarray(raw, dtype=float)` (`None` becomes NaN); `TypeError`/`ValueError` →
  `ValueError(f"'{name}' must be a number or a regular array of numbers.")`.
- `_warn_not_applicable(name, values, applies)`: `values` is `(F, L, ...)`; a cell "has a value" if some entry is not
  NaN (float) or not `None`/`''` (selector). If such cells exist where `applies` is false: `warnings.warn(f"'{name}'
  has values at {n} cell(s) where it does not apply, for example cell (i={i}, l={l}); they are ignored.",
  UserWarning, stacklevel=3)`.
- `_selector(data, key, F, L, applies, default, allowed, fill="") -> (F, L) object array`. Absent → `default` (a
  `str`) at `applies` cells. A `str` → every cell; a 1-D list of length `L` → tiled over trains; an `(F, L)` array →
  per cell (then `_warn_not_applicable`); other shape → `ValueError(f"'{key}' shape {shape} must be a string, ({L},),
  or ({F},{L}).")`. `None` and `''` are null. Every non-null entry, at every cell, is in `allowed`, else
  `ValueError(f"'{key}' has the value {v!r} at cell (i={i}, l={l}); allowed: {allowed}.")`. Null at an `applies`
  cell → `ValueError(f"'{key}' is null at cell (i={i}, l={l}), where it applies.")`. Cells outside `applies` hold
  `fill`.
- `_masks(model, bound_method, repair_model, reliability_impl=None) -> dict` of `(F, L)` bool arrays: `"all"`; one per
  model name; one per bound (`model == "rainflow"` and `bound_method == b`); `"ard1"`; `"v_profile"` = G | Ca | Be;
  `"variance"` = `v_profile` | W; `"variance_anchor"` = `ard1` & `v_profile`; `"support"` = Ho | Be; `"quadratic"` =
  G | W | Ca | Ho | Be; with `reliability_impl`: `"tangent"`, `"pwl"` (`reliability_impl == ...`).
- `_cell_value(data, key, F, L, applies, default=None, optional=False) -> (F, L) float64`. Absent: `optional` → all
  NaN; `default is None` and some `applies` cell → `ValueError(f"missing required key '{key}', needed at cell (i={i},
  l={l}).")`; else `default` (a number or an `(F, L)` array) broadcast. Present: `_float_array`; shapes `()`, `(L,)` (tiled), `(F, L)` (per cell,
  `_warn_not_applicable`); other → `ValueError(f"'{key}' shape {shape} must be scalar, ({L},), or ({F},{L}).")`. At
  `applies` cells: NaN → null error as in `_selector` (not when `optional`: there NaN means "not given", Question Q2);
  infinite → `ValueError(f"'{key}' must be a finite number at cell (i={i}, l={l}).")`. Return the values at `applies`
  cells and NaN elsewhere.
- `_profile(data, key, F, L, M, Hs, applies, rates=False, required=False) -> (F, L, M, Hs) float64 or None`. Absent →
  `None`, or the missing-key error if `required` and some cell applies. Accepted shapes, matched in this order:
  profiles `()`, `(M,)`, `(L, M)`, `(L, M, Hs)`, `(F, L, M)`, `(F, L, M, Hs)`; rates (`gamma_beta*`) `()`, `(L,)`,
  `(F, L)`, `(F, L, M)`, `(F, L, M, Hs)`. Expand eagerly (a missing axis is repeated). Per-cell shapes are those that
  start with `F`; they get `_warn_not_applicable`. Shape error: list the accepted shapes as in c:258-261. Null and
  infinite entries at `applies` cells → the two errors above. Return NaN at the other cells.
- Note: with `F > M`, the shapes `(L, M, Hs)` and `(F, L, M)` never coincide, so the spec's tie rule (spec 426) cannot
  be tested; the matching order keeps it.
- Check: a scratch script reproduces the shape table of each helper on `F=3, L=2, M=2, Hs=3`.

**A5. `FleetConfig` and `load_config`** (replace c:45-143 and c:290-432; delete `_fl_scalar`, `_fl_str`,
`_component_cost`, `_flmh_prof`, `_gamma_rate_prof`, `_validate_cells`).
- `FleetConfig` fields, in this order (the seven sizes without default; `component_names` with
  `field(default_factory=list)`; `options` with `field(default_factory=dict)`; the rest `= None`); keep the property
  `models`; delete `cell`, `describe`, `is_single_model`:
  ```
  F, M, L: int; H: object (int or [H1, H2]); H1, H2, T: int; component_names: list
  model, repair_model, bound_method, reliability_impl, encoding        # (F, L) object, see the interface
  tau, epsilon, tangent_ref, rho, anchor_mu_0, anchor_v_0, anchor_R_0, anchor_K_0,
  mu_0, v_0, support_0, R_0, cgf_0, replacement_mu, replacement_v, support_new, R_new, cgf_new,
  s_chernoff, sigma, eta, v_max_user, gamma_beta_0, gamma_beta_new, C_R, C_rep    # (F, L) float64, NaN off-cell
  pwl_points: int; allow_replacement, replacement_as_new: bool; C_M, C_D: float; penalty_type: str
  mu, v, support, cgf, gamma_beta                                       # (F, L, M, H2) or None
  mu_trans, v_trans, support_trans, cgf_trans, gamma_beta_trans         # (F, L, M, H1) or None
  options: dict
  support_const, range_mu, range_v, range_R, range_K                    # (F, L) float64, set by _derive (A6)
  ```
- `load_config(data)` body, in this order (`cv(key, applies, default=None, optional=False)` and `pf(key, Hs, applies,
  rates=False, required=False)` are local shorthands for `_cell_value` and `_profile`):
  ```
  _check_keys(data)
  F, M = _integer(data["F"], "F"), _integer(data["M"], "M")
  F < 1 or M < 1  -> ValueError(f"'F' and 'M' must be >= 1 (got F={F}, M={M}).")
  F <= M          -> ValueError(f"'F' must be > 'M' (got F={F}, M={M}).")
  H, H1, H2, T = _parse_horizon(data["H"]);  L = _infer_L(data, F)
  model = _selector(data, "model", F, L, all, None, SUPPORTED_MODELS)
  repair_model = _selector(..., "repair_model", all, "ardinf", REPAIR_MODELS)
  bound_method = _selector(..., "bound_method", model == "rainflow", "cantelli", RAINFLOW_BOUNDS)
  m = _masks(model, bound_method, repair_model)
  reliability_impl = _selector(..., "reliability_impl", m["quadratic"], "exact", RELIABILITY_IMPLS, fill="exact")
  encoding = _selector(..., "encoding", all, "bigm", ENCODINGS)
  m = _masks(model, bound_method, repair_model, reliability_impl)
  tau, epsilon, rho, mu_0 = cv(k, m["all"]) for each
  s_chernoff = cv("s_chernoff", m["chernoff"])
  v_0 = cv("v_0", m["variance"], 0.0); support_0 = cv("support_0", m["bernstein"], 0.0); R_0 = cv("R_0", m["hoeffding"], 0.0)
  cgf_0 = cv("cgf_0", m["chernoff"], s_chernoff * mu_0)
  replacement_mu = cv(.., m["all"], 0.0); replacement_v = cv(.., m["variance"], 0.0)
  support_new = cv(.., m["bernstein"], 0.0); R_new = cv(.., m["hoeffding"], 0.0)
  cgf_new = cv("cgf_new", m["chernoff"], s_chernoff * replacement_mu)
  anchor_mu_0 = cv(.., m["ard1"], mu_0); anchor_v_0 = cv(.., m["variance_anchor"], v_0)
  anchor_R_0 = cv(.., m["ard1"] & m["hoeffding"], R_0); anchor_K_0 = cv(.., m["ard1"] & m["chernoff"], cgf_0)
  tangent_ref = cv(.., m["tangent"], 0.5); sigma = cv(.., m["wiener"]); eta = cv(.., m["inverse_gaussian"])
  v_max_user = cv(.., m["variance"], optional=True)
  gamma_beta_0, gamma_beta_new = cv(.., m["gamma"], optional=True)
  C_R, C_rep = cv(.., m["all"])
  mu = pf("mu", H2, all, required=True); v = pf("v", H2, m["v_profile"], required=True)
  support = pf("support", H2, m["support"], required=True); cgf = pf("cgf", H2, m["chernoff"], required=True)
  gamma_beta = pf("gamma_beta", H2, m["gamma"], rates=True, required=True)
  <key>_trans = pf(<key>_trans, H1, same mask, rates as above); given without <key> ->
      ValueError(f"'{key}_trans' needs the operating profile '{key}'.")
  pwl_points = _integer(data.get("pwl_points", 8), ..); < 1 -> ValueError("'pwl_points' must be an integer >= 1.")
  allow_replacement = _flag(data.get(.., True)); replacement_as_new = _flag(data.get(.., False))
  C_M, C_D = _number(data[k], k); not finite or < 0 -> ValueError(f"'{k}' must be a finite number >= 0.")
  penalty_type = data.get("penalty_type", "inf_norm"); not in PENALTY_TYPES ->
      ValueError(f"'penalty_type' must be one of {PENALTY_TYPES} (got {penalty_type!r}).")
  options = _run_options(data, H1)
  cfg = FleetConfig(...); later steps add the checks here; return cfg
  ```
- Check: tests A5. The old checks are gone until A6 to A9, so only A1, A2 and A5 tests run here.

**A6. State checks and derived fields** (new section "Consistency checks"; spec 542-552, interface "Derived").
- `_TOL = 1e-12` and `_le(a, b)`: `a <= b + _TOL * np.maximum(1.0, np.abs(b))` (Question Q3). Use `_le` for every
  non-strict inequality between an input and a computed value; strict inequalities stay exact.
- `_require(cfg, ok, key, rule)`: `ok` is an `(F, L)` bool array, true where the check passes or does not apply. At the
  first false cell: `ValueError(f"cell (i={i}, l={l}) model={cfg.model[i, l]}: '{key}' {rule}.")`.
- `_check_state(cfg)` (`m = _masks(...)` of `cfg`; `when` = the check runs only then):

  | Key | Cells | `ok` where the cells apply | Rule text |
  |---|---|---|---|
  | `tau` | all | `tau > 0` | "must be > 0" |
  | `epsilon` | all | `0 < eps < 1` | "must be in (0, 1)" |
  | `epsilon` | G, W | `eps < 0.5` | "must be < 0.5 at a gaussian or wiener cell" |
  | `rho` | all | `0 < rho <= 1` | "must be in (0, 1]" |
  | `mu_0` | all | `0 <= mu_0 < tau` | "must satisfy 0 <= mu_0 < tau" |
  | `v_0`, `support_0`, `R_0` | variance, Be, Ho | `>= 0` | "must be >= 0" |
  | `replacement_mu`, `replacement_v`, `support_new`, `R_new` | all, variance, Be, Ho; when `allow_replacement` | `>= 0` | "must be >= 0 when allow_replacement is true" |
  | `tangent_ref` | tangent | `0 <= t < 1` | "must be in [0, 1) at a cell with the form tangent" |
  | `anchor_mu_0`, `anchor_v_0`, `anchor_R_0`, `anchor_K_0` | ard1, variance_anchor, ard1 & Ho, ard1 & Ch | `0 <= a <= mu_0, v_0, R_0, cgf_0` | f"must satisfy 0 <= {key} <= {initial key}" |
  | `replacement_mu`, `replacement_v`, `support_new`, `R_new`, `cgf_new` | all, variance, Be, Ho, Ch; when `replacement_as_new` | `== 0` | "must be 0 when replacement_as_new is true" |
  | `C_R`, `C_rep` | all | `>= 0` | "must be >= 0" |
  | `reliability_impl` | quadratic; when `C_D > 0` and `penalty_type == "quadratic"` | `== "exact"` | "must be 'exact' when penalty_type is 'quadratic' and C_D > 0" |

- `_derive(cfg)`, inside `np.errstate(divide="ignore", invalid="ignore")`; `kappa = ln(1/eps)`,
  `kappa_g = ndtri(1 - eps)`:
  - `support_const` at Be cells: elementwise max of `support_0`, `max over (M, H2) of support`, the same of
    `support_trans` if given, and `support_new` if `allow_replacement` (use `np.nan_to_num(..., nan=0.0)` before the
    max); NaN elsewhere (spec Eq. `rf_bsupport`, 1494-1500).
  - `c = kappa * support_const / 3`; `range_mu = tau`; `range_v` = `tau**2 / kappa_g**2` at G and W,
    `eps * tau**2 / (1 - eps)` at Ca, `tau * (tau - 2c) / (2 kappa)` at Be, NaN elsewhere; `range_R = 2 tau**2 / kappa`
    at Ho; `range_K = s_chernoff * tau - kappa` at Ch; NaN elsewhere (spec 556, 821-825).
- In `load_config`: `_check_state(cfg)`, then `_derive(cfg)`.
- Check: tests A6.

**A7. Increment checks** (spec 560-564, 569, 578, 582).
- `_lookup(op, trans, H1, H2)`: `idx = np.concatenate([np.arange(H1) % H2, np.arange(H2)])`; `out = op[..., idx]`; if
  `trans` is not `None`: `out[..., :H1] = trans`. This is Eq. `c_phase`; WP5 keeps its own accessor (Question Q7).
- `_check_increments(cfg)`; a cell fails if any mission and day fails (`.all(axis=(2, 3))`):

  | Key | Cells | Fails where | Rule text |
  |---|---|---|---|
  | `mu`, `mu_trans` | all | not `0 <= mu < tau` | "must satisfy 0 <= mu < tau for every mission and day" |
  | `v`, `support`, `cgf` and `*_trans` | where given | `< 0` | "must be >= 0 for every mission and day" |
  | `v`, `support`, `cgf` (looked up) | where not NaN | `mu == 0` and value `!= 0` | "must be 0 where the mean increment 'mu' is 0" |
  | `v` (looked up) | G | not `_le(3*sqrt(v), mu)` | "must satisfy mu >= 3*sqrt(v) for every mission and day" |
  | `v` (looked up) | Be | not `_le(v, support**2/4)` | "must satisfy v <= support^2/4 for every mission and day" |
  | `cgf` (looked up) | Ch | not `_le(s_chernoff*mu, cgf)` | "must satisfy cgf >= s_chernoff*mu for every mission and day" |

  "Looked up" means `_lookup(op, trans, H1, H2)` of each key on its own; `mu` is looked up the same way. Run the
  single-key rows first.
- In `load_config`: after `_check_state`, before `_derive`.
- Check: tests A7.

**A8. Model checks** (spec 567-582).
- `_check_models(cfg)`; `R` = when `allow_replacement`; `s` = `s_chernoff`; `c` and `kappa` as in A6:

  | Key | Cells | `ok` | Rule text |
  |---|---|---|---|
  | `mu_0` | G | `_le(3*sqrt(v_0), mu_0)` | "must be >= 3*sqrt(v_0) at a gaussian cell" |
  | `replacement_mu` | G; R | `_le(3*sqrt(replacement_v), replacement_mu)` | "must be >= 3*sqrt(replacement_v) at a gaussian cell" |
  | `eta` | IG | `> 0` | "must be > 0" |
  | `sigma` | W | `> 0` | "must be > 0" |
  | `gamma_beta`, `gamma_beta_trans` | Ga | every entry `> 0` | "must be > 0 for every mission and day" |
  | `gamma_beta_0`, `gamma_beta_new` | Ga, where not NaN | `> 0` | "must be > 0 where given" |
  | `v_max_user` | variance, where not NaN | `> 0`; `_le(v_0, v)`; R: `_le(replacement_v, v)` | "must be > 0", "must be >= v_0", "must be >= replacement_v" |
  | `v_0` | Be | `_le(v_0, support_0**2/4)` | "must be <= support_0^2/4 at a bernstein cell" |
  | `replacement_v` | Be; R | `_le(replacement_v, support_new**2/4)` | "must be <= support_new^2/4 at a bernstein cell" |
  | `tau` | Be | `tau > 2*kappa*support_const/3` | "must be > 2*ln(1/epsilon)*b/3, b the support constant of the cell" |
  | `tangent_ref` | Be & tangent | `tangent_ref*tau < sqrt(tau*(tau - 2c))` | "must satisfy tangent_ref*tau < sqrt(tau*(tau - 2c)) at a bernstein cell" |
  | `pwl_points` | Be & pwl | `tau/(2*pwl_points) < sqrt(tau*(tau - 2c))` | "must satisfy tau/(2*pwl_points) < sqrt(tau*(tau - 2c)) at a bernstein cell" |
  | `s_chernoff` | Ch | `s > ln(1/eps)/tau` | "must be > ln(1/epsilon)/tau" |
  | `cgf_0` | Ch | `_le(s*mu_0, cgf_0)` | "must be >= s_chernoff*mu_0" |
  | `anchor_K_0` | ard1 & Ch | `_le(s*anchor_mu_0, anchor_K_0)` | "must be >= s_chernoff*anchor_mu_0" |
  | `cgf_new` | Ch; R | `_le(s*replacement_mu, cgf_new)` | "must be >= s_chernoff*replacement_mu" |

  Run the `tau` row of Be before the two `sqrt` rows. No check of `gamma_beta_0` against `mu_0` (G27).
- In `load_config`: after `_derive`.
- Check: tests A8.

**A9. Natural ranges and replacement state** (spec 554-557, 851-911, model sections).
- `_check_ranges(cfg)`:
  - Natural ranges, rule "must lie in [0, M], M the natural range of its state at this cell": `mu_0` vs `range_mu`
    (all), `v_0` vs `range_v` (variance), `R_0` vs `range_R` (Ho), `cgf_0` vs `range_K` (Ch); when
    `allow_replacement` also `replacement_mu`, `replacement_v`, `R_new`, `cgf_new` against the same ranges.
    `ok = (value >= 0) & _le(value, range)`.
  - Then, only when `allow_replacement`, the rows of a replacement day:
    - `_quad_coefficients(cfg, m) -> (chi, c2, c1, side)`, `(F, L)` arrays, NaN off the quadratic cells: G and W:
      `replacement_v`, `1/kappa_g**2`, `0`, `tau`; Ca: `replacement_v`, `eps/(1 - eps)`, `0`, `tau`; Ho: `R_new`,
      `2/kappa`, `0`, `tau`; Be: `replacement_v`, `1/(2 kappa)`, `-support_const/3`, `tau - kappa*support_const/3`
      (spec table 863-874).
    - `_cap(mu, tau, c2, c1, form, tangent_ref, P)`: `f(x) = c2*(tau - x)**2 + c1*(tau - x)`,
      `fp(x) = -2*c2*(tau - x) - c1`; `exact`: `f(mu)`; `tangent`: `r = tangent_ref*tau`, `f(r) + fp(r)*(mu - r)`;
      `pwl`: the largest `f(h) + fp(h)*(mu - h)`, `h = (p - 0.5)*tau/P`, over the segments `p = 1..P` with
      `(p - 1)*tau/P <= mu <= p*tau/P` (`-inf` if none); pick per cell with `np.where` on `reliability_impl`.
    - Side row: key `replacement_mu`, cells quadratic, `_le(replacement_mu, side)`, rule "must satisfy the side row of
      the reliability row (mu <= tau, or mu <= tau - c at a bernstein cell)".
    - Reliability row: key `replacement_v` (G, W, Ca, Be) and key `R_new` (Ho), `_le(chi, cap)`, rule "must satisfy
      the reliability row on a replacement day, in the form of reliability_impl".
    - Markov: key `replacement_mu`, cells Mk, `_le(replacement_mu, eps*tau)`, rule "must be <= epsilon*tau at a markov
      cell".
    - `v_max_user` and Chernoff are covered by A8 and the natural ranges. IG and Gamma: section 4.
- In `load_config`: last, after `_check_models`.
- Check: tests A9.

**A10. Finish.** Rewrite the module docstring (what `config.py` holds, with the spec section names) and the docstrings
of `load_config` and `FleetConfig`. Check: `grep` finds none of `H_prof`, `cfg.costs`, `self.raw`, `_validate_cells`,
`alias(`, `GAMMA_CALIBRATION_METHODS`, `_ARD1_UNSUPPORTED` in `config.py`, and `gamma_beta_bound` and
`gamma_calibration_method` only inside `REJECTED_KEYS`; all tests of `tests/test_config.py` pass.

### Task B: `solver.py` and `tests/test_io.py` (3 steps)

**B1. Solve path** (replace s:1-310; spec 206-220, 1851-1865).
- Delete `_solve_mixed`, `_identify_result`, `_solve_gamma_fleet`, `_require_uniform`, `_uniform_over_vehicles`,
  `_legacy_gamma_beta`, `_cfg_to_gamma_kwargs`, `_read_input`, `_read_hdf5`, `_resolve_results_path`, and the
  module-level import of `base`.
- Imports: `json`, `math`, `os`, `Path`, `h5py`, `numpy as np`, `yaml`, and
  `from fleet_management.config import INPUT_EXTENSIONS, load_config, read_input`.
- `_read_input = read_input` with the comment "Old name, used by the validation tools." (G47).
- Constants: `RESULT_EXTENSIONS = (".yaml", ".yml", ".json", ".h5", ".hdf5")`; `WARM_START_MODES = ("start", "hint",
  "both")`; `RESULT_KEYS` (the 51 keys of spec 1701-1738 in table order: `status objective mip_gap bound F M L H1 H2 T
  H model component_names tau mu_0 v_0 bound_method repair_model reliability_impl encoding objective_mode
  evaluation_horizon x m r idle mu v z R K g_mu g_v g_R g_K u step_costs J_initialization J_op J_op_average
  peak_damage damage_penalty operating_objective J_total projected_evaluation_cost evaluation_objective
  component_costs performance optimization_progress warm_start relaxation_warm_start`); `_HDF5_DATASET_KEYS = (x mu v
  R K g_mu g_v g_R g_K z m r idle step_costs tau mu_0 v_0)`; `_HDF5_JSON_KEYS = (model component_names component_costs
  performance optimization_progress warm_start relaxation_warm_start)`; `_SELECTOR_KEYS = (bound_method repair_model
  reliability_impl encoding)` (spec 2686-2690).
- `solve(input_path, results_path="output.yaml", *, warm_start=None, warm_start_mode="start") -> dict`: input file
  missing → `FileNotFoundError` (message as s:41); extension not in `INPUT_EXTENSIONS` → `ValueError` (message as
  s:43-46); `path = _check_results_path(results_path)`; `data = read_input(input_file)`; return
  `solve_mapping(data, path, warm_start=warm_start, warm_start_mode=warm_start_mode)`. Keep the numpy-style docstring.
- `solve_mapping(data, results_path=None, *, warm_start=None, warm_start_mode="start") -> dict`: mode not in
  `WARM_START_MODES` → `ValueError(f"warm_start_mode must be one of {WARM_START_MODES} (got {warm_start_mode!r}).")`;
  `path = _check_results_path(results_path)`; `cfg = load_config(data)`;
  `from fleet_management.degradation_model import base` (inside the function); `result = base.solve_mixed(cfg,
  warm_start=warm_start, warm_start_mode=warm_start_mode)`; write with `_save_results(result, path)` if `path` is not
  `None`; return `result` unchanged (no added keys, G40).
- `_check_results_path(results_path) -> Path | None`: `None` → `None`; no suffix → `.yaml` added; suffix (lower case)
  not in `RESULT_EXTENSIONS` → `ValueError(f"Unsupported results file type '{suffix}'. Supported types:
  {list(RESULT_EXTENSIONS)}")`; `path.parent` missing → `FileNotFoundError` (message as s:51); not writable
  (`os.access(..., os.W_OK)`) → `PermissionError` (message as s:53).
- Check: tests B1. `import fleet_management.solver` works in the runner (no gurobipy).

**B2. YAML and JSON writers** (replace s:314-483; spec 1862-1867, Questions Q4, Q5).
- `_to_plain(value)`: `None` → `None`; `dict` → `{str(k): _to_plain(v)}`; `list`/`tuple` → list; `np.ndarray` →
  `_to_plain(value.tolist())`; numpy scalar → `_to_plain(value.item())`; `float` → itself if finite, else `None`;
  `bool`, `int`, `str` → itself; else `TypeError(f"cannot write a value of type {type(value).__name__} to a result
  file.")`.
- `_text_output(result) -> dict`: for each key of `RESULT_KEYS` in order, skip a value that is `None` or a missing
  key, except `objective`, always written (`None` → `null`); value `_to_plain(...)`. Keys outside `RESULT_KEYS` are not
  written.
- `_save_results(result, path)`: by `Path(path).suffix.lower()` → `_save_yaml`, `_save_json`, `_save_hdf5`; another
  extension → the `ValueError` of `_check_results_path`.
- `_save_yaml`: `open(path, "w", encoding="utf-8")`; `yaml.safe_dump(output, f, default_flow_style=False,
  sort_keys=False, allow_unicode=True)`. `_save_json`: same `open`; `json.dump(output, f, indent=2, allow_nan=False)`.
- Check: tests B2.

**B3. HDF5 writer** (replace s:486-579; spec 2683-2690).
- `_save_hdf5(result, path)`: `h5py.File(path, "w")`; for each key of `RESULT_KEYS` with a value that is not `None`
  (also `objective`, Question Q8): `_HDF5_DATASET_KEYS` → `f.create_dataset(key, data=np.asarray(value,
  dtype=np.float64))`; `_HDF5_JSON_KEYS`, and a selector that is not a `str` → `f.attrs[key] =
  json.dumps(_to_plain(value), allow_nan=False)`; `H` → `np.asarray(value, dtype=np.int64)`; any other key (numbers,
  strings, a selector that is a `str`) → the value, with numpy scalars converted by `.item()`; NaN and infinite
  numbers are kept.
- Check: tests B3.

## 3. Tests

Plain pytest functions; only the fixture `tmp_path`; no `parametrize` (a loop inside the test); `pytest.raises(...,
match=...)` and `pytest.warns(UserWarning, match=...)` only; "no warning" is checked with
`warnings.catch_warnings(record=True)` and `warnings.simplefilter("always")`. Every value below was checked by a
script on 28 Oct 2026 (`numbers_check.py` in the scratchpad folder). "error 'x'" means `pytest.raises(ValueError,
match="'x'")`.

**Helpers of `tests/test_config.py`**
- `EXAMPLE_YAML`: spec 2716-2774 verbatim. It passes every check (spec 2713; checked by hand: G three-sigma
  `0.02 >= 0.01897` at the replacement, Chernoff anchor `0.5 >= 5*0.1` at equality).
- `_example(**changes)`: `yaml.safe_load(EXAMPLE_YAML)` updated with `changes`.
- `_small(model, bound=None, **changes)`: `{"F": 2, "M": 1, "H": [1, 2], "model": model, "tau": 1.0, "epsilon": 0.1,
  "rho": 0.5, "mu_0": 0.1, "mu": 0.05, "C_M": 1.0, "C_R": 1.0, "C_rep": 1.0, "C_D": 1.0}` plus: G `v: 0.0001`; IG
  `eta: 0.15`; W `sigma: 0.03`; Ga `gamma_beta: 20.0`; RF `bound_method: bound` and Ca `v: 0.0001`, Ho
  `support: 0.05`, Be `v: 0.0001, support: 0.05`, Ch `cgf: 0.3, s_chernoff: 5.0`; then `changes`. All nine are valid.
- `_write_input_hdf5(data, path)`: `str`, `bool`, `int`, `float` → attribute; `H` list → `int64` attribute; `dict`,
  and a list that holds strings at any depth → JSON text attribute; other lists → `float64` dataset (`None` → NaN).

**A1 (reader)**
- `test_read_input_yaml_and_json_agree`: the example as `.yaml` and as `.json` (`json.dump` of the parsed YAML) give
  equal mappings.
- `test_read_input_hdf5_layout`: a file with `F` int attribute, `H` `[2, 3]` attribute, `mu` dataset with a NaN, a
  `str` attribute, `gurobi_params` JSON text, `fast` bool, `bound_method` as a string dataset `[["", "cantelli"]]`
  (`dtype=h5py.string_dtype()`). Read back: `F == 3` and `type(F) is int`; `H == [2, 3]`; NaN kept; `bound_method ==
  [["", "cantelli"]]`; `gurobi_params == {"Threads": 4}`; `fast is False`.
- `test_read_input_rejects_extension` (`.txt`, "Unsupported input file type"), `test_read_input_rejects_non_mapping`
  (YAML `- 1`, `TypeError`), `test_read_input_rejects_hdf5_group` ("must be flat").

**A2 (keys)**
- `test_rejected_keys`: loop over the 21 keys; `_example(**{key: 1})`; message contains f"key '{key}' is not accepted"
  and the canonical key or phrase: `maintenance_type` repair_model, `tail_bound` and `method` bound_method,
  `formulation` reliability_impl, `beta` and `alpha_inc` gamma_beta, `b_inc` support, `b_0` support_0, `b_new`
  support_new, `n_workers` and `warm_start` sweep, `C_S` C_D, `alpha` tau, `xi` and `repair_rho` rho, `mu_new`
  replacement_mu, `v_new` replacement_v, `gamma_beta_bound`, `gamma_calibration_method`, `depot_capacity`, `C_P`
  "Remove it".
- `test_unknown_key` (`colour`: "unknown key", message lists `bound_method`); `test_missing_required_keys` (loop over
  the 13 keys: "missing required key"); `test_key_check_runs_before_arrays` (`mu="abc"` and `xi=0.5`: error names
  `xi`); `test_input_must_be_a_mapping` (`load_config([1])`: `TypeError`).

**A5 (parse, shapes, `null`)**
- `test_example_loads`: no `UserWarning`; `F, M, L, H1, H2, T == 3, 2, 2, 2, 3, 5`; `H == [2, 3]`; `mu.shape ==
  (3, 2, 2, 3)`, `mu_trans.shape == (3, 2, 2, 2)`; `bound_method[0, 0] == ""`; `reliability_impl[0, 1] == "exact"`;
  `v_0[0, 1]` NaN; `cgf_0[2, 1] == 5.0 * 0.14`; `anchor_v_0[2, 0] == 0.0002`; `tangent_ref[0, 0] == 0.4`, NaN elsewhere;
  `v_max_user[0, 0]` NaN and `[1, 0] == 0.05` (Q2); `C_M == 1.0`; `C_R == [[2.0, 1.5]] * 3`; `pwl_points == 6`;
  `options` = `RUN_OPTION_DEFAULTS` updated with the example's nine run options.
- `test_example_three_formats`: YAML, JSON and HDF5 (`_write_input_hdf5`) give equal `FleetConfig` fields
  (`np.testing.assert_array_equal`, NaN equal).
- `test_profile_shapes`: on the example, `ref = 0.040 + 0.0001 * np.arange(36).reshape(3, 2, 2, 3)`; `mu` given as
  `ref[0, 0, :, 0]` (M,), `ref[0, :, :, 0]` (L, M), `ref[0]` (L, M, H2), `ref[..., 0]` (F, L, M), `ref` → `cfg.mu`
  equals the numpy broadcast of each; shape `(4,)` → error "'mu' shape".
- `test_cell_value_shapes`: `tau` as `1.0`, `[1.0, 1.0]`, `[[1.0, 1.0]] * 3` → `(3, 2)`; `[1.0] * 3` → error "'tau'
  shape". `test_cost_shapes`: `C_R` scalar, `(L,)`, `(F, L)`; `C_M=[1.0, 1.0]` → error "'C_M' must be a number".
- `test_gamma_beta_shapes`: `_small("gamma")`, `gamma_beta` as `20.0`, `[20.0]`, `[[20.0], [20.0]]`, `(2, 1, 1)`,
  `(2, 1, 1, 2)` → shape `(2, 1, 1, 2)`; `gamma_beta_trans=20.0` → `(2, 1, 1, 1)`.
- `test_null_where_key_applies` (`tau=[[1.0, None], [1.0, 1.0], [1.0, 1.0]]`: "'tau' is null at cell \(i=0, l=1\)");
  `test_value_where_key_does_not_apply_warns` (`sigma=[[0.03, None], [0.03, None], [None, None]]`: warns "'sigma' has
  values at 1 cell", `sigma[0, 0]` NaN); `test_broadcast_value_does_not_warn` (`sigma=0.03`: no warning, `sigma[1, 0]
  == 0.03`, NaN elsewhere); `test_profile_value_where_key_does_not_apply_warns` (`v` of shape `(F, L, M)` with values
  at component 1: warns "'v' has values").
- `test_non_finite_and_non_numeric` (`rho=float("inf")`: "finite"; `rho="high"`: "'rho' must be").
- `test_selector_defaults` (`_small("gaussian")`: `repair_model` `"ardinf"`, `encoding` `"bigm"`,
  `reliability_impl` `"exact"`; `_small("rainflow")` without `bound_method`: `"cantelli"`).
- `test_selector_values`: errors for `model="weibull"`, `repair_model="ARD1"`, `encoding="big"`,
  `penalty_type="l2"`; `bound_method` `(F, L)` with `"foo"` at a G cell; `reliability_impl` `(F, L)` with `"lp"` at the
  IG cell.
- `test_component_names` (the example without the key: `["Component 1", "Component 2"]`; wrong length, `""`, `5` →
  errors).
- `test_cell_type_required_keys`: loop over (G, `v`), (IG, `eta`), (W, `sigma`), (Ga, `gamma_beta`), (Ho, `support`),
  (Be, `support`), (Ch, `cgf`), (Ch, `s_chernoff`): `_small` without the key → "missing required key '<key>'".
- `test_transitory_needs_operating` (`_small("inverse_gaussian", v_trans=0.0001)`: "'v_trans' needs").
- `test_sizes`: on `_small("gaussian")` errors for `M=2` ("'F' must be > 'M'"), `F=0`, `F=2.5`, `F=True`, `H=0`,
  `H=[1]`, `H=[1, 0]`, `H=1.5`; `H=2.0` accepted. On the example, `L=3` → error "'L'" (with a string `model`, a given
  `L` is used).
- `test_run_options`: `_small("gaussian")` gives `RUN_OPTION_DEFAULTS`; errors for `objective_mode="avg"`,
  `objective_mode="evaluation_total"` without `evaluation_horizon`, `evaluation_horizon=1` (`= H1`) and `2.5`,
  `mip_gap=-0.1`, `time_limit=0`, `progress_interval_seconds=0`, `relaxation_time_limit=0`, `gurobi_params=[1]`,
  `validation="x"`, `fast="yes"`, `relaxation_warm_start=1`, `verbose=2`, `allow_replacement="no"`,
  `replacement_as_new=0`.
- `test_cost_values`: errors for `C_M=-1.0`, `C_D=-1.0`, `C_D=float("inf")`, `C_R` negative at one cell.

**A6 (state, derived fields)** — on `_small` unless stated:
- `test_tau_epsilon_rho`: errors for `tau=0.0`, `epsilon=0.0`, `epsilon=1.0`, G `epsilon=0.5` ("must be < 0.5"),
  `rho=0.0`, `rho=1.5`; accepted: Ca `epsilon=0.6`, `rho=1.0`.
- `test_mu_0_range` (`mu_0=1.0`, `mu_0=-0.1`: errors).
- `test_nonnegative_values`: errors for Be `support_0=-0.01`, Ho `R_0=-0.01`, G `replacement_v=-0.0001`; the last is
  accepted with `allow_replacement=False`.
- `test_tangent_ref_and_pwl_points`: G `reliability_impl="tangent", tangent_ref=1.0` → error; `tangent_ref=1.0` with
  the form `exact` → accepted; `pwl_points=0` → error.
- `test_anchors`, all with `repair_model="ard1"`: G defaults `anchor_mu_0 == mu_0`, `anchor_v_0 == v_0`; errors for
  `anchor_mu_0=0.2`, `anchor_mu_0=-0.1`, G `v_0=0.0001, anchor_v_0=0.001`, Ho `anchor_R_0=0.01` (`R_0 = 0`), Ch
  `anchor_K_0=0.6` (`K_0 = 0.5`). Without `repair_model` (default `ardinf`), `anchor_mu_0=0.2` → accepted.
- `test_replacement_as_new`: G `replacement_as_new=True, replacement_mu=0.02` → error; Ch `replacement_as_new=True` →
  accepted and `cgf_new == 0`; with `cgf_new=0.1` → error.
- `test_quadratic_penalty`: G `penalty_type="quadratic", reliability_impl="tangent"` → error; with `C_D=0.0` →
  accepted; IG with the same two keys → accepted.
- `test_derived_fields` (example; `isclose`, `rtol=1e-12`): `range_v[0, 0] = 0.3696115094681951`, `range_v[2, 0] =
  0.20048057428495922`, `range_K[2, 1] = 2.697414907005954`, `range_R` all NaN, `range_v[0, 1]` NaN, `range_mu ==
  tau`; `support_const[2, 0] == 0.05`, NaN elsewhere; with `support_trans=[[[0.06, 0.06], [0.06, 0.06]], [[None, None],
  [None, None]]]` → `0.06`; plus `support_new=[[None, None], [None, None], [0.08, None]]` → `0.08`; plus
  `allow_replacement=False` → `0.06`.

**A7 (increments)**
- `test_increment_mean_range` (`mu=-0.01`, `mu=1.0`: errors; IG `mu=0.0`: accepted).
- `test_zero_mean_needs_zero_spread`: G `mu=0.0` (with `v=0.0001`) → error "'v' must be 0 where"; G `mu=0.0, v=0.0` →
  accepted; Ho `mu=0.0` → error 'support'; Ch `mu=0.0` → error 'cgf'.
- `test_negative_spread` (Ca `v=-0.0001`, Ho `support=-0.01`, Ch `cgf=-0.1`: errors).
- `test_lookup_pairs_transitory_mean_with_operating_variance`: G `mu_trans=0.0` → error "'v' must be 0 where" (day 1
  takes `mu` from `mu_trans` and `v` from the operating profile).
- `test_gaussian_three_sigma_increments` (G `mu=0.02`: error 'v'); `test_bernstein_popoviciu_increments` (Be
  `v=0.001`: error 'v'); `test_chernoff_cgf_increments` (Ch `cgf=0.2`: error 'cgf').
- `test_tolerance` (Ch `mu_0=0.14, cgf_0=0.7`: accepted, although `5*0.14 = 0.7000000000000001`).

**A8 (models)**
- G: `mu_0=0.02, v_0=0.0001` → error 'mu_0'; `replacement_mu=0.01, replacement_v=0.0001` → error 'replacement_mu',
  accepted with `allow_replacement=False`.
- `eta=0.0` (IG), `sigma=0.0` (W) → errors; W `v_0=0.0` accepted.
- Ga: errors for `gamma_beta=0.0`, `gamma_beta_trans=float("inf")`, `gamma_beta_0=0.0`, `gamma_beta_new=-1.0`; `mu_0=0.1`
  without `gamma_beta_0` accepted.
- `v_max_user`: errors for `0.0`, G `v_0=0.0001, v_max_user=0.00005`, G `replacement_mu=0.05, replacement_v=0.0001,
  v_max_user=0.00005` (`replacement_mu` keeps the three-sigma row true); the last accepted with
  `allow_replacement=False`.
- Be: `v_0=0.0001` (with `support_0 = 0`) → error 'v_0', accepted with `support_0=0.05`; `replacement_v=0.0001` →
  error, accepted with `support_new=0.05`; `support=1.0` → error 'tau'; with `support=0.5, reliability_impl="tangent"`:
  `tangent_ref=0.6` → error (`0.6 > 0.48215`), `0.4` → accepted; with `support=0.5, reliability_impl="pwl"`:
  `pwl_points=1` → error (`0.5 > 0.48215`), `2` → accepted.
- Ch: errors for `s_chernoff=2.0`, `cgf_0=0.4`, `repair_model="ard1", anchor_mu_0=0.1, anchor_K_0=0.4`,
  `replacement_mu=0.02, cgf_new=0.0`.

**A9 (natural ranges, replacement state)**
- Ranges: errors for W `v_0=0.7` (`V_max = 0.60887`), Ca `v_0=0.2` (`0.11111`), Ho `R_0=1.0` (`0.86859`), Ch
  `cgf_0=3.0` (`2.69741`), G `replacement_mu=1.5`; the last accepted with `allow_replacement=False`.
- Rows: errors for G `replacement_mu=0.9, replacement_v=0.01` (cap `0.0060887`; accepted with `0.005`); G
  `reliability_impl="tangent", tangent_ref=0.4, replacement_mu=0.9, replacement_v=0.0001` (cap `-0.14613`; accepted in
  the form `exact`); W `reliability_impl="pwl", pwl_points=6, replacement_mu=0.95, replacement_v=0.001` (cap
  `0.00084566`; accepted with `0.0005`); Mk `replacement_mu=0.2` (accepted with `0.05`); Ca `replacement_mu=0.9,
  replacement_v=0.002` (cap `0.0011111`); Ho `replacement_mu=0.9, R_new=0.01` (cap `0.0086859`); Be
  `replacement_mu=0.97` (side row `0.96162`).
- `allow_replacement=False` with Mk `replacement_mu=0.2` → accepted.

**Helpers of `tests/test_io.py`**
- `_result(solution=True)`: all 51 keys, `F=2, M=1, L=2, H1=1, H2=2, T=3, H=[1, 2]`; `model` an `(2, 2)` object array
  of `"gaussian"`, `"rainflow"`; `component_names=["Wheelset", "Brake pad"]`; `v_0`, `v` with NaN at component 1;
  `R`, `K`, `g_*` all NaN; `bound_method=[[None, "cantelli"], [None, "cantelli"]]`; `repair_model="ardinf"`;
  `encoding=[["bigm", "indicator"], ["bigm", "bigm"]]`; `evaluation_horizon`, `projected_evaluation_cost`,
  `evaluation_objective` `None`; `step_costs` length 3; `component_costs` with `(2, 2)` arrays; `performance` with a
  `None` entry and an `np.float64`; `optimization_progress` one record with `None`; `warm_start` and
  `relaxation_warm_start` mappings as in spec 1816-1849; an extra key `"extra_key": 1`. With `solution=False`: the
  solution arrays, the cost report, `objective`, `mip_gap`, `bound` are `None`.
- `_stub_base(result, calls)`: `mock.patch.dict(sys.modules, {"fleet_management.degradation_model": package})`, where
  `package` is a `types.ModuleType` whose attribute `base` has `solve_mixed(cfg, *, warm_start=None,
  warm_start_mode="start")` that appends its arguments to `calls` and returns `result`.

**B1 (solve path)** — the two tests that write a file pass once B2 is done.
- `test_results_path_checks`: `_check_results_path(None) is None`; `"out"` → `out.yaml`; `.txt` → `ValueError`;
  missing folder → `FileNotFoundError`; `mock.patch("fleet_management.solver.os.access", return_value=False)` →
  `PermissionError`.
- `test_solve_checks_input_file` (missing → `FileNotFoundError`; existing `in.txt` → `ValueError`).
- `test_solve_checks_results_path_before_reading_input`: input `in.yaml` holding `"{{{"`, `results_path` `r.txt` →
  `ValueError` that names the results file type.
- `test_solve_mapping_rejects_warm_start_mode` (`solve_mapping({}, warm_start_mode="warm")`: "warm_start_mode").
- `test_solve_mapping_checks_input_without_gurobi` (`solve_mapping({"F": 2})`: `ValueError`; passes once A2 is done).
- `test_solve_mapping_wiring`: patch `fleet_management.solver.load_config` to return a sentinel; `_stub_base`;
  `solve_mapping({"F": 2}, tmp_path / "r.json", warm_start={"x": None}, warm_start_mode="hint")` returns the same
  object as the stub, writes `r.json`, and `calls == [(sentinel, {"x": None}, "hint")]`.
- `test_solve_writes_or_not` (same stubs; a YAML input file; `results_path=tmp_path / "out"` writes `out.yaml`;
  `results_path=None` writes nothing); `test_solve_default_results_path` (`inspect.signature`: `"output.yaml"`);
  `test_read_input_alias` (`solver._read_input is config.read_input`).

**B2 (YAML, JSON)**
- `test_yaml_round_trip`: keys of the file `==` the keys of `RESULT_KEYS` whose value is not `None`, in that order;
  `v` has `None` where NaN; `model`, selectors, `H` as lists; `performance` keeps its `None` entry; no `extra_key`.
- `test_json_round_trip`: the same with `json.load`; the file text has no `NaN` and no `Infinity`.
- `test_no_solution_text_files`: `objective` present and `None`; no `x`, `mu`, `step_costs`, `mip_gap`.
- `test_infinite_float_written_as_null` (`bound=float("inf")` → `None`, Q4); `test_save_results_rejects_extension`
  (`.csv`).

**B3 (HDF5)**
- `test_hdf5_round_trip`: the set of datasets `==` the 17 dataset keys, all `float64`; NaN kept in `v`; attributes
  `status`, `objective`, `u`, `H == [1, 2]`; `repair_model == "ardinf"` (attribute); `bound_method` JSON text →
  `[[None, "cantelli"], [None, "cantelli"]]`; `model`, `component_names`, `component_costs`, `performance`,
  `optimization_progress`, `warm_start`, `relaxation_warm_start` parse as JSON; no `evaluation_horizon`, no
  `extra_key`.
- `test_no_solution_hdf5`: `status` present; no `objective` attribute; no `x` dataset.

## 4. Deferred

- **WP6, checks that need a builder constant** (spec 557, the replacement state on a replacement day), only when
  `allow_replacement` is true:
  - IG cells: `replacement_mu <= mu_bar = tau / E[Z | Z > q]` (spec Eq. `ig_row`, 1078-1106).
  - Gamma cells: `replacement_mu <= A_1(beta* tau, eps) / beta*` (spec Eq. `gm_row`, `gm_betastar`, 1258-1301);
    a cell in which no term has a rate is already covered by the natural range `replacement_mu <= tau`.
  - Place: a new block at the end of `_check_ranges`, same `_require` style. `config.py` must still import without
    Gurobi, so the cap functions must live in a module without a module-level `gurobipy` import.
- **G44 (WP5)** `base.py:1202-1247`: `resolve_run_options` defaults (`mip_gap` 0.12, `time_limit` None,
  `replacement_as_new` True) contradict spec 417, 477-478; read `cfg.options` and the attributes instead.
- **G45 (WP5)** `base.py:1250-1279`: `resolve_costs` reads `cfg.costs` and defaults `C_rep` (spec 468: required); read
  `cfg.C_M`, `cfg.C_R`, `cfg.C_rep`, `cfg.C_D`.
- **G46 (WP5)** the phase lookup accessor, Eq. `c_phase` (it may reuse `config._lookup`, Q7); the result dictionary:
  `bound_method` `None` at cells that are not rainflow (`cfg.bound_method` holds `''`), one `str` per selector when all
  cells agree, `model` as names, `H` from `cfg.H`, no key outside the 51 of `RESULT_KEYS`. `base.solve_mixed` may also
  check `warm_start_mode`.
- **G47 (WP7)** callers of `solver._read_input`: `validation/validator.py:13, 108, 449, 631, 899`,
  `validation/dashboard_rainflow.py:468-470`; `utils/mixed_plotter.py:768-842` has its own reader (spec 2664: one
  reader); `gamma_stochastic_validator.py:81` reads `cfg.raw` (gone; use `cfg.options["validation"]`);
  `greedy_warm_start.py:120, 140` and `gamma_tail_validator.py:189, 214` read `gamma_beta_bound`;
  `horizon_sweep.py:416-422` writes temporary inputs and calls `solve` (spec 1933: `solve_mapping`, no file).
- **G48 (WP6)** `rainflow.py:702-713` calls `solve(load_config(...))`.
- **WP8 and the README**: scripts in `examples/regression/` and `experiments/` that use removed keys or
  `solver._save_results`.

## 5. Questions

None blocks a step: each step follows the recommendation, and a different answer changes only the step named.

- **Q1. Where the input reader lives.** The spec (183) puts it in `config.py`; the brief puts it in `solver.py`, and
  the interface says `config.py` imports only the standard library, numpy and scipy. Options: `config.py` with `yaml`
  and `h5py` imported inside the reader; `solver.py`. Recommendation: `config.py` (spec authority; the interface rule
  still holds at module level), with `solver._read_input` kept as an alias for the old callers. Step: A1 (it moves to
  Task B if the answer is `solver.py`).
- **Q2. `null` at a cell where an optional key without default applies.** Spec 352 says `null` there is an error; the
  example (2756) writes `v_max_user: null` at its G and Be cells, and spec 2713 says the example passes every check.
  Options: `null` means "not given at this cell" for `v_max_user`, `gamma_beta_0`, `gamma_beta_new`; or the example
  is wrong. Recommendation: the first, and spec 352 names these three keys as the exception. Step: A4.
- **Q3. Rounding in non-strict checks.** The spec has no tolerance. Example: `cgf_0 = 0.7`, `s = 5`, `mu_0 = 0.14`
  fails `K_0 >= s*mu_0` without one, because `5*0.14 = 0.7000000000000001` in floating point. Options: exact
  comparisons; a tolerance `1e-12 * max(1, |b|)` on `a <= b`. Recommendation: the tolerance, on non-strict
  inequalities only; the spec gets one sentence. Steps: A6 to A9.
- **Q4. Infinite numbers in YAML and JSON results.** The spec turns only NaN into `null`; JSON cannot hold infinity.
  Recommendation: every non-finite number becomes `null` in YAML and JSON; HDF5 keeps it. Step: B2.
- **Q5. Result keys outside the spec's table** (for example an entry that a builder's `extract` hook adds).
  Options: write only the 51 keys (the decided "fixed key list" of `main`); write the others too. Recommendation: the
  51 keys; a builder that needs a new key adds it to the spec's table first. Step: B2, B3.
- **Q6. Case of selector values.** `main` lower-cases `objective_mode`; the spec is silent. Recommendation: exact
  match, like key names (`ARD1` then fails with the list of valid values). Steps: A3, A4.
- **Q7. The phase lookup in the checks.** The interface says `config.py` checks profiles entry by entry. That is not
  enough for the checks that pair two keys (G three-sigma, Popoviciu, Chernoff `cgf >= s*mu`, zero spread where
  `mu = 0`) when only one key of the pair has a transitory profile: spec 562-582 quantify over the days after the
  lookup. Recommendation: a private `_lookup` in `config.py` for these checks; WP5 keeps its accessor or imports
  `_lookup`. No field of the interface changes. Step: A7.
- **Q8. `objective` in HDF5 when it is `None`.** An HDF5 attribute cannot be `null`. Recommendation: not stored (spec
  2684: "a key whose value is absent is not stored"); readers treat a missing key as `null` (spec 1867). Step: B3.
- **Q9. Integers written as floats** (`F: 3.0`, common in HDF5 files written by numeric tools). Recommendation:
  accept a float with an integer value and store an `int`; refuse `2.5` and `True`. Step: A3.
- **Q10. "No file for a failed solve"** (interface, section "Solve path"). The spec has no such case: `solve` writes
  the file whenever `results_path` is not `None`, also without a solution (status and `objective: null`). The plan
  follows the spec. No step depends on it.

## 7. Review (29 Oct 2026)

Read: the full diff of `config.py` and `solver.py`, the two new test files, both progress files, the four passages
of the spec on `lore` that record the user's answers (lines 352-357, 525, 1865-1871, 2671-2690), the last section of
`wp4_wp5_interface.md`. Line numbers below are those of the corrected files in the worktree.

**State found.** All 13 steps were done once: no duplicate definition, no leftover of the old code
(`H_prof`, `cfg.costs`, `alias(`, legacy dispatch, `model_assignment`), no function defined and never used. The
resume after the credit limit left nothing half done. Before the review: 55 + 18 tests passed.

**Defects corrected**
- `config.py:357-376` (`_float_array`): a numeric string (`"0.5"`) and a flag (`true`) passed as numbers in cell
  values and profiles, because `np.asarray(..., dtype=float)` converts both; the spec's types are numbers. New helper
  `_holds_text_or_flag`; `_float_array` refuses a `str`, `bytes` or `bool` anywhere in the value, with the message
  "'<key>' must be a number or a regular array of numbers.".
- `config.py:544-552, 573-600, 631-636, 798, 326, 893` (non-strict inequalities against a constant): `rho <= 1`,
  `mu_0 >= 0`, every `>= 0` of initial, replacement and increment values, `tangent_ref >= 0`, `anchor >= 0`,
  `C_R, C_rep, C_M, C_D >= 0`, the lower end of the natural ranges, `mip_gap >= 0` were exact. The spec on `lore`
  (line 525) gives every non-strict inequality the tolerance `1e-12·max(1, |a|, |b|)`. All now use `_le`. This
  follows the plan's narrower rule ("input against a computed value"), which the user's answer widened; it is not an
  executor error.
- `config.py:646, 697, 701` (consequence of the line above): a tolerated variance such as `-1e-13` gave
  `sqrt(-1e-13) = NaN`, so the three-sigma checks failed with a false error. The square root now takes
  `max(v, 0)`.
- `config.py:326`: a `mip_gap` within the tolerance below 0 is stored as `0.0`, because Gurobi refuses a negative
  `MIPGap`. No other tolerated value is changed: a constant such as `v_0 = -1e-13` reaches the model, where Gurobi's
  feasibility tolerance (1e-6) absorbs it.
- `tests/test_config.py:440` (`_fails`): it matched only the key, so a range or replacement-row test could pass on
  an earlier error that names the same key. It now takes the rule text; `test_natural_ranges` and
  `test_replacement_rows` match the reason.
- New test lines: strings and flags refused (`test_non_finite_and_non_numeric`); tolerance at 0 and at 1, and the
  `mip_gap` clamp (`test_tolerance`); `replacement_as_new` with `allow_replacement=False` (`test_replacement_as_new`).

**Points raised by the executors**
1. Strings and flags as numbers: refused (above).
2. Exact comparisons with 0: they did not match the spec text; corrected (above). Equalities (`== 0` under
   `replacement_as_new`, zero spread where `mu = 0`) and strict inequalities stay exact, as the spec says.
3. `replacement_as_new` check with `allow_replacement` false: kept. Spec line 551 makes the flag the only condition
   ("the check does not run when the flag is false"); a test now records it.
4. `_first_cell` (`config.py`) and `_result_format` (`solver.py`): kept. Each removes a repetition (the first cell
   of a mask; the one message for an unsupported results file), and both follow the files' helper style.
5. `base.solve_mixed` still has the signature `solve_mixed(cfg, **overrides)`: WP5. `solve_mapping` calls it with
   `warm_start=` and `warm_start_mode=`, as the interface fixes; the tests stub it. Until WP5 changes it, an end-to-end
   `solve` fails at that call. `base.py` not touched.

**Package import and pytest collection.** `import fleet_management` works on the branch today, also without
gurobipy (checked with `python` on this computer: the package, `config` and `solver` import; `gurobipy` and
`degradation_model.base` stay unloaded), and with a stub gurobipy. So `tests/test_config.py` and `tests/test_io.py`
are collected by real pytest once the package is installed (`pip install -e ".[dev]"`, as for the existing tests).
No change is needed in the four files. Risk for WP7: `__init__.py` imports the sweeps, the plotter and the validators;
if one of them fails to import later, every test module in `tests/` fails to collect.

**Line endings.** No file has mixed endings: `config.py` CRLF throughout; `solver.py`, `tests/test_config.py`,
`tests/test_io.py` LF throughout (git warns that it will convert `solver.py` to CRLF).

**Tests.** `tests/test_config.py`: 55 passed; `tests/test_io.py`: 18 passed (runner `run_tests.py`, 29 Oct 2026). The
test count is unchanged; the new cases extend existing tests.

**Left for other packages**
- WP5: the signature of `base.solve_mixed`; `bound_method` as `None` at cells that are not rainflow in the result
  (`cfg.bound_method` holds `''`); read run options and costs from `cfg` (section 4).
- WP6: the IG and Gamma replacement-state checks (section 4), at the end of `_check_ranges`.
- WP7: the callers of `solver._read_input` and of `cfg.raw` (section 4, G47); keep `__init__.py` importable without
  gurobipy.
