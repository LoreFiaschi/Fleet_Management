# WP5 plan: the shared model (`base.py`)

Status (29 Oct 2026): done. All six sections written and checked once for consistency. Next: Sonnet executors carry
out Task A and Task B of section 3; then the planner reviews their work against this plan and the spec.

## 0. Scope and rules for the executors

- **Where.** Code in the worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge`, branch
  `merge/lore-into-main`. Authority: `spec/spec.tex` of that worktree ("the spec" below; line numbers refer to it).
  Interface with WP4 (input and output layer): `merge/wp4_wp5_interface.md` in the `lore` checkout. Read it first.
- **Files.** Task A edits only `src/fleet_management/degradation_model/base.py`. Task B creates only
  `tests/test_base.py`. Nothing else in the repository changes. No git command that changes state; no commit.
- **Parts of `base.py` that stay as they are.**
  - The Gamma block: `_add_binary_scaled_product`, `_gamma_reachable_upper_bounds`, `GammaCellBuilder` and the line
    `register_cell_builder("gamma", GammaCellBuilder())` (today lines 164-1187). WP6 (cell builders) replaces it.
  - The warm-start functions `apply_binary_warm_start`, `complete_binary_warm_start`, `apply_relaxation_warm_start`.
    WP7 (tools) rewrites them. Only one forced edit is allowed: step A11 removes the lines that set `ctx.idle`.
- **`FleetConfig`.** Read it as the interface file defines it (attribute = input key, canonical shape, `NaN` where a
  key does not apply, `''` in a selector where it does not apply). Do not read `cfg.costs`, `cfg.models` or any field
  the interface does not list.
- **Day index.** In code, `k = 0, ..., T-1` is the day index, and spec day = `k + 1`. Index `k = -1` stands for the
  spec's day 0 (initial values and starting anchors): code uses the constants there. The loop row compares index
  `T-1` with index `H1-1`.
- **Code style.** Write like the existing `base.py`: section banners `# ====...`, a one-line docstring summary plus a
  short paragraph where needed, the unpacking idiom `md, T = ctx.model, ctx.T`, `float(...)` around every parameter
  read from an array, plain `for` loops over `i`, `l`, `k`, `gp.quicksum`, row names as f-strings that end with the
  indices (`f"{q}_carry_{i}_{l}_{k}"`). Few comments; a comment cites the spec equation label (`# Eq. a_bigm_carry`).
  Every variable and row name is unique (the relaxation start maps variables by name).
- **Safety rules.** No installs. Do not delete any file. Temporary files only in the scratchpad
  `C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp5\`.
  Never put the user's name or email address in any request. Do not spawn sub-agents.
- **Local tools** (already in that scratchpad folder; run them with `python`, from that folder):
  - `load_base.py`: loads `base.py` through a stub `gurobipy` (`stubs/gurobipy/`); fails on a module-level error.
  - `names_check.py`: lists global names that a function uses but nobody defines (a substitute for pyflakes).
  - `local_checks.py`: checks the pure functions of section 4 against hand-computed values; prints `failures: 0`.
  - `hand_case.py`: the exact day maps of the hand-worked test case (prints the expected arrays of section 4).
  - If the folder is gone, rebuild the stub from section 4.5 and skip the other scripts; say so in the report.

## 1. Gaps

Current code: `base.py` of the worktree (2180 lines, commit 5f29660). "Pkg" names the package that closes the gap;
WP5 plans only its own rows. Lines of `base.py` are approximate after the first edit.

| Gap | Code (`base.py`) | Spec | Pkg |
|---|---|---|---|
| Context lacks `opts`, `encoding_of`, initial, replacement and starting-anchor values of every tracked quantity, natural ranges, per-cell `tangent_ref` | 55-109 | 251; 385, 393-396, 402-416 | WP5 |
| Context carries rainflow-only fields (`Le`, `ln_eps`, `latch_of`, `track_v_of`, `support_max_of`, `s_chernoff`, `v_inc`, `w2_inc`, `cgf_inc`) and a scalar `tangent_ref`; no anchor `gK` | 87-107 | 247 (builders store their own arrays); 623-625 | WP5 |
| Variable `idle` is built; the spec has no such variable (`idle` is a computed output) | 70, 1627 | 607-632, 1719 | WP5 |
| Selector `nb` built at every non-Gamma cell; spec builds `n` only at cells with `encoding = indicator` | 71, 1628-1635 | 613-614 | WP5 |
| `z` built at every cell and day, with upper bound `tau` | 73, 1637, 1659 | 628-630, 1548-1550 | WP5 |
| `u` always built | 74, 1638 | 631-632, 1567-1569 | WP5 |
| Increment accessor `mu_inc(i, j0, l, k)`, one per profile, mission index `j0 = j-1`; spec: one accessor `(i, j, l, k)` after Eq. `c_phase`, `j = 0` the depot (the lookup arithmetic itself is correct) | 83, 1291-1304 | 251, 431-440 | WP5 |
| `resolve_run_options`: own defaults (`mip_gap` 0.12, `time_limit` None, `replacement_as_new` True), keyword overrides, and `allow_replacement`, `reliability_impl`, `pwl_points`, `tangent_ref` read as run options | 1202-1247 | 475-487, 375, 385-386, 417-418 | WP5 (defaults filled by WP4) |
| `resolve_costs`: `cfg.costs`, `C_M`/`C_R`/`C_rep` of shape `(L,)`, default `C_rep`, alias `C_S` | 1250-1279 | 466-470, 511 | WP5 |
| Progress callback raises an interval below 0.1 s to 0.1 s | 1326 | 483, 1811 | WP5 |
| Depot row is `idle + m + r = x_i0k` per cell inside each builder; spec: `m + r <= x_i0k` (Eq. `a_depot`), built once in step 3 | 1697-1721 | 714-722, 257 | WP5 |
| Selector row `nb + m + r = 1` at every non-Gamma cell; spec: Eq. `a_ind_sum` at indicator cells only | 1722-1732 | 794-799 | WP5 |
| Damage-penalty rows: `inf_norm` only, always built, inside `add_base_constraints` | 1685-1694 | 1567-1584 (Eqs. `a_u_inf`, `a_u_quad`) | WP5 |
| Objective: `C_M[l]·m` per repaired component; `C_R[l]`, `C_rep[l]`; `C_D·u` always | 1735-1810 | 647-685 (Eqs. `a_daycost`, `a_phases`, `a_objective`) | WP5 |
| No generic dynamics rows (big-M and indicator), anchor rows, replacement rows, loop rows; only rainflow's own indicator equalities exist | -- (rainflow.py:191-338, 604-641) | 770-849, 1587-1602 | WP5 (helpers); WP6 (use) |
| No generic repair-cost rows (Eq. `a_z`, lower rows only, `z = 0` on replacement days) | -- (rainflow.py:277-286) | 1543-1560 | WP5 |
| No generic `tangent` and `pwl` forms; rainflow's own versions clip `tangent_ref` and accept an unknown form | -- (rainflow.py:470-570) | 851-911 | WP5 |
| `_load_builders` imports `rainflow` only; spec: one module per model | 1978-1982 | 192, 260 | WP5 |
| `extract_solution`: keys `method`, `models`, `model_assignment`; `model` holds the Gurobi object | 1858-1869 | 1696, 1709 | WP5 |
| `extract_solution`: no `encoding`; `objective_mode` and `evaluation_horizon` only when a solution exists; `bound_method` not `null` at non-rainflow cells | 1855-1972 | 1712-1713 | WP5 |
| `extract_solution`: `v = 0` where untracked; no `R`, `K`, `g_mu`, `g_v`, `g_R`, `g_K`; `idle` read from a variable | 1887-1905 | 1719-1724 | WP5 |
| `extract_solution`: `z` at every cell; no Eq. `c_zpost` on transitory days of `operating_average` | 1901 | 1722, 1747-1755 | WP5 |
| `extract_solution`: `u` by `inf_norm` only | 1906-1912 | 1756-1761 (Eq. `c_u`) | WP5 |
| `extract_solution`: no solution gives no cost-report keys; with a solution, `step_costs` uses `C_M[l]·m` and `component_costs` has the old shapes | 1875-1971 | 1727-1733, 1744, 1762 | WP5 |
| `solve_mixed(cfg, **overrides)`; no `warm_start_mode` | 2007-2017 | 209-211; interface "Solve path" | WP5 |
| Report `warm_start` lacks `mode`; report `relaxation_warm_start` never has status `skipped` | 2021-2036 | 1822, 1845 | WP5 |
| `performance` comes from `gamma_utils/gamma_diagnostics.py` (outside the spec's layout) and carries `gamma_calibration_seconds`; result carries `gamma_formulation` | 2058-2152 | 194-198, 1773-1797 | WP5 |
| `requested_mip_gap`, `requested_time_limit_seconds` from the options, not from Gurobi after `gurobi_params` | 2070-2075 | 1795, 1889 | WP5 |
| Unused leftovers: `FleetModel.rainflow_cells`, `collapse_dict`, `AccumulatedDegradationModel`; module docstring describes the old objective | 1-37, 120, 1842, 2164-2180 | -- | WP5 |
| Warm start: `idle` start values (variable removed) | 1448-1452 | 2132-2138 | WP5 (forced edit only) |
| Warm-start modes `hint` and `both`; `source_horizon` as a pair | 1403-1464 | 2111-2147, 1823 | WP7 |
| Gamma block reads removed names (`add_maintenance_gating`, `ctx.mu_inc`, `ctx.mu_new`, `ctx.mu_0`, dict `impl_of`) and builds `z` at every cell | 744-1063 | 1192-1301 | WP6 |
| `rainflow.py` imports removed names (`add_base_constraints`, `add_maintenance_gating`, `cell_max`, `make_accessor`, `pick`, `resolve_costs`) | rainflow.py:25-38 | 1304-1541 | WP6 |
| Run-option defaults, cost shapes, natural ranges, `penalty_type` and form checks | -- | 475-590 | WP4 |

## 2. Helper API for WP6

This section is the contract between `base.py` and the cell builders of WP6. A tracked quantity is named by a key
`q`: `"mu"` (mean), `"v"` (variance), `"R"` (Hoeffding budget), `"K"` (Chernoff quantity).

### 2.1 The context `FleetModel` (spec line 251)

| Field | Content |
|---|---|
| `model`; `F`, `M`, `L`, `H1`, `H2`, `T` | the Gurobi model; the sizes |
| `x` | tupledict `(i, j, k)`, `j = 0` the depot, binary, always |
| `m_rep` | tupledict `(i, l, k)`, repair `x^m`, binary, always |
| `r_rep` | tupledict `(i, l, k)`, replacement `x^r`; `None` if `allow_replacement` is false |
| `mu_var` | tupledict `(i, l, k)`, the shared mean, bounds `[0, tau]` |
| `z_var` | tupledict over the keys `(i, l, k)` of cells with `C_R > 0` and days `k` in `cost_days(...)` only |
| `u_var` | the scalar `u`; `None` if `C_D == 0` |
| `model_of`, `repair_of`, `bound_of`, `impl_of`, `encoding_of` | `(F, L)` str arrays: `cfg.model`, `cfg.repair_model`, `cfg.bound_method`, `cfg.reliability_impl`, `cfg.encoding` |
| `tau`, `eps`, `rho`, `tangent_ref` | `(F, L)` float arrays from `cfg` (`eps` is `cfg.epsilon`); costs stay in `cfg` |
| `initial[q]` | `(F, L)`: `mu_0`, `v_0`, `R_0`, `cgf_0` for `q` = `mu`, `v`, `R`, `K` |
| `replacement[q]` | `(F, L)`: `replacement_mu`, `replacement_v`, `R_new`, `cgf_new` |
| `anchor_start[q]` | `(F, L)`: `anchor_mu_0`, `anchor_v_0`, `anchor_R_0`, `anchor_K_0` |
| `state_range[q]` | `(F, L)`: `range_mu`, `range_v`, `range_R`, `range_K` (the natural ranges `M_s`) |
| `allow_replacement`, `pwl_points`, `opts` | `cfg.allow_replacement`, `cfg.pwl_points`, the run options (dict) |
| `profiles` | `{key: (operating, transitory or None)}` for every profile key whose operating array is not `None` (`mu`, `v`, `support`, `cgf`, `gamma_beta`) |
| `state_vars[q]` | dict `(i, l, k) -> Var`; `state_vars["mu"] is mu_var`; the others are filled by `add_state_vars` |
| `anchor_vars[q]` | dict `(i, l, k) -> Var`, filled by `add_state_vars` at `ard1` cells |
| `nb` | dict `(i, l, k) -> Var`, the selector `n`, filled by `add_selector_vars` |
| `pwl_w` | dict `(i, l, k, p) -> Var`, `p = 0, ..., P-1`, filled by `add_pwl_vars` |
| `extras` | free dict for a builder's own data |

Methods: `cells_of(name)`, `all_cells()` (kept), and

- `increment(key, i, j, l, k) -> float`. The value of profile `key` for train `i`, activity `j`, component `l`, day
  index `k`, after the phase lookup Eq. `c_phase` (spec 431-438). `j = 0` (the depot) returns `0.0`. It uses
  `phase_day(k, H1, H2, transitory is not None)` and reads `array[i, l, j - 1, h]`. Called by `day_increment`, and by a
  builder that needs raw values (Gamma rates `gamma_beta`). Models: all.

### 2.2 Helpers for `prepare(ctx, cfg, cells, opts)` (create variables)

- **`add_state_vars(ctx, q, cells, *, anchored=True)`**. For every cell in `cells` and every `k`: creates the state of
  `q` in `[0, state_range[q][i, l]]` (not for `q = "mu"`, which is shared), and, where `repair_of[i, l] == "ard1"` and
  `anchored` is true, the anchor `g^q` in the same range. Spec: variable table 615-625, Eq. `a_anchor`, ranges 821-825.
  Models: every builder for `"mu"` (anchors); G, W (with `anchored=False`), RF Ca, RF Be for `"v"`; RF Ho for `"R"`;
  RF Ch for `"K"`.
- **`add_selector_vars(ctx, cells)`**. Creates the binary `n` for every cell of `cells` with `encoding_of == "indicator"`
  and every `k`. Spec: 613-614, Eq. `a_ind_sum`. Models: all.
- **`add_pwl_vars(ctx, cells)`**. If `pwl_points >= 2`: creates the binaries `w` for every cell of `cells` with
  `impl_of == "pwl"`, every `k` and every segment `p`. `cells` must hold only cells with a quadratic reliability row.
  Spec: 626-627, Eq. `a_pwl_sum`. Models: G, W, RF Ca, RF Ho, RF Be.
- **Convention, no helper**: `prepare` sets `ctx.model.Params.NonConvex = 2` when one of its cells has a quadratic row
  and `impl_of == "exact"` (spec 247, 879-881).

### 2.3 Helpers for `add_cell(ctx, i, l)` (add rows)

Call order in `add_cell`: `add_selector_rows`, then `add_quantity_rows` once per tracked quantity (`"mu"` first), then
the reliability rows, the optional `v_max_user` row, and `add_repair_cost_rows`.

- **`mission_count(ctx, i, k) -> LinExpr`**: `sum_{j=1}^{M} x[i, j, k]`. Models: W (`delta v = sigma^2 * mission_count`,
  Eq. `w_map_v`).
- **`day_increment(ctx, i, l, k, key, *, squared=False) -> LinExpr`**: `sum_{j=1}^{M} x[i, j, k] * p` with
  `p = ctx.increment(key, i, j, l, k)`, or `p^2` if `squared`. Terms with `p == 0` are left out; a value that is not
  finite raises `ValueError` naming the key and the cell. Spec: notation 297-299, 1361-1365. Models: all (`"mu"`); G and
  RF Ca, Be (`"v"`); RF Ho (`"support"`, `squared=True`); RF Ch (`"cgf"`).
- **`add_selector_rows(ctx, i, l)`**: at an indicator cell, `n + x^m + x^r = 1` for every `k` (no `x^r` without
  replacement); nothing at a big-M cell. Spec: Eq. `a_ind_sum`. Models: all.
- **`add_quantity_rows(ctx, i, l, q, increments, *, repair_keeps_state=False)`**. All generic rows of one tracked
  quantity of one cell, for every `k`: the dynamics in the cell's encoding, the replacement rows (only if replacement
  is allowed), the anchor rows (only at an `ard1` cell, not when `repair_keeps_state`), and the loop rows of the state
  and of its anchor. `increments[k]` is the day increment `delta s` (a list of `T` expressions, usually from
  `day_increment`). The repair factor is `a = repair_factor(q, rho)`; the repair map is `a*s + (1-a)*g` at an `ard1`
  cell and `a*s` at an `ardinf` cell (Eq. `a_psi`). `repair_keeps_state=True` is for the Wiener variance only: `a = 1`,
  no anchor, and in the big-M form one row `s_k >= s_{k-1} + delta s_k - M_s x^r_k` replaces the carry and repair rows.
  Spec: Eqs. `a_psi`, `a_anchor`, `a_bigm_carry`, `a_bigm_repair`, `a_bigm_repl`, `a_bigm_anchor`, `a_ind_state`,
  `a_ind_anchor`, `a_loop`, `w_vrow`; the model rows `g_mu_carry`...`g_anchor_keep`, `ig_*`, `w_*`, `gm_*`, `rf_*`.
  Models: all.
- **`add_repair_cost_rows(ctx, i, l)`**: for every key `(i, l, k)` of `z_var`,
  `z_k >= rho (mu_{k-1} - g^mu_{k-1}) - rho tau (1 - x^m_k)`, with `g^mu = 0` at an `ardinf` cell; nothing at a cell
  without `z`. Spec: Eq. `a_z`. Models: all.
- **`add_quadratic_reliability_rows(ctx, i, l, q, *, c2, c1, mu_side)`**. The row `chi_k <= f(mu_k)` with
  `chi = state_vars[q]`, `f(mu) = c2 (tau - mu)^2 + c1 (tau - mu)`, and the side row `mu_k <= mu_side`, for every `k`,
  in the form `impl_of[i, l]`: `exact`, `tangent` (at `tangent_ref * tau`) or `pwl` (`pwl_points` segments; one tangent
  at `tau/2` if `pwl_points == 1`). Another form raises `ValueError`. Spec: Eqs. `a_quad`, `a_tangent`, `a_pwl_sum`,
  `a_pwl_seg`, `a_pwl_cap`; table 866-873. Models and arguments: G and W (`"v"`, `1/kappa_G^2`, `0`, `tau`);
  RF Ca (`"v"`, `eps/(1-eps)`, `0`, `tau`); RF Ho (`"R"`, `2/kappa`, `0`, `tau`); RF Be (`"v"`, `1/(2 kappa)`,
  `-b/3`, `tau - c`).
- **`add_upper_cap_rows(ctx, i, l, q, cap, *, name)`**: `state_vars[q][i, l, k] <= cap` for every `k`. Spec and models:
  IG Eq. `ig_row` (`"mu"`, `mu_bar`); Ga Eq. `gm_row` (`"mu"`, `A_1/beta*`, or `tau` without rates); RF Mk Eq.
  `rf_markov` (`"mu"`, `eps tau`); RF Ch Eq. `rf_chernoff` (`"K"`, `s tau + ln eps`); the optional row
  `v <= v_max_user` (`"v"`; G, W, RF Ca, RF Be; spec 1016-1017, 1539-1540).

### 2.4 Pure functions (no Gurobi object; checked locally)

`phase_day(k, H1, H2, has_transitory) -> (bool, int)`; `cost_days(objective_mode, H1, T) -> range`;
`repair_factor(q, rho) -> float`; `collapse_selector(arr, *, empty_as_none=False)`;
`transitory_repair_cost(mu, g_mu, m, mu_0, anchor_mu_0, rho, ard1, H1) -> ndarray (F, L, H1)`;
`peak_damage(mu, penalty_type) -> float`; `cost_report(x, r, z, C_M, C_R, C_rep, C_D, u, H1, H2, evaluation_horizon)
-> dict`. Formulas in step A8.

### 2.5 What base does itself (builders do not call these)

`resolve_run_options(cfg)`, `build_context(cfg, opts)`, `build_objective(ctx, cfg, opts)`, `add_assignment_rows(ctx)`,
`add_depot_rows(ctx)`, `add_damage_penalty_rows(ctx, penalty_type)`, `build_fleet(cfg, opts)`,
`extract_solution(ctx, cfg)`, `solve_mixed(cfg, *, warm_start=None, warm_start_mode="start")`. The shared extraction
reads every `state_vars[q]` and `anchor_vars[q]`, so a builder needs an `extract` hook only for keys outside the
spec's result table. A hook must accept `out["x"] is None` (no solution).

## 3. Executor tasks

Two tasks for Sonnet executors. They edit different files and can run in parallel: Task B needs only the API of
section 2, not Task A's code. Each executor re-reads its file from disk before each edit.

**Checks after every step of Task A** (from the scratchpad folder of section 0):
`python -c "import ast; ast.parse(open(r'<worktree>\src\fleet_management\degradation_model\base.py', encoding='utf-8').read())"`,
then `python load_base.py`, `python names_check.py` (expected: 0 undefined names outside the Gamma block) and
`python local_checks.py` (expected: `failures: 0`; `MISSING` only for functions of later steps). The step's own check
is listed with the step.

### Task A: `base.py` (11 steps, in this order)

**A1. Context.** Replace the class `FleetModel` (section "Shared context").
- Add the module constant `TRACKED_QUANTITIES = ("mu", "v", "R", "K")`.
- Add, before the class, the pure function `phase_day(k: int, H1: int, H2: int, has_transitory: bool) -> tuple[bool,
  int]` (Eq. `c_phase` with code index `k` = spec day - 1): `if k < H1: return (True, k) if has_transitory else
  (False, k % H2)`; otherwise `return (False, (k - H1) % H2)`. The first item says whether the transitory profile is
  read; the second is the index on the profile's day axis.
- Fields, in this order (no default first, then with `field(default_factory=dict)`): `model`; `F`, `H1`, `H2`, `M`,
  `L`, `T`; `x`, `m_rep`, `r_rep`, `mu_var`, `z_var`, `u_var`; `model_of`, `repair_of`, `bound_of`, `impl_of`,
  `encoding_of`; `tau`, `eps`, `rho`, `tangent_ref`; `initial`, `replacement`, `anchor_start`,
  `state_range`; `allow_replacement`, `pwl_points`, `opts`, `profiles`; then `state_vars`, `anchor_vars`, `nb`,
  `pwl_w`, `extras`. One trailing comment per field, as in section 2.1. Remove every other field (`idle`, `mu_inc`,
  `mu_0`, `mu_new`, `v_var`, `gmu`, `gv`, `gR`, `R_var`, `K_var`, `v_0`, `v_new`, `s_chernoff`, `support_max_of`, `Le`,
  `ln_eps`, `track_v_of`, `latch_of`, `v_inc`, `w2_inc`, `cgf_inc`).
- Keep `cells_of` and `all_cells`; remove `rainflow_cells`.
- Add the method `increment(self, key: str, i: int, j: int, l: int, k: int) -> float`: `if j == 0: return 0.0`;
  `operating, transitory = self.profiles[key]`; `use_transitory, h = phase_day(k, self.H1, self.H2, transitory is not
  None)`; `source = transitory if use_transitory else operating`; `return float(source[i, l, j - 1, h])`. A missing
  key raises `KeyError` with the message `f"profile {key!r} is not in the input"`.
- Check: `local_checks.py` passes the three `phase_day` lines.

**A2. Options and shared variables.** Section "Options, costs, accessors" and `build_context`.
- `resolve_run_options(cfg) -> dict`: `return dict(cfg.options)`. Docstring: the options of "Run Options", defaults
  filled by `load_config`. No keyword overrides.
- Remove `pick`, `resolve_costs`, `cell_max`, `make_accessor`. Keep `apply_performance_params` unchanged.
- Add the pure function `cost_days(objective_mode: str, H1: int, T: int) -> range`: `range(H1, T)` for
  `"operating_average"`, else `range(T)` (spec 1548-1550).
- Rewrite `build_context(cfg, opts, model_name="fleet_management") -> FleetModel` (spec step 1 of the building order,
  line 255; variable table 607-632):
  1. `md = gp.Model(model_name)`; `md.Params.OutputFlag = int(opts["verbose"])`; `apply_performance_params(md,
     opts["time_limit"], opts["mip_gap"], opts["fast"], opts["gurobi_params"])`.
  2. `allow_replacement = bool(cfg.allow_replacement)`. `x = md.addVars(F, M + 1, T, vtype=GRB.BINARY, name="x")`;
     `m_rep = md.addVars(F, L, T, vtype=GRB.BINARY, name="m")`; `r_rep` likewise with `name="r"` if
     `allow_replacement`, else `None`. Do not create `idle` or `nb`.
  3. `mu_var = md.addVars(F, L, T, lb=0.0, name="mu")`, then `mu_var[i, l, k].UB = float(cfg.tau[i, l])`.
  4. `z_keys = [(i, l, k) for i in range(F) for l in range(L) if C_R[i, l] > 0 for k in cost_days(...)]` with
     `C_R = np.asarray(cfg.C_R, dtype=float)`; `z_var = md.addVars(z_keys, lb=0.0, name="z") if z_keys else
     gp.tupledict()`. No upper bound.
  5. `u_var = md.addVar(lb=0.0, name="u") if float(cfg.C_D) > 0 else None`.
  6. `profiles`: for `key` in `("mu", "v", "support", "cgf", "gamma_beta")` whose `getattr(cfg, key, None)` is not
     `None`: the pair (its float array, the float array of `getattr(cfg, key + "_trans", None)` or `None`).
  7. The four dicts of section 2.1 (`initial`, `replacement`, `anchor_start`, `state_range`), each value
     `np.asarray(..., dtype=float)`; the selector arrays as `np.asarray(..., dtype=str)`.
  8. Build `FleetModel(...)` with `opts=dict(opts)`, then `ctx.state_vars = {"mu": mu_var, "v": {}, "R": {}, "K": {}}`
     and `ctx.anchor_vars = {q: {} for q in TRACKED_QUANTITIES}`. Return `ctx`.
- Check: `names_check.py` reports none of the removed names outside the Gamma block; `local_checks.py` passes
  `cost_days`.

**A3. Variable helpers for `prepare`.** New section banner "Helpers for the cell builders: variables", after
`build_context`. Functions of section 2.2, plus:
- `_cell_value(values, i: int, l: int, label: str) -> float`: `value = float(values[i, l])`; if not
  `math.isfinite(value)`: `raise ValueError(f"{label} is not finite at cell (i={i}, l={l})")`; return `value`.
- `repair_factor(q: str, rho: float) -> float`: `1.0 - rho` for `"mu"` and `"K"`; `(1.0 - rho) ** 2` for `"v"` and
  `"R"`; another `q` raises `ValueError` (Eq. `a_psi`, table 1668-1682).
- `add_state_vars(ctx, q, cells, *, anchored=True)`: for each `(i, l)` in `cells`: `upper = _cell_value(
  ctx.state_range[q], i, l, f"range_{q}")`; `keys = [(i, l, k) for k in range(T)]`; if `q != "mu"`:
  `ctx.state_vars[q].update(md.addVars(keys, lb=0.0, ub=upper, name=q))`; if `anchored` and
  `str(ctx.repair_of[i, l]) == "ard1"`: `ctx.anchor_vars[q].update(md.addVars(keys, lb=0.0, ub=upper,
  name=f"g{q}"))`.
- `add_selector_vars(ctx, cells)`: for each cell with `str(ctx.encoding_of[i, l]) == "indicator"`:
  `ctx.nb.update(md.addVars(keys, vtype=GRB.BINARY, name="n"))`.
- `add_pwl_vars(ctx, cells)`: `P = int(ctx.pwl_points)`; return if `P < 2`; for each cell with
  `str(ctx.impl_of[i, l]) == "pwl"`: `ctx.pwl_w.update(md.addVars([(i, l, k, p) for k in range(T) for p in
  range(P)], vtype=GRB.BINARY, name="w"))`.
- Check: `local_checks.py` passes `repair_factor`.

**A4. Dynamics helpers.** New section banner "Helpers for the cell builders: rows". Functions of section 2.3:
- `mission_count(ctx, i, k)`: `gp.quicksum(ctx.x[i, j, k] for j in range(1, ctx.M + 1))`.
- `day_increment(ctx, i, l, k, key, *, squared=False)`: loop `j = 1..M`, `value = ctx.increment(key, i, j, l, k)`;
  non-finite: `raise ValueError(f"increment {key!r} is not finite at cell (i={i}, l={l}), mission {j}, day {k + 1}")`;
  `value = value * value` if `squared`; keep `value * ctx.x[i, j, k]` if `value != 0.0`; return `gp.quicksum(terms)`.
- `add_selector_rows(ctx, i, l)`: return unless `encoding_of[i, l] == "indicator"`; for each `k`:
  `md.addConstr(ctx.nb[i, l, k] + xm + xr == 1, name=f"sel_{i}_{l}_{k}")`, with `xr` omitted when `r_rep is None`.
- `add_quantity_rows(ctx, i, l, q, increments, *, repair_keeps_state=False)`. Set up once:
  `S = ctx.state_vars[q]`; `G = ctx.anchor_vars[q]`; `big_m = _cell_value(ctx.state_range[q], i, l, f"range_{q}")`;
  `start = _cell_value(ctx.initial[q], i, l, f"initial {q}")`; `rho = float(ctx.rho[i, l])`;
  `a = 1.0 if repair_keeps_state else repair_factor(q, rho)`;
  `anchored = (not repair_keeps_state) and str(ctx.repair_of[i, l]) == "ard1"`; if `anchored` and `(i, l, 0) not in
  G`: `raise ValueError(f"anchor of {q!r} missing at cell (i={i}, l={l}); call add_state_vars in prepare")`;
  `g_start = _cell_value(ctx.anchor_start[q], i, l, f"starting anchor {q}")` if `anchored`;
  `s_new = _cell_value(ctx.replacement[q], i, l, f"replacement {q}")` if `ctx.allow_replacement`;
  `indicator = str(ctx.encoding_of[i, l]) == "indicator"`; if `indicator` and `(i, l, 0) not in ctx.nb`: raise
  `ValueError` ("call add_selector_vars in prepare"); `len(increments) == T` or `ValueError`.
  Then for each `k`: `s = S[i, l, k]`; `s_prev = start if k == 0 else S[i, l, k - 1]`; `xm = ctx.m_rep[i, l, k]`;
  `xr = ctx.r_rep[i, l, k] if ctx.allow_replacement else 0.0`; if `anchored`: `g = G[i, l, k]`,
  `g_prev = g_start if k == 0 else G[i, l, k - 1]`, `psi = a * s_prev + (1.0 - a) * g_prev`; else `psi = a * s_prev`.
  - Big-M form (Eqs. `a_bigm_carry`, `a_bigm_repair`, `a_bigm_repl`, `a_bigm_anchor`, `w_vrow`), names in brackets:
    - if `repair_keeps_state`: `s >= s_prev + increments[k] - big_m * xr` [`{q}_carry_{i}_{l}_{k}`];
    - else: `s >= s_prev + increments[k] - big_m * (xm + xr)` [`{q}_carry_...`] and
      `s >= psi - big_m * (1 - xm)` [`{q}_repair_...`];
    - if replacement allowed: `s >= s_new - big_m * (1 - xr)` [`{q}_repl_lb_...`] and
      `s <= s_new + big_m * (1 - xr)` [`{q}_repl_ub_...`];
    - if `anchored`, with `e = xm + xr`: `g - s <= big_m * (1 - e)` [`g{q}_set_lb_...`], `s - g <= big_m * (1 - e)`
      [`g{q}_set_ub_...`], `g - g_prev <= big_m * e` [`g{q}_keep_lb_...`], `g_prev - g <= big_m * e`
      [`g{q}_keep_ub_...`].
  - Indicator form (Eqs. `a_ind_state`, `a_ind_anchor`), `n = ctx.nb[i, l, k]`, each with
    `md.addGenConstrIndicator(binary, True, temp_constr, name=...)`:
    `n => s >= s_prev + increments[k]` [`{q}_carry_...`]; `xm => s >= psi` [`{q}_repair_...`]; if replacement
    allowed: `xr => s == s_new` [`{q}_repl_...`]; if `anchored`: `n => g == g_prev` [`g{q}_keep_...`],
    `xm => g == s` [`g{q}_setm_...`], and if replacement allowed `xr => g == s` [`g{q}_setr_...`].
  - After the loop (Eq. `a_loop`): `S[i, l, T - 1] <= S[i, l, H1 - 1]` [`loop_{q}_{i}_{l}`]; if `anchored`:
    `G[i, l, T - 1] <= G[i, l, H1 - 1]` [`loop_g{q}_{i}_{l}`].
- Check: `load_base.py` loads; `names_check.py` reports 0 names; each row matches its spec equation (re-read 780-807).

**A5. Repair cost, caps and reliability forms.** Same section as A4.
- `add_repair_cost_rows(ctx, i, l)` (Eq. `a_z`): `ard1 = str(ctx.repair_of[i, l]) == "ard1"`; `rho`, `tau` of the
  cell; `mu_start = _cell_value(ctx.initial["mu"], ...)`; `g_start = _cell_value(ctx.anchor_start["mu"], ...)` if
  `ard1` (and the anchor must exist, as in A4); for each `k` with `(i, l, k) in ctx.z_var`: `mu_prev = mu_start if
  k == 0 else ctx.mu_var[i, l, k - 1]`; `g_prev = (g_start if k == 0 else ctx.anchor_vars["mu"][i, l, k - 1]) if ard1
  else 0.0`; `md.addConstr(ctx.z_var[i, l, k] >= rho * (mu_prev - g_prev) - rho * tau * (1 - ctx.m_rep[i, l, k]),
  name=f"z_{i}_{l}_{k}")`. No upper row; no row on replacement days.
- `add_upper_cap_rows(ctx, i, l, q, cap, *, name)`: for each `k`: `ctx.state_vars[q][i, l, k] <= float(cap)`,
  name `f"{name}_{i}_{l}_{k}"`.
- `add_quadratic_reliability_rows(ctx, i, l, q, *, c2, c1, mu_side)` (Eqs. `a_quad`, `a_tangent`, `a_pwl_*`):
  `chi = ctx.state_vars[q]`; `tau`; local functions `f(m) = c2 * (tau - m) ** 2 + c1 * (tau - m)` and
  `df(m) = -2.0 * c2 * (tau - m) - c1`; `form = str(ctx.impl_of[i, l])`.
  - Side row, every form, every `k`: `mu[k] <= mu_side` [`rel_side_{i}_{l}_{k}`].
  - `"exact"`: `md.addQConstr(chi[i, l, k] <= c2 * (tau - mu) * (tau - mu) + c1 * (tau - mu),
    name=f"rel_{i}_{l}_{k}")` with `mu = ctx.mu_var[i, l, k]`.
  - `"tangent"`: `mu_r = _cell_value(ctx.tangent_ref, i, l, "tangent_ref") * tau` (no clipping);
    `chi <= f(mu_r) + df(mu_r) * (mu - mu_r)` [`rel_{i}_{l}_{k}`].
  - `"pwl"` with `P = int(ctx.pwl_points)`: if `P == 1`, the tangent row at `mu_r = tau / 2` for every `k`. Else, for
    every `k`: `quicksum(w[i, l, k, p] for p) == 1` [`rel_seg_{i}_{l}_{k}`]; for `p = 0..P-1` with `lo = p * tau / P`,
    `hi = (p + 1) * tau / P`, `mid = (p + 0.5) * tau / P`: indicators `w => mu >= lo` [`rel_lo_{i}_{l}_{k}_{p}`],
    `w => mu <= hi` [`rel_hi_..._{p}`], `w => chi <= f(mid) + df(mid) * (mu - mid)` [`rel_cap_..._{p}`]. A missing
    `w` raises `ValueError` ("call add_pwl_vars in prepare").
  - Another `form`: `raise ValueError(f"unknown reliability_impl {form!r} at cell (i={i}, l={l})")`.
  - The helper does not set `NonConvex` (section 2.2).
- Check: `names_check.py` reports 0 names; read the three forms once more against spec 854-907.

**A6. Fleet rows and objective.** Section "General constraints and problem equations".
- `add_assignment_rows(ctx)`: the two loops of today's `add_base_constraints` (Eqs. `a_one_activity`, `a_coverage`),
  unchanged, names `assign_{i}_{k}` and `demand_{j}_{k}`.
- `add_depot_rows(ctx)` (Eq. `a_depot`): for every `i`, `l`, `k`: `m + r <= x[i, 0, k]`, or `m <= x[i, 0, k]` when
  `r_rep is None`, name `depot_{i}_{l}_{k}`.
- `add_damage_penalty_rows(ctx, penalty_type)` (Eqs. `a_u_inf`, `a_u_quad`): return if `ctx.u_var is None`; for every
  `i`, `k`: `"inf_norm"`: `md.addConstr(u >= quicksum(mu[i, l, k] for l), name=f"u_{i}_{k}")`; `"quadratic"`:
  `md.addQConstr(u >= quicksum(mu[i, l, k] * mu[i, l, k] for l), name=f"u_{i}_{k}")`; another value: `ValueError`.
  No `NonConvex` (the row is convex, spec 1580-1581).
- Remove `add_base_constraints` and `add_maintenance_gating`.
- Rewrite `build_objective(ctx, cfg, opts)` (Eqs. `a_daycost`, `a_phases`, `a_objective`): `C_M = float(cfg.C_M)`,
  `C_D = float(cfg.C_D)`, `C_R`, `C_rep` as `(F, L)` arrays. Day cost of index `k`: `C_M * quicksum(x[i, 0, k] for
  i)`, plus `C_R[i, l] * z[i, l, k]` for each key in `z_var`, plus `C_rep[i, l] * r[i, l, k]` for every cell when
  `r_rep` is not `None`. `J_init` sums indices `k < H1`, `J_op` the others. Mode `"total"`: `J_init + J_op`;
  `"operating_average"`: `(1.0 / H2) * J_op`; `"evaluation_total"`: `J_init + ((E - H1) / H2) * J_op` with
  `E = int(opts["evaluation_horizon"])`; another mode: `ValueError`. Add `C_D * u_var` only if `u_var` is not `None`.
  `md.setObjective(objective, GRB.MINIMIZE)`. Do not store `extras["phase_costs"]` (the extraction recomputes).
- Check: `names_check.py` reports 0 names; no remaining reference to `ctx.idle` outside the warm-start function.

**A7. Registry and assembly.** Sections "Cell-builder registry" and "Assembly".
- `CellBuilder` protocol: add the optional hook `extract(self, ctx, cfg, out) -> None` to the docstring (spec 249);
  keep `register_cell_builder`, `get_cell_builder`, `dispatch_cell` unchanged.
- `_load_builders(names) -> None`: for each name not in `CELL_BUILDERS`: `module =
  f"fleet_management.degradation_model.{name}"`; `importlib.import_module(module)`; catch `ModuleNotFoundError as
  error` and re-raise unless `error.name == module`. Add `import importlib` at the top.
- `build_fleet(cfg, opts, model_name="fleet_management")` (building order, spec 253-259): `present = sorted({str(m)
  for m in np.asarray(cfg.model).ravel()})`; `_load_builders(present)`; call `get_cell_builder(name)` for every name
  (fails before any Gurobi object exists); `ctx = build_context(cfg, opts, model_name)`; `prepare` of each present
  model in that order with `ctx.cells_of(name)`; `build_objective(ctx, cfg, opts)`; `add_assignment_rows(ctx)`;
  `add_depot_rows(ctx)`; `add_damage_penalty_rows(ctx, str(cfg.penalty_type))`; `dispatch_cell(ctx, i, l)` with `i`
  outer and `l` inner. Return `ctx`.
- Check: `load_base.py` loads; `names_check.py` reports 0 names.

**A8. Pure post-processing.** Section "Results". Replace `collapse` and `collapse_dict` by:
- `collapse_selector(arr, *, empty_as_none=False)`: `values = [[str(v) for v in row] for row in np.asarray(arr)]`;
  with `empty_as_none`, `""` becomes `None`; if every entry is equal, return that entry (a string or `None`); else
  return the nested list. (Spec 1712.)
- `transitory_repair_cost(mu, g_mu, m, mu_0, anchor_mu_0, rho, ard1, H1) -> np.ndarray` (Eq. `c_zpost`): array
  `(F, L, H1)`; for `k < H1`: `mu_prev = mu_0 if k == 0 else mu[:, :, k - 1]`; `g_prev = anchor_mu_0 if k == 0 else
  g_mu[:, :, k - 1]`, then `np.where(ard1, g_prev, 0.0)`; value `np.where(m[:, :, k] > 0.5, rho * (mu_prev - g_prev),
  0.0)`.
- `peak_damage(mu, penalty_type) -> float` (Eq. `c_u`): `float(np.max(np.sum(mu, axis=1)))` for `"inf_norm"`,
  `float(np.max(np.sum(mu * mu, axis=1)))` for `"quadratic"`, else `ValueError`.
- `cost_report(x, r, z, C_M, C_R, C_rep, C_D, u, H1, H2, evaluation_horizon) -> dict` (Eqs. `a_daycost`, `a_phases`,
  `d_projected`; spec 1727-1733, 1762): `z0 = np.where(np.isnan(z), 0.0, z)`; `step_costs[k] = C_M * x[:, 0, k].sum()
  + (C_R * z0[:, :, k]).sum() + (C_rep * r[:, :, k]).sum()`; `J_initialization = step_costs[:H1].sum()`; `J_op =
  step_costs[H1:].sum()`; `J_op_average = J_op / H2`; `peak_damage = u`; `damage_penalty = C_D * u`;
  `operating_objective = J_op_average + damage_penalty`; `J_total = J_initialization + J_op + damage_penalty`;
  `projected_evaluation_cost = J_initialization + (E - H1) * J_op_average` or `None` without `E`;
  `evaluation_objective = projected_evaluation_cost + damage_penalty` or `None`; `component_costs = {"C_M": C_M,
  "C_D": C_D, "C_R": np.asarray(C_R, dtype=float), "C_rep": np.asarray(C_rep, dtype=float)}`. Scalars as `float`.
- Check: `local_checks.py` prints `failures: 0` with no `MISSING` line.

**A9. Result extraction.** Rewrite `extract_solution(ctx, cfg) -> dict` (spec 1699-1769). Keep `status_string`.
- Module constants: `_SOLUTION_KEYS = ("x", "m", "r", "idle", "mu", "v", "z", "R", "K", "g_mu", "g_v", "g_R", "g_K",
  "u")` and `_COST_KEYS` = the eleven keys of `cost_report`.
- `_read_values(variables, shape) -> np.ndarray`: `np.full(shape, np.nan)`, then `values[key] = var.X` for each item.
- Always present: `status`; `F`, `M`, `L`, `H1`, `H2`, `T`, `H` (`cfg.H`); `model`
  (`np.asarray(cfg.model).astype(str).tolist()`); `component_names` (list); `tau`, `mu_0`, `v_0` (float arrays from
  `cfg`); `bound_method` = `collapse_selector(np.where(model == "rainflow", cfg.bound_method, ""),
  empty_as_none=True)`; `repair_model`, `reliability_impl`, `encoding` = `collapse_selector(...)`; `objective_mode`,
  `evaluation_horizon` from `ctx.opts`; `bound` = `float(md.ObjBound)` or `None` on `AttributeError` or
  `gp.GurobiError`. No key `method`, `models`, `model_assignment`; no Gurobi object.
- If `md.SolCount == 0`: `objective`, `mip_gap` and every key of `_SOLUTION_KEYS` and `_COST_KEYS` are `None`; return.
- Else: `objective = float(md.ObjVal)`; `mip_gap = float(md.MIPGap)` or `None` on error; `x`, `m`, `mu` by
  `_read_values`; `r` by `_read_values` or zeros when `r_rep is None`; `v`, `R`, `K` from `state_vars`; `g_mu`, `g_v`,
  `g_R`, `g_K` from `anchor_vars` (NaN where absent); `z` from `z_var`; `idle = x[:, 0, :][:, None, :] - m - r`.
- Mode `"operating_average"`, with `C_R = np.asarray(cfg.C_R, dtype=float)` and `ard1 =
  np.asarray(cfg.repair_model).astype(str) == "ard1"`: `z[:, :, :H1] = np.where(C_R[:, :, None] > 0,
  transitory_repair_cost(mu, g_mu, m, ctx.initial["mu"], ctx.anchor_start["mu"], ctx.rho, ard1, H1), np.nan)`.
- `u = peak_damage(mu, str(cfg.penalty_type))` (also when `C_D == 0`); then `out.update(cost_report(...))`.
- Check: `names_check.py` reports 0 names; by reading the code, the keys of both branches are the 51 keys of
  section 4 minus the four report keys (`performance`, `optimization_progress`, `warm_start`,
  `relaxation_warm_start`), which `solve_mixed` adds.

**A10. Reports and `solve_mixed`.** Sections "Options, costs, accessors" (callback) and "Assembly".
- `_make_progress_callback`: `interval = float(interval_seconds)` (remove `max(0.1, ...)`; spec 1811).
- New `_model_statistics(model) -> dict` with the 16 size and work keys of spec 1776-1786, in this order:
  `gurobi_version` (`".".join(str(v) for v in gp.gurobi.version())`), `variables` (`NumVars`),
  `continuous_variables` (`NumVars - NumIntVars`), `integer_variables` (`NumIntVars`), `binary_variables`
  (`NumBinVars`), `linear_constraints` (`NumConstrs`), `general_constraints` (`NumGenConstrs`),
  `indicator_constraints` (count of `getGenConstrs()` with `GenConstrType == GRB.GENCONSTR_INDICATOR`),
  `quadratic_constraints` (`NumQConstrs`), `nonzeros` (`NumNZs`), `solutions_found` (`SolCount`),
  `gurobi_runtime_seconds` (`Runtime`), `branch_and_bound_nodes` (`NodeCount`), `simplex_iterations` (`IterCount`),
  `barrier_iterations` (`BarIterCount`), `work_units` (`Work`). Each read is wrapped: `None` on `AttributeError`,
  `TypeError` or `gp.GurobiError`. Call `model.update()` first.
- Rewrite `solve_mixed(cfg, *, warm_start=None, warm_start_mode="start") -> dict` (spec 209-220, 253-259, 1735-1849):
  1. `warm_start_mode` not in `("start", "hint", "both")`: `ValueError`. `has_source = warm_start is not None and
     warm_start.get("x") is not None`. If `has_source` and the mode is not `"start"`: `NotImplementedError` (see
     section 6, question on warm-start modes).
  2. Timers as today (`backend_start`, `construction_start`); `opts = resolve_run_options(cfg)`;
     `ctx = build_fleet(cfg, opts)`; `ctx.model.update()`.
  3. `warm = apply_binary_warm_start(ctx, warm_start)`; `warm = complete_binary_warm_start(ctx.model, warm_start,
     warm)`; `warm["mode"] = warm_start_mode if has_source else None`.
  4. Relaxation report: option false: `{"enabled": False, "status": "disabled", "relaxation_objective": None,
     "values_copied": 0, "runtime_seconds": None}`; option true and `warm["applied"]`: the same with `"enabled": True,
     "status": "skipped"`; else `apply_relaxation_warm_start(ctx.model, time_limit=opts["relaxation_time_limit"])`.
  5. Optimise with the progress callback, then `_append_final_progress`, as today.
  6. `out = extract_solution(ctx, cfg)`; add `optimization_progress`, `warm_start`, `relaxation_warm_start`; call the
     `extract` hook of each present model, in sorted order.
  7. `performance = _model_statistics(ctx.model)` plus `model_construction_seconds`, `optimizer_call_seconds`,
     `solution_extraction_seconds`, `backend_wall_seconds`, `requested_time_limit_seconds =
     float(ctx.model.Params.TimeLimit)`, `requested_mip_gap = float(ctx.model.Params.MIPGap)`, `threads`, `seed`.
     Remove the import of `gamma_diagnostics`, `gamma_calibration_seconds` and `gamma_formulation`.
  8. `out["performance"] = performance`; `ctx.model.dispose()`; return `out`.
- Check: `names_check.py` reports 0 names; `grep gamma_diagnostics base.py` finds nothing.

**A11. Clean-up.**
- `apply_binary_warm_start`: delete the five lines that set `ctx.idle[i, l, k].Start` (`depot_value = ...` to
  `values += 1`). Change nothing else in the three warm-start functions.
- Remove `AccumulatedDegradationModel` and its section banner.
- Rewrite the module docstring (about 25 lines): what `base.py` owns (context, shared variables, registry, fleet rows,
  objective, helpers for the builders, extraction, reports), the building order of the spec, the two encodings, and
  that the Gamma block stays until WP6.
- Final check: all four local tools; `names_check.py` lists inside the Gamma block only names that A1-A10 removed
  (expected: `add_maintenance_gating`); report that list.

### Task B: `tests/test_base.py` (4 steps)

Pytest style, plain `assert`, `pytest.mark.parametrize`, `np.testing.assert_allclose(..., atol=1e-6)`. The tests use
only the API of section 2 and the fixture of section 4.1; they never import `config.py` (WP4 writes it in parallel).
Checks after every step: `python -c "import ast; ast.parse(open(r'<worktree>\tests\test_base.py',
encoding='utf-8').read())"` and `python names_check.py <worktree>\tests\test_base.py` (0 undefined names).

- **B1.** Module docstring (the tests need `gurobipy`; every model is far below 2000 variables and rows); imports
  (`gp = pytest.importorskip("gurobipy")` before `from fleet_management.degradation_model import base`); the fixture
  `_cfg` (4.1); the constants and helpers of 4.2; the three builders of 4.3. About 150 lines.
- **B2.** Tests T1, T2, T3, T6 of 4.4.
- **B3.** Tests T4, T5, T7.
- **B4.** Tests T8 to T13. Last check: every `base.<name>` in the file is a name of section 2.

## 4. Tests

The tests of `tests/test_base.py` run only on the licensed machine (gurobipy and pytest are not on this computer).
Every model below has at most about 120 variables, within the free limit of 2000 variables and 2000 rows.

### 4.1 Fixture `_cfg(**changes) -> SimpleNamespace` (the hand-worked case)

It mirrors `FleetConfig` of the interface file. Cell values are `(F, L)` arrays (`np.full`); `changes` replace any
attribute, and a scalar given for a cell value is broadcast. With `repair_model="ardinf"` the starting anchors are
`NaN`.

| Attributes | Default |
|---|---|
| `F, M, L, H, H1, H2, T` | `2, 1, 1, [2, 2], 2, 2, 4` |
| `component_names` | `["Component 1"]` |
| `model`, `repair_model`, `bound_method`, `reliability_impl`, `encoding` | `"gaussian"`, `"ard1"`, `""`, `"tangent"`, `"bigm"` |
| `tau, epsilon, tangent_ref, rho, mu_0, v_0` | `1.0, 0.1, 0.5, 0.5, 0.2, 0.0` |
| `anchor_mu_0, anchor_v_0`; `anchor_R_0, anchor_K_0` | `0.2, 0.0`; `NaN` |
| `replacement_mu, replacement_v`; `R_0, cgf_0, R_new, cgf_new` | `0.1, 0.0`; `NaN` |
| `support_0, support_new, s_chernoff, eta, v_max_user, gamma_beta_0, gamma_beta_new, support_const` | `NaN` |
| `sigma` | `[[0.1], [0.0]]` (read only by the Wiener-like builder) |
| `C_M, C_D, penalty_type`; `C_R, C_rep` | `1.0, 0.0, "inf_norm"`; `2.0, 3.0` per cell |
| `pwl_points, allow_replacement, replacement_as_new` | `8, True, False` |
| `mu`, `v` (shape `(2, 1, 1, 2)`) | train 0: `0.3` and `0.01`; train 1: `0.0` and `0.0` |
| `support, cgf, gamma_beta` and every `*_trans` | `None` |
| `range_mu, range_v`; `range_R, range_K` | `1.0, 1 / norm.ppf(0.9) ** 2` (0.6088746); `NaN` |
| `options` | `verbose 0, mip_gap 0.0, time_limit 60.0, fast False, gurobi_params None, objective_mode "total", evaluation_horizon None, progress_interval_seconds 5.0, relaxation_warm_start False, relaxation_time_limit 60.0, validation None` |

### 4.2 Constants and helpers

- `_SCHEDULE`: `x` of shape `(2, 2, 4)` with ones at `[0, 1, 0]`, `[0, 0, 1]`, `[0, 1, 2]`, `[0, 0, 3]`, `[1, 1, 1]`,
  `[1, 1, 3]`; `m` of shape `(2, 1, 4)` with a one at `[0, 0, 1]`; `r` with a one at `[0, 0, 3]`. Train 0 flies,
  is repaired, flies, is replaced; train 1 flies on days 2 and 4 with zero increments.
- `_fix_schedule(ctx)`: `LB = UB = value` on every variable of `ctx.x`, `ctx.m_rep`, `ctx.r_rep`.
- `_min_states(ctx)`: objective `sum(mu) + sum(state_vars["v"]) + sum(z_var)`, minimised. With fixed binaries the
  one-sided rows then give the exact trajectory.
- `_register(monkeypatch, builder)`: `monkeypatch.setitem(base.CELL_BUILDERS, "gaussian", builder)`.
- `_RESULT_KEYS`: the 51 keys of spec 1701-1739. `_PERFORMANCE_KEYS`: the 24 keys of spec 1776-1797.

### 4.3 Test builders (they show WP6 how the helpers fit together)

- `_MeanVarianceBuilder` (Gaussian-like). `prepare`: `add_state_vars(ctx, "mu", cells)`, `add_state_vars(ctx, "v",
  cells)`, `add_selector_vars`, `add_pwl_vars`, `NonConvex = 2` if a cell has `impl_of == "exact"`. `add_cell`:
  `add_selector_rows`; `add_quantity_rows` for `"mu"` and `"v"` with `day_increment(..., "mu")` and `(..., "v")`;
  `add_quadratic_reliability_rows(ctx, i, l, "v", c2=1 / kappa2, c1=0.0, mu_side=tau)` with
  `kappa2 = norm.ppf(1 - eps) ** 2`; `add_repair_cost_rows`.
- `_WienerLikeBuilder`: as above, but `add_state_vars(ctx, "v", cells, anchored=False)` and `add_quantity_rows(ctx, i,
  l, "v", [sigma ** 2 * base.mission_count(ctx, i, k) for k in range(T)], repair_keeps_state=True)`, with
  `sigma = cfg.sigma[i, l]` stored in `ctx.extras` by `prepare`.
- `_RowOnlyBuilder(c2, c1, mu_side)`: `prepare` creates `"v"` (`anchored=False`) and the `pwl` binaries; `add_cell`
  adds only `add_quadratic_reliability_rows(ctx, i, l, "v", ...)`.

### 4.4 Test list

Expected arrays of the hand-worked case (train 0; checked with `hand_case.py`). Train 1: `mu = 0.2`, `v = 0`,
`z = 0` on every day; `g_mu = 0.2` and `g_v = 0` under `ard1`.

| Array | `ard1` | `ardinf` |
|---|---|---|
| `mu` | 0.5, 0.35, 0.65, 0.1 | 0.5, 0.25, 0.55, 0.1 |
| `v` (Gaussian-like) | 0.01, 0.0025, 0.0125, 0.0 | 0.01, 0.0025, 0.0125, 0.0 |
| `v` (Wiener-like) | 0.01, 0.01, 0.02, 0.0 | 0.01, 0.01, 0.02, 0.0 |
| `g_mu` | 0.2, 0.35, 0.35, 0.1 | NaN |
| `g_v` (Gaussian-like) | 0.0, 0.0025, 0.0025, 0.0 | NaN |
| `z` (`C_R = 2`) | 0.0, 0.15, 0.0, 0.0 | 0.0, 0.25, 0.0, 0.0 |

- **T1 registry.** `base.get_cell_builder("no_such_model")` raises `NotImplementedError` whose message contains
  `"gamma"`.
- **T2 phase lookup.** `H = [2, 3]`, `v = None`; operating `mu` of train 0 = `[1, 2, 3]` over the days.
  `build_context` only.
  `ctx.increment("mu", 0, 1, 0, k)`, `k = 0..4`: `[1, 2, 1, 2, 3]` without `mu_trans`; `[10, 20, 1, 2, 3]` with
  `mu_trans = [10, 20]`. `ctx.increment("mu", 0, 0, 0, 3) == 0.0` (depot).
- **T3 variables only where used** (`build_fleet` with `_MeanVarianceBuilder`, then `model.update()`), one parametrized
  case per line: `C_D = 0` gives `u_var is None` and no row `u_0_0`; `C_D = 0.5` gives `u` and 8 rows `u_*`;
  `C_R = [[2], [0]]` gives `z_var` keys `{(0, 0, k)}`; mode `operating_average` gives keys only for `k = 2, 3`;
  `allow_replacement=False` gives `r_rep is None` and row `depot_0_0_0`; `bigm` gives empty `nb`, `indicator` gives 8;
  `pwl` with `pwl_points = 4` gives 32 `w`, with `pwl_points = 1` none; `ardinf` gives empty `anchor_vars["mu"]`,
  `ard1` gives 8.
- **T4 dynamics** (`_MeanVarianceBuilder`; parametrized over `bigm`/`indicator` and `ard1`/`ardinf`):
  `build_fleet`, `_fix_schedule`, `_min_states`, optimise, status `"optimal"`, `out = base.extract_solution(ctx,
  cfg)`. Compare `mu`, `v`, `g_mu`, `g_v`, `z` with the table; `R`, `K`, `g_R`, `g_K` all NaN; `idle` all zero.
- **T5 Wiener-like variance** (`_WienerLikeBuilder`; same four cases): `v` as in the table; `g_v` all NaN; `mu` as T4.
- **T6 loop rows.** `ard1`, `bigm`: rows `loop_mu_0_0`, `loop_v_0_0`, `loop_gmu_0_0`, `loop_gv_0_0` exist;
  `ardinf`: `loop_gmu_0_0` does not.
- **T7 repair cost on transitory days.** Mode `operating_average`, `ard1`, `C_R = [[2], [0]]`; T4's procedure:
  `z_var` has no key with `k < 2`; `out["z"][0, 0] == [0, 0.15, 0, 0]` (the first two by Eq. `c_zpost`);
  `out["z"][1, 0]` all NaN.
- **T8 reliability forms** (`_RowOnlyBuilder`; `H = [1, 1]`; options `gurobi_params = {"DualReductions": 0}`;
  `mu_var[0, 0, 0]` and `state_vars["v"][0, 0, 0]` fixed by bounds; base objective). Infeasible: status `"infeasible"` or
  `"inf_or_unbounded"`.
  - Gaussian type (`c2 = 1/kappa_G^2 = 1/1.6423744`, `c1 = 0`, side `1.0`), point `mu = 0.375`,
    `v = 0.99 f(0.375) = 0.2354632`: `exact` optimal; `tangent` with `tangent_ref = 0.375` optimal; `tangent` with
    `0.5` infeasible (cap 0.2283280); `pwl` with `pwl_points = 4` optimal (cap 0.2378416); `pwl_points = 1` infeasible.
  - Bernstein type, `exact` (`kappa = ln 10`, `b = 0.3`: `c2 = 0.2171472`, `c1 = -0.1`, side `0.7697415`,
    `range_v = 0.1171472`): `(0.3, 0.99 * 0.0364021)` optimal; `(0.3, 1.01 * 0.0364021)` infeasible; `(0.8, 0.0)`
    infeasible (side row).
- **T9 objective equals the report** (fixed schedule, base objective, `C_D = 0.5`), parametrized:
  `total`/`inf_norm` with key `J_total`; `operating_average`/`inf_norm` with `operating_objective`;
  `evaluation_total` (`E = 6`)/`inf_norm` with `evaluation_objective`; `total`/`quadratic` (form `exact`) with
  `J_total`. `abs(out["objective"] - out[key]) <= 1e-6 * max(1, abs(out[key]))`; `out["u"]` equals
  `max_{i,k} sum_l mu` (or `mu^2`) of `out["mu"]`.
- **T10 keys of `solve_mixed`** (free binaries, `C_D = 0.5`): status `"optimal"`; `set(out) == _RESULT_KEYS`;
  `set(out["performance"]) == _PERFORMANCE_KEYS`; `out["performance"]["requested_mip_gap"] == 0.0`;
  `out["warm_start"]["mode"] is None`; `out["relaxation_warm_start"]["status"] == "disabled"`; `out["model"] ==
  [["gaussian"], ["gaussian"]]`; `out["R"]` all NaN; `out["v"]` finite.
- **T11 no solution** (`solve_mixed`; `mu` increments 0.9 for both trains, `allow_replacement=False`, `H = [1, 1]`,
  `DualReductions = 0`): status infeasible as in T8; `objective`, `mip_gap` and every solution and cost key are
  `None`; `performance` and `objective_mode` are present.
- **T12 warm-start mode.** `solve_mixed(cfg, warm_start_mode="bogus")` raises `ValueError`.
- **T13 warm-start reports.** Source `{"x": ..., "m": ..., "r": ...}` of `_SCHEDULE` with `"H1": 2, "H2": 2`, option
  `relaxation_warm_start=True`: `out["warm_start"]["applied"] is True`, `["mode"] == "start"`;
  `out["relaxation_warm_start"]` has `enabled True` and status `"skipped"`.

### 4.5 Local checks on this computer

- Task A: the four tools of section 0 after every step. `local_checks.py` holds, among others: `phase_day` at
  `H1 = 2, H2 = 3`; `transitory_repair_cost` `[0, 0.15]` (`ard1`) and `[0, 0.25]` (`ardinf`) on the hand case;
  `cost_report` with `step_costs = [0, 1.3, 0, 4]`, `J_total = 5.725`, `P_E = 9.3` (`E = 6`, `C_D = 0.5`, `u = 0.85`).
  Task B: `ast` parsing and `names_check.py` on the test file.
- The stub, if it must be rebuilt: `stubs/gurobipy/__init__.py` with `GurobiError`, a class `GRB` (the status, type
  and callback constants that `base.py` uses), `tupledict(dict)`, classes `Model`, `Var`, `LinExpr`, `QuadExpr` that
  raise `RuntimeError` when called, `quicksum`, and `gurobi.version()`.

## 5. Deferred

- **WP6 (cell builders).**
  - New modules `gaussian.py`, `inverse_gaussian.py`, `wiener.py`; `gamma.py` with the sharp cap `A_1`; `rainflow.py`
    rewritten. Each builds its cells only with the helpers of section 2, registers itself, and sets `NonConvex` in
    `prepare`. `_load_builders` imports a module by the model's name, so each module must be named as its model.
  - The constants the builders compute (IG cap `mu_bar`, Gamma `beta*` and `A_1`, `kappa`, Bernstein `c` from
    `cfg.support_const`, Chernoff `s tau + ln eps`) and the checks that need them in `config.py` (interface file).
  - Changes WP6 must make in `base.py`: delete the Gamma block (`_add_binary_scaled_product`,
    `_gamma_reachable_upper_bounds`, `GammaCellBuilder`, its registration). Until then the block calls the removed
    `add_maintenance_gating` and reads removed context fields, so a Gamma cell fails at build time; it already fails
    today, because `prepare` imports `gamma_utils.gamma_repeated_calibration`, which WP1 removed.
  - `rainflow.py` imports names that A2 and A6 remove; a rainflow fleet fails at import until WP6 rewrites it. No
    other module imports `rainflow.py`.
- **WP7 (tools).**
  - Warm starts: the modes `"hint"` and `"both"` (then remove the `NotImplementedError` of A10 and pass the mode to the
    warm-start functions), `source_horizon` as a pair, and the rewrite of the three warm-start functions. The greedy
    start for every model.
  - `examples/regression/*.py` and `experiments/` scripts import `build_fleet`, `extract_solution(ctx, cfg, model)`
    and `resolve_run_options` with their old meanings; they are Gamma-specific and are updated or dropped with WP6 and
    WP7. `gamma_utils/gamma_diagnostics.py` is no longer used by `base.py`.
- **WP4.** Fills every field of `FleetConfig` that section 2.1 reads, the option defaults and the natural ranges;
  `solve_mapping` passes `warm_start_mode`.
- **WP8.** Runs `tests/test_base.py` on the licensed machine; may switch the fixture to `load_config` once WP4 is in.

## 6. Questions

None blocks a step; each step follows the recommendation unless the user decides otherwise.

- **Warm-start modes before WP7.** The spec defines `"hint"` and `"both"` (2111-2147), but the warm-start functions
  belong to WP7. Options: (1) `solve_mixed` raises `NotImplementedError` for these modes when a source is given, until
  WP7; (2) treat them as `"start"` without a message; (3) WP5 adds hints to `apply_binary_warm_start`, which crosses
  WP7's ownership. Recommendation: (1). It is explicit and lasts only until WP7. Step A10 follows it.
- **`NonConvex` and `gurobi_params`.** The spec sets `gurobi_params` "after every other option" (480), but the
  building order applies the options in step 1 and lets `prepare` set `NonConvex = 2` in step 2 (247, 255-256). A
  user's `NonConvex` in `gurobi_params` is then overwritten where an exact quadratic row exists. Options: (1) keep the
  spec's order; (2) apply `gurobi_params` again after step 2. Recommendation: (1). `NonConvex` is not a run option,
  and an exact quadratic row needs the value 2. The plan follows (1).
- **`bound_method` of a fleet without rainflow cells.** The spec says "one string if all cells agree, else a list",
  and `null` at a non-rainflow cell (1712). It does not say what a fleet without rainflow cells gets. Options: (1) one
  `None`; (2) an `F x L` list of `None`. Recommendation: (1), the rule "one value if all agree" applied to `null`. The
  WP4 writers then omit the key, and readers treat a missing key as `null` (1867). Step A9 follows (1).
