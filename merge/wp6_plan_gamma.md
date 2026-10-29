# WP6 plan: the Gamma cell builder

Status (29 Oct 2026): done. Plan carried out (Task G; section 5 by the shared executor) and reviewed (section 8).
Next: Claude adds the spec sentences of section 8, removes the two old Gamma test files with `git rm`, and commits;
WP8 runs the Gurobi tests.

## 0. Scope, sources and tools

- **Where.** Code: the worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge` (`<wt>` below), branch
  `merge/lore-into-main`, commit 5c8de19. Authority: the spec copy on `lore`,
  `C:\Users\utente\Documents\GitHub\Fleet_Management\spec\spec.tex` (`wp6_interface.md`: it wins over the worktree's
  copy). Spec line numbers below are those of the `lore` copy; after line 352 the worktree's are 1 to 3 lower.
- **Names.** WP: a work package of the merge (`merge/code_plan.md`). GIW: the Gaussian, inverse Gaussian (IG) and
  Wiener builders, planned in `merge/wp6_plan_giw.md`.
- **Files that Task G changes:** `<wt>\src\fleet_management\degradation_model\gamma.py` (rewritten),
  `degradation_model\__init__.py`, `degradation_model\base.py` (only the removal of the old Gamma block), and the new
  `<wt>\tests\test_gamma.py`. `config.py` and `tests\test_config.py` change later (section 5, shared executor).
- **Code style:** as `base.py` after WP5 (`merge/wp5_plan.md`, section 0): banners `# ====`, a one-line docstring and
  a short paragraph, `float(...)` around every parameter read from an array, plain `for` loops, row names that end
  with the indices, few comments, each citing an equation label. Lines of at most 99 characters. Windows line endings
  (CRLF), as the files they replace.
- **Rules for the executor:** no installs; no git command that changes state; do not delete any file; temporary
  files only in the scratchpad `<sp>` =
  `C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\`;
  never put the user's name or email address in any request; do not spawn sub-agents; re-read a file from disk
  before each edit.
- **The planner's files** in `<g>` = `<sp>\wp6\gamma\` (each script prints its result; outputs in `*_output.txt`):

| File | Content |
|---|---|
| `gamma_planned.py` | the reference `gamma.py` (sections 1 and 2), 202 lines |
| `test_gamma_planned.py` | the reference `tests/test_gamma.py` (section 6), 256 lines |
| `remove_gamma_block.py` | removes the old Gamma block from `base.py`; checks its markers before it writes |
| `copy_crlf.py` | copies a text file and writes CRLF line endings |
| `cap_proto.py` | the cap against brute force and against the records (`cap_proto_output.txt`) |
| `model_cases.py` | builder rows and the check of section 5, through `fakegp` (the linear emulator of `gurobipy` in `<sp>\wp5\fakegp`) and the real `config.py`; prints `failures: 0` |
| `run_tests_gamma.py` | runs a pytest-style file without pytest; adds `importorskip` (skips) and `parametrize` |
| `single_peak_small_r.py` | the single-peak property for small `r`; the cap at very small `eps` |
| `pkgcopy\` | a copy of the package with every planned change; the planner's own check, not for the executor |

## 1. What the builder builds

Spec: "Gamma Model" (1195-1304); "Cell-Builder Interface" (235-260); repair laws, encoding, replacement (727-852);
repair cost (1546-1563); loop (1590-1613). Decision "Gamma formulation" in `merge/decisions.md`.

- **Tracked quantities.** The shared mean `mu` (`ctx.mu_var`, bounds `[0, tau]`) and, at an `ard1` cell (repair of the
  damage since the last intervention), its anchor `gmu` in `[0, tau]`. The shape is not tracked; there is no variance.
- **Day increment.** `delta mu_k = sum_j x[i, j, k] mu_inc[j, k]` from `base.day_increment(ctx, i, l, k, "mu")`.
  The big-M constant (the constant that switches a row off) is `tau`, read by the helper from `ctx.state_range["mu"]`.
- **Per-cell constants**, computed in `prepare` and kept in `ctx.extras["gamma"]` (two dicts keyed by `(i, l)`):
  - `beta_star`: Eq. `gm_betastar`, by `smallest_rate` (section 2); `None` if no term of the cell has a rate.
  - `cap`: `mean_cap(beta_star, tau, eps)`, that is `A_1(beta* tau, eps) / beta*`, or `tau` without a rate (spec 1269).
  - Arguments of `smallest_rate` for cell `(i, l)`: the `M x T` lists `[[ctx.increment("mu", i, j, l, k) ...]]` and
    `[[ctx.increment("gamma_beta", i, j, l, k) ...]]` for `j = 1..M`, `k = 0..T-1` (values after the phase lookup,
    Eq. `c_phase`, each key on its own); `ctx.initial["mu"][i, l]`; `cfg.gamma_beta_0[i, l]`;
    `ctx.replacement["mu"][i, l]`; `cfg.gamma_beta_new[i, l]` (NaN where not given); `ctx.allow_replacement`.
- **`prepare(ctx, cfg, cells, opts)`:** `base.add_state_vars(ctx, "mu", cells)` (anchors at `ard1` cells; `mu` is
  shared), `base.add_selector_vars(ctx, cells)` (the selector `n` at `indicator` cells), then the constants. No
  segment binaries. `NonConvex` is not set: every row of the model is linear.
- **`add_cell(ctx, i, l)`**, in the order of the WP5 contract (`merge/wp5_plan.md`, section 2.3):
  1. `base.add_selector_rows`: Eq. `a_ind_sum` at an `indicator` cell, nothing at a `bigm` cell.
  2. `base.add_quantity_rows(ctx, i, l, "mu", increments)`: Eqs. `gm_mu_carry`, `gm_mu_inf` (`ardinf`, repair of the
     whole state) or `gm_mu_one` (`ard1`), `gm_mu_repl`, `gm_anchor_set`, `gm_anchor_keep` in the big-M form; Eqs.
     `a_ind_state`, `a_ind_anchor` in the indicator form; the loop rows `loop_mu_{i}_{l}` and, at `ard1`,
     `loop_gmu_{i}_{l}` (Eq. `a_loop`). Repair factor `1 - rho`; the replacement rows only if replacement is allowed.
  3. `base.add_upper_cap_rows(ctx, i, l, "mu", cap, name="rel")`: Eq. `gm_row`, rows `rel_{i}_{l}_{k}`.
  4. `base.add_repair_cost_rows`: Eq. `a_z`, at cells with `C_R > 0` only.
- **No `extract` hook.** The shared extraction reads `mu`, `g_mu` and `z`. The spec's result table has no Gamma key.
- **`reliability_impl`** is ignored (spec 1288); `config.py` already stores `exact` at Gamma cells.
- **What `config.py` guarantees** at a Gamma cell (spec 520-593): `tau > 0`; `0 < eps < 1` (no `eps < 0.5` rule);
  `0 < rho <= 1`; `0 <= mu_0 < tau`; `0 <= mu_inc < tau`; every rate finite and `> 0`; `gamma_beta_0` and
  `gamma_beta_new` `> 0` where given; `0 <= mu_new <= tau` when replacement is allowed. So every call of the cap has
  `t = beta* tau > 0` and `0 < eps < 1`.
- **Code** (the end of `gamma.py` after the review of section 8; docstrings left out here):

```python
class GammaCellBuilder:
    """Rows of the Gamma cells: tracked mean and anchor, cap row, repair cost."""

    name = "gamma"

    def prepare(self, ctx: base.FleetModel, cfg, cells, opts: dict) -> None:
        base.add_state_vars(ctx, "mu", cells)
        base.add_selector_vars(ctx, cells)
        M, T = ctx.M, ctx.T
        beta_star, cap = {}, {}
        for i, l in cells:
            mu_inc = [[ctx.increment("mu", i, j, l, k) for k in range(T)]
                      for j in range(1, M + 1)]
            beta_inc = [[ctx.increment("gamma_beta", i, j, l, k) for k in range(T)]
                        for j in range(1, M + 1)]
            beta_star[i, l] = smallest_rate(
                mu_inc, beta_inc,
                float(ctx.initial["mu"][i, l]), float(cfg.gamma_beta_0[i, l]),
                float(ctx.replacement["mu"][i, l]), float(cfg.gamma_beta_new[i, l]),
                ctx.allow_replacement,
            )
            cap[i, l] = mean_cap(beta_star[i, l], float(ctx.tau[i, l]), float(ctx.eps[i, l]))
        ctx.extras[self.name] = {"beta_star": beta_star, "cap": cap}

    def add_cell(self, ctx: base.FleetModel, i: int, l: int) -> None:
        T = ctx.T
        base.add_selector_rows(ctx, i, l)
        # Eqs. gm_mu_carry ... gm_anchor_keep
        base.add_quantity_rows(ctx, i, l, "mu",
                               [base.day_increment(ctx, i, l, k, "mu") for k in range(T)])
        # Eq. gm_row
        base.add_upper_cap_rows(ctx, i, l, "mu", ctx.extras[self.name]["cap"][i, l],
                                name="rel")
        base.add_repair_cost_rows(ctx, i, l)


if base is not None:
    base.register_cell_builder("gamma", GammaCellBuilder())
```

- **Against `lore`'s builder** (`src/fleet_management/models/gamma.py`): `lore` tracks the shape `alpha` with one rate
  per component, caps it by the exact one-rate cap (found by `brentq`), and writes its own anchor rows. The spec tracks
  the mean, takes the smallest rate of the cell, uses the sharp cap, and takes the dynamics, anchors and loop rows from
  the WP5 helpers. Nothing of `lore`'s builder is ported.
- **Checked** (`model_cases.py`, emulator, 88 checks, `failures: 0`): the hand-worked case of `test_base.py` with Gamma
  cells gives the trajectories of the WP5 table under both laws and both encodings; the cap rows have the right
  right-hand sides; the cap makes the fixed schedule infeasible exactly where it should (section 6).

## 2. The cap computation

Spec 1272-1304 (what to compute); scoping document, "Gamma: a sharp cap on the mean" and its derivations of the
properties of `w`, of the computation and of the cap below the threshold (why). `Q(a, x)` is `gammaincc(a, x)` of
`scipy.special`; `t = beta* tau`.

| Function | Computes | Method |
|---|---|---|
| `one_rate_cap(t, eps)` | `A_hat`, the root of `Q(A_hat, t) = eps` (step 1) | `ValueError` unless `t` is finite and `> 0` and `0 < eps < 1`. Bracket: `lo = 0`, `hi = max(1, 2t)`; while `Q(hi, t) <= eps`: `lo = hi`, `hi = 2 hi`. Bisection on the first argument until `hi - lo <= 1e-12 hi`; returns `lo`, so `Q(A_hat, t) <= eps` |
| `worst_one_term_tail(A, t)` | `w(A)`, the maximum over `c` in `(0, 1]` of `Q(A/c, t/c)` (step 2) | Log grid of 401 values of `c` in `[min(1e-4, t/100), 1]`. While the best point is the smallest `c`: a new grid of 401 values on `[c_min/100, c_min]`, at most 10 times, then `RuntimeError`. Golden-section search in `log c` between the neighbours of the best point, 60 steps; returns the largest value seen |
| `sharp_cap(t, eps)` | `A_1` (step 3) | `A_hat` if `A_hat < t` and `w(A_hat) <= eps`. Otherwise bisection on `A` in `[eps t, min(A_hat, t))` until `hi - lo <= 1e-9 hi`, with the test `w(mid) <= eps`; returns `lo`. For `eps >= 1/2` the test is true without computing `w`. Cached per `(t, eps)` by `functools.lru_cache` |
| `smallest_rate(mu_inc, beta_inc, mu_0, beta_0, mu_new, beta_new, allow_replacement)` | `beta*`, Eq. `gm_betastar` | the smallest of: the rates where `mu_inc > 0`; `beta_0` if `mu_0 > 0` and given; `beta_new` if replacement is allowed, `mu_new > 0` and given. `None` if there is none |
| `mean_cap(beta_star, tau, eps)` | right-hand side of Eq. `gm_row` | `tau` if `beta_star is None`; else `sharp_cap(beta_star * tau, eps) / beta_star` |

- **Special cases.**
  - `eps >= 1/2`: `w(A) < 1/2 <= eps` for every `A < t` (scoping, properties of `w`), so the test holds and the result
    equals the spec's step 3: about `t (1 - 1e-9)` (`A_1/t = 0.999999999069` at `eps = 0.5`). Computing `w` there
    would chase a maximum towards `c -> 0`, where `Q` tends to `1/2`.
  - `A_hat >= t` (for `eps` near or above `1/2`): the bisection runs on `[eps t, t)`.
  - `w(eps t) <= eps` by Markov's inequality, so the lower end of the bisection is safe.
  - No term with a rate: the cap is `tau`.
- **Threshold.** `t_1(eps)` is the value of `t` below which `A_1 < A_hat`: 0.436, 0.448, 0.614 for `eps` = 0.001,
  0.01, 0.1 (scoping). Below it, `A_1(t, eps) = r*(eps) t` with a constant `r*`; above it, `A_1 = A_hat`. The code
  does not use `t_1`; the tests do.
- **Code** (the top of `gamma_planned.py`; docstrings left out here):

```python
try:
    from fleet_management.degradation_model import base
except ModuleNotFoundError as error:      # no gurobipy: the cap functions still work
    if error.name != "gurobipy":
        raise
    base = None

_GRID_POINTS = 401           # values of c in one grid (spec: at least 400)
_GRID_EXTENSIONS = 10        # downward extensions of the grid, by a factor 100 each
_GOLDEN_STEPS = 60           # golden-section steps in log c
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
_SHAPE_RTOL = 1e-12          # bisection of step 1
_CAP_RTOL = 1e-9             # bisection of step 3


def one_rate_cap(t: float, eps: float) -> float:
    if not (math.isfinite(t) and t > 0.0 and 0.0 < eps < 1.0):
        raise ValueError(
            f"the Gamma cap needs t > 0 and 0 < epsilon < 1 (t={t!r}, epsilon={eps!r})"
        )
    lo, hi = 0.0, max(1.0, 2.0 * t)
    while gammaincc(hi, t) <= eps:
        lo, hi = hi, 2.0 * hi
    while hi - lo > _SHAPE_RTOL * hi:
        mid = 0.5 * (lo + hi)
        if gammaincc(mid, t) <= eps:
            lo = mid
        else:
            hi = mid
    return lo


def worst_one_term_tail(A: float, t: float) -> float:
    grid = np.geomspace(min(1e-4, t / 100.0), 1.0, _GRID_POINTS)
    values = gammaincc(A / grid, t / grid)
    best = int(np.argmax(values))
    extensions = 0
    while best == 0:
        if extensions == _GRID_EXTENSIONS:
            raise RuntimeError(...)            # message: the maximum lies below c = grid[0]
        grid = np.geomspace(grid[0] / 100.0, grid[0], _GRID_POINTS)
        values = gammaincc(A / grid, t / grid)
        best = int(np.argmax(values))
        extensions += 1

    def tail(u):
        c = math.exp(u)
        return float(gammaincc(A / c, t / c))

    lo = math.log(grid[max(best - 1, 0)])
    hi = math.log(grid[min(best + 1, _GRID_POINTS - 1)])
    u1, u2 = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
    f1, f2 = tail(u1), tail(u2)
    for _ in range(_GOLDEN_STEPS):
        if f1 < f2:
            lo, u1, f1 = u1, u2, f2
            u2 = lo + _GOLDEN * (hi - lo)
            f2 = tail(u2)
        else:
            hi, u2, f2 = u2, u1, f1
            u1 = hi - _GOLDEN * (hi - lo)
            f1 = tail(u1)
    return max(float(values[best]), f1, f2)


def _within_level(A: float, t: float, eps: float) -> bool:
    return eps >= 0.5 or worst_one_term_tail(A, t) <= eps


@functools.lru_cache(maxsize=None)
def sharp_cap(t: float, eps: float) -> float:
    a_hat = one_rate_cap(t, eps)
    if a_hat < t and _within_level(a_hat, t, eps):
        return a_hat
    lo, hi = eps * t, min(a_hat, t)
    while hi - lo > _CAP_RTOL * hi:
        mid = 0.5 * (lo + hi)
        if _within_level(mid, t, eps):
            lo = mid
        else:
            hi = mid
    return lo
```

  `smallest_rate` and `mean_cap` follow the table above (13 lines in `gamma_planned.py`).
- **Placement.** The five functions sit at the top of `gamma.py` and use numpy and scipy only. `gamma.py` imports
  `base` under the guard above and registers the builder only if `base` loaded. So `config.py` (section 5) and WP7's
  validators import `mean_cap` and `smallest_rate` without Gurobi, and the spec's layout holds ("`gamma.py` computes
  the cap A_1", spec 192-193). `base._load_builders` imports `gamma.py` after `base` itself is loaded, so the builder is
  always registered when a model is built. Other placements: question Q1.
- **Cost.** About 30 ms per distinct `(t, eps)`, 16 evaluations of `w` per cap (`cap_timing.py`). The cache makes the
  builder's call free after the check in `config.py`.
- **Reference values** (`cap_proto_output.txt`, computed by the code above):

| Quantity | Value | Agrees with |
|---|---|---|
| `A_1(t, 0.01)`, `t` = 0.01, 0.1, 0.3, 1, 10 | 3.528887629e-4, 3.52888763e-3, 1.058666289e-2, 4.365272936e-2, 3.971715971 | `verify_gamma_sharp/single_peak_check_output.txt`: every printed digit (8) |
| `r* = A_1(0.01, eps)/0.01`, `eps` = 0.001, 0.01, 0.1 | 0.00355012891, 0.0352888763, 0.33061707 | same file: 0.00355013, 0.0352889, 0.330617; scoping: 0.00355, 0.0353, 0.331 |
| `A_1/A_hat`, `eps` = 0.001, 0.01, 0.1 and `t` = 0.01, 0.1, 0.3, 1, 3, 10, 100 | at `eps` = 0.01: 0.1419, 0.6415, 0.9606, 1, 1, 1, 1 | `verify_gamma_sharp/check_output.txt`: all 21 values (4 digits) |
| `A_hat`, 72 pairs (`eps` 1e-6 to 0.99, `t` 1e-6 to 1e4) | | `brentq` at relative tolerance 1e-15: largest relative difference 8.8e-13 |
| `A_1`, 32 pairs (`eps` 0.001 to 0.3, `t` 1e-4 to 1000) | | brute force (20001 values of `s = t/c`, bounded refinement, bisection to 1e-13): 6.7e-10 |
| `w(A_hat(0.1, 0.01)) / 0.01` | 1.564868 | `check_output.txt`: 1.564868; scoping: 1.565 |
| `w(A_1)/eps`, `w(A_1 (1 + 1e-6))/eps`, nine pairs below `t_1` | 1.000000000 and 1.000001 | the cap is safe and sharp |
| mean caps at `tau = 1`, `eps = 0.1`, `beta*` = 20, 5, 2, 0.2 | 0.744517324566, 0.553864057892, 0.414912231273, 0.330617069585 | used by the tests |

- **Beyond the records.** The scoping checks the single peak of `s -> Q(rs, s)` for `r` in `[1e-5, 0.999]`, which
  covers `eps >= 1e-5`. `single_peak_small_r.py` extends it to `r` in `[1e-10, 1e-5]` (200 values, no second peak);
  the cap agrees with brute force at `eps` = 1e-6 and 1e-8 (largest difference 2.2e-10). At `eps` = 0.3, 0.45, 0.49,
  0.499 and `t` from 1e-6 to 1e4, no grid reaches its last extension and the cap agrees with brute force to 7.4e-10
  (`large_eps_output.txt`).

## 3. Removals and import paths

- **`base.py`** (2358 lines at 5c8de19). Remove lines 184-1209: the banner of the Gamma block,
  `_add_binary_scaled_product`, `_gamma_reachable_upper_bounds`, `GammaCellBuilder` with its `extract`,
  `register_cell_builder("gamma", GammaCellBuilder())` and the two blank lines after it. Remove also lines 26-29 of
  the module docstring ("The Gamma block below stays until ..."), which describe the block. Nothing else changes:
  every import is still used (checked). Result: 1328 lines; the diff holds 1030 deletions and no addition. The only
  remaining "gamma" is the profile key `"gamma_beta"` in `build_context`.
- **`gamma.py`**: rewritten. Gone: `GammaAction` and `GammaModel` (a shape state with a common-rate repair
  approximation, dropped by the decision "Gamma formulation"), `maximum_reliable_shape`,
  `maximum_reliable_expected_damage`, `_validate_positive`. New: the five functions of section 2, `GammaCellBuilder`
  and its registration.
- **`degradation_model/__init__.py`**: today it imports `GammaAction` and `GammaModel`. Afterwards it imports nothing:

```python
"""
Degradation models: the shared model ``base`` and one cell builder per model.

The package imports no module, so that it loads without Gurobi; ``base`` imports
the builder of a model by its name when a fleet needs it.
"""
```

  Reason: every import of a submodule (`base`, `gamma_utils.*`) runs this file first. If it imported `gamma.py`, the
  package would load `base` and `gurobipy` whenever they exist, and the registration would depend on the import order.
- **Who imports what** (grep of `<wt>`, 29 Oct 2026):
  - `gamma_utils/*.py` (WP7) import nothing from `gamma.py`. WP7's replay validator and the Gamma dashboard need the cap
    (spec 2459, 2513); they can import `from fleet_management.degradation_model.gamma import mean_cap, smallest_rate`
    without Gurobi.
  - `GammaModel`, `GammaAction`, `maximum_reliable_*`: only the two old tests and `__init__.py` use them (and the
    ignored `src/fleet_management.egg-info/SOURCES.txt`).
  - `import fleet_management` runs `degradation_model/__init__.py` through the validators; it stays importable without
    Gurobi. Checked on `pkgcopy` with `gurobipy` blocked: the package imports, `gamma.base is None`, `base` is not
    loaded.
- **The two old tests.** `tests/test_gamma_transitions.py` tests `GammaModel` and `GammaAction`: no replacement, the
  classes leave. `tests/test_gamma_reliability_bound.py` tests `maximum_reliable_shape`; its checks (the tail at the cap
  equals `eps`; 0.9 times the cap is below, 1.1 times above; three parameter sets; invalid inputs refused) move to
  `tests/test_gamma.py` as tests of `one_rate_cap`, with `t = beta tau` = 1.25, 11, 40. Both old files are then
  removed with `git rm` when Claude commits; the executor deletes no file. Until then pytest cannot collect them
  (question Q3).
- **`tests/test_base.py`, test T1** (`test_unknown_model_has_no_cell_builder`) expects "gamma" in the message of
  `get_cell_builder("no_such_model")`. Today the old block registers "gamma" whenever `base` is imported. After the
  removal nothing is registered at import time, the message ends with "available: ()", and T1 fails when
  `test_base.py` runs alone (checked on `pkgcopy`). Question Q2; step G2b.

## 4. Executor task

**Task G**, one Sonnet executor. One task, not two: the old `__init__.py` imports `GammaAction` from `gamma.py`, so the
new module cannot be loaded before `__init__.py` changes; and while the old block registers "gamma" at import time,
`build_fleet` never loads the new builder. Steps G2 to G4 change more than about 150 lines each, but by a checked
script or by copying a verified file; nothing is edited by hand. Paths: `<wt>`, `<sp>`, `<g>` as in section 0; `<dm>` =
`<wt>\src\fleet_management\degradation_model`. Run every `python` command from `<g>` unless said otherwise.

**G1. `__init__.py`.** Write the docstring of section 3 as the whole content of `<dm>\__init__.py`, with CRLF line
endings. Check: the file has 6 lines and no `import`.

**G2. Remove the old Gamma block.** Run `python remove_gamma_block.py <dm>\base.py`. Expected output: "removed
docstring lines 26-29 and block lines 184-1209 (1026 lines); 2358 -> 1328 lines". If the script stops at a marker,
stop and report: `base.py` is not the file of commit 5c8de19. Checks:
- from `<sp>\wp5`: `python load_base.py` loads `base.py`; `python names_check.py` reports 0 undefined names, outside
  and inside the old block; `python local_checks.py` prints `failures: 0`;
- `git -C <wt> diff --stat -- src/fleet_management/degradation_model/base.py` shows 1030 deletions and no insertion;
- `grep -n -i "gamma" <dm>\base.py` finds one line, the profile key `"gamma_beta"` in `build_context`.

**G2b. [Blocked by Q2.] Test T1.** Only if Claude agrees: in `<wt>\tests\test_base.py`, function
`test_unknown_model_has_no_cell_builder`, insert `base._load_builders(["gamma"])` as its first line. Check:
`python -c "import ast; ast.parse(open(r'<wt>\tests\test_base.py', encoding='utf-8').read())"`.

**G3. `gamma.py`.** Run `python copy_crlf.py gamma_planned.py <dm>\gamma.py`. Read the new file once against sections 1
and 2: the guard, the six constants, the five functions, the class, the row name `rel`, the registration. Checks:
- `python model_cases.py <dm>\gamma.py` prints `failures: 0` (88 checks: section 6, cases G and C);
- `python <sp>\wp5\names_check.py <dm>\gamma.py` reports only `base` in `GammaCellBuilder.prepare`, `add_cell` and
  its list comprehension. This is expected: the guarded import binds `base`, and the tool does not see it. Any other
  name is an error;
- with `PYTHONPATH=<wt>\src`: `python -c "import sys; sys.modules['gurobipy'] = None; import fleet_management; from
  fleet_management.degradation_model import gamma; print(gamma.base, round(gamma.mean_cap(20.0, 1.0, 0.1), 9))"`
  prints `None 0.744517325`.

**G4. `tests/test_gamma.py`.** Run `python copy_crlf.py test_gamma_planned.py <wt>\tests\test_gamma.py`. Checks:
`python run_tests_gamma.py <wt>\tests\test_gamma.py` prints `21 passed, 15 skipped, 0 failed` (the 15 model cases need
gurobipy); no line is longer than 99 characters.

**G5. Report.** `git -C <wt> status --short` lists: modified `__init__.py`, `base.py`, `gamma.py` (and
`tests/test_base.py` after G2b); new `tests/test_gamma.py`; nothing else. The report gives the output of every check
and every point where the work differed from the plan.

## 5. config.py change

Applied by the shared executor after Task G (the import needs the new `gamma.py`), together with the IG check of
`merge/wp6_plan_giw.md`. Spec 557-561 ("Natural ranges and replacement states"), Eqs. `gm_betastar`, `gm_row`;
deferred by `merge/wp4_plan.md`, section 4.

**C1. The check.** File `<wt>\src\fleet_management\config.py`, function `_check_ranges`, at its end: after the Markov
check (today lines 811-812), inside the block `with np.errstate(...)`, indented by 8 spaces. The function has
already returned when `allow_replacement` is false, as the spec asks; `mu` is `cfg.replacement_mu` (line 802).

```python
        if m["gamma"].any():
            # Gamma cap, Eqs. gm_betastar and gm_row; gamma.py loads without Gurobi
            from fleet_management.degradation_model.gamma import mean_cap, smallest_rate
            mu_inc = _lookup(cfg.mu, cfg.mu_trans, cfg.H1, cfg.H2)
            beta_inc = _lookup(cfg.gamma_beta, cfg.gamma_beta_trans, cfg.H1, cfg.H2)
            gamma_cap = np.full(mu.shape, np.nan)
            for i, l in np.argwhere(m["gamma"]):
                beta_star = smallest_rate(mu_inc[i, l], beta_inc[i, l], float(cfg.mu_0[i, l]),
                                          float(cfg.gamma_beta_0[i, l]), float(mu[i, l]),
                                          float(cfg.gamma_beta_new[i, l]), True)
                gamma_cap[i, l] = mean_cap(beta_star, float(cfg.tau[i, l]),
                                           float(cfg.epsilon[i, l]))
            check(m["gamma"], _le(mu, gamma_cap), "replacement_mu",
                  "must be <= the Gamma cap A_1(beta_star*tau, epsilon)/beta_star at a gamma cell")
```

- `_lookup` is `config.py`'s phase lookup (Eq. `c_phase`); WP5 checked it against `base.phase_day` (98 cases, no
  disagreement; `merge/wp5_plan.md`, section 7). So the check and the builder see the same pairs of mean and rate, and
  both call the same `smallest_rate` and `mean_cap`.
- `_le` gives the tolerance `1e-12 max(1, |a|, |b|)` of spec 525. A cell without a rate gets the cap `tau`, which the
  natural-range check above already enforces.
- The message reads, for example: "cell (i=0, l=0) model=gamma: 'replacement_mu' must be <= the Gamma cap
  A_1(beta_star*tau, epsilon)/beta_star at a gamma cell."

**C2. The module docstring** of `config.py`. Replace its line 18,
"(``yaml`` and ``h5py`` are imported inside the reader), so it runs without Gurobi.", by the two lines
"(``yaml`` and ``h5py`` are imported inside the reader, and the Gamma cap of" and
"``degradation_model.gamma`` inside its check), so it runs without Gurobi."

**C3. The test**, appended to `<wt>\tests\test_config.py` after `test_replacement_rows_off`:

```python
def test_gamma_replacement_cap():
    # mean cap A_1(beta* tau, epsilon)/beta* at tau = 1, epsilon = 0.1: 0.744517 for beta* = 20,
    # 0.414912 for beta* = 2, 0.330617 for beta* = 0.2 (below the threshold t_1)
    cap = "must be <= the Gamma cap"
    _fails("gamma", None, "replacement_mu", cap, replacement_mu=0.7453)
    load_config(_small("gamma", replacement_mu=0.7437))
    _fails("gamma", None, "replacement_mu", cap, gamma_beta_new=2.0, replacement_mu=0.5)
    load_config(_small("gamma", gamma_beta_new=2.0, replacement_mu=0.41))
    load_config(_small("gamma", replacement_mu=0.5))
    load_config(_small("gamma", gamma_beta_new=2.0, replacement_mu=0.5, allow_replacement=False))
    # day 1 pairs the transitory rate with the operating mean (phase lookup)
    _fails("gamma", None, "replacement_mu", cap, gamma_beta_trans=2.0, replacement_mu=0.5)
    load_config(_small("gamma", gamma_beta_trans=2.0, mu_trans=0.0, replacement_mu=0.5))
    _fails("gamma", None, "replacement_mu", cap, gamma_beta_0=2.0, replacement_mu=0.5)
    load_config(_small("gamma", gamma_beta_0=2.0, mu_0=0.0, replacement_mu=0.5))
    _fails("gamma", None, "replacement_mu", cap, gamma_beta=0.2, replacement_mu=0.34)
    load_config(_small("gamma", gamma_beta=0.2, replacement_mu=0.32))
    # no term with a rate: only the natural range mu_new <= tau
    load_config(_small("gamma", mu=0.0, replacement_mu=0.9))
```

`_small("gamma")` has `tau = 1`, `eps = 0.1`, `mu_0 = 0.1`, `mu = 0.05`, `gamma_beta = 20`, `H = [1, 2]`. Check:
`python <sp>\wp4\run_tests.py <wt>\tests\test_config.py` prints `56 passed` before the IG test is added (55 old cases
and this one; checked on `pkgcopy`, where `test_example_loads` also passes: the spec's example has a Gamma cell with
cap 0.6727 and `replacement_mu = 0.02`). The run has no `gurobipy`, so it also shows that the check works without
Gurobi.

## 6. Tests

`tests/test_gamma.py` = `test_gamma_planned.py` (pytest style, as `tests/test_base.py`). It imports `load_config` and
`gamma` at module level, both without Gurobi. Every model test calls `pytest.importorskip("gurobipy")` inside its body,
so the cap tests run everywhere. "Local" means this computer, through `run_tests_gamma.py` (21 cases pass, checked
on `pkgcopy`); "Gurobi" means the licensed machine (WP8).

| Test (cases) | Checks | Where |
|---|---|---|
| `test_one_rate_cap_reaches_the_level` (3) | `(t, eps)` = (1.25, 0.10), (11, 0.05), (40, 0.01): `A_hat` = 0.452900672495, 6.36399044578, 26.5638726679 (relative 1e-9); `abs(Q(A_hat, t) - eps) <= 1e-10`; `Q(0.9 A_hat, t) < eps < Q(1.1 A_hat, t)` | local |
| `test_one_rate_cap_rejects_invalid_input` (9) | `t` in {0, -1, inf, NaN}, `eps` in {0, 1, -0.1, 1.1, NaN}: `ValueError` "Gamma cap" | local |
| `test_sharp_cap_reference_values` (1) | `A_1(t, 0.01)` of section 2, relative 1e-8 | local |
| `test_sharp_cap_ratio_table` (1) | the 21 values of `A_1/A_hat`, within 6e-5; `A_1 == A_hat` exactly for `t >= 1` | local |
| `test_sharp_cap_below_the_threshold_is_proportional` (1) | `A_1(t, eps)/t = r*(eps)` for `t` = 0.01, 0.1, 0.3, relative 1e-7 | local |
| `test_sharp_cap_is_safe_and_sharp` (1) | nine pairs below `t_1`: `w(A_1) <= eps < w(A_1 (1 + 1e-6))` | local |
| `test_worst_one_term_tail` (1) | `w(A_hat(0.1, 0.01))/0.01 = 1.564868` (relative 1e-6); at `t = 10`, `w(A_hat) = Q(A_hat, 10)` | local |
| `test_sharp_cap_at_large_epsilon` (1) | `eps` = 0.5, 0.7 and `t` = 0.01, 1, 100: `t (1 - 2e-9) <= A_1 < t` | local |
| `test_smallest_rate` (1) | eight cases: a rate with mean 0 is left out; `beta_0` only with `mu_0 > 0`; a known damage has no rate; `beta_new` only with replacement and `mu_new > 0`; no rate gives `None` | local |
| `test_mean_cap` (1) | `None` gives `tau`; the four mean caps of section 2; `mean_cap(4, 0.5, 0.1) = 0.5 mean_cap(2, 1, 0.1)` | local |
| `test_cap_without_gurobi` (1) | a subprocess with `gurobipy` blocked imports `gamma` (`gamma.base is None`), computes a cap and runs `load_config` on a Gamma input with replacement | local (after Task G) and Gurobi |
| `test_builder_is_registered` (1) | `base.get_cell_builder("gamma").name == "gamma"` | Gurobi |
| `test_dynamics_and_cap` (4) | both laws times both encodings; hand-worked case below: `mu`, `g_mu`, `z` of train 0 as in the table, train 1 `mu = 0.2`, `z = 0`; `beta_star == {(0, 0): 20, (1, 0): None}`; right-hand side (RHS) of `rel_0_0_k` = 0.744517324566 and of `rel_1_0_k` = 1.0; `loop_gmu_0_0` only at `ard1`; no quadratic row; `NonConvex == -1` (not set) | Gurobi |
| `test_cap_binds` (2) | `gamma_beta_new = 2`: `beta*` = 2 at both cells, cap 0.414912231273 < 0.5, status `infeasible` or `inf_or_unbounded` | Gurobi |
| `test_cap_depends_on_the_repair_law` (1) | `gamma_beta_0 = 5`: cap 0.553864057892; `ard1` (peak 0.65) has no solution, `ardinf` (peak 0.55) is optimal | Gurobi |
| `test_smallest_rate_of_each_cell` (7) | `prepare` only: `beta_new` left out without replacement or with `mu_new = 0`; a transitory rate counts where the mean of the day is positive, also when the mean comes from the operating profile; `beta_0` left out with `mu_0 = 0`; rates per cell | Gurobi |

**The hand-worked case** (the case of `merge/wp5_plan.md`, section 4, with Gamma cells), input of `load_config`:
`F = 2`, `M = 1`, `L = 1`, `H = [2, 2]`, `tau = 1`, `eps = 0.1`, `rho = 0.5`, `mu_0 = 0.2`, `replacement_mu = 0.1`,
`mu` = 0.3 for train 0 and 0 for train 1 (shape `(F, L, M)`), `gamma_beta = 20`, `C_M = 1`, `C_R = 2`, `C_rep = 3`,
`C_D = 0`, `gurobi_params = {DualReductions: 0}`. Schedule fixed by bounds: train 0 flies on day 1, is repaired on day 2,
flies on day 3 and is replaced on day 4; train 1 flies on days 2 and 4. Objective: minimise the sum of `mu`, `gmu`
and `z`, so the one-sided rows give the exact trajectory.

| Train 0 | `ard1` | `ardinf` |
|---|---|---|
| `mu` | 0.5, 0.35, 0.65, 0.1 | 0.5, 0.25, 0.55, 0.1 |
| `g_mu` | 0.2, 0.35, 0.35, 0.1 | NaN |
| `z` | 0, 0.15, 0, 0 | 0, 0.25, 0, 0 |

The model tests cannot run here: the emulator returns row names, not row objects with `ConstrName` and `RHS`.
`model_cases.py` checks the same cases through the emulator (88 checks, `failures: 0`), and the check of section 5
on the inputs of C3 (cases C1 to C7). So every expected value of the Gurobi tests has been reproduced once.

## 7. Questions

The spec, the scoping document and the decision "Gamma formulation" agree on the Gamma model; I found no
contradiction. The WP5 helpers cover every row: no helper is missing. Five points need Claude's answer. Only G2b
waits for one (Q2); every other step follows the recommendation.

- **Q1. Where the Gurobi-free cap lives.** `config.py` and WP7's validators need `A_1` and `beta*` without Gurobi; a
  builder module imports `base`, and `base` imports `gurobipy`.
  - Options: (1) `gamma.py`, with the guarded import of `base` (section 2): the spec's layout holds as written (spec
    192-193), and one module holds the model. (2) A new module without Gurobi, for example
    `degradation_model/gamma_cap.py`: plainer imports, but the spec's layout gains a file. (3) `config.py`, as the GIW
    plan proposes for the IG cap (its Q1): the input layer then holds about 110 lines of Gamma numerics, and spec 193
    changes.
  - Recommendation: (1), and the same rule for the IG cap, so that each model's cap lives in its model's module. The
    GIW plan's objection to (1) ("that import still loads `base` and `gurobipy`") does not hold with the guard:
    checked on `pkgcopy` with `gurobipy` blocked. Claude aligns the two plans.
  - Steps that change with (2) or (3): G3 (where the five functions go), C1 (one import line), the import line of
    `tests/test_gamma.py`.
- **Q2. Test T1 of `tests/test_base.py` fails after the removal** (section 3). Options: (1) the Gamma executor adds
  `base._load_builders(["gamma"])` as the first line of T1 (step G2b; the file is outside this plan's ownership);
  (2) T1 matches `"no_such_model"` instead of `"gamma"` (it then no longer tests that the message lists the
  builders); (3) Claude edits T1 at review. Recommendation: (1). It keeps T1's intent with one line. Blocks G2b only.
- **Q3. Who removes the two old test files.** They import names that leave, so pytest cannot collect them after
  Task G. Options: (1) Claude, with `git rm` at the commit, as for the removals of WP1; (2) the executor moves them to
  the Recycle Bin. Recommendation: (1): no executor deletes files, and git keeps their history. Blocks no step; both
  must be gone before WP8 runs pytest.
- **Q4. Numerical choices the spec leaves open** (spec 1291-1303): the bracket and tolerance of step 1 (relative
  1e-12, the lower end returned); the downward extension of the grid (a factor 100, at most 10 times, then
  `RuntimeError`); 60 golden-section steps; no computation of `w` when `eps >= 1/2`; the cache per `(t, eps)`. Each
  is checked in section 2, and none changes a cap beyond the tolerance 1e-9 of step 3. Recommendation: accept them, and
  Claude adds one sentence with these constants to the spec's "Computation of `A_1`", since the spec describes the
  code. Blocks no step.
- **Q5. Names of the linear cap rows.** This plan and the rainflow plan use `name="rel"` (rows `rel_{i}_{l}_{k}`,
  the name of the quadratic rows too); the GIW plan uses `rel_ig` for the IG cap. Each cell has one model, so every
  choice gives unique names. Recommendation: `rel` for every model, so that one prefix finds the reliability rows of
  any cell. With `rel_gm`, step G3 changes one string, and `test_dynamics_and_cap` and `model_cases.py` two lines
  each. Blocks no step.
- **Answers** (Claude, 29 Oct 2026; `wp6_interface.md`, last section): Q1 option (1), for both caps; Q2 option (1);
  Q3 option (1); Q4 accepted, with spec sentences (section 8); Q5 `rel` for every model.

## 8. Review (29 Oct 2026)

Read: the diffs of `base.py`, `__init__.py` and `tests/test_base.py`; `gamma.py` and `tests/test_gamma.py` in full
against sections 1 and 2 and the spec's Gamma section; the Task C diff of `config.py` and `tests/test_config.py`; the
four other builders; the progress files of Tasks G and C. **State found:** Task G followed the plan exactly. `base.py`
lost the 1026 lines of the block and the 4 docstring lines (1030 deletions, no insertion). `__init__.py` holds the
docstring only. T1 has the one agreed line. `gamma.py` and `tests/test_gamma.py` equal the reference files byte for
byte, apart from the line endings. Task C applied section 5 as written, with the IG check placed before it.

| File:line | What was wrong | Change |
|---|---|---|
| `config.py:815-824` (IG check) | `ig_mean_cap` was imported for every input with replacement, also without IG cells: with gurobipy installed, `load_config` of any fleet loaded `inverse_gaussian.py`, `base` and `gurobipy` | the block runs under `if ig.any():`, with a comment line as in the Gamma block. Checked: a Gaussian input no longer loads the module |
| `config.py:18-20` (docstring) | named the Gamma cap only | names both caps and says that both modules load without Gurobi |
| `tests/test_config.py:678` (banner) | cited `merge/wp6_plan_giw.md`, a record that the merge branch does not carry | "Mean caps on a replacement day: inverse Gaussian and Gamma cells (spec, "Consistency Checks")" |
| `gamma.py:189-201` (style only) | `ctx.extras["gamma"]` and trailing equation comments, unlike the GIW builders | `ctx.extras[self.name]`, `T = ctx.T`, the comments `# Eqs. gm_mu_carry ... gm_anchor_keep` and `# Eq. gm_row` above the calls; section 1 shows the new code |

No defect in the rows, the cap, the removal or the tests.

- **Point 1, `config.py` and `tests/test_config.py`.** Both checks run after the Markov check, inside `np.errstate`,
  only when replacement is allowed, with `_le` (the tolerance of spec 525). Each calls the function of its builder
  module, so the check and the row use one cap. The Gamma check pairs mean and rate after `_lookup`, as the builder
  does through `ctx.increment`. The IG test's cap 0.5493234 equals `ig_mean_cap(0.15, 1, 0.1)`; the IG numerics are
  the GIW planner's review. Both tests pass, and so does `test_example_loads` (its Gamma cell has cap 0.6727, above
  `replacement_mu = 0.02`). With gurobipy refused, `import fleet_management` and `load_config` on IG and Gamma inputs
  work, and `base` stays unloaded.
- **Point 2, consistency.** The five builders agree on: the order of the helper calls (selector rows, `mu` rows
  first, reliability rows, `v_max_user`, repair cost); the row names `rel` (linear caps of IG, Gamma, Markov,
  Chernoff; the quadratic helper's own `rel_*`) and `v_max_user` (G, W, Cantelli, Bernstein); the `NonConvex` rule
  (set to 2 by G, W and rainflow when a quadratic row is `exact`, never by IG and Gamma); anchors and selectors through
  the helpers. The guard is in IG and Gamma only, as decided; G, W and rainflow import `base` directly, and nothing
  imports them without Gurobi. No defect found in another builder. Style differences in `rainflow.py` only: its
  constants go to `ctx.extras["rainflow_rows"]` (the others use `ctx.extras[self.name]`), and its `add_cell` has no
  equation comments.
- **Point 3, spec sentences** (spec on `lore`, "Computation of $A_1$", lines 1291-1303). Replace the three items by:

```latex
\item $\hat A$: the root of $Q(\hat A, t) = \varepsilon$, by bisection on $\hat A$ (\texttt{gammaincinv} solves for the second argument and does not apply). The interval starts as $[0, \max(1, 2t)]$; while $Q(b, t) \le \varepsilon$ at its upper end $b$, it becomes $[b, 2b]$. The bisection stops when the interval is at most $10^{-12}$ times its upper end, and returns the lower end, so $Q(\hat A, t) \le \varepsilon$.
\item $w(A)$: maximum of $c \mapsto Q(A/c, t/c)$ over a logarithmic grid of $401$ values of $c$ in $[\min(10^{-4}, t/100),\, 1]$. While the best grid point is the smallest value $c_0$ of the grid, the grid is replaced by $401$ logarithmic values in $[c_0/100,\, c_0]$, at most $10$ times; then the computation stops with an error. A golden-section search of $60$ steps in $\log c$, between the neighbours of the best grid point, refines the maximum; $w(A)$ is the largest value found.
\item If $\hat A < t$ and $w(\hat A) \le \varepsilon$: $A_1 = \hat A$. Otherwise: bisection on $A$ in $[\varepsilon t,\, \min(\hat A, t))$ until the interval is at most $10^{-9}$ times its upper end; $w(\varepsilon t) \le \varepsilon$ holds. For $\varepsilon \ge 1/2$ the test $w(A) \le \varepsilon$ holds without computing $w$, because $w(A) < 1/2$ for every $A < t$. The bisection returns the lower end of its last interval.
```

  After the list, add: "$A_1$ is computed once for each pair $(t, \varepsilon)$ in a run and then reused.
  \texttt{prepare} computes $\beta^*$ and the cap $A_1(\beta^*\tau, \varepsilon)/\beta^*$ of each cell and keeps them in
  \texttt{ctx.extras["gamma"]}. A value $t$ that is not finite and positive, or $\varepsilon \notin (0, 1)$, raises
  \texttt{ValueError}. The functions of the cap need numpy and scipy only: \texttt{gamma.py} imports \texttt{base}
  only if \texttt{gurobipy} is installed and registers its builder only then, so \texttt{config.py} imports the cap
  for the check of the replacement state without Gurobi." Package Layout (spec 193): "gamma.py computes the cap A_1"
  becomes "gamma.py computes beta* and the cap A_1 and loads without gurobipy" (to merge with the IG wording).
- **Point 4, removed names.** No module of the package imports a removed name (`GammaModel`, `GammaAction`,
  `maximum_reliable_*`, `_validate_positive`, the old block's functions); only the two old test files do (grep of
  `src`, `tests`, `examples`, `experiments`). WP7 modules still read data that only the removed `extract` hook or the
  removed inputs gave, all known to WP7 (`merge/wp4_plan.md` G47, `merge/wp5_plan.md` section 5), none an import error:
  `gamma_utils/gamma_replay_validator.py:75-83` and `gamma_tail_validator.py:187-216` (keys `gamma_shape_bound`,
  `gamma_tail_bound`, `gamma_beta_bound`, `gamma_mean_latch`, `gamma_shape_latch`, `gamma_calibration`, and
  `cfg.gamma_beta_bound`); `greedy_warm_start.py:76` (imports `gamma_repeated_calibration`, removed in WP1, inside a
  function) and `:120-152`; `horizon_sweep.py:29-62, 426-492` and `utils/mixed_plotter.py:460` (`gamma_calibration*`,
  `gamma_formulation`). `gamma_utils/gamma_diagnostics.py` has no caller in the package; `examples/regression` and
  `experiments` call it.
- **Test counts (29 Oct 2026, this computer).** `tests/test_gamma.py`: 21 passed, 15 skipped (the model tests need
  gurobipy). `tests/test_config.py`: 57 passed (55 old, the IG test, the Gamma test). `model_cases.py`: 88 checks,
  `failures: 0`. `names_check.py`: `base.py` 0 undefined names; `gamma.py` only the expected `base`. Not run here: the
  15 model tests of `test_gamma.py` and `tests/test_base.py` (WP8, with Gurobi).
