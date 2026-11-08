# WP6 plan: Gaussian, inverse Gaussian and Wiener builders

Status (29 Oct 2026): done. Plan carried out (Tasks G, IG, W, C) and reviewed by the planner (section 7). Next:
Claude reviews the diff, adds the spec sentences of section 7, and commits; WP8 runs the [Gurobi] cases.

## 0. Scope, sources and tools

- **Where.** Code in the worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge` (branch
  `merge/lore-into-main`, commit 5c8de19). Authority: `spec/spec.tex`; where the worktree's spec and the `lore` copy
  differ, the `lore` copy wins (`merge/wp6_interface.md`). Line numbers below are those of the worktree's spec.
- **Files.** Task G creates `src/fleet_management/degradation_model/gaussian.py` and `tests/test_gaussian.py`; Task
  IG creates `inverse_gaussian.py` and `tests/test_inverse_gaussian.py`; Task W creates `wiener.py` and
  `tests/test_wiener.py`. Task C (later, with the Gamma section) edits `config.py` and `tests/test_config.py`.
  Nothing else changes. No git command that changes state.
- **Code style.** As `base.py` after WP5 (section 0 of `merge/wp5_plan.md`): section banners `# ====...`, a one-line
  docstring summary plus a short paragraph, `float(...)` around every parameter read from an array, plain `for`
  loops, comments that cite the spec equation label (`# Eq. g_row`), lines of at most 100 characters. Every variable
  and row name is unique. Line endings: git normalises them (`core.autocrlf = true`), so LF from the Write tool is
  fine; no file may mix endings.
- **Safety rules for executors.** No installs. Do not delete any file. Temporary files only in the scratchpad folder
  `C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp6\giw\`.
  Never put the user's name or email address in any request. Do not spawn sub-agents.
- **Terms.** G, IG, W: the Gaussian, inverse Gaussian and Wiener models. `pwl`: the piecewise-linear form of a
  reliability row. CVaR cap: the inverse Gaussian cap built on the conditional tail mean `E[Z | Z > q]` (conditional
  value at risk).
- **Local tools** (in the scratchpad folder `wp6\giw\`; run with `python` from that folder):
  - `run_giw_tests.py <test file> [--proto] [name]`: runs a pytest-style test file with a stub `pytest` and the
    emulator `fakegp\` (a copy of WP5's, extended: it builds quadratic rows but does not solve them). A test that
    needs a real solve (quadratic row, free binaries) prints `NEEDS GUROBI`; it is not a failure. If `config.py` has
    no `ig_mean_cap` yet, the prototype of `ig_cap.py` is injected. `--proto` loads the planner's prototypes
    `proto\degradation_model\*.py` instead of the worktree's builders (planner only).
  - `ig_cap.py`: the IG cap functions of section 2 and their numeric checks. `hand_giw.py`: the expected arrays of
    section 5. `proto\tests\test_*.py`: the planner's prototypes of the three test files.
  - If the folder is gone: the plan text is complete without it; skip the local runs and say so in the report.

## 1. Per model: what the builder builds

All three builders use only the helpers of section 2 of `merge/wp5_plan.md`, in the call order fixed there. Day
index `k = 0, ..., T-1` (spec day `k + 1`). `kappa_G = ndtri(1 - eps)` (`scipy.special.ndtri`, the function that
`config.py` uses for `range_v`). None of the three needs an `extract` hook: the shared extraction reads every
`state_vars[q]` and `anchor_vars[q]`, and the spec's result table has no key of these models alone (spec 1701-1739).

### 1.1 Gaussian (G; spec 914-1017)

- **Tracked**: `mu`, `v`; at an `ard1` cell also `g_mu`, `g_v`. Increments `delta mu = day_increment(..., "mu")`,
  `delta v = day_increment(..., "v")` (profile `v`, present at every G cell).
- **`prepare`**: `add_state_vars(ctx, "mu", cells)` (anchors `g_mu`), `add_state_vars(ctx, "v", cells)` (`v` and
  `g_v`, range `V_max = range_v`), `add_selector_vars`, `add_pwl_vars`; store `cfg.v_max_user` as an `(F, L)` float
  array in `ctx.extras["gaussian"]["v_max_user"]` (NaN where not given); `ctx.model.Params.NonConvex = 2` if one of
  `cells` has `impl_of == "exact"` (spec 247, 1011-1012).
- **`add_cell`**: `add_selector_rows` (Eq. `a_ind_sum`); `add_quantity_rows(..., "mu", ...)` (Eqs. `g_mu_carry`,
  `g_mu_inf`, `g_mu_one`, `g_mu_repl`, `g_anchor_set`, `g_anchor_keep`, `a_loop`); `add_quantity_rows(..., "v", ...)`
  (Eqs. `g_v_carry` to `g_v_repl`, the `g_v` anchor rows, `a_loop`); `add_quadratic_reliability_rows(ctx, i, l, "v",
  c2=1/kappa_G^2, c1=0, mu_side=tau)` (Eq. `g_row`, forms `exact`, `tangent`, `pwl`; table spec 866-868); if
  `v_max_user` is finite, `add_upper_cap_rows(ctx, i, l, "v", v_max_user, name="v_max_user")` (spec 1016-1017);
  `add_repair_cost_rows` (Eq. `a_z`).
- **Differences from `lore`** (`models/gaussian.py`):
  - `lore` accepts only its "ARD1" (the spec's `ardinf`); the builder accepts `ardinf` and `ard1`, with the anchors
    `g_mu`, `g_v` and the `ard1` variance repair `(1-rho)^2 v + (1-(1-rho)^2) g_v` (Eq. `g_v_one`; decision "Names
    and set of repair laws").
  - `lore` writes big-M rows only; the builder also has the indicator form (input `encoding`).
  - `lore` builds `2H` days on doubled increments and loops `2H-1` on `H-1`; the builder has `T = H1 + H2` days with
    the phase lookup (Eq. `c_phase`) and loop rows `T-1` on `H1-1` for `mu`, `v`, `g_mu`, `g_v`.
  - `lore`'s forms are `exact` and `lp` (the tangent at `mu = 0`, without the side row `mu <= tau`); the builder has
    `exact`, `tangent` (at `tangent_ref * tau`), `pwl`, all with the side row (decision "Linear forms").
  - `lore`'s repair cost is a McCormick envelope (two upper rows and one lower row) at every cell and day; the builder
    has the lower row only, at cells with `C_R > 0`, on the days of `cost_days` (spec 1547-1560).
  - `lore` creates its own unbounded `mu_g`, `v_g` variables; the builder uses the shared `mu` in `[0, tau]` and `v`
    in `[0, V_max]`.
  - `lore` reads one scalar `epsilon`; the builder reads `eps` per cell.
  - `lore` builds replacement rows always; the builder only if `allow_replacement` (inside the helpers).
  - `lore`'s `validate_inputs` moves to `config.py` (done in WP4: three-sigma, `eps < 0.5`, ranges, replacement row).

### 1.2 Inverse Gaussian (IG; spec 1020-1106)

- **Tracked**: `mu`; at an `ard1` cell also `g_mu`. Increment `delta mu = day_increment(..., "mu")`.
- **`prepare`**: `add_state_vars(ctx, "mu", cells)`, `add_selector_vars`; store `ctx.extras["inverse_gaussian"]
  ["mu_bar"]`, a dict `(i, l) -> ig_mean_cap(eta, tau, eps)` over `cells` (section 2). No `pwl` binaries and no
  `NonConvex`: the row is linear, and `config.py` already sets `reliability_impl = "exact"` at IG cells (spec 375,
  1086-1087).
- **`add_cell`**: `add_selector_rows`; `add_quantity_rows(..., "mu", ...)` (Eqs. `ig_mu_carry` to `ig_anchor_keep`,
  `a_loop`); `add_upper_cap_rows(ctx, i, l, "mu", mu_bar, name="rel_ig")` (Eq. `ig_row`); `add_repair_cost_rows`.
- **Differences from `lore`** (`models/inverse_gaussian.py`):
  - The cap. `lore`: `mu_bar` solves `Pr(mu Z > tau) = eps`, so `mu_bar = tau / q`, found by `brentq` in `mu`, with a
    "non-binding" branch (cap `tau (1 - 1e-9)`) and an "unachievable" error. Spec and builder: the CVaR cap
    `mu_bar = tau / E[Z | Z > q]` (decision "Inverse Gaussian reliability cap"). Since `E[Z | Z > q] > E[Z] = 1`, the
    cap always lies in `(0, tau)`, so neither branch exists. It is 9.8 % to 22.5 % below `lore`'s cap at `eps = 0.01`
    (section 2).
  - `lore` accepts only "ARD1" (`ardinf`); the builder accepts both laws (decision of 26 Oct 2026 in "Follow-up
    decisions from writing the spec").
  - Encoding, horizon and loop, repair cost, shared `mu`, per-cell `eps`, `allow_replacement`, input checks: as for G.

### 1.3 Wiener (W; spec 1109-1189)

- **Tracked**: `mu`, `v`; at an `ard1` cell also `g_mu`; no variance anchor (spec 625, 754-755, 811-815).
  Increments `delta mu = day_increment(..., "mu")`; `delta v = sigma^2 * mission_count(ctx, i, k)` (Eq. `w_map_v`;
  there is no profile `v` at a W cell).
- **`prepare`**: `add_state_vars(ctx, "mu", cells)`, `add_state_vars(ctx, "v", cells, anchored=False)`,
  `add_selector_vars`, `add_pwl_vars`; store `cfg.sigma` and `cfg.v_max_user` as `(F, L)` float arrays in
  `ctx.extras["wiener"]`; `NonConvex = 2` as for G.
- **`add_cell`**: `add_selector_rows`; `add_quantity_rows(..., "mu", ...)` (Eqs. `w_mu_carry` to `w_anchor_keep`);
  `add_quantity_rows(..., "v", increments, repair_keeps_state=True)` (big-M: the single row Eq. `w_vrow` and Eq.
  `w_v_repl`; indicator: Eq. `a_ind_state` with `psi = v_{k-1}`; loop row of `v`); the Gaussian row as for G (spec
  1179-1183); the optional `v_max_user` row as for G; `add_repair_cost_rows`.
- **Differences from `lore`** (`models/wiener.py`, `maintenance/ara1.py`, `maintenance/ard1.py`):
  - `lore`'s "ARA1" is the spec's `ard1`, with another anchor: `lore` sets the anchor to the state *before* the repair
    (`mu_{k-1}`), and to `mu_new` on a replacement day, starting from `mu_0`. The spec's anchor is the state *after*
    the intervention (Eq. `a_anchor`), starting from `anchor_mu_0` (default `mu_0`). This is the known defect of
    `CLAUDE.md` ("an ARA1 repair cannot lower the state"), decided in "Anchor of the repair law".
  - `lore` requires `v_0 > 0`; the spec requires `v_0 >= 0` (checked in `config.py`).
  - The variance rows are `lore`'s (`v_k >= v_{k-1} + delta v - V_max x^r` and the replacement rows), now also in the
    indicator form.
  - Repair cost, forms, horizon and loop, shared variables, per-cell `eps`, `allow_replacement`: as for G.

## 2. The IG cap

- **Formula** (spec 1080-1106). `Z ~ IG(1, 1/eta)`, distribution function
  `F(x) = Phi(r (x - 1)) + e^{2/eta} Phi(-r (x + 1))`, `r = 1/sqrt(eta x)`. `q` solves `Pr(Z > q) = eps`. The tail
  mean is `E[Z; Z > q] = F(1/q)` (substitute `w = 1/z`: `z f(z) dz = f(w) dw`, because `f(1/z) = z^3 f(z)`), so
  `mu_bar = tau / E[Z | Z > q] = tau eps / F(1/q)`.
- **Computation** (the code is in section 4):
  - `_ig_unit_log_cdf(x, eta)`: `ln F(x) = logaddexp(log_ndtr(r (x-1)), 2/eta + log_ndtr(-r (x+1)))`. Both terms
    are positive, so the sum loses nothing; `e^{2/eta}` never overflows.
  - `_ig_unit_log_sf(x, eta)`: `ln Pr(Z > x) = a + log1p(-exp(b - a))`, with `a = log_ndtr(-r (x-1))` and
    `b = 2/eta + log_ndtr(-r (x+1))`. This avoids the cancellation of `1 - F(x)` when `eps` is small.
  - `_ig_unit_quantile(eta, eps)`: `brentq` on `u = ln q`, function `_ig_unit_log_sf(e^u, eta) - ln eps`, bracket
    `[-50, ln(1 + sqrt(eta (2/eps - 1)))]`. The right end holds by Cantelli's inequality: `Var Z = eta`, so
    `Pr(Z > b) <= eps/2`. Tolerances `xtol = 1e-14`, `rtol = 1e-15`, `maxiter = 200`. A sign check before `brentq`
    raises `ValueError` (it never fired on the grid below).
  - `ig_mean_cap(eta, tau, eps) = tau * exp(ln eps - _ig_unit_log_cdf(1/q, eta))`.
  - The spec allows `brentq` on `ln q` or `scipy.stats.invgauss.isf`. `brentq` was chosen: at `eta = 1e-6`, `eps = 0.5`
    the high-precision survival (mpmath, 60 digits) at `invgauss.isf`'s quantile is off by a relative `1.6e-6`, at
    ours by `2e-13`; at `eta = 1e-3`, `eps = 1e-12`, `1.4e-5` against `5e-14` (scratchpad `ig_mp.py`).
- **Placement**: `config.py`, new section "Model constants"; the IG builder imports `ig_mean_cap` from there. The
  spec's "Package Layout" does not change. Reason and other options: question Q1.
- **Numeric check** (scratchpad `ig_cap.py`, `ig_extreme.py`; output in `ig_cap_out.txt`):
  - Grid `eta` in {1e-3, ..., 1e3} (11 values) times `eps` in {1e-6, ..., 0.9} (9 values): `q` agrees with
    `invgauss.isf` to a relative `2.2e-12`; the closed-form tail mean agrees with the quadrature of
    `int_q^inf z f(z) dz` (scipy `quad`, relative tolerance 1e-12) to `4e-14`; every cap lies in `(0, tau/q]`.
  - Extremes `eta` in {1e-6, ..., 1e6}, `eps` in {1e-12, ..., 0.999}: no failure; `0 < mu_bar <= tau/q`.
  - Monte Carlo (2e6 draws of `invgauss`): `E[Z | Z > q]` 2.61568 against the sample mean 2.61699 (`eta = 0.5`,
    `eps = 0.1`); 5.76507 against 5.76704 (`eta = 2`, `eps = 0.05`).
  - Cost against `lore`'s cap `tau/q` at `eps = 0.01`: 9.8 % (`eta = 0.1`), 18.2 % (0.5), 22.5 % (1). These are the
    values of `decisions.md`, item "Inverse Gaussian reliability cap".
- **Reference values** (used by the tests):

| `eta` | `tau` | `eps` | `q` | `mu_bar` | `lore`'s `tau/q` |
|---|---|---|---|---|---|
| 0.5 | 1.0 | 0.1 | 1.8859953361 | 0.3823093163 | 0.5302240047 |
| 0.1 | 1.0 | 0.01 | 1.9488253601 | 0.4627586840 | 0.5131296115 |
| 2.0 | 2.0 | 0.05 | 3.5525960458 | 0.3469170694 | 0.5629685937 |
| 0.15 | 1.0 | 0.1 | 1.5123693654 | 0.5493233512 | 0.6612141338 |

## 3. Executor tasks

Three tasks for Sonnet executors, run in parallel: Task G, Task IG, Task W. Each creates one builder module and one
test file; no task edits an existing file. Task C (section 4) runs after them. The code blocks of this plan are
identical to the tested prototypes (checked by the scratchpad script `plan_blocks_check.py`).

**Checks after every step** (scratchpad folder `wp6\giw\`; `<wt>` is the worktree):
- `python -c "import ast; ast.parse(open(r'<file>', encoding='utf-8').read())"`;
- `python ..\..\wp5\names_check.py <file>`: expected `undefined names outside the Gamma block: 0`;
- no line longer than 100 characters.

### Task G: `gaussian.py` and `tests/test_gaussian.py` (3 steps)

**G1. Builder.** Create `<wt>\src\fleet_management\degradation_model\gaussian.py` with exactly this text (it is the
scratchpad file `proto\degradation_model\gaussian.py`; copy that file if it exists, then compare):

```python
"""
Gaussian cell builder (spec, section "Gaussian Model").

A cell tracks the mean ``mu`` and the variance ``v`` of its degradation; under
the law ``ard1`` also their anchors ``g_mu`` and ``g_v``. The day maps are
Eqs. g_map_mu and g_map_v, built by the helpers of ``base`` in the cell's
encoding. The reliability row is Eq. g_row, ``kappa_G^2 v <= (tau - mu)^2``
with ``kappa_G = Phi^{-1}(1 - epsilon)``, in the form ``reliability_impl``;
the optional row ``v <= v_max_user`` holds where the input gives it.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri

from fleet_management.degradation_model import base


# ===========================================================================
# Cell builder
# ===========================================================================
class GaussianCellBuilder:
    """Rows of the cells with ``model = gaussian``."""

    name = "gaussian"

    def prepare(self, ctx: base.FleetModel, cfg, cells, opts: dict) -> None:
        """Variables of ``v``, the anchors, selectors and segment binaries; ``NonConvex``."""
        base.add_state_vars(ctx, "mu", cells)
        base.add_state_vars(ctx, "v", cells)
        base.add_selector_vars(ctx, cells)
        base.add_pwl_vars(ctx, cells)
        ctx.extras[self.name] = {"v_max_user": np.asarray(cfg.v_max_user, dtype=float)}
        if any(str(ctx.impl_of[i, l]) == "exact" for i, l in cells):
            ctx.model.Params.NonConvex = 2

    def add_cell(self, ctx: base.FleetModel, i: int, l: int) -> None:
        """Dynamics, replacement, anchor and loop rows; reliability rows; repair cost."""
        T = ctx.T
        tau = float(ctx.tau[i, l])
        kappa = float(ndtri(1.0 - float(ctx.eps[i, l])))
        base.add_selector_rows(ctx, i, l)
        # Eqs. g_mu_carry ... g_anchor_keep
        base.add_quantity_rows(ctx, i, l, "mu",
                               [base.day_increment(ctx, i, l, k, "mu") for k in range(T)])
        base.add_quantity_rows(ctx, i, l, "v",
                               [base.day_increment(ctx, i, l, k, "v") for k in range(T)])
        # Eq. g_row
        base.add_quadratic_reliability_rows(ctx, i, l, "v", c2=1.0 / kappa ** 2, c1=0.0,
                                            mu_side=tau)
        v_max_user = float(ctx.extras[self.name]["v_max_user"][i, l])
        if math.isfinite(v_max_user):
            base.add_upper_cap_rows(ctx, i, l, "v", v_max_user, name="v_max_user")
        base.add_repair_cost_rows(ctx, i, l)


base.register_cell_builder("gaussian", GaussianCellBuilder())
```

Check: the three checks above; `python -c "import sys; sys.path.insert(0, r'<wt>\src'); import fleet_management;
print('fleet_management.degradation_model.gaussian' in sys.modules)"` prints `False` (the package still imports
without Gurobi, and the builder loads lazily).

**G2. Tests.** Create `<wt>\tests\test_gaussian.py` as section 5.1 lists it. The scratchpad file
`proto\tests\test_gaussian.py` implements that list: copy it, then check it against section 5.1. Check:
`python run_giw_tests.py <wt>\tests\test_gaussian.py` prints `11 passed, 0 failed, 3 need Gurobi`; the three are
`test_free_schedule`, `test_reliability_forms['exact', True]`, `test_reliability_forms['pwl', True]`.

**G3. Report.** List any deviation from this plan, and the output of the G2 check.

### Task IG: `inverse_gaussian.py` and `tests/test_inverse_gaussian.py` (3 steps)

**IG1. Builder.** Create `<wt>\src\fleet_management\degradation_model\inverse_gaussian.py` (scratchpad
`proto\degradation_model\inverse_gaussian.py`). It is G1's file with these changes:
- Docstring: "Inverse Gaussian cell builder (spec, section "Inverse Gaussian Model")." then the paragraph: a cell
  tracks `mu`, under `ard1` also `g_mu`; day map Eq. ig_map; reliability row the linear cap Eq. ig_row,
  `mu <= mu_bar = tau / E[Z | Z > q]`, `Z ~ IG(1, 1/eta)`, `q` its `(1 - epsilon)`-quantile; `config.ig_mean_cap`
  computes `mu_bar` once per cell and lives in `config.py` because the input check of the replacement state needs it
  without Gurobi; `reliability_impl` is ignored.
- Imports: only `from __future__ import annotations`, `from fleet_management.config import ig_mean_cap` and the
  `base` import (no `math`, `numpy`, `ndtri`).
- Class `InverseGaussianCellBuilder`, `name = "inverse_gaussian"`, docstring "Rows of the cells with
  ``model = inverse_gaussian``."
- `prepare` (docstring "The mean anchors and selectors; the cap ``mu_bar`` of every cell."): `add_state_vars(ctx,
  "mu", cells)`, `add_selector_vars(ctx, cells)`, then
  `ctx.extras[self.name] = {"mu_bar": {(i, l): ig_mean_cap(float(cfg.eta[i, l]), float(ctx.tau[i, l]),
  float(ctx.eps[i, l])) for i, l in cells}}`. No `add_pwl_vars`, no `NonConvex`.
- `add_cell` (docstring "Dynamics, replacement, anchor and loop rows; the cap row; repair cost."): `T = ctx.T`;
  `add_selector_rows`; comment `# Eqs. ig_mu_carry ... ig_anchor_keep` and the `"mu"` call of G1; comment
  `# Eq. ig_row` and `base.add_upper_cap_rows(ctx, i, l, "mu", ctx.extras[self.name]["mu_bar"][i, l],
  name="rel_ig")`; `add_repair_cost_rows`.
- Last line: `base.register_cell_builder("inverse_gaussian", InverseGaussianCellBuilder())`.

Check: the three checks above and G1's import check (module name `inverse_gaussian`). Until Task C, `config.py` has
no `ig_mean_cap`, so the builder module cannot be imported outside `run_giw_tests.py` (which injects it); this is
expected.

**IG2. Tests.** Create `<wt>\tests\test_inverse_gaussian.py` as section 5.2 lists it (scratchpad
`proto\tests\test_inverse_gaussian.py`). Check: `python run_giw_tests.py <wt>\tests\test_inverse_gaussian.py` prints
`8 passed, 0 failed, 1 need Gurobi` (`test_free_schedule`).

**IG3. Report**, as G3.

### Task W: `wiener.py` and `tests/test_wiener.py` (3 steps)

**W1. Builder.** Create `<wt>\src\fleet_management\degradation_model\wiener.py` (scratchpad
`proto\degradation_model\wiener.py`). It is G1's file with these changes:
- Docstring: "Wiener cell builder (spec, section "Wiener Model")." then: a cell tracks `mu` and `v`, under `ard1`
  also `g_mu` and no variance anchor; a mission day adds `sigma^2` to the variance (Eq. w_map_v); a repair acts on the
  mean only, so the variance falls only at a replacement (Eq. w_vrow in the big-M form); the reliability row is the
  Gaussian row Eq. g_row in the form `reliability_impl`; the optional row `v <= v_max_user`.
- Class `WienerCellBuilder`, `name = "wiener"`, docstring "Rows of the cells with ``model = wiener``."
- `prepare` (docstring "Variables of ``v`` (no anchor), the mean anchors, selectors and segment binaries."): the
  second call is `base.add_state_vars(ctx, "v", cells, anchored=False)`; the extras are
  `{"sigma": np.asarray(cfg.sigma, dtype=float), "v_max_user": np.asarray(cfg.v_max_user, dtype=float)}`.
- `add_cell`: after `kappa`, `sigma = float(ctx.extras[self.name]["sigma"][i, l])`; the first comment reads
  `# Eqs. w_mu_carry ... w_anchor_keep`; the `"v"` call becomes, under the comment `# Eqs. w_map_v, w_vrow, w_v_repl`:
  `base.add_quantity_rows(ctx, i, l, "v", [sigma ** 2 * base.mission_count(ctx, i, k) for k in range(T)],
  repair_keeps_state=True)`. The rest as G1.
- Last line: `base.register_cell_builder("wiener", WienerCellBuilder())`.

Check: the three checks above and G1's import check (module name `wiener`).

**W2. Tests.** Create `<wt>\tests\test_wiener.py` as section 5.3 lists it (scratchpad `proto\tests\test_wiener.py`).
Check: `python run_giw_tests.py <wt>\tests\test_wiener.py` prints `10 passed, 0 failed, 1 need Gurobi`
(`test_free_schedule`).

**W3. Report**, as G3.

## 4. config.py change

Task C (one Sonnet executor, after Tasks G, IG, W and the Gamma builder; it also applies the Gamma plan's section).
Files: `<wt>\src\fleet_management\config.py`, `<wt>\tests\test_config.py`. The scratchpad script
`config_patch.py` applies C1 to C4 to a copy and runs the whole `test_config.py` on it: `57 passed, 0 failed`
(55 existing tests and the 2 new ones). Spec: 557 (replacement state on a replacement day), 1080-1106.

**C1. Imports.** Replace the line `from scipy.special import ndtri` by the two lines
`from scipy.optimize import brentq` and `from scipy.special import log_ndtr, ndtri`.

**C2. New section.** Insert, just before the banner `# ---...` / `# Consistency checks`, this section:

```python
# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
def _ig_unit_log_cdf(x: float, eta: float) -> float:
    """ln F(x) of Z ~ IG(1, 1/eta), x > 0 (spec, "Inverse Gaussian Model")."""
    r = 1.0 / math.sqrt(eta * x)
    return float(np.logaddexp(log_ndtr(r * (x - 1.0)), 2.0 / eta + log_ndtr(-r * (x + 1.0))))


def _ig_unit_log_sf(x: float, eta: float) -> float:
    """ln (1 - F(x)) of Z ~ IG(1, 1/eta), x > 0, without the cancellation of 1 - F."""
    r = 1.0 / math.sqrt(eta * x)
    first = float(log_ndtr(-r * (x - 1.0)))
    second = 2.0 / eta + float(log_ndtr(-r * (x + 1.0)))
    return first + math.log1p(-math.exp(second - first))


def _ig_unit_quantile(eta: float, epsilon: float) -> float:
    """The (1 - epsilon)-quantile q of Z ~ IG(1, 1/eta): Pr(Z > q) = epsilon.

    Root of ln Pr(Z > e^u) = ln epsilon in u = ln q, by brentq on [-50, ln b], with
    b = 1 + sqrt(eta (2/epsilon - 1)): Var Z = eta, so Pr(Z > b) <= epsilon/2 (Cantelli).
    """
    target = math.log(epsilon)

    def gap(u):
        return _ig_unit_log_sf(math.exp(u), eta) - target

    lo, hi = -50.0, math.log(1.0 + math.sqrt(eta * (2.0 / epsilon - 1.0)))
    if not (gap(lo) > 0.0 > gap(hi)):
        raise ValueError(f"no quantile bracket for eta={eta}, epsilon={epsilon}")
    return math.exp(brentq(gap, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200))


def ig_mean_cap(eta: float, tau: float, epsilon: float) -> float:
    """Mean cap mu_bar = tau / E[Z | Z > q] of an inverse Gaussian cell (spec, Eq. ig_row).

    E[Z; Z > q] = F(1/q), so mu_bar = tau * epsilon / F(1/q), computed in log space.
    The builder ``degradation_model/inverse_gaussian.py`` imports it; it lives here
    because the check of the replacement state needs it without Gurobi.
    """
    q = _ig_unit_quantile(eta, epsilon)
    return tau * math.exp(math.log(epsilon) - _ig_unit_log_cdf(1.0 / q, eta))
```

**C3. The check.** In `_check_ranges`, after the Markov check (`check(m["markov"], ... "must be <= epsilon*tau at a
markov cell")`), inside the `with` block, add:

```python
        ig = m["inverse_gaussian"]
        mu_bar = np.full(mu.shape, np.nan)
        for i, l in zip(*np.nonzero(ig)):
            mu_bar[i, l] = ig_mean_cap(float(cfg.eta[i, l]), float(cfg.tau[i, l]),
                                       float(cfg.epsilon[i, l]))
        check(ig, _le(mu, mu_bar), "replacement_mu",
              "must be <= the mean cap tau/E[Z | Z > q] at an inverse_gaussian cell")
```

The function returns before this point when `allow_replacement` is false, as the spec asks. `_check_models` has
already checked `eta > 0`, `eps` in `(0, 1)`, `tau > 0`. Message, for example: `cell (i=0, l=0)
model=inverse_gaussian: 'replacement_mu' must be <= the mean cap tau/E[Z | Z > q] at an inverse_gaussian cell.`

**C4. Tests.** Append to `tests/test_config.py`, after `test_replacement_rows_off`:

```python
# ---------------------------------------------------------------------------
# Inverse Gaussian mean cap (merge/wp6_plan_giw.md, section 2)
# ---------------------------------------------------------------------------
def test_inverse_gaussian_cap():
    from scipy.stats import invgauss

    # (eta, tau, epsilon, q, mu_bar), computed by brentq and checked by quadrature
    for eta, tau, eps, q, mu_bar in ((0.5, 1.0, 0.1, 1.8859953361, 0.3823093163),
                                     (0.1, 1.0, 0.01, 1.9488253601, 0.4627586840),
                                     (2.0, 2.0, 0.05, 3.5525960458, 0.3469170694)):
        assert abs(config._ig_unit_quantile(eta, eps) - q) <= 1e-9 * q
        assert abs(config.ig_mean_cap(eta, tau, eps) - mu_bar) <= 1e-9 * mu_bar
        assert abs(invgauss.sf(q, eta, scale=1.0 / eta) - eps) <= 1e-8 * eps
    for eta in (1e-3, 1.0, 1e3):
        for eps in (1e-9, 0.5, 0.99):
            cap = config.ig_mean_cap(eta, 1.0, eps)
            assert 0.0 < cap <= 1.0 / config._ig_unit_quantile(eta, eps)


def test_inverse_gaussian_replacement_cap():
    # cap 0.5493234 for eta = 0.15, epsilon = 0.1, tau = 1
    rule = "must be <= the mean cap"
    _fails("inverse_gaussian", None, "replacement_mu", rule, replacement_mu=0.56)
    load_config(_small("inverse_gaussian", replacement_mu=0.54))
    load_config(_small("inverse_gaussian", replacement_mu=0.56, allow_replacement=False))
```

Check: `python ..\..\wp4\run_tests.py <wt>\tests\test_config.py` prints `57 passed, 0 failed` plus the Gamma plan's new
tests; then `python run_giw_tests.py <wt>\tests\test_inverse_gaussian.py` no longer prints the line "note:
config.ig_mean_cap injected" and gives the counts of IG2.

## 5. Tests

Pytest style, as `tests/test_base.py`: plain `assert`, `pytest.mark.parametrize`, `np.testing.assert_allclose(...,
atol=1e-6)`, `gp = pytest.importorskip("gurobipy")` before the package imports. Unlike `test_base.py`, the input goes
through `load_config`, so the builders read the fields that WP4 fills. Expected arrays: `hand_giw.py` (exact day
maps; output in `hand_giw_out.txt`). Every model has at most about 150 variables and rows (free Gurobi licence).

**Shared parts of the three files** (each file has its own copy; there is no `conftest.py`):
- `_data(**changes) -> dict`: the input mapping of the hand case: `F 2, M 1, H [2, 2]`, the model, `repair_model
  "ard1"`, `tau 1.0, epsilon 0.1, rho 0.5`, `C_M 1.0, C_R 2.0, C_rep 3.0, C_D 0.0`, `verbose 0, mip_gap 0.0,
  time_limit 60.0`, and the model's keys below. `changes` replace keys.
- `_build(data) -> (cfg, ctx)`: `load_config`, `base.build_fleet(cfg, base.resolve_run_options(cfg))`,
  `ctx.model.update()`.
- `_fix(ctx, schedule)`: `LB = UB` on every variable of `ctx.x`, `ctx.m_rep`, `ctx.r_rep` (1 if its key is listed in
  `schedule["x"]`, `["m"]`, `["r"]`, else 0), and on every selector of `ctx.nb` (`1 - m - r`).
- `_solve_fixed(data, schedule=_SCHEDULE) -> (ctx, out)`: `_build`, `_fix`, objective "minimise the sum of `mu`, `v`
  (G, W) and `z`" (with fixed binaries the one-sided rows then give the exact trajectory), `optimize`,
  `base.extract_solution(ctx, cfg)`.
- `_row_names(ctx)`: `{row.ConstrName for row in ctx.model.getConstrs()}`.
- `_INFEASIBLE = ("infeasible", "inf_or_unbounded")`, compared with `out["status"]`.
- `_SCHEDULE` (keys `(train, activity, day index)`, activity 0 = depot): `x` at `(0,1,0), (0,0,1), (0,1,2), (0,0,3),
  (1,1,1), (1,1,3)`; `m` at `(0,0,1)`; `r` at `(0,0,3)`. Train 0 flies, is repaired, flies, is replaced; train 1 is
  idle, flies, is idle, flies. Wiener adds `x (1,0,2)` and `r (1,0,2)` (train 1 replaced on day index 2).

Local runs (`run_giw_tests.py`, emulator): every test with linear rows and fixed binaries. Licensed machine (WP8):
also the tests marked **[Gurobi]** (quadratic rows, `pwl` binaries, or free binaries through `solve_mixed`).

### 5.1 `tests/test_gaussian.py` (11 local cases, 3 [Gurobi])

Keys: `model "gaussian"`, `reliability_impl "tangent"`, `mu_0 0.2`, `replacement_mu 0.1`,
`mu [[[[0.3, 0.3]]], [[[0.0, 0.0]]]]`, `v [[[[0.01, 0.01]]], [[[0.0, 0.0]]]]` (shape `(F, L, M, H2)`).

| Train 0 | `ard1` | `ardinf` |
|---|---|---|
| `mu` | 0.5, 0.35, 0.65, 0.1 | 0.5, 0.25, 0.55, 0.1 |
| `v` | 0.01, 0.0025, 0.0125, 0.0 | same |
| `g_mu`; `g_v` | 0.2, 0.35, 0.35, 0.1; 0.0, 0.0025, 0.0025, 0.0 | all NaN |
| `z` | 0.0, 0.15, 0.0, 0.0 | 0.0, 0.25, 0.0, 0.0 |

Train 1: `mu = 0.2`, `v = 0` on every day.
- `test_registered`: `base.get_cell_builder("gaussian")` is a `gaussian.GaussianCellBuilder` with `name "gaussian"`.
- `test_dynamics` (encoding `bigm`/`indicator` x law `ard1`/`ardinf`): `_solve_fixed`; status `"optimal"`; the table;
  train 1; `ardinf`: `g_mu`, `g_v` all NaN; `R`, `K` all NaN.
- `test_rows_and_variables` (defaults): 8 `v`, 8 `g_mu`, 8 `g_v`; `nb` and `pwl_w` empty; rows `rel_side`, `rel`,
  `v_carry`, `v_repair`, `gv_set_lb`, `z` with suffix `_0_0_{k}`, `k = 0..3`; `loop_mu_0_0`, `loop_v_0_0`,
  `loop_gmu_0_0`, `loop_gv_0_0`; no row `v_max_user*`; `NumQConstrs == 0`; `getattr(Params, "NonConvex", None) != 2`.
- `test_exact_form_is_quadratic` (`exact`): `NonConvex == 2`; the `QCName`s of `getQConstrs()` are exactly
  `rel_{i}_0_{k}`, `i = 0, 1`, `k = 0..3` (the emulator builds them without solving).
- `test_pwl_form` (`pwl`, `pwl_points 4`): 32 `w`; rows `rel_seg_0_0_{k}`; no quadratic row; `NonConvex != 2`.
- `test_v_max_user` (`v_max_user [[x], [None]]`; `None` means "not given" at train 1, spec on `lore` line 353):
  rows `v_max_user_0_0_{k}` exist, none `v_max_user_1_*`; `x = 0.013` optimal; `x = 0.012` infeasible (`v` reaches
  0.0125).
- `test_reliability_forms` (`epsilon 0.001`; at `mu = 0.65`, `v = 0.0125` the exact cap is 0.0128278, the tangent at
  `0.5 tau` 0.0104717, the `pwl` segment `[0.625, 0.75]` of `P = 8` 0.0126806): `tangent` infeasible (local);
  `exact` optimal **[Gurobi]**; `pwl` optimal **[Gurobi]**.
- `test_free_schedule` **[Gurobi]**: `_data(mu=0.3, v=0.01, reliability_impl="exact")`, `base.solve_mixed`; status
  `"optimal"`; `kappa_G^2 v <= (1 - mu)^2 + 1e-6` and `mu <= 1 + 1e-6` everywhere; loop rows hold on `mu`, `v`,
  `g_mu`, `g_v` (index `T-1` against `H1-1`, tolerance 1e-6).

### 5.2 `tests/test_inverse_gaussian.py` (8 local cases, 1 [Gurobi])

Keys: `model "inverse_gaussian"`, `eta 0.5`, `mu_0 0.1`, `replacement_mu 0.05`,
`mu [[[[0.15, 0.15]]], [[[0.0, 0.0]]]]`. Cap `_MU_BAR = 0.3823093163` (section 2).

| Train 0 | `ard1` | `ardinf` |
|---|---|---|
| `mu` | 0.25, 0.175, 0.325, 0.05 | 0.25, 0.125, 0.275, 0.05 |
| `g_mu` | 0.1, 0.175, 0.175, 0.05 | all NaN |
| `z` | 0.0, 0.075, 0.0, 0.0 | 0.0, 0.125, 0.0, 0.0 |

Train 1: `mu = 0.1` on every day. `v`, `g_v`, `R`, `K` all NaN.
- `test_registered`: as G, class `InverseGaussianCellBuilder`.
- `test_dynamics` (encoding x law): as G, with the table above.
- `test_cap_rows` (`reliability_impl "pwl"`, which an IG cell ignores): rows `rel_ig_{i}_0_{k}` for `i = 0, 1`,
  `k = 0..3` with `RHS` within 1e-9 of `_MU_BAR` and `Sense "<"`; no row `rel_side*`; `state_vars["v"]` and `pwl_w`
  empty; `NumQConstrs == 0`; `NonConvex != 2`.
- `test_cap_boundary` (`H [1, 1]`, `mu [[[[a]]], [[[0.0]]]]`, schedule `x (0,1,0), (1,1,1)`, no `m`, no `r`): train 0
  holds `mu = 0.1 + a` on both days. `a = 0.99 _MU_BAR - 0.1` optimal; `a = 1.01 _MU_BAR - 0.1` infeasible.
- `test_free_schedule` **[Gurobi]**: `solve_mixed` on `_data(mu=0.15)`; status `"optimal"`; `mu <= _MU_BAR + 1e-6`
  everywhere; loop rows on `mu`, `g_mu`.

### 5.3 `tests/test_wiener.py` (10 local cases, 1 [Gurobi])

Keys: `model "wiener"`, `reliability_impl "tangent"`, `sigma 0.1` (both trains: `config.py` requires `sigma > 0`),
`mu_0 0.2`, `replacement_mu 0.1`, `mu [[[[0.3, 0.3]]], [[[0.0, 0.0]]]]`. Train 1 flies on day indices 1 and 3, so
its variance grows; without the replacement on day index 2 its loop row `v_3 <= v_1` fails (spec 1185-1189).

| | `ard1` | `ardinf` |
|---|---|---|
| train 0 `mu` | 0.5, 0.35, 0.65, 0.1 | 0.5, 0.25, 0.55, 0.1 |
| train 0 `g_mu` | 0.2, 0.35, 0.35, 0.1 | all NaN |
| train 0 `z` | 0.0, 0.15, 0.0, 0.0 | 0.0, 0.25, 0.0, 0.0 |
| `v`, trains 0 and 1 | 0.01, 0.01, 0.02, 0.0; 0.0, 0.01, 0.0, 0.01 | same |
| train 1 `mu` (`g_mu` under `ard1`) | 0.2, 0.2, 0.1, 0.1 | same |

`g_v` all NaN under both laws.
- `test_registered`: as G, class `WienerCellBuilder`.
- `test_dynamics` (encoding x law): the table.
- `test_rows_and_variables`: 8 `v`, 8 `g_mu`, no `g_v`; rows `rel_side`, `rel`, `v_carry`, `v_repl_lb`,
  `mu_repair`, `gmu_set_lb` with suffix `_0_0_{k}`; no row `v_repair_0_0_{k}` (Eq. `w_vrow` is one row);
  `loop_v_0_0`, `loop_gmu_0_0`; no name starting with `gv`, `loop_gv` or `v_max_user`; `NonConvex != 2`.
- `test_exact_form_is_quadratic`: as G.
- `test_loop_needs_replacement`: `_SCHEDULE` without `x (1,0,2)` and `r (1,0,2)`: infeasible.
- `test_v_max_user` (`[[x], [None]]`): rows as G; `x = 0.021` optimal, `x = 0.019` infeasible (`v` reaches 0.02).
- `test_free_schedule` **[Gurobi]**: `_data(mu=0.3, reliability_impl="exact")`; status `"optimal"`; the Gaussian row
  and the loop rows on `mu`, `v`, `g_mu`; every train that flies in the operating phase is replaced there.

Checked by the planner: all local cases pass on the prototypes; three mutations of the prototypes (Wiener variance
repaired like a Gaussian one; Gaussian `c2 = 1/kappa_G` instead of `1/kappa_G^2`; IG cap set to `tau`) each make at
least one local case fail.

## 6. Questions

The spec, the decisions and the WP5 helpers fit the three builders: every generic row has its helper, and no helper
is missing. I found no contradiction in the spec for G, IG and W. Three points need Claude's agreement; none blocks
a step if the recommendation holds.

- **Q1. Where `ig_mean_cap` lives.** `config.py` needs the cap without Gurobi; a builder module imports `base`, hence
  `gurobipy`. The interface file suggests "in the builder module, imported by `config.py` inside the check function":
  that import still loads `base` and `gurobipy`, so `load_config` would fail without Gurobi on every input with an IG
  cell (`test_example_loads` has one).
  - Options: (1) `config.py`, new section "Model constants"; the builder imports it from there (recommended: no
    change to the spec's "Package Layout", one function for the check and the row). (2) A new Gurobi-free module, for
    example `degradation_model/inverse_gaussian_cap.py`: the spec's layout (spec 192) gains a file. (3) `_derive`
    stores a field `cfg.ig_mean_cap` and the builder reads it: the `FleetConfig` interface of WP4 and WP5 changes.
  - The Gamma plan meets the same problem for `A_1` (spec 193: "gamma.py computes the cap A_1"). One rule for both
    caps would be clearer; Claude aligns the two plans.
  - Steps that change with the answer: IG1 (one import line), C2 (the module that receives the code).
- **Q2. Order of Task IG and Task C.** The IG builder imports `ig_mean_cap`, which Task C adds; `wp6_interface.md`
  runs Task C after the builders. Options: (1) keep that order; `run_giw_tests.py` injects the prototype, so IG2 runs
  locally, and nothing is committed before Task C (recommended); (2) apply C1 and C2 before Task IG. No step is
  blocked.
- **Q3. Row names shared with the rainflow plan.** The optional row `v <= v_max_user` is named
  `v_max_user_{i}_{l}_{k}` here; the rainflow builder adds the same row at Cantelli and Bernstein cells. The same
  prefix there keeps the names uniform (each cell has one model, so the names stay unique either way). The IG cap row
  is `rel_ig_{i}_{l}_{k}`; the quadratic rows keep the helper's names `rel_*`. No step is blocked.
- **Answers.** Claude's decisions 1 and 2 (`wp6_interface.md`, 29 Oct 2026) override sections 2 to 5 where they
  differ: the cap functions live in `inverse_gaussian.py` under a guarded import of `base`; `config.py` imports
  `ig_mean_cap` inside its check; the IG cap row is named `rel`.

## 7. Review (29 Oct 2026)

Read in full: the six files of Tasks G, IG, W, against this plan, decisions 1 and 2, and spec 914-1191. `base.py`
lost its old Gamma block meanwhile (deletions only; the helpers of WP5 section 2 are unchanged); all checks rerun.

**State found.** `gaussian.py`, `wiener.py`, `tests/test_gaussian.py`, `tests/test_wiener.py` are byte-identical to
the prototypes. `inverse_gaussian.py` follows decisions 1 and 2; its four cap functions have the code of C2 (AST
comparison, docstrings aside). No defect in the rows, constants or helper calls of the three builders.

**Defects corrected**

| File:line | What was wrong | Change |
|---|---|---|
| `tests/test_inverse_gaussian.py:18` | module-level `importorskip("gurobipy")`: without Gurobi pytest skipped the cap test too | Gamma's pattern: module imports need no Gurobi; `_gurobi()` skips per test; new `test_cap_without_gurobi` (subprocess with `gurobipy` blocked: the module loads, `base is None`, the cap value, `load_config` of an IG input with replacement) |
| `inverse_gaussian.py:9-12` | docstring: "the functions below it" (they come before `ig_mean_cap`); "so the module also loads" reversed cause and effect | "before it"; "so the module must load without gurobipy" |
| scratchpad `run_giw_tests.py` | still injected `config.ig_mean_cap` (obsolete after decision 1) | injection removed |

**Point 1 (tests without Gurobi).** Only the IG file has tests that need no Gurobi; it now follows `test_gamma.py`.
Run as pytest would without `gurobipy` (scratchpad `run_without_gurobi.py`): IG 2 passed, 9 skipped. Every test of
the G and W files builds a Gurobi model, so their module-level skip stays.

**Point 2 (consistency with `gamma.py`, `rainflow.py`).** The same in all builders: the guard of `base` (IG and Gamma
have the same text; G, W and rainflow need no guard, since `config.py` imports nothing from them); cap rows `rel`; the
row `v_max_user`; the `NonConvex` rule (exact form at a quadratic row); the helper call order; per-cell inputs read
from `cfg` in `prepare` into `ctx.extras`. Differences that are style only: rainflow's extras key is `rainflow_rows`
(the others use the model name); Gamma writes its equation comments at the end of the line. I found no defect in
`gamma.py` or `rainflow.py`: their row constants match the spec's table (spec 866-873) and Eqs. `rf_markov`,
`rf_chernoff`, `gm_row`.

**Point 3 (spec sentences).** Replace the paragraph "Computation of $\bar\mu$" (worktree spec 1089-1106, `lore`
1092-1109) by:

```latex
\textbf{Computation of $\bar\mu$} (offline, once per cell, by \texttt{ig\_mean\_cap(eta, tau, epsilon)}
in \texttt{inverse\_gaussian.py}). With $r(x) = 1/\sqrt{\eta x}$, the distribution function of $Z$ is
\[ F(x) = \Phi\bigl(r(x)\,(x - 1)\bigr) + e^{2/\eta}\,\Phi\bigl(-r(x)\,(x + 1)\bigr). \]
Every $\ln\Phi$ below is \texttt{scipy.special.log\_ndtr}.
\begin{enumerate}[leftmargin=1.5em, nosep]
  \item $\ln F(x) = \mathrm{logaddexp}\bigl(\ln\Phi(r(x)(x-1)),\ 2/\eta + \ln\Phi(-r(x)(x+1))\bigr)$.
  \item $\ln\bigl(1 - F(x)\bigr) = a + \ln\bigl(1 - e^{\,b-a}\bigr)$ (\texttt{log1p}), with
        $a = \ln\Phi(-r(x)(x-1))$ and $b = 2/\eta + \ln\Phi(-r(x)(x+1))$.
  \item $q$: the root in $u = \ln q$ of $\ln\bigl(1 - F(e^u)\bigr) = \ln\varepsilon$, by
        \texttt{scipy.optimize.brentq} on $\bigl[-50,\ \ln\bigl(1 + \sqrt{\eta\,(2/\varepsilon - 1)}\bigr)\bigr]$,
        with \texttt{xtol} $= 10^{-14}$, \texttt{rtol} $= 10^{-15}$, \texttt{maxiter} $= 200$; a
        \texttt{ValueError} is raised if the function does not change sign on the bracket.
  \item $\bar\mu = \tau\,\exp\bigl(\ln\varepsilon - \ln F(1/q)\bigr)$, since $\mathbb{E}[Z;\, Z > q] = F(1/q)$.
\end{enumerate}
The module imports \texttt{base} under a guard: without \texttt{gurobipy} it loads, \texttt{base} is
\texttt{None} and no builder is registered. \texttt{config.py} imports \texttt{ig\_mean\_cap} inside the check
of the replacement state (Section~\ref{sec:consistency_checks}).
```

The spec's alternatives (bisection, `invgauss.isf`) and the quadrature sentence leave the spec: the code uses neither.
The reasons go to the scoping document: Cantelli's inequality for the right end (`Var Z = eta`, so
`Pr(Z > b) <= eps/2`), the identity `E[Z; Z > q] = F(1/q)`, the accuracy figures of section 2. Two further
sentences: in "Package Layout" (spec 193) "cell builders; gamma.py computes the cap A_1, inverse_gaussian.py the cap
mu_bar"; in "Gaussian Model", "Reliability row": "$\kappa_G$ is computed by \texttt{scipy.special.ndtri}$(1 -
\varepsilon)$, as $V_{\max}$ in \texttt{config.py}". Nothing else that the G, IG, W builders do is missing.

**Point 4 (`config.py`, Task C; not edited).** Correct: the IG block sits after the return for `allow_replacement`
false, inside `np.errstate`; the message and the test (0.56 refused, 0.54 and `allow_replacement` false accepted)
fit the cap 0.5493234; `tests/test_config.py` 57 passed. Two defects found in Task C's version, both already
corrected by the Gamma reviewer while this review ran (checked on the file, 29 Oct 2026):
- the import of `inverse_gaussian` ran at every `load_config` with replacement, also without IG cells (with
  `gurobipy` installed it then loaded `base` for any fleet); now under `if ig.any():`, as the Gamma block
  (`config.py:815-823`);
- the module docstring named only the Gamma cap; it now names both caps (`config.py:18-20`).

**Final counts** (29 Oct 2026). Emulator (`run_giw_tests.py`): G 11 passed, 0 failed, 3 need Gurobi; W 10, 0, 1; IG
10, 0, 1. Without `gurobipy`: IG 2 passed, 9 skipped; G and W skipped as a whole. `tests/test_config.py` 57 passed.
`ast` and `names_check.py`: 0 undefined names (IG: 3 reports of `base`, the guarded import, expected). Longest line
96 characters. For WP8 (licensed machine): the cases marked [Gurobi] in section 5.
