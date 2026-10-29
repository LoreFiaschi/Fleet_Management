# WP6 plan: the rainflow builder

Status (29 Oct 2026): done. Plan carried out (Task R) and reviewed by the planner (section 5). Next: Claude reviews
the diff and commits; WP8 runs `tests/test_rainflow.py` with Gurobi.

## 0. Scope and rules for the executor

- **Where.** Code in the worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge` (branch
  `merge/lore-into-main`). Authority: `spec/spec.tex` of that worktree, section "Rainflow Model" (lines 1304-1541),
  "Linear Forms of the Reliability Rows" (851-911), "Encoding of the Dynamics" (770-825). Line numbers below refer to
  it. The `lore` copy of the spec differs only outside these sections (checked 29 Oct 2026).
- **Files.** The task replaces `src/fleet_management/degradation_model/rainflow.py` and creates
  `tests/test_rainflow.py`. Nothing else changes. No git command that changes state; no commit.
- **Contract.** The builder calls only the helpers of `base.py` (section 2 of `merge/wp5_plan.md`, in the `lore`
  checkout). It does not change `base.py` or `config.py`. No module imports `rainflow.py` (checked 29 Oct 2026);
  `base._load_builders` imports it lazily, so `import fleet_management` still works without Gurobi.
- **Code style.** As `base.py` after WP5 (section 0 of `merge/wp5_plan.md`): section banners `# ====...`, a one-line
  docstring summary plus a short paragraph, `float(...)` around every value read from an array, plain `for` loops,
  few comments, each citing a spec label (`# Eq. rf_markov`). Lines of at most 100 characters. CRLF line endings,
  as every other file of the package.
- **Safety rules.** No installs. Do not delete any file. Do not spawn sub-agents. Temporary files only in the
  scratchpad `C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp6\rainflow\`.
  Never put the user's name or email address in any request.
- **Local tools** (this computer has no Gurobi and no pytest; run with `python`, from the folder of the tool):
  - `wp6\rainflow\rf_rows.py <path of rainflow.py>`: loads `base.py` and the builder through the linear emulator
    `wp5\fakegp`, builds the hand case of section 3 with the schedule fixed, and compares trajectories, statuses and
    row names. Linear forms only. Prints `failures: N`.
  - `wp5\names_check.py <path>`: global names that a function uses but nobody defines. Prints two counts.
  - `wp6\rainflow\proto_rainflow.py`: the prototype that section 2.1 copies; `fixture_check.py` checks the fixture of
    section 3.2; `t12_check.py` checks T12 and the segment counts of T4; `hand_values.py` computes section 3.1. If
    the scratchpad folder is gone, skip these tools and say so in the report.

## 1. Per bound: what the builder builds

**Common to every bound** (spec 1333-1470). Day index `k = 0, ..., T-1` in code; spec day `k + 1`.

- **Mean `mu`.** The shared variable `ctx.mu_var`, range `tau`. Day increment `day_increment(..., "mu")`; repair
  factor `1 - rho`; start `mu_0`; replacement `replacement_mu`; starting anchor `anchor_mu_0` (default `mu_0`).
  Rows: Eqs. `rf_mu_carry`, `rf_mu_inf`, `rf_mu_one`, `rf_mu_repl`, and the anchor rows `rf_anchor_set`,
  `rf_anchor_keep`.
- **Encoding** `bigm` (default) or `indicator`; **laws** `ardinf` and `ard1` for every bound (decision "Names and set
  of repair laws"). Both come from `base.add_quantity_rows`, with the selector rows of `base.add_selector_rows`.
- **Loop rows** (Eq. `a_loop`): one per tracked quantity and one per anchor, from `add_quantity_rows`.
- **Repair cost** (Eq. `a_z`): lower rows only, at cells with `C_R > 0`, from `base.add_repair_cost_rows`.
- **Starting and replacement values** come from `FleetConfig` through `ctx.initial[q]`, `ctx.replacement[q]`,
  `ctx.anchor_start[q]`. `config.py` sets the defaults (`R_0 = 0`, `cgf_0 = s mu_0`, `R_new = 0`,
  `cgf_new = s mu_new`, anchors equal to the initial values) and checks the ranges.
- **Natural ranges** (big-M constants and variable bounds) come from `ctx.state_range[q]` (`config._derive`).
- **No `extract` hook.** The shared extraction writes `mu`, `v`, `R`, `K` and their anchors (spec result table).

**Per bound.** `kappa = ln(1/eps)`; `b` = `cfg.support_const` (Eq. `rf_bsupport`, computed by `config._derive`);
`c = kappa b / 3`; `s` = `cfg.s_chernoff`.

| Bound | Second quantity: increment, factor `a`, start; replacement; range | Reliability row (spec label) | Forms | `NonConvex = 2` | `v_max_user` row |
|---|---|---|---|---|---|
| `markov` | none | `mu <= eps tau` (`rf_markov`) | `exact` only | no | no |
| `cantelli` | `v`: `day_increment("v")`, `(1-rho)^2`, `v_0`; `replacement_v`; `eps tau^2/(1-eps)` | `v <= c2 (tau-mu)^2`, `c2 = eps/(1-eps)`; side `mu <= tau` (`rf_cantelli`) | exact, tangent, pwl | if `exact` | if given |
| `hoeffding` | `R`: `day_increment("support", squared=True)`, `(1-rho)^2`, `R_0`; `R_new`; `2 tau^2/kappa` | `R <= c2 (tau-mu)^2`, `c2 = 2/kappa`; side `mu <= tau` (`rf_hoeffding`) | exact, tangent, pwl | if `exact` | no |
| `bernstein` | `v`: `day_increment("v")`, `(1-rho)^2`, `v_0`; `replacement_v`; `tau(tau-2c)/(2 kappa)` | `v <= c2 (tau-mu)^2 + c1 (tau-mu)`, `c2 = 1/(2 kappa)`, `c1 = -b/3`; side `mu <= tau - c` (`rf_bernstein`) | exact, tangent, pwl | if `exact` | if given |
| `chernoff` | `K`: `day_increment("cgf")`, `1 - rho`, `cgf_0`; `cgf_new`; `s tau - kappa` | `K <= s tau + ln eps` (`rf_chernoff`) | `exact` only | no | no |

- Under `ard1` the second quantity has its anchor (`g^v`, `g^R` or `g^K`, Eqs. `rf_v_one`, `rf_R_one`, `rf_K_one`).
  The Chernoff law `ard1`, `K+ = (1-rho) K + rho g^K`, is new (an upper bound; scoping, Proposition "Day maps of the
  statistics").
- The quadratic rows go through `base.add_quadratic_reliability_rows(ctx, i, l, q, c2=..., c1=..., mu_side=...)`,
  which builds the side row in every form (`rel_side_i_l_k`) and the row in the form `ctx.impl_of[i, l]`
  (spec table 866-873). The linear rows go through `base.add_upper_cap_rows(ctx, i, l, q, cap, name="rel")`.
- `config.py` writes `reliability_impl = "exact"` at Markov and Chernoff cells, whatever the input. So `NonConvex`
  and the segment binaries must look at the bound, not only at `impl_of`.
- The Chernoff cap equals the upper bound `range_K` of `K`. The row is redundant with that bound; the spec builds it
  (question 3).

**Differences from `main`'s builder** (worktree `rainflow.py`; each change follows the spec or the decision named).

| `main` (line) | New builder |
|---|---|
| imports `add_base_constraints`, `add_maintenance_gating`, `make_accessor`, ... that WP5 removed (25-38) | `base` helpers of WP5 only |
| entry point `solve` (55-76), demo `__main__` (701-714) | removed: the spec has no per-model solve; no module imports them |
| `replacement_as_new` default true, applied in the builder (104-109) | `config.py` applies it; default false (decision "Replacement") |
| `ard1` refused at Chernoff cells (122-127) | accepted, with the anchor `g^K` (decision "Names and set of repair laws") |
| indicator equalities only (217-338) | one-sided rows, `bigm` or `indicator` (decision "Encoding of the dynamics") |
| starting anchors 0 (222, 244, 259, 268, 303) | `anchor_*_0`, default the initial value (decision "Starting value of the anchor") |
| `R` and `K` start at 0 and reset to 0 (290, 312, 327, 337) | `R_0`, `R_new`, `cgf_0`, `cgf_new` (decision "Replacement", Defects A and B) |
| `z` as an equality on repair and replacement days (277-286) | lower rows, repair days only (decision "Repair cost on replacement days") |
| Bernstein `b` = largest increment support (130, 394) | `cfg.support_const`: also `support_0` and `support_new` (decision "Rainflow tail bounds") |
| Bernstein side row `mu <= tau` (454) | `mu <= tau - c` in every form (same decision) |
| `tangent_ref` clipped to [0, 1]; unknown form becomes `exact` (478, 561-570) | the helper raises on an unknown form; `config.py` checks `tangent_ref` (decision "Linear forms") |
| `pwl` binaries created in `add_cell` (511) | `base.add_pwl_vars` in `prepare` |
| `_tighten_bounds`: reachability bounds (646-680) | not ported; the natural ranges (question 1) |
| no `v_max_user` row | the optional row of spec 1539-1540 |

**Differences from `lore`'s builder** (`lore`: `src/fleet_management/models/rainflow.py`).

| `lore` (line) | New builder |
|---|---|
| Cantelli and Bernstein only, key `tail_bound` (57-62) | the five bounds, key `bound_method` |
| one law (`lore`'s `ARD1`, the spec's `ardinf`), no anchors (44-48) | both laws, anchors under `ard1` |
| big-M form only (144-153) | `bigm` or `indicator`, same one-sided rows |
| `formulation`: `exact` or `lp`, the tangent at `mu = 0` without the side row (155-180) | `exact`, `tangent` (at `tangent_ref tau`), `pwl`; the side row in every form |
| `b = max(b_0, b_new, max b_inc)` always (26-28) | `support_new` enters only when replacement is allowed (spec 1500) |
| Bhatia-Davis checks in the builder (66-102) | Popoviciu checks `v <= b^2/4` in `config.py` (spec 578) |
| repair cost by a McCormick envelope with upper rows | lower rows only (Eq. `a_z`) |
| scalar `epsilon`; horizon `2H` | `epsilon` per cell; `T = H1 + H2` |
| loop rows on `mu` and `v` | also on every anchor |
| `v_max_user` row (182-183) | kept |

## 2. Executor tasks

One task, **Task R**, for one Sonnet executor: it writes `rainflow.py` first (the most valuable part), then
`tests/test_rainflow.py`. Re-read a file from disk before each edit. After each step, run the step's check; if it
fails, fix it before the next step. The two files are independent, so Claude may give R1-R2 and R3-R7 to two
executors if the limit of four agents allows it.

### Steps

1. **R1. Replace `rainflow.py`.** Write `src/fleet_management/degradation_model/rainflow.py` with exactly the code of
   section 2.1 (Write tool, whole file). It replaces all 714 lines of `main`'s module.
   Check: `python -c "import ast; ast.parse(open(r'<path>', encoding='utf-8').read())"` succeeds.
2. **R2. Line endings and local checks.** Convert the file to CRLF (Bash; tested on the prototype):
   `python -c "p=r'<path>'; b=open(p,'rb').read().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'); open(p,'wb').write(b)"`.
   Check: `python -c "b=open(r'<path>','rb').read(); print(b.count(b'\n'), b.count(b'\r\n'), max(len(x) for x in b.decode().splitlines()))"`
   prints two equal counts and a length of at most 100; `python names_check.py <path>` (in `wp5\`) prints 0 and 0;
   `python rf_rows.py <path>` (in `wp6\rainflow\`) prints `failures: 0`.
3. **R3. Head of the test file.** Write `tests/test_rainflow.py`: a verbatim copy of `fixture_draft.py`
   (section 3.2). Check: `ast.parse` succeeds; `names_check.py` prints 0 and 0.
4. **R4. Structure tests.** Append tests T1-T5 of section 3.3. Check: `ast.parse`; `names_check.py`.
5. **R5. Solve tests.** Append tests T6-T9 of section 3.3. Check: `ast.parse`; `names_check.py`.
6. **R6. Remaining tests.** Append tests T10-T12 of section 3.3. Convert the file to CRLF as in R2. Check:
   `ast.parse`; `names_check.py`; CRLF and line length as in R2.
7. **R7. Report** (at most 40 lines): the steps done, the output of every check, any deviation from this plan and
   its reason. If a check could not run, say which and why.

### 2.1 Code of `rainflow.py`

Checked through the emulator as `proto_rainflow.py` (294 checks, 0 failures). Copy it exactly.

```python
"""
Rainflow cell builder (spec, "Rainflow Model").

A rainflow cell tracks the mean ``mu`` and, by its tail bound
``bound_method``, at most one more quantity: the variance ``v`` (Cantelli,
Bernstein), the Hoeffding budget ``R`` (Hoeffding) or the Chernoff quantity
``K`` (Chernoff). Markov tracks the mean only. Under the law ``ard1`` every
tracked quantity has an anchor. Both repair laws are accepted for every bound.

The dynamics, replacement, anchor, loop and repair-cost rows come from the
helpers of ``base``. This module adds the reliability row of the bound, in the
form ``reliability_impl`` for a quadratic row (Cantelli, Hoeffding, Bernstein)
and in its exact form for a linear row (Markov, Chernoff), and the optional row
``v <= v_max_user`` at Cantelli and Bernstein cells.
"""

from __future__ import annotations

import math

import numpy as np

from fleet_management.degradation_model import base


# ===========================================================================
# Tail bounds
# ===========================================================================
# bound -> (tracked quantity besides mu, increment profile, squared) or None
BOUND_QUANTITY = {
    "markov": None,
    "cantelli": ("v", "v", False),
    "hoeffding": ("R", "support", True),
    "bernstein": ("v", "v", False),
    "chernoff": ("K", "cgf", False),
}
QUADRATIC_BOUNDS = ("cantelli", "hoeffding", "bernstein")


def row_constants(bound: str, tau: float, eps: float, b: float = math.nan,
                  s: float = math.nan) -> dict:
    """Constants of the reliability row of a rainflow cell.

    Quadratic row (Eq. a_quad): ``q``, ``c2``, ``c1``, ``mu_side`` for
    ``chi <= c2 (tau - mu)^2 + c1 (tau - mu)`` and ``mu <= mu_side``, with
    ``chi = state_vars[q]``. Linear row: ``q`` and ``cap`` for
    ``state_vars[q] <= cap``. ``b`` is the Bernstein support constant and ``s``
    the Chernoff parameter; each is read only by its bound.
    """
    kappa = math.log(1.0 / eps)
    if bound == "markov":
        return {"q": "mu", "cap": eps * tau}                                   # Eq. rf_markov
    if bound == "cantelli":
        return {"q": "v", "c2": eps / (1.0 - eps), "c1": 0.0, "mu_side": tau}  # Eq. rf_cantelli
    if bound == "hoeffding":
        return {"q": "R", "c2": 2.0 / kappa, "c1": 0.0, "mu_side": tau}        # Eq. rf_hoeffding
    if bound == "bernstein":
        c = kappa * b / 3.0                                                    # Eq. rf_bsupport
        return {"q": "v", "c2": 1.0 / (2.0 * kappa), "c1": -b / 3.0,
                "mu_side": tau - c}                                            # Eq. rf_bernstein
    if bound == "chernoff":
        return {"q": "K", "cap": s * tau + math.log(eps)}                      # Eq. rf_chernoff
    raise ValueError(f"unknown bound_method {bound!r}")


# ===========================================================================
# Cell builder
# ===========================================================================
class RainflowCellBuilder:
    """Rainflow implementation of the ``base.CellBuilder`` interface."""

    name = "rainflow"

    def prepare(self, ctx: base.FleetModel, cfg, cells, opts: dict) -> None:
        """Variables of the rainflow cells, their row constants and ``NonConvex``.

        The anchors of ``mu`` (ard1 cells); ``v``, ``R`` or ``K`` with its anchor,
        by bound; the selector ``n`` (indicator cells); the segment binaries
        (quadratic rows in the form ``pwl``). ``NonConvex = 2`` if a quadratic row
        is built in the form ``exact``. The row constants of each cell go to
        ``ctx.extras["rainflow_rows"]``.
        """
        cells = list(cells)
        by_bound = {bound: [] for bound in BOUND_QUANTITY}
        for i, l in cells:
            bound = str(ctx.bound_of[i, l])
            if bound not in BOUND_QUANTITY:
                raise ValueError(f"unknown bound_method {bound!r} at cell (i={i}, l={l})")
            by_bound[bound].append((i, l))

        base.add_state_vars(ctx, "mu", cells)
        for bound, own in by_bound.items():
            if BOUND_QUANTITY[bound] is not None:
                base.add_state_vars(ctx, BOUND_QUANTITY[bound][0], own)
        base.add_selector_vars(ctx, cells)
        quadratic = [cell for bound in QUADRATIC_BOUNDS for cell in by_bound[bound]]
        base.add_pwl_vars(ctx, quadratic)
        if any(str(ctx.impl_of[i, l]) == "exact" for i, l in quadratic):
            ctx.model.Params.NonConvex = 2

        support_const = np.asarray(cfg.support_const, dtype=float)
        s_chernoff = np.asarray(cfg.s_chernoff, dtype=float)
        v_max_user = np.asarray(cfg.v_max_user, dtype=float)
        rows = {}
        for i, l in cells:
            bound = str(ctx.bound_of[i, l])
            b = float(support_const[i, l]) if bound == "bernstein" else math.nan
            s = float(s_chernoff[i, l]) if bound == "chernoff" else math.nan
            if bound == "bernstein" and not math.isfinite(b):
                raise ValueError(f"support_const is not finite at cell (i={i}, l={l})")
            if bound == "chernoff" and not math.isfinite(s):
                raise ValueError(f"s_chernoff is not finite at cell (i={i}, l={l})")
            constants = row_constants(bound, float(ctx.tau[i, l]), float(ctx.eps[i, l]), b, s)
            constants["v_max_user"] = (float(v_max_user[i, l]) if constants["q"] == "v"
                                       else math.nan)
            rows[i, l] = constants
        ctx.extras["rainflow_rows"] = rows

    def add_cell(self, ctx: base.FleetModel, i: int, l: int) -> None:
        """Rows of one rainflow cell, in the call order of ``base``."""
        T = ctx.T
        bound = str(ctx.bound_of[i, l])
        rows = ctx.extras["rainflow_rows"][i, l]

        base.add_selector_rows(ctx, i, l)
        base.add_quantity_rows(ctx, i, l, "mu",
                               [base.day_increment(ctx, i, l, k, "mu") for k in range(T)])
        if BOUND_QUANTITY[bound] is not None:
            q, key, squared = BOUND_QUANTITY[bound]
            base.add_quantity_rows(
                ctx, i, l, q,
                [base.day_increment(ctx, i, l, k, key, squared=squared) for k in range(T)])
        if bound in QUADRATIC_BOUNDS:
            base.add_quadratic_reliability_rows(ctx, i, l, rows["q"], c2=rows["c2"],
                                                c1=rows["c1"], mu_side=rows["mu_side"])
        else:
            base.add_upper_cap_rows(ctx, i, l, rows["q"], rows["cap"], name="rel")
        if math.isfinite(rows["v_max_user"]):
            base.add_upper_cap_rows(ctx, i, l, "v", rows["v_max_user"], name="v_max_user")
        base.add_repair_cost_rows(ctx, i, l)


base.register_cell_builder("rainflow", RainflowCellBuilder())
```

Row and variable names that the code produces (all end with the cell and day indices, so they are unique):
`rel_i_l_k` (reliability row, linear or quadratic), `rel_side_i_l_k`, `rel_seg_*`, `rel_lo_*`, `rel_hi_*`,
`rel_cap_*` (pwl), `v_max_user_i_l_k`, and the names of the `base` helpers (`mu_carry_*`, `gK_set_lb_*`,
`loop_K_i_l`, `z_i_l_k`, ...).

## 3. Tests

`tests/test_rainflow.py`, pytest style, as `tests/test_base.py`. Every test needs `gurobipy`
(`pytest.importorskip`); WP8 runs the file on the licensed machine. Every model has fewer than 200 variables.

### 3.1 The hand case and its expected values

Two trains, one mission, one component, `H = [2, 2]` (`T = 4`). `tau = 1`, `rho = 0.5`, `mu_0 = 0.2`,
`replacement_mu = 0.1`, mean increment `0.3` for train 0 and `0` for train 1, `C_R = 2`, `C_rep = 3`, `C_M = 1`,
`C_D = 0`. Schedule: train 0 flies on day 1, is repaired on day 2, flies on day 3, is replaced on day 4; train 1 flies
on days 2 and 4. Train 1 never changes its state. The inputs of each bound (increments are those of train 0):

| Bound | `eps` that holds; that fails under `ard1` only | Second quantity: start; replacement; increment | Other inputs |
|---|---|---|---|
| `markov` | 0.7; 0.6 | none | -- |
| `cantelli` | 0.2; 0.11 | `v_0 = 0.004`; `replacement_v = 0.002`; `v = 0.01` | -- |
| `hoeffding` | 0.05; 0.01 | `R_0 = 0.01`; `R_new = 0.005`; `support = 0.2` (so `R` grows by 0.04) | -- |
| `bernstein` | 0.4; 0.3 | as Cantelli | `support = 0.25`, `support_0 = 0.3`, `support_new = 0.1`, so `b = 0.3` |
| `chernoff` | 0.5; 0.4 | `cgf_0 = 1.0`; `cgf_new = 0.5`; `cgf = 1.5` | `s_chernoff = 4.0` |

Starting anchors (`ard1`) equal the starting values. These inputs pass every check of `config.py` (Popoviciu,
`cgf >= s mu`, `s > kappa/tau`, natural ranges, Bernstein `tau > 2c` and tangent condition). Expected values of train
0 on days 1-4 (exact day maps; `hand_values.py` in the scratchpad):

| Quantity | `ard1`: state; anchor | `ardinf`: state |
|---|---|---|
| `mu` | 0.5, 0.35, 0.65, 0.1; 0.2, 0.35, 0.35, 0.1 | 0.5, 0.25, 0.55, 0.1 |
| `v` (Ca, Be) | 0.014, 0.0065, 0.0165, 0.002; 0.004, 0.0065, 0.0065, 0.002 | 0.014, 0.0035, 0.0135, 0.002 |
| `R` (Ho) | 0.05, 0.02, 0.06, 0.005; 0.01, 0.02, 0.02, 0.005 | 0.05, 0.0125, 0.0525, 0.005 |
| `K` (Ch) | 2.5, 1.75, 3.25, 0.5; 1.0, 1.75, 1.75, 0.5 | 2.5, 1.25, 2.75, 0.5 |
| `z` | 0, 0.15, 0, 0 | 0, 0.25, 0, 0 |

Margins of the reliability row at the worst day (cap minus state; `hand_values.py`). Holding `eps`: all positive,
smallest +0.0031 (Bernstein, tangent, `ard1`). Failing `eps` under `ard1`: Markov -0.05; Cantelli -0.0014 (exact),
-0.0041 (tangent), -0.0014 (pwl, 4 segments); Hoeffding -0.0068, -0.0166, -0.0071; Bernstein -0.0006, -0.0100,
-0.0009; Chernoff -0.166. Under `ardinf` the failing `eps` still holds (smallest margin +0.0108, Cantelli, pwl).

### 3.2 Head of the file (step R3)

Copy `wp6\rainflow\fixture_draft.py` of the scratchpad verbatim (214 lines, checked by `fixture_check.py` through
the emulator; `names_check.py` 0 and 0). If the file is gone, write the same from this description.

- Docstring: "Tests of the rainflow cell builder, `degradation_model/rainflow.py`." Then: the tests need `gurobipy`;
  the hand case (two trains, one mission, one component, `H1 = H2 = 2`, the schedule above); expected values from
  the exact day maps of the spec, section "Rainflow Model".
- Imports: `math`, `SimpleNamespace`, `numpy as np`, `pytest`; `gp = pytest.importorskip("gurobipy")`; then `base`
  and `rainflow` from `fleet_management.degradation_model`.
- `_BOUNDS`: the table of section 3.1. `_OPTIONS`: as in `test_base.py`, but
  `"gurobi_params": {"DualReductions": 0}` (a definite status for an infeasible case).
- `_cfg(bound, repair_model="ard1", *, reliability_impl="tangent", encoding="bigm", failing=False, **changes)`: the
  `FleetConfig` of `load_config` for the hand case. Selectors are `(2, 1)` object arrays; cell values `(2, 1)` float
  arrays, `NaN` where the key does not apply; `reliability_impl` is `"exact"` at Markov and Chernoff cells. Profiles
  `(2, 1, 1, 2)`: the value of `_BOUNDS` for train 0, zero for train 1, `None` where the bound has no such profile.
  Derived fields as `config._derive`: `support_const = b` (Bernstein), `range_mu = 1`, `range_v = eps/(1-eps)` (Ca)
  or `(1 - 2c)/(2 kappa)` (Be), `range_R = 2/kappa`, `range_K = s - kappa`. `changes` replace attributes after the
  defaults; a cell value in `changes` must be a `(2, 1)` array.
- `_build_schedule(L=1)`, `_SCHEDULE`, `_INFEASIBLE = ("infeasible", "inf_or_unbounded")`, `_EXPECTED`, `_START`
  (train 1: `mu 0.2`, `v 0.004`, `R 0.01`, `K 1.0`), `_Z`, `_RESULT_KEYS` (`q -> (state key, anchor key)` of the
  result), `_tracked(bound)`, `_build`, `_fix_schedule(ctx, schedule)`, `_min_states`, `_solve_fixed(cfg,
  schedule=_SCHEDULE)`, `_row_names`, `_stack(first, second)` (two one-component cases as one fleet with `L = 2`).

### 3.3 Test list

Tolerance `atol = 1e-6` for arrays (`np.testing.assert_allclose`). "Local" means checked on this computer by
`rf_rows.py` through the emulator; the others need Gurobi.

| Test | Cases | Assertions | Local |
|---|---|---|---|
| **T1** `test_builder_is_registered` | 1 | after `base._load_builders(["rainflow"])`, `base.get_cell_builder("rainflow")` is a `rainflow.RainflowCellBuilder` with `name == "rainflow"` | -- |
| **T2** `test_row_constants` | 5 bounds | `row_constants("markov", 1, 0.6) == {"q": "mu", "cap": 0.6}`; cantelli `(1, 0.2)`: `v`, `c2 = 0.25`, `c1 = 0`, side 1; hoeffding `(1, 0.05)`: `R`, `c2 = 2/ln 20`, `c1 = 0`, side 1; bernstein `(1, 0.4, b=0.3)`: `v`, `c2 = 1/(2 ln 2.5)`, `c1 = -0.1`, side `1 - 0.1 ln 2.5`; chernoff `(1, 0.5, s=4)`: `K`, `cap = 4 + ln 0.5`. Numbers by `pytest.approx` of these formulas | yes |
| **T3** `test_variables_only_where_the_bound_tracks_them` | bound x law (10) | for `q` in `v`, `R`, `K`: `len(ctx.state_vars[q])` is 8 if `q in _tracked(bound)`, else 0; `len(ctx.anchor_vars[q])` is 8 if also `ard1`, else 0; `len(ctx.anchor_vars["mu"])` is 8 under `ard1`, else 0 | yes |
| **T4** `test_segment_binaries_and_nonconvex` | 9 | `(bound, impl, len(ctx.pwl_w), NonConvex == 2)`: `(cantelli, exact, 0, True)`, `(hoeffding, exact, 0, True)`, `(bernstein, exact, 0, True)`, `(cantelli, tangent, 0, False)`, `(cantelli, pwl, 64, False)`, `(hoeffding, pwl, 64, False)`, `(bernstein, pwl, 64, False)`, `(markov, pwl, 0, False)`, `(chernoff, pwl, 0, False)`; the fixture turns `pwl` into `exact` at Markov and Chernoff, as `config.py` does | `pwl` and tangent cases |
| **T5** `test_rows_of_a_cell` | bound x law (10) | with `v_max_user = 0.05` at both cells: the `loop_*` row names are exactly `loop_{q}_{i}_0` for `q` in `_tracked(bound)`, `i` in 0, 1, plus `loop_g{q}_{i}_0` under `ard1`; `rel_side_0_0_0` exists exactly when the bound is quadratic; `rel_0_0_3` exists; `v_max_user_1_0_3` exists exactly at Cantelli and Bernstein; `z_0_0_1` exists | yes |
| **T6** `test_trajectories` | bound x law x encoding (20) | `_solve_fixed(_cfg(bound, law, encoding=encoding))`: status `"optimal"`; for each tracked `q`: train 0 equals `_EXPECTED`, train 1 equals `_START[q]` on every day; anchors likewise under `ard1`, all NaN under `ardinf`; untracked state and anchor keys all NaN; `z[0, 0] == _Z[law]`, `z[1, 0]` all 0 | yes |
| **T7** `test_reliability_row_binds` | 22: Mk and Ch with `exact`; Ca, Ho, Be with `exact`, `tangent`, `pwl` (`pwl_points = 4`); both laws | `failing=True`: status in `_INFEASIBLE` under `ard1`, `"optimal"` under `ardinf` | tangent, linear |
| **T8** `test_quadratic_forms_keep_the_trajectories` | Ca, Ho, Be x (`exact`, `pwl` with 4 segments) x law (12) | status `"optimal"`; `mu` and the second quantity of train 0 equal `_EXPECTED` | -- |
| **T9** `test_v_max_user_row` | law (2) | Cantelli with `v_max_user = 0.015` (below the `ard1` peak 0.0165, above the `ardinf` peak 0.0135): `ard1` infeasible, `ardinf` optimal | yes |
| **T10** `test_bernstein_side_row_is_tau_minus_c` | `b` in 0.3, 0.0 (2) | Bernstein, `ardinf`, `exact`; `mu_trans` of train 0 `[0.8, 0.0]` (shape `(2, 1, 1, 2)`); `mu`, `v`, `support`, `v_0`, `replacement_v` zero; `support_const = b`: the mean reaches `tau` on day 1 with `v = 0`. `b = 0.3`: status in `_INFEASIBLE`; `b = 0`: optimal. In both, `getConstrByName("rel_side_0_0_0").RHS == approx(1 - ln(2.5) b / 3)`. With `main`'s side row `mu <= tau`, `b = 0.3` would be feasible | -- |
| **T11** `test_two_bounds_in_one_fleet` | law (2) | `_stack(_cfg("cantelli", law), _cfg("chernoff", law))` with `_build_schedule(L=2)`: optimal; the keys of `state_vars["v"]` have `l = 0` only, those of `state_vars["K"]` `l = 1` only; `v[0, 0]`, `K[0, 1]`, `mu[0, 0]`, `mu[0, 1]` equal `_EXPECTED`; `v[:, 1]` and `K[:, 0]` all NaN | yes |
| **T12** `test_bad_cell_data_is_an_error` | 2 | `pytest.raises(ValueError)` from `_build`: `bound_method` `"weibull"` at a Cantelli case (match `unknown bound_method`); `support_const` NaN at a Bernstein case (match `support_const is not finite`) | yes |

Steps: R4 writes T1-T5, R5 writes T6-T9, R6 writes T10-T12. The "Local" checks were run on the prototype on
29 Oct 2026 (`rf_rows.py`, `fixture_check.py`, `t12_check.py`: 0 failures). Gurobi only: the `exact` form (the
emulator has no quadratic rows), `pwl` solves (free segment binaries), `NonConvex`, `getConstrByName`.

## 4. Questions

No question blocks a step: the plan follows each recommendation unless the user decides otherwise. The WP5 helpers
fit all five bounds; no new helper is needed (prototype run, section 3.3).

1. **`main`'s variable-bound tightening (`_tighten_bounds`, `main` 646-680).** The spec lets `prepare` "tighten
   variable bounds" (spec 247), and its variable table gives the domains `[0, M_s]` (spec 615-622).
   - Option 1 (recommended): do not port it. The variable bounds stay the natural ranges, set by
     `base.add_state_vars`.
   - Option 2: port it, with the reachability bounds corrected for the starting and replacement values.
   - Reasons for option 1 (the planner's analysis). `main`'s bounds come from the increments only, so under the merged rules
     they cut feasible schedules. Variance: `v_0 = 0`, `v = 0.001` per mission, `T = 4` give `v <= 0.004`; with
     `replacement_v = 0.005` (allowed: below `V^C`) every replacement becomes infeasible. Chernoff: `s = 5`,
     `tau = 1`, `eps = 0.1` (`range_K = 2.697`), `mu_0 = 0.4`, so `K_0 = 2.0`; a mission with `cgf = 0.62`; `T = 2`
     gives `K <= 1.24 < K_0`, which forces an intervention on day 1, while the spec's model also admits a flight
     (`K_1 = 2.62 <= 2.697`). A corrected bound is still unsafe for the one-sided rows: the tracked state may lie
     above the exact one, and the loop row may need that slack (decision "Encoding of the dynamics").
2. **Names of the linear rows.** This plan names the Markov and Chernoff rows `rel_i_l_k`, the name that
   `base.add_quadratic_reliability_rows` gives the quadratic row, and the optional row `v_max_user_i_l_k`.
   Recommendation: the GIW and Gamma plans use the same names (IG and Gamma caps `rel`; G and W `v_max_user`), so one
   name finds the reliability row of every cell. Claude aligns the three plans.
3. **Redundant rows that the spec builds.** The Chernoff row `K <= s tau + ln eps` equals the upper bound
   `range_K = s tau - kappa` of `K`. The side row `mu <= tau` of Cantelli and Hoeffding equals the upper bound of
   `mu`. Option 1 (recommended): build them, as the spec says; they cost `T` rows per cell, and every reliability row
   then has the name `rel`. Option 2: skip them, which changes the spec.
- **Answers (Claude, 29 Oct 2026; `wp6_interface.md`):** no tightening; the redundant rows are built; every cap row
  is named `rel` and the optional variance cap `v_max_user` in all models.

## 5. Review (29 Oct 2026)

Reviewed in full: `rainflow.py` (the executor's copy of section 2.1, byte-identical to the prototype apart from CRLF)
and `tests/test_rainflow.py` (head from `fixture_draft.py`; T1-T12 written by the executor), against this plan and
spec 1304-1542. All checks rerun on the current `base.py`, which no longer holds the old Gamma block.

**Defects found and corrected.** No wrong row, sign, increment, repair factor, start or replacement value, and no
wrong expected value. The defects are in coverage and in consistency.

| File:line (after the change) | What was wrong | Change |
|---|---|---|
| `test_rainflow.py:19` (executor) | `pytest.importorskip("gurobipy")` at module level: without gurobipy every test was skipped, `test_row_constants` included | guarded import of `gurobipy` and `base` (lines 26-34); `@_needs_gurobi` (`skipif`) on each test that builds a model |
| `rainflow.py:23` | plain `import base`: `row_constants` could not load without gurobipy | the guarded import of `inverse_gaussian.py` and `gamma.py` (lines 26-31); registration under `if base is not None` (157) |
| `rainflow.py:117, 123` | constants in `ctx.extras["rainflow_rows"]`; the other four builders use `ctx.extras[self.name]` | `ctx.extras[self.name] = {"rows": rows}` (125, 131) |
| `rainflow.py:125-148` | no `# Eq.` comments above the helper calls, unlike the other builders | comments citing `rf_mu_*`, `rf_v_*`/`rf_R_*`/`rf_K_*`, `rf_anchor_*`, the reliability rows and the optional row |
| tests (gap) | no test read the constants inside the model; a wrong argument passed to a helper went unseen, unless it changed a feasibility result | `test_reliability_rows_carry_the_constants` (429): RHS of `rel_0_0_k` and `rel_side_0_0_k` for the five bounds |
| tests (gap) | the claim "these inputs pass `config.py`" was not tested; the fixture could drift from `load_config` | `test_fixture_matches_load_config` (282): 51 fields (52 with `tangent_ref` at a quadratic bound) of `load_config(_input(...))` equal `_cfg(...)`, holding and failing `eps`, both laws (20 cases; no Gurobi) |
| tests (gap) | `row_constants` raising on an unknown bound, and loading without gurobipy, untested | `test_row_constants_rejects_an_unknown_bound` (299); `test_module_loads_without_gurobi` (264, subprocess, as `test_gamma.py`) |

The executor's T1-T12 are correct as written; T4 compares `NonConvex == 2` with Gurobi's default (-1), which is
right. **Mutation check** (`mutants.py`): 14 defects planted in `rainflow.py` are all caught by tests that run
locally. Examples: `main`'s Bernstein side row, the wrong sign of `c1`, a missing square on the Hoeffding increment,
`v_max_user` read at every bound, `NonConvex` at linear rows, no anchors. `main`'s side row is also caught by T10 with
Gurobi.

**Consistency with the other builders** (read in full; not edited).
- Same for all five after the changes: `prepare` calls `add_state_vars`, `add_selector_vars`, `add_pwl_vars` (quadratic
  cells); `add_cell` calls `add_selector_rows`, `add_quantity_rows`, then `add_quadratic_reliability_rows` or
  `add_upper_cap_rows(..., name="rel")`, the optional `v_max_user` row, `add_repair_cost_rows`. Constants in
  `ctx.extras[<model name>]`. `NonConvex = 2` only for an exact quadratic row (G, W: any `exact` cell; RF: quadratic
  bounds only, because `config.py` writes `exact` at Markov and Chernoff; IG, Ga: never).
- Differences that are not defects: `gaussian.py` and `wiener.py` import `base` without the guard (they hold no function
  needed without Gurobi; decision 1 concerns the caps). `gamma.py` puts its `# Eq.` comments at the end of the line.
- No defect found in `gaussian.py`, `wiener.py`, `inverse_gaussian.py` or `gamma.py`.

**Sentences for the spec** (the code does this; the Rainflow section does not say it).
1. After Eq. `rf_bsupport`: "`load_config` computes $b$ once per Bernstein cell; the builder reads this value."
2. Package layout or Cell-Builder Interface, with the matching sentence for `inverse_gaussian.py` and `gamma.py`:
   "`rainflow.py` also loads without `gurobipy`: its function `row_constants`, the constants of the reliability row
   of a cell, needs no Gurobi object; the module then registers no builder."
3. Optional, for all models (Cell-Builder Interface): "The reliability row of cell $(i,\ell)$ on day $k$ is named
   `rel_i_l_k`, its side row `rel_side_i_l_k`, and the optional variance cap `v_max_user_i_l_k`."
- Not needed: the builder's `ValueError` on an unknown `bound_method` or a non-finite $b$ or $s$; `load_config`
  rejects these inputs first.

**Final counts.** `tests/test_rainflow.py`: 16 test functions, 124 cases (580 lines, CRLF, longest line 97).
- Under pytest with gurobipy: 124 cases. Without gurobipy: 27 run, 97 are skipped.
- Local runs (`review_run.py`): emulator 95 passed, 29 need Gurobi, 0 failed; real package without gurobipy 27
  passed, 0 failed.
- Need Gurobi: T4 `exact` (3), T7 `exact` and `pwl` of the quadratic bounds (12), T8 (12), T10 (2). T10's logic also
  passes with its quadratic row reduced to the side row (`t10_check_review.py`).
- `rainflow.py`: 158 lines, CRLF; `rf_rows.py` 0 failures. `names_check_try.py` gives 0 for all five builders. It is
  `names_check.py` patched to see a guarded import; the original tool flags `base` in IG, Gamma and rainflow.
- `test_fixture_matches_load_config` reads the current `config.py`, which Task C is editing; rerun it after Task C.
