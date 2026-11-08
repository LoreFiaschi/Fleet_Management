# Start of the repetition (tighter loop rows): plan

Status (8 Nov 2026): plan written by Claude; the user adopted the change as the formulation (`merge/decisions.md`,
"Start of the repetition"). Next: tasks Y1 (code core) and Y2 (reference model) in parallel; Y3 (tools) after coder 5
of step 3; the Euler run; then the spec and scoping texts.

- **Why.** The one-sided rows refuse safe schedules that replace a component on day `H1`: the replacement fixes
  `s_{H1}`, so the loop row `s_T <= s_{H1}` cannot use a higher start for the repetitions (WP8 tightness check).
- **Idea.** Decouple the start of the repetitions from day `H1`: a variable `y >= s_{H1}` is the state at the start
  of every repetition; the operating phase starts from `y`, and the loop row returns to `y`.
- **Safety** (the same induction as now): the true state on day `H1` is at most `s_{H1} <= y`; the tracked operating
  phase from `y` bounds the true one; the loop row gives `s_T <= y`, so the next repetition starts at most at `y`.

## 1 Rows (the interface; every file follows it)

For every cell `(i, l)`, every tracked quantity `s` of the cell (`mu`, and `v`, `R` or `K` where tracked) and, at an
`ard1` cell, every anchor `g^s` (none for the Wiener variance):

| Item | Definition |
|---|---|
| Variables | `y^s_{il}` and `y^{g,s}_{il}`, continuous, in `[0, M_s]` (the natural range of the quantity); one per cell and quantity or anchor; no day index |
| Start rows | `y^s >= s_{H1}`; `y^{g,s} >= g^s_{H1}` (1-based days; in 0-based code arrays, index `H1 - 1`) |
| Day `H1 + 1` | every row of day `k = H1 + 1` that reads the previous day reads `y` instead: `s_{k-1} -> y^s`, `g_{k-1} -> y^{g,s}` (dynamics rows of each event, both encodings; the Wiener variance row; the anchor keep rows; the repair-cost row `z >= rho (y^mu - y^{g,mu}) - rho tau (1 - x^m)`, with `y^{g,mu} = 0` under `ardinf`) |
| Loop rows | `s_T <= y^s`; `g^s_T <= y^{g,s}` (they replace `s_T <= s_{H1}` and `g_T <= g_{H1}`) |
| Unchanged | rows of days `1..H1` and `H1 + 2..T`; replacement and anchor-set rows (they read no previous day); reliability rows (no row on `y`); objective; `u`; the post-solve `z` of the transitory days (`c_zpost`) |

## 2 Result keys

`y_mu`, `y_v`, `y_R`, `y_K`, `y_g_mu`, `y_g_v`, `y_g_R`, `y_g_K`: arrays `F x L` of floats, `NaN` where the cell does
not track the quantity or anchor (the pattern of `v`, `R`, `K`, `g_*`). Upper bounds, as the other tracked values.
In every result file, as the other arrays (HDF5: `float64` datasets); `read_result` returns them as float arrays.

## 3 Tasks

- **Y1, code core** (Sonnet). Files: `src/fleet_management/degradation_model/base.py` (variables, rows, extraction;
  `add_quantity_rows` holds the dynamics, anchor and loop rows; `add_repair_cost_rows` the cost rows), the five cell
  builders only if they build a previous-day row themselves (the Wiener variance row, for example),
  `src/fleet_management/solver.py` (`RESULT_KEYS`, the HDF5 dataset keys). Tests: a new
  `tests/test_step3_start.py` (Gurobi): (a) a one-cell input whose best schedule replaces the component on day `H1`
  and then rises in the operating phase (mean of 0.2 per mission, repair `rho = 0.5`, `ardinf`; derive it): solved,
  status optimal, `y_mu` above `mu_{H1}`; (b) every row of section 1 by name and coefficient on a small model; (c)
  the result keys and `NaN` pattern. Update existing tests that assert the old loop rows or the exact key set (list
  each change).
- **Y2, reference model** (Sonnet). `tests/reference_model.py`: `row_violations` (start rows, day `H1 + 1` from `y`,
  loop rows to `y`; for a solution without `y` keys, use `y = ` the value on day `H1`, the old rule), `least_solution`
  (the Kleene iteration includes `y >= s_{H1}`, `y >= s_T` and the day `H1 + 1` rows from `y`; it returns `y`),
  `reference_trajectory` unchanged (equalities: `y` equals the value on day `H1`). Unit tests: the hand case of Y1
  (a) accepted now, refused by the old rule; update the cases that assert the old loop rows. Then rerun the tightness
  check (`wp45_work/wp8/tightness_check.md`, scripts in the scratchpad `wp8/tight/`) with the new `least_solution`:
  expected, no safe schedule refused because of a replacement on day `H1`; report what remains.
- **Y3, tools** (Sonnet, after coder 5 of step 3 finishes `dashboard_inspector.py`). `validation/validator.py`: the
  checks `loop_q` become `max{0, max(q_T - y_q), max(q_{H1} - y_q)}` when the result has `y_q`, else the old
  `max{0, max(q_T - q_{H1})}`; `gamma_replay_validator.py`: the field `repeatability` likewise (expected value `y`);
  `validation/dashboard_inspector.py`: the tab Repeatability shows `q_{H1}`, `y_q`, `q_T` and the margins
  `y_q - q_T`, `y_q - q_{H1}`. The greedy heuristic does not change (its loop excess on the reference trajectory is
  stricter than the rows; the spec will say so). Tests in a new `tests/test_step3_start_tools.py`; update existing
  tests that assert the old checks.
- **Spec and scoping** (Sonnet, after the code passes on Euler): Decision Variables, Encoding (day `H1 + 1`),
  Loop Constraint, Repair Cost, Full Optimization Problem, Result Dictionary and Rules, HDF5 Layout, Structural
  Validator, Replay validator, Dashboards, Greedy (note); scoping: "What the one-sided rows accept" (the replacement
  case goes; Proposition "Acceptance" (b) loses that condition), the proof of the perpetual theorem (start from `y`),
  the loop-constraint section and its example.
