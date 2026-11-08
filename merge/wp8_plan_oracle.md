# WP8 plan, part `oracle`: the meaning of a schedule

Status (7 Nov 2026): done; see section Review.

- **Goal.** Show that every schedule the solver returns means what the spec says, and that the solver finds the best
  schedule the spec allows. The tool is a **reference model** ("oracle"): a short re-implementation of the spec's
  formulation in numpy and scipy, written from the spec only (`wp8_interface.md`, the independence rule).
- **Spec** (`lore` copy, `C:\Users\utente\Documents\GitHub\Fleet_Management\spec\spec.tex`): Index Sets 270-348;
  Input Specification 349-602 (meaning of the inputs); Decision Variables 603-650; Objective 653-695; Constraints
  698-1625 (assignment, depot day, repair laws, encoding, replacement, linear forms, the five model sections, repair
  cost, damage penalty, loop); Full Optimization Problem 1628-1707; Result Dictionary and Rules for the Values
  1714-1784. Read all of it before writing code.
- **Out of scope**: parsing and checks (part `io`), the tools (part `tools`).

## 1 Files

| File | Content | Task |
|---|---|---|
| `tests/reference_model.py` | the oracle (section 2); not a test file, pytest does not collect it | O1, O2 |
| `tests/test_oracle_unit.py` | unit tests of the oracle on hand-derived numbers; no Gurobi | O1, O2 |
| `tests/test_oracle_replay.py` | solves of small inputs, each result checked by the oracle; Gurobi | O3 |
| `tests/test_oracle_bruteforce.py` | every schedule of tiny fleets enumerated; Gurobi | O4 |
| `tests/data/oracle/*.yaml` | the inputs of O3 and O4, one per case, each with a first comment line saying what it covers | O3, O4 |

The test files import the oracle with `from reference_model import ...` (pytest puts `tests/` on `sys.path`, because
`tests/` has no `__init__.py`). The oracle reads the configuration of `fleet_management.config.load_config`, because
the part `io` tests the parsing. It must **not** use the derived fields of `FleetConfig` (`support_const`,
`range_mu`, `range_v`, `range_R`, `range_K`) or any function of the package except `load_config`, `read_input` and
`solve_mapping`/`solve`: every constant is computed again from the spec.

## 2 The oracle (`tests/reference_model.py`)

Notation as the spec: cell `(i, l)`, day `k = 1..T`, index 0 for the initial values; arrays are 0-based, so day `k`
is index `k` of an array of length `T + 1` whose index 0 holds the initial value.

| Function | Spec | What it returns |
|---|---|---|
| `day_profile(cfg, key)` | Eq. `c_phase`, 438-449 | array `(F, L, M, T)` of the profile `key` (`mu`, `v`, `support`, `cgf`, `gamma_beta`) after the phase lookup; `gamma_beta` without a day axis is repeated (row 457) |
| `cell_kind(cfg, i, l)` | 1673-1700 | the model and, for rainflow, the bound; the tracked quantities in order: G `mu, v`; IG `mu`; W `mu, v`; Ga `mu`; RF Mk `mu`; Ca `mu, v`; Ho `mu, R`; Be `mu, v`; Ch `mu, K` |
| `constants(cfg)` | 324-343, 1022, 1094-1119, 1274-1310, 1399-1405, 1508-1529, 862-918 | per cell: `kappa_G`, `V_max`, `kappa_C`, `kappa`, `b`, `c`, `V_C`, `V_B`, the ranges `M_s` of every tracked quantity (832-836), the IG cap `mu_bar`, the Gamma `beta_star` and cap `A1/beta_star`, the quadratic-row constants `c2`, `c1` and side cap (table 875-884; Bernstein `c2 = 1/(2 kappa)`, `c1 = -b/3`) |
| `initial_and_new(cfg)` | 408-428, 774-778, 1389-1396 | per cell and quantity: initial value, starting anchor, replacement value (`replacement_as_new` sets all replacement values to 0) |
| `reference_trajectory(cfg, x, m, r)` | 829 and the day maps `g_map_mu`, `g_map_v`, `ig_map`, `w_map_v`, `gm_map`, rainflow "Day maps" 1381-1387; anchors `a_anchor` | dict of arrays `(F, L, T+1)`: `mu`, `v`, `R`, `K`, `g_mu`, `g_v`, `g_R`, `g_K`, with NaN where the cell does not track the quantity; every row at equality |
| `reliability_slack(cfg, i, l, state)` | rows `g_row`, `ig_row`, `gm_row`, `rf_markov`, `rf_cantelli`, `rf_hoeffding`, `rf_bernstein`, `rf_chernoff`; forms 887-922 | for one day's state of a cell: the smallest slack among the reliability row **in the form built at the cell** (`exact`; `tangent` Eq. `a_tangent`; `pwl`: the largest cap among the segments that contain the mean, Eqs. `a_pwl_sum`-`a_pwl_cap`, `P = 1` is the tangent at `tau/2`), the side row, and `v <= v_max_user` where given; a linear row is always exact |
| `row_violations(cfg, sol, tol)` | 698-1625, 1606-1617, `a_z` | list of records `(row, i, l, k, violation)` for every row of the spec evaluated at a solution `sol` (binaries `x`, `m`, `r` and the tracked arrays and `z` as the result reports them, day 0 from the inputs): one activity, coverage, depot day, every dynamics row as the spec's **one-sided** inequality (`s_k >= map(s_{k-1})` for the day's event; equality on a replacement day; anchor rows), reliability (via `reliability_slack`), loop rows `s_T <= s_{H1}` for every tracked quantity and anchor, the repair-cost row `a_z` where `z` is built, `z >= 0` |
| `repeated_safety(cfg, x, m, r, cycles)` | 1612-1617 (claim: "every day of every repetition of the operating phase satisfies the reliability requirement") | extends the schedule by repeating days `H1+1..T` `cycles` times, computes the reference trajectory over the extended horizon, and returns the smallest `reliability_slack` over all days; the loop rows are not applied to the extension |
| `minimal_z(cfg, traj, m)` | `a_z`, 1572 | `(F, L, T)`: `max(0, rho (mu_{k-1} - g_{k-1}))` on repair days, `0` otherwise; NaN where `C_R = 0`; `g = 0` under `ardinf` |
| `u_value(cfg, mu)` | `c_u` | `max_{i,k} sum_l mu` (`inf_norm`) or `max_{i,k} sum_l mu^2` (`quadratic`), days 1..T |
| `costs(cfg, x, r, z, u)` | `a_daycost`, `a_phases`, `a_objective`, `d_projected` | `step_costs`, `J_initialization`, `J_op`, `J_op_average`, `objective` for the mode of `cfg` (`C_D u` only if `C_D > 0`), `projected_evaluation_cost` when `E` is given; `z = 0` where NaN |
| `enumerate_schedules(cfg)` | 704-732 | every binary schedule `(x, m, r)` that satisfies one activity, coverage and depot day; on a depot day each component gets none, repair, or (if `allow_replacement`) replacement; an idle day and a depot day without intervention are both listed |
| `best_reference(cfg)` | -- | the smallest `costs(...)["objective"]` over the enumerated schedules whose reference trajectory has no row violation, with `z = minimal_z`, `u = u_value(reference mu)`; returns the cost, the schedule and the number of feasible schedules |

Caps computed independently of the spec's algorithm (the expected values must not repeat the code's method):
- **IG cap** `mu_bar = tau / E[Z | Z > q]`, `Z ~ IG(1, 1/eta)`: with `scipy.stats.invgauss(mu=eta, scale=1/eta)`
  (mean 1, shape `1/eta`), `q = isf(epsilon)` and `E[Z; Z > q]` by `scipy.integrate.quad` of `z f(z)` on
  `[q, inf)`; then `mu_bar = tau * epsilon / E[Z; Z > q]`.
- **Gamma cap** `A1(t, eps)`: `w(A) = max over c in (0, 1] of gammaincc(A/c, t/c)`, by `scipy.optimize
  .minimize_scalar` on `log c` started from a grid of 2000 values of `log c` in `[log 1e-8, 0]`; `A1` is the largest
  `A < t` with `w(A) <= eps`, by `scipy.optimize.brentq` on `w(A) - eps` when `w` crosses `eps`, else the root `A^`
  of `gammaincc(A^, t) = eps` (spec 1293-1308). The test compares with the package's cap within `1e-6` relative.
- **beta_star** Eq. `gm_betastar`: over both phases' rates where the mean is positive, `beta_0` and `beta_new` only
  where given and their mean is positive (`beta_new` only if `allow_replacement`); no rate → row `mu <= tau`.

## 3 Tests

### 3.1 `test_oracle_unit.py` (no Gurobi), tasks O1 and O2

Each expected number has a comment with its derivation; each case is checked first by a short separate computation
in the scratchpad. Hand-derived anchors (Claude, from the spec):
- **Gaussian, one cell**: `tau = 1`, `eps = 0.05`, `rho = 0.5`, `mu_0 = 0.1`, `v_0 = 0.0004`, increment `mu = 0.2`,
  `v = 0.001`. Day 1 mission: `mu = 0.3`, `v = 0.0014`. Day 2 repair, `ardinf`: `mu = 0.15`, `v = 0.00035`,
  `z = 0.15`. Day 2 repair, `ard1` with `anchor_mu_0 = 0.05`, `anchor_v_0 = 0.0002`: `mu = 0.5*0.3 + 0.5*0.05 =
  0.175`, `v = 0.25*0.0014 + 0.75*0.0002 = 0.0005`, anchors after the repair `0.175` and `0.0005`, `z = 0.5*(0.3 -
  0.05) = 0.125`.
- **Constants**: `eps = 0.05`: `kappa_G = 1.6448536`, `V_max = 1/kappa_G^2 = 0.3696115` (`tau = 1`); Cantelli
  `eps = 0.1`: `kappa_C^2 = 9`, `V_C = 1/9`; Hoeffding `eps = 0.1`: `kappa = ln 10`, range of `R` `= 2/ln 10`;
  Chernoff `s = 5`, `eps = 0.1`, `tau = 1`: range of `K` `= 5 - ln 10`.
- **Tangent** (Gaussian, `tau = 1`, `eps = 0.05`, `tangent_ref = 0.5`): `v <= 0.0924029 - 0.3696115 (mu - 0.5)`;
  at `mu = 0.3` the cap is `0.1663252`, below the exact `0.1811096`.

Tests (at least these):
1. `day_profile`: operating only, transitory given, `k <= H1` without a transitory profile (wraps with period `H2`),
   one per branch of `c_phase`, with distinct numbers per day.
2. Per model and law, the reference trajectory of a hand schedule of 4 days (mission, repair, idle, replacement):
   G, IG, W (repair keeps `v`; replacement resets), Ga, RF with each bound (factor `(1-rho)^2` for `v` and `R`,
   `1-rho` for `K`); `ardinf` and `ard1` for each; the anchor after a replacement equals the replacement value.
3. `reliability_slack`: each row and form at one point inside and one outside; Bernstein with `b > 0`; `pwl` at a
   segment boundary takes the larger cap; `P = 1`.
4. Property (spec 921-922, "each linear form implies the exact row"): for 1000 random points with `chi` within the
   tangent or `pwl` cap and `mu` within the side row, the exact row holds; every quadratic row.
5. `minimal_z`, `u_value`, `costs` in the three modes on a hand schedule (numbers derived in the comment).
6. IG cap: three `(eta, eps)` pairs; Gamma cap: three `(t, eps)` pairs, one with `eps >= 1/2`; each against the
   package's functions (`ig_mean_cap` in `inverse_gaussian.py`; the Gamma cap function of `gamma.py`, by its name),
   within `1e-6` relative.
7. `enumerate_schedules`: the count for `F = 2, M = 1, L = 1, T = 2` with and without replacement (derive it:
   per day 2 choices of the flying train times 4 choices of the other train with replacement, 3 without).
8. `repeated_safety` on a schedule whose loop rows hold: slack never negative for 20 cycles (property of the spec).

### 3.2 `test_oracle_replay.py` (Gurobi), task O3

Each case: an input in `tests/data/oracle/`, solved with `solve_mapping(load_input(path), None)` after setting
`verbose: 0`, `mip_gap: 0`, `time_limit: 60`, `gurobi_params: {Threads: 1}`. Checks on the result `res`:
- A. `res["status"] == "optimal"`.
- B. `row_violations(cfg, res, tol)` is empty. Binaries: within `1e-5` of `{0, 1}`, then rounded. Tolerance of a row:
  `1e-6 + 2e-5 * M_s` for a row with a big-M constant `M_s` (Gurobi's integrality tolerance lets a binary leak),
  `1e-6` otherwise, `1e-5` relative for a quadratic row.
- C. `repeated_safety(cfg, x, m, r, cycles=20) >= -1e-9`: the schedule is safe on every repetition (the spec's claim).
- D. Each reported tracked array is at least the reference trajectory of the reported schedule (`>= ref - tol`), as
  "Upper bounds" (1761) says.
- E. `z`: rows as B; on the transitory days of `operating_average`, `z` equals Eq. `c_zpost` on the reported `mu`,
  `g_mu`; `u` equals `u_value(cfg, res["mu"])`; `step_costs`, `J_*`, `operating_objective`, `J_total`,
  `projected_evaluation_cost`, `evaluation_objective` equal `costs(...)` on the reported arrays (`1e-8` relative).
- F. `objective` equals the cost of the mode (Rules 1778-1783), within `1e-6` relative (at `mip_gap = 0` the variable
  `u` has no slack when `C_D > 0`).

Cases (one input each; `F = 3`, `M = 1` or `2`, `L = 2`, `H = [2, 3]` unless stated; every case non-trivial: the
cheapest schedule without interventions breaks a row, checked by the oracle in the test):
G `ardinf` exact; G `ard1` tangent; G pwl `P = 4`; G `indicator` encoding; IG `ardinf`; IG `ard1`; W `ardinf` (a
replacement must appear in the operating phase); W `ard1` with `v_max_user`; Ga `ardinf` with `gamma_beta_0`; Ga
`ard1` with rates per mission and a transitory rate; RF Markov; RF Cantelli tangent; RF Hoeffding exact; RF Bernstein
pwl; RF Chernoff `ard1`; the spec's mixed example (2768-2827, with `mip_gap: 0`, `time_limit: 120`); mode
`operating_average`; mode `evaluation_total`; `penalty_type: quadratic` with every quadratic cell `exact`;
`allow_replacement: false`; `replacement_as_new: true`; a transitory profile `mu_trans` with distinct days.

### 3.3 `test_oracle_bruteforce.py` (Gurobi), task O4

Tiny fleets: `F = 2`, `M = 1`, `L = 1` or `2`, `T <= 4`; at most about 10 000 schedules per case. Per case:
1. `best = best_reference(cfg)`; assert at least one feasible schedule, and that the best one has an intervention
   or a binding reliability row (else the case is trivial: change the numbers).
2. Solve as in 3.2; checks A to F of 3.2.
3. **G1**: `res["objective"] <= best.cost + 1e-6 max(1, |best.cost|)`. A larger value means the code forbids a
   schedule that the spec allows: a code defect.
4. **G2**: if `res["objective"] < best.cost - 1e-6 max(1, |best.cost|)`, the returned schedule's reference trajectory
   breaks a row (the spec's one-sided rows allow a larger tracked state, which can satisfy a loop row that the
   reference breaks). Then B and C must hold; the test records the case in its output with `print` (it is allowed by
   the spec, and it is a finding for the report).
5. Infeasible case: when `best_reference` finds no feasible schedule, the solve must not report `optimal` with an
   invalid point: either a status without a solution, or B and C hold.

Cases: each model and law (G, IG, W, Ga, RF Mk, Ca, Ho, Be, Ch; `ardinf` and `ard1`), each form for one quadratic
model (exact, tangent, pwl `P = 1` and `P = 3`), `indicator` for one model of each tracked set, each objective mode,
both penalty types, `allow_replacement` false, a mixed `L = 2` cell pair, a W cell (loop forces a replacement).
About 35 cases; each solve under 2 s.

## 4 Executor tasks (Sonnet agents)

Common: read `merge/wp8_interface.md`, this plan, and the spec lines named; the independence rule binds. Local runs:
copy `merge/wp45_work/wp6/gamma/run_tests_gamma.py` to the scratchpad and run the test file with it (Gurobi tests
skip); add `tests/` to `sys.path` for the oracle import. Write a progress file
`merge/wp45_work/wp8/oracle/task<id>_progress.md` with a `Status:` line, updated after each step.

- **O1** (oracle core and G, IG, W): `reference_model.py` sections `day_profile`, `cell_kind`, `constants` (all
  models), `initial_and_new`, `reference_trajectory` and `reliability_slack` for G, IG, W, `minimal_z`, `u_value`,
  `costs`, `enumerate_schedules`; unit tests 1, 2 (G, IG, W), 3 (G, W), 4, 5, 6 (IG), 7.
- **O2** (Ga and rainflow, rows, brute force, safety): `reference_trajectory` and `reliability_slack` for Ga and the
  five bounds; `row_violations`, `repeated_safety`, `best_reference`; unit tests 2 (Ga, RF), 3 (RF, Ga), 6 (Gamma), 8.
  Starts after O1.
- **O3** (replay): the inputs of 3.2 in `tests/data/oracle/replay_*.yaml` and `test_oracle_replay.py`. Each input is
  checked locally with `load_config` (it must pass the checks) and with the oracle (non-trivial). Starts after O2.
- **O4** (brute force): the inputs of 3.3 in `tests/data/oracle/bf_*.yaml` and `test_oracle_bruteforce.py`; local
  run of `best_reference` on each input, with the number of schedules and the time, in the progress file. Starts
  after O2; may run beside O3.

## 5 Questions

- Q1. Reported tracked values versus the reference trajectory: the spec allows larger values ("Upper bounds"), so D
  checks `>=`, never `==`. Decided by Claude.
- Q2. G2 cases (the solver beats the reference optimum through larger tracked states) are allowed by the spec and are
  reported, not failed. Decided by Claude; the report to the user lists them.

## 6 Removals

None.

## 7 Review

Claude, 7 Nov 2026. Tasks O1 to O5 done. Euler: unit tests, 22 replay cases and 42 brute-force cases pass (jobs 20154950, 20153605). In 19 brute-force cases the solver's optimum is below the best exact-trajectory schedule (finding F1 of `wp8_defects.md`); task O5's least one-sided solution gives the spec's optimum, equal to Gurobi's within 1e-11 (check G3). No code defect.
