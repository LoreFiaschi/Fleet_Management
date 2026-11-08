# WP8 plan, part `legacy`: earlier tests and scripts

Status (7 Nov 2026): done; see section Review.

- **Goal.** Bring the material written before the merge into the merged package, or remove it with a reason:
  `lore`'s test suite, the smoke test, and `main`'s regression scripts, experiments and inputs.
- **Decisions** (`merge/decisions.md`): "Tests and test inputs" (the smoke test is ported, its inputs under
  `examples/`, its scenarios extended to the merge's changes; `main`'s check scripts kept where they still apply),
  "Cluster jobs and experiments" (the definitions of the experiments stay; results computed before the merge are
  removed).
- **What earlier packages left to WP8**: `merge/wp1_removals.md` items U4 and U8 and the lists below them;
  `merge/wp4_plan.md` (item on `examples/regression/` and `experiments/`); the sections "Callers outside src" or
  "Broken by" of `merge/wp7_plan_sweeps.md`, `wp7_plan_plots.md`, `wp7_plan_validator.md`,
  `wp7_plan_gamma_validators.md` (grep "WP8").
- This part ports existing material: the independence rule does not bind it; it may read the code.

## 1 Task L1: inventory (Sonnet; no edits outside the inventory file)

Write `merge/wp8_legacy_inventory.md` (a `Status:` line first; one table per group; one row per item; short cells):

1. **`lore`'s tests**, `C:\Users\utente\Documents\GitHub\Fleet_Management\test\` (9 files, `conftest.py`,
   `TEST_DOCUMENTATION.md`; written for `lore`'s old API). Columns: file, test function, what it checks (one line),
   spec section of the merged spec it maps to, does the merged package still have that behaviour (yes / changed on
   purpose, with the decision item / removed), the inputs it needs, proposed action (port / drop) and reason.
2. **`main`'s scripts and inputs** in the worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge`:
   `examples/regression/` (every file), `experiments/` (every folder, file by file), `input/` (5 files), and
   `remote_smoke_test.py`. Columns: path, what it does (one line), keys or functions it uses that the merge removed or
   renamed (check against `config.load_config`, the spec's Rejected Keys 500-525, and the public API), still
   meaningful for the merged package (yes / no and why), proposed action (keep as is / rewrite: what changes / remove)
   and the size of the rewrite (lines).
3. **Checks**: for each YAML input of group 2, run `config.load_config` locally (with the `gurobipy` stub of
   `merge/wp45_work/wp5/stubs/` on `sys.path` if an import needs it) and record the first error; for each Python
   script, record whether it imports (`python -c "import ast; ast.parse(open(p).read())"` for syntax, then grep its
   imports against `src/fleet_management/`).

## 2 Task L2: ports and rewrites (after Claude's decisions)

Claude writes the decision for each row of the inventory into the inventory file (column "decision"). L2 carries
them out: ported `lore` tests in `tests/test_ported_lore.py` (one file; inputs in `tests/data/ported/`); rewritten
scripts and inputs in place; the files to remove listed in section 4 for Claude.

## 3 Task L3: the smoke test (Sonnet)

`remote_smoke_test.py` in the worktree root, rewritten for the merged package; inputs in `examples/smoke/`; a job
script `euler/run_smoke.sh`.
- **Behaviour** (kept from the current file): writes everything under `smoke_test_output/` (`smoke_test.log`,
  `summary.json` with pass or fail per scenario and the pytest result, `pytest_output.txt`, `plots/*.png`,
  `results/*`); always exits 0; runs `pytest tests -v` first unless `--no-pytest` is given.
- **Scenarios** (one input each, `examples/smoke/<name>.yaml`, a first comment line saying what it covers; each under
  about one minute; `mip_gap: 0.01`; each solved with `solve`, then checked with `validate` (must pass), then plotted
  with `plot_management`). Together they cover the merge's changes listed in the decision:
  1. `five_models`: one fleet with G, IG, W, Ga and RF cells, `ardinf` and `ard1` mixed, `H = [2, 4]`.
  2. `rainflow_bounds`: one RF component per bound (Mk, Ca, Ho, Be, Ch).
  3. `forms`: G and RF Ca cells in `exact`, `tangent`, `pwl`; and the same fleet with `encoding: indicator`.
  4. `anchors`: `ard1` cells with starting anchors below the initial values.
  5. `replacement`: `replacement_mu`, `replacement_v` and the Gamma `gamma_beta_new`; and a variant with
     `replacement_as_new: true`; and one with `allow_replacement: false` that stays feasible.
  6. `caps`: IG and Ga cells whose mean must stop at the cap (`mu` close to it); the result's `mu` never above the cap
     computed by the package's cap functions.
  7. `costs`: the three objective modes (`total`, `operating_average`, `evaluation_total` with `E`) and both penalty
     types; check that each cost key of the result equals the formula of the spec (Eqs. `a_daycost`, `a_phases`,
     `d_projected`, `c_u`).
  8. `two_phase`: a transitory profile `mu_trans` different from the operating one, `H = [3, 4]`.
  9. `sweep`: `sweep_operating_horizons` over `H2 in [2, 3, 4]` with `warm_start=True` and an `output_path`;
     `plot_horizon_sweep` of the report.
  10. `greedy_start`: `build_greedy_warm_start` on scenario 1, then `solve(..., warm_start=source)`; the
      `warm_start` report shows `applied: true` and a completion status.
  11. `gamma_validators`: `validate_gamma_schedule` in mode `both` on the Gamma result of scenario 1 (mode
      `stochastic` with `repetitions: 20000`).
  12. `mixed_example`: the spec's YAML example (spec lines 2768-2827) as is.
- **Job script** `euler/run_smoke.sh`: the header and module lines of `euler/run_tests.sh` (job name `fm-smoke`,
  `--time=02:00:00`, outputs `logs/smoke_%j.out` and `.err`), the same environment check, then
  `python remote_smoke_test.py`; Unix line endings (`.gitattributes` keeps them for `*.sh`).
- **Local check**: `python -c "import ast; ..."` on the script; each input passes `load_config` locally. The real run
  is on Euler (Claude submits it).

## 4 Removals

Filled in by task L2 (7 Nov 2026). Nothing is deleted yet: Claude removes these files at the commit (to the Recycle
Bin, never a hard delete). Paths are in the worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge`.
28 files: 24 in `examples/regression/`, 4 in `input/` (the inventory counts 27: it proposed `gamma_tail_bound_public.yaml`
"rewrite or remove with its last user", and every user is removed). The reason is the row of `merge/wp8_legacy_inventory.md`.
The 7 Nov 2026 decisions of that file apply; the `lore` test folder (`Fleet_Management\test\`) is not in the
worktree and is not listed.

### `examples/regression/` (24 files)

| file | reason (inventory row) |
|---|---|
| `check_gamma_ard1_integration.py` | Tests removed shape and latch states; its two cases are ported to `tests/test_ported.py` (complete repair, consecutive repairs) |
| `check_gamma_ard1_replacement_truth_table.py` | Audits the removed exact-hull rows; imports the removed `gamma_diagnostics` |
| `check_gamma_ard1_truth_table.py` | Same: removed hull rows and latch keys |
| `check_gamma_ardinf_replacement_truth_table.py` | Same: removed hull rows |
| `check_gamma_big_m_formulation.py` | Product-hull rows replaced by one-sided rows and the sharp cap; rejected keys |
| `check_gamma_cell_integration.py` | Rejected key and removed result keys; covered by `tests/test_config.py::test_gamma_beta_shapes` and the smoke scenario `five_models` |
| `check_gamma_complexity_diagnostics.py` | Checks the report of the removed calibration step; goes with `report_gamma_complexity.py` |
| `check_gamma_exact_validator.py` | Rejected keys, 3-argument `extract_solution`, old report keys; covered by `tests/test_gamma_validators.py::test_exact_*` |
| `check_gamma_public_interface.py` | Tests the routing away from the removed legacy Gamma backend; public Gamma solve is in the smoke test |
| `check_gamma_repair_integration.py` | Rejected key, `ctx.extras["gamma"]`; covered by `tests/test_gamma.py::test_dynamics_and_cap` |
| `check_gamma_stochastic_validator.py` | Rejected key; covered by `tests/test_gamma_validators.py::test_monte_carlo_*` and `test_clopper_pearson` |
| `check_gamma_validator_edge_cases.py` | Rejected keys, old report keys; covered by `tests/test_gamma_validators.py::test_exact_full_repair`, `test_exact_tail_edge_cases` |
| `check_greedy_warm_start.py` | Imports the removed `_simulate_cell`; covered by `tests/test_warm_start.py` |
| `check_greedy_warm_start_acceptance.py` | Rejected keys, old report fields; replaced by the smoke scenario `greedy_start` |
| `check_horizon_sweep.py` | Imports the removed `_annotate_latest_gradient`; covered by `tests/test_horizon_sweep.py` and the smoke scenario `sweep` |
| `check_mixed_gamma_ard1_public.py` | Duplicate of `check_gamma_replay_validator.py` (README rule: extend one check); removed result keys |
| `check_mixed_public_interface.py` | Same duplicate; removed result keys |
| `check_operating_average_objective.py` | Reads an input with rejected keys; covered by the smoke scenario `costs` and `tests/test_base.py::test_objective_equals_the_cost_report` |
| `check_result_visualization.py` | Imports the removed `plot_mixed_management`; covered by `tests/test_plotter.py::test_schedule_layout` |
| `gamma_replacement_truth_table_common.py` | Helper of the three truth-table checks and of `report_gamma_validator_slide8.py` |
| `report_gamma_complexity.py` | Estimates of the removed formulation; its three inputs are rejected |
| `report_gamma_validator_slide8.py` | One-off evidence for the old formulation; imports the truth-table helper |
| `gamma_tail_bound_public.yaml` | Rejected keys; every script that read it is removed (`check_gamma_big_m_formulation`, `check_gamma_exact_validator`, `check_gamma_public_interface`, `check_operating_average_objective`, `report_gamma_complexity`) |
| `mixed_gamma_rainflow_public.yaml` | Only users are removed (`check_mixed_public_interface`, `report_gamma_complexity`) |

### `input/` (4 files)

| file | reason (inventory row) |
|---|---|
| `input/data.yaml` | Old format (`alpha`, `xi`, `C_S`, `C_P`); the README quick start uses the spec example (`examples/smoke/mixed_example.yaml`, task L3) |
| `input/data_test_baseline.yaml` | Input of the two legacy validator tests, replaced by `tests/test_validator.py`; old keys |
| `input/tiny.yaml` | Old format; no reference in the repository |
| `input/vbz_man12e_year.yaml` | Input of the dropped family A (decision 2 of the inventory) |

Not removed, kept as decided: `experiments/vbz_F8_feasibility_transfer/starts/vbz_F6_M1_L4_H2_48_result.yaml`
(decision 7).

## 5 Review

Claude, 7 Nov 2026. Tasks L1 to L3 done. 8 ported tests, 38 scripts and inputs rewritten, 28 files moved to the Recycle Bin. Euler: smoke test 12 of 12, regression scripts 6 of 6. The scalar `C_M` of four experiment inputs: the largest entry, the user's choice (commit 47178b8).
