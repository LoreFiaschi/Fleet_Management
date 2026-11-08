# WP8 plan, part `io`: inputs, checks, outputs

Status (7 Nov 2026): done; see section Review.

- **Goal.** Show that the package reads every input the spec accepts, refuses every input the spec refuses with the
  error the spec names, and writes and reads results as the spec says.
- **Spec** (`lore` copy, `C:\Users\utente\Documents\GitHub\Fleet_Management\spec\spec.tex`): Data Flow 213-227;
  Public API 229-240; Input Specification 349-600 (all subsections, Rejected Keys 500-525, Consistency Checks
  527-600); Output Specification 1709-1885 (Result Dictionary, Rules for the Values, the reports `performance` and
  `optimization_progress` 1788-1829, Result Files); Input File Schema 2712-2828 (formats, HDF5 layout, keys by cell
  type, the YAML example).
- **Independence rule** (`wp8_interface.md`): expected values come from the spec. `src/` may be read only for names,
  signatures, exception types and the field names of `FleetConfig` (`config.py`, lines 150-216), so the tests can
  read the normalised configuration. Do not read the existing `tests/` files before your tests are written.
- **Out of scope**: the meaning of a schedule (part `oracle`); the reports `warm_start` and
  `relaxation_warm_start` (part `tools`).

## 1 Files

| File | Content | Task | Gurobi |
|---|---|---|---|
| `tests/data/io/valid/*.yaml` | the library of valid inputs (section 2) | I1 | -- |
| `tests/test_spec_inputs.py` | keys, defaults, shapes, broadcasting, `null` rules, the three input formats | I1 | no |
| `tests/test_spec_checks.py` | rejected keys and every consistency check | I2 | no |
| `tests/test_spec_outputs.py` | result dictionary, rules for the values, reports, result files, `read_result`, public API | I3 | most |

Helper in each test file: `_base(name)` loads `tests/data/io/valid/<name>.yaml` with `yaml.safe_load` and returns a
fresh mapping; `_cfg(data)` calls `load_config`. An invalid input is a valid mapping with one key changed.

## 2 The library of valid inputs (`tests/data/io/valid/`)

Each file: a first comment line saying what it covers; `verbose: 0`, `mip_gap: 0`, `time_limit: 60`; numbers chosen so
that the file passes every check (verify with `load_config` locally) and solves in under 2 s. Sizes `F = 3`, `M = 1`,
`L = 2`, `H = [2, 3]` unless stated.

| File | Covers |
|---|---|
| `gaussian.yaml` | G cells, `ardinf`, exact form; `L` given; `component_names` |
| `gaussian_ard1_forms.yaml` | G cells, `ard1`, one cell `tangent`, one `pwl`, starting anchors given |
| `inverse_gaussian.yaml` | IG cells, `eta` |
| `wiener.yaml` | W cells, `sigma`, `v_max_user` at one cell, `null` at the other |
| `gamma.yaml` | Ga cells, `gamma_beta` per mission `(F, L, M)`, `gamma_beta_0` at one cell, `gamma_beta_new` |
| `rainflow_bounds.yaml` | `F = 3`, `L = 5`: one component per bound (Mk, Ca, Ho, Be, Ch), with every bound-specific key, `null` where a key does not apply |
| `mixed_example.yaml` | the YAML example of the spec, lines 2768-2827, copied verbatim (comments included) |
| `minimal.yaml` | only the required keys of a G fleet (`F, M, H, model, tau, epsilon, rho, mu_0, mu, v, C_M, C_R, C_rep, C_D`), for the defaults tests |

## 3 Tests

### 3.1 `test_spec_inputs.py` (no Gurobi), task I1

- **Library**: every file of section 2 passes `load_config` (parametrised).
- **Defaults** (each table of 366-496): load `minimal.yaml` (and `rainflow_bounds.yaml`, `gamma.yaml` for their keys)
  and assert the normalised value: `repair_model` `ardinf`; `bound_method` `cantelli` at an RF cell without it;
  `reliability_impl` `exact`; `encoding` `bigm`; `component_names` `Component 1, Component 2`; `tangent_ref` 0.5;
  `pwl_points` 8; anchors equal the initial values; `v_0`, `support_0`, `R_0` 0; `cgf_0 = s mu_0`; `replacement_mu`,
  `replacement_v`, `support_new`, `R_new` 0; `cgf_new = s mu_new`; `replacement_as_new` false; `allow_replacement`
  true; `penalty_type` `inf_norm`; run options `verbose` 1 (in a file without it), `mip_gap` 0.1, `time_limit` 3600,
  `fast` false, `objective_mode` `total`, `progress_interval_seconds` 5.0, `relaxation_warm_start` false,
  `relaxation_time_limit` 60.0.
- **Sizes**: `L` inferred from `model`; `L` required when `model` is one string and `L > 1`; a scalar `H` gives
  `H1 = H2 = H`, `T = 2H`; `[H1, H2]` gives `T = H1 + H2`.
- **Cell values** (354): scalar, list of `L`, `F x L` give the same `F x L` array.
- **Profiles** (432-436): the six accepted shapes expand to `(F, L, M, H*)` with the same numbers; the ambiguous shape
  (`F = L`, an array fitting both `(L, M, H*)` and `(F, L, M)`) is read as `(L, M, H*)`; transitory profiles have `H1`
  entries; `gamma_beta` and `gamma_beta_trans`: scalar, `(L)` one rate per component, `(F, L)` one per cell,
  `(F, L, M)`, `(F, L, M, H2)`.
- **Integers** (364): `F: 3.0` accepted; `F: 2.5` and `F: true` refused (`ValueError`).
- **`null` rules** (359-362): `null` where a key does not apply is accepted (for example `v` at an IG cell); `null`
  where it applies is an error; at `v_max_user`, `gamma_beta_0`, `gamma_beta_new` a `null` cell means "not given"
  (NaN in the configuration); a non-`null` entry of an `F x L` array at a cell where the key does not apply gives a
  `UserWarning` and is ignored; a selector entry at such a cell is still checked (`bound_method: chernof` at a G cell
  is an error, 547).
- **Case sensitivity** (363): `model: Gaussian` and the key `Tau` are errors.
- **Formats** (2714-2743): `mixed_example.yaml` written as JSON (`.json`) and as HDF5 (`.h5` and `.hdf5`, built in
  the test with `h5py` by the layout of 2729-2734: scalars as attributes, arrays as `float64` datasets with NaN for
  `null`, selectors as string datasets with `""` for `null`, `component_names` and `gurobi_params` as JSON text) give
  the same `FleetConfig` as the YAML file (compare every field; NaN equal to NaN). `.yml` accepted; `.txt` refused;
  the extension is read in lower case (`.YAML` accepted).

### 3.2 `test_spec_checks.py` (no Gurobi), task I2

One test per bullet of 527-600, and one per row of 506-525; parametrise where natural. Each asserts `ValueError`
whose message contains the key name (and nothing more specific than the spec states).
- **Rejected keys**: each of the 22 rejected keys, added to a valid input, raises an error whose message names the
  canonical key of its row (for `maintenance_type`: `repair_model`; for `C_S`: `C_D`; and so on); an unknown key
  raises an error containing `unknown key`; the key check runs before any array is read (an unknown key together with
  a malformed `mu` gives the unknown-key error).
- **Required keys** (536): removing each required key, and each key a cell type requires (table 2750-2760), is an
  error.
- **Tolerance** (532): for one non-strict check of each kind (for example `anchor_mu_0 <= mu_0`): the value
  `b (1 + 5e-13)` passes, `b (1 + 1e-10)` fails; for a strict check (`tau > 0`, `F > M`): equality fails.
- **Each check** of 541-600 with its boundary: sizes and selectors; `tau`; `epsilon` in `(0, 1)` and `< 0.5` at G and
  W; `rho` in `(0, 1]` (`rho = 1` passes); `0 <= mu_0 < tau`; non-negative initial and replacement values (the
  replacement ones only when `allow_replacement` is true: a negative `replacement_v` passes with
  `allow_replacement: false`); `tangent_ref` in `[0, 1)`; `pwl_points >= 1`; the starting anchors; `replacement_as_new`
  with a non-zero replacement input; natural ranges (566) with the ranges computed in the test from the spec's
  formulas; the replacement state against the row of the form built (567), one case per form, the `pwl` case at a
  mean inside a segment; increments (571-575); G `mu >= 3 sqrt(v)`; IG `eta > 0`; W `sigma > 0`; Ga rates;
  `v_max_user`; Be `v <= b^2/4`, `tau > 2 kappa b / 3`, the tangent and `pwl` conditions of 590; Ch `s > ln(1/eps)
  / tau`, `K >= s mu` at the four places of 592; run options (597-598, including `mip_gap = -1e-13` accepted and set
  to 0); costs (599); `evaluation_horizon` (597).
- **Checks of one model** (531): a value that breaks a G check at an IG cell passes.

### 3.3 `test_spec_outputs.py`, task I3

Gurobi tests call `pytest.importorskip("gurobipy")` first. Solves use files of section 2 and `solve(path, out)` with
`out` in `tmp_path`.
- **Data flow** (221, 1880, no Gurobi): a missing input file gives `FileNotFoundError`; input extension `.txt` gives
  `ValueError`; the result path is checked before the input is read (bad result extension with a missing input file
  gives `ValueError`, not `FileNotFoundError`); a missing result folder gives `FileNotFoundError`.
- **Result dictionary** (Gurobi, one solve of `mixed_example.yaml` with `mip_gap: 0.05`, and one of each other library
  file): exactly the keys of 1716-1753; the shapes; `model` as in the input; selectors one string when all cells agree
  and a list otherwise, `bound_method` `null` at a non-RF cell; `v` NaN at IG, Ga, RF Mk, Ho, Ch cells; `R` and `K`
  NaN except at Ho and Ch cells; `g_*` NaN at `ardinf` cells and at the cells 1739 excludes; `z` NaN where `C_R = 0`;
  `idle = x_{i0k} - m - r`; `u` by Eq. `c_u` from `mu`, also with `C_D = 0`; the cost report from Eqs.
  `a_daycost`, `a_phases`, `d_projected` on the arrays; `projected_evaluation_cost` and `evaluation_objective` `null`
  without `evaluation_horizon`; `component_costs`; `objective` against the key of the mode (1778-1783).
- **No solution** (1759, Gurobi): an infeasible input (W cells only, `allow_replacement: false`, a mission in the
  operating phase; spec 1201-1205) gives `status` `infeasible` (or `inf_or_unbounded`), and `null` solution arrays,
  cost report, `objective`, `mip_gap`.
- **Reports** (1788-1829, Gurobi): `performance` has every key of the table (values may be `null`);
  `requested_mip_gap` and `requested_time_limit_seconds` equal the run options; `optimization_progress` is a list
  of records with the five keys, its last record is the final state (its `solutions` equals
  `performance["solutions_found"]`).
- **Result files** (1866-1884): the five extensions; another extension gives `ValueError`; no extension gives
  `.yaml`; default `output.yaml` in the working folder (`monkeypatch.chdir(tmp_path)`); `None` writes no file;
  YAML and JSON load with `yaml.safe_load` and `json.load` and hold `null` for NaN (for example `v` at an IG cell);
  `objective` written as `null` when `None` (infeasible case), other `None` keys absent; HDF5 layout of 2736-2743
  (attributes, `float64` datasets with NaN, JSON text attributes), `objective` absent when `None`.
- **`read_result`** (1883, no Gurobi for hand-written files): exactly the keys of the result table, a missing key is
  `None`, an extra key is dropped, a `float64` key read from YAML with a `null` entry becomes a float array with NaN;
  `.txt` gives `ValueError`; round trip writer to `read_result` for the three formats on a solved result (Gurobi):
  equal arrays (NaN equal), equal scalars.
- **Public API** (232-240, no Gurobi): `fleet_management` exports `solve`, `sweep_operating_horizons`,
  `sweep_horizon_grid`, `plot_management`, `plot_horizon_sweep`, `validate`, `validate_gamma_replay_schedule`,
  `validate_gamma_replay_files`, `validate_gamma_stochastic_schedule`, `validate_gamma_schedule`,
  `validate_gamma_schedule_files`; and not `solve_mapping`, `load_config`, `build_greedy_warm_start`,
  `validate_gamma_tail_bound_schedule` (check `__all__` if it exists, else the module attributes); `import
  fleet_management` works without `gurobipy` (run `python -c` in a subprocess with `sys.modules['gurobipy'] = None`).

## 4 Executor tasks (Sonnet agents)

Common: read `merge/wp8_interface.md`, this plan, and the spec lines named. Local runs: copy
`merge/wp45_work/wp6/gamma/run_tests_gamma.py` to the scratchpad and run each test file with it (Gurobi tests
skip). Progress file `merge/wp45_work/wp8/io/task<id>_progress.md` with a `Status:` line.

- **I1**: the library (section 2) and `test_spec_inputs.py` (3.1).
- **I2**: `test_spec_checks.py` (3.2). May start at once; it builds its inputs from `minimal.yaml` and the library,
  so it waits for I1's library files (I1 writes them first and says so in its progress file).
- **I3**: `test_spec_outputs.py` (3.3). After I1.

A test that fails locally is either a test error (fix it) or a code defect: keep the test, mark nothing as skipped,
and list it in the progress file with the spec line and the observed behaviour.

## 5 Questions

None open.

## 6 Removals

None.

## 7 Review

Claude, 7 Nov 2026. Tasks I1 to I3 done. Euler: all pass after two test defects were fixed (X1: the plan expected the wrong error; X2: an infeasible library input). No code defect. Spec findings S1, S3, S5.
