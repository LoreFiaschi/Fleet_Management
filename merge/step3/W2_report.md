Status: done (all four items). File edited: `merge/spec_drafts/fragments_C.tex` only; line endings unchanged (0 CR before and after). Not built.

Locations by section of the fragment.

| Item | Location | Change | Code |
|---|---|---|---|
| M20 | Rejected Keys, intro | message form "key 'k' is not accepted." + column text; `ValueError` | `_check_keys` |
| M20 | Rejected Keys, rows `gamma_beta_bound`, `gamma_calibration_method`, `n_workers`/`warm_start`, `depot_capacity`, `C_P` | pointers (`sec:dyn_gamma`, `sec:sweeps`, `sec:warm_starts`, `sec:assignment`, `sec:loop`) removed from the table; one sentence below the table lists them | `_REJECTED_ROWS` |
| M21 | Result Files, "Plain data and NaN"; HDF5 Layout, "Output file" | datasets and number attributes keep NaN and inf; JSON text in an attribute writes `null` | `_save_hdf5`, `_to_plain` |
| A | Selectors table, `component_names`; Result Dictionary `component_names` | blanks at both ends removed | `_component_names` |
| A | Result Dictionary `tau, mu_0, v_0` | `v_0` is NaN at a cell without a variance | `cv("v_0", m["variance"])` |
| A | Format Rules | new item: data that are not a mapping raise `TypeError` | `read_input` |
| A | HDF5 Layout intro | three `ValueError`s of the reader (group, attribute and dataset of one name, invalid JSON text) | `_read_hdf5`, `_hdf5_value` |
| A | `performance` table | `integer_variables` includes binaries; `general_constraints` includes indicators; node and iteration counts are floats; `threads`, `seed` are the Gurobi parameter values | `_model_statistics`, `solve_mixed` |
| A | Increment table `*_trans` row; Consistency, Increments | a transitory profile needs its operating profile also where the key applies at no cell | `load_config`, line 907 |
| A | Result Dictionary `bound_method...` | scalar `null` if no cell is rainflow | `collapse_selector(..., empty_as_none=True)` |
| A | `optimization_progress` | `nodes` float, `solutions` integer; the two formulas of `relative_gap` (during the solve; last record = `MIPGap`) | `_make_progress_callback`, `_append_final_progress` |
| A | Rules for the Values | new item "Entries of a builder": `extract` hook entries returned, not written | `solve_mixed`, `_text_output` |
| M1 | Run Options, `gurobi_params` | set also after the builders' preparation; a `NonConvex` entry replaces the 2 | `build_fleet` |
| M5 | Result Dictionary `objective, mip_gap, bound` | non-finite best bound or gap is `null` | `finite_or_none`, `extract_solution` |
| M18 | Result Dictionary `x`, `m`, `r`, `idle`; Rules for the Values, new item "Binary values" | rounded to exactly 0 or 1; `idle` and the cost report from the rounded values | `_read_binary_values`, `extract_solution` |
| Start | Result Dictionary: two rows `y_mu..y_K`, `y_g_mu..y_g_K` (`F x L`, NaN pattern; refer to `sec:loop`); Rules, "Upper bounds" list; HDF5 Layout dataset row | the eight keys | `RESULT_KEYS`, `_HDF5_DATASET_KEYS`, `extract_solution` |
| M2, M3 | none | already stated: Inputs intro "An integer key ... refuses ... a flag"; "where it applies, `null` is an error" | `_integer`, `_component_names` |

Disagreements and open points
- The last record of `optimization_progress` takes `best_bound` as `float(model.ObjBound)` without `finite_or_none`: an infinite bound there is kept in the list returned by `solve()` (and written as `null` in every file). M5 covers only the keys `bound` and `mip_gap`. The spec text says only "`null` while Gurobi has none". Code to decide.
- `y_g_v` pattern is written as the pattern of `g_v` (the code builds the anchors where `add_state_vars(anchored=True)`); not checked against each cell builder.
- The plan says `read_result` returns the y arrays as floats: covered by the existing sentence on `float64` datasets.
- Section labels used and not defined in this fragment: `sec:loop`, `sec:dyn_gamma`, `sec:sweeps`, `sec:warm_starts`, `sec:assignment`, `sec:variables` (other writers).
