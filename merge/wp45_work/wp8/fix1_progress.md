Status: done. D1, D2, X1, X2 fixed; local tests pass (counts below). Not run: Gurobi solves (Euler).

## Fixes
- **D1** `gamma_replay_validator._schedule_violations`: one loop over the interventions; one string per cell and day with a mission or without a depot day ("... with a mission and without a depot day" when both). Test `test_replay_global_violation_intervention_on_mission_day` (test_spec_validation.py) now expects exactly 1. Older test `test_replay_global_violations` (test_gamma_validators.py) asserted the two strings; changed to the one string.
- **D2** `plotter._draw_statistics`: values wrapped at 22 characters (`_STATISTICS_VALUE_WIDTH`, `textwrap`), the next row moves down by 0.044 per extra line; `wrap=True` removed. New test `test_schedule_summary_panel_texts_do_not_overlap` (test_spec_plots.py): five models, both laws, no two text boxes of the panel overlap; it fails on the old plotter and passes now. PNG: scratchpad `wp8\fix1\summary.png`.
- **X1** `test_spec_outputs.py`: renamed `test_missing_input_is_checked_before_bad_result_extension`, expects `FileNotFoundError`.
- **X2** `tests/data/io/valid/rainflow_bounds.yaml`: `mu_0: [0.02, 0.10, 0.10, 0.10, 0.10]`, `mu: [[0.02], [0.25], [0.25], [0.25], [0.25]]` (Markov component only changed). Reference-model check (scratchpad `wp8\fix1\feas.py`): trains fly in turn (day k: train k mod 3); a train in the depot repairs all components once it has flown; `row_violations(cfg, sol, tol=1e-9, bigm_tol=0)` is empty (max Markov mean 0.04 <= 0.1). The old numbers give reliability violations. Test changed: `test_default_anchors_equal_initial_values_rainflow` (test_spec_inputs.py), `anchor_mu_0` is 0.02 at the Markov column, 0.10 elsewhere.

## Local counts (before -> after; runner with warns/approx stubs)
- test_spec_validation 133 -> 133; test_spec_plots 76 -> 77; test_spec_inputs 83 -> 83; test_spec_checks 350 -> 350
- test_spec_outputs 14 passed, 128 skipped, 1 failed -> 15 passed, 128 skipped, 0 failed
- test_validator 38 -> 38; test_gamma_validators 60 -> 60 (after the one-line change); test_plotter 20 -> 20; test_oracle_unit 138 -> 138
