Status: done. 76 tests, all pass locally (0 skipped, 0 failed; runner in the scratchpad, no Gurobi needed). No code defect found.

- Files: tests/test_spec_plots.py; tests/data/tools/plots/ (schedule_marks.yaml, schedule_precedence.yaml, schedule_no_mr.yaml, sweep_a.yaml, sweep_b.yaml, sweep_nogap.yaml), in the worktree Fleet_Management-merge.
- PNGs: scratchpad wp8/T4/schedule_plot_example.png and sweep_plot_example.png.
- Code defects: none.
- Spec findings / open points:
  - 2376: the line at maximum_mip_gap_for_stopping is drawn at 5 for 0.05 (axis in percent); the spec does not say the unit of the line. The test asserts 5.
  - 2343: the spec does not say which of the presolved counts is the number of variables; the test takes the 'columns' of the last Presolved line.
  - 2339-2341 and 2331: summary labels, the wording of the source line and the format of values are not fixed; tests find rows by loose label words.
  - 2341: counts of repairs and replacements for non-binary values (0.5 < v) are not defined; tested only with 0/1 data.
  - HDF5 result input not tested (schema outside the assigned lines); PermissionError test is a no-op where the system cannot make a folder read-only (Windows).
