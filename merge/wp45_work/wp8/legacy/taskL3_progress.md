Status: done. Next: Claude reviews and submits `euler/run_smoke.sh` on Euler (real run).

- [x] 10 inputs in `Fleet_Management-merge/examples/smoke/` (all pass `load_config`; `mixed_example.yaml` body verbatim from spec 2769-2827, checked with diff)
- [x] `remote_smoke_test.py` rewritten (12 scenarios; options `--no-pytest`, `--dry-run`, `--out-dir`)
- [x] `euler/run_smoke.sh` (0 CR characters, `bash -n` ok)
- [x] local checks: `ast.parse` ok; `--dry-run` from a scratchpad copy: 16 inputs (10 files + 6 script variants) load, 0 fail

Open points (not testable locally; no Gurobi):
- Feasibility of the loop constraint (state at T <= state at H1) was argued by hand, not solved. Design rule used: F=3, M=1, mu_0 (0.15) >= replacement value + one increment, so a replacement before the last-day flight keeps the loop. Wiener and Hoeffding initial values (v_0, R_0) set above replacement + one increment for the same reason.
- `replacement_no_replacement` swaps the Wiener cell for a Gaussian cell (Wiener variance falls only by replacement).
- `mixed_example` is the spec example as is (mip_gap 0.05, time_limit 600); its feasibility is claimed by the spec only.
- Scenario `forms` asserts bigm and indicator objectives agree within 2.1 %.
- Scenarios 10 and 11 reuse `five_models.yaml`; the `validation` block there (stochastic, 20000) drives the second validator call.
