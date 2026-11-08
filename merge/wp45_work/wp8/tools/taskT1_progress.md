Status: done. 133 tests passed, 0 skipped, 0 failed locally (runner: scratchpad run_t1.py, real package, no gurobipy). No code defect found.

# T1 progress (validation tests)

## Files (worktree C:\Users\utente\Documents\GitHub\Fleet_Management-merge)
- tests/test_spec_validation.py (133 test cases, parametrised ones counted separately).
- tests/data/tools/validation/: gaussian_{input,result}.yaml, gamma_ardinf_{input,result}.yaml, gamma_ard1_{input,result}.yaml,
  margins_{input,result}.yaml (ten cells, one per row of the margin table), mixed_mc_{input,result}.yaml (one Gamma cell + three Gaussian).
  Each file has a first comment line; each result derives its numbers by the spec's day maps in comments.
- Broken copies are made in tmp_path from these files (one change per check).

## Coverage
validate (valid pair, order, each check incl. 8 loop_q, objective, tol, status rule, objective in 3 modes, C_D = 0, saved u, null z);
errors (missing file, extensions, no x, failing input, sizes F/M/L/H1/H2, shapes, missing m/r/mu/z/u/objective, non-finite x/m/r/idle);
report path (.json, .yml, no extension, folder, default); check_result; dashboards (import without streamlit, load_inspector,
margins of all rows G, IG, Ga, Mk, Ca, Ho, Be, Ch, W, overview, tolerance, NaN left out, repeatability table, model mismatch,
load_gamma_diagnostic incl. no-Gamma ValueError and the tail/mean tables); replay validator (valid pairs, each error field, each
global violation, events, raise, tolerance, input errors, files: suffix before reading, write before raising); exact validator (tails
vs scipy: single term, equal rates, convolution by quad, repair x 1/(1-rho), ard1 frozen part, known constant, c >= tau rules,
include_steps, keys, RuntimeError, argument errors); Monte Carlo (4 standard errors, same report, Clopper-Pearson incl. f = n, pass
rule, argument errors, excluded_cells); dispatcher (three modes, default, None rule, errors, raise, files).
Gamma cap A1 computed in the test file (helper gamma_A1, as wp8_plan_oracle.md section 2); IG cap by scipy.invgauss + quad.

## Code defects
None found. Observed behaviour matched every assertion.

## Spec questions / assumptions the tests make (for Claude to classify)
1. Spec 2428/2435: when the result has no key `idle`, is the check `idle` listed (violation 0) or left out? The test accepts both
   (only the order of the listed names is asserted).
2. Spec 2427: with allow_replacement false and a replacement saved, depot_gating violation is asserted as 1.0 ("excess of x^r over 0").
3. Spec 2540 "a result horizon other than T": tested by saving `T: 4`; code reports one global violation.
4. Spec 2544: an intervention on a mission day gives two strings in the code (with a mission, without a depot day); the spec
   says "one string per occurrence". The test accepts 1 or 2.
5. Spec 2602 c >= tau: input checks forbid such constants, so the test sets them on a deep copy of the loaded configuration
   (cfg.mu_0, cfg.replacement_mu).
6. PermissionError (2413) is not tested (not portable). The null-for-non-finite rule of the report file is tested only as "no NaN/Infinity
   token in the JSON file" (no case produced a non-finite report value).
7. Column names of the dashboard tables (`margin`, `holds`, `value_at_H1`, `tail_upper_bound`, `slack`...) and the report key
   `worst_time_step` were taken from the returned mappings; the spec names them only in prose.
8. Exact validator on a repair of a known constant: the constant is multiplied by 1 - rho also under ardinf (needed for the exact mean to
   equal the day map); spec 2601 states it under ard1 only.
