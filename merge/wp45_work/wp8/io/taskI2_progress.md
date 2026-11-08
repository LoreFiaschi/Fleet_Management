# Task I2 progress

Status: done. `tests/test_spec_checks.py` written; local run 350 passed, 0 skipped, 0 failed. No code defect found.

- File: `C:\Users\utente\Documents\GitHub\Fleet_Management-merge\tests\test_spec_checks.py` (about 1480 lines, 350 cases after parametrisation).
- Runner: copy of `run_tests_gamma.py` in the scratchpad `wp8\I2\`.
- Content, by spec location:
  - Rejected keys 506-525: 21 keys (the plan said 22; the table has 21), the message names the canonical key and is not "unknown key"; unknown key 503; key check before arrays 504 (unknown and rejected key with a malformed `mu`).
  - Required keys 536 and 2750-2760: the 13 common keys, and per cell type (G v, IG eta, W sigma, Ga gamma_beta, Ca v, Ho support, Be v and support, Ch cgf and s_chernoff); Markov needs only common keys.
  - Shapes and entries 537-538; sizes and selectors 541-549 (F > M both sides, L, component_names, H, selectors, quadratic penalty by cell type).
  - Reliability, repair, initial state 554-561 (each check with its boundary; anchors; replacement_as_new).
  - Natural ranges 566 (G, W, Ca, Be, Ho, Ch bounds computed in the test: `ndtri` for kappa_G, formulas of 566).
  - Replacement-state rows 567: G and W exact, G tangent, G pwl (segment of mu_new), Ca, Ho, Be, Mk, Ch, v_max_user, IG, Ga.
    IG: the cap by `scipy.stats.invgauss(mu=eta, scale=1/eta)` and `scipy.integrate.quad`, with a sanity test against the spec's F (1107); 0.5 percent margin.
    Ga: no sharp cap; inside = eps tau (A_1 >= eps t, spec 1308), outside = above A_hat / beta* (A_1 <= A_hat, 1306-1308), A_hat by `brentq` on `gammaincc`.
  - Increments 571-575 (every day and mission, transitory lookup 439-448).
  - G, IG, W, Ga checks 577-584; rainflow checks 586-593 (Be b^2/4 three places, tau > 2 kappa b/3, tangent and pwl conditions 590 with c = 0.42, Ch s, K >= s mu at four places).
  - Run options and costs 597-599 (evaluation_horizon in every mode, mip_gap -1e-13 set to 0 and -1e-10 refused, positive options, mappings, flags, costs).
  - Tolerance 532: non-strict with pass at 5e-13 and fail at 1e-10 (anchor, rho, v_0, mu >= 3 sqrt v, K >= s mu, Be, v_max_user), scale test at tau = 1000, strict checks without tolerance.
  - Checks of one model run only at that model (531): G, W, IG, Ga, Ho, Be, Ch values at other cells, mixed fleet columns.
- Code defects found: none.
- Test errors fixed during the run (all my own): NaN at `gamma_beta_0` / `gamma_beta_new` means "not given" in the stored form, so it is not tested as an error; an input with one model string and no `L` has `L = 1` (the spec's "required if L > 1" cannot be tested by omission); helper and parameter slips.
- Open points:
  - Message content: the tests assert the key name (a word-boundary match), or, where the spec says the key of a transitory profile or two keys can be named, any one of them; for rows 512, 513, 517, 518, 519 the spec names no canonical key, so only "not unknown key" is asserted.
  - Ga replacement row (567) is tested with the two bounds the spec gives, not with the sharp cap A_1; the sharp value would need a re-implementation of 1306-1308.
  - `pwl_points >= 1` is tested only at a pwl cell; the spec (559) does not say whether it is checked when no cell has the form pwl.
  - The tests import `fleet_management.config.load_config`; under real pytest the package `__init__` needs gurobipy (Euler has it), as for I1.
