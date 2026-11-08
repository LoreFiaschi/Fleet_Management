Status: done (items 1, 2, 3). Not built. Line endings unchanged (0 CR before and after).

## Item 1: start of the repetition (code: `base.py` `add_state_vars`, `add_quantity_rows`, `add_repair_cost_rows`)

| Fragment | Location | Old | New |
|---|---|---|---|
| fragments_A | Notation table, after the anchors row | none | row for `y^s_{il}`, `y^{g,s}_{il}`, pointing to the Loop Constraint |
| fragments_A | Decision Variables table, after the anchor row | none | two rows: `y^mu, y^v, y^R, y^K` (F x L, continuous, `[0, M_s]`, every cell, per tracked quantity) and `y^{g,.}` (ARD1 cells, per anchor) |
| fragments_A | Decision Variables, closing paragraph | "$M_v$, $M_R$, $M_K$ ..." | adds `M_s`; the output reports the tracked values and `y` |
| fragments_A | Encoding, "Rules for both forms" | none | new bullet "Day H1+1": rows read `y`, `y^g` instead of day H1, both forms (carry, repair, anchor, Wiener row); replacement and anchor-set rows unchanged |
| fragments_A | Encoding, reference trajectory bullet | no `y` | adds: in zeta, `y^s = s_{H1}`, `y^{g,s} = g^s_{H1}` |
| fragments_A | Repair Cost, after Eq. a_z | "with g^mu = 0 under ARDinf" | adds: day H1+1 reads `y^mu`, `y^{g,mu}` (0 under ARDinf) |
| fragments_A | Loop Constraint | row `s_T <= s_{H1}` (Eq. a_loop) | new Eq. `a_start` (`y >= s_{H1}`, `y^g >= g_{H1}`); Eq. `a_loop` now `s_T <= y^s`, `g^s_T <= y^{g,s}`; the reliability statement, the pointer and the Consequences paragraph are kept |
| fragments_B | Gaussian, intro of the big-M rows | none | sentence: day H1+1 reads `y`, `y^g` (covers the next four sections) |
| fragments_B | Wiener note | "loop row `v_T <= v_{H1}`" | "start rows `y^v >= v_{H1}` and loop row `v_T <= y^v`" (no other comparison of day T with H1 remains) |
| claude_full_problem | Rows of a cell, items 1, 2, 5, 6 | loop rows only | day H1+1 reads `y`/`y^g`; start rows and loop rows; repair-cost row of day H1+1 |
| claude_full_problem | Domains and table text | none | `y` continuous `[0, M_s]`, no day index; no reliability row reads `y` |

## Item 2: M16 (code: `gaussian.py` line 44, `wiener.py` line 48: `kappa = -float(ndtri(eps))`)

- fragments_B, Gaussian section: "`ndtri`(1 - eps)" became "-`ndtri`(eps) (the value of Phi^{-1}(1 - eps), exact for small eps)". No other `ndtri(1 - eps)` in my files.

## Item 3: B1-3 (code: `rainflow.py` lines 16-28, 48-69)

- fragments_B, Rainflow, after the table of tracked quantities: new paragraph on the guarded import of `base` and on `row_constants(bound, tau, eps, b, s)` (constants of the quadratic rows and caps of the linear rows; unknown bound is a `ValueError`).

## Disagreements and open points

- No disagreement with the code found in these items.
- `ndtri(1 - eps)` still appears in `spec_assembled.tex` only (generated); fragments C and D not checked for it.
- Result keys `y_mu` ... `y_g_K` (Result Dictionary, HDF5, validators, dashboards, greedy note) belong to other writers.
- The model sections write rows with `mu_{k-1}`; the day H1+1 exception is stated once (Gaussian intro and Encoding), not in each model section.
