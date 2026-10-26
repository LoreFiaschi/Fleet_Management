# Explicit big-M rows in the model sections (fragments_B.tex)

Status: done (26 Oct 2026). Done: Gaussian, inverse Gaussian, Wiener, Gamma, rainflow. Next: nothing; the notes
below wait for Claude. `fragments_B.tex` grew from 459 to 628 lines.

Task (user: "I don't see the equations in the case of the Big-M representation"): in each model section, the table
(state, increment, factor a, range, initial value, replacement value) and the sentence that points to the generic
rows are replaced by the explicit big-M rows of the model, with the model's own symbols and big-M constants.

## Models

| Model | Done | Labels added | Build |
|---|---|---|---|
| Gaussian | yes | `eq:g_mu_carry`, `eq:g_mu_inf`, `eq:g_mu_one`, `eq:g_mu_repl`, `eq:g_v_carry`, `eq:g_v_inf`, `eq:g_v_one`, `eq:g_v_repl`, `eq:g_anchor_set`, `eq:g_anchor_keep` | 0 errors, 0 overfull |
| Inverse Gaussian | yes | `eq:ig_mu_carry`, `eq:ig_mu_inf`, `eq:ig_mu_one`, `eq:ig_mu_repl`, `eq:ig_anchor_set`, `eq:ig_anchor_keep` | 0 errors, 0 overfull |
| Wiener | yes | `eq:w_mu_carry`, `eq:w_mu_inf`, `eq:w_mu_one`, `eq:w_mu_repl`, `eq:w_v_repl`, `eq:w_anchor_set`, `eq:w_anchor_keep` (`eq:w_vrow` kept) | 0 errors, 0 overfull |
| Gamma | yes | `eq:gm_mu_carry`, `eq:gm_mu_inf`, `eq:gm_mu_one`, `eq:gm_mu_repl`, `eq:gm_anchor_set`, `eq:gm_anchor_keep` | 0 errors, 0 overfull |
| Rainflow | yes | `eq:rf_mu_*`, `eq:rf_v_*`, `eq:rf_R_*`, `eq:rf_K_*` (each `_carry`, `_inf`, `_one`, `_repl`), `eq:rf_anchor_set`, `eq:rf_anchor_keep` | 0 errors, 0 overfull |

47 new labels in all; `eq:w_vrow` is kept. Every label that other fragments reference is kept
(`sec:dyn_*`, `eq:g_row`, `eq:ig_row`, `eq:w_vrow`, `eq:gm_row`, `eq:gm_betastar`, `eq:rf_markov`, `eq:rf_cantelli`,
`eq:rf_hoeffding`, `eq:rf_bsupport`, `eq:rf_bernstein`, `eq:rf_chernoff`), and the internal ones (`eq:g_map_mu`,
`eq:g_map_v`, `eq:ig_map`, `eq:w_map_v`, `eq:gm_map`).

Final build (twice): 0 errors, 0 overfull or underfull boxes, 8 pages. Undefined references are only other writers'
labels: `eq:a_ind_sum`, `eq:a_ind_anchor`, `sec:consistency_checks`, `sec:encoding`, `sec:linear_forms`,
`sec:loop`, `sec:repair_laws`, `sec:replacement`. Pages 1, 3, 4, 6, 7 rendered and read.

## Check of the rows

- Script: `C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\bigm\check_bigm_rows.py`,
  output `check_bigm_rows_output.txt` beside it (session scratchpad; copy into `checks/` if it should be kept).
- It parses every labelled big-M row from the LaTeX source, turns it into a Python expression, and compares it with
  the generic row of `fragments_A.tex` (Eqs. `eq:a_bigm_carry`, `eq:a_bigm_repair`, `eq:a_bigm_repl`,
  `eq:a_bigm_anchor`) with the quantity, factor a (`eq:a_psi`) and big-M constant of the model. Equal slacks at 200
  random real points per inequality, binaries included: 83 comparisons, 0 failures. The anchor rows are checked for
  every pair (y, M_y) of the model.
- `eq:w_vrow`: at the three binary points (no intervention, repair, replacement), on `v in [0, V_max]` with a zero
  increment on a depot day, it gives the same lower bound on `v` as the generic carry and repair rows with a = 1.
- Mutation test: two injected errors (a wrong coefficient in `eq:rf_K_one`, a missing `x^r` term in `eq:g_v_carry`)
  were both reported.

## Notes and questions for Claude

- **One edit outside the replaced blocks.** The rainflow paragraph "Day maps" said "with the factor a of the table
  below"; that table is removed. The sentence now gives the factors: "1-rho for mu and K, (1-rho)^2 for v and R".
- **Conditions stated once.** The Gaussian "Big-M rows" paragraph holds the conditions for all five sections ("In
  this section and the next four": k = 1..T, e_k = x^m_k + x^r_k, which rows each law builds, allow_replacement =
  false). The other four sections point to it ("conditions in Section dyn_gaussian").
- **Symbols.** The anchor pattern uses `y` (not `s`), because `s` is the Chernoff parameter in the rainflow section;
  the same letter is used in the Gaussian section for one term per concept. The rainflow variance rows use `M_v`,
  defined as the value `V^C` (Cantelli) or `V^B` (Bernstein), because one row serves both bounds.
- **Wiener variance: "may" in fragment A.** `fragments_A.tex`, `sec:encoding`, says the Wiener no-intervention and
  repair rows "may be replaced by the single row" `eq:w_vrow`. `lore`'s code builds only the single row
  (`src/fleet_management/models/wiener.py`, lines 94-95); `main` has no Wiener model. Fragment B now states
  `eq:w_vrow` as the built row. Suggest "are replaced" in A.
- **Full problem section.** `claude_full_problem.tex`, item 1 of "Rows of a cell", still points to the generic rows
  and the factor table. It could point to the model rows instead (for example `eq:g_mu_carry`--`eq:g_anchor_keep`).
