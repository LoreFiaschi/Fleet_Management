# Map of the blocks moved from the spec to `scoping.tex`

Status: done (26 Oct 2026). All 145 blocks have a row below; line numbers refer to `scoping.tex` after the final build
of 26 Oct 2026 (60 pages, 0 errors, 0 undefined, 0 bibtex warnings, no `Float too large`).
Every block of `spec_drafts/moved_A.md`, `moved_B.md`, `moved_C.md`, `moved_D.md` and `moved_Claude.md` has one row.

## Spec pointers to the scoping document (for Claude to fix in the spec)

| Spec line (`spec_assembled.tex`) | Pointer as written | Correct section in `scoping.tex` |
|---|---|---|
| 137 | "Stochastic loop constraint" (forced replacements) | correct: subsection "Stochastic loop constraint", subsubsection "Replacements required by the loop constraint" (`sec:loop_replacements`) |
| 536 | "Stochastic loop constraint" (tracked values are upper bounds) | correct: its subsubsection "One-sided encoding of the dynamics" (`sec:encoding`, Proposition "Domination") |
| 808 | "Conservative surrogates for the empirical damage-increment distribution" (each linear form implies the exact row) | **change to** "Linear forms of the reliability rows" (`sec:linear_forms`, Proposition "Safety of the linear forms") |
| 890 | "Stochastic orderings" (inverse Gaussian, no independence) | correct (`sec:stoch_orders`, paragraph "Inverse Gaussian: a cap on the mean") |
| 1004 | "Stochastic orderings", loop constraint for each degradation model (Wiener replacement) | correct (`sec:loop_per_model`, paragraph "Wiener") |
| 1177 | "Conservative surrogates ..." (random initial damage needs `R_0`, `cgf_0`) | correct (`sec:cons_surrogates`, paragraph "Starting and replacement values of the Chernoff statistic") |
| 1289 | "Cost function" (`z` upper bound, starting-anchor exception) | correct (`sec:cost_function`, paragraph "Repair cost", Proposition "What z measures") |
| 1329 | "Stochastic loop constraint" (perpetual safety) | correct (subsubsection "Perpetual safety", `sec:perpetual`) |
| 1336 | "Stochastic loop constraint" (ARD1 replacement, `allow_replacement = false`) | correct (`sec:loop_replacements`) |

## Facts of the key and result rows that the condensed spec does not hold (for Claude to restore in the spec)

Checked 26 Oct 2026 against `spec_assembled.tex` (rows of M-C-13, M-C-14, M-C-29). These are specification facts, so
they are not placed in the scoping document:
- `L`: "a given `L` must equal the length of the component axis of `model`" (no such rule found in the spec).
- `component_names`: "a name is not empty after blanks are removed" (no such check found).
- `v_max_user`: "if it is not given, only the reliability row bounds the variance" (implied; optional to restore).
- All other wording of the three blocks is present in the spec rows, the checks or the output section, or is a reason
  now in the scoping document (the meaning of the default MIP gap: `sec:impl_outputs`).

- **Actions**: *added* (new text in the scoping); *merged into <label>* (the scoping held part of it; the rest is
  added there); *already present at <label>* (nothing to add); *not placed* (spec content; see the note).
- Notation of the scoping is kept: "component" for the spec's "cell", "member" for "train", `\pi_k` for the day map
  (the scoping's `\phi` is the aggregation function), `\Pi` for the map of the operating phase.

## Planned structure of the new scoping text

| Destination (section of `scoping.tex`) | Blocks |
|---|---|
| 1.1 Scheduling and maintenance variables | M-A-5, M-A-6, M-C-27 |
| 1.2 Damage dynamics: models paragraph; horizon; state after a replacement | M-A-1, M-C-5, M-C-7, M-A-22, M-C-4 |
| 1.3 Stochastic loop constraint, new subsubsection "One-sided encoding of the dynamics" | M-A-12, M-A-15, M-A-16, M-A-17, M-A-18, M-A-21 |
| 1.3, subsubsection "Perpetual safety" | M-A-37, M-A-38 (full proof replaces the sketch), M-A-39, M-A-40 |
| 1.3, subsubsection "What the one-sided rows accept" | M-A-19, M-A-20 |
| 1.3, subsubsection "Replacements required by the loop constraint" | M-A-41 (with the proposition moved from 2.3.4), M-A-23 |
| 1.4 Cost function: objective modes; repair cost; damage penalty | M-A-2, M-A-3, M-D-1, M-D-2, M-C-26, M-A-31, M-A-32, M-A-33, M-A-34, M-A-35, M-A-36, M-C-10 |
| 1.5 Abstract problem formulation | M-A-4 |
| 2.1.4 Conservative surrogates: rainflow model, assumptions, width, day maps, derivations of the five rows | M-B-39, M-B-40, M-B-41, M-C-8, M-C-23, M-B-42, M-B-43, M-C-3, M-B-44, M-B-45, M-B-46, M-B-47, M-B-48, M-B-49, M-B-50 |
| 2.1.5 Loop under the Route-I bounds | (cites M-B-42) |
| 2.1.7 new subsubsection "Linear forms of the reliability rows" | M-A-24, M-A-25, M-B-6, M-A-26, M-A-27, M-B-51, M-A-28, M-A-29, M-A-30 |
| 2.2.1 Within each family: inverse Gaussian | M-B-9, M-B-10, M-B-11, M-B-12, M-B-13, M-B-14, M-B-15, M-B-16 |
| 2.2.1 Within each family: Gamma | M-B-25, M-B-26, M-B-27, M-C-9, M-B-28, M-B-29 (full proof replaces the sketch), M-B-30, M-B-31, M-B-32, M-B-33, M-B-34, M-B-35, M-B-36, M-B-37 |
| 2.2.2 Loop per model: Gaussian, Wiener, facts (M1)-(M3) of each model | M-B-1, M-B-2, M-B-3, M-B-4, M-B-5, M-B-7, M-C-1, M-B-8, M-B-18, M-B-19, M-B-20, M-B-21, M-B-22, M-B-23, M-B-24, M-B-17, M-B-38, M-B-52 |
| 2.3.3 Reduction-based models; 2.3.4 Connection | M-A-7, M-A-8, M-A-9, M-A-10, M-C-2, M-A-11 |
| 5 Implementation details, new subsections | encoding forms: M-A-13, M-A-14; checks: M-C-15 to M-C-22; outputs: M-C-24, M-C-25, M-C-28, M-C-30; sweeps: M-D-3, M-D-4, M-D-5, M-D-18; warm starts: M-D-6 to M-D-10, M-D-19; validation: M-D-14 to M-D-17; package: M-D-11, M-D-12, M-D-13, M-C-6, M-C-11, M-C-12, M-C-31; open questions: M-Cl-1 |
| not placed (spec content) or dropped | M-C-13, M-C-14, M-C-29 (key and result rows: spec content; to be checked row by row against the spec); M-Cl-2 (bibliography already complete) |

## Rows

| Block | Destination (section, label, lines) | Action |
|---|---|---|
| M-A-1 | 1.2 Damage dynamics, paragraph "Degradation models of the formulation", 235-243 | added |
| M-A-2 | 1.4 Cost function, `sec:cost_function`, 555-560 and 589-590 | merged into `sec:cost_function` (C_M, C_R present; "C_D u added once in every mode" added at 589); "C_rep has no default" is spec content (key table) |
| M-A-3 | 1.4, paragraph "Objective modes", 582-599 | added |
| M-A-4 | 1.5 Abstract problem formulation, 697-701 | added |
| M-A-5 | 1.1 Scheduling, 140-146 | merged into 1.1 (idle row and pricing present; "no damage on idle and depot days, depot day explicit" added at 144-145) |
| M-A-6 | 1.1, Remark `rem:one_depot_row`, 158-162 | added |
| M-A-7 | 2.3.3 Reduction-based models and 2.3.4, 2127-2166, 2180 | merged into 2.3.3 (citations present from step 5; "No law of the formulation acts on a virtual age" added at 2180) |
| M-A-8 | 2.3.4, paragraph "Repair maps of the tracked statistics", 2195-2203 | merged (derivation of the mean factor added) |
| M-A-9 | 2.3.4, paragraph "Anchors", 2182-2193 | merged (worked example 0.6/0.2 added) |
| M-A-10 | 2.3.4, paragraph "Anchors", 2187-2193 | already present at 2.3.4 (default starting anchor, example 0.5) |
| M-A-11 | 2.3.4, paragraph "Laws per model", table `tab:laws_per_model`, 2205-2224 | added |
| M-A-12 | 1.3.1 `sec:encoding`, 284-293 | added |
| M-A-13 | 5.1 `sec:impl_encoding`, paragraph "Validity of the big-M constants", 2537-2541 | added (with the rows stated in words at 2525-2529) |
| M-A-14 | 5.1, paragraph "Choice of the default form", 2543-2548 | added |
| M-A-15 | 1.3.1, list of objects, 295-306 | added (day map written `\pi_k`) |
| M-A-16 | 1.3.1, facts (M1)-(M3) and table `tab:chance_rows`, 308-337 | added |
| M-A-17 | 1.3.1, Lemma `lem:monotone`, Proposition `prop:domination`, 339-363 | added |
| M-A-18 | 1.3.1, 365-368 | added |
| M-A-19 | 1.3.3 `sec:acceptance`, Proposition `prop:acceptance`, 432-473 | added |
| M-A-20 | 1.3.3, two examples, 475-502 | added |
| M-A-21 | 1.3.1, paragraph "Meaning of the reported values", 369-379 | added |
| M-A-22 | 1.2, paragraph "State after a replacement", 245-252 | added (the check itself: 5.2) |
| M-A-23 | 1.3.4 `sec:loop_replacements`, 507-510 | added |
| M-A-24 | 2.1.7 `sec:linear_forms`, Eq. `eq:quad_row`, table `tab:quad_constants`, 1319-1342 | added |
| M-A-25 | 2.1.7, item `exact`, 1346-1350 | added (with the Gaussian example of M-B-6) |
| M-A-26 | 2.1.7, item `tangent` 1351-1353, paragraph "The tangent point of a Bernstein row" 1384-1393 | added |
| M-A-27 | 2.1.7, paragraph "The tangent point of a Bernstein row", 1391-1393 | added |
| M-A-28 | 2.1.7, Proposition `prop:tangent_safe`, 1360-1369 | added |
| M-A-29 | 2.1.7, paragraph "Accuracy", 1371-1377 | merged (my step-5 text replaced by the block's fuller wording) |
| M-A-30 | 2.1.7, Example (Cantelli), 1379-1382 | added |
| M-A-31 | 1.4, paragraph "Repair cost", Eq. `eq:z_row`, 601-614 | added |
| M-A-32 | 1.4, paragraph "Repair cost", 611-614 | added |
| M-A-33 | 1.4, Proposition `prop:repair_cost` and the exception, 616-645 | added |
| M-A-34 | 1.4, 576 | added |
| M-A-35 | 1.4, 571 | already present at `sec:cost_function` (step 5) |
| M-A-36 | 1.4, 573-575 | added |
| M-A-37 | 1.3.2 `sec:perpetual`, 384-386 | added (endless schedule itself: 1.2, 230) |
| M-A-38 | 1.3.2, Theorem `thm:perpetual` with full proof and scope, 388-419 | merged: replaces my proof sketch (`prop:perpetual`, removed) |
| M-A-39 | 1.3.2, paragraph "Why the anchors enter the loop", 421-428 | merged (peak 0.525 sentence added) |
| M-A-40 | 1.3.2, 429-430, and 2.1.5 `sec:loop_routeI` (example) | merged (example already in 2.1.5 footnote; closing sentence added) |
| M-A-41 | 1.3.4, Proposition `prop:ard1_replacement`, 512-540 | merged: proposition moved here from 2.3.4; `allow_replacement = false` consequence added at 538-540 |
| M-B-1 | 2.2.2 `sec:loop_per_model`, paragraph "Gaussian", 1971-1975 | added (model statement, N1, negative damage) |
| M-B-2 | 2.2.2, Proposition `prop:gauss_exact`, 1976-1990 | added (replaces my prose) |
| M-B-3 | 2.2.2, 1992 | added |
| M-B-4 | 2.2.2, 1993-1994 | merged |
| M-B-5 | 2.2.2, "Validity", 1995-1998 | added |
| M-B-6 | 2.1.7, item `exact` (nonconvexity example), 1347-1350 | added |
| M-B-7 | 2.2.2, Remark `rem:vmax`, 1997-2005 | added (merged with M-C-1) |
| M-B-8 | 2.2.2, Gaussian facts (M1)-(M3), 2007-2010 | added |
| M-B-9 | 2.2.1 `sec:stoch_orders`, paragraph "Inverse Gaussian: a cap on the mean", 1657-1663 | merged (density, variance, squared coefficient of variation added) |
| M-B-10 | 2.2.1, 1667-1672 | merged ("mean at most the tracked mean" added) |
| M-B-11 | 2.2.1, Proposition `prop:ig_order`, 1674-1690 | already present at `prop:ig_order` (step 5) |
| M-B-12 | 2.2.1, Theorem `thm:ig_cap`, 1694-1715 | already present at `thm:ig_cap`; "built exact, form ignored" is spec content |
| M-B-13 | 2.2.1, 1716-1717 | merged (change-of-variable step added) |
| M-B-14 | 2.2.1, table `tab:ig_cap`, 1718-1733 | added |
| M-B-15 | 2.2.1, 1736-1738 | merged (Jensen step, "can reject safe schedules" added) |
| M-B-16 | 2.2.1, 1739-1742 | merged (ratio 0.93-0.59 added) |
| M-B-17 | 2.2.2, paragraph "Inverse Gaussian", facts, 2042-2043 | added |
| M-B-18 | 2.2.2, paragraph "Wiener", 2013-2016 | added |
| M-B-19 | 2.2.2, paragraph "Wiener", 2016-2018 | merged (shift formulas added) |
| M-B-20 | 2.2.2, paragraph "Wiener", 2019 | added |
| M-B-21 | 2.2.2, paragraph "Wiener", 2020-2021 | added |
| M-B-22 | 2.2.2, paragraph "Wiener", 2019-2020 | merged; optional row `v_max_user` is spec content |
| M-B-23 | 2.2.2, paragraph "Wiener", 2022 | added |
| M-B-24 | 2.2.2, paragraph "Wiener", 2023-2025 | merged ("no input check; no flight in the operating phase without replacements" added) |
| M-B-25 | 2.2.1, paragraph "Gamma: a sharp cap on the mean", 1747-1749 | added |
| M-B-26 | 2.2.1, 1750 | added |
| M-B-27 | 2.2.1, 1751-1755 | added |
| M-B-28 | 2.2.1, Definition `def:gamma_class`, Lemma `lem:gamma_class`, 1759-1783 | merged (c to 0 remark, proof steps added) |
| M-B-29 | 2.2.1, W, atoms, Lemma `lem:gamma_rates`, Theorem `thm:gamma_oneterm` with full proof, 1785-1838 | merged: full proof replaces my sketch |
| M-B-30 | 2.2.1, Lemma `lem:gamma_monotone`, 1841-1848 | added |
| M-B-31 | 2.2.1, 1850-1858 | merged (w continuous, < 1/2, limit 1/2, eps >= 1/2) |
| M-B-32 | 2.2.1, Proposition `prop:gamma_safe`, 1860-1871 | merged (proof with W and the monotone lemma) |
| M-B-33 | 2.2.1, paragraph "Computation of the cap", 1873-1879 | added (justifications only; the steps stay in the spec) |
| M-B-34 | 2.2.1, paragraph "The sharp cap and the exact one-rate cap", 1881-1901 | merged (definition of t1 and table added) |
| M-B-35 | 2.2.1, paragraph "Below the threshold", 1903-1906 | merged (the reason Q(rs, s) added) |
| M-B-36 | 2.2.1, paragraph "The choice of beta*", 1908-1911 | merged (the reason added) |
| M-B-37 | 2.2.1, 1912-1913 | merged |
| M-B-38 | 2.2.2, paragraph "Gamma", facts, 1955 | added |
| M-B-39 | 2.1.4 `sec:cons_surrogates`, paragraph "The five bounds of the formulation", 1064-1068 | added |
| M-B-40 | 2.1.4, assumptions A0-A3, 1074-1086 | added (input names left to the spec) |
| M-B-41 | 2.1.4, paragraph "Width of a range", 1106-1115 | added |
| M-B-42 | 2.1.4, paragraph "Day maps of the statistics", Proposition `prop:rf_maps`, 1117-1155 | added; my duplicate derivation in 2.1.5 now cites it (1250) |
| M-B-43 | 2.1.4, paragraph "Starting and replacement values of the Chernoff statistic", 1190-1205 | merged (input-check sentence and random budget added) |
| M-B-44 | 2.1.4, 1153-1155 | added |
| M-B-45 | 2.1.4, "Derivation and validity of the rows", Markov, 1159-1160 | added |
| M-B-46 | 2.1.4, Cantelli, 1161-1167 | added |
| M-B-47 | 2.1.4, Hoeffding, 1168-1172 | added |
| M-B-48 | 2.1.4, Bernstein, 1173-1179; support constant at 1029-1031 | merged |
| M-B-49 | 2.1.4, paragraph "The Bernstein side row", 1185-1189 | merged ("at mu = v = 0 the row needs tau >= 2c" added) |
| M-B-50 | 2.1.4, Chernoff, 1180-1183 | added |
| M-B-51 | 2.1.7, paragraph "The tangent point of a Bernstein row", 1384-1393 | merged with M-A-26 and M-A-27 (one place) |
| M-B-52 | 2.2.2, paragraph "Rainflow", 2045-2053 | added |
| M-C-1 | 2.2.2, Remark `rem:vmax`, 1997-2005 | merged with M-B-7 |
| M-C-2 | 2.3.4, paragraph "Anchors", 2187-2193 | already present at 2.3.4 (same content as M-A-10) |
| M-C-3 | 2.1.4, 1203-1205 | merged ("the defaults describe a known damage" added) |
| M-C-4 | 1.2, paragraph "State after a replacement", 245-252 | merged; pointers to spec sections dropped; Gamma replacement rate already at 1746 |
| M-C-5 | 1.2, horizon paragraph and Eq. `eq:dyn_full` (initial condition), 221-230 | already present (step 5) |
| M-C-6 | 5.7 `sec:impl_package`, 2697-2698 | added |
| M-C-7 | 1.2, horizon paragraph, 231-233 | added |
| M-C-8 | 2.1.4, paragraph "Width of a range", 1107 | added |
| M-C-9 | 2.2.1, 1751-1752 | merged ("no input gives it" added) |
| M-C-10 | 5.2, "All costs are finite", 2613 | dropped as a pointer; the cost rule is in 5.2 and the modes in 1.4 |
| M-C-11 | 5.7, 2694-2696 | added |
| M-C-12 | 5.7, 2696-2697 | added |
| M-C-13 | — | not placed: key-table wording is spec content; checked against the spec (two facts to restore: see the section above) |
| M-C-14 | — | not placed: key-table wording is spec content; the meaning of the default MIP gap added to 5.3 (2620) |
| M-C-15 | 5.2 `sec:check_reasons`, 2549-2565 | added |
| M-C-16 | 5.2, "Sizes and selectors", 2567-2572 | added |
| M-C-17 | 5.2, "Reliability, repair and initial state", 2574-2584 | added |
| M-C-18 | 5.2, "Natural ranges and replacement states", 2586-2590 | added |
| M-C-19 | 5.2, "Increments", 2592-2597 | added |
| M-C-20 | 5.2, "Model-specific checks", 2599-2604 | added |
| M-C-21 | 5.2, "Model-specific checks" (Bernstein, Chernoff), 2605-2610 | added |
| M-C-22 | 5.2, "Run options and costs", 2612-2615 | added |
| M-C-23 | 2.1.4, paragraph "Width of a range" (Popoviciu, proof and example), 1110-1115 | added |
| M-C-24 | 5.3 `sec:impl_outputs`, 2618-2620 | added |
| M-C-25 | 5.3, 2621-2622 | added |
| M-C-26 | 1.4, paragraph "Objective modes", 597-598 | added |
| M-C-27 | 1.1, 156 | added (the key `idle` and its plot marks are spec content) |
| M-C-28 | 5.3, 2623-2624 | added |
| M-C-29 | — | not placed: result-table wording is spec content, present in the spec rows (checked) |
| M-C-30 | 5.3, 2625 | added |
| M-C-31 | 5.7, input example, 2699-2704 | added |
| M-D-1 | 1.4, paragraph "Objective modes", 587-589, and 5.4 `sec:impl_sweeps`, 2629-2633 | added |
| M-D-2 | 1.4, objective modes, 588-590; 5.4, 2632 | added |
| M-D-3 | 5.4, "What a proof certifies", 2635-2637 | added |
| M-D-4 | 5.4, "Early stop", 2638-2639 | added |
| M-D-5 | 5.4, "Parallel cases", 2640-2641 | added |
| M-D-6 | 5.5 `sec:impl_warm`, 2650 | added |
| M-D-7 | 5.5, "Day mapping", 2652-2656 | added |
| M-D-8 | 5.5, "Start and hints", 2657-2660 | added (the hint priority not set is spec content) |
| M-D-9 | 5.5, "Relaxation start", 2661-2662 | added |
| M-D-10 | 5.5, "Greedy heuristic", 2663-2664 | added |
| M-D-11 | 5.7, 2690-2691 | added |
| M-D-12 | 5.7, 2687 | added |
| M-D-13 | 5.7, 2692-2693 | added |
| M-D-14 | 5.6 `sec:impl_validation`, "Objective check" and "Loop checks", 2670-2673 | added |
| M-D-15 | 5.6, "Margins in the dashboards", 2674-2676 | added |
| M-D-16 | 5.6, "Exact Gamma validator", 2677-2679 | added (the operational rule stays in the spec) |
| M-D-17 | 5.6, "Monte Carlo validator", 2680 | added |
| M-D-18 | 5.4, 2629-2630; 5.7, 2685-2686, 2692-2694 | added |
| M-D-19 | 5.4 "Report" 2642-2643; 5.5 "Start and hints" 2658; 5.7 builders 2688-2689; refusal with parallel cases 2641 | added |
| M-Cl-1 | 5.8 `sec:open_questions`, 2706-2731 | added ("train level" written "member level") |
| M-Cl-2 | `scoping_bib.bib` | dropped: the four sources are already entries `DoyenGaudoin2004`, `MercierCastro2019`, `RoostaKhorasaniSzekely2015`, `SzekelyBakirov2003` |

## Notes

- New labels: `rem:one_depot_row`, `sec:encoding`, `tab:chance_rows`, `lem:monotone`, `prop:domination`,
  `sec:perpetual`, `thm:perpetual`, `sec:acceptance`, `prop:acceptance`, `sec:loop_replacements`, `eq:z_row`,
  `prop:repair_cost`, `prop:rf_maps`, `sec:linear_forms`, `eq:quad_row`, `tab:quad_constants`, `prop:tangent_safe`,
  `tab:ig_cap`, `lem:gamma_rates`, `lem:gamma_monotone`, `prop:gauss_exact`, `rem:vmax`, `tab:laws_per_model`,
  `sec:impl_encoding`, `sec:check_reasons`, `sec:impl_outputs`, `sec:impl_sweeps`, `sec:impl_warm`,
  `sec:impl_validation`, `sec:impl_package`, `sec:open_questions`. Removed: `prop:perpetual` (now `thm:perpetual`).
- The Moschopoulos series (M-D-16) is named without a citation: no source of it was read, and the brief allows no
  new citation. Question for Claude: cite it once a source is read.
- Line numbers are approximate to two or three lines for the merged rows; the section and label are exact.