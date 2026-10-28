# D. Documents: `lore` versus `main`

Status: done (all seven items, 30 Sep 2026). Next: none.

Merge base: 462ac1d (25 Jun 2026, "rainflow"). Evidence format: `branch:path:line` or a commit hash.

**Key findings**
- Main changed **no** `.tex` or `.bib` file after the merge base. All differences in `spec/spec.tex` and `scoping.tex` come from lore.
- `spec/main.tex` and `spec/review_report.tex` are May 2026 files that lore deleted. They are a literature review and an automated panel review of it, not a newer specification.
- Main therefore has no current specification: its README describes a different product (two horizons, five bounds, three encodings), and its `spec/` holds only the v0.2 spec and the May documents.
- Lore's spec v0.5 and lore's scoping disagree on the idle convention.

## 0. Correction to the brief (fact)

- `spec/main.tex`, `spec/review_report.tex` and `lore_check.txt` are **not new on main**.
  All three exist at the merge base 462ac1d. Lorenzo created them in 61be6c3 (26 May 2026, "refactoring for scoping").
- Main never changed them: `git diff --numstat 462ac1d main` is empty for all three.
- Lore **deleted** them: `spec/main.tex` and `spec/review_report.tex` in 90e9ef0 (7 Jul 2026), `lore_check.txt` in 651aa71 (30 Jul 2026).
- The same holds for `spec/spec.tex` and `scoping.tex`: main never changed them after the merge base. So every difference in these two files comes from lore.

## 1. spec/spec.tex

Evidence basis: both diffs read in full (`git diff 462ac1d main` is empty; `git diff 462ac1d lore` is 1859 lines, +1081/-166).

**Who changed what (fact)**

| Branch | Commits touching the file | Version on title page |
|---|---|---|
| main | none since 462ac1d | "Version 0.2 (revised)" (`main:spec/spec.tex:56`), 1867 lines |
| lore | 98721b7 (2 Jul, "added cantelli") gives v0.4 with 46 `RF-` markers; 90e9ef0 (7 Jul) gives v0.5 with 18 `LPR-` and 22 `REV-` markers | "Version 0.5" (`lore:spec/spec.tex:60`, header `:32`), 2782 lines |

Lore's change log states that the "v0.3" log entry corresponds to the page titled "v0.2 (revised)" (`lore:spec/spec.tex:2721-2740`). So main's spec is v0.3 in content.

**Section-by-section differences (lore relative to main; main = merge base)**

| Topic | main (v0.2 revised) | lore (v0.5) | Evidence |
|---|---|---|---|
| Models | 4: Gaussian, inverse Gaussian (IG), Wiener, Gamma | 5: adds **Rainflow**, distribution-free; per-mission mean, variance and optional support bound from rainflow counting | `lore:spec/spec.tex:83-88,103-117,1330-1680` |
| Reliability bounds | Gaussian/Wiener exact form called "rotated SOC" (second-order cone) | Same constraint re-labelled **nonconvex quadratic**, with a counter-example to convexity; Gurobi `NonConvex=2` | `main:spec/spec.tex:547-560`; `lore:spec/spec.tex:744-770` |
| Rainflow bounds | absent | **Cantelli** (default) and **Bernstein** (`tail_bound` input), each with exact and single-tangent LP (linear programme) forms; Bernstein support constant b is a schedule-independent maximum | `lore:spec/spec.tex:1482-1625` |
| Formulation token | `"socp"` (default) or `"lp"` | `"exact"` (default) or `"lp"`; `"socp"` kept as a deprecated alias with a warning | `main:spec/spec.tex:174`; `lore:spec/spec.tex:226-240` |
| Loop constraint | hard constraint **plus** soft penalty `C_P * L_loop` in the objective | hard constraint only; `C_P` removed everywhere; new remark that the guarantee survives slack | `main:spec/spec.tex:168,388,1132`; `lore:spec/spec.tex:520-540,656-697,1776` |
| Loop proof | FSD (first-order stochastic dominance) wording for Wiener | Moment-domination induction; "not FSD"; warning against looping on the scalar bound, with a numerical counter-example | `lore:spec/spec.tex:1088-1110,1627-1680` |
| Maintenance (ARA1, ARD1) | ARA1 for Wiener/Gamma; ARD1 for all | Unchanged set; rainflow is ARD1 only. Adds anchor loop rows for ARA1 Wiener and Gamma; corrects the ARA1 repair-cost anchor index k to k-1; opens an item on pre- versus post-repair anchor | `lore:spec/spec.tex:1115-1155,1305-1327,1745-1760,2671` |
| Idle days | implicit | explicit "idle = all-zero row"; `C_M` is a real decision cost | `lore:spec/spec.tex:562-575` |
| Horizon | `H` scalar or `[Hmin, Hmax]`, called "half-horizon"; two halves `[1,H]`, `[H+1,2H]` | **unchanged** structure | `main:spec/spec.tex:157`; `lore:spec/spec.tex:206` |
| Inputs | `tau` scalar; `alpha_target` in shape list; `C_P` | `tau` scalar or F x L; `alpha_target` removed; new `tail_bound`, `b_inc`, `b_0`, `b_new`; Bhatia-Davis and `tau > 2 kappa b / 3` checks | `main:spec/spec.tex:162`; `lore:spec/spec.tex:212,256-270,372-470` |
| Outputs | no `tail_bound`; HDF5 stores `alpha_target` | echoes `tail_bound`; HDF5 stores `epsilon`, `tail_bound`, `b_*` | `lore:spec/spec.tex:2114-2120,2330-2340` |
| Plot | 4 border styles | 5th style (long dash) for rainflow | `lore:spec/spec.tex:2281-2283` |
| Open questions | Hierarchical failure bounds (FUTURE) | adds 6 FUTURE items: schedule-dependent Bernstein support, multi-tangent PWL (piecewise-linear) tightening, ARA1 for rainflow variance, ARA1 anchor convention, more `tail_bound` values, sampling error of empirical moments | `lore:spec/spec.tex:2628-2710` |
| Change log | v0.3 only | adds v0.4 (rainflow) and v0.5 (loop-penalty removal plus review pass `REV-001..022`) | `lore:spec/spec.tex:2712-2780` |

**Interpretation.** Lore's spec extends and revises main's spec on the same formulation (two half-horizons of length `H` per solve). It also removes material (`C_P`, `alpha_target`). It keeps the in-file `%% CHANGE` markers, so the source file carries its own revision history.

## 2. spec/main.tex (on main; deleted on lore)

Evidence basis: read in full (2180 lines). Byte-identical to the merge-base version (checked with `cmp`).

**What it is (fact)**

- A **literature review**, not a specification and not an article draft.
  Title: "Stochastic Degradation Models for Train Fleet Maintenance Scheduling: Taxonomy, Literature, and Mathematical Foundations. A Comprehensive Literature Review" (`main:spec/main.tex:65-66`).
- Byline "Compiled for L. Fiaschi, ETH Zürich", dated "May 2026" (`:67-68`).
- Generated by a multi-writer process: comments name "Writer 1 ... Round 1 revision", "writer_3_round_1.tex", "writer_4_round_1.tex" and reviewer items R1-R4 (`:84,1231-1239,1904`).
- Five sections: macro taxonomy of stochastic orders (`:87`), micro taxonomy (`:638`), framing of the fleet problem (`:1242`), related work in six streams (`:1908`), conclusion (`:2152`). Bibliography `spec/references.bib` (`:2178`).

**Formulation it describes (fact)**

| Item | spec/main.tex | Same as main's spec.tex (v0.2 revised)? |
|---|---|---|
| Models | Gamma, Wiener, IG only; "Wiener" also covers Gaussian marginals (`:1301-1307`) | No: spec has a separate Gaussian model |
| Horizon | 2H days, two identical halves `[1,H]`, `[H+1,2H]` (`:1261-1264`) | Yes |
| Loop constraint | FSD of the laws, `L(D_2H) <=_st L(D_H)` (`:1366-1372`) | Spec states hard parameter rows plus soft penalty |
| Loop penalty | `C_P * L_loop` with `max(0, mu_2H - mu_H)` terms (`:1403,1415-1419`) | No: spec's penalty is the signed sum `mu_H - mu_2H` |
| Wiener variance on repair | unchanged (`:1331-1333`) | Yes for Wiener; spec's Gaussian ARD1 scales by `(1-rho)^2` |
| Reliability, Wiener | "second-order cone", MI-RSOCP (`:1349,1431-1435`) | Yes (same label that lore later corrects) |
| Reliability, Gamma and IG | precomputed caps `alpha_hat`, `mu_bar` (`:1350-1355`) | Yes |
| epsilon range | `(0, 0.5)` (`:1344`) | No: spec v0.3 tightened it to `(0, 0.01]` |
| Repair term z | linearised by big-M, "not McCormick" (`:1405-1409,1425-1430`) | No: spec says McCormick |
| Wiener caveat | admits Gaussian FSD fails with unequal variances; calls the two rows a "tractable approximation" (`:1537-1548,1746-1752`) | Consistent with lore's later wording |

**Does it supersede spec.tex on main? (interpretation)**

- No. It is a literature review, and its content is older than main's spec.tex.
  The `epsilon` range `(0, 0.5)` is the value that spec's v0.3 change log says was replaced.
- It does not mention rainflow, Cantelli, Bernstein, electric vehicles or a transitory horizon.
- Main's README label for it is checked in item 5.

## 3. spec/review_report.tex (on main; deleted on lore)

Evidence basis: read in full (721 lines). Unchanged since 61be6c3.

**What was reviewed, by whom (fact)**

- Object: `spec/main.tex`, the literature review of item 2 ("~2,181 lines"; line numbers match) (`main:spec/review_report.tex:53-55,71`).
- Reviewers: a five-role panel (Mathematician, Domain Expert, Enthusiast, Devil's Advocate, Computer Scientist), dated 13 May 2026 (`:5,56-57,73-81`).
  Interpretation: this is the output of the `panel-review` skill, not a human review.
- 88 comments: 27 "must accept" (Group A), 37 "requires judgment" (Group B); 8 debate clusters, all settled in one round (`:83-108`).

**Main verdicts (fact)**

| Item | Verdict | Evidence |
|---|---|---|
| Overall | "major revision"; "No theorem is outright false" | `:110-120` |
| A-01 | Supermartingale does not imply marginal FSD (counter-example given) | `:130-139` |
| A-02 (major) | Wiener rows `mu_2H <= mu_H`, `v_2H <= v_H` are not FSD; panel names them the increasing convex order | `:144-159` |
| A-04 | Binary-variable count wrong: 187,200, not 144,000 | `:184-195` |
| A-05 (major) | IG reduction to one mean row needs a fixed `lambda/mu` ratio; not verified | `:200-212` |
| A-06, A-07 | Lattice argument confuses tail probability with support; order preservation not proved | `:217-238` |
| A-08 to A-10 (major) | Big-M, three-case dynamics and Wiener LP approximation never derived | `:243-274` |
| A-11 (major) | Missing rolling-stock maintenance-routing literature (Maróti and Kroon 2005; Wagenaar et al. 2017) | `:279-288` |
| B-06 | The model does not enforce schedule periodicity; the guarantee needs repetition of the policy | `:483-490` |
| Reviewer scores | Enthusiast 9/10; Domain Expert 6/10; Computer Scientist 4/10 for computational content | `:681,690,708` |

**Was it applied? (fact)**

- Not to `spec/main.tex`: the linear hierarchy of A-01 is still there (`main:spec/main.tex:146-151`), and so is the 144,000 count of A-04 (`:1789-1790`).
- Lore's spec applies A-02 in its own form. It cites "accepted review item A-02" and writes "moment bounds, not FSD", without naming the increasing convex order (`lore:spec/spec.tex:1090-1100,2736`).
- Main's spec.tex does not cite A-02.

**Interpretation.** On main, the review report is a leftover from May 2026. It reviews a literature review, not the specification or the code.

## 4. scoping.tex

Evidence basis: both diffs read in full (`git diff 462ac1d main` is empty; `git diff 462ac1d lore` is 570 lines, +194/-77).

**Who changed what (fact)**

| Branch | Commits | Size |
|---|---|---|
| main | none since 462ac1d; `scoping.pdf` also unchanged | 1323 lines |
| lore | 98721b7 (2 Jul, +6/-5); 90e9ef0 (7 Jul, +188/-72); `scoping.pdf` rebuilt in both | 1440 lines |

Both versions carry the same title (lore fixes only the hyphen: "Degradation-Aware ... Management", `lore:scoping.tex:37`) and the same "Application to Electric Vehicles".

**Lore's changes, grouped (fact)**

| Group | Change | Evidence |
|---|---|---|
| Wording | "route" becomes "mission (route)"; `j=0` becomes "depot day, i.e. maintenance or idle" | `lore:scoping.tex:44-46,82` |
| New variable | `y_ik` = maintenance-day indicator; `C_M` charges `y_ik`, not `x_i0k`; reason: with equality constraints exactly `F-M` members are in the depot, so `sum x_i0k` is a constant | `lore:scoping.tex:113,127-146` |
| `tau` | "uniform across components (tau = 1 in the Palmgren-Miner normalisation)" | diff line 52 |
| Initial condition | the commented-out row `D_i,l,0 = D_start` is re-enabled | diff lines 172-177 |
| Loop constraint | new paragraph: a single-threshold comparison does not chain for two-parameter states; chaining needs a monotone sufficient statistic (Gamma shape, IG mean, moment pair, or `<=_st`) | `lore:scoping.tex:215-216` |
| Curvature | the Bernstein constraint is re-described: not SOC-representable but reverse-convex (concave square root of affine moments); tangents give conservative linear rows | diff lines 239-241 |
| New subsection | "Two refinements of the Route-I constraints": schedule-dependent Bernstein support (indicator `zeta_ij` per member-mission pair) and multi-tangent PWL tightening as a union (disjunction), with worked numbers | `lore:scoping.tex:702-746` |
| New subsection | "Random fatigue life in the Palmgren-Miner rule": inverse moments of three scatter families; block-independent versus unit-specific scatter; unit-specific case gives an exact linear chance constraint `M_k <= q_eps` | `lore:scoping.tex:748-825` |
| Corrections | IG closure formula exponent 2 becomes 3; compound Poisson ordering table corrected (`<=_st` iff `nu1 <= nu2`; `<=_lr`, `<=_hr` can fail, with counter-example); Gaussian `sigma_2H = sigma_H` claim corrected | diff lines 199-205, 419-452, 486-489 |
| ARD in parameter space | pathwise ARD leaves the shared-rate Gamma and shared-dispersion IG families; repair must act on parameters | `lore:scoping.tex:1009-1012` |
| Bounds section | old Markov/Cantelli/Chernoff list for the loop removed; replaced by "use bounds for reliability, never for the loop" | diff lines 497-520 |
| Checklists | "Stochastic orderings" and two EV-component items ticked as done | diff lines 546-558 |

- Several new footnotes say that test scripts "are kept in the verification workspace" (`lore:scoping.tex:637,660,706,816`). No such workspace is in the repository. UNVERIFIED where it lives.

**Interpretation**

- Lore's scoping and lore's spec use **two different idle conventions**.
  Scoping: `sum_j x_ijk = 1` (equality), `x_i0k` = depot day, `y_ik` = maintenance (`lore:scoping.tex:127-146`).
  Spec v0.5: `sum_j x_ijk <= 1`, idle = all-zero row, `x_i0k` = maintenance (`lore:spec/spec.tex:562-575`).
  Both make `C_M` a real decision cost, but a reader must not mix them.
- Main's scoping still has the equality constraint and charges `C_M` on `x_i0k` (merge-base text), so on main the `C_M` term is a constant per day (lore's own diagnosis, diff line 74).

## 5. README.md

Evidence basis: both READMEs read in full (main 327 lines, 12 commits since base; lore 211 lines, 3 commits: 53a9ab5, 91dd2ef, 98e3f20). Code behaviour not checked here.

**Claims side by side (fact)**

| Topic | main README | lore README |
|---|---|---|
| Source of truth | "The authoritative formulation is the one in `base.py`, `rainflow.py`, and `solver.py`" (`main:README.md:326-327`) | `spec/spec.tex` v0.5 (`lore:README.md:7,172,209`) |
| Models implemented | `rainflow`, `gamma`; `gaussian`, `inverse_gaussian` "Reserved", raise `NotImplementedError`; Wiener not mentioned (`:12-16`) | all five: Gaussian, IG, Wiener, Gamma, Rainflow (`:11-15`) |
| Horizon `H` | int or `[H1, H2]` = **transitory** (run-up from `mu_0`) plus **operating** (repeatable cycle), `T = H1 + H2` (`:67`) | int or `[H_min, H_max]` = **interval loop**, one solve per `H`; horizon is `2H` (`:61-63,115`) |
| Reliability bounds | `bound_method`: markov, cantelli (default), hoeffding, bernstein, chernoff (`:87,245-256`) | `tail_bound`: cantelli (default), bernstein (`:139`) |
| Encoding | `reliability_impl`: exact, tangent, pwl (`pwl_points` default 8) (`:106-109,261-273`) | `formulation`: exact, lp; `socp` deprecated alias (`:120`) |
| Maintenance | `repair_model`: ard1, ardinf (`:88`) | `maintenance_type`: ARD1, ARA1 (ARA1 Wiener/Gamma only) (`:123`) |
| `epsilon` | per cell, in (0, 1) (`:70`) | scalar, in (0, 0.01] (`:117`) |
| `tau` | scalar, (L,) or (F,L) (`:69`) | float or (F,L) (`:116`) |
| Costs | `C_M`, `C_R`, `C_rep` scalar or (L,) (`:74`) | `C_M` float; `C_R`, `C_rep` float, (F,) or (F,L) (`:118,125`) |
| `C_P` (loop penalty) | not in the key table, but present in the YAML example `C_P: 3.0` (`:142`) | absent (removed in spec v0.5) |
| Output keys | `m`, `r`, `H1`, `H2`, `T`, `component_names`, `model_assignment`, `bound`, `mip_gap` (`:187-206`) | `x_m`, `x_r`, `tail_bound`, `v` = NaN where untracked (`:153-160`) |
| Plot | one row per vehicle-component pair, `T+1` columns, labels `M_j`/`I`/`R`/`P`/`D`, rows labelled with component names and model (`:178-183`) | `F x 2H` grid, cells split in `L` strips, **border style** per model (`:98-103`) |
| Python | >= 3.10 (`:23`) | >= 3.9 (`:33`) |
| Tests | not described | `cd test && pytest -v` (`:195-203`) |

**Internal problems found (fact)**

- main: API section documents `plot_mixed_management` (`main:README.md:169`), but the quick start and the tree name `plot_management` (`:44,50,297`).
- main: the tree lists only `spec/main.tex`, "article/report source" (`:284-285`). `spec/spec.tex` and `spec/review_report.tex` also exist on main but are not listed.
- main: "The corresponding equations ... are documented in the report source under `spec/`" (`:258-259`). No file under `spec/` on main contains them. `grep -c -i 'cantelli|hoeffding|chernoff|bernstein'`: spec.tex 0, review_report.tex 0, main.tex 1 (the unrelated "Fréchet-Hoeffding bound"). The bounds appear only in the root `scoping.tex` (33 matches), and there only in the merge-base form.
- main: links `docs/PROJECT_LAYOUT.md` and `euler/README.md` (`:283,293,318`) do not exist on main (`git cat-file -e`).
- lore: links `test/TEST_DOCUMENTATION.md`, `input/data_example.yaml`, `input/data_example_loop.yaml` (`lore:README.md:63,146-147,203,210`) are not tracked on lore, because `lore:.gitignore` ignores `input/*` and `test/*`.
- main: the dashboard and validator sections say they "predate the new schema" (`:212-214,230`).

**Interpretation.** The two READMEs describe two different products with a partly shared vocabulary. The same input syntax `H: [a, b]` means two different things (transitory plus operating horizon on main; a sweep of horizons on lore). Main has no up-to-date specification document: its README points to `spec/`, but `spec/` holds the May 2026 literature review and the v0.2 spec.

## 6. Project files

Evidence basis: every file below read in full on both branches.

| File | main | lore | Evidence |
|---|---|---|---|
| `ASSESSMENT.md` | absent | added in 53a9ab5 (7 Jul), never edited; 61 lines | `lore:ASSESSMENT.md` |
| `Things to change.txt` | absent | added in 651aa71 (30 Jul); 5 items | `lore:"Things to change.txt":1-9` |
| `lore_check.txt` | present, merge-base text | deleted in 651aa71 | `main:lore_check.txt:1-9` |
| `pyproject.toml` | 20ae7f3: Python >= 3.10, adds `pandas`, extra `dashboard = ["streamlit"]` | merge-base: Python >= 3.9 | `git diff 462ac1d main -- pyproject.toml` |
| `.gitignore` | +35 lines, among them `docs/*`, `tests/*`, `results/*`, `mixed_plotter.py`, experiment run folders | merge-base (ignores `input/*`, `test/*`, `agentic/*`) | `git diff 462ac1d main -- .gitignore`; `lore:.gitignore` |
| `src/fleet_management.egg-info/*` | regenerated (db79e86, a42e906) | merge-base | as above |
| `.vscode/settings.json` | present (32ef801, 11 Jun): venv and pip as Python environment manager | absent | `main:.vscode/settings.json:1-4` |

**ASSESSMENT.md (fact).** Dated 2026-07-07; refers to `scoping.tex` and spec v0.5 (`lore:ASSESSMENT.md:3`).
- Verdict: the spec is "implementation-grade" (`:7-9`).
- Main risk: "Spec-code drift. The shipped code implements ~40% of the spec" (`:29-33`).
- Recommends rolling-horizon deployment, scenario (SAA, sample average approximation) chance constraints, a distributionally robust framing, decomposition, and a policy baseline (`:41-56`).
- It points to verification scripts in `agentic/workspace/verification/rainflow_bounds/` (`:25`). `agentic/*` is git-ignored, so the scripts are not in the repository.
- Lore's README calls it a "historical gap-analysis snapshot ... superseded" (`lore:README.md:211`).

**lore_check.txt versus Things to change.txt (fact).** The contents are unrelated; it is not a rename (0 shared lines).

| main:lore_check.txt (May 2026, part Italian) | lore:Things to change.txt (30 Jul 2026) |
|---|---|
| Check that a constant `lambda/mu` is reasonable for IG | Targets are electric vehicles (buses, cars), not trains |
| Check that a constant `sigma` is reasonable for Wiener | Split the horizon search into a transitory and an operating horizon; interface and formulation change |
| "Verificarlo anche per gamma" (check it for Gamma too) | Better plots: a label at the start of each row, not border styles |
| "Capire come il lavoro di castillo si pone rispetto al nostro" (understand how Castillo's work relates to ours) | Tighten the linear surrogates beyond one tangent |
| "Capire come inserire i cicli di vita" (understand how to include life cycles) | Schedule-dependent Bernstein support |

**Stale metadata (fact).**
- Both branches keep `description = "Fleet management optimization with Gaussian degradation via MILP"` in `pyproject.toml`.
- Lore's `SOURCES.txt` lists `src/fleet_management/gaussian.py` and `inverse_gaussian.py`, which lore moved to `models/`.
- Main's `SOURCES.txt` lists `src/fleet_management/test.py` and `tests/test_new_gamma.py`, which do not exist on main.
- Main's `.gitignore` ignores `docs/*`. This explains why `docs/PROJECT_LAYOUT.md`, linked from main's README, is not in the repository (interpretation).

**Interpretation.** The egg-info folder is a build artefact. Tracking it in git on either branch will produce merge conflicts with no value.

## 7. Cross-check: "Things to change.txt" against main

Evidence basis: main's README and `git grep` on main; code read only in part (`main:src/fleet_management/degradation_model/rainflow.py:108-134,350-366`). Every "main has it" is a **lead** for the code agent, not a verified fact.

| Item in `lore:"Things to change.txt"` | Does main already have it? | Evidence on main |
|---|---|---|
| Targets are electric vehicles, not trains | **Partly.** README still says "Vehicle/train fleet" (`README.md:3`). Inputs use vehicle components: `component_names: [Tires, Motor insulation]`, "Gamma traction-battery" | `examples/regression/vbz_progressive/vbz_00_reference.yaml:6`; `experiments/gamma_h1_h2_daily_5400s/README.md:13` |
| Split horizon search into transitory and operating horizon | **Yes (lead).** `H: [H1, H2]`, `T = H1 + H2`; parsed in `config.py`; H1 x H2 grid experiments; commit 70d57f0 (3 Aug) "Variable Horizon Length H_transitional and H_operating" | `README.md:67`; `src/fleet_management/config.py:53,150-156`; `experiments/gamma_h1_h2_daily_5400s/README.md:5-10` |
| Plot: label at row start instead of border styles | **Yes (lead).** Rows "labelled with `component_names` plus the assigned Gamma or remaining-life model" | `README.md:178-183` |
| Tighten linear surrogates beyond one tangent | **Yes (lead).** `reliability_impl: pwl`, `pwl_points` default 8; code comment: "one tangent active per segment (segment chosen by binaries)", which is the union (disjunction) form that lore's scoping recommends | `README.md:261-273`; `rainflow.py:360-363`; `base.py:103` |
| Schedule-dependent Bernstein support | **No (lead).** Support is one per-cell constant, `support_max_of[i,l] = _cell_max(cfg.support, cfg.support_trans, i, l)`; no member-mission indicator (`zeta`) found by `git grep` | `rainflow.py:115,130,394,450` |

**Other overlaps (lead).**
- Idle days: main has commit f166a21 (16 Sep) "Add explicit idle variable", and its plot marks `I` (idle) and `D` (depot) separately (`README.md:180-181`). Lore handles the same issue twice, in two different ways (item 4).
- Loop penalty: lore removed `C_P` (spec v0.5). Main's README example still sets `C_P: 3.0` (`README.md:142`); commit a2bb17b (22 Sep) "Use a single horizon-wide damage penalty". Whether main's code still uses `C_P` is for the code agent.
- Bounds: main offers five bounds (Markov, Cantelli, Hoeffding, Bernstein, Chernoff); lore's spec specifies two. Lore's scoping discusses all five, so main follows lore's scoping more closely than lore's spec (interpretation).

**Interpretation.** Of the five items in `Things to change.txt`, main appears to cover three fully and one partly. The open one is schedule-dependent Bernstein support. Lore's scoping (`lore:scoping.tex:702-746`) and spec FUTURE item (`lore:spec/spec.tex:2628-2649`) already specify it, so it could be added on main without new theory.
