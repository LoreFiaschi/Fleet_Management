Status: done (7 Nov 2026). Fragments edited, assembled, copied over spec/spec.tex (CRLF kept, 2834 lines), built.

Before edits, spec/spec.tex equalled the assembly (ignoring line endings). Code read: worktree commit 8cf16c1.

| Id | Fragment | Old -> new | What the code does |
|---|---|---|---|
| S1 | fragments_C.tex ~96 | sentence "An array whose shape fits both (L,M,H*) and (F,L,M) is read as (L,M,H*)" deleted | `_profile` tests shapes in order; both fit only if F=L=M, excluded by F > M |
| S2 | fragments_D.tex ~477 | "(the larger cap where two segments contain it, ...)" -> "(where two segments contain it, at a boundary, their caps agree; ...)" | no code check; maths in the row |
| S3 | fragments_C.tex ~220 | "pwl_points is an integer >= 1" -> "... in every input, also if no cell has the form pwl" | DISAGREES with the proposal: `load_config` checks `pwl_points >= 1` always (config.py 911-913), default 8 |
| S4 | fragments_D.tex ~650 | "only if the result has the key idle" -> "listed only if the result has the key idle, not null" | validator.py 125: `if result.get("idle") is not None` |
| S5 | fragments_C.tex ~283 | added "reliability_impl is the form built at the cell ("exact" at a cell with a linear row, whatever the input says)" | `_selector(..., fill="exact")` outside quadratic cells |
| S6 | fragments_D.tex ~191 | added after Eq. d_gradient: absolute gradient (J_n - J_{n-1})/(H2_n - H2_{n-1}), key cost_gradient_per_H2, null where the relative one is not available | horizon_sweep.py `_annotate_gradients` |
| S7 | fragments_D.tex ~206, ~267 | Annotation item: error row carries the keys too (class "unavailable", eligibility false, null gradients); error-row item: "except the sizes and the gradient keys" | the annotation resets every row, error rows included |
| S8 | fragments_D.tex ~237-258 | formulation_growth: whole entry null with fewer than two rows with counts; a null count in first or last row gives null entries; optimizer_seconds/solutions_found = entries optimizer_call_seconds, solutions_found of performance; warm_start.mode null if off; n_workers integer >= 1 | `_formulation_growth`, `_case_row`, report assembly, `_is_integer(n_workers)` |
| S9 | fragments_D.tex ~493-494, fragments_C.tex ~394 | proxy ratio: "both are taken over the days, so the initial state mu_0/tau is not counted"; assignment_order "numbered from 1"; source_horizon "list [H1, H2]" | `_construction_pass` (j+1), traces of length T, base.py 419 returns `[H1_src, H2_src]` |

Package layout (fragments_D.tex 8-50): added `remote_smoke_test.py` line, `examples/` text, and `run_smoke.sh`, `run_regression.sh` in the `euler/` entry.

Build (spec/, pdflatex, bibtex, pdflatex, pdflatex; no editor build was running): 54 pages; errors 0; undefined citations or references 0; bibtex warnings 0; overfull boxes 0; "Float too large" 0.
Pages looked at (PDF index): 5, 36, 41 (also rendered 35, 45).
