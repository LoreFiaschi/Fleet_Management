Status: done. Rule M19 written into the SDD and built (0 / 0 / 0).

## Edits

Fragments are in `merge/spec_drafts/`; `spec/spec.tex` is the assembled copy (CRLF).

| Location | New text (short) |
|---|---|
| `fragments_C.tex`, Run Options, row `validation` | now "settings of the validators"; key list gains `tolerance`, with its meaning and a pointer to `sec:validation_tolerance` |
| `fragments_C.tex`, Consistency Checks, Run options and costs | new item: `validation.tolerance`, if given and not `null`, is a finite number > 0, else `ValueError` naming it |
| `fragments_D.tex`, new subsection "Tolerance of the checks" (label `sec:validation_tolerance`) at the start of Validation Tools | the fact behind the rule (a binary within `IntFeasTol` in a big-M row moves a state by `IntFeasTol x M_s`); the three-step order; table of defaults; per-cell pass rule; dashboards pass the sidebar value explicitly |
| Structural Validator | `tol=None` in `validate` and `check_result`; row `tol`; Pass rule per cell and which tolerance each check uses; report key `tolerance` (single value or mapping `{FeasibilityTol, IntFeasTol}`); `checks` records (largest violation, tolerance at its cell) |
| Gamma common rules | `tolerance` default `None`, positive if given, resolution, per-cell pass, report key; binary-array error now "within the tolerance for binaries" |
| Replay validator | `tolerance=None`; step 5 uses the tolerance of its cell (all five fields); new report keys `tolerance` and `maximum_error_tolerances` |
| Exact validator | `tolerance=None`; pass rule: probability tolerance for the bound, tracked-quantity tolerance for mean and anchor; report keys as in replay |
| Monte Carlo validator | `tolerance=None`; the tolerance is used only for binaries |
| Dispatcher | `tolerance=None`; None rule: default for `tolerance` is the resolution; error list gains `tolerance` |

## Build

`assemble.py`, copy, `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`: 0 errors, 0 undefined references, 0 bibtex warnings, 0 overfull boxes, no "Float too large". 62 pages. (The log has one font-shape message, not a reference.) One first build failed because `\typerow` takes three arguments; the new table uses `\rejrow`.

## Pages looked at (pdftocairo, 70 dpi)

49 (new subsection), 50, 55 (Gamma errors and replay validator), 12, 57. The rendered tables and text are in order.
