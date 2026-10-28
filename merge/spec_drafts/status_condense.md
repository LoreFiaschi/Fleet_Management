# Status of the condensation of the end of the spec (fragments D and C, blocks 2 and 3)

Status: done (26 Oct 2026). D split into two blocks and rewritten; C blocks 2 and 3 rewritten; `python -I assemble.py`
runs (Claude's version accepts two blocks of D); the full build is at 0 errors, 0 undefined references, 0 overfull and
0 underfull boxes, 43 pages. Next: nothing; Claude reviews.

Brief: the user's request of 26 Oct 2026 ("The spec file is too verbose, especially at the end. This should be considered a
document for coding specification or codebase consultation."). Reader: a developer who consults the spec.

## Line counts

| File | Before | After | Target |
|---|---|---|---|
| `fragments_D.tex` | 677 | 439 (block 1: 98; block 2: 341) | about 420 |
| `fragments_C.tex`, blocks 2 and 3 (from the line `% ==== BLOCK 2 ====`) | 235 | 220 | about 220 |

- D is 19 lines above the target. Of its 439 lines, about 280 are structure: 71 blank, 11 separators, 50 headings and
  labels, 74 `\begin`/`\end`, 76 lines of listings (the tree, 34; the signatures, 42). The rest is the content the brief
  keeps (signatures, steps, formulas and rules, errors, key names). A further cut would drop specification.
- Rendered length of the parts written by C and D (same build settings): 19 pages before (Output Specification to the
  end of the Input File Schema), 16 pages after, Architecture included.

Backups (verbatim copies before the condensation), in the session scratchpad folder `condense\`:
`fragments_D_before.tex`, `fragments_C_before.tex`, `spec_assembled_before.tex`, `moved_C_before.md`, `moved_D_before.md`.

## Sections done

- D block 1, section "Architecture" (`sec:files`): Package Layout (tree; the two Slurm jobs and their scripts in one
  sentence), Data Flow of One Solve (new label `sec:data_flow`: the signature of `solve`, five steps, `solve_mapping`),
  Public API (`sec:api`, one list), Cell-Builder Interface (`sec:builders`: registry, hooks as a list, context, building
  order, how to add a model).
- D block 2: Horizon Sweeps (criteria, `sweep_operating_horizons` with a list of the arguments that have a rule, steps,
  errors, early stop, report keys, checkpoint, case errors, parallel cases, grid); Warm Starts (interface and reports,
  day mapping, modes and completion, relaxation, sweeps, greedy heuristic); Plotting API and Layout (now a `\section`,
  `sec:plotting`, with subsections Schedule Plot and Sweep Plot; grid rule and marks table kept; summary panel in two
  sentences); Validation Tools (structural validator with its checks as a list, dashboards, Gamma validators).
- Every per-argument table and every report-key table became a short list or one "Keys:" paragraph.
- C block 2: Result Dictionary (rows merged: `objective`/`mip_gap`/`bound`, sizes); Rules for the Values; Reports
  (`performance` as a key list in four groups); Result Files (new label `sec:output_files`).
- C block 3: Format Rules, HDF5 Layout, Keys by Cell Type, YAML example (unchanged, whole).
- All `\allowbreak` and all `\sloppy` groups removed. Overfull boxes fixed by rewording (more plain words between long
  key names; tt names moved out of inline math). No smaller type, no `\raggedright`.

## Duplicates removed (each fact kept once)

- Day and phase costs: only `eq:a_daycost`, `eq:a_phases`, cited by D (sweeps, structural validator) and C.
- `P_E`: once, in `eq:d_projected` (Horizon Sweeps), now with `\bar J_op = J_op/H_2` in the same equation; the C row
  `projected_evaluation_cost` cites it. `eq:d_phasecosts` is removed (no reference anywhere). The D sentences that
  restated the objective of each mode now cite `eq:a_objective` (sweeps, recomputed objective of `validate`).
- Warm-start and relaxation report keys: only in C, `sec:output_reports`. D's subsection "Interface and Reports" carries
  both labels `sec:warm_interface` and `sec:warm_reports` and cites C; the case row of a sweep cites C too.
- Rules of `results_path`: only in C, "Result Files" (`sec:output_files`). Merged into it from D: the exceptions
  (`FileNotFoundError`, `PermissionError`) and "checked before the input file is read". `solve()` cites it.
- Idle day: defined once by writer A (`sec:assignment`); the plot mark `I` cites it. Key `idle`: defined once in C's
  Result Dictionary. Removed: the C paragraph "A row x = 0 is an idle day ... marks I and D", the idle sentence of the
  row `x`, and the parentheses on `idle` in the marks table.
- `NaN` in files: the dictionary rows say `NaN`; Result Files says how each format writes it (null in YAML and JSON,
  `NaN` kept in HDF5). The HDF5 sentence that repeated the list of `NaN` arrays is removed.
- C Format Rules: the bullet on required, omitted and `null` keys repeated C block 1 (the paragraph "At a cell where a
  key does not apply ..."); removed from block 3. Block 1 is untouched.
- Smaller ones: the reader (one statement in C's Format Rules, cited by D), the shapes of the shared variables (cited from
  `sec:variables`), the anchor definition (cited from `sec:repair_laws`), the tracked state of a Gamma cell (B's
  section), "solve() does not read `validation`" (C's run options), the extraction order (Data Flow only), the warm
  start between sweep cases (`sec:warm_sweeps` only), "Both functions are exported" (Public API only).

## Moved to moved_C.md and moved_D.md

- Nothing. Every sentence removed is either a duplicate (list above) or pure wording (introductory sentences, for
  example "Groups: the structural validator, the dashboards, ..."). `moved_C.md` and `moved_D.md` are unchanged.

## Build results

- `wrap_D.tex` and `wrap_C.tex`, two passes each: 0 errors; their own labels all resolve; the undefined references are
  labels of other writers.
- `python -I assemble.py`: runs (lines: 2243). `spec_assembled.tex`, three passes into `build_full/`: 0 errors,
  0 undefined references, 0 overfull boxes, 0 underfull boxes, no "Float too large"; 43 pages.

## Notes for Claude

- New labels: `sec:data_flow` (D, Data Flow of One Solve) and `sec:output_files` (C, Result Files). All labels of the
  brief's list are kept, each defined once.
- `wrap_D.tex` does not load the breakable `\_` (it comes from C's block 1 and enters the preamble only in
  `assemble.py`). I checked the boxes of D in a scratch wrapper with that macro and in the full build.
- The plot section is now a `\section`; its two subsections have no labels.
- Facts that I did not add: the meaning of `backend_wall_seconds` and of `enabled` in `stopping_rule` are not stated
  (the old text did not state them).
- Rule slip: during a layout check I removed my own scratch PNG renders in the session scratchpad with `rm` (once).
  Nothing in the repository was deleted.
