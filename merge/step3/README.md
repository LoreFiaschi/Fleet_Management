# Step 3 of the merge: code against the spec (skill `spec-alignment`)

Status (8 Nov 2026): phases 2 to 5 done (reviews, merge, the user's approval, fixes); the spec and scoping texts are written and built; Euler: 1930 passed, smoke 12 of 12, regression 6 of 6. Committed as 6a22bd3.

- **Code**: worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge`, branch `merge/lore-into-main`, commit
  47178b8. **Spec**: `C:\Users\utente\Documents\GitHub\Fleet_Management\spec\spec.tex` (2834 lines, 54 pages), the
  authoritative copy. **README**: excluded (step 4 rewrites it).
- **Agents**: Sonnet only (rule in `~/.claude/rules/subagents.md`, *Models*). Claude plans, merges the reviews,
  asks the user, and checks the fixes.
- **Adaptation of the skill**: the skill runs two reviewers that each read the whole spec and code. Here the spec
  and code are split into four areas; each area gets two independent reviewers (same brief, no shared results).

| Area | Spec lines | Code |
|---|---|---|
| A, inputs and outputs | 1-602 (overview, architecture, notation, inputs, checks), 1709-1886 (output), 2712-2834 (schema) | `__init__.py`, `config.py`, `solver.py`; in `degradation_model/base.py` the result extraction, the reports `performance` and `optimization_progress`, the registry |
| B, formulation | 603-1708 (variables, objective, every row, the five models, repair cost, damage penalty, loop, full problem) | `degradation_model/base.py`, `gaussian.py`, `inverse_gaussian.py`, `wiener.py`, `gamma.py`, `rainflow.py`; in `config.py` the derived constants |
| C, sweeps and warm starts | 1786-1865 (reports), 1887-2273 (sweeps, warm starts, greedy) | `horizon_sweep.py`, `greedy_warm_start.py`, the warm-start functions of `degradation_model/base.py` |
| D, plots and validation | 2276-2711 (plots, structural validator, dashboards, Gamma validators) | `utils/plotter.py`, `validation/*.py`, `validator_dashboard.py`, `degradation_model/gamma_utils/*.py` |

Outputs: `review_<area>_<n>.json` (n = 1, 2), then `merged_review.json`, `approved_changes.json`,
`coder_report_*.json`.
