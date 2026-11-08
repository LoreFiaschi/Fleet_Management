# WP8 (tests): split, file ownership and shared rules

Status (7 Nov 2026): done, commit 8cf16c1; findings in `wp8_defects.md`.

- **Purpose.** WP8 confirms that the merged code does what the spec says and that its schedules mean what the spec
  says. User, 7 Nov 2026: *"Create your own unit tests, the input files to test it etc. Thus, you can confirm that the
  code abide by the spec and is semantically correct"*. `lore`'s `input/` folder is not needed: WP8 writes its own
  inputs.
- **Where.** Code and tests: the worktree `C:\Users\utente\Documents\GitHub\Fleet_Management-merge`, branch
  `merge/lore-into-main`, at commit fb6097a (Euler scripts). Records (plans, reviews, defects): the `lore` checkout,
  `C:\Users\utente\Documents\GitHub\Fleet_Management\merge\`. The merge branch never edits its own `merge/` folder.
- **Authority.** The `lore` copy of the spec, `C:\Users\utente\Documents\GitHub\Fleet_Management\spec\spec.tex`
  (2830 lines; newer than the worktree's copy). Line numbers below refer to it.
- **Decisions** (`merge/decisions.md`): "Tests and test inputs" (one suite in `tests/`, `testpaths = ["tests"]`; the
  smoke test ported with its inputs under `examples/` and extended to the merge's changes; `main`'s check scripts kept
  where they still apply), "Cluster jobs and experiments", "General rules for the merge".

## Workflow (user, 7 Nov 2026: *"again use sonnet only, you look at the results, plan and orchestrate"*)

1. Claude plans each part: `merge/wp8_plan_<part>.md`.
2. Sonnet agents carry out the plans, at most four at once. They run the tests that need no Gurobi on this computer.
3. Euler run: a Sonnet agent copies the worktree to Euler (`EULER.md`, "One test cycle", steps 2 and 3); Claude
   submits the job and reads the results.
4. Claude classifies each failing test: a **test defect** (the test contradicts the spec), a **code defect** (the code
   contradicts the spec), or a **spec question** (the spec is silent, ambiguous or looks wrong).
5. Claude records the defects in `merge/wp8_defects.md` and gives each file one owner; Sonnet agents fix the tests
   and the code. Spec questions go to the user. Steps 3 to 5 repeat until the suite passes.
6. Claude reviews the diff and commits on `merge/lore-into-main`; the final code goes to the Euler main folder.

## The independence rule (parts `io`, `oracle`, `tools`)

- Every expected value comes from the spec, derived by hand or by an independent computation written from the spec.
  Never from running the code under test, and never from reading a function body in `src/`.
- `src/` may be read for names, signatures, docstrings, exception types and the keys of dictionaries, to call the
  code. When the spec does not fix a value or a behaviour, the test does not assert it; the plan lists it under
  "Questions".
- The existing files in `tests/` were written from the code. Do not read them before the plan is written. Afterwards a
  planned test may be dropped only if an existing test asserts the same spec statement with the same expected value;
  the plan says which.
- Part `legacy` ports existing material and may read the code to adapt it.

## Parts and file ownership

| Part | Spec (lines of the `lore` copy) | Creates or changes |
|---|---|---|
| `io`: inputs, checks, outputs | Public API 229-241; Index Sets 270-348; Input Specification 349-602; Output Specification 1709-1886; Input File Schema 2712-2830 | `tests/test_spec_inputs.py`, `tests/test_spec_checks.py`, `tests/test_spec_outputs.py`, `tests/test_spec_schema.py` (names may change, prefix `test_spec_`); data in `tests/data/io/` |
| `oracle`: meaning of a schedule | Index Sets 270-348; Input Specification 349-602 (meaning only); Decision Variables to Full Optimization Problem 603-1708; Result Dictionary 1714-1785 | the reference model `tests/oracle/` (or one module `tests/reference_model.py`); `tests/test_oracle_*.py`; data in `tests/data/oracle/` |
| `tools`: sweeps, warm starts, plots, validators | Architecture 165-269; Reports 1786-1865; Horizon Sweeps 1887-2108; Warm Starts 2109-2275; Plotting 2276-2380; Validation Tools 2381-2711 | `tests/test_spec_sweeps.py`, `tests/test_spec_warm_starts.py`, `tests/test_spec_plots.py`, `tests/test_spec_validation.py` (prefix `test_spec_`); data in `tests/data/tools/` |
| `legacy`: earlier tests and scripts | as each file needs | `lore`'s `test/` (lore checkout) ported into `tests/test_ported_*.py`; `remote_smoke_test.py` with its inputs in `examples/smoke/` and a job script `euler/run_smoke.sh`; `examples/regression/`, `experiments/`, `input/` (rewrite, or list for removal) |

- **Nobody edits `src/`, the spec, `pyproject.toml` or the existing files of `tests/`** in steps 1 to 4. Code fixes
  come in step 5, under Claude's assignment. A change to `pyproject.toml` or a `conftest.py` is asked under
  "Questions".
- **Removals**: a plan names the files to remove in a section "Removals"; Claude removes them at the commit.
  Executors do not delete files.
- Test file names are unique across `tests/` (pytest imports test files by base name).

## Rules for every plan

- **What the oracle means.** An independent re-implementation, from the spec, of what a schedule implies: the state of
  every component on every day under each model and repair law, the replacement resets, the reliability rows in each
  form, the loop rows, the repair cost and the damage penalty, the objective. It checks every solved result, and on
  tiny fleets it enumerates every schedule and compares the best feasible cost with the solver's optimum.
- **Small and fast.** Each solve under 5 s; the new tests together under 15 min on Euler (4 CPUs). Fixed seeds.
- **Gurobi.** A test that needs Gurobi is skipped per test, not per module (the pattern of `tests/test_gamma.py`), so
  the rest runs on this computer.
- **Each test cites its spec location** (section and line) in its docstring or a comment.
- Test inputs are small YAML files (JSON and HDF5 where a format is the point), each with a comment line saying what
  it covers.
- Code style: as the existing test files of `tests/` (pytest functions, `tmp_path`, `pytest.raises`).
- Every plan ends with: "Questions", "Removals", and an empty section "Review" for step 4.

## Local tools (this computer: numpy, scipy, pyyaml, h5py, pandas, matplotlib; no pytest, no gurobipy; nothing may be installed)

- `merge/wp45_work/wp4/run_tests.py <test module> [name]` runs pytest-style tests without pytest;
  `merge/wp45_work/wp6/gamma/run_tests_gamma.py` and `merge/wp45_work/wp6/rainflow/review_run.py` add `pytest.mark`,
  `skipif` and `importorskip`. Copy a runner to your scratchpad folder before use, because it writes temporary files
  beside itself.
- `merge/wp45_work/wp5/stubs/gurobipy/`: a stub that lets modules importing `gurobipy` load.
- Euler (Gurobi 13.0.0, full licence): `C:\Users\utente\Documents\GitHub\Fleet_Management\EULER.md`.
  `sbatch run_tests.sh [pytest targets]` runs the given tests, or the whole suite. Only Claude submits jobs.

## Execution order

Step 1: Claude writes the four plans. Step 2, batch 1: `io`, `oracle` core, `tools`, `legacy` (lore's `test/`). Later
batches as the plans order them, at most four agents at once.
