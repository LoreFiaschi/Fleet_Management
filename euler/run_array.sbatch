#!/usr/bin/bash
#SBATCH --job-name=bound_tests
#SBATCH --array=0-19                  # 20 shards (submit.sh overrides this)
#SBATCH --ntasks=1                    # always 1 on Euler
#SBATCH --cpus-per-task=4             # Gurobi threads (see --threads below)
#SBATCH --mem-per-cpu=4g              # MIQCP branch-and-bound trees get large
#SBATCH --time=24:00:00               # per ARRAY TASK, not for the whole array
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# One array task = one shard of the work. Each shard writes its own results
# folder; euler/merge.sbatch combines them and evaluates the hypotheses.
set -euo pipefail

# Slurm copies the batch script to a spool dir, so $0 is useless for locating the
# project. SLURM_SUBMIT_DIR is the directory sbatch was called from -- submit.sh
# always calls it from the project root. PROJECT can also be passed explicitly.
PROJECT=${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}

TEST=${TEST:-sweep}                   # analytic | base | sweep | impl | failure
# One stamp for the whole run, minted by submit.sh: every task must write into the
# SAME results/<stamp>_<test>/ folder or the shards can never be merged.
RUN_STAMP=${RUN_STAMP:?RUN_STAMP must be exported by euler/submit.sh}
NAME=${NAME:-bound_tightness}
OUT=${OUT:-$PROJECT/results}
PARAMS=${PARAMS:-L,H,M,F}
IMPLS=${IMPLS:-exact}
# MILP encoding of the logical constraints (rainflow_v2):
#   FORM   one encoding for every solve in this shard: indicator | bigm
#   FORMS  only for TEST=formulation, which solves BOTH and compares them
#   BIGM   fallback big-M for a state with no finite bound (states are in [0,1.1])
# Leaving FORM empty keeps the harness default ('indicator'), i.e. exactly the
# model rainflow.py has always built -- so an old submission reproduces bit for
# bit and only an explicit FORM= changes the encoding.
FORM=${FORM:-}
FORMS=${FORMS:-indicator,bigm}
BIGM=${BIGM:-}
EXTRA=${EXTRA:-}
# --tests case only: comma-separated input file names, resolved under INPUT_DIR
CASES=${CASES:-}
INPUT_DIR=${INPUT_DIR:-input}
# Per-solve Gurobi limit, in seconds. This is NOT the same as the Slurm --time:
# without it, one intractable MIQCP can consume the whole 24 h task (and grow a
# branch-and-bound tree until the node is out of memory), taking every later solve
# in that shard down with it -- the CSV is written per finished solve, so a killed
# task loses everything it had not yet completed.
# The gap is recorded per run, and the summary reports INCONCLUSIVE where the gap
# is too wide to rank two bounds, so a truncated solve degrades the conclusion
# honestly instead of silently.
# Set SOLVE_TL=0 for no per-solve limit (only sensible for the failure test, where
# feasibility rather than cost is the measurement).
SOLVE_TL=${SOLVE_TL:-900}

# `module` is a shell function set up by the login profile. A batch script runs a
# NON-login shell, so on some setups it is undefined here and every task dies
# instantly with "module: command not found" -- which under `set -e` looks like a
# mass failure with no useful message. Bootstrap it if needed.
if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh \
                /cluster/apps/modules/init/bash \
                /usr/share/lmod/lmod/init/bash; do
        [ -f "$init" ] && . "$init" && break
    done
fi
if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: 'module' is unavailable in this batch shell. Change the shebang" >&2
    echo "       of this script to '#!/usr/bin/bash -l' (login shell) and retry." >&2
    exit 1
fi

# module commands can return non-zero on harmless warnings; do not let set -e
# turn that into a failed task.
set +e
module purge
module load stack/2024-06 python/3.12.8 gurobi/12.0.1
MODULE_RC=$?
set -e
if [ $MODULE_RC -ne 0 ]; then
    echo "ERROR: 'module load' returned $MODULE_RC. Without the gurobi module" >&2
    echo "       GRB_LICENSE_FILE is unset and every solve fails. Aborting." >&2
    module list 2>&1 >&2
    exit 1
fi

if [ ! -f "$PROJECT/.venv-euler/bin/activate" ]; then
    echo "ERROR: no venv at $PROJECT/.venv-euler -- run euler/setup_euler.sh" >&2
    exit 1
fi
source "$PROJECT/.venv-euler/bin/activate"
cd "$PROJECT"

# Fail loudly here rather than 85 times inside the harness.
# NOTE `import gurobipy` succeeds WITHOUT a licence -- the licence is only checked
# when an environment is created. So the preflight has to actually solve something,
# otherwise a missing or unreachable token server surfaces later as an opaque
# exception from inside the harness.
set +e
python - <<'PREFLIGHT'
import numpy, yaml, gurobipy as gp
m = gp.Model()                     # <- this is what needs the licence
m.Params.OutputFlag = 0
x = m.addVar(ub=2)
m.setObjective(x, gp.GRB.MAXIMIZE)
m.optimize()
assert abs(m.ObjVal - 2.0) < 1e-9
print(f"preflight ok: gurobipy {gp.gurobi.version()}, numpy {numpy.__version__}")
PREFLIGHT
PF_RC=$?
set -e
if [ $PF_RC -ne 0 ]; then
    echo "ERROR: preflight failed (exit $PF_RC). Diagnostics:" >&2
    echo "  python           : $(which python)" >&2
    echo "  GRB_LICENSE_FILE : ${GRB_LICENSE_FILE:-<UNSET -- gurobi module not loaded>}" >&2
    module list 2>&1 >&2 || true
    gurobi_cl --tokens 2>&1 >&2 || echo "  gurobi_cl --tokens failed" >&2
    exit 1
fi

# NSHARDS is exported by submit.sh. Falling back to SLURM_ARRAY_TASK_COUNT is a
# last resort: it is not set by every Slurm build, and if it is missing or reports
# the %throttle instead of the array size, the shards silently stop partitioning
# the work correctly (some tasks take zero units and write empty CSVs).
# Version check inside the task too, not only at submit time: a task can start
# minutes after submission, and argparse's failure mode is a usage dump that buries
# the real problem. Costs ~1 s.
for flag in --threads --shard --gurobi-params --formulation; do
    if ! python test.py --help 2>/dev/null | grep -q -- "$flag"; then
        echo "ERROR: this test.py does not support $flag -- it is older than these" >&2
        echo "       job scripts. On the login node:  git pull  (or scp the current" >&2
        echo "       test.py up), then verify with:" >&2
        echo "         python test.py --help | grep -c -- '--gurobi-params'   # 1" >&2
        exit 1
    fi
done

NSHARDS=${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-0}}
if [ "$NSHARDS" -lt 1 ] 2>/dev/null || [ -z "$NSHARDS" ]; then
    echo "ERROR: NSHARDS is unset and SLURM_ARRAY_TASK_COUNT is unavailable." >&2
    echo "       Submit via 'bash euler/submit.sh <test> <nshards>'." >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "$NSHARDS" ]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} >= NSHARDS ${NSHARDS}: the array size" >&2
    echo "       and the shard count disagree." >&2
    exit 1
fi
echo "project=$PROJECT"
echo "python=$(which python)  gurobi=$(python -c 'import gurobipy;print(gurobipy.gurobi.version())')"
echo "host=$(hostname)  task=${SLURM_ARRAY_TASK_ID:-0}/${NSHARDS}  cpus=${SLURM_CPUS_PER_TASK:-1}"
echo "solve time limit=${SOLVE_TL}s  mip gap=${MIP_GAP:-1e-4}"
if [ "$TEST" = "formulation" ]; then
    echo "formulations=${FORMS} (compared against each other)"
else
    echo "formulation=${FORM:-indicator (harness default)}"
fi
echo "started $(date)"

# A nonconvex MIQCP with a weak dual bound grows an enormous search tree (8e6
# nodes measured). Two defences, both cheap:
#   NodefileStart  spill the tree to node-local scratch past 1 GB of RAM
#   SoftMemLimit   stop GRACEFULLY at ~80 % of the allocation and return the
#                  incumbent, instead of being SIGKILLed by the OOM killer --
#                  a killed task loses every row it had not yet written
MEM_PER_CPU_MB=${SLURM_MEM_PER_CPU:-4096}
MEM_TOTAL_GB=$(( MEM_PER_CPU_MB * ${SLURM_CPUS_PER_TASK:-1} / 1024 ))
SOFT_MEM_GB=$(( MEM_TOTAL_GB * 8 / 10 ))
[ "$SOFT_MEM_GB" -lt 1 ] && SOFT_MEM_GB=1
GUROBI_EXTRA="NodefileStart=1,NodefileDir=${TMPDIR:-/tmp},SoftMemLimit=${SOFT_MEM_GB}"
if [ -n "${GUROBI_PARAMS:-}" ]; then
    GUROBI_EXTRA="$GUROBI_EXTRA,$GUROBI_PARAMS"
fi
echo "gurobi memory guard: ${SOFT_MEM_GB} GB soft limit of ${MEM_TOTAL_GB} GB allocated"

# --threads: without it Gurobi spawns a thread per core it can SEE on the node,
#   not per core Slurm gave us. On Euler nodes are shared, so that oversubscribes
#   the machine, slows everyone down, and shows up in get_inefficient_jobs.
# --time-limit: see SOLVE_TL above. Bounds each solve so one hard instance cannot
#   take the whole task, and its Slurm allocation, down with it.
# --no-plots: plotting happens once, in the merge step.
ARGS=(--tests "$TEST" --name "$NAME" --out "$OUT"
      --mip-gap "${MIP_GAP:-1e-4}"
      --time-limit "$SOLVE_TL"
      --threads "${SLURM_CPUS_PER_TASK:-1}"
      --no-plots
      --shard "${SLURM_ARRAY_TASK_ID:-0}/${NSHARDS}"
      --run-stamp "$RUN_STAMP"
      --gurobi-params "$GUROBI_EXTRA")

# The case test takes input files instead of a generated grid, so --params /
# --impls do not apply to it and --case does. For every other test it is the
# other way round. Note the file is authoritative for a case: --mip-gap and
# --time-limit above only fill keys the YAML leaves out.
if [ "$TEST" = "case" ]; then
    if [ -z "$CASES" ]; then
        echo "ERROR: TEST=case needs CASES=name1,name2 (files under $INPUT_DIR/)" >&2
        exit 1
    fi
    ARGS+=(--case "$CASES" --input-dir "$INPUT_DIR")
    echo "cases: $CASES   input dir: $INPUT_DIR"
else
    ARGS+=(--params "$PARAMS" --impls "$IMPLS")
fi

# The 'formulation' test needs BOTH encodings (it compares them), every other
# test needs at most one. Passing --formulation to the formulation test would be
# silently overridden by --formulations, so the two are mutually exclusive here.
if [ "$TEST" = "formulation" ]; then
    ARGS+=(--formulations "$FORMS")
elif [ -n "$FORM" ]; then
    ARGS+=(--formulation "$FORM")
fi
[ -n "$BIGM" ] && ARGS+=(--bigM "$BIGM")

python test.py "${ARGS[@]}" $EXTRA

echo "finished $(date)"
