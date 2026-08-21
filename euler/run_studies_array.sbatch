#!/usr/bin/bash
#SBATCH --job-name=fleet_studies
#SBATCH --array=0-19                  # 20 shards (submit_studies.sh overrides this)
#SBATCH --ntasks=1                    # always 1 on Euler
#SBATCH --cpus-per-task=4             # Gurobi threads (see --threads below)
#SBATCH --mem-per-cpu=4g              # B&B trees plus the relaxation copy
#SBATCH --time=24:00:00               # per ARRAY TASK, not for the whole array
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# One array task = one shard of the work. Every task writes its own
# results_shard<k>.csv into the SAME run folder; euler/merge_studies.sbatch
# combines them, aggregates over seeds and draws the figures.
set -euo pipefail

# Slurm copies the batch script to a spool dir, so $0 is useless for locating the
# project. SLURM_SUBMIT_DIR is the directory sbatch was called from --
# submit_studies.sh always calls it from the project root.
PROJECT=${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}

# Comma list: scaling,horizon,heatmap,convergence. Each study gets its OWN run
# folder and its OWN shard partition (run_studies restarts the work-unit counter
# per study), so several studies in one submission is safe -- it just means each
# task does the sum of their work.
STUDIES=${STUDIES:-scaling}
# One stamp for the whole run, minted by submit_studies.sh: every task must write
# into the SAME results/<stamp>_<study>/ folder or the shards can never be merged.
RUN_STAMP=${RUN_STAMP:?RUN_STAMP must be exported by euler/submit_studies.sh}
NAME=${NAME:-studies}
OUT=${OUT:-$PROJECT/results}
EXTRA=${EXTRA:-}
# Per-solve Gurobi limit, in seconds. NOT the same as the Slurm --time: without it
# one hard instance can consume the whole task allocation and take every later
# solve in that shard with it. run_studies records status=time_limit and
# censored=True for those rows, plots them as hollow up-arrows and keeps them out
# of the medians, so a truncated solve degrades the figure honestly.
SOLVE_TL=${SOLVE_TL:-300}
MIP_GAP=${MIP_GAP:-1e-4}
# Time limit for the CONTINUOUS relaxation, which is solved on a copy of each
# model to produce the lp_gap column. Set LP_RELAX=0 to skip it entirely.
LP_TL=${LP_TL:-60}
LP_RELAX=${LP_RELAX:-1}

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

# run_studies.py imports test.py BY PATH for the Scenario, the run-folder layout
# and the CSV schema. Without it the script exits with a clear message, but it is
# cheaper to catch here than 20 times in parallel.
if [ ! -f "$PROJECT/test.py" ]; then
    echo "ERROR: run_studies.py needs test.py next to it (not found in $PROJECT)." >&2
    exit 1
fi

# Fail loudly here rather than hundreds of times inside the harness.
# NOTE `import gurobipy` succeeds WITHOUT a licence -- the licence is only checked
# when an environment is created. So the preflight has to actually solve something.
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

# Version check inside the task too, not only at submit time: a task can start
# minutes after submission, and argparse's failure mode is a usage dump that
# buries the real problem. Costs ~1 s.
for flag in --studies --shard --run-stamp --threads --gurobi-params; do
    if ! python run_studies.py --help 2>/dev/null | grep -q -- "$flag"; then
        echo "ERROR: this run_studies.py does not support $flag -- it is older than" >&2
        echo "       these job scripts. On the login node:  git pull  (or scp the" >&2
        echo "       current run_studies.py up), then verify with:" >&2
        echo "         python run_studies.py --help | grep -c -- '--studies'   # 1" >&2
        exit 1
    fi
done

# NSHARDS is exported by submit_studies.sh. Falling back to
# SLURM_ARRAY_TASK_COUNT is a last resort: it is not set by every Slurm build, and
# if it reports the %throttle instead of the array size the shards silently stop
# partitioning the work correctly (some tasks take zero units and write empty CSVs
# -- run_studies detects that and exits 2 rather than letting the merge report
# "0 rows" with no explanation).
NSHARDS=${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-0}}
if [ -z "$NSHARDS" ] || [ "$NSHARDS" -lt 1 ] 2>/dev/null; then
    echo "ERROR: NSHARDS is unset and SLURM_ARRAY_TASK_COUNT is unavailable." >&2
    echo "       Submit via 'bash euler/submit_studies.sh <studies> <nshards>'." >&2
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
echo "studies=$STUDIES  stamp=$RUN_STAMP"
echo "solve time limit=${SOLVE_TL}s  mip gap=${MIP_GAP}  lp_relax=${LP_RELAX} (tl ${LP_TL}s)"
echo "started $(date)"

# Two cheap memory defences:
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
#   the machine and shows up in get_inefficient_jobs. It must ALSO be identical
#   across the whole study: a timing curve measured on a varying thread count is
#   not a timing curve, so never mix --cpus-per-task between submissions that
#   feed the same figure.
# --no-plots: plotting needs every seed of a design point, so it happens once in
#   the merge step. run_studies skips plots under --shard anyway; this is explicit.
ARGS=(--studies "$STUDIES" --name "$NAME" --out "$OUT"
      --mip-gap "$MIP_GAP"
      --time-limit "$SOLVE_TL"
      --lp-time-limit "$LP_TL"
      --threads "${SLURM_CPUS_PER_TASK:-1}"
      --no-plots
      --shard "${SLURM_ARRAY_TASK_ID:-0}/${NSHARDS}"
      --run-stamp "$RUN_STAMP"
      --gurobi-params "$GUROBI_EXTRA")

if [ "$LP_RELAX" = "0" ]; then
    ARGS+=(--no-lp-relax)
fi

python run_studies.py "${ARGS[@]}" $EXTRA

echo "finished $(date)"
