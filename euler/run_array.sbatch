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
NAME=${NAME:-bound_tightness}
OUT=${OUT:-$PROJECT/results}
PARAMS=${PARAMS:-L,H,M,F}
IMPLS=${IMPLS:-exact}
EXTRA=${EXTRA:-}

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
[ $MODULE_RC -eq 0 ] || echo "warning: module load returned $MODULE_RC, continuing"

if [ ! -f "$PROJECT/.venv-euler/bin/activate" ]; then
    echo "ERROR: no venv at $PROJECT/.venv-euler -- run euler/setup_euler.sh" >&2
    exit 1
fi
source "$PROJECT/.venv-euler/bin/activate"
cd "$PROJECT"

# Fail loudly here rather than 85 times inside the harness.
python -c "import gurobipy, numpy, yaml" || {
    echo "ERROR: python environment is incomplete in this job" >&2
    module list 2>&1 >&2
    exit 1
}

NSHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
echo "project=$PROJECT"
echo "python=$(which python)  gurobi=$(python -c 'import gurobipy;print(gurobipy.gurobi.version())')"
echo "host=$(hostname)  task=${SLURM_ARRAY_TASK_ID:-0}/${NSHARDS}  cpus=${SLURM_CPUS_PER_TASK:-1}"
echo "started $(date)"

# --threads: without it Gurobi spawns a thread per core it can SEE on the node,
#   not per core Slurm gave us. On Euler nodes are shared, so that oversubscribes
#   the machine, slows everyone down, and shows up in get_inefficient_jobs.
# --no-time-limit: let each solve run to --mip-gap. The Slurm --time above is the
#   real wall-clock guard; a Gurobi time limit would silently turn slow solves
#   into "no solution" rows and corrupt the failure thresholds.
# --no-plots: plotting happens once, in the merge step.
python test.py \
    --tests "$TEST" \
    --name "$NAME" \
    --out "$OUT" \
    --params "$PARAMS" \
    --impls "$IMPLS" \
    --mip-gap 1e-4 \
    --no-time-limit \
    --threads "${SLURM_CPUS_PER_TASK:-1}" \
    --no-plots \
    --shard "${SLURM_ARRAY_TASK_ID:-0}/${NSHARDS}" \
    $EXTRA

echo "finished $(date)"
