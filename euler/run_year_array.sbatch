#!/usr/bin/bash
#SBATCH --job-name=vbz_year
#SBATCH --array=0-4                   # 5 strategies (submit_year.sh overrides)
#SBATCH --ntasks=1                    # always 1 on Euler
#SBATCH --cpus-per-task=8             # Gurobi threads (see --threads below)
#SBATCH --mem-per-cpu=4g              # 37 804 vars / 27 456 indicators
#SBATCH --time=24:00:00               # per ARRAY TASK, not for the whole array
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# One array task = one SOLVER STRATEGY on the SAME instance. This array does not
# shard work; it races configurations, because the open question is not "how long
# does it take" but "which configuration produces an incumbent at all". The first
# cluster run (600 s, MIPFocus default, no start) ended with sol_count=0 and
# nodes=1 -- the root node never finished.
#
# Each task writes results_<case>_<strategy>.csv into the same run folder;
# euler/merge_year.sbatch concatenates them and ranks by incumbent quality.
set -euo pipefail

PROJECT=${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}

CASE=${CASE:-vbz_man12e_year_solve}   # tau = 1200 twin; see derivation.md 9.3
INPUT_DIR=${INPUT_DIR:-input}
RUN_STAMP=${RUN_STAMP:?RUN_STAMP must be exported by euler/submit_year.sh}
NAME=${NAME:-vbz_case}
TEST=${TEST:-year}
OUT=${OUT:-$PROJECT/results}
# Comma list, one strategy per array task, index = SLURM_ARRAY_TASK_ID.
STRATEGIES=${STRATEGIES:-cold,warm,warm-norel,warm-bound,warm-pwl}
# Per-solve Gurobi limit. NOT the Slurm --time: leave headroom so the task can
# still write its CSV and schedule after Gurobi stops.
SOLVE_TL=${SOLVE_TL:-72000}
MIP_GAP=${MIP_GAP:-0.05}
EXTRA=${EXTRA:-}

# `module` is a shell function from the login profile; a batch script runs a
# NON-login shell, so bootstrap it or every task dies with "module: not found".
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

YAML="$INPUT_DIR/${CASE}.yaml"
if [ ! -f "$YAML" ]; then
    echo "ERROR: no instance at $PROJECT/$YAML" >&2
    exit 1
fi

# `import gurobipy` succeeds WITHOUT a licence -- it is only checked when an
# environment is created, so the preflight must actually solve something.
set +e
python - <<'PREFLIGHT'
import numpy, yaml, gurobipy as gp
m = gp.Model()
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

# Version check inside the task too: a task can start long after submission.
for flag in --strategy --gurobi-params --run-stamp --threads; do
    if ! python run_year.py --help 2>/dev/null | grep -q -- "$flag"; then
        echo "ERROR: this run_year.py does not support $flag -- it is older than" >&2
        echo "       these job scripts. git pull (or scp it up) and verify with:" >&2
        echo "         python run_year.py --help | grep -c -- '--strategy'   # 1" >&2
        exit 1
    fi
done

IFS=',' read -r -a STRAT_ARR <<< "$STRATEGIES"
NSTRAT=${#STRAT_ARR[@]}
TASK=${SLURM_ARRAY_TASK_ID:-0}
if [ "$TASK" -ge "$NSTRAT" ]; then
    echo "ERROR: task $TASK >= ${NSTRAT} strategies: the array size and the" >&2
    echo "       STRATEGIES list disagree." >&2
    exit 1
fi
STRATEGY=${STRAT_ARR[$TASK]}

echo "project=$PROJECT"
echo "python=$(which python)  gurobi=$(python -c 'import gurobipy;print(gurobipy.gurobi.version())')"
echo "host=$(hostname)  task=$TASK/$NSTRAT  cpus=${SLURM_CPUS_PER_TASK:-1}"
echo "case=$CASE  strategy=$STRATEGY  stamp=$RUN_STAMP  test=$TEST"
echo "checkpoint every ${CKPT:-900}s -> checkpoints_<strategy>.csv + snapshots/"
echo "solve time limit=${SOLVE_TL}s  mip gap=${MIP_GAP}"
echo "started $(date)"

# Two cheap memory defences, same as the other arrays:
#   NodefileStart  spill the tree to node-local scratch past 1 GB of RAM
#   SoftMemLimit   stop GRACEFULLY at ~80 % of the allocation and return the
#                  incumbent, instead of being SIGKILLed by the OOM killer
MEM_PER_CPU_MB=${SLURM_MEM_PER_CPU:-4096}
MEM_TOTAL_GB=$(( MEM_PER_CPU_MB * ${SLURM_CPUS_PER_TASK:-1} / 1024 ))
SOFT_MEM_GB=$(( MEM_TOTAL_GB * 8 / 10 ))
[ "$SOFT_MEM_GB" -lt 1 ] && SOFT_MEM_GB=1
GUROBI_EXTRA="NodefileStart=1,NodefileDir=${TMPDIR:-/tmp},SoftMemLimit=${SOFT_MEM_GB}"
if [ -n "${GUROBI_PARAMS:-}" ]; then
    GUROBI_EXTRA="$GUROBI_EXTRA,$GUROBI_PARAMS"
fi
echo "gurobi memory guard: ${SOFT_MEM_GB} GB soft limit of ${MEM_TOTAL_GB} GB allocated"

python run_year.py "$YAML" \
    --strategy "$STRATEGY" \
    --name "$NAME" --test "$TEST" --out "$OUT" --run-stamp "$RUN_STAMP" \
    --checkpoint-every "${CKPT:-900}" \
    --time-limit "$SOLVE_TL" --mip-gap "$MIP_GAP" \
    --threads "${SLURM_CPUS_PER_TASK:-1}" \
    --gurobi-params "$GUROBI_EXTRA" $EXTRA

echo "finished $(date)"
